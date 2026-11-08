"""Explicit second-round clone adapter; the first-round shard stays frozen.

One deterministic clone setup is attempted after the ORIGINAL scene preflight.
Failure ends the scene; no object is deleted and the same Controller is never
reused for another clone. Only the common baseline contains the new candidate;
the counterfactual transition still edits the distinct anchor X alone.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

from . import pair_v2_shard as base
from . import pair_v2_simulator as engine
from .pair_v2_clone import V2CloneSimulator, prepare_clone_baseline, clone_replay_worker
from .pair_v2_clone_policy import clone_pair_check
from .pair_v2_construct import identity_sets as natural_identity_sets, distance
from .thor_common import Worker


CLONE_RULE = {
    'version': 1,
    'selection': 'first (objectType,name) singleton pickupable with a distinct singleton pickupable anchor',
    'proposal': 'nearest/name-ordered open receptacles; native xyz order; source-ID and editable AABB footprint proposal filters',
    'attempts_per_controller': 1,
    'natural_fallback': 'only if no eligible singleton source/anchor structure exists',
    'failure': 'retain evidence and stop scene; never retry/delete in this Controller',
    'acceptance': 'unchanged v2 policy plus authenticated common clone baseline and fresh lineage replay',
}
_PREPARATION = None  # Private spawned generator process only; reset in finally.
_REPLAY_CLEANUP = None


def enable_config(config):
    """Add the opt-in at top level without altering the audited shard manifest."""
    config['enable_candidate_clone'] = True
    config['max_raw_bytes'] = 2 * 1024 * 1024 * 1024
    config['clone_raw_storage_limit'] = {
        'max_stored_payload_bytes_per_simulator': config['max_raw_bytes'],
        'first_round_limit_bytes': 512 * 1024 * 1024,
        'scope': 'each scratch EvidenceStore (construction or fresh replay); incremental index overhead is additional',
        'purpose': 'hard storage guard, not a scientific acceptance threshold',
        'on_limit': 'fail and retain completed evidence; no deletion or acceptance override',
    }
    config['clone_rule'] = copy.deepcopy(CLONE_RULE)
    config['clone_adapter_source'] = {'file': __file__, 'sha256': base.sha(__file__)}
    config['clone_portable_archive'] = {
        'scope': 'referenced JSON proofs only; final PNG/masks remain in ordinary pair export',
        'max_file_bytes': JsonProofArchive.MAX_FILE_BYTES,
        'max_run_bytes': JsonProofArchive.MAX_TOTAL_BYTES,
        'max_pair_json_files': JsonProofArchive.MAX_PAIR_FILES,
        'on_limit': 'reject export, retain prior raw evidence and partial archive; never count as accepted',
    }
    return config


def clone_source(objects):
    groups = {}
    for obj in objects:
        groups.setdefault(obj['objectType'], []).append(obj)
    singletons = [items[0] for _, items in sorted(groups.items())
                  if len(items) == 1 and items[0].get('pickupable') is True]
    return sorted(singletons, key=lambda obj: (obj['objectType'], obj['name']))[0] if len(singletons) >= 2 else None


def footprint_clear(source, point, objects, receptacle_name):
    """AABB proposal rejection only. Native placement/rest/parent still decide."""
    source_box = source.get('axisAlignedBoundingBox') or {}
    source_size = source_box.get('size')
    if not source_size:
        return True  # Unknown geometry is never reported as physical clearance.
    for obj in objects:
        if obj['name'] == receptacle_name or not (obj.get('pickupable') or obj.get('moveable')):
            continue
        box = obj.get('axisAlignedBoundingBox') or {}
        center, size = box.get('center'), box.get('size')
        if center and size and all(abs(point[axis] - center[axis]) < (source_size[axis] + size[axis]) / 2
                                   for axis in ('x', 'z')):
            return False
    return True


def select_clone_token(sim, baseline, source, record):
    objects = baseline['objects']
    receptacles = sorted((obj for obj in objects if obj.get('receptacle') is True
        and obj['name'] != source['name'] and not (obj.get('openable') and not obj.get('isOpen'))),
        key=lambda obj: (distance(obj['position'], source['position']), obj['name']))
    record['proposal_queries'] = []
    for receptacle in receptacles[:sim.config['max_receptacles']]:
        query = {'receptacle_name': receptacle['name'], 'selected': None, 'rejected': []}
        record['proposal_queries'].append(query)
        try:
            tokens = sim.spawn_points(receptacle['name'])
        except (TimeoutError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            query['error'] = f'{type(error).__name__}: {error}'
            continue
        for token in sorted(tokens, key=lambda t: tuple(t['selected_position'][axis] for axis in ('x', 'y', 'z'))):
            point = token['selected_position']
            reason = ('source_id_proximity_proposal' if distance(source['position'], point) <= .02
                      else 'editable_aabb_footprint_proposal' if not footprint_clear(source, point, objects, receptacle['name'])
                      else None)
            if reason:
                query['rejected'].append({'position': copy.deepcopy(point), 'reason': reason})
                continue
            query['selected'] = copy.deepcopy(token)
            return token
    return None


class PreparationSimulator(V2CloneSimulator):
    def preflight_scene(self):
        global _PREPARATION
        original = super().preflight_scene()
        record = {'enabled': True, 'status': 'original_preflight_failed', 'passed': False,
                  'rule': copy.deepcopy(CLONE_RULE), 'clone_attempted': False,
                  'original_scene_preflight_passed': original.get('passed'),
                  'source_name': None, 'lineage': None, 'error': None}
        _PREPARATION = record
        if not original['passed']:
            return original
        # Never mutate self.scene_preflight or its raw file: its object inventory
        # must continue to prove the original, unexpanded scene preflight.
        baseline = original['snapshot']
        source = clone_source(baseline['objects'])
        try:
            if source is None:
                natural = natural_identity_sets(baseline['objects'])
                record.update(status='natural_fallback' if natural else 'no_candidate_structure',
                              passed=bool(natural), natural_identity_count=len(natural))
                return original if natural else {**original, 'passed': False}
            record['source_name'] = source['name']
            token = select_clone_token(self, baseline, source, record)
            if token is None:
                record['status'] = 'clone_no_native_proposal'
                return {**original, 'passed': False}
            record.update(status='clone_setup_started', clone_attempted=True)
            prepared = prepare_clone_baseline(self, baseline, source['name'], token)
            record.update(status='clone_ready' if prepared['passed'] else 'clone_setup_failed',
                          passed=prepared['passed'] is True, lineage=copy.deepcopy(prepared.get('lineage')),
                          baseline_validation_reference=copy.deepcopy(prepared.get('record_reference')),
                          clone_record=copy.deepcopy(prepared['record']))
            if not prepared['passed']:
                return {**original, 'passed': False}
            return {**original, 'snapshot': prepared['snapshot'],
                    'clone_baseline_reference': prepared['record_reference']}
        except BaseException as error:
            record.update(status='clone_setup_failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            reference = self.store.save_json('clone_selection.json', record)
            record['record_reference'] = reference


def cloned_identity_sets(objects):
    identities = natural_identity_sets(objects)
    record = _PREPARATION or {}
    if record.get('status') == 'natural_fallback':
        return identities
    lineage = record.get('lineage') or {}
    if record.get('status') != 'clone_ready' or lineage.get('passed') is not True:
        return []
    candidates = set(lineage['candidate_names'])
    return [identity for identity in identities if set(identity['candidate_names']) == candidates
            and identity['anchor_name'] not in candidates]


def wait_replay(config, snapshot, actions, target, identities, directory):
    child_config = {**config, 'enable_candidate_clone': True, 'snapshot': snapshot,
        'actions': actions, 'target_name': target, **identities,
        'instruction': f"Select the {identities['target_type']} closest to the {identities['anchor_type']}.",
        'raw_dir': str(directory)}
    target_worker = clone_replay_worker if snapshot.get('clone_context') is not None else engine.replay_worker
    child = Worker(target_worker, child_config)
    result = None
    try:
        while result is None:
            child.receive()
            if child.result is not None:
                result = child.result
            elif child.error:
                result = {'status': 'worker_failed', 'error': child.error, 'success': False}
            elif time.time() >= config['deadline'] - 10:
                raise TimeoutError('Witness replay deadline')
            else:
                time.sleep(.05)
    finally:
        try:
            cleanup = child.stop()
        except BaseException as error:
            if _REPLAY_CLEANUP is not None:
                _REPLAY_CLEANUP.append(f'witness cleanup raised {type(error).__name__}: {error}')
            raise
        if _REPLAY_CLEANUP is not None:
            _REPLAY_CLEANUP.extend(cleanup)
    result.setdefault('cleanup_errors', []).extend(cleanup)
    if cleanup:
        result['success'] = False
    return result


def references(value):
    if isinstance(value, dict):
        if isinstance(value.get('file'), str) and isinstance(value.get('sha256'), str):
            yield {'file': value['file'], 'sha256': value['sha256']}
        for item in value.values():
            yield from references(item)
    elif isinstance(value, list):
        for item in value:
            yield from references(item)


class JsonProofArchive:
    """Per-run content-addressed JSON copies, with a separate index per pair."""
    MAX_FILE_BYTES = 32 * 1024 * 1024
    MAX_TOTAL_BYTES = 128 * 1024 * 1024
    MAX_PAIR_FILES = 1024

    def __init__(self, config):
        self.root = base.ROOT.resolve()
        self.scratch = Path(config['runtime']['scratch']).resolve()
        self.directory = self.root / 'data/world_pairs/v2' / config['name'] / 'clone_evidence'
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / 'objects').mkdir()
        self.total_bytes = 0
        self.blobs = {}

    def archive(self, pair_id, seeds):
        if Path(pair_id).name != pair_id or pair_id in ('', '.', '..'):
            raise ValueError('Invalid clone proof pair ID')
        pending = list(references(seeds))
        entries, skipped, seen = [], [], {}
        while pending:
            ref = pending.pop()
            source = Path(ref['file']).resolve()
            if not source.is_relative_to(self.scratch) or source == self.scratch:
                raise ValueError('Clone proof source must be inside allocated scratch')
            digest = ref['sha256']
            key = str(source)
            if key in seen:
                if seen[key] != digest:
                    raise ValueError('One clone proof source has conflicting hashes')
                continue
            seen[key] = digest
            if source.suffix.lower() != '.json':
                skipped.append({'source_file': key, 'sha256': digest, 'reason': 'non_JSON; final pair RGB/masks archived separately'})
                continue
            if len(entries) >= self.MAX_PAIR_FILES or source.stat().st_size > self.MAX_FILE_BYTES:
                raise ValueError('Clone JSON proof archive exceeded declared size/file limit')
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError('Clone JSON proof source hash changed')
            value = json.loads(data)
            pending.extend(references(value))
            destination = self.directory / 'objects' / (digest + '.json')
            if digest not in self.blobs:
                if self.total_bytes + len(data) > self.MAX_TOTAL_BYTES:
                    raise ValueError('Clone JSON proof archive exceeded total byte limit')
                with destination.open('xb') as stream:
                    stream.write(data)
                self.total_bytes += len(data)
                self.blobs[digest] = destination
            elif base.sha(destination) != digest:
                raise ValueError('Portable clone proof blob changed')
            entries.append({'source_file': key, 'file': str(destination.relative_to(self.root)), 'sha256': digest})
        entries.sort(key=lambda entry: entry['source_file'])
        index = {'schema_version': 1, 'pair_id': pair_id, 'seeds': copy.deepcopy(seeds),
                 'entries': entries, 'excluded_non_json_references': sorted(skipped, key=lambda entry: entry['source_file']),
                 'scope': 'complete referenced JSON proof closure; final images/masks are in portable_files',
                 'max_file_bytes': self.MAX_FILE_BYTES, 'max_run_bytes': self.MAX_TOTAL_BYTES}
        path = self.directory / (pair_id + '_index.json')
        content = (json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        if len(content) > self.MAX_FILE_BYTES or self.total_bytes + len(content) > self.MAX_TOTAL_BYTES:
            raise ValueError('Clone proof index exceeded the declared archive byte limit')
        with path.open('xb') as stream:
            stream.write(content)
        self.total_bytes += len(content)
        return {'file': str(path.relative_to(self.root)), 'sha256': base.sha(path)}, entries


def export_pair(original_export, archive, config, row, worlds, prefix, after, witnesses, constructions):
    contexts = [world.get('clone_context') for world in worlds]
    row['enable_candidate_clone'] = True
    row['clone_selection'] = copy.deepcopy(_PREPARATION)
    if all(context is None for context in contexts):
        if (_PREPARATION or {}).get('status') != 'natural_fallback':
            raise ValueError('Missing clone context without the declared natural fallback')
        row['candidate_origin'] = 'natural_fallback'
        return original_export(config, row, worlds, prefix, after, witnesses, constructions)
    if any(not isinstance(context, dict) or context.get('setup_passed') is not True for context in contexts):
        raise ValueError('Both worlds require the validated common clone context')
    if contexts[0] != contexts[1]:
        raise ValueError('The two worlds do not share the same clone context')
    context = contexts[0]
    check = clone_pair_check(context['lineage'], worlds[0]['objects'], worlds[1]['objects'],
                             worlds[0]['poses'], constructions[1]['requested_poses'], row['identities']['anchor_name'])
    row.update(candidate_origin='common_baseline_clone', clone_pair_check=check)
    if not check['passed']:
        raise ValueError('Clone pair inventory/anchor-only transition failed: ' + repr(check['reasons']))
    restored = []
    for witness in witnesses:
        ref = (witness.get('restoration') or {}).get('restored_snapshot_reference')
        if not ref or base.sha(ref['file']) != ref['sha256']:
            raise ValueError('Fresh cloned replay lacks unchanged restored snapshot evidence')
        snapshot = json.loads(Path(ref['file']).read_text())
        measured = (witness.get('restoration') or {}).get('clone_recreation') or {}
        if measured.get('passed') is not True or measured.get('already_recreated') is True or not snapshot.get('clone_recreation_reference'):
            raise ValueError('Fresh replay must independently create and verify the actual clone lineage')
        restored.append({'restored_snapshot_reference': ref,
                         'clone_recreation_reference': snapshot['clone_recreation_reference']})
    seeds = {'context': context, 'original_selection': _PREPARATION,
             'world_snapshots': [world['reference'] for world in worlds], 'fresh_replays': restored}
    index, entries = archive.archive(row['pair_id'], seeds)
    row['clone_evidence'] = {'schema_version': 1, 'context': copy.deepcopy(context), 'index_reference': index,
        'scope': 'JSON proof closure; final RGB/masks remain portable_files'}
    row['clone_evidence_files'] = entries
    return original_export(config, row, worlds, prefix, after, witnesses, constructions)


def generator_worker(config, connection):
    """Rebind only this new spawned process; restore every binding on all exits."""
    global _PREPARATION, _REPLAY_CLEANUP
    if config.get('enable_candidate_clone') is not True:
        raise ValueError('Second-round clone generation requires explicit opt-in')
    previous = (engine.V2Simulator, base.identity_sets, base.wait_replay, base.export_pair, _PREPARATION, _REPLAY_CLEANUP)
    archive = None

    def export_adapter(*args):
        nonlocal archive
        if archive is None:
            archive = JsonProofArchive(config)
        return export_pair(previous[3], archive, *args)

    try:
        _PREPARATION = None
        _REPLAY_CLEANUP = []
        engine.V2Simulator = PreparationSimulator
        base.identity_sets = cloned_identity_sets
        base.wait_replay = wait_replay
        base.export_pair = export_adapter
        result = base.generator_worker(config, connection)
        result['enable_candidate_clone'] = True
        result['clone_preparation'] = copy.deepcopy(_PREPARATION)
        result.setdefault('cleanup_errors', []).extend(_REPLAY_CLEANUP)
        if _PREPARATION and _PREPARATION.get('original_scene_preflight_passed') is True and not _PREPARATION.get('passed'):
            result['scene_excluded'] = {'scene': config['scene'], 'reason': _PREPARATION['status'],
                'evidence': copy.deepcopy(_PREPARATION.get('record_reference')),
                'original_scene_preflight_passed': True, 'requires_new_controller': True}
            result['scene_exclusion_count'] = 1
            if not result.get('black_frame_detected') and result.get('status') in ('completed_search', 'completed_quota'):
                result['status'] = 'clone_setup_failed'
        return result
    finally:
        engine.V2Simulator, base.identity_sets, base.wait_replay, base.export_pair, _PREPARATION, _REPLAY_CLEANUP = previous


def main():
    original_worker, original_writer = base.Worker, base.write_result

    def worker_adapter(target, config):
        if target is not base.generator_worker:
            raise ValueError('Unexpected parent worker target in clone entry point')
        enable_config(config)
        return original_worker(generator_worker, config)

    def write_adapter(name, metrics, config):
        enable_config(config)
        metrics['enable_candidate_clone'] = True
        return original_writer(name, metrics, config)

    try:
        base.Worker, base.write_result = worker_adapter, write_adapter
        return base.main()
    finally:
        base.Worker, base.write_result = original_worker, original_writer


if __name__ == '__main__':
    raise SystemExit(main())
