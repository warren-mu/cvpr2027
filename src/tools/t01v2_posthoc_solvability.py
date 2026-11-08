"""CPU-only positive supplement-4 certificates from already recorded witnesses.

This does not reconstruct an action catalogue, run a simulator, or alter a
source. A visible first action can prove existence; an invisible/missing first
action cannot prove that no other action sees the anchor.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re

from src.tools import merge_pair_v2 as merge

VERSION = 'legacy_selected_action_positive_v1'
MODE = 'legacy_selected_action_positive'
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024


def certify_selected_replay(row, worlds, replays, read_json, read_blob):
    """Require authenticated world/action/capture inputs supplied by the caller."""
    import numpy as np
    from src.envs.pair_v2_policy import restore_state_check
    from src.envs.pair_v2_solvable_export import capture_observation
    from src.envs.pair_simulator import path_complete, point
    if not row['kind'].startswith('look') or len(worlds) != 2 or len(replays) != 2:
        raise ValueError('Positive posthoc proof is only for a two-world Look pair')
    first = [plan[0] for plan in row['witness_actions']]
    if first[0] != first[1] or re.fullmatch(r'(turn_left|turn_right|move [1-9][0-9]*)', first[0]) is None:
        raise ValueError('Require the same existing public first observation in both worlds')
    observations, proofs = [], []
    for index, (world, replay) in enumerate(zip(worlds, replays)):
        if (replay.get('status') != 'completed' or replay.get('cleanup_errors') != []
                or replay.get('error') is not None or replay.get('planned_actions') != row['witness_actions'][index]
                or replay.get('executed_actions') != replay.get('planned_actions')
                or replay.get('process_id') == world.get('worker_pid')
                or replay.get('unity_pid') == world.get('unity_pid')):
            raise ValueError('Require an independently completed, clean existing witness')
        restored = read_json(replay['restoration']['restored_snapshot_reference'])
        checked = restore_state_check(world['objects'], restored['objects'], world['agent'], restored['agent'])
        if checked['restore_deterministic'] is not True:
            raise ValueError('First observation was not made after the required world restore')
        action_ref = replay['actions'][0]['action_reference']
        native = read_json(action_ref)
        capture = replay['captures'][1]
        observation = capture_observation(capture, read_blob, row['identities'], row['instruction'])
        before = native['before_metadata']
        before_check = restore_state_check(world['objects'], before['objects'], world['agent'], before['agent'])
        if (not before_check['restore_deterministic'] or native.get('exception') is not None
                or native.get('capture') != capture or native.get('after_metadata') != observation['metadata']
                or observation['metadata'].get('lastActionSuccess') is not True
                or native.get('index') != capture.get('action_index')
                or native.get('action') != capture.get('source_action')
                or replay['actions'][0].get('success') is not True):
            raise ValueError('First-action metadata/mask capture is not its successful native output')
        query_ref = None
        if first[0].startswith('move '):
            arguments = native['arguments']
            target, start = arguments['position'], world['agent']['position']
            expected = {'position': target, 'rotation': restored['agent']['rotation'],
                'horizon': restored['agent']['horizon'], 'standing': restored['agent']['standing'], 'forceAction': False}
            displacement = math.dist([start[k] for k in 'xyz'], [target[k] for k in 'xyz'])
            if (native['action'] != 'TeleportFull' or arguments != expected or not 0 < displacement <= 1.5
                    or abs(target['y'] - start['y']) > .01
                    or min(abs(target['x'] - start['x']), abs(target['z'] - start['z'])) > .01):
                raise ValueError('Selected move is not the original adjacent observation motion')
            # The exact frozen replay performs a native path query immediately
            # before each move. Its final index supplies the original file SHA.
            raw_index = read_json(replay['raw_evidence'])
            expected_path = Path(action_ref['file']).with_name(f"action_{native['index'] - 1:06d}.json")
            entries = [ref for ref in raw_index['files'] if Path(ref['file']) == expected_path]
            if len(entries) != 1:
                raise ValueError('Selected move lacks its unique prior native path record')
            query_ref = entries[0]
            query = read_json(query_ref)
            if (query.get('action') != 'GetShortestPathToPoint' or query.get('exception') is not None
                    or query.get('arguments') != {'position': start, 'target': target, 'allowedError': .05}
                    or query.get('after_metadata') != before or before.get('lastActionSuccess') is not True
                    or not path_complete(before.get('actionReturn'), point(start), point(target), .05)
                    or not restore_state_check(world['objects'], query['before_metadata']['objects'],
                                               world['agent'], query['before_metadata']['agent'])['restore_deterministic']):
                raise ValueError('Selected move lacks its restored-world complete native path')
        else:
            expected = 'RotateLeft' if first[0] == 'turn_left' else 'RotateRight'
            if native['action'] != expected or native['arguments'] != {'degrees': 45}:
                raise ValueError('Selected turn changed the original 45-degree observation action')
        count = int(observation['masks_by_name'].get(row['identities']['anchor_name'], np.zeros((255, 255), bool)).sum())
        proofs.append({'world_index': index, 'action': first[index], 'anchor_pixels': count,
            'visible': count >= 250, 'action_reference': action_ref, 'capture_reference': copy.deepcopy(capture),
            'restored_snapshot_reference': replay['restoration']['restored_snapshot_reference'],
            'native_path_reference': query_ref, 'restore_state': checked})
        observations.append(observation)
    if first[0].startswith('move '):
        positions = [read_json(proof['action_reference'])['arguments']['position'] for proof in proofs]
        if positions[0] != positions[1]:
            raise ValueError('The same public first move reached different world destinations')
    visible_both = all(item['visible'] for item in proofs)
    # Old replay records bind motion and native feasibility, but do not retain
    # the construction-time reachable list that gave a move its public index.
    # Do not substitute proximity for that missing action-domain certificate.
    domain_verified = not first[0].startswith('move ')
    positive = visible_both and domain_verified
    return {'version': VERSION, 'mode': MODE, 'status': 'positive' if positive else 'unknown',
        'solvability': True if positive else None, 'complete_catalog': False,
        'other_actions': 'unknown_not_enumerated', 'worlds': proofs,
        'anchor_visible_actions_w0': [first[0]] if proofs[0]['visible'] else [],
        'anchor_visible_actions_w1': [first[1]] if proofs[1]['visible'] else [],
        'selected_action_observed_visible_both': visible_both,
        'selected_action_domain_verified': domain_verified,
        'reasons': ([] if positive else
            (['legacy_move_domain_mapping_unavailable'] if not domain_verified else
             ['selected_action_does_not_certify_both_worlds; other_actions_unknown'])),
        'measurement_scope': 'existing fresh-witness first action, not a new rendering or all-action search',
        'public_move_index_scope': 'native motion/path rechecked; original indexed adjacency list unavailable, so moves remain unknown',
        'full_metrics_spec_compliance': None}


class Archive:
    def __init__(self, directory, raw_root, root, budget):
        self.directory, self.raw_root, self.root = directory, Path(raw_root).resolve(), root
        self.files, self.used, self.budget = {}, set(), budget

    def read(self, reference):
        path = merge.reference_path(reference, root=self.root, within=self.raw_root)
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError('Posthoc evidence file exceeds the declared byte cap')
        data = path.read_bytes()
        key = (str(path), reference['sha256'])
        self.used.add(key)
        if key not in self.files:
            target = self.directory / (reference['sha256'] + path.suffix)
            if not target.exists():
                if self.budget['bytes'] + len(data) > MAX_ARCHIVE_BYTES:
                    raise ValueError('Posthoc evidence archive exceeded its declared byte cap')
                with target.open('xb') as handle:
                    handle.write(data)
                self.budget['bytes'] += len(data)
            if hashlib.sha256(target.read_bytes()).hexdigest() != reference['sha256']:
                raise ValueError('Posthoc archive SHA mismatch')
            self.files[key] = {'source_file': str(path), 'file': str(target.relative_to(self.root)),
                               'sha256': reference['sha256']}
        return data

    def json(self, reference):
        return json.loads(self.read(reference))


def verified_passed_rows(data):
    """Authenticate the complete journal, including all rejected attempts."""
    cfg, metrics = data['config'], data['metrics']
    eligible = merge.source_eligibility(data)
    ref = merge.journal_reference(data, eligible)
    if ref is None:
        return eligible, None, []
    path = merge.reference_path(ref, within=merge.ROOT / 'data/world_pairs/v2')
    rows, seen, passed = [], set(), {}
    for item in metrics.get('passed_pair_files') or []:
        target = merge.reference_path(item, within=merge.ROOT / 'data/world_pairs/v2' / cfg['name'])
        if target in passed:
            raise ValueError('Duplicate passed pair in source')
        passed[target] = item
    matched, selected = set(), []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row['scene'] != cfg['scene'] or row['pair_id'] in seen:
            raise ValueError('Duplicate or wrong-scene journal row')
        seen.add(row['pair_id'])
        raw = merge.read_reference(row['raw_attempt'], within=cfg['raw_evidence_dir'])
        if any(raw.get(key) != row.get(key) for key in ('pair_id', 'scene', 'kind', 'status',
                'discard_reasons', 'generation_seconds', 'pair_file', 'pair_sha256')):
            raise ValueError('Journal differs from authenticated raw attempt')
        rows.append(row)
        if row['status'] == 'passed':
            pair_path = (merge.ROOT / row['pair_file']).resolve()
            if pair_path not in passed or passed[pair_path]['sha256'] != row.get('pair_sha256'):
                raise ValueError('Passed journal/file references disagree')
            matched.add(pair_path)
            selected.append((row, passed[pair_path], pair_path))
    counts = {kind: sum(row.get('kind') == kind and row['status'] == 'passed' for row in rows) for kind in merge.KINDS}
    if (matched != set(passed)
            or (metrics.get('attempts') is not None and metrics['attempts'] != len(rows))
            or (metrics.get('passed_by_class') is not None and metrics['passed_by_class'] != counts)
            or (eligible['eligible'] and any(metrics.get(key) is None for key in ('attempts', 'passed_by_class')))):
        raise ValueError('Whole-journal attempt/class counts differ from recorded source')
    return eligible, ref, selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source', nargs='+', required=True)
    args = parser.parse_args(argv)
    if re.fullmatch(r'[A-Za-z0-9_]+', args.name) is None:
        parser.error('Use a new alphanumeric run name')
    output = merge.ROOT / 'results' / (args.name + '.json')
    evidence = merge.ROOT / 'logs/runs' / args.name / 'evidence'
    if output.exists() or evidence.parent.exists():
        raise FileExistsError('Existing posthoc evidence is preserved; use a new name')
    evidence.mkdir(parents=True, exist_ok=False)
    metrics = {'kind': 't01v2_posthoc_solvability', 'status': 'failed', 'sources': [],
        'source_runs': [], 'pairs': [], 'errors': [], 'acceptance_passed': None,
        'full_metrics_spec_compliance': None, 'complete_catalog': False}
    config = {'name': args.name, 'requested_sources': args.source, 'version': VERSION,
        'scope': 'positive existential supplement-4(a) check of existing selected first witness actions only',
        'missing_or_invisible': 'unknown; never a proof of unsolvability',
        'source_artifacts_modified': False, 'simulator_actions_executed': False,
        'archive_max_file_bytes': MAX_FILE_BYTES, 'archive_max_unique_bytes': MAX_ARCHIVE_BYTES}
    seen, budget = set(), {'bytes': 0}
    for source in args.source:
        try:
            path = (merge.ROOT / source).resolve()
            if path in seen:
                raise ValueError('Duplicate explicit source')
            seen.add(path)
            if not path.exists():
                metrics['source_runs'].append({'source_file': source, 'status': 'not_yet_completed', 'pair_count': None})
                continue
            data, source_ref = merge.load_source(path)
            metrics['sources'].append(source_ref)
            cfg, measured = data['config'], data['metrics']
            eligibility, journal_ref, rows = verified_passed_rows(data)
            metrics['source_runs'].append({'source': source_ref, **eligibility, 'journal': journal_ref,
                'source_status': measured['status'], 'source_exit_code': data['_meta']['exit_code'],
                'reported_passed_by_class': measured.get('passed_by_class'), 'reported_rejections': measured.get('discard_reason_counts')})
            if not eligibility['eligible']:
                continue
            ledger = None
            if cfg.get('search_version') is not None and rows:
                inventory = merge.search_source_inventory(measured, lambda ref: merge.read_reference(ref, within=cfg['raw_evidence_dir']))
                ledger = merge.verify_search_ledger(measured, inventory)
            archive = Archive(evidence, cfg['raw_evidence_dir'], merge.ROOT, budget)
            for compact, pair_ref, pair_path in rows:
                pair = merge.read_reference(pair_ref, within=pair_path.parent)
                if any(pair.get(key) != compact.get(key) for key in ('pair_id', 'scene', 'kind', 'status', 'discard_reasons')):
                    raise ValueError('Pair and journal identities differ')
                if not pair['kind'].startswith('look') or cfg.get('solvability_version') is not None:
                    continue
                entry = {'pair_id': pair['pair_id'], 'scene': pair['scene'], 'kind': pair['kind'],
                    'pair_reference': pair_ref, 'source_reference': source_ref, 'partial_source': eligibility['partial_source']}
                metrics['pairs'].append(entry)
                archive.used.clear()
                try:
                    checked = merge.check_pair(pair, pair_path, raw_root=cfg['raw_evidence_dir'],
                        clone_enabled=cfg.get('enable_candidate_clone') is True, preflight_version=cfg.get('preflight_version'),
                        evidence_version=cfg.get('evidence_version'), search_version=cfg.get('search_version'),
                        search_policy=cfg.get('search_policy'), search_clone_source=(ledger or {}).get('successful_source_name'))
                    entry['independent_construction_verification'] = checked
                    refs = {Path(ref['file']).name: ref for ref in pair['portable_files']}
                    worlds = [merge.read_reference(refs[f'snapshot_w{i}.json'], within=pair_path.parent) for i in range(2)]
                    replays = [merge.read_reference(refs[f'witness_w{i}.json'], within=pair_path.parent) for i in range(2)]
                    entry['certificate'] = certify_selected_replay(pair, worlds, replays, archive.json, archive.read)
                except Exception as error:
                    entry['certificate'] = {'version': VERSION, 'mode': MODE, 'status': 'unknown', 'solvability': None,
                        'complete_catalog': False, 'other_actions': 'unknown_not_enumerated',
                        'error': f'{type(error).__name__}: {error}'}
                    metrics['errors'].append({'source': source, 'pair_id': pair['pair_id'], 'error': entry['certificate']['error']})
                finally:
                    entry['proof_files'] = [value for key, value in archive.files.items() if key in archive.used]
        except Exception as error:
            metrics['errors'].append({'source': source, 'error': f'{type(error).__name__}: {error}'})
    metrics['certificate_status_counts'] = dict(Counter(row['certificate']['status'] for row in metrics['pairs']))
    metrics['archived_unique_bytes'] = budget['bytes']
    metrics['status'] = 'reported' if not metrics['errors'] else 'incomplete_evidence'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({'metrics': metrics, 'config': config}, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False))
    return int(bool(metrics['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
