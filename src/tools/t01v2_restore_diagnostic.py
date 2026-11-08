"""Read authenticated ended-shard restoration failures without changing rules.

Run through run_logged. This is a diagnostic of saved metadata, not a new
simulation, acceptance decision, tolerance change, or repair of old results.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

from src.envs import pair_v2_simulator as engine
from src.tools import merge_pair_v2 as merge
from src.tools.t01v2_posthoc_solvability import verified_passed_rows

MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_SETUP_RECORDS = 256
MISSING = object()


def describe(value):
    if value is MISSING:
        return {'present': False, 'value': None, 'type': None, 'repr': None, 'float_hex': None}
    return {'present': True, 'value': value, 'type': type(value).__name__, 'repr': repr(value),
            'float_hex': value.hex() if type(value) is float and math.isfinite(value) else None}


def compare_scalar(requested=MISSING, realized=MISSING):
    result = {'requested': describe(requested), 'realized': describe(realized),
              'numeric_delta': None, 'numeric_equal': None, 'encoded_equal': None}
    if requested is MISSING or realized is MISSING:
        result['difference'] = 'unknown_missing_field'
        return result
    exact = engine.encoded(requested) == engine.encoded(realized)
    result['encoded_equal'] = exact
    numeric = all(type(v) in (int, float) and math.isfinite(v) for v in (requested, realized))
    if numeric:
        result.update(numeric_delta=realized - requested, numeric_equal=requested == realized)
    if exact:
        kind = 'identical'
    elif numeric and requested == realized == 0 and math.copysign(1, requested) != math.copysign(1, realized):
        kind = 'signed_zero_representation'
    elif numeric and requested == realized:
        kind = 'numeric_equal_different_representation'
    elif numeric:
        kind = 'numeric_difference'
    else:
        kind = 'non_numeric_difference'
    result['difference'] = kind
    return result


def fields_at(value, path):
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            return MISSING
        value = value[key]
    return value


def field_table(requested, realized, paths):
    return {path: compare_scalar(fields_at(requested, path), fields_at(realized, path)) for path in paths}


POSE_PATHS = tuple(f'{part}.{axis}' for part in ('position', 'rotation') for axis in 'xyz')
AGENT_PATHS = POSE_PATHS + ('horizon', 'standing')
CAMERA_PATHS = tuple('cameraPosition.' + axis for axis in 'xyz') + ('fov', 'screenWidth', 'screenHeight')


def compare_snapshots(baseline, restored):
    """Exact parsed-value diagnostics; the old encoded-agent rule stays intact."""
    before, after = engine.objects_by_name(baseline['objects']), engine.objects_by_name(restored['objects'])
    editable = engine.editable_names(baseline['objects'])
    object_rows = []
    for name in sorted(editable):
        old, new = before[name], after.get(name, {})
        axes = field_table(old, new, POSE_PATHS)
        known = all(axes['position.' + axis]['numeric_delta'] is not None for axis in 'xyz')
        displacement = engine.distance(old['position'], new['position']) if known else None
        object_rows.append({'name': name, 'object_type': old.get('objectType'), 'axes': axes,
            'position_delta_m': displacement,
            'within_original_1cm_rule': None if displacement is None else displacement <= .01})
    agent = field_table(baseline.get('agent', {}), restored.get('agent', {}), AGENT_PATHS)
    camera = field_table(baseline.get('camera', {}), restored.get('camera', {}), CAMERA_PATHS)
    changes = {key: value['difference'] for key, value in agent.items() if value['difference'] != 'identical'}
    missing = (any(agent[key]['numeric_delta'] is None for key in AGENT_PATHS[:-1])
               or any(type(fields_at(snap.get('agent', {}), 'standing')) is not bool for snap in (baseline, restored)))
    measured, measurement_error = None, None
    if not missing and all(item['position_delta_m'] is not None for item in object_rows):
        try:
            measured = engine.restoration_state(baseline, restored['objects'], restored['agent'])
        except (KeyError, TypeError, ValueError) as error:
            measurement_error = f'{type(error).__name__}: {error}'
    positions = [item['position_delta_m'] for item in object_rows if item['position_delta_m'] is not None]
    return {'agent_axes': agent, 'camera_axes': camera, 'agent_difference_fields': changes,
        'agent_difference_only_signed_zero': None if missing else bool(changes) and set(changes.values()) == {'signed_zero_representation'},
        'agent_has_numeric_difference': None if missing else 'numeric_difference' in changes.values(),
        'all_object_inventory_exact': set(before) == set(after),
        'missing_object_names': sorted(set(before) - set(after)), 'added_object_names': sorted(set(after) - set(before)),
        'object_identity_changes': [{'name': name,
            'requested_object_id': before[name].get('objectId'), 'realized_object_id': after[name].get('objectId'),
            'requested_object_type': before[name].get('objectType'), 'realized_object_type': after[name].get('objectType')}
            for name in sorted(set(before) & set(after))
            if any(before[name].get(key) != after[name].get(key) for key in ('objectId', 'objectType'))],
        'identity_scope': 'stable-name inventory and observed ID/type changes; no ID stability rule added',
        'movable_or_pickupable_positions': object_rows,
        'maximum_observed_movable_position_delta_m': max(positions) if positions else None,
        'movable_position_violations': [item['name'] for item in object_rows if item['within_original_1cm_rule'] is False],
        'unknown_movable_positions': [item['name'] for item in object_rows if item['position_delta_m'] is None],
        'unchanged_restoration_rule_recomputed': measured,
        'restoration_recomputation_error': measurement_error,
        'scope': 'reported state differences only; no attribution to hidden physics or renderer internals'}


class Evidence:
    def __init__(self, raw_root, archive, *, root=merge.ROOT):
        self.raw_root, self.archive, self.root = Path(raw_root).resolve(), Path(archive), Path(root)
        self.references, self.saved, self.bytes = [], set(), 0

    def read(self, ref):
        candidate = (self.root / ref['file']).resolve()
        if (not candidate.is_relative_to(self.raw_root) or candidate.suffix != '.json'
                or candidate.stat().st_size > MAX_FILE_BYTES):
            raise ValueError('Diagnostic JSON is outside the source raw root or exceeds its bound')
        source = merge.reference_path(ref, root=self.root, within=self.raw_root)
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != ref['sha256']:
            raise ValueError('Raw diagnostic JSON changed during reading')
        value = json.loads(content)
        target = self.archive / (ref['sha256'] + '.json')
        if ref['sha256'] not in self.saved:
            if self.bytes + len(content) > MAX_ARCHIVE_BYTES:
                raise ValueError('Bounded diagnostic archive is full')
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError('Existing diagnostic archive blob differs')
            else:
                with target.open('xb') as stream:
                    stream.write(content)
            self.bytes += len(content)
            self.saved.add(ref['sha256'])
        row = {'source_file': str(source), 'file': str(target.relative_to(self.root)), 'sha256': ref['sha256']}
        if row not in self.references:
            self.references.append(row)
        return value


def inspect_setup(record, scene, read):
    restoration = record.get('restoration')
    if not isinstance(restoration, dict):
        raise ValueError('Proposal setup has no recorded restoration measurement')
    baseline = read(record['baseline_reference'])
    restored = read(restoration['restored_snapshot_reference'])
    if (baseline.get('scene') != scene or restored.get('scene') != scene
            or baseline.get('worker_pid') != restored.get('worker_pid')
            or baseline.get('unity_pid') != restored.get('unity_pid')
            or type(baseline.get('worker_pid')) is not int or type(baseline.get('unity_pid')) is not int):
        raise ValueError('Source and restored snapshots do not bind the same scene/process')
    result = compare_snapshots(baseline, restored)
    errors = []
    for key, snapshot_value in (('agent_requested', baseline.get('agent')), ('agent_realized', restored.get('agent'))):
        if engine.encoded(restoration.get(key)) != engine.encoded(snapshot_value):
            errors.append('Recorded restoration ' + key + ' differs from authenticated snapshot')
    native = read(restored['last_action_reference'])
    after = native.get('after_metadata') or {}
    actual_agent = {'position': after.get('agent', {}).get('position'),
        'rotation': after.get('agent', {}).get('rotation'),
        'horizon': after.get('agent', {}).get('cameraHorizon'),
        'standing': after.get('agent', {}).get('isStanding')}
    expected = copy.deepcopy(baseline.get('agent_setup_command'))
    if expected is None:
        expected = {**copy.deepcopy(baseline['agent']), 'forceAction': False}
    if (native.get('action') != 'TeleportFull' or native.get('exception') is not None
            or after.get('lastActionSuccess') is not True
            or engine.encoded(native.get('arguments')) != engine.encoded(expected)
            or engine.encoded(native.get('arguments')) != engine.encoded(restoration.get('executed_agent_setup_arguments'))
            or engine.encoded(after.get('objects')) != engine.encoded(restored['objects'])
            or engine.encoded(actual_agent) != engine.encoded(restored.get('agent'))
            or native.get('capture') != restored.get('capture_reference')):
        errors.append('Restored snapshot is not bound to its recorded successful canonical TeleportFull')
    recomputed = result['unchanged_restoration_rule_recomputed']
    keys = ('restore_deterministic', 'agent_pose_exact', 'restored_positions_within_1cm', 'pose_measurements')
    if recomputed is not None:
        for key in keys:
            if engine.encoded(restoration.get(key)) != engine.encoded(recomputed[key]):
                errors.append('Recorded restoration measurement differs on ' + key)
    else:
        errors.append('Unknown: incomplete state fields prevent recomputing the original restoration rule')
    result.update(recorded_restoration={key: copy.deepcopy(restoration.get(key)) for key in keys},
        canonical_agent_setup_command=baseline.get('agent_setup_command'),
        executed_agent_setup_arguments=native.get('arguments'),
        recorded_rgb_diagnostics={key: restoration.get(key) for key in ('restore_max_abs_diff', 'restore_rgb_exact')},
        rgb_scope='copied existing measurement only; no images rendered or RGB numbers recomputed',
        baseline_reference=record['baseline_reference'], restored_snapshot_reference=restoration['restored_snapshot_reference'],
        native_action_reference=restored['last_action_reference'], binding_errors=errors)
    return result


def scene_diagnostic(data, evidence):
    cfg, measured = data['config'], data['metrics']
    result = {'scene': cfg['scene'], 'source_status': measured.get('status'),
        'source_exit_code': data['_meta']['exit_code'], 'source_error': measured.get('error'),
        'source_cleanup_errors': measured.get('cleanup_errors'), 'journal_audit': None,
        'proposal_setup_failure': None, 'errors': []}
    try:
        eligibility, reference, _ = verified_passed_rows(data)
        result['journal_audit'] = {'passed': True if reference else None,
            'reference': reference, 'source_eligibility': eligibility,
            'scope': 'existing whole journal/raw-attempt agreement; not pair acceptance'}
    except Exception as error:
        result['journal_audit'] = {'passed': False, 'error': f'{type(error).__name__}: {error}'}
        result['errors'].append('Source journal audit failed; original source is not repaired')
    visits = measured.get('controller_visits')
    if not isinstance(visits, list):
        result['errors'].append('Unknown: source does not bind Controller visits and final raw indexes')
        return result
    scanned = 0
    for visit in visits:
        try:
            index_ref = visit.get('raw_index')
            if not isinstance(index_ref, dict):
                raise ValueError('No hashed final raw index; do not trust an unbound directory scan')
            index = evidence.read(index_ref)
            refs = [ref for ref in index['files'] if re.fullmatch(r'proposal_setup_[0-9]+\.json', Path(ref['file']).name)]
            refs.sort(key=lambda ref: int(re.search(r'([0-9]+)\.json$', ref['file']).group(1)))
            for ref in refs:
                if Path(ref['file']).resolve().parent != Path(index_ref['file']).resolve().parent:
                    raise ValueError('Proposal setup escaped its authenticated Controller store')
                scanned += 1
                if scanned > MAX_SETUP_RECORDS:
                    raise ValueError('Bounded setup-record scan exhausted')
                record = evidence.read(ref)
                if (record.get('restoration') or {}).get('restore_deterministic') is False:
                    result['proposal_setup_failure'] = {'controller_id': visit.get('controller_id'),
                        'setup_reference': ref, 'index_reference': index_ref,
                        **inspect_setup(record, cfg['scene'], evidence.read)}
                    result['errors'].extend(result['proposal_setup_failure']['binding_errors'])
                    return result  # first actual failed setup; no large raw traversal
        except Exception as error:
            result['errors'].append(f'{visit.get("controller_id")}: {type(error).__name__}: {error}')
    result['errors'].append('Unknown: no retained authenticated failed proposal_setup record was found')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source', nargs='+', required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r't01v2_restore_diag_[A-Za-z0-9_]+', args.name):
        parser.error('Use a unique t01v2_restore_diag_* name')
    output = merge.ROOT / 'results' / (args.name + '.json')
    archive = merge.ROOT / 'logs/runs' / args.name / 'evidence'
    if output.exists() or archive.parent.exists():
        raise FileExistsError('Preserve prior diagnostics; use a new run name')
    archive.mkdir(parents=True)
    metrics = {'kind': 't01v2_restore_diagnostic', 'status': 'diagnostic_completed',
        'scenes': [], 'source_errors': [], 'acceptance_changed': False, 'restoration_rule_changed': False}
    sources, seen = [], set()
    for supplied in args.source:
        try:
            path = (merge.ROOT / supplied).resolve()
            if path in seen or not path.is_relative_to(merge.ROOT / 'results') or path.suffix != '.json':
                raise ValueError('Require unique explicit ended result source paths')
            seen.add(path)
            data, source = merge.load_source(path)
            sources.append(source)
            raw = Path(data['config']['raw_evidence_dir']).resolve()
            scratch = Path(data['config']['runtime']['scratch']).resolve()
            if raw == scratch or not raw.is_relative_to(scratch):
                raise ValueError('Source raw directory is not its declared private scratch directory')
            evidence = Evidence(raw, archive)
            result = scene_diagnostic(data, evidence)
            result.update(source=source, portable_evidence=evidence.references)
            metrics['scenes'].append(result)
        except Exception as error:
            metrics['source_errors'].append({'file': supplied, 'error': f'{type(error).__name__}: {error}'})
    code = int(bool(metrics['source_errors'] or any(row['errors'] for row in metrics['scenes'])))
    if code:
        metrics['status'] = 'diagnostic_partial_unknown'
    config = {'name': args.name, 'requested_sources': args.source,
        'scope': 'read-only saved-state diagnosis before any fix; original failure and acceptance unchanged',
        'agent_rule': 'original encoded JSON equality, including signed-zero/type representation; no tolerance added',
        'object_rule': 'original all movable OR pickupable position <=0.01m; rotations diagnostic only',
        'representation': 'parsed JSON values, repr and float.hex; original lexical bytes remain in archived SHA-bound JSON',
        'archive_max_file_bytes': MAX_FILE_BYTES, 'archive_max_unique_bytes_per_source': MAX_ARCHIVE_BYTES,
        'maximum_setup_records_per_source': MAX_SETUP_RECORDS, 'simulator_actions_executed': False}
    with output.open('x') as stream:
        json.dump({'metrics': metrics, 'config': config, 'sources': sources}, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
