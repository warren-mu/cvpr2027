"""Describe retained, recorded Look-search measurements from explicit ended runs.

No simulator, re-rendering, acceptance upgrade, causal attribution or source
rewrite. Compact omissions remain unknown; retained examples are not a random
sample. Invoke through run_logged after committing this producer.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re

from src.tools import merge_pair_v2 as merge

MAX_ATTEMPT_BYTES = 49_000_000
EXAMPLES_PER_GROUP = 3
MATCH_FIELDS = ('pair_id', 'scene', 'kind', 'status', 'discard_reasons',
                'generation_seconds', 'pair_file', 'pair_sha256')


def observed_bool(value):
    return value if type(value) is bool else None


def stage(row):
    report = row.get('look_solvability') or row.get('observation_search')
    if isinstance(report, dict):
        return 'look_search_recorded'
    reasons = set(row.get('discard_reasons') or [])
    if reasons & {'target_uniqueness_or_ratio', 'realized_target_mode'}:
        return 'before_look_target'
    if reasons & {'w0_physical', 'w1_physical'}:
        return 'before_look_physics'
    if reasons & {'v0_candidates_visibility', 'v0_candidate_outside_select_range',
                  'neither_hidden_shared_prefix_nor_direct_evidence'}:
        return 'before_look_prefix'
    if str(row.get('kind', '')).startswith('direct_'):
        return 'direct_branch'
    return 'unknown'  # A Look kind alone does not prove choose_look was entered.


def prefix_details(row):
    value = row.get('prefix_measurement')
    if not isinstance(value, dict):
        return None
    masks = value.get('prefix_edit_mask_pixels')
    counts = [masks[0].get(key) for key in ('w0', 'w1')] if isinstance(masks, list) and len(masks) == 1 and isinstance(masks[0], dict) else None
    valid_counts = counts is not None and all(type(n) is int and 0 <= n <= 255 * 255 for n in counts)
    maximum = value.get('prefix_max_abs_diff')
    valid_diff = type(maximum) is int and 0 <= maximum <= 255
    category = None
    if valid_counts:
        if counts == [0, 0]:
            if valid_diff:
                category = 'x_hidden_both_rgb_above_3' if maximum > 3 else 'x_hidden_both_rgb_within_3'
        else:
            category = 'x_visible_both' if all(counts) else 'x_visible_w0_only' if counts[0] else 'x_visible_w1_only'
    frames = value.get('frames')
    frame = frames[0] if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], dict) else {}
    return {'classification': category, 'anchor_pixels_w0_w1': counts if valid_counts else None,
        'prefix_max_abs_diff': maximum if valid_diff else None,
        'render_qualified_for_shared_prefix': observed_bool(value.get('render_qualified_for_shared_prefix')),
        'candidates_visible_and_marked': observed_bool(value.get('candidates_visible_and_marked')),
        'construction_context_verified': observed_bool(value.get('construction_context_verified')),
        'recorded_reasons': copy.deepcopy(value.get('reasons')),
        'recorded_frame': {key: copy.deepcopy(frame.get(key)) for key in (
            'label_bindings_exact', 'public_context_exact', 'evidence_pixels', 'evidence_passed',
            'marked_evidence_pixels', 'marked_evidence_passed', 'resolution')},
        'shadow_or_other_render_cause': None}


def look_details(row):
    value = row.get('look_solvability') or row.get('observation_search')
    if not isinstance(value, dict):
        return None
    actions = value.get('actions')
    return {'complete': observed_bool(value.get('complete')),
        'solvable_w0': observed_bool(value.get('solvable_w0')),
        'solvable_w1': observed_bool(value.get('solvable_w1')),
        'anchor_visible_actions_w0': copy.deepcopy(value.get('anchor_visible_actions_w0')),
        'anchor_visible_actions_w1': copy.deepcopy(value.get('anchor_visible_actions_w1')),
        'selected_action': value.get('selected_action'), 'recorded_reasons': copy.deepcopy(value.get('reasons')),
        'actions': [{'action': item.get('action'), 'recorded_common_choice_passed': observed_bool(item.get('passed')),
            'worlds': [{key: copy.deepcopy(world.get(key)) for key in (
                'world_index', 'status', 'anchor_pixels', 'anchor_visible', 'error', 'continuation')}
                for world in item.get('worlds', [])]} for item in actions] if isinstance(actions, list) else None}


def read_attempt(reference, raw_root):
    if (not isinstance(reference, dict) or not isinstance(reference.get('file'), str)
            or re.fullmatch(r'[0-9a-f]{64}', str(reference.get('sha256'))) is None):
        raise ValueError('Attempt reference lacks its recorded path/SHA')
    path = (merge.ROOT / reference['file']).resolve()
    if not path.is_relative_to(Path(raw_root).resolve()):
        raise ValueError('Attempt evidence escaped its declared raw directory')
    if not path.exists():
        return None  # Missing retained bytes are not an observed zero.
    if path.stat().st_size > MAX_ATTEMPT_BYTES:
        raise ValueError('Attempt JSON exceeds the diagnostic read cap')
    return json.loads(merge.reference_path(reference, within=raw_root).read_text())


def journal_rows(data):
    cfg, measured = data['config'], data['metrics']
    reference = merge.journal_reference(data, merge.source_eligibility(data))
    if reference is None:
        return reference, None
    path = merge.reference_path(reference, within=merge.ROOT / 'data/world_pairs/v2')
    rows, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get('scene') != cfg['scene'] or not isinstance(row.get('pair_id'), str)
                or re.fullmatch(re.escape(cfg['scene'].lower()) + r'_[0-9]{5}', row['pair_id']) is None
                or row['pair_id'] in seen or row.get('status') not in ('passed', 'rejected')
                or not isinstance(row.get('discard_reasons'), list)
                or any(not isinstance(reason, str) for reason in row['discard_reasons'])
                or (row['status'] == 'passed' and row.get('kind') not in merge.KINDS)):
            raise ValueError('Malformed, duplicate or wrong-scene journal attempt')
        seen.add(row['pair_id'])
        rows.append(row)
    return reference, rows


def describe_source(data, reference, *, attempt_reader=read_attempt):
    cfg, measured = data['config'], data['metrics']
    journal, rows = journal_rows(data)
    result = {'source': reference, 'scene': cfg['scene'], 'pbs_jobid': cfg['shard']['pbs_jobid'],
        'source_status': measured.get('status'), 'source_exit_code': data['_meta']['exit_code'],
        'cleanup_errors': copy.deepcopy(measured.get('cleanup_errors')), 'journal': journal,
        'reported_attempts': measured.get('attempts'), 'journal_attempts': None,
        'counter_consistency': None, 'counter_errors': [], 'unknown_source_counters': [],
        'raw_available_attempts': 0, 'raw_unavailable_attempts': 0, 'raw_errors': [],
        'stage_counts': {}, 'prefix_classification_counts': {}, 'neither_hidden_breakdown': {},
        'target_reason_counts_in_retained_subset': {}, 'target_measurement_attempts': 0,
        'look_report_attempts': 0, 'look_report_completeness': {}, 'per_world_solvability': {},
        'prefix_measurement_attempts': 0, 'prefix_classification_unknown_attempts': 0,
        'examples': {key: [] for key in ('target', 'hidden_rgb_failure', 'visible_prefix', 'look_search', 'unknown_prefix')},
        'shadow_cause_verified': None, 'scientific_acceptance': None}
    if rows is None:
        for key in ('raw_available_attempts', 'raw_unavailable_attempts', 'stage_counts',
                    'prefix_classification_counts', 'neither_hidden_breakdown',
                    'target_reason_counts_in_retained_subset', 'target_measurement_attempts',
                    'look_report_attempts', 'look_report_completeness', 'per_world_solvability',
                    'prefix_measurement_attempts', 'prefix_classification_unknown_attempts'):
            result[key] = None
        return result
    result['journal_attempts'] = len(rows)
    expected = {'attempts': len(rows),
        'passed_by_class': {kind: sum(row['status'] == 'passed' and row.get('kind') == kind for row in rows) for kind in merge.KINDS},
        'discard_reason_counts': dict(Counter(reason for row in rows for reason in row['discard_reasons']))}
    for key, value in expected.items():
        if measured.get(key) is None:
            result['unknown_source_counters'].append(key)
        elif measured[key] != value:
            result['counter_errors'].append({'field': key, 'source_reported': measured[key], 'authenticated_journal': value})
    result['counter_consistency'] = False if result['counter_errors'] else None if result['unknown_source_counters'] else True
    stages, prefixes, neither, targets, complete = Counter(), Counter(), Counter(), Counter(), Counter()
    solvability = {world: Counter() for world in ('w0', 'w1')}
    for compact in rows:
        try:
            raw = attempt_reader(compact.get('raw_attempt'), cfg['raw_evidence_dir'])
            if raw is not None and not isinstance(raw, dict):
                raise ValueError('Attempt JSON is not an object')
            if raw is not None and any(raw.get(key) != compact.get(key) for key in MATCH_FIELDS):
                raise ValueError('Authenticated raw attempt differs from its journal row')
        except (OSError, ValueError, TypeError, KeyError) as error:
            result['raw_errors'].append({'pair_id': compact['pair_id'], 'reference': compact.get('raw_attempt'),
                                         'error': f'{type(error).__name__}: {error}'})
            raw = None
        if raw is None:
            result['raw_unavailable_attempts'] += 1
        else:
            result['raw_available_attempts'] += 1
        stages[stage(raw if raw is not None else compact)] += 1
        prefix, look = prefix_details(raw or {}), look_details(raw or {})
        category = (prefix or {}).get('classification')
        if prefix is not None:
            result['prefix_measurement_attempts'] += 1
        if category is None:
            result['prefix_classification_unknown_attempts'] += 1
        else:
            prefixes[category] += 1
        if 'neither_hidden_shared_prefix_nor_direct_evidence' in compact['discard_reasons']:
            neither[category or 'unknown_missing_retained_prefix_measurements'] += 1
        values = (raw or {}).get('target_measurements')
        if (isinstance(values, list) and len(values) == 2 and all(isinstance(v, dict)
                and isinstance(v.get('reasons'), list) and all(isinstance(reason, str) for reason in v['reasons']) for v in values)):
            result['target_measurement_attempts'] += 1
            targets.update(set(reason for value in values for reason in value.get('reasons', [])))
        if look is not None:
            result['look_report_attempts'] += 1
            complete[str(look['complete']).lower() if look['complete'] is not None else 'unknown'] += 1
            for world in solvability:
                value = look['solvable_' + world]
                solvability[world][str(value).lower() if value is not None else 'unknown'] += 1
        group = ('target' if 'target_uniqueness_or_ratio' in compact['discard_reasons'] else
                 'hidden_rgb_failure' if category == 'x_hidden_both_rgb_above_3' else
                 'visible_prefix' if category and category.startswith('x_visible') else
                 'look_search' if look is not None else 'unknown_prefix' if category is None else None)
        if group and len(result['examples'][group]) < EXAMPLES_PER_GROUP:
            result['examples'][group].append({'pair_id': compact['pair_id'], 'raw_attempt': compact.get('raw_attempt'),
                'raw_available': raw is not None, 'discard_reasons': compact['discard_reasons'],
                'prefix': prefix, 'target_measurements': copy.deepcopy(values), 'look_search': look})
    result.update(stage_counts=dict(stages), prefix_classification_counts=dict(prefixes),
        neither_hidden_breakdown=dict(neither), target_reason_counts_in_retained_subset=dict(targets),
        look_report_completeness=dict(complete), per_world_solvability={key: dict(value) for key, value in solvability.items()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source', nargs='+', required=True)
    args = parser.parse_args(argv)
    if re.fullmatch(r'[A-Za-z0-9_]+', args.name) is None:
        parser.error('Use a unique safe result name')
    output = merge.ROOT / 'results' / (args.name + '.json')
    if output.exists() or output.is_symlink():
        raise FileExistsError('Existing evidence is preserved; use a new name')
    sources, runs, errors, seen = [], [], [], set()
    for value in args.source:
        try:
            path = (merge.ROOT / value).resolve()
            if path in seen:
                raise ValueError('Duplicate explicit source')
            seen.add(path)
            data, ref = merge.load_source(path)
            sources.append(ref)
            runs.append(describe_source(data, ref))
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append({'source': value, 'error': f'{type(error).__name__}: {error}'})
    failed = bool(errors or any(row['counter_errors'] or row['raw_errors'] for row in runs))
    metrics = {'kind': 't01v2_recorded_look_diagnostic', 'aggregation_status': 'failed' if failed else 'passed',
               'source_runs': runs, 'source_errors': errors, 'scientific_acceptance': None, 'shadow_cause_verified': None}
    config = {'name': args.name, 'explicit_sources': args.source,
        'scope': 'Source-recorded measurements in authenticated finalized journals/retained attempt JSON; not independent native remeasurement or pair acceptance.',
        'denominators': 'Stage counts use authenticated journal attempts; prefix/target/action details use explicitly counted retained subsets only. Unknown observations are separate.',
        'retention_bias': 'Full rejected evidence generally covers only the first three examples of a rejection reason; this is not a random subset or a population estimate.',
        'prefix_note': 'shared_prefix_not_verified alone is expected in the initial pre-reuse check; use render_qualified_for_shared_prefix and the separately recorded actual masks/RGB/context fields.',
        'causal_limit': 'X mask zero plus RGB difference above three does not establish shadows or any rendering cause.',
        'examples': 'First three journal-order examples per fixed diagnostic group; raw references remain at their original source paths, not copied or modified.',
        'max_attempt_json_bytes': MAX_ATTEMPT_BYTES, 'simulator_actions_executed': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump({'metrics': metrics, 'config': config, 'sources': sources}, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
