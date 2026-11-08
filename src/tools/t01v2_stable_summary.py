"""Explicit stable-shard progress/report ledger; no acceptance or job mutation."""
import argparse
import copy
import json
from pathlib import Path
import re

from src.envs.pair_v2_stable import PREFLIGHT_VERSION
from src.tools.merge_pair_v2 import KINDS, load_source, reference_path
from src.tools.t01_partition import ROOT
from src.tools.t01v2_preflight_diagnostic import archive_reference


def preflight_numbers(record):
    diagnostics = record.get('diagnostics') or {}
    initial = diagnostics.get('initial_settling') or {}
    window = diagnostics.get('stability_window') or {}
    def largest(rows):
        values = [item['baseline_displacement_m'] for row in rows
                  for item in (row.get('measurements') or {}).get('objects', {}).values()]
        return max(values) if values else None
    initial_rows, window_rows = initial.get('observations'), window.get('observations')
    initial_measurements = list(initial_rows or [])
    if diagnostics.get('pause_measurements') is not None:
        initial_measurements.insert(0, {'measurements': diagnostics['pause_measurements']})
    return {'passed': record.get('passed'), 'reasons': record.get('reasons'), 'error': record.get('error'),
        'initial_settled': initial.get('settled'), 'settling_criterion': initial.get('criterion'),
        'observed_initial_steps': len(initial_rows) if isinstance(initial_rows, list) else None,
        'initial_max_displacement_m': largest(initial_measurements),
        'window_completed': window.get('completed'),
        'observed_window_steps': len(window_rows) if isinstance(window_rows, list) else None,
        'window_max_displacement_m': largest(window_rows or []), 'preflight_version': record.get('preflight_version')}


def aggregate(rows, pending, errors):
    known = {kind: 0 for kind in KINDS}
    unknown = []
    categories = ('edited_position_error', 'other_object_displacement', 'parent_receptacle_unverified')
    physical = {key: 0 for key in categories}
    physical_unknown = []
    for row in rows:
        counts = row.get('reported_passed_by_class')
        if not isinstance(counts, dict) or any(type(counts.get(k)) is not int or counts[k] < 0 for k in KINDS):
            unknown.append(row['scene'])
        else:
            for key in KINDS:
                known[key] += counts[key]
        distribution = row.get('physical_rejection_distribution')
        if not isinstance(distribution, dict) or any(type(distribution.get(k)) is not int or distribution[k] < 0 for k in categories):
            physical_unknown.append(row['scene'])
        else:
            for key in categories:
                physical[key] += distribution[key]
    w0, w0_unknown = {}, []
    for row in rows:
        values = row.get('w0_by_anchor')
        keys = ('measured_attempts', 'passed', 'rejected', 'measurement_unavailable')
        valid = isinstance(values, dict) and all(isinstance(v, dict)
            and all(type(v.get(k)) is int and v[k] >= 0 for k in keys)
            and v['passed'] + v['rejected'] == v['measured_attempts'] for v in values.values())
        if not valid:
            w0_unknown.append(row['scene'])
            continue
        for anchor, values_for_anchor in values.items():
            totals = w0.setdefault(anchor, {key: 0 for key in keys})
            for key in keys:
                totals[key] += values_for_anchor[key]
    return {'known_source_reported_passed_by_class': known, 'unknown_count_scenes': unknown,
        'all_requested_counts_known': not pending and not errors and not unknown,
        'known_physical_rejection_distribution': physical, 'unknown_physical_distribution_scenes': physical_unknown,
        'all_requested_physical_distributions_known': not pending and not errors and not physical_unknown,
        'physical_distribution_denominator': 'rejected attempts; each category once per attempt; categories may overlap',
        'source_counts_independently_revalidated_as_accepted_pairs': False, 'task_acceptance': None,
        'known_w0_by_anchor': w0, 'unknown_w0_scenes': w0_unknown,
        'all_requested_w0_counts_known': not pending and not errors and not w0_unknown,
        'w0_denominator': 'completed physical measurements only; unavailable measurements remain separate'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source', nargs='+', required=True)
    args = parser.parse_args()
    if re.fullmatch(r'[A-Za-z0-9_]+', args.name) is None:
        parser.error('Use a unique alphanumeric result name')
    output = ROOT / 'results' / (args.name + '.json')
    if output.exists():
        raise FileExistsError('Preserve previous summary; choose a new name')
    evidence = ROOT / 'logs/runs' / args.name / 'evidence'
    evidence.mkdir(parents=True, exist_ok=False)
    rows, pending, errors, sources, seen = [], [], [], [], set()
    for value in args.source:
        try:
            path = (ROOT / value).resolve()
            if not path.is_relative_to(ROOT / 'results') or path in seen or path.suffix != '.json':
                raise ValueError('Require unique explicit result source paths')
            seen.add(path)
            if not path.is_file():
                pending.append({'source_file': value, 'status': 'result_not_yet_observed', 'counts': None})
                continue
            data, ref = load_source(path)
            config, metrics, meta = data['config'], data['metrics'], data['_meta']
            if config.get('preflight_version') != PREFLIGHT_VERSION:
                raise ValueError('Only explicitly versioned stable shards are reported')
            row = {'source_file': value, 'scene': config['scene'], 'pbs_jobid': config['shard']['pbs_jobid'],
                'status': metrics.get('status'), 'exit_code': meta['exit_code'],
                'reported_passed_by_class': copy.deepcopy(metrics.get('passed_by_class')),
                'reported_attempts': metrics.get('attempts'),
                'discard_reason_counts': copy.deepcopy(metrics.get('discard_reason_counts')),
                'scene_excluded': copy.deepcopy(metrics.get('scene_excluded')),
                'cleanup_errors': copy.deepcopy(metrics.get('cleanup_errors')), 'error': copy.deepcopy(metrics.get('error')),
                'physical_rejection_distribution': copy.deepcopy((metrics.get('evidence_retention') or {}).get('physical_rejection_distribution')),
                'complete_rejection_reason_counts': copy.deepcopy((metrics.get('evidence_retention') or {}).get('complete_rejection_reason_counts')),
                'physical_check_unavailable_rejections': (metrics.get('evidence_retention') or {}).get('physical_check_unavailable_rejections'),
                'preflight': None, 'clone_preparation': None,
                'search_version': metrics.get('search_version'),
                'w0_by_anchor': copy.deepcopy(metrics.get('w0_by_anchor')),
                'clone_attempts': copy.deepcopy(metrics.get('clone_attempts')),
                'controller_visits': copy.deepcopy(metrics.get('controller_visits')),
                'raw_evidence_errors': []}
            row['reported_clone_attempt_count'] = (len(row['clone_attempts'])
                if isinstance(row['clone_attempts'], list) else None)
            preflight_ref = metrics.get('scene_preflight_reference')
            if preflight_ref:
                try:
                    raw = json.loads(reference_path(preflight_ref, within=config['raw_evidence_dir']).read_text())
                    row['preflight'] = preflight_numbers(raw)
                    row['preflight']['evidence'] = archive_reference(preflight_ref, evidence)
                except (OSError, ValueError) as error:
                    row['raw_evidence_errors'].append({'scope': 'preflight', 'reference': preflight_ref,
                        'error': f'{type(error).__name__}: {error}'})
            clone = metrics.get('clone_preparation')
            if clone is not None:
                row['clone_preparation'] = {key: copy.deepcopy(clone.get(key)) for key in
                    ('status', 'passed', 'clone_attempted', 'source_name', 'error')}
                check = (clone.get('clone_record') or {}).get('clone_baseline_check') or {}
                row['clone_preparation']['physical_reasons'] = copy.deepcopy(check.get('reasons'))
                row['clone_preparation']['physical_edits'] = copy.deepcopy((check.get('physical') or {}).get('edits'))
                if clone.get('record_reference'):
                    try:
                        reference_path(clone['record_reference'], within=config['raw_evidence_dir'])
                        row['clone_preparation']['evidence'] = archive_reference(clone['record_reference'], evidence)
                    except (OSError, ValueError) as error:
                        row['raw_evidence_errors'].append({'scope': 'clone_preparation', 'reference': clone['record_reference'],
                            'error': f'{type(error).__name__}: {error}'})
            rows.append(row)
            sources.append(ref)
        except Exception as error:
            errors.append({'source_file': value, 'error': f'{type(error).__name__}: {error}'})
    evidence_errors = [{'source_file': row['source_file'], 'scene': row['scene'], **error}
        for row in rows for error in row['raw_evidence_errors']]
    metrics = {'kind': 't01v2_stable_progress', 'shards': rows, 'pending_sources': pending,
        'raw_evidence_errors': evidence_errors,
        'source_errors': errors, **aggregate(rows, pending, errors)}
    config = {'name': args.name, 'explicit_sources': args.source,
        'scope': 'Audited source-reported progress plus authenticated preflight/clone JSON; final pair acceptance belongs to the independent final merger.',
        'missing_policy': 'Not observed or unavailable means unknown, never zero.',
        'physical_counts_scope': 'Construction-attempt categories; clone setup failures and scene preflight exclusions reported separately.'}
    with output.open('x') as stream:
        json.dump({'metrics': metrics, 'config': config, 'sources': sources}, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return int(bool(errors or evidence_errors))


if __name__ == '__main__':
    raise SystemExit(main())
