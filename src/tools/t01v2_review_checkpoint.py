"""Publish a uniquely named early review checkpoint, never final pilot acceptance.

Only formally ended, audited scene sources are read. The frozen final merger's
independent construction checks remain mandatory even for a single candidate.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re
import shutil

from src.tools import merge_pair_v2 as merge
from src.tools.t01v2_posthoc_solvability import verified_passed_rows


def output_paths(name):
    if re.fullmatch(r't01v2_checkpoint_[A-Za-z0-9_]+', name) is None:
        raise ValueError('Require a new t01v2_checkpoint_* name')
    result = merge.ROOT / 'results' / (name + '.json')
    directory = merge.ROOT / 'logs/review' / name
    # run_logged owns its currently open .log/.meta.json. Do not reject those.
    if result.exists() or directory.exists():
        raise FileExistsError('Existing checkpoint evidence is preserved; use a new name')
    return result, directory


def compact_pair(row):
    return {key: value for key, value in row.items() if key not in ('pair', 'independent_verification')}


def checkpoint_summary(qualified, scenes, errors):
    """No quota trimming: retain every independently checked unique candidate."""
    selected, surplus, duplicates = merge.balanced_selection(qualified)
    unique = sorted(selected + surplus, key=lambda item:
                    (merge.scene_number(item['scene']), item['journal_index'], item['source_file']))
    counts = {kind: sum(row['kind'] == kind for row in unique) for kind in merge.KINDS}
    valid = not errors
    return unique, {'independently_verified_by_class': counts,
        'independently_verified_pair_count': len(unique),
        'duplicate_pairs': [compact_pair(row) for row in duplicates],
        'target_balance_unique_qualified': merge.target_balance(unique, scenes),
        'e2e_data_threshold_met': sum(value >= 1 for value in counts.values()) >= 2 if valid else None,
        'g01_data_threshold_met': all(value >= 5 for value in counts.values()) if valid else None,
        'final_pilot_acceptance': False, 'full_t02_acceptance': None,
        'full_metrics_spec_compliance': None, 'human_review_approved': None,
        'execution_authorized': False}


def collect_sources(paths):
    """Validate the complete journal before admitting any of its passed pairs."""
    references, runs, qualified, legacy, attempts, errors = [], [], [], [], [], []
    for path in paths:
        try:
            data, reference = merge.load_source(path)
            if type(data['_meta'].get('exit_code')) is not int or not data['_meta'].get('end'):
                raise ValueError('Checkpoint requires an explicitly completed source')
            references.append(reference)
            cfg, measured = data['config'], data['metrics']
            eligibility, journal_ref, passed = verified_passed_rows(data)
            run = {'source': reference, **eligibility, 'scene': cfg['scene'],
                'ended_at_source_time': data['_meta']['end'], 'source_time_zone': 'not_recorded_by_run_logged',
                'pbs_jobid': cfg['shard']['pbs_jobid'],
                'ledger_only': not eligibility['eligible'],
                'recorded_attempts': measured.get('attempts'),
                'recorded_passed_by_class': copy.deepcopy(measured.get('passed_by_class')),
                'recorded_rejection_counts': copy.deepcopy(measured.get('discard_reason_counts')),
                'scene_excluded': copy.deepcopy(measured.get('scene_excluded')),
                'attempt_journal': journal_ref, 'verified_attempts': None,
                'versions': {key: cfg.get(key) for key in ('preflight_version', 'evidence_version',
                    'search_version', 'solvability_version', 'generation_version')}}
            runs.append(run)
            merge.scene_number(cfg['scene'])
            if journal_ref is None:
                continue  # Audited early failure; its unobserved attempts stay unknown.
            journal = merge.reference_path(journal_ref, within=merge.ROOT / 'data/world_pairs/v2')
            rows = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
            run['verified_attempts'] = len(rows)
            indices = {row['pair_id']: index for index, row in enumerate(rows)}
            attempts.extend({'source_file': reference['file'], **row} for row in rows)
            if not eligibility['eligible']:
                continue
            ledger = None
            if cfg.get('search_version') is not None and passed:
                inventory = merge.search_source_inventory(measured,
                    lambda ref: merge.read_reference(ref, within=cfg['raw_evidence_dir']))
                ledger = merge.verify_search_ledger(measured, inventory)
                run['search_ledger_verification'] = ledger
            for journal_row, pair_ref, pair_path in passed:
                try:
                    pair = merge.read_reference(pair_ref, within=pair_path.parent)
                    if pair.get('kind') not in merge.KINDS or any(pair.get(key) != journal_row.get(key)
                            for key in ('pair_id', 'scene', 'kind', 'status', 'discard_reasons')):
                        raise ValueError('Pair and authenticated journal identities/outcomes differ')
                    if not merge.final_pair_eligible(pair['kind'], cfg.get('solvability_version')):
                        legacy.append({'source_file': reference['file'], 'pair_id': pair['pair_id'],
                            'scene': pair['scene'], 'kind': pair['kind'], 'pair_reference': pair_ref,
                            'original_status': pair['status'], 'solvability': None,
                            'reason': 'legacy_insufficient_solvability',
                            'pending_check': 'separate authenticated posthoc proof; absence is not unsolvability'})
                        continue
                    preflight = measured['scene_preflight_reference']
                    merge.reference_path(preflight, within=cfg['raw_evidence_dir'])
                    if pair['scene_preflight_reference']['sha256'] != preflight['sha256']:
                        raise ValueError('Portable pair preflight differs from the logged source preflight')
                    checked = merge.check_pair(pair, pair_path, raw_root=cfg['raw_evidence_dir'],
                        clone_enabled=cfg.get('enable_candidate_clone') is True,
                        preflight_version=cfg.get('preflight_version'), evidence_version=cfg.get('evidence_version'),
                        search_version=cfg.get('search_version'), search_policy=cfg.get('search_policy'),
                        search_clone_source=(ledger or {}).get('successful_source_name'),
                        solvability_version=cfg.get('solvability_version'), solvability_policy=cfg.get('solvability_policy'),
                        generation_version=cfg.get('generation_version'), generation_policy=cfg.get('generation_policy'),
                        attempt_setup_version=cfg.get('attempt_setup_version'), attempt_setup_policy=cfg.get('attempt_setup_policy'),
                        candidate_policy=cfg.get('candidate_policy'), candidate_policy_spec=cfg.get('candidate_policy_spec'),
                        projection_pool_version=cfg.get('projection_pool_version'), projection_pool_policy=cfg.get('projection_pool_policy'))
                    qualified.append({'pair_id': pair['pair_id'], 'scene': pair['scene'], 'kind': pair['kind'],
                        'target_w0': pair['target_w0'], 'target_w1': pair['target_w1'],
                        'source_file': reference['file'], 'source_metrics_sha256': reference['metrics_sha256'],
                        'source_sha256': reference['sha256'], 'partial_source': eligibility['partial_source'],
                        'journal_index': indices[pair['pair_id']], 'generation_seconds': journal_row.get('generation_seconds'),
                        'pair_reference': pair_ref, 'review_image': checked['review_image'],
                        'proposal_identity': checked['proposal_identity'], 'independent_verification': checked, 'pair': pair})
                except Exception as error:
                    errors.append({'stage': 'pair', 'source_file': reference['file'],
                        'pair_id': journal_row.get('pair_id'), 'error': f'{type(error).__name__}: {error}'})
        except Exception as error:
            errors.append({'stage': 'source', 'source_file': str(path), 'error': f'{type(error).__name__}: {error}'})
    return references, runs, qualified, legacy, attempts, errors


def publish_reviews(rows, directory):
    # Verify every source before creating the output directory. All pair/native
    # checks have already succeeded; no candidate is published on a contradiction.
    sources = [merge.reference_path(row['review_image'], within=merge.ROOT / 'data/world_pairs/v2') for row in rows]
    directory.mkdir(parents=True, exist_ok=False)
    files = []
    for index, (row, source) in enumerate(zip(rows, sources)):
        destination = directory / f'{index:05d}_{row["kind"]}.png'
        with source.open('rb') as incoming, destination.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        digest = merge.sha_file(destination)
        if digest != row['review_image']['sha256']:
            raise ValueError('Copied review image differs from authenticated source')
        files.append({'file': str(destination.relative_to(merge.ROOT)), 'sha256': digest,
            'pair_id': row['pair_id'], 'scene': row['scene'], 'kind': row['kind'],
            'source_file': row['source_file'], 'source_image': row['review_image']})
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source', nargs='+', action='extend', required=True)
    args = parser.parse_args(argv)
    result, review_directory = output_paths(args.name)
    from src.envs.pair_v2_solvable_policy import SOLVABILITY_VERSION
    config = {'name': args.name, 'requested_sources': args.source,
        'review_directory': str(review_directory.relative_to(merge.ROOT)),
        'scope': 'early construction-stage plus supplement 4(a) review checkpoint; not final pilot/T02 acceptance',
        'final_requirements': {'look_solvability_version': SOLVABILITY_VERSION,
            'legacy_look': 'unknown/excluded here; separate posthoc report does not silently upgrade this checkpoint',
            'legacy_direct': 'all unchanged independent construction checks required'},
        'thresholds': {'e2e': 'at least two classes with one independently qualified unique pair each',
                       'g01': 'at least five independently qualified unique pairs in each of four classes'},
        'human_review': 'separate human relay of Claude image review required; never inferred from counts',
        'order': 'numeric scene, journal row, source filename; all unique qualified pairs retained',
        'metrics_spec_sha256': merge.sha_file(merge.ROOT / 'docs/METRICS_SPEC.md'),
        'construction_compatible_spec_references': list(merge.CONSTRUCTION_SPEC_REFERENCES),
        'evidence_retention': 'original scratch references remain necessary; images alone are not a portable proof archive'}
    metrics = {'kind': 't01v2_review_checkpoint', 'status': 'invalid_evidence', 'errors': [],
        'source_runs': [], 'review_files': [], 'candidate_pairs': [], 'published_pair_count': 0,
        'published_by_class': {kind: 0 for kind in merge.KINDS},
        'final_pilot_acceptance': False, 'full_t02_acceptance': None, 'full_metrics_spec_compliance': None,
        'human_review_approved': None, 'execution_authorized': False,
        'e2e_data_threshold_met': None, 'g01_data_threshold_met': None,
        't02_pending_checks': {key: 'not_run_T02_pending' for key in
            ('P_table', 'BFS_budget_D', 'navigation_graph', 'no_leak', 'cache_at_least_200')}}
    references, code = [], 1
    try:
        paths = [(merge.ROOT / path).resolve() for path in args.source]
        if len(paths) != len(set(paths)):
            raise ValueError('Duplicate explicit source')
        references, runs, qualified, legacy, attempts, errors = collect_sources(paths)
        metrics.update(source_runs=runs, legacy_insufficient_solvability=legacy, errors=errors)
        unique, summary = checkpoint_summary(qualified, [row['scene'] for row in runs], errors)
        metrics.update(summary)
        reasons = Counter(reason for row in attempts for reason in (row.get('discard_reasons') or []))
        metrics.update(observed_attempts=len(attempts),
            attempts=(len(attempts) if not errors and all(row['verified_attempts'] is not None for row in runs) else None),
            rejection_counts=dict(reasons), unknown_attempt_source_count=sum(row['verified_attempts'] is None for row in runs),
            attempt_generation_time=merge.durations([row.get('generation_seconds') for row in attempts]),
            qualified_generation_time=merge.durations([row.get('generation_seconds') for row in unique]),
            timing_scope='recorded construction durations only; excludes PBS wait/allocation and unmeasured setup')
        if not errors:
            metrics['review_files'] = publish_reviews(unique, review_directory)
            metrics['candidate_pairs'] = [{**compact_pair(row), 'independent_verification': row['independent_verification']}
                                          for row in unique]
            metrics.update(status='checkpoint_ready_for_human_review' if unique else 'checkpoint_no_qualified_pairs',
                published_pair_count=len(unique), published_by_class=summary['independently_verified_by_class'])
            code = 0
    except Exception as error:
        metrics['errors'].append({'stage': 'checkpoint', 'error': f'{type(error).__name__}: {error}'})
        metrics.update(e2e_data_threshold_met=None, g01_data_threshold_met=None)
    result.parent.mkdir(parents=True, exist_ok=True)
    with result.open('x') as stream:
        json.dump({'metrics': metrics, 'config': config, 'sources': references}, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
