"""Look-only checkpoint with strict per-Controller preflight association.

Collection is the original checkpoint flow with exactly one association change;
all source, journal, eligibility, pair and publication checks remain in place.
Original source records and active producers are never modified.
"""
import copy
import json
from src.tools import merge_pair_v2 as merge
from src.envs import pair_v2_look_unit_policy as policy
from src.tools import t01v2_review_checkpoint as original
from src.tools.t01v2_posthoc_solvability import verified_passed_rows
from src.tools.t01v2_look_unit_review_checkpoint import source_support
from src.tools.t01v2_controller_association import verify_pair_controller


def collect_sources(paths):
    """Validate the complete journal before admitting any of its passed pairs."""
    references, runs, qualified, legacy, attempts, errors = [], [], [], [], [], []
    for path in paths:
        try:
            data, reference = merge.load_source(path)
            if data['config'].get('producer_module') != policy.PRODUCER:
                raise ValueError('Per-controller checkpoint requires an explicit Look-only source')
            policy.validate_source(data['config'], data['metrics'])
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
                    association = verify_pair_controller(data, reference, journal_row, pair, pair_path)
                    checked = merge.check_pair(pair, pair_path, raw_root=cfg['raw_evidence_dir'],
                        clone_enabled=cfg.get('enable_candidate_clone') is True,
                        preflight_version=cfg.get('preflight_version'), evidence_version=cfg.get('evidence_version'),
                        search_version=cfg.get('search_version'), search_policy=cfg.get('search_policy'),
                        search_clone_source=(ledger or {}).get('successful_source_name'),
                        solvability_version=cfg.get('solvability_version'), solvability_policy=cfg.get('solvability_policy'),
                        generation_version=cfg.get('generation_version'), generation_policy=cfg.get('generation_policy'),
                        attempt_setup_version=cfg.get('attempt_setup_version'), attempt_setup_policy=cfg.get('attempt_setup_policy'),
                        candidate_policy=cfg.get('candidate_policy'), candidate_policy_spec=cfg.get('candidate_policy_spec'),
                        projection_pool_version=cfg.get('projection_pool_version'), projection_pool_policy=cfg.get('projection_pool_policy'),
                                      frozen_render_version=cfg.get('frozen_render_version'), frozen_render_policy=cfg.get('frozen_render_policy'),
                                      field_of_view=cfg.get('field_of_view', 100))
                    qualified.append({'pair_id': pair['pair_id'], 'scene': pair['scene'], 'kind': pair['kind'],
                        'target_w0': pair['target_w0'], 'target_w1': pair['target_w1'],
                        'source_file': reference['file'], 'source_metrics_sha256': reference['metrics_sha256'],
                        'source_sha256': reference['sha256'], 'partial_source': eligibility['partial_source'],
                        'journal_index': indices[pair['pair_id']], 'generation_seconds': journal_row.get('generation_seconds'),
                        'pair_reference': pair_ref, 'review_image': checked['review_image'],
                        'proposal_identity': checked['proposal_identity'], 'independent_verification': checked,
                        'controller_preflight_association': association, 'pair': pair})
                except Exception as error:
                    errors.append({'stage': 'pair', 'source_file': reference['file'],
                        'pair_id': journal_row.get('pair_id'), 'error': f'{type(error).__name__}: {error}'})
        except Exception as error:
            errors.append({'stage': 'source', 'source_file': str(path), 'error': f'{type(error).__name__}: {error}'})
    return references, runs, qualified, legacy, attempts, errors



def main(argv=None):
    previous = original.collect_sources
    with source_support():
        try:
            original.collect_sources = collect_sources
            return original.main(argv)
        finally:
            original.collect_sources = previous


if __name__ == '__main__':
    raise SystemExit(main())
