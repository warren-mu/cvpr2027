"""Synthetic CPU fixtures only; no simulator or scientific result generation."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools import t01v2_review_checkpoint as checkpoint


def candidate(kind='direct_flip', identity='one', index=0):
    pair = {'pair_id': identity, 'scene': 'FloorPlan1', 'kind': kind,
        'identities': {'candidate_names': ['a', 'b']},
        'target_w0': 'a', 'target_w1': 'a' if kind.endswith('stay') else 'b'}
    return {**pair, 'pair': pair, 'proposal_identity': identity, 'source_file': 'results/scene.json',
        'journal_index': index, 'independent_verification': {'validity': {'acceptance_passed': True}}}


class CheckpointTests(unittest.TestCase):
    def test_two_classes_allow_data_threshold_without_final_or_human_acceptance(self):
        rows, measured = checkpoint.checkpoint_summary([candidate(), candidate('direct_stay', 'two')], ['FloorPlan1'], [])
        self.assertEqual(len(rows), 2)
        self.assertTrue(measured['e2e_data_threshold_met'])
        self.assertFalse(measured['g01_data_threshold_met'])
        self.assertFalse(measured['final_pilot_acceptance'])
        self.assertIsNone(measured['full_t02_acceptance'])
        self.assertIsNone(measured['human_review_approved'])
        self.assertFalse(measured['execution_authorized'])

    def test_no_pairs_is_valid_observed_zero_not_a_quota_failure(self):
        rows, measured = checkpoint.checkpoint_summary([], ['FloorPlan1'], [])
        self.assertEqual(rows, [])
        self.assertFalse(measured['e2e_data_threshold_met'])
        self.assertIsNone(measured['target_balance_unique_qualified']['rows'][0]['correct_target_a_probability_at_rho_half'])

    def test_g01_counts_deduplicate_and_do_not_trim_at_final_twenty(self):
        rows = [candidate(kind, f'{kind}_{index}', index) for kind in checkpoint.merge.KINDS for index in range(5)]
        unique, measured = checkpoint.checkpoint_summary(rows + [copy.deepcopy(rows[0])], ['FloorPlan1'], [])
        self.assertEqual(len(unique), 20)
        self.assertTrue(measured['g01_data_threshold_met'])
        self.assertEqual(len(measured['duplicate_pairs']), 1)
        many = [candidate(identity=str(index), index=index) for index in range(21)]
        self.assertEqual(len(checkpoint.checkpoint_summary(many, ['FloorPlan1'], [])[0]), 21)

    def test_conflicting_duplicate_cannot_be_published(self):
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            checkpoint.checkpoint_summary([candidate(), candidate('direct_stay')], ['FloorPlan1'], [])

    def test_any_evidence_error_leaves_threshold_unknown(self):
        _, measured = checkpoint.checkpoint_summary([candidate(), candidate('direct_stay', 'two')],
                                                    ['FloorPlan1'], [{'error': 'source SHA differs'}])
        self.assertIsNone(measured['e2e_data_threshold_met'])
        self.assertIsNone(measured['g01_data_threshold_met'])

    def test_name_and_output_collision_preserve_evidence(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(checkpoint.merge, 'ROOT', Path(folder)):
            for name in ('pilot_pairs_v2', '../t01v2_checkpoint_x', 't01v2_checkpoint_'):
                with self.assertRaises(ValueError):
                    checkpoint.output_paths(name)
            result, directory = checkpoint.output_paths('t01v2_checkpoint_fixture')
            directory.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                checkpoint.output_paths('t01v2_checkpoint_fixture')
            self.assertFalse(result.exists())

    def test_audited_early_failure_keeps_attempts_unknown(self):
        data = {'_meta': {'exit_code': 1, 'end': 'synthetic-end'},
            'config': {'scene': 'FloorPlan1', 'shard': {'pbs_jobid': 'synthetic'}},
            'metrics': {'status': 'failed_before_generation'}}
        eligibility = {'eligible': False, 'partial_source': True, 'cleanup_errors': []}
        with patch.object(checkpoint.merge, 'load_source', return_value=(data, {'file': 'results/scene.json'})), \
                patch.object(checkpoint, 'verified_passed_rows', return_value=(eligibility, None, [])):
            _, runs, rows, _, attempts, errors = checkpoint.collect_sources([Path('synthetic.json')])
        self.assertEqual(errors, [])
        self.assertEqual(rows, [])
        self.assertEqual(attempts, [])
        self.assertIsNone(runs[0]['recorded_attempts'])
        self.assertIsNone(runs[0]['verified_attempts'])
        self.assertEqual(runs[0]['ended_at_source_time'], 'synthetic-end')
        self.assertEqual(runs[0]['source_time_zone'], 'not_recorded_by_run_logged')
        self.assertNotIn('ended_at_utc', runs[0])

    def test_portable_preflight_must_match_source_before_independent_pair_check(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            journal = root / 'journal.jsonl'
            compact = {'pair_id': 'p', 'scene': 'FloorPlan1', 'kind': 'direct_flip',
                       'status': 'passed', 'discard_reasons': []}
            journal.write_text(json.dumps(compact) + '\n')
            pair = {**compact, 'scene_preflight_reference': {'sha256': 'different'}}
            cfg = {'scene': 'FloorPlan1', 'shard': {'pbs_jobid': 'synthetic'}, 'raw_evidence_dir': str(root)}
            data = {'_meta': {'exit_code': 0, 'end': 'synthetic-end'}, 'config': cfg,
                    'metrics': {'scene_preflight_reference': {'file': 'raw.json', 'sha256': 'original'}}}
            pair_ref = {'file': 'pair.json', 'sha256': 'synthetic'}
            eligibility = {'eligible': True, 'partial_source': False}
            with patch.object(checkpoint.merge, 'load_source', return_value=(data, {'file': 'results/scene.json'})), \
                    patch.object(checkpoint, 'verified_passed_rows', return_value=(eligibility, {'file': 'journal'},
                                  [(compact, pair_ref, root / 'pair.json')])), \
                    patch.object(checkpoint.merge, 'reference_path', return_value=journal), \
                    patch.object(checkpoint.merge, 'read_reference', return_value=pair), \
                    patch.object(checkpoint.merge, 'check_pair') as check:
                _, _, qualified, _, _, errors = checkpoint.collect_sources([root / 'scene.json'])
            self.assertEqual(qualified, [])
            self.assertIn('preflight differs', errors[0]['error'])
            check.assert_not_called()

    def test_source_error_prevents_any_review_publication_and_writes_failed_result(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(checkpoint.merge, 'ROOT', Path(folder)):
            with patch.object(checkpoint.merge, 'sha_file', return_value='synthetic'), \
                    patch.object(checkpoint, 'collect_sources', return_value=([], [], [], [], [], [{'error': 'hash mismatch'}])), \
                    patch.object(checkpoint, 'publish_reviews') as publish, contextlib.redirect_stdout(io.StringIO()) as output:
                code = checkpoint.main(['--name', 't01v2_checkpoint_fixture', '--source', 'results/scene.json'])
            self.assertEqual(code, 1)
            publish.assert_not_called()
            result = json.loads((Path(folder) / 'results/t01v2_checkpoint_fixture.json').read_text())
            self.assertEqual(result['metrics']['candidate_pairs'], [])
            self.assertEqual(result['metrics']['status'], 'invalid_evidence')
            self.assertIn('FINAL_METRICS_JSON: ', output.getvalue())

    def test_current_look_uses_full_proof_while_legacy_stays_unknown(self):
        from src.envs.pair_v2_solvable_policy import SOLVABILITY_VERSION
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            journal = root / 'journal.jsonl'
            compact = {'pair_id': 'p', 'scene': 'FloorPlan1', 'kind': 'look_flip',
                       'status': 'passed', 'discard_reasons': []}
            journal.write_text(json.dumps(compact) + '\n')
            pair = {**compact, 'scene_preflight_reference': {'sha256': 'same'},
                    'target_w0': 'a', 'target_w1': 'b'}
            source = {'file': 'results/scene.json', 'sha256': 'source', 'metrics_sha256': 'metrics'}
            checked = {'review_image': {'file': 'review.png', 'sha256': 'image'}, 'proposal_identity': 'proposal'}
            for version in (SOLVABILITY_VERSION, None):
                cfg = {'scene': 'FloorPlan1', 'shard': {'pbs_jobid': 'synthetic'},
                       'raw_evidence_dir': str(root), 'solvability_version': version,
                       'solvability_policy': {'synthetic': True}, 'generation_version': 'synthetic-role-version'}
                data = {'_meta': {'exit_code': 0, 'end': 'synthetic-end'}, 'config': cfg,
                        'metrics': {'scene_preflight_reference': {'file': 'raw.json', 'sha256': 'same'}}}
                with patch.object(checkpoint.merge, 'load_source', return_value=(data, source)), \
                        patch.object(checkpoint, 'verified_passed_rows', return_value=(
                            {'eligible': True, 'partial_source': False}, {'file': 'journal'},
                            [(compact, {'file': 'pair.json', 'sha256': 'pair'}, root / 'pair.json')])), \
                        patch.object(checkpoint.merge, 'reference_path', return_value=journal), \
                        patch.object(checkpoint.merge, 'read_reference', return_value=pair), \
                        patch.object(checkpoint.merge, 'check_pair', return_value=checked) as check:
                    _, _, qualified, legacy, _, errors = checkpoint.collect_sources([root / 'scene.json'])
                self.assertEqual(errors, [])
                if version:
                    self.assertEqual(len(qualified), 1)
                    self.assertEqual(legacy, [])
                    self.assertEqual(check.call_args.kwargs['solvability_version'], SOLVABILITY_VERSION)
                    self.assertEqual(check.call_args.kwargs['solvability_policy'], cfg['solvability_policy'])
                    self.assertEqual(check.call_args.kwargs['generation_version'], cfg['generation_version'])
                else:
                    check.assert_not_called()
                    self.assertEqual(qualified, [])
                    self.assertIsNone(legacy[0]['solvability'])


if __name__ == '__main__':
    unittest.main()
