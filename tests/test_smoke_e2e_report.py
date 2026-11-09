"""Synthetic CPU checks of E2E report denominators and missing-run handling."""
import unittest

from src.tools.smoke_e2e_report import summarize_decisions, summarize_training_updates


class E2EReportTests(unittest.TestCase):
    def test_shared_physical_sample_keeps_both_logical_world_decisions(self):
        result = summarize_decisions([
            {'decision_id': 'pair1_w0:0', 'sample_id': 'shared', 'response': 'select 1', 'available': True},
            {'decision_id': 'pair1_w1:0', 'sample_id': 'shared', 'response': 'select 1', 'available': True},
            {'decision_id': 'pair2_w0:0', 'sample_id': 'independent', 'response': 'select 1', 'available': True}])
        self.assertEqual(result['logical_model_decisions'], 3)
        self.assertEqual(result['physical_model_outputs'], 2)
        self.assertEqual(result['physical_syntax_parse_failure_rate'], 0.)

    def test_syntax_failure_and_valid_unavailable_action_are_separate(self):
        result = summarize_decisions([
            {'decision_id': 'a:0', 'sample_id': 's0', 'response': 'I choose select 1', 'available': False},
            {'decision_id': 'a:1', 'sample_id': 's1', 'response': 'select 99', 'available': False},
            {'decision_id': 'b:0', 'sample_id': 's2', 'response': ' select 1\n', 'available': True}])
        self.assertEqual(result['logical_syntax_parse_failures'], 1)
        self.assertEqual(result['syntax_valid_but_unavailable_logical_actions'], 1)
        self.assertEqual(result['physical_syntax_parse_failure_rate'], 1 / 3)
        self.assertEqual(result['parse_failure_decision_ids'], ['a:0'])
        self.assertEqual(result['syntax_valid_unavailable_decision_ids'], ['a:1'])

    def test_unfinished_update_does_not_acquire_imputed_time(self):
        result = summarize_training_updates([
            {'kind': 'update_begin', 'update': 1},
            {'kind': 'decision', 'update': 1, 'episode_id': 'a', 'decision_index': 0,
             'sample_id': 's0', 'response': 'select 1', 'after': {'action_valid': True}},
            {'kind': 'update_failed', 'update': 1},
            {'kind': 'update_begin', 'update': 2}])
        self.assertFalse(result['1']['completed'])
        self.assertIsNone(result['1']['timing_seconds'])
        self.assertEqual(result['1']['physical_model_outputs'], 1)
        self.assertIsNone(result['2']['timing_seconds'])
        self.assertIsNone(result['2']['physical_syntax_parse_failure_rate'])


if __name__ == '__main__':
    unittest.main()
