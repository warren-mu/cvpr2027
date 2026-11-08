import unittest
from src.tools.t01v2_stable_summary import aggregate, preflight_numbers, KINDS


class StableProgressTests(unittest.TestCase):
    def test_missing_counts_never_become_zero_complete_result(self):
        row = {'scene': 'FloorPlan1', 'reported_passed_by_class': {kind: 0 for kind in KINDS},
               'physical_rejection_distribution': {'edited_position_error': 4, 'other_object_displacement': 2,
                                                    'parent_receptacle_unverified': 1}}
        result = aggregate([row, {'scene': 'FloorPlan2'}], [{'source_file': 'missing'}], [])
        self.assertFalse(result['all_requested_counts_known'])
        self.assertEqual(result['unknown_count_scenes'], ['FloorPlan2'])
        self.assertFalse(result['all_requested_physical_distributions_known'])
        self.assertEqual(result['known_physical_rejection_distribution']['edited_position_error'], 4)
        self.assertIsNone(result['task_acceptance'])

    def test_interrupted_window_reports_observed_data_without_completion(self):
        record = {'passed': False, 'diagnostics': {'initial_settling': {'settled': True,
            'criterion': 'native_rest', 'observations': []}, 'stability_window': {'completed': False,
            'observations': [{'measurements': {'objects': {'a': {'baseline_displacement_m': .015}}}}]}}}
        row = preflight_numbers(record)
        self.assertEqual(row['observed_window_steps'], 1)
        self.assertFalse(row['window_completed'])
        self.assertEqual(row['window_max_displacement_m'], .015)
        self.assertIsNone(row['initial_max_displacement_m'])

    def test_pause_failure_and_missing_observations_are_not_zero(self):
        row = preflight_numbers({'passed': False, 'diagnostics': {'pause_measurements':
            {'objects': {'a': {'baseline_displacement_m': .7}}}}})
        self.assertEqual(row['initial_max_displacement_m'], .7)
        self.assertIsNone(row['observed_initial_steps'])
        self.assertIsNone(row['observed_window_steps'])


if __name__ == '__main__':
    unittest.main()
