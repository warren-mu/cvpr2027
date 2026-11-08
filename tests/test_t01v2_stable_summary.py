import unittest
from src.tools.t01v2_stable_summary import aggregate, preflight_numbers, KINDS


class StableProgressTests(unittest.TestCase):
    def test_w0_report_separates_measured_rejection_from_unavailable(self):
        rows = [{'scene': 's1', 'w0_by_anchor': {'Book': {'measured_attempts': 5,
            'passed': 2, 'rejected': 3, 'measurement_unavailable': 4}}},
            {'scene': 's2', 'w0_by_anchor': {'Book': {'measured_attempts': 1,
            'passed': 1, 'rejected': 0, 'measurement_unavailable': 0}}}, {'scene': 's3'}]
        report = aggregate(rows, [], [])
        self.assertEqual(report['known_w0_by_anchor']['Book'], {'measured_attempts': 6,
            'passed': 3, 'rejected': 3, 'measurement_unavailable': 4})
        self.assertEqual(report['unknown_w0_scenes'], ['s3'])
        self.assertFalse(report['all_requested_w0_counts_known'])
        known = aggregate(rows[:2], [], [])
        self.assertTrue(known['all_requested_w0_counts_known'])
        self.assertEqual(known['known_w0_by_anchor']['Book']['measurement_unavailable'], 4)

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
