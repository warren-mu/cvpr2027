"""Synthetic prefix/progress checks only; no simulator or production journal."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.tools.t01v2_live_progress import (check_expected_shard, complete_prefix,
    expected_identity, observed_counts, read_prefix)


def attempt(index=0, **updates):
    return {'scene': 'FloorPlan1', 'pair_id': f'floorplan1_{index:05d}', 'status': 'rejected',
            'discard_reasons': ['w0_physical'], 'anchor_type': 'Book', 'w0_valid_physical': False, **updates}


def encoded(rows):
    return b''.join(json.dumps(row).encode() + b'\n' for row in rows)


class LiveProgressTests(unittest.TestCase):
    def test_incomplete_last_line_is_not_repaired_or_counted(self):
        whole = encoded([attempt()])
        prefix = complete_prefix(whole + b'{"scene":"FloorPlan1","pair_id":')
        self.assertEqual(prefix, whole)
        self.assertEqual(observed_counts(prefix, 'FloorPlan1')['observed_complete_attempts'], 1)

    def test_measured_boolean_denominator_keeps_missing_separate(self):
        raw = encoded([attempt(), attempt(1, w0_valid_physical=True, discard_reasons=['no_view']),
                       attempt(2, w0_valid_physical=None, discard_reasons=['interrupt']),
                       attempt(3, w0_valid_physical=1, discard_reasons=['unknown'])])
        result = observed_counts(raw, 'FloorPlan1')
        self.assertEqual(result['w0_by_anchor']['Book'],
            {'measured_attempts': 2, 'passed': 1, 'rejected': 1, 'measurement_unavailable': 2})
        self.assertEqual(result['discard_reason_counts']['w0_physical'], 1)

    def test_log_fallback_only_counts_exact_complete_attempt_lines(self):
        row = attempt(status='passed', kind='look_flip', discard_reasons=[], w0_valid_physical=True)
        raw = b'ordinary output\nnot T01V2_ATTEMPT: {}\nT01V2_ATTEMPT: ' + encoded([row])
        result = observed_counts(raw, 'FloorPlan1', log=True)
        self.assertEqual(result['observed_reported_passed_by_class']['look_flip'], 1)
        self.assertNotIn('acceptance_passed', result)
        self.assertEqual(observed_counts(b'no observations\n', 'FloorPlan1', log=True)['observed_complete_attempts'], 0)

    def test_wrong_scene_duplicate_and_malformed_complete_row_fail_closed(self):
        for raw in (encoded([attempt(), attempt()]), encoded([attempt(scene='FloorPlan2')]),
                    b'{broken}\n', encoded([attempt(status='passed', kind=None)])):
            with self.assertRaises(ValueError):
                observed_counts(raw, 'FloorPlan1')

    def test_expected_name_binds_exact_scene_job_and_logical_gpu(self):
        name = 'pilot_pairs_v2_fixture_j707123_hopper_m_02_g1_floorplan201'
        expected = expected_identity(name)
        self.assertEqual(expected['pbs_jobid'], '707123.hopper-m-02')
        self.assertEqual(expected['scene'], 'FloorPlan201')
        check_expected_shard(expected, expected)
        for key, value in (('name', name + '_retry'), ('scene', 'FloorPlan1'),
                           ('pbs_jobid', '707124.hopper-m-02'), ('logical_index', 0)):
            bad = copy.deepcopy(expected)
            bad[key] = value
            with self.assertRaises(ValueError):
                check_expected_shard(expected, bad)
        with self.assertRaises(ValueError):
            expected_identity('../results/unsafe')

    def test_bounded_prefix_snapshot_does_not_touch_source(self):
        data = encoded([attempt(), attempt(1)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.jsonl'
            path.write_bytes(data)
            raw, evidence = read_prefix(path, limit=len(encoded([attempt()])) + 4)
            self.assertTrue(evidence['read_cap_reached'])
            self.assertEqual(raw, encoded([attempt()]))
            self.assertEqual(path.read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
