"""Synthetic CPU contracts only; none of these fixtures verifies a real cache."""
from dataclasses import replace
from functools import lru_cache
import unittest
from unittest.mock import patch

from src.envs.paired_env import RGBFrame
from src.envs.pair_table import RenderedView, label_assignment
from src.envs.pair_table_cache import (CACHE_POLICY, CacheSample, _mask_check,
    check_cached_state, require_cache_validation, sample_state_keys, validate_cache_sample)

PIXELS = 255 * 255
BASE_RGB = RGBFrame(bytes([10]) * (PIXELS * 3), 255, 255)


@lru_cache(maxsize=64)
def mask(count, offset=0):
    return bytes(offset) + bytes([1]) * count + bytes(PIXELS - offset - count)


def sample(pair_id='pair0', state_id='s0', *, anchor_count=0):
    assignments = label_assignment(pair_id, ('a', 'b'))
    def view(world, fresh):
        return RenderedView(BASE_RGB,
            {'x': mask(anchor_count), 'a': mask(500, 1000), 'b': mask(500, 2000)},
            {'a': 1.0, 'b': 1.0}, {'position': {'x': 0, 'y': 1, 'z': 0}, 'yaw': 0},
            'fresh:start9' if fresh else 'cached:start1', f'{pair_id}:{state_id}:{world}:{fresh}',
            {'pair_id': pair_id, 'state_id': state_id, 'world_id': world,
             'snapshot_sha256': ('0' if world == 'w0' else '1') * 64}, synthetic=True)
    return CacheSample(pair_id, state_id, 'x', ('a', 'b'),
        (view('w0', False), view('w1', False)), (view('w0', True), view('w1', True)),
        (dict(assignments), dict(assignments)), (dict(assignments), dict(assignments)))


def change_view(item, which, world, **changes):
    views = list(getattr(item, which))
    views[world] = replace(views[world], **changes)
    return replace(item, **{which: tuple(views)})


class CacheMaskTests(unittest.TestCase):
    def test_exact_two_percent_boundary_without_ceil(self):
        self.assertTrue(_mask_check(mask(500), mask(510))['passed'])
        self.assertFalse(_mask_check(mask(500), mask(511))['count_passed'])
        self.assertTrue(_mask_check(mask(501), mask(511))['count_passed'])
        self.assertFalse(_mask_check(mask(501), mask(512))['count_passed'])
        # A symmetric max-count denominator would incorrectly allow this increase.
        self.assertFalse(_mask_check(mask(990), mask(1010))['count_passed'])

    def test_iou_boundary_and_positive_vs_empty(self):
        self.assertTrue(_mask_check(mask(19), mask(19, 1))['iou_passed'])  # 18/20
        self.assertFalse(_mask_check(mask(18), mask(18, 1))['iou_passed'])
        absent = _mask_check(mask(0), mask(1))
        self.assertTrue(absent['count_passed'])
        self.assertFalse(absent['passed'])  # 10-pixel count allowance cannot hide a new mask.
        empty = _mask_check(mask(0), mask(0))
        self.assertIsNone(empty['iou'])
        self.assertFalse(empty['iou_required'])
        self.assertTrue(empty['passed'])

    def test_equal_counts_displaced_mask_fails(self):
        result = _mask_check(mask(300), mask(300, 400))
        self.assertTrue(result['count_passed'])
        self.assertFalse(result['passed'])


class CacheStateTests(unittest.TestCase):
    def test_complete_both_world_fixture_does_not_certify_live(self):
        result = check_cached_state(sample())
        self.assertTrue(result['checks_passed'])
        self.assertIsNone(result['cache_matches_live'])
        self.assertEqual(set(result['worlds']), {'w0', 'w1'})
        self.assertEqual(set(result['worlds']['w1']['masks']), {'x', 'a', 'b'})

    def test_large_same_world_rgb_difference_is_diagnostic(self):
        item = sample(anchor_count=250)
        for world in (0, 1):
            item = change_view(item, 'fresh', world, rgb=RGBFrame(bytes([200]) * (PIXELS * 3), 255, 255))
        result = check_cached_state(item)
        self.assertTrue(result['checks_passed'])
        self.assertEqual(result['worlds']['w0']['cache_max_abs_diff'], 190)

    def test_same_process_classification_still_rejects_nonanchor_divergence(self):
        item = change_view(sample(), 'fresh', 1, rgb=RGBFrame(bytes([14]) * (PIXELS * 3), 255, 255))
        result = check_cached_state(item)
        self.assertFalse(result['checks_passed'])
        self.assertIn('fresh_non_anchor_divergent', result['reasons'])

    def test_one_world_mask_failure_blocks_state(self):
        item = sample(anchor_count=250)
        masks = dict(item.fresh[1].masks, b=mask(500, 4000))
        result = check_cached_state(change_view(item, 'fresh', 1, masks=masks))
        self.assertFalse(result['checks_passed'])
        self.assertIn('w1:mask_mismatch:b', result['reasons'])

    def test_label_eligibility_changes_are_not_ignored(self):
        item = sample()
        item = change_view(item, 'fresh', 1, distances={'a': 1.5001, 'b': 1.0})
        labels = ({**item.fresh_labels[0]}, {'b': item.fresh_labels[1]['b']})
        result = check_cached_state(replace(item, fresh_labels=labels))
        self.assertFalse(result['checks_passed'])
        self.assertIn('w1:label_set_changed', result['reasons'])
        with self.assertRaises(ValueError):
            check_cached_state(replace(item, fresh_labels=item.cached_labels))

    def test_missing_masks_or_distances_are_unknown(self):
        item = sample()
        masks = {name: value for name, value in item.fresh[0].masks.items() if name != 'x'}
        with self.assertRaises(ValueError):
            check_cached_state(change_view(item, 'fresh', 0, masks=masks))
        with self.assertRaises(ValueError):
            check_cached_state(change_view(item, 'fresh', 0, distances={'a': None, 'b': 1.0}))

    def test_new_single_process_and_same_camera_are_required(self):
        item = sample()
        for changes in ({'process_id': item.cached[0].process_id},
                        {'process_id': 'other_fresh:start10'}, {'agent': {'yaw': 1}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                check_cached_state(change_view(item, 'fresh', 1, **changes))
        with self.assertRaises(ValueError):
            check_cached_state(change_view(item, 'fresh', 1, render_id=item.fresh[0].render_id))

    def test_capture_binding_rejects_wrong_world_or_snapshot(self):
        item = sample()
        for source in (dict(item.fresh[1].source, world_id='w0'),
                       dict(item.fresh[1].source, snapshot_sha256='a' * 64),
                       dict(item.fresh[1].source, state_id='another')):
            with self.subTest(source=source), self.assertRaises(ValueError):
                check_cached_state(change_view(item, 'fresh', 1, source=source))


class CacheSampleTests(unittest.TestCase):
    def setUp(self):
        self.population = [('pair', f's{index}') for index in range(205)]
        self.plan = sample_state_keys(self.population, seed=0)
        self.records = [sample(*key) for key in self.plan['selected_keys']]

    def measured(self, record):
        # This fixture isolates aggregation, not mask/render verification.
        return {'pair_id': record.pair_id, 'state_id': record.state_id,
                'checks_passed': True, 'cache_matches_live': None,
                'synthetic': True, 'reasons': [], 'worlds': {}}

    def test_fixed_sample_and_minimum_distinct_domain(self):
        self.assertEqual(self.plan, sample_state_keys(reversed(self.population), seed=0))
        self.assertEqual(len({tuple(k) for k in self.plan['selected_keys']}), 200)
        for population, size in ((self.population[:199], 200), (self.population, 199),
                                 (self.population + [self.population[0]], 200)):
            with self.assertRaises(ValueError):
                sample_state_keys(population, seed=0, sample_size=size)

    def test_synthetic_200_states_cannot_authorize_downstream(self):
        with patch('src.envs.pair_table_cache.check_cached_state', side_effect=self.measured):
            result = validate_cache_sample(self.records, population=self.population, sampling_plan=self.plan)
        self.assertTrue(result['sample_complete'])
        self.assertTrue(result['checks_passed'])
        self.assertFalse(result['downstream_allowed'])
        self.assertIsNone(result['cache_matches_live'])
        with self.assertRaises(ValueError):
            require_cache_validation(result)

    def test_one_failure_blocks_whole_sample_without_filtering(self):
        def measured(record):
            result = self.measured(record)
            if record is self.records[-1]:
                result.update(checks_passed=False, reasons=['w1:mask_mismatch:x'])
            return result
        with patch('src.envs.pair_table_cache.check_cached_state', side_effect=measured):
            result = validate_cache_sample(self.records, population=self.population, sampling_plan=self.plan)
        self.assertFalse(result['checks_passed'])
        self.assertTrue(result['stop_downstream'])
        self.assertEqual(len(result['states']), 200)
        self.assertFalse(result['states'][-1]['checks_passed'])

    def test_missing_or_duplicate_or_replaced_state_never_qualifies(self):
        with patch('src.envs.pair_table_cache.check_cached_state', side_effect=self.measured):
            missing = validate_cache_sample(self.records[:-1], population=self.population, sampling_plan=self.plan)
        self.assertFalse(missing['sample_complete'])
        self.assertFalse(missing['checks_passed'])
        with self.assertRaises(ValueError):
            validate_cache_sample(self.records + self.records[:1], population=self.population, sampling_plan=self.plan)
        changed = dict(self.plan, population_sha256='f' * 64)
        with self.assertRaises(ValueError):
            validate_cache_sample(self.records, population=self.population, sampling_plan=changed)


if __name__ == '__main__':
    unittest.main()
