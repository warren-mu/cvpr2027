"""Small synthetic pool tests; these do not generate experimental results."""
import copy
import unittest

from src.envs.pair_table_balance import (balance_pairs, balance_statistics,
    left_half_from_mask, normalize_table, seeded_digest)


def pair(pair_id, kind='look_flip', scene='FloorPlan1', targets=None, left=None):
    return {'pair_id': pair_id, 'scene': scene, 'kind': kind, 'candidate_names': ['a', 'b'],
        'target_names': targets if targets is not None else (['a', 'b'] if kind.endswith('flip') else ['a', 'a']),
        's0_target_left': left if left is not None else [True, False]}


class PairTableBalanceTests(unittest.TestCase):
    def test_pixel_center_left_rule_preserves_midline_and_unknown(self):
        self.assertTrue(left_half_from_mask(bytes([1, 0, 0]), 3, 1)['left'])
        self.assertFalse(left_half_from_mask(bytes([0, 1, 0]), 3, 1)['left'])
        self.assertFalse(left_half_from_mask(bytes([1, 0, 1]), 3, 1)['left'])
        self.assertIsNone(left_half_from_mask(bytes(3), 3, 1)['left'])
        self.assertEqual(left_half_from_mask(bytes(3), 3, 1)['visible_pixels'], 0)
        self.assertIsNone(left_half_from_mask(None, 3, 1)['visible_pixels'])
        with self.assertRaises(ValueError):
            left_half_from_mask(bytes([2, 0, 0]), 3, 1)

    def test_normalize_real_schema_uses_each_world_target_and_preserves_table(self):
        from src.envs.pair_core import ViewState
        from src.envs.paired_env import RGBFrame
        from src.envs.pair_table import PairTable, RenderedView, TableState
        masks = {'a': b'\x01' + bytes(255 * 255 - 1),
                 'b': bytes(254) + b'\x01' + bytes(255 * 255 - 255), 'x': bytes(255 * 255)}
        captures = tuple(RenderedView(RGBFrame(bytes([10]) * (255 * 255 * 3), 255, 255), masks,
            {'a': 1., 'b': 1.}, {'position': {'x': 0, 'y': 0, 'z': 0}},
            'synthetic:one', f'frame:{index}', {'synthetic': True}, True) for index in range(2))
        state = TableState('s0', ViewState('v0', 0), captures, {}, (), ({'synthetic': True}, {'synthetic': True}), True, 0, (0, 0))
        value = PairTable(pair_id='fixture', scene='FloorPlan1', kind='look_flip', instruction='Synthetic task',
            anchor_name='x', candidate_names=('a', 'b'), target_names=('a', 'b'), initial_state='s0',
            positions={'v0': {'x': 0, 'y': 0, 'z': 0}}, states={'s0': state})
        original_targets = value.target_names
        row = normalize_table(value)
        self.assertEqual(row['target_names'], list(original_targets))
        self.assertEqual(value.target_names, original_targets)
        self.assertTrue(row['synthetic'])
        for index, target in enumerate(original_targets):
            capture = value.states[value.initial_state].captures[index]
            expected = left_half_from_mask(capture.masks[target], 255, 255)
            self.assertEqual(row['s0_left_measurements'][index], expected)

    def test_deterministic_pair_id_swap_is_full_world_mapping_and_input_unchanged(self):
        rows = [pair('flip_' + str(index)) for index in range(8)]
        original = copy.deepcopy(rows)
        result = balance_pairs(rows)
        self.assertEqual(rows, original)
        self.assertEqual(result, balance_pairs(list(reversed(rows))))
        for selected in result['selected']:
            digest = seeded_digest(selected['pair_id'], 0, 'flip_world_swap')
            order = [1, 0] if int(digest[:2], 16) & 1 else [0, 1]
            self.assertEqual(selected['world_order'], order)
            self.assertEqual(selected['target_names'], [['a', 'b'][i] for i in order])
            self.assertEqual(selected['s0_target_left'], [[True, False][i] for i in order])

    def test_stay_downsample_is_per_scene_and_pools_classes(self):
        rows = [pair('a1', 'look_stay'), pair('a2', 'direct_stay'), pair('a3', 'look_stay'),
                pair('b1', 'direct_stay', targets=['b', 'b']),
                pair('other_a', 'look_stay', scene='FloorPlan2')]
        result = balance_pairs(rows)
        kept = result['selected']
        self.assertEqual(len(kept), 2)
        self.assertEqual(sorted(row['target_names'][0] for row in kept), ['a', 'b'])
        self.assertEqual({row['scene'] for row in kept}, {'FloorPlan1'})
        self.assertTrue(all(row['world_order'] == [0, 1] for row in kept))
        self.assertEqual(len(result['dropped']), 3)

    def test_selection_never_uses_left_measurement(self):
        rows = [pair('a1', 'look_stay'), pair('a2', 'look_stay'),
                pair('b1', 'look_stay', targets=['b', 'b'])]
        modified = copy.deepcopy(rows)
        for row in modified:
            row['s0_target_left'] = [None, None]
        first, second = balance_pairs(rows), balance_pairs(modified)
        self.assertEqual([row['pair_id'] for row in first['selected']], [row['pair_id'] for row in second['selected']])
        self.assertEqual(first['dropped'], second['dropped'])

    def test_unknown_left_not_zero_and_known_subtotal_is_separate(self):
        result = balance_pairs([pair('f', left=[True, None])])['after']
        self.assertIsNone(result['overall']['target_left_proportion'])
        self.assertEqual(result['overall']['target_left_observed_only_proportion'], 1)
        self.assertEqual(result['overall']['target_left_unknown_world_count'], 1)
        self.assertIsNone(result['global_balance_within_0_4_to_0_6'])
        self.assertEqual(result['overall']['target_a_proportion'], .5)

    def test_rho_half_uses_actual_both_world_targets(self):
        result = balance_statistics([pair('f'), pair('s', 'look_stay', targets=['a', 'a'])], ['FloorPlan1'])
        self.assertEqual(result['overall']['target_a_proportion'], .75)
        self.assertFalse(result['global_balance_within_0_4_to_0_6'])
        self.assertEqual(result['scenes_outside_0_3_to_0_7'][0]['outside_bounds'], ['target_a_proportion'])

    def test_empty_scene_and_pool_remain_unknown(self):
        result = balance_pairs([pair('only_a', 'look_stay')])
        self.assertEqual(result['selected'], [])
        self.assertIsNone(result['after']['overall']['target_a_proportion'])
        self.assertIsNone(result['after']['overall']['target_left_proportion'])
        self.assertIsNone(result['after']['global_balance_within_0_4_to_0_6'])
        self.assertEqual(result['after']['by_scene'][0]['pair_count'], 0)

    def test_duplicate_id_wrong_mode_and_fabricated_boolean_rejected(self):
        for rows in ([pair('same'), pair('same')], [pair('bad', targets=['a', 'a'])], [pair('bad', left=[1, False])]):
            with self.assertRaises(ValueError):
                balance_pairs(rows)

    def test_left_imbalance_is_reported_not_reselected(self):
        rows = [pair('f1', left=[True, True]), pair('f2', left=[True, True])]
        result = balance_pairs(rows)
        self.assertEqual(len(result['selected']), 2)
        self.assertEqual(result['after']['overall']['target_left_proportion'], 1)
        self.assertFalse(result['after']['global_balance_within_0_4_to_0_6'])
        self.assertIsNone(result['full_T02_acceptance'])

    def test_reporting_includes_zero_count_scene_class_without_fake_probability(self):
        stats = balance_pairs([pair('f')])['after']['by_scene_class']
        absent = next(row for row in stats if row['kind'] == 'direct_stay')
        self.assertEqual(absent['pair_count'], 0)
        self.assertIsNone(absent['target_a_proportion'])
        self.assertIsNone(absent['target_left_proportion'])

    def test_balanced_global_threshold_does_not_claim_full_T02(self):
        result = balance_pairs([pair('f1'), pair('f2')])
        self.assertTrue(result['after']['global_balance_within_0_4_to_0_6'])
        self.assertIsNone(result['after']['full_T02_acceptance'])
        self.assertIsNone(result['full_T02_acceptance'])
