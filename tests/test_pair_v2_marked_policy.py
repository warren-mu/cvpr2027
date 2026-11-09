"""Synthetic marked-object policy only; no production render verification."""
import copy
import unittest
import numpy as np

from src.envs import pair_v2_marked_policy as policy


def obj(name, kind, x):
    return {'name': name, 'objectType': kind, 'position': {'x': x, 'y': 0., 'z': 0.}, 'pickupable': True}


def capture():
    objects = [obj('A', 'Mug', .2), obj('B', 'Book', 1.), obj('C', 'Mug', .7), obj('X', 'Vase', 0.)]
    mask = np.zeros((255, 255), dtype=bool); mask.flat[:250] = True
    return {'rgb': np.ones((255, 255, 3), dtype=np.uint8),
        'metadata': {'objects': objects, 'agent': {'position': {'x': 0., 'y': 0., 'z': 0.}}},
        'masks_by_name': {o['name']: mask.copy() for o in objects}}


class MarkedPolicyTests(unittest.TestCase):
    def test_different_categories_and_unmarked_same_category_do_not_change_target_domain(self):
        objects = capture()['metadata']['objects']
        measured = policy.closest_target(objects, ['A', 'B'], 'X')
        self.assertTrue(measured['passed'])
        self.assertEqual(measured['target_name'], 'A')
        self.assertFalse(measured['same_class'])
        self.assertEqual(measured['candidate_types'], ['Mug', 'Book'])

    def test_ratio_tie_unique_anchor_and_nonrolling_filters_remain(self):
        original = capture()['metadata']['objects']
        tied = copy.deepcopy(original); tied[1]['position']['x'] = -.2
        self.assertIn('nearest_target_tie', policy.closest_target(tied, ['A', 'B'], 'X')['reasons'])
        weak = copy.deepcopy(original); weak[1]['position']['x'] = .29
        self.assertIn('closest_ratio_below_1_5', policy.closest_target(weak, ['A', 'B'], 'X')['reasons'])
        duplicate = original + [obj('Y', 'Vase', 5)]
        self.assertIn('category_only_anchor_not_unique', policy.closest_target(duplicate, ['A', 'B'], 'X')['reasons'])
        rolling = copy.deepcopy(original); rolling[-1]['objectType'] = 'Apple'
        self.assertFalse(policy.closest_target(rolling, ['A', 'B'], 'X')['passed'])

    def test_visible_actual_pairs_same_class_first_and_foreground_requirements(self):
        actual = capture()
        groups = policy.identity_sets_at_view(actual)
        self.assertTrue(groups)
        self.assertEqual(groups[0]['candidate_names'], ['A', 'C'])
        self.assertTrue(groups[0]['same_class'])
        self.assertTrue(any(group['candidate_names'] == ['A', 'B'] for group in groups))
        actual['masks_by_name']['B'].flat[249] = False
        self.assertFalse(any('B' in group['candidate_names'] for group in policy.identity_sets_at_view(actual)))
        actual = capture(); actual['metadata']['objects'][1]['position']['x'] = 1.50001
        self.assertFalse(any('B' in group['candidate_names'] for group in policy.identity_sets_at_view(actual)))

    def test_no_new_stay_displacement_threshold_and_only_two_names(self):
        objects = capture()['metadata']['objects']
        measurement = policy.stay_horizontal_displacement([{'objects': objects}, {'objects': copy.deepcopy(objects)}], 'X')
        self.assertEqual(measurement['distance_m'], 0)
        self.assertIsNone(measurement['minimum_required_m'])
        with self.assertRaises(ValueError):
            policy.closest_target(objects, ['A', 'B', 'C'], 'X')


if __name__ == '__main__':
    unittest.main()
