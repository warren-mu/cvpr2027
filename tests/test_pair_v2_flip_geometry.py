"""Synthetic geometry regressions; predictions are never rendered outcomes."""
import copy
import inspect
import json
import math
import unittest
from unittest.mock import patch

from src.envs import pair_v2_flip_geometry as geometry
from src.envs import pair_v2_projection_pool as pools


def token(x=1.35, y=.9, z=.9, *, query='synthetic-query'):
    point = {'x': x, 'y': y, 'z': z}
    return {'position': point, 'receptacle_name': 'Surface', 'spawn': {
        'action': 'GetSpawnCoordinatesAboveReceptacle', 'success': True, 'scene': 'FloorPlan203',
        'receptacle_name': 'Surface', 'selected_position': dict(point), 'coordinates': [dict(point)],
        'query_reference': {'file': query, 'sha256': 'a' * 64}}}


def setup(*, opposite_edges=False, actions=True):
    camera = pools.camera_from_metadata({'agent': {'position': {'x': 0., 'y': 1., 'z': 0.},
        'rotation': {'x': 0., 'y': 0., 'z': 0.}, 'cameraHorizon': 0.},
        'cameraPosition': {'x': 0., 'y': 1., 'z': 0.}, 'fov': 90.,
        'screenWidth': 100, 'screenHeight': 100}, {'file': 'synthetic-camera', 'sha256': 'b' * 64})
    a = {'name': 'CandidateA', 'position': {'x': -.6 if opposite_edges else .6, 'y': 1., 'z': .9}}
    b = {'name': 'CandidateB', 'position': {'x': .6, 'y': 1. if opposite_edges else 1.85, 'z': .9}}
    anchor = {'name': 'Anchor', 'axisAlignedBoundingBox': {'size': {'x': .2, 'y': .2, 'z': .2}}}
    direct = {'A': [{'unchanged': 'direct-A'}], 'B': []}
    stay = {'A': [{'unchanged': 'historical-stay-A'}], 'B': [{'unchanged': 'historical-stay-B'}]}
    record = {'camera': camera, 'view': {'position': camera['agent_position'], 'yaw': 0, 'horizon': 0},
        'observation_cameras': [pools.camera_after_action(camera, {'kind': kind})
            for kind in ('turn_left', 'turn_right')] if actions else [],
        'points': {'look': {'A': [], 'B': []}, 'direct': direct}, 'stay_points': stay,
        'diagnostics': {'A': [], 'B': []}, 'counts': {'A': {'stay_retained': 1}, 'B': {'stay_retained': 1}}}
    return record, a, b, anchor


def filtered(values=None, **kwargs):
    record, a, b, anchor = setup(**kwargs)
    values = [token(), token(y=1.75)] if values is None else values
    return geometry.filter_pool(record, a, b, anchor, all_native_tokens=values, check_deadline=lambda: None)


class CompleteGeometryTests(unittest.TestCase):
    def test_complete_native_carriers_do_not_inherit_old_horizontal_pruning(self):
        values = [token(), token(y=1.75)]
        record, a, b, anchor = setup()
        historical_a, historical_b = [], [values[0]]
        calls = []
        def old(points, ca, cb, x, **kwargs):
            calls.append(tuple(points)); return historical_a, historical_b
        left, right = geometry.placement_groups(old, values, a, b, anchor, maximum_per_side=2)
        self.assertEqual(calls, [tuple(values)])
        self.assertIs(left.stay_values, historical_a); self.assertIs(right.stay_values, historical_b)
        self.assertEqual(list(geometry.original_tokens(left, right)), values)
        self.assertTrue(all(first is second for first, second in zip(left, values)))
        outcome = geometry.filter_pool(record, a, b, anchor,
            all_native_tokens=geometry.original_tokens(left, right), check_deadline=lambda: None)
        self.assertEqual(outcome['flip_geometry_counts']['retained_by_distance_role'], {'A': 1, 'B': 1})
        left.pop()
        with self.assertRaises(ValueError): geometry.original_tokens(left, right)
        with self.assertRaises(ValueError): geometry.original_tokens(values, values)

    def test_candidates_and_both_anchor_points_can_share_one_screen_side(self):
        record, a, b, _ = setup()
        for candidate in (a, b):
            projected = pools.project_point(candidate['position'], record['camera'])
            self.assertTrue(projected['in_frame']); self.assertGreater(projected['u'], 50)
            self.assertLessEqual(math.dist(list(candidate['position'].values()),
                list(record['camera']['agent_position'].values())), 1.5)
        outcome = filtered()
        first, second = (outcome['points']['look'][side][0]['token'] for side in ('A', 'B'))
        proof = geometry.complete_pair(outcome, first, second)
        self.assertTrue(proof['proposal_tuple_passed'])
        for world in proof['worlds'].values():
            start = world['bbox_prediction']['start']
            self.assertGreater(start['rectangle_px']['left'], 100)
            self.assertGreaterEqual(start['signed_outside_margin_px'], 10)
            self.assertTrue(world['visible_one_step_action_indices'])
        self.assertIs(outcome['points']['look']['A'][0]['token'], first)
        self.assertNotIn('acceptance_passed', proof)

    def test_anchor_center_ratio_can_pass_when_native_surface_point_ratio_fails(self):
        record, a, b, anchor = setup()
        native = token(y=1.75)
        old_role, _ = geometry.predicted_role(native['position'], a['position'], b['position'])
        self.assertIsNone(old_role)
        prediction = geometry.predict(native, a, b, anchor, record['camera'], record['observation_cameras'])
        self.assertTrue(prediction['proposal_geometry_passed'])
        self.assertEqual(prediction['distance_role'], 'B')
        self.assertAlmostEqual(prediction['predicted_anchor_center']['y'], 1.85)
        self.assertEqual(native['position'], native['spawn']['selected_position'])
        self.assertIn('not physical or settled measurement', prediction['center_definition'])

    def test_point_outside_does_not_substitute_for_whole_bbox_margin(self):
        record, a, b, anchor = setup(); leaking = token(x=1.2)
        center = {**leaking['position'], 'y': 1.}
        self.assertGreaterEqual(pools.project_point(center, record['camera'])['outside_edge_margin_px'], 10)
        outcome = geometry.filter_pool(record, a, b, anchor,
            all_native_tokens=[leaking], check_deadline=lambda: None)
        row = outcome['flip_geometry_prefilter'][0]
        self.assertFalse(row['proposal_geometry_passed'])
        self.assertIn('whole_start_bbox_not_outside_by10px', row['reasons'])
        self.assertLess(row['start_bbox']['signed_outside_margin_px'], 10)
        self.assertEqual(outcome['points']['look'], {'A': [], 'B': []})

    def test_each_world_may_use_a_different_known_one_step_camera(self):
        record = filtered([token(x=-1.35), token()], opposite_edges=True)
        first, second = (record['points']['look'][side][0]['token'] for side in ('A', 'B'))
        proof = geometry.complete_pair(record, first, second)
        visible = [set(world['visible_one_step_action_indices']) for world in proof['worlds'].values()]
        self.assertTrue(all(visible)); self.assertFalse(visible[0] & visible[1])
        swapped = geometry.complete_pair(record, second, first, ('B', 'A'))
        self.assertEqual(swapped['source_sides'], ['B', 'A'])
        self.assertEqual(swapped['worlds']['w0']['distance_role'], 'B')
        with self.assertRaises(ValueError): geometry.complete_pair(record, first, second, ('B', 'A'))
        with self.assertRaises(ValueError): geometry.complete_pair(record, first, first, ('A', 'A'))

    def test_unknown_boxes_empty_actions_and_original_range_do_not_pass(self):
        no_actions = filtered(actions=False)
        self.assertTrue(all('no_known_positive_area_one_step_bbox' in row['reasons']
                            for row in no_actions['flip_geometry_prefilter']))
        unknown = filtered([token(z=-.9)])['flip_geometry_prefilter'][0]
        self.assertFalse(unknown['proposal_geometry_passed'])
        self.assertIn('start_bbox_projection_unknown', unknown['reasons'])
        self.assertIsNone(unknown['start_bbox']['signed_outside_margin_px'])
        far = filtered([token(x=4.)])['flip_geometry_prefilter'][0]
        self.assertIn('original_native_point_agent_range_exceeds3m', far['reasons'])
        record, a, b, anchor = setup(); anchor['axisAlignedBoundingBox']['size']['y'] = float('nan')
        value = geometry.filter_pool(record, a, b, anchor, all_native_tokens=[token()], check_deadline=lambda: None)
        self.assertIn('unknown_geometric_metadata', value['flip_geometry_prefilter'][0]['reasons'])
        self.assertEqual(value['points']['look'], {'A': [], 'B': []})

    def test_all_input_occurrences_and_reasons_are_compact_and_stay_direct_unchanged(self):
        record, a, b, anchor = setup()
        good, bad = token(), token(x=1.2)
        values = [good, token(y=1.75), bad, copy.deepcopy(good)]
        outcome = geometry.filter_pool(record, a, b, anchor, all_native_tokens=values, check_deadline=lambda: None)
        self.assertEqual(outcome['flip_geometry_counts']['input_occurrences'], 4)
        self.assertEqual([row['input_index'] for row in outcome['flip_geometry_prefilter']], [0, 1, 2, 3])
        self.assertTrue(outcome['flip_geometry_prefilter'][3]['duplicate_native_key'])
        self.assertFalse(outcome['flip_geometry_prefilter'][3]['retained_in_flip_domain'])
        self.assertIn('whole_start_bbox_not_outside_by10px', outcome['flip_geometry_prefilter'][2]['reasons'])
        encoded = json.dumps(outcome['flip_geometry_prefilter'], allow_nan=False)
        self.assertNotIn('"coordinates"', encoded)
        self.assertNotIn('bbox_proposal_rank_policy', encoded)
        self.assertNotIn('candidate_positions', encoded)
        self.assertNotIn('after_actions', encoded)
        self.assertIs(outcome['stay_points'], record['stay_points'])
        self.assertIs(outcome['points']['direct'], record['points']['direct'])
        self.assertIs(outcome['diagnostics'], record['diagnostics'])
        self.assertEqual(record['points']['look'], {'A': [], 'B': []})
        self.assertEqual(record['counts']['A'], {'stay_retained': 1})

    def test_complete_pairs_is_lazy_and_seed_swapping_remains_available(self):
        record = filtered([token(), token(x=1.351), token(y=1.75), token(x=1.351, y=1.75)])
        native = geometry.complete_pair
        with patch.object(geometry, 'complete_pair', wraps=native) as checked:
            stream = geometry.complete_pairs(record, ('B', 'A'))
            self.assertTrue(inspect.isgenerator(stream)); self.assertEqual(checked.call_count, 0)
            _, _, proof = next(stream)
            self.assertEqual(checked.call_count, 1)
            self.assertEqual(proof['source_sides'], ['B', 'A'])

    def test_retry_domain_bound_and_actual_native_point_guard(self):
        values = [token(x=1.35 + index / 10000) for index in range(45)] + [token(y=1.75)]
        record = filtered(values)
        first, second = values[5], values[-1]
        retries = geometry.retry_points(record, 'A', first)
        self.assertEqual(len(retries), 40); self.assertIs(retries[0], first)
        for value in (retries[0], retries[-1]):
            geometry.validate_placement(record, value['spawn'], 'A', opposite=second, opposite_role='B')
        with self.assertRaises(ValueError):
            geometry.validate_placement(record, token(x=1.2)['spawn'], 'A', opposite=second, opposite_role='B')
        with self.assertRaises(ValueError):
            geometry.validate_placement(record, first, 'A', opposite=first, opposite_role='A')
        with self.assertRaises(TypeError): geometry.validate_placement(record, first, 'A')
        with self.assertRaises(ValueError): geometry.retry_points(record, 'A', first, max_trials=41)
        altered = copy.deepcopy(first); altered['spawn']['query_reference']['file'] = 'another-query'
        with self.assertRaises(ValueError):
            geometry.validate_placement(record, altered, 'A', opposite=second, opposite_role='B')
        changed_camera = copy.deepcopy(record); changed_camera['camera'] = pools.camera_after_action(
            record['camera'], {'kind': 'turn_right'})['camera']
        with self.assertRaises(ValueError):
            geometry.validate_placement(changed_camera, first, 'A', opposite=second, opposite_role='B')

    def test_soft_deadline_propagates_without_partial_pool_or_input_mutation(self):
        record, a, b, anchor = setup(); before = copy.deepcopy(record)
        def stop():
            raise TimeoutError('original deadline')
        with self.assertRaisesRegex(TimeoutError, 'original deadline'):
            geometry.filter_pool(record, a, b, anchor, all_native_tokens=[token()], check_deadline=stop)
        self.assertEqual(record, before)
        with self.assertRaises(ValueError):
            geometry.filter_pool(record, a, b, anchor, all_native_tokens=[], check_deadline=None)
        geometry.validate_fields(geometry.fields())
        changed = geometry.fields(); changed['flip_geometry_policy']['distance'] = 'screen left/right'
        with self.assertRaises(ValueError): geometry.validate_fields(changed)


if __name__ == '__main__':
    unittest.main()
