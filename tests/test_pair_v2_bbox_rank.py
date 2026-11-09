"""Synthetic proposal geometry only; no measured visibility or simulator."""
import copy
import unittest
from unittest.mock import Mock, patch
from src.envs import pair_v2_bbox_rank as rank
from src.envs import pair_v2_projection_pool as pools


def camera(yaw=0):
    return pools.camera_from_metadata({'agent': {'position': {'x': 0., 'y': 0., 'z': 0.},
        'rotation': {'x': 0., 'y': yaw, 'z': 0.}, 'cameraHorizon': 0.},
        'cameraPosition': {'x': 0., 'y': 1., 'z': 0.}, 'fov': 90.,
        'screenWidth': 100, 'screenHeight': 100}, {'file': 'synthetic', 'sha256': '0' * 64})


def point(x, z=2., tag='point'):
    native = {'x': x, 'y': .8, 'z': z}
    return {'projection_point': {'x': x, 'y': 1., 'z': z},
        'token': {'position': native, 'receptacle_name': 'Surface',
            'spawn': {'selected_position': dict(native), 'query_reference': {'file': tag}}},
        'original_tag': tag}


ANCHOR = {'name': 'X', 'axisAlignedBoundingBox': {'size': {'x': .4, 'y': .4, 'z': .4}}}


def actions(cam):
    return [pools.camera_after_action(cam, {'kind': 'turn_right'})]


class BboxRankTests(unittest.TestCase):
    def test_center_outside_does_not_claim_entire_box_hidden(self):
        cam = camera()
        item = point(2.1)
        self.assertFalse(pools.project_point(item['projection_point'], cam)['in_frame'])
        prediction = rank.predict(item, ANCHOR, cam, actions(cam))
        self.assertTrue(prediction['ranking_prediction_known'])
        self.assertLess(prediction['start']['signed_outside_margin_px'], 0)
        self.assertFalse(prediction['start']['whole_rectangle_outside_prediction'])
        self.assertGreater(prediction['start']['clipped_rectangle_area_px2'], 0)

    def test_full_box_outside_precedes_larger_body_that_leaks_into_start(self):
        cam = camera(); leaking, hidden = point(2.1, tag='leaking'), point(2.8, tag='hidden')
        result = rank.rank_look_points([leaking, hidden], ANCHOR, cam, actions(cam))
        self.assertEqual([item['original_tag'] for item in result], ['hidden', 'leaking'])
        self.assertIs(result[0]['token'], hidden['token'])
        self.assertGreater(result[0]['bbox_proposal_prediction']['start']['signed_outside_margin_px'], 0)

    def test_one_step_area_continuously_orders_hidden_points_without_cutoff(self):
        cam = camera(); far, near = point(5.6, 4., 'far'), point(2.8, 2., 'near')
        result = rank.rank_look_points([far, near], ANCHOR, cam, actions(cam))
        self.assertEqual([item['original_tag'] for item in result], ['near', 'far'])
        predictions = [item['bbox_proposal_prediction'] for item in result]
        self.assertGreater(predictions[0]['best_one_step_rectangle_area_px2'],
                           predictions[1]['best_one_step_rectangle_area_px2'])
        self.assertTrue(all(item['start']['whole_rectangle_outside_prediction'] for item in predictions))

    def test_projection_matches_perspective_extrema_and_clips_to_frame(self):
        box = rank.project_box({'x': 0., 'y': 1., 'z': 2.}, {'x': 2., 'y': 2., 'z': 0.}, camera())
        for key, expected in {'left': 25., 'right': 75., 'top': 25., 'bottom': 75.}.items():
            self.assertAlmostEqual(box['rectangle_px'][key], expected)
        self.assertAlmostEqual(box['clipped_rectangle_area_px2'], 2500.)
        large = rank.project_box({'x': 0., 'y': 1., 'z': 2.}, {'x': 20., 'y': 20., 'z': 0.}, camera())
        self.assertAlmostEqual(large['clipped_rectangle_area_px2'], 10000.)

    def test_camera_plane_unknown_is_retained_without_fabricated_zero_area(self):
        cam = camera(); crossing = point(.1, 0., 'unknown'); hidden = point(2.8, tag='known')
        result = rank.rank_look_points([crossing, hidden], ANCHOR, cam, actions(cam))
        self.assertEqual([item['original_tag'] for item in result], ['known', 'unknown'])
        prediction = result[1]['bbox_proposal_prediction']
        self.assertFalse(prediction['ranking_prediction_known'])
        self.assertIsNone(prediction['start']['clipped_rectangle_area_px2'])
        self.assertIs(result[1]['token'], crossing['token'])

    def test_missing_geometry_and_empty_actions_do_not_filter_points(self):
        cam = camera(); entries = [point(2.8, tag='first'), point(3., tag='second')]
        missing = rank.rank_look_points(entries, {'name': 'X'}, cam, actions(cam))
        empty = rank.rank_look_points(entries, ANCHOR, cam, [])
        for result in (missing, empty):
            self.assertEqual([item['original_tag'] for item in result], ['first', 'second'])
            self.assertTrue(all(not item['bbox_proposal_prediction']['ranking_prediction_known'] for item in result))

    def test_signed_margin_breaks_equal_area_ties_and_exact_ties_are_stable(self):
        known = {'ranking_prediction_known': True, 'start': {'whole_rectangle_outside_prediction': True,
            'signed_outside_margin_px': 1.}, 'best_one_step_rectangle_area_px2': 100.}
        safer = copy.deepcopy(known); safer['start']['signed_outside_margin_px'] = 2.
        self.assertLess(rank.rank_key(safer), rank.rank_key(known))
        cam = camera(); a, b = point(2.8, tag='a'), point(2.8, tag='b')
        result = rank.rank_look_points([a, b], ANCHOR, cam, actions(cam))
        self.assertEqual([item['original_tag'] for item in result], ['a', 'b'])

    def test_native_arrays_not_copied_and_full_inventory_stay_direct_unchanged(self):
        class CannotCopy:
            def __deepcopy__(self, memo):
                raise AssertionError('Do not copy native coordinate arrays for ranking')
        cam = camera(); entries = [point(2.1, tag='a'), point(2.8, tag='b'), point(.1, 0., 'c')]
        original = copy.deepcopy(entries)
        for item in entries:
            item['token']['spawn']['coordinates'] = CannotCopy()
        direct, stay = {'A': [], 'B': []}, {'A': entries[:1], 'B': entries[1:]}
        stays_by_pool = {'look': stay, 'direct': direct}
        record = {'camera': cam, 'observation_cameras': actions(cam),
            'points': {'look': {'A': entries, 'B': []}, 'direct': direct},
            'stay_points': stay, 'stay_points_by_pool': stays_by_pool,
            'counts': {'original': 'unchanged'}, 'diagnostics': {'A': entries}}
        output = rank.rank_pool(record, ANCHOR)
        self.assertEqual(sorted(id(item['token']) for item in output['points']['look']['A']),
                         sorted(id(item['token']) for item in entries))
        self.assertIs(output['points']['direct'], direct)
        self.assertIs(output['stay_points'], stay)
        self.assertIs(output['stay_points_by_pool'], stays_by_pool)
        self.assertIs(output['counts'], record['counts'])
        self.assertIs(output['diagnostics'], record['diagnostics'])
        self.assertEqual([item['original_tag'] for item in entries], [item['original_tag'] for item in original])
        self.assertTrue(all('bbox_proposal_prediction' not in item for item in entries))
        for item in entries:
            self.assertEqual(item['token']['position'], item['token']['spawn']['selected_position'])

    def test_deadline_runs_before_each_point_and_propagates_without_mutation(self):
        cam = camera(); a, b = point(2.8, tag='a'), point(3., tag='b')
        record = {'camera': cam, 'observation_cameras': actions(cam),
            'points': {'look': {'A': [a], 'B': [b]}, 'direct': {'A': [], 'B': []}}}
        original = copy.deepcopy(record)
        deadline = Mock(side_effect=[None, TimeoutError('original soft deadline')])
        with patch.object(rank, 'predict', wraps=rank.predict) as prediction:
            with self.assertRaisesRegex(TimeoutError, 'original soft deadline'):
                rank.rank_pool(record, ANCHOR, check_deadline=deadline)
            self.assertEqual(prediction.call_count, 1)
        self.assertEqual(deadline.call_count, 2)
        self.assertEqual(record, original)
        first = Mock(side_effect=TimeoutError('before first point'))
        with patch.object(rank, 'predict') as prediction:
            with self.assertRaisesRegex(TimeoutError, 'before first point'):
                rank.rank_look_points([a], ANCHOR, cam, actions(cam), check_deadline=first)
            prediction.assert_not_called()
        self.assertEqual(record, original)

    def test_explicit_version_and_prediction_scope_no_acceptance_claim(self):
        rank.validate_fields(rank.fields())
        with self.assertRaises(ValueError): rank.validate_fields({})
        modified = rank.fields(); modified['bbox_proposal_rank_policy']['domain'] = 'drop small objects'
        with self.assertRaises(ValueError): rank.validate_fields(modified)
        prediction = rank.predict(point(2.8), ANCHOR, camera(), actions(camera()))
        self.assertEqual(prediction['scope'], rank.POLICY['scope'])
        self.assertNotIn('acceptance_passed', prediction)
        self.assertNotIn('anchor_mask_pixels', prediction)


if __name__ == '__main__':
    unittest.main()
