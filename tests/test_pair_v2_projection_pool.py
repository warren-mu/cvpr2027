"""Synthetic geometry/scheduling only; no Unity or measured qualification."""
import copy
import itertools
import unittest

from src.envs import pair_v2_projection_pool as pools
from src.envs import pair_v2_role_policy as roles


def camera(width=200, height=100, yaw=0, pitch=0):
    return pools.camera_from_metadata({'agent': {'position': {'x': 0., 'y': 0., 'z': 0.},
        'rotation': {'x': 0., 'y': yaw, 'z': 0.}, 'cameraHorizon': pitch},
        'cameraPosition': {'x': 0., 'y': 1., 'z': 0.}, 'fov': 90.,
        'screenWidth': width, 'screenHeight': height}, {'file': 'synthetic.json', 'sha256': '0' * 64})


def token(x, z, label, y=.5):
    point = {'x': x, 'y': y, 'z': z}
    return {'position': point, 'receptacle_name': 'Surface',
        'spawn': {'selected_position': dict(point), 'query_reference': {'file': label, 'sha256': 'a' * 64}}}


def record():
    cam = camera(100, 100)
    a = {'position': {'x': -1., 'y': .5, 'z': 1.}}
    b = {'position': {'x': 1., 'y': .5, 'z': 1.}}
    anchor = {'name': 'X', 'axisAlignedBoundingBox': {'size': {'x': .2, 'y': 1., 'z': .2}}}
    left = [token(-1.3, 1., 'la'), token(-1.4, 1., 'lb'), token(-.5, 1., 'da'), token(-.6, 1., 'db')]
    right = [token(1.3, 1., 'ra'), token(1.4, 1., 'rb'), token(.5, 1., 'ea'), token(.6, 1., 'eb')]
    return pools.build_view_pools(left, right, a, b, anchor, camera=cam,
        view={'position': cam['agent_position'], 'yaw': 0, 'horizon': 0},
        observation_cameras=[pools.camera_after_action(cam, {'kind': kind}) for kind in ('turn_left', 'turn_right')])


class ProjectionPoolTests(unittest.TestCase):
    def test_saved_pool_geometry_references_queries_without_duplicate_coordinate_payload(self):
        class RawCoordinates:
            def __deepcopy__(self, memo):
                raise AssertionError('Native coordinate arrays must not be recopied into the diagnostic pool')
        rec = record()
        raw = RawCoordinates()
        for side in ('A', 'B'):
            for point in rec['diagnostics'][side]:
                point['token']['spawn']['coordinates'] = raw
        stored = pools.pool_evidence_record(rec)
        self.assertEqual(stored['counts'], rec['counts'])
        self.assertEqual(stored['view'], rec['view'])
        self.assertEqual(stored['policy'], rec['policy'])
        for side in ('A', 'B'):
            for original, saved in zip(rec['diagnostics'][side], stored['diagnostics'][side]):
                self.assertIs(original['token']['spawn']['coordinates'], raw)
                self.assertEqual(saved['start'], original['start'])
                self.assertEqual(saved['after_actions'], original['after_actions'])
                self.assertNotIn('spawn', saved['token'])
                self.assertNotIn('coordinates', saved['token']['spawn_reference'])
                self.assertEqual(saved['token']['spawn_reference']['selected_position'],
                                 original['token']['spawn']['selected_position'])
                self.assertEqual(saved['token']['spawn_reference']['query_reference'],
                                 original['token']['spawn']['query_reference'])
            for pool in ('look', 'direct'):
                self.assertEqual([p['token']['position'] for p in stored['points'][pool][side]],
                                 [p['token']['position'] for p in rec['points'][pool][side]])

    def test_actual_vertical_fov_aspect_yaw_pitch_and_camera_height(self):
        cam = camera()
        center = pools.project_point({'x': 0, 'y': 1, 'z': 1}, cam)
        self.assertAlmostEqual(center['u'], 100)
        self.assertAlmostEqual(center['v'], 50)
        self.assertAlmostEqual(pools.project_point({'x': 1, 'y': 1, 'z': 1}, cam)['u'], 150)
        turned = pools.project_point({'x': 1, 'y': 1, 'z': 0}, camera(yaw=90))
        self.assertAlmostEqual(turned['u'], 100)
        downward = pools.project_point({'x': 0, 'y': 0, 'z': 1}, camera(pitch=45))
        self.assertAlmostEqual(downward['v'], 50)

    def test_actual_metadata_missing_or_unsupported_pose_fails(self):
        with self.assertRaises((KeyError, ValueError)):
            pools.camera_from_metadata({'agent': {}}, {'file': 'fixture'})
        cam = camera()
        for p in ({'x': 0, 'y': 1, 'z': 0}, {'x': 0, 'y': 1, 'z': -1}):
            projection = pools.project_point(p, cam)
            self.assertFalse(projection['projectable'])
            self.assertIsNone(projection['outside_edge_margin_px'])

    def test_native_point_unchanged_halfheight_only_projection(self):
        rec = record()
        for side in ('A', 'B'):
            for item in rec['diagnostics'][side]:
                self.assertEqual(item['token']['position'], item['token']['spawn']['selected_position'])
                self.assertEqual(item['token']['position']['y'], .5)
                self.assertEqual(item['projection_point']['y'], 1.)
        self.assertTrue(rec['points']['look']['A'])
        self.assertTrue(rec['points']['direct']['A'])

    def test_margin_and_distance_and_actual_existing_action_are_required(self):
        rec = record(); cam = camera(100, 100)
        a = {'position': {'x': 0, 'y': .5, 'z': 1}}
        anchor = {'name': 'X', 'axisAlignedBoundingBox': {'size': {'y': 1.}}}
        inputs = [token(1.19, 1, 'near-edge'), token(1.3, 1, 'look'), token(4, 1, 'too-far')]
        result = pools.build_view_pools(inputs, [], a, a, anchor, camera=cam, view=rec['view'],
            observation_cameras=[pools.camera_after_action(cam, {'kind': 'turn_right'})])
        self.assertEqual([x['token']['spawn']['query_reference']['file'] for x in result['points']['look']['A']], ['look'])
        none = pools.build_view_pools(inputs, [], a, a, anchor, camera=cam, view=rec['view'], observation_cameras=[])
        self.assertEqual(none['points']['look']['A'], [])
        with self.assertRaises(ValueError):
            pools.camera_after_action(cam, {'kind': 'move', 'index': 1, 'position': a['position']})

    def test_pool_caps_sorting_full_tokens_and_retry_no_fabrication(self):
        rec = record()
        original = copy.deepcopy(rec)
        selected = rec['points']['look']['A'][-1]['token']
        values = pools.placement_alternatives(rec, 'look', 'A', selected)
        self.assertEqual(values[0], selected)
        self.assertEqual(len(values), 2)
        self.assertEqual(pools.placement_alternatives(rec, 'look', 'A', selected,
            exclude_positions=[selected['position']]), values[1:])
        self.assertEqual(rec, original)
        with self.assertRaises(ValueError):
            pools.placement_alternatives(rec, 'look', 'A', selected, max_trials=41)
        many = copy.deepcopy(rec)
        cam = camera(100, 100); a = {'position': {'x': 0., 'y': .5, 'z': 1.}}
        look = [token(1.3 + i / 1000, 1., str(i)) for i in range(15)]
        direct = [token(i / 1000, 1., 'd' + str(i)) for i in range(45)]
        many = pools.build_view_pools(look + direct, [], a, a,
            {'name': 'X', 'axisAlignedBoundingBox': {'size': {'y': 1.}}}, camera=cam,
            view=rec['view'], observation_cameras=[pools.camera_after_action(cam, {'kind': 'turn_right'})])
        self.assertEqual(many['counts']['A']['look'], {'before_cap': 15, 'retained': 12, 'cap': 12})
        self.assertEqual(many['counts']['A']['direct'], {'before_cap': 45, 'retained': 40, 'cap': 40})

    def test_global_pool_alternation_avoids_mode_phase_lock_and_preserves_roles(self):
        rec = record(); records = [rec, copy.deepcopy(rec)]
        records[1]['view']['horizon'] = 30
        stream = pools.PoolProposalIterator(records)
        rows = list(stream)
        self.assertEqual([r[0]['projection_pool']['pool'] for r in rows[:8]], ['look', 'direct'] * 4)
        self.assertEqual({(r[0]['projection_pool']['pool'], r[0]['mode']) for r in rows},
                         set(itertools.product(('look', 'direct'), ('flip', 'stay'))))
        for pool in ('look', 'direct'):
            stays = [r[2]['stay_side'] for r in rows if r[0]['mode'] == 'stay' and r[0]['projection_pool']['pool'] == pool]
            self.assertEqual(stays, ['A', 'B'] * 2)
        for index, (proposal, _, domain) in enumerate(rows):
            _, role = roles.orient_proposal(proposal, 'FloorPlan1', index)
            if proposal['mode'] == 'flip':
                self.assertEqual(role['assignment_digest'], roles.flip_orientation('FloorPlan1', index)['assignment_digest'])
        self.assertTrue(all(value['all_exhausted'] for value in stream.summary()['pools'].values()))

    def test_empty_pool_truthful_fallback_and_no_duplicate_pairs(self):
        rec = record()
        rec['points']['look'] = {'A': [], 'B': []}
        stream = pools.PoolProposalIterator([rec])
        rows = list(stream)
        self.assertTrue(rows)
        self.assertTrue(all(row[0]['projection_pool']['pool'] == 'direct' for row in rows))
        self.assertTrue(rows[0][0]['projection_pool']['fallback_due_to_exhaustion'])
        self.assertEqual(stream.summary()['yielded']['look'], 0)
        self.assertEqual(list(pools.PoolProposalIterator([])), [])


if __name__ == '__main__':
    unittest.main()
