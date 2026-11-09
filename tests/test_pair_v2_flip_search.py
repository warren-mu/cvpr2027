"""CPU behavior fixtures for new-only Flip domains, quota, labels and sources."""
import copy
import sys
import unittest
from unittest.mock import patch
from src.envs import pair_v2_flip_search as search
from src.envs import pair_v2_flip_policy as policy
from src.envs import pair_v2_flip_shard as shard
from src.envs import pair_v2_projection_pool as pools
from src.envs import pair_v2_role_policy as roles
from src.envs import pair_v2_stay_order as stay
from src.envs import pair_v2_fixed_labels as labels
from src.tools import t01v2_flip_parallel as parallel
from src.tools import t01v2_flip_review_checkpoint as review


def token(x, y=0., z=1.):
    point = {'x': x, 'y': y, 'z': z}
    return {'position': point, 'receptacle_name': 'Surface', 'spawn': {
        'selected_position': dict(point), 'receptacle_name': 'Surface',
        'coordinates': [dict(point)], 'query_reference': {'file': str((x, y, z)), 'sha256': 'a' * 64}}}


def config(scene='FloorPlan204'):
    fields = {**policy.camera_fields(60, scene), **stay.fields(), 'producer_module': policy.PRODUCER}
    return {**fields, 'scene': scene, 'raw_evidence_dir': '/synthetic/raw', 'shard': {**fields, 'scene': scene}}


def record():
    left, right = [token(-2.), token(-2.2), token(-2.4)], [token(2.), token(2.2), token(2.4)]
    return {'version': pools.POOL_VERSION, 'view': {}, 'counts': {},
        'flip_search_version': search.VERSION,
        'points': {'look': {'A': [{'token': value} for value in left], 'B': [{'token': value} for value in right]},
                   'direct': {'A': [], 'B': []}},
        'stay_points': {'A': [{'token': value} for value in left], 'B': [{'token': value} for value in right]}}


def argv():
    return ['--name', 'pilot_pairs_v2_synthetic', '--gpus', '1', '--scenes', 'FloorPlan204',
        '--field-of-view', '60', '--timeout-seconds', '1350', '--scene-timeout-seconds', '1200',
        '--gate-timeout-seconds', '120']


class FlipSearchTests(unittest.TestCase):
    def test_full_native3d_domain_keeps_points_rejected_by_horizontal_geometry(self):
        a = {'position': {'x': 0., 'y': 0., 'z': 1.}}
        b = {'position': {'x': 0., 'y': 2., 'z': 1.}}
        points = [token(0., .1), token(0., 1.9)]
        original = copy.deepcopy(points)
        left, right = search.placement_groups(lambda *args, **kw: ([], []), points, a, b, {}, maximum_per_side=1)
        self.assertEqual(left, [points[0]]); self.assertEqual(right, [points[1]])
        self.assertIs(left[0], points[0]); self.assertEqual(points, original)
        self.assertEqual(left.stay_values, [])
        self.assertIsNone(search.side_for_point({'x': 0, 'y': 1, 'z': 1}, [a['position'], b['position']]))

    def test_full_flip_pool_has_no_twelve_point_cap_and_compact_native_refs(self):
        a, b = {'position': {'x': -1., 'y': 0., 'z': 1.}}, {'position': {'x': 1., 'y': 0., 'z': 1.}}
        anchor = {'name': 'X', 'axisAlignedBoundingBox': {'size': {'y': .2}}}
        metadata = {'agent': {'position': {'x': 0, 'y': 0, 'z': 0}, 'rotation': {'x': 0, 'y': 0, 'z': 0}, 'cameraHorizon': 0},
            'cameraPosition': {'x': 0, 'y': 0, 'z': 0}, 'fov': 60, 'screenWidth': 255, 'screenHeight': 255}
        camera = pools.camera_from_metadata(metadata, {'file': 'native.json'})
        left = [token(-2. - i / 100) for i in range(20)]
        right = [token(2. + i / 100) for i in range(20)]
        values = search.full_pool(shard._ORIGINAL_POOLS,
            search.PointGroup(left, left), search.PointGroup(right, right), a, b, anchor,
            camera=camera, view={}, observation_cameras=[pools.camera_after_action(camera, {'kind': kind})
                                                       for kind in ('turn_left', 'turn_right')])
        self.assertEqual(len(values['points']['look']['A']), 20)
        self.assertEqual(len(values['stay_points']['A']), 12)
        self.assertIs(values['points']['look']['A'][0]['token'], left[0])
        saved = search.compact_pool(shard._ORIGINAL_POOL_RECORD, values)
        self.assertIn('spawn_reference', saved['points']['look']['A'][0]['token'])
        self.assertNotIn('spawn', saved['stay_points']['A'][0]['token'])
        self.assertNotIn('coordinates', saved['stay_points']['A'][0]['token']['spawn_reference'])
        self.assertIn('coordinates', values['points']['look']['A'][0]['token']['spawn'])

    def test_scene_quota_survives_stream_restarts_and_ignores_native_failures(self):
        quota = search.AttemptQuota()
        first = search.FlipIterator([record()], quota)
        self.assertEqual(next(first)[0]['mode'], 'flip')
        quota.record('/first', {'counts_as_pair_attempt': False, 'mode': 'flip'})
        second = search.FlipIterator([record()], quota)  # another Controller/domain
        self.assertEqual(next(second)[0]['mode'], 'flip')
        quota.record('/second', {'counts_as_pair_attempt': True, 'mode': 'flip'})
        self.assertEqual(next(first)[0]['mode'], 'stay')
        quota.record('/third', {'counts_as_pair_attempt': True, 'mode': 'stay'})
        self.assertEqual(next(second)[0]['mode'], 'flip')
        quota.record('/third', {'counts_as_pair_attempt': True, 'mode': 'stay'})
        self.assertEqual((quota.flip, quota.stay), (1, 1))
        with self.assertRaises(ValueError):
            quota.record('/unearned', {'counts_as_pair_attempt': True, 'mode': 'stay'})

    def test_one_sided_pool_cannot_substitute_stay_for_required_flip(self):
        value = record(); value['points']['look']['B'] = []
        stream = search.FlipIterator([value], search.AttemptQuota())
        with self.assertRaises(StopIteration):
            next(stream)

    def test_stay_side_alternation_and_far_first_inventory_preserved(self):
        quota = search.AttemptQuota(); stream = search.FlipIterator([record()], quota)
        modes = []
        for index in range(4):
            proposal, _, domain = next(stream)
            modes.append((proposal['mode'], domain['stay_side']))
            quota.record(str(index), {'counts_as_pair_attempt': True, 'mode': proposal['mode']})
        self.assertEqual(modes, [('flip', None), ('stay', 'A'), ('flip', None), ('stay', 'B')])

    def test_edge_priority_orders_opposite_edges_without_mutating_geometry(self):
        camera = {'camera_position': {'x': 0, 'y': 0, 'z': 0}, 'yaw_degrees': 0,
            'horizon_degrees': 0, 'vertical_fov_degrees': 60, 'width': 255, 'height': 255}
        obj = lambda x: {'position': {'x': x, 'y': 0, 'z': 1}}
        self.assertLess(search.edge_rank(obj(-.5), obj(.5), camera), search.edge_rank(obj(-.1), obj(.1), camera))
        self.assertLess(search.edge_rank(obj(-.1), obj(.1), camera), search.edge_rank(obj(.1), obj(.5), camera))

    def test_explicit_native3d_role_keeps_seed_and_rejects_tampering(self):
        positions = {'A': {'x': 0., 'y': 0., 'z': 1.}, 'B': {'x': 0., 'y': 2., 'z': 1.}}
        original = roles.RoleProposalIterator([token(0, .1)], [token(0, 1.9)], [{}])
        proposal, _, domain = next(original); proposal['role_domain'] = domain
        proposal, assignment = roles.orient_proposal(proposal, 'FloorPlan204', 0)
        assignment.update(candidate_a_name='A', candidate_b_name='B')
        row = {'pair_id': 'floorplan204_00000', 'scene': 'FloorPlan204', 'mode': 'flip',
            'identities': {'candidate_names': ['A', 'B']}, 'proposal': proposal, 'proposal_role': assignment,
            'generation_version': roles.ROLE_VERSION, 'generation_policy': copy.deepcopy(roles.ROLE_POLICY),
            'flip_search_version': search.VERSION, 'flip_search_policy': copy.deepcopy(search.POLICY),
            'target_w0': assignment['w0_source_side'], 'target_w1': assignment['w1_source_side']}
        self.assertTrue(search.verify_flip_role(row, positions)['passed'])
        with self.assertRaises(ValueError):
            roles.verify_proposal_role(row, positions)  # historical geometry stays strict
        row['proposal_role']['assignment_digest'] = 'altered'
        with self.assertRaises(ValueError):
            search.verify_flip_role(row, positions)

    def test_single_selected_scene_and_camera_binding_rejects_mixed_or_truncated_plan(self):
        policy.validate_source(config(), {**config(), 'shard': None})
        parallel.validate_plan(argv())
        for flag, value in (('--gpus', '3'), ('--field-of-view', '100'), ('--scene-timeout-seconds', '1199')):
            values = argv(); values[values.index(flag) + 1] = value
            with self.assertRaises(ValueError): parallel.validate_plan(values)
        values = argv(); values.insert(values.index('--field-of-view'), 'FloorPlan203')
        with self.assertRaises(ValueError): parallel.validate_plan(values)
        cfg = config(); cfg['shard']['flip_unit_scene'] = 'FloorPlan203'
        with self.assertRaises(ValueError): policy.validate_source(cfg)

    def test_parent_refuses_scheduler_partial_grant_before_child_launch(self):
        def main():
            command = ['python3', '-m', policy.PRODUCER, '--timeout-seconds', '1199']
            with self.assertRaises(RuntimeError):
                parallel.original.launch_logged('unit', command, {}, '/scratch')
            return 0
        with patch.object(sys, 'argv', ['module', *argv()]), patch.object(parallel.marked, 'main', side_effect=main), \
                patch.object(parallel.original, 'launch_logged') as launch:
            self.assertEqual(parallel.main(), 0)
            launch.assert_not_called()

    def test_pair_context_and_replay_are_explicit_in_spawn_entry(self):
        cfg = config(); identity = {'candidate_names': ['A', 'B']}
        proposal = {'mode': 'flip'}; row = {}
        def build(sim, pair_config, baseline, identities, value, view, pair_id, directory, reachable, attempt_row):
            labels.validate_pair(pair_config, pair_id, ['A', 'B'])
            self.assertEqual(labels._CONTEXT.get()[0], pair_id)
            return attempt_row
        with labels.installed(), patch.object(shard.look, 'measured_build', side_effect=build):
            result = shard.measured_build(None, cfg, {}, identity, proposal, {}, 'floorplan204_00000', '/raw', [], row)
        self.assertEqual(result['mode'], 'flip'); self.assertNotIn('fixed_label_pair_id', cfg)
        child = {**cfg, **labels.pair_fields('floorplan204_00000', ['A', 'B']), **identity}
        def replay(config, connection):
            self.assertEqual(labels._CONTEXT.get()[0], 'floorplan204_00000')
            return {'status': 'observed'}
        with patch.object(shard.look, '_ORIGINAL_REPLAY', side_effect=replay):
            outcome = shard.replay_worker(child, None)
        labels.validate_pair(outcome, 'floorplan204_00000', ['A', 'B'])
        self.assertIsNone(labels._CONTEXT.get())

    def test_spawn_installs_quota_hooks_and_restores_them_after_failure(self):
        old = shard.marked._BASE_WRITE
        def generated(cfg, connection):
            self.assertIs(shard.marked.measured_build, shard.measured_build)
            self.assertIs(shard.marked.replay_worker, shard.replay_worker)
            self.assertIsNot(shard.marked._BASE_WRITE, old)
            raise RuntimeError('native failure preserved')
        with patch.object(shard.look.stay, '_ORIGINAL_GENERATOR', side_effect=generated):
            with self.assertRaisesRegex(RuntimeError, 'native failure preserved'):
                shard.generator_worker(config(), None)
        self.assertIs(shard.marked._BASE_WRITE, old)

    def test_source_adapter_keeps_original_validator_failure_and_all_kwargs(self):
        cfg = config(); data = {'config': cfg, 'metrics': {**cfg, 'shard': None}}
        row = {**labels.pair_fields('floorplan204_00000', ['A', 'B']),
            'pair_id': 'floorplan204_00000', 'identities': {'candidate_names': ['A', 'B']},
            'flip_search_version': search.VERSION, 'flip_search_policy': copy.deepcopy(search.POLICY)}
        arguments = {'raw_root': '/synthetic/raw', 'field_of_view': 60,
                     'frozen_render_version': 'must_reach_original', 'solvability_policy': {'unaltered': True}}
        with patch.object(review, '_BASE_LOAD', return_value=(data, {'file': 'source'})), \
                patch.object(review.merge, 'check_pair', side_effect=ValueError('physical proof failed')) as check:
            with review.source_support():
                review.merge.load_source('source')
                with self.assertRaisesRegex(ValueError, 'physical proof failed'):
                    review.merge.check_pair(row, '/pair.json', **arguments)
            check.assert_called_once_with(row, '/pair.json', **arguments)

    def test_new_pair_cannot_use_adapter_without_authenticated_source(self):
        with review.source_support():
            with self.assertRaisesRegex(ValueError, 'authenticated source'):
                review.merge.check_pair({'flip_search_version': search.VERSION}, '/pair.json', raw_root='/unknown')


if __name__ == '__main__':
    unittest.main()
