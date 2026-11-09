"""CPU-only explicit unit-plan, lifecycle and unknown-stage fixtures."""
import copy
import sys
import unittest
from unittest.mock import patch
from src.envs import pair_v2_fov as fov
from src.envs import pair_v2_projection_pool as pools
from src.envs import pair_v2_stay_order as stay
from src.envs import pair_v2_look_unit_policy as policy
from src.envs import pair_v2_look_unit_shard as shard
from src.tools import t01v2_look_unit_parallel as parallel
from src.tools import t01v2_look_unit_review_checkpoint as review


def record():
    def token(x):
        point = {'x': x, 'y': .5, 'z': 1.}
        return {'position': point, 'receptacle_name': 'Surface', 'spawn': {
            'selected_position': dict(point), 'receptacle_name': 'Surface',
            'query_reference': {'file': str(x), 'sha256': 'a' * 64}}}
    return {'version': pools.POOL_VERSION, 'view': {'position': {'x': 0, 'y': 0, 'z': 0},
        'yaw': 0, 'horizon': 0}, 'counts': {}, 'points': {
        kind: {side: [{'token': token(x)} for x in (0, 1, 4, 7)]
               for side in ('A', 'B')} for kind in ('look', 'direct')}}


def config(scene='FloorPlan203', camera=60):
    fields = {**policy.camera_fields(camera, 'FloorPlan2'), **stay.fields(), 'producer_module': policy.PRODUCER}
    return {**fields, 'scene': scene, 'shard': {**fields, 'scene': scene}}


def argv(gpus=1, scenes=('FloorPlan203', 'FloorPlan204', 'FloorPlan2'), camera=60):
    return ['--name', 'pilot_pairs_v2_synthetic', '--gpus', str(gpus), '--scenes', *scenes,
        '--field-of-view', str(camera), '--timeout-seconds', '3870', '--scene-timeout-seconds', '1200',
        '--gate-timeout-seconds', '120']


class LookUnitTests(unittest.TestCase):
    def test_same_scene_both_cameras_require_explicit_new_policy(self):
        for scene in ('FloorPlan203', 'FloorPlan204', 'FloorPlan2'):
            for camera in (60, 100):
                cfg = config(scene, camera)
                self.assertEqual(policy.declared_fields(cfg)['field_of_view'], camera)
                policy.validate_source(cfg, {**cfg, 'shard': None})
        with self.assertRaises(ValueError):
            fov.declared_fields({'scene': 'FloorPlan203', **fov.contrast_fields(60)})
        cfg = config(); cfg['fov_contrast_policy']['search'] = 'mixed'
        with self.assertRaises(ValueError):
            policy.declared_fields(cfg)
        cfg = config('FloorPlan205')
        with self.assertRaises(ValueError):
            policy.validate_source(cfg)

    def test_old_parity_sources_and_legacy100_stay_unchanged(self):
        old = {'scene': 'FloorPlan203', **fov.contrast_fields(100)}
        before = fov.declared_fields
        with policy.camera_support():
            self.assertEqual(fov.declared_fields(old), before(old))
            self.assertEqual(fov.declared_fields({'scene': 'FloorPlan1'}), {'field_of_view': 100})
            self.assertEqual(fov.declared_fields(config())['field_of_view'], 60)
        self.assertIs(fov.declared_fields, before)

    def test_exactly_three_ordered_units_on_one_gpu_with_bounded_timing(self):
        parallel.validate_plan(argv(), 'FloorPlan2')
        for bad in (argv(2), argv(3), argv(1, ('FloorPlan203',)),
                    argv(1, ('FloorPlan204', 'FloorPlan203', 'FloorPlan2')),
                    argv(1, ('FloorPlan203', 'FloorPlan204', 'FloorPlan1'))):
            with self.assertRaises(ValueError):
                parallel.validate_plan(bad, 'FloorPlan2')
        for flag in ('--timeout-seconds', '--scene-timeout-seconds', '--gate-timeout-seconds'):
            bad = argv(); bad[bad.index(flag) + 1] = '1201'
            with self.assertRaises(ValueError):
                parallel.validate_plan(bad, 'FloorPlan2')
        self.assertEqual(len(policy.unit_fields('FloorPlan2')['look_unit_plan']), 6)

    def test_iterator_never_falls_back_to_direct_even_when_look_is_empty(self):
        rec = record(); old = copy.deepcopy(rec)
        stream = shard.LookOnlyIterator([rec])
        rows = list(stream)
        self.assertTrue(rows)
        self.assertEqual({proposal['projection_pool']['pool'] for proposal, _, _ in rows}, {'look'})
        self.assertEqual({proposal['mode'] for proposal, _, _ in rows}, {'flip', 'stay'})
        sides = [domain['stay_side'] for proposal, _, domain in rows if proposal['mode'] == 'stay']
        self.assertEqual(sides, ['A', 'B'] * (len(sides) // 2))
        self.assertEqual(rec, old)
        rec['points']['look'] = {'A': [], 'B': []}
        self.assertEqual(list(shard.LookOnlyIterator([rec])), [])
        self.assertTrue(rec['points']['direct']['A'])

    def test_look_prefix_rejects_visible_or_divergent_start_before_direct_branch(self):
        cases = [({'w0': 1, 'w1': 0}, 0, 'anchor_visible_at_v0'),
                 ({'w0': 0, 'w1': 0}, 4, 'anchor_hidden_but_prefix_rgb_diff_gt3')]
        for masks, maximum, reason in cases:
            measured = {'prefix_edit_mask_pixels': [masks], 'prefix_max_abs_diff': maximum,
                'render_qualified_for_shared_prefix': False, 'candidates_visible_and_marked': True}
            row = {'mode': 'stay', 'physical_w0': {'valid_physical': True}, 'physical_w1': {'valid_physical': True}}
            def build(*args, **kwargs):
                shard.original_policy.prefix_check()
                self.fail('Non-Look start must not continue into Direct collection')
            with patch.object(shard.original_policy, 'prefix_check', return_value=measured):
                with patch.object(shard, '_ORIGINAL_BUILD', build):
                    result = shard.measured_build(attempt_row=row)
            self.assertEqual(result['discard_reasons'], [reason])
            self.assertEqual(result['prefix_measurement'], measured)
            self.assertNotIn('look_solvability', result)
            self.assertNotIn('witnesses', result)

    def test_physical_rejection_does_not_invent_prefix_or_later_measurements(self):
        row = {'mode': 'flip', 'discard_reasons': ['w0_physical'], 'physical_w0': {'valid_physical': False}}
        with patch.object(shard, '_ORIGINAL_BUILD', return_value=row):
            self.assertIs(shard.measured_build(attempt_row=row), row)
        compact = shard.compact_attempt(row)
        self.assertIsNone(compact['prefix_measurement'])
        self.assertIsNone(compact['look_solvability'])
        self.assertIsNone(compact['witnesses'])
        self.assertIsNone(compact['restore_deterministic'])
        self.assertEqual(compact['mode'], 'flip')

    def test_compact_retains_actual_per_action_evidence_and_unknown_others(self):
        row = {'mode': 'stay', 'prefix_measurement': {'prefix_edit_mask_pixels': [{'w0': 0, 'w1': 0}],
            'prefix_max_abs_diff': 0, 'render_qualified_for_shared_prefix': True},
            'look_solvability': {'complete': True, 'solvable_w0': True, 'solvable_w1': True, 'actions': [
                {'action': 'turn_left', 'worlds': [], 'evidence': {'evidence_pixels': 219,
                 'marked_evidence_pixels': 219, 'evidence_passed': False, 'marked_evidence_passed': False}},
                {'action': 'turn_right', 'worlds': [], 'evidence': None}]}}
        compact = shard.compact_attempt(row)
        self.assertEqual(compact['look_solvability']['actions'][0]['evidence']['evidence_pixels'], 219)
        self.assertFalse(compact['look_solvability']['actions'][0]['evidence']['evidence_passed'])
        self.assertIsNone(compact['look_solvability']['actions'][1]['evidence'])
        self.assertIsNone(compact['witnesses'])

    def test_spawned_generator_and_replay_install_same_scene_camera_guard(self):
        cfg = config(); before = fov.declared_fields, pools.PoolProposalIterator, shard.marked.generator_worker
        def generate(config, connection):
            self.assertIs(shard.marked.generator_worker, shard.generator_worker)
            self.assertIs(shard.marked.replay_worker, shard.replay_worker)
            self.assertIs(pools.PoolProposalIterator, shard.LookOnlyIterator)
            self.assertEqual(fov.declared_fields(config)['field_of_view'], 60)
            return 'generated'
        with patch.object(shard.stay, '_ORIGINAL_GENERATOR', generate):
            self.assertEqual(shard.generator_worker(cfg, None), 'generated')
        def replay(config, connection):
            self.assertEqual(fov.declared_fields(config)['field_of_view'], 60)
            return 'replayed'
        with patch.object(shard, '_ORIGINAL_REPLAY', replay):
            self.assertEqual(shard.replay_worker(cfg, None), 'replayed')
        self.assertEqual(before, (fov.declared_fields, pools.PoolProposalIterator, shard.marked.generator_worker))

    def test_source_camera_and_metrics_binding_rejects_mismatch(self):
        cfg = config(); measured = copy.deepcopy(cfg)
        policy.validate_source(cfg, measured)
        measured['field_of_view'] = 100
        with self.assertRaises(ValueError):
            policy.validate_source(cfg, measured)
        cfg['shard']['look_unit_kitchen_scene'] = 'FloorPlan1'
        with self.assertRaises(ValueError):
            policy.validate_source(cfg)

    def test_parent_strips_only_explicit_kitchen_option_and_restores_hooks(self):
        before = parallel.marked.SCENE_MODULE, parallel.marked.validate_plan
        entry = ['look_entry', *argv(), '--kitchen', 'FloorPlan2']
        def main():
            self.assertNotIn('--kitchen', sys.argv)
            self.assertEqual(parallel.marked.SCENE_MODULE, policy.PRODUCER)
            self.assertEqual(parallel.marked.validate_plan(sys.argv[1:])['look_unit_kitchen_scene'], 'FloorPlan2')
            return 0
        with patch.object(sys, 'argv', entry), patch.object(parallel.marked, 'main', main):
            self.assertEqual(parallel.main(), 0)
            self.assertIs(sys.argv, entry)
        self.assertEqual(before, (parallel.marked.SCENE_MODULE, parallel.marked.validate_plan))

    def test_parent_refuses_truncated_unit_before_launch_and_keeps_full_grants(self):
        entry = ['look_entry', *argv(), '--kitchen', 'FloorPlan2']
        launched = []
        def launch(name, command, environment, directory):
            launched.append(name)
            return name
        def main():
            command = ['python3', '-m', policy.PRODUCER, '--timeout-seconds', '1200']
            self.assertEqual(parallel.original.launch_logged('complete_grant', command, {}, 'synthetic'), 'complete_grant')
            command[-1] = '1199.99'
            with self.assertRaisesRegex(RuntimeError, 'Unstarted Look unit'):
                parallel.original.launch_logged('truncated', command, {}, 'synthetic')
            return 0
        with patch.object(sys, 'argv', entry), patch.object(parallel.marked, 'main', main):
            with patch.object(parallel.original, 'launch_logged', launch):
                self.assertEqual(parallel.main(), 0)
        self.assertEqual(launched, ['complete_grant'])

    def test_new_source_dispatch_preserves_original_command_and_policy_guards(self):
        cfg = config(); original = copy.deepcopy(cfg)
        with patch.object(review.merge, 'marked_policy_configuration', return_value={}):
            with review.source_support():
                result = review.merge.shard_producer(['python3', '-m', policy.PRODUCER, '--scene', 'FloorPlan203', '--timeout-seconds', '1200'], cfg)
                self.assertEqual(result, (['--scene', 'FloorPlan203', '--timeout-seconds', '1200'], False))
                with self.assertRaises(ValueError):
                    review.merge.shard_producer(['python3', '-m', 'src.envs.pair_v2_marked_shard'], cfg)
        self.assertEqual(cfg, original)


if __name__ == '__main__':
    unittest.main()
