"""Explicit new-source boundaries; no PBS, native rendering, or result claims."""
import copy
import unittest

from src.envs.pair_v2_marked_shard import tags
from src.tools import merge_pair_v2 as merge
from src.tools import phase_watch as watch
from src.tools import t01v2_marked_parallel as outer


class MarkedIntegrationTests(unittest.TestCase):
    def test_explicit_policy_and_producer_are_both_required(self):
        config = tags()
        command = ['python3', '-u', '-m', outer.SCENE_MODULE, '--name', 'fixture']
        args, clone = merge.shard_producer(command, config)
        self.assertFalse(clone)
        self.assertEqual(args, ['--name', 'fixture'])
        for key in ('candidate_policy', 'candidate_policy_spec', 'projection_pool_version', 'projection_pool_policy',
                    'preflight_version', 'solvability_policy', 'enable_candidate_clone', 'producer_module'):
            changed = copy.deepcopy(config); changed[key] = 'wrong'
            with self.subTest(key=key), self.assertRaises(ValueError):
                merge.shard_producer(command, changed)
        for key in ('search_version', 'proposal_order_version'):
            changed = copy.deepcopy(config); changed[key] = 'old_policy'
            with self.assertRaises(ValueError):
                merge.shard_producer(command, changed)
        with self.assertRaises(ValueError):
            merge.shard_producer(['python3', '-m', 'src.envs.pair_v2_shard'], config)

    def test_fixed_forty_train_scenes_and_one_gpu(self):
        outer.validate_plan(['--gpus', '1', '--scenes', 'FloorPlan1', 'FloorPlan220', '--timeout-seconds', '500'])
        for args in (['--gpus', '2', '--scenes', 'FloorPlan1'],
                     ['--gpus', '1', '--scenes', 'FloorPlan21'],
                     ['--gpus', '1', '--scenes', 'FloorPlan221'],
                     ['--gpus', '1', '--scenes', 'FloorPlan1', 'FloorPlan1']):
            with self.assertRaises(ValueError):
                outer.validate_plan(args)

    def test_new_dispatch_never_relabels_historical_results(self):
        data = {'config': outer.marked_tags(), 'metrics': outer.marked_tags()}
        self.assertEqual(watch.effective_kind('t01v2_marked', data), 't01v2_marked')
        for old in ('t01v2', 't01v2_isolated', 't01v2_search'):
            with self.assertRaises(ValueError):
                watch.effective_kind(old, data)
        for location in ('config', 'metrics'):
            changed = copy.deepcopy(data); changed[location]['enable_candidate_clone'] = True
            with self.assertRaises(ValueError):
                watch.effective_kind('t01v2_marked', changed)

    def test_native_retry_is_not_enabled_for_old_source_command(self):
        self.assertTrue(watch.authorized_shard_command(['python3', '-m', outer.SCENE_MODULE], v2=True, marked=True))
        self.assertFalse(watch.authorized_shard_command(['python3', '-m', outer.SCENE_MODULE], v2=True))
        self.assertFalse(watch.authorized_shard_command(['python3', '-m', 'src.envs.pair_v2_isolated_shard'], v2=True, marked=True))


if __name__ == '__main__':
    unittest.main()
