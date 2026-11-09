"""Explicit new-source/spawn wiring; never silently upgrade old runs."""
import unittest
from unittest.mock import patch
from src.envs import pair_v2_flip_shard as previous
from src.envs import pair_v2_flip_policy as old_policy
from src.envs import pair_v2_flip_rounds_policy as policy
from src.envs import pair_v2_flip_rounds_shard as producer
from src.envs import pair_v2_flip_rounds_search as cadence
from src.tools import t01v2_flip_review_checkpoint as old_checker
from src.tools import t01v2_flip_rounds_review_checkpoint as checker
from src.envs import pair_v2_fixed_labels as labels
from src.envs import pair_v2_stay_order as stay
from src.envs import pair_v2_fov as fov


class RoundSourceWiringTests(unittest.TestCase):
    def test_explicit_policy_binds_cadence_and_training_scene(self):
        fields = policy.camera_fields(60, 'FloorPlan203')
        config = {**fields, 'scene': 'FloorPlan203'}
        self.assertEqual(policy.declared_fields(config), fields)
        cadence.validate_fields(config)
        for change in ({'field_of_view': 100}, {'scene': 'FloorPlan21'},
                       {'flip_unit_version': old_policy.VERSION}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                policy.declared_fields({**config, **change})

    def test_spawn_targets_and_rollback_leave_active_old_code_unchanged(self):
        marked = previous.marked
        before = (previous.policy, marked.run_search, marked.generator_worker,
                  marked.replay_worker, marked.tags)
        with producer.installed():
            self.assertIs(previous.policy, policy)
            self.assertIs(marked.run_search, cadence.run_search)
            self.assertEqual(marked.generator_worker.__module__, producer.__name__)
            self.assertEqual(marked.replay_worker.__module__, producer.__name__)
            self.assertEqual(marked.tags()['producer_module'], policy.PRODUCER)
            cadence.validate_fields(marked.tags())
        self.assertEqual(before, (previous.policy, marked.run_search, marked.generator_worker,
                                  marked.replay_worker, marked.tags))

    def test_source_adapter_restores_old_checker_policy(self):
        before = old_checker.policy
        with checker.source_support():
            self.assertIs(old_checker.policy, policy)
        self.assertIs(old_checker.policy, before)

    def test_spawned_generator_installs_new_scheduler_before_native_entry(self):
        shard = {**policy.camera_fields(60, 'FloorPlan203'), **stay.fields(),
                 'scene': 'FloorPlan203', 'producer_module': policy.PRODUCER}
        config = {**shard, 'shard': shard}
        def native(actual, connection):
            self.assertIs(actual, config)
            self.assertIs(previous.marked.run_search, cadence.run_search)
            self.assertIs(previous.policy, policy)
            return 'entered'
        with patch.object(previous.look.stay, '_ORIGINAL_GENERATOR', side_effect=native):
            self.assertEqual(producer.generator_worker(config, None), 'entered')
        self.assertIs(previous.policy, old_policy)

    def test_spawned_replay_keeps_new_camera_and_fixed_pair_context(self):
        names = ['candidate-a', 'candidate-b']
        pair_id = 'synthetic-rounds-pair'
        config = {**labels.pair_fields(pair_id, names), **cadence.fields(),
                  'candidate_names': names}
        def native(actual, connection):
            self.assertIs(fov.declared_fields, policy.declared_fields)
            self.assertIs(actual, config)
            return {'scope': 'synthetic-worker'}
        with patch.object(previous.look, '_ORIGINAL_REPLAY', side_effect=native):
            observed = producer.replay_worker(config, None)
        self.assertEqual(observed['candidate_label_assignment'], config['candidate_label_assignment'])
        cadence.validate_fields(observed)


if __name__ == '__main__':
    unittest.main()
