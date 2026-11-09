"""Pure pilot budget, namespace and observed-denominator fixtures."""
import copy
import unittest

from src.tools import smoke_e2e_pilot_contract as pilot


class PilotContractTests(unittest.TestCase):
    def test_engineering_mode_is_only_an_explicit_t4_prefix_of_original_schedule(self):
        for arm in pilot.ARM_UPDATES:
            self.assertEqual(pilot.execution_schedule(arm),
                (pilot.ARM_UPDATES[arm], pilot.HOLDOUT_UPDATES[arm]))
        self.assertEqual(pilot.execution_schedule('T4', pilot.ENGINEERING_SMOKE), (5, (0,)))
        self.assertEqual(pilot.ARM_UPDATES, {'T0': 50, 'T4': 50, 'T2': 5})
        self.assertEqual(pilot.HOLDOUT_UPDATES['T4'], (0, 20, 40, 50))
        for arm, mode in [('T0', pilot.ENGINEERING_SMOKE), ('T2', pilot.ENGINEERING_SMOKE),
                          ('T4', True), ('T4', 'five_updates'), ('unknown', None)]:
            with self.subTest(arm=arm, mode=mode), self.assertRaises(ValueError):
                pilot.execution_schedule(arm, mode)

    def test_training_uses_only_explicit_train_keys_and_exact_frozen_arm_budget(self):
        train = ['dataset:one', 'dataset:two']
        for arm, end in pilot.ARM_UPDATES.items():
            with self.subTest(arm=arm):
                rows = pilot.training_plan(train, arm=arm, update=end)
                self.assertEqual(rows, pilot.training_plan(train, arm=arm, update=end))
                self.assertEqual(len(rows), 64)
                self.assertEqual(len({r['episode_id'] for r in rows}), 64)
                self.assertLessEqual({r['pair_id'] for r in rows}, set(train))
                self.assertTrue(all(r['trajectory_loss_weight'] == 1. for r in rows))
                instances = {r['pair_instance_id'] for r in rows}
                self.assertEqual(len(instances), 32 if arm == 'T4' else 64)
                if arm == 'T4':
                    for identity in instances:
                        members = [r for r in rows if r['pair_instance_id'] == identity]
                        self.assertEqual({r['world_id'] for r in members}, {'w0', 'w1'})
                with self.assertRaises(ValueError):
                    pilot.training_plan(train, arm=arm, update=end + 1)

    def test_holdout_is_exact_both_worlds_without_training_draws_or_fake_t2_endpoint(self):
        keys = ['dataset:b', 'dataset:a']
        for arm in ('T0', 'T4'):
            for update in (0, 20, 40, 50):
                rows = pilot.holdout_plan(keys, arm=arm, update=update)
                self.assertEqual([(r['pair_id'], r['world_id']) for r in rows],
                    [('dataset:a', 'w0'), ('dataset:a', 'w1'), ('dataset:b', 'w0'), ('dataset:b', 'w1')])
        with self.assertRaises(ValueError): pilot.holdout_plan(keys, arm='T2', update=50)

    def observed(self, *, invalid_select=False, seen=None):
        plan = {'episode_id': 'e', 'pair_id': 'dataset:a', 'world_id': 'w0'}
        traces = [{**plan, 'decision_index': 0, 'parse_valid': False, 'action_valid': False,
                   'action_kind': 'invalid', 'reward': 0., 'sample_id': 'sample0'},
                  {**plan, 'decision_index': 1, 'parse_valid': True, 'action_valid': True,
                   'action_kind': 'observe', 'reward': -.05, 'sample_id': 'sample1'},
                  {**plan, 'decision_index': 2, 'parse_valid': True, 'action_valid': not invalid_select,
                   'action_kind': 'select', 'reward': 0. if invalid_select else 1., 'sample_id': 'sample2'}]
        return pilot.episode_observation(plan, traces, kind='look_flip',
            final_info={'tau': 1, 'x_seen_step': seen, 'selected': True},
            selected_name=None if invalid_select else 'A', success=not invalid_select)

    def test_unparseable_unavailable_and_terminal_select_are_distinct(self):
        normal, invalid = self.observed(), self.observed(invalid_select=True)
        self.assertEqual(normal['unparsable_outputs'], 1)
        self.assertEqual(invalid['unparsable_outputs'], 1)
        self.assertEqual(invalid['unavailable_parsed_actions'], 1)
        self.assertTrue(invalid['actual_terminal_selection'])
        self.assertFalse(invalid['valid_terminal_selection'])
        self.assertTrue(normal['premature_commit'])  # tau reached, own X never seen.
        self.assertTrue(invalid['premature_commit'])
        self.assertAlmostEqual(normal['task_reward_sum'], .95)
        self.assertEqual(normal['valid_observation_actions'], 1)

    def test_learning_seen_uses_strict_parsed_select_boundary_and_all_episode_denominator(self):
        earlier = self.observed(seen=1)
        simultaneous = self.observed(seen=2)
        invalid = self.observed(seen=1, invalid_select=True)
        self.assertTrue(earlier['seen_before_select'])
        self.assertFalse(simultaneous['seen_before_select'])
        self.assertFalse(simultaneous['premature_commit'])
        self.assertTrue(invalid['seen_before_select'])
        self.assertIsNone(invalid['seen_before_select_reason'])
        simultaneous['episode_id'], invalid['episode_id'] = 'e2', 'e3'
        row = pilot.classified_observations([earlier, simultaneous, invalid])['look_flip']
        self.assertEqual(row['episode_count'], 3)
        self.assertEqual(row['seen_before_select_count'], 2)
        self.assertEqual(row['seen_before_select_fraction_of_episodes'], 2 / 3)

    def test_seen_without_any_parsed_select_still_counts_false(self):
        plan = {'episode_id': 'e', 'pair_id': 'dataset:a', 'world_id': 'w0'}
        traces = [{**plan, 'decision_index': 0, 'parse_valid': False, 'action_valid': False,
                   'action_kind': 'invalid', 'reward': 0., 'sample_id': 'sample0'}]
        row = pilot.episode_observation(plan, traces, kind='look_stay',
            final_info={'tau': 0, 'x_seen_step': 0, 'selected': False},
            selected_name=None, success=False)
        self.assertFalse(row['seen_before_select'])
        self.assertEqual(row['seen_before_select_reason'], 'no_parseable_select')
        self.assertEqual(row['unparsable_outputs'], 1)

    def test_empty_classes_remain_unknown_and_denominators_are_explicit(self):
        episode = self.observed()
        rows = pilot.classified_observations([episode])
        row = rows['look_flip']
        self.assertEqual(row['observation_episode_fraction'], 1.)
        self.assertEqual(row['observation_action_fraction_of_decisions'], 1 / 3)
        self.assertEqual(row['unparsable_fraction_of_logical_decisions'], 1 / 3)
        self.assertIsNone(rows['look_stay']['mean_task_reward'])
        self.assertIsNone(rows['look_stay']['observation_episode_fraction'])
        self.assertIsNone(rows['direct_flip']['premature_commit_fraction_of_episodes'])
        with self.assertRaises(ValueError): pilot.classified_observations([episode, copy.deepcopy(episode)])


if __name__ == '__main__':
    unittest.main()
