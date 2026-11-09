"""Synthetic CPU contracts only; actual PPO integration runs under PBS."""
import unittest
from types import SimpleNamespace

from src.tools.smoke_e2e_training_contract import (episode_plan, generation_groups,
    prompt_identity, decision_token_spans, task_reward, require_direct, available_actions)


class TinyImage:
    mode, size = 'RGB', (1, 1)

    def __init__(self, value):
        self.value = value

    def tobytes(self):
        return self.value


class TrainingContracts(unittest.TestCase):
    def test_empty_final_turn_public_action_domain_is_valid(self):
        self.assertEqual(available_actions('Instruction\nAvailable actions: []'), [])
        self.assertEqual(available_actions('Available actions: ["select 1"]'), ['select 1'])
        for text in ('', 'Available actions: [1]', 'Available actions: ["select 1", "select 1"]',
                     'Available actions: []\nAvailable actions: []'):
            with self.assertRaises(ValueError):
                available_actions(text)

    def test_fixed_budget_and_paired_worlds(self):
        for arm in ('T0', 'T3'):
            plan = episode_plan(['b', 'a'], arm=arm, update=1)
            self.assertEqual(len(plan), 64)
            self.assertEqual(len({row['episode_id'] for row in plan}), 64)
            self.assertEqual(plan, episode_plan(['a', 'b'], arm=arm, update=1))
            self.assertTrue(all(row['rho'] == .5 and row['trajectory_loss_weight'] == 1 for row in plan))
            if arm == 'T3':
                self.assertEqual(len({row['pair_instance_id'] for row in plan}), 32)
                for index in range(0, 64, 2):
                    self.assertEqual([row['world_id'] for row in plan[index:index + 2]], ['w0', 'w1'])
                    self.assertEqual(plan[index]['pair_id'], plan[index + 1]['pair_id'])

    def test_dedup_only_exact_history_within_same_pair(self):
        plan = episode_plan(['a'], arm='T3', update=1)[:4]
        image = TinyImage(b'abc')
        a = prompt_identity([1, 2], [image])[0]
        b = prompt_identity([1, 3], [image])[0]
        c = prompt_identity([1, 2], [TinyImage(b'abd')])[0]
        self.assertEqual(generation_groups(plan, [a, a, a, a], paired=True), [[0, 1], [2, 3]])
        self.assertEqual(generation_groups(plan, [a, b, a, c], paired=True), [[0], [1], [2], [3]])
        self.assertEqual(generation_groups(plan, [a] * 4, paired=False), [[0], [1], [2], [3]])
        with self.assertRaises(ValueError):
            generation_groups(plan, [a], paired=True)

    def test_complete_token_spans_and_no_silent_truncation(self):
        self.assertEqual(decision_token_spans([0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 0, 0, 1, 0]), [[1, 2], [4, 5]])
        for mask, ends in [([1, 1], [1, 0]), ([0, 1], [1, 1]), ([1], [0, 1])]:
            with self.assertRaises(ValueError):
                decision_token_spans(mask, ends)

    def test_reward_choice_cannot_be_inferred(self):
        self.assertEqual(task_reward(-.05, action_valid=True, format_policy='explicit_task_zero'), -.05)
        self.assertEqual(task_reward(0, action_valid=False, format_policy='explicit_task_zero'), 0.)
        self.assertEqual(task_reward(1, action_valid=True, format_policy='vagen_valid_bonus'), 1.5)
        with self.assertRaises(ValueError):
            task_reward(0, action_valid=False, format_policy=None)

    def test_direct_smoke_cannot_claim_look_credit_coverage(self):
        direct = SimpleNamespace(kind='direct_flip', initial_state='v0', states={'v0': SimpleNamespace(shared=False)})
        require_direct({'a': direct})
        for kind, shared in [('look_flip', True), ('direct_flip', True), ('look_flip', False)]:
            with self.assertRaises(ValueError):
                require_direct({'a': SimpleNamespace(kind=kind, initial_state='v0', states={'v0': SimpleNamespace(shared=shared)})})


if __name__ == '__main__':
    unittest.main()
