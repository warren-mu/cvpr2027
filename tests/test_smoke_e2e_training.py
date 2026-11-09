"""CPU contracts and pinned worker wiring; actual PPO runs under PBS."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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


class PinnedWorkerWiring(unittest.TestCase):
    """Exercise real pinned Ray/verl wiring without creating a Ray cluster/GPU.

    Only model constructors/initializers are patched below. Class construction,
    remote-class wrapping, colocation and registered dispatch use the installed
    pinned implementation that rejected the first actual T0 attempt.
    """

    @classmethod
    def setUpClass(cls):
        scratch = os.environ.get('CVPR_SCRATCH')
        if not scratch:
            raise unittest.SkipTest('Source scratch_env.sh to check the actual pinned Hopper workers')
        from src.tools.smoke_e2e_train import upstream_roots
        upstream_roots(scratch)  # Checks the exact VAGEN and verl commits.
        import ray
        from src.tools import smoke_e2e_training_runtime as runtime
        from verl.single_controller.ray.base import RayClassWithInitArgs, create_colocated_worker_cls
        cls.ray, cls.runtime = ray, runtime
        cls.ray_args = RayClassWithInitArgs
        cls.colocate = staticmethod(create_colocated_worker_cls)

    def test_actual_workers_share_base_and_preserve_upstream_constructors(self):
        r = self.runtime
        self.assertIs(r.ObservedActorWorker.__base__, r.ObservedWorkerHooks)
        self.assertIs(r.ObservedCriticWorker.__base__, r.ObservedWorkerHooks)
        self.assertIs(r.ObservedWorkerHooks.__init__, r.Worker.__init__)
        self.assertIs(r.ObservedActorWorker.__init__, r.ActorRolloutRefWorker.__init__)
        self.assertIs(r.ObservedCriticWorker.__init__, r.CriticWorker.__init__)
        self.assertIs(r.ObservedActorWorker.update_actor, r.ActorRolloutRefWorker.update_actor)
        self.assertIs(r.ObservedCriticWorker.update_critic, r.CriticWorker.update_critic)

    def test_actual_registered_model_init_seeds_then_dispatches_upstream(self):
        r, calls = self.runtime, []
        for concrete, upstream, label in (
                (r.ObservedActorWorker, r.ActorRolloutRefWorker, 'actor'),
                (r.ObservedCriticWorker, r.CriticWorker, 'critic')):
            with self.subTest(worker=label), \
                    patch.object(r, 'seed_everything', side_effect=lambda seed: calls.append(('seed', seed))), \
                    patch.object(upstream, 'init_model', lambda self: calls.append(('model', label))):
                calls.clear()
                concrete.init_model(object.__new__(concrete))
                self.assertEqual(calls, [('seed', 0), ('model', label)])

    def test_actual_colocated_factory_builds_and_dispatches_all_three_roles(self):
        r, calls = self.runtime, []

        def actor_init(worker, config, role):
            worker.test_role, worker.test_config = role, config
            calls.append(('construct', role, config))

        def critic_init(worker, config):
            worker.test_role, worker.test_config = 'critic', config
            calls.append(('construct', 'critic', config))

        with patch.object(r.Worker, '__init__', return_value=None), \
                patch.object(r.ActorRolloutRefWorker, '__init__', actor_init), \
                patch.object(r.CriticWorker, '__init__', critic_init), \
                patch.object(r, 'seed_everything', side_effect=lambda seed: calls.append(('seed', seed))), \
                patch.object(r.ActorRolloutRefWorker, 'init_model', lambda self: self.test_role), \
                patch.object(r.CriticWorker, 'init_model', lambda self: self.test_role):
            roles = {
                'actor_rollout': self.ray_args(self.ray.remote(r.ObservedActorWorker), config='actor_cfg', role='actor_rollout'),
                'critic': self.ray_args(self.ray.remote(r.ObservedCriticWorker), config='critic_cfg'),
                'ref': self.ray_args(self.ray.remote(r.ObservedActorWorker), config='ref_cfg', role='ref')}
            wrapped = self.colocate(roles)
            worker_dict_cls = wrapped.cls.__ray_actor_class__
            self.assertIs(worker_dict_cls.__base__, r.ObservedWorkerHooks)
            workers = object.__new__(worker_dict_cls)
            worker_dict_cls.__init__(workers)
            self.assertEqual(calls, [('construct', 'actor_rollout', 'actor_cfg'),
                                     ('construct', 'critic', 'critic_cfg'),
                                     ('construct', 'ref', 'ref_cfg')])
            calls.clear()
            self.assertEqual(workers.actor_rollout_init_model(), 'actor_rollout')
            self.assertEqual(workers.critic_init_model(), 'critic')
            self.assertEqual(workers.ref_init_model(), 'ref')
            self.assertEqual(calls, [('seed', 0), ('seed', 0), ('seed', 0)])
            self.assertTrue(callable(workers.actor_rollout_smoke_memory))
            self.assertFalse(self.ray.is_initialized())


if __name__ == '__main__':
    unittest.main()
