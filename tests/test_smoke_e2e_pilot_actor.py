"""Actor hook scheduling fixtures; no surrogate PPO/model claim."""
import copy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.tools import smoke_e2e_pilot_actor as adapter
try:
    import torch
except ImportError:
    torch = None


class Optimizer:
    def __init__(self): self.calls = []
    def step(self): self.calls.append('original_step')


class Actor:
    def __init__(self):
        self.actor_optimizer = Optimizer()
        self.config = SimpleNamespace(ppo_mini_batch_size=32, ppo_epochs=1,
            use_dynamic_bsz=False, ulysses_sequence_parallel_size=1)
    def _optimizer_step(self):
        self.actor_optimizer.step()
        return 'original_norm'


def data(arm='T0'):
    ids = [f'e{i:02d}' for i in range(64)]
    return SimpleNamespace(batch=SimpleNamespace(batch_size=(64,)),
        non_tensor_batch={'pilot_episode_id': ids},
        meta_info={'pilot_update': {'arm': arm}, 'temperature': .7})


class ActorScheduleTests(unittest.TestCase):
    def test_original_minibatch_order_and_optimizer_methods_survive_context(self):
        actor, batch = Actor(), data()
        original = actor._optimizer_step.__func__, actor.actor_optimizer.step.__func__
        with adapter.update_support(actor, batch) as record:
            self.assertEqual(actor._optimizer_step(), 'original_norm')
            self.assertEqual(actor._optimizer_step(), 'original_norm')
        self.assertEqual(record['native_optimizer_step_calls'], 2)
        self.assertEqual(record['minibatches'][0]['episode_ids'], batch.non_tensor_batch['pilot_episode_id'][:32])
        self.assertEqual(record['minibatches'][1]['episode_ids'], batch.non_tensor_batch['pilot_episode_id'][32:])
        self.assertEqual((actor._optimizer_step.__func__, actor.actor_optimizer.step.__func__), original)
        self.assertNotIn('_optimizer_step', vars(actor))
        self.assertNotIn('step', vars(actor.actor_optimizer))

    def test_native_skipped_optimizer_is_retained_but_never_completed(self):
        actor, batch = Actor(), data()
        native_skipped_step = lambda: 'native_nonfinite_skip'
        actor._optimizer_step = native_skipped_step
        with adapter.update_support(actor, batch) as record:
            self.assertEqual(actor._optimizer_step(), 'native_nonfinite_skip')
            self.assertEqual(actor._optimizer_step(), 'native_nonfinite_skip')
        self.assertEqual(record['native_minibatch_steps'], 2)
        self.assertEqual(record['native_optimizer_step_calls'], 0)
        self.assertTrue(all(row['optimizer_step_called'] is False for row in record['minibatches']))
        self.assertIs(actor._optimizer_step, native_skipped_step)
        self.assertEqual(actor.actor_optimizer.calls, [])
        with self.assertRaisesRegex(ValueError, 'optimizer_calls=0, native_minibatches=2'):
            adapter.require_completed_optimizer_steps(record)
        actor, batch = Actor(), data()
        with adapter.update_support(actor, batch) as complete:
            actor._optimizer_step(); actor._optimizer_step()
        adapter.require_completed_optimizer_steps(complete)

    def test_unfinished_or_extra_minibatch_is_not_complete_and_hooks_restore(self):
        for calls in (1, 3):
            actor, batch = Actor(), data()
            with self.subTest(calls=calls), self.assertRaises(ValueError):
                with adapter.update_support(actor, batch):
                    for _ in range(calls): actor._optimizer_step()
            self.assertNotIn('_optimizer_step', vars(actor))
            self.assertNotIn('step', vars(actor.actor_optimizer))

    def test_missing_t4_episode_or_unapproved_dynamic_batch_fails_before_update(self):
        actor, batch = Actor(), data('T4')
        with self.assertRaises(ValueError):
            with adapter.update_support(actor, batch): pass
        batch = data(); actor.config.use_dynamic_bsz = True
        with self.assertRaises(ValueError):
            with adapter.update_support(actor, batch): pass
        self.assertEqual(actor.actor_optimizer.calls, [])


@unittest.skipIf(torch is None, 'real torch unavailable locally; execute in logged pinned environment')
class ActorGradientOrderTests(unittest.TestCase):
    def test_real_auxiliary_gradient_accumulates_before_original_step_with_full_N(self):
        actor, batch = Actor(), data('T4')
        parameter = torch.tensor(-2., dtype=torch.float64, requires_grad=True)
        snapshots = []
        def original_step():
            snapshots.append(None if parameter.grad is None else float(parameter.grad))
            actor.actor_optimizer.step()
            parameter.grad = None
            return 0.
        actor._optimizer_step = original_step
        descriptors = {key: {'applicable': False, 'skip_reason': 'no_legal_select'}
                       for key in batch.non_tensor_batch['pilot_episode_id']}
        descriptors['e00'] = {'applicable': True, 'skip_reason': None, 'change': 'flip',
            'candidate_names': ['A', 'B'], 'target_name': 'A', 'other_target_name': 'B',
            'trajectory_loss_weight': 1.}
        batch.meta_info['pilot_update'].update(consistency_descriptors=descriptors,
            candidate_inputs={'e00': {'fixture': 'native forward is a separate integration boundary'}})
        with patch.object(adapter, 'candidate_scores', return_value={'A': parameter, 'B': parameter * 0 - 1.}):
            with adapter.update_support(actor, batch) as record:
                actor._optimizer_step(); actor._optimizer_step()
        self.assertAlmostEqual(snapshots[0], -.1 / 32)
        self.assertIsNone(snapshots[1])
        self.assertEqual(record['terms'][0]['minibatch_N'], 32)
        self.assertIs(actor._optimizer_step, original_step)


class PinnedPilotWorkerWiring(unittest.TestCase):
    """Real pinned colocation/registration; no Ray cluster, model, or GPU start."""
    @classmethod
    def setUpClass(cls):
        scratch = os.environ.get('CVPR_SCRATCH')
        if not scratch:
            raise unittest.SkipTest('Requires the logged pinned Hopper CPU dependencies')
        from src.tools.smoke_e2e_train import upstream_roots
        upstream_roots(scratch)
        import ray
        from src.tools import smoke_e2e_training_runtime as old
        from src.tools import smoke_e2e_pilot_runtime as runtime
        from verl.single_controller.ray.base import RayClassWithInitArgs, create_colocated_worker_cls
        cls.old, cls.runtime, cls.ray = old, runtime, ray
        cls.ray_args, cls.colocate = RayClassWithInitArgs, staticmethod(create_colocated_worker_cls)

    def test_original_default_hooks_and_new_arm_dispatch_are_explicit(self):
        old, new = self.old, self.runtime
        self.assertEqual(old.TableRollouts.paired_arms, ('T3',))
        self.assertEqual(old.TableRollouts.q_commit_arms, ('T3',))
        self.assertEqual(old.TableRollouts.expected_episode_count, 64)
        self.assertEqual(old.TableRollouts.generation_metadata, {})
        self.assertEqual(new.PilotRollouts.paired_arms, ('T4',))
        self.assertEqual(new.PilotRollouts.q_commit_arms, ('T2', 'T4'))
        self.assertIs(new.PilotActorWorker.__base__, old.ObservedWorkerHooks)
        self.assertIs(new.PilotActorWorker.__init__, old.ActorRolloutRefWorker.__init__)

    def test_actual_colocated_factory_keeps_actor_critic_and_reference_role_dispatch(self):
        old, runtime, calls = self.old, self.runtime, []
        def actor_init(worker, config, role):
            worker.fixture_role = role
            calls.append(('construct', role, config))
        def critic_init(worker, config):
            worker.fixture_role = 'critic'
            calls.append(('construct', 'critic', config))
        with patch.object(old.Worker, '__init__', return_value=None), \
                patch.object(old.ActorRolloutRefWorker, '__init__', actor_init), \
                patch.object(old.CriticWorker, '__init__', critic_init), \
                patch.object(old, 'seed_everything', side_effect=lambda seed: calls.append(('seed', seed))), \
                patch.object(old.ActorRolloutRefWorker, 'init_model', lambda self: self.fixture_role), \
                patch.object(old.CriticWorker, 'init_model', lambda self: self.fixture_role):
            roles = {
                'actor_rollout': self.ray_args(self.ray.remote(runtime.PilotActorWorker), config='actor', role='actor_rollout'),
                'critic': self.ray_args(self.ray.remote(old.ObservedCriticWorker), config='critic'),
                'ref': self.ray_args(self.ray.remote(runtime.PilotActorWorker), config='reference', role='ref')}
            wrapped = self.colocate(roles)
            concrete = wrapped.cls.__ray_actor_class__
            self.assertIs(concrete.__base__, old.ObservedWorkerHooks)
            workers = object.__new__(concrete); concrete.__init__(workers)
            self.assertEqual(workers.actor_rollout_init_model(), 'actor_rollout')
            self.assertEqual(workers.critic_init_model(), 'critic')
            self.assertEqual(workers.ref_init_model(), 'ref')
            self.assertTrue(callable(workers.actor_rollout_update_actor))
            self.assertTrue(callable(workers.actor_rollout_smoke_memory))
            self.assertEqual(calls[-3:], [('seed', 0)] * 3)
            self.assertFalse(self.ray.is_initialized())


if __name__ == '__main__':
    unittest.main()
