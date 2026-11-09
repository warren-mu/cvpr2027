"""Frozen section1b CPU integration against the real pinned PPO dependencies."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from src.tools.smoke_e2e_training_contract import (FROZEN_PIPELINE, require_training_pool,
    pre_evidence, decision_token_spans)


class FrozenScope(unittest.TestCase):
    def test_opt_in_keeps_legacy_direct_boundary(self):
        look = SimpleNamespace(kind='look_flip', initial_state='v0', states={'v0': SimpleNamespace(shared=True)})
        require_training_pool({'p': look}, FROZEN_PIPELINE)
        with self.assertRaises(ValueError):
            require_training_pool({'p': look}, 'legacy_direct')
        with self.assertRaises(ValueError):
            require_training_pool({'p': look}, 'unapproved')
        look.states['v0'].shared = False
        with self.assertRaises(ValueError):
            require_training_pool({'p': look}, FROZEN_PIPELINE)
        self.assertTrue(pre_evidence(0, None))
        self.assertTrue(pre_evidence(0, 1))
        self.assertFalse(pre_evidence(1, 1))


class FrozenRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = os.environ.get('CVPR_SCRATCH')
        if not scratch:
            raise unittest.SkipTest('Requires the real pinned CPU dependencies on Hopper')
        from src.tools.smoke_e2e_train import upstream_roots
        cls.roots = upstream_roots(scratch)
        import torch
        from src.tools import smoke_e2e_training_runtime as runtime
        cls.torch, cls.runtime = torch, runtime

    def test_frozen_configuration_retains_common_actor_kl_and_task_only_rewards(self):
        from src.tools.smoke_e2e_train import build_config
        cfg, proof = build_config(Path('/unused/model'), Path('/unused/checkpoint'),
            roots=self.roots, critic_offload=False, pipeline=FROZEN_PIPELINE)
        self.assertFalse(cfg.algorithm.use_kl_in_reward)
        self.assertTrue(cfg.actor_rollout_ref.actor.use_kl_loss)
        self.assertEqual(cfg.actor_rollout_ref.actor.kl_loss_type, 'low_var_kl')
        self.assertEqual(cfg.actor_rollout_ref.actor.kl_loss_coef, .001)
        self.assertEqual((cfg.algorithm.gamma, cfg.algorithm.lam), (1, 1))
        self.assertTrue(proof['frozen_common_reference']['sha256'])
        self.assertFalse(cfg.critic.model.fsdp_config.param_offload)

    def test_actual_batch_reordering_keeps_paired_credit_and_original_critic_returns(self):
        t = self.torch
        plan = [{'episode_id': 'draw_' + w, 'pair_id': 'p', 'pair_instance_id': 'draw',
                 'world_id': w, 'rho': .5, 'trajectory_loss_weight': 1.} for w in ('w0', 'w1')]
        traces = {}
        for env, reward in ((0, 1.), (1, 0.)):
            traces[env] = []
            for turn, immediate in ((0, -.05), (1, reward)):
                traces[env].append({'decision_index': turn,
                    'before': {'tau': None if turn == 0 else 1, 'x_seen_step': None if turn == 0 else 1,
                               'target_name': 'a' if env == 0 else 'b',
                               'table_temporal_proof': {'decision_index': turn, 'shared': turn == 0, 'tau': None if turn == 0 else 1}},
                    'input': {'input_sha256': ('a' if turn == 0 else str(env)) * 64},
                    'sample_id': 'shared0' if turn == 0 else f'private{env}', 'shared_sample': turn == 0,
                    'response': 'turn_left' if turn == 0 else 'select 1', 'parse_valid': True,
                    'action_valid': True, 'action_kind': 'observe' if turn == 0 else 'select',
                    'reward': immediate, 'public_select_action_texts': ['select 1', 'select 2']})
        manager = SimpleNamespace(decisions=traces, scores_by_decision={(0, 0): .7, (1, 0): .3},
            envs={i: SimpleNamespace(env=SimpleNamespace(info={'tau': 1})) for i in (0, 1)})
        # Balancing reordered the actual tensor rows, while traces retain env IDs.
        batch = SimpleNamespace(batch={'responses': t.tensor([[10, 11, 20, 21], [10, 11, 30, 31]]),
            'loss_mask': t.ones((2, 4)), 'gae_mask': t.ones((2, 4)),
            'end_of_response_position_mask': t.tensor([[0, 1, 0, 1], [0, 1, 0, 1]]),
            'token_level_rewards': t.tensor([[0., -.05, 0., 0.], [0., -.05, 0., 1.]]),
            'values': t.tensor([[.1, .2, .3, .4], [.5, .6, .7, .8]])},
            non_tensor_batch={'smoke_env_id': [1, 0]})
        logs = []
        count = self.runtime.apply_frozen_credit(manager, batch, plan, 'T3',
            SimpleNamespace(algorithm=SimpleNamespace(gamma=1., lam=1.)), logs.append, 1)
        self.assertEqual(count, 2)
        self.assertTrue(t.equal(batch.batch['advantages'][0, :2], batch.batch['advantages'][1, :2]))
        self.assertEqual([row['world_id'] for row in logs], ['w1', 'w1', 'w0', 'w0'])
        self.assertAlmostEqual(logs[0]['G'], -.05)
        self.assertAlmostEqual(logs[2]['G'], .95)
        self.assertEqual(logs[0]['Q_commit'], .3)
        self.assertEqual(logs[2]['Q_commit'], .7)
        self.assertTrue(t.allclose(batch.batch['returns'], t.tensor([[-.05, -.05, 0., 0.], [.95, .95, 1., 1.]])))
        self.assertEqual([row['pre_evidence'] for row in logs], [True, False, True, False])
        self.assertTrue(all(row['token_world_loss_weights'] == [1., 1.] for row in logs))


if __name__ == '__main__':
    unittest.main()
