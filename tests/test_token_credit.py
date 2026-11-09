"""T05 §1b synthetic CPU fixtures against the actual pinned torch functions.

Run after commit through run_logged, with scratch_env.sh sourced. These are
arithmetic contracts, not measured model/experiment results. No GPU is used.
"""
import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from src.rl.credit import CreditContractError
from src.rl import token_credit as credit


def fixture(torch, *, rho=.5, dtype=None, value_dtype=None):
    dtype = dtype or torch.float32
    mask = torch.tensor([[0, 1, 1, 0, 1, 1, 1, 0], [1, 1, 0, 1, 1, 1, 0, 0]], dtype=torch.long)
    rewards = torch.zeros((2, 8), dtype=dtype)
    values = torch.tensor([[.12, .3, .5, .2, .8, .2, .4, .1],
                           [.2, .4, .7, .3, .8, .1, .9, .6]], dtype=value_dtype or dtype)
    records = []
    for world, spans in enumerate(([[1, 2], [4, 5, 6]], [[0, 1], [3, 4, 5]])):
        tasks = [-.05, 1.0 if world == 0 else 0.0]
        for turn, indices in enumerate(spans):
            rewards[world, indices[-1]] = tasks[turn]
            records.append({'row_index': world, 'episode_id': 'episode_' + str(world),
                'pair_id': 'pool_pair', 'pair_instance_id': 'draw_0', 'world_id': 'w' + str(world),
                'rho': rho, 'decision_index': turn, 'token_indices': indices,
                'response_token_ids': [11, 12] if turn == 0 else [21 + world, 22, 23],
                'tau': 1, 'x_seen_step': None if turn == 0 else 1,
                'input_sha256': 'a' * 64 if turn == 0 else str(world) * 64,
                'shared_sample_id': 'shared_turn_0' if turn == 0 else 'independent_' + str(world),
                'shared_input_verified_exact': turn == 0,
                'action_text': 'turn_left' if turn == 0 else 'select 1',
                'parse_valid': True, 'action_valid': True,
                'action_kind': 'observe' if turn == 0 else 'select',
                'task_reward': tasks[turn], 'G': sum(tasks[turn:]),
                'Q_commit': .6 if world == 0 else .4,
                'public_select_action_texts': ['select 1', 'select 2'],
                'target_name': 'candidate_A' if world == 0 else 'candidate_B',
                'temporal_proof': {'synthetic_T02_state': turn, 'shared': turn == 0}})
    return {'token_level_rewards': rewards, 'values': values, 'loss_mask': mask,
            'gae_mask': mask.clone(), 'decisions': records, 'gamma': 1.0, 'lam': 1.0}


class FrozenTokenCredit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = os.environ.get('CVPR_SCRATCH')
        if not scratch:
            raise unittest.SkipTest('Source scratch_env.sh to test the actual pinned verl torch functions')
        from src.tools.smoke_e2e_train import upstream_roots
        cls.roots = upstream_roots(scratch)  # Authenticate both fixed commits; no source stubs.
        import torch
        from verl.trainer.ppo.core_algos import compute_gae_advantage_return_with_loss_mask
        from verl.utils import torch_functional
        cls.torch, cls.F = torch, torch_functional
        cls.upstream = staticmethod(compute_gae_advantage_return_with_loss_mask)

    def upstream_result(self, data):
        return self.upstream(token_level_rewards=data['token_level_rewards'], values=data['values'],
            loss_mask=data['gae_mask'], gamma=data['gamma'], lam=data['lam'])

    def test_no_replacement_exactly_matches_actual_pinned_function_including_mixed_dtype(self):
        t = self.torch
        for dtype, value_dtype in ((t.float32, t.float32), (t.float64, t.float64), (t.float32, t.bfloat16)):
            for arm in ('T0', 'T1'):
                for gamma, lam in ((1.0, 1.0), (.91, .83)):
                    with self.subTest(dtype=dtype, values=value_dtype, arm=arm, gamma=gamma, lam=lam):
                        data = fixture(t, dtype=dtype, value_dtype=value_dtype)
                        data.update(gamma=gamma, lam=lam)
                        # Preserve a padded empty trajectory, as the original function does.
                        for key in ('token_level_rewards', 'values', 'loss_mask', 'gae_mask'):
                            data[key] = t.cat((data[key], t.zeros_like(data[key][:1])), dim=0)
                        baseline_adv, baseline_returns = self.upstream_result(data)
                        with patch.object(self.F, 'masked_whiten', wraps=self.F.masked_whiten) as whiten:
                            result = credit.credit_pipeline(arm=arm, **data)
                        self.assertEqual(whiten.call_count, 1)
                        self.assertTrue(t.equal(result['advantages'], baseline_adv))
                        self.assertTrue(t.equal(result['returns'], baseline_returns))
                        self.assertTrue(t.equal(result['raw_advantages'], result['replaced_advantages']))
                        self.assertFalse(result['advantages'].requires_grad)

    def test_q_credit_broadcast_prefix_equal_and_post_uses_same_global_affine(self):
        t = self.torch
        for arm in ('T3', 'T4'):
            data = fixture(t, rho=.25)
            result = credit.credit_pipeline(arm=arm, **data)
            expected = .75 * (.95 - .6) + .25 * (-.05 - .4)
            a, b = data['decisions'][0], data['decisions'][2]
            prefix = result['replaced_advantages'][0, a['token_indices']]
            self.assertTrue(t.equal(prefix, t.full_like(prefix, expected)))
            self.assertTrue(t.equal(prefix, result['replaced_advantages'][1, b['token_indices']]))
            self.assertTrue(t.equal(result['advantages'][0, a['token_indices']],
                                    result['advantages'][1, b['token_indices']]))
            mean = self.F.masked_mean(result['replaced_advantages'], data['loss_mask'])
            variance = self.F.masked_var(result['replaced_advantages'], data['loss_mask'])
            affine = (result['raw_advantages'] - mean) * t.rsqrt(variance + 1e-8)
            for record in (data['decisions'][1], data['decisions'][3]):
                index = (record['row_index'], record['token_indices'])
                self.assertTrue(t.equal(result['raw_advantages'][index], result['replaced_advantages'][index]))
                self.assertTrue(t.equal(result['advantages'][index], affine[index]))
            self.assertTrue(t.equal(result['returns'], self.upstream_result(data)[1]))
            self.assertTrue(t.equal(result['world_loss_weights'][0], t.full((8,), 1.5)))
            self.assertTrue(t.equal(result['world_loss_weights'][1], t.full((8,), .5)))
            self.assertFalse(result['normalization']['world_weights_applied_to_advantages'])

    def test_fallback_is_same_position_raw_tokens_not_decision_mean(self):
        t = self.torch
        for fallback in ('no_candidates', 'unparseable'):
            with self.subTest(fallback=fallback):
                data = fixture(t, rho=.25)
                for record in data['decisions'][::2]:
                    record['Q_commit'] = None
                    if fallback == 'no_candidates':
                        record['public_select_action_texts'] = []
                    else:
                        record.update(parse_valid=False, action_kind='invalid', action_text='')
                result = credit.credit_pipeline(arm='T3', **data)
                first, second = data['decisions'][0], data['decisions'][2]
                a, b = (0, first['token_indices']), (1, second['token_indices'])
                expected = .75 * result['raw_advantages'][a] + .25 * result['raw_advantages'][b]
                self.assertNotEqual(expected[0].item(), expected[1].item())
                self.assertTrue(t.equal(result['replaced_advantages'][a], expected))
                self.assertTrue(t.equal(result['replaced_advantages'][b], expected))
                self.assertTrue(t.equal(result['advantages'][a], result['advantages'][b]))
                self.assertIsNone(result['decision_records'][0]['decision_advantage'])
                self.assertEqual(result['decision_records'][0]['credit_mode'],
                                 'prior_weighted_same_position_token_gae_fallback')

    def test_single_world_fallback_and_semantic_unavailable_action_are_distinct(self):
        t = self.torch
        data = fixture(t)
        data['decisions'][0].update(public_select_action_texts=[], Q_commit=None)
        # Parsing succeeds in the second world, even though the environment did
        # not make this observation action available. This is not parse failure.
        data['decisions'][2]['action_valid'] = False
        result = credit.credit_pipeline(arm='T2', **data)
        self.assertTrue(t.equal(result['raw_advantages'][0], result['replaced_advantages'][0]))
        index = (1, data['decisions'][2]['token_indices'])
        self.assertTrue(t.equal(result['replaced_advantages'][index], t.full((2,), -.45)))
        self.assertTrue(t.equal(result['world_loss_weights'], t.ones_like(data['values'])))

    def test_select_credit_uses_immediate_reward_and_task_G_is_verified(self):
        t = self.torch
        data = fixture(t)
        for row in data['decisions'][::2]:
            # A parseable unavailable select can leave the episode unfinished.
            row.update(action_text='select 99', action_kind='select', action_valid=False,
                       task_reward=0., G=data['decisions'][row['row_index'] * 2 + 1]['task_reward'])
            data['token_level_rewards'][row['row_index'], row['token_indices'][-1]] = 0.
        result = credit.credit_pipeline(arm='T3', **data)
        self.assertTrue(t.equal(result['replaced_advantages'][0, [1, 2]], t.full((2,), -.5)))
        broken = copy.deepcopy(data)
        broken['decisions'][0]['G'] += .1
        with self.assertRaisesRegex(CreditContractError, 'task-reward suffix'):
            credit.credit_pipeline(arm='T3', **broken)

    def test_unproven_alignment_masks_or_reward_extras_fail_closed(self):
        mutations = {
            'different_masks': lambda d: d['gae_mask'].__setitem__((0, 3), 1),
            'missing_reply': lambda d: d['decisions'][0]['token_indices'].pop(),
            'duplicate_reply': lambda d: d['decisions'][1]['token_indices'].__setitem__(0, 2),
            'different_shared_tokens': lambda d: d['decisions'][2]['response_token_ids'].__setitem__(0, 99),
            'unverified_input': lambda d: d['decisions'][0].__setitem__('shared_input_verified_exact', False),
            'missing_Q': lambda d: d['decisions'][0].__setitem__('Q_commit', None),
            'different_prefix_boundary': lambda d: d['decisions'][2].__setitem__('tau', 0),
            'reward_KL_extra': lambda d: d['token_level_rewards'].__setitem__((0, 1), .001),
            'missing_temporal_proof': lambda d: d['decisions'][0].__setitem__('temporal_proof', {}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                data = fixture(self.torch)
                mutate(data)
                with self.assertRaises(CreditContractError):
                    credit.credit_pipeline(arm='T3', **data)

    def test_common_config_matches_approved_actor_kl_and_actual_vagen_default(self):
        from omegaconf import OmegaConf
        root = Path(__file__).resolve().parents[1]
        common = OmegaConf.load(root / 'configs/rl_common.yaml')
        base = OmegaConf.load(Path(self.roots['VAGEN']['path']) / 'vagen/trainer/config/ppo_trainer.yaml')
        self.assertEqual(common.algorithm.adv_estimator, 'masked_gae')
        self.assertEqual((common.algorithm.gamma, common.algorithm.lam), (1., 1.))
        self.assertIs(common.algorithm.use_kl_in_reward, False)
        self.assertIs(common.actor_rollout_ref.actor.use_kl_loss, True)
        self.assertEqual(common.actor_rollout_ref.actor.kl_loss_type, 'low_var_kl')
        self.assertEqual(common.actor_rollout_ref.actor.kl_loss_coef, base.actor_rollout_ref.actor.kl_loss_coef)
        self.assertEqual(common.actor_rollout_ref.actor.kl_loss_coef, .001)


if __name__ == '__main__':
    unittest.main()
