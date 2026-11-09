"""Frozen §4b eligibility plus genuine torch gradients when torch is installed."""
import copy
import unittest

from src.rl import t4_consistency as t4
try:
    import torch
except ImportError:
    torch = None


def pair(change='stay', rho=.5):
    return [{'episode_id': world, 'pair_instance_id': 'instance', 'pair_id': 'dataset:key',
        'world_id': world, 'kind': 'look_' + change, 'rho': rho,
        'candidate_names': ['stableA', 'stableB'],
        'target_name': 'stableA' if change == 'stay' or world == 'w0' else 'stableB',
        'last_legal_select': {'action_kind': 'select', 'action_valid': True,
            'decision_index': 2, 'x_seen_step': 2,
            'public_select_candidate_names': ['stableA', 'stableB']}}
        for world in ('w0', 'w1')]


class EligibilityTests(unittest.TestCase):
    def test_inclusive_own_seen_boundary_and_mapped_flip_targets(self):
        rows = pair('flip'); rows[0]['target_name'], rows[1]['target_name'] = 'stableB', 'stableA'
        out = t4.descriptors(rows)
        self.assertTrue(all(row['applicable'] for row in out.values()))
        self.assertEqual(out['w0']['target_name'], 'stableB')
        self.assertEqual(out['w0']['other_target_name'], 'stableA')
        rows[0]['last_legal_select']['x_seen_step'] = None
        out = t4.descriptors(rows)
        self.assertEqual(out['w0']['skip_reason'], 'own_anchor_not_seen')
        self.assertTrue(out['w1']['applicable'])

    def test_flip_invisible_target_zero_and_stay_failure_zeroes_both_worlds(self):
        rows = pair('flip'); rows[0]['last_legal_select']['public_select_candidate_names'] = ['stableA']
        out = t4.descriptors(rows)
        self.assertEqual(out['w0']['skip_reason'], 'flip_required_target_unselectable')
        self.assertTrue(out['w1']['applicable'])
        for failure in ('missing_select', 'unseen', 'missing_candidate'):
            rows = pair()
            if failure == 'missing_select': rows[0]['last_legal_select'] = None
            elif failure == 'unseen': rows[0]['last_legal_select']['x_seen_step'] = None
            else: rows[0]['last_legal_select']['public_select_candidate_names'] = ['stableA']
            out = t4.descriptors(rows)
            self.assertFalse(any(row['applicable'] for row in out.values()))
            self.assertEqual(out['w0']['skip_reason'], out['w1']['skip_reason'])

    def test_legal_select_proof_cannot_be_replaced_by_a_parsed_invalid_action(self):
        rows = pair(); rows[0]['last_legal_select']['action_valid'] = False
        with self.assertRaises(ValueError): t4.descriptors(rows)
        rows = pair(); rows[1]['candidate_names'] = ['otherA', 'stableB']
        with self.assertRaises(ValueError): t4.descriptors(rows)


@unittest.skipIf(torch is None, 'real torch unavailable locally; requires the logged pinned environment')
class TorchGradientTests(unittest.TestCase):
    def test_flip_both_current_scores_have_real_gradients_and_zero_rows_still_divide_N(self):
        rows = pair('flip', rho=.25); rows[1]['last_legal_select'] = None
        descriptors = t4.descriptors(rows)
        a = torch.tensor(-2., dtype=torch.float64, requires_grad=True)
        b = torch.tensor(-1., dtype=torch.float64, requires_grad=True)
        term = t4.term(descriptors['w0'], {'stableA': a, 'stableB': b})
        loss = t4.weighted_minibatch_loss({'w0': term}, minibatch_episode_ids=['w0', 'w1'],
                                         descriptors_by_episode=descriptors)
        self.assertAlmostEqual(loss.item(), .1 * 1.5 * 2 / 2)
        loss.backward()
        self.assertAlmostEqual(a.grad.item(), -.075)
        self.assertAlmostEqual(b.grad.item(), .075)

    def test_stay_direction_is_current_to_detached_sampled_other_not_joint_symmetric(self):
        descriptors = t4.descriptors(pair())
        a = torch.tensor(-2., dtype=torch.float64, requires_grad=True)
        b = torch.tensor(-1., dtype=torch.float64, requires_grad=True)
        oa = torch.tensor(-1., dtype=torch.float64, requires_grad=True)
        ob = torch.tensor(-3., dtype=torch.float64, requires_grad=True)
        got = t4.term(descriptors['w0'], {'stableA': a, 'stableB': b},
                      sampled_other_log_probs={'stableA': oa, 'stableB': ob})
        current = torch.stack((a, b)); opposite = torch.stack((oa, ob)).detach()
        lp, lq = torch.log_softmax(current, 0), torch.log_softmax(opposite, 0)
        expected = (lp.exp() * (lp - lq)).sum()
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        got.backward()
        self.assertIsNotNone(a.grad); self.assertIsNotNone(b.grad)
        self.assertIsNone(oa.grad); self.assertIsNone(ob.grad)

    def test_missing_current_scores_or_cross_minibatch_terms_are_rejected(self):
        descriptors = t4.descriptors(pair('flip'))
        x = torch.tensor(-1., requires_grad=True)
        with self.assertRaises(ValueError): t4.term(descriptors['w0'], {'stableA': x})
        with self.assertRaises(ValueError):
            t4.weighted_minibatch_loss({'w0': x, 'w1': x}, minibatch_episode_ids=['w0'],
                                      descriptors_by_episode=descriptors)


if __name__ == '__main__':
    unittest.main()
