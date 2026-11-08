"""Synthetic role/schedule fixtures; no rendered acceptance or measured balance."""
import copy
import hashlib
import itertools
import unittest

from src.envs import pair_v2_role_policy as policy


def token(x, label):
    point = {'x': x, 'y': 1.0, 'z': 0.0}
    return {'position': point, 'receptacle_name': 'SyntheticSurface',
            'spawn': {'selected_position': dict(point), 'synthetic_label': label}}


def views(count=3):
    return [{'position': {'x': 5., 'y': 0., 'z': float(i)}, 'yaw': 0, 'horizon': 0}
            for i in range(count)]


def proposal(mode='flip'):
    for item, view, domain in policy.view_expanded_proposals(
            [token(0., 'a0'), token(1., 'a1')], [token(9., 'b0'), token(10., 'b1')], views(1)):
        if item['mode'] == mode:
            item['role_domain'] = domain
            return item
    raise AssertionError('Synthetic domain lacks requested mode')


def row_for(mode='flip', attempt=0):
    item, role = policy.orient_proposal(proposal(mode), 'FloorPlan1', attempt)
    role.update(candidate_a_name='CandidateA', candidate_b_name='CandidateB')
    by_side = {'A': 'CandidateA', 'B': 'CandidateB'}
    return {'scene': 'FloorPlan1', 'pair_id': f'floorplan1_{attempt:05d}', 'mode': mode,
        'identities': {'candidate_names': ['CandidateA', 'CandidateB']},
        'proposal': item, 'proposal_role': role,
        'generation_version': policy.ROLE_VERSION, 'generation_policy': copy.deepcopy(policy.ROLE_POLICY),
        'target_w0': by_side[role['w0_source_side']], 'target_w1': by_side[role['w1_source_side']]}


POSITIONS = {'CandidateA': {'x': 0., 'y': 1., 'z': 0.},
             'CandidateB': {'x': 10., 'y': 1., 'z': 0.}}


class RolePolicyTests(unittest.TestCase):
    def test_fixed_seed_encoding_and_repeatability(self):
        expected = hashlib.sha256(b'{"attempt_index":17,"scene":"FloorPlan1","seed":0}').hexdigest()
        first = policy.flip_orientation('FloorPlan1', 17)
        self.assertEqual(first['assignment_digest'], expected)
        self.assertEqual(first, policy.flip_orientation('FloorPlan1', 17))
        self.assertEqual(first['flip_swapped'], bool(int(expected[:2], 16) & 1))
        self.assertNotEqual(first['assignment_digest'], policy.flip_orientation('FloorPlan2', 17)['assignment_digest'])
        self.assertNotEqual(first['assignment_digest'], policy.flip_orientation('FloorPlan1', 18)['assignment_digest'])

    def test_synthetic_balanced_eligible_domain_has_nonconstant_flip_assignment(self):
        # This is a CPU hash fixture, never a claim about retained or valid pairs.
        eligible = proposal()  # Both fixed native-token sides exist for every synthetic trial.
        oriented = [policy.orient_proposal(eligible, 'FloorPlan1', index)[0] for index in range(1000)]
        fraction = sum(item['w0']['position']['x'] < 5 for item in oriented) / len(oriented)
        self.assertGreaterEqual(fraction, .4)
        self.assertLessEqual(fraction, .6)

    def test_only_approved_seed_and_numeric_attempt_are_accepted(self):
        for scene, index, seed in [('FloorPlan1', True, 0), ('FloorPlan1', -1, 0),
                                   ('FloorPlan1', '1', 0), ('FloorPlan1', 1, 2),
                                   ('FloorPlan1', 1, False), ('not-a-scene', 1, 0)]:
            with self.assertRaises(ValueError):
                policy.flip_orientation(scene, index, seed)

    def test_flip_swaps_only_two_actual_native_tokens_and_never_mutates_input(self):
        original = proposal()
        saved = copy.deepcopy(original)
        for index in range(20):
            oriented, role = policy.orient_proposal(original, 'FloorPlan1', index)
            self.assertEqual(oriented['w0'], original['w1' if role['flip_swapped'] else 'w0'])
            self.assertEqual(oriented['w1'], original['w0' if role['flip_swapped'] else 'w1'])
            self.assertEqual(oriented['role_domain'], original['role_domain'])
        self.assertEqual(original, saved)

    def test_stay_alternates_after_multiple_views_are_expanded(self):
        stream = policy.view_expanded_proposals([token(0, 'a0'), token(1, 'a1')],
            [token(9, 'b0'), token(10, 'b1')], views(3))
        rows = list(stream)
        stays = [(view, domain) for item, view, domain in rows if item['mode'] == 'stay']
        self.assertEqual([domain['stay_side'] for _, domain in stays], ['A', 'B'] * 3)
        self.assertEqual([domain['view_index'] for _, domain in stays], [0, 0, 1, 1, 2, 2])
        self.assertEqual([domain['stay_yield_index'] for _, domain in stays], list(range(6)))
        self.assertTrue(all(not domain['fallback_due_to_exhaustion'] for _, domain in stays))
        self.assertTrue(stream.summary()['all_exhausted'])
        self.assertIsNone(stream.summary()['accepted_balance'])

    def test_original_eligible_pairs_and_views_are_preserved_once(self):
        left, right, camera_views = [token(0, 'a0'), token(1, 'a1')], [token(9, 'b0'), token(10, 'b1')], views(2)
        rows = list(policy.view_expanded_proposals(left, right, camera_views))
        def identity(item, view):
            return item['mode'], item['w0']['spawn']['synthetic_label'], item['w1']['spawn']['synthetic_label'], view['position']['z']
        observed = [identity(item, view) for item, view, _ in rows]
        expected = [identity({'mode': mode, 'w0': a, 'w1': b}, view)
                    for mode, pairs in [('flip', itertools.product(left, right)),
                        ('stay', itertools.chain(itertools.combinations(left, 2), itertools.combinations(right, 2)))]
                    for a, b in pairs for view in camera_views if a['position'] != b['position']]
        self.assertCountEqual(observed, expected)
        self.assertEqual(len(observed), len(set(observed)))

    def test_unequal_domains_report_exhaustion_without_fabricating_tokens(self):
        left = [token(0, 'a0'), token(.5, 'a1'), token(1, 'a2')]
        right = [token(9, 'b0'), token(10, 'b1')]
        rows = list(policy.view_expanded_proposals(left, right, views(2)))
        stays = [(item, domain) for item, _, domain in rows if item['mode'] == 'stay']
        self.assertEqual([domain['stay_side'] for _, domain in stays[:4]], ['A', 'B', 'A', 'B'])
        self.assertTrue(all(domain['stay_side'] == 'A' for _, domain in stays[4:]))
        self.assertTrue(any(domain['fallback_due_to_exhaustion'] for _, domain in stays[4:]))
        for item, domain in stays[4:]:
            self.assertIn('stay_b', domain['exhausted_after_yield'])
            self.assertEqual(domain['remaining_after_yield']['stay_b'], 0)
            self.assertIn(item['w0'], left)
            self.assertIn(item['w1'], left)

    def test_empty_side_and_no_views_are_explicit_known_domains(self):
        left = [token(0, 'a0'), token(1, 'a1')]
        stream = policy.view_expanded_proposals(left, [], views(2))
        rows = list(stream)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(item['mode'] == 'stay' and domain['stay_side'] == 'A' for item, _, domain in rows))
        self.assertEqual(stream.summary()['unavailable_at_start'], ['flip', 'stay_b'])
        no_views = policy.view_expanded_proposals(left, [token(9, 'b0')], [])
        self.assertEqual(list(no_views), [])
        self.assertEqual(no_views.summary()['view_count'], 0)
        self.assertTrue(no_views.summary()['all_exhausted'])

    def test_original_same_coordinate_rejection_is_retained(self):
        first, second = token(0, 'one'), token(0, 'different-native-token')
        rows = list(policy.view_expanded_proposals([first, second], [], views()))
        self.assertEqual(rows, [])

    def test_stay_has_no_random_reorientation_or_hidden_new_acceptance_gate(self):
        first = proposal('stay')
        for index in (0, 1, 55):
            oriented, role = policy.orient_proposal(first, 'FloorPlan1', index)
            self.assertEqual(oriented, first)
            self.assertIsNone(role['assignment_digest'])
            self.assertIsNone(role['flip_swapped'])
        row = row_for('stay')
        row['target_w0'] = row['target_w1'] = 'CandidateB'
        check = policy.verify_proposal_role(row, POSITIONS)
        self.assertTrue(check['passed'])
        self.assertFalse(check['actual_targets_match_intended'])
        self.assertEqual(check['actual_candidate_roles'], ['B', 'B'])

    def test_verifier_binds_hash_native_coordinates_and_stable_names(self):
        row = row_for()
        self.assertTrue(policy.verify_proposal_role(row, POSITIONS)['passed'])
        changed = copy.deepcopy(row); changed['proposal_role']['assignment_digest'] = '0' * 64
        with self.assertRaises(ValueError):
            policy.verify_proposal_role(changed, POSITIONS)
        changed = copy.deepcopy(row); changed['proposal']['w0']['spawn']['selected_position']['x'] += 3
        with self.assertRaises(ValueError):
            policy.verify_proposal_role(changed, POSITIONS)
        changed = copy.deepcopy(row); changed['identities']['candidate_names'].reverse()
        with self.assertRaises(ValueError):
            policy.verify_proposal_role(changed, POSITIONS)
        swapped_baseline = {'CandidateA': POSITIONS['CandidateB'], 'CandidateB': POSITIONS['CandidateA']}
        with self.assertRaises(ValueError):
            policy.verify_proposal_role(row, swapped_baseline)

    def test_stay_false_fallback_claim_is_rejected(self):
        row = row_for('stay')
        domain = row['proposal']['role_domain']
        domain['fallback_due_to_exhaustion'] = True
        # Keep redundant metadata consistent so the schedule check is exercised.
        row['proposal_role']['original_role_domain'] = copy.deepcopy(domain)
        with self.assertRaises(ValueError):
            policy.verify_proposal_role(row, POSITIONS)


if __name__ == '__main__':
    unittest.main()
