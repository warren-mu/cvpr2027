"""Synthetic finite-domain checks only; never claims rendered Look coverage."""
from collections import Counter
import copy
import itertools
import json
import unittest

from src.envs import pair_v2_coverage_order as coverage
from src.envs import pair_v2_role_policy as roles


def token(side, ordinal):
    position = {'x': (0 if side == 'A' else 10) + ordinal / 10, 'y': 1., 'z': 0.}
    return {'position': position, 'receptacle_name': 'Surface',
            'spawn': {'selected_position': copy.deepcopy(position), 'source': f'{side}{ordinal}'}}


def views(positions=3):
    return [{'position': {'x': float(index), 'y': 0., 'z': 2.}, 'yaw': heading, 'horizon': horizon}
            for index in range(positions) for heading in (0, 315, 45) for horizon in (0, 30)]


def identity(row):
    proposal, view, _ = row
    return json.dumps([proposal, view], sort_keys=True)


class CoverageOrderTests(unittest.TestCase):
    def test_product_bijection_including_non_coprime_and_empty_domains(self):
        for pair_count, view_count in ((0, 4), (4, 0), (1, 6), (6, 1), (4, 6), (6, 6), (3, 8)):
            order = coverage.product_index_order(pair_count, list(reversed(range(view_count))))
            expected = list(itertools.product(range(pair_count), range(view_count)))
            self.assertCountEqual(order, expected)
            self.assertEqual(len(order), len(set(order)))
        for invalid in ([0, 0], [1], [-1]):
            with self.assertRaises(ValueError):
                coverage.product_index_order(2, invalid)

    def test_exact_original_multiset_no_new_or_lost_native_tokens(self):
        for a_count, b_count, position_count in ((3, 3, 3), (3, 2, 2), (2, 0, 1), (1, 1, 1), (2, 2, 0)):
            left, right = [token('A', i) for i in range(a_count)], [token('B', i) for i in range(b_count)]
            camera_views = views(position_count)
            before = copy.deepcopy((left, right, camera_views))
            old = list(roles.RoleProposalIterator(left, right, camera_views))
            new = list(coverage.CoverageProposalIterator(left, right, camera_views))
            self.assertEqual(Counter(map(identity, old)), Counter(map(identity, new)))
            self.assertEqual(len(new), len(set(map(identity, new))))
            self.assertEqual((left, right, camera_views), before)

    def test_short_prefix_reaches_distinct_geometry_positions_and_headings(self):
        stream = coverage.CoverageProposalIterator(
            [token('A', i) for i in range(3)], [token('B', i) for i in range(3)], views(3))
        first = list(itertools.islice((row for row in stream if row[0]['mode'] == 'flip'), 3))
        self.assertEqual(len({(r[0]['w0']['spawn']['source'], r[0]['w1']['spawn']['source']) for r in first}), 3)
        self.assertEqual(len({r[1]['position']['x'] for r in first}), 3)
        self.assertEqual([row[1]['yaw'] for row in first], [0, 315, 45])
        self.assertEqual([row[2]['pair_index'] for row in first], [0, 1, 2])

    def test_inherited_mode_stay_alternation_and_truthful_unequal_exhaustion(self):
        left, right, camera_views = [token('A', i) for i in range(3)], [token('B', i) for i in range(2)], views(1)
        old = list(roles.RoleProposalIterator(left, right, camera_views))
        new = list(coverage.CoverageProposalIterator(left, right, camera_views))
        keys = ('mode', 'stay_side', 'stay_yield_index', 'requested_stay_side',
                'fallback_due_to_exhaustion', 'requested_mode', 'mode_fallback',
                'remaining_after_yield', 'exhausted_after_yield', 'unavailable_at_start')
        self.assertEqual([{key: row[2][key] for key in keys} for row in old],
                         [{key: row[2][key] for key in keys} for row in new])

    def test_same_actual_attempt_keeps_original_role_seed_assignment(self):
        stream = coverage.CoverageProposalIterator([token('A', i) for i in range(2)],
            [token('B', i) for i in range(2)], views(1))
        for attempt, (proposal, _, domain) in enumerate(stream):
            proposal['role_domain'] = domain
            oriented, role = roles.orient_proposal(proposal, 'FloorPlan1', attempt, seed=0)
            self.assertEqual(role['attempt_index'], attempt)
            if proposal['mode'] == 'flip':
                original = roles.flip_orientation('FloorPlan1', attempt, seed=0)
                self.assertEqual(role['assignment_digest'], original['assignment_digest'])
                self.assertEqual(oriented['w0'], proposal['w1' if original['flip_swapped'] else 'w0'])
            else:
                self.assertEqual(oriented, proposal)
                self.assertIsNone(role['assignment_digest'])

    def test_deterministic_indices_and_duplicate_input_occurrences_preserved(self):
        camera_views = views(2)
        camera_views.append(copy.deepcopy(camera_views[0]))
        order = coverage.view_index_order(camera_views)
        self.assertEqual(order, coverage.view_index_order(copy.deepcopy(camera_views)))
        self.assertCountEqual(order, range(len(camera_views)))
        left, right = [token('A', 0), token('A', 1)], [token('B', 0)]
        old = list(roles.RoleProposalIterator(left, right, camera_views))
        new = list(coverage.CoverageProposalIterator(left, right, camera_views))
        self.assertEqual(Counter(map(identity, old)), Counter(map(identity, new)))
        self.assertEqual(new, list(coverage.CoverageProposalIterator(left, right, camera_views)))

    def test_same_position_filter_and_empty_view_summary_are_unchanged(self):
        same = token('A', 0)
        self.assertEqual(list(coverage.CoverageProposalIterator([same, copy.deepcopy(same)], [], views(1))), [])
        empty = coverage.CoverageProposalIterator([token('A', 0)], [token('B', 0)], [])
        self.assertEqual(list(empty), [])
        self.assertTrue(empty.summary()['all_exhausted'])
        self.assertEqual(empty.summary()['proposal_order_version'], coverage.ORDER_VERSION)
        self.assertIsNone(empty.summary()['accepted_balance'])

    def test_private_factory_restored_on_failure_and_configuration_is_explicit(self):
        original = roles.view_expanded_proposals
        self.assertIsNot(original, coverage.coverage_view_expanded_proposals)
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            with coverage.install_coverage_order():
                self.assertIs(roles.view_expanded_proposals, coverage.coverage_view_expanded_proposals)
                with self.assertRaises(ValueError):
                    with coverage.install_coverage_order():
                        pass
                raise RuntimeError('synthetic')
        self.assertIs(roles.view_expanded_proposals, original)
        config = coverage.enable_config({'seed': 0, 'max_attempts': 160})
        self.assertEqual(config['seed'], 0)
        self.assertEqual(config['max_attempts'], 160)
        self.assertEqual(config['proposal_order_policy'], coverage.ORDER_POLICY)


if __name__ == '__main__':
    unittest.main()
