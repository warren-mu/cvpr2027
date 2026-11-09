"""CPU fixtures for discovery cadence; original native/attempt semantics remain."""
import copy
import io
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from src.envs import pair_v2_flip_rounds_search as rounds
from src.envs.pair_v2_native_retry import NativePlacementExhausted, FreshControllerRequired


class Stream:
    def __init__(self, name, count, unavailable=False):
        self.name, self.count, self.index, self.unavailable = name, count, 0, unavailable
        self.next_pool = 'look'

    def summary(self):
        return {'pools': {'look': {'all_exhausted': self.index >= self.count},
                          'direct': {'all_exhausted': True}}}

    def __next__(self):
        if self.unavailable or self.index >= self.count:
            raise StopIteration
        point = {'position': {'x': float(self.index), 'y': 0., 'z': 0.}, 'receptacle_name': self.name}
        self.index += 1
        return {'mode': 'flip', 'w0': point, 'w1': {**point, 'position': {**point['position'], 'z': 1.}},
                'projection_pool': {'pool': 'look'}}, {}, {}


class RoundsTests(unittest.TestCase):
    def exercise(self, specifications, *, failures=None, physical=False, seen=None):
        events, written, calls = [], [], []
        state = {'rows': [], 'excluded': None, 'status': 'completed_search', 'w0_by_anchor': {},
            'selected_controller_id': 'marked_controller_000', 'proposal_trials': 0,
            'placement_exhausted_proposals': 0, 'placement_point_failures': [], 'failed_native_point_count': 0}
        sim = Mock(); sim.reachable_positions.return_value = {'positions': []}
        journal, connection = io.StringIO(), Mock()
        failures = {} if failures is None else failures
        def discovery(*args, **kwargs):
            for name, count, unavailable in specifications:
                events.append(('discover', name))
                yield {'candidate_names': ['A', 'B'], 'anchor_name': name, 'anchor_type': 'Box'}, {
                    'fixture': name, 'stream': Stream(name, count, unavailable)}
            events.append(('discovery_exhausted', None))
        def build(sim, cfg, baseline, identities, proposal, view, pair_id, directory, reachable, attempt_row):
            anchor = identities['anchor_name']
            events.append(('attempt', anchor)); calls.append((anchor, pair_id, Path(directory).name))
            attempt_row.update(mode='flip', physical_w0={'valid_physical': not physical},
                               discard_reasons=['w0_physical'] if physical else ['synthetic_rejection'])
            error = failures.get(len(calls) - 1)
            if error is not None:
                raise error
            return attempt_row
        def write(path, value):
            written.append(copy.deepcopy(value))
            return {'file': str(path), 'sha256': 'a' * 64}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            cfg = {'max_attempts': 1000, 'scene_pairs_per_class': 1000,
                   'scene': 'FloorPlan203', 'raw_evidence_dir': directory}
            stack.enter_context(patch.object(rounds.core, 'existing_views', return_value=[{}]))
            stack.enter_context(patch.object(rounds.core, 'discover_view', side_effect=discovery))
            stack.enter_context(patch.object(rounds.pools, 'PoolProposalIterator', side_effect=lambda values: values[0]['stream']))
            stack.enter_context(patch.object(rounds.base, 'build_pair', side_effect=build))
            stack.enter_context(patch.object(rounds.base, 'write_json', side_effect=write))
            stack.enter_context(patch.object(rounds.core, 'summary', side_effect=lambda state, cfg: {'status': state['status']}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            error = None
            try:
                rounds.run_search(sim, cfg, {}, state, journal, connection, set() if seen is None else seen)
            except BaseException as caught:
                error = caught
        return events, written, calls, state, error

    def test_every_active_combination_gets_one_visit_before_next_discovery(self):
        events, rows, calls, state, error = self.exercise([('A', 2, False), ('B', 2, False), ('C', 2, False)])
        self.assertIsNone(error)
        self.assertEqual(events, [('discover', 'A'), ('attempt', 'A'),
            ('discover', 'B'), ('attempt', 'A'), ('attempt', 'B'),
            ('discover', 'C'), ('attempt', 'B'), ('attempt', 'C'),
            ('discovery_exhausted', None), ('attempt', 'C')])
        self.assertEqual([item[1] for item in calls], [f'floorplan203_{i:05d}' for i in range(6)])
        self.assertTrue(all(row['counts_as_pair_attempt'] for row in rows))

    def test_required_flip_empty_domain_does_not_block_next_discovery(self):
        events, rows, calls, state, error = self.exercise([('empty', 3, True), ('good', 1, False)])
        self.assertIsNone(error)
        self.assertEqual(events[:3], [('discover', 'empty'), ('discover', 'good'), ('attempt', 'good')])
        self.assertEqual([item[0] for item in calls], ['good'])

    def test_round_snapshot_excludes_domains_added_after_round_start(self):
        first = {'stream': Stream('A', 1)}; second = {'stream': Stream('B', 1)}
        active = [first]; current = rounds.DiscoveryRound(active)
        active.append(second)
        self.assertIs(current.pop('look'), first)
        self.assertFalse(current)
        self.assertEqual(active, [first, second])
        with self.assertRaises(StopIteration): current.pop('look')

    def test_existing_requested_pool_preference_is_preserved(self):
        look = {'stream': Stream('look', 1)}
        direct = {'stream': Mock()}
        direct['stream'].summary.return_value = {'pools': {'look': {'all_exhausted': True}, 'direct': {'all_exhausted': False}}}
        current = rounds.DiscoveryRound([look, direct])
        self.assertIs(current.pop('direct'), direct)
        self.assertIs(current.pop('look'), look)

    def test_native_exhaustion_is_visited_but_does_not_consume_pair_id(self):
        failure = NativePlacementExhausted({'failed_point_count': 2, 'attempts': [], 'status': 'exhausted'})
        events, rows, calls, state, error = self.exercise([('A', 2, False), ('B', 1, False)], failures={0: failure})
        self.assertIsNone(error)
        self.assertEqual([item[1] for item in calls], ['floorplan203_00000', 'floorplan203_00000', 'floorplan203_00001'])
        self.assertEqual([row['counts_as_pair_attempt'] for row in rows], [False, True, True])
        self.assertEqual(state['placement_exhausted_proposals'], 1)
        self.assertEqual(state['failed_native_point_count'], 2)

    def test_ten_consecutive_physical_rejections_still_remove_combination(self):
        events, rows, calls, state, error = self.exercise([('A', 20, False)], physical=True)
        self.assertIsNone(error)
        self.assertEqual(len(calls), 10)
        self.assertEqual(state['search_stop_reason'], 'all_existing_views_and_pool_domains_exhausted_or_skipped')

    def test_fresh_controller_exception_preserves_raw_and_propagates(self):
        failure = FreshControllerRequired({'failed_point_count': 1, 'attempts': [], 'fresh_controller_required': True})
        events, rows, calls, state, error = self.exercise([('A', 2, False), ('B', 1, False)], failures={0: failure})
        self.assertIs(error, failure)
        self.assertEqual([event for event in events if event[0] == 'attempt'], [('attempt', 'A')])
        self.assertFalse(rows[0]['counts_as_pair_attempt'])
        self.assertEqual(state['rows'], [])

    def test_seen_proposal_is_not_rebuilt_or_counted(self):
        identity = {'candidate_names': ['A', 'B'], 'anchor_name': 'A', 'anchor_type': 'Box'}
        signature = rounds.engine.encoded([identity, 'flip', 'look', {},
            [['A', {'x': 0., 'y': 0., 'z': 0.}], ['A', {'x': 0., 'y': 0., 'z': 1.}]]])
        events, rows, calls, state, error = self.exercise([('A', 2, False)], seen={signature})
        self.assertIsNone(error)
        self.assertEqual([item[1] for item in calls], ['floorplan203_00000'])
        self.assertEqual(state['proposal_trials'], 1)

    def test_explicit_schedule_policy_is_not_silently_optional(self):
        rounds.validate_fields(rounds.fields())
        with self.assertRaises(ValueError): rounds.validate_fields({})
        altered = rounds.fields(); altered['discovery_schedule_policy']['round'] = 'one visit only'
        with self.assertRaises(ValueError): rounds.validate_fields(altered)


if __name__ == '__main__':
    unittest.main()
