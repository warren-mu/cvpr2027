"""CPU scheduling fixtures, not rendered or accepted scientific observations."""
import copy
import time
import unittest
from unittest.mock import patch

from src.envs import pair_v2_stay_priority as priority
from src.envs import pair_v2_stay_priority_policy as policy
from src.envs import pair_v2_stay_priority_shard as producer
from src.envs import pair_v2_flip_search as flip
from src.envs import pair_v2_projection_pool as pools
from src.envs import pair_v2_stay_order as stay
from src.envs import pair_v2_fixed_labels as labels
from src.tools import t01v2_stay_priority_parallel as launcher


def token(x):
    point = {'x': float(x), 'y': 0., 'z': 1.}
    return {'position': point, 'receptacle_name': 'Surface', 'spawn': {
        'selected_position': dict(point), 'receptacle_name': 'Surface', 'coordinates': [dict(point)],
        'query_reference': {'file': 'synthetic-query', 'sha256': 'a' * 64}}}


def record(name):
    entries = {side: [{'token': token(sign * value),
                      'projection_point': {'x': float(sign * value), 'y': .1, 'z': 1.}}
                     for value in (1, 2, 3)] for side, sign in (('A', -1), ('B', 1))}
    return {'version': pools.POOL_VERSION, 'flip_search_version': flip.VERSION,
        'view': {'fixture': name}, 'reference': {'file': name + '.json', 'sha256': 'b' * 64},
        'camera': {'fixture': name}, 'observation_cameras': [], 'counts': {},
        'flip_geometry_context': {'anchor': {'name': 'X'}},
        'points': {'look': entries, 'direct': {'A': [], 'B': []}}, 'stay_points': entries}


def item(name, quota):
    value = record(name)
    return {'identity': {'anchor_name': name}, 'record': value,
            'stream': priority.PriorityIterator([value], quota), 'consecutive_physical_rejections': 0}


def prediction(scores):
    def predict(entry, anchor, camera, actions):
        selected = scores[camera['fixture']]
        value = selected(entry['token']['position']['x']) if callable(selected) else selected
        return {'after_actions': [{'projection': {'projectable': value is not None,
            'clipped_rectangle_area_px2': value}}]}
    return predict


def counted(mode):
    return {'counts_as_pair_attempt': True, 'mode': mode}


def configuration():
    declared = {**policy.camera_fields(60, 'FloorPlan203'), **stay.fields(),
                'scene': 'FloorPlan203', 'producer_module': policy.PRODUCER}
    return {**declared, 'shard': copy.deepcopy(declared), 'deadline': time.time() + 60}


class StayPriorityTests(unittest.TestCase):
    def test_peek_reads_original_far_first_row_without_consuming_stream_or_native_tokens(self):
        quota, order = flip.AttemptQuota(), priority.DomainOrder()
        domain = item('a', quota); order.ordered([domain])
        stream = domain['stream'].streams['look']
        before = copy.deepcopy((stream.cursor, stream.stay_count, stream.next_kind, stream.summary()))
        native = domain['record']['stay_points']['A'][0]['token']['spawn']['coordinates']
        with patch.object(priority.boxes, 'predict', side_effect=prediction({'a': 12.})):
            proof = priority.peek_stay(domain, check_deadline=lambda: None)
        self.assertEqual(before, (stream.cursor, stream.stay_count, stream.next_kind, stream.summary()))
        self.assertIs(native, domain['record']['stay_points']['A'][0]['token']['spawn']['coordinates'])
        self.assertEqual(proof['planned_native_references']['w0']['position']['x'], -1.)
        self.assertEqual(proof['planned_native_references']['w1']['position']['x'], -3.)
        self.assertEqual(proof['priority_min_area_px2'], 12.)
        self.assertNotIn('coordinates', str(proof))

    def test_real_partial_round_prioritizes_min_world_area_and_next_round_restores_flip_order(self):
        quota, order = flip.AttemptQuota(), priority.DomainOrder()
        quota.record('earned', counted('flip'))
        a, b, unknown = (item(name, quota) for name in ('a', 'b', 'unknown'))
        scores = {'a': lambda x: 100. if x == -1. else 1., 'b': 20., 'unknown': None}
        with patch.object(priority.boxes, 'predict', side_effect=prediction(scores)):
            current = priority.PriorityRound([a, b, unknown], quota, order, check_deadline=lambda: None)
            chosen = current.pop('direct')  # the old Look-only cadence also requests Direct here
            self.assertIs(chosen, b)
            proposal, _, domain = next(chosen['stream'])
            self.assertEqual((proposal['mode'], domain['stay_side']), ('stay', 'A'))
            self.assertEqual(proposal[priority.SELECTION]['priority_min_area_px2'], 20.)
            quota.record('b_stay', counted('stay'))
            self.assertIs(current.pop('look'), a)  # original Flip branch order
            flip_proposal, _, _ = next(a['stream'])
            self.assertEqual(flip_proposal['mode'], 'flip')
            self.assertIsNone(flip_proposal[priority.SELECTION])
            quota.record('a_flip', counted('flip'))
            self.assertIs(current.pop('look'), unknown)
            proposal, _, _ = next(unknown['stream'])
            self.assertIsNone(proposal[priority.SELECTION]['priority_min_area_px2'])
            quota.record('unknown_stay', counted('stay'))
            self.assertFalse(current)
            with self.assertRaises(StopIteration): current.pop('look')
            following = priority.PriorityRound([b, a, unknown], quota, order, check_deadline=lambda: None)
            self.assertEqual(following.pending, [a, b, unknown])
            self.assertIs(following.pop('look'), a)
            self.assertEqual(next(a['stream'])[0]['mode'], 'flip')

    def test_stable_ties_unknown_retention_and_no_phantom_counted_attempt(self):
        quota, order = flip.AttemptQuota(), priority.DomainOrder()
        quota.record('earned', counted('flip'))
        a, b, unknown = (item(name, quota) for name in ('a', 'b', 'unknown'))
        with patch.object(priority.boxes, 'predict', side_effect=prediction({'a': 0., 'b': 0., 'unknown': None})):
            current = priority.PriorityRound([unknown, a, b], quota, order, check_deadline=lambda: None)
            self.assertIs(current.pop('look'), a)
            proposal, _, _ = next(a['stream'])
            quota.record('native-failure', {'counts_as_pair_attempt': False, 'mode': proposal['mode']})
            self.assertEqual(quota.wanted(), 'stay')
            self.assertIs(current.pop('look'), b)
            next(b['stream'])
            self.assertIs(current.pop('look'), unknown)
            next(unknown['stream'])
            self.assertFalse(current)
            self.assertEqual(quota.summary()['counted_stay'], 0)

    def test_deadline_propagates_without_selecting_or_consuming_any_domain(self):
        quota, order = flip.AttemptQuota(), priority.DomainOrder()
        quota.record('earned', counted('flip'))
        a = item('a', quota); calls = []
        def check():
            calls.append(None)
            if len(calls) == 3:
                raise TimeoutError('original deadline before world prediction')
        current = priority.PriorityRound([a], quota, order, check_deadline=check)
        before = copy.deepcopy(a['stream'].summary())
        with self.assertRaisesRegex(TimeoutError, 'original deadline'):
            current.pop('look')
        self.assertEqual(current.pending, [a])
        self.assertEqual(a['stream'].summary(), before)
        self.assertFalse(hasattr(a['stream'], '_stay_priority_pending'))

    def test_selection_is_planned_only_and_schema_cannot_impute_unknown_or_change_original_row(self):
        quota, order = flip.AttemptQuota(), priority.DomainOrder()
        quota.record('earned', counted('flip')); a = item('a', quota)
        with patch.object(priority.boxes, 'predict', side_effect=prediction({'a': None})):
            current = priority.PriorityRound([a], quota, order, check_deadline=lambda: None)
            current.pop('look'); proposal, _, _ = next(a['stream'])
        row = {**priority.fields(), 'mode': 'stay', 'proposal': proposal,
            'projection_pool_reference': a['record']['reference'], priority.SELECTION: proposal[priority.SELECTION]}
        priority.validate_selection(row)
        fallback = copy.deepcopy(row); fallback['proposal']['w1'] = token(-2.)
        priority.validate_selection(fallback)  # actual native fallback is checked by the original scientific chain
        for key, value in (('priority_min_area_px2', 0.), ('known_both', True), ('pair_index', 99)):
            bad = copy.deepcopy(row); bad[priority.SELECTION][key] = value
            with self.assertRaises(ValueError): priority.validate_selection(bad)
        with self.assertRaises(ValueError): priority.validate_selection({**row, 'mode': 'flip'})
        priority.validate_selection({'mode': 'flip', priority.SELECTION: None})

    def test_source_tags_scope_and_actual_spawn_entry_hooks_restore(self):
        cfg = configuration(); policy.validate_source(cfg, cfg)
        old = producer.ranked.rounds.original.marked
        previous = (producer.ranked.policy, producer.ranked.cadence.DiscoveryRound,
                    old.generator_worker, old.replay_worker, old.pools.PoolProposalIterator)
        def native(actual, connection):
            self.assertIs(actual, cfg)
            self.assertIs(old.generator_worker, producer.generator_worker)
            self.assertIs(old.replay_worker, producer.replay_worker)
            priority.validate_fields(old.tags())
            stream = old.pools.PoolProposalIterator([record('a')])
            self.assertIsInstance(stream, priority.PriorityIterator)
            raise RuntimeError('original native failure')
        with patch.object(producer.ranked.rounds.original.look.stay, '_ORIGINAL_GENERATOR', side_effect=native):
            with self.assertRaisesRegex(RuntimeError, 'original native failure'):
                producer.generator_worker(cfg, None)
        self.assertEqual(previous, (producer.ranked.policy, producer.ranked.cadence.DiscoveryRound,
                    old.generator_worker, old.replay_worker, old.pools.PoolProposalIterator))
        for key in priority.FIELDS:
            changed = copy.deepcopy(cfg); changed.pop(key)
            with self.assertRaises(ValueError): policy.validate_source(changed)
        for camera, scene in ((100, 'FloorPlan203'), (60, 'FloorPlan1')):
            with self.assertRaises(ValueError): policy.camera_fields(camera, scene)

    def test_replay_and_parent_delegate_complete_existing_paths_with_explicit_new_policy(self):
        cfg = configuration(); cfg.update(labels.pair_fields('synthetic', ['A', 'B']), candidate_names=['A', 'B'])
        def replay(actual, connection):
            self.assertEqual(labels._CONTEXT.get()[0], 'synthetic')
            return {'status': 'original synthetic replay'}
        with patch.object(producer.ranked.rounds.original.look, '_ORIGINAL_REPLAY', side_effect=replay):
            result = producer.replay_worker(cfg, None)
        priority.validate_fields(result); labels.validate_pair(result, 'synthetic', ['A', 'B'])
        previous = launcher.original.policy
        def parent():
            self.assertIs(launcher.original.policy, policy)
            return 'original guarded parent'
        with patch.object(launcher.original, 'main', side_effect=parent):
            self.assertEqual(launcher.main(), 'original guarded parent')
        self.assertIs(launcher.original.policy, previous)


if __name__ == '__main__':
    unittest.main()
