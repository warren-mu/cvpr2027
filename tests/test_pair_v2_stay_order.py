"""Synthetic CPU-only supplement 11 search and isolated-entry fixtures."""
import copy
import itertools
import unittest
from unittest.mock import patch

from src.envs import pair_v2_projection_pool as pools
from src.envs import pair_v2_role_policy as roles
from src.envs import pair_v2_stay_order as order
from src.envs import pair_v2_stay_shard as shard
from src.envs import pair_v2_marked_shard as marked
from src.envs import pair_v2_native_retry as native
from src.tools import t01v2_stay_parallel as parallel
from src.tools import t01v2_stay_review_checkpoint as review


def token(x, label=None):
    point = {'x': x, 'y': .5, 'z': 1.}
    return {'position': point, 'receptacle_name': 'Surface', 'spawn': {
        'receptacle_name': 'Surface', 'selected_position': dict(point),
        'query_reference': {'file': label or str(x), 'sha256': 'a' * 64}}}


def record():
    return {'version': pools.POOL_VERSION, 'reference': {'file': 'pool', 'sha256': 'c' * 64},
        'view': {'position': {'x': 0., 'y': 0., 'z': 0.}, 'yaw': 0, 'horizon': 0},
        'identity': {'candidate_names': ['A', 'B'], 'anchor_name': 'X'}, 'counts': {},
        'points': {pool: {side: [{'token': token(x)} for x in (0., 1., 4., 7.)]
                          for side in ('A', 'B')} for pool in ('look', 'direct')}}


def proposal(rec, w0=0, w1=1, mode='stay'):
    values = rec['points']['look']['A']
    return {'mode': mode, 'w0': copy.deepcopy(values[w0]['token']),
        'w1': copy.deepcopy(values[w1]['token']), 'projection_pool': {'pool': 'look'}}


class Simulator:
    """No renderer or physical claims: explicit recorded-ledger CPU fixture."""
    def __init__(self, w0=2, settled=3.9):
        self.w0_index, self.settled = w0, settled
        self.sequence, self.offers, self.error_after_native = 0, [], False
        self.values = [token(x) for x in (0., 1., 4., 7.)]

    def construct_world(self, baseline, key, placements, require_agent=None):
        self.sequence += 1
        if key.endswith('_w0'):
            actual = self.values[self.w0_index]['spawn']
            position = {**actual['selected_position'], 'x': self.settled}
            retry = {}
        else:
            supplied = [placements[0]['spawn'], *placements[0]['alternatives']]
            self.offers.append(copy.deepcopy(supplied))
            retry = native.place_native_with_retries(self, 'X', supplied, rotation={}, record_key=key)
            if self.error_after_native:
                raise RuntimeError('synthetic later settling failure')
            actual = retry['successful_token']
            position = actual['selected_position']
        snap = {'reference': {'file': 'actual' + str(self.sequence), 'sha256': 'd' * 64},
            'objects': [{'name': 'X', 'objectType': 'Vase', 'position': copy.deepcopy(position)}]}
        return {'snapshot': snap, 'passed': True, 'successful_tokens_by_name': {'X': copy.deepcopy(actual)},
            'retry_records': {'X': retry}}


def place_fixture(sim, name, tokens, **kwargs):
    return {'object_name': name, 'successful_token': copy.deepcopy(tokens[0]),
        'ledger_reference': {'file': 'ledger' + str(sim.sequence), 'sha256': 'e' * 64},
        'attempts': [{'spawn_token': copy.deepcopy(tokens[0]), 'native_success': True}]}


def run_construction(sim, rec, selected=1, mode='stay'):
    prop = proposal(rec, w1=selected, mode=mode)
    role = {'w0_source_side': 'A', 'w1_source_side': 'A', 'candidate_a_name': 'A', 'candidate_b_name': 'B'}
    row = {}
    _, observed = shard.construct_observer(sim, prop, role, rec, row)
    baseline = {'reference': {'file': 'common_baseline', 'sha256': 'b' * 64}}
    first = observed(baseline, 'pair_w0', [{'object_name': 'X', 'spawn': prop['w0']['spawn']}])
    # w1 baseline deliberately has a fresh reference every time, just as production.
    second = observed(first['snapshot'], 'pair_w1', [{'object_name': 'X', 'spawn': prop['w1']['spawn']}])
    return row, prop, second


class StayOrderTests(unittest.TestCase):
    def test_enumeration_keeps_inventory_flip_sequence_and_far_first_suffix(self):
        rec = record(); original = pools._PoolRoles([rec], 'look'); changed = order.FarFirstPoolRoles([rec], 'look')
        self.assertEqual(original.rows['flip'], changed.rows['flip'])
        self.assertEqual(original.totals, changed.totals)
        for key in ('stay_a', 'stay_b'):
            old = [(pools._point_key(a), pools._point_key(b)) for a, b in original.pairs[key]]
            new = [(pools._point_key(a), pools._point_key(b)) for a, b in changed.pairs[key]]
            self.assertEqual(set(old), set(new)); self.assertEqual(len(new), len(set(new)))
            self.assertEqual([b['position']['x'] for a, b in changed.pairs[key] if a['position']['x'] == 0], [7, 4, 1])

    def test_pool_mode_role_schedules_and_seed0_unchanged(self):
        records = [record(), record()]; records[1]['view']['horizon'] = 30
        old = list(pools.PoolProposalIterator(records))
        with shard.installed():
            new = list(pools.PoolProposalIterator(records))
        signature = lambda rows: [(p['mode'], p['projection_pool'], d) for p, _, d in rows]
        self.assertEqual(signature(old), signature(new))
        for index, ((before, _, _), (after, _, _)) in enumerate(zip(old, new)):
            if before['mode'] == 'flip':
                self.assertEqual(before, after)
                self.assertEqual(roles.orient_proposal(before, 'FloorPlan1', index, seed=0),
                                 roles.orient_proposal(after, 'FloorPlan1', index, seed=0))

    def test_native_order_ties_tokens_and_tried_filter_are_deterministic(self):
        rec = record(); items = [token(0), token(-1), token(1)]
        rec['points']['look']['A'] = [{'token': value} for value in items]
        source = copy.deepcopy(rec)
        ordered = order.alternatives(rec, 'look', 'A', items[2], items[0], items[0]['position'], set(), w0_fell_back=True)
        self.assertEqual(ordered, [items[1], items[2]])
        self.assertEqual(rec, source)
        self.assertEqual(order.alternatives(rec, 'look', 'A', items[2], items[0], items[0]['position'],
            {pools._point_key(items[1])}, w0_fell_back=True), [items[2]])

    def test_actual_fallback_and_settled_origin_do_not_repeat_previous_w1(self):
        rec, sim = record(), Simulator(w0=2, settled=3.4)
        with patch.object(native, 'place_native_with_retries', place_fixture):
            first, prop1, _ = run_construction(sim, rec)
            second, prop2, _ = run_construction(sim, rec)
        self.assertEqual(prop1['w0']['position']['x'], 4)
        # Requested w0=4 would prefer 0, but actual settled x=3.4 prefers 7.
        self.assertEqual(prop1['w1']['position']['x'], 7)
        self.assertEqual(prop2['w1']['position']['x'], 0)
        self.assertNotEqual(first['stay_w1_search_order']['w0_snapshot_reference'], second['stay_w1_search_order']['w0_snapshot_reference'])
        self.assertEqual(first['stay_w1_search_order']['common_baseline_reference'], second['stay_w1_search_order']['common_baseline_reference'])
        self.assertEqual(len(second['stay_w1_search_order']['previously_attempted_native_keys']), 1)
        self.assertEqual([t['selected_position']['x'] for t in sim.offers[0]], [7, 0, 1])
        self.assertEqual([t['selected_position']['x'] for t in sim.offers[1]], [0, 1])
        self.assertEqual(len(sim._stay_w1_tried), 1)

    def test_unexecuted_offers_remain_available_and_later_physical_failure_marks_executed(self):
        rec, sim = record(), Simulator(w0=2, settled=3.4)
        sim.error_after_native = True
        with patch.object(native, 'place_native_with_retries', place_fixture):
            with self.assertRaisesRegex(RuntimeError, 'settling'):
                run_construction(sim, rec)
            sim.error_after_native = False
            _, proposal_next, _ = run_construction(sim, rec)
        self.assertEqual(proposal_next['w1']['position']['x'], 0)
        self.assertEqual(len(next(iter(sim._stay_w1_tried.values()))), 2)

    def test_partial_exception_ledger_is_marked_and_fresh_controller_resets_state(self):
        rec, sim = record(), Simulator(w0=2, settled=3.4)
        def fail(sim, name, tokens, **kwargs):
            ledger = place_fixture(sim, name, tokens, **kwargs)
            ledger['attempts'][0]['native_success'] = False
            ledger['status'] = 'exhausted'
            raise native.NativePlacementExhausted(ledger)
        with patch.object(native, 'place_native_with_retries', fail):
            with self.assertRaises(native.NativePlacementExhausted):
                run_construction(sim, rec)
        with patch.object(native, 'place_native_with_retries', place_fixture):
            _, prop_old, _ = run_construction(sim, rec)
            _, prop_new, _ = run_construction(Simulator(w0=2, settled=3.4), rec)
        self.assertEqual(prop_old['w1']['position']['x'], 0)
        self.assertEqual(prop_new['w1']['position']['x'], 7)

    def test_flip_keeps_selected_first_and_never_allocates_stay_state(self):
        rec, sim = record(), Simulator(w0=0, settled=0)
        with patch.object(native, 'place_native_with_retries', place_fixture):
            row, prop, _ = run_construction(sim, rec, mode='flip')
        self.assertEqual(prop['w1']['position']['x'], 1)
        self.assertNotIn('stay_w1_search_order', row)
        self.assertFalse(hasattr(sim, '_stay_w1_tried'))

    def test_scheduled_distinct_points_are_not_reset_to_same_farthest_point(self):
        rec, sim = record(), Simulator(w0=0, settled=0)
        with patch.object(native, 'place_native_with_retries', place_fixture):
            actual = [run_construction(sim, rec, selected=i)[1]['w1']['position']['x'] for i in (3, 2, 1)]
        self.assertEqual(actual, [7, 4, 1])

    def test_worker_installs_hooks_in_its_own_process_and_restores_them(self):
        config = {**order.fields(), 'producer_module': order.PRODUCER,
                  'shard': {**order.fields(), 'producer_module': order.PRODUCER}}
        before = pools._PoolRoles, marked.construct_observer, marked.generator_worker
        def original(cfg, connection):
            self.assertIs(pools._PoolRoles, order.FarFirstPoolRoles)
            self.assertIs(marked.construct_observer, shard.construct_observer)
            self.assertIs(marked.generator_worker, shard.generator_worker)
            return 'synthetic worker result'
        with patch.object(shard, '_ORIGINAL_GENERATOR', original):
            self.assertEqual(shard.generator_worker(config, None), 'synthetic worker result')
        self.assertEqual(before, (pools._PoolRoles, marked.construct_observer, marked.generator_worker))

    def test_parent_explicit_child_module_shard_tags_and_scoped_restoration(self):
        before = parallel.marked.SCENE_MODULE, parallel.marked.marked_tags, parallel.original.shard_configuration
        def fake_main():
            self.assertEqual(parallel.marked.SCENE_MODULE, order.PRODUCER)
            self.assertEqual(parallel.marked.marked_tags()['stay_w1_order_version'], order.VERSION)
            value = parallel.original.shard_configuration()
            self.assertEqual(value['producer_module'], order.PRODUCER)
            order.validate_fields(value)
            return 0
        with patch.object(parallel.original, 'shard_configuration', lambda: {'unchanged': True}):
            with patch.object(parallel.marked, 'main', fake_main):
                self.assertEqual(parallel.main(), 0)
        self.assertEqual(before, (parallel.marked.SCENE_MODULE, parallel.marked.marked_tags, parallel.original.shard_configuration))

    def test_checkpoint_dispatch_requires_explicit_matching_new_source_and_keeps_old_dispatch(self):
        config = {**order.fields(), 'producer_module': order.PRODUCER,
                  'shard': {**order.fields(), 'producer_module': order.PRODUCER}}
        original = review.merge.shard_producer
        with patch.object(review.merge, 'marked_policy_configuration', return_value={}) as common:
            with review.source_support():
                self.assertEqual(review.merge.shard_producer(['python3', '-m', order.PRODUCER, '--name', 'fixture'], config),
                                 (['--name', 'fixture'], False))
                self.assertEqual(common.call_args.args[0]['producer_module'], 'src.envs.pair_v2_marked_shard')
                self.assertEqual(config['producer_module'], order.PRODUCER)
                invalid = copy.deepcopy(config); invalid['shard']['stay_w1_order_version'] = 'unknown'
                with self.assertRaises(ValueError):
                    review.merge.shard_producer(['python3', '-m', order.PRODUCER], invalid)
        self.assertIs(review.merge.shard_producer, original)

    def test_metrics_mismatch_and_unknown_policy_fail(self):
        config = {**order.fields(), 'producer_module': order.PRODUCER,
                  'shard': {**order.fields(), 'producer_module': order.PRODUCER}}
        metrics = {**order.fields(), 'producer_module': order.PRODUCER}
        order.validate_source(config, metrics)
        metrics['stay_w1_order_policy']['distance'] = 'altered'
        with self.assertRaises(ValueError):
            order.validate_source(config, metrics)


if __name__ == '__main__':
    unittest.main()
