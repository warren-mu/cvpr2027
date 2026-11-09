"""CPU interface/lifecycle fixtures only; no renderer, tests run by run_logged."""
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from src.envs import pair_v2_marked_shard as shard
from src.envs import pair_v2_projection_pool as pools
from src.envs import pair_v2_role_policy as roles
from src.envs.pair_v2_native_retry import NativePlacementExhausted


def record():
    metadata = {'agent': {'position': {'x': 0., 'y': 0., 'z': 0.},
        'rotation': {'x': 0., 'y': 0., 'z': 0.}, 'cameraHorizon': 0.},
        'cameraPosition': {'x': 0., 'y': 1., 'z': 0.}, 'fov': 90., 'screenWidth': 100, 'screenHeight': 100}
    camera = pools.camera_from_metadata(metadata, {'file': 'synthetic', 'sha256': 'a' * 64})
    def token(x):
        position = {'x': x, 'y': .5, 'z': 1.}
        return {'position': position, 'receptacle_name': 'Surface', 'spawn': {'selected_position': dict(position)}}
    return pools.build_view_pools([token(-1.3), token(-1.4), token(-.5), token(-.6)],
        [token(1.3), token(1.4), token(.5), token(.6)],
        {'position': {'x': -1., 'y': .5, 'z': 1.}}, {'position': {'x': 1., 'y': .5, 'z': 1.}},
        {'name': 'X', 'axisAlignedBoundingBox': {'size': {'y': 1.}}}, camera=camera,
        view={'position': camera['agent_position'], 'yaw': 0, 'horizon': 0},
        observation_cameras=[pools.camera_after_action(camera, {'kind': name}) for name in ('turn_left', 'turn_right')])


class Store:
    def __init__(self):
        self.saved = []
    def save_json(self, name, value):
        if name in [row[0] for row in self.saved]:
            raise FileExistsError(name)
        self.saved.append((name, copy.deepcopy(value)))
        return {'file': '/synthetic/' + name, 'sha256': 'a' * 64}


def state():
    return {'rows': [], 'excluded': None, 'status': 'completed_search', 'w0_by_anchor': {},
        'controller_visits': [], 'selected_controller_id': 'fixture', 'search_stop_reason': None,
        'cleanup_errors': [], 'pool_catalogs': [], 'placement_exhausted_proposals': 0,
        'placement_point_failures': [], 'controller_restarts': 0, 'discovery_failures': [],
        'proposal_trials': 0, 'failed_native_point_count': 0}


class MarkedShardTests(unittest.TestCase):
    def test_config_explicit_natural_source_and_no_old_search_or_coverage_claim(self):
        config = shard.enable_config({'max_attempts': 10})
        self.assertIs(config['enable_candidate_clone'], False)
        self.assertIsNone(config['search_version'])
        self.assertIsNone(config['search_policy'])
        self.assertNotIn('proposal_order_version', config)
        self.assertNotIn('clone_rule', config)
        self.assertEqual(config['candidate_policy'], 'two_marked_pickupables_v1')
        self.assertEqual(config['max_attempts'], 100)
        self.assertEqual(config['safe_retry_version'], shard.NATIVE_RETRY_VERSION)

    def test_canonical_initial_setup_executes_original_command_and_keeps_actual_response(self):
        initial = {'position': {'x': 1., 'y': .90000006, 'z': 2.},
                   'rotation': {'x': -0., 'y': 45., 'z': 0.}, 'cameraHorizon': -0.}
        sim = SimpleNamespace(controller=SimpleNamespace(last_event=SimpleNamespace(metadata={'agent': initial})),
                              store=Store(), last_action_reference={'file': 'action0', 'sha256': 'b' * 64})
        calls = []
        def position_agent(position, yaw, horizon):
            calls.append((copy.deepcopy(position), yaw, horizon))
            sim.agent_setup_command = {'position': position, 'rotation': {'x': 0, 'y': yaw, 'z': 0},
                                       'horizon': horizon, 'standing': True, 'forceAction': False}
            sim.agent_setup_actual = {'position': {**position, 'y': .900000119}, 'rotation': {'x': 0., 'y': yaw, 'z': 0.},
                                      'horizon': 0., 'standing': True}
        sim.position_agent = position_agent
        source = copy.deepcopy(initial)
        ref = shard.canonical_initial_agent_setup(sim)
        self.assertEqual(calls, [(initial['position'], 45., -0.)])
        self.assertEqual(initial, source)
        self.assertEqual(sim.canonical_initial_setup_reference, ref)
        stored = sim.store.saved[0][1]
        self.assertEqual(stored['actual_command'], sim.agent_setup_command)
        self.assertEqual(stored['actual_agent'], sim.agent_setup_actual)
        self.assertNotEqual(stored['actual_command']['position']['y'], stored['actual_agent']['position']['y'])

    def test_fresh_replay_installs_marked_semantics_and_restores_on_exception(self):
        previous = shard.original_policy.closest_target
        config = {'candidate_policy': shard.candidates.CANDIDATE_POLICY, 'snapshot': {}, 'enable_candidate_clone': True}
        def replay(actual, connection):
            self.assertIs(shard.original_policy.closest_target, shard.candidates.closest_target)
            self.assertIs(actual['enable_candidate_clone'], False)
            raise RuntimeError('synthetic replay failure')
        with patch.object(shard.search, 'replay_worker', side_effect=replay):
            with self.assertRaisesRegex(RuntimeError, 'synthetic replay'):
                shard.replay_worker(config, None)
        self.assertIs(shard.original_policy.closest_target, previous)
        self.assertIs(config['enable_candidate_clone'], True)

    def test_retry_actual_token_replaces_proposal_and_tail_has_no_duplicate_first(self):
        rec = record()
        proposal, _, _ = next(pools.PoolProposalIterator([rec]))
        proposal, role = roles.orient_proposal(proposal, 'FloorPlan1', 0)
        row = {}; seen = []
        def construct(baseline, key, placements, require_agent=None):
            request = placements[0]; seen.append(copy.deepcopy(request))
            self.assertTrue(request['alternatives'])
            self.assertNotEqual(request['spawn'], request['alternatives'][0])
            chosen = request['alternatives'][0]
            return {'successful_tokens_by_name': {'X': chosen}, 'retry_records': {'X': {'failed_point_count': 1}},
                    'snapshot': {'objects': [{'name': 'X', 'objectType': 'Vase', 'position': chosen['selected_position']}]}}
        sim = SimpleNamespace(construct_world=construct)
        _, wrapped = shard.construct_observer(sim, proposal, role, rec, row)
        wrapped({}, 'pair_w0', [{'object_name': 'X', 'spawn': proposal['w0']['spawn']}])
        self.assertEqual(row['proposal']['w0']['spawn'], seen[0]['alternatives'][0])
        self.assertEqual(row['native_retry_records']['w0']['X']['failed_point_count'], 1)

    def test_stay_retry_cannot_reuse_actual_w0_native_point(self):
        rec = record()
        proposal = next(p for p, _, _ in pools.PoolProposalIterator([rec]) if p['mode'] == 'stay')
        proposal, role = roles.orient_proposal(proposal, 'FloorPlan1', 0)
        intended_w1 = copy.deepcopy(proposal['w1']['spawn'])
        requests = []
        def construct(baseline, key, placements, require_agent=None):
            requests.append(copy.deepcopy(placements[0]))
            chosen = intended_w1 if key.endswith('_w0') else placements[0]['spawn']
            return {'successful_tokens_by_name': {'X': chosen}, 'retry_records': {},
                'snapshot': {'objects': [{'name': 'X', 'objectType': 'Vase', 'position': chosen['selected_position']}]}}
        _, wrapped = shard.construct_observer(SimpleNamespace(construct_world=construct), proposal, role, rec, {})
        wrapped({}, 'pair_w0', [{'object_name': 'X', 'spawn': proposal['w0']['spawn']}])
        wrapped({}, 'pair_w1', [{'object_name': 'X', 'spawn': proposal['w1']['spawn']}])
        self.assertNotEqual(requests[1]['spawn']['selected_position'], intended_w1['selected_position'])
        self.assertEqual(requests[1]['alternatives'], [])

    def test_prefix_includes_failed_points_and_ends_at_first_actual_success(self):
        sim = SimpleNamespace(store=Store(), query_log=[], action_count=0)
        def action(name, **kwargs):
            index = sim.action_count; sim.action_count += 1
            ref = {'file': str(index), 'sha256': 'b' * 64}
            sim.query_log.append({'action': name, 'reference': ref})
            if index == 0:
                raise RuntimeError('native recorded false')
            return SimpleNamespace(metadata={'lastActionSuccess': True})
        sim.action = action
        row = {'pair_id': 'floorplan1_00000', 'scene': 'FloorPlan1',
               'attempt_setup_reference': {'file': 'setup', 'sha256': 'c' * 64}}
        def original(*args, attempt_row):
            try:
                sim.action('PlaceObjectAtPoint')
            except RuntimeError:
                pass
            self.assertNotIn('attempt_setup_native_prefix_reference', row)
            sim.action('PlaceObjectAtPoint')
            sim.action('PlaceObjectAtPoint')
            return row
        shard.prefix_observer(original, sim, row, 0, ())
        self.assertIs(sim.action, action)
        self.assertEqual(len(sim.store.saved), 1)
        proof = sim.store.saved[0][1]
        self.assertEqual([ref['reference']['file'] for ref in proof['action_references']], ['0', '1'])
        self.assertEqual(proof['first_place_reference']['file'], '1')
        self.assertEqual(proof['safe_retry_version'], shard.NATIVE_RETRY_VERSION)

    def test_uncounted_trials_have_unique_setup_filenames_and_restore_private_methods(self):
        rec = record(); rec['reference'] = {'file': 'pool', 'sha256': 'd' * 64}
        proposal, view, _ = next(pools.PoolProposalIterator([rec])); proposal['_private_pool_record'] = rec
        sim = SimpleNamespace(store=Store(), construct_world=lambda *a, **kw: None)
        old_construct, old_observer = sim.construct_world, shard.isolated.build_with_prefix_observer
        identity = {'candidate_names': ['A', 'B'], 'candidate_types': ['Mug', 'Book'], 'same_class': False}
        def isolated_build(original, actual, config, baseline, identities, prop, camera_view, pair_id, directory, reachable, attempt_row):
            actual.store.save_json('attempt_setup_' + pair_id + '.json', {'pair_id': pair_id})
            raise NativePlacementExhausted({'status': 'exhausted', 'failed_point_count': 40})
        with patch.object(shard.isolated, 'isolated_build', side_effect=isolated_build):
            for trial in (0, 1):
                with self.assertRaises(NativePlacementExhausted):
                    shard.measured_build(sim, {'scene': 'FloorPlan1'}, {}, identity, proposal, view,
                        'floorplan1_00000', Path(f'floorplan1_00000_trial{trial:06d}'), [], attempt_row={})
        self.assertEqual(len({name for name, _ in sim.store.saved}), 2)
        self.assertIs(sim.construct_world, old_construct)
        self.assertIs(shard.isolated.build_with_prefix_observer, old_observer)

    def test_exhausted_native_points_do_not_consume_pair_attempt_or_repeat_forever(self):
        rec = record(); rec['reference'] = {'file': 'pool', 'sha256': 'd' * 64}
        identity = {'candidate_names': ['A', 'B'], 'anchor_name': 'X', 'anchor_type': 'Vase'}
        proposal, view, _ = next(pools.PoolProposalIterator([rec]))
        class Single:
            next_pool = 'look'
            def __init__(self, ignored): self.done = False
            def summary(self):
                return {'pools': {'look': {'all_exhausted': self.done}, 'direct': {'all_exhausted': True}}}
            def __next__(self):
                if self.done: raise StopIteration
                self.done = True; return copy.deepcopy(proposal), view, {}
        sim = SimpleNamespace(check_deadline=lambda: None, reachable_positions=lambda: {'positions': []})
        current = state(); channel = SimpleNamespace(send=lambda value: None)
        with tempfile.TemporaryDirectory() as folder:
            config = {'scene': 'FloorPlan1', 'raw_evidence_dir': folder, 'max_attempts': 160,
                      'scene_pairs_per_class': 5, 'attempt_journal': 'synthetic'}
            with patch.object(shard, 'existing_views', return_value=[view]), \
                 patch.object(shard, 'discover_view', return_value=iter([(identity, rec)])), \
                 patch.object(shard.pools, 'PoolProposalIterator', Single), \
                 patch.object(shard.base, 'build_pair', side_effect=NativePlacementExhausted({'failed_point_count': 40})), \
                 patch.object(shard.base, 'write_json', return_value={'file': 'raw', 'sha256': 'e' * 64}), \
                 patch.object(shard, 'summary', return_value={}):
                journal = io.StringIO()
                shard.run_search(sim, config, {}, current, journal, channel, set())
            self.assertEqual(current['rows'], [])
            self.assertEqual(journal.getvalue(), '')
            self.assertEqual(current['failed_native_point_count'], 40)
            self.assertEqual(current['proposal_trials'], 1)

    def test_parent_private_bindings_restore_and_missing_receipt_cannot_succeed(self):
        old = shard.base.generator_worker, shard.base.Worker, shard.base.write_result
        config = {'max_attempts': 160}
        instance = object()
        def base_main():
            shard.base.Worker(shard.base.generator_worker, config)
            self.assertFalse(config['enable_candidate_clone'])
            shard.base.write_result('fixture', {}, config)
            return 0
        with patch.object(shard.base, 'main', side_effect=base_main), \
             patch.object(shard.terminal, 'TerminalWorker', return_value=instance), \
             patch.object(shard.terminal, 'final_metrics', return_value=({'status': 'terminal_receipt_invalid'}, False)), \
             patch.object(shard.base, 'write_result', return_value=None):
            writer = shard.base.write_result
            self.assertEqual(shard.main(), 1)
            self.assertIs(shard.base.write_result, writer)
        self.assertIs(shard.base.generator_worker, old[0])
        self.assertIs(shard.base.Worker, old[1])
        self.assertIs(shard.base.write_result, old[2])


if __name__ == '__main__':
    unittest.main()
