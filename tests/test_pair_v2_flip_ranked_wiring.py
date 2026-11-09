"""CPU wiring fixtures for a fresh combined producer; no simulator outcomes."""
import copy
import multiprocessing as mp
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.envs import pair_v2_flip_shard as old
from src.envs import pair_v2_flip_rounds_shard as rounds
from src.envs import pair_v2_flip_rounds_policy as prior_policy
from src.envs import pair_v2_flip_rounds_search as cadence
from src.envs import pair_v2_flip_ranked_policy as policy
from src.envs import pair_v2_flip_ranked_shard as producer
from src.envs import pair_v2_spawn_sharing as sharing
from src.envs import pair_v2_bbox_rank as rank
from src.envs import pair_v2_flip_geometry as geometry
from src.envs import pair_v2_native_retry as native_retry
from src.envs import pair_v2_stay_shard as stay_shard
from src.envs import pair_v2_fixed_labels as labels
from src.envs import pair_v2_stay_order as stay
from src.envs import pair_v2_fov as fov
from src.tools import t01v2_flip_ranked_parallel as launcher
from src.tools import t01v2_flip_parallel as old_launcher
from src.tools import t01v2_flip_ranked_review_checkpoint as checker
from src.tools import t01v2_flip_review_checkpoint as old_checker


def configuration():
    fields = {**policy.camera_fields(60, 'FloorPlan203'), **stay.fields(),
        'scene': 'FloorPlan203', 'producer_module': policy.PRODUCER}
    return {**fields, 'shard': copy.deepcopy(fields), 'deadline': time.time() + 60,
            'raw_evidence_dir': '/synthetic/ranked-source'}


def pool_arguments():
    def token(x):
        point = {'x': x, 'y': .8, 'z': 2.}
        return {'position': point, 'receptacle_name': 'Table',
            'spawn': {'action': 'GetSpawnCoordinatesAboveReceptacle', 'success': True,
                      'scene': 'FloorPlan203', 'receptacle_name': 'Table',
                      'selected_position': dict(point), 'coordinates': [dict(point)],
                      'query_reference': {'file': str(x), 'sha256': 'a' * 64}}}
    left, right = [token(-1.6), token(-1.8)], [token(1.6), token(1.8)]
    a, b = ({'name': name, 'position': {'x': x, 'y': .8, 'z': 1.}} for name, x in (('A', -1.), ('B', 1.)))
    anchor = {'name': 'X', 'axisAlignedBoundingBox': {'size': {'x': .4, 'y': .4, 'z': .4}}}
    camera = old.marked.pools.camera_from_metadata({'agent': {'position': {'x': 0., 'y': 0., 'z': 0.},
        'rotation': {'x': 0., 'y': 0., 'z': 0.}, 'cameraHorizon': 0.},
        'cameraPosition': {'x': 0., 'y': 1., 'z': 0.}, 'fov': 60., 'screenWidth': 255, 'screenHeight': 255},
        {'file': 'synthetic-camera', 'sha256': '0' * 64})
    complete = tuple(left + right)
    return (geometry.NativePointGroup(complete, left), geometry.NativePointGroup(complete, right), a, b, anchor), {
        'camera': camera, 'view': {}, 'observation_cameras': [old.marked.pools.camera_after_action(camera, {'kind': kind})
            for kind in ('turn_left', 'turn_right')]}


def bindings():
    marked = old.marked
    return (rounds.policy, old.policy, marked.run_search, marked.generator_worker, marked.replay_worker,
            marked.tags, marked.compact_attempt, marked.pools.build_view_pools, marked.base.placement_groups,
            marked.measured_build, marked.construct_observer, marked.engine.validate_spawn,
            old.marked.engine.V2Simulator.spawn_points, fov.declared_fields, labels._CONTEXT.get())


def retry_fixture(directory, outcomes, record):
    # Reuse the original native action/evidence fixture, including its real
    # PlaceObjectAtPoint failure records and unchanged-object verification.
    spec = importlib.util.spec_from_file_location('ranked_native_fixture',
        Path(__file__).with_name('test_pair_v2_native_retry.py'))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    sim = fixture.Sim(directory, outcomes)
    sim.scene = 'FloorPlan203'
    for side in ('A', 'B'):
        for item in record['points']['look'][side]:
            token = item['token']['spawn']
            sim.spawn_tokens[hashlib.sha256(old.marked.engine.encoded(token)).hexdigest()] = copy.deepcopy(token)
    def construct(baseline, key, placements, require_agent=None):
        request = placements[0]
        result = native_retry.place_native_with_retries(sim, request['object_name'],
            [request['spawn'], *request.get('alternatives', [])],
            rotation={'x': 0., 'y': 0., 'z': 0.}, record_key=key)
        return {'successful_tokens_by_name': {request['object_name']: result['successful_token']},
            'retry_records': {request['object_name']: {key: value for key, value in result.items() if key != 'event'}},
            'snapshot': sim.snapshot(key), 'passed': True}
    sim.construct_world = construct
    return sim


def runtime_pool():
    args, kwargs = pool_arguments()
    with producer.installed(configuration()):
        return old.marked.pools.build_view_pools(*args, **kwargs)


def spawned_entry(connection, kind):
    """Exercise actual top-level entry functions in a fresh Python interpreter."""
    before = bindings()
    config = configuration()
    try:
        if kind.startswith('generator'):
            if kind == 'generator_deadline':
                config['deadline'] = time.time() - 1
            def native(actual, pipe):
                if (old.policy is not policy or old.marked.run_search is not cadence.run_search
                        or old.marked.engine.V2Simulator.spawn_points is not sharing.spawn_points
                        or old.marked.generator_worker is not producer.generator_worker):
                    raise AssertionError('Spawned generator did not install the combined hooks')
                args, kwargs = pool_arguments()
                record = old.marked.pools.build_view_pools(*args, **kwargs)
                rank.validate_fields(record); sharing.validate_fields(record); geometry.validate_fields(record)
                if not all(record['points']['look'][side] for side in ('A', 'B')):
                    raise AssertionError('Synthetic full Look domain unexpectedly empty')
                return {'scope': 'synthetic fresh-process wiring only'}
            with patch.object(old.look.stay, '_ORIGINAL_GENERATOR', side_effect=native):
                if kind == 'generator_deadline':
                    try:
                        producer.generator_worker(config, None)
                    except TimeoutError:
                        result = {'scope': 'synthetic fresh-process original-deadline propagation only'}
                    else:
                        raise AssertionError('Spawned ranker ignored the original expired scene deadline')
                else:
                    result = producer.generator_worker(config, None)
        else:
            names = ['A', 'B']; pair_id = 'synthetic-ranked-pair'
            config.update(labels.pair_fields(pair_id, names), candidate_names=names, enable_candidate_clone=True)
            def native(actual, pipe):
                if fov.declared_fields is not policy.declared_fields or labels._CONTEXT.get()[0] != pair_id:
                    raise AssertionError('Spawned replay lost its camera/fixed-label context')
                if actual is not config or actual['enable_candidate_clone'] is not True:
                    raise AssertionError('Wrapper changed original replay transport inputs')
                return {'scope': 'synthetic fresh-process replay wiring only'}
            with patch.object(old.look, '_ORIGINAL_REPLAY', side_effect=native):
                result = producer.replay_worker(config, None)
            sharing.validate_fields(result); rank.validate_fields(result); geometry.validate_fields(result)
            labels.validate_pair(result, pair_id, names)
        if before != bindings():
            raise AssertionError('Fresh-process hooks leaked after the entry returned')
        connection.send({'passed': True, 'scope': result['scope']})
    except BaseException as error:
        connection.send({'passed': False, 'error': f'{type(error).__name__}: {error}'})
    finally:
        connection.close()


class RankedSourceWiringTests(unittest.TestCase):
    def test_installed_grouping_keeps_points_rejected_by_historical_horizontal_roles(self):
        args, _ = pool_arguments()
        complete = list(args[0].all_native_values)
        historical = ([], [])
        with patch.object(old, '_ORIGINAL_GROUPS', return_value=historical):
            with producer.installed(configuration()):
                left, right = old.marked.base.placement_groups(complete, *args[2:], maximum_per_side=len(complete))
        self.assertEqual(list(geometry.original_tokens(left, right)), complete)
        self.assertIs(left.stay_values, historical[0]); self.assertIs(right.stay_values, historical[1])

    def test_actual_native_retry_guard_tracks_seed_swapped_worlds_and_successful_w0_fallback(self):
        record = runtime_pool()
        # This is the original seed-swapped world order, with two genuine
        # native failures followed by successful same-domain fallbacks.
        proposal = {'mode': 'flip', 'projection_pool': {'pool': 'look'},
            'w0': copy.deepcopy(record['points']['look']['B'][0]['token']),
            'w1': copy.deepcopy(record['points']['look']['A'][0]['token'])}
        original_w0, original_w1 = copy.deepcopy(proposal['w0']), copy.deepcopy(proposal['w1'])
        role = {'w0_source_side': 'B', 'w1_source_side': 'A'}
        row = {}
        with tempfile.TemporaryDirectory() as directory:
            sim = retry_fixture(directory, [False, True, False, True], record)
            initial_validator = old.marked.engine.validate_spawn
            _, construct = producer.guarded_observer(stay_shard.construct_observer,
                sim, proposal, role, record, row)
            baseline = sim.snapshot('guard_base')
            first = construct(baseline, 'guard_w0', [{'object_name': 'X', 'spawn': proposal['w0']['spawn']}])
            actual_w0 = copy.deepcopy(proposal['w0'])
            self.assertNotEqual(original_w0, actual_w0)
            construct(first['snapshot'], 'guard_w1', [{'object_name': 'X', 'spawn': proposal['w1']['spawn']}])
            self.assertNotEqual(original_w1, proposal['w1'])
            self.assertIs(old.marked.engine.validate_spawn, initial_validator)
            self.assertEqual([name for name, _ in sim.calls], ['PlaceObjectAtPoint'] * 4)
            self.assertEqual([len(row['flip_geometry_native_guards'][world]) for world in ('w0', 'w1')], [2, 2])
            for proof in row['flip_geometry_native_guards']['w0']:
                self.assertEqual(proof['opposite_reference'], geometry.native_reference(original_w1))
                self.assertEqual((proof['source_side'], proof['opposite_side']), ('B', 'A'))
            for proof in row['flip_geometry_native_guards']['w1']:
                self.assertEqual(proof['opposite_reference'], geometry.native_reference(actual_w0))
                self.assertEqual((proof['source_side'], proof['opposite_side']), ('A', 'B'))
            self.assertEqual(row['flip_geometry_final_tuple'], geometry.complete_pair(
                record, proposal['w0'], proposal['w1'], ('B', 'A')))

    def test_actual_native_retry_cannot_escape_certified_domain_and_original_registry_still_applies(self):
        for escape in ('wrong_role', 'unregistered', 'invalid_native_membership'):
            with self.subTest(escape=escape), tempfile.TemporaryDirectory() as directory:
                record = runtime_pool()
                proposal = {'mode': 'flip', 'projection_pool': {'pool': 'look'},
                    'w0': copy.deepcopy(record['points']['look']['A'][0]['token']),
                    'w1': copy.deepcopy(record['points']['look']['B'][0]['token'])}
                row, role = {}, {'w0_source_side': 'A', 'w1_source_side': 'B'}
                sim = retry_fixture(directory, [False, True], record)
                bad = copy.deepcopy(record['points']['look']['B'][1 if escape == 'wrong_role' else 0]['token']
                    if escape == 'wrong_role' else record['points']['look']['A'][1]['token'])
                if escape == 'unregistered':
                    sim.spawn_tokens.pop(hashlib.sha256(old.marked.engine.encoded(bad['spawn'])).hexdigest())
                elif escape == 'invalid_native_membership':
                    bad['spawn']['coordinates'] = []
                original_validator = old.marked.engine.validate_spawn
                _, construct = producer.guarded_observer(stay_shard.construct_observer,
                    sim, proposal, role, record, row)
                with patch.object(old.marked.pools, 'placement_alternatives', return_value=[proposal['w0'], bad]):
                    with self.assertRaises(ValueError):
                        construct(sim.snapshot('base'), 'escape_w0', [{'object_name': 'X', 'spawn': proposal['w0']['spawn']}])
                self.assertEqual(len(sim.calls), 1)
                self.assertIs(old.marked.engine.validate_spawn, original_validator)

    def test_stay_uses_identical_observer_and_flip_rejects_before_physical_build(self):
        sentinel = (object(), object())
        with patch.object(stay_shard, 'construct_observer', return_value=sentinel) as observer:
            actual = producer.guarded_observer(observer, object(), {'mode': 'stay'}, {}, {}, {})
        self.assertIs(actual, sentinel)
        record = runtime_pool()
        invalid = {'mode': 'flip', '_private_pool_record': record,
                   'w0': record['points']['look']['A'][0]['token'],
                   'w1': record['points']['look']['A'][1]['token']}
        with patch.object(old, 'measured_build', side_effect=AssertionError('physical construction must not start')) as build:
            with producer.installed(configuration()):
                with self.assertRaises(ValueError):
                    old.marked.measured_build(None, {}, {}, {}, invalid, {}, 'synthetic', None, [])
        build.assert_not_called()

    def test_parent_config_propagates_storage_fields_and_compact_keeps_pair_labels_and_cadence(self):
        complete = configuration()
        cfg = {'scene': complete['scene'], 'field_of_view': 60, 'shard': complete['shard']}
        with producer.installed():
            old.marked.enable_config(cfg)
            policy.validate_source(cfg)
            fake = SimpleNamespace(config=cfg, scene=cfg['scene'], check_deadline=lambda: None,
                controller=SimpleNamespace(last_event=SimpleNamespace(metadata={'objects': [
                    {'name': 'Surface', 'objectId': 'surface-id', 'receptacle': True,
                     'position': {'x': 0., 'y': 0., 'z': 0.}}]})),
                action=lambda *args, **kwargs: SimpleNamespace(metadata={'actionReturn': []}))
            self.assertEqual(old.marked.engine.V2Simulator.spawn_points(fake, 'Surface'), [])
            row = {**old.marked.tags(), **labels.pair_fields('synthetic-compact', ['A', 'B']),
                   'pair_id': 'synthetic-compact', 'status': 'rejected'}
            compact = old.marked.compact_attempt(row)
        labels.validate_pair(compact, 'synthetic-compact', ['A', 'B'])
        cadence.validate_fields(compact); sharing.validate_fields(compact); rank.validate_fields(compact); geometry.validate_fields(compact)

    def test_explicit_complete_source_policy_rejects_old_or_partial_sources(self):
        cfg = configuration(); metrics = {key: value for key, value in cfg.items() if key != 'shard'}
        policy.validate_source(cfg, metrics)
        old_cfg = {**prior_policy.camera_fields(60, 'FloorPlan203'), **stay.fields(),
                   'scene': 'FloorPlan203', 'producer_module': prior_policy.PRODUCER}
        with self.assertRaises(ValueError): policy.validate_source({**old_cfg, 'shard': old_cfg})
        for key in (*sharing.FIELDS, *rank.FIELDS, *cadence.FIELDS, *geometry.FIELDS):
            changed = copy.deepcopy(cfg); changed.pop(key)
            with self.assertRaises(ValueError): policy.validate_source(changed)
            changed = copy.deepcopy(metrics); changed.pop(key)
            with self.assertRaises(ValueError): policy.validate_source(cfg, changed)
        with self.assertRaises(ValueError): policy.camera_fields(100, 'FloorPlan203')
        with self.assertRaises(ValueError): policy.camera_fields(60, 'FloorPlan221')

    def test_generator_builds_original_stay_then_filters_complete_native_flip_domain(self):
        cfg, before = configuration(), bindings()
        args, kwargs = pool_arguments()
        original_full = old.search.full_pool
        built = []
        def full(*args, **kwargs):
            result = original_full(*args, **kwargs); built.append(result)
            return result
        def native(actual, connection):
            self.assertIs(actual, cfg)
            self.assertIs(old.policy, policy)
            self.assertIs(old.marked.run_search, cadence.run_search)
            self.assertIs(old.marked.engine.V2Simulator.spawn_points, sharing.spawn_points)
            result = old.marked.pools.build_view_pools(*args, **kwargs)
            source = built[0]
            for side in ('A', 'B'):
                self.assertEqual(source['points']['look'][side], [])
                self.assertTrue(result['points']['look'][side])
                expected_ids = {id(token) for token in args[0].all_native_values}
                self.assertTrue(all(id(item['token']) in expected_ids for item in result['points']['look'][side]))
                historical = original_full(old._ORIGINAL_POOLS,
                    old.search.PointGroup([], args[0].stay_values),
                    old.search.PointGroup([], args[1].stay_values), *args[2:], **kwargs)
                self.assertEqual(source['stay_points'][side], historical['stay_points'][side])
            self.assertIs(result['points']['direct'], source['points']['direct'])
            self.assertIs(result['stay_points'], source['stay_points'])
            self.assertEqual(result['flip_geometry_counts']['input_occurrences'], len(args[0].all_native_values))
            sharing.validate_fields(old.marked.tags()); rank.validate_fields(old.marked.tags())
            return result
        with patch.object(old.search, 'full_pool', side_effect=full) as full_call, \
                patch.object(old.look.stay, '_ORIGINAL_GENERATOR', side_effect=native):
            result = producer.generator_worker(cfg, None)
        self.assertEqual(full_call.call_count, 1)
        rank.validate_fields(result); sharing.validate_fields(result); geometry.validate_fields(result)
        self.assertEqual(before, bindings())

    def test_original_soft_deadline_cancels_before_partial_rank_and_restores_hooks(self):
        cfg, before = configuration(), bindings(); cfg['deadline'] = 100.
        args, kwargs = pool_arguments(); built = []
        original_full = old.search.full_pool
        def full(*args, **kwargs):
            result = original_full(*args, **kwargs); built.append(result); return result
        def native(actual, connection):
            return old.marked.pools.build_view_pools(*args, **kwargs)
        with patch.object(old.search, 'full_pool', side_effect=full), \
                patch.object(old.look.stay, '_ORIGINAL_GENERATOR', side_effect=native), \
                patch.object(producer.time, 'time', return_value=101.):
            with self.assertRaisesRegex(TimeoutError, 'Original scene soft deadline'):
                producer.generator_worker(cfg, None)
        self.assertEqual(len(built), 1)
        self.assertTrue(all('bbox_proposal_prediction' not in item
            for items in built[0]['points']['look'].values() for item in items))
        self.assertEqual(before, bindings())

    def test_generator_and_replay_native_exceptions_restore_process_local_state(self):
        cfg, before = configuration(), bindings()
        with patch.object(old.look.stay, '_ORIGINAL_GENERATOR', side_effect=RuntimeError('native fixture')):
            with self.assertRaisesRegex(RuntimeError, 'native fixture'): producer.generator_worker(cfg, None)
        self.assertEqual(before, bindings())
        cfg.update(labels.pair_fields('synthetic', ['A', 'B']), candidate_names=['A', 'B'])
        with patch.object(old.look, '_ORIGINAL_REPLAY', side_effect=RuntimeError('replay fixture')):
            with self.assertRaisesRegex(RuntimeError, 'replay fixture'): producer.replay_worker(cfg, None)
        self.assertEqual(before, bindings())
        with producer.installed():
            args, kwargs = pool_arguments()
            with self.assertRaisesRegex(RuntimeError, 'spawned generator'):
                old.marked.pools.build_view_pools(*args, **kwargs)
        self.assertEqual(before, bindings())

    def test_actual_spawned_generator_and_replay_entries_install_their_own_hooks(self):
        context = mp.get_context('spawn')
        for kind in ('generator', 'generator_deadline', 'replay'):
            with self.subTest(kind=kind):
                incoming, outgoing = context.Pipe(duplex=False)
                process = context.Process(target=spawned_entry, args=(outgoing, kind))
                process.start(); outgoing.close()
                try:
                    self.assertTrue(incoming.poll(30), 'Synthetic spawned entry timed out')
                    result = incoming.recv()
                    self.assertTrue(result['passed'], result)
                    process.join(5)
                    self.assertEqual(process.exitcode, 0)
                finally:
                    if process.is_alive():
                        process.terminate(); process.join(5)
                    incoming.close()

    def test_launcher_and_source_context_restore_without_upgrading_old_policy(self):
        before_launch, before_check, before_camera = old_launcher.policy, old_checker.policy, fov.declared_fields
        def launch():
            self.assertIs(old_launcher.policy, policy)
            raise RuntimeError('launcher fixture')
        with patch.object(old_launcher, 'main', side_effect=launch):
            with self.assertRaisesRegex(RuntimeError, 'launcher fixture'): launcher.main()
        self.assertIs(old_launcher.policy, before_launch)
        with self.assertRaisesRegex(RuntimeError, 'source fixture'):
            with checker.source_support():
                self.assertIs(old_checker.policy, policy)
                self.assertIs(fov.declared_fields, policy.declared_fields)
                raise RuntimeError('source fixture')
        self.assertIs(old_checker.policy, before_check); self.assertIs(fov.declared_fields, before_camera)


if __name__ == '__main__':
    unittest.main()
