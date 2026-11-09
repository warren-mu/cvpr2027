"""CPU wiring fixtures for a fresh combined producer; no simulator outcomes."""
import copy
import multiprocessing as mp
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
        return {'position': point, 'receptacle_name': 'Surface',
            'spawn': {'selected_position': dict(point), 'coordinates': [dict(point)],
                      'query_reference': {'file': str(x), 'sha256': 'a' * 64}}}
    left, right = [token(-1.6), token(-2.1)], [token(1.6), token(2.1)]
    a, b = ({'position': {'x': x, 'y': .8, 'z': 1.}} for x in (-1., 1.))
    anchor = {'name': 'X', 'axisAlignedBoundingBox': {'size': {'x': .4, 'y': .4, 'z': .4}}}
    camera = old.marked.pools.camera_from_metadata({'agent': {'position': {'x': 0., 'y': 0., 'z': 0.},
        'rotation': {'x': 0., 'y': 0., 'z': 0.}, 'cameraHorizon': 0.},
        'cameraPosition': {'x': 0., 'y': 1., 'z': 0.}, 'fov': 60., 'screenWidth': 255, 'screenHeight': 255},
        {'file': 'synthetic-camera', 'sha256': '0' * 64})
    return (old.search.PointGroup(left, left), old.search.PointGroup(right, right), a, b, anchor), {
        'camera': camera, 'view': {}, 'observation_cameras': [old.marked.pools.camera_after_action(camera, {'kind': kind})
            for kind in ('turn_left', 'turn_right')]}


def bindings():
    marked = old.marked
    return (rounds.policy, old.policy, marked.run_search, marked.generator_worker, marked.replay_worker,
            marked.tags, marked.compact_attempt, marked.pools.build_view_pools,
            old.marked.engine.V2Simulator.spawn_points, fov.declared_fields, labels._CONTEXT.get())


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
                rank.validate_fields(record); sharing.validate_fields(record)
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
            sharing.validate_fields(result); rank.validate_fields(result)
            labels.validate_pair(result, pair_id, names)
        if before != bindings():
            raise AssertionError('Fresh-process hooks leaked after the entry returned')
        connection.send({'passed': True, 'scope': result['scope']})
    except BaseException as error:
        connection.send({'passed': False, 'error': f'{type(error).__name__}: {error}'})
    finally:
        connection.close()


class RankedSourceWiringTests(unittest.TestCase):
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
        cadence.validate_fields(compact); sharing.validate_fields(compact); rank.validate_fields(compact)

    def test_explicit_complete_source_policy_rejects_old_or_partial_sources(self):
        cfg = configuration(); metrics = {key: value for key, value in cfg.items() if key != 'shard'}
        policy.validate_source(cfg, metrics)
        old_cfg = {**prior_policy.camera_fields(60, 'FloorPlan203'), **stay.fields(),
                   'scene': 'FloorPlan203', 'producer_module': prior_policy.PRODUCER}
        with self.assertRaises(ValueError): policy.validate_source({**old_cfg, 'shard': old_cfg})
        for key in (*sharing.FIELDS, *rank.FIELDS, *cadence.FIELDS):
            changed = copy.deepcopy(cfg); changed.pop(key)
            with self.assertRaises(ValueError): policy.validate_source(changed)
            changed = copy.deepcopy(metrics); changed.pop(key)
            with self.assertRaises(ValueError): policy.validate_source(cfg, changed)
        with self.assertRaises(ValueError): policy.camera_fields(100, 'FloorPlan203')
        with self.assertRaises(ValueError): policy.camera_fields(60, 'FloorPlan221')

    def test_generator_builds_prior_full_domain_then_ranks_only_look(self):
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
                self.assertTrue(source['points']['look'][side])
                self.assertEqual(sorted(id(item['token']) for item in result['points']['look'][side]),
                                 sorted(id(item['token']) for item in source['points']['look'][side]))
                self.assertTrue(all('bbox_proposal_prediction' not in item for item in source['points']['look'][side]))
            self.assertIs(result['points']['direct'], source['points']['direct'])
            self.assertIs(result['stay_points'], source['stay_points'])
            self.assertIs(result['counts'], source['counts'])
            sharing.validate_fields(old.marked.tags()); rank.validate_fields(old.marked.tags())
            return result
        with patch.object(old.search, 'full_pool', side_effect=full) as full_call, \
                patch.object(old.look.stay, '_ORIGINAL_GENERATOR', side_effect=native):
            result = producer.generator_worker(cfg, None)
        self.assertEqual(full_call.call_count, 1)
        rank.validate_fields(result); sharing.validate_fields(result)
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
