"""Synthetic CPU fixtures for explicit G01 mixed production and source admission."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from src.envs import pair_v2_g01_policy as policy
from src.envs import pair_v2_g01_search as search
from src.envs import pair_v2_g01_shard as shard
from src.envs import pair_v2_flip_search as flip
from src.envs import pair_v2_fixed_labels as labels
from src.envs import pair_v2_projection_pool as pools
from src.envs import pair_v2_role_policy as roles
from src.envs import pair_v2_stay_order as stay
from src.envs import pair_v2_flip_geometry as geometry
from src.envs import pair_v2_flip_rounds_search as cadence
from src.envs import pair_v2_spawn_sharing as sharing
from src.envs import pair_v2_bbox_rank as rank
from src.tools import t01v2_g01_association as association
from src.tools import t01v2_g01_review_checkpoint as review
_spec = importlib.util.spec_from_file_location('g01_controller_fixtures',
    Path(__file__).with_name('test_t01v2_controller_association.py'))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
Fixture, original_preflight_fixture = _fixtures.Fixture, _fixtures.original_preflight_fixture


def fixture_module(filename):
    spec = importlib.util.spec_from_file_location('g01_' + filename,
        Path(__file__).with_name(filename + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(scene='FloorPlan203'):
    declared = {**policy.declared(scene), **stay.fields(), 'producer_module': policy.PRODUCER}
    return {**declared, 'scene': scene, 'raw_evidence_dir': '/synthetic/raw', 'deadline': time.time() + 60,
        'enable_candidate_clone': False, 'scene_pairs_per_class': 8,
        'shard': {**declared, 'scene': scene}}


def token(x, y=0., z=1.):
    point = {'x': x, 'y': y, 'z': z}
    return {'position': point, 'receptacle_name': 'Surface', 'spawn': {
        'selected_position': dict(point), 'receptacle_name': 'Surface',
        'coordinates': [dict(point)], 'query_reference': {'file': str((x, y, z)), 'sha256': 'a' * 64}}}


def record():
    sides = {'A': [{'token': token(-2. - index / 10)} for index in range(3)],
             'B': [{'token': token(2. + index / 10)} for index in range(3)]}
    return {**search.fields(), 'version': pools.POOL_VERSION, 'view': {}, 'counts': {},
        'points': {pool: copy.deepcopy(sides) for pool in ('look', 'direct')},
        'stay_points_by_pool': {pool: copy.deepcopy(sides) for pool in ('look', 'direct')}}


def counted(pool, mode, kind=None, passed=False):
    return {**search.fields(), 'counts_as_pair_attempt': True, 'mode': mode, 'proposed_pool': pool,
        'kind': kind, 'status': 'passed' if passed else 'rejected', 'discard_reasons': []}


class MixedProductionTests(unittest.TestCase):
    def test_only_fixed_training_partition_and_no_clone(self):
        for start in (1, 201, 301, 401):
            for number in range(start, start + 20):
                cfg = config(f'FloorPlan{number}')
                policy.validate_source(cfg, cfg)
            for number in range(start + 20, start + 30):
                with self.assertRaises(ValueError): policy.declared(f'FloorPlan{number}')
        for key, value in (('field_of_view', 100), ('enable_candidate_clone', True), ('scene_pairs_per_class', 9)):
            cfg = config(); cfg[key] = value
            with self.assertRaises(ValueError): policy.validate_source(cfg)
        cfg = config(); cfg['shard']['scene'] = 'FloorPlan204'
        with self.assertRaises(ValueError): policy.validate_source(cfg)
        cfg = config(); cfg['flip_search_version'] = flip.VERSION
        with self.assertRaises(ValueError): policy.validate_source(cfg)
        for key in (*geometry.FIELDS, *sharing.FIELDS, *cadence.FIELDS, *rank.FIELDS):
            cfg = config(); cfg.pop(key)
            with self.assertRaises(ValueError): policy.validate_source(cfg)

    def test_look_keeps_same_side_native_point_that_surface_role_would_drop(self):
        fixture = fixture_module('test_pair_v2_flip_geometry')
        source, a, b, anchor = fixture.setup()
        points = (fixture.token(), fixture.token(y=1.75))
        self.assertIsNone(flip.side_for_point(points[1]['position'], [a['position'], b['position']]))
        args = [geometry.NativePointGroup(points, []), geometry.NativePointGroup(points, []), a, b, anchor]
        record = search.build_pool(shard.flip._ORIGINAL_POOLS, *args,
            camera=source['camera'], view=source['view'], observation_cameras=source['observation_cameras'],
            check_deadline=lambda: None)
        self.assertIs(record['points']['look']['B'][0]['token'], points[1])
        self.assertTrue(geometry.complete_pair(record, points[0], points[1])['proposal_tuple_passed'])
        self.assertEqual(record['points']['direct'], {'A': [], 'B': []})
        self.assertEqual(record['stay_points_by_pool'],
                         {'look': {'A': [], 'B': []}, 'direct': {'A': [], 'B': []}})
        self.assertEqual(record['flip_geometry_counts']['input_occurrences'], 2)

    def test_original_soft_deadline_prevents_partial_g01_pool(self):
        fixture = fixture_module('test_pair_v2_flip_ranked_wiring')
        args, kwargs = fixture.pool_arguments()
        before = copy.deepcopy(args)
        def expired():
            raise TimeoutError('original scene deadline')
        with self.assertRaisesRegex(TimeoutError, 'original scene deadline'):
            search.build_pool(shard.flip._ORIGINAL_POOLS, *args, check_deadline=expired, **kwargs)
        self.assertEqual(args, before)

    def test_complete_flip_domains_with_original_stay_caps_and_native_inputs(self):
        a, b = {'position': {'x': -1., 'y': 0., 'z': 1.}}, {'position': {'x': 1., 'y': 0., 'z': 1.}}
        anchor = {'name': 'X', 'axisAlignedBoundingBox': {'size': {'x': .02, 'y': .2, 'z': .02}}}
        camera = pools.camera_from_metadata({'agent': {'position': {'x': 0, 'y': 0, 'z': 0},
            'rotation': {'x': 0, 'y': 0, 'z': 0}, 'cameraHorizon': 0},
            'cameraPosition': {'x': 0, 'y': 0, 'z': 0}, 'fov': 60, 'screenWidth': 255, 'screenHeight': 255},
            {'file': 'synthetic-native.json'})
        left = [token(-2. - i / 100) for i in range(20)] + [token(-.4 - i / 1000) for i in range(45)]
        right = [token(2. + i / 100) for i in range(20)] + [token(.4 + i / 1000) for i in range(45)]
        before = copy.deepcopy((left, right))
        complete = tuple(left + right)
        value = search.build_pool(shard.flip._ORIGINAL_POOLS,
            geometry.NativePointGroup(complete, left), geometry.NativePointGroup(complete, right),
            a, b, anchor, check_deadline=lambda: None, camera=camera, view={}, observation_cameras=[pools.camera_after_action(camera, {'kind': kind})
                for kind in ('turn_left', 'turn_right')])
        self.assertEqual((left, right), before)
        self.assertEqual(len(value['points']['look']['A']), 20)
        self.assertEqual(len(value['points']['direct']['A']), 45)
        self.assertEqual(len(value['stay_points_by_pool']['look']['A']), 12)
        self.assertEqual(len(value['stay_points_by_pool']['direct']['A']), 40)
        self.assertNotIn('flip_search_version', value)
        geometry.validate_fields(value); rank.validate_fields(value); sharing.validate_fields(value)
        compact = search.compact_pool(shard.flip._ORIGINAL_POOL_RECORD, value)
        self.assertEqual(compact['g01_search_policy'], search.POLICY)
        for pool in ('look', 'direct'):
            self.assertIsNone(value['counts']['A'][pool]['cap'])
            self.assertIn('spawn_reference', compact['points'][pool]['A'][0]['token'])
            self.assertNotIn('spawn', compact['stay_points_by_pool'][pool]['A'][0]['token'])

    def test_mixed_stream_alternates_pools_and_preserves_stay_sides(self):
        quota = search.SceneQuota(); stream = search.MixedIterator([record()], quota)
        sequence = []
        for index in range(8):
            proposal, _, domain = next(stream)
            pool = proposal['projection_pool']['pool']
            sequence.append((pool, proposal['mode'], domain['stay_side']))
            quota.record(str(index), counted(pool, proposal['mode']))
        self.assertEqual(sequence, [('look', 'flip', None), ('direct', 'flip', None),
            ('look', 'stay', 'A'), ('direct', 'stay', 'A'), ('look', 'flip', None),
            ('direct', 'flip', None), ('look', 'stay', 'B'), ('direct', 'stay', 'B')])

    def test_actual_quota_survives_controller_restart_and_native_failure(self):
        quota = search.SceneQuota()
        first, second = search.MixedIterator([record()], quota), search.MixedIterator([record()], quota)
        self.assertEqual(next(first)[0]['mode'], 'flip')
        quota.record('/native-fail', {**counted('look', 'flip'), 'counts_as_pair_attempt': False})
        self.assertEqual(next(second)[0]['mode'], 'flip')
        quota.record('/counted-flip', counted('look', 'flip'))
        first.next_pool = 'look'
        self.assertEqual(next(first)[0]['mode'], 'stay')
        quota.record('/counted-stay', counted('look', 'stay'))
        quota.record('/counted-stay', counted('look', 'stay'))
        with self.assertRaises(ValueError): quota.record('/unearned-stay', counted('look', 'stay'))
        self.assertEqual(quota.attempts['look_stay'], 1)

    def test_full_three_dimensional_role_keeps_seed0_and_independent_geometry_check(self):
        positions = {'A': {'x': 0., 'y': 0., 'z': 1.}, 'B': {'x': 0., 'y': 2., 'z': 1.}}
        stream = roles.RoleProposalIterator([token(0., .1)], [token(0., 1.9)], [{}])
        proposal, _, domain = next(stream); proposal['role_domain'] = domain
        proposal, assignment = roles.orient_proposal(proposal, 'FloorPlan203', 0)
        assignment.update(candidate_a_name='A', candidate_b_name='B')
        row = {**search.fields(), 'pair_id': 'floorplan203_00000', 'scene': 'FloorPlan203', 'mode': 'flip', 'proposed_pool': 'direct',
            'identities': {'candidate_names': ['A', 'B']}, 'proposal': proposal, 'proposal_role': assignment,
            'generation_version': roles.ROLE_VERSION, 'generation_policy': copy.deepcopy(roles.ROLE_POLICY),
            'target_w0': assignment['w0_source_side'], 'target_w1': assignment['w1_source_side']}
        self.assertTrue(search.verify_flip_role(row, positions)['passed'])
        with self.assertRaisesRegex(ValueError, 'independent complete-tuple'):
            search.verify_flip_role({**row, 'proposed_pool': 'look'}, positions)
        with self.assertRaises(ValueError): roles.verify_proposal_role(row, positions)
        row['proposal_role']['assignment_digest'] = 'tampered'
        with self.assertRaises(ValueError): search.verify_flip_role(row, positions)

    def test_one_sided_look_never_backfills_unearned_stay_but_direct_continues(self):
        value = record(); value['points']['look']['B'] = []
        proposal, _, _ = next(search.MixedIterator([value], search.SceneQuota()))
        self.assertEqual(proposal['projection_pool']['pool'], 'direct')
        self.assertTrue(proposal['projection_pool']['fallback_due_to_exhaustion'])
        self.assertEqual(proposal['mode'], 'flip')

    def test_per_actual_class_cap_allows_balance_attempts_but_not_more_exports(self):
        quota = search.SceneQuota()
        for index in range(8):
            quota.record(f'/flip{index}', counted('look', 'flip', 'look_flip', True))
        with self.assertRaises(search.CollectionQuotaFull): quota.before_export({'kind': 'look_flip'})
        for index in range(8): quota.record(f'/stay-rejection{index}', counted('look', 'stay'))
        self.assertEqual(quota.desired_modes('look', 'stay'), ['flip'])
        quota.record('/capacity-skipped-flip', counted('look', 'flip', 'look_flip'))
        self.assertEqual(quota.desired_modes('look', 'flip'), ['stay'])
        self.assertEqual(quota.passed['look_flip'], 8)
        quota.passed = dict.fromkeys(search.KINDS, 8)
        with self.assertRaises(StopIteration): next(search.MixedIterator([record()], quota))

    def test_proposal_pool_and_actual_output_class_remain_separate(self):
        quota = search.SceneQuota()
        quota.record('/direct-produced-look', counted('direct', 'flip', 'look_flip', True))
        self.assertEqual(quota.passed['look_flip'], 1)
        self.assertEqual(quota.attempts['look_flip'], 0)
        with self.assertRaises(ValueError): quota.record('/stay-unearned', counted('look', 'stay'))
        quota.record('/look-produced-direct', counted('look', 'flip', 'direct_flip', True))
        quota.record('/stay-earned', counted('look', 'stay'))
        self.assertEqual(quota.passed['direct_flip'], 1)
        self.assertEqual(quota.attempts['look_flip'], quota.attempts['look_stay'])

    def test_look_retry_guard_binds_real_failed_native_fallbacks_and_direct_stays_original(self):
        fixture = fixture_module('test_pair_v2_flip_ranked_wiring')
        args, kwargs = fixture.pool_arguments()
        record = search.build_pool(shard.flip._ORIGINAL_POOLS, *args, check_deadline=lambda: None, **kwargs)
        proposal = {'mode': 'flip', 'projection_pool': {'pool': 'look'},
            'w0': copy.deepcopy(record['points']['look']['B'][0]['token']),
            'w1': copy.deepcopy(record['points']['look']['A'][0]['token'])}
        role = {'w0_source_side': 'B', 'w1_source_side': 'A'}
        row = {}
        with tempfile.TemporaryDirectory() as directory:
            sim = fixture.retry_fixture(directory, [False, True, False, True], record)
            previous = shard.marked.engine.validate_spawn
            with shard.installed(config()):
                _, construct = shard.marked.construct_observer(sim, proposal, role, record, row)
                first = construct(sim.snapshot('g01_base'), 'g01_w0', [{'object_name': 'X', 'spawn': proposal['w0']['spawn']}])
                actual_w0 = copy.deepcopy(proposal['w0'])
                construct(first['snapshot'], 'g01_w1', [{'object_name': 'X', 'spawn': proposal['w1']['spawn']}])
            self.assertIs(shard.marked.engine.validate_spawn, previous)
            self.assertEqual([action for action, _ in sim.calls], ['PlaceObjectAtPoint'] * 4)
            self.assertEqual([len(row['flip_geometry_native_guards'][w]) for w in ('w0', 'w1')], [2, 2])
            for guard in row['flip_geometry_native_guards']['w1']:
                self.assertEqual(guard['opposite_reference'], geometry.native_reference(actual_w0))
        for mode, pool in (('flip', 'direct'), ('stay', 'look'), ('stay', 'direct')):
            sentinel = (object(), object())
            with patch.object(shard.stay, 'construct_observer', return_value=sentinel) as original:
                with shard.installed(config()), patch.object(geometry, 'complete_pair', side_effect=AssertionError('wrong branch')):
                    value = shard.marked.construct_observer(None,
                        {'mode': mode, 'projection_pool': {'pool': pool}}, {}, {}, {})
            self.assertIs(value, sentinel)
            original.assert_called_once()

    def test_look_precheck_happens_before_build_and_does_not_override_measured_class(self):
        fixture = fixture_module('test_pair_v2_flip_ranked_wiring')
        args, kwargs = fixture.pool_arguments()
        record = search.build_pool(shard.flip._ORIGINAL_POOLS, *args, check_deadline=lambda: None, **kwargs)
        proposal = {'mode': 'flip', 'projection_pool': {'pool': 'look'}, '_private_pool_record': record,
            'w0': record['points']['look']['A'][0]['token'], 'w1': record['points']['look']['B'][0]['token']}
        def original(*args, **kwargs):
            row = kwargs['attempt_row']; row.update(status='passed', kind='direct_flip')
            return row
        with patch.object(shard, '_ORIGINAL_BUILD', side_effect=original) as build:
            row = shard.measured_build(None, config(), {}, {'candidate_names': ['A', 'B']}, proposal,
                {}, 'floorplan203_00000', '/synthetic', [], {})
            self.assertEqual((row['proposed_pool'], row['kind']), ('look', 'direct_flip'))
            build.reset_mock()
            invalid = {**proposal, 'w1': record['points']['look']['A'][1]['token']}
            with self.assertRaises(ValueError):
                shard.measured_build(None, config(), {}, {'candidate_names': ['A', 'B']}, invalid,
                    {}, 'floorplan203_00001', '/synthetic', [], {})
            build.assert_not_called()

    def test_fixed_label_context_keeps_direct_branch_and_new_spawn_replay(self):
        cfg, identity = config(), {'candidate_names': ['A', 'B']}
        def original(sim, pair_cfg, baseline, identities, proposal, view, pair_id, directory, reachable, attempt_row):
            labels.validate_pair(pair_cfg, pair_id, ['A', 'B'])
            self.assertEqual(labels._CONTEXT.get()[0], pair_id)
            self.assertIs(shard.look.original_policy.prefix_check, prefix)
            attempt_row.update(kind='direct_flip', status='passed')
            return attempt_row
        prefix = shard.look.original_policy.prefix_check
        with shard.installed(), patch.object(shard, '_ORIGINAL_BUILD', side_effect=original):
            row = shard.measured_build(None, cfg, {}, identity, {'mode': 'flip', 'projection_pool': {'pool': 'direct'}},
                {}, 'floorplan203_00000', '/raw', [], {})
        self.assertEqual(row['kind'], 'direct_flip')
        child = {**cfg, **labels.pair_fields('floorplan203_00000', ['A', 'B']), **identity,
            'candidate_policy': shard.marked.candidates.CANDIDATE_POLICY,
            'snapshot': {'clone_context': None}, 'enable_candidate_clone': True}
        def replay(child_cfg, connection):
            self.assertEqual(labels._CONTEXT.get()[0], 'floorplan203_00000')
            policy.validate_source(child_cfg)
            return {'status': 'observed'}
        with patch.object(shard.look, '_ORIGINAL_REPLAY', side_effect=replay):
            outcome = shard.replay_worker(child, None)
            with self.assertRaisesRegex(ValueError, 'natural candidate snapshot'):
                shard.replay_worker({**child, 'snapshot': {'clone_context': {'unexpected': True}}}, None)
        labels.validate_pair(outcome, 'floorplan203_00000', ['A', 'B'])
        self.assertIs(child['enable_candidate_clone'], True)
        self.assertIsNone(labels._CONTEXT.get())

    def test_quota_skip_preserves_scientific_verdict_in_compact_raw(self):
        def original(*args, **kwargs):
            row = kwargs['attempt_row']; row.update(kind='direct_flip', status='passed',
                validity={'acceptance_passed': True, 'scientific_diagnostic': 'retained'})
            raise search.CollectionQuotaFull('direct_flip')
        with patch.object(shard, '_ORIGINAL_BUILD', side_effect=original):
            row = shard.measured_build(None, config(), {}, {'candidate_names': ['A', 'B']},
                {'mode': 'flip', 'projection_pool': {'pool': 'direct'}}, {}, 'floorplan203_00000', '/raw', [], {})
        self.assertEqual(row['discard_reasons'], ['collection_quota_full'])
        self.assertTrue(row['validity']['acceptance_passed'])
        self.assertEqual(shard.compact_attempt(row)['validity'], row['validity'])
        self.assertNotIn('pair_file', row)

    def test_complete_caps_stop_discovery_without_native_call_and_hooks_restore(self):
        old = shard.marked.discover_view
        with patch.object(shard, '_ORIGINAL_DISCOVER') as discover:
            with shard.installed() as quota:
                quota.passed = dict.fromkeys(search.KINDS, 8)
                self.assertEqual(list(shard.marked.discover_view(None)), [])
            discover.assert_not_called()
        self.assertIs(shard.marked.discover_view, old)

    def test_native_spawn_installs_mixed_hooks_and_keeps_no_clone_eight_cap(self):
        from src.tools import g01_admission
        old = shard.marked._BASE_WRITE
        def worker(cfg, connection):
            self.assertIs(shard.marked.measured_build, shard.measured_build)
            self.assertIs(shard.marked.replay_worker, shard.replay_worker)
            self.assertIsNot(shard.marked._BASE_WRITE, old)
            self.assertIs(shard.marked.run_search, cadence.run_search)
            self.assertIs(shard.marked.engine.V2Simulator.spawn_points, sharing.spawn_points)
            raise RuntimeError('native failure retained')
        @contextmanager
        def bridge(cfg):
            yield None
        with patch.object(shard.stay, '_ORIGINAL_GENERATOR', side_effect=worker), \
                patch.object(g01_admission, 'witness_bridge', side_effect=bridge):
            with self.assertRaisesRegex(RuntimeError, 'native failure retained'):
                shard.generator_worker(config(), None)
        self.assertIs(shard.marked._BASE_WRITE, old)


class G01SourceTests(unittest.TestCase):
    def test_original_scientific_failure_and_all_arguments_are_preserved(self):
        cfg = config(); data = {'config': cfg, 'metrics': cfg}
        row = {**policy.fields(), **search.fields(), **labels.pair_fields('floorplan203_00000', ['A', 'B']),
            **cadence.fields(), **sharing.fields(), **rank.fields(), **geometry.fields(),
            'producer_module': policy.PRODUCER, 'mode': 'stay', 'kind': 'direct_stay', 'proposed_pool': 'direct',
            'proposal': {'mode': 'stay', 'projection_pool': {'pool': 'direct'}},
            'pair_id': 'floorplan203_00000', 'identities': {'candidate_names': ['A', 'B']}}
        arguments = {'raw_root': cfg['raw_evidence_dir'], 'field_of_view': 60,
            'frozen_render_version': 'unaltered', 'solvability_policy': {'unchanged': True}}
        with patch.object(review, '_BASE_LOAD', return_value=(data, {'file': 'source'})), \
                patch.object(review, 'verify_collection', return_value={}), \
                patch.object(review, 'verify_admission', return_value={}), \
                patch.object(review, 'verify_pair_source', return_value={}), \
                patch.object(review.merge, 'check_pair', side_effect=ValueError('physical proof failed')) as check:
            with review.source_support():
                review.merge.load_source('source')
                with self.assertRaisesRegex(ValueError, 'physical proof failed'):
                    review.merge.check_pair(row, '/pair.json', **arguments)
            check.assert_called_once_with(row, '/pair.json', **arguments)

    def test_new_source_or_role_cannot_bypass_authentication(self):
        previous = roles.verify_proposal_role
        with review.source_support():
            with self.assertRaisesRegex(ValueError, 'authenticated source'):
                review.merge.check_pair(search.fields(), '/pair.json', raw_root='/unknown')
            with self.assertRaises(ValueError): roles.verify_proposal_role({'g01_search_version': 'tampered'}, {})
        self.assertIs(roles.verify_proposal_role, previous)

    def test_direct_pair_uses_the_same_strict_controller_native_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fields = {**policy.declared('FloorPlan203'), **stay.fields(), 'producer_module': policy.PRODUCER,
                'enable_candidate_clone': False, 'scene_pairs_per_class': 8}
            for value in (fixture.data['config'], fixture.data['config']['shard'], fixture.data['metrics']):
                for key in tuple(value):
                    if key.startswith('look_unit_'):
                        value.pop(key)
            fixture.data['config'].update(fields)
            fixture.data['config']['shard'].update(fields)
            fixture.data['metrics'].update(fields)
            fixture.pair.update(kind='direct_stay', **policy.fields(), **search.fields(),
                producer_module=policy.PRODUCER, proposed_pool='direct',
                proposal={'mode': 'stay', 'projection_pool': {'pool': 'direct'}})
            owner = {'pid': 11, 'start_ticks': 90, 'pgid': 11, 'sid': 11}
            leases, portable = [], []
            for world in (0, 1):
                portable.append(fixture.write(fixture.directory / f'witness_w{world}.json',
                    {'process_id': 100 + world, 'unity_pid': 200 + world}))
                leases.append({'token': f'lease{world}', 'owner': owner,
                    'worker_identity': {'pid': 100 + world}, 'unity_identity': {'pid': 200 + world},
                    'release_confirmed': True, 'cleanup_errors': [], 'remaining_owned_group': False})
            fixture.pair['portable_files'] = portable
            fixture.data['metrics']['g01_witness_admission'] = {'owner': owner, 'lease_records': leases}
            fixture.refresh_pair()
            with patch.object(association.merge, 'verify_preflight', side_effect=original_preflight_fixture) as check:
                value = association.verify_pair_controller(fixture.data, fixture.source_ref, fixture.journal,
                    fixture.pair, fixture.pair_path, root=fixture.root)
                self.assertEqual(value['controller_id'], 'marked_controller_000')
                self.assertEqual(value['source_final_controller_id'], 'marked_controller_001')
                self.assertFalse(value['source_record_mutated'])
                check.assert_called_once()
                fixture.data['metrics']['g01_witness_admission']['lease_records'] = []
                with self.assertRaisesRegex(ValueError, 'actual admitted resource lease'):
                    association.verify_pair_controller(fixture.data, fixture.source_ref, fixture.journal,
                        fixture.pair, fixture.pair_path, root=fixture.root)
                fixture.data['metrics']['g01_witness_admission']['lease_records'] = leases
                fixture.raw_attempt['search_controller_id'] = 'marked_controller_001'; fixture.refresh_raw()
                with self.assertRaises(ValueError):
                    association.verify_pair_controller(fixture.data, fixture.source_ref, fixture.journal,
                        fixture.pair, fixture.pair_path, root=fixture.root)

    def test_collection_replays_real_counted_journal_and_rejects_forged_totals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); raw_root = root / 'raw'; raw_root.mkdir()
            journal = root / 'data/world_pairs/v2/journal.jsonl'; journal.parent.mkdir(parents=True)
            quota, rows, raw = search.SceneQuota(), [], {}
            for index, mode in enumerate(('flip', 'stay', 'flip')):
                value = counted('look', mode)
                path = raw_root / f'attempt{index}.json'; path.write_text(json.dumps(value))
                reference = {'file': str(path), 'sha256': review.merge.sha_file(path)}
                raw[str(path)] = value
                rows.append({'raw_attempt': reference, 'proposed_pool': 'look'})
                quota.record(path, value)
            journal.write_text('\n'.join(json.dumps(row) for row in rows))
            journal_ref = {'file': str(journal), 'sha256': review.merge.sha_file(journal)}
            data = {'config': {**config(), 'raw_evidence_dir': str(raw_root)},
                'metrics': {'g01_collection': quota.summary(), 'passed_by_class': quota.passed}}
            with patch.object(review, 'verified_passed_rows', return_value=({}, journal_ref, [])), \
                    patch.object(review.merge, 'ROOT', root):
                self.assertEqual(review.verify_collection(data), quota.summary())
                data['metrics']['g01_collection']['counted_by_proposed_pool_mode']['look_flip'] += 1
                with self.assertRaisesRegex(ValueError, 'actual attempts'): review.verify_collection(data)

    def test_early_failure_stays_unknown_and_collector_bindings_restore(self):
        with patch.object(review, 'verified_passed_rows', return_value=({}, None, [])):
            self.assertIsNone(review.verify_collection({'config': config(), 'metrics': {}}))
            with self.assertRaises(ValueError):
                review.verify_collection({'config': config(), 'metrics': {'g01_collection': search.SceneQuota().summary()}})
        old = (review.original.collect_sources, review.controller.policy, review.controller.verify_pair_controller)
        with review.collector_support():
            self.assertIs(review.controller.policy, policy)
            self.assertIs(review.controller.verify_pair_controller, association.verify_pair_controller)
        self.assertEqual(old, (review.original.collect_sources, review.controller.policy, review.controller.verify_pair_controller))

    def test_missing_resource_report_is_only_a_zero_claim_failed_ledger(self):
        from src.tools import g01_admission
        data = {'config': config(), 'metrics': {}}
        with patch.object(review.merge, 'source_eligibility', return_value={'eligible': False}), \
                patch.object(g01_admission, 'validate_source_admission', return_value=None) as validate:
            self.assertIsNone(review.verify_admission(data))
            validate.assert_called_once_with(data['config'])
            data['metrics']['passed_by_class'] = {'look_flip': 1}
            with self.assertRaisesRegex(ValueError, 'positive pair claims'): review.verify_admission(data)
        data['metrics'] = {}
        with patch.object(review.merge, 'source_eligibility', return_value={'eligible': True}), \
                patch.object(g01_admission, 'validate_source_admission', side_effect=ValueError('missing certificate')) as validate:
            with self.assertRaisesRegex(ValueError, 'missing certificate'): review.verify_admission(data)
            validate.assert_called_once_with(data['config'], data['metrics'])


if __name__ == '__main__':
    unittest.main()
