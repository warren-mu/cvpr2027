"""Synthetic indexed evidence; no Unity or scientific outcome claims."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools import t01v2_flip_ranked_review_checkpoint as check
from src.envs import pair_v2_flip_geometry as geometry
from src.envs import pair_v2_projection_pool as pools
from src.envs import pair_v2_bbox_rank as rank
from src.envs import pair_v2_spawn_sharing as sharing
from src.envs import pair_v2_flip_ranked_policy as policy
from src.envs import pair_v2_native_retry as retry


def native_fixture():
    spec = importlib.util.spec_from_file_location('geometry_checkpoint_native_fixture',
        Path(__file__).with_name('test_pair_v2_native_retry.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root / 'raw' / 'marked_controller_000' / 'construction'
        self.directory.mkdir(parents=True)
        self.pairdir = self.root / 'portable'; self.pairdir.mkdir()
        self.pair_path = self.pairdir / 'pair.json'
        self.scene = 'FloorPlan203'
        native = native_fixture()
        self.sim = native.Sim(self.directory, [False, True, False, True])
        self.sim.scene = self.scene
        metadata = self.sim.controller.last_event.metadata
        objects = retry.engine.objects_by_name(metadata['objects'])
        objects['A']['position'] = native.vector(-1., .8, 1.)
        objects['B']['position'] = native.vector(1., .8, 1.)
        objects['X']['axisAlignedBoundingBox'] = {'center': native.vector(0., 1., 0.),
            'size': native.vector(.4, .4, .4)}
        metadata.update(cameraPosition=native.vector(0., 1., 0.), fov=60., screenWidth=255, screenHeight=255)
        self.baseline = {'objects': copy.deepcopy(metadata['objects']), 'scene': self.scene}
        self.baseline_ref = self.write(self.directory / 'baseline_geometry.json', self.baseline)
        self.view = {'position': native.vector(), 'yaw': 0., 'horizon': 0.}
        self.camera_ref = self.write(self.directory / 'actual_view_metadata.json', metadata)
        self.camera = pools.camera_from_metadata(metadata, self.camera_ref)
        self.write(self.directory / 'action_000000.json', {'index': 0, 'action': 'TeleportFull',
            'arguments': {'position': self.view['position'], 'rotation': native.vector(),
                          'horizon': 0., 'standing': True, 'forceAction': False},
            'before_metadata': copy.deepcopy(metadata), 'after_metadata': copy.deepcopy(metadata),
            'exception': None, 'capture': {'files': {'metadata': self.camera_ref}}})
        coordinates = [native.vector(x, .8, 2.) for x in (-1.6, -1.8, 1.6, 1.8)]
        before = copy.deepcopy(metadata)
        metadata['actionReturn'] = coordinates
        query = self.write(self.directory / 'action_000001.json', {'index': 1,
            'action': 'GetSpawnCoordinatesAboveReceptacle', 'arguments': {'objectId': objects['Table']['objectId'], 'anywhere': True},
            'before_metadata': before, 'after_metadata': copy.deepcopy(metadata), 'exception': None})
        self.tokens = []
        for position in coordinates:
            spawn = {'action': 'GetSpawnCoordinatesAboveReceptacle', 'success': True,
                'scene': self.scene, 'receptacle_name': 'Table', 'receptacle_id_at_query': objects['Table']['objectId'],
                'receptacle_position_at_query': objects['Table']['position'], 'selected_position': position,
                'coordinates': coordinates, 'query_reference': query,
                'source': retry.engine.SOURCE + 'PhysicsRemoteFPSAgentController.cs#L3423',
                'scope': 'native receptacle samples; placement requires subsequent settling/membership checks'}
            token = {'position': position, 'receptacle_name': 'Table', 'spawn': spawn}
            self.tokens.append(token)
            self.sim.spawn_tokens[hashlib.sha256(retry.engine.encoded(spawn)).hexdigest()] = copy.deepcopy(spawn)
        self.sim.action_count = 2
        self.pool = geometry.filter_pool({'camera': self.camera, 'view': self.view,
            'observation_cameras': [pools.camera_after_action(self.camera, {'kind': kind})
                                    for kind in ('turn_left', 'turn_right')],
            'points': {'look': {'A': [], 'B': []}}, 'diagnostics': {'A': [], 'B': []}},
            objects['A'], objects['B'], objects['X'], all_native_tokens=self.tokens, check_deadline=lambda: None)
        self.pool = rank.rank_pool(self.pool, objects['X'])
        self.pool.update(sharing.fields())
        self.pool_ref = self.write(self.directory / 'projection_pools_000002.json', pools.pool_evidence_record(self.pool))
        proposal, _, domain = next(check.roles.view_expanded_proposals(
            [self.pool['points']['look']['A'][0]['token']],
            [self.pool['points']['look']['B'][0]['token']], [self.view]))
        proposal.update(role_domain=domain, projection_pool={'pool': 'look'})
        proposal, role = check.roles.orient_proposal(proposal, self.scene, 0, 0)
        role.update(candidate_a_name='A', candidate_b_name='B')
        self.sides = [role[world + '_source_side'] for world in ('w0', 'w1')]
        self.initial = [copy.deepcopy(proposal[world]) for world in ('w0', 'w1')]
        self.row = {**sharing.fields(), **rank.fields(), **geometry.fields(),
            'producer_module': policy.PRODUCER, 'flip_unit_version': policy.VERSION, 'flip_unit_policy': copy.deepcopy(policy.POLICY),
            'flip_search_version': check.original.search.VERSION, 'flip_search_policy': copy.deepcopy(check.original.search.POLICY),
            'generation_version': check.roles.ROLE_VERSION, 'generation_policy': copy.deepcopy(check.roles.ROLE_POLICY),
            'scene': self.scene, 'pair_id': self.scene.lower() + '_00000', 'kind': 'look_flip', 'mode': 'flip',
            'identities': {'candidate_names': ['A', 'B'], 'anchor_name': 'X'}, 'view': self.view,
            'projection_pool_reference': self.pool_ref, 'proposal': proposal, 'proposal_role': role,
            'flip_geometry_initial_tuple': geometry.complete_pair(self.pool, *self.initial, self.sides),
            'native_retry_records': {}, 'flip_geometry_native_guards': {}, 'portable_files': []}
        self.constructions = {}
        for index, world in enumerate(('w0', 'w1')):
            source_objects = copy.deepcopy(metadata['objects'])
            selected = self.initial[index]
            alternative = next(item['token'] for item in self.pool['points']['look'][self.sides[index]]
                               if item['token']['position'] != selected['position'])
            result = retry.place_native_with_retries(self.sim, 'X', [selected['spawn'], alternative['spawn']],
                rotation=copy.deepcopy(retry.engine.objects_by_name(source_objects)['X']['rotation']),
                record_key=self.row['pair_id'] + '_' + world + '_0')
            metadata = self.sim.controller.last_event.metadata
            record = {key: value for key, value in result.items() if key != 'event'}
            records = {'X': record}
            self.row['native_retry_records'][world] = records
            request = self.write(self.directory / ('request_' + world + '.json'), {'native_retry_records': records})
            final = {'position': result['successful_token']['selected_position'], 'receptacle_name': 'Table',
                     'spawn': result['successful_token']}
            placement = {**copy.deepcopy(final['spawn']), 'requested_position': metadata['objects'][2]['position'],
                         'geometry': {'placement_reference': result['action_reference']}}
            construction = {'request_reference': request, 'before_objects': source_objects,
                            'placement_evidence_by_name': {'X': placement}}
            self.constructions[world] = construction
            ref = self.write(self.pairdir / ('construction_' + world + '.json'), construction)
            self.row['portable_files'].append(ref)
            opposite = self.initial[1] if world == 'w0' else self.row['proposal']['w0']
            self.row['flip_geometry_native_guards'][world] = [
                {'native_reference': geometry.native_reference(attempt['spawn_token']),
                 'opposite_reference': geometry.native_reference(opposite),
                 'source_side': self.sides[index], 'opposite_side': self.sides[1-index]}
                for attempt in record['attempts']]
            self.row['proposal'][world] = final
        self.row['flip_geometry_final_tuple'] = geometry.complete_pair(self.pool,
            self.row['proposal']['w0'], self.row['proposal']['w1'], self.sides)
        # Deliberately actual targets opposite the proposal roles: this remains
        # a diagnostic, never an extra settled-target acceptance condition.
        self.row['target_w0'], self.row['target_w1'] = (('B', 'A') if self.sides == ['A', 'B'] else ('A', 'B'))
        self.config = {'field_of_view': 60.}
        self.refresh_index()

    def write(self, path, value):
        data = retry.engine.encoded(value)
        Path(path).write_bytes(data)
        return {'file': str(path), 'sha256': hashlib.sha256(data).hexdigest()}

    def refresh_index(self):
        files = [{'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
                 for path in sorted(self.directory.glob('*.json')) if path.name != 'evidence_index.json']
        self.index = self.write(self.directory / 'evidence_index.json', {'files': files})
        self.read = check.IndexedReader({'controller_index_reference': self.index}, root=self.root)

    def verify(self):
        return check.verify_used_geometry(self.row, self.pair_path, self.baseline, self.read, self.config, root=self.root)


class GeometryCheckpointTests(unittest.TestCase):
    def test_real_retry_ledgers_initial_final_and_seed_orientation_are_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            proof = fixture.verify()
            self.assertEqual(proof['verified_native_attempts_by_world'], {'w0': 2, 'w1': 2})
            self.assertFalse(proof['unused_pool_diagnostics_revalidated'])
            result = check.verify_role(fixture.row, proof['candidate_positions'], proof)
            self.assertTrue(result['passed'])
            self.assertFalse(result['actual_targets_match_intended'])

    def test_initial_successful_and_failed_retry_guards_cannot_be_substituted(self):
        for change in ('initial', 'final', 'w0_opposite', 'w1_opposite', 'omit_failed', 'seed'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                f = Fixture(directory)
                if change in ('initial', 'final'):
                    f.row['flip_geometry_' + change + '_tuple']['worlds']['w0']['predicted_anchor_center']['y'] += .1
                elif change == 'w0_opposite':
                    f.row['flip_geometry_native_guards']['w0'][0]['opposite_reference'] = geometry.native_reference(f.row['proposal']['w1'])
                elif change == 'w1_opposite':
                    f.row['flip_geometry_native_guards']['w1'][0]['opposite_reference'] = geometry.native_reference(f.initial[0])
                elif change == 'omit_failed':
                    f.row['flip_geometry_native_guards']['w1'].pop(0)
                else:
                    f.row['proposal_role']['assignment_digest'] = '0' * 64
                with self.assertRaises(ValueError):
                    proof = f.verify()
                    check.verify_role(f.row, proof['candidate_positions'], proof)

    def test_baseline_camera_native_membership_and_controller_index_are_authenticated(self):
        for change in ('baseline_position', 'baseline_box', 'camera', 'coordinates', 'foreign_controller', 'old_version'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                f = Fixture(directory)
                if change == 'baseline_position':
                    f.baseline['objects'][0]['position']['y'] += .1
                elif change == 'baseline_box':
                    f.baseline['objects'][2]['axisAlignedBoundingBox']['size']['y'] += .1
                elif change == 'camera':
                    pool = copy.deepcopy(f.read(f.pool_ref)); pool['camera']['yaw_degrees'] += 1
                    f.row['projection_pool_reference'] = f.write(Path(f.pool_ref['file']), pool); f.refresh_index()
                elif change == 'coordinates':
                    f.row['proposal']['w0']['spawn']['coordinates'] = []
                elif change == 'foreign_controller':
                    foreign = f.root / 'other_controller.json'
                    ref = f.write(foreign, f.read(f.pool_ref))
                    f.row['projection_pool_reference'] = ref
                else:
                    f.row['flip_unit_version'] = 'historical-ranked-v1'
                with self.assertRaises(ValueError): f.verify()

    def test_original_failed_retry_action_must_match_recorded_unchanged_state(self):
        with tempfile.TemporaryDirectory() as directory:
            f = Fixture(directory)
            record = f.row['native_retry_records']['w1']['X']
            record['attempts'][0]['before_object_poses']['X']['position']['x'] += .01
            with self.assertRaises(ValueError): f.verify()

    def test_independent_role_does_not_use_old_native_surface_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            f = Fixture(directory)
            proof = f.verify()
            with patch.object(check.original.search, 'side_for_point', side_effect=AssertionError('obsolete surface role')):
                self.assertTrue(check.verify_role(f.row, proof['candidate_positions'], proof)['passed'])

    def test_source_support_preserves_full_checker_and_restores_role_after_failure(self):
        # Expensive source/science functions are isolated here only to verify
        # wiring: actual geometry/index tests above use real bytes and retries.
        with tempfile.TemporaryDirectory() as directory:
            f = Fixture(directory)
            cfg = {**policy.camera_fields(60, f.scene), 'scene': f.scene,
                'producer_module': policy.PRODUCER, 'raw_evidence_dir': str(f.root / 'raw')}
            cfg['shard'] = copy.deepcopy(cfg)
            data = {'config': cfg, 'metrics': {**policy.camera_fields(60, f.scene), 'producer_module': policy.PRODUCER}}
            pairref = f.write(f.pair_path, f.row)
            association = {'controller_index_reference': f.index, 'baseline_reference': f.baseline_ref}
            previous_role = check.roles.verify_proposal_role
            def science(row, path, **kwargs):
                self.assertEqual(kwargs, {'raw_root': cfg['raw_evidence_dir'], 'sentinel': 'all-original-kwargs'})
                self.assertTrue(check.roles.verify_proposal_role(row, {name: obj['position'] for name, obj in
                    retry.engine.objects_by_name(f.baseline['objects']).items()})['passed'])
                raise RuntimeError('original scientific gate failed')
            with patch.object(check.original, '_BASE_LOAD', return_value=(data, {'file': 'synthetic-source'})), \
                    patch.object(check.policy, 'validate_source'), \
                    patch.object(check.original.labels, 'checked_pair', side_effect=lambda function, row, path, declared_source, **kw: function(row, path, **kw)), \
                    patch.object(check, 'verified_passed_rows', return_value=({'eligible': True}, {}, [({}, pairref, f.pair_path)])), \
                    patch.object(check.original.association, 'verify_pair_controller', return_value=association), \
                    patch.object(check.merge, 'ROOT', f.root), patch.object(check.merge, 'check_pair', side_effect=science):
                with check.source_support():
                    check.merge.load_source('synthetic')
                    installed_role = check.roles.verify_proposal_role
                    with self.assertRaisesRegex(RuntimeError, 'original scientific gate failed'):
                        check.merge.check_pair(f.row, f.pair_path, raw_root=cfg['raw_evidence_dir'], sentinel='all-original-kwargs')
                    self.assertIs(check.roles.verify_proposal_role, installed_role)
            self.assertIs(check.roles.verify_proposal_role, previous_role)

    def test_stay_delegates_to_existing_checker_without_flip_geometry_reinterpretation(self):
        row = {**sharing.fields(), **rank.fields(), **geometry.fields(),
            'producer_module': policy.PRODUCER, 'flip_unit_version': policy.VERSION,
            'flip_unit_policy': copy.deepcopy(policy.POLICY), 'mode': 'stay',
            'flip_search_version': check.original.search.VERSION,
            'flip_search_policy': copy.deepcopy(check.original.search.POLICY)}
        cfg = {'producer_module': policy.PRODUCER, 'raw_evidence_dir': '/synthetic/stay'}
        data = {'config': cfg, 'metrics': {}}
        with patch.object(check.original, '_BASE_LOAD', return_value=(data, {})), \
                patch.object(check.policy, 'validate_source'), \
                patch.object(check.original.labels, 'checked_pair', side_effect=lambda function, row, path, declared_source, **kw: function(row, path, **kw)), \
                patch.object(check, 'verify_used_geometry', side_effect=AssertionError('Stay must keep original domain')), \
                patch.object(check.merge, 'check_pair', return_value={'stay': 'original'}) as science:
            with check.source_support():
                check.merge.load_source('synthetic')
                result = check.merge.check_pair(row, 'unused', raw_root=cfg['raw_evidence_dir'])
        self.assertEqual(result, {'stay': 'original'})
        science.assert_called_once_with(row, 'unused', raw_root=cfg['raw_evidence_dir'])


if __name__ == '__main__':
    unittest.main()
