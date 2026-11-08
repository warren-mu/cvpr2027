"""Synthetic CPU integration. Run only after commit through run_logged."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.envs import pair_v2_clone_shard as shard
from src.envs.pair_v2_clone_policy import make_clone_recipe, verify_clone_creation


def point(x=0.0, y=0.5, z=0.0):
    return {'x': x, 'y': y, 'z': z}


def objects(natural=False):
    result = []
    for name, category, x, pickup in [('A', 'Mug', 0., True), ('X', 'Apple', 1., True), ('Table', 'Table', 2., False)]:
        result.append({'name': name, 'objectId': name + '_id', 'objectType': category,
            'position': point(x), 'rotation': point(0, 0), 'pickupable': pickup, 'moveable': False,
            'receptacle': not pickup, 'parentReceptacles': ['Table_id'],
            'axisAlignedBoundingBox': {'center': point(x), 'size': point(.2, .2, .2)}})
    if natural:
        clone = copy.deepcopy(result[0]); clone.update(name='B', objectId='B_id', position=point(.6))
        result.append(clone)
    return result


def poses(rows):
    return [{'objectName': row['name'], 'position': copy.deepcopy(row['position']),
             'rotation': copy.deepcopy(row['rotation'])} for row in rows if row['pickupable'] or row['moveable']]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(data)
    return {'file': str(path), 'sha256': hashlib.sha256(data).hexdigest()}


class MemoryStore:
    def __init__(self):
        self.rows = []

    def save_json(self, name, row):
        self.rows.append((name, copy.deepcopy(row)))
        return {'file': '/synthetic/' + name, 'sha256': 'synthetic'}


def make_sim():
    sim = object.__new__(shard.PreparationSimulator)
    sim.scene = 'FloorPlan1'
    sim.config = {'max_receptacles': 4}
    sim.store = MemoryStore()
    sim.scene_preflight = None
    return sim


def original_preflight(rows, passed=True):
    value = {'passed': passed, 'snapshot': {'scene': 'FloorPlan1', 'objects': copy.deepcopy(rows), 'poses': poses(rows)}}

    def call(sim):
        sim.scene_preflight = copy.deepcopy(value)
        sim.store.save_json('scene_preflight.json', value)
        return copy.deepcopy(value)
    return value, call


class CloneShardTests(unittest.TestCase):
    def tearDown(self):
        shard._PREPARATION = None
        shard._REPLAY_CLEANUP = None

    def test_original_preflight_failure_never_queries_or_clones(self):
        original, call = original_preflight(objects(), False)
        sim = make_sim()
        with patch.object(shard.V2CloneSimulator, 'preflight_scene', call), \
             patch.object(shard, 'select_clone_token') as select, patch.object(shard, 'prepare_clone_baseline') as prepare:
            result = sim.preflight_scene()
        self.assertEqual(result, original)
        select.assert_not_called(); prepare.assert_not_called()
        self.assertEqual(shard._PREPARATION['status'], 'original_preflight_failed')

    def test_success_returns_expanded_baseline_without_replacing_original_preflight(self):
        original, call = original_preflight(objects())
        sim = make_sim()
        enlarged = copy.deepcopy(original['snapshot']); enlarged['objects'].append({'name': 'actual_clone'})
        lineage = {'passed': True, 'candidate_names': ['X', 'actual_clone']}
        token = {'selected_position': point(3)}
        prepared = {'passed': True, 'snapshot': enlarged, 'lineage': lineage,
                    'record': {'passed': True}, 'record_reference': {'file': '/raw/validation.json', 'sha256': 'fixture'}}
        with patch.object(shard.V2CloneSimulator, 'preflight_scene', call), \
             patch.object(shard, 'select_clone_token', return_value=token), \
             patch.object(shard, 'prepare_clone_baseline', return_value=prepared) as prepare:
            result = sim.preflight_scene()
        self.assertEqual(result['snapshot'], enlarged)
        self.assertEqual(sim.scene_preflight, original)
        self.assertEqual(sim.store.rows[0], ('scene_preflight.json', original))
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(prepare.call_args.args[2], 'X')  # Apple before Mug, fixed category order.
        self.assertEqual(shard._PREPARATION['status'], 'clone_ready')
        self.assertEqual(sim.store.rows[-1][0], 'clone_selection.json')

    def test_failed_clone_is_retained_without_natural_fallback_or_retry(self):
        original, call = original_preflight(objects())
        sim = make_sim()
        with patch.object(shard.V2CloneSimulator, 'preflight_scene', call), \
             patch.object(shard, 'select_clone_token', return_value={'selected_position': point(3)}), \
             patch.object(shard, 'prepare_clone_baseline', return_value={'passed': False, 'record': {'reason': 'settling'}}) as prepare:
            result = sim.preflight_scene()
        self.assertFalse(result['passed'])
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(shard._PREPARATION['status'], 'clone_setup_failed')
        self.assertTrue(shard._PREPARATION['clone_attempted'])
        self.assertEqual(sim.scene_preflight, original)

    def test_clone_exception_records_error_and_propagates_for_scene_stop(self):
        _, call = original_preflight(objects())
        sim = make_sim()
        with patch.object(shard.V2CloneSimulator, 'preflight_scene', call), \
             patch.object(shard, 'select_clone_token', return_value={'selected_position': point(3)}), \
             patch.object(shard, 'prepare_clone_baseline', side_effect=RuntimeError('native failed')) as prepare:
            with self.assertRaisesRegex(RuntimeError, 'native failed'):
                sim.preflight_scene()
        self.assertEqual(prepare.call_count, 1)
        self.assertIn('native failed', sim.store.rows[-1][1]['error'])

    def test_natural_fallback_is_explicit_only_without_two_singletons(self):
        original, call = original_preflight(objects(natural=True))
        sim = make_sim()
        with patch.object(shard.V2CloneSimulator, 'preflight_scene', call), \
             patch.object(shard, 'prepare_clone_baseline') as prepare:
            result = sim.preflight_scene()
        self.assertEqual(result, original)
        self.assertEqual(shard._PREPARATION['status'], 'natural_fallback')
        self.assertTrue(shard.cloned_identity_sets(result['snapshot']['objects']))
        prepare.assert_not_called()

    def test_proposal_filters_source_footprint_and_keeps_literal_native_token(self):
        rows = objects()
        source = rows[0]
        blocked = {'selected_position': point(.05), 'receptacle_name': 'Table'}
        selected = {'selected_position': point(.5), 'receptacle_name': 'Table'}
        sim = SimpleNamespace(config={'max_receptacles': 4}, spawn_points=lambda name: [selected, blocked])
        record = {}
        actual = shard.select_clone_token(sim, {'objects': rows}, source, record)
        self.assertIs(actual, selected)
        self.assertEqual(record['proposal_queries'][0]['rejected'][0]['reason'], 'editable_aabb_footprint_proposal')
        self.assertNotIn('valid_physical', record)

    def test_identity_filter_uses_actual_fixed_source_and_clone_names(self):
        rows = objects(natural=True)
        extra = copy.deepcopy(rows[1]); extra.update(name='X_copy', objectId='X_copy_id')
        rows.append(extra)
        other = copy.deepcopy(rows[1]); other.update(name='anchor', objectId='anchor_id', objectType='Book')
        rows.append(other)
        shard._PREPARATION = {'status': 'clone_ready', 'lineage': {'passed': True, 'candidate_names': ['X', 'X_copy']}}
        identities = shard.cloned_identity_sets(rows)
        self.assertTrue(identities)
        self.assertTrue(all(set(row['candidate_names']) == {'X', 'X_copy'} for row in identities))
        self.assertTrue(all(row['anchor_name'] == 'anchor' for row in identities))

    def test_replay_uses_clone_worker_and_retains_cleanup_failure(self):
        child = SimpleNamespace(result={'success': True}, error=None, receive=lambda: None, stop=lambda: ['cleanup failed'])
        config = {'deadline': 100, 'shard': {'original': True}}
        identities = {'candidate_names': ['A', 'B'], 'anchor_name': 'X', 'target_type': 'Mug', 'anchor_type': 'Apple'}
        with patch.object(shard, 'Worker', return_value=child) as worker:
            result = shard.wait_replay(config, {'clone_context': {'setup_passed': True}}, [], 'A', identities, Path('/raw'))
        self.assertIs(worker.call_args.args[0], shard.clone_replay_worker)
        self.assertTrue(worker.call_args.args[1]['enable_candidate_clone'])
        self.assertEqual(worker.call_args.args[1]['shard'], config['shard'])
        self.assertFalse(result['success'])
        with patch.object(shard, 'Worker', return_value=child) as worker:
            shard.wait_replay(config, {}, [], 'A', identities, Path('/raw'))
        self.assertIs(worker.call_args.args[0], shard.engine.replay_worker)

    def test_worker_bindings_restore_even_on_exception(self):
        previous = (shard.engine.V2Simulator, shard.base.identity_sets, shard.base.wait_replay, shard.base.export_pair)

        def fail(config, connection):
            self.assertIs(shard.engine.V2Simulator, shard.PreparationSimulator)
            self.assertIs(shard.base.wait_replay, shard.wait_replay)
            raise RuntimeError('synthetic failure')
        with patch.object(shard.base, 'generator_worker', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                shard.generator_worker({'enable_candidate_clone': True}, None)
        self.assertEqual(previous, (shard.engine.V2Simulator, shard.base.identity_sets, shard.base.wait_replay, shard.base.export_pair))
        self.assertIsNone(shard._PREPARATION)

    def test_replay_timeout_keeps_actual_cleanup_errors_in_final_worker_result(self):
        child = SimpleNamespace(result=None, error=None, receive=lambda: None, stop=lambda: ['Unity exit not confirmed'])
        identities = {'candidate_names': ['A', 'B'], 'anchor_name': 'X', 'target_type': 'Mug', 'anchor_type': 'Apple'}

        def generate(config, connection):
            with self.assertRaisesRegex(TimeoutError, 'replay deadline'):
                shard.wait_replay({'deadline': 0}, {'clone_context': {}}, [], 'A', identities, Path('/raw'))
            return {'status': 'incomplete', 'cleanup_errors': []}
        with patch.object(shard, 'Worker', return_value=child), \
             patch.object(shard.base, 'generator_worker', side_effect=generate):
            result = shard.generator_worker({'enable_candidate_clone': True}, None)
        self.assertEqual(result['cleanup_errors'], ['Unity exit not confirmed'])
        self.assertIsNone(shard._REPLAY_CLEANUP)

    def test_clone_exclusion_does_not_mislabel_original_preflight_or_black_failure(self):
        def result(config, connection):
            shard._PREPARATION = {'original_scene_preflight_passed': True, 'passed': False,
                                  'status': 'clone_setup_failed', 'record_reference': {'file': '/raw/failed.json'}}
            return {'status': 'completed_search', 'kind': 'pair_v2_shard', 'black_frame_detected': False}
        with patch.object(shard.base, 'generator_worker', side_effect=result):
            measured = shard.generator_worker({'enable_candidate_clone': True, 'scene': 'FloorPlan1'}, None)
        self.assertEqual(measured['status'], 'clone_setup_failed')
        self.assertTrue(measured['scene_excluded']['original_scene_preflight_passed'])
        self.assertEqual(measured['kind'], 'pair_v2_shard')

    def test_parent_opt_in_preserves_shard_and_restores_original_bindings(self):
        original_manifest = {'scene': 'FloorPlan1', 'gpu_identity': {'gpu_uuid': 'GPU-synthetic'}}
        config = {'shard': copy.deepcopy(original_manifest), 'max_raw_bytes': 512 * 1024 * 1024,
                  'resolution': 255, 'seed': 0}
        original_worker, original_writer = shard.base.Worker, shard.base.write_result
        captured = {}

        def worker(target, child_config):
            captured.update(target=target, config=copy.deepcopy(child_config))
            return object()

        def fake_main():
            shard.base.Worker(shard.base.generator_worker, config)
            shard.base.write_result('fixture', {'kind': 'pair_v2_shard'}, config)
            return 1
        with patch.object(shard.base, 'Worker', side_effect=worker) as saved_worker, \
             patch.object(shard.base, 'write_result') as saved_writer, patch.object(shard.base, 'main', side_effect=fake_main):
            self.assertEqual(shard.main(), 1)
            self.assertIs(shard.base.Worker, saved_worker)
            self.assertIs(shard.base.write_result, saved_writer)
            self.assertTrue(saved_writer.call_args.args[2]['enable_candidate_clone'])
            self.assertEqual(saved_writer.call_args.args[2]['max_raw_bytes'], 2 * 1024 * 1024 * 1024)
        self.assertEqual(config['shard'], original_manifest)
        self.assertEqual(captured['config']['max_raw_bytes'], 2 * 1024 * 1024 * 1024)
        self.assertEqual(config['clone_raw_storage_limit']['max_stored_payload_bytes_per_simulator'], config['max_raw_bytes'])
        self.assertEqual(config['clone_portable_archive']['max_run_bytes'], 128 * 1024 * 1024)
        self.assertEqual((config['resolution'], config['seed']), (255, 0))
        self.assertIs(captured['target'], shard.generator_worker)
        self.assertIs(shard.base.Worker, original_worker)
        self.assertIs(shard.base.write_result, original_writer)

    def test_json_archive_recurses_and_authenticates_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); raw = root / 'scratch'
            leaf = save(raw / 'leaf.json', {'actual_name': 'native_copy'})
            source = save(raw / 'parent.json', {'action_reference': leaf})
            with patch.object(shard.base, 'ROOT', root):
                archive = shard.JsonProofArchive({'name': 'fixture', 'runtime': {'scratch': str(raw)}})
                index, entries = archive.archive('pair0', {'creation': source})
                self.assertEqual(len(entries), 2)
                self.assertEqual(json.loads((root / index['file']).read_text())['entries'], entries)
                first_bytes = archive.total_bytes
                _, again = archive.archive('pair1', {'creation': source})
                self.assertEqual(entries, again)
                self.assertGreater(archive.total_bytes, first_bytes)  # New index is charged; blobs are shared.
                self.assertEqual(len(list((archive.directory / 'objects').iterdir())), 2)
                self.assertEqual(archive.total_bytes, sum(path.stat().st_size for path in archive.directory.rglob('*.json')))
                with self.assertRaises(FileExistsError):
                    archive.archive('pair0', {'creation': source})
                Path(leaf['file']).write_text('{}')
                with self.assertRaisesRegex(ValueError, 'hash changed'):
                    archive.archive('pair2', {'creation': source})

    def test_clone_export_uses_realized_w0_and_archives_both_fresh_recreations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); raw = root / 'scratch'
            original = objects()
            recipe = make_clone_recipe(original, 'A', point(.7))
            created = copy.deepcopy(original)
            clone = copy.deepcopy(created[0]); clone.update(name='actual_copy', objectId='new_id', position=point(.7))
            created.append(clone)
            lineage = verify_clone_creation(original, created, recipe, action_success=True)
            reference = save(raw / 'baseline.json', {'synthetic': True})
            context = {'enabled': True, 'setup_passed': True, 'lineage': lineage, 'recipe': recipe,
                       'source_baseline_reference': reference, 'baseline_validation_reference': reference}
            worlds = [{'objects': copy.deepcopy(created), 'poses': poses(created), 'clone_context': copy.deepcopy(context)} for _ in range(2)]
            worlds[0]['objects'][0]['position']['x'] = .004  # Realized settling is the w1 request baseline.
            worlds[0]['poses'] = poses(worlds[0]['objects'])
            worlds[1]['objects'] = copy.deepcopy(worlds[0]['objects'])
            worlds[1]['objects'][1]['position'] = point(1.2)
            worlds[1]['poses'] = poses(worlds[1]['objects'])
            for index, world in enumerate(worlds):
                world['reference'] = save(raw / f'world{index}.json', world)
            witnesses = []
            for index in range(2):
                action = save(raw / f'recreate_action{index}.json', {'synthetic_action': index})
                recreation = save(raw / f'recreate{index}.json', {'passed': True, 'action_reference': action})
                restored = save(raw / f'restored{index}.json', {'clone_recreation_reference': recreation})
                witnesses.append({'restoration': {'restored_snapshot_reference': restored,
                                                  'clone_recreation': {'passed': True}}})
            row = {'pair_id': 'fixture_pair', 'identities': {'anchor_name': 'X'}}
            constructions = [{'requested_poses': poses(created)}, {'requested_poses': worlds[1]['poses']}]
            shard._PREPARATION = {'status': 'clone_ready', 'passed': True}
            config = {'name': 'fixture', 'runtime': {'scratch': str(raw)}}
            with patch.object(shard.base, 'ROOT', root):
                archive = shard.JsonProofArchive(config)
                result = shard.export_pair(lambda *args: 'exported', archive, config, row, worlds, [], [], witnesses, constructions)
            self.assertEqual(result, 'exported')
            self.assertTrue(row['clone_pair_check']['passed'])
            originals = {entry['source_file'] for entry in row['clone_evidence_files']}
            self.assertIn(str(raw / 'recreate_action0.json'), originals)
            self.assertIn(str(raw / 'recreate_action1.json'), originals)
            self.assertEqual(row['candidate_origin'], 'common_baseline_clone')
            for witness in witnesses:
                witness['restoration']['clone_recreation']['already_recreated'] = True
            with self.assertRaisesRegex(ValueError, 'independently create'):
                shard.export_pair(lambda *args: 'never', archive, config, row, worlds, [], [], witnesses, constructions)


if __name__ == '__main__':
    unittest.main()
