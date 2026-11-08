"""Synthetic CPU checks for buffering/entry wiring; no simulator is launched."""
import copy
import builtins
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.envs import pair_v2_stable_shard as shard
from src.envs.pair_v2_evidence import EvidenceSession, RetentionPolicy, compact_attempt


def physical_row(name='X', error=.03):
    reason = 'edited_position_error_above_0_02m:' + name
    return {'status': 'rejected', 'discard_reasons': ['w0_physical'],
        'physical_w0': {'valid_physical': False, 'native_is_scene_at_rest': True,
            'reasons': [reason], 'edited_names': [name],
            'edits': {name: {'position_error_m': error, 'parent_receptacle_verified': True}},
            'all_object_changes': {name: {'position_change_m': .5}, 'A': {'position_change_m': .001}}}}


class EvidenceTests(unittest.TestCase):
    def test_common_is_immediate_attempt_reads_memory_then_can_be_omitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            session = EvidenceSession(scratch)
            store = session.store(scratch / 'worker', 10000)
            common = store.save_json('common.json', {'baseline': True})
            self.assertTrue(Path(common['file']).is_file())
            session.begin()
            transient = store.save_json('attempt.json', {'actual': 2})
            self.assertFalse(Path(transient['file']).exists())
            old_open, old_stat = Path.open, Path.stat
            with session.reading():
                self.assertEqual(json.loads(Path(transient['file']).read_text()), {'actual': 2})
                self.assertEqual(Path(transient['file']).stat().st_size, transient['bytes'])
                self.assertTrue(Path(transient['file']).is_file())
            self.assertIs(Path.open, old_open); self.assertIs(Path.stat, old_stat)
            saved = session.finish(False)
            self.assertFalse(saved['full_raw_retained'])
            self.assertFalse(Path(transient['file']).exists())
            self.assertTrue(Path(common['file']).is_file())
            self.assertEqual(saved['disk_files_deleted'], 0)

    def test_keep_flushes_exact_payloads_once_and_final_index_omits_unmaterialized(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            session = EvidenceSession(scratch); store = session.store(scratch / 'worker', 10000)
            session.begin(); dropped = store.save('omitted.bin', b'omitted'); session.finish(False)
            session.begin(); kept = store.save('kept.bin', b'kept'); session.finish(True)
            index = store.index()
            data = json.loads(Path(index['file']).read_text())
            self.assertEqual([ref['file'] for ref in data['files']], [kept['file']])
            self.assertFalse(Path(dropped['file']).exists())
            self.assertEqual(Path(kept['file']).read_bytes(), b'kept')
            with self.assertRaises(FileExistsError):
                store.save('kept.bin', b'new')

    def test_fresh_child_bundle_can_be_absorbed_without_early_disk_materialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            child = EvidenceSession(scratch, buffer_all=True)
            store = child.store(scratch / 'replay', 10000)
            ref = store.save_json('metadata.json', {'object_name': 'actual_clone'})
            store.index()
            self.assertFalse((scratch / 'replay').exists())
            parent = EvidenceSession(scratch); parent.begin(); parent.absorb(child.bundle())
            with parent.reading():
                self.assertEqual(json.loads(Path(ref['file']).read_text()), {'object_name': 'actual_clone'})
            parent.finish(True)
            self.assertTrue(Path(ref['file']).is_file())
            with self.assertRaises(ValueError):
                parent.begin(); parent.absorb(child.bundle())

    def test_read_view_restores_on_exception_and_rejects_scratch_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            session = EvidenceSession(scratch, buffer_all=True)
            store = session.store(scratch / 'worker', 5)
            store.save('five', b'12345')
            with self.assertRaises(ValueError):
                store.save('six', b'6')
            with self.assertRaises(ValueError):
                store.save('../escape', b'')
            old_open, old_stat = Path.open, Path.stat
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with session.reading():
                    raise RuntimeError('fixture')
            self.assertIs(Path.open, old_open); self.assertIs(Path.stat, old_stat)
            with self.assertRaises(ValueError):
                EvidenceSession(scratch, imports={str(scratch.parent / 'outside'): b'x'})

    def test_real_png_decodes_from_path_bytes_and_pil_string_filename_in_view(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            session = EvidenceSession(scratch, buffer_all=True)
            store = session.store(scratch / 'worker', 100000)
            buffer = io.BytesIO()
            original = np.full((255, 255, 3), 79, dtype=np.uint8)
            Image.fromarray(original).save(buffer, format='PNG')
            ref = store.save('prefix.png', buffer.getvalue())
            old_builtin = builtins.open
            with session.reading():
                # Exact frozen restore_snapshot form, then PIL's alternate str path.
                first = np.asarray(Image.open(io.BytesIO(Path(ref['file']).read_bytes())).convert('RGB'))
                with Image.open(ref['file']) as image:
                    second = np.asarray(image).copy()
                self.assertTrue(np.array_equal(first, original))
                self.assertTrue(np.array_equal(second, original))
            self.assertIs(builtins.open, old_builtin)
            self.assertFalse(Path(ref['file']).exists())

    def test_retention_counts_complete_physical_reason_and_categories(self):
        policy = RetentionPolicy()
        decisions = [policy.decide(physical_row())[0] for _ in range(4)]
        self.assertEqual(decisions, [True, True, True, False])
        self.assertEqual(policy.reason_counts, {'edited_position_error_above_0_02m:X': 4})
        self.assertEqual(policy.physical_distribution['edited_position_error'], 4)
        self.assertTrue(policy.decide(physical_row('different_object'))[0])
        self.assertTrue(policy.decide(physical_row(), black=True)[0])
        before = dict(policy.reason_counts)
        self.assertTrue(policy.decide({'status': 'passed'})[0])
        self.assertEqual(policy.reason_counts, before)

    def test_compact_rejection_keeps_journal_identity_and_key_numbers(self):
        row = {**physical_row(), 'pair_id': 'p', 'scene': 'FloorPlan1', 'kind': 'direct_flip',
            'generation_seconds': 1.5, 'full_metadata': {'large': 'excluded'},
            'evidence_retention': {'full_raw_retained': False}, 'raw_complete': True}
        compact = compact_attempt(row)
        self.assertNotIn('full_metadata', compact)
        for key in ('pair_id', 'scene', 'kind', 'status', 'discard_reasons', 'generation_seconds'):
            self.assertEqual(compact[key], row[key])
        self.assertEqual(compact['physical_key_numbers']['physical_w0']['max_other_position_change_m'], .001)


class StableEntryTests(unittest.TestCase):
    def tearDown(self):
        shard._SESSION = None
        shard._REPLAY_RAW_COMPLETE = None
        shard.clone._REPLAY_CLEANUP = None

    def test_config_and_mro_select_new_preflight_without_mutating_manifest(self):
        manifest = {'name': 'fixture', 'gpu_identity': {'gpu_uuid': 'GPU-fixture'}}
        config = {'shard': copy.deepcopy(manifest)}
        shard.enable_config(config)
        self.assertEqual(config['shard'], manifest)
        self.assertTrue(config['enable_candidate_clone'])
        self.assertEqual(config['preflight_version'], shard.PREFLIGHT_VERSION)
        self.assertEqual(config['preflight_policy'], shard.PREFLIGHT_POLICY)
        self.assertEqual(config['evidence_version'], shard.stable_policy.EVIDENCE_VERSION)
        self.assertEqual(config['evidence_policy'], shard.stable_policy.EVIDENCE_POLICY)
        self.assertFalse(config['raw_evidence_policy']['disk_deletion'])
        mro = shard.StablePreparationSimulator.__mro__
        self.assertLess(mro.index(shard.clone.PreparationSimulator), mro.index(shard.StableCloneSimulator))
        self.assertLess(mro.index(shard.StableCloneSimulator), mro.index(shard.clone.V2CloneSimulator))

    def test_replay_reads_private_source_bytes_and_returns_private_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve(); image = scratch / 'source.png'
            config = {'runtime': {'scratch': str(scratch)}, 'snapshot': {},
                      '_private_source_image_bytes': {str(image): b'actual encoded source PNG'},
                      'raw_dir': str(scratch / 'replay')}
            old_store = shard.engine.EvidenceStore

            def replay(actual_config, connection):
                self.assertNotIn('_private_source_image_bytes', actual_config)
                self.assertEqual(image.read_bytes(), b'actual encoded source PNG')
                store = shard.engine.EvidenceStore(actual_config['raw_dir'], max_bytes=10000)
                ref = store.save_json('metadata.json', {'real_capture': True})
                store.index()
                return {'status': 'completed', 'success': True, 'reference': ref}
            with patch.object(shard.engine, 'replay_worker', side_effect=replay):
                result = shard.replay_worker(config, None)
            self.assertTrue(result['buffered_raw_complete'])
            self.assertTrue(result['_private_evidence_bundle']['payloads'])
            self.assertFalse((scratch / 'replay').exists())
            self.assertIs(shard.engine.EvidenceStore, old_store)

    def test_replay_black_flushes_before_parent_reads_86_evidence(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            config = {'runtime': {'scratch': str(scratch)}, 'snapshot': {}, 'raw_dir': str(scratch / 'replay')}

            def replay(actual_config, connection):
                store = shard.engine.EvidenceStore(actual_config['raw_dir'], max_bytes=10000)
                buffer = io.BytesIO()
                Image.fromarray(np.zeros((255, 255, 3), dtype=np.uint8)).save(buffer, format='PNG')
                image = store.save('black.png', buffer.getvalue())
                ref = store.save_json('invalid_rgb.json', {'files': {'rgb': image}, 'black': True})
                return {'exit_code': 86, 'black_frame_reference': ref}
            with patch.object(shard.engine, 'replay_worker', side_effect=replay):
                result = shard.replay_worker(config, None)
            self.assertIsNone(result['_private_evidence_bundle'])
            self.assertTrue(Path(result['black_frame_reference']['file']).is_file())
            self.assertTrue((scratch / 'replay' / 'black.png').is_file())
            with Image.open(scratch / 'replay' / 'black.png') as image:
                self.assertEqual(image.size, (255, 255))
                self.assertFalse(np.any(np.asarray(image)))

    def test_wait_removes_private_bundle_before_recording_witness_and_preserves_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            shard._SESSION = EvidenceSession(scratch); shard._SESSION.begin()
            child_session = EvidenceSession(scratch, buffer_all=True)
            store = child_session.store(scratch / 'replay', 10000)
            ref = store.save('frame.png', b'actual')
            child = SimpleNamespace(result={'success': True, 'buffered_raw_complete': True,
                '_private_evidence_bundle': child_session.bundle()}, error=None,
                receive=lambda: None, stop=lambda: ['cleanup error'])
            shard.clone._REPLAY_CLEANUP = []
            identities = {'target_type': 'Mug', 'anchor_type': 'Apple'}
            with patch.object(shard, 'Worker', return_value=child) as worker:
                result = shard.wait_replay({'deadline': time.time() + 1000}, {}, [], 'A', identities, scratch / 'replay')
            self.assertNotIn('_private_evidence_bundle', result)
            self.assertFalse(result['success'])
            self.assertEqual(shard.clone._REPLAY_CLEANUP, ['cleanup error'])
            self.assertIs(worker.call_args.args[0], shard.replay_worker)
            self.assertEqual(shard._SESSION.data(ref['file']), b'actual')

    def test_full_generator_four_rejections_write_three_raw_examples_and_four_thumbnails(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve(); raw = scratch / 'run'; raw.mkdir()
            config = {'runtime': {'scratch': str(scratch)}, 'raw_evidence_dir': str(raw),
                      'enable_candidate_clone': True, 'preflight_version': shard.PREFLIGHT_VERSION,
                      'evidence_version': shard.stable_policy.EVIDENCE_VERSION, 'scene': 'FloorPlan1'}
            actual_refs = []

            def original_build(sim, cfg, baseline, identities, proposal, view, pair_id, directory, reachable, attempt_row=None):
                attempt_row.update(physical_row(), pair_id=pair_id, scene='FloorPlan1')
                actual_refs.append(sim.store.save_json(pair_id + '.json', {'measured': pair_id}))
                # Same-process restore/export reads can resolve real buffered bytes.
                self.assertEqual(json.loads(Path(actual_refs[-1]['file']).read_text()), {'measured': pair_id})
                return attempt_row

            def generate(cfg, connection):
                store = shard.engine.EvidenceStore(raw / 'construction', max_bytes=1000000)
                store.save_json('scene_preflight.json', {'preflight_version': shard.PREFLIGHT_VERSION,
                    'passed': True, 'reasons': [], 'error': None})
                sim = SimpleNamespace(store=store, last_capture={'rgb': np.full((255, 255, 3), 90, dtype=np.uint8)})
                for index in range(4):
                    directory = raw / f'p{index}'; directory.mkdir()
                    row = {'pair_id': f'p{index}', 'scene': 'FloorPlan1', 'status': 'rejected'}
                    shard.base.build_pair(sim, cfg, {}, {}, {}, {}, f'p{index}', directory, [], attempt_row=row)
                    row['generation_seconds'] = 1.
                    shard.base.write_json(directory / 'attempt.json', row)
                store.index()
                return {'kind': 'pair_v2_shard', 'status': 'completed_search', 'cleanup_errors': []}

            original_store, original_build_function = shard.engine.EvidenceStore, shard.base.build_pair
            with patch.object(shard.base, 'build_pair', side_effect=original_build), \
                 patch.object(shard, '_CLONE_GENERATOR', side_effect=generate):
                result = shard.generator_worker(config, None)
            self.assertEqual([Path(ref['file']).exists() for ref in actual_refs], [True, True, True, False])
            self.assertTrue(all((raw / f'p{i}' / 'thumbnail.jpg').is_file() for i in range(4)))
            compact = json.loads((raw / 'p3' / 'attempt.json').read_text())
            self.assertNotIn('physical_w0', compact)
            self.assertIn('physical_key_numbers', compact)
            self.assertEqual(result['evidence_retention']['physical_rejection_distribution']['edited_position_error'], 4)
            self.assertIs(shard.engine.EvidenceStore, original_store)
            self.assertIs(shard.base.build_pair, original_build_function)

    def test_receive_exception_marks_unreturned_child_raw_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            shard._SESSION = EvidenceSession(scratch); shard._SESSION.begin()
            shard._REPLAY_RAW_COMPLETE = True
            child = SimpleNamespace(result=None, error=None, receive=lambda: (_ for _ in ()).throw(EOFError('lost pipe')),
                                    stop=lambda: [])
            with patch.object(shard, 'Worker', return_value=child):
                with self.assertRaises(EOFError):
                    shard.wait_replay({'deadline': time.time() + 1000}, {}, [], 'A',
                        {'target_type': 'Mug', 'anchor_type': 'Apple'}, scratch / 'replay')
            self.assertFalse(shard._REPLAY_RAW_COMPLETE)

    def test_common_baseline_measurement_is_applied_in_new_entry_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            config = {'runtime': {'scratch': str(scratch)}, 'raw_evidence_dir': str(scratch / 'run'),
                      'enable_candidate_clone': True, 'preflight_version': shard.PREFLIGHT_VERSION,
                      'evidence_version': shard.stable_policy.EVIDENCE_VERSION, 'scene': 'FloorPlan1'}
            check = {'passed': False, 'reasons': ['common_baseline_other_object_position_change_above_0_01m:A']}

            def generate(cfg, connection):
                self.assertIs(shard.base.choose_look, shard.stable_policy.choose_look)
                self.assertIs(shard.original_policy.classify_pair, shard.stable_policy.classify_pair)
                measured = shard.base.physical_measurement({'physical': {'common_baseline_check': check}})
                self.assertFalse(measured['valid_physical'])
                self.assertEqual(measured['reasons'], check['reasons'])
                return {'status': 'completed_search'}
            with patch.object(shard.base, 'physical_measurement', return_value={'valid_physical': True, 'reasons': []}), \
                 patch.object(shard, '_CLONE_GENERATOR', side_effect=generate):
                shard.generator_worker(config, None)
            self.assertIs(shard.original_policy.classify_pair, shard.stable_policy._ORIGINAL_CLASSIFY)

    def test_saved_stable_failure_replaces_old_preflight_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve(); directory = scratch / 'run' / 'construction'; directory.mkdir(parents=True)
            (directory / 'scene_preflight.json').write_text(json.dumps({'preflight_version': shard.PREFLIGHT_VERSION,
                'passed': False, 'reasons': ['initial_not_settled_within_50_steps'], 'error': None}))
            config = {'runtime': {'scratch': str(scratch)}, 'raw_evidence_dir': str(directory.parent),
                      'enable_candidate_clone': True, 'preflight_version': shard.PREFLIGHT_VERSION,
                      'evidence_version': shard.stable_policy.EVIDENCE_VERSION, 'scene': 'FloorPlan1'}
            with patch.object(shard, '_CLONE_GENERATOR', return_value={'status': 'completed_search',
                    'scene_excluded': {'reason': 'original_scene_not_settled_within_1cm'}}):
                result = shard.generator_worker(config, None)
            self.assertEqual(result['scene_excluded']['reason'], 'settled_baseline_window_preflight_failed')
            self.assertEqual(result['scene_excluded']['reasons'], ['initial_not_settled_within_50_steps'])


if __name__ == '__main__':
    unittest.main()
