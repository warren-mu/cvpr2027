"""Original-byte storage fixtures; no simulator or scientific result claims."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.envs import pair_v2_construction_metadata as metadata
from src.envs.pair_v2_evidence import SearchEvidenceSession, SearchBufferedStore, encoded
from src.envs import pair_v2_search_shard as search
from src.envs import pair_v2_marked_shard as marked


def session(root, *, raw=10, total=100000, single=50000):
    return metadata.ConstructionMetadataSession(root, construction_root=Path(root)/'scene',
        raw_budget_bytes=raw, terminal_reserve_bytes=0,
        metadata_budget_bytes=total, metadata_file_limit_bytes=single)


def construction(value, index=0):
    return value.store(value.construction_root/f'marked_controller_{index:03d}'/'construction', 100000)


class ConstructionMetadataTests(unittest.TestCase):
    def test_cap_and_controller_retirement_preserve_original_identity_files_not_missing_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            current = session(directory); store = construction(current)
            payload = store.save('baseline.json', b'12345')
            preflight_value = {'passed': True, 'snapshot_reference': payload, 'unicode': '原始', 'zero': -0.0}
            preflight = store.save_json('scene_preflight.json', preflight_value)
            current.begin(); store.save('retained-scientific.bin', b'abc')
            self.assertTrue(current.finish(True)['evidence_complete_for_acceptance'])
            late = store.save('omitted-scientific.bin', b'xxxx')
            self.assertTrue(current.raw_saving_stopped)
            self.assertFalse(Path(late['file']).exists())
            index = store.index()
            original_preflight = encoded(preflight_value)
            self.assertEqual(Path(preflight['file']).read_bytes(), original_preflight)
            self.assertEqual(preflight['sha256'], hashlib.sha256(original_preflight).hexdigest())
            index_value = json.loads(Path(index['file']).read_bytes())
            self.assertIn(late, index_value['files'])
            self.assertEqual(index_value['index_scope'],
                'references may remain memory-only; use retention certificate before admission')
            self.assertEqual(index_value['raw_bytes'], 8)
            self.assertEqual(current.status()['stored_payload_bytes'], 8)
            current.release_runtime_references([])
            self.assertIsNone(current.data(late['file']))
            self.assertEqual(Path(preflight['file']).read_bytes(), original_preflight)
            self.assertEqual(Path(index['file']).read_bytes(), encoded(index_value))
            # A later noncontributing Controller can still identify its real
            # preflight/final index; this does not flush its missing evidence.
            final = construction(current, 1)
            last_preflight = final.save_json('scene_preflight.json', {'passed': True, 'late': True})
            last_index = final.index()
            current.release_runtime_references([])
            self.assertTrue(Path(last_preflight['file']).is_file())
            self.assertTrue(Path(last_index['file']).is_file())
            current.begin(); missing = final.save('new-attempt.bin', b'xx')
            retained = current.finish(True)
            self.assertFalse(retained['evidence_complete_for_acceptance'])
            self.assertFalse(retained['full_raw_retained'])
            self.assertFalse(Path(missing['file']).exists())
            self.assertEqual(current.status()['stored_payload_bytes'], 8)
            reserve = current.status()['construction_metadata_retention']
            self.assertEqual(reserve['written_bytes'], sum(item['bytes'] for item in reserve['files']))
            self.assertEqual(reserve['errors'], [])
            self.assertFalse(reserve['scientific_payload_materialization_or_acceptance_granted'])

    def test_original_session_still_keeps_post_cap_index_memory_only(self):
        with tempfile.TemporaryDirectory() as directory:
            old = SearchEvidenceSession(directory, raw_budget_bytes=1)
            store = old.store(Path(directory)/'legacy', 10000)
            store.save('cap.bin', b'xx')
            index = store.index()
            self.assertFalse(Path(index['file']).exists())
            self.assertIsNotNone(old.data(index['file']))
            self.assertNotIn('construction_metadata_retention', old.status())

    def test_whitelist_does_not_fund_replay_or_arbitrary_json_after_raw_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            current = session(directory, raw=0)
            store = construction(current)
            for name in ('snapshot.json', 'initial_canonical_agent_setup.json', 'action_000000.json'):
                ref = store.save_json(name, {'original': True})
                self.assertFalse(Path(ref['file']).exists())
            witness = current.store(Path(directory)/'scene'/'replay_w0', 10000)
            self.assertIs(type(witness), SearchBufferedStore)
            ref = witness.save_json('scene_preflight.json', {'original': True})
            self.assertFalse(Path(ref['file']).exists())
            self.assertEqual(current.metadata_written_bytes, 0)
            with self.assertRaises(ValueError):
                store.save('../scene_preflight.json', b'xx')

    def test_shared_scene_and_single_file_limits_fail_without_missing_reference_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            current = session(directory, total=5, single=4)
            first, second = construction(current), construction(current, 1)
            first.save('scene_preflight.json', b'123')
            with self.assertRaises(metadata.MetadataRetentionFailure):
                second.save('scene_preflight.json', b'456')
            self.assertFalse((second.directory/'scene_preflight.json').exists())
            self.assertEqual(current.metadata_claimed_bytes, 3)
            self.assertEqual(current.metadata_written_bytes, 3)
            self.assertTrue(current.metadata_errors)
        with tempfile.TemporaryDirectory() as directory:
            current = session(directory, total=10, single=4); store = construction(current)
            with self.assertRaises(metadata.MetadataRetentionFailure):
                store.save('scene_preflight.json', b'12345')
            self.assertFalse((store.directory/'scene_preflight.json').exists())
            self.assertFalse(current.resident)

    def test_reserve_never_overwrites_existing_bytes_or_reclaims_failed_io(self):
        with tempfile.TemporaryDirectory() as directory:
            current = session(directory); store = construction(current)
            first = store.save('scene_preflight.json', b'original')
            with self.assertRaises(FileExistsError): store.save('scene_preflight.json', b'replacement')
            self.assertEqual(Path(first['file']).read_bytes(), b'original')
            target = store.directory/'evidence_index.json'
            original_open = Path.open
            def opened(path, *args, **kwargs):
                if path == target and (args[0] if args else kwargs.get('mode')) == 'xb':
                    raise OSError('synthetic write failure')
                return original_open(path, *args, **kwargs)
            before = current.metadata_claimed_bytes
            with patch.object(Path, 'open', opened), self.assertRaises(metadata.MetadataRetentionFailure):
                store.save('evidence_index.json', b'index')
            self.assertEqual(current.metadata_claimed_bytes, before + len(b'index'))
            self.assertEqual(current.metadata_written_bytes, len(b'original'))
            self.assertTrue(current.metadata_errors)
            self.assertFalse(target.exists())
            with self.assertRaises(FileExistsError): store.save('evidence_index.json', b'index')

    def test_active_attempt_cannot_use_common_identity_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            current = session(directory); store = construction(current)
            current.begin()
            with self.assertRaises(metadata.MetadataRetentionFailure):
                store.save_json('scene_preflight.json', {'not_common_phase': True})
            self.assertFalse((store.directory/'scene_preflight.json').exists())
            self.assertEqual(current.metadata_written_bytes, 0)
            current.finish(False)

    def test_factory_opt_in_and_declared_engineering_limits_leave_legacy_replay_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            old = search.session_factory(directory)
            current = search.session_factory(directory, construction_root=Path(directory)/'future')
            self.assertIs(type(old), SearchEvidenceSession)
            self.assertIsInstance(current, metadata.ConstructionMetadataSession)
            self.assertEqual(current.raw_budget_bytes, old.raw_budget_bytes)
            self.assertEqual(current.terminal_reserve_bytes, old.terminal_reserve_bytes)
            self.assertEqual(current.metadata_file_limit_bytes, 16*1024*1024)
            self.assertEqual(current.metadata_budget_bytes, 64*1024*1024)
            for key, value in metadata.fields().items():
                self.assertEqual(marked.tags()[key], value)
            with self.assertRaises(ValueError):
                search.session_factory(directory, construction_root=Path(directory)/'bad', buffer_all=True)


if __name__ == '__main__':
    unittest.main()
