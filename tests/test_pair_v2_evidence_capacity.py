"""Supplement-3 CPU fixtures; root commits before formal run_logged execution."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.envs.pair_v2_evidence import EvidenceSession, SearchEvidenceSession, SearchBufferedStore


class SearchEvidenceCapacityTests(unittest.TestCase):
    def test_common_cap_keeps_runtime_bytes_and_later_search_attempts_continue(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            session = SearchEvidenceSession(scratch, raw_budget_bytes=5)
            store = session.store(scratch / 'controller', 100)
            saved = store.save('saved.bin', b'12345')
            resident = store.save('common.bin', b'needed later')
            self.assertTrue(Path(saved['file']).is_file())
            self.assertFalse(Path(resident['file']).exists())
            self.assertTrue(session.raw_saving_stopped)
            for index in range(3):
                session.begin()
                current = store.save(f'attempt{index}.bin', b'measured')
                with session.reading():
                    self.assertEqual(Path(resident['file']).read_bytes(), b'needed later')
                    self.assertEqual(Path(current['file']).read_bytes(), b'measured')
                result = session.finish(True)
                self.assertFalse(result['evidence_complete_for_acceptance'])
                self.assertTrue(result['search_may_continue'])
                self.assertTrue(result['candidate_requires_evidence_incomplete_label'])
                self.assertFalse(Path(current['file']).exists())
                # Compact output is independent of the raw payload writer/cap.
                (scratch / f'compact{index}.json').write_text(json.dumps({'evidence_incomplete': True}))
            self.assertEqual(len(session.cap_events), 1)
            self.assertEqual(session.data(resident['file']), b'needed later')

    def test_complete_package_gets_certificate_only_after_materialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            session = SearchEvidenceSession(scratch, raw_budget_bytes=100)
            store = session.store(scratch / 'controller', 100)
            common = store.save('common.bin', b'common')
            session.begin()
            refs = [store.save(f'actual{i}.bin', value) for i, value in enumerate((b'RGB', b'metadata'))]
            self.assertTrue(session.retention_plan()['can_materialize_complete'])
            self.assertTrue(all(not Path(ref['file']).exists() for ref in refs))
            result = session.finish(True)
            self.assertTrue(result['evidence_complete_for_acceptance'])
            for ref in [common, *refs]:
                self.assertEqual(hashlib.sha256(Path(ref['file']).read_bytes()).hexdigest(), ref['sha256'])

    def test_package_over_store_cap_is_not_partially_written_or_thrown(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            session = SearchEvidenceSession(scratch, raw_budget_bytes=100)
            store = session.store(scratch / 'controller', 10)
            session.begin()
            refs = [store.save(f'frame{i}.bin', b'123456') for i in range(2)]
            plan = session.retention_plan()
            self.assertFalse(plan['can_materialize_complete'])
            self.assertEqual(plan['per_store_capacity_exceeded'], [str(store.directory)])
            result = session.finish(True)
            self.assertFalse(result['full_raw_retained'])
            self.assertTrue(all(not Path(ref['file']).exists() for ref in refs))
            self.assertEqual(store.bytes, 0)

    def test_replay_bundle_over_cap_can_be_read_and_rejected_without_stopping_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            child = EvidenceSession(scratch, buffer_all=True)
            store = child.store(scratch / 'replay', 100)
            ref = store.save('frame.bin', b'x' * 80)
            parent = SearchEvidenceSession(scratch, raw_budget_bytes=10)
            parent.begin(); parent.absorb(child.bundle())
            with parent.reading():
                self.assertEqual(Path(ref['file']).read_bytes(), b'x' * 80)
            self.assertFalse(parent.finish(True)['evidence_complete_for_acceptance'])
            parent.begin()
            self.assertTrue(parent.active)
            self.assertFalse(parent.finish(False)['evidence_complete_for_acceptance'])

    def test_release_common_ram_authenticates_closure_and_never_deletes_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            session = SearchEvidenceSession(scratch, raw_budget_bytes=5)
            store = session.store(scratch / 'controller', 10000)
            saved = store.save('existing.bin', b'12345')
            leaf = store.save('source.png', b'actual source encoded bytes')
            source = store.save_json('snapshot.json', {'image': leaf})
            unused = store.save('unused.bin', b'unreferenced')
            session.release_runtime_references([source])
            self.assertIsNone(session.data(unused['file']))
            with session.reading():
                self.assertEqual(json.loads(Path(source['file']).read_text())['image'], leaf)
                self.assertEqual(Path(leaf['file']).read_bytes(), b'actual source encoded bytes')
            result = session.release_runtime_references([])
            self.assertEqual(result['remaining_common_references'], 0)
            self.assertEqual(result['disk_files_deleted'], 0)
            self.assertEqual(Path(saved['file']).read_bytes(), b'12345')

    def test_changed_runtime_hash_fails_closed_instead_of_silently_releasing(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            session = SearchEvidenceSession(scratch, raw_budget_bytes=0)
            store = session.store(scratch / 'controller', 100)
            ref = store.save('runtime.bin', b'actual')
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                session.release_runtime_references([{**ref, 'sha256': '0' * 64}])
            self.assertEqual(session.data(ref['file']), b'actual')

    def test_reserved_terminal_png_and_json_are_real_without_claiming_full_attempt(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            session = SearchEvidenceSession(scratch, raw_budget_bytes=10000, terminal_reserve_bytes=5000)
            store = session.store(scratch / 'controller', 10000)
            store.save('common.bin', b'x' * 5000)
            session.begin()
            unrelated = store.save('large_attempt.bin', b'x' * 6000)
            image = io.BytesIO(); Image.new('RGB', (255, 255), (0, 0, 0)).save(image, format='PNG')
            png = store.save('black.png', image.getvalue())
            metadata = store.save_json('metadata.json', {'source': 'actual fixture metadata'})
            invalid = store.save_json('invalid_rgb.json', {'black': True, 'files': {'rgb': png, 'metadata': metadata}})
            session.mark_terminal(invalid)
            result = session.finish(True)
            self.assertTrue(result['terminal_evidence']['preserved'])
            self.assertFalse(result['full_raw_retained'])
            self.assertFalse(result['evidence_complete_for_acceptance'])
            self.assertFalse(Path(unrelated['file']).exists())
            for ref in (png, metadata, invalid):
                self.assertEqual(hashlib.sha256(Path(ref['file']).read_bytes()).hexdigest(), ref['sha256'])
            self.assertEqual(result['terminal_evidence']['packet_bytes'], sum(ref['bytes'] for ref in (png, metadata, invalid)))
            self.assertLessEqual(result['terminal_evidence']['packet_bytes'], 5000)
            with Image.open(png['file']) as loaded:
                self.assertEqual(loaded.size, (255, 255))
                self.assertEqual(loaded.getextrema(), ((0, 0), (0, 0), (0, 0)))
            self.assertLessEqual(session.status()['stored_payload_bytes'], 10000)

    def test_terminal_packet_over_total_cap_remains_unproven(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            session = SearchEvidenceSession(scratch, raw_budget_bytes=4, terminal_reserve_bytes=4)
            store = session.store(scratch / 'controller', 100)
            session.begin(); ref = store.save('invalid.json', b'{"black":true}')
            session.mark_terminal(ref)
            result = session.finish(True)
            self.assertFalse(result['terminal_evidence']['preserved'])
            self.assertFalse(result['evidence_complete_for_acceptance'])
            self.assertFalse(Path(ref['file']).exists())

    def test_terminal_reserve_zero_never_uses_other_available_raw_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            session = SearchEvidenceSession(scratch, raw_budget_bytes=10000, terminal_reserve_bytes=0)
            store = session.store(scratch / 'controller', 10000)
            session.begin(); ref = store.save('terminal.bin', b'actual terminal proof')
            result = session.minimal_black_flush(ref)
            self.assertFalse(result['preserved'])
            self.assertEqual(result['capacity_plan']['terminal_packet_limit_bytes'], 0)
            self.assertFalse(Path(ref['file']).exists())

    def test_external_child_packet_is_bounded_even_without_parent_new_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            external = scratch / 'child_terminal.bin'
            content = b'0123456789abcdef'; external.write_bytes(content)
            ref = {'file': str(external), 'sha256': hashlib.sha256(content).hexdigest()}
            parent = SearchEvidenceSession(scratch, raw_budget_bytes=10000, terminal_reserve_bytes=8)
            report = parent.minimal_black_flush(ref)
            self.assertFalse(report['preserved'])
            self.assertEqual(report['capacity_plan']['required_payload_bytes'], 0)
            self.assertEqual(report['packet_bytes'], len(content))
            self.assertEqual(external.read_bytes(), content)

    def test_path_identity_errors_and_old_cap_behavior_are_not_relaxed(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            session = SearchEvidenceSession(scratch, raw_budget_bytes=0)
            store = session.store(scratch / 'new', 0)
            store.save('allowed.bin', b'any size remains memory')
            with self.assertRaises(ValueError):
                store.save('../escape', b'x')
            with self.assertRaises(FileExistsError):
                store.save('allowed.bin', b'overwrite')
            old = EvidenceSession(scratch, buffer_all=True).store(scratch / 'old', 0)
            with self.assertRaises(ValueError):
                old.save('old.bin', b'x')

    def test_io_error_retains_partial_files_and_cannot_return_complete_certificate(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            session = SearchEvidenceSession(scratch, raw_budget_bytes=100)
            store = session.store(scratch / 'controller', 100)
            session.begin(); first = store.save('first', b'a'); second = store.save('second', b'b')
            original = SearchBufferedStore.persist

            def persist(actual, path, content):
                if path.name == 'second':
                    raise OSError('synthetic IO failure')
                return original(actual, path, content)
            with patch.object(SearchBufferedStore, 'persist', persist):
                with self.assertRaisesRegex(OSError, 'synthetic IO failure'):
                    session.finish(True)
            self.assertEqual(Path(first['file']).read_bytes(), b'a')
            self.assertFalse(Path(second['file']).exists())


if __name__ == '__main__':
    unittest.main()
