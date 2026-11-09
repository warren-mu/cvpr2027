"""Real file/SHA fixtures for scoped byte reuse; no rendered acceptance claims."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.tools import merge_pair_v2 as merge
from src.tools import t02_cache_check as cache
from src.tools import t02_evidence_read_session as reuse
from src.envs import pair_table_checks as checks


class EvidenceReuseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.path = self.root / 'native.json'
        self.path.write_bytes(b'{"value":[1,2],"seen":false}\n')
        self.ref = self.reference()

    def reference(self):
        return {'file': str(self.path), 'sha256': hashlib.sha256(self.path.read_bytes()).hexdigest()}

    def test_original_sha_reader_once_and_mutable_json_not_shared(self):
        original = merge.reference_path
        with patch.object(merge, 'reference_path', wraps=original) as checker:
            with reuse.authenticated_read_reuse(cache) as session:
                first = merge.read_reference(self.ref, root=self.root, within=self.root)
                first['value'].append(99)
                second = merge.read_reference(self.ref, root=self.root, within=self.root)
                self.assertEqual(second, {'value': [1, 2], 'seen': False})
                self.assertEqual(checker.call_count, 1)
                self.assertIsInstance(cache.verified_read(self.ref, [self.root]), bytes)
                self.assertEqual(session.retained, self.path.stat().st_size)
        self.assertTrue(session.closed)
        self.assertEqual(session.retained, 0)
        self.assertFalse(session.cache or session.accessed)

    def test_same_path_different_digest_never_reuses_success(self):
        with reuse.authenticated_read_reuse(cache):
            merge.read_reference(self.ref, within=self.root)
            with self.assertRaisesRegex(ValueError, 'SHA-mismatched'):
                merge.read_reference({**self.ref, 'sha256': '0' * 64}, within=self.root)

    def test_roots_and_identity_checked_on_every_hit(self):
        elsewhere = self.root / 'other'
        elsewhere.mkdir()
        with reuse.authenticated_read_reuse(cache):
            merge.read_reference(self.ref, within=self.root)
            cache.verified_read(self.ref, [self.root])
            with self.assertRaisesRegex(ValueError, 'escaped'):
                merge.read_reference(self.ref, within=elsewhere)
            with self.assertRaisesRegex(ValueError, 'escaped'):
                cache.verified_read(self.ref, [elsewhere])
            for bad in ({'file': str(self.path)}, {**self.ref, 'sha256': 'not-a-digest'}):
                with self.assertRaises(ValueError):
                    merge.read_reference(bad, within=self.root)

    def test_changed_file_on_hit_is_rechecked_not_reused(self):
        with self.assertRaisesRegex(ValueError, 'SHA-mismatched'):
            with reuse.authenticated_read_reuse(cache):
                merge.read_reference(self.ref, within=self.root)
                self.path.write_bytes(b'{"value":[3,4],"seen":false}\n')
                merge.read_reference(self.ref, within=self.root)

    def test_final_hash_catches_mutation_without_another_read(self):
        originals = merge.read_reference, checks.verified_read, cache.verified_read
        with self.assertRaisesRegex(ValueError, 'before cache-load completion'):
            with reuse.authenticated_read_reuse(cache) as session:
                cache.verified_read(self.ref, [self.root])
                self.path.write_bytes(b'changed after the last verifier')
        self.assertEqual((merge.read_reference, checks.verified_read, cache.verified_read), originals)
        self.assertTrue(session.closed)

    def test_final_hash_is_not_only_a_stat_fingerprint_check(self):
        fixed = reuse.fingerprint(self.path)
        with patch.object(reuse, 'fingerprint', return_value=fixed):
            with self.assertRaisesRegex(ValueError, 'before cache-load completion'):
                with reuse.authenticated_read_reuse(cache):
                    cache.verified_read(self.ref, [self.root])
                    self.path.write_bytes(b'{"value":[8,9],"seen":false}\n')

    def test_removed_file_cannot_return_a_certificate(self):
        with self.assertRaises(OSError):
            with reuse.authenticated_read_reuse(cache):
                cache.verified_read(self.ref, [self.root])
                self.path.unlink()

    def test_evicted_and_large_files_remain_in_final_authentication_set(self):
        session = reuse.EvidenceBytes(max_bytes=4)
        self.addCleanup(session.close)
        session.read(self.path, self.ref['sha256'], self.path.read_bytes)
        self.assertFalse(session.cache)
        self.assertEqual(session.retained, 0)
        self.assertIn((self.path, self.ref['sha256']), session.accessed)
        self.path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'before cache-load completion'):
            session.verify_unchanged()

    def test_byte_budget_evicts_without_dropping_the_final_recheck(self):
        session = reuse.EvidenceBytes(max_bytes=3)
        self.addCleanup(session.close)
        a, b = self.root / 'a', self.root / 'b'
        a.write_bytes(b'aaa'); b.write_bytes(b'bbb')
        for path in (a, b):
            session.read(path, hashlib.sha256(path.read_bytes()).hexdigest(), path.read_bytes)
        self.assertEqual(session.retained, 3)
        self.assertEqual(len(session.cache), 1)
        self.assertEqual(len(session.accessed), 2)
        a.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'before cache-load completion'):
            session.verify_unchanged()

    def test_reader_exception_and_nested_scope_restore_original_entrypoints(self):
        originals = merge.read_reference, checks.verified_read, cache.verified_read
        with self.assertRaisesRegex(RuntimeError, 'original science failure'):
            with reuse.authenticated_read_reuse(cache):
                with self.assertRaisesRegex(RuntimeError, 'Overlapping'):
                    with reuse.authenticated_read_reuse(cache):
                        self.fail('nested scope accepted')
                merge.read_reference(self.ref, within=self.root)
                raise RuntimeError('original science failure')
        self.assertEqual((merge.read_reference, checks.verified_read, cache.verified_read), originals)
        with reuse.authenticated_read_reuse(cache):
            self.assertEqual(merge.read_reference(self.ref)['value'], [1, 2])

    def test_wrong_consumer_fails_without_leaving_patches_or_lock(self):
        originals = merge.read_reference, checks.verified_read
        with self.assertRaisesRegex(ValueError, 'original verified'):
            with reuse.authenticated_read_reuse(SimpleNamespace(verified_read=lambda *_: b'fake')):
                self.fail('unrelated consumer accepted')
        self.assertEqual((merge.read_reference, checks.verified_read), originals)
        with reuse.authenticated_read_reuse(cache):
            cache.verified_read(self.ref, [self.root])

    def test_other_threads_keep_the_original_readers(self):
        original = checks.verified_read
        calls, results = [], []
        def counted(ref, roots):
            calls.append(threading.get_ident())
            return original(ref, roots)
        with patch.object(checks, 'verified_read', counted), patch.object(cache, 'verified_read', counted):
            with reuse.authenticated_read_reuse(cache):
                cache.verified_read(self.ref, [self.root])
                worker = threading.Thread(target=lambda: results.append(cache.verified_read(self.ref, [self.root])))
                worker.start(); worker.join()
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0], calls[1])
        self.assertEqual(results, [self.path.read_bytes()])

    def test_load_exports_runs_original_body_and_seals_before_return(self):
        output = object()
        originals = merge.read_reference, checks.verified_read, cache.verified_read
        def body(paths, scratch):
            self.assertEqual((paths, scratch), (['original-source'], 'original-scratch'))
            self.assertIsNot(merge.read_reference, originals[0])
            self.assertEqual(merge.read_reference(self.ref)['value'], [1, 2])
            return output
        with patch.object(cache, '_load_exports', side_effect=body) as complete_body:
            self.assertIs(cache.load_exports(['original-source'], 'original-scratch'), output)
            complete_body.assert_called_once_with(['original-source'], 'original-scratch')
        self.assertEqual((merge.read_reference, checks.verified_read, cache.verified_read), originals)
        def changed(*args):
            cache.verified_read(self.ref, [self.root])
            self.path.write_bytes(b'changed before returning')
            return output
        with patch.object(cache, '_load_exports', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'before cache-load completion'):
                cache.load_exports(['original-source'], 'original-scratch')
        self.assertEqual((merge.read_reference, checks.verified_read, cache.verified_read), originals)


if __name__ == '__main__':
    unittest.main()
