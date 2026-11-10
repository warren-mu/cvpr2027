"""Call-local proof-byte reuse; synthetic science boundaries remain explicit."""
from contextlib import ExitStack, contextmanager
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.envs import pair_table_checks as checks
from src.tools import smoke_e2e_pilot_dataset as dataset
from src.tools import t02_cache_check as cache
from src.tools import t02_evidence_read_session as reuse
# Bind the adjacent fixture explicitly; Hopper may contain an unrelated tests package.
_fixture_spec = importlib.util.spec_from_file_location(
    'pilot_dataset_fresh_reuse_fixtures', Path(__file__).with_name('test_smoke_e2e_pilot_dataset.py'))
fixtures = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(fixtures)


class PilotFreshReadReuseTests(unittest.TestCase):
    def exercise(self, *, change_after_read=False, native_error=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, source, table, observed, plan, actual = fixtures.FormalBatchTests().fixture(root)
            row = data['metrics']['worker_runs'][0]
            phases, calls, seals = [], [], []
            original_scope = reuse.authenticated_read_reuse
            original_seal = reuse.EvidenceBytes.verify_unchanged
            original_reader = cache.verified_read
            original_native_reader = checks.verified_read
            @contextmanager
            def scope(consumer):
                self.assertIs(consumer, cache)
                self.assertFalse(reuse._INSTALL_LOCK.locked(), 'load_exports/fresh scopes overlap')
                number = len(phases)
                phases.append(number)
                with original_scope(consumer) as session:
                    calls.append(('enter', number))
                    yield session
                calls.append(('exit', number))
            def seal(session):
                seals.append(set(session.accessed))
                return original_seal(session)
            def load_original(paths, scratch):
                self.assertTrue(reuse._INSTALL_LOCK.locked())
                self.assertEqual(paths, data['config']['source'])
                calls.append(('original_load_exports', 0))
                # A real original byte read within the existing loader's scope.
                cache.read_json_ref(data['metrics']['sampling_plan'], Path(data['config']['scratch_directory']))
                return ({'dataset': table}, {'dataset': {'cache_evidence_roots': []}},
                        {'dataset': {'passed': True}}, [], data['metrics']['population_ledger'])
            def samples(*args, **kwargs):
                self.assertTrue(reuse._INSTALL_LOCK.locked())
                self.assertEqual(kwargs, {'use_dataset_key': True})
                self.assertEqual(args[4], ['state'])
                calls.append(('original_load_samples', 1))
                return observed
            def native(*args, **kwargs):
                self.assertTrue(reuse._INSTALL_LOCK.locked())
                self.assertEqual(args, ([table], observed))
                calls.append(('original_native_authentication', 1))
                if native_error:
                    raise ValueError('unchanged native check rejects')
                if change_after_read:
                    # This original report has already been read and compared;
                    # only final rehash can catch this later mutation.
                    Path(row['report_reference']['file']).write_bytes(b'{}')
                return {('authenticated',)}
            def validate(samples_arg, **kwargs):
                self.assertTrue(reuse._INSTALL_LOCK.locked())
                self.assertEqual(samples_arg, observed)
                self.assertEqual(kwargs, {'population': [('dataset', 'state')], 'sampling_plan': plan})
                calls.append(('original_validate_cache_sample', 1))
                return copy.deepcopy(actual)
            def authorize(*args, **kwargs):
                self.assertTrue(reuse._INSTALL_LOCK.locked())
                self.assertEqual(args, ([table],))
                self.assertEqual(kwargs['authenticated_cache_source_ids'], {('authenticated',)})
                calls.append(('original_authorize_tables', 1))
            with ExitStack() as stack:
                stack.enter_context(patch.object(dataset, 'ROOT', root))
                stack.enter_context(patch.object(dataset, 'audited_result', side_effect=lambda path:
                    (data, source) if path.name == 'cache.json' else ({}, data['metrics']['gate'])))
                stack.enter_context(patch.object(cache, 'gate_runtime'))
                # Real cache.load_exports + original reuse manager, rather than
                # mocking the outer loader and hiding accidental nesting.
                stack.enter_context(patch.object(cache, '_load_exports', side_effect=load_original))
                stack.enter_context(patch.object(reuse, 'authenticated_read_reuse', scope))
                stack.enter_context(patch.object(reuse.EvidenceBytes, 'verify_unchanged', seal))
                stack.enter_context(patch('src.envs.pair_table_cache.sample_state_keys', return_value=plan))
                stack.enter_context(patch.object(cache, 'load_samples', side_effect=samples))
                stack.enter_context(patch.object(checks, 'authenticate_cache_sources', side_effect=native))
                stack.enter_context(patch('src.envs.pair_table_cache.validate_cache_sample', side_effect=validate))
                stack.enter_context(patch.object(checks, 'authorize_tables', side_effect=authorize))
                if change_after_read or native_error:
                    expected = 'Evidence changed before cache-load completion' if change_after_read else 'unchanged native check rejects'
                    with self.assertRaisesRegex(ValueError, expected):
                        dataset.authenticate_cache_batch(source)
                else:
                    tables, _, proof = dataset.authenticate_cache_batch(source)
                    self.assertEqual(set(tables), {'dataset'})
                    self.assertEqual(proof['source'], source)
            self.assertFalse(reuse._INSTALL_LOCK.locked())
            self.assertIs(cache.verified_read, original_reader)
            self.assertIs(checks.verified_read, original_native_reader)
            self.assertEqual(phases, [0, 1])
            self.assertLess(calls.index(('exit', 0)), calls.index(('enter', 1)))
            if not native_error:
                self.assertIn(('original_authorize_tables', 1), calls)
                self.assertEqual(len(seals), 2)
                self.assertIn((Path(row['report_reference']['file']).resolve(), row['report_reference']['sha256']), seals[1])
            return calls

    def test_original_checks_execute_in_two_sequential_scopes_and_finally_rehash(self):
        calls = self.exercise()
        self.assertEqual(calls, [('enter', 0), ('original_load_exports', 0), ('exit', 0),
            ('enter', 1), ('original_load_samples', 1), ('original_native_authentication', 1),
            ('original_validate_cache_sample', 1), ('original_authorize_tables', 1), ('exit', 1)])

    def test_changed_fresh_evidence_after_native_read_fails_final_rehash(self):
        self.exercise(change_after_read=True)

    def test_real_native_failure_is_not_replaced_by_certificate_and_restores_readers(self):
        self.exercise(native_error=True)


if __name__ == '__main__':
    unittest.main()
