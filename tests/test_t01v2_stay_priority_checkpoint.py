"""CPU adapter fixtures; mocked scientific boundaries make no GPU claims.

The final test also reuses the indexed native geometry fixture, so the new
source identity cannot replace the existing used-point/retry proof.
"""
import copy
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.envs import pair_v2_stay_priority as priority
from src.envs import pair_v2_stay_priority_policy as policy
from src.envs import pair_v2_fov as fov
from src.tools import t01v2_stay_priority_review_checkpoint as check
from src.tools import t01v2_flip_ranked_review_checkpoint as ranked


def row_fixture(mode='flip'):
    return {**priority.fields(), 'producer_module': policy.PRODUCER,
        'flip_unit_version': policy.VERSION, 'flip_unit_policy': copy.deepcopy(policy.POLICY),
        'mode': mode, 'kind': 'look_' + mode, 'stay_priority_selection': None}


def bindings():
    original = ranked.original
    return (ranked.policy, original.policy, check.merge.load_source,
        check.merge.check_pair, check.merge.shard_producer, ranked.roles.verify_proposal_role,
        fov.declared_fields, original.original.collect_sources,
        original.controller.policy, original.association.policy)


class StayPriorityCheckpointTests(unittest.TestCase):
    def test_actual_context_installs_explicit_policy_and_restores_on_error(self):
        before = bindings()
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            with check.source_support():
                self.assertIs(ranked.policy, policy)
                self.assertIs(ranked.original.policy, policy)
                self.assertIs(fov.declared_fields, policy.declared_fields)
                raise RuntimeError('fixture failure')
        self.assertEqual(bindings(), before)

    def test_both_modes_retain_complete_checker_inputs_and_failure(self):
        # Only the delegated scientific boundary and detailed selection
        # validator are mocked; the adapter's exact source/field guards run.
        @contextmanager
        def support():
            self.assertIs(ranked.policy, policy)
            yield

        kwargs = {'raw_root': '/synthetic/raw', 'field_of_view': 60,
            'frozen_render_version': 'fixture-frozen', 'candidate_policy': 'fixture-candidates'}
        for mode in ('flip', 'stay'):
            for failure in (False, True):
                with self.subTest(mode=mode, failure=failure):
                    row = row_fixture(mode); before = copy.deepcopy(row)
                    def scientific(actual, path, **received):
                        self.assertIs(actual, row); self.assertEqual(path, '/synthetic/pair.json')
                        self.assertEqual(received, kwargs)
                        if failure:
                            raise ValueError('original scientific gate failed')
                        return {'unchanged_full_check': True}
                    with patch.object(ranked, 'source_support', side_effect=support), \
                            patch.object(check.merge, 'check_pair', side_effect=scientific) as delegated, \
                            patch.object(priority, 'validate_selection') as selection:
                        with check.source_support():
                            if failure:
                                with self.assertRaisesRegex(ValueError, 'original scientific gate failed'):
                                    check.merge.check_pair(row, '/synthetic/pair.json', **kwargs)
                            else:
                                self.assertEqual(check.merge.check_pair(row, '/synthetic/pair.json', **kwargs),
                                                 {'unchanged_full_check': True})
                        selection.assert_called_once_with(row)
                        delegated.assert_called_once()
                    self.assertEqual(row, before)

    def test_wrong_producer_version_or_missing_order_fields_fail_before_science(self):
        @contextmanager
        def support():
            yield
        for change in ('producer', 'version', 'policy', 'priority'):
            row = row_fixture()
            if change == 'producer': row['producer_module'] = 'src.envs.pair_v2_flip_ranked_shard'
            elif change == 'version': row['flip_unit_version'] = 'historical-version'
            elif change == 'policy': row['flip_unit_policy'] = {}
            else: row.pop(priority.FIELDS[0])
            with self.subTest(change=change), \
                    patch.object(ranked, 'source_support', side_effect=support), \
                    patch.object(check.merge, 'check_pair') as delegated:
                with check.source_support(), self.assertRaises(ValueError):
                    check.merge.check_pair(row, '/synthetic/pair.json', raw_root='/synthetic/raw')
                delegated.assert_not_called()

    def test_selection_failure_cannot_be_bypassed_by_scientific_success(self):
        @contextmanager
        def support():
            yield
        with patch.object(ranked, 'source_support', side_effect=support), \
                patch.object(check.merge, 'check_pair', return_value={'passed': True}) as delegated, \
                patch.object(priority, 'validate_selection', side_effect=ValueError('invalid proposal diagnostic')):
            with check.source_support(), self.assertRaisesRegex(ValueError, 'invalid proposal diagnostic'):
                check.merge.check_pair(row_fixture('stay'), '/synthetic/pair.json')
            delegated.assert_not_called()

    def test_main_uses_original_controller_collector_and_restores_ownership(self):
        before = bindings(); original = ranked.original
        def main(argv):
            self.assertEqual(argv, ['--name', 'fixture'])
            self.assertIs(original.original.collect_sources, original.controller.collect_sources)
            self.assertIs(original.controller.policy, policy)
            self.assertIs(original.association.policy, policy)
            self.assertIs(ranked.policy, policy)
            raise RuntimeError('publication fixture')
        with patch.object(original.original, 'main', side_effect=main):
            with self.assertRaisesRegex(RuntimeError, 'publication fixture'):
                check.main(['--name', 'fixture'])
        self.assertEqual(bindings(), before)

    def test_new_identity_still_requires_original_byte_backed_geometry_and_old_scope_stays_strict(self):
        spec = importlib.util.spec_from_file_location('stay_priority_geometry_fixture',
            Path(__file__).with_name('test_t01v2_flip_geometry_checkpoint.py'))
        fixture_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture_module)
        with tempfile.TemporaryDirectory() as directory:
            fixture = fixture_module.Fixture(directory)
            historical = copy.deepcopy(fixture.row)
            fixture.row.update(row_fixture())
            with self.assertRaisesRegex(ValueError, 'explicit new v2 Look Flip source'):
                fixture.verify()
            with check.source_support():
                proof = fixture.verify()
                self.assertEqual(proof['verified_native_attempts_by_world'], {'w0': 2, 'w1': 2})
                current = fixture.row
                fixture.row = historical
                with self.assertRaisesRegex(ValueError, 'explicit new v2 Look Flip source'):
                    fixture.verify()
                fixture.row = current
                fixture.row['flip_geometry_native_guards']['w1'][0]['opposite_reference'] = {}
                with self.assertRaises(ValueError):
                    fixture.verify()
            fixture.row = historical
            self.assertEqual(fixture.verify()['verified_native_attempts_by_world'], {'w0': 2, 'w1': 2})


if __name__ == '__main__':
    unittest.main()
