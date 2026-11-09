"""Native-prefix fixtures for safe failed-point retries; no renderer calls."""
import copy
import importlib.util
from pathlib import Path
import unittest

from src.envs import pair_v2_marked_verify as verify

spec = importlib.util.spec_from_file_location('isolated_verify_fixtures', Path(__file__).with_name('test_pair_v2_isolated_verify.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def fixture(failures=0):
    args, values = fixtures.fixture()
    prefix = values['/synthetic/prefix.json']
    prefix['safe_retry_version'] = verify.NATIVE_RETRY_VERSION
    final = copy.deepcopy(values['/synthetic/action_000005.json'])
    before = copy.deepcopy(final['before_metadata'])
    refs = prefix['action_references'][:5]
    for index in range(failures):
        after = copy.deepcopy(before)
        after.update(sequenceId=5 + index, lastActionSuccess=False, errorMessage='Spawn area not clear')
        error = 'RuntimeError: PlaceObjectAtPoint: Spawn area not clear'
        action = {'index': 5 + index, 'action': 'PlaceObjectAtPoint',
                  'arguments': {**copy.deepcopy(final['arguments']), 'position': fixtures.xyz(5.0 + index)},
                  'before_metadata': before, 'after_metadata': after, 'exception': error}
        path = '/synthetic/failed_' + str(index) + '.json'
        values[path] = action
        refs.append({'action': 'PlaceObjectAtPoint', 'reference': {'file': path, 'sha256': 'fixture'}, 'exception': error})
        before = copy.deepcopy(after)
    final['index'] = 5 + failures
    final['before_metadata'] = before
    final['after_metadata']['sequenceId'] = 5 + failures
    values['/synthetic/action_000005.json'] = final
    refs.append(prefix['action_references'][-1])
    prefix['action_references'] = refs
    return args, values


class MarkedVerifyTests(unittest.TestCase):
    def test_zero_and_thirty_nine_safe_failures_then_native_success(self):
        for count in (0, 1, 39):
            args, _ = fixture(count)
            result = verify.verify_attempt_setup(*args)
            self.assertTrue(result['passed'])
            self.assertEqual(result['safe_failed_native_points'], count)

    def test_forty_failures_plus_success_exceeds_forty_point_cap(self):
        args, _ = fixture(40)
        with self.assertRaises(ValueError):
            verify.verify_attempt_setup(*args)

    def test_failed_point_cannot_move_any_object_or_change_identity(self):
        for field, value in (('position', {'x': 1e-12, 'y': 0.0, 'z': 0.0}), ('objectId', 'changed')):
            args, values = fixture(1)
            values['/synthetic/failed_0.json']['after_metadata']['objects'][0][field] = value
            with self.assertRaisesRegex(ValueError, 'preserve'):
                verify.verify_attempt_setup(*args)

    def test_failed_point_cannot_hide_transport_or_black_capture_exception(self):
        for error in ('BlackFrame: black', 'TimeoutError: deadline', None):
            args, values = fixture(1)
            values['/synthetic/failed_0.json']['exception'] = error
            with self.assertRaises(ValueError):
                verify.verify_attempt_setup(*args)

    def test_failure_cannot_change_target_or_rotation(self):
        for key, value in (('objectId', 'A_native'), ('rotation', fixtures.xyz(90.0))):
            args, values = fixture(1)
            values['/synthetic/failed_0.json']['arguments'][key] = value
            with self.assertRaises(ValueError):
                verify.verify_attempt_setup(*args)

    def test_signed_zero_restore_not_silently_relaxed(self):
        args, values = fixture(1)
        values['/synthetic/ready.json']['agent']['horizon'] = -0.0
        with self.assertRaisesRegex(ValueError, 'exact-agent'):
            verify.verify_attempt_setup(*args)

    def test_old_proof_cannot_claim_retry_validation(self):
        args, _ = fixtures.fixture()
        with self.assertRaises(ValueError):
            verify.verify_attempt_setup(*args)


if __name__ == '__main__':
    unittest.main()
