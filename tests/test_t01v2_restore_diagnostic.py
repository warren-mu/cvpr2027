"""Synthetic saved-evidence fixtures; no renderer or measured research result."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools import t01v2_restore_diagnostic as diag


def vector(x=0.0, y=0.0, z=0.0):
    return {'x': x, 'y': y, 'z': z}


def snapshot():
    return {'scene': 'FloorPlan3', 'worker_pid': 100, 'unity_pid': 101,
        'objects': [{'name': 'Mug_fixture', 'objectType': 'Mug', 'pickupable': True, 'moveable': False,
                     'position': vector(), 'rotation': vector(), 'isMoving': False}],
        'agent': {'position': vector(y=.9), 'rotation': vector(), 'horizon': -0.0, 'standing': True},
        'agent_setup_command': None, 'camera': {'cameraPosition': vector(y=1.5), 'fov': 100,
                                              'screenWidth': 255, 'screenHeight': 255}}


def reference(name):
    return {'file': name + '.json', 'sha256': '0' * 64}


def failed_setup():
    baseline, restored = snapshot(), snapshot()
    restored['agent']['horizon'] = 0.0
    restored['agent']['rotation']['x'] = -0.0
    native_ref = reference('native')
    restored.update(last_action='TeleportFull', last_action_reference=native_ref,
                    capture_reference={'files': {'metadata': reference('capture_metadata')}})
    expected = {**copy.deepcopy(baseline['agent']), 'forceAction': False}
    native = {'action': 'TeleportFull', 'arguments': expected, 'exception': None,
        'after_metadata': {'objects': copy.deepcopy(restored['objects']), 'lastActionSuccess': True,
            'agent': {'position': restored['agent']['position'], 'rotation': restored['agent']['rotation'],
                      'cameraHorizon': restored['agent']['horizon'], 'isStanding': True}},
        'capture': copy.deepcopy(restored['capture_reference'])}
    measurement = diag.engine.restoration_state(baseline, restored['objects'], restored['agent'])
    measurement.update(restored_snapshot_reference=reference('restored'),
        executed_agent_setup_arguments=expected, restore_max_abs_diff=3, restore_rgb_exact=False)
    record = {'baseline_reference': reference('baseline'), 'restoration': measurement}
    values = {'baseline.json': baseline, 'restored.json': restored, 'native.json': native}
    return record, values


class ScalarTests(unittest.TestCase):
    def test_signed_zero_is_numeric_equal_but_encoded_unequal(self):
        result = diag.compare_scalar(-0.0, 0.0)
        self.assertEqual(result['difference'], 'signed_zero_representation')
        self.assertTrue(result['numeric_equal'])
        self.assertFalse(result['encoded_equal'])
        self.assertEqual(result['numeric_delta'], 0.0)
        self.assertNotEqual(result['requested']['float_hex'], result['realized']['float_hex'])

    def test_tiny_y_difference_stays_numeric_difference_without_tolerance(self):
        first, second = .9009998440742493, .9009997844696045
        result = diag.compare_scalar(first, second)
        self.assertEqual(result['requested']['value'], first)
        self.assertEqual(result['realized']['value'], second)
        self.assertEqual(result['numeric_delta'], second - first)
        self.assertEqual(result['difference'], 'numeric_difference')
        self.assertFalse(result['numeric_equal'])

    def test_int_float_representation_and_missing_are_distinct(self):
        result = diag.compare_scalar(0, 0.0)
        self.assertEqual(result['difference'], 'numeric_equal_different_representation')
        unknown = diag.compare_scalar(0.0)
        self.assertEqual(unknown['difference'], 'unknown_missing_field')
        self.assertIsNone(unknown['numeric_equal'])
        self.assertIsNone(unknown['numeric_delta'])


class StateTests(unittest.TestCase):
    def test_signed_zero_only_explains_agent_gate_not_object_motion(self):
        old, new = snapshot(), snapshot()
        new['agent']['horizon'] = 0.0
        new['agent']['rotation']['x'] = -0.0
        result = diag.compare_snapshots(old, new)
        self.assertTrue(result['agent_difference_only_signed_zero'])
        self.assertFalse(result['agent_has_numeric_difference'])
        checked = result['unchanged_restoration_rule_recomputed']
        self.assertTrue(checked['restored_positions_within_1cm'])
        self.assertFalse(checked['agent_pose_exact'])
        self.assertFalse(checked['restore_deterministic'])

    def test_movable_one_cm_boundary_unchanged_and_rotation_is_only_reported(self):
        old, new = snapshot(), snapshot()
        new['objects'][0]['position']['x'] = .01
        new['objects'][0]['rotation']['y'] = 360.0
        result = diag.compare_snapshots(old, new)
        self.assertTrue(result['unchanged_restoration_rule_recomputed']['restore_deterministic'])
        self.assertEqual(result['movable_or_pickupable_positions'][0]['axes']['rotation.y']['numeric_delta'], 360)
        new['objects'][0]['position']['x'] = .0101
        result = diag.compare_snapshots(old, new)
        self.assertEqual(result['movable_position_violations'], ['Mug_fixture'])

    def test_missing_agent_axis_remains_unknown_and_missing_object_remains_unknown(self):
        old, new = snapshot(), snapshot()
        del new['agent']['rotation']['z']
        new['objects'] = []
        result = diag.compare_snapshots(old, new)
        self.assertIsNone(result['unchanged_restoration_rule_recomputed'])
        self.assertIsNone(result['agent_difference_only_signed_zero'])
        self.assertEqual(result['unknown_movable_positions'], ['Mug_fixture'])
        self.assertIsNone(result['maximum_observed_movable_position_delta_m'])

    def test_authenticated_snapshot_and_native_action_binding_preserve_rgb_record(self):
        record, values = failed_setup()
        result = diag.inspect_setup(record, 'FloorPlan3', lambda ref: values[ref['file']])
        self.assertEqual(result['binding_errors'], [])
        self.assertTrue(result['agent_difference_only_signed_zero'])
        self.assertEqual(result['recorded_rgb_diagnostics'], {'restore_max_abs_diff': 3, 'restore_rgb_exact': False})
        self.assertEqual(result['executed_agent_setup_arguments']['horizon'], -0.0)

    def test_wrong_native_result_or_recorded_flag_is_reported_as_conflict(self):
        record, values = failed_setup()
        values['native.json']['arguments']['position']['x'] = 99
        record['restoration']['agent_pose_exact'] = True
        result = diag.inspect_setup(record, 'FloorPlan3', lambda ref: values[ref['file']])
        self.assertTrue(any('canonical TeleportFull' in value for value in result['binding_errors']))
        self.assertTrue(any('agent_pose_exact' in value for value in result['binding_errors']))


class ProvenanceTests(unittest.TestCase):
    def test_only_hash_verified_json_is_archived_and_original_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, archive = root / 'scratch/raw', root / 'logs/evidence'
            raw.mkdir(parents=True); archive.mkdir(parents=True)
            source = raw / 'setup.json'
            content = b'{"value": -0.0}\n'
            source.write_bytes(content)
            ref = {'file': str(source), 'sha256': hashlib.sha256(content).hexdigest()}
            evidence = diag.Evidence(raw, archive, root=root)
            self.assertEqual(evidence.read(ref), {'value': -0.0})
            self.assertEqual(source.read_bytes(), content)
            self.assertEqual((root / evidence.references[0]['file']).read_bytes(), content)
            source.write_bytes(b'{"value": 0.0}\n')
            with self.assertRaises(ValueError):
                evidence.read(ref)

    def test_no_bound_final_index_remains_unknown_without_directory_scan(self):
        data = {'config': {'scene': 'FloorPlan212'}, '_meta': {'exit_code': 1},
                'metrics': {'status': 'worker_failed', 'controller_visits': [{'controller_id': 'controller0'}]}}
        with patch.object(diag, 'verified_passed_rows', return_value=({'eligible': False}, None, [])):
            result = diag.scene_diagnostic(data, unittest.mock.Mock())
        self.assertIsNone(result['proposal_setup_failure'])
        self.assertIsNone(result['journal_audit']['passed'])
        self.assertTrue(any('No hashed final raw index' in text for text in result['errors']))

    def test_journal_failure_is_retained_while_independent_hashed_restore_can_be_diagnosed(self):
        record, values = failed_setup()
        index_ref = reference('/private/raw/evidence_index')
        setup_ref = reference('/private/raw/proposal_setup_000007')
        values.update({index_ref['file']: {'files': [setup_ref]}, setup_ref['file']: record})
        evidence = unittest.mock.Mock()
        evidence.read.side_effect = lambda ref: values[ref['file']]
        data = {'config': {'scene': 'FloorPlan3'}, '_meta': {'exit_code': 1},
            'metrics': {'status': 'worker_failed', 'controller_visits': [
                {'controller_id': 'controller0', 'raw_index': index_ref}]}}
        with patch.object(diag, 'verified_passed_rows', side_effect=ValueError('original counter mismatch')):
            result = diag.scene_diagnostic(data, evidence)
        self.assertFalse(result['journal_audit']['passed'])
        self.assertTrue(result['proposal_setup_failure']['agent_difference_only_signed_zero'])
        self.assertTrue(result['errors'])


if __name__ == '__main__':
    unittest.main()
