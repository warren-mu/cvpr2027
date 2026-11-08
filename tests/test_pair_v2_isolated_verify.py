"""Synthetic native-schema proof fixtures; no simulator or numeric experiment."""
import copy
import hashlib
import unittest

from src.envs import pair_v2_isolated_verify as verify
from src.envs.pair_v2_isolated_shard import ATTEMPT_SETUP_VERSION, ATTEMPT_SETUP_POLICY


def xyz(x=0.0):
    return {'x': x, 'y': 0.0, 'z': 0.0}


def fixture():
    values = {}
    def save(name, value):
        path = '/synthetic/' + name
        values[path] = copy.deepcopy(value)
        return {'file': path, 'sha256': hashlib.sha256(verify.engine.encoded(value)).hexdigest()}
    def read(ref):
        # Authentication is the caller's contract. These fixtures deliberately
        # vary authenticated content to test independent semantic bindings.
        return copy.deepcopy(values[ref['file']])
    objects = [{'name': name, 'objectId': name + '_native', 'objectType': category,
                'moveable': False, 'pickupable': True, 'position': xyz(float(index)), 'rotation': xyz()}
               for index, (name, category) in enumerate((('A', 'Mug'), ('B', 'Mug'), ('X', 'Book')))]
    agent = {'position': xyz(), 'rotation': xyz(), 'horizon': 0.0, 'standing': True}
    baseline = {'scene': 'FloorPlan1', 'key': 'common', 'objects': copy.deepcopy(objects),
        'poses': [{'objectName': obj['name'], 'position': obj['position'], 'rotation': obj['rotation']} for obj in objects],
        'agent': copy.deepcopy(agent), 'agent_setup_command': {**copy.deepcopy(agent), 'forceAction': False},
        'openness': {}, 'worker_pid': 100, 'unity_pid': 101}
    baseline_ref = save('baseline.json', baseline)
    metadata = {'objects': copy.deepcopy(objects), 'agent': {'position': xyz(), 'rotation': xyz(),
                 'cameraHorizon': 0.0, 'isStanding': True}, 'lastActionSuccess': True}
    refs = []
    def action(name, arguments, after_agent=None, placed=False):
        nonlocal metadata
        before = copy.deepcopy(metadata)
        metadata = copy.deepcopy(metadata)
        if after_agent is not None:
            metadata['agent'] = copy.deepcopy(after_agent)
        if placed:
            metadata['objects'][2]['position'] = xyz(.3)
        metadata['sequenceId'] = len(refs)
        value = {'index': len(refs), 'action': name, 'arguments': copy.deepcopy(arguments),
                 'before_metadata': before, 'after_metadata': copy.deepcopy(metadata), 'exception': None}
        reference = save(f'action_{len(refs):06d}.json', value)
        refs.append({'action': name, 'reference': reference, 'exception': None})
        return value
    action('PausePhysicsAutoSim', {})
    action('SetObjectPoses', {'objectPoses': baseline['poses'], 'placeStationary': True})
    action('TeleportFull', baseline['agent_setup_command'])
    restored = {**copy.deepcopy(baseline), 'last_action_reference': refs[-1]['reference']}
    restored_ref = save('restored.json', restored)
    ready_ref = save('ready.json', restored)
    measured = verify.engine.restoration_state(baseline, objects, agent)
    row = {'pair_id': 'floorplan1_00000', 'scene': 'FloorPlan1',
        'view': {'position': xyz(), 'yaw': 45, 'horizon': 0.0}, 'identities': {'anchor_name': 'X'},
        'attempt_setup_version': ATTEMPT_SETUP_VERSION, 'attempt_setup_policy': copy.deepcopy(ATTEMPT_SETUP_POLICY)}
    record = {'version': ATTEMPT_SETUP_VERSION, 'policy': copy.deepcopy(ATTEMPT_SETUP_POLICY),
        'pair_id': row['pair_id'], 'scene': row['scene'], 'phase': 'before_v0_and_w0',
        'baseline_reference': baseline_ref, 'status': 'passed', 'setup_passed': True,
        'restoration': {**copy.deepcopy(measured), 'restored_snapshot_reference': restored_ref,
                        'executed_agent_setup_arguments': baseline['agent_setup_command']},
        'restored_snapshot_reference': ready_ref, 'independent_state': measured,
        'action_references': copy.deepcopy(refs), 'error': None}
    row['attempt_setup'] = copy.deepcopy(record)
    row['attempt_setup_reference'] = save('setup.json', record)
    actual_view = {'position': xyz(), 'rotation': {'x': 0.0, 'y': 45.0, 'z': 0.0},
                   'cameraHorizon': 0.0, 'isStanding': True}
    action('TeleportFull', {'position': xyz(), 'rotation': {'x': 0, 'y': 45, 'z': 0},
                           'horizon': 0.0, 'standing': True, 'forceAction': False}, actual_view)
    action('PausePhysicsAutoSim', {})
    placement_args = {'objectId': 'X_native', 'position': xyz(.3), 'rotation': xyz(), 'forceKinematic': True}
    action('PlaceObjectAtPoint', placement_args, placed=True)
    prefix = {'version': ATTEMPT_SETUP_VERSION, 'pair_id': row['pair_id'], 'scene': row['scene'],
              'setup_reference': row['attempt_setup_reference'], 'action_references': copy.deepcopy(refs),
              'first_place_reference': refs[-1]['reference']}
    row['attempt_setup_native_prefix_reference'] = save('prefix.json', prefix)
    view_pose = verify.agent_pose({'agent': actual_view})
    w0 = {**copy.deepcopy(baseline), 'agent': view_pose, 'objects': copy.deepcopy(metadata['objects'])}
    w1 = copy.deepcopy(w0)
    w1['objects'][2]['position'] = xyz(.7)
    constructions = [{'before_objects': copy.deepcopy(objects), 'agent_before': copy.deepcopy(view_pose),
        'placement_evidence_by_name': {'X': {'geometry': {'placement_arguments': placement_args,
                                                       'placement_reference': refs[-1]['reference']}}}},
        {'before_objects': copy.deepcopy(w0['objects'])}]
    return (row, [w0, w1], constructions, objects, read), values


class IsolatedVerifyTests(unittest.TestCase):
    def test_native_schema_complete_chain_passes(self):
        arguments, _ = fixture()
        result = verify.verify_attempt_setup(*arguments)
        self.assertTrue(result['passed'])
        self.assertEqual(result['native_prefix_action_count'], 6)
        self.assertTrue(result['agent_pose_exact'])

    def test_full_setup_record_must_match_immutable_reference(self):
        arguments, _ = fixture()
        arguments[0]['attempt_setup']['setup_passed'] = False
        with self.assertRaisesRegex(ValueError, 'authenticate'):
            verify.verify_attempt_setup(*arguments)

    def test_wrong_common_baseline_or_world_process_rejected(self):
        for mode in ('common', 'process', 'clone'):
            arguments, _ = fixture()
            if mode == 'common':
                arguments[3][0]['position']['x'] = 7
            elif mode == 'process':
                arguments[1][1]['unity_pid'] = 999
            else:
                arguments[1][1]['clone_context'] = {'unrelated': True}
            with self.assertRaises(ValueError):
                verify.verify_attempt_setup(*arguments)

    def test_signed_zero_restore_camera_difference_remains_failure(self):
        arguments, values = fixture()
        values['/synthetic/ready.json']['agent']['horizon'] = -0.0
        with self.assertRaisesRegex(ValueError, 'exact-agent'):
            verify.verify_attempt_setup(*arguments)

    def test_missing_or_inserted_native_action_not_admitted(self):
        for mode in ('missing', 'inserted'):
            arguments, values = fixture()
            refs = values['/synthetic/prefix.json']['action_references']
            if mode == 'missing':
                refs.pop(3)
            else:
                refs.insert(3, copy.deepcopy(refs[3]))
            with self.assertRaisesRegex(ValueError, 'omitted or inserted'):
                verify.verify_attempt_setup(*arguments)

    def test_wrong_before_metadata_cannot_splice_other_attempt(self):
        arguments, values = fixture()
        values['/synthetic/action_000003.json']['before_metadata']['agent']['position']['x'] = 5
        with self.assertRaisesRegex(ValueError, 'continuous'):
            verify.verify_attempt_setup(*arguments)

    def test_current_place_objects_must_still_match_restored_common(self):
        arguments, values = fixture()
        # Preserve the adjacency chain while moving X during the view setup.
        for number, part in ((3, 'after_metadata'), (4, 'before_metadata'),
                             (4, 'after_metadata'), (5, 'before_metadata')):
            values[f'/synthetic/action_{number:06d}.json'][part]['objects'][2]['position']['x'] += .02
        with self.assertRaisesRegex(ValueError, 'unchanged bound'):
            verify.verify_attempt_setup(*arguments)

    def test_first_place_reference_and_native_anchor_identity_bound(self):
        for mode in ('reference', 'object'):
            arguments, values = fixture()
            if mode == 'reference':
                values['/synthetic/prefix.json']['first_place_reference'] = {'file': 'other', 'sha256': 'other'}
            else:
                values['/synthetic/action_000005.json']['arguments']['objectId'] = 'wrong'
                arguments[2][0]['placement_evidence_by_name']['X']['geometry']['placement_arguments']['objectId'] = 'wrong'
            with self.assertRaises(ValueError):
                verify.verify_attempt_setup(*arguments)

    def test_position_agent_args_and_paired_camera_cannot_change(self):
        for mode in ('args', 'world'):
            arguments, values = fixture()
            if mode == 'args':
                values['/synthetic/action_000003.json']['arguments']['horizon'] = 30
            else:
                arguments[1][1]['agent']['horizon'] = 30
            with self.assertRaises(ValueError):
                verify.verify_attempt_setup(*arguments)

    def test_paired_camera_retains_existing_signed_zero_mapping_equality(self):
        arguments, _ = fixture()
        arguments[1][1]['agent']['horizon'] = -0.0
        arguments[1][0]['agent']['rotation']['x'] = -0.0
        arguments[2][0]['agent_before']['position']['z'] = -0.0
        self.assertTrue(verify.verify_attempt_setup(*arguments)['passed'])

    def test_incomplete_setup_or_wrong_policy_cannot_be_claimed_passed(self):
        for mode in ('policy', 'failed_action'):
            arguments, values = fixture()
            if mode == 'policy':
                arguments[0]['attempt_setup_policy']['signed_zero'] = 'ignore'
            else:
                values['/synthetic/action_000005.json']['after_metadata']['lastActionSuccess'] = False
            with self.assertRaises(ValueError):
                verify.verify_attempt_setup(*arguments)


if __name__ == '__main__':
    unittest.main()
