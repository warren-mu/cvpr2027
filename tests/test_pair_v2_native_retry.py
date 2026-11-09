"""Synthetic native-record/control-flow tests; no Unity/GPU execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from src.envs import pair_v2_native_retry as retry


def vector(x=0.0, y=0.0, z=0.0):
    return {'x': x, 'y': y, 'z': z}


def obj(name, *, pickup=True, receptacle=False):
    return {'name': name, 'objectId': name + '|id', 'objectType': name,
        'pickupable': pickup, 'moveable': False, 'receptacle': receptacle,
        'position': vector(), 'rotation': vector(), 'isMoving': False,
        'parentReceptacles': ['Table|id'] if pickup else None}


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.fail_ledger = False

    def save_json(self, name, value):
        if self.fail_ledger and name.startswith('native_retry_'):
            raise OSError('synthetic ledger failure')
        path = self.directory / name
        content = retry.engine.encoded(value)
        with path.open('xb') as stream:
            stream.write(content)
        return {'file': str(path), 'sha256': hashlib.sha256(content).hexdigest()}


class Sim(retry.RetryPlacementSimulator):
    def __init__(self, directory, outcomes):
        self.store = Store(directory)
        self.scene = 'FloorPlan1'
        self.config = {'preflight_version': retry.PREFLIGHT_VERSION,
                       'preflight_policy': copy.deepcopy(retry.PREFLIGHT_POLICY)}
        self.max_steps, self.time_step, self.step_seconds = 50, .02, .1
        self.action_count, self.frame_count, self.snapshot_count = 0, 0, 0
        self.query_log, self.calls, self.spawn_tokens = [], [], {}
        self.last_action_reference, self.last_capture = None, None
        self.clone_context, self.clone_preparation_attempted = None, False
        self.outcomes, self.failed_mutation, self.fatal = list(outcomes), None, None
        self.invalid_capture, self.corrupt_record = False, False
        self.settle_delta, self.settled = {}, True
        metadata = {'objects': [obj('A'), obj('B'), obj('X'), obj('Table', pickup=False, receptacle=True)],
            'agent': {'position': vector(), 'rotation': vector(), 'cameraHorizon': 0.0, 'isStanding': True},
            'inventoryObjects': [], 'lastActionSuccess': True, 'isSceneAtRest': True}
        self.controller = SimpleNamespace(last_event=SimpleNamespace(metadata=metadata))
        self.scene_preflight = {'passed': True, 'snapshot': self.snapshot('common')}

    def tokens(self, number):
        points = [vector(x=index / 10, y=1.5) for index in range(number)]
        result = []
        for point in points:
            token = {'action': 'GetSpawnCoordinatesAboveReceptacle', 'success': True, 'scene': self.scene,
                'receptacle_name': 'Table', 'receptacle_id_at_query': 'Table|id',
                'selected_position': point, 'coordinates': copy.deepcopy(points),
                'query_reference': {'file': 'synthetic_native_query', 'sha256': 'a' * 64}}
            self.spawn_tokens[hashlib.sha256(retry.engine.encoded(token)).hexdigest()] = copy.deepcopy(token)
            result.append(token)
        return result

    def action(self, action, **arguments):
        index = self.action_count
        self.action_count += 1
        self.calls.append((action, copy.deepcopy(arguments)))
        before = copy.deepcopy(self.controller.last_event.metadata)
        after = copy.deepcopy(before)
        caught = None
        if action == 'PlaceObjectAtPoint':
            outcome = self.outcomes.pop(0)
            after['lastActionSuccess'] = outcome
            if outcome:
                target = next(o for o in after['objects'] if o['objectId'] == arguments['objectId'])
                target['position'] = {**arguments['position'], 'y': .42}
                target['rotation'] = copy.deepcopy(arguments['rotation'])
            else:
                after['errorMessage'] = 'Spawn area not clear (Bowl is in the way)'
                if self.failed_mutation:
                    self.failed_mutation(after)
                caught = self.fatal or RuntimeError('PlaceObjectAtPoint: ' + after['errorMessage'])
        elif action == 'SetObjectPoses':
            for pose in arguments['objectPoses']:
                target = next(o for o in after['objects'] if o['name'] == pose['objectName'])
                target['position'], target['rotation'] = copy.deepcopy(pose['position']), copy.deepcopy(pose['rotation'])
            after['lastActionSuccess'] = True
        elif action == 'AdvancePhysicsStep':
            for name, delta in self.settle_delta.items():
                next(o for o in after['objects'] if o['name'] == name)['position']['x'] += delta
            after['lastActionSuccess'] = True
            after['isSceneAtRest'] = self.settled
        else:
            after['lastActionSuccess'] = True
        event = SimpleNamespace(metadata=after)
        self.controller.last_event = event
        capture = {'valid': not self.invalid_capture, 'black': False,
                   'source_action': action, 'action_index': index}
        self.last_capture = {'evidence': capture}
        record = {'index': index, 'action': action, 'arguments': copy.deepcopy(arguments),
            'before_metadata': before, 'after_metadata': copy.deepcopy(after), 'capture': capture,
            'exception': None if caught is None else f'{type(caught).__name__}: {caught}'}
        self.last_action_reference = self.store.save_json(f'action_{index:06d}.json', record)
        if self.corrupt_record and action == 'PlaceObjectAtPoint':
            self.last_action_reference['sha256'] = 'b' * 64
        self.query_log.append({'action': action, 'reference': copy.deepcopy(self.last_action_reference),
                               'exception': record['exception']})
        if caught is not None:
            raise caught
        return event

    def snapshot(self, key):
        metadata = self.controller.last_event.metadata
        value = {'key': key, 'scene': self.scene, 'objects': copy.deepcopy(metadata['objects']),
            'poses': [{'objectName': o['name'], 'position': copy.deepcopy(o['position']),
                      'rotation': copy.deepcopy(o['rotation'])} for o in metadata['objects'] if o['pickupable']],
            'agent': retry.engine.agent_pose(self.controller.last_event),
            'native_is_scene_at_rest': metadata['isSceneAtRest']}
        ref = self.store.save_json(f'snapshot_{self.snapshot_count:06d}.json', value)
        self.snapshot_count += 1
        return {**value, 'reference': ref}

    def _settle(self, before):
        self.action('AdvancePhysicsStep')
        return {'settled': self.settled, 'observations': []}


def invoke(sim, tokens, key='fixture'):
    return retry.place_native_with_retries(sim, 'X', tokens, rotation=vector(), record_key=key)


class NativeRetryTests(unittest.TestCase):
    def test_unchanged_native_failure_retries_and_returns_actual_second_token_and_event(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory, [False, True])
            tokens = sim.tokens(2)
            original = copy.deepcopy(tokens)
            result = invoke(sim, tokens)
            self.assertEqual(result['status'], 'placed')
            self.assertEqual(result['successful_token'], tokens[1])
            self.assertEqual(result['arguments']['position'], tokens[1]['selected_position'])
            self.assertEqual(result['arguments']['position']['y'], 1.5)
            self.assertIs(result['event'], sim.controller.last_event)
            self.assertEqual(result['failed_point_count'], 1)
            self.assertTrue(result['attempts'][0]['unchanged_check']['unchanged'])
            self.assertEqual(tokens, original)
            saved = json.loads(Path(result['ledger_reference']['file']).read_text())
            self.assertEqual(saved['successful_token'], tokens[1])
            self.assertNotIn('event', saved)
            self.assertFalse(saved['pair_attempt_counter_modified'])

    def test_any_numeric_pose_identity_or_inventory_change_requires_fresh_controller(self):
        def move(meta): meta['objects'][0]['position']['x'] += 1e-12
        def rotate(meta): meta['objects'][0]['rotation']['z'] = 360.0
        def rename_id(meta): meta['objects'][0]['objectId'] += '_changed'
        def remove(meta): meta['objects'].pop(0)
        for mutate in (move, rotate, rename_id, remove):
            with self.subTest(change=mutate.__name__), tempfile.TemporaryDirectory() as directory:
                sim = Sim(directory, [False, True])
                sim.failed_mutation = mutate
                with self.assertRaises(retry.FreshControllerRequired) as caught:
                    invoke(sim, sim.tokens(2))
                self.assertEqual(len(sim.calls), 1)
                self.assertTrue(caught.exception.result['fresh_controller_required'])
                self.assertFalse(caught.exception.result['attempts'][0]['unchanged_check']['unchanged'])

    def test_signed_zero_numeric_equality_is_allowed_and_encoding_difference_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory, [False, True])
            sim.failed_mutation = lambda meta: meta['objects'][0]['rotation'].update(x=-0.0)
            result = invoke(sim, sim.tokens(2))
            check = result['attempts'][0]['unchanged_check']
            self.assertTrue(check['unchanged'])
            self.assertEqual(check['representation_only_names'], ['A'])

    def test_missing_pose_is_unknown_and_never_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory, [False, True])
            sim.failed_mutation = lambda meta: meta['objects'][0]['position'].pop('x')
            with self.assertRaises(retry.FreshControllerRequired) as caught:
                invoke(sim, sim.tokens(2))
            self.assertEqual(len(sim.calls), 1)
            self.assertIsNone(caught.exception.result['attempts'][0]['unchanged_check']['unchanged'])

    def test_forty_point_bound_and_all_failures_are_separate_from_pair_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory, [False] * 41)
            result = invoke(sim, sim.tokens(41))
            self.assertEqual(result['status'], 'exhausted')
            self.assertEqual(result['attempted_point_count'], 40)
            self.assertEqual(result['failed_point_count'], 40)
            self.assertEqual(len(sim.outcomes), 1)
            self.assertFalse(result['pair_attempt_counter_modified'])

    def test_unsampled_or_modified_input_never_dispatches_native_action(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory, [True])
            tokens = sim.tokens(1)
            tokens[0]['selected_position']['y'] += 1
            with self.assertRaises(ValueError):
                invoke(sim, tokens)
            self.assertEqual(sim.calls, [])

    def test_black_and_timeout_propagate_identity_and_never_retry_even_if_ledger_fails(self):
        for error in (retry.engine.BlackFrame({'file': 'black_raw'}), TimeoutError('synthetic deadline')):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                sim = Sim(directory, [False, True])
                sim.fatal, sim.store.fail_ledger = error, True
                with self.assertRaises(type(error)) as caught:
                    invoke(sim, sim.tokens(2))
                self.assertIs(caught.exception, error)
                self.assertEqual(len(sim.calls), 1)
                self.assertIn('ledger_save_error', error.native_retry_result)

    def test_invalid_capture_or_bad_action_hash_is_not_counted_as_safe_native_failure(self):
        for field in ('invalid_capture', 'corrupt_record'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                sim = Sim(directory, [False, True])
                setattr(sim, field, True)
                with self.assertRaises(RuntimeError) as caught:
                    invoke(sim, sim.tokens(2))
                result = caught.exception.native_retry_result
                self.assertEqual(result['failed_point_count'], 0)
                self.assertTrue(result['fresh_controller_required'])
                self.assertEqual(len(sim.calls), 1)

    def test_success_without_ledger_is_not_returned_as_valid_proposal(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory, [True])
            sim.store.fail_ledger = True
            with self.assertRaises(retry.NativePlacementEvidenceError) as caught:
                invoke(sim, sim.tokens(1))
            self.assertEqual(caught.exception.result['status'], 'evidence_failed')


class ConstructionTests(unittest.TestCase):
    def test_actual_success_token_and_pivot_bind_geometry_and_complete_pose_request(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory, [False, True])
            tokens = sim.tokens(2)
            baseline = sim.scene_preflight['snapshot']
            result = sim.construct_world(baseline, 'w0', [{'object_name': 'X', 'spawn': tokens[0], 'alternatives': tokens[1:]}])
            self.assertTrue(result['passed'])
            self.assertEqual(result['successful_tokens_by_name']['X'], tokens[1])
            proof = result['placement_evidence_by_name']['X']
            self.assertEqual(proof['selected_position'], tokens[1]['selected_position'])
            self.assertEqual(proof['geometry']['placement_arguments']['position'], tokens[1]['selected_position'])
            self.assertEqual(proof['requested_position']['y'], .42)
            self.assertEqual(proof['geometry']['placement_reference'], result['retry_records']['X']['action_reference'])
            set_poses = next(args for action, args in sim.calls if action == 'SetObjectPoses')
            self.assertEqual({item['objectName'] for item in set_poses['objectPoses']}, {'A', 'B', 'X'})
            self.assertFalse(set_poses['placeStationary'])
            self.assertEqual([action for action, _ in sim.calls],
                ['PausePhysicsAutoSim', 'PlaceObjectAtPoint', 'PlaceObjectAtPoint', 'SetObjectPoses', 'AdvancePhysicsStep'])

    def test_exhausted_placement_does_not_set_poses_or_settle_or_claim_a_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory, [False])
            token = sim.tokens(1)[0]
            with self.assertRaises(retry.NativePlacementExhausted) as caught:
                sim.construct_world(sim.scene_preflight['snapshot'], 'w0', [{'object_name': 'X', 'spawn': token}])
            self.assertEqual(caught.exception.result['failed_point_count'], 1)
            self.assertEqual([action for action, _ in sim.calls], ['PausePhysicsAutoSim', 'PlaceObjectAtPoint'])

    def test_worlds_keep_same_common_baseline_and_do_not_restore_between_them(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory, [True, True])
            tokens = sim.tokens(2)
            sim.settle_delta = {'A': .006}
            w0 = sim.construct_world(sim.scene_preflight['snapshot'], 'w0', [{'object_name': 'X', 'spawn': tokens[0]}])
            self.assertTrue(w0['passed'])
            w1 = sim.construct_world(w0['snapshot'], 'w1', [{'object_name': 'X', 'spawn': tokens[1]}], require_agent=w0['snapshot']['agent'])
            self.assertFalse(w1['passed'])
            self.assertFalse(w1['physical']['common_baseline_check']['passed'])
            self.assertEqual(w1['physical']['common_baseline_check']['violations_above_1cm'], ['A'])
            self.assertFalse(any(action in ('Reset', 'TeleportFull') for action, _ in sim.calls))

    def test_native_success_still_requires_original_two_cm_settling_and_rest_rules(self):
        for case in ('drift', 'not_settled'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                sim = Sim(directory, [True])
                if case == 'drift':
                    sim.settle_delta = {'X': .021}
                else:
                    sim.settled = False
                token = sim.tokens(1)[0]
                result = sim.construct_world(sim.scene_preflight['snapshot'], 'w0', [{'object_name': 'X', 'spawn': token}])
                self.assertFalse(result['passed'])
                self.assertEqual(result['retry_records']['X']['status'], 'placed')


def canonical_setup(sim):
    before = copy.deepcopy(sim.controller.last_event.metadata['agent'])
    sim.position_agent(before['position'], before['rotation']['y'], before['cameraHorizon'])
    record = {'requested_from_initial_metadata': before,
        'actual_command': copy.deepcopy(sim.agent_setup_command),
        'actual_agent': copy.deepcopy(sim.agent_setup_actual),
        'action_reference': copy.deepcopy(sim.last_action_reference)}
    sim.canonical_initial_setup_reference = sim.store.save_json('initial_canonical_agent_setup.json', record)
    return record


class CanonicalPreflightTests(unittest.TestCase):
    def test_one_real_canonical_setup_keeps_indices_and_full_fifty_step_window(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory, [])
            canonical_setup(sim)
            result = sim.preflight_scene()
            self.assertTrue(result['passed'])
            self.assertEqual(result['initial_agent_setup_reference'], sim.canonical_initial_setup_reference)
            self.assertEqual(len(result['diagnostics']['stability_window']['observations']), 50)
            pause = json.loads(Path(result['diagnostics']['pause_action_reference']['file']).read_text())
            self.assertEqual(pause['index'], 1)
            self.assertEqual(sim.calls[0][0], 'TeleportFull')
            self.assertEqual(sum(action == 'TeleportFull' for action, _ in sim.calls), 1)
            self.assertEqual(sim.action_count, len(sim.calls))

    def test_extra_preflight_action_is_not_hidden_by_resetting_action_count(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory, [])
            canonical_setup(sim)
            sim.action('PausePhysicsAutoSim')
            with self.assertRaises(ValueError):
                sim.preflight_scene()
            self.assertEqual(sim.action_count, 2)

    def test_unbound_initial_setup_or_changed_canonical_command_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory, [])
            record = canonical_setup(sim)
            record['actual_command']['position']['x'] += 1
            sim.canonical_initial_setup_reference = sim.store.save_json('changed_setup.json', record)
            with self.assertRaisesRegex(ValueError, 'canonical action'):
                sim.preflight_scene()
            self.assertEqual(sim.action_count, 1)


if __name__ == '__main__':
    unittest.main()
