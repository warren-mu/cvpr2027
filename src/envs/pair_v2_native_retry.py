"""Opt-in native placement retries for supplement 6; no historical mutation.

Only a recorded native failure with unchanged metadata identities and poses may
try the next literal native point. Successful coordinates remain proposals:
the existing complete-pose, settling, parent and common-baseline checks follow.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import re

from . import pair_v2_simulator as engine
from .pair_v2_stable import (StableCloneSimulator, common_baseline_check, PREFLIGHT_VERSION,
                             PREFLIGHT_POLICY, position_step, exceeded)

NATIVE_RETRY_VERSION = 'native_placement_unchanged_state_retry_v1'
FROZEN_RENDER_VERSION = 'w0_nonanchor_frozen_render_v1'
FROZEN_RENDER_POLICY = {
    'version': FROZEN_RENDER_VERSION,
    'physics': 'unchanged two native placements and settling checks; retain both original physical snapshots',
    'requested_render_worlds': 'complete physical w0 pose list for both worlds; replace only X position and rotation with physically settled w1 X',
    'render_actions': 'PausePhysicsAutoSim; SetObjectPoses(placeStationary=True), PhysicsSyncTransforms, Pass for w0 then w1; no agent action or physics advance between captures',
    'readback': 'all requested editable poses checked against actual metadata; exact equality and maximum position/rotation differences recorded',
    'acceptance': 'unchanged restoration position<=1cm/exact agent and physical/image/witness criteria; no new exact-pose acceptance threshold',
    'actual_cross_world': 'all nonanchor native position/rotation differences recorded without replacing measurements by zero',
}


def frozen_pose_lists(worlds, anchor):
    if len(worlds) != 2 or worlds[0]['scene'] != worlds[1]['scene']:
        raise ValueError('Frozen render needs two original physical worlds in one scene')
    poses = [engine.complete_pose_list(world['objects'], world['poses']) for world in worlds]
    source = {item['objectName']: item for item in poses[1]}
    if set(source) != {item['objectName'] for item in poses[0]} or anchor not in source:
        raise ValueError('Frozen render physical inventories differ or anchor is absent')
    first = copy.deepcopy(poses[0])
    second = [copy.deepcopy(source[anchor] if item['objectName'] == anchor else item) for item in first]
    return [first, second]


def pose_readback(poses, objects):
    """Read-only diagnostics; Euler representation differences remain visible."""
    actual = engine.objects_by_name(objects)
    requested = {item['objectName']: item for item in poses}
    if len(requested) != len(poses) or not set(requested) <= set(actual):
        raise ValueError('Pose readback has duplicate or missing identities')
    rows = {}
    for name, desired in requested.items():
        observed = actual[name]
        rotation = {axis: abs(desired['rotation'][axis] - observed['rotation'][axis]) for axis in 'xyz'}
        rows[name] = {'position_delta_m': engine.distance(desired['position'], observed['position']),
            'rotation_component_abs_diff_degrees': rotation,
            'rotation_shortest_component_diff_degrees': {a: min(v % 360, 360 - v % 360) for a, v in rotation.items()},
            'position_exact': desired['position'] == observed['position'],
            'rotation_exact': desired['rotation'] == observed['rotation']}
    return {'objects': rows, 'all_pose_components_exact': all(v['position_exact'] and v['rotation_exact'] for v in rows.values()),
        'max_position_delta_m': max((v['position_delta_m'] for v in rows.values()), default=0.),
        'max_rotation_component_abs_diff_degrees': max((n for v in rows.values() for n in v['rotation_component_abs_diff_degrees'].values()), default=0.),
        'max_rotation_shortest_component_diff_degrees': max((n for v in rows.values() for n in v['rotation_shortest_component_diff_degrees'].values()), default=0.),
        'scope': 'actual metadata versus requested pose values; diagnostic, not a new acceptance threshold'}


def frozen_requested_snapshot(worlds, poses, index):
    """Explicit desired-state template used only by the existing restore check."""
    template = copy.deepcopy(worlds[0])
    template['poses'] = copy.deepcopy(poses[index])
    values = {row['objectName']: row for row in poses[index]}
    for obj in template['objects']:
        if obj['name'] in values:
            for key in ('position', 'rotation'):
                obj[key] = copy.deepcopy(values[obj['name']][key])
    template['serialization_scope'] = 'derived requested render template only; not an observed native snapshot'
    template.pop('reference', None)
    return template


def record_restore_readback(sim, snapshot, measured):
    readback = pose_readback(snapshot['poses'], sim.controller.last_event.metadata['objects'])
    proof = {'version': FROZEN_RENDER_VERSION, 'requested_snapshot_reference': snapshot['reference'],
             'restored_snapshot_reference': measured['restored_snapshot_reference'], 'readback': readback}
    measured['frozen_pose_readback_reference'] = sim.store.save_json(f'frozen_restore_readback_{sim.action_count:06d}.json', proof)
    measured['frozen_pose_readback'] = readback
    return measured
NATIVE_RETRY_POLICY = {
    'version': NATIVE_RETRY_VERSION, 'maximum_points_per_placement': 40,
    'point_arguments': 'literal authenticated native spawn coordinates; no height or pivot offset added',
    'retry': 'recorded lastActionSuccess=false and all metadata object identities/positions/rotations unchanged',
    'pose_equality': 'exact numeric equality, no tolerance; signed-zero/type encoding differences recorded separately',
    'changed_or_unknown_state': 'no same-Controller retry; fresh_controller_required',
    'other_exceptions': 'propagate unchanged, including black frame and deadline; preserve partial ledger',
    'attempt_count': 'failed native points are not pair attempts; caller records their separate count',
    'validity': 'native success is a proposal only; unchanged downstream physical/image/witness rules remain mandatory',
}


class FreshControllerRequired(RuntimeError):
    def __init__(self, result):
        self.result = self.native_retry_result = result
        super().__init__('fresh_controller_required: failed native placement changed or lost object state')


class NativePlacementExhausted(RuntimeError):
    def __init__(self, result):
        self.result = self.native_retry_result = result
        super().__init__('Native placement exhausted its declared point list without success')


class NativePlacementEvidenceError(RuntimeError):
    def __init__(self, message, result):
        self.result = self.native_retry_result = result
        super().__init__(message)


def object_pose_state(metadata):
    objects = engine.objects_by_name(metadata['objects'])
    if not objects:
        raise ValueError('Cannot verify an empty object inventory')
    result, ids = {}, set()
    for name, obj in sorted(objects.items()):
        object_id, category = obj.get('objectId'), obj.get('objectType')
        if not isinstance(object_id, str) or not object_id or object_id in ids or not isinstance(category, str):
            raise ValueError('Missing or duplicate native object identity')
        ids.add(object_id)
        row = {key: copy.deepcopy(obj[key]) for key in ('name', 'objectId', 'objectType', 'position', 'rotation')}
        for key in ('position', 'rotation'):
            if any(type(row[key].get(axis)) not in (int, float) or not math.isfinite(row[key][axis]) for axis in 'xyz'):
                raise ValueError('Missing/nonfinite object pose: ' + name)
        result[name] = row
    return result


def unchanged_objects(before, after):
    """Numeric equality treats signed zero equally; it never permits an epsilon."""
    old, new = object_pose_state(before), object_pose_state(after)
    shared = sorted(set(old) & set(new))
    identity = [name for name in shared if any(old[name][key] != new[name][key]
                                              for key in ('name', 'objectId', 'objectType'))]
    pose = [name for name in shared if any(old[name][key][axis] != new[name][key][axis]
                                          for key in ('position', 'rotation') for axis in 'xyz')]
    representation = [name for name in shared if old[name] == new[name]
                      and engine.encoded(old[name]) != engine.encoded(new[name])]
    inventory = set(old) == set(new)
    return {'verified': True, 'unchanged': inventory and not identity and not pose,
        'inventory_exact': inventory, 'identity_changed_names': identity, 'pose_changed_names': pose,
        'missing_names': sorted(set(old) - set(new)), 'added_names': sorted(set(new) - set(old)),
        'representation_only_names': representation,
        'before_object_poses': old, 'after_object_poses': new,
        'scope': 'all objects in native metadata; velocities and unreported Unity state are not serialized'}


def _native_record(sim, before, after, arguments, action_index, query_count):
    reference = copy.deepcopy(sim.last_action_reference)
    if (len(sim.query_log) != query_count + 1 or sim.query_log[-1].get('action') != 'PlaceObjectAtPoint'
            or sim.query_log[-1].get('reference') != reference):
        raise ValueError('Native placement lacks its unique recorded action')
    path = Path(reference['file']).resolve()
    if path.parent != Path(sim.store.directory).resolve():
        raise ValueError('Native placement action escaped its own evidence store')
    raw = path.read_bytes()  # Existing evidence-session read view also supports buffered records.
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('Native placement action hash mismatch')
    record = json.loads(raw)
    if (record.get('index') != action_index or record.get('action') != 'PlaceObjectAtPoint'
            or engine.encoded(record.get('arguments')) != engine.encoded(arguments)
            or engine.encoded(record.get('before_metadata')) != engine.encoded(before)
            or engine.encoded(record.get('after_metadata')) != engine.encoded(after)
            or (record.get('capture') or {}).get('valid') is not True
            or (record.get('capture') or {}).get('black') is not False
            or (record.get('capture') or {}).get('source_action') != 'PlaceObjectAtPoint'
            or (record.get('capture') or {}).get('action_index') != action_index):
        raise ValueError('Native placement result is not its authenticated fresh action/capture')
    return record, reference


def canonical_initial_setup(sim):
    """Authenticate the sole allowed preflight-prefix action; never reset indices."""
    reference = copy.deepcopy(getattr(sim, 'canonical_initial_setup_reference', None))
    if sim.action_count != 1 or len(sim.query_log) != 1 or not isinstance(reference, dict):
        raise ValueError('Preflight may follow exactly one recorded canonical initial TeleportFull')
    def read(ref):
        path = Path(ref['file']).resolve()
        if path.parent != Path(sim.store.directory).resolve():
            raise ValueError('Initial setup proof escaped its own evidence store')
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != ref['sha256']:
            raise ValueError('Initial setup proof hash mismatch')
        return json.loads(content)
    record = read(reference)
    native_ref = record['action_reference']
    native = read(native_ref)
    initial = native['before_metadata']['agent']
    expected = {'position': copy.deepcopy(initial['position']),
        'rotation': {'x': 0, 'y': initial['rotation']['y'], 'z': 0},
        'horizon': initial['cameraHorizon'], 'standing': True, 'forceAction': False}
    current = sim.controller.last_event
    actual = engine.agent_pose(current)
    if (native_ref != sim.last_action_reference or sim.query_log[0]['reference'] != native_ref
            or sim.query_log[0]['action'] != 'TeleportFull'
            or native.get('index') != 0 or native.get('action') != 'TeleportFull'
            or native.get('exception') is not None
            or native.get('after_metadata', {}).get('lastActionSuccess') is not True
            or (native.get('capture') or {}).get('valid') is not True
            or (native.get('capture') or {}).get('black') is not False
            or (native.get('capture') or {}).get('source_action') != 'TeleportFull'
            or (native.get('capture') or {}).get('action_index') != 0
            or engine.encoded(native.get('after_metadata')) != engine.encoded(current.metadata)
            or engine.encoded(record.get('requested_from_initial_metadata')) != engine.encoded(initial)
            or any(engine.encoded(value) != engine.encoded(expected) for value in
                   (record.get('actual_command'), native.get('arguments'), sim.agent_setup_command))
            or any(engine.encoded(value) != engine.encoded(actual) for value in
                   (record.get('actual_agent'), sim.agent_setup_actual))):
        raise ValueError('Initial canonical action does not bind original input and actual current metadata')
    return reference


def place_native_with_retries(sim, object_name, tokens, *, rotation, record_key, max_points=40):
    """Return the actual successful token/args/event, or an exhausted ledger.

    Caller already pauses physics and keeps hands empty. No reset, teleport,
    restore, simulation step or hidden action retry occurs here. The input list
    order is preserved and at most its first max_points entries are executed.
    """
    if type(max_points) is not int or not 1 <= max_points <= 40 or not isinstance(tokens, (list, tuple)):
        raise ValueError('Require an explicit native-token sequence and a point cap between 1 and 40')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', str(record_key)):
        raise ValueError('Use a unique safe native retry record key')
    if any(type(rotation.get(axis)) not in (int, float) or not math.isfinite(rotation[axis]) for axis in 'xyz'):
        raise ValueError('Require the unchanged finite source rotation')
    start = sim.action_count
    result = {'version': NATIVE_RETRY_VERSION, 'policy': copy.deepcopy(NATIVE_RETRY_POLICY),
        'scene': sim.scene, 'record_key': record_key, 'object_name': object_name,
        'rotation': copy.deepcopy(rotation), 'status': 'running', 'max_points': max_points,
        'supplied_point_count': len(tokens), 'attempted_point_count': 0, 'failed_point_count': 0,
        'attempts': [], 'successful_token': None, 'arguments': None, 'action_reference': None,
        'fresh_controller_required': False, 'pair_attempt_counter_modified': False}
    original_error, success_event = None, None
    try:
        for token_index, supplied in enumerate(tokens[:max_points]):
            token = engine.validate_spawn({'spawn': supplied}, sim.scene)
            if hashlib.sha256(engine.encoded(token)).hexdigest() not in sim.spawn_tokens:
                raise ValueError('Placement token was not sampled by this Simulator instance')
            before = copy.deepcopy(sim.controller.last_event.metadata)
            before_state = object_pose_state(before)
            indexed = engine.objects_by_name(before['objects'])
            target, destination = indexed.get(object_name), indexed.get(token['receptacle_name'])
            if target is None or target.get('pickupable') is not True or destination is None or destination.get('receptacle') is not True:
                raise ValueError('Current metadata lacks the target or declared native receptacle')
            if before.get('inventoryObjects') != []:
                raise ValueError('Native placement retries require explicitly empty hands')
            args = {'objectId': target['objectId'], 'position': copy.deepcopy(token['selected_position']),
                    'rotation': copy.deepcopy(rotation), 'forceKinematic': True}
            attempt = {'token_index': token_index, 'spawn_token': token, 'arguments': args,
                'action_index': sim.action_count, 'action_reference': None, 'native_success': None,
                'before_object_poses': before_state, 'after_object_poses': None,
                'unchanged_check': None, 'error': None}
            result['attempts'].append(attempt)
            result['attempted_point_count'] += 1
            query_count, caught, event = len(sim.query_log), None, None
            try:
                event = sim.action('PlaceObjectAtPoint', **args)
            except BaseException as error:
                caught = error
            after = copy.deepcopy(sim.controller.last_event.metadata)
            try:
                attempt['after_object_poses'] = object_pose_state(after)
            except (KeyError, TypeError, ValueError) as error:
                attempt['pose_read_error'] = f'{type(error).__name__}: {error}'
            attempt['action_reference'] = copy.deepcopy(sim.last_action_reference)
            attempt['error'] = None if caught is None else f'{type(caught).__name__}: {caught}'
            if caught is not None and isinstance(caught, (TimeoutError, KeyboardInterrupt, SystemExit)):
                raise caught
            try:
                native, reference = _native_record(sim, before, after, args, attempt['action_index'], query_count)
            except Exception:
                if caught is not None:
                    raise caught
                raise
            flag = after.get('lastActionSuccess')
            if flag is True:
                if caught is not None:
                    raise caught
                if native.get('exception') is not None or event is None or engine.encoded(event.metadata) != engine.encoded(after):
                    raise ValueError('Native success lacks the actual returned event')
                attempt['native_success'] = True
                result.update(status='placed', successful_token=copy.deepcopy(token), arguments=copy.deepcopy(args),
                              action_reference=reference)
                success_event = event
                break
            if flag is not False:
                if caught is not None:
                    raise caught
                raise ValueError('Unknown native placement success flag')
            # Distinguish the existing Simulator's explicit native-failure
            # RuntimeError from RGB/recording/transport exceptions. Never hide
            # a rendering or evidence failure merely because metadata is false.
            if caught is not None and (type(caught) is not RuntimeError
                    or str(caught) != f'PlaceObjectAtPoint: {after.get("errorMessage")}'):
                raise caught
            if native.get('exception') != (None if caught is None else f'RuntimeError: {caught}'):
                raise ValueError('Native failure exception disagrees with the action record')
            attempt['native_success'] = False
            result['failed_point_count'] += 1
            try:
                check = unchanged_objects(before, after)
            except (KeyError, TypeError, ValueError) as error:
                check = {'verified': False, 'unchanged': None, 'error': f'{type(error).__name__}: {error}'}
            attempt['unchanged_check'] = check
            if check.get('unchanged') is not True:
                result.update(status='fresh_controller_required', fresh_controller_required=True)
                raise FreshControllerRequired(result)
        else:
            result['status'] = 'exhausted'
    except BaseException as error:
        original_error = error
        if result['status'] == 'running':
            result['status'] = 'aborted'
            # A non-native/rendering/transport error after dispatch is not a
            # measured unchanged failure. Caller must abandon this Controller
            # if it chooses to recover, and terminal errors still propagate.
            result['fresh_controller_required'] = bool(result['attempted_point_count'])
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        try:
            result['ledger_reference'] = sim.store.save_json(f'native_retry_{record_key}_{start:06d}.json', result)
        except BaseException as error:
            result['ledger_save_error'] = f'{type(error).__name__}: {error}'
            if original_error is None:
                result.update(status='evidence_failed', fresh_controller_required=True)
                original_error = NativePlacementEvidenceError('Native placement retry ledger could not be saved', result)
    if original_error is not None:
        original_error.native_retry_result = result
        raise original_error
    result['event'] = success_event
    return result


class RetryPlacementSimulator(StableCloneSimulator):
    """Same stable physics rules, with explicit actual-token native proposals.

    The new natural collector never invokes clone preparation. Inherited clone
    safeguards remain available for explicit future callers; no clone is made
    by this class. A native failure does not count as a pair construction.
    """
    def restore_snapshot(self, snapshot):
        capture, measured = super().restore_snapshot(snapshot)
        if self.config.get('frozen_render_version') == FROZEN_RENDER_VERSION:
            record_restore_readback(self, snapshot, measured)
        return capture, measured

    def freeze_render_pair(self, worlds, anchor, key):
        """Render actual paused worlds; original physical results stay separate."""
        poses = frozen_pose_lists(worlds, anchor)
        beginning = len(self.query_log)
        self.action('PausePhysicsAutoSim')
        rendered, captures, readbacks, restores = [], [], [], []
        for index in range(2):
            self.action('SetObjectPoses', objectPoses=copy.deepcopy(poses[index]), placeStationary=True)
            self.action('PhysicsSyncTransforms')
            self.action('Pass')
            capture = self.last_capture
            snapshot = self.snapshot(key + f'_frozen_w{index}')
            desired = frozen_requested_snapshot(worlds, poses, index)
            restored = engine.restoration_state(desired, snapshot['objects'], snapshot['agent'])
            readbacks.append(pose_readback(poses[index], snapshot['objects']))
            rendered.append(snapshot); captures.append(capture); restores.append(restored)
        nonanchor = [{'objectName': obj['name'], 'position': obj['position'], 'rotation': obj['rotation']}
                     for obj in rendered[0]['objects'] if obj['name'] != anchor]
        proof = {'version': FROZEN_RENDER_VERSION, 'policy': copy.deepcopy(FROZEN_RENDER_POLICY),
            'anchor_name': anchor, 'physical_world_references': [w['reference'] for w in worlds],
            'requested_pose_lists': poses, 'actual_world_references': [w['reference'] for w in rendered],
            'actual_capture_references': [c['evidence'] for c in captures],
            'action_references': copy.deepcopy(self.query_log[beginning:]),
            'readbacks': readbacks, 'restoration_checks': restores,
            'nonanchor_cross_world_readback': pose_readback(nonanchor, rendered[1]['objects'])}
        reference = self.store.save_json(key + '_frozen_render.json', proof)
        return {'worlds': rendered, 'captures': captures, 'proof': proof, 'reference': reference,
                'passed': all(check['restore_deterministic'] for check in restores)}

    def preflight_scene(self):
        if not self.action_count:
            return super().preflight_scene()
        initial_reference = canonical_initial_setup(self)
        if self.config.get('preflight_version') != PREFLIGHT_VERSION:
            raise RuntimeError('Require the explicit stable preflight version')
        if (self.max_steps != PREFLIGHT_POLICY['initial_max_steps']
                or self.config.get('preflight_policy') != PREFLIGHT_POLICY
                or (self.time_step, self.step_seconds) != (.02, .1)):
            raise ValueError('Require the explicit approved N/M policy and unchanged physics step durations')
        # Same N=50/M=50 implementation as the frozen stable backend. The sole
        # difference is the authenticated canonical setup before its source
        # snapshot; original action indices and exact state measurements stay.
        self._stable_common = None
        record = {'preflight_version': PREFLIGHT_VERSION, 'policy': copy.deepcopy(PREFLIGHT_POLICY),
            'initial_agent_setup_reference': initial_reference,
            'passed': False, 'snapshot': None, 'window_end_snapshot': None, 'reasons': [], 'error': None,
            'physics': {'time_step_seconds': self.time_step, 'seconds_per_step': self.step_seconds},
            'diagnostics': {'source_snapshot': None, 'pause_action_reference': None, 'pause_measurements': None,
                'initial_settling': {'settled': False, 'criterion': None, 'observations': []},
                'stability_window': {'completed': False, 'observations': []}}}
        initial, window = record['diagnostics']['initial_settling'], record['diagnostics']['stability_window']
        try:
            source = self.snapshot('original_scene')
            record['diagnostics']['source_snapshot'] = source
            event = self.action('PausePhysicsAutoSim')
            measured = position_step(source['objects'], event.metadata['objects'], source['objects'])
            record['diagnostics'].update(pause_action_reference=copy.deepcopy(self.last_action_reference), pause_measurements=measured)
            previous = copy.deepcopy(event.metadata['objects'])
            travelled = {name: row['observed_path_length_m'] for name, row in measured['objects'].items()}
            if not measured['inventory_exact'] or exceeded(measured, .5):
                record['reasons'].append('initial_inventory_changed_or_displacement_above_0_5m')
                return copy.deepcopy(record)
            streak = 0
            for index in range(50):
                event = self.action('AdvancePhysicsStep', timeStep=self.time_step,
                    simSeconds=self.step_seconds, allowAutoSimulation=False)
                measured = position_step(previous, event.metadata['objects'], source['objects'], travelled)
                streak = streak + 1 if measured['all_steps_below_1mm'] else 0
                row = {'index': index, 'native_is_scene_at_rest': event.metadata.get('isSceneAtRest'),
                    'small_step_streak': streak, 'measurements': measured,
                    'action_reference': copy.deepcopy(self.last_action_reference)}
                initial['observations'].append(row)
                previous = copy.deepcopy(event.metadata['objects'])
                travelled = {name: item['observed_path_length_m'] for name, item in measured['objects'].items()}
                if not measured['inventory_exact'] or exceeded(measured, .5):
                    record['reasons'].append('initial_inventory_changed_or_displacement_above_0_5m')
                    break
                if row['native_is_scene_at_rest'] is True or streak >= 10:
                    initial.update(settled=True, criterion='native_rest' if row['native_is_scene_at_rest'] is True else 'ten_small_steps')
                    record['snapshot'] = self.snapshot('preflight_settled_baseline')
                    break
            if not initial['settled']:
                if not record['reasons']:
                    record['reasons'].append('initial_not_settled_within_50_steps')
                return copy.deepcopy(record)
            baseline = record['snapshot']['objects']
            previous, travelled = copy.deepcopy(baseline), {}
            for index in range(50):
                event = self.action('AdvancePhysicsStep', timeStep=self.time_step,
                    simSeconds=self.step_seconds, allowAutoSimulation=False)
                measured = position_step(previous, event.metadata['objects'], baseline, travelled)
                window['observations'].append({'index': index, 'native_is_scene_at_rest': event.metadata.get('isSceneAtRest'),
                    'measurements': measured, 'action_reference': copy.deepcopy(self.last_action_reference)})
                previous = copy.deepcopy(event.metadata['objects'])
                travelled = {name: item['observed_path_length_m'] for name, item in measured['objects'].items()}
            window['completed'] = True
            record['window_end_snapshot'] = self.snapshot('preflight_window_end')
            if any(not row['measurements']['inventory_exact'] or exceeded(row['measurements'], .01)
                   for row in window['observations']):
                record['reasons'].append('stability_window_inventory_changed_or_displacement_above_0_01m')
            record['passed'] = not record['reasons']
            return copy.deepcopy(record)
        except BaseException as error:
            record['error'] = f'{type(error).__name__}: {error}'
            record['reasons'].append('preflight_interrupted_or_action_failed')
            raise
        finally:
            self.scene_preflight = copy.deepcopy(record)
            self.store.save_json('scene_preflight.json', record)

    def construct_world(self, baseline, key, placements, require_agent=None):
        if not self.scene_preflight or self.scene_preflight.get('passed') is not True:
            raise RuntimeError('Construction requires a passed original-scene preflight')
        if baseline['scene'] != self.scene:
            raise ValueError('Snapshot scene mismatch')
        if ((self.clone_preparation_attempted or self.clone_context is not None)
                and (self.clone_context or {}).get('setup_passed') is not True):
            raise RuntimeError('Clone common baseline has not passed physical validation')
        context = self.clone_context
        if context is not None:
            reference = context['baseline_validation_reference']
            content = Path(reference['file']).read_bytes()
            if hashlib.sha256(content).hexdigest() != reference['sha256']:
                raise ValueError('Common clone baseline reference changed')
            common = json.loads(content)['construction_physical']['realized_objects']
        else:
            common = self.scene_preflight['snapshot']['objects']
            reference = self.scene_preflight['snapshot']['reference']
        cached = getattr(self, '_stable_common', None)
        if cached is not None and cached != (reference, common):
            raise ValueError('Common baseline changed within one scene controller')
        self._stable_common = copy.deepcopy((reference, common))
        current = self.controller.last_event
        fixed_agent = copy.deepcopy(require_agent if require_agent is not None else engine.agent_pose(current))
        initial_agent = engine.agent_pose(current)
        if initial_agent != fixed_agent:
            raise ValueError('Agent already differs from required prefix pose')
        current_objects = current.metadata['objects']
        poses = engine.complete_pose_list(current_objects, baseline['poses'])
        if engine.editable_names(current_objects) != engine.editable_names(baseline['objects']):
            raise ValueError('Construction inventory differs from baseline')
        requested = {pose['objectName']: pose for pose in poses}
        evidence, successful, retries, proposals = {}, {}, {}, []
        for placement in placements:
            name = placement['object_name']
            if placement.get('clone') or name in evidence or name not in requested:
                raise ValueError('Require a unique existing editable object for native placement')
            if context is not None and name in context['lineage']['candidate_names']:
                raise ValueError('Shared cloned candidates cannot be edited between worlds')
            obj = engine.objects_by_name(baseline['objects'])[name]
            if 'rotation' in placement and placement['rotation'] != obj['rotation']:
                raise ValueError('v2 single-position edit preserves the source rotation')
            alternatives = placement.get('alternatives', [])
            if not isinstance(alternatives, (list, tuple)):
                raise ValueError('Native alternatives must be an explicit ordered token list')
            proposals.append((name, obj, [placement['spawn'], *alternatives]))
            evidence[name] = None
        self.action('PausePhysicsAutoSim')
        if self.controller.last_event.metadata.get('inventoryObjects') != []:
            raise RuntimeError('Constructive placement requires explicitly empty hands')
        for name, base_object, tokens in proposals:
            placement = place_native_with_retries(self, name, tokens, rotation=base_object['rotation'],
                record_key=key + '_' + str(len(retries)))
            record = {k: v for k, v in placement.items() if k != 'event'}
            retries[name] = record
            if placement['status'] != 'placed':
                raise NativePlacementExhausted(record)
            token, args, event = placement['successful_token'], placement['arguments'], placement['event']
            returned = engine.objects_by_name(event.metadata['objects'])[name]
            proof = copy.deepcopy(token)
            proof.update(requested_position=copy.deepcopy(returned['position']), geometry={
                'method': 'native_PlaceObjectAtPoint_forceKinematic', 'object_name': name,
                'source_rotation': copy.deepcopy(base_object['rotation']),
                'placement_action': 'PlaceObjectAtPoint', 'placement_arguments': copy.deepcopy(args),
                'placement_success': event.metadata.get('lastActionSuccess'),
                'returned_position': copy.deepcopy(returned['position']),
                'returned_rotation': copy.deepcopy(returned['rotation']),
                'placement_reference': copy.deepcopy(placement['action_reference']),
                'captured_before_complete_pose_set_and_settling': True,
                'source': engine.SOURCE + 'PhysicsRemoteFPSAgentController.cs#L3178'})
            requested[name]['position'] = copy.deepcopy(returned['position'])
            evidence[name], successful[name] = proof, copy.deepcopy(token)
        poses = engine.complete_pose_list(current_objects, poses)
        before = copy.deepcopy(baseline['objects'])
        request = {'key': key, 'baseline_key': baseline['key'], 'complete_requested_poses': poses,
            'placement_evidence_by_name': evidence, 'required_agent': fixed_agent,
            'native_retry_version': NATIVE_RETRY_VERSION, 'native_retry_records': retries}
        request_ref = self.store.save_json(f'construction_{self.action_count:06d}_request.json', request)
        self.action('SetObjectPoses', objectPoses=poses, placeStationary=False)
        settling = self._settle(before)
        observed, snap = self.last_capture, self.snapshot(key)
        changes = engine.pose_measurements(before, snap['objects'])
        actual, edits = engine.objects_by_name(snap['objects']), {}
        for name in sorted(evidence):
            obj, token = actual[name], evidence[name]
            destination = actual.get(token['receptacle_name'])
            edits[name] = {'position_error_m': engine.distance(requested[name]['position'], obj['position']),
                'requested_position': requested[name]['position'], 'actual_position': obj['position'],
                'requested_rotation': requested[name]['rotation'], 'actual_rotation': obj['rotation'],
                'parentReceptacles': copy.deepcopy(obj.get('parentReceptacles')),
                'target_receptacle_name': token['receptacle_name'],
                'target_receptacle_id': None if destination is None else destination['objectId'],
                'parent_membership': destination is not None and isinstance(obj.get('parentReceptacles'), list)
                    and destination['objectId'] in obj['parentReceptacles']}
        unchanged = all(row['position_delta_m'] <= .01 for name, row in changes['objects'].items() if name not in evidence)
        same_agent = initial_agent == snap['agent'] == fixed_agent
        passed = bool(settling['settled'] and changes['inventory_exact'] and changes['editable_inventory_exact']
            and unchanged and same_agent
            and all(row['position_error_m'] <= .02 and row['parent_membership'] for row in edits.values()))
        physical = {'valid_physical': passed, 'settling': settling, 'edited': edits,
            'collateral_position_within_1cm': unchanged, 'pose_measurements': changes,
            'agent_unchanged': same_agent, 'agent_before': initial_agent, 'agent_after': snap['agent'],
            'native_is_scene_at_rest': snap['native_is_scene_at_rest'],
            'before_objects': before, 'requested_poses': poses, 'realized_objects': snap['objects'],
            'edited_names': sorted(evidence), 'placement_evidence_by_name': evidence,
            'request_reference': request_ref,
            'physical_validity_method': 'approved receptacle construction + native rest + position bounds + parentReceptacles; no penetration-depth measurement'}
        check = common_baseline_check(common, snap['objects'], physical['edited_names'], reference)
        physical['common_baseline_check'] = check
        physical['common_baseline_check_reference'] = self.store.save_json(
            f'common_baseline_check_{self.action_count:06d}.json', check)
        physical['valid_physical'] = passed = bool(passed and check['passed'])
        result = {'passed': passed, 'snapshot': snap, 'physical': physical, 'capture': observed,
            'placement_evidence_by_name': evidence, 'successful_tokens_by_name': successful,
            'retry_records': retries, 'native_retry_version': NATIVE_RETRY_VERSION}
        self.store.save_json(f'stable_construction_{self.action_count:06d}_outcome.json',
                            {k: v for k, v in result.items() if k != 'capture'})
        return result
