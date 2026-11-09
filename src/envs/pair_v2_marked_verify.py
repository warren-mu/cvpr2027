"""Independent setup verification for marked pairs with safe native retries.

The caller's read_json must enforce each explicit file/SHA and its allowed
evidence roots. Remaining pair physics, w0/w1 prefixes and witnesses stay under
the existing full merger checks; this helper does not replace them.
"""
import copy

from . import pair_v2_simulator as engine
from .pair_v2_isolated_shard import ATTEMPT_SETUP_VERSION, ATTEMPT_SETUP_POLICY
from .pair_v2_native_retry import NATIVE_RETRY_VERSION, unchanged_objects


def verify_frozen_render(row, worlds, constructions, read_json):
    """Authenticate physical originals separately from actual paused renders."""
    from .pair_v2_native_retry import (FROZEN_RENDER_VERSION, FROZEN_RENDER_POLICY,
        frozen_pose_lists, frozen_requested_snapshot, pose_readback)
    if (row.get('frozen_render_version') != FROZEN_RENDER_VERSION
            or row.get('frozen_render_policy') != FROZEN_RENDER_POLICY):
        raise ValueError('Unknown frozen-render contract')
    refs = row.get('physical_world_references')
    if not isinstance(refs, list) or len(refs) != 2:
        raise ValueError('Both original physical snapshot references are required')
    physical = [{**read_json(ref), 'reference': ref} for ref in refs]
    if any(world['objects'] != construction['realized_objects'] for world, construction in zip(physical, constructions)):
        raise ValueError('Frozen rendering cannot replace the original measured physics')
    anchor = row['identities']['anchor_name']
    poses = frozen_pose_lists(physical, anchor)
    proof = read_json(row['frozen_render_reference'])
    if (proof.get('version') != FROZEN_RENDER_VERSION or proof.get('policy') != FROZEN_RENDER_POLICY
            or proof.get('anchor_name') != anchor or proof.get('physical_world_references') != refs
            or not same(proof.get('requested_pose_lists'), poses)
            or proof.get('actual_world_references') != [world['reference'] for world in worlds]):
        raise ValueError('Frozen render recipe differs from physical w0 plus actual settled w1 anchor')
    action_refs = proof.get('action_references')
    if not isinstance(action_refs, list) or len(action_refs) != 7:
        raise ValueError('Frozen render needs its complete paused two-world native action chain')
    expected = [('PausePhysicsAutoSim', {})]
    for values in poses:
        expected += [('SetObjectPoses', {'objectPoses': values, 'placeStationary': True}),
                     ('PhysicsSyncTransforms', {}), ('Pass', {})]
    actions = []
    for ref, (name, arguments) in zip(action_refs, expected):
        action = read_json(ref['reference'])
        if (ref.get('action') != name or action.get('action') != name or action.get('exception') is not None
                or not same(action.get('arguments'), arguments)
                or action['after_metadata'].get('lastActionSuccess') is not True
                or (action.get('capture') or {}).get('valid') is not True
                or (action.get('capture') or {}).get('black') is not False):
            raise ValueError('Frozen render contains an unverified, failed or unexpected native action')
        if actions and (action['index'] != actions[-1]['index'] + 1
                        or not same(action['before_metadata'], actions[-1]['after_metadata'])):
            raise ValueError('Frozen native action chain is not continuous')
        if agent_pose(action['before_metadata']) != physical[0]['agent'] or agent_pose(action['after_metadata']) != physical[0]['agent']:
            raise ValueError('Frozen paired renders changed the agent')
        actions.append(action)
    if actions[0]['before_metadata']['objects'] != physical[1]['objects']:
        raise ValueError('Frozen render does not follow the original w1 physical realization')
    readbacks, restoration = [], []
    for index, action_index in enumerate((3, 6)):
        world, action = worlds[index], actions[action_index]
        if (world.get('last_action') != 'Pass' or world.get('last_action_reference') != action_refs[action_index]['reference']
                or not same(world['objects'], action['after_metadata']['objects'])
                or not same(world['agent'], agent_pose(action['after_metadata']))
                or world.get('capture_reference') != action.get('capture')
                or proof['actual_capture_references'][index] != action.get('capture')):
            raise ValueError('Saved frozen world is not its actual paused native capture')
        actual = engine.objects_by_name(world['objects'])
        for item in engine.complete_pose_list(world['objects'], world['poses']):
            if any(item[key] != actual[item['objectName']][key] for key in ('position', 'rotation')):
                raise ValueError('Saved frozen snapshot poses differ from actual native metadata')
        readbacks.append(pose_readback(poses[index], world['objects']))
        measured = engine.restoration_state(frozen_requested_snapshot(physical, poses, index), world['objects'], world['agent'])
        if not measured['restore_deterministic']:
            raise ValueError('Frozen actual world fails unchanged restoration position/agent checks')
        restoration.append(measured)
    nonanchor = [{'objectName': obj['name'], 'position': obj['position'], 'rotation': obj['rotation']}
                 for obj in worlds[0]['objects'] if obj['name'] != anchor]
    cross = pose_readback(nonanchor, worlds[1]['objects'])
    if (not same(proof.get('readbacks'), readbacks) or not same(row.get('frozen_render_readbacks'), readbacks)
            or not same(proof.get('restoration_checks'), restoration)
            or not same(proof.get('nonanchor_cross_world_readback'), cross)
            or not same(row.get('nonanchor_cross_world_readback'), cross)):
        raise ValueError('Frozen pose diagnostics were altered or rounded to zero')
    return physical, {'version': FROZEN_RENDER_VERSION, 'passed': True, 'readbacks': readbacks,
        'nonanchor_cross_world_readback': cross, 'physical_world_references': refs,
        'frozen_render_reference': row['frozen_render_reference']}


def verify_frozen_restore_diagnostic(restoration, snapshot, read_json):
    from .pair_v2_native_retry import FROZEN_RENDER_VERSION, pose_readback
    proof = read_json(restoration['frozen_pose_readback_reference'])
    realized = read_json(restoration['restored_snapshot_reference'])
    actual = pose_readback(snapshot['poses'], realized['objects'])
    if (proof.get('version') != FROZEN_RENDER_VERSION
            or proof.get('requested_snapshot_reference') != snapshot['reference']
            or proof.get('restored_snapshot_reference') != restoration['restored_snapshot_reference']
            or not same(proof.get('readback'), actual) or not same(restoration.get('frozen_pose_readback'), actual)):
        raise ValueError('Frozen restore readback diagnostic differs from its actual metadata')
    return actual


def verify_canonical_preflight_prefix(preflight, pause, read_json):
    """The sole initial Teleport is real; no indices or baseline values reset."""
    record = read_json(preflight['initial_agent_setup_reference'])
    action = read_json(record['action_reference'])
    before, after = action['before_metadata'], action['after_metadata']
    agent = before['agent']
    expected = {'position': agent['position'], 'rotation': {'x': 0, 'y': agent['rotation']['y'], 'z': 0},
                'horizon': agent['cameraHorizon'], 'standing': True, 'forceAction': False}
    source = preflight['diagnostics']['source_snapshot']
    if (pause.get('index') != 1 or action.get('index') != 0 or action.get('action') != 'TeleportFull'
            or action.get('exception') is not None or after.get('lastActionSuccess') is not True
            or not same(action.get('arguments'), expected) or not same(record.get('actual_command'), expected)
            or not same(record.get('requested_from_initial_metadata'), agent)
            or not same(record.get('actual_agent'), agent_pose(after))
            or not same(source.get('agent'), agent_pose(after)) or not same(source.get('objects'), after['objects'])
            or not same(source.get('agent_setup_command'), expected)
            or source.get('last_action_reference') != record['action_reference']
            or not same(pause['before_metadata'], after)
            or (action.get('capture') or {}).get('valid') is not True
            or (action.get('capture') or {}).get('black') is not False):
        raise ValueError('Marked preflight lacks its exact actual canonical setup before the unchanged settling window')


def same(first, second):
    return engine.encoded(first) == engine.encoded(second)


def agent_pose(metadata):
    agent = metadata['agent']
    return {'position': agent['position'], 'rotation': agent['rotation'],
            'horizon': agent['cameraHorizon'], 'standing': agent['isStanding']}


def positions_match(objects, baseline):
    changes = engine.pose_measurements(baseline, objects)
    if (not changes['inventory_exact'] or not changes['editable_inventory_exact']
            or any(item['position_delta_m'] > .01 for item in changes['objects'].values())):
        raise ValueError('Native pre-w0 object state differs from the common baseline by the unchanged bound')
    before, actual = engine.objects_by_name(baseline), engine.objects_by_name(objects)
    if any(any(before[name].get(key) != actual[name].get(key) for key in
               ('objectType', 'pickupable', 'moveable')) for name in before):
        raise ValueError('Native pre-w0 stable inventory roles changed')
    return changes


def verify_attempt_setup(row, worlds, constructions, common_objects, read_json):
    if (row.get('attempt_setup_version') != ATTEMPT_SETUP_VERSION
            or row.get('attempt_setup_policy') != ATTEMPT_SETUP_POLICY or len(worlds) != 2 or len(constructions) != 2):
        raise ValueError('Require exact isolated producer setup policy and both constructed worlds')
    record = read_json(row['attempt_setup_reference'])
    if (not same(record, row.get('attempt_setup')) or record.get('version') != ATTEMPT_SETUP_VERSION
            or record.get('policy') != ATTEMPT_SETUP_POLICY or record.get('pair_id') != row['pair_id']
            or record.get('scene') != row['scene'] or record.get('phase') != 'before_v0_and_w0'
            or record.get('status') != 'passed' or record.get('setup_passed') is not True
            or record.get('error') is not None or record.get('proof_save_error') is not None):
        raise ValueError('Saved native setup does not authenticate a completed matching attempt')
    baseline = read_json(record['baseline_reference'])
    ready = read_json(record['restored_snapshot_reference'])
    restored = read_json(record['restoration']['restored_snapshot_reference'])
    if (not same(baseline['objects'], common_objects)
            or not same(constructions[0]['before_objects'], common_objects)
            or not same(constructions[1]['before_objects'], worlds[0]['objects'])):
        raise ValueError('Attempt baseline/construction input is not the validated shared common world')
    for snapshot in (baseline, ready, restored, *worlds):
        if snapshot.get('scene') != row['scene']:
            raise ValueError('Setup snapshot belongs to another scene')
        for key in ('worker_pid', 'unity_pid'):
            if type(snapshot.get(key)) is not int or snapshot[key] <= 0 or snapshot[key] != baseline.get(key):
                raise ValueError('Setup/world snapshots do not bind the same actual worker and Unity')
        if not same(snapshot.get('clone_context'), baseline.get('clone_context')):
            raise ValueError('Setup changed the independently validated clone baseline lineage')
    measured = engine.restoration_state(baseline, ready['objects'], ready['agent'])
    if (measured['restore_deterministic'] is not True
            or record['restoration'].get('restore_deterministic') is not True
            or not same(measured, record['independent_state'])
            or not same(ready['objects'], restored['objects']) or not same(ready['agent'], restored['agent'])):
        raise ValueError('Actual pre-view restore fails the unchanged position/exact-agent requirement')
    prefix = read_json(row['attempt_setup_native_prefix_reference'])
    if (prefix.get('version') != ATTEMPT_SETUP_VERSION or prefix.get('pair_id') != row['pair_id']
            or prefix.get('scene') != row['scene'] or prefix.get('setup_reference') != row['attempt_setup_reference']
            or prefix.get('safe_retry_version') != NATIVE_RETRY_VERSION):
        raise ValueError('Native prefix does not bind its exact attempt setup')
    refs = prefix.get('action_references')
    setup_refs = record.get('action_references')
    if (not isinstance(refs, list) or not isinstance(setup_refs, list) or len(setup_refs) < 3
            or refs[:len(setup_refs)] != setup_refs or not len(setup_refs) + 3 <= len(refs) <= len(setup_refs) + 42):
        raise ValueError('Native prefix omitted or inserted actions between restore and first placement')
    actions = []
    for action_number, info in enumerate(refs):
        action = read_json(info['reference'])
        failed_retry = len(setup_refs) + 2 <= action_number < len(refs) - 1
        if info.get('action') != action.get('action') or type(action.get('index')) is not int:
            raise ValueError('Native prefix contains an unbound action')
        if failed_retry:
            after = action['after_metadata']
            expected_error = 'RuntimeError: PlaceObjectAtPoint: ' + str(after.get('errorMessage'))
            if (action['action'] != 'PlaceObjectAtPoint' or after.get('lastActionSuccess') is not False
                    or action.get('exception') != expected_error or info.get('exception') != expected_error
                    or not unchanged_objects(action['before_metadata'], after)['unchanged']):
                raise ValueError('Failed native point did not preserve every object identity and pose')
        elif (info.get('exception') is not None or action.get('exception') is not None
                or action['after_metadata'].get('lastActionSuccess') is not True):
            raise ValueError('Native prefix contains an unsuccessful required action')
        if actions and (action['index'] != actions[-1]['index'] + 1
                        or not same(action['before_metadata'], actions[-1]['after_metadata'])):
            raise ValueError('Native prefix is not a continuous original action/metadata chain')
        actions.append(action)

    def action_at(index, name, arguments):
        value = actions[index]
        if value['action'] != name or not same(value.get('arguments'), arguments):
            raise ValueError('Unexpected native setup action/arguments: ' + name)
        return value

    action_at(0, 'PausePhysicsAutoSim', {})
    poses = engine.complete_pose_list(baseline['objects'], baseline['poses'])
    action_at(1, 'SetObjectPoses', {'objectPoses': poses, 'placeStationary': True})
    cursor = 2
    for name, desired in sorted(baseline.get('openness', {}).items()):
        obj = engine.objects_by_name(actions[cursor - 1]['after_metadata']['objects'])[name]
        actual = obj.get('openness', float(bool(obj.get('isOpen'))))
        if actual != desired:
            arguments = {'objectId': obj['objectId'], 'forceAction': True}
            if desired > 0:
                arguments['openness'] = desired
            action_at(cursor, 'OpenObject' if desired > 0 else 'CloseObject', arguments)
            cursor += 1
    canonical = copy.deepcopy(baseline.get('agent_setup_command'))
    if canonical is None:
        canonical = {**copy.deepcopy(baseline['agent']), 'forceAction': False}
    restored_action = action_at(cursor, 'TeleportFull', canonical)
    if (cursor + 1 != len(setup_refs)
            or not same(record['restoration'].get('executed_agent_setup_arguments'), canonical)
            or ready.get('last_action_reference') != setup_refs[-1]['reference']
            or restored.get('last_action_reference') != setup_refs[-1]['reference']
            or not same(restored_action['after_metadata']['objects'], ready['objects'])
            or not same(agent_pose(restored_action['after_metadata']), ready['agent'])):
        raise ValueError('Ready snapshot does not follow the actual canonical restore action')
    view = row['view']
    position_args = {'position': view['position'], 'rotation': {'x': 0, 'y': view['yaw'], 'z': 0},
                     'horizon': view['horizon'], 'standing': True, 'forceAction': False}
    positioned = action_at(cursor + 1, 'TeleportFull', position_args)
    action_at(cursor + 2, 'PausePhysicsAutoSim', {})
    geometry = constructions[0]['placement_evidence_by_name'][row['identities']['anchor_name']]['geometry']
    placed = action_at(len(actions) - 1, 'PlaceObjectAtPoint', geometry['placement_arguments'])
    for failed in actions[cursor + 3:-1]:
        arguments = failed.get('arguments', {})
        # Only the sampled position may differ during the bounded native retry.
        if (set(arguments) != set(geometry['placement_arguments']) or any(arguments.get(key) != value
                for key, value in geometry['placement_arguments'].items() if key != 'position')
                or agent_pose(failed['before_metadata']) != agent_pose(failed['after_metadata'])):
            raise ValueError('Native retry changed the target, rotation, force setting, or camera')
    if (refs[-1]['reference'] != prefix.get('first_place_reference')
            or refs[-1]['reference'] != geometry.get('placement_reference')):
        raise ValueError('Prefix does not terminate at the actual first w0 constructive placement')
    before = placed['before_metadata']
    positions = positions_match(before['objects'], common_objects)
    anchor = engine.objects_by_name(before['objects'])[row['identities']['anchor_name']]
    if geometry['placement_arguments'].get('objectId') != anchor['objectId']:
        raise ValueError('First placement does not address the actual current anchor identity')
    camera = agent_pose(positioned['after_metadata'])
    # Preserve construct_world's existing Python mapping equality for paired
    # camera poses. Encoded equality above belongs only to baseline restore;
    # do not introduce a new signed-zero rejection for the paired prefixes.
    if (agent_pose(before) != camera or any(world['agent'] != camera for world in worlds)
            or constructions[0]['agent_before'] != camera):
        raise ValueError('First w0 placement or paired prefixes changed the actual v0 camera')
    return {'passed': True, 'version': ATTEMPT_SETUP_VERSION,
        'setup_reference': copy.deepcopy(row['attempt_setup_reference']),
        'native_prefix_reference': copy.deepcopy(row['attempt_setup_native_prefix_reference']),
        'baseline_reference': copy.deepcopy(record['baseline_reference']),
        'ready_snapshot_reference': copy.deepcopy(record['restored_snapshot_reference']),
        'first_place_reference': copy.deepcopy(prefix['first_place_reference']),
        'restore_deterministic': measured['restore_deterministic'], 'agent_pose_exact': measured['agent_pose_exact'],
        'native_prefix_action_count': len(actions), 'safe_failed_native_points': len(actions) - len(setup_refs) - 3, 'pre_w0_positions': positions,
        'scope': 'actual restore and native prefix through first w0 Place; remaining pair checks stay mandatory'}
