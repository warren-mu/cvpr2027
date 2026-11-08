"""Independent CPU verification of the versioned pre-w0 native setup proof.

The caller's read_json must enforce each explicit file/SHA and its allowed
evidence roots. Remaining pair physics, w0/w1 prefixes and witnesses stay under
the existing full merger checks; this helper does not replace them.
"""
import copy

from . import pair_v2_simulator as engine
from .pair_v2_isolated_shard import ATTEMPT_SETUP_VERSION, ATTEMPT_SETUP_POLICY


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
            or prefix.get('scene') != row['scene'] or prefix.get('setup_reference') != row['attempt_setup_reference']):
        raise ValueError('Native prefix does not bind its exact attempt setup')
    refs = prefix.get('action_references')
    setup_refs = record.get('action_references')
    if (not isinstance(refs, list) or not isinstance(setup_refs, list) or len(setup_refs) < 3
            or refs[:len(setup_refs)] != setup_refs or len(refs) != len(setup_refs) + 3):
        raise ValueError('Native prefix omitted or inserted actions between restore and first placement')
    actions = []
    for info in refs:
        action = read_json(info['reference'])
        if (info.get('action') != action.get('action') or info.get('exception') is not None
                or action.get('exception') is not None or action['after_metadata'].get('lastActionSuccess') is not True
                or type(action.get('index')) is not int):
            raise ValueError('Native prefix contains an unsuccessful or unbound action')
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
    placed = action_at(cursor + 3, 'PlaceObjectAtPoint', geometry['placement_arguments'])
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
        'native_prefix_action_count': len(actions), 'pre_w0_positions': positions,
        'scope': 'actual restore and native prefix through first w0 Place; remaining pair checks stay mandatory'}
