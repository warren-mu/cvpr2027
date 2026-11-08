"""CPU contracts for an optional clone prepared before BOTH T01v2 worlds.

This file does not enable cloning or change first-round generation. A successful
SetObjectPoses duplicate is only an identity-creation measurement; legal native
placement, settling, prefix/evidence and independent replay remain mandatory.
"""
import copy
import math

from .pair_v2_policy import physical_check, single_edit_check

BUILD = 'f0825767cd50d69f666c7f282e54abfe58f1e917'
SOURCE = 'https://github.com/allenai/ai2thor/blob/' + BUILD + '/unity/Assets/Scripts/PhysicsSceneManager.cs#L334'
KINDS = ('direct_flip', 'direct_stay', 'look_flip', 'look_stay')


def clone_retry_required(passed_by_class):
    """The caller must supply an audited prior shard; unknown counts do not decide."""
    if set(passed_by_class) != set(KINDS) or any(type(passed_by_class[k]) is not int or passed_by_class[k] < 0 for k in KINDS):
        raise ValueError('Require all four explicit nonnegative integer prior shard counts')
    return any(passed_by_class[k] < 5 for k in KINDS)


def _index(objects):
    result = {}
    for obj in objects:
        name = obj.get('name')
        if not isinstance(name, str) or not name or name in result:
            raise ValueError('Missing or duplicate stable object name')
        result[name] = obj
    return result


def _editable(objects):
    return {name for name, obj in _index(objects).items() if obj.get('moveable') is True or obj.get('pickupable') is True}


def _vector(value):
    if not isinstance(value, dict) or set(value) != {'x', 'y', 'z'} or any(
            type(value[axis]) not in (int, float) or not math.isfinite(value[axis]) for axis in ('x', 'y', 'z')):
        raise ValueError('Require a finite xyz vector')
    return copy.deepcopy(value)


def _pose(obj):
    return {'objectName': obj['name'], 'position': _vector(obj['position']), 'rotation': _vector(obj['rotation'])}


def make_clone_recipe(original_objects, source_name, clone_position):
    """Append exactly one duplicate source after the complete, sorted old list.

    clone_position is an explicit construction proposal, never a settled value
    silently substituted afterward. Existing original objects keep their poses.
    """
    indexed = _index(original_objects)
    if source_name not in indexed or indexed[source_name].get('pickupable') is not True:
        raise ValueError('Clone source must be an existing pickupable object')
    source = indexed[source_name]
    category = source.get('objectType')
    if not isinstance(category, str) or not category:
        raise ValueError('Source category is unknown')
    if [name for name, obj in indexed.items() if obj.get('objectType') == category] != [source_name]:
        raise ValueError('Clone only a category with exactly one existing candidate')
    names = sorted(_editable(original_objects))
    poses = [_pose(indexed[name]) for name in names]
    extra = _pose(source)
    extra['position'] = _vector(clone_position)
    poses.append(extra)
    return {'schema_version': 1, 'stage': 'shared_baseline_setup_before_w0_and_w1',
            'source_name': source_name, 'object_type': category,
            'original_names': sorted(indexed), 'original_editable_names': names,
            'duplicate_request_index': len(names), 'objectPoses': poses,
            'source_pose': _pose(source), 'clone_creation_position': copy.deepcopy(extra['position']),
            'source_build': BUILD, 'source_url': SOURCE,
            'expected_added_objects': 1, 'physical_validity': None,
            'world_pair_edit': False, 'object_id_policy': 'bind actual metadata; SetupScene regenerates position-derived IDs'}


def verify_clone_creation(original_objects, created_objects, recipe, *, action_success):
    """Bind the one actually added stable name; never predict Unity clone/ID text."""
    before, after = _index(original_objects), _index(created_objects)
    expected = make_clone_recipe(original_objects, recipe['source_name'], recipe['clone_creation_position'])
    reasons = []
    if recipe != expected:
        reasons.append('clone_recipe_changed_or_incomplete')
    if action_success is not True:
        reasons.append('native_clone_action_not_successful')
    missing, added = sorted(set(before) - set(after)), sorted(set(after) - set(before))
    if missing or len(added) != 1:
        reasons.append('clone_must_add_exactly_one_object_without_losing_originals')
    ids = [obj.get('objectId') for obj in created_objects]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        reasons.append('actual_object_ids_missing_or_duplicated')
    source_name = recipe['source_name']
    clone_name = added[0] if len(added) == 1 else None
    if clone_name is not None:
        clone = after[clone_name]
        if clone.get('pickupable') is not True or clone.get('objectType') != recipe['object_type']:
            reasons.append('added_object_is_not_a_pickupable_source_category_copy')
        if _editable(created_objects) != _editable(original_objects) | {clone_name}:
            reasons.append('editable_inventory_changed_beyond_clone')
    if source_name not in after or any(after[source_name].get(field) != before[source_name].get(field)
                                        for field in ('objectType', 'pickupable', 'moveable')):
        reasons.append('original_source_identity_or_role_changed')
    category_names = sorted(name for name, obj in after.items() if obj.get('objectType') == recipe['object_type'])
    if clone_name is None or category_names != sorted([source_name, clone_name]):
        reasons.append('candidate_category_not_exactly_source_and_clone')
    return {'passed': not reasons, 'reasons': reasons, 'source_name': source_name, 'clone_name': clone_name,
            'object_type': recipe['object_type'], 'candidate_names': [source_name, clone_name],
            'added_names': added, 'missing_original_names': missing,
            'baseline_names': sorted(after), 'baseline_editable_names': sorted(_editable(created_objects)),
            'duplicate_request_index': recipe['duplicate_request_index'], 'physical_validity': None,
            'scope': 'creation identity only; native IDs may change after later SetObjectPoses'}


def clone_baseline_check(original_objects, created_objects, settled_objects, recipe, *,
                         creation_action_success, requested_clone_pose, placement_evidence,
                         native_is_scene_at_rest):
    """Check the enlarged shared baseline before constructing either world.

    requested_clone_pose and native placement proof must be captured BEFORE
    settling. Only the new clone may move intentionally; all original objects
    retain the original-scene positions within the existing 1cm limit.
    """
    lineage = verify_clone_creation(original_objects, created_objects, recipe, action_success=creation_action_success)
    if not lineage['passed']:
        return {'passed': False, 'lineage': lineage, 'physical': None, 'reasons': lineage['reasons']}
    clone_name = lineage['clone_name']
    if requested_clone_pose.get('objectName') != clone_name:
        raise ValueError('Requested placement must name the actual newly created clone')
    created = _index(created_objects)
    before = copy.deepcopy(original_objects) + [copy.deepcopy(created[clone_name])]
    requests = [_pose(_index(original_objects)[name]) for name in sorted(_editable(original_objects))]
    requests.append(copy.deepcopy(requested_clone_pose))
    physical = physical_check(before, requests, settled_objects, [clone_name],
        native_is_scene_at_rest=native_is_scene_at_rest, placement_evidence_by_name={clone_name: placement_evidence})
    return {'passed': physical['valid_physical'], 'lineage': lineage, 'physical': physical,
            'complete_post_clone_requested_poses': requests,
            'reasons': physical['reasons'], 'scope': 'shared baseline construction, never the counterfactual X edit'}


def clone_pair_check(lineage, objects_w0, objects_w1, requested_poses_w0, requested_poses_w1, anchor_name):
    """After creation, never repeat a source name again: use actual clone once."""
    reasons = []
    candidates = lineage.get('candidate_names') or []
    if lineage.get('passed') is not True or len(candidates) != 2 or len(set(candidates)) != 2:
        reasons.append('clone_lineage_not_verified')
    if anchor_name in candidates:
        reasons.append('edited_anchor_cannot_be_a_cloned_candidate')
    for label, objects, requests in (('w0', objects_w0, requested_poses_w0), ('w1', objects_w1, requested_poses_w1)):
        indexed = _index(objects)
        names = [pose['objectName'] for pose in requests]
        if sorted(indexed) != lineage.get('baseline_names') or sorted(_editable(objects)) != lineage.get('baseline_editable_names'):
            reasons.append(label + '_inventory_differs_from_shared_clone_baseline')
        if len(set(names)) != len(names) or set(names) != _editable(objects):
            reasons.append(label + '_must_use_each_actual_editable_name_exactly_once')
        if any(name not in indexed or indexed[name].get('objectType') != lineage.get('object_type')
               or indexed[name].get('pickupable') is not True for name in candidates):
            reasons.append(label + '_candidate_identity_or_role_changed')
        ids = [obj.get('objectId') for obj in objects]
        if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
            reasons.append(label + '_actual_object_ids_missing_or_duplicated')
    if not reasons:
        edit = single_edit_check(requested_poses_w0, requested_poses_w1, anchor_name)
        if not edit['passed']:
            reasons.append('world_pair_request_must_only_move_anchor_X')
    else:
        edit = None
    return {'passed': not reasons, 'reasons': reasons, 'candidate_names': candidates,
            'anchor_name': anchor_name, 'single_edit': edit,
            'scope': 'shared clone identity and requested edit scope; all ordinary T01v2 acceptance checks still required'}


def clone_replay_check(original_objects, recreated_objects, recipe, expected_lineage, *, action_success):
    """A fresh process recreates the same actual-name inventory before restore."""
    actual = verify_clone_creation(original_objects, recreated_objects, recipe, action_success=action_success)
    matched = actual['passed'] and expected_lineage.get('passed') is True and all(
        actual[key] == expected_lineage.get(key) for key in
        ('source_name', 'clone_name', 'object_type', 'candidate_names', 'baseline_names', 'baseline_editable_names', 'duplicate_request_index'))
    return {'passed': matched, 'actual_lineage': actual,
            'reasons': [] if matched else ['fresh_process_clone_lineage_does_not_match_saved_baseline'],
            'scope': 'clone recreation only; subsequent snapshot state restore and witness must also pass'}
