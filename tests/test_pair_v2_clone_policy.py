"""Synthetic clone preparation contracts; no scene/controller or file writes."""
import copy
import unittest

from src.envs.pair_v2_clone_policy import (KINDS, clone_retry_required, make_clone_recipe,
    verify_clone_creation, clone_baseline_check, clone_pair_check, clone_replay_check)


def xyz(x, y=0.5, z=0.0):
    return {'x': x, 'y': y, 'z': z}


def original():
    rows = []
    for name, category, x, pickup, movable, receptacle in (
        ('A', 'Mug', 1.0, True, False, False), ('X', 'Apple', 3.0, True, False, False),
        ('Chair', 'Chair', 0.0, False, True, False), ('Table', 'Table', 2.0, False, False, True)):
        rows.append({'name': name, 'objectId': name + '_id', 'objectType': category,
                     'position': xyz(x), 'rotation': xyz(0.0, 0.0), 'pickupable': pickup,
                     'moveable': movable, 'receptacle': receptacle, 'parentReceptacles': ['Table_id']})
    return rows


def poses(objects):
    return [{'objectName': obj['name'], 'position': copy.deepcopy(obj['position']),
             'rotation': copy.deepcopy(obj['rotation'])} for obj in objects if obj['pickupable'] or obj['moveable']]


def creation():
    before = original()
    recipe = make_clone_recipe(before, 'A', xyz(0.3))
    clone = copy.deepcopy(before[0])
    clone.update(name='actual_unity_clone_name_copy_3', objectId='measured_clone_native_id', position=xyz(0.3))
    created = copy.deepcopy(before) + [clone]
    return before, created, recipe


def settled_fixture():
    before, created, recipe = creation()
    clone_name = created[-1]['name']
    requested = {'objectName': clone_name, 'position': xyz(0.0), 'rotation': xyz(0.0, 0.0)}
    settled = copy.deepcopy(created)
    settled[-1]['position'] = xyz(0.02)
    sample = xyz(0.0, 0.7)
    proof = {'action': 'GetSpawnCoordinatesAboveReceptacle', 'success': True,
        'receptacle_name': 'Table', 'selected_position': sample, 'coordinates': [sample],
        'requested_position': requested['position'], 'geometry': {
            'method': 'native_PlaceObjectAtPoint_forceKinematic', 'object_name': clone_name,
            'placement_action': 'PlaceObjectAtPoint', 'placement_success': True,
            'source_rotation': requested['rotation'], 'placement_reference': {'file': 'synthetic_placement', 'sha256': 'fixture'},
            'placement_arguments': {'objectId': created[-1]['objectId'], 'position': sample,
                                    'rotation': requested['rotation'], 'forceKinematic': True},
            'returned_position': requested['position'], 'returned_rotation': requested['rotation'],
            'captured_before_complete_pose_set_and_settling': True}}
    return before, created, recipe, settled, requested, proof


class ClonePolicyContracts(unittest.TestCase):
    def test_trigger_requires_all_four_recorded_counts(self):
        counts = {kind: 5 for kind in KINDS}
        self.assertFalse(clone_retry_required(counts))
        counts['look_stay'] = 4
        self.assertTrue(clone_retry_required(counts))
        for bad in ({}, {**counts, 'look_stay': None}, {**counts, 'look_stay': True}, {**counts, 'look_stay': -1}):
            with self.assertRaises(ValueError):
                clone_retry_required(bad)

    def test_recipe_preserves_all_originals_and_duplicates_only_source_last(self):
        before = original()
        recipe = make_clone_recipe(before, 'A', xyz(0.3))
        requests = recipe['objectPoses']
        self.assertEqual([row['objectName'] for row in requests], ['A', 'Chair', 'X', 'A'])
        self.assertEqual(recipe['duplicate_request_index'], 3)
        self.assertEqual(requests[-1]['position'], xyz(0.3))
        self.assertEqual(requests[0]['position'], before[0]['position'])
        self.assertEqual(recipe, make_clone_recipe(before[::-1], 'A', xyz(0.3)))
        self.assertFalse(recipe['world_pair_edit'])
        self.assertIsNone(recipe['physical_validity'])
        self.assertNotIn('clone_name', recipe)

    def test_clone_only_single_pickupable_category(self):
        before = original()
        with self.assertRaises(ValueError):
            make_clone_recipe(before, 'Chair', xyz(0.3))
        duplicate = copy.deepcopy(before[0])
        duplicate['name'], duplicate['objectId'] = 'already_B', 'already_B_id'
        with self.assertRaises(ValueError):
            make_clone_recipe(before + [duplicate], 'A', xyz(0.3))

    def test_creation_binds_actual_name_without_predicting_position_derived_id(self):
        before, created, recipe = creation()
        result = verify_clone_creation(before, created, recipe, action_success=True)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['clone_name'], created[-1]['name'])
        self.assertIsNone(result['physical_validity'])
        created[-1]['objectId'] = 'different_native_id_after_SetupScene'
        self.assertTrue(verify_clone_creation(before, created, recipe, action_success=True)['passed'])

    def test_creation_rejects_omission_extra_child_id_collision_and_failed_native_action(self):
        before, created, recipe = creation()
        cases = [created[1:], created + [dict(created[-1], name='unexpected_child', objectId='child_id')]]
        duplicate_id = copy.deepcopy(created)
        duplicate_id[-1]['objectId'] = duplicate_id[0]['objectId']
        cases.append(duplicate_id)
        wrong_category = copy.deepcopy(created)
        wrong_category[-1]['objectType'] = 'Apple'
        cases.append(wrong_category)
        for rows in cases:
            self.assertFalse(verify_clone_creation(before, rows, recipe, action_success=True)['passed'])
        self.assertFalse(verify_clone_creation(before, created, recipe, action_success=False)['passed'])

    def test_recipe_cannot_reposition_old_object_or_repeat_another_source(self):
        before, created, recipe = creation()
        for alter in (lambda r: r['objectPoses'][0]['position'].update(x=9.0),
                      lambda r: r['objectPoses'].pop(1),
                      lambda r: r['objectPoses'][-1].update(objectName='X')):
            changed = copy.deepcopy(recipe)
            alter(changed)
            self.assertFalse(verify_clone_creation(before, created, changed, action_success=True)['passed'])

    def test_shared_baseline_uses_actual_clone_legal_placement_and_existing_position_thresholds(self):
        before, created, recipe, settled, requested, proof = settled_fixture()
        settled[2]['position']['x'] = 0.01
        settled[-1]['rotation']['x'] = 0.00003
        result = clone_baseline_check(before, created, settled, recipe, creation_action_success=True,
            requested_clone_pose=requested, placement_evidence=proof, native_is_scene_at_rest=True)
        self.assertTrue(result['passed'], result)
        names = [pose['objectName'] for pose in result['complete_post_clone_requested_poses']]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn(created[-1]['name'], names)
        self.assertEqual(names.count('A'), 1)
        self.assertEqual(result['physical']['edited_names'], [created[-1]['name']])

    def test_clone_physical_failures_cannot_be_hidden_by_creation_success(self):
        before, created, recipe, settled, requested, proof = settled_fixture()
        for mode in ('clone_error', 'collateral', 'parent', 'rest', 'post_settle_refill', 'wrong_source_proof'):
            actual, evidence, rest = copy.deepcopy(settled), copy.deepcopy(proof), True
            if mode == 'clone_error': actual[-1]['position']['x'] = 0.02001
            if mode == 'collateral': actual[2]['position']['x'] = 0.01001
            if mode == 'parent': actual[-1]['parentReceptacles'] = []
            if mode == 'rest': rest = None
            if mode == 'post_settle_refill': evidence['geometry']['returned_position'] = actual[-1]['position']
            if mode == 'wrong_source_proof': evidence['geometry']['object_name'] = 'A'
            result = clone_baseline_check(before, created, actual, recipe, creation_action_success=True,
                requested_clone_pose=requested, placement_evidence=evidence, native_is_scene_at_rest=rest)
            self.assertFalse(result['passed'], mode)

    def test_pair_has_shared_clone_and_only_anchor_requested_edit(self):
        before, created, recipe = creation()
        lineage = verify_clone_creation(before, created, recipe, action_success=True)
        w0, w1 = copy.deepcopy(created), copy.deepcopy(created)
        requests0, requests1 = poses(w0), poses(w1)
        next(row for row in requests1 if row['objectName'] == 'X')['position']['x'] = 3.5
        for row in w1: row['objectId'] += '_new_current_id'
        result = clone_pair_check(lineage, w0, w1, requests0, requests1, 'X')
        self.assertTrue(result['passed'], result)
        changed = copy.deepcopy(requests1)
        next(row for row in changed if row['objectName'] == created[-1]['name'])['position']['x'] = 4.0
        self.assertFalse(clone_pair_check(lineage, w0, w1, requests0, changed, 'X')['passed'])
        self.assertFalse(clone_pair_check(lineage, w0, w1, requests0, requests1, 'A')['passed'])
        self.assertFalse(clone_pair_check(lineage, w0, w1, requests0, recipe['objectPoses'], 'X')['passed'])

    def test_replay_must_recreate_same_actual_name_then_restore_and_replay_separately(self):
        before, created, recipe = creation()
        saved = verify_clone_creation(before, created, recipe, action_success=True)
        fresh = copy.deepcopy(created)
        for obj in fresh: obj['objectId'] += '_fresh'
        self.assertTrue(clone_replay_check(before, fresh, recipe, saved, action_success=True)['passed'])
        fresh[-1]['name'] = 'different_actual_clone_name'
        result = clone_replay_check(before, fresh, recipe, saved, action_success=True)
        self.assertFalse(result['passed'])
        self.assertNotIn('restore_deterministic', result)


if __name__ == '__main__':
    unittest.main()
