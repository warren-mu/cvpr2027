"""Supplement-9 CPU recipes/readback proofs only; no native-render claims."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.envs.pair_table_export import frozen_w1_request, verify_frozen_world_request, frozen_target_measurement
from src.envs.pair_table_export import SimulatorTableRenderer, point_id, build_table
from src.envs.pair_table_checks import (world_metadata_matches, pose_restore_measurements,
    frozen_non_anchor_measurements, verify_paused_setup)


def worlds():
    def object(name, x):
        return {'name': name, 'objectType': name, 'moveable': True, 'pickupable': True,
                'position': {'x': x, 'y': 0., 'z': 0.}, 'rotation': {'x': 0., 'y': 0., 'z': 0.}}
    result = []
    for world in (0, 1):
        objects = [object('A', .00012 * world), object('B', 1.), object('X', 2. + world)]
        result.append({'scene': 'FloorPlan1', 'objects': objects,
            'poses': [{'objectName': obj['name'], 'position': copy.deepcopy(obj['position']),
                       'rotation': copy.deepcopy(obj['rotation'])} for obj in objects],
            'agent': {'position': {'x': 0, 'y': 1, 'z': 0}}, 'agent_setup_command': {'fixture': True},
            'clone_context': None, 'reference': {'file': str(world), 'sha256': str(world) * 64}})
    return result


class FrozenTableTests(unittest.TestCase):
    def test_native_teleport_collision_stops_complete_graph_without_force_or_skip(self):
        originals = worlds()
        position = {'x': 0., 'y': 1., 'z': 0.}
        for snapshot in originals:
            snapshot['agent_setup_command'] = {'position': position, 'rotation': {'x': 0., 'y': 0., 'z': 0.},
                'horizon': 0., 'standing': True, 'forceAction': False}
        actions = []
        sim = SimpleNamespace(controller=SimpleNamespace(last_event=SimpleNamespace(metadata={
            'objects': originals[0]['objects']})), query_log=[], last_action_reference={})
        def action(name, **arguments):
            actions.append((name, arguments))
            if name == 'TeleportFull':
                raise RuntimeError('TeleportFull: InvalidOperationException: Collided with: Box')
        sim.action = action
        renderer = object.__new__(SimulatorTableRenderer)
        renderer.sim, renderer.worlds, renderer.positions = sim, originals, {point_id(position): position}
        with patch('src.envs.pair_table_checks.world_metadata_matches', return_value={}), \
                self.assertRaisesRegex(RuntimeError, 'Collided with: Box'):
            build_table(renderer, pair_id='collision', scene='FloorPlan1', kind='look_flip',
                instruction='Select the marked object closest to the anchor.', anchor_name='X',
                candidate_names=('A', 'B'), target_names=('A', 'B'),
                initial_position=position, initial_heading=0)
        self.assertEqual([name for name, _ in actions],
                         ['PausePhysicsAutoSim', 'SetObjectPoses', 'PhysicsSyncTransforms', 'TeleportFull'])
        self.assertIs(actions[-1][1]['forceAction'], False)

    def test_actual_restore_rechecks_target_and_original_ratio_without_new_tolerance(self):
        objects = copy.deepcopy(worlds()[0]['objects'])
        for obj, category, x in zip(objects, ('Mug', 'Mug', 'Bowl'), (0., 1.25, .5)):
            obj['objectType'] = category; obj['position']['x'] = x
        self.assertTrue(frozen_target_measurement(objects, ('A', 'B'), 'X', 'A', None)['passed'])
        with self.assertRaisesRegex(ValueError, 'frozen_render_target_changed'):
            frozen_target_measurement(objects, ('A', 'B'), 'X', 'B', None)
        objects[0]['position']['x'] = -.0001
        with self.assertRaisesRegex(ValueError, 'closest_ratio_below_1_5'):
            frozen_target_measurement(objects, ('A', 'B'), 'X', 'A', None)
        objects[0]['position']['x'] = 0.
        objects[1]['objectType'] = 'Plate'
        with self.assertRaisesRegex(ValueError, 'candidate_categories_differ'):
            frozen_target_measurement(objects, ('A', 'B'), 'X', 'A', None)
        self.assertTrue(frozen_target_measurement(objects, ('A', 'B'), 'X', 'A',
                                                 'two_marked_pickupables_v1')['passed'])

    def test_recipe_copies_only_original_anchor_and_preserves_physical_sources(self):
        original = worlds(); before = copy.deepcopy(original)
        requested = frozen_w1_request(original, 'X')
        self.assertEqual(requested['poses'][0], original[0]['poses'][0])
        self.assertEqual(requested['poses'][2], original[1]['poses'][2])
        self.assertEqual(requested['objects'][0], original[0]['objects'][0])
        self.assertEqual(requested['objects'][2], original[1]['objects'][2])
        self.assertIn('not a native observation', requested['serialization_scope'])
        self.assertNotIn('capture_reference', requested)
        self.assertEqual(original, before)
        raw = {world['reference']['file']: {k: v for k, v in world.items() if k != 'reference'} for world in original}
        read = lambda ref: json.dumps(raw[ref['file']]).encode()
        self.assertEqual(verify_frozen_world_request(requested, read), [world['reference'] for world in original])
        forged = copy.deepcopy(requested)
        forged['poses'][0]['position']['x'] = .00012
        with self.assertRaisesRegex(ValueError, 'authenticated original'):
            verify_frozen_world_request(forged, read)

    def test_missing_anchor_or_different_inventory_fails_recipe(self):
        original = worlds()
        with self.assertRaises(ValueError):
            frozen_w1_request(original, 'absent')
        original[1]['objects'].pop()
        with self.assertRaises(ValueError):
            frozen_w1_request(original, 'X')

    def test_exact_numeric_signed_zero_and_submillimeter_diagnostics(self):
        original = worlds()[0]
        actual = {'objects': copy.deepcopy(original['objects'])}
        actual['objects'][0]['rotation']['x'] = -0.
        equal = pose_restore_measurements(actual, original)
        self.assertTrue(equal['all_object_positions_exact'])
        self.assertTrue(equal['all_object_rotations_exact'])
        actual['objects'][0]['position']['x'] = .00012
        actual['objects'][0]['rotation']['y'] = .00005
        measured = world_metadata_matches(actual, original)
        self.assertFalse(measured['all_object_positions_exact'])
        self.assertFalse(measured['all_object_rotations_exact'])
        self.assertEqual(measured['max_position_delta_m'], .00012)
        self.assertEqual(measured['max_raw_rotation_component_delta_degrees'], .00005)
        # Existing 1 cm restore guard is unchanged; diagnostics add no threshold.
        actual['objects'][0]['position']['x'] = .010001
        with self.assertRaisesRegex(ValueError, 'declared world'):
            world_metadata_matches(actual, original)

    def test_actual_nonanchor_comparison_excludes_x_without_hiding_other_movement(self):
        original = worlds()
        measured = frozen_non_anchor_measurements(original[0], original[1], 'X')
        self.assertEqual(measured['max_position_delta_m'], .00012)
        self.assertEqual(measured['position_mismatch_names'], ['A'])
        self.assertFalse(measured['all_object_positions_exact'])

    def test_paused_setup_requires_successful_uninterrupted_native_actions(self):
        names = ['PausePhysicsAutoSim', 'SetObjectPoses', 'PhysicsSyncTransforms', 'TeleportFull']
        records, sequence, previous = {}, [], {'before': True}
        for index, name in enumerate(names):
            ref = {'file': str(index), 'sha256': str(index) * 64}
            after = {'lastActionSuccess': True, 'native_step': index}
            records[str(index)] = {'index': index, 'action': name, 'exception': None,
                                   'before_metadata': previous, 'after_metadata': after}
            sequence.append({'action': name, 'reference': ref}); previous = after
        evidence = {'paused_setup_actions': sequence, 'physics_pause_action': sequence[0]['reference'],
                    'agent_setup_action': sequence[-1]['reference']}
        read = lambda ref: json.dumps(records[ref['file']]).encode()
        self.assertEqual(len(verify_paused_setup(evidence, read)), 4)
        records['2']['index'] = 10
        with self.assertRaisesRegex(ValueError, 'interrupted'):
            verify_paused_setup(evidence, read)
        records['2']['index'] = 2
        records['0']['after_metadata']['lastActionSuccess'] = False
        with self.assertRaisesRegex(ValueError, 'Failed'):
            verify_paused_setup(evidence, read)


if __name__ == '__main__':
    unittest.main()
