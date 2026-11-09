"""Synthetic native-action contracts only; no THOR or measured GPU claims."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.envs import pair_v2_native_retry as frozen
from src.envs import pair_v2_marked_verify as verify
from src.envs import pair_v2_marked_shard as shard
from src.envs import pair_v2_simulator as engine


def xyz(x=0.):
    return {'x': x, 'y': 0., 'z': 0.}


class Store:
    def __init__(self):
        self.values = {}
    def save_json(self, name, value):
        path = '/synthetic/' + name
        if path in self.values:
            raise FileExistsError(path)
        self.values[path] = copy.deepcopy(value)
        return {'file': path, 'sha256': 'synthetic_fixture'}
    def read(self, ref):
        return copy.deepcopy(self.values[ref['file']])


def make_objects():
    return [{'name': name, 'objectId': name, 'objectType': category,
        'position': xyz(position), 'rotation': xyz(), 'pickupable': name != 'Static',
        'moveable': False} for name, category, position in
        [('A', 'Mug', 0.), ('B', 'Book', 2.), ('X', 'Vase', .2), ('Static', 'Wall', 5.)]]


def poses(objects):
    return [{'objectName': obj['name'], 'position': copy.deepcopy(obj['position']),
        'rotation': copy.deepcopy(obj['rotation'])} for obj in objects if obj['pickupable'] or obj['moveable']]


class NativeFixture:
    def __init__(self, restore_delta=0.):
        self.store = Store()
        self.agent = {'position': xyz(-1), 'rotation': xyz(), 'horizon': 0., 'standing': True}
        self.metadata = {'agent': {'position': xyz(-1), 'rotation': xyz(), 'cameraHorizon': 0., 'isStanding': True},
            'objects': make_objects(), 'lastActionSuccess': True, 'sequenceId': 0}
        self.query_log, self.action_count, self.snapshot_count = [], 0, 0
        self.restore_delta, self.set_count = restore_delta, 0
        self.last_capture = {'evidence': None}
        self.controller = SimpleNamespace(last_event=SimpleNamespace(metadata=self.metadata))
        self.last_action, self.last_action_reference = None, None
        self.physical = []
        self.physical.append(self.snapshot('physical_w0'))
        self.metadata['objects'][0]['position']['x'] += .005  # genuine collateral physical drift
        self.metadata['objects'][2]['position']['x'] = 1.8
        self.metadata['objects'][2]['rotation']['y'] = .02
        self.physical.append(self.snapshot('physical_w1'))
    def action(self, name, **kwargs):
        before = copy.deepcopy(self.metadata)
        if name == 'SetObjectPoses':
            self.set_count += 1
            objects = engine.objects_by_name(self.metadata['objects'])
            for pose in kwargs['objectPoses']:
                for key in ('position', 'rotation'):
                    objects[pose['objectName']][key] = copy.deepcopy(pose[key])
            if self.set_count == 2:
                objects['A']['position']['x'] += self.restore_delta
        self.metadata['sequenceId'] += 1
        self.metadata['lastActionSuccess'] = True
        self.last_action = name
        evidence = {'valid': True, 'black': False, 'source_action': name, 'action_index': self.action_count,
                    'files': {'metadata': self.store.save_json(f'frame{self.action_count}.json', self.metadata)}}
        self.last_capture = {'metadata': copy.deepcopy(self.metadata), 'evidence': evidence}
        record = {'index': self.action_count, 'action': name, 'arguments': copy.deepcopy(kwargs),
            'before_metadata': before, 'after_metadata': copy.deepcopy(self.metadata), 'exception': None,
            'capture': copy.deepcopy(evidence)}
        self.last_action_reference = self.store.save_json(f'action{self.action_count}.json', record)
        self.query_log.append({'action': name, 'reference': self.last_action_reference})
        self.action_count += 1
        return self.controller.last_event
    def snapshot(self, key):
        result = {'key': key, 'scene': 'FloorPlan1', 'objects': copy.deepcopy(self.metadata['objects']),
            'poses': poses(self.metadata['objects']), 'agent': copy.deepcopy(self.agent),
            'worker_pid': 10, 'unity_pid': 20, 'last_action': self.last_action,
            'last_action_reference': self.last_action_reference, 'capture_reference': copy.deepcopy(self.last_capture['evidence'])}
        result['reference'] = self.store.save_json(f'snapshot{self.snapshot_count}.json', result)
        self.snapshot_count += 1
        return result
    def freeze(self):
        return frozen.RetryPlacementSimulator.freeze_render_pair(self, self.physical, 'X', 'pair')


def verified_fixture(delta=0.):
    sim = NativeFixture(delta)
    original = copy.deepcopy(sim.physical)
    result = sim.freeze()
    row = {'identities': {'anchor_name': 'X'}, 'frozen_render_version': frozen.FROZEN_RENDER_VERSION,
        'frozen_render_policy': copy.deepcopy(frozen.FROZEN_RENDER_POLICY),
        'physical_world_references': [w['reference'] for w in original],
        'frozen_render_reference': result['reference'],
        'frozen_render_readbacks': result['proof']['readbacks'],
        'nonanchor_cross_world_readback': result['proof']['nonanchor_cross_world_readback']}
    constructions = [{'realized_objects': w['objects']} for w in original]
    return sim, original, result, row, constructions


class FrozenRenderTests(unittest.TestCase):
    def test_only_anchor_requested_pose_changes_and_original_physics_stays(self):
        sim, original, result, row, constructions = verified_fixture()
        requested = result['proof']['requested_pose_lists']
        self.assertEqual(requested[0], original[0]['poses'])
        self.assertEqual(requested[1][0], requested[0][0])
        self.assertEqual(requested[1][2], original[1]['poses'][2])
        self.assertNotEqual(original[1]['objects'][0]['position'], original[0]['objects'][0]['position'])
        self.assertEqual(sim.physical, original)
        self.assertEqual(result['worlds'][1]['objects'][0]['position'], original[0]['objects'][0]['position'])
        self.assertEqual([x['action'] for x in sim.query_log],
            ['PausePhysicsAutoSim', 'SetObjectPoses', 'PhysicsSyncTransforms', 'Pass',
             'SetObjectPoses', 'PhysicsSyncTransforms', 'Pass'])
        physical, checked = verify.verify_frozen_render(row, result['worlds'], constructions, sim.store.read)
        self.assertEqual(physical, original)
        self.assertTrue(checked['passed'])

    def test_actual_readback_difference_is_reported_without_new_exact_pose_gate(self):
        sim, original, result, row, constructions = verified_fixture(.0001)
        self.assertTrue(result['passed'])
        cross = result['proof']['nonanchor_cross_world_readback']
        self.assertFalse(cross['all_pose_components_exact'])
        self.assertAlmostEqual(cross['max_position_delta_m'], .0001)
        _, checked = verify.verify_frozen_render(row, result['worlds'], constructions, sim.store.read)
        self.assertFalse(checked['nonanchor_cross_world_readback']['all_pose_components_exact'])
        row['nonanchor_cross_world_readback']['max_position_delta_m'] = 0.
        with self.assertRaisesRegex(ValueError, 'diagnostics'):
            verify.verify_frozen_render(row, result['worlds'], constructions, sim.store.read)

    def test_existing_one_cm_restore_bound_is_retained(self):
        sim, original, result, row, constructions = verified_fixture(.01001)
        self.assertFalse(result['passed'])
        with self.assertRaisesRegex(ValueError, 'unchanged restoration'):
            verify.verify_frozen_render(row, result['worlds'], constructions, sim.store.read)

    def test_proof_cannot_claim_original_w1_furniture_drift_was_frozen_request(self):
        sim, original, result, row, constructions = verified_fixture()
        proof = sim.store.values[result['reference']['file']]
        proof['requested_pose_lists'][1] = original[1]['poses']
        with self.assertRaisesRegex(ValueError, 'recipe'):
            verify.verify_frozen_render(row, result['worlds'], constructions, sim.store.read)

    def test_proof_cannot_insert_physics_or_camera_actions(self):
        for name in ('AdvancePhysicsStep', 'TeleportFull', 'UnpausePhysicsAutoSim'):
            sim, original, result, row, constructions = verified_fixture()
            ref = result['proof']['action_references'][2]['reference']
            sim.store.values[ref['file']]['action'] = name
            with self.assertRaisesRegex(ValueError, 'unexpected'):
                verify.verify_frozen_render(row, result['worlds'], constructions, sim.store.read)

    def test_hidden_rgb_rejection_and_visible_anchor_are_distinct_and_unknown_not_imputed(self):
        cases = [([0, 0], 4, 'anchor_hidden_but_prefix_rgb_diff_gt3'),
                 ([0, 1], 4, 'anchor_visible_at_v0_but_direct_evidence_failed'),
                 ([0, 0], None, 'neither_hidden_shared_prefix_nor_direct_evidence')]
        for masks, maximum, expected in cases:
            row = {'discard_reasons': ['neither_hidden_shared_prefix_nor_direct_evidence'],
                'prefix_measurement': {'prefix_edit_mask_pixels': [{'w0': masks[0], 'w1': masks[1]}],
                                       'prefix_max_abs_diff': maximum}}
            shard.split_prefix_rejection(row)
            self.assertEqual(row['discard_reasons'], [expected])

    def test_rotation_readback_preserves_euler_representation_difference(self):
        objects = make_objects()
        requested = poses(objects)
        objects[0]['rotation']['y'] = 359.999
        report = frozen.pose_readback(requested, objects)
        self.assertFalse(report['all_pose_components_exact'])
        self.assertAlmostEqual(report['max_rotation_component_abs_diff_degrees'], 359.999)
        self.assertAlmostEqual(report['max_rotation_shortest_component_diff_degrees'], .001)

    def test_actual_frozen_target_ratio_is_rechecked_by_production_builder(self):
        # Synthetic boundary fixture: genuine 5mm candidate drift could make
        # original physical w1 pass while its frozen render world fails 1.5.
        from src.envs import pair_v2_marked_policy as marked
        sim = NativeFixture()
        sim.metadata['objects'][2]['position']['x'] = .801
        sim.physical[1] = sim.snapshot('physical_w1_ratio_boundary')
        self.assertTrue(marked.closest_target(sim.physical[1]['objects'], ['A', 'B'], 'X')['passed'])
        native_results = []
        def construct(baseline, key, placements, require_agent=None):
            index = int(key.endswith('_w1'))
            result = {'snapshot': sim.physical[index], 'passed': True,
                'capture': {'original_physical_capture': index}, 'physical': {'original_physics': index},
                'successful_tokens_by_name': {'X': copy.deepcopy(placements[0]['spawn'])}, 'retry_records': {}}
            native_results.append(result)
            return result
        sim.construct_world = construct
        sim.position_agent = lambda *args: None
        sim.freeze_render_pair = lambda worlds, anchor, key: frozen.RetryPlacementSimulator.freeze_render_pair(sim, worlds, anchor, key)
        def point(x):
            return {'position': xyz(x), 'spawn': {'selected_position': xyz(x)}}
        proposal = {'mode': 'flip', 'w0': point(.2), 'w1': point(.801), 'projection_pool': {'pool': 'look'}}
        view = {'position': xyz(-1), 'yaw': 0., 'horizon': 0.}
        role = {'w0_source_side': 'a', 'w1_source_side': 'b'}
        row = {'pair_id': 'pair'}
        _, sim.construct_world = shard.construct_observer(sim, proposal, role, {}, row, frozen=True)
        identities = {'candidate_names': ['A', 'B'], 'anchor_name': 'X',
            'anchor_type': 'Vase', 'target_type': 'marked object'}
        with patch.object(shard.original_policy, 'closest_target', marked.closest_target), \
             patch.object(shard.pools, 'placement_alternatives', side_effect=lambda record, pool, side, selected, **kwargs: [selected]), \
             patch.object(shard.base, 'physical_measurement', return_value={'valid_physical': True}), \
             patch.object(shard, '_BASE_PHYSICAL', return_value={'valid_physical': True}), \
             patch.object(shard, 'augment_physical_result', side_effect=lambda result, raw: result):
            result = shard._BASE_BUILD(sim, {'scene': 'FloorPlan1'}, sim.physical[0], identities,
                                      proposal, view, 'pair', None, [], attempt_row=row)
        self.assertEqual(result['discard_reasons'], ['target_uniqueness_or_ratio'])
        self.assertFalse(result['target_measurements'][1]['passed'])
        self.assertEqual([item['physical'] for item in native_results],
                         [{'original_physics': 0}, {'original_physics': 1}])

    def test_restore_readback_proof_rejects_zeroed_actual_difference(self):
        sim, original, result, row, constructions = verified_fixture(.0001)
        requested = result['worlds'][0]
        measured = {'restored_snapshot_reference': result['worlds'][1]['reference']}
        frozen.record_restore_readback(sim, requested, measured)
        actual = verify.verify_frozen_restore_diagnostic(measured, requested, sim.store.read)
        self.assertGreater(actual['max_position_delta_m'], 0.)
        measured['frozen_pose_readback']['max_position_delta_m'] = 0.
        with self.assertRaisesRegex(ValueError, 'actual metadata'):
            verify.verify_frozen_restore_diagnostic(measured, requested, sim.store.read)

    def test_historical_source_cannot_silently_claim_frozen_policy(self):
        from src.tools.merge_pair_v2 import check_pair
        with self.assertRaisesRegex(ValueError, 'Historical pairs'):
            check_pair({'frozen_render_version': frozen.FROZEN_RENDER_VERSION}, '/synthetic/pair.json')

    def test_parent_and_scene_declare_same_explicit_frozen_contract(self):
        from src.tools.t01v2_marked_parallel import marked_tags
        parent = marked_tags()
        config = shard.enable_config({})
        for key in ('frozen_render_version', 'frozen_render_policy'):
            self.assertEqual(parent[key], config[key])
            self.assertEqual(shard.tags()[key], config[key])


if __name__ == '__main__':
    unittest.main()
