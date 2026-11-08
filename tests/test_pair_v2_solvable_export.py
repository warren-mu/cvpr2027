"""Synthetic native-record/image fixtures, never a Unity experiment."""
import copy
import io
import json
import unittest

from src.envs.pair_v2_solvable_export import capture_observation, digest, verify_look_solvability


def fixture(unavailable_neighbor=False):
    import numpy as np
    from PIL import Image
    from src.envs.pair_v2_solvable_policy import SOLVABILITY_VERSION, SOLVABILITY_POLICY, solvability_summary
    from src.envs.pair_v2_stable_policy import look_evidence
    values, blobs = {}, {}
    def store(name, value):
        path = '/synthetic/raw/' + name
        data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
        blobs[path] = data
        if not isinstance(value, bytes):
            values[path] = copy.deepcopy(value)
        return {'file': path, 'sha256': digest(data)}
    def point(x=0):
        return {'x': x, 'y': 0, 'z': 0}
    agent = {'position': point(), 'rotation': point(), 'horizon': 0, 'standing': True}
    worlds = [{'objects': [{'name': name, 'objectType': kind, 'objectId': name,
        'position': point(x), 'rotation': point(), 'pickupable': True, 'moveable': False}
        for name, kind, x in (('a', 'Mug', -.5), ('b', 'Mug', .5), ('x', 'Book', -.25 if index == 0 else .25))],
        'agent': copy.deepcopy(agent)} for index in range(2)]
    before = [{'objects': copy.deepcopy(world['objects']), 'agent': copy.deepcopy(agent),
               'lastActionSuccess': True} for world in worlds]
    restored = [{'restore_deterministic': True,
                 'restored_snapshot_reference': store(f'restored{index}.json', world)} for index, world in enumerate(worlds)]
    reachable = [point()] + ([point(.25)] if unavailable_neighbor else [])
    reach_ref = store('reachable.json', {'action': 'GetReachablePositions', 'exception': None,
        'after_metadata': {'lastActionSuccess': True, 'actionReturn': reachable}})
    actions = [{'kind': 'turn_left'}, {'kind': 'turn_right'}]
    domains = [{'world_index': index, 'restoration': copy.deepcopy(restored[index]),
                'reachable_positions': copy.deepcopy(reachable), 'reachable_reference': reach_ref,
                'actions': copy.deepcopy(actions), 'unavailable_moves': []} for index in range(2)]
    if unavailable_neighbor:
        for index in range(2):
            native = {'action': 'GetShortestPathToPoint', 'arguments': {'position': point(), 'target': point(.25), 'allowedError': .05},
                      'before_metadata': before[index], 'after_metadata': {'lastActionSuccess': False, 'actionReturn': None},
                      'exception': 'RuntimeError: native path unavailable'}
            ref = store(f'unavailable{index}.json', native)
            domains[index]['unavailable_moves'] = [{'position': point(.25), 'reason': 'native path failed', 'query_reference': ref,
                'query_outcome': {'action': native['action'], 'arguments': native['arguments'], 'exception': native['exception'],
                                  'last_action_success': False, 'action_return': None}}]
    identities = {'candidate_names': ['a', 'b'], 'anchor_name': 'x', 'target_type': 'Mug', 'anchor_type': 'Book'}
    instruction = 'Select the Mug closest to the Book.'
    report = {'solvability_version': SOLVABILITY_VERSION, 'solvability_policy': copy.deepcopy(SOLVABILITY_POLICY),
              'anchor_name': 'x', 'domains': domains, 'enumerated_actions': copy.deepcopy(actions), 'actions': [],
              'selected_action': 'turn_left'}
    captures, observations = [], {}
    for action_index, action in enumerate(actions):
        slots = []
        for index in range(2):
            visible = action_index == index
            rgb = np.full((255, 255, 3), 50, dtype=np.uint8)
            masks = {name: np.zeros((255, 255), bool) for name in ('a', 'b', 'x')}
            masks['a'][50:60, :25] = True
            masks['b'][60:70, :25] = True
            if visible:
                masks['x'][:10, :25] = True
                rgb[:10, :25] = 100
            metadata = {**copy.deepcopy(before[index]), 'sequenceId': action_index * 2 + index}
            label = f'a{action_index}_w{index}'
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format='PNG')
            image_ref = store(label + '.png', buffer.getvalue())
            buffer = io.BytesIO()
            np.savez_compressed(buffer, names=np.asarray(['a', 'b', 'x']), masks=np.stack([masks[name] for name in ('a', 'b', 'x')]))
            mask_ref = store(label + '.npz', buffer.getvalue())
            meta_ref = store(label + '.json', metadata)
            native_name = 'RotateLeft' if action_index == 0 else 'RotateRight'
            capture = {'files': {'metadata': meta_ref, 'rgb': image_ref, 'masks': mask_ref},
                'valid': True, 'black': False, 'rgb_shape': [255, 255, 3], 'rgb_sha256': digest(rgb.tobytes()),
                'sequence_id': metadata['sequenceId'], 'source_action': native_name,
                'action_index': action_index * 2 + index,
                'mask_pixels_by_name': {name: int(mask.sum()) for name, mask in masks.items()}}
            observed = capture_observation(capture, lambda ref: blobs[ref['file']], identities, instruction)
            observations[(action_index, index)] = observed
            buffer = io.BytesIO()
            Image.fromarray(observed['marked']).save(buffer, format='PNG')
            marked_ref = store(label + '_marked.png', buffer.getvalue())
            action_ref = store(label + '_action.json', {'index': capture['action_index'], 'action': native_name,
                'arguments': {'degrees': 45}, 'exception': None, 'before_metadata': before[index],
                'after_metadata': metadata, 'capture': capture})
            slots.append({'world_index': index, 'status': 'observed', 'anchor_pixels': 250 if visible else 0,
                'anchor_visible': visible, 'capture_reference': capture, 'action_reference': action_ref,
                'restoration': copy.deepcopy(restored[index]), 'executed_action': copy.deepcopy(action)})
            captures.append({'action_index': action_index, 'world_index': index, 'action': action['kind'],
                'capture_reference': capture, 'observation': {**{key: copy.deepcopy(observed[key]) for key in
                    ('metadata', 'marks', 'mark_labels_by_name', 'input_text', 'history')},
                    'files': {'rgb': image_ref, 'masks': mask_ref, 'marked': marked_ref}}})
        report['actions'].append({'action_index': action_index, 'action': action['kind'],
            'requested_action': copy.deepcopy(action), 'worlds': slots,
            'evidence': look_evidence(observations[(action_index, 0)], observations[(action_index, 1)], 'x')})
    report.update(solvability_summary(report))
    reviews = [{'world_index': index, 'action_index': index, 'action': actions[index]['kind']} for index in range(2)]
    row = {'identities': identities, 'instruction': instruction, 'observation_search': report, 'look_solvability': copy.deepcopy(report),
           'evidence': {'look_solvability': copy.deepcopy(report)}, 'look_review_views': reviews,
           'witness_actions': [['turn_left', 'select 1'], ['turn_left', 'turn_right', 'select 2']]}
    proof = {'schema_version': 1, 'search': copy.deepcopy(report), 'captures': captures, 'review_views': reviews}
    after = [copy.deepcopy(observations[(0, index)]) for index in range(2)]
    args = (row, worlds, after, proof, lambda ref: copy.deepcopy(values[ref['file']]), lambda ref: blobs[ref['file']])
    return args, values, blobs


class SolvableExportTests(unittest.TestCase):
    def test_each_world_can_see_anchor_via_different_native_actions(self):
        args, _, _ = fixture()
        result = verify_look_solvability(*args)
        self.assertTrue(result['passed'])
        self.assertEqual(result['summary']['anchor_visible_actions_w0'], ['turn_left'])
        self.assertEqual(result['summary']['anchor_visible_actions_w1'], ['turn_right'])

    def test_selected_after_mask_cannot_inflate_evidence_with_identical_public_rgb(self):
        args, _, _ = fixture()
        args[2][1]['masks_by_name']['x'][:20, :20] = True
        with self.assertRaises(ValueError):
            verify_look_solvability(*args)

    def test_selected_after_metadata_and_native_action_arguments_are_bound(self):
        args, values, _ = fixture()
        values['/synthetic/raw/a0_w0_action.json']['arguments']['degrees'] = 90
        with self.assertRaises(ValueError):
            verify_look_solvability(*args)
        args, _, _ = fixture()
        args[2][0]['metadata']['objects'][0]['position']['x'] += .01
        with self.assertRaises(ValueError):
            verify_look_solvability(*args)

    def test_missing_capture_does_not_become_zero_or_an_observed_action(self):
        args, _, _ = fixture()
        args[3]['captures'].pop()
        with self.assertRaises(ValueError):
            verify_look_solvability(*args)

    def test_domain_native_unavailable_proof_can_be_complete_without_fabricated_masks(self):
        args, _, _ = fixture(unavailable_neighbor=True)
        self.assertTrue(verify_look_solvability(*args)['passed'])

    def test_wrong_world_path_proof_and_unknown_successful_payload_are_rejected(self):
        args, values, _ = fixture(unavailable_neighbor=True)
        values['/synthetic/raw/unavailable0.json']['before_metadata'] = copy.deepcopy(values['/synthetic/raw/unavailable1.json']['before_metadata'])
        with self.assertRaises(ValueError):
            verify_look_solvability(*args)
        args, values, _ = fixture(unavailable_neighbor=True)
        values['/synthetic/raw/unavailable0.json']['after_metadata']['lastActionSuccess'] = True
        values['/synthetic/raw/unavailable0.json']['exception'] = None
        with self.assertRaises(ValueError):
            verify_look_solvability(*args)

    def test_selected_common_action_must_match_both_public_witness_plans(self):
        args, _, _ = fixture()
        args[0]['witness_actions'][1][0] = 'turn_right'
        with self.assertRaises(ValueError):
            verify_look_solvability(*args)


if __name__ == '__main__':
    unittest.main()
