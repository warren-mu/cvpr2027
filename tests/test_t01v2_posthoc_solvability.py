"""CPU synthetic evidence only; no environment or scientific measurements."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from src.tools.t01v2_posthoc_solvability import Archive, certify_selected_replay


def fixture(counts=(250, 250), move=False):
    import numpy as np
    from PIL import Image
    blobs, values = {}, {}
    def save(name, value):
        data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
        path = '/synthetic/' + name
        blobs[path] = data
        if not isinstance(value, bytes):
            values[path] = copy.deepcopy(value)
        return {'file': path, 'sha256': hashlib.sha256(data).hexdigest()}
    point = lambda x=0: {'x': x, 'y': 0, 'z': 0}
    agent = {'position': point(), 'rotation': point(), 'horizon': 0, 'standing': True}
    worlds, replays = [], []
    action_text = 'move 1' if move else 'turn_left'
    row = {'kind': 'look_flip', 'identities': {'candidate_names': ['a', 'b'], 'anchor_name': 'x'},
           'instruction': 'Select the Mug closest to the Book.',
           'witness_actions': [[action_text, 'select 1'], [action_text, 'select 2']]}
    for index, count in enumerate(counts):
        objects = [{'name': name, 'objectId': name, 'objectType': kind, 'position': point(x),
                    'rotation': point(), 'moveable': False, 'pickupable': True}
                   for name, kind, x in [('a', 'Mug', -.5), ('b', 'Mug', .5), ('x', 'Book', -.2 if index == 0 else .2)]]
        world = {'objects': objects, 'agent': copy.deepcopy(agent), 'worker_pid': 100, 'unity_pid': 101}
        worlds.append(world)
        restored = save(f'w{index}/restored.json', world)
        before = {'objects': copy.deepcopy(objects), 'agent': copy.deepcopy(agent), 'lastActionSuccess': True}
        query_ref = None
        if move:
            query_after = {**copy.deepcopy(before), 'actionReturn': {'status': 'PathComplete', 'corners': [point(), point(.25)]}}
            query_ref = save(f'w{index}/action_000005.json', {'action': 'GetShortestPathToPoint', 'exception': None,
                'arguments': {'position': point(), 'target': point(.25), 'allowedError': .05},
                'before_metadata': before, 'after_metadata': query_after})
            before = query_after
        metadata = {**copy.deepcopy(before), 'sequenceId': index + 50}
        rgb = np.full((255, 255, 3), 50, np.uint8)
        masks = np.zeros((3, 255, 255), bool)
        masks[0, 50:60, :25] = True
        masks[1, 60:70, :25] = True
        masks[2].flat[:count] = True
        png, npz = io.BytesIO(), io.BytesIO()
        Image.fromarray(rgb).save(png, format='PNG')
        np.savez_compressed(npz, names=np.asarray(['a', 'b', 'x']), masks=masks)
        capture = {'files': {'metadata': save(f'w{index}/meta.json', metadata),
                            'rgb': save(f'w{index}/rgb.png', png.getvalue()), 'masks': save(f'w{index}/mask.npz', npz.getvalue())},
            'valid': True, 'black': False, 'rgb_shape': [255, 255, 3], 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
            'sequence_id': metadata['sequenceId'], 'action_index': 6, 'source_action': 'TeleportFull' if move else 'RotateLeft',
            'mask_pixels_by_name': dict(zip(['a', 'b', 'x'], [250, 250, count]))}
        native = {'index': 6, 'action': capture['source_action'], 'exception': None, 'before_metadata': before,
            'after_metadata': metadata, 'capture': capture,
            'arguments': ({'position': point(.25), 'rotation': point(), 'horizon': 0, 'standing': True,
                           'forceAction': False} if move else {'degrees': 45})}
        action_ref = save(f'w{index}/action_000006.json', native)
        raw_index = save(f'w{index}/index.json', {'files': [query_ref] if move else []})
        replays.append({'status': 'completed', 'cleanup_errors': [], 'error': None,
            'process_id': 200 + index, 'unity_pid': 300 + index,
            'planned_actions': row['witness_actions'][index], 'executed_actions': row['witness_actions'][index],
            'restoration': {'restored_snapshot_reference': restored}, 'raw_evidence': raw_index,
            'captures': [None, capture], 'actions': [{'action_reference': action_ref, 'success': True}]})
    return (row, worlds, replays, lambda ref: copy.deepcopy(values[ref['file']]), lambda ref: blobs[ref['file']]), values, blobs


class PosthocSolvabilityTests(unittest.TestCase):
    def test_positive_existence_never_claims_catalog_or_full_compliance(self):
        args, _, _ = fixture()
        result = certify_selected_replay(*args)
        self.assertIs(result['solvability'], True)
        self.assertEqual(result['mode'], 'legacy_selected_action_positive')
        self.assertFalse(result['complete_catalog'])
        self.assertEqual(result['other_actions'], 'unknown_not_enumerated')
        self.assertIsNone(result['full_metrics_spec_compliance'])

    def test_invisible_one_world_is_unknown_not_negative(self):
        args, _, _ = fixture((250, 0))
        result = certify_selected_replay(*args)
        self.assertIsNone(result['solvability'])
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['worlds'][1]['anchor_pixels'], 0)
        self.assertNotIn('look_not_solvable_w1', result['reasons'])

    def test_249_boundary_is_unknown(self):
        args, _, _ = fixture((249, 250))
        self.assertIsNone(certify_selected_replay(*args)['solvability'])

    def test_changed_native_degrees_or_wrong_world_cannot_certify(self):
        for alteration in ('degrees', 'world'):
            args, values, _ = fixture()
            record = values[args[2][0]['actions'][0]['action_reference']['file']]
            if alteration == 'degrees':
                record['arguments']['degrees'] = 90
            else:
                record['before_metadata']['objects'] = copy.deepcopy(args[1][1]['objects'])
            with self.assertRaises(ValueError):
                certify_selected_replay(*args)

    def test_changed_mask_metadata_binding_or_missing_capture_fails_closed(self):
        args, values, _ = fixture()
        record = values[args[2][0]['actions'][0]['action_reference']['file']]
        record['after_metadata']['sequenceId'] += 1
        with self.assertRaises(ValueError):
            certify_selected_replay(*args)
        args, _, _ = fixture()
        args[2][1]['captures'].pop()
        with self.assertRaises(IndexError):
            certify_selected_replay(*args)

    def test_real_move_path_without_original_domain_mapping_remains_unknown(self):
        args, _, _ = fixture(move=True)
        result = certify_selected_replay(*args)
        self.assertTrue(result['selected_action_observed_visible_both'])
        self.assertFalse(result['selected_action_domain_verified'])
        self.assertIsNone(result['solvability'])

    def test_archive_hash_and_scope_and_reused_proof_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, evidence = root / 'raw', root / 'evidence'
            raw.mkdir(); evidence.mkdir()
            path = raw / 'actual.json'
            path.write_bytes(b'{"measured":true}')
            ref = {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            budget = {'bytes': 0}
            archive = Archive(evidence, raw, root, budget)
            self.assertEqual(archive.json(ref), {'measured': True})
            archive.used.clear()
            archive.read(ref)
            self.assertEqual(len(archive.used), 1)
            self.assertEqual(budget['bytes'], path.stat().st_size)
            path.write_bytes(b'{"measured":false}')
            with self.assertRaises(ValueError):
                archive.read(ref)
