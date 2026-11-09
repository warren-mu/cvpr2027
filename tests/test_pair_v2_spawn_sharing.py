"""CPU equivalence oracle: call the real old method without constructing Unity."""
import copy
import hashlib
import json
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.envs import pair_v2_spawn_sharing as sharing
from src.envs import pair_v2_simulator as engine
from src.envs import pair_v2_native_retry as retry


class Simulator:
    def __init__(self, values):
        self.scene = 'FloorPlan203'
        self.config, self.spawn_tokens, self.calls = sharing.fields(), {}, []
        self.values = values
        self.reference = {'file': '/scratch/原生/"query".json', 'sha256': 'a' * 64,
            'nested': {'tuple': (1, -0.0), 'list': [True, None, 'é\\x']}}
        self.obj = {'name': 'Café_柜子', 'objectId': 'native|柜子', 'receptacle': True,
            'position': {'x': -0.0, 'y': 1e-30, 'z': 1e20}}
        self.controller = SimpleNamespace(last_event=SimpleNamespace(metadata={'objects': [self.obj]}))
        self.last_action_reference = None
        self.deadline_hook = lambda: None

    def check_deadline(self):
        self.deadline_hook()

    def action(self, name, **arguments):
        self.check_deadline()
        self.calls.append((name, arguments))
        self.last_action_reference = copy.deepcopy(self.reference)
        return SimpleNamespace(metadata={'actionReturn': self.values})


def points():
    return [{'x': -0.0, 'y': 1e-30, 'z': 2.0},
            {'x': 0, 'y': -3.25, 'z': 1e20},
            {'x': 1, 'y': 2, 'z': 3, 'extra': {'array': [1, (2, '雪')]}},
            {'x': -0.0, 'y': 1e-30, 'z': 2.0}]


class SpawnSharingTests(unittest.TestCase):
    def test_full_tokens_registered_hashes_and_queries_match_real_old_method(self):
        old, new = Simulator(points()), Simulator(points())
        expected = sharing._ORIGINAL_SPAWN(old, old.obj['name'])
        observed = sharing.spawn_points(new, new.obj['name'])
        self.assertEqual(expected, observed)
        self.assertEqual(engine.encoded(expected), engine.encoded(observed))
        self.assertEqual(engine.encoded(old.spawn_tokens), engine.encoded(new.spawn_tokens))
        self.assertEqual(old.calls, new.calls)
        for token in observed:
            validated = engine.validate_spawn({'spawn': token}, new.scene)
            digest = hashlib.sha256(engine.encoded(validated)).hexdigest()
            self.assertIn(digest, new.spawn_tokens)
            # A serialized/deserialized genuine token is admitted too; there is
            # no hidden object-identity capability or lazy authorization gate.
            roundtrip = json.loads(engine.encoded(token))
            self.assertIn(hashlib.sha256(engine.encoded(roundtrip)).hexdigest(), new.spawn_tokens)

    def test_default_json_framing_corners_are_byte_exact(self):
        values = ({'x': -0.0, 'y': 0.0, 'z': -1e-20}, {'x': 0, 'y': 1e20, 'z': 2.5})
        for coordinate_container in (list(values), tuple(values)):
            template = {'action': 'quoted"\\é', 'coordinates': coordinate_container,
                'selected_position': None, 'source': '雪/é', 'success': True,
                'nested': {'z': None, 'a': [False, (1, 2)]}}
            prefix, suffix = sharing.canonical_parts(template)
            for selected in values:
                complete = {**template, 'selected_position': selected}
                self.assertEqual(prefix + engine.encoded(selected) + suffix, engine.encoded(complete))
                digest = hashlib.sha256(prefix); digest.update(engine.encoded(selected)); digest.update(suffix)
                self.assertEqual(digest.hexdigest(), hashlib.sha256(engine.encoded(complete)).hexdigest())

    def test_coordinate_tree_detaches_from_native_metadata_and_shares_across_copies(self):
        sim = Simulator(points()); rows = sharing.spawn_points(sim, sim.obj['name'])
        shared = rows[0]['coordinates']
        self.assertIs(copy.deepcopy(shared), shared)
        self.assertTrue(all(token['coordinates'] is shared for token in rows))
        self.assertTrue(all(token['coordinates'] is shared for token in sim.spawn_tokens.values()))
        original = engine.encoded(shared)
        sim.values[0]['x'] = 98
        self.assertEqual(engine.encoded(shared), original)
        with self.assertRaises(TypeError): shared.append({'x': 1, 'y': 1, 'z': 1})
        with self.assertRaises(TypeError): shared[0]['x'] = 99
        self.assertEqual(engine.encoded(pickle.loads(pickle.dumps(shared))), original)

    def test_mutation_never_registers_a_new_caller_digest(self):
        sim = Simulator(points()); rows = sharing.spawn_points(sim, sim.obj['name'])
        original_keys = set(sim.spawn_tokens)
        token = copy.deepcopy(rows[0]); token['query_reference']['sha256'] = 'b' * 64
        # The original schema check alone accepts the valid native point, but
        # its unchanged full-token registry guard rejects the modified header.
        validated = engine.validate_spawn({'spawn': token}, sim.scene)
        self.assertNotIn(hashlib.sha256(engine.encoded(validated)).hexdigest(), sim.spawn_tokens)
        self.assertEqual(set(sim.spawn_tokens), original_keys)
        token = copy.deepcopy(rows[0]); token['selected_position']['x'] = 123
        with self.assertRaises(ValueError): engine.validate_spawn({'spawn': token}, sim.scene)
        # Python deliberately exposes base-class mutators on subclasses. Even
        # bypassing read-only methods must not bypass the fresh full SHA guard.
        dict.__setitem__(rows[0]['coordinates'][0], 'x', 99)
        validated = engine.validate_spawn({'spawn': rows[1]}, sim.scene)
        self.assertNotIn(hashlib.sha256(engine.encoded(validated)).hexdigest(), sim.spawn_tokens)

    def test_shared_array_is_encoded_once_during_registration_not_once_per_point(self):
        sim = Simulator([{'x': float(index), 'y': 0., 'z': 1.} for index in range(25)])
        encoded = engine.encoded
        arrays = []
        def watched(value):
            if isinstance(value, sharing.ReadOnlyList): arrays.append(value)
            return encoded(value)
        with patch.object(engine, 'encoded', side_effect=watched):
            rows = sharing.spawn_points(sim, sim.obj['name'])
        self.assertEqual(arrays, [rows[0]['coordinates']])
        self.assertEqual(len(sim.spawn_tokens), len(rows))

    def test_original_deadline_interrupts_between_points_and_no_extra_query_occurs(self):
        sim = Simulator(points())
        def stop_after_first_registration():
            if sim.spawn_tokens: raise TimeoutError('original soft deadline')
        sim.deadline_hook = stop_after_first_registration
        with self.assertRaisesRegex(TimeoutError, 'original soft deadline'):
            sharing.spawn_points(sim, sim.obj['name'])
        self.assertEqual(len(sim.spawn_tokens), 1)
        self.assertEqual(sim.calls, [('GetSpawnCoordinatesAboveReceptacle', {'objectId': sim.obj['objectId'], 'anywhere': True})])

    def test_empty_invalid_and_nonreceptacle_inputs_keep_native_contract(self):
        for values, error in (({}, RuntimeError), ([{'x': float('nan'), 'y': 0, 'z': 1}], ValueError)):
            for function in (sharing._ORIGINAL_SPAWN, sharing.spawn_points):
                sim = Simulator(values)
                with self.assertRaises(error): function(sim, sim.obj['name'])
        sim = Simulator([])
        self.assertEqual(sharing.spawn_points(sim, sim.obj['name']), [])
        self.assertEqual(sim.spawn_tokens, {})
        sim.obj['receptacle'] = False
        with self.assertRaises(ValueError): sharing.spawn_points(sim, sim.obj['name'])

    def test_opt_in_restores_old_methods_and_never_patches_placement_validation(self):
        before = engine.V2Simulator.spawn_points, engine.validate_spawn, retry.place_native_with_retries
        with sharing.installed():
            self.assertIs(engine.V2Simulator.spawn_points, sharing.spawn_points)
            self.assertIs(engine.validate_spawn, before[1])
            self.assertIs(retry.place_native_with_retries, before[2])
            sim = Simulator(points()); sim.config = {}
            with self.assertRaises(ValueError): sharing.spawn_points(sim, sim.obj['name'])
        self.assertEqual(before, (engine.V2Simulator.spawn_points, engine.validate_spawn, retry.place_native_with_retries))


if __name__ == '__main__':
    unittest.main()
