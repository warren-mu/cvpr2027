"""Handmade CPU tables; these fixtures never certify THOR rendering."""
from dataclasses import replace
import copy
import hashlib
import tempfile
from pathlib import Path
import unittest

from src.envs.pair_core import ViewState
from src.envs.paired_env import RGBFrame, table_environment
from src.envs.pair_table import (PairTable, RenderedView, TableObservationBackend,
    classify_views, label_assignment, mark_view)
from src.envs.pair_table_export import build_table, load_table, save_table
from src.envs.pair_table_checks import (authorize_tables, check_pair, min_solution_turns,
                                      replay_witness, visible_first_actions, export_cost_estimate)
from src.envs.pair_table_checks import verify_cached_binding, world_metadata_matches, verify_agent_setup


def mask(count, start=0):
    value = bytearray(255 * 255)
    value[start:start + count] = b'\x01' * count
    return bytes(value)


class FixtureRenderer:
    def __init__(self, *, mismatch=False, leak=False, missing=False, distant=False):
        self.mismatch, self.leak, self.missing, self.distant = mismatch, leak, missing, distant
        self.order = []

    def domain(self, world, state):
        self.order.append(('domain', world, state.heading_index))
        moves = []
        if self.mismatch and world == 1:
            moves = [{'index': 1, 'position': {'x': .25, 'y': 0, 'z': 0},
                      'verified': True, 'path_reference': {'fixture': True}}]
        return {'complete': True, 'moves': moves, 'evidence': {'synthetic': True}}

    def render_pair(self, state):
        self.order.append(('render_pair', state.heading_index))
        views = []
        for world in (0, 1):
            # w0 sees X looking right; w1 sees X looking left. Other states are shared.
            count = 250 if state.heading_index == (1 if world == 0 else 7) else 0
            delta = (20 if state.heading_index in (1, 7) else (4 if self.leak else 3)) if world else 0
            masks = {'X': mask(count, 1000), 'A': mask(300), 'B': mask(300, 500)}
            if self.missing:
                del masks['X']
            views.append(RenderedView(RGBFrame(bytes([30 + delta, 40, 50]) * (255 * 255), 255, 255),
                masks, {'A': 1.5001 if self.distant else 1.5, 'B': 1, 'X': 2},
                {'position': {'x': 0, 'y': 0, 'z': 0}, 'heading': state.heading_index},
                'synthetic-process', f'capture:{state.heading_index}:{world}', {'synthetic': True}, True))
        return tuple(views)


def table(**options):
    return build_table(FixtureRenderer(**options), pair_id='pair_fixture', scene='FloorPlan1',
        kind='look_flip', instruction='Select the cup beside the kettle.', anchor_name='X',
        candidate_names=('A', 'B'), target_names=('A', 'B'),
        initial_position={'x': 0, 'y': 0, 'z': 0}, initial_heading=0,
        source={'synthetic': True})


class TableTests(unittest.TestCase):
    def env(self, value=None):
        value = value or table()
        return table_environment({value.pair_id: value}, allow_synthetic=True)

    def test_shared_inputs_fork_and_return_preserve_tau(self):
        value = table()
        left, right = self.env(value), self.env(value)
        self.assertEqual(left.reset(value.pair_id, 'w0'), right.reset(value.pair_id, 'w1'))
        self.assertIsNone(left.info['tau'])
        a, _, _, _ = left.step('turn_right')
        b, _, _, _ = right.step('turn_right')
        self.assertNotEqual(a.image, b.image)
        self.assertEqual(left.info['tau'], 1)
        self.assertEqual(right.info['tau'], 1)
        self.assertEqual(left.info['x_seen_step'], 1)
        self.assertIsNone(right.info['x_seen_step'])
        a, _, _, _ = left.step('turn_left')
        b, _, _, _ = right.step('turn_left')
        self.assertEqual(a, b)
        self.assertEqual(left.info['tau'], 1)
        self.assertEqual(left.info['x_seen_step'], 1)
        self.assertIsNone(right.info['x_seen_step'])

    def test_labels_require_actual_visible_success_eligibility(self):
        value = table(distant=True)
        env = self.env(value)
        observation = env.reset(value.pair_id, 'w0')
        self.assertEqual([m.label for m in observation.marks], [value.labels['B']])
        self.assertNotIn(f'select {value.labels["A"]}', observation.text)
        self.assertEqual(label_assignment('pair_fixture', ('A', 'B')),
                         label_assignment('pair_fixture', ('B', 'A')))
        _, reward, done, info = env.step(f'select {value.labels["A"]}')
        self.assertEqual(reward, 0)
        self.assertTrue(done)
        self.assertFalse(info['success'])

    def test_correct_reward_and_fifth_turn_observation_no_new_state(self):
        value = table()
        env = self.env(value)
        env.reset(value.pair_id, 'w0')
        _, reward, done, info = env.step(f'select {value.labels["A"]}')
        self.assertEqual((reward, done, info['success']), (1, True, True))
        env.reset(value.pair_id, 'w0')
        for _ in range(4):
            observation, reward, done, _ = env.step('turn_right')
            self.assertEqual(reward, -.05)
            self.assertFalse(done)
        self.assertNotIn('"turn_right"', observation.text)
        self.assertEqual(env.frame.observation_actions, ())
        before = env.info['state_id']
        _, reward, done, info = env.step('turn_right')
        self.assertEqual((reward, done, info['terminal_reward']), (-.05, True, 0))
        self.assertEqual(info['state_id'], before)
        with self.assertRaises(RuntimeError):
            env.step('select 1')

    def test_invalid_output_consumes_turn_not_state_or_extra_penalty(self):
        env = self.env()
        env.reset('pair_fixture', 'w0')
        before = env.info['state_id']
        for _ in range(5):
            _, reward, _, _ = env.step('not an action')
            self.assertEqual(reward, 0)
        self.assertTrue(env.done)
        self.assertEqual(before, env.info['state_id'])

    def test_belief_bfs_requires_own_seen_then_selection(self):
        value = table()
        self.assertEqual(min_solution_turns(value), 4)
        self.assertEqual(visible_first_actions(value, 0), ['turn_right'])
        self.assertEqual(visible_first_actions(value, 1), ['turn_left'])
        # Only seeing the absence of X in the other branch is not own seen.
        altered = dict(value.states)
        for key, state in altered.items():
            if state.anchor_pixels[1]:
                capture = replace(state.captures[1], masks={**state.captures[1].masks, 'X': mask(249)})
                measured = classify_views((state.captures[0], capture), 'X', kind='look_flip')
                altered[key] = replace(state, captures=(state.captures[0], capture), anchor_pixels=measured['anchor_pixels'])
        value.states = altered
        self.assertIsNone(min_solution_turns(value))

    def test_export_rejects_graph_mismatch_leak_and_missing_mask(self):
        for option, error in [('mismatch', 'graph_mismatch'), ('leak', 'non_anchor_divergent'), ('missing', 'Missing mask')]:
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, error):
                table(**{option: True})

    def test_direct_hidden_divergence_forks_counts_and_preserves_each_world(self):
        class DirectRenderer(FixtureRenderer):
            def render_pair(self, state):
                captures = super().render_pair(state)
                if state.heading_index == 0:
                    captures = tuple(replace(view, masks={**view.masks, 'X': mask(250, 1000)})
                                     for view in captures)
                return captures
        value = build_table(DirectRenderer(leak=True), pair_id='direct_fixture', scene='FloorPlan1',
            kind='direct_stay', instruction='Select the marked object closest to the kettle.',
            anchor_name='X', candidate_names=('A', 'B'), target_names=('A', 'A'),
            initial_position={'x': 0, 'y': 0, 'z': 0}, initial_heading=0, source={'synthetic': True})
        self.assertEqual(value.export_measurements['non_anchor_divergent_states'], 5)
        source = {'pair_id': value.pair_id, 'witnesses': [
            {'actions': [{'kind': 'select', 'target_name': 'A'}]} for _ in (0, 1)]}
        checked = check_pair(value, source)
        self.assertTrue(checked['passed'])
        self.assertEqual(checked['non_anchor_divergent_states'], 5)
        for state in value.states.values():
            if state.anchor_pixels == (0, 0):
                self.assertFalse(state.shared)
                self.assertGreater(state.frame_max_abs_diff, 3)
        with tempfile.TemporaryDirectory() as temporary:
            reference = save_table(value, Path(temporary) / 'table')
            loaded = load_table(reference['file'], expected_sha256=reference['sha256'])
            left, right = self.env(loaded), self.env(loaded)
            left.reset(loaded.pair_id, 'w0'); right.reset(loaded.pair_id, 'w1')
            for _ in range(2):
                observation0, _, _, _ = left.step('turn_right')
                observation1, _, _, _ = right.step('turn_right')
            self.assertEqual(left.info['anchor_pixels_w0'], 0)
            self.assertEqual(left.info['anchor_pixels_w1'], 0)
            self.assertNotEqual(observation0.image, observation1.image)
            self.assertEqual((left.info['tau'], right.info['tau']), (0, 0))
        # Identical captures remain forbidden for Look regardless of Flip/Stay.
        hidden = next(state for state in value.states.values() if state.anchor_pixels == (0, 0))
        for kind in ('look_flip', 'look_stay'):
            with self.assertRaisesRegex(ValueError, 'non_anchor_divergent'):
                classify_views(hidden.captures, 'X', kind=kind)

    def test_kind_is_explicit_and_direct_noise_boundary_still_shares(self):
        captures = FixtureRenderer().render_pair(ViewState('v', 0))
        for kind in ('direct_flip', 'direct_stay'):
            self.assertTrue(classify_views(captures, 'X', kind=kind)['shared'])
        with self.assertRaisesRegex(ValueError, 'pair kind'):
            classify_views(captures, 'X', kind='direct_unknown')

    def test_d4_graph_and_real_provenance_not_certified_by_mock(self):
        value = table()
        self.assertEqual({s.view.heading_index for s in value.states.values()}, set(range(8)))
        self.assertFalse(value.export_verified)
        with self.assertRaisesRegex(ValueError, 'CPU-test'):
            TableObservationBackend({value.pair_id: value})
        with self.assertRaisesRegex(ValueError, 'stop all downstream'):
            authorize_tables([value], pair_checks={}, cache_report={'downstream_allowed': False})

    def test_native_pair_camera_and_render_identity_required(self):
        captures = FixtureRenderer().render_pair(ViewState('v', 0))
        with self.assertRaises(ValueError):
            classify_views((captures[0], replace(captures[1], process_id='other')), 'X', kind='look_flip')
        with self.assertRaises(ValueError):
            classify_views((captures[0], replace(captures[1], agent={'heading': 1})), 'X', kind='look_flip')

    def test_explicit_world_relabel_maps_targets_and_temporal_proof_not_pixels(self):
        value = table()
        env = table_environment({value.pair_id: value}, allow_synthetic=True,
                                world_orders={value.pair_id: [1, 0]})
        env.reset(value.pair_id, 'w0')
        env.step('turn_right')
        self.assertEqual(env.info['world_id'], 'w0')
        self.assertEqual(env.info['original_world_id'], 'w1')
        self.assertEqual(env.info['target_name'], 'B')
        self.assertIsNone(env.info['x_seen_step'])
        self.assertEqual(env.info['table_temporal_proof']['original_world_index'], 1)
        self.assertTrue(env.info['table_temporal_proof']['synthetic'])

    def test_immutable_storage_roundtrip_hash_and_no_overwrite(self):
        value = table()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'pair'
            reference = save_table(value, directory)
            loaded = load_table(reference['file'], expected_sha256=reference['sha256'])
            self.assertEqual(loaded.labels, value.labels)
            self.assertEqual(self.env(loaded).reset(value.pair_id, 'w1'), self.env(value).reset(value.pair_id, 'w0'))
            with self.assertRaises(FileExistsError):
                save_table(value, directory)
            first = next(directory.glob('*.rgb'))
            first.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'payload changed'):
                load_table(reference['file'], expected_sha256=reference['sha256'])

    def test_witness_target_name_not_label_and_long_witness_not_false_failure(self):
        value = table()
        result = replay_witness(value, 0, {'actions': [{'kind': 'turn_right'}, {'kind': 'select', 'target_name': 'A'}]})
        self.assertTrue(result['passed'])
        self.assertEqual(result['trace'][-1]['action'], f'select {value.labels["A"]}')
        self.assertIsNone(replay_witness(value, 0, {'actions': [{'kind': 'turn_left'}] * 6})['passed'])

    def test_check_pair_cannot_replace_missing_t01_visibility(self):
        value = table()
        source = {'pair_id': value.pair_id, 'selected_action': 'turn_right',
            'anchor_visible_actions_w0': ['turn_right'], 'anchor_visible_actions_w1': ['turn_left'],
            'witnesses': [{'actions': [{'kind': 'select', 'target_name': name}]} for name in ('A', 'B')]}
        self.assertTrue(check_pair(value, source)['passed'])
        source.pop('anchor_visible_actions_w1')
        self.assertIn('t01_first_step_visibility_mismatch_w1', check_pair(value, source)['reasons'])

    def test_direct_does_not_require_an_extra_observation(self):
        value = table()
        value.kind = 'direct_flip'
        start = value.states[value.initial_state]
        captures = tuple(replace(c, masks={**c.masks, 'X': mask(250, 1000)}) for c in start.captures)
        measurement = classify_views(captures, 'X', kind='direct_flip')
        value.states[value.initial_state] = replace(start, captures=captures,
            shared=False, anchor_pixels=measurement['anchor_pixels'])
        self.assertEqual(min_solution_turns(value), 1)
        env = self.env(value)
        env.reset(value.pair_id, 'w1')
        self.assertEqual((env.info['tau'], env.info['x_seen_step']), (0, 0))

    def test_move_witness_resolves_destination_not_old_inverse_index(self):
        value = table()
        initial = value.states[value.initial_state]
        destination_view = ViewState('second', initial.view.heading_index)
        destination_id = 'second:h0'
        value.positions['second'] = {'x': .25, 'y': 0, 'z': 0}
        value.states[destination_id] = replace(initial, state_id=destination_id, view=destination_view,
                                               transitions={}, neighbors=())
        value.states[value.initial_state] = replace(initial,
            transitions={**initial.transitions, 'move 2': destination_id}, neighbors=((2, 'right', 'second'),))
        replay = replay_witness(value, 0, {'actions': [
            {'kind': 'move', 'index': 1, 'position': value.positions['second']},
            {'kind': 'select', 'target_name': 'A'}]})
        self.assertTrue(replay['passed'])
        self.assertEqual(replay['trace'][0]['action'], 'move 2')

    def test_no_cost_estimate_from_missing_or_synthetic_measurements(self):
        self.assertIsNone(export_cost_estimate([])['estimate'])
        row = {'pair_id': 'fixture', 'export_elapsed_seconds': 1, 'allocated_gpus': 1,
               'stored_bytes': 10, 'state_count': 2, 'export_completed': True, 'synthetic_only': True}
        self.assertIsNone(export_cost_estimate([row])['estimate'])
        row.pop('allocated_gpus')
        with self.assertRaises(ValueError):
            export_cost_estimate([row])

    def test_cached_binding_rejects_same_rgb_with_other_mask_or_labels(self):
        value = table()
        state = value.states[value.initial_state]
        state = replace(state, captures=tuple(replace(c, source={'snapshot_sha256': str(w) * 64})
                                              for w, c in enumerate(state.captures)))
        value.states[value.initial_state] = state
        row = {'pair_id': value.pair_id, 'kind': value.kind, 'state_id': value.initial_state, 'worlds': {}}
        for world, capture in enumerate(state.captures):
            row['worlds'][f'w{world}'] = {'cached_source': {'rgb_sha256': capture.rgb.sha256,
                'render_id': capture.render_id, 'snapshot_sha256': capture.source['snapshot_sha256']},
                'cached_labels': dict(value.labels), 'masks': {name: {'cached_mask_sha256': hashlib.sha256(mask).hexdigest()}
                    for name, mask in capture.masks.items()}}
        verify_cached_binding(value, row)
        with self.assertRaisesRegex(ValueError, 'pair kind'):
            verify_cached_binding(value, {**row, 'kind': 'direct_flip'})
        altered = copy.deepcopy(row)
        altered['worlds']['w1']['masks']['X']['cached_mask_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'different masks'):
            verify_cached_binding(value, altered)
        altered = copy.deepcopy(row)
        altered['worlds']['w1']['cached_labels'] = {}
        with self.assertRaisesRegex(ValueError, 'candidate labels'):
            verify_cached_binding(value, altered)

    def test_world_and_camera_bindings_do_not_accept_other_state(self):
        obj = {'name': 'X', 'position': {'x': 0, 'y': 0, 'z': 0},
               'objectType': 'Kettle', 'moveable': True, 'pickupable': True}
        snapshot = {'objects': [obj], 'poses': [{'objectName': 'X', 'position': obj['position']}]}
        world_metadata_matches({'objects': [obj]}, snapshot)
        moved = copy.deepcopy(obj)
        moved['position']['x'] = .5
        with self.assertRaisesRegex(ValueError, 'declared world'):
            world_metadata_matches({'objects': [moved]}, snapshot)
        value = table()
        state = value.states[value.initial_state]
        action = {'action': 'TeleportFull', 'exception': None,
            'arguments': {'position': value.positions[state.view.viewpoint_id], 'rotation': {'y': 0}},
            'after_metadata': {'lastActionSuccess': True, 'agent': {'actual': True}}}
        self.assertEqual(verify_agent_setup(action, state, value), {'actual': True})
        action['arguments']['rotation']['y'] = 45
        with self.assertRaisesRegex(ValueError, 'declared discrete'):
            verify_agent_setup(action, state, value)


if __name__ == '__main__':
    unittest.main()
