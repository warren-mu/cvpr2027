"""Synthetic CPU measurements only; no simulator or actual result is opened."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from src.envs import pair_v2_shard as base
from src.envs import pair_v2_solvable_policy as policy


def point(x=0, z=0):
    return {'x': x, 'y': .5, 'z': z}


def reference(name):
    return {'file': '/synthetic/' + name, 'sha256': hashlib.sha256(name.encode()).hexdigest()}


class Sim:
    """Actual policy/control flow; native responses are explicit CPU fixtures."""
    def __init__(self, counts, *, blocked=(), selection_fails=(), missing=()):
        self.counts, self.blocked, self.selection_fails, self.missing = counts, set(blocked), set(selection_fails), set(missing)
        self.query_log, self.records, self.observed = [], {}, []
        self.world, self.current_action, self.last_action_reference = 0, None, None
        self.reachable = [point(), point(.25)]
        self.native('GetReachablePositions', {}, self.reachable)

    def native(self, action, arguments, value):
        ref = reference(f'native_{len(self.records)}.json')
        self.records[ref['file']] = {'action': action, 'arguments': copy.deepcopy(arguments),
            'exception': None, 'after_metadata': {'lastActionSuccess': True, 'actionReturn': copy.deepcopy(value)}}
        self.query_log.append({'action': action, 'reference': ref})
        self.last_action_reference = ref
        return ref

    def read(self, ref):
        return copy.deepcopy(self.records[ref['file']])

    def restore_snapshot(self, world):
        self.world = world['world_index']
        return {}, {'restore_deterministic': True, 'restored_snapshot_reference': reference(f'restore_{self.world}.json')}

    def shortest_path(self, source, target):
        complete = self.world not in self.blocked
        raw = {'status': 'PathComplete' if complete else 'PathInvalid',
               'corners': [copy.deepcopy(source), copy.deepcopy(target)] if complete else []}
        ref = self.native('GetShortestPathToPoint', {'position': source, 'target': target, 'allowedError': .05}, raw)
        return {'source': copy.deepcopy(source), 'target': copy.deepcopy(target), 'raw': raw,
                'complete': complete, 'query_reference': ref, 'endpoint_tolerance_m': .05}

    def observe(self, action):
        public = base.action_text(action)
        self.current_action = public
        self.observed.append((self.world, public))
        rgb = np.full((255, 255, 3), 60, np.uint8)
        if self.world == 1:
            rgb.reshape(-1, 3)[30000:30250, 0] += 11
        masks = {name: np.zeros((255, 255), bool) for name in ('A', 'B', 'X')}
        masks['A'][10:20, 10:35] = True
        masks['B'][10:20, 100:125] = True
        count = self.counts.get((self.world, public), 0)
        masks['X'].reshape(-1)[30000:30000 + count] = True
        self.native('RotateLeft' if public == 'turn_left' else 'RotateRight' if public == 'turn_right' else 'TeleportFull', {}, None)
        files = {key: reference(f'{self.world}_{public}_{key}') for key in ('metadata', 'rgb', 'masks')}
        if (self.world, public) in self.missing:
            files.pop('masks')
        return {'rgb': rgb, 'masks_by_name': masks, 'metadata': {'objects': []},
            'evidence': {'valid': True, 'black': False, 'rgb_shape': [255, 255, 3], 'files': files}}

    def select(self, target):
        return {'success': self.world not in self.selection_fails, 'target_name': target,
                'distance_m': .5, 'visible_mask_pixels': 250}


WORLDS = [{'world_index': index, 'agent': {'position': point(), 'rotation': point(), 'horizon': 0, 'standing': True}}
          for index in range(2)]
IDENTITIES = {'candidate_names': ['A', 'B'], 'anchor_name': 'X', 'target_type': 'Mug', 'anchor_type': 'Book'}


def choose(sim):
    with patch.object(policy, 'read_reference', sim.read):
        return policy.choose_look(sim, copy.deepcopy(WORLDS), ['A', 'B'], IDENTITIES, sim.reachable, [])


class SolvablePolicyTests(unittest.TestCase):
    def tearDown(self):
        policy.reset_report()

    def test_different_visibility_actions_allowed_and_every_action_evaluated(self):
        sim = Sim({(0, 'turn_left'): 250, (1, 'turn_right'): 250})
        choice, report = choose(sim)
        self.assertIsNotNone(choice)
        self.assertEqual(report['anchor_visible_actions_w0'], ['turn_left'])
        self.assertEqual(report['anchor_visible_actions_w1'], ['turn_right'])
        self.assertEqual(sim.observed, [(0, 'turn_left'), (1, 'turn_left'),
            (0, 'turn_right'), (1, 'turn_right'), (0, 'move 1'), (1, 'move 1')])
        self.assertTrue(report['complete'])
        self.assertEqual(choice['anchor_pixels'], [250, 0])
        self.assertTrue(choice['evidence']['evidence_passed'])

    def test_w1_only_native_move_is_retained_without_fabricating_w0_zero(self):
        sim = Sim({(0, 'turn_left'): 250, (1, 'move 1'): 250}, blocked=(0,))
        choice, report = choose(sim)
        self.assertIsNotNone(choice)
        self.assertEqual(report['anchor_visible_actions_w1'], ['move 1'])
        slot = report['actions'][-1]['worlds'][0]
        self.assertEqual(slot['status'], 'not_in_domain')
        self.assertIsNone(slot['anchor_pixels'])
        self.assertIsNone(slot['anchor_visible'])
        self.assertNotIn((0, 'move 1'), sim.observed)
        self.assertIn((1, 'move 1'), sim.observed)
        self.assertTrue(report['complete'])

    def test_forever_invisible_world_is_rejected_with_specific_reason(self):
        choice, report = choose(Sim({(0, 'turn_left'): 250}))
        self.assertIsNone(choice)
        self.assertEqual(report['reasons'], ['look_not_solvable_w1'])
        self.assertFalse(report['solvable_w1'])
        self.assertTrue(report['complete'])

    def test_249_is_not_visible_and_250_is(self):
        for pixels, expected in ((249, False), (250, True)):
            choice, report = choose(Sim({(0, 'turn_left'): 250, (1, 'turn_right'): pixels}))
            self.assertIs(report['solvable_w1'], expected)
            self.assertIs(choice is not None, expected)

    def test_continuation_failure_does_not_hide_other_world_first_view(self):
        sim = Sim({(0, 'turn_left'): 250, (1, 'turn_right'): 250}, selection_fails=(0,))
        # Keep the inverse turn deterministic; a move inverse would require the
        # fixture's unrelated snapshot helper. The native domain stays two turns.
        sim.reachable = [point()]
        sim.native('GetReachablePositions', {}, sim.reachable)
        choice, report = choose(sim)
        self.assertIsNone(choice)  # No complete selected continuation.
        self.assertEqual(report['anchor_visible_actions_w1'], ['turn_right'])
        self.assertEqual(report['actions'][0]['worlds'][0]['status'], 'observed')
        self.assertFalse(report['actions'][0]['worlds'][0]['continuation']['passed'])
        self.assertEqual(report['actions'][0]['worlds'][1]['status'], 'observed')
        self.assertTrue(report['complete'])

    def test_missing_raw_mask_is_unknown_not_zero_or_accepted(self):
        choice, report = choose(Sim({(0, 'turn_left'): 250, (1, 'turn_right'): 250}, missing=((1, 'turn_right'),)))
        self.assertIsNone(choice)
        self.assertFalse(report['complete'])
        self.assertIsNone(report['solvable_w1'])
        self.assertIn('look_solvability_unverified', report['reasons'])
        self.assertNotIn('look_not_solvable_w1', report['reasons'])

    def test_malformed_native_path_is_not_a_proved_unavailable_action(self):
        sim = Sim({}, blocked=(0,))
        _, report = choose(sim)
        entry = copy.deepcopy(report['domains'][0]['unavailable_moves'][0])
        self.assertTrue(policy.unavailable_proved(entry))
        for raw in (None, {}, {'status': 0, 'corners': []}, {'status': False, 'corners': []},
                    {'status': 1, 'corners': [{'x': float('nan'), 'y': 0, 'z': 0}]}):
            changed = copy.deepcopy(entry)
            changed['path']['raw'] = raw
            changed['query_outcome']['action_return'] = raw
            self.assertFalse(policy.unavailable_proved(changed))

    def test_summary_is_recomputed_not_trusted_and_report_contains_no_arrays(self):
        _, report = choose(Sim({(0, 'turn_left'): 250, (1, 'turn_right'): 250}))
        json.dumps(report, allow_nan=False)
        changed = copy.deepcopy(report)
        changed['anchor_visible_actions_w1'] = ['made_up']
        checked = policy.solvability_summary(changed)
        self.assertEqual(checked['anchor_visible_actions_w1'], ['turn_right'])
        changed['actions'][1]['worlds'][1]['anchor_pixels'] = 0
        self.assertFalse(policy.solvability_summary(changed)['complete'])

    def test_direct_classification_is_unchanged(self):
        with patch.object(policy, '_OLD_CLASSIFY', return_value={'direct': 'unchanged'}) as prior:
            self.assertEqual(policy.classify_pair('direct_flip', evidence={}), {'direct': 'unchanged'})
            prior.assert_called_once_with('direct_flip', evidence={})

    def test_selected_common_action_and_anchor_cannot_be_replaced(self):
        choice, report = choose(Sim({(0, 'turn_left'): 250, (1, 'turn_right'): 250}))
        with patch.object(policy, '_OLD_CLASSIFY', return_value={'reasons': [], 'acceptance_passed': True}):
            args = {'evidence': choice['evidence'], 'target_w0': {'anchor_name': 'X'},
                    'common_observation_action': report['selected_action']}
            self.assertTrue(policy.classify_pair('look_flip', **args)['acceptance_passed'])
            args['common_observation_action'] = 'made_up'
            self.assertFalse(policy.classify_pair('look_flip', **args)['acceptance_passed'])


if __name__ == '__main__':
    unittest.main()
