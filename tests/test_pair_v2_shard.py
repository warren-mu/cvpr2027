"""CPU mock integration only. Root runs after commit through run_logged."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.envs import pair_v2_shard as shard
from src.envs import pair_v2_simulator as backend


def point(x, y=0.5, z=0.0):
    return {'x': x, 'y': y, 'z': z}


def baseline():
    rows = []
    for name, category, x in [('A', 'Mug', 0.0), ('B', 'Mug', 1.0), ('X', 'Apple', 0.2), ('table', 'Table', 0.5)]:
        rows.append({'name': name, 'objectId': name + '_id', 'objectType': category,
            'position': point(x), 'rotation': point(0, 0), 'pickupable': name != 'table',
            'moveable': False, 'receptacle': name == 'table', 'parentReceptacles': ['table_id']})
    return {'key': 'source', 'scene': 'FloorPlan1', 'objects': rows,
            'poses': [{'objectName': obj['name'], 'position': copy.deepcopy(obj['position']),
                       'rotation': copy.deepcopy(obj['rotation'])} for obj in rows if obj['pickupable']],
            'agent': {'position': point(0.5), 'rotation': point(0, 0), 'horizon': 0, 'standing': True},
            'worker_pid': 10, 'unity_pid': 11}


IDENTITIES = {'candidate_names': ['A', 'B'], 'anchor_name': 'X', 'target_type': 'Mug', 'anchor_type': 'Apple'}
VIEW = {'position': point(0.5), 'yaw': 0, 'horizon': 0}


def token(x):
    p = point(x)
    native = {'action': 'GetSpawnCoordinatesAboveReceptacle', 'success': True,
              'selected_position': p, 'coordinates': [p], 'receptacle_name': 'table'}
    return {'position': p, 'receptacle_name': 'table', 'spawn': native}


PROPOSAL = {'mode': 'flip', 'w0': token(0.2), 'w1': token(0.8)}


class SyntheticSimulator:
    """Native calls are mocked; policy, marking and classification remain real."""
    def __init__(self, *, look=False, fail_second=False):
        self.current = baseline()
        self.look, self.fail_second = look, fail_second
        self.phase, self.constructs = 'prefix', 0
        self.calls, self.path_queries = [], []

    def position_agent(self, position, yaw, horizon):
        self.calls.append('setup_agent')
        self.current['agent'] = {'position': copy.deepcopy(position), 'rotation': point(0, 0),
                                 'horizon': horizon, 'standing': True}

    def snapshot(self, key):
        value = copy.deepcopy(self.current)
        value['key'] = key
        return value

    def capture(self):
        is_w1 = self.current['key'].endswith('_w1')
        rgb = np.full((255, 255, 3), 60, dtype=np.uint8)
        if self.look and self.phase == 'prefix' and is_w1:
            rgb += 2
        masks = {name: np.zeros((255, 255), dtype=bool) for name in ('A', 'B', 'X')}
        masks['A'][10:20, 10:35] = True
        masks['B'][10:20, 100:125] = True
        if not self.look or self.phase == 'after':
            masks['X'].reshape(-1)[30000:30250] = True
            if is_w1:
                rgb.reshape(-1, 3)[30000:30250, 0] += 11
        return {'rgb': rgb, 'masks_by_name': masks, 'metadata': {'objects': copy.deepcopy(self.current['objects'])},
                'evidence': {'synthetic_fixture': True}}

    def construct_world(self, source, key, placements, require_agent):
        self.constructs += 1
        self.calls.append('construct')
        if self.fail_second and self.constructs == 2:
            raise RuntimeError('synthetic second-world failure')
        if len(placements) != 1 or placements[0]['object_name'] != 'X':
            raise AssertionError('Wrong construction interface')
        # Exact wrapping prevents accidentally sending a proposal wrapper as a native token.
        proof = placements[0]['spawn']
        if proof.get('action') != 'GetSpawnCoordinatesAboveReceptacle':
            raise AssertionError('Native spawn token was not unwrapped')
        self.current = copy.deepcopy(source)
        self.current.update(key=key, agent=copy.deepcopy(require_agent))
        self.current['objects'][2]['position'] = copy.deepcopy(proof['selected_position'])
        self.current['poses'][2]['position'] = copy.deepcopy(proof['selected_position'])
        self.phase = 'prefix'
        return {'passed': True, 'snapshot': copy.deepcopy(self.current), 'capture': self.capture(),
                'physical': {'before_objects': copy.deepcopy(source['objects']),
                    'requested_poses': copy.deepcopy(self.current['poses']),
                    'realized_objects': copy.deepcopy(self.current['objects']), 'edited_names': ['X'],
                    'native_is_scene_at_rest': True, 'placement_evidence_by_name': {'X': copy.deepcopy(proof)}}}

    def restore_snapshot(self, world):
        self.calls.append('restore')
        self.current, self.phase = copy.deepcopy(world), 'prefix'
        return self.capture(), {'restore_deterministic': True}

    def observe(self, action):
        self.calls.append('observe')
        if action['kind'] == 'move' and action['path']['world_key'] != self.current['key']:
            raise AssertionError('A move certificate was reused across worlds')
        self.phase = 'after'
        return self.capture()

    def select(self, target):
        return {'success': True, 'target_name': target, 'distance_m': 0.5, 'visible_mask_pixels': 250}

    def shortest_path(self, source, destination):
        self.path_queries.append(self.current['key'])
        return {'complete': True, 'world_key': self.current['key'],
                'source': copy.deepcopy(source), 'target': copy.deepcopy(destination), 'endpoint_tolerance_m': 0.05}


def successful_replay(config, world, actions, target, identities, directory):
    public = [shard.action_text(action) for action in actions]
    return {'planned_actions': public, 'executed_actions': public, 'action_successes': [True] * len(public),
            'process_id': 30, 'construction_process_id': world['worker_pid'], 'snapshot_restored': True,
            'restore_deterministic': True, 'restore_max_abs_diff': 3, 'selected_name': target,
            'selected_mask_pixels': 250, 'selected_distance_m': 0.5, 'success': True, 'done': True, 'cleanup_errors': []}


def build(sim, *, shared=None):
    return shard.build_pair(sim, {'name': 'synthetic', 'scene': 'FloorPlan1'}, baseline(), IDENTITIES,
        copy.deepcopy(PROPOSAL), copy.deepcopy(VIEW), 'fixture', Path('/unused_cpu_fixture'), [], attempt_row=shared)


class PairV2ShardIntegration(unittest.TestCase):
    def test_direct_build_uses_real_policy_and_one_select_per_world(self):
        sim = SyntheticSimulator()
        with patch.object(shard, 'wait_replay', side_effect=successful_replay), \
             patch.object(shard, 'export_pair', return_value='synthetic_pair.json') as export, \
             patch.object(shard, 'sha', return_value='synthetic_sha'):
            row = build(sim)
        self.assertEqual(row['status'], 'passed', row)
        self.assertEqual(row['kind'], 'direct_flip')
        self.assertEqual(row['witness_actions'], [['select 1'], ['select 2']])
        self.assertFalse(row['validity']['same_input_at_t'])
        self.assertEqual(row['restore_max_abs_diff'], [3, 3])
        self.assertEqual(sim.calls[:3], ['setup_agent', 'construct', 'construct'])
        self.assertEqual(export.call_count, 1)

    def test_look_build_verifies_shared_input_without_relabelling_native_noise(self):
        sim = SyntheticSimulator(look=True)
        with patch.object(shard, 'observation_actions', return_value=([{'kind': 'turn_right'}], [])), \
             patch.object(shard, 'wait_replay', side_effect=successful_replay), \
             patch.object(shard, 'export_pair', return_value='synthetic_pair.json'), \
             patch.object(shard, 'sha', return_value='synthetic_sha'):
            row = build(sim)
        self.assertEqual(row['status'], 'passed', row)
        self.assertEqual(row['kind'], 'look_flip')
        self.assertTrue(row['prefix_measurement']['shared_prefix_verified'])
        self.assertFalse(row['prefix_measurement']['native_public_prefix_exact'])
        self.assertEqual(row['prefix_measurement']['prefix_max_abs_diff'], 2)
        self.assertEqual(row['witness_actions'], [['turn_right', 'select 1'], ['turn_right', 'select 2']])

    def test_new_process_witness_failure_cannot_export_pair(self):
        def failed(*args):
            record = successful_replay(*args)
            record['snapshot_restored'] = False
            record['restore_deterministic'] = False
            return record
        with patch.object(shard, 'wait_replay', side_effect=failed), patch.object(shard, 'export_pair') as export:
            row = build(SyntheticSimulator())
        self.assertEqual(row['status'], 'rejected')
        self.assertIn('state_restore', row['discard_reasons'])
        export.assert_not_called()

    def test_shared_attempt_keeps_completed_measurements_when_later_construction_raises(self):
        row = {}
        with self.assertRaisesRegex(RuntimeError, 'second-world failure'):
            build(SyntheticSimulator(fail_second=True), shared=row)
        self.assertEqual(row['pair_id'], 'fixture')
        self.assertEqual(row['identities'], IDENTITIES)
        self.assertTrue(row['physical_w0']['valid_physical'])
        self.assertNotIn('physical_w1', row)

    def test_common_move_gets_independent_certificate_in_each_world(self):
        sim = SyntheticSimulator(look=True)
        worlds = [baseline(), baseline()]
        worlds[0]['key'], worlds[1]['key'] = 'fixture_w0', 'fixture_w1'
        worlds[1]['objects'][2]['position'] = point(0.8)
        first_action = {'kind': 'move', 'index': 1, 'position': point(0.75),
                        'path': {'world_key': 'unusable_other_world', 'complete': True}}
        with patch.object(shard, 'observation_actions', return_value=([first_action], [])):
            choice, _ = shard.choose_look(sim, worlds, ['A', 'B'], IDENTITIES, [], [])
        self.assertIsNotNone(choice)
        self.assertEqual(sim.path_queries, ['fixture_w0', 'fixture_w1'])
        self.assertEqual([plan[0]['path']['world_key'] for plan in choice['plans']], ['fixture_w0', 'fixture_w1'])
        self.assertEqual([shard.action_text(plan[0]) for plan in choice['plans']], ['move 1', 'move 1'])

    def test_failed_planner_restore_cannot_supply_evidence(self):
        sim = SyntheticSimulator(look=True)
        with patch.object(shard, 'observation_actions', return_value=([{'kind': 'turn_right'}], [])), \
             patch.object(sim, 'restore_snapshot', return_value=({}, {'restore_deterministic': False})):
            choice, search = shard.choose_look(sim, [baseline(), baseline()], ['A', 'B'], IDENTITIES, [], [])
        self.assertIsNone(choice)
        self.assertIn('restoration', search['actions'][0]['error'])
        self.assertNotIn('observe', sim.calls)

    def test_replay_wait_always_stops_child_and_cleanup_error_invalidates_success(self):
        child = SimpleNamespace(result={'success': True}, error=None, receive=lambda: None)
        calls = []
        child.stop = lambda: calls.append('stop') or ['synthetic cleanup error']
        with patch.object(shard, 'Worker', return_value=child):
            result = shard.wait_replay({'deadline': 100}, baseline(), [{'kind': 'select', 'label': 1}],
                                        'A', IDENTITIES, Path('/unused'))
        self.assertEqual(calls, ['stop'])
        self.assertFalse(result['success'])
        self.assertEqual(result['cleanup_errors'], ['synthetic cleanup error'])

    def test_generator_journals_partial_attempt_after_failure(self):
        class Store:
            def save_json(self, *args):
                return {'file': 'synthetic'}
        sim = SimpleNamespace(store=Store(), action_count=0,
            preflight_scene=lambda: {'passed': True, 'snapshot': baseline()},
            reachable_positions=lambda: {'positions': [point(0.5)]},
            spawn_points=lambda name: [token(0.2)['spawn']], check_deadline=lambda: None, finish=lambda: {})
        controller = SimpleNamespace(_build=SimpleNamespace(commit_id=backend.BUILD))
        def fail(*args, attempt_row):
            attempt_row.update(physical_w0={'measured_before_failure': True}, proposal=copy.deepcopy(PROPOSAL),
                               kind='direct_flip', status='passed', pair_file='synthetic_incomplete_export.json')
            raise RuntimeError('synthetic failure retained')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'raw').mkdir()
            config = {'name': 'synthetic', 'scene': 'FloorPlan1', 'attempt_journal': 'journal.jsonl',
                      'raw_evidence_dir': str(root / 'raw'), 'max_identity_sets': 1, 'max_receptacles': 1,
                      'max_points_per_side': 1, 'max_view_positions': 1, 'max_attempts': 5, 'scene_pairs_per_class': 1}
            with patch.object(shard, 'ROOT', root), patch.object(shard, 'controller_session', return_value=nullcontext(controller)), \
                 patch.object(backend, 'V2Simulator', return_value=sim), \
                 patch.object(shard, 'identity_sets', return_value=[IDENTITIES]), \
                 patch.object(shard, 'placement_groups', return_value=([token(0.2)], [token(0.8)])), \
                 patch.object(shard, 'view_proposals', return_value=[VIEW]), \
                 patch.object(shard, 'construction_pairs', return_value=[PROPOSAL]), \
                 patch.object(shard, 'build_pair', side_effect=fail), patch.object(shard.traceback, 'print_exc'):
                result = shard.generator_worker(config, SimpleNamespace(send=lambda value: None))
            journal = json.loads((root / 'journal.jsonl').read_text())
            raw = json.loads(Path(journal['raw_attempt']['file']).read_text())
            self.assertTrue(raw['physical_w0']['measured_before_failure'])
            self.assertEqual(raw['proposal'], PROPOSAL)
            self.assertIn('synthetic failure retained', raw['discard_reasons'][0])
            self.assertEqual(raw['status'], 'rejected')
            self.assertEqual(raw['incomplete_export']['pair_file'], 'synthetic_incomplete_export.json')
            self.assertEqual(journal['raw_attempt']['sha256'], shard.sha(journal['raw_attempt']['file']))
            self.assertEqual(result['attempts'], 1)
            self.assertEqual(result['passed_pair_files'], [])
            self.assertEqual(result['passed_by_class']['direct_flip'], 0)


if __name__ == '__main__':
    unittest.main()
