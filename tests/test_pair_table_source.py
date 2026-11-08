"""CPU-only synthetic adapter fixtures; no scene, renderer or experiment run."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from src.envs import pair_table_source as source


def xyz(x=0):
    return {'x': x, 'y': 0, 'z': 0}


def canonical_fixture():
    command = {'position': xyz(), 'rotation': xyz(0), 'horizon': 30,
               'standing': True, 'forceAction': False}
    agent = {'position': xyz(), 'rotation': xyz(), 'horizon': 30.0000038, 'standing': True}
    world = {'agent': agent, 'agent_setup_command': command}
    return {'view': {'position': xyz(), 'yaw': 0, 'horizon': 30}}, [world, copy.deepcopy(world)]


def witness_fixture(kinds=('turn_left',)):
    records = {}
    def put(name, data):
        path = '/synthetic/' + name
        records[path] = copy.deepcopy(data)
        return {'file': path, 'sha256': name}
    def capture(number, metadata):
        return {'action_index': number, 'files': {'metadata': put(f'metadata_{number}.json', metadata)}}
    meta = {'objects': [{'name': 'a'}], 'agent': {'position': xyz(), 'rotation': xyz(),
            'cameraHorizon': 30.0000038, 'isStanding': True}, 'lastActionSuccess': True}
    initial = {'objects': copy.deepcopy(meta['objects']), 'agent': {'position': xyz(), 'rotation': xyz(),
               'horizon': 30.0000038, 'standing': True}}
    restored = put('restored.json', initial)
    captures, actions, public, native_refs = [capture(0, meta)], [], [], []
    number = 1
    for index, kind in enumerate(kinds):
        prior = copy.deepcopy(meta)
        if kind == 'move':
            # Repeated legacy move 1 can denote a different literal destination.
            destination = xyz(.25 if index == 0 else 0)
            meta = {**copy.deepcopy(prior), 'actionReturn': {'status': 'PathComplete',
                    'corners': [prior['agent']['position'], destination]}}
            query = {'index': number, 'action': 'GetShortestPathToPoint', 'exception': None,
                'arguments': {'position': prior['agent']['position'], 'target': destination, 'allowedError': .05},
                'before_metadata': prior, 'after_metadata': copy.deepcopy(meta)}
            native_refs.append(put(f'action_{number:06d}.json', query))
            number += 1
            arguments = {'position': destination, 'rotation': prior['agent']['rotation'],
                         'horizon': prior['agent']['cameraHorizon'], 'standing': True, 'forceAction': False}
            native_name, text = 'TeleportFull', 'move 1'
        else:
            arguments = {'degrees': 45}
            native_name, text = ('RotateLeft' if kind == 'turn_left' else 'RotateRight'), kind
        before = copy.deepcopy(meta)
        meta = copy.deepcopy(meta)
        if kind == 'move':
            meta['agent']['position'] = destination
        else:
            meta['agent']['rotation']['y'] = (meta['agent']['rotation']['y'] + (-45 if kind == 'turn_left' else 45)) % 360
        meta['sequenceId'] = number
        observed = capture(number, meta)
        native = {'index': number, 'action': native_name, 'exception': None, 'arguments': arguments,
                  'before_metadata': before, 'after_metadata': copy.deepcopy(meta), 'capture': observed}
        action_ref = put(f'action_{number:06d}.json', native)
        native_refs.append(action_ref)
        actions.append({'kind': kind, 'success': True, 'action_reference': action_ref})
        captures.append(observed)
        public.append(text)
        number += 1
    public.append('select 2')
    actions.append({'kind': 'select', 'label': 2, 'success': True, 'selected_object_name': 'a'})
    row = {'witness_actions': [public, public], 'target_w0': 'a', 'target_w1': 'a'}
    replay = {'planned_actions': public, 'executed_actions': public, 'actions': actions, 'captures': captures,
        'status': 'completed', 'cleanup_errors': [], 'passed': True, 'success': True, 'selected_name': 'a',
        'restoration': {'restored_snapshot_reference': restored},
        'raw_evidence': put('index.json', {'files': native_refs})}
    return row, replay, records, lambda ref: copy.deepcopy(records[ref['file']])


class PairTableSourceTests(unittest.TestCase):
    def test_canonical_horizon_preserved_without_readback_rounding(self):
        row, worlds = canonical_fixture()
        result = source.canonical_start(row, worlds)
        self.assertEqual(result['agent_setup_command']['horizon'], 30)
        self.assertEqual(result['initial_actual_agent']['horizon'], 30.0000038)
        self.assertEqual(result['initial_heading'], 0)

    def test_missing_canonical_or_non_discrete_yaw_rejected(self):
        for mode in ('missing', 'yaw', 'world'):
            row, worlds = canonical_fixture()
            if mode == 'missing':
                worlds[0]['agent_setup_command'] = None
            elif mode == 'yaw':
                for world in worlds:
                    world['agent_setup_command']['rotation']['y'] = 45.0000038
                row['view']['yaw'] = 45.0000038
            else:
                worlds[1]['agent']['position']['x'] = .001
            with self.assertRaises(ValueError):
                source.canonical_start(row, worlds)

    def test_original_turn_and_name_selection_mapped(self):
        row, replay, _, read = witness_fixture()
        mapped = source.native_witness(row, 0, replay, read)
        self.assertEqual(mapped['actions'][0]['kind'], 'turn_left')
        self.assertEqual(mapped['actions'][1]['target_name'], 'a')

    def test_reused_move_index_maps_two_actual_targets(self):
        row, replay, _, read = witness_fixture(('move', 'move'))
        mapped = source.native_witness(row, 0, replay, read)
        self.assertEqual([item['original_public_action'] for item in mapped['actions'][:2]], ['move 1', 'move 1'])
        self.assertEqual([item['position'] for item in mapped['actions'][:2]], [xyz(.25), xyz()])
        self.assertTrue(all(item['path_reference'] for item in mapped['actions'][:2]))

    def test_native_path_prior_capture_or_arguments_mismatch_rejected(self):
        for key in ('start', 'capture'):
            row, replay, records, read = witness_fixture(('move',))
            query = records['/synthetic/action_000001.json']
            if key == 'start':
                query['arguments']['position'] = xyz(9)
            else:
                query['before_metadata']['agent']['position'] = xyz(9)
            with self.assertRaisesRegex(ValueError, 'path'):
                source.native_witness(row, 0, replay, read)

    def test_no_guessed_coordinate_when_native_move_record_missing(self):
        row, replay, records, read = witness_fixture(('move',))
        records['/synthetic/index.json']['files'] = []
        with self.assertRaisesRegex(ValueError, 'path certificate'):
            source.native_witness(row, 0, replay, read)

    def test_initial_capture_must_bind_actual_restored_world(self):
        row, replay, records, read = witness_fixture()
        records['/synthetic/restored.json']['objects'] = [{'name': 'other_world'}]
        with self.assertRaisesRegex(ValueError, 'restored state'):
            source.native_witness(row, 0, replay, read)

    def test_selection_label_cannot_be_silently_changed_to_actual_target(self):
        row, replay, _, read = witness_fixture(())
        replay['actions'][-1]['label'] = 3
        with self.assertRaisesRegex(ValueError, 'terminal selection'):
            source.native_witness(row, 0, replay, read)

    def test_container_rejects_failed_or_unfinished_checkpoint(self):
        reference = {'file': 'results/t01v2_checkpoint_fixture.json', 'audit': {'warn': []}}
        for mode in ('failed', 'unfinished', 'wrong_producer'):
            name = 't01v2_checkpoint_fixture'
            data = {'_meta': {'name': name, 'end': 'recorded', 'exit_code': 0,
                'command': ['python3', '-m', 'src.tools.t01v2_review_checkpoint', '--name', name]},
                'config': {'name': name}, 'metrics': {'kind': 't01v2_review_checkpoint',
                    'status': 'checkpoint_ready_for_human_review', 'errors': [], 'final_pilot_acceptance': False,
                    'candidate_pairs': [{'pair_id': 'p'}]}}
            if mode == 'failed':
                data['metrics']['errors'] = [{'error': 'hash mismatch'}]
            elif mode == 'unfinished':
                data['_meta']['end'] = None
            else:
                data['_meta']['command'][2] = 'src.tools.other'
            with patch.object(source, 'audited_result', return_value=(data, reference)), self.assertRaises(ValueError):
                source._container('unused')

    def test_ambiguous_pair_id_requires_explicit_pair_file(self):
        entries = [{'pair_id': 'p', 'pair_reference': {'file': f'{name}/pair.json'}} for name in ('one', 'two')]
        with patch.object(source, '_container', return_value=({}, {}, entries)), \
                patch.object(source.merge, 'load_source') as load, self.assertRaisesRegex(ValueError, 'exactly one'):
            source.load_reviewed_pair('unused', 'p')
        load.assert_not_called()

    def test_changed_scene_hash_stops_before_any_pair_evidence(self):
        entry = {'pair_id': 'p', 'source_file': 'results/scene.json', 'source_metrics_sha256': 'original'}
        data = {'config': {}, 'metrics': {}}
        with patch.object(source, '_container', return_value=({}, {}, [entry])), \
                patch.object(source.merge, 'load_source', return_value=(data, {'metrics_sha256': 'changed'})), \
                patch.object(source, 'verified_passed_rows') as journal, self.assertRaisesRegex(ValueError, 'changed source'):
            source.load_reviewed_pair('unused', 'p')
        journal.assert_not_called()


if __name__ == '__main__':
    unittest.main()
