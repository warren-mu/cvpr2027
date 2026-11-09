"""CPU scheduler fixtures; original science/auditor boundaries are not simulated passes."""
from contextlib import ExitStack
import copy
import io
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.tools import t02_export_parallel as parent


UUID = 'GPU-11111111-2222-3333-4444-555555555555'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return {'file': str(path), 'sha256': parent.sha_file(path)}


class Fixture:
    def __init__(self, directory, count=2, partition='train'):
        self.root = Path(directory)
        self.identity = {'pbs_jobid': '700000.hopper-m-02', 'cuda_visible_devices': UUID,
                         'scratch': str(self.root / 'scratch'), 'cpu_affinity': list(range(12)),
                         'node': 'fixture-node'}
        self.entries = []
        scene = 'FloorPlan1' if partition == 'train' else 'FloorPlan21'
        for index in range(count):
            # Different authenticated sources can legitimately reuse the original ID.
            pair_id = scene.lower() + '_00001'
            source = write_json(self.root / 'results' / f'source_{index}.json', {'fixture_source': index})
            pair = write_json(self.root / 'data/world_pairs/v2' / f'source_{index}' / 'pair.json',
                              {'pair_id': pair_id, 'fixture_source': index})
            for ref in (source, pair):
                ref['file'] = str(Path(ref['file']).relative_to(self.root))
            self.entries.append({'scene': scene, 'field_of_view': 60, 'pair_id': pair_id,
                                 'source': source, 'pair': pair, 'proposal_identity': f'{index + 1:064x}'})
        self.manifest = {'version': parent.MANIFEST_VERSION, 'partition': partition,
                         'entries': self.entries}
        self.manifest_ref = write_json(self.root / 'manifest.json', self.manifest)
        self.commands = []
        self.children = []
        self.events = []
        self.clock = 100.0
        self.behavior = {}
        self.hold = set()
        self.launch_failure = None
        self.uncertain_cleanup = False
        self.finish_failure = None

    def child_data(self, index, command, status='exported'):
        name = command[command.index('--name') + 1]
        entry = self.entries[index]
        code = 0 if status in ('exported', 'rejected') else 86 if status == 'black' else 1
        package = {'pair_id': entry['pair_id'], 'scene': entry['scene'],
                   'source_config': {'field_of_view': 60},
                   'independent_verification': {'proposal_identity': entry['proposal_identity']},
                   'source_provenance': {'container': entry['source'], 'pair': entry['pair']}}
        package_ref = write_json(self.root / 'scratch' / name / 'authenticated_source.json', package)
        rows = [] if status in ('black', 'interrupted') else [
            {'pair_id': entry['pair_id'], 'status': status, 'source_package': package_ref}]
        metrics = {'kind': 't02_table_export', 'status': status, 'producer_exit_code': code,
                   'pairs': rows, 'unstarted_pair_ids': [] if rows else [entry['pair_id']],
                   'cleanup_errors': [], 'errors': [] if code in (0, 86) else ['original failure'],
                   'black_frame_detected': status == 'black', 'export_checks_complete': code == 0}
        cfg = {**copy.deepcopy(self.identity), 'name': name, 'version': parent.export.VERSION,
               'gpus': 1, 'pair_id': [entry['pair_id']], 'source': entry['source']['file'],
               'pair_file': entry['pair']['file']}
        if rows:
            cfg['runtime'] = {**copy.deepcopy(self.identity), 'gpu_uuid': UUID}
        return {'config': cfg, 'metrics': metrics,
                '_meta': {'command': command, 'name': name, 'end': 'fixture-ended', 'exit_code': code}}

    def launch(self, name, command, environment, directory):
        index = len(self.children)
        if index == self.launch_failure:
            raise OSError('fixture launch failure')
        self.assert_environment = dict(environment)
        self.commands.append(copy.deepcopy(command))
        process = SimpleNamespace(returncode=None, pid=1000 + index)
        process.poll = lambda: process.returncode
        child = {'process': process, 'name': name, 'index': index, 'command': command,
                 'stream': io.StringIO(), 'start_monotonic': self.clock,
                 'launcher_log': self.root / f'{name}.launcher.log'}
        self.children.append(child)
        self.events.append(('launch', index))
        return child

    def tick(self, _seconds):
        self.clock += .1
        for child in self.children:
            index = child['index']
            if child['process'].returncode is None and index not in self.hold:
                status = self.behavior.get(index, 'exported')
                child['data'] = self.child_data(index, child['command'], status)
                child['process'].returncode = child['data']['_meta']['exit_code']

    def finish(self, child, sources):
        if child['index'] == self.finish_failure:
            raise OSError('fixture launcher log unavailable')
        if child['process'].returncode is None:
            raise RuntimeError('fixture cannot finish live child')
        ref = {'file': f"results/{child['name']}.json", 'sha256': 'a' * 64}
        sources.append(ref)
        self.events.append(('finish', child['index']))
        return ({'run_name': child['name'], 'exit_code': child['process'].returncode,
                 'source': ref, 'result_path': ref['file']}, child['data'])

    def request(self, process):
        self.events.append(('request', process.pid - 1000))

    def stop(self, process, **kwargs):
        index = process.pid - 1000
        self.events.append(('stop', index))
        if self.uncertain_cleanup:
            return ['fixture owned child still alive']
        child = self.children[index]
        child['data'] = self.child_data(index, child['command'], 'interrupted')
        process.returncode = 1
        return []

    def run(self, concurrency=6, deadline=10000, stopping=None):
        self.sources = []
        with ExitStack() as stack:
            stack.enter_context(patch.object(parent, 'ROOT', self.root))
            for name in ('launch_logged', 'finish_logged', 'request_stop_logged', 'stop_logged'):
                stack.enter_context(patch.object(parent, name, getattr(self, {
                    'launch_logged': 'launch', 'finish_logged': 'finish',
                    'request_stop_logged': 'request', 'stop_logged': 'stop'}[name])))
            stack.enter_context(patch.object(parent.time, 'time', lambda: self.clock))
            stack.enter_context(patch.object(parent.time, 'monotonic', lambda: self.clock))
            stack.enter_context(patch.object(parent.time, 'sleep', self.tick))
            stack.enter_context(patch.dict(os.environ, {'PBS_JOBID': self.identity['pbs_jobid'],
                'CUDA_VISIBLE_DEVICES': UUID, 'CVPR_SCRATCH': self.identity['scratch']}))
            return parent.run_units(self.entries, name='t02_export_fixture', concurrency=concurrency,
                directory=self.root, identity=self.identity, deadline=deadline, unit_seconds=400,
                pair_seconds=300, gate_seconds=30, sources=self.sources,
                stopping=[] if stopping is None else stopping)


class LauncherTests(unittest.TestCase):
    def test_actual_t02_namespace_retains_logged_wrapper_and_gpu_mask(self):
        with tempfile.TemporaryDirectory() as text:
            root = Path(text); name = 't02_export_fixture_u0000'
            command = ['python3', '-m', 'src.tools.t02_export', '--name', name]
            process = SimpleNamespace(pid=123)
            with patch.object(parent, 'ROOT', root), patch.object(parent.subprocess, 'Popen', return_value=process) as popen:
                running = parent.launch_logged(name, command, {'CUDA_VISIBLE_DEVICES': UUID}, root)
            args, kwargs = popen.call_args
            self.assertEqual(args[0], [parent.sys.executable, '-u', str(root / 'scripts/run_logged.py'),
                                      '--name', name, '--seed', '0', '--', *command])
            self.assertEqual(kwargs['cwd'], root)
            self.assertTrue(kwargs['start_new_session'])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], UUID)
            self.assertEqual(kwargs['env']['CVPR_RUN_NAME'], name)
            self.assertEqual(kwargs['env']['CVPR_RESULTS_PATH'], str(root / 'results' / (name + '.json')))
            running['stream'].close()

    def test_existing_evidence_and_invalid_names_never_launch(self):
        for suffix in ('result', 'directory', '.log', '.meta.json', '.diff', 'launcher', 'dangling'):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as text:
                root = Path(text); name = 't02_export_fixture'
                (root / 'results').mkdir(); (root / 'logs/runs').mkdir(parents=True)
                path = (root / 'results' / (name + '.json') if suffix in ('result', 'dangling') else
                        root / 'logs/runs' / name if suffix == 'directory' else
                        root / (name + '.launcher.log') if suffix == 'launcher' else
                        root / 'logs/runs' / (name + suffix))
                if suffix == 'directory': path.mkdir()
                elif suffix == 'dangling': path.symlink_to(root / 'absent')
                else: path.write_text('preserved')
                with patch.object(parent, 'ROOT', root), patch.object(parent.subprocess, 'Popen') as popen:
                    with self.assertRaises(FileExistsError): parent.launch_logged(name, ['python3'], {}, root)
                    popen.assert_not_called()
        for name in ('pilot_pairs_fixture', 't02_export_', 't02_export_bad/path'):
            with self.assertRaises(ValueError): parent.launch_logged(name, [], {}, Path('/unused'))


class ManifestTests(unittest.TestCase):
    def test_hash_split_and_original_identity_bound_without_pair_id_collapse(self):
        for partition in ('train', 'test'):
            with tempfile.TemporaryDirectory() as text:
                f = Fixture(text, partition=partition)
                with patch.object(parent, 'ROOT', f.root):
                    data, reference = parent.read_manifest(**{'path': f.manifest_ref['file'],
                                                            'expected_sha': f.manifest_ref['sha256']})
                    self.assertEqual(data['entries'], f.entries)
                    self.assertEqual(data['partition'], partition)
                    self.assertEqual(reference['sha256'], f.manifest_ref['sha256'])
                    self.assertEqual(f.entries[0]['pair_id'], f.entries[1]['pair_id'])
                    with self.assertRaises(ValueError): parent.read_manifest(f.manifest_ref['file'], '0' * 64)
                    for edit in ('split', 'camera', 'source_sha', 'escape', 'duplicate', 'alias_duplicate'):
                        bad = copy.deepcopy(f.manifest)
                        if edit == 'split': bad['partition'] = 'test' if partition == 'train' else 'train'
                        elif edit == 'camera': bad['entries'][0]['field_of_view'] = 100
                        elif edit == 'source_sha': bad['entries'][0]['source']['sha256'] = '0' * 64
                        elif edit == 'escape': bad['entries'][0]['pair'] = f.entries[0]['source']
                        else:
                            bad['entries'][1] = copy.deepcopy(bad['entries'][0])
                            if edit == 'alias_duplicate':
                                bad['entries'][1]['source']['file'] = str(f.root / bad['entries'][1]['source']['file'])
                        ref = write_json(f.root / f'{edit}.json', bad)
                        with self.subTest(edit=edit), self.assertRaises(ValueError):
                            parent.read_manifest(ref['file'], ref['sha256'])

    def test_exact_original_cli_and_terminal_allocation_package_binding(self):
        with tempfile.TemporaryDirectory() as text:
            f = Fixture(text); entry = f.entries[0]
            command = parent.unit_command('t02_export_x_u0000', entry, 400, 300, 30)
            self.assertEqual(command, [parent.sys.executable, '-u', '-m', 'src.tools.t02_export', '--name',
                't02_export_x_u0000', '--source', entry['source']['file'], '--pair-id', entry['pair_id'],
                '--pair-file', entry['pair']['file'], '--timeout-seconds', '400',
                '--pair-timeout-seconds', '300', '--gate-timeout-seconds', '30'])
            data = f.child_data(0, command); record = {'exit_code': 0}
            self.assertTrue(parent.child_outcome(record, data, f.identity, command, entry)['export_checks_complete'])
            for field, value in (('cuda_visible_devices', 'GPU-wrong'), ('pbs_jobid', 'another'),
                                  ('cpu_affinity', [0]), ('node', 'different-node'),
                                  ('source', f.entries[1]['source']['file']), ('name', 't02_wrong')):
                bad = copy.deepcopy(data); bad['config'][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    parent.child_outcome(record, bad, f.identity, command, entry)
            for edit in ('unfinished', 'command', 'false_success', 'wrong_gpu_runtime', 'package_sha', 'other_package'):
                bad = copy.deepcopy(data)
                if edit == 'unfinished': bad['_meta']['end'] = None
                elif edit == 'command': bad['_meta']['command'] = command + ['--changed']
                elif edit == 'false_success': bad['metrics']['errors'] = ['real failure']
                elif edit == 'wrong_gpu_runtime': bad['config']['runtime']['gpu_uuid'] = 'different'
                elif edit == 'package_sha': bad['metrics']['pairs'][0]['source_package']['sha256'] = '0' * 64
                else:
                    other = f.child_data(1, parent.unit_command('t02_export_other', f.entries[1], 400, 300, 30))
                    bad['metrics']['pairs'][0]['source_package'] = other['metrics']['pairs'][0]['source_package']
                with self.subTest(edit=edit), self.assertRaises(ValueError):
                    parent.child_outcome(record, bad, f.identity, command, entry)


class SchedulerTests(unittest.TestCase):
    def test_six_same_uuid_units_keep_duplicate_original_pair_ids_separate(self):
        with tempfile.TemporaryDirectory() as text:
            f = Fixture(text, count=10); result = f.run()
            self.assertTrue(result['all_units_ended'])
            self.assertEqual(result['maximum_simultaneous_unit_processes'], 6)
            self.assertEqual(len(result['units']), 10)
            self.assertEqual(len({r['run_name'] for r in result['units']}), 10)
            self.assertEqual(len({r['result_path'] for r in result['units']}), 10)
            self.assertEqual(len({r['entry']['pair_id'] for r in result['units']}), 1)
            self.assertEqual(len(f.sources), 10)
            self.assertEqual(f.assert_environment['CUDA_VISIBLE_DEVICES'], UUID)
            self.assertEqual(f.assert_environment['PBS_JOBID'], f.identity['pbs_jobid'])
            self.assertEqual(result['cleanup_errors'], [])
            self.assertFalse(result['downstream_allowed'])
            self.assertIsNone(result['cache_matches_live'])
            # Original producer gates then runs one Worker, sequentially; no second
            # native worker is launched by this scheduler inside a unit.
            self.assertTrue(all(command[3] == 'src.tools.t02_export' for command in f.commands))

    def test_real_failed_unit_is_retained_and_other_units_continue(self):
        with tempfile.TemporaryDirectory() as text:
            f = Fixture(text, count=4); f.behavior[0] = 'error'
            result = f.run(concurrency=2)
            self.assertTrue(result['all_units_ended'])
            self.assertEqual(result['units'][0]['producer_exit_code'], 1)
            self.assertEqual(result['units'][0]['errors'], ['original failure'])
            self.assertFalse(result['units'][0]['export_checks_complete'])
            self.assertEqual(len(result['units']), 4)
            self.assertEqual(result['errors'], [])
            self.assertEqual(result['unstarted_unit_indices'], [])

    def test_black_stops_new_units_and_requests_all_owned_children_before_waiting(self):
        with tempfile.TemporaryDirectory() as text:
            f = Fixture(text, count=8); f.behavior[0] = 'black'; f.hold = set(range(1, 6))
            result = f.run(concurrency=6)
            self.assertTrue(result['black_frame_detected'])
            self.assertEqual(result['unstarted_unit_indices'], [6, 7])
            self.assertEqual([r['producer_exit_code'] for r in result['units']], [86, 1, 1, 1, 1, 1])
            self.assertTrue(all(r['stopped_by_parent'] for r in result['units'][1:]))
            self.assertEqual([e for e in f.events if e[0] == 'request'], [('request', i) for i in range(1, 6)])
            self.assertLess(max(i for i, e in enumerate(f.events) if e[0] == 'request'),
                            min(i for i, e in enumerate(f.events) if e[0] == 'stop'))
            self.assertFalse(result['all_units_ended'])
            self.assertEqual(result['cleanup_errors'], [])

    def test_uncertain_cleanup_or_early_deadline_never_claim_completion(self):
        with tempfile.TemporaryDirectory() as text:
            f = Fixture(text); result = f.run(concurrency=2, deadline=544)
            self.assertEqual(result['units'], [])
            self.assertEqual(result['unstarted_unit_indices'], [0, 1])
            self.assertEqual(result['stopping_reasons'], ['insufficient_parent_time_for_full_next_unit_grant'])
            self.assertFalse(result['all_units_ended'])
        with tempfile.TemporaryDirectory() as text:
            f = Fixture(text, count=3); f.behavior[0] = 'black'; f.hold = {1}; f.uncertain_cleanup = True
            result = f.run(concurrency=2)
            self.assertFalse(result['all_units_ended'])
            self.assertEqual(result['cleanup_errors'], ['fixture owned child still alive'])
            self.assertTrue(result['errors'])
            self.assertEqual(result['unstarted_unit_indices'], [2])

    def test_unreadable_finished_evidence_does_not_interrupt_other_owned_cleanup(self):
        with tempfile.TemporaryDirectory() as text:
            f = Fixture(text, count=4); f.finish_failure = 0; f.hold = {1}
            result = f.run(concurrency=2)
            self.assertEqual(result['unstarted_unit_indices'], [2, 3])
            self.assertEqual(len(result['units']), 2)
            self.assertIn('fixture launcher log unavailable', result['units'][0]['provenance_error'])
            self.assertTrue(result['units'][1]['stopped_by_parent'])
            self.assertIn(('stop', 1), f.events)
            self.assertFalse(result['all_units_ended'])

    def test_launch_failure_keeps_prior_child_record_and_cleans_it(self):
        with tempfile.TemporaryDirectory() as text:
            f = Fixture(text, count=3); f.launch_failure = 1
            result = f.run(concurrency=2)
            self.assertEqual(result['unstarted_unit_indices'], [1, 2])
            self.assertEqual(len(result['units']), 1)
            self.assertTrue(result['units'][0]['stopped_by_parent'])
            self.assertIn('fixture launch failure', result['errors'][0])
            self.assertFalse(result['all_units_ended'])
        with tempfile.TemporaryDirectory() as text:
            f = Fixture(text)
            with self.assertRaises(ValueError): f.run(concurrency=7)


class MainTests(unittest.TestCase):
    def invoke(self, f, *, gate_exit=0, unit_result=None):
        name = 't02_export_main_fixture'
        gate = {'_meta': {'exit_code': gate_exit}, 'metrics': {'cleanup_errors': []}}
        ref = {'file': 'results/original_gate.json', 'sha256': 'e' * 64}
        events = []
        def gate_call(_name, _seconds, _directory, _deadline, sources):
            events.append('parent_gate'); sources.append(ref)
            return gate, ref, []
        def units(*args, **kwargs):
            events.append('units')
            self.assertEqual(args[0], f.entries)
            self.assertEqual(kwargs['identity'], f.identity)
            return copy.deepcopy(unit_result)
        old_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        with ExitStack() as stack:
            stack.enter_context(patch.object(parent, 'ROOT', f.root))
            stack.enter_context(patch.object(parent.socket, 'gethostname', return_value=f.identity['node']))
            stack.enter_context(patch.object(parent.os, 'sched_getaffinity', return_value=set(range(12)), create=True))
            stack.enter_context(patch.object(parent.export, 'execution_environment', return_value=f.identity))
            stack.enter_context(patch.object(parent.export, 'logged_gate', side_effect=gate_call))
            runtime = stack.enter_context(patch.object(parent.export, 'gate_runtime'))
            run = stack.enter_context(patch.object(parent, 'run_units', side_effect=units))
            stack.enter_context(patch('sys.stdout', io.StringIO()))
            code = parent.main(['--name', name, '--manifest', f.manifest_ref['file'],
                '--manifest-sha256', f.manifest_ref['sha256'], '--concurrency', '6',
                '--timeout-seconds', '1000', '--unit-timeout-seconds', '400',
                '--pair-timeout-seconds', '300', '--gate-timeout-seconds', '30'])
        self.assertEqual({sig: signal.getsignal(sig) for sig in old_signals}, old_signals)
        return code, json.loads((f.root / 'results' / (name + '.json')).read_text()), events, runtime, run

    def test_parent_gate_failure_starts_no_children_and_preserves_all_unstarted(self):
        with tempfile.TemporaryDirectory() as text:
            f = Fixture(text)
            code, data, events, runtime, run = self.invoke(f, gate_exit=86)
            self.assertEqual(code, 86)
            self.assertEqual(events, ['parent_gate'])
            runtime.assert_not_called(); run.assert_not_called()
            self.assertEqual(data['metrics']['units'], [])
            self.assertEqual(data['metrics']['unstarted_unit_indices'], [0, 1])
            self.assertEqual(len(data['sources']), 1)
            self.assertEqual(data['config']['maximum_unity_instances'], 6)

    def test_only_original_checks_complete_can_finish_and_never_authorizes_cache(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as text:
                f = Fixture(text); result = f.run(concurrency=2)
                if failure:
                    result['units'][0]['export_checks_complete'] = False
                    result['units'][0]['producer_exit_code'] = 1
                code, data, events, runtime, _ = self.invoke(f, unit_result=result)
                self.assertEqual(code, 1 if failure else 0)
                self.assertEqual(events, ['parent_gate', 'units'])
                runtime.assert_called_once()
                self.assertFalse(data['metrics']['downstream_allowed'])
                self.assertIsNone(data['metrics']['cache_matches_live'])
                self.assertEqual(data['config']['manifest_reference']['sha256'], f.manifest_ref['sha256'])
                self.assertEqual(data['config']['partition'], 'train')


if __name__ == '__main__':
    unittest.main()
