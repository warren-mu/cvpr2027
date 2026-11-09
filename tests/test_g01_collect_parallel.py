"""CPU fixtures: no renderer, live resource claim, or synthetic throughput result."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools import g01_collect_parallel as collect
from src.tools import g01_parallel as parent
from src.tools import g01_admission as admission

GPU = 'GPU-11111111-2222-3333-4444-555555555555'


def reference(name, code=0):
    return {'file': f'results/{name}.json', 'sha256': name + '-sha',
            'metrics_sha256': name + '-metrics', 'git_commit': 'synthetic-fixture', 'exit_code': code}


class BenchmarkFixture:
    """Audit is the mock boundary; plans, shard files, policy and cleanup are real checks."""
    def __init__(self, root, concurrency=4):
        self.root, self.name, self.concurrency = root, 'gen_train_fixture_benchmark', concurrency
        stage_index = collect.BENCHMARK_CONCURRENCIES.index(concurrency)
        self.reference = reference(self.name, 1)
        self.lookup = {}
        job = '700000.hopper-m-02'
        allocation = {'pbs_jobid': job, 'cuda_visible_devices': GPU, 'devices': [{'gpu_uuid': GPU}]}
        gate_name = self.name + '_gate'
        gate_ref = reference(gate_name)
        gate = {'_meta': {'name': gate_name, 'exit_code': 0, 'end': 'fixture-ended'},
            'config': {'allocation': allocation, 'node': 'fixture-node', 'resolution': 255, 'segmentation': True},
            'metrics': {'passed': True, 'cleanup_errors': [], 'devices': [{'observation': {'captures': [
                {'segmentation': {'valid': True}}, {'segmentation': {'valid': True}}]}}]}}
        self.gate_record = {'run_name': gate_name, 'result_path': gate_ref['file'], 'source': gate_ref, 'exit_code': 0}
        self.lookup[gate_ref['file']] = gate, gate_ref
        directory = root / 'scratch/c4'
        directory.mkdir(parents=True)
        assignments = parent.assignments(self.name, stage_index, concurrency)
        plan, plan_ref = admission.create_plan(directory, parent_run_name=self.name, stage_index=stage_index,
            concurrency=concurrency, assignments=assignments, pbs_jobid=job, node='fixture-node',
            gpu_uuid=GPU, cpu_affinity=list(range(12)), scenes=list(collect.BENCHMARK_SCENES))
        self.plan, self.plan_ref = plan, plan_ref
        timeout = parent.required_timeout()
        payload = {'name': self.name, 'dry_run': False, 'budget_group': 'G01', 'task_id': 'T01',
            'budget_batch': 't00b_t01_20261108', 'queue': 'autox', 'requested_host': None,
            'requested_gpu_count': 1, 'requested_wall_seconds': timeout + 30,
            'command': ['python3', '-m', 'src.tools.g01_parallel', '--name', self.name,
                        '--timeout-seconds', str(timeout)]}
        receipt = root / 'logs/pbs/fixture'
        receipt.mkdir(parents=True)
        (receipt / 'payload.json').write_text(json.dumps(payload))
        (receipt / 'submission.json').write_text(json.dumps({'stdout': job + '\n'}))
        cfg = {'name': self.name, 'pbs_jobid': job, 'node': 'fixture-node', 'producer_module': parent.policy.PRODUCER,
            'gpus': 1, 'cpu_cores_per_gpu': 12, 'cpu_affinity': list(range(12)),
            'parent_cuda_visible_devices': GPU, 'scenes': list(collect.BENCHMARK_SCENES),
            'concurrencies': list(collect.BENCHMARK_CONCURRENCIES), 'timeout_seconds': timeout,
            'required_timeout_seconds': timeout, 'scene_timeout_seconds': parent.SCENE_SECONDS,
            'child_finalization_seconds': parent.CHILD_FINALIZATION_SECONDS,
            'gate_timeout_seconds': parent.GATE_SECONDS, 'gate_wrapper_seconds': parent.GATE_WRAPPER_SECONDS,
            'stage_finalization_seconds': parent.STAGE_FINALIZATION_SECONDS,
            'parent_cleanup_seconds': parent.PARENT_CLEANUP_SECONDS,
            'stage_plans': [{**plan_ref, 'plan': plan}],
            'submission': parent.submission_identity(root, self.name, job, timeout)}
        clean = {'poison': None, 'active_leases': {}}
        self.stage = {'stage_index': stage_index, 'concurrency': concurrency, 'completed': True,
            'fixed_scenes': list(collect.BENCHMARK_SCENES), 'plan': plan_ref,
            'admission_state_at_stage_end': copy.deepcopy(clean), 'wall_seconds_observed': 10.0}
        self.rows, self.starts, self.children = [], [], []
        sources = [gate_ref]
        for assignment in assignments:
            scene, name = assignment['scene'], assignment['child_run_name']
            raw = directory / scene
            raw.mkdir()
            shard = parent.make_shard(name, scene, list(range(12)), gate, self.gate_record, raw, plan, plan_ref)
            path = raw / 'shard_config.json'
            path.write_text(json.dumps(shard))
            child_cfg = {**copy.deepcopy(shard), 'shard': shard, 'enable_candidate_clone': False,
                         'scene_pairs_per_class': 8}
            report = {'version': admission.VERSION, 'assignment': shard['g01_admission'],
                'cleanup_confirmed': True, 'active_leases': [], 'errors': [],
                'owner': {'pid': 900, 'start_ticks': 1000, 'pgid': 900, 'sid': 900}, 'lease_records': []}
            child = {'_meta': {'name': name, 'exit_code': 1, 'end': 'fixture-ended',
                'command': ['python3', '-u', '-m', parent.policy.PRODUCER, '--name', name, '--scene', scene,
                            '--shard-config', str(path), '--timeout-seconds', '1200']},
                'config': child_cfg, 'metrics': {**copy.deepcopy(child_cfg), 'kind': 'pair_v2_shard',
                    'cleanup_errors': [], 'g01_witness_admission': report}}
            child_ref = reference(name, 1)
            self.children.append(child)
            self.lookup[child_ref['file']] = child, child_ref
            sources.append(child_ref)
            common = {'scene': scene, 'run_name': name, 'stage_index': stage_index, 'concurrency': concurrency,
                      'lane_index': assignment['lane_index'], 'logical_index': 0, 'gpu_uuid': GPU}
            self.rows.append({**common, 'source': child_ref, 'result_path': child_ref['file'], 'exit_code': 1,
                              'granted_timeout_seconds': 1200, 'stopped_by_orchestrator': False})
            self.starts.append({**common, 'cpu_affinity': list(range(12)), 'shard_config': str(path),
                                'shard_config_sha256': collect.sha_file(path)})
        self.data = {'_meta': {'name': self.name, 'exit_code': 1, 'end': 'fixture-ended', 'command': payload['command']},
            'config': cfg, 'sources': sources, 'metrics': {'kind': 'g01_parallel_benchmark',
                'exit_code': 1, 'status': 'completed', 'cleanup_errors': [], 'black_frame_detected': False,
                'gate': self.gate_record, 'stages': [self.stage], 'scene_runs': self.rows,
                'started_shards': self.starts, 'admission_final': [{'stage_index': stage_index, **clean}]}}
        self.lookup[self.reference['file']] = self.data, self.reference

    def audit(self, path):
        path = Path(path)
        key = str(path.relative_to(self.root)) if path.is_absolute() else str(path)
        return self.lookup[key]

    def verify(self):
        with patch.object(collect, 'ROOT', self.root), patch.object(collect, 'audited_result', self.audit):
            return collect.benchmark_stage(self.reference['file'], self.concurrency)


class CollectionTests(unittest.TestCase):
    def test_only_unique_training_scenes_and_whole_wave_deadlines(self):
        scenes = ['FloorPlan1', 'FloorPlan7', 'FloorPlan203', 'FloorPlan204', 'FloorPlan301']
        self.assertEqual(collect.runner_timeout(scenes, 4), 2740)
        self.assertEqual(collect.required_timeout(scenes, 4), 2860)
        for values, count in (([], 2), (['FloorPlan1', 'FloorPlan1'], 2), (['FloorPlan21'], 2), (scenes, 3)):
            with self.assertRaises(ValueError): collect.required_timeout(values, count)
        with collect.installed(scenes=scenes, concurrency=4, benchmark='results/benchmark.json', proof={'source': {}}):
            self.assertEqual(parent.required_timeout(), collect.runner_timeout(scenes, 4))
            self.assertEqual(parent.CONCURRENCIES, (4,))
            self.assertEqual([row['scene'] for row in parent.assignments('gen_train_fixture', 0, 4)], scenes)
            with self.assertRaises(TimeoutError): parent.full_scene_grant(1319, 0)

    def test_real_plan_and_resource_checks_accept_clean_ended_partial_scenes_without_inventing_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BenchmarkFixture(Path(temporary))
            proof = fixture.verify()
            self.assertEqual(proof['parent_exit_code'], 1)
            self.assertEqual(proof['concurrency'], 4)
            self.assertEqual(len(proof['children']), 6)
            self.assertTrue(all(row['exit_code'] == 1 for row in proof['children']))
            self.assertIn('no scientific pair acceptance', proof['scope'])

    def test_incomplete_shortened_or_unknown_cleanup_benchmark_is_rejected(self):
        for mutate in (
            lambda f: f.stage.update(completed=False),
            lambda f: f.rows[0].update(granted_timeout_seconds=1199),
            lambda f: f.rows[0].update(stopped_by_orchestrator=True),
            lambda f: f.rows.pop(),
            lambda f: f.children[0]['metrics']['g01_witness_admission'].update(cleanup_confirmed=False),
            lambda f: f.children[0]['metrics'].update(g01_witness_admission=None),
            lambda f: f.data['sources'].pop(),
        ):
            with tempfile.TemporaryDirectory() as temporary:
                fixture = BenchmarkFixture(Path(temporary))
                mutate(fixture)
                with self.subTest(mutate=mutate), self.assertRaises(ValueError): fixture.verify()

    def test_exact_collection_payload_prevents_silent_scene_or_concurrency_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'logs/pbs/collect'
            folder.mkdir(parents=True)
            scenes, count, source = ['FloorPlan1', 'FloorPlan201'], 2, 'results/gen_train_fixture.json'
            timeout = collect.required_timeout(scenes, count)
            payload = {'name': 'gen_train_collect_fixture', 'budget_group': 'G01', 'task_id': 'T01',
                'budget_batch': 't00b_t01_20261108', 'queue': 'autox', 'requested_gpu_count': 1,
                'requested_wall_seconds': timeout + 30, 'requested_host': None,
                'command': ['python3', '-u', '-m', collect.MODULE, '--name', 'gen_train_collect_fixture',
                    '--scenes', *scenes, '--concurrency', '2', '--benchmark-source', source,
                    '--timeout-seconds', str(timeout)]}
            (folder / 'payload.json').write_text(json.dumps(payload))
            (folder / 'submission.json').write_text(json.dumps({'stdout': '700001.hopper-m-02\n'}))
            def verify():
                return collect.submission_identity(root, payload['name'], '700001.hopper-m-02', timeout,
                    scenes=scenes, concurrency=count, benchmark=source)
            self.assertEqual(verify()['requested_wall_seconds'], timeout + 30)
            payload['command'][payload['command'].index('--concurrency') + 1] = '4'
            (folder / 'payload.json').write_text(json.dumps(payload))
            with self.assertRaises(ValueError): verify()

    def test_scoped_runner_restores_hooks_on_exception_and_emits_collection_not_benchmark(self):
        keys = ('SCENES', 'CONCURRENCIES', 'submission_identity', 'write_result', 'finish_logged', 'launch_logged')
        previous = {key: getattr(parent, key) for key in keys}
        proof = {'source': reference('gen_train_fixture_benchmark')}
        saved = []
        def write(name, metrics, config):
            saved.append(copy.deepcopy((name, metrics, config)))
        with patch.object(parent, 'write_result', write):
            with self.assertRaisesRegex(RuntimeError, 'fixture interruption'):
                with collect.installed(scenes=['FloorPlan1'], concurrency=2, benchmark='results/b.json', proof=proof):
                    self.assertIs(parent.launch_logged, previous['launch_logged'])
                    parent.write_result('gen_train_fixture', {'kind': 'g01_parallel_benchmark'},
                                        {'timeout_seconds': parent.required_timeout()})
                    raise RuntimeError('fixture interruption')
            self.assertIs(parent.write_result, write)
        self.assertTrue(all(getattr(parent, key) is value for key, value in previous.items()))
        self.assertEqual(saved[0][1]['kind'], 'g01_parallel_collection')
        self.assertEqual(saved[0][2]['benchmark_selection'], proof)
        self.assertEqual(saved[0][2]['timeout_seconds'], collect.required_timeout(['FloorPlan1'], 2))

    def test_actual_runner_pre_gate_failure_preserves_collection_scope_and_benchmark_source(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(temporary)
            scenes, proof = ['FloorPlan1'], {'source': reference('gen_train_fixture_benchmark')}
            stack.enter_context(patch.object(collect, 'ROOT', root))
            stack.enter_context(patch.object(parent, 'ROOT', root))
            from src.envs import thor_common
            stack.enter_context(patch.object(thor_common, 'ROOT', root))
            stack.enter_context(patch.object(collect, 'benchmark_stage', return_value=proof))
            stack.enter_context(patch.dict(parent.os.environ, {}, clear=True))
            launched = stack.enter_context(patch.object(parent, 'launch_logged'))
            code = collect.main(['--name', 'gen_train_collect_fixture', '--scenes', *scenes,
                '--concurrency', '2', '--benchmark-source', 'results/benchmark.json',
                '--timeout-seconds', str(collect.required_timeout(scenes, 2))])
            self.assertEqual(code, 1)
            launched.assert_not_called()
            result = json.loads((root / 'results/gen_train_collect_fixture.json').read_text())
            self.assertEqual(result['metrics']['kind'], 'g01_parallel_collection')
            self.assertEqual(result['config']['concurrencies'], [2])
            self.assertEqual(result['config']['scenes'], scenes)
            self.assertEqual(result['sources'], [proof['source']])
            self.assertEqual(result['metrics']['scene_runs'], [])
            self.assertEqual(len(result['metrics']['unstarted_assignments']), 1)


if __name__ == '__main__':
    unittest.main()
