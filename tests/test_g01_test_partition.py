"""CPU-only split/source fixtures. No rendering or evaluation outcomes."""
import copy
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from src.envs import pair_v2_g01_policy as policy
from src.envs import pair_v2_g01_search as search
from src.envs import pair_v2_g01_shard as shard
from src.envs import pair_v2_fixed_labels as labels
from src.envs import pair_v2_stay_order as stay
from src.tools import g01_admission as admission
from src.tools import g01_parallel as parent
from src.tools import g01_collect_parallel as collect
from src.tools import t01v2_g01_review_checkpoint as checkpoint

GPU = 'GPU-11111111-2222-3333-4444-555555555555'


def config(scene='FloorPlan221', partition='test'):
    declared = {**policy.declared(scene, partition), **stay.fields(), 'producer_module': policy.PRODUCER}
    name = f'gen_{partition}_fixture_{scene.lower()}'
    return {**declared, 'name': name, 'scene': scene, 'raw_evidence_dir': '/synthetic/' + name,
        'deadline': time.time() + 60, 'enable_candidate_clone': False,
        'scene_pairs_per_class': policy.class_cap(partition),
        'shard': {**declared, 'scene': scene, 'name': name}}


def counted(index, kind):
    pool, mode = kind.split('_')
    return {**search.fields(), 'counts_as_pair_attempt': True, 'proposed_pool': pool,
            'mode': mode, 'kind': kind, 'status': 'passed'}


class PartitionTests(unittest.TestCase):
    def test_fixed_test_inventory_is_disjoint_and_not_available_to_default_train(self):
        expected = {f'FloorPlan{i}' for start in (21, 221, 321, 421) for i in range(start, start + 10)}
        self.assertEqual(policy.TEST_SCENES, expected)
        self.assertFalse(policy.TRAIN_SCENES & policy.TEST_SCENES)
        self.assertEqual(policy.fields()['g01_version'], policy.VERSION)
        self.assertEqual(policy.fields()['g01_policy'], policy.POLICY)
        for scene in expected:
            cfg = config(scene)
            policy.validate_source(cfg, cfg)
            self.assertEqual(cfg['field_of_view'], 60)
            self.assertEqual(cfg['scene_pairs_per_class'], 6)
            self.assertEqual(cfg['g01_version'], policy.TEST_VERSION)
            self.assertEqual(cfg['g01_search_policy'], search.POLICY)
            with self.assertRaises(ValueError): policy.declared(scene)
        for scene in policy.TRAIN_SCENES:
            policy.validate_source(config(scene, 'train'))
            with self.assertRaises(ValueError): policy.declared(scene, 'test')
        with self.assertRaises(ValueError): policy.split_scenes('auto')

    def test_tampered_partition_cap_camera_namespace_and_metrics_rejected(self):
        for change in ({'g01_partition': 'train'}, {'g01_version': policy.VERSION},
                       {'scene_pairs_per_class': 8}, {'field_of_view': 100},
                       {'name': 'gen_train_disguised'}, {'enable_candidate_clone': True}):
            cfg = config(); cfg.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): policy.validate_source(cfg)
        cfg = config(); cfg['shard']['scene'] = 'FloorPlan203'
        with self.assertRaises(ValueError): policy.validate_source(cfg)
        cfg = config(); metrics = copy.deepcopy(cfg); metrics['g01_partition'] = 'train'
        with self.assertRaises(ValueError): policy.validate_source(cfg, metrics)

    def test_six_cap_counts_actual_classes_and_keeps_original_attempt_balance(self):
        quota = search.SceneQuota(6)
        for i in range(6):
            for kind in search.KINDS:
                quota.record(f'{i}-{kind}', counted(i, kind))
        self.assertTrue(quota.complete())
        self.assertEqual(quota.summary()['per_scene_class_cap'], 6)
        self.assertEqual(quota.passed, dict.fromkeys(search.KINDS, 6))
        with self.assertRaises(search.CollectionQuotaFull): quota.before_export(counted(7, 'direct_flip'))
        with self.assertRaisesRegex(ValueError, 'unearned'):
            search.SceneQuota(6).record('first', counted(0, 'look_stay'))
        self.assertEqual(search.SceneQuota().per_class, 8)
        for bad in (True, 5, 7, 9):
            with self.assertRaises(ValueError): search.SceneQuota(bad)

    def test_spawn_scope_and_fresh_replay_retain_test_declaration_and_fixed_labels(self):
        cfg = config()
        original_tags = shard.marked.tags
        with shard.installed(cfg) as quota:
            self.assertEqual(quota.per_class, 6)
            self.assertEqual(shard.marked.tags()['g01_partition'], 'test')
            self.assertEqual(shard.marked.tags()['g01_version'], policy.TEST_VERSION)
        self.assertIs(shard.marked.tags, original_tags)
        child = {**cfg, **labels.pair_fields('floorplan221_00000', ['A', 'B']),
            'candidate_names': ['A', 'B'], 'candidate_policy': shard.marked.candidates.CANDIDATE_POLICY,
            'snapshot': {'clone_context': None}, 'enable_candidate_clone': True}
        def replay(actual, connection):
            policy.validate_source(actual)
            self.assertEqual(labels._CONTEXT.get()[0], 'floorplan221_00000')
            self.assertEqual(actual['scene_pairs_per_class'], 6)
            return {'status': 'observed'}
        with patch.object(shard.look, '_ORIGINAL_REPLAY', side_effect=replay):
            result = shard.replay_worker(child, None)
        self.assertEqual(result['g01_partition'], 'test')
        self.assertEqual(result['g01_version'], policy.TEST_VERSION)
        self.assertIsNone(labels._CONTEXT.get())

    def test_enable_config_selects_six_without_changing_original_search_or_measurement(self):
        cfg = config()
        # Original marked infrastructure is a mock boundary, not a scientific verdict.
        with patch.object(shard.stay, '_ORIGINAL_CONFIG', side_effect=lambda actual: actual):
            shard.enable_config(cfg)
        self.assertEqual(cfg['scene_pairs_per_class'], 6)
        self.assertIsNone(cfg['minimum_eligible_attempts'])
        policy.validate_source(cfg)
        self.assertEqual(cfg['g01_search_policy'], policy.declared('FloorPlan203')['g01_search_policy'])

    def test_source_context_cannot_combine_train_and_test_or_override_original_scientific_check(self):
        values = {'train': {'config': config('FloorPlan203', 'train')}, 'test': {'config': config()}}
        for value in values.values(): value['metrics'] = copy.deepcopy(value['config'])
        with patch.object(checkpoint, '_BASE_LOAD', side_effect=lambda path: (values[path], {'file': path})), \
                patch.object(checkpoint, 'verify_collection'), patch.object(checkpoint, 'verify_admission'):
            original_check = checkpoint.merge.check_pair
            with checkpoint.source_support():
                checkpoint.merge.load_source('test')
                with self.assertRaisesRegex(ValueError, 'cannot share'): checkpoint.merge.load_source('train')
                row = {**policy.fields('train'), 'producer_module': policy.PRODUCER}
                with self.assertRaisesRegex(ValueError, 'authenticated G01 source policy'):
                    checkpoint.merge.check_pair(row, 'unused', raw_root=values['test']['config']['raw_evidence_dir'])
            self.assertIs(checkpoint.merge.check_pair, original_check)


class AdmissionAndParentTests(unittest.TestCase):
    def test_real_immutable_test_plan_binds_scene_namespace_and_same_gpu_cpu_resources(self):
        scenes = ['FloorPlan21', 'FloorPlan221', 'FloorPlan321', 'FloorPlan421']
        name = 'gen_test_fixture'
        with tempfile.TemporaryDirectory() as directory:
            with collect.installed(scenes=scenes, concurrency=6, benchmark='results/benchmark.json',
                    proof={'source': {}}, partition='test', explicit_partition=True):
                rows = parent.assignments(name, 0, 6)
                plan, reference = admission.create_plan(directory, parent_run_name=name, stage_index=0,
                    concurrency=6, assignments=rows, pbs_jobid='700001.hopper-m-02', node='synthetic-node',
                    gpu_uuid=GPU, cpu_affinity=list(range(12)), scenes=scenes, partition='test')
                self.assertEqual((plan['max_unity_instances'], plan['witness_capacity']), (8, 2))
                self.assertEqual(plan['per_scene_class_cap'], 6)
                self.assertEqual(parent.PARTITION, 'test')
                assignment = admission.child_assignment(plan, reference, scenes[0])
                self.assertEqual(admission.load_plan(assignment), plan)
                gate = {'config': {'node': 'synthetic-node', 'allocation': {
                    'pbs_jobid': '700001.hopper-m-02', 'cuda_visible_devices': GPU, 'devices': [{'gpu_uuid': GPU}]}}}
                gate_record = {'source': {'file': 'results/gate.json', 'sha256': 'a'*64}, 'result_path': 'results/gate.json'}
                sh = parent.make_shard(rows[0]['child_run_name'], scenes[0], list(range(12)), gate,
                    gate_record, Path(directory) / scenes[0], plan, reference)
                cfg = {**config(scenes[0]), 'name': sh['name'], 'shard': sh}
                admission.validate_source_admission(cfg)
                self.assertEqual(sh['g01_partition'], 'test')
                cfg['g01_partition'] = 'train'
                with self.assertRaises(ValueError): admission.validate_source_admission(cfg)
                for mutate in (lambda p: p.update(g01_partition='train'),
                               lambda p: p.update(parent_run_name='gen_train_disguised'),
                               lambda p: p.update(scenes=['FloorPlan1', *scenes[1:]]),
                               lambda p: p.update(concurrency=4),
                               lambda p: p.update(per_scene_class_cap=8)):
                    altered = copy.deepcopy(plan); mutate(altered)
                    with self.assertRaises(ValueError): admission.validate_partition_plan(altered)
            self.assertEqual(parent.PARTITION, 'train')

    def test_actual_main_dispatches_test_runner_and_retains_pre_gate_failure(self):
        with tempfile.TemporaryDirectory() as text, ExitStack() as stack, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(text)
            from src.envs import thor_common
            for module in (collect, parent, thor_common): stack.enter_context(patch.object(module, 'ROOT', root))
            proof = {'source': {'file': 'results/benchmark.json'}}
            stack.enter_context(patch.object(collect, 'benchmark_stage', return_value=proof))
            stack.enter_context(patch.dict(parent.os.environ, {}, clear=True))
            launch = stack.enter_context(patch.object(parent, 'launch_logged'))
            code = collect.main(['--name', 'gen_test_fixture', '--scenes', 'FloorPlan21',
                '--concurrency', '6', '--benchmark-source', 'results/benchmark.json',
                '--timeout-seconds', '1600', '--partition', 'test'])
            self.assertEqual(code, 1)
            launch.assert_not_called()
            data = json.loads((root / 'results/gen_test_fixture.json').read_text())
            self.assertEqual(data['config']['g01_partition'], 'test')
            self.assertEqual(data['config']['per_scene_class_cap'], 6)
            self.assertEqual(data['config']['scenes'], ['FloorPlan21'])
            self.assertEqual(data['config']['concurrencies'], [6])
            self.assertEqual(data['config']['timeout_seconds'], 1600)
            self.assertIn(proof['source'], data['sources'])

    def test_committed_initial_manifest_covers_every_test_scene_once_without_outcome_selection(self):
        path = Path(__file__).resolve().parents[1] / 'configs/g01_test40_initial.json'
        plan = json.loads(path.read_text())
        scenes = [scene for unit in plan['units'] for scene in unit['scenes']]
        self.assertEqual(set(scenes), policy.TEST_SCENES)
        self.assertEqual(len(scenes), len(set(scenes)))
        self.assertFalse(set(scenes) & policy.TRAIN_SCENES)
        self.assertEqual([len(unit['scenes']) for unit in plan['units']], [12, 12, 12, 4])
        for unit in plan['units']:
            self.assertEqual(unit['gpus'], 1)
            self.assertEqual(unit['concurrency'], 6)
            self.assertIsNone(unit['requested_host'])
            self.assertEqual(unit['scene_timeout_seconds'], 1200)
            self.assertEqual(unit['parent_timeout_seconds'], collect.required_timeout(unit['scenes'], 6, 'test'))
            self.assertEqual(unit['command'][-2:], ['--partition', 'test'])
            self.assertTrue(unit['name'].startswith('gen_test_'))

    def test_test_selection_frozen_six_lanes_full_deadline_and_exact_payload(self):
        scenes = [f'FloorPlan{i}' for i in range(21, 31)]
        self.assertEqual(collect.required_timeout(scenes, 6, 'test'), 2860)
        self.assertEqual(collect.required_timeout(scenes[:4], 6, 'test'), 1600)
        for values, concurrency, partition in ((scenes, 4, 'test'), (scenes, 6, 'train'),
                (['FloorPlan1'], 6, 'test'), (['FloorPlan21', 'FloorPlan21'], 6, 'test')):
            with self.assertRaises(ValueError): collect.validate_selection(values, concurrency, partition)
        with tempfile.TemporaryDirectory() as text:
            root = Path(text); folder = root / 'logs/pbs/receipt'; folder.mkdir(parents=True)
            payload = {'name': 'gen_test_fixture', 'budget_group': 'G01', 'task_id': 'T01',
                'budget_batch': 't00b_t01_20261108', 'queue': 'autox', 'requested_gpu_count': 1,
                'requested_host': None, 'requested_wall_seconds': 2890,
                'command': ['python3', '-u', '-m', collect.MODULE, '--name', 'gen_test_fixture',
                    '--scenes', *scenes, '--concurrency', '6', '--benchmark-source', 'results/benchmark.json',
                    '--timeout-seconds', '2860', '--partition', 'test']}
            (folder / 'payload.json').write_text(json.dumps(payload))
            (folder / 'submission.json').write_text(json.dumps({'stdout': '700001.hopper-m-02\n'}))
            def verify():
                return collect.submission_identity(root, 'gen_test_fixture', '700001.hopper-m-02', 2860,
                    scenes=scenes, concurrency=6, benchmark='results/benchmark.json', partition='test', explicit_partition=True)
            self.assertEqual(verify()['budget_group'], 'G01')
            payload['command'][-1] = 'train'
            (folder / 'payload.json').write_text(json.dumps(payload))
            with self.assertRaises(ValueError): verify()


if __name__ == '__main__':
    unittest.main()
