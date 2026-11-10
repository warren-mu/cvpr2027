"""DIAG02 native opening-black evidence; original audit is the explicit fixture boundary."""
import copy
import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from PIL import Image
from src.tools import pbs_black_retry as policy

_spec = importlib.util.spec_from_file_location('original_diag02_black_fixtures',
    Path(__file__).with_name('test_pbs_black_retry.py'))
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)


class DiagEvidence(fixtures.Evidence):
    def __init__(self, root):
        super().__init__(root)
        self.name = 'diag_diag02_baselines_3b_fixture'
        self.new_name = self.name + '_retry1'
        self.manifest = {'file': '/fixed/manifest.json', 'sha256': '1' * 64}
        self.holdout = {'file': '/fixed/holdout.json', 'sha256': '2' * 64}
        self.model_source = {'file': 'results/model.json', 'sha256': '3' * 64}
        self.model_argument = dict(self.model_source, audit_passed=True, exit_code=0,
            commit='original-commit', metrics_sha256='4' * 64)
        self.command = ['python3', '-u', '-m', 'src.tools.diag02_baselines', '--name', self.name,
            '--manifest', json.dumps(self.manifest), '--holdout-reference', json.dumps(self.holdout),
            '--model-source', json.dumps(self.model_argument), '--timeout-seconds', '1440',
            '--gate-timeout-seconds', '180', '--model-size', '3b', '--model-path', '/scratch/fixture/model']
        self.payload.update(name=self.name, command=self.command, budget_group='DIAG02E3',
            requested_wall_seconds=1500, environment={'CVPR_SCRATCH': '/scratch/fixture'})
        (root/self.payload_ref['file']).chmod(0o644)
        self.payload_ref = self.write(self.payload_ref['file'], self.payload, True)
        receipt = json.loads((root/self.receipt_ref['file']).read_text())
        receipt['command'][receipt['command'].index('walltime=00:30:30')] = 'walltime=00:25:00'
        self.receipt_ref = self.write(self.receipt_ref['file'], receipt)
        self.account['config']['jobs'][0]['fields']['Submit_arguments'] = shlex.join(receipt['command'][1:])
        self.account_ref = self.logged('account', self.account,
            ['python3', 'src/tools/pbs_usage.py', '--name', 'account', self.job], 0)
        self.gate['config']['allocation']['scratch'] = '/scratch/fixture'
        self.gate_command = ['python3', '-u', '-m', 'src.tools.render_gate', '--name', self.name+'_gate',
            '--gpus', '1', '--resolution', '255', '--segmentation', '--timeout-seconds', '180.0']
        self.gate_ref = self.logged(self.name+'_gate', self.gate, self.gate_command, 86)
        directory = root/'logs/runs'/self.name
        directory.mkdir()
        self.parent = {'config': {'name': self.name, 'version': 'diag02_fixed83_cov_full_input_once_v1',
            'gpus': 1, 'node': self.node, 'manifest': json.dumps(self.manifest),
            'manifest_reference': self.manifest, 'holdout_reference': self.holdout,
            'model_source': self.model_source, 'timeout_seconds': 1440., 'gate_timeout_seconds': 180.,
            'model': 'Qwen/Qwen2.5-VL-3B-Instruct', 'model_size': '3b', 'model_path': '/scratch/fixture/model', 'maximum_requested_gpu_hours': .5,
            'strategies': ['obs_never', 'obs_trigger', 'obs_trigger_cov'],
            'training_config_changed': False, 'model_settings_reused_from': 'src.tools.smoke_e2e_eval.VllmPolicy',
            'episode_directory': str(directory), 'identity': {'pbs_jobid': self.job,
                'cuda_visible_devices': self.gpu, 'cpu_affinity': list(range(12)), 'scratch': '/scratch/fixture'}},
            'metrics': {'kind': 'diag02_original_model', 'completed': False, 'errors': [],
                'cleanup_errors': [], 'black_frame_detected': True, 'paper_result': False,
                'gate': self.gate_ref, 'journals': [], 'producer_exit_code': 86}}
        self.proof.update(native_kind='diag02_original_model', original_payload=self.payload_ref,
            original_receipt=self.receipt_ref, original_command_unchanged=self.command,
            pbs_accounting=self.account_ref)
        self.refresh_parent()

    def refresh_parent(self, exit_code=86):
        self.parent_ref = self.logged(self.name, self.parent, self.command, exit_code)
        self.proof['source_results'] = [self.parent_ref, self.gate_ref]
        self.refresh_guard()

    def arguments(self, host='hopper-28'):
        args = super().arguments(host)
        args.update(group='DIAG02E3', wall_seconds=1500)
        return args


class DiagBlackRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.e = DiagEvidence(self.root)
    def tearDown(self):
        self.temp.cleanup()

    def test_native_parent_exact_none_group_and_runtime_replay(self):
        self.assertNotIn('pbs_jobid', self.e.parent['config'])
        self.assertIn('metrics_sha256', self.e.model_argument)
        self.assertEqual(set(self.e.parent['config']['model_source']), {'file', 'sha256'})
        binding = self.e.validate()
        p = dict(self.e.payload, name=self.e.new_name, command=self.e.arguments()['command'],
            requested_host='hopper-28', retry_of=self.e.job, black_retry_host_authorization=binding)
        self.assertEqual(policy.validate_payload_host(self.root, p), binding)
        self.assertEqual(binding['original_gpu_uuid'], self.e.gpu)
        self.assertIsNone(policy.validate_payload_host(self.root, {'requested_host': None}))

    def test_changed_command_group_wall_or_gpu_rejected(self):
        for update in ({'group': 'T01v2'}, {'group': 'E2Epilot'}, {'task': 'T00b'},
                       {'gpus': 2}, {'wall_seconds': 3299}, {'name': 'smoke_alias'}):
            args = self.e.arguments(); args.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                policy.validate_request_host(self.root, **args)
        for flag, value in [('--manifest', '{}'), ('--model-source', '{}'),
                            ('--timeout-seconds', '3239'), ('-m', 'src.tools.smoke_e2e_eval')]:
            args = self.e.arguments(); args['command'][args['command'].index(flag)+1] = value
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                policy.validate_request_host(self.root, **args)

    def test_completed_model_journals_cleanup_or_nonblack_rejected(self):
        original = copy.deepcopy(self.e.parent)
        for update in ({'completed': True}, {'journals': [{'file': 'real.jsonl'}]},
                       {'model_initialization_seconds': .1}, {'errors': ['TimeoutError']},
                       {'cleanup_errors': ['owned worker remains']}, {'black_frame_detected': False},
                       {'producer_exit_code': 1}):
            self.e.parent = copy.deepcopy(original); self.e.parent['metrics'].update(update)
            self.e.refresh_parent()
            with self.subTest(update=update), self.assertRaises(ValueError): self.e.validate()
        self.e.parent = original; self.e.refresh_parent(1)
        with self.assertRaises(ValueError): self.e.validate()

    def test_actual_hidden_journal_and_wrong_job_uuid_rejected(self):
        path = self.root/'logs/runs'/self.e.name/'progress.jsonl'; path.write_text('model began\n')
        with self.assertRaisesRegex(ValueError, 'journals'): self.e.validate()
        path.unlink()
        original = copy.deepcopy(self.e.parent)
        for key, value in [('pbs_jobid', 'other'), ('cuda_visible_devices', 'GPU-other'),
                           ('cpu_affinity', [0]), ('scratch', '/wrong')]:
            self.e.parent = copy.deepcopy(original); self.e.parent['config']['identity'][key] = value
            self.e.refresh_parent()
            with self.subTest(key=key), self.assertRaises(ValueError): self.e.validate()

    def test_nonblack_original_png_still_rejected(self):
        path = self.root/self.e.png_refs[0]['file']; Image.new('RGB', (255,255), (1,0,0)).save(path)
        ref = self.e.ref(path); self.e.proof['captures'][0] = ref
        self.e.gate['metrics']['devices'][0]['observation']['captures'][0]['png']['sha256'] = ref['sha256']
        self.e.gate_ref = self.e.logged(self.e.name+'_gate', self.e.gate, self.e.gate_command, 86)
        self.e.parent['metrics']['gate'] = self.e.gate_ref; self.e.refresh_parent()
        with self.assertRaisesRegex(ValueError, 'decoded PNG'): self.e.validate()

    def test_old_g01_path_unchanged_and_diag_cannot_use_old_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            old = fixtures.Evidence(Path(folder).resolve())
            self.assertEqual(old.validate()['version'], policy.VERSION)
        with self.assertRaises(ValueError):
            policy.validate_request_host(self.root, **self.e.arguments('hopper-22'),
                _recorded_rotation_index=0, _recorded_version=policy.LEGACY_VERSION)

if __name__ == '__main__': unittest.main()
