"""Synthetic original-artifact chains; no scheduler/network/GPU operations."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('external_budget_fixture',
    Path(__file__).resolve().parents[1] / 'scripts/pbs/submit_phase.py')
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)
JOB = '709137.hopper-m-02'
POLICY = {'rates': {'smallx': 9, 'mediumx': 9, 'largex': 9}}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __init__(self, repo, rate='1'):
        repo = repo.resolve()
        self.repo = repo
        self.registry = {'version': submit.EXTERNAL_FINAL_VERSION, 'budget_batch': submit.BATCH,
                         'entries': [dict(name='test_round2_cpu_01', job_id=JOB, task_id='T01',
                             budget_group='G01', submission_transport='direct_qsub', requested_gpu_count=0)]}
        self.entry = self.registry['entries'][0]
        # The real production auditor is a separate tested boundary. This fixture
        # marks audit eligibility only; all SHA/command/log/final-cost code is real.
        self.write('scripts/audit_provenance.py', 'def audit_file(path):\n    return {"fail": []}\n')
        self.command = ['qsub', '-N', 'cvpr27_cupre01', '-o', str(repo/'logs/pbs/test_round2_cpu_01/pbs.out'),
                        'experiments/g01_test_round2_cpu_prepare.sh']
        self.account = {
            'config': {'jobs': [{'job_id': JOB, 'qstat_exit': 0, 'fields': {
                'job_state': 'F', 'project': 'CFP04-SF-125', 'Job_Owner': 'e1583364@fixture',
                'Job_Name': 'cvpr27_cupre01', 'queue': 'interactive', 'run_count': '1',
                'exec_host': 'hopper-24/0*12', 'Resource_List.ngpus': '1',
                'Resource_List.walltime': '00:10:00', 'resources_used.walltime': '00:00:03',
                'Resource_List.queue_charge_rate': rate, 'Submit_arguments': shlex.join(self.command[1:]),
                'Output_Path': 'fixture:' + str(repo/'logs/pbs/test_round2_cpu_01/pbs.out')}}], 'errors': []},
            'metrics': {'complete_allocation_accounting': True, 'allocation_gpu_hours_by_job': {JOB: 3/3600}},
            '_meta': {'command': ['python3', 'src/tools/pbs_usage.py', '--name', 'g01_round2_cpu_attempt_accounting_01', JOB]}}
        self.submission = {'config': {'command': self.command}, 'metrics': {'command_exit_code': 0, 'elapsed_seconds': 1},
            '_meta': {'command': ['python3', '-m', 'src.tools.record_command', '--name', 'diag_submit_test_round2_cpu_01', '--', *self.command]}}
        self.save_logged('submission_result', 'diag_submit_test_round2_cpu_01', self.submission, job_line=True)
        self.save_logged('accounting_result', 'g01_round2_cpu_attempt_accounting_01', self.account)
        self.save_ref('submission_transcript', 'logs/pbs/test_round2_cpu_01/submission_transcript.txt', JOB + '\n')
        self.save_ref('request_script', 'experiments/g01_test_round2_cpu_prepare.sh',
            '#!/bin/bash\n#PBS -P CFP04-SF-125\n#PBS -q interactive\n'
            '#PBS -l select=1:ncpus=1:mem=8gb:ngpus=0\n#PBS -l walltime=00:10:00\n')
        self.save_ref('pbs_output', 'logs/pbs/test_round2_cpu_01/pbs.out', 'Original CPU allocation guard failed\n')
        self.save_registry()

    def write(self, relative, text):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def save_ref(self, key, relative, text):
        path = self.write(relative, text)
        self.entry[key] = {'file': relative, 'sha256': digest(path)}

    def save_logged(self, key, name, data, job_line=False):
        meta = data['_meta']
        meta.update(name=name, end='2026-11-09T00:00:00', exit_code=0, log_path='logs/runs/' + name + '.log',
                    metrics_sha256=hashlib.sha256(json.dumps(data['metrics'], sort_keys=True).encode()).hexdigest())
        self.write(meta['log_path'], json.dumps(data['config'], indent=2) + '\n' + (JOB + '\n' if job_line else '')
                   + 'FINAL_METRICS_JSON: ' + json.dumps(data['metrics'], sort_keys=True) + '\n')
        self.save_ref(key, 'results/' + name + '.json', json.dumps(data))

    def save_registry(self):
        self.write(submit.EXTERNAL_FINAL_CONFIG, json.dumps(self.registry))

    def save_account(self):
        self.save_logged('accounting_result', 'g01_round2_cpu_attempt_accounting_01', self.account)
        self.save_registry()

    def budget(self):
        return submit.budget_state(self.repo, policy=POLICY, live_rates={})


class ExternalFinalBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name))

    def test_actual_rate_not_current_premium_rate_counts_once_in_three_ledgers(self):
        for rate in ('1', '2.5'):
            with self.subTest(rate=rate):
                f = self.fixture
                f.account['config']['jobs'][0]['fields']['Resource_List.queue_charge_rate'] = rate
                f.save_account()
                result = f.budget()
                expected = 3/3600 * float(rate)
                self.assertEqual(result['accounted_or_reserved_by_task']['T01'], expected)
                self.assertEqual(result['accounted_or_reserved_by_subbudget']['G01'], expected)
                self.assertEqual(result['accounted_or_reserved_total'], expected)
                self.assertEqual(result['accounted_or_reserved_by_subbudget']['T02'], 0)
                self.assertEqual(len(result['submissions']), 1)
                row = result['submissions'][0]
                self.assertEqual((row['requested_gpu_count'], row['actual_gpu_count']), (0, 1))
                self.assertEqual(row['submission_transport'], 'direct_qsub')
                self.assertEqual(row['actual_queue'], 'interactive')
                self.assertIsNone(row['payload'])
                self.assertEqual(row['basis'], 'final_accounting')
                self.assertEqual(f.budget()['accounted_or_reserved_total'], expected)

    def test_no_final_incomplete_cost_or_wrong_rate_never_reserves_zero(self):
        cases = [('job_state', 'R'), ('resources_used.walltime', ''),
                 ('Resource_List.queue_charge_rate', 'nan'), ('Resource_List.queue_charge_rate', '0')]
        for field, value in cases:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(Path(tmp)); f.account['config']['jobs'][0]['fields'][field] = value; f.save_account()
                with self.assertRaises(RuntimeError): f.budget()
        for metric, value in [('complete_allocation_accounting', False), ('allocation_gpu_hours_by_job', {JOB: 0})]:
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(Path(tmp)); f.account['metrics'][metric] = value; f.save_account()
                with self.assertRaises(RuntimeError): f.budget()

    def test_duplicate_registry_or_normal_payload_cannot_double_count(self):
        f = self.fixture
        f.registry['entries'].append(copy.deepcopy(f.entry)); f.save_registry()
        with self.assertRaises(RuntimeError): f.budget()
        f.registry['entries'].pop(); f.save_registry()
        for batch in (submit.BATCH, 'another_batch'):
            f.write('logs/pbs/normal/payload.json', json.dumps({'budget_batch': batch, 'dry_run': False,
                'task_id': 'T01', 'name': 'gen_test_normal', 'command': [], 'budget_group': 'G01',
                'requested_gpu_count': 1, 'requested_wall_seconds': 600}))
            f.write('logs/pbs/normal/submission.json', json.dumps({'stdout': JOB, 'returncode': 0}))
            with self.subTest(batch=batch), self.assertRaises(RuntimeError): f.budget()

    def test_missing_registration_or_original_evidence_is_not_silently_ignored(self):
        f = self.fixture
        path = f.repo/submit.EXTERNAL_FINAL_CONFIG
        path.unlink()
        with self.assertRaises(RuntimeError): f.budget()
        f.save_registry()
        f.registry['entries'] = []; f.save_registry()
        with self.assertRaises(RuntimeError): f.budget()

    def test_tampered_bytes_and_wrong_qsub_binding_are_rejected(self):
        for role in ('submission_result', 'submission_transcript', 'request_script', 'pbs_output', 'accounting_result'):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(Path(tmp)); path = f.repo/f.entry[role]['file']; path.write_text(path.read_text()+' ')
                with self.assertRaises(RuntimeError): f.budget()
        for field, value in [('Submit_arguments', '-N unrelated'), ('Output_Path', 'host:/elsewhere'),
                             ('project', 'wrong'), ('Resource_List.ngpus', '0'), ('queue', 'smallx')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(Path(tmp)); f.account['config']['jobs'][0]['fields'][field] = value; f.save_account()
                with self.assertRaises(RuntimeError): f.budget()

    def test_exact_source_audit_and_original_request_still_required(self):
        f = self.fixture
        f.write('scripts/audit_provenance.py', 'def audit_file(path):\n    return {"fail": ["fixture provenance failure"]}\n')
        with self.assertRaises(RuntimeError): f.budget()
        f.write('scripts/audit_provenance.py', 'def audit_file(path):\n    return {"fail": []}\n')
        path = f.repo/f.entry['request_script']['file']
        f.save_ref('request_script', f.entry['request_script']['file'], path.read_text().replace('ngpus=0', 'ngpus=1'))
        f.save_registry()
        with self.assertRaises(RuntimeError): f.budget()

    def test_missing_logged_end_and_unlogged_final_fields_are_rejected(self):
        f = self.fixture
        f.account['_meta']['end'] = None
        f.save_ref('accounting_result', 'results/g01_round2_cpu_attempt_accounting_01.json', json.dumps(f.account))
        f.save_registry()
        with self.assertRaises(RuntimeError): f.budget()
        f.save_account()
        f.write(f.account['_meta']['log_path'], 'No original PBS config here\n')
        with self.assertRaises(RuntimeError): f.budget()

    def test_regular_unfinished_reservation_remains_separate(self):
        f = self.fixture
        f.write('logs/pbs/normal/payload.json', json.dumps({'budget_batch': submit.BATCH, 'dry_run': False,
            'task_id': 'T01', 'name': 'gen_test_normal', 'command': [], 'budget_group': 'G01',
            'requested_gpu_count': 1, 'requested_wall_seconds': 600, 'requested_queue_charge_rate': 1}))
        f.write('logs/pbs/normal/submission.json', json.dumps({'stdout': '999.hopper-m-02', 'returncode': 0}))
        result = submit.budget_state(f.repo, policy=POLICY, live_rates={'999.hopper-m-02': 2})
        self.assertEqual(result['accounted_or_reserved_total'], 600/3600*2 + 3/3600)
        self.assertEqual([r['basis'] for r in result['submissions']], ['full_request_reservation', 'final_accounting'])


if __name__ == '__main__':
    unittest.main()
