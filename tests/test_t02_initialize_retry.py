"""CPU lifecycle fixtures, not real-render acceptance certificates."""
from contextlib import contextmanager
import copy
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src.tools import t02_cache_check as cache


def init_error(action='Initialize', scene='FloorPlan203_physics'):
    return TimeoutError('Error encountered when running action ' + repr({'action': action, 'sequenceId': 2}) + ' in scene ' + scene + '.')


def failure():
    return {'pair_id': 'p', 'dataset_key': 'key', 'status': 'error', 'states': [],
        'requested_states': ['s0', 's1'], 'error': {'type': 'TimeoutError', 'message': str(init_error())},
        'cleanup_errors': [], 'controller_initialized': False, 'initialize_timeout': True}


def success():
    return {'pair_id': 'p', 'dataset_key': 'key', 'status': 'captured', 'states': [],
        'requested_states': ['s0', 's1'], 'error': None, 'cleanup_errors': [],
        'controller_initialized': True, 'initialize_timeout': False}


class InitializeRetryTests(unittest.TestCase):
    def test_exact_action_timeout_only(self):
        self.assertTrue(cache.initialize_timeout(init_error(), 'FloorPlan203'))
        for error in (TimeoutError('Worker deadline expired'), RuntimeError(str(init_error())),
                      init_error('Teleport'), init_error(scene='FloorPlan204_physics'), TimeoutError('Initialize timed out')):
            self.assertFalse(cache.initialize_timeout(error, 'FloorPlan203'))

    def lifecycle(self, rows, *, cleanup=None, group_alive=False, deadline_delta=120):
        created, stopped, events = [], [], []
        with tempfile.TemporaryDirectory() as tmp:
            config = {'name': 't02_fixture', 'scene': 'FloorPlan203', 'dataset_key': 'key',
                'directory': tmp, 'package': {'pair_id': 'p'}, 'runtime': {'cpu_affinity': [0]},
                'state_ids': ['s0', 's1'], 'deadline': time.time() + deadline_delta}
            original = copy.deepcopy(config)
            class FakeWorker:
                def __init__(self, fn, child):
                    events.append(('start', len(created)))
                    self.config = copy.deepcopy(child)
                    self.identity = {'pid': 200 + len(created), 'start_ticks': 5,
                                     'pgid': 200 + len(created), 'sid': 200 + len(created)}
                    self.process = SimpleNamespace(pid=self.identity['pid'], exitcode=0)
                    self.unity_identity = None
                    self.latest, self.error = {}, None
                    self.result = copy.deepcopy(rows[len(created)])
                    filename = ('worker_report.json' if self.result['controller_initialized'] else
                                f"initialization_attempt_{child['initialization_attempt']:02d}.json")
                    self.result['report_reference'] = cache.save_json(Path(tmp) / filename, self.result)
                    created.append(self)
                def receive(self):
                    pass
            def stop(workers):
                events.append(('stop', len(stopped))); stopped.extend(workers)
                return list(cleanup or [])
            def identity(pid):
                return copy.deepcopy(next(w.identity for w in created if w.process.pid == pid))
            def killpg(pid, signum):
                self.assertEqual(signum, 0)
                if not group_alive:
                    raise ProcessLookupError()
            with patch.object(cache, 'Worker', FakeWorker), patch.object(cache, 'stop_workers', stop), \
                 patch.object(cache, 'process_identity', identity), patch.object(cache.os, 'killpg', killpg):
                result = cache.capture_fresh_pair(config)
            self.assertEqual(config, original)
            records = [json.loads(p.read_text()) for p in sorted(Path(tmp).glob('initialization_record_*.json'))]
            self.assertEqual(len(records), len(created)); self.assertEqual(len(stopped), len(created))
            self.assertEqual([e[0] for e in events], ['start', 'stop'] * len(created))
            self.assertEqual(len({w.config['name'] for w in created}), len(created))
            self.assertEqual(len({w.config['runtime']['log_dir'] for w in created}), len(created))
            for worker in created:
                for key in ('deadline', 'state_ids', 'dataset_key', 'package'):
                    self.assertEqual(worker.config[key], original[key])
            for attempt in result['initialization_attempts']:
                stored = cache.read_json_ref(attempt['report_reference'], tmp)
                self.assertEqual(stored['status'], attempt['status'])
                self.assertEqual(stored['error'], attempt['error'])
            return result, created

    def test_two_new_workers_then_success_same_sample_deadline_and_separate_evidence(self):
        result, workers = self.lifecycle([failure(), failure(), success()])
        self.assertEqual(len(workers), 3); self.assertEqual(result['status'], 'captured')
        self.assertEqual([a['retry_started_next'] for a in result['initialization_attempts']], [True, True, False])
        self.assertTrue(all(a['cleanup_proof']['confirmed'] for a in result['initialization_attempts'][:2]))
        self.assertEqual(Path(result['report_reference']['file']).name, 'worker_report.json')

    def test_maximum_two_retries_preserves_infra_incomplete(self):
        result, workers = self.lifecycle([failure(), failure(), failure()])
        self.assertEqual(len(workers), 3); self.assertEqual(result['status'], 'error')
        self.assertFalse(result['initialization_attempts'][-1]['retry_started_next'])
        self.assertEqual(result['error']['type'], 'TimeoutError')

    def test_cleanup_errors_or_live_group_stop_before_retry(self):
        for kw in ({'cleanup': ['unknown']}, {'group_alive': True}):
            result, workers = self.lifecycle([failure()], **kw)
            self.assertEqual(len(workers), 1); self.assertTrue(result['cleanup_errors'])
            self.assertFalse(result['initialization_attempts'][0]['cleanup_proof']['confirmed'])

    def test_expired_original_deadline_prevents_retry(self):
        result, workers = self.lifecycle([failure()], deadline_delta=10)
        self.assertEqual(len(workers), 1)
        self.assertFalse(result['initialization_attempts'][0]['retry_started_next'])

    def test_masks_iou_divergence_or_noninitialize_error_never_retry(self):
        rows = []
        for reason in ('mask_iou', 'shared_divergent'):
            row = success(); row.update(status='cache_mismatch', states=[{'checks_passed': False, 'reasons': [reason]}]); rows.append(row)
        row = failure(); row['error']['message'] = str(init_error('Teleport')); rows.append(row)
        row = failure(); row['controller_initialized'] = True; rows.append(row)
        row = failure(); row['states'] = [{'partial': True}]; rows.append(row)
        for row in rows:
            result, workers = self.lifecycle([row])
            self.assertEqual(len(workers), 1); self.assertEqual(result['status'], row['status'])
            self.assertEqual(result['states'], row['states'])

    def test_cleanup_requires_original_pid_start_group_identity(self):
        worker = SimpleNamespace(identity={'pid': 20, 'start_ticks': 9}, process=SimpleNamespace(pid=20, exitcode=0))
        valid = {'pid': 20, 'start_ticks': 9, 'pgid': 20, 'sid': 20}
        with patch.object(cache.os, 'killpg', side_effect=ProcessLookupError()):
            self.assertTrue(cache.initialization_cleanup_proof(worker, valid, [])['confirmed'])
            for scope in (None, {**valid, 'start_ticks': 10}, {**valid, 'sid': 21}):
                self.assertFalse(cache.initialization_cleanup_proof(worker, scope, [])['confirmed'])
            worker.process.exitcode = None
            self.assertFalse(cache.initialization_cleanup_proof(worker, valid, [])['confirmed'])

    def test_real_fresh_worker_entry_marks_only_initialization_failure(self):
        value = SimpleNamespace(pair_id='p', scene='FloorPlan203', dataset_key='key')
        @contextmanager
        def fail_session(*args, **kwargs):
            raise init_error()
            yield
        with tempfile.TemporaryDirectory() as tmp:
            config = {'name': 't02_fixture', 'scene': 'FloorPlan203', 'directory': tmp, 'dataset_key': 'key',
                'package': {'cached_table_manifest': {'file': '/fixture/table', 'sha256': 'a'*64}},
                'runtime': {'cpu_affinity': [0]}, 'state_ids': ['s0', 's1'], 'initialization_attempt': 1}
            with patch.object(cache.os, 'sched_setaffinity', create=True), patch.object(cache, 'load_table', return_value=value), \
                 patch.object(cache, 'controller_session', fail_session), patch('sys.stderr', new_callable=io.StringIO):
                row = cache.fresh_worker(config, SimpleNamespace(send=lambda _: None))
            self.assertTrue(row['initialize_timeout']); self.assertFalse(row['controller_initialized'])
            self.assertEqual(row['states'], []); self.assertNotIn('raw_index', row)
            self.assertEqual(Path(row['report_reference']['file']).name, 'initialization_attempt_01.json')
            self.assertFalse((Path(tmp)/'worker_report.json').exists())


    def test_timeout_after_controller_body_entry_is_not_initialization_retry(self):
        from src.envs import pair_v2_simulator
        value = SimpleNamespace(pair_id='p', scene='FloorPlan203', dataset_key='key')
        @contextmanager
        def active_session(*args, **kwargs):
            yield object()
        with tempfile.TemporaryDirectory() as tmp:
            config = {'name': 't02_fixture', 'scene': 'FloorPlan203', 'directory': tmp, 'dataset_key': 'key',
                'package': {'cached_table_manifest': {'file': '/fixture/table', 'sha256': 'a'*64}, 'clone_context': None},
                'runtime': {'cpu_affinity': [0]}, 'state_ids': ['s0', 's1']}
            with patch.object(cache.os, 'sched_setaffinity', create=True), patch.object(cache, 'load_table', return_value=value), \
                 patch.object(cache, 'controller_session', active_session), \
                 patch.object(pair_v2_simulator, 'V2Simulator', side_effect=init_error()), \
                 patch('sys.stderr', new_callable=io.StringIO):
                row = cache.fresh_worker(config, SimpleNamespace(send=lambda _: None))
            self.assertFalse(row['initialize_timeout']); self.assertTrue(row['controller_initialized'])
            self.assertEqual(row['status'], 'error')
            self.assertEqual(Path(row['report_reference']['file']).name, 'worker_report.json')


if __name__ == '__main__':
    unittest.main()
