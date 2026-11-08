"""Synthetic terminal-protocol checks only; no Unity or research measurements."""
import copy
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.envs import pair_v2_terminal as terminal
from src.envs import pair_v2_isolated_shard as isolated


def row(index=0, *, passed=False, measured=None):
    return {'pair_id': f'floorplan1_{index:05d}', 'scene': 'FloorPlan1',
        'status': 'passed' if passed else 'rejected', 'kind': 'look_flip' if passed else None,
        'discard_reasons': [] if passed else ['synthetic_obstruction'],
        'generation_seconds': 2.0, 'anchor_type': 'Mug', 'w0_valid_physical': measured,
        'pair_file': 'data/world_pairs/v2/fixture/pair.json' if passed else None,
        'pair_sha256': 'a' * 64 if passed else None}


def setup(directory, rows=None):
    rows = [row()] if rows is None else rows
    config = {'name': 'fixture', 'scene': 'FloorPlan1', 'deadline': 1000.0,
        'max_attempts': 160, 'shard': {'pbs_jobid': 'synthetic.job'},
        'attempt_journal': 'data/world_pairs/v2/fixture_attempts.jsonl'}
    terminal.terminal_config(config, now=900)
    path = Path(directory) / config['attempt_journal']
    path.parent.mkdir(parents=True)
    path.write_bytes(b''.join(json.dumps(item, sort_keys=True).encode() + b'\n' for item in rows))
    # These are hand-specified fixture observations, not results of a simulator.
    metrics = {'kind': 'pair_v2_shard', 'status': 'completed_search', 'cleanup_errors': [],
        'black_frame_detected': False, 'acceptance_passed': False,
        'attempt_journal': config['attempt_journal'], 'attempts': 1,
        'passed_by_class': {kind: 0 for kind in terminal.KINDS},
        'discard_reason_counts': {'synthetic_obstruction': 1}, 'passed_pair_files': [],
        'mean_attempt_generation_seconds': 2.0,
        'w0_by_anchor': {'Mug': {'measured_attempts': 0, 'passed': 0, 'rejected': 0,
                               'measurement_unavailable': 1}}}
    return config, path, metrics


def sealed(metrics, config, root):
    return {'metrics': copy.deepcopy(metrics), 'receipt': terminal.seal_terminal(metrics, config, root)}


def worker(payload, progress=None):
    return SimpleNamespace(terminal_payload=payload, last_progress=progress or {}, error=None,
        process=SimpleNamespace(pid=os.getpid()), owned_cleanup_complete=True)


class TerminalTests(unittest.TestCase):
    def test_soft_deadline_reserves_existing_thirty_seconds_and_does_not_mutate_parent(self):
        config = {'deadline': 500.0}
        child = terminal.terminal_config(config, now=100)
        self.assertEqual(config['deadline'], 500)
        self.assertEqual(child['deadline'], 470)
        self.assertEqual(child['terminal_protocol']['parent_hard_deadline'], 500)
        child['terminal_protocol']['policy']['version'] = 'modified'
        self.assertEqual(config['terminal_protocol']['policy']['version'], terminal.TERMINAL_VERSION)
        with self.assertRaises(TimeoutError):
            terminal.terminal_config({'deadline': 500}, now=470)

    def test_valid_receipt_checks_complete_actual_counters_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            config, path, metrics = setup(directory)
            payload = sealed(metrics, config, directory)
            result, valid = terminal.final_metrics(metrics, config, worker(payload), directory)
            self.assertTrue(valid)
            self.assertEqual(result['attempts'], 1)
            self.assertEqual(result['attempt_journal_sha256'], payload['receipt']['journal']['sha256'])
            self.assertEqual(payload['receipt']['journal']['bytes'], len(path.read_bytes()))
            self.assertTrue(result['terminal_verified_after_cleanup'])

    def test_original_race_n_plus_one_journal_cannot_seal_n_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            config, path, metrics = setup(directory)
            with path.open('ab') as stream:
                stream.write(json.dumps(row(1)).encode() + b'\n')
            with self.assertRaisesRegex(ValueError, 'attempts'):
                terminal.seal_terminal(metrics, config, directory)
            self.assertEqual(len(path.read_bytes().splitlines()), 2)

    def test_child_class_discard_w0_or_timing_mismatch_is_rejected_not_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, metrics = setup(directory)
            changes = [('passed_by_class', {kind: 1 for kind in terminal.KINDS}),
                       ('discard_reason_counts', {}), ('w0_by_anchor', {}),
                       ('mean_attempt_generation_seconds', 1.0)]
            for key, value in changes:
                with self.subTest(field=key):
                    changed = copy.deepcopy(metrics)
                    changed[key] = value
                    with self.assertRaisesRegex(ValueError, key):
                        terminal.seal_terminal(changed, config, directory)

    def test_incomplete_last_line_and_wrong_scene_never_become_zero_or_final(self):
        with tempfile.TemporaryDirectory() as directory:
            config, path, metrics = setup(directory)
            original = path.read_bytes()
            for data in (original + b'{"unfinished":', original.replace(b'FloorPlan1', b'FloorPlan2')):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    terminal.seal_terminal(metrics, config, directory)

    def test_empty_existing_journal_can_seal_actual_zero_but_missing_journal_cannot(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, metrics = setup(directory, [])
            metrics.update(attempts=0, discard_reason_counts={}, mean_attempt_generation_seconds=None,
                           w0_by_anchor={})
            self.assertEqual(sealed(metrics, config, directory)['receipt']['journal']['row_count'], 0)
            missing = copy.deepcopy(config)
            missing.update(name='missing', attempt_journal='data/world_pairs/v2/missing_attempts.jsonl')
            with self.assertRaises(FileNotFoundError):
                terminal.journal_summary(missing, directory)

    def test_full_metrics_or_worker_identity_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, metrics = setup(directory)
            payload = sealed(metrics, config, directory)
            with self.assertRaises(ValueError):
                terminal.verify_terminal(payload, config, directory, expected_pid=os.getpid() + 1)
            payload['metrics']['status'] = 'incomplete'
            with self.assertRaises(ValueError):
                terminal.verify_terminal(payload, config, directory)

    def test_post_cleanup_append_fails_and_does_not_pair_old_counts_with_new_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            config, path, metrics = setup(directory)
            payload = sealed(metrics, config, directory)
            with path.open('ab') as stream:
                stream.write(json.dumps(row(1)).encode() + b'\n')
            parent = {**metrics, 'attempt_journal_sha256': 'newer_unverified_hash'}
            result, valid = terminal.final_metrics(parent, config, worker(payload, metrics), directory)
            self.assertFalse(valid)
            self.assertEqual(result['status'], 'worker_failed')
            for key in ('attempts', 'passed_by_class', 'discard_reason_counts', 'attempt_journal_sha256'):
                self.assertNotIn(key, result)
            self.assertNotIn('attempt_journal_sha256', result['last_progress_nonfinal'])
            self.assertEqual(result['unverified_terminal_payload'], payload)

    def test_late_queued_terminal_is_authoritative_instead_of_stale_parent_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, metrics = setup(directory)
            payload = sealed(metrics, config, directory)
            parent = {**metrics, 'attempts': 0, 'status': 'shard_failed', 'error': 'parent deadline'}
            result, valid = terminal.final_metrics(parent, config, worker(payload), directory)
            self.assertTrue(valid)
            self.assertEqual(result['attempts'], 1)
            self.assertEqual(result['parent_before_terminal_publication']['error'], 'parent deadline')

    def test_missing_receipt_preserves_black_evidence_without_final_counts(self):
        metrics = {'attempts': 7, 'cleanup_errors': [], 'black_frame_detected': True,
                   'black_frame_reference': {'file': 'actual_black.json', 'sha256': 'b' * 64}}
        result, valid = terminal.final_metrics(metrics, {}, worker(None, metrics))
        self.assertFalse(valid)
        self.assertTrue(result['black_frame_detected'])
        self.assertEqual(result['black_frame_reference'], metrics['black_frame_reference'])
        self.assertNotIn('attempts', result)
        self.assertEqual(result['last_progress_nonfinal']['attempts'], 7)

    def test_latest_n_progress_with_n_plus_one_journal_without_receipt_stays_nonfinal(self):
        with tempfile.TemporaryDirectory() as directory:
            config, path, metrics = setup(directory)
            with path.open('ab') as stream:
                stream.write(json.dumps(row(1)).encode() + b'\n')
            parent = {**metrics, 'attempt_journal_sha256': 'old-main-post-cleanup-hash'}
            result, valid = terminal.final_metrics(parent, config, worker(None, metrics), directory)
            self.assertFalse(valid)
            self.assertEqual(result['last_progress_nonfinal']['attempts'], 1)
            self.assertNotIn('attempts', result)
            self.assertNotIn('attempt_journal_sha256', result)
            self.assertEqual(len(path.read_bytes().splitlines()), 2)

    def test_cleanup_errors_never_become_authoritative_even_with_matching_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, metrics = setup(directory)
            metrics['cleanup_errors'] = ['actual child cleanup uncertainty']
            payload = sealed(metrics, config, directory)
            result, valid = terminal.final_metrics(metrics, config, worker(payload), directory)
            self.assertFalse(valid)
            self.assertEqual(result['cleanup_errors'], metrics['cleanup_errors'])

    def test_terminal_worker_seals_only_after_generator_returns_through_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, metrics = setup(directory)
            events, sent = [], []
            def generator(cfg, connection):
                try:
                    events.append('generate')
                    return metrics
                finally:
                    events.append('cleanup')
            original = terminal.seal_terminal
            def seal(value, cfg):
                self.assertEqual(events, ['generate', 'cleanup'])
                events.append('seal')
                return original(value, cfg, directory)
            with patch.object(terminal.signal, 'setitimer'), patch.object(terminal, 'seal_terminal', seal):
                terminal.terminal_worker(config, SimpleNamespace(send=sent.append), generator=generator)
            self.assertEqual(sent[0][0], 'terminal_receipt')
            self.assertEqual(events[-1], 'seal')


class ReceiverTests(unittest.TestCase):
    def test_partial_frame_blocks_only_reader_and_reader_does_not_close_owned_pipe(self):
        entered, release = threading.Event(), threading.Event()
        class Connection:
            closed = False
            def recv(self):
                entered.set()
                release.wait()
                raise EOFError()
            def close(self):
                self.closed = True
        connection = Connection()
        reader = terminal.PipeReader(connection)
        try:
            self.assertTrue(entered.wait(1))
            self.assertEqual(reader.drain(), [])
            self.assertFalse(reader.done.is_set())
            self.assertFalse(connection.closed)
        finally:
            release.set()
            self.assertTrue(reader.done.wait(1))
            self.assertEqual(reader.after_owned_cleanup(), [])
        self.assertFalse(connection.closed)

    def test_drain_has_fixed_message_bound(self):
        reader = terminal.PipeReader.__new__(terminal.PipeReader)
        reader.messages = queue.Queue()
        for index in range(40):
            reader.messages.put(('progress', {'index': index}))
        self.assertEqual(len(reader.drain()), 16)
        self.assertFalse(reader.messages.empty())

    def test_full_queue_at_hard_cleanup_cannot_turn_undelivered_terminal_into_success(self):
        reached = threading.Event()
        class Connection:
            index = 0
            def recv(self):
                self.index += 1
                if self.index <= 8:
                    return ('progress', {'attempts': self.index})
                reached.set()
                return ('terminal_receipt', {'synthetic': 'not-yet-delivered'})
        reader = terminal.PipeReader(Connection())
        self.assertTrue(reached.wait(1))
        # No process or connection is owned by the reader. Closing its queue
        # at the hard boundary may forfeit a receipt, but must fail closed.
        self.assertEqual(reader.after_owned_cleanup(), [])
        messages = reader.drain()
        self.assertEqual([kind for kind, _ in messages], ['progress'] * 8)
        item = terminal.TerminalWorker.__new__(terminal.TerminalWorker)
        item.reader = SimpleNamespace(drain=lambda: messages, done=reader.done, messages=queue.Queue())
        item.terminal_payload, item.result, item.error = None, None, None
        item.receive()
        self.assertIsNone(item.result)
        self.assertEqual(item.error['type'], 'WorkerExit')

    def test_plain_result_is_not_a_terminal_receipt(self):
        item = terminal.TerminalWorker.__new__(terminal.TerminalWorker)
        item.reader = SimpleNamespace(drain=lambda: [('result', {'attempts': 99})],
            done=threading.Event(), messages=queue.Queue())
        item.reader.done.set()
        item.result, item.error = None, None
        item.receive()
        self.assertIsNone(item.result)
        self.assertEqual(item.error['type'], 'WorkerExit')


class MainIntegrationTests(unittest.TestCase):
    def test_writer_post_cleanup_mismatch_forces_nonzero_for_old_zero_and_black_codes(self):
        for old_code in (0, 86):
            with self.subTest(old_code=old_code), tempfile.TemporaryDirectory() as directory:
                config, path, metrics = setup(directory)
                payload = sealed(metrics, config, directory)
                instance = worker(payload, metrics)
                outputs = []
                actual = terminal.final_metrics
                def finalize(m, c, w):
                    return actual(m, c, w, directory)
                def main():
                    isolated.base.Worker(None, config)
                    with path.open('ab') as stream:
                        stream.write(json.dumps(row(1)).encode() + b'\n')
                    isolated.base.write_result('fixture', metrics, config)
                    return old_code
                with patch.object(isolated.previous, 'main', main), \
                        patch.object(isolated.terminal, 'TerminalWorker', lambda f, c: instance), \
                        patch.object(isolated.terminal, 'final_metrics', finalize), \
                        patch.object(isolated.base, 'write_result', lambda n, m, c: outputs.append(m)):
                    self.assertEqual(isolated.main(), 1)
                self.assertEqual(len(outputs), 1)
                self.assertFalse(outputs[0]['final_metrics_verified'])
                self.assertNotIn('attempt_journal_sha256', outputs[0])

    def test_verified_late_receipt_keeps_parent_failure_exit_and_valid_black_keeps_86(self):
        for old_code, black in ((1, False), (86, True)):
            with self.subTest(old_code=old_code), tempfile.TemporaryDirectory() as directory:
                config, _, metrics = setup(directory)
                metrics['black_frame_detected'] = black
                if black:
                    metrics.update(status='black_frame', black_frame_reference={'file': 'actual.json'})
                payload = sealed(metrics, config, directory)
                instance = worker(payload, metrics)
                outputs = []
                actual = terminal.final_metrics
                def finalize(m, c, w):
                    return actual(m, c, w, directory)
                def main():
                    isolated.base.Worker(None, config)
                    parent = copy.deepcopy(metrics)
                    if old_code == 1:
                        parent.update(status='shard_failed', attempts=0, error='parent deadline')
                    else:
                        parent['black_frame_stop'] = {'synthetic': 'parent separately verified portable proof'}
                    isolated.base.write_result('fixture', parent, config)
                    return old_code
                with patch.object(isolated.previous, 'main', main), \
                        patch.object(isolated.terminal, 'TerminalWorker', lambda f, c: instance), \
                        patch.object(isolated.terminal, 'final_metrics', finalize), \
                        patch.object(isolated.base, 'write_result', lambda n, m, c: outputs.append(m)):
                    self.assertEqual(isolated.main(), old_code)
                self.assertTrue(outputs[0]['final_metrics_verified'])
                self.assertEqual(outputs[0]['attempts'], 1)


if __name__ == '__main__':
    unittest.main()
