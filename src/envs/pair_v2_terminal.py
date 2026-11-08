"""Future-only bounded terminal handshake; progress never certifies a journal.

The unchanged generator owns all scientific measurements and closes its journal
and Controllers before returning. This adapter only authenticates that return.
It never repairs old metrics, truncates a journal, or deletes failed evidence.
"""
from collections import Counter
import copy
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import threading
import time

from . import thor_common as common

TERMINAL_VERSION = 'sealed_journal_terminal_v1'
FINALIZATION_GRACE_SECONDS = 30
MAX_JOURNAL_BYTES = 32 * 1024 * 1024
KINDS = ('direct_flip', 'direct_stay', 'look_flip', 'look_stay')
TERMINAL_POLICY = {
    'version': TERMINAL_VERSION,
    'finalization_grace_seconds': FINALIZATION_GRACE_SECONDS,
    'deadline': 'worker soft deadline is parent hard deadline minus grace; no added allocation time',
    'authoritative': 'generator returned after cleanup; matching full metrics and complete sealed journal',
    'post_cleanup': 'parent verifies identical seal after owned process cleanup',
    'missing_receipt': 'nonzero; last progress explicitly nonfinal; no final counters or journal hash',
    'journal_read_limit_bytes': MAX_JOURNAL_BYTES,
}


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def terminal_config(config, now=None):
    """Keep the original parent allowance; only shorten generation itself."""
    now = time.time() if now is None else now
    hard = config['deadline']
    if not isinstance(hard, (int, float)) or not math.isfinite(hard):
        raise ValueError('Invalid parent terminal deadline')
    soft = hard - FINALIZATION_GRACE_SECONDS
    if soft <= now:
        raise TimeoutError('Insufficient existing allowance for terminal finalization grace')
    protocol = {'version': TERMINAL_VERSION, 'policy': copy.deepcopy(TERMINAL_POLICY),
                'worker_soft_deadline': soft, 'parent_hard_deadline': hard}
    config['terminal_protocol'] = copy.deepcopy(protocol)
    child = copy.deepcopy(config)
    child['deadline'] = soft
    return child


def journal_summary(config, root=common.ROOT):
    """Read one bounded, complete journal; derive, never repair, its counters."""
    expected = f'data/world_pairs/v2/{config["name"]}_attempts.jsonl'
    if config['attempt_journal'] != expected:
        raise ValueError('Terminal journal path does not match the explicit run')
    path = (Path(root) / expected).resolve()
    if not path.is_relative_to((Path(root) / 'data/world_pairs/v2').resolve()):
        raise ValueError('Terminal journal escaped its source directory')
    with path.open('rb') as stream:
        raw = stream.read(MAX_JOURNAL_BYTES + 1)
    if len(raw) > MAX_JOURNAL_BYTES or (raw and not raw.endswith(b'\n')):
        raise ValueError('Terminal journal exceeds bound or ends with an incomplete row')
    rows = []
    for index, line in enumerate(raw.splitlines()):
        row = json.loads(line)
        if (not isinstance(row, dict) or row.get('scene') != config['scene']
                or row.get('pair_id') != f'{config["scene"].lower()}_{index:05d}'
                or row.get('status') not in ('passed', 'rejected')
                or not isinstance(row.get('discard_reasons'), list)
                or any(not isinstance(reason, str) for reason in row['discard_reasons'])):
            raise ValueError('Terminal journal identity/order/status is invalid')
        seconds = row.get('generation_seconds')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Terminal attempt lacks measured finite generation time')
        if row['status'] == 'passed' and (row.get('kind') not in KINDS
                or not row.get('pair_file') or not row.get('pair_sha256')):
            raise ValueError('Passed terminal row lacks its class or saved pair reference')
        if row['status'] != 'passed' and (row.get('pair_file') or row.get('pair_sha256')):
            raise ValueError('Rejected terminal row still claims a passed pair')
        rows.append(row)
    if len(rows) > config['max_attempts']:
        raise ValueError('Terminal journal exceeds the declared attempt bound')
    w0 = {}
    for row in rows:
        anchor = row.get('anchor_type')
        if not isinstance(anchor, str) or not anchor:
            raise ValueError('Search terminal row lacks its observed anchor type')
        item = w0.setdefault(anchor, {'measured_attempts': 0, 'passed': 0,
                                     'rejected': 0, 'measurement_unavailable': 0})
        value = row.get('w0_valid_physical')
        if type(value) is bool:
            item['measured_attempts'] += 1
            item['passed' if value else 'rejected'] += 1
        elif value is None:
            item['measurement_unavailable'] += 1
        else:
            raise ValueError('Terminal w0 measurement is neither an actual bool nor unknown')
    counters = {
        'attempts': len(rows),
        'passed_by_class': {kind: sum(row['status'] == 'passed' and row.get('kind') == kind
                                     for row in rows) for kind in KINDS},
        'discard_reason_counts': dict(Counter(reason for row in rows for reason in row['discard_reasons'])),
        'passed_pair_files': [{'file': row['pair_file'], 'sha256': row['pair_sha256']}
                              for row in rows if row['status'] == 'passed'],
        'mean_attempt_generation_seconds': sum(row['generation_seconds'] for row in rows) / len(rows) if rows else None,
        'w0_by_anchor': w0,
    }
    return {'file': expected, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
            'row_count': len(rows), 'counters': counters}


def checked_journal(metrics, config, root=common.ROOT):
    summary = journal_summary(config, root)
    if metrics.get('attempt_journal') != summary['file']:
        raise ValueError('Returned metrics name another journal')
    for key, expected in summary['counters'].items():
        if key not in metrics or encoded(metrics[key]) != encoded(expected):
            raise ValueError(f'Returned metrics disagree with complete journal: {key}')
    return summary


def seal_terminal(metrics, config, root=common.ROOT):
    if not isinstance(metrics, dict) or not isinstance(metrics.get('cleanup_errors'), list):
        raise ValueError('Terminal metrics lack actual cleanup observations')
    journal = checked_journal(metrics, config, root)
    return {'version': TERMINAL_VERSION, 'name': config['name'], 'scene': config['scene'],
            'pbs_jobid': config['shard']['pbs_jobid'],
            'worker_pid': os.getpid(),
            'protocol': copy.deepcopy(config['terminal_protocol']),
            'metrics_sha256': digest(metrics), 'journal': journal,
            'generator_returned_after_cleanup': True}


def verify_terminal(payload, config, root=common.ROOT, expected_pid=None):
    if not isinstance(payload, dict) or set(payload) != {'metrics', 'receipt'}:
        raise ValueError('Missing explicit terminal envelope')
    metrics, receipt = payload['metrics'], payload['receipt']
    if (receipt.get('version') != TERMINAL_VERSION or receipt.get('name') != config['name']
            or receipt.get('scene') != config['scene']
            or receipt.get('pbs_jobid') != config['shard']['pbs_jobid']
            or receipt.get('protocol') != config['terminal_protocol']
            or receipt.get('generator_returned_after_cleanup') is not True
            or type(receipt.get('worker_pid')) is not int
            or (expected_pid is not None and receipt['worker_pid'] != expected_pid)
            or receipt.get('metrics_sha256') != digest(metrics)):
        raise ValueError('Terminal receipt identity or full-metrics digest mismatch')
    if checked_journal(metrics, config, root) != receipt.get('journal'):
        raise ValueError('Sealed journal changed or receipt counters disagree')
    return copy.deepcopy(metrics)


def terminal_worker(config, connection, *, generator):
    metrics = generator(config, connection)
    # Generation has returned through its original finally blocks. An early
    # return must not leave the soft alarm armed during sealing/IPC.
    signal.setitimer(signal.ITIMER_REAL, max(.1, config['terminal_protocol']['parent_hard_deadline'] - time.time()))
    receipt = seal_terminal(metrics, config)
    connection.send(('terminal_receipt', {'metrics': metrics, 'receipt': receipt}))
    # thor_common emits a conventional result afterwards. It is deliberately
    # not authoritative; the parent accepts only terminal_receipt.
    return {'terminal_protocol': TERMINAL_VERSION, 'receipt_sent': True}


class PipeReader:
    """One recv owner; a partial large pipe frame cannot block the parent loop."""
    def __init__(self, connection):
        self.connection = connection
        self.messages = queue.Queue(maxsize=8)
        self.closed, self.done = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True, name='t01v2-terminal-pipe')
        self.thread.start()

    def _read(self):
        try:
            while not self.closed.is_set():
                value = self.connection.recv()
                while not self.closed.is_set():
                    try:
                        self.messages.put(value, timeout=.05)
                        break
                    except queue.Full:
                        pass
        except (EOFError, OSError):
            pass
        finally:
            self.done.set()

    def drain(self):
        result = []
        for _ in range(16):
            try:
                result.append(self.messages.get_nowait())
            except queue.Empty:
                break
        return result

    def after_owned_cleanup(self):
        # This reader never closes the shared connection or signals a process.
        # common.stop_workers remains the only owner of both operations.
        self.closed.set()
        self.thread.join(timeout=.2)
        return [] if not self.thread.is_alive() else ['Terminal pipe reader did not exit after owned cleanup']


class TerminalWorker(common.Worker):
    def __init__(self, generator, config):
        child = terminal_config(config)
        self.parent_config = config
        self.terminal_payload = None
        self.last_progress = {}
        self.owned_cleanup_complete = False
        super().__init__(partial(terminal_worker, generator=generator), child)
        self.reader = PipeReader(self.connection)

    def receive(self):
        for message in self.reader.drain():
            if not isinstance(message, tuple) or len(message) != 2:
                self.error = {'type': 'TerminalProtocolError', 'message': 'Invalid worker message'}
                continue
            kind, value = message
            if kind == 'progress':
                if self.terminal_payload is not None:
                    self.error = {'type': 'TerminalProtocolError', 'message': 'Progress arrived after seal'}
                else:
                    self.last_progress = copy.deepcopy(value)
                    self.latest = copy.deepcopy(value)
            elif kind == 'unity_identity':
                self.unity_identity = value
            elif kind == 'ready':
                self.ready = True
            elif kind == 'terminal_receipt':
                try:
                    if self.terminal_payload is not None:
                        raise ValueError('Duplicate terminal receipt')
                    verified = verify_terminal(value, self.parent_config, expected_pid=self.process.pid)
                    self.terminal_payload = copy.deepcopy(value)
                    self.result = verified
                    self.latest = copy.deepcopy(verified)
                except Exception as error:
                    self.error = {'type': type(error).__name__, 'message': str(error)}
            elif kind == 'error':
                self.error = value
            elif kind != 'result':
                self.error = {'type': 'TerminalProtocolError', 'message': 'Unknown worker message'}
        if self.error is not None:
            self.result = None
        if self.reader.done.is_set() and self.reader.messages.empty() and self.result is None and self.error is None:
            self.error = {'type': 'WorkerExit', 'message': 'Worker closed without a verified terminal receipt'}

    def stop(self):
        try:
            errors = super().stop()
        except BaseException as error:
            errors = [f'Owned cleanup raised {type(error).__name__}: {error}']
        errors.extend(self.reader.after_owned_cleanup())
        # A terminal message already in the queue must not be lost merely
        # because the parent's last loop iteration reached its hard deadline.
        self.receive()
        self.owned_cleanup_complete = not errors
        return errors


def final_metrics(metrics, config, worker, root=common.ROOT):
    """Called after unchanged owned cleanup, immediately before writing result."""
    failure = None
    try:
        if worker is None or worker.terminal_payload is None:
            raise ValueError('No verified terminal receipt; progress is not final')
        if worker.error is not None or not worker.owned_cleanup_complete:
            raise ValueError('Worker protocol or owned process cleanup did not complete')
        verified = verify_terminal(worker.terminal_payload, config, root, worker.process.pid)
        if verified.get('cleanup_errors') or metrics.get('cleanup_errors'):
            raise ValueError('Cleanup errors prevent authoritative terminal publication')
        # A last queued receipt can be drained during owned cleanup. Its full
        # metrics, never the older parent progress object, are authoritative.
        result = verified
        for key in ('cleanup_errors', 'black_frame_stop', 'black_frame_evidence_error',
                    'source_black_frame_detected', 'capture_black_frame_detected'):
            if key in metrics:
                result[key] = copy.deepcopy(metrics[key])
        if metrics.get('error') != verified.get('error') or metrics.get('status') != verified.get('status'):
            result['parent_before_terminal_publication'] = {
                'status': metrics.get('status'), 'error': copy.deepcopy(metrics.get('error')),
                'scope': 'parent observation retained separately from the authenticated terminal metrics'}
        result.update(attempt_journal_sha256=worker.terminal_payload['receipt']['journal']['sha256'],
            terminal_receipt=copy.deepcopy(worker.terminal_payload['receipt']),
            terminal_verified_after_cleanup=True, final_metrics_verified=True,
            terminal_generator_status=verified['status'])
        return result, True
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
    progress = copy.deepcopy(worker.last_progress if worker is not None else metrics)
    # The old main may have attached a newer journal hash to this stale object.
    progress.pop('attempt_journal_sha256', None)
    cleanup = copy.deepcopy(metrics.get('cleanup_errors'))
    if worker is not None and worker.terminal_payload is not None:
        child_cleanup = worker.terminal_payload['metrics'].get('cleanup_errors')
        if isinstance(cleanup, list) and isinstance(child_cleanup, list):
            cleanup.extend(item for item in child_cleanup if item not in cleanup)
    result = {'kind': 'pair_v2_shard', 'status': 'worker_failed', 'acceptance_passed': False,
        'final_metrics_verified': False, 'terminal_verified_after_cleanup': False,
        'terminal_finalization_error': failure, 'last_progress_nonfinal': progress,
        'progress_scope': 'observed progress only; no final attempt/class/discard counts or final journal hash',
        'cleanup_errors': cleanup,
        'black_frame_detected': metrics.get('black_frame_detected'),
        'terminal_protocol': copy.deepcopy(config.get('terminal_protocol'))}
    for key in ('error', 'black_frame_reference', 'black_frame_stop', 'terminal_evidence',
                'black_frame_evidence_error', 'source_black_frame_detected', 'capture_black_frame_detected'):
        if key in metrics:
            result[key] = copy.deepcopy(metrics[key])
    for key in ('preflight_version', 'evidence_version', 'search_version', 'search_policy',
                'solvability_version', 'solvability_policy', 'generation_version', 'generation_policy',
                'attempt_setup_version', 'attempt_setup_policy', 'proposal_order_version',
                'proposal_order_policy', 'enable_candidate_clone'):
        if key in config:
            result[key] = copy.deepcopy(config[key])
    if worker is not None and worker.terminal_payload is not None:
        result['unverified_terminal_payload'] = copy.deepcopy(worker.terminal_payload)
    return result, False
