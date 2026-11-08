"""Append-only common evidence and verdict-dependent in-memory attempt evidence.

No staging files are deleted. Only an explicitly completed retention decision
materializes buffered payloads. Unexpected process loss leaves an explicit
incomplete record rather than claiming that unreturned bytes were preserved.
"""
from contextlib import contextmanager
import builtins
import hashlib
import io
import json
import os
from pathlib import Path
import stat


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def reference(path, content):
    return {'file': str(path), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}


class EvidenceSession:
    def __init__(self, scratch, *, buffer_all=False, imports=None):
        self.scratch = Path(scratch).resolve()
        self.active = buffer_all
        self.stores = {}
        self.pending = {}
        self.imports = dict(imports or {})
        for path, data in self.imports.items():
            self.checked_path(path)
            if not isinstance(data, bytes):
                raise TypeError('Imported private evidence must contain bytes')

    def checked_path(self, path):
        value = Path(path).resolve()
        if value == self.scratch or not value.is_relative_to(self.scratch):
            raise ValueError('Evidence payload escaped allocated scratch')
        return value

    def store(self, directory, max_bytes):
        directory = self.checked_path(directory)
        if str(directory) in self.stores or directory.exists():
            raise FileExistsError('Private evidence directory already exists: ' + str(directory))
        result = BufferedStore(self, directory, max_bytes)
        self.stores[str(directory)] = result
        if not self.active:
            result.ensure_directory()
        return result

    def begin(self):
        if self.active or self.pending:
            raise RuntimeError('Nested or unfinished attempt evidence session')
        self.active = True

    def data(self, path):
        # Do not call Path.resolve here: older pathlib.resolve calls Path.stat,
        # which is itself temporarily routed through this read-only view.
        key = os.path.realpath(os.fsdecode(path))
        if key in self.pending:
            return self.pending[key]['content']
        return self.imports.get(key)

    def bundle(self):
        return {'schema_version': 1,
                'stores': [{'directory': str(store.directory), 'max_bytes': store.max_bytes}
                           for store in self.stores.values()],
                'payloads': [{'file': path, **row} for path, row in self.pending.items()]}

    def absorb(self, bundle):
        if bundle.get('schema_version') != 1 or not self.active:
            raise ValueError('Cannot absorb an unknown or out-of-attempt evidence bundle')
        for info in bundle['stores']:
            directory = self.checked_path(info['directory'])
            key = str(directory)
            if key in self.stores:
                raise ValueError('Replay evidence store identity reused')
            self.stores[key] = BufferedStore(self, directory, int(info['max_bytes']))
        for row in bundle['payloads']:
            path = self.checked_path(row['file'])
            store = self.stores[str(path.parent)]
            if row['directory'] != str(path.parent) or not isinstance(row['content'], bytes):
                raise ValueError('Replay evidence bundle path/type mismatch')
            store.save(path.name, row['content'])

    def finish(self, keep):
        if not self.active:
            raise RuntimeError('No active evidence session')
        files, size = len(self.pending), sum(len(row['content']) for row in self.pending.values())
        if keep:
            # Each byte is written once with exclusive creation; any partial IO
            # failure is retained, never removed or reported as a full archive.
            for path, row in list(self.pending.items()):
                store = self.stores[row['directory']]
                store.persist(Path(path), row['content'])
                del self.pending[path]
        else:
            self.pending.clear()  # In-memory buffers only; never unlink a file.
        for store in self.stores.values():
            store.pending_bytes = 0
        self.active = False
        return {'full_raw_retained': bool(keep), 'buffered_payload_files': files,
                'buffered_payload_bytes': size, 'disk_files_deleted': 0}

    @contextmanager
    def reading(self):
        """Private-process read-only view; writes always use the real filesystem."""
        old_open, old_stat, old_builtin_open = Path.open, Path.stat, builtins.open
        session = self

        def opened(path, mode='r', buffering=-1, encoding=None, errors=None, newline=None):
            data = session.data(path)
            if data is not None and mode in ('r', 'rt', 'rb'):
                return io.BytesIO(data) if 'b' in mode else io.TextIOWrapper(
                    io.BytesIO(data), encoding=encoding, errors=errors, newline=newline)
            return old_open(path, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline)

        def stated(path, *, follow_symlinks=True):
            data = session.data(path)
            if data is not None:
                return os.stat_result((stat.S_IFREG | 0o600, 0, 0, 1, 0, 0, len(data), 0, 0, 0))
            return old_stat(path, follow_symlinks=follow_symlinks)

        def builtin_opened(file, mode='r', buffering=-1, encoding=None, errors=None,
                           newline=None, closefd=True, opener=None):
            data = session.data(file) if isinstance(file, (str, bytes, os.PathLike)) else None
            if data is not None and mode in ('r', 'rt', 'rb'):
                return io.BytesIO(data) if 'b' in mode else io.TextIOWrapper(
                    io.BytesIO(data), encoding=encoding, errors=errors, newline=newline)
            return old_builtin_open(file, mode=mode, buffering=buffering, encoding=encoding,
                                    errors=errors, newline=newline, closefd=closefd, opener=opener)

        try:
            Path.open, Path.stat = opened, stated
            builtins.open = builtin_opened
            yield
        finally:
            Path.open, Path.stat = old_open, old_stat
            builtins.open = old_builtin_open


class BufferedStore:
    def __init__(self, session, directory, max_bytes):
        self.session, self.directory, self.max_bytes = session, Path(directory), int(max_bytes)
        self.files, self.bytes, self.pending_bytes = [], 0, 0
        self.reserved = set()
        self.incremental = self.directory / 'evidence_index.jsonl'
        self.initialized = False

    def ensure_directory(self):
        if not self.initialized:
            self.directory.mkdir(parents=True, exist_ok=False)
            with self.incremental.open('xb'):
                pass
            self.initialized = True

    def persist(self, path, content):
        self.ensure_directory()
        with path.open('xb') as stream:
            stream.write(content)
        ref = reference(path, content)
        self.files.append(ref)
        self.bytes += len(content)
        with self.incremental.open('ab') as stream:
            stream.write(encoded(ref))
        return ref

    def save(self, name, content):
        path = self.directory / name
        if path.parent != self.directory or self.bytes + self.pending_bytes + len(content) > self.max_bytes:
            raise ValueError('Raw evidence path or per-store payload byte limit exceeded; prior files retained')
        key = str(path)
        if key in self.reserved:
            raise FileExistsError('Evidence payload name reused: ' + key)
        self.reserved.add(key)
        if self.session.active:
            if key in self.session.pending:
                raise FileExistsError('Evidence buffer identity collision')
            self.session.pending[key] = {'directory': str(self.directory), 'content': content}
            self.pending_bytes += len(content)
            return reference(path, content)
        return self.persist(path, content)

    def save_json(self, name, value):
        return self.save(name, encoded(value))

    def index(self):
        pending = [reference(path, row['content']) for path, row in self.session.pending.items()
                   if row['directory'] == str(self.directory)]
        return self.save_json('evidence_index.json', {'files': self.files + pending,
            'raw_bytes': self.bytes + self.pending_bytes, 'max_raw_bytes': self.max_bytes,
            'raw_files_not_deleted': True, 'omitted_attempt_payloads_not_materialized': True})


class SearchEvidenceSession(EvidenceSession):
    """Supplement-3 storage cap: stop raw writes, never stop scientific search.

Only this explicit new class changes cap behavior. The earlier EvidenceSession
remains unchanged for frozen producers. A retention result is NOT scientific
acceptance: callers must require evidence_complete_for_acceptance before export
and keep their ordinary independent checks. Compact journals/results use their
own writers and are never suppressed by this raw-payload budget.
"""
    def __init__(self, scratch, *, raw_budget_bytes, terminal_reserve_bytes=0,
                 buffer_all=False, imports=None):
        super().__init__(scratch, buffer_all=buffer_all, imports=imports)
        if (type(raw_budget_bytes) is not int or type(terminal_reserve_bytes) is not int
                or not 0 <= terminal_reserve_bytes <= raw_budget_bytes):
            raise ValueError('Require explicit nonnegative raw budget and contained terminal reserve')
        self.raw_budget_bytes = raw_budget_bytes
        self.terminal_reserve_bytes = terminal_reserve_bytes
        self.raw_saving_stopped = False
        self.cap_events = []
        self.resident = {}  # Unsaved common runtime references, retained across attempts.
        self.terminal_reference = None

    def store(self, directory, max_bytes):
        directory = self.checked_path(directory)
        if str(directory) in self.stores or directory.exists():
            raise FileExistsError('Private evidence directory already exists: ' + str(directory))
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('Require a nonnegative per-store raw-payload limit')
        result = SearchBufferedStore(self, directory, max_bytes)
        self.stores[str(directory)] = result
        if not self.active:
            result.ensure_directory()
        return result

    def data(self, path):
        key = os.path.realpath(os.fsdecode(path))
        row = self.resident.get(key)
        return row['content'] if row is not None else super().data(path)

    def status(self):
        return {'raw_budget_bytes': self.raw_budget_bytes,
                'terminal_reserve_bytes': self.terminal_reserve_bytes,
                'stored_payload_bytes': sum(store.bytes for store in self.stores.values()),
                'resident_common_bytes': sum(len(row['content']) for row in self.resident.values()),
                'current_attempt_buffer_bytes': sum(len(row['content']) for row in self.pending.values()),
                'raw_saving_stopped': self.raw_saving_stopped,
                'cap_events': list(self.cap_events), 'disk_files_deleted': 0,
                'scope': 'raw payload bytes; compact records and append-only index overhead are outside this cap'}

    def _plan(self, payloads, *, terminal=False):
        added = sum(len(row['content']) for row in payloads.values())
        used = sum(store.bytes for store in self.stores.values())
        limit = self.raw_budget_bytes if terminal else self.raw_budget_bytes - self.terminal_reserve_bytes
        per_store = {}
        for row in payloads.values():
            per_store[row['directory']] = per_store.get(row['directory'], 0) + len(row['content'])
        exhausted = [directory for directory, count in per_store.items()
                     if self.stores[directory].bytes + count > self.stores[directory].max_bytes]
        allowed = not exhausted and used + added <= limit and (terminal or not self.raw_saving_stopped)
        return {'can_materialize_complete': bool(allowed), 'required_payload_bytes': added,
                'stored_payload_bytes': used, 'payload_limit_bytes': limit,
                'remaining_payload_bytes': max(0, limit - used),
                'per_store_capacity_exceeded': exhausted, 'terminal_reserve_requested': terminal,
                'raw_saving_already_stopped': self.raw_saving_stopped}

    def retention_plan(self):
        """Conservative complete package: all unsaved common + current attempt."""
        return self._plan({**self.resident, **self.pending})

    def _hit_cap(self, plan):
        if not self.raw_saving_stopped:
            self.cap_events.append(dict(plan))
        self.raw_saving_stopped = True

    def _materialize(self, payloads):
        for path, row in payloads.items():
            self.stores[row['directory']].persist(Path(path), row['content'])
            self.pending.pop(path, None)
            self.resident.pop(path, None)

    def finish(self, keep):
        if not self.active:
            raise RuntimeError('No active evidence session')
        files, size = len(self.pending), sum(len(row['content']) for row in self.pending.values())
        terminal = None
        if self.terminal_reference is not None:
            terminal = self.minimal_black_flush(self.terminal_reference)
        plan = self.retention_plan()
        complete = bool(keep and plan['can_materialize_complete'])
        if complete:
            self._materialize({**self.resident, **self.pending})
        elif keep:
            self._hit_cap(plan)
        # Current attempt bytes are no longer needed once its verdict is saved.
        # Common baseline/query references stay available for later candidates.
        self.pending.clear()
        for store in self.stores.values():
            store.pending_bytes = 0
        self.active = False
        self.terminal_reference = None
        return {'full_raw_retained': complete, 'evidence_complete_for_acceptance': complete,
                'retention_requested': bool(keep), 'cap_hit': bool(keep and not complete),
                'capacity_plan': plan, 'buffered_payload_files': files,
                'buffered_payload_bytes': size, 'search_may_continue': True,
                'candidate_requires_evidence_incomplete_label': bool(keep and not complete),
                'terminal_evidence': terminal, 'disk_files_deleted': 0}

    def absorb(self, bundle):
        if bundle.get('schema_version') != 1 or not self.active:
            raise ValueError('Cannot absorb an unknown or out-of-attempt evidence bundle')
        for info in bundle['stores']:
            directory = self.checked_path(info['directory'])
            if str(directory) in self.stores:
                raise ValueError('Replay evidence store identity reused')
            if type(info['max_bytes']) is not int or info['max_bytes'] < 0:
                raise ValueError('Replay store limit is not a nonnegative integer')
            self.stores[str(directory)] = SearchBufferedStore(self, directory, info['max_bytes'])
        for row in bundle['payloads']:
            path = self.checked_path(row['file'])
            if row['directory'] != str(path.parent) or not isinstance(row['content'], bytes):
                raise ValueError('Replay evidence bundle path/type mismatch')
            self.stores[str(path.parent)].save(path.name, row['content'])

    def _reference_closure(self, seeds, *, include_sizes=False):
        """Authenticate runtime reference closure using original bytes/hashes."""
        pending, seen, sizes = [], {}, {}

        def collect(value):
            if isinstance(value, dict):
                if isinstance(value.get('file'), str) and isinstance(value.get('sha256'), str):
                    pending.append(value)
                for nested in value.values():
                    collect(nested)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    collect(nested)
        collect(seeds)
        while pending:
            ref = pending.pop()
            path = self.checked_path(ref['file']); key = str(path)
            if key in seen:
                if seen[key] != ref['sha256']:
                    raise ValueError('Conflicting hashes for one runtime reference')
                continue
            data = self.data(key)
            if data is None:
                data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != ref['sha256']:
                raise ValueError('Runtime evidence reference hash changed')
            seen[key] = ref['sha256']
            sizes[key] = len(data)
            if path.suffix.lower() == '.json':
                collect(json.loads(data))
        return (seen, sizes) if include_sizes else seen

    def release_runtime_references(self, seeds):
        """Release ONLY RAM outside an explicit dependency closure, never disk.

        Use [] after a Controller retires and its compact failure/retention
        records have been written. For a continuing Controller, supply its
        common snapshot, native placement tokens and other still-needed refs.
        """
        if self.active:
            raise RuntimeError('Finish the current attempt before releasing common references')
        keep = self._reference_closure(seeds)
        omitted = [key for key in self.resident if key not in keep]
        released = sum(len(self.resident[key]['content']) for key in omitted)
        for key in omitted:
            self.resident.pop(key)
        return {'released_in_memory_payloads': len(omitted), 'released_in_memory_bytes': released,
                'disk_files_deleted': 0, 'remaining_common_references': len(self.resident)}

    def preserve_terminal_references(self, seeds):
        """Preserve a bounded black-frame proof from an explicitly reserved budget.

        This never grants pair acceptance or claims full attempt retention. A
        caller may emit black exit86 only if this returns preserved=True and
        its existing independent PNG/UUID/cleanup guard also succeeds.
        """
        closure, sizes = self._reference_closure(seeds, include_sizes=True)
        candidates = {**self.resident, **self.pending}
        needed = {key: candidates[key] for key in closure if key in candidates}
        plan = self._plan(needed, terminal=True)
        # Bound the ENTIRE unique-path proof, including files materialized by a
        # separate replay worker. Otherwise a parent with only an 8MiB reserve
        # could silently accept a much larger external child packet. Zero means
        # no terminal packet allowance; it never defaults to the whole raw cap.
        packet_bytes = sum(sizes.values())
        packet_fits = bool(closure) and packet_bytes <= self.terminal_reserve_bytes
        plan.update(terminal_packet_bytes=packet_bytes,
                    terminal_packet_limit_bytes=self.terminal_reserve_bytes,
                    terminal_packet_limit_exceeded=not packet_fits,
                    can_materialize_complete=bool(plan['can_materialize_complete'] and packet_fits))
        if plan['can_materialize_complete']:
            self._materialize(needed)
        return {'preserved': plan['can_materialize_complete'], 'capacity_plan': plan,
                'packet_bytes': packet_bytes,
                'newly_materialized_payload_bytes': plan['required_payload_bytes'] if plan['can_materialize_complete'] else 0,
                'references': [{'file': key, 'sha256': value} for key, value in sorted(closure.items())],
                'evidence_complete_for_acceptance': False, 'scope': 'terminal guard proof only'}

    def mark_terminal(self, reference):
        """Call from a private black-frame retention hook BEFORE finish clears RAM."""
        if not isinstance(reference, dict) or not isinstance(reference.get('sha256'), str):
            raise ValueError('Terminal evidence needs a measured file/hash reference')
        self.checked_path(reference['file'])
        self.terminal_reference = dict(reference)

    def minimal_black_flush(self, reference):
        return self.preserve_terminal_references([reference])


class SearchBufferedStore(BufferedStore):
    """Cap-aware writes retain usable bytes instead of raising a capacity error."""
    def save(self, name, content):
        path = self.directory / name
        if path.parent != self.directory:
            raise ValueError('Raw evidence path escaped its store')
        if not isinstance(content, bytes):
            raise TypeError('Raw evidence payload must contain bytes')
        key = str(path)
        if key in self.reserved:
            raise FileExistsError('Evidence payload name reused: ' + key)
        self.reserved.add(key)
        row = {'directory': str(self.directory), 'content': content}
        if self.session.active:
            self.session.pending[key] = row
            self.pending_bytes += len(content)
        else:
            plan = self.session._plan({key: row})
            if plan['can_materialize_complete']:
                return self.persist(path, content)
            self.session._hit_cap(plan)
            self.session.resident[key] = row
        return reference(path, content)

    def index(self):
        waiting = {**self.session.resident, **self.session.pending}
        refs = [reference(path, row['content']) for path, row in waiting.items()
                if row['directory'] == str(self.directory)]
        return self.save_json('evidence_index.json', {'files': self.files + refs,
            'raw_bytes': self.bytes, 'unmaterialized_payload_bytes': sum(ref['bytes'] for ref in refs),
            'max_raw_bytes': self.max_bytes, 'raw_files_not_deleted': True,
            'raw_saving_stopped': self.session.raw_saving_stopped,
            'index_scope': 'references may remain memory-only; use retention certificate before admission'})


def rejection_reasons(row, error=None):
    """Use complete native/policy reasons, including object identity, not w0_physical."""
    result = set()
    for key in ('physical_w0', 'physical_w1'):
        result.update((row.get(key) or {}).get('reasons') or [])
    if not result:
        result.update(row.get('discard_reasons') or [])
    if not result and error is not None:
        result.add(f'{type(error).__name__}: {error}')
    return sorted(result)


def physical_numbers(row):
    result = {}
    for key in ('physical_w0', 'physical_w1'):
        source = row.get(key)
        if not isinstance(source, dict):
            continue
        changes = source.get('all_object_changes') or {}
        edited = set(source.get('edited_names') or [])
        other = {name: value.get('position_change_m') for name, value in changes.items() if name not in edited}
        result[key] = {'valid_physical': source.get('valid_physical'), 'reasons': source.get('reasons'),
            'native_is_scene_at_rest': source.get('native_is_scene_at_rest'), 'edits': source.get('edits'),
            'max_other_position_change_m': max((value for value in other.values() if value is not None), default=None),
            'other_position_changes_above_1cm': {name: value for name, value in other.items() if value is not None and value > .01}}
        common = source.get('common_baseline_check')
        if isinstance(common, dict):
            rows = (common.get('measurements') or {}).get('objects') or {}
            selected = set(common.get('edited_names') or [])
            deltas = {name: value.get('baseline_displacement_m') for name, value in rows.items() if name not in selected}
            result[key]['common_baseline_key_numbers'] = {
                'passed': common.get('passed'), 'reasons': common.get('reasons'),
                'baseline_reference': common.get('baseline_reference'),
                'max_other_displacement_m': max((value for value in deltas.values() if value is not None), default=None),
                'violations': {name: value for name, value in deltas.items() if value is not None and value > .01}}
    return result


class RetentionPolicy:
    def __init__(self):
        self.reason_counts = {}
        self.physical_check_unavailable_attempts = 0
        self.physical_distribution = {'edited_position_error': 0, 'other_object_displacement': 0,
                                      'parent_receptacle_unverified': 0}

    def decide(self, row, error=None, *, black=False):
        reasons = rejection_reasons(row, error)
        passed = row.get('status') == 'passed' and error is None
        if passed:
            return True, {'reason': 'qualified_pair', 'rejection_reasons': [], 'reason_occurrences': {}}
        occurrences = {}
        for reason in reasons:
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
            occurrences[reason] = self.reason_counts[reason]
        physical = [row.get(key) or {} for key in ('physical_w0', 'physical_w1')]
        if not any(physical):
            self.physical_check_unavailable_attempts += 1
        edits = [edit for record in physical for edit in (record.get('edits') or {}).values()]
        flags = {
            'edited_position_error': any((edit.get('position_error_m') or 0) > .02 for edit in edits),
            'other_object_displacement': any(reason.startswith(('other_object_position_change_above_0_01m:',
                'common_baseline_other_object_position_change_above_0_01m:')) for reason in reasons),
            'parent_receptacle_unverified': any(edit.get('parent_receptacle_verified') is False for edit in edits),
        }
        for key, value in flags.items():
            self.physical_distribution[key] += int(value)
        # A new reason qualifies for a full example even when this attempt also
        # contains another reason already observed more than three times.
        keep = bool(black or any(count <= 3 for count in occurrences.values()))
        return keep, {'reason': 'black_frame_stop' if black else 'first_three_of_actual_reason' if keep else 'compact_rejection',
                      'rejection_reasons': reasons, 'reason_occurrences': occurrences,
                      'physical_rejection_categories': flags}


def compact_attempt(row):
    fields = ('pair_id', 'scene', 'kind', 'status', 'discard_reasons', 'generation_seconds',
              'pair_file', 'pair_sha256', 'error', 'evidence_retention', 'thumbnail', 'raw_complete')
    result = {key: row[key] for key in fields if key in row}
    result['physical_key_numbers'] = physical_numbers(row)
    result['evidence_policy'] = 'compact rejection; omitted raw payloads existed only in memory'
    return result
