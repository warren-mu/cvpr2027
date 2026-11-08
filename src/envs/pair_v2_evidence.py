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
