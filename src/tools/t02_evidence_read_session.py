"""Bounded, call-local reuse of authenticated evidence bytes during cache load.

This caches no scientific verdict, decoded mutable object, source eligibility,
or cross-run result. Every original scientific checker still executes. Before
the owning call may return, every accessed file is hashed again from disk.
"""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import threading

MAX_RETAINED_BYTES = 128 * 1024 * 1024
_INSTALL_LOCK = threading.Lock()


def fingerprint(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


class EvidenceBytes:
    def __init__(self, max_bytes=MAX_RETAINED_BYTES):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('Evidence byte cache requires a positive finite byte bound')
        self.limit, self.retained = max_bytes, 0
        self.cache, self.accessed = OrderedDict(), set()
        self.closed = False

    def read(self, path, expected, original_reader):
        if self.closed:
            raise RuntimeError('Evidence read session is already closed')
        path = Path(path).resolve()
        key = (path, expected)
        before = fingerprint(path)
        prior = self.cache.get(key)
        if prior is not None and prior[0] == before:
            self.cache.move_to_end(key)
            return prior[1]
        if prior is not None:
            self.retained -= len(self.cache.pop(key)[1])
        data = original_reader()
        if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Authenticated evidence bytes changed while being read')
        after = fingerprint(path)
        if before != after:
            raise ValueError('Evidence file changed during its authenticated read')
        self.accessed.add(key)
        if len(data) <= self.limit:
            while self.cache and self.retained + len(data) > self.limit:
                _, (_, old) = self.cache.popitem(last=False)
                self.retained -= len(old)
            self.cache[key] = (after, data)
            self.retained += len(data)
        return data

    def verify_unchanged(self):
        if self.closed:
            raise RuntimeError('Cannot seal a closed evidence session')
        for path, expected in sorted(self.accessed):
            before = fingerprint(path)
            if not path.is_file() or digest(path) != expected or fingerprint(path) != before:
                raise ValueError('Evidence changed before cache-load completion: ' + str(path))

    def close(self):
        self.cache.clear()
        self.accessed.clear()
        self.retained, self.closed = 0, True


@contextmanager
def authenticated_read_reuse(consumer, *, max_bytes=MAX_RETAINED_BYTES):
    """Scope only the current cache-loading thread; restore readers on all exits.

    ``consumer`` is the actual running cache module, including ``__main__``
    when invoked with Python ``-m``. Other threads use the original readers.
    """
    from src.tools import merge_pair_v2 as merge
    from src.envs import pair_table_checks as checks
    if not _INSTALL_LOCK.acquire(blocking=False):
        raise RuntimeError('Overlapping evidence read scopes are not supported')
    session = None
    previous = []
    try:
        session = EvidenceBytes(max_bytes)
        owner = threading.get_ident()
        original_json = merge.read_reference
        original_path = merge.reference_path
        original_blob = checks.verified_read

        def read_json(reference, *, root=merge.ROOT, within=None):
            if threading.get_ident() != owner:
                return original_json(reference, root=root, within=within)
            # Same per-call identity and directory conditions as reference_path;
            # the original complete path/SHA check remains mandatory on a miss.
            if (not isinstance(reference, dict) or not isinstance(reference.get('file'), str)
                    or not re.fullmatch(r'[0-9a-f]{64}', str(reference.get('sha256', '')))):
                raise ValueError('Artifact requires an explicit file and SHA-256')
            path = (Path(root) / reference['file']).resolve()
            if within is not None and not path.is_relative_to(Path(within).resolve()):
                raise ValueError('Artifact escaped its declared evidence directory')
            if not path.is_file():
                raise ValueError('Missing or SHA-mismatched evidence: ' + reference['file'])
            data = session.read(path, reference['sha256'],
                lambda: original_path(reference, root=root, within=within).read_bytes())
            # No mutable dictionaries are shared between original validators.
            return json.loads(data.decode('utf-8'))

        def read_blob(reference, roots):
            if threading.get_ident() != owner:
                return original_blob(reference, roots)
            roots = tuple(roots)
            path = Path(reference['file']).resolve()
            if not any(path.is_relative_to(Path(root).resolve()) for root in roots):
                raise ValueError('Raw table proof escaped explicit evidence roots')
            if path.stat().st_size > 48 * 1024 * 1024:
                raise ValueError('Oversized raw table proof')
            return session.read(path, reference['sha256'], lambda: original_blob(reference, roots))

        targets = [(merge, 'read_reference', read_json), (checks, 'verified_read', read_blob),
                   (consumer, 'verified_read', read_blob)]
        # Explicitly reject an unrelated consumer rather than patch a fake
        # success/read interface or silently miss the actual __main__ alias.
        if consumer.verified_read is not original_blob:
            raise ValueError('Cache consumer does not use the original verified byte reader')
        previous = [(module, name, getattr(module, name)) for module, name, _ in targets]
        for module, name, value in targets:
            setattr(module, name, value)
        yield session
        session.verify_unchanged()
    finally:
        for module, name, value in reversed(previous):
            setattr(module, name, value)
        if session is not None:
            session.close()
        _INSTALL_LOCK.release()
