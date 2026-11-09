"""Bounded original Controller identity metadata for future collection runs.

This reserve never materializes missing scientific payloads or changes the
ordinary raw-retention acceptance decision. Old evidence is not reconstructed.
"""
import copy
from pathlib import Path
import re

from .pair_v2_evidence import (SearchEvidenceSession, SearchBufferedStore,
                               encoded, reference)

VERSION = 'bounded_original_construction_identity_metadata_v1'
METADATA_BUDGET_BYTES = 64 * 1024 * 1024
METADATA_FILE_LIMIT_BYTES = 16 * 1024 * 1024
NAMES = frozenset(('scene_preflight.json', 'evidence_index.json'))
POLICY = {
    'version': VERSION,
    'whitelist': '<declared raw root>/marked_controller_NNN/construction/{scene_preflight.json,evidence_index.json}',
    'encoding': 'original save_json bytes and SHA; exclusive creation; no reconstruction or reference substitution',
    'accounting': 'one separate bounded metadata reserve per scene, with per-file limit; excluded from raw payload bytes',
    'raw_cap': 'unchanged raw budget, black reserve and evidence_complete_for_acceptance; no resident/pending payload flush',
    'index': 'original index generation and memory-only scope retained; every referenced scientific file still requires original independent authentication',
    'on_metadata_limit': 'explicit scene failure; never silently leave a claimed metadata reference only in memory',
    'phase': 'construction common/closing phase only; replay stores and active attempt evidence unchanged',
}
FIELDS = ('construction_metadata_retention_version', 'construction_metadata_retention_policy',
          'construction_metadata_budget')


def fields():
    return {FIELDS[0]: VERSION, FIELDS[1]: copy.deepcopy(POLICY), FIELDS[2]: {
        'scene_bytes': METADATA_BUDGET_BYTES, 'single_file_bytes': METADATA_FILE_LIMIT_BYTES}}


class MetadataRetentionFailure(SystemExit):
    """Infrastructure stop, not a scientific rejection or successful cleanup."""


class ConstructionMetadataStore(SearchBufferedStore):
    def save(self, name, content):
        if name not in NAMES:
            return super().save(name, content)
        if self.session.active:
            raise MetadataRetentionFailure('Construction identity metadata written inside an active attempt')
        if not isinstance(content, bytes):
            raise TypeError('Original construction metadata must contain bytes')
        path = self.directory / name
        if path.parent != self.directory:
            raise ValueError('Construction metadata escaped its original store')
        key = str(path)
        if key in self.reserved:
            raise FileExistsError('Construction metadata identity reused: ' + key)
        self.reserved.add(key)
        self.session.claim_metadata(path, content)
        try:
            # Deliberately do not call BufferedStore.persist: its raw byte
            # counter belongs to the unchanged scientific payload budget.
            self.ensure_directory()
            with path.open('xb') as stream:
                stream.write(content)
            ref = reference(path, content)
            self.files.append(ref)
            with self.incremental.open('ab') as stream:
                stream.write(encoded(ref))
            self.session.metadata_files.append(ref)
            self.session.metadata_written_bytes += len(content)
            return ref
        except BaseException as error:
            self.session.metadata_errors.append({
                'file': key, 'reason': f'{type(error).__name__}: {error}',
                'scope': 'partial bytes, if any, remain; no rewrite or acceptance claim'})
            raise MetadataRetentionFailure('Construction metadata persistence failed: ' + key) from error


class ConstructionMetadataSession(SearchEvidenceSession):
    def __init__(self, scratch, *, construction_root, metadata_budget_bytes=METADATA_BUDGET_BYTES,
                 metadata_file_limit_bytes=METADATA_FILE_LIMIT_BYTES, **kwargs):
        super().__init__(scratch, **kwargs)
        if self.active or self.imports:
            raise ValueError('Construction reserve is not a buffered replay session')
        self.construction_root = self.checked_path(construction_root)
        if (type(metadata_budget_bytes) is not int or type(metadata_file_limit_bytes) is not int
                or not 0 < metadata_file_limit_bytes <= metadata_budget_bytes):
            raise ValueError('Require positive bounded metadata file and scene limits')
        self.metadata_budget_bytes = metadata_budget_bytes
        self.metadata_file_limit_bytes = metadata_file_limit_bytes
        self.metadata_claimed_bytes = 0
        self.metadata_written_bytes = 0
        self.metadata_files = []
        self.metadata_errors = []

    def construction_directory(self, directory):
        if not directory.is_relative_to(self.construction_root):
            return False
        parts = directory.relative_to(self.construction_root).parts
        return (len(parts) == 2 and re.fullmatch(r'marked_controller_[0-9]{3}', parts[0]) is not None
                and parts[1] == 'construction')

    def store(self, directory, max_bytes):
        directory = self.checked_path(directory)
        if str(directory) in self.stores or directory.exists():
            raise FileExistsError('Private evidence directory already exists: ' + str(directory))
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('Require a nonnegative per-store raw-payload limit')
        cls = ConstructionMetadataStore if self.construction_directory(directory) else SearchBufferedStore
        result = cls(self, directory, max_bytes)
        self.stores[str(directory)] = result
        if not self.active:
            result.ensure_directory()
        return result

    def claim_metadata(self, path, content):
        size = len(content)
        if (size > self.metadata_file_limit_bytes
                or self.metadata_claimed_bytes + size > self.metadata_budget_bytes):
            self.metadata_errors.append({'file': str(path), 'requested_bytes': size,
                'scene_claimed_bytes': self.metadata_claimed_bytes,
                'reason': 'bounded construction metadata reserve exhausted'})
            raise MetadataRetentionFailure('Bounded construction metadata reserve exhausted: ' + str(path))
        # Charge before IO, retaining the claim on failure. A partial write is
        # never silently reclaimed or retried under a different budget.
        self.metadata_claimed_bytes += size

    def status(self):
        return {**super().status(), 'construction_metadata_retention': {
            'version': VERSION, 'policy': copy.deepcopy(POLICY),
            'scene_budget_bytes': self.metadata_budget_bytes,
            'single_file_limit_bytes': self.metadata_file_limit_bytes,
            'claimed_bytes': self.metadata_claimed_bytes,
            'written_bytes': self.metadata_written_bytes,
            'files': copy.deepcopy(self.metadata_files), 'errors': copy.deepcopy(self.metadata_errors),
            'scientific_payload_materialization_or_acceptance_granted': False}}
