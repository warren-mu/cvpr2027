"""Future-only retention of rejected T02 binary payloads; never a scientific check.

The caller first completes native validation and stops its owned Worker/Unity.
All JSON, indices, source packages, results and logs remain byte-identical.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat

POLICY = {'version': 't02_scientific_reject_binary_seed0_tenth_v1', 'seed': 0,
          'selection': 'SHA256 canonical attempt identity below floor(2**256/10)',
          'eligible_status': 'rejected', 'accepted_full_raw': True,
          'error_timeout_black_or_unknown_cleanup_full_raw': True,
          'metadata_indices_logs_results_retained': True}
RAW_SUFFIXES = frozenset(('.png', '.npz'))
TABLE_SUFFIXES = frozenset(('.rgb', '.mask'))


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def save(path, value):
    body = encoded(value)
    with Path(path).open('xb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    return {'file': str(Path(path).absolute()), 'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)}


def read(reference):
    path = Path(reference['file'])
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise ValueError('Retention reference is not an exact real path')
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != reference['sha256'] or len(body) != reference['bytes']:
        raise ValueError('Retention metadata bytes differ')
    return json.loads(body)


def selection(identity):
    digest = hashlib.sha256(encoded({'seed': 0, 'attempt': identity})).hexdigest()
    return {'seed': 0, 'sha256': digest, 'selected': int(digest, 16) < (1 << 256) // 10}


def start_attempt(config, package, pair_dir, index, parent_result):
    """Record ownership immediately after the parent exclusively creates pair_dir."""
    path, root = Path(pair_dir), Path(config['scratch_directory'])
    expected = root / ('pair_%04d' % index)
    if path != expected or path.resolve() != path or not path.is_dir() or Path(parent_result).exists():
        raise ValueError('Retention applies only before publication in this new private pair directory')
    info = path.lstat()
    if info.st_uid != os.getuid():
        raise ValueError('Private pair directory belongs to another account')
    pair_ref = package['source_provenance']['pair']
    identity = {'run_name': config['name'], 'pbs_jobid': config['pbs_jobid'],
                'pair_index': index, 'pair_id': package['pair_id'], 'scene': package['scene'],
                'source_pair': {key: pair_ref[key] for key in ('file', 'sha256')}}
    value = {'policy': copy.deepcopy(POLICY), 'identity': identity, 'pair_directory': str(path),
             'directory_identity': [info.st_dev, info.st_ino], 'owner_uid': info.st_uid,
             'parent_result': str(Path(parent_result).absolute()), 'selection': selection(identity)}
    return save(path / 'raw_retention_owner.json', value)


def _payload(reference, parent, suffixes):
    path = Path(reference['file'])
    if (path.parent != parent or path.suffix not in suffixes or path.resolve() != path
            or path.is_symlink()):
        raise ValueError('Payload is not a direct private raw/table binary')
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        raise ValueError('Payload is foreign, shared by hardlink, or not regular')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != reference['sha256'] or info.st_size != reference['bytes']:
        raise ValueError('Payload differs from its original index')
    return {'original_path': str(path), 'sha256': digest.hexdigest(), 'bytes': info.st_size,
            'identity': [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns],
            'availability': 'planned_omission_after_scientific_rejection'}


def apply_retention(row, owner_reference, *, cleanup_confirmed):
    """No omission for accepted, unknown/error, black or unconfirmed cleanup."""
    result = copy.deepcopy(row)
    owner = read(owner_reference)
    if owner['policy'] != POLICY or owner['selection'] != selection(owner['identity']):
        raise ValueError('Retention policy/selection changed')
    root = Path(owner['pair_directory'])
    info = root.lstat()
    if (root.resolve() != root or [info.st_dev, info.st_ino] != owner['directory_identity']
            or info.st_uid != os.getuid() or Path(owner['parent_result']).exists()
            or row.get('pair_id') != owner['identity']['pair_id']):
        raise ValueError('Retention owner/current unpublished attempt identity changed')
    eligible = (cleanup_confirmed is True and row.get('cleanup_errors') == []
                and row.get('status') == 'rejected' and bool(row.get('rejection_reasons'))
                and not row.get('black_frame_reference') and not row.get('raw_index_error'))
    decision = {'policy': copy.deepcopy(POLICY), 'owner_reference': owner_reference,
                'draw': owner['selection'], 'eligible_scientific_rejection': eligible,
                'binary_payloads_retained': True, 'omitted_payload_count': 0,
                'omitted_payload_bytes': 0, 'cleanup_confirmed': cleanup_confirmed is True,
                'scientific_status_unchanged': row.get('status')}
    result['raw_retention'] = decision
    if not eligible or owner['selection']['selected']:
        return result
    if row.get('downstream_allowed') is True or row.get('task_acceptance') is True:
        raise ValueError('Never omit accepted/shared evidence')
    index_ref = row['raw_index']
    if Path(index_ref['file']) != root / 'raw/evidence_index.json':
        raise ValueError('Raw index does not belong to this private attempt')
    index = read(index_ref)
    candidates = []
    for reference in index['files']:
        if Path(reference['file']).suffix in RAW_SUFFIXES:
            candidates.append(_payload(reference, root / 'raw', RAW_SUFFIXES))
    table_suffixes = TABLE_SUFFIXES
    manifest_ref = row.get('table_manifest')
    if manifest_ref is not None:
        if Path(manifest_ref['file']) != root / 'table/table.json':
            raise ValueError('Table is not private to the rejected attempt')
        # save_table's extra payload_bytes is metadata, not part of the file reference.
        manifest = read(manifest_ref)
        if manifest.get('storage_encoding') is not None:
            from src.envs.pair_table_codec import POLICY as compression_policy
            if manifest['storage_encoding'] != compression_policy or any(
                    ref.get('encoding') != compression_policy for ref in manifest['files']):
                raise ValueError('Unknown rejected table compression declaration')
            table_suffixes = frozenset(('.npz',))
        for reference in manifest['files']:
            name = reference['file']
            if Path(name).name != name:
                raise ValueError('Table payload name escapes its private directory')
            candidates.append(_payload({**reference, 'file': str(root / 'table' / name)},
                                       root / 'table', table_suffixes))
    paths = [item['original_path'] for item in candidates]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate payload in original raw/table index')
    # Unexpected binaries are retained and block ALL omission before the first unlink.
    for folder, suffixes in ((root / 'raw', RAW_SUFFIXES), (root / 'table', table_suffixes | TABLE_SUFFIXES)):
        if folder.exists():
            if folder.is_symlink() or folder.resolve() != folder:
                raise ValueError('Private payload directory became a symlink')
            discovered = {str(p) for p in folder.iterdir() if p.suffix in suffixes}
            expected = {p for p in paths if Path(p).parent == folder}
            if discovered != expected:
                raise ValueError('Private binaries are not fully indexed')
    intent = {'policy': POLICY, 'owner_reference': owner_reference,
              'scientific_rejection_reasons': copy.deepcopy(row['rejection_reasons']),
              'raw_index_metadata_reference': index_ref, 'table_metadata_reference': manifest_ref,
              'payloads': candidates, 'scope': 'Current unpublished rejected attempt only; metadata/logs never removed'}
    intent_ref = save(root / 'raw_retention_intent.json', intent)
    omitted, error = [], None
    try:
        for item in candidates:
            path = Path(item['original_path']); info = path.lstat()
            actual = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
            if (actual != item['identity'] or path.resolve() != path or path.is_symlink()
                    or info.st_nlink != 1 or info.st_uid != os.getuid()):
                raise ValueError('Payload changed after complete pre-delete validation')
            path.unlink()
            omitted.append({**item, 'availability': 'omitted_by_declared_rejected_raw_policy'})
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    receipt = {'intent_reference': intent_ref, 'omitted_payloads': omitted,
               'completed': error is None, 'error': error, 'metadata_indices_logs_unchanged': True,
               'metadata_scope': 'Original indices describe pre-retention writes; this receipt records current payload availability. No full native/cache proof is claimed.'}
    decision.update(binary_payloads_retained=not omitted, omitted_payload_count=len(omitted),
                    omitted_payload_bytes=sum(item['bytes'] for item in omitted),
                    intent_reference=intent_ref, omitted_payloads=omitted, retention_error=error)
    try:
        decision['receipt_reference'] = save(root / 'raw_retention_receipt.json', receipt)
    except BaseException as exc:
        decision['deletion_error'] = error
        decision['retention_error'] = {'type': type(exc).__name__, 'message': str(exc),
                                       'stage': 'write_retention_receipt'}
    if omitted:
        # Original metadata remains immutable and auditable, but is not advertised
        # as an available full table/raw payload after explicit sampling.
        for key in ('raw_index', 'raw_incremental_index', 'table_manifest', 'detail', 'report_reference'):
            if key in result:
                result['retained_' + key + '_metadata'] = result.pop(key)
        result['raw_payloads_available'] = False
        result['table_payloads_available'] = False
        result['downstream_allowed'] = False
        try:
            result['report_reference'] = save(root / 'pair_retention_report.json', result)
        except BaseException as exc:
            decision['report_write_error'] = {'type': type(exc).__name__, 'message': str(exc),
                                              'stage': 'write_retention_report'}
            if decision['retention_error'] is None:
                decision['retention_error'] = decision['report_write_error']
    return result


def summary(rows):
    status_counts, reasons, errors = {}, {}, {}
    sampled = omitted = 0
    for row in rows:
        status = str(row.get('status', 'unknown')); status_counts[status] = status_counts.get(status, 0) + 1
        for reason in set(row.get('rejection_reasons') or []):
            reasons[reason] = reasons.get(reason, 0) + 1
        error = row.get('error')
        if isinstance(error, dict):
            key = str(error.get('type', 'unknown')); errors[key] = errors.get(key, 0) + 1
        decision = row.get('raw_retention', {})
        if decision.get('eligible_scientific_rejection'):
            sampled += int(decision['draw']['selected'])
            omitted += int(decision.get('omitted_payload_count', 0) > 0)
    return {'policy': copy.deepcopy(POLICY), 'attempts': len(rows), 'status_counts': status_counts,
            'rejection_reason_counts': reasons, 'error_type_counts': errors,
            'sampled_rejected_attempts': sampled, 'omitted_rejected_attempts': omitted,
            'scope': 'Storage decisions only; original scientific results and all attempt denominators unchanged'}
