"""Read-only snapshots of explicit T01v2 scene runs; never final pair acceptance.

Invoke periodically through run_logged with a NEW report name. This command
does not watch, submit, cancel, or modify a running source. Complete sources
use the strict final auditor; live counts cover only archived complete lines.
"""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex

from src.tools.merge_pair_v2 import (KINDS, load_source, logged_configuration,
    option, producer_arguments, verify_source_identity)
from src.tools.t01_partition import ROOT, audited_result
from src.tools.t01v2_parallel import submission_identity

MAX_PREFIX_BYTES = 8 * 1024 * 1024
MODULES = tuple('src.envs.' + name for name in ('pair_v2_shard', 'pair_v2_clone_shard',
    'pair_v2_stable_shard', 'pair_v2_search_shard'))
RUN_NAME = re.compile(r'(?P<collection>pilot_pairs_v2_[A-Za-z0-9_]+)_j(?P<number>[0-9]+)_hopper_m_(?P<server>[0-9]+)_g(?P<gpu>[0-9]+)_floorplan(?P<scene>[1-9][0-9]*)')


def expected_identity(name):
    match = RUN_NAME.fullmatch(name)
    if match is None:
        raise ValueError('Require an explicit per-scene T01v2 run name')
    row = match.groupdict()
    return {'name': name, 'collection': row['collection'], 'scene': 'FloorPlan' + row['scene'],
            'pbs_jobid': row['number'] + '.hopper-m-' + row['server'], 'logical_index': int(row['gpu'])}


def check_expected_shard(expected, shard):
    if any(shard.get(key) != expected[key] for key in ('name', 'scene', 'pbs_jobid', 'logical_index')):
        raise ValueError('Shard name/scene/job/GPU differs from the explicit requested source name')


def complete_prefix(raw):
    """A partial final JSON/log line is unobserved, never repaired or counted."""
    return raw[:raw.rfind(b'\n') + 1]


def read_prefix(path, limit=MAX_PREFIX_BYTES):
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(min(before.st_size, limit))
        after = os.fstat(stream.fileno())
    current = path.stat()
    if ((before.st_dev, before.st_ino) != (current.st_dev, current.st_ino)
            or after.st_size < before.st_size or current.st_size < before.st_size):
        raise ValueError('Source replaced or truncated during snapshot')
    prefix = complete_prefix(raw)
    return prefix, {'source_file': str(path), 'size_at_open_bytes': before.st_size,
        'snapshot_bytes': len(prefix), 'read_cap_reached': before.st_size > limit,
        'trailing_incomplete_bytes_not_counted': len(raw) - len(prefix),
        'append_only_contract': 'producer appends; snapshot is not a final-file hash'}


def archive(data, directory, filename):
    path = directory / filename
    with path.open('xb') as stream:
        stream.write(data)
    return {'file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(data).hexdigest()}


def observed_counts(raw, scene, *, log=False):
    rows, seen, reasons, passed, w0 = [], set(), Counter(), {kind: 0 for kind in KINDS}, {}
    for line in raw.decode('utf-8').splitlines():
        if log:
            if not line.startswith('T01V2_ATTEMPT: '):
                continue
            line = line[len('T01V2_ATTEMPT: '):]
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get('scene') != scene or not isinstance(row.get('pair_id'), str)
                or re.fullmatch(scene.lower() + r'_[0-9]{5}', row['pair_id']) is None
                or row['pair_id'] in seen or row.get('status') not in ('passed', 'rejected')
                or not isinstance(row.get('discard_reasons'), list)
                or any(not isinstance(reason, str) for reason in row['discard_reasons'])):
            raise ValueError('Wrong-scene, duplicate, or malformed complete attempt row')
        seen.add(row['pair_id'])
        rows.append(row)
        reasons.update(row['discard_reasons'])
        if row['status'] == 'passed':
            if row.get('kind') not in KINDS:
                raise ValueError('Reported pass has no recognized class')
            passed[row['kind']] += 1
        anchor = row.get('anchor_type')
        if anchor is not None and (not isinstance(anchor, str) or not anchor):
            raise ValueError('Malformed reported anchor category')
        item = w0.setdefault(anchor or '__anchor_unavailable__',
            {'measured_attempts': 0, 'passed': 0, 'rejected': 0, 'measurement_unavailable': 0})
        value = row.get('w0_valid_physical')
        if type(value) is bool:
            item['measured_attempts'] += 1
            item['passed' if value else 'rejected'] += 1
        else:
            item['measurement_unavailable'] += 1
    return {'observed_complete_attempts': len(rows), 'observed_reported_passed_by_class': passed,
            'discard_reason_counts': dict(reasons), 'w0_by_anchor': w0}


def bind_live(name, log_text, directory):
    expected = expected_identity(name)
    lines = log_text.splitlines()
    if not lines or not lines[0].startswith('# run_logged: '):
        raise ValueError('Live source lacks its original run_logged command header')
    command = shlex.split(lines[0][len('# run_logged: '):])
    matches = []
    for module in MODULES:
        try:
            matches.append((module, producer_arguments(command, module)))
        except ValueError:
            pass
    if len(matches) != 1:
        raise ValueError('Unknown live shard producer')
    module, args = matches[0]
    if option(args, '--name') != name or option(args, '--scene') != expected['scene']:
        raise ValueError('Log command differs from the explicitly requested scene name')
    shard_path = Path(option(args, '--shard-config'))
    tag = expected['pbs_jobid'].replace('.', '_').replace('-', '_')
    suffix = f"/t01v2_parallel/{expected['collection']}/{tag}/{expected['scene']}/shard_config.json"
    if (not shard_path.is_absolute() or not str(shard_path).startswith('/scratch/')
            or not str(shard_path).endswith(suffix) or shard_path.stat().st_mode & 0o222):
        raise ValueError('Expected the immutable scratch shard configuration for this scene/job')
    raw = shard_path.read_bytes()
    shard = json.loads(raw)
    check_expected_shard(expected, shard)
    if Path(shard['raw_evidence_dir']).resolve() != shard_path.resolve().parent:
        raise ValueError('Shard configuration name/scene/job/GPU differs from expected identity')
    gate, gate_ref = audited_result(shard['gate_result'])
    if any(shard['gate_source'].get(key) != gate_ref[key]
           for key in ('file', 'sha256', 'metrics_sha256', 'exit_code', 'git_commit')):
        raise ValueError('Shard gate source differs from its audited original')
    gate_log = ROOT / gate['_meta']['log_path']
    runtime_lines = [line for line in lines if line.startswith('RUNTIME_CONFIG:')]
    runtime = logged_configuration(log_text, 'RUNTIME_CONFIG') if runtime_lines else None
    data = {'_meta': {'name': name}, 'metrics': {'status': 'running'},
        'config': {**{key: shard[key] for key in ('name', 'scene', 'seed', 'resolution', 'raw_evidence_dir')},
                   'shard': shard, 'runtime': runtime}}
    verify_source_identity(data, gate, logged_runtime=runtime,
        logged_allocation=logged_configuration(gate_log.read_text(), 'ALLOCATION_CONFIG'))
    submission = submission_identity(ROOT, expected['collection'], expected['pbs_jobid'],
        len(shard['allocation']['devices']), float(option(args, '--timeout-seconds')))
    payload = ROOT / submission['file']
    receipt = payload.with_name('submission.json')
    evidence = {'shard_config': archive(raw, directory, name + '_shard_config.json'),
        'submission_payload': archive(payload.read_bytes(), directory, name + '_payload.json'),
        'submission_receipt': archive(receipt.read_bytes(), directory, name + '_submission.json')}
    return {**expected, 'producer_module': module, 'gpu_uuid': shard['gpu_identity']['gpu_uuid'],
        'gate_source': gate_ref, 'submission_identity': submission, 'identity_evidence': evidence,
        'actual_runtime_observed': runtime is not None,
        'binding_scope': 'immutable shard assignment plus audited gate and submit receipt; live child is not finally audited'}, gate_ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source', required=True, nargs='+')
    args = parser.parse_args()
    if re.fullmatch(r'[A-Za-z0-9_]+', args.name) is None:
        parser.error('Use a unique alphanumeric report name')
    output = ROOT / 'results' / (args.name + '.json')
    if output.exists():
        raise FileExistsError('Preserve prior report; use a new name')
    directory = ROOT / 'logs/runs' / args.name / 'evidence'
    directory.mkdir(parents=True, exist_ok=False)
    rows, errors, sources, seen = [], [], {}, set()
    for value in args.source:
        try:
            path = (ROOT / value).resolve()
            if path.parent != ROOT / 'results' or path.suffix != '.json' or path in seen:
                raise ValueError('Require unique explicit current per-scene results paths')
            seen.add(path)
            name = path.stem
            expected = expected_identity(name)
            if path.is_file() and (ROOT / 'logs/runs' / (name + '.meta.json')).exists():
                data, reference = load_source(path)
                check_expected_shard(expected, data['config']['shard'])
                sources[reference['file']] = reference
                measured = data['metrics']
                rows.append({**expected, 'source_file': value, 'partial': False, 'finally_audited': True,
                    'status': measured.get('status'), 'exit_code': data['_meta']['exit_code'],
                    'ended_at_utc': data['_meta']['end'],
                    'source_reported_attempts': measured.get('attempts'),
                    'source_reported_passed_by_class': measured.get('passed_by_class'),
                    'discard_reason_counts': measured.get('discard_reason_counts'),
                    'w0_by_anchor': measured.get('w0_by_anchor'), 'cleanup_errors': measured.get('cleanup_errors'),
                    'source': reference})
                continue
            row = {**expected, 'source_file': value, 'partial': True, 'finally_audited': False,
                   'status': 'not_yet_observed', 'counts': None, 'cleanup_errors': None}
            rows.append(row)
            log = ROOT / 'logs/runs' / (name + '.log')
            if not log.is_file():
                continue
            log_raw, log_extent = read_prefix(log)
            row['log_prefix'] = {**log_extent, 'snapshot': archive(log_raw, directory, name + '_log.prefix')}
            binding, gate_ref = bind_live(name, log_raw.decode('utf-8'), directory)
            row['binding'] = binding
            sources[gate_ref['file']] = gate_ref
            journal = ROOT / 'data/world_pairs/v2' / (name + '_attempts.jsonl')
            journal_observed = journal.exists()
            if journal_observed:
                raw, extent = read_prefix(journal)
                row['journal_prefix'] = {**extent, 'snapshot': archive(raw, directory, name + '_journal.prefix')}
                row['count_source'] = 'complete journal byte prefix'
            else:
                raw = log_raw
                row['count_source'] = 'complete T01V2_ATTEMPT log lines'
            counts = observed_counts(raw, expected['scene'], log=not journal_observed)
            # No journal and no observed attempt lines is unknown, not a zero-run result.
            row.update(status='observed_partial', counts=(counts if journal_observed or counts['observed_complete_attempts'] else None))
        except Exception as error:
            errors.append({'source_file': value, 'error': f'{type(error).__name__}: {error}'})
    metrics = {'kind': 't01v2_live_progress', 'snapshot_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'scenes': rows, 'errors': errors, 'task_acceptance': None,
        'partial_observation_is_final_acceptance': False,
        'scope': 'Explicit sources only; observed/source-reported passes are not independently accepted pairs.'}
    config = {'name': args.name, 'explicit_sources': args.source, 'maximum_prefix_bytes': MAX_PREFIX_BYTES,
        'missing_policy': 'unknown; only complete observed attempt rows counted',
        'source_mutations': False, 'raw_physics_evidence_read': False}
    with output.open('x') as stream:
        json.dump({'metrics': metrics, 'config': config, 'sources': list(sources.values())}, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
