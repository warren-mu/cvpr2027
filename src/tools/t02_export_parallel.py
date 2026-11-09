"""Bounded single-GPU fan-out of independently logged original T02 exports.

Each unit keeps its own render gate, source authentication, native worker,
scratch directory, table checks and result. The parent is only a scheduler:
its completion never authorizes a table, cache, evaluation or training pool.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

from src.envs.thor_common import ROOT, deadline_after
from src.tools.t01_partition import sha_file
from src.tools.t01_parallel_approved import finish_logged, stop_logged, request_stop_logged
from src.tools import t02_export as export

VERSION = 't02_single_gpu_parallel_original_exports_v1'
MANIFEST_VERSION = 't02_explicit_source_units_v1'


def launch_logged(name, command, environment, directory):
    """Keep the original logged child contract in T02's own unique namespace."""
    if re.fullmatch(r't02_export_[A-Za-z0-9_]+', name) is None:
        raise ValueError('Require an explicit unique t02_export_ child name')
    paths = [ROOT / 'results' / (name + '.json'), ROOT / 'logs/runs' / name]
    paths += [ROOT / 'logs/runs' / (name + suffix) for suffix in ('.log', '.meta.json', '.diff')]
    if any(path.exists() or path.is_symlink() for path in paths):
        raise FileExistsError('Prior T02 evidence exists; choose a new child name')
    path = Path(directory) / (name + '.launcher.log')
    stream = path.open('x')
    env = dict(environment, CVPR_RUN_NAME=name,
               CVPR_RESULTS_PATH=str(ROOT / 'results' / (name + '.json')))
    argv = [sys.executable, '-u', str(ROOT / 'scripts/run_logged.py'),
            '--name', name, '--seed', '0', '--', *command]
    try:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        stream.close()
        raise
    return {'process': process, 'stream': stream, 'launcher_log': path, 'name': name,
            'start_monotonic': time.monotonic(), 'command': argv}


def read_manifest(path, expected_sha):
    path = Path(path).resolve()
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha) or sha_file(path) != expected_sha:
        raise ValueError('Require the exact prepared manifest bytes')
    data = json.loads(path.read_text())
    if data.get('version') != MANIFEST_VERSION or data.get('partition') not in ('train', 'test'):
        raise ValueError('Require an explicit separate training or evaluation manifest')
    entries = data.get('entries')
    if not isinstance(entries, list) or not 1 <= len(entries) <= 64:
        raise ValueError('Require a bounded nonempty explicit pair list')
    seen = set()
    low, high = (1, 20) if data['partition'] == 'train' else (21, 30)
    allowed = {f'FloorPlan{base + i}' for base in (0, 200, 300, 400) for i in range(low, high + 1)}
    for entry in entries:
        if (not isinstance(entry, dict) or entry.get('scene') not in allowed
                or entry.get('field_of_view') != 60
                or not re.fullmatch(r'floorplan[0-9]+_[0-9]+', str(entry.get('pair_id')))
                or not re.fullmatch(r'[0-9a-f]{64}', str(entry.get('proposal_identity')))):
            raise ValueError('Pair lacks its explicit split, camera or construction identity')
        for key, base in (('source', ROOT / 'results'), ('pair', ROOT / 'data/world_pairs/v2')):
            ref = entry.get(key)
            if (not isinstance(ref, dict) or not isinstance(ref.get('file'), str)
                    or not re.fullmatch(r'[0-9a-f]{64}', str(ref.get('sha256')))):
                raise ValueError('Explicit original result and pair file hashes are required')
            original = (ROOT / ref['file']).resolve()
            if not original.is_relative_to(base.resolve()) or sha_file(original) != ref['sha256']:
                raise ValueError('Missing, escaped or changed original source bytes')
        identity = (str((ROOT / entry['source']['file']).resolve()), entry['pair_id'],
                    str((ROOT / entry['pair']['file']).resolve()))
        if identity in seen:
            raise ValueError('Duplicate explicit export unit')
        seen.add(identity)
    return data, {'file': str(path), 'sha256': expected_sha}


def unit_command(name, entry, unit_seconds, pair_seconds, gate_seconds):
    return [sys.executable, '-u', '-m', 'src.tools.t02_export', '--name', name,
            '--source', entry['source']['file'], '--pair-id', entry['pair_id'],
            '--pair-file', entry['pair']['file'], '--timeout-seconds', str(unit_seconds),
            '--pair-timeout-seconds', str(pair_seconds), '--gate-timeout-seconds', str(gate_seconds)]


def child_outcome(record, data, identity, command, entry):
    """Authenticate scheduling identity; original native science stays in child."""
    if record.get('provenance_error') or data is None:
        raise ValueError('Missing or unauthenticated original child result')
    cfg, measured, meta = data['config'], data['metrics'], data['_meta']
    expected_name = command[command.index('--name') + 1]
    if (meta.get('command') != command or not meta.get('end')
            or meta.get('name') != expected_name or cfg.get('name') != expected_name
            or meta.get('exit_code') != record.get('exit_code')
            or measured.get('producer_exit_code') != meta.get('exit_code')
            or cfg.get('version') != export.VERSION or measured.get('kind') != 't02_table_export'
            or cfg.get('pbs_jobid') != identity['pbs_jobid']
            or cfg.get('cuda_visible_devices') != identity['cuda_visible_devices']
            or cfg.get('cpu_affinity') != identity['cpu_affinity']
            or cfg.get('scratch') != identity['scratch']
            or cfg.get('node') != identity['node']
            or cfg.get('gpus') != 1 or cfg.get('pair_id') != [entry['pair_id']]
            or cfg.get('source') != entry['source']['file'] or cfg.get('pair_file') != entry['pair']['file']):
        raise ValueError('Original child command, source or allocation differs from its plan')
    runtime = cfg.get('runtime')
    if runtime is not None and any(runtime.get(key) != identity[key] for key in
            ('pbs_jobid', 'cuda_visible_devices', 'cpu_affinity', 'scratch')):
        raise ValueError('Original child runtime changed the shared allocation')
    if runtime is not None and runtime.get('gpu_uuid') != identity['cuda_visible_devices']:
        raise ValueError('Original child renderer changed the allocated GPU UUID')
    rows = measured.get('pairs', [])
    if not isinstance(rows, list):
        raise ValueError('Original pairs must be an explicit list')
    if rows:
        if len(rows) != 1 or rows[0].get('pair_id') != entry['pair_id']:
            raise ValueError('Single-pair unit reported another pair')
        reference = rows[0].get('source_package')
        if not isinstance(reference, dict) or sha_file(reference['file']) != reference['sha256']:
            raise ValueError('Original authenticated source package is missing or changed')
        package = json.loads(Path(reference['file']).read_text())
        provenance = package['source_provenance']
        if (package['scene'] != entry['scene'] or package['source_config'].get('field_of_view') != 60
                or package['independent_verification']['proposal_identity'] != entry['proposal_identity']
                or any(provenance['container'].get(k) != entry['source'][k] for k in ('file', 'sha256'))
                or any(provenance['pair'].get(k) != entry['pair'][k] for k in ('file', 'sha256'))):
            raise ValueError('Independently authenticated pair differs from prepared source identity')
    cleanup = measured.get('cleanup_errors')
    if not isinstance(cleanup, list):
        raise ValueError('Missing original cleanup observations')
    errors, unstarted = measured.get('errors'), measured.get('unstarted_pair_ids')
    if not isinstance(errors, list) or not isinstance(unstarted, list):
        raise ValueError('Missing original errors or unstarted-pair observations')
    complete = measured.get('export_checks_complete') is True
    if complete and (meta['exit_code'] != 0 or cleanup or errors or unstarted
                     or len(rows) != 1 or rows[0].get('status') not in ('exported', 'rejected')
                     or measured.get('black_frame_detected') is True):
        raise ValueError('Original completion flag contradicts observed terminal result')
    return {'source_status': measured.get('status'), 'producer_exit_code': meta['exit_code'],
            'cleanup_errors': copy.deepcopy(cleanup), 'errors': copy.deepcopy(measured.get('errors')),
            'black_frame_detected': measured.get('black_frame_detected') is True,
            'original_pairs': copy.deepcopy(rows),
            'original_unstarted_pair_ids': copy.deepcopy(measured.get('unstarted_pair_ids')),
            'export_checks_complete': measured.get('export_checks_complete') is True,
            'cache_matches_live': None, 'downstream_allowed': False}


def run_units(entries, *, name, concurrency, directory, identity, deadline,
              unit_seconds, pair_seconds, gate_seconds, sources, stopping):
    if concurrency not in (1, 2, 4, 6) or len(identity['cpu_affinity']) != 12:
        raise ValueError('Require at most six single-Unity units on the shared twelve-CPU allocation')
    active, finished, errors, cleanup = {}, [], [], []
    next_index, peak, black = 0, 0, False
    started = time.monotonic()

    def finish(index, running, stopped_by_parent=False):
        nonlocal black
        try:
            record, data = finish_logged(running, sources)
        except Exception as error:
            # Preserve the failed unit even if its wrapper log/result cannot be
            # authenticated; still clean every other owned child in finally.
            running['stream'].close()
            record = {'run_name': running['name'], 'exit_code': running['process'].poll(),
                      'result_path': f"results/{running['name']}.json",
                      'provenance_error': f'{type(error).__name__}: {error}'}
            data = None
        record.update(unit_index=index, entry=copy.deepcopy(entries[index]), command=running['producer_command'],
                      stopped_by_parent=stopped_by_parent)
        try:
            record.update(child_outcome(record, data, identity, running['producer_command'], entries[index]))
        except (ValueError, KeyError, TypeError, OSError) as error:
            record['authentication_error'] = f'{type(error).__name__}: {error}'
            errors.append(record['authentication_error'])
        finished.append(record)
        cleanup.extend(record.get('cleanup_errors', []))
        black = black or record.get('black_frame_detected', False)
        return record

    def reap():
        for index, running in list(active.items()):
            if running['process'].poll() is not None:
                finish(index, running)
                del active[index]

    try:
        while active or next_index < len(entries):
            reap()
            if stopping or black or cleanup or errors or time.time() >= deadline - 45:
                break
            while next_index < len(entries) and len(active) < concurrency:
                reap()
                if stopping or black or cleanup or errors:
                    break
                # Never silently shorten a unit's predeclared complete grant.
                if time.time() + unit_seconds + 45 > deadline:
                    stopping.append('insufficient_parent_time_for_full_next_unit_grant')
                    break
                index = next_index
                child_name = f'{name}_u{index:04d}'
                command = unit_command(child_name, entries[index], unit_seconds, pair_seconds, gate_seconds)
                child = launch_logged(child_name, command, os.environ, directory)
                child['producer_command'] = command
                active[index] = child
                next_index += 1
                peak = max(peak, len(active))
            if active:
                time.sleep(.1)
            elif next_index < len(entries):
                break
    except Exception as error:
        errors.append(f'{type(error).__name__}: {error}')
    finally:
        if active:
            was_running = {index: running['process'].poll() is None for index, running in active.items()}
            # Ask every owned producer to clean up before waiting on any one.
            for running in active.values():
                try:
                    request_stop_logged(running['process'])
                except Exception as error:
                    cleanup.append(f'Owned child stop request: {error}')
            grace, hard = time.monotonic() + 20, time.monotonic() + 28
            for index, running in list(active.items()):
                try:
                    cleanup.extend(stop_logged(running['process'], request=False,
                                               grace_deadline=grace, hard_deadline=hard))
                except Exception as error:
                    cleanup.append(f'Owned child cleanup: {type(error).__name__}: {error}')
                if running['process'].poll() is not None:
                    finish(index, running, stopped_by_parent=was_running[index])
                else:
                    errors.append('Owned logged child remained alive after bounded cleanup')
            active.clear()
    finished.sort(key=lambda row: row['unit_index'])
    return {'units': finished, 'unstarted_unit_indices': list(range(next_index, len(entries))),
            'stopping_reasons': list(stopping), 'cleanup_errors': cleanup, 'errors': errors,
            'black_frame_detected': black, 'maximum_simultaneous_unit_processes': peak,
            'parallel_phase_elapsed_seconds': time.monotonic() - started,
            'all_units_ended': len(finished) == len(entries) and not cleanup and not errors,
            'downstream_allowed': False, 'cache_matches_live': None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--concurrency', type=int, choices=(1, 2, 4, 6), default=6)
    parser.add_argument('--timeout-seconds', type=float, required=True)
    parser.add_argument('--unit-timeout-seconds', type=float, required=True)
    parser.add_argument('--pair-timeout-seconds', type=float, required=True)
    parser.add_argument('--gate-timeout-seconds', type=float, default=180)
    args = parser.parse_args(argv)
    if (not re.fullmatch(r't02_export_[A-Za-z0-9_]+', args.name)
            or not all(math.isfinite(x) and x > 0 for x in (args.timeout_seconds, args.unit_timeout_seconds,
                                                           args.pair_timeout_seconds, args.gate_timeout_seconds))
            or not 15 <= args.gate_timeout_seconds <= 600
            or args.pair_timeout_seconds > args.unit_timeout_seconds - 30
            or args.unit_timeout_seconds <= args.gate_timeout_seconds + 60
            or args.timeout_seconds < args.gate_timeout_seconds + args.unit_timeout_seconds + 60):
        parser.error('Require a unique T02 export name, full finite unit grants and parent cleanup reserve')
    output = ROOT / 'results' / (args.name + '.json')
    if output.exists() or output.is_symlink():
        raise FileExistsError('Previous evidence is preserved; use a fresh name')
    config = {**vars(args), 'manifest': str(args.manifest), 'version': VERSION,
              'gpus': 1, 'cpu_cores_per_gpu': 12, 'node': socket.gethostname(),
              'maximum_unity_instances': args.concurrency,
              'unit_producer': 'src.tools.t02_export', 'unit_producer_version': export.VERSION,
              'gate_policy': 'parent gate completes first; each original child retains its own original gate',
              'scope': 'same GPU UUID and shared12CPU; independent original results; no scientific acceptance inferred'}
    metrics = {'kind': 't02_parallel_export', 'status': 'failed_before_gate', 'units': [],
               'gate': None, 'cleanup_errors': [], 'errors': [], 'downstream_allowed': False,
               'black_frame_detected': False, 'cache_matches_live': None}
    sources, stopping, code = [], [], 1
    old_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in old_signals:
        signal.signal(sig, lambda signum, frame: stopping.append(f'signal:{signum}'))
    try:
        identity = {**export.execution_environment(os.environ, os.sched_getaffinity(0)), 'node': config['node']}
        config.update(identity)
        deadline = deadline_after(args.timeout_seconds)
        config['absolute_deadline_epoch'] = deadline
        directory = Path(identity['scratch']) / 't02_export_parallel' / args.name / identity['pbs_jobid']
        directory.mkdir(parents=True, exist_ok=False)
        config['scratch_directory'] = str(directory)
        # Pin manifest bytes before GPU work; do not run source science until the gate passes.
        manifest, reference = read_manifest(args.manifest, args.manifest_sha256)
        config.update(manifest_reference=reference, partition=manifest['partition'],
                      planned_units=copy.deepcopy(manifest['entries']))
        metrics['unstarted_unit_indices'] = list(range(len(manifest['entries'])))
        gate, gate_ref, cleanup = export.logged_gate(args.name + '_gate', args.gate_timeout_seconds,
                                                    directory, deadline, sources)
        metrics['gate'], metrics['cleanup_errors'] = gate_ref, cleanup
        if gate['_meta']['exit_code'] != 0:
            metrics.update(status='blocked_by_render_gate', black_frame_detected=gate['_meta']['exit_code'] == 86)
            code = 86 if metrics['black_frame_detected'] and not cleanup and gate['metrics'].get('cleanup_errors') == [] else 1
        else:
            export.gate_runtime(gate, identity, config['node'], directory / 'unused_parent_unity')
            if cleanup:
                raise RuntimeError('Original parent gate cleanup is uncertain')
            result = run_units(manifest['entries'], name=args.name, concurrency=args.concurrency,
                directory=directory, identity=identity, deadline=deadline,
                unit_seconds=args.unit_timeout_seconds, pair_seconds=args.pair_timeout_seconds,
                gate_seconds=args.gate_timeout_seconds, sources=sources, stopping=stopping)
            metrics.update(result)
            complete = result['all_units_ended'] and all(row.get('export_checks_complete') for row in result['units'])
            code = 0 if complete else 1
            metrics['status'] = 'original_exports_completed_cache_pending' if complete else 'incomplete'
            if (result['black_frame_detected'] and not result['cleanup_errors'] and not result['errors']
                    and all(row.get('producer_exit_code') in (0, 86) or row.get('stopped_by_parent')
                            for row in result['units'])):
                code, metrics['status'] = 86, 'black_frame'
    except BaseException as error:
        traceback.print_exc()
        metrics['errors'].append(f'{type(error).__name__}: {error}')
        code = 1
    finally:
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)
    metrics['producer_exit_code'] = code
    export.save_json(output, {'metrics': metrics, 'config': config, 'sources': sources})
    print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
