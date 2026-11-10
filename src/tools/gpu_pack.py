"""One or four allocated GPUs consuming the same immutable scientific queue.

Only scheduling changes. Each claimed unit keeps its original command, complete
time grant, render gate, checks, and logged outcome. Empty queues never wait.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback

from src.tools.gpu_shared_claim import (ROOT, UUID, SAFE, load_plan, read_reference, reference,
    require, validate_runtime_claim, write_immutable)
from src.tools.t01_parallel_approved import finish_logged, request_stop_logged, stop_logged

VERSION = 'shared_gpu_pack_v1'


def queue_module():
    spec = importlib.util.spec_from_file_location('original_gpu_queue_for_pack', ROOT / 'scripts/gpu_queue.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def submission_binding(name, plan_ref, gpus, environment, actual_arguments, timeout_seconds):
    job = environment.get('PBS_JOBID', '')
    matches = []
    for directory in (ROOT / 'logs/pbs').glob(name + '-*'):
        p, r = directory / 'payload.json', directory / 'submission.json'
        if not p.exists() or not r.exists():
            continue
        payload, receipt = json.loads(p.read_text()), json.loads(r.read_text())
        if receipt.get('returncode') == 0 and receipt.get('stdout', '').strip() == job:
            matches.append((p, r, payload))
    require(len(matches) == 1, 'Pack requires one exact immutable successful PBS receipt')
    p, r, payload = matches[0]
    from src.tools.pbs_mixed_accounting import validate_payload
    binding = validate_payload(ROOT, payload)
    if payload.get('requested_host') is not None:
        from src.tools.pbs_black_retry import validate_payload_host
        validate_payload_host(ROOT, payload)
    command = payload['command']
    require(command[command.index('-m') + 2:] == actual_arguments
            and timeout_seconds <= payload['requested_wall_seconds'] - 30,
            'Actual pack CLI or full parent deadline differs from the immutable submission')
    require(payload['name'] == name and payload['requested_gpu_count'] == gpus
            and binding['plan_reference'] == plan_ref
            and payload.get('environment', {}).get('CVPR_SCRATCH') == environment.get('CVPR_SCRATCH'),
            'Original pack submission/allocation differs')
    return payload, reference(p), reference(r)


def lane_environment(parent, gpu, claim=None):
    env = dict(parent, CUDA_VISIBLE_DEVICES=gpu, CVPR_QUEUE_PARENT_MASK=parent['CUDA_VISIBLE_DEVICES'])
    env.pop('CVPR_SHARED_CLAIM_FILE', None);env.pop('CVPR_SHARED_CLAIM_SHA256', None)
    if claim is not None:
        env.update(CVPR_SHARED_CLAIM_FILE=claim['claim_reference']['file'],
                   CVPR_SHARED_CLAIM_SHA256=claim['claim_reference']['sha256'])
    return env


def verify_gate(data, identity):
    """Original per-card gate; a black card is retired without consuming work."""
    from src.tools.t02_export import gate_runtime
    cfg, metrics, meta = data['config'], data['metrics'], data['_meta']
    require(cfg['allocation']['cpu_affinity'] == identity['cpu_affinity'], 'Gate CPU mask differs')
    require(metrics.get('cleanup_errors') == [] and metrics.get('errors') == [], 'Gate cleanup or setup is uncertain')
    if meta['exit_code'] == 0:
        gate_runtime(data, identity, identity['node'], Path(identity['scratch']) / 'unused_pack_gate')
        return 'healthy'
    require(meta['exit_code'] == 86 and metrics.get('exit_code') == 86
            and metrics.get('black_frame_detected') is True and metrics.get('passed') is False,
            'Original gate failed for a nonblack reason')
    allocation = cfg['allocation']
    require(allocation['pbs_jobid'] == identity['pbs_jobid']
            and allocation['cuda_visible_devices'] == identity['cuda_visible_devices']
            and allocation['cuda_driver_uuids'] == [identity['cuda_visible_devices']]
            and cfg['node'] == identity['node'] and metrics['expected_gpu_count'] == metrics['completed_gpu_count'] == 1
            and len(metrics['devices']) == 1 and cfg['resolution'] == 255 and cfg['segmentation'] is True,
            'Original black gate allocation differs')
    row = metrics['devices'][0];observed = row['observation']
    require(row['worker_error'] is None and observed['gpu_identity'] == allocation['devices'][0]
            and observed['gpu_identity']['gpu_uuid'] == identity['cuda_visible_devices']
            and [capture['stage'] for capture in observed['captures']] == ['after_constructor', 'after_Pass'],
            'Black gate lacks the original per-card fresh captures')
    from PIL import Image
    for capture in observed['captures']:
        p = Path(capture['png']['path']).resolve();p.relative_to(ROOT / 'logs/runs')
        require(hashlib.sha256(p.read_bytes()).hexdigest() == capture['png']['sha256']
                and capture['shape'] == [255, 255, 3] and capture['action_success'] is True
                and capture['black'] is True and capture['nonzero_rgb_values'] == 0,
                'Black frame measurement differs')
        with Image.open(p) as image:
            require(image.mode == 'RGB' and image.size == (255, 255)
                    and image.getextrema() == ((0, 0),) * 3, 'Original gate PNG is not black')
    return 'black'


def end_unit(queue, running, sources):
    record, data = finish_logged(running, sources)
    claim = running['claim']
    require(data is not None and not record.get('provenance_error'), 'Original unit result did not authenticate')
    mask_key = ('parent_cuda_visible_devices' if claim['unit']['producer_module'] == 'src.tools.g01_collect_parallel'
                else 'cuda_visible_devices')
    require(data['_meta']['command'] == claim['command'] and data['_meta']['name'] == claim['name']
            and data['config']['pbs_jobid'] == claim['pbs_jobid']
            and data['config'][mask_key] == claim['gpu_uuid']
            and data['config']['cpu_affinity'] == claim['cpu_affinity']
            and data['config']['node'] == claim['node'], 'Unit result differs from its actual claim')
    require(data['metrics'].get('cleanup_errors') == [], 'Unit did not confirm owned-process cleanup')
    finished, mono = time.time(), time.monotonic()
    release = {key: copy.deepcopy(claim[key]) for key in
        ('unit_id', 'name', 'budget_group', 'pbs_jobid', 'gpu_uuid', 'cpu_affinity',
         'claim_reference', 'started_epoch', 'started_monotonic')}
    release.update(finished_epoch=finished, finished_monotonic=mono,
        duration_seconds_monotonic=mono - claim['started_monotonic'], result_reference=record['source'],
        exit_code=record['exit_code'], cleanup_errors=[])
    release_ref = queue.finish(claim, release)
    return {**release, 'release_reference': release_ref}, data


def launch_failure(queue, claim, error):
    mono = time.monotonic()
    release = {key: copy.deepcopy(claim[key]) for key in
        ('unit_id', 'name', 'budget_group', 'pbs_jobid', 'gpu_uuid', 'cpu_affinity',
         'claim_reference', 'started_epoch', 'started_monotonic')}
    release.update(finished_epoch=time.time(), finished_monotonic=mono,
        duration_seconds_monotonic=mono - claim['started_monotonic'], result_reference=None,
        exit_code=None, cleanup_errors=[], process_started=False,
        launch_error=f'{type(error).__name__}: {error}')
    return {**release, 'release_reference': queue.finish(claim, release)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--queue-plan', required=True)
    parser.add_argument('--queue-plan-sha256', required=True)
    parser.add_argument('--queue-root', type=Path, required=True)
    parser.add_argument('--gpus', type=int, choices=(1, 4), required=True)
    parser.add_argument('--timeout-seconds', type=float, required=True)
    parser.add_argument('--gate-timeout-seconds', type=float, default=180)
    actual_arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(actual_arguments)
    require(args.name.startswith('gpu_pack_') and SAFE.fullmatch(args.name) and all(math.isfinite(x) and x > 0 for x in
            (args.timeout_seconds, args.gate_timeout_seconds)), 'Invalid pack name or deadline')
    output = ROOT / 'results' / (args.name + '.json')
    require(not output.exists(), 'Preserved pack evidence cannot be overwritten')
    config = {**vars(args), 'queue_root': str(args.queue_root), 'version': VERSION,
              'pbs_jobid': os.environ.get('PBS_JOBID'), 'node': socket.gethostname(),
              'scope': 'shared scheduling and allocation cost only; original unit science unchanged'}
    metrics = {'kind': 'gpu_shared_pack', 'status': 'initializing', 'unit_runs': [], 'unit_claims': [], 'gates': [],
               'retired_gpu_uuids': [], 'errors': [], 'cleanup_errors': [], 'all_owned_units_ended': False,
               'downstream_allowed': False}
    sources, active, stopping, code = [], {}, [], 1
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in old_handlers:
        signal.signal(sig, lambda number, _frame: stopping.append(f'signal:{number}'))
    queue = None
    deadline = time.monotonic() + args.timeout_seconds
    try:
        plan, plan_ref = load_plan(args.queue_plan, args.queue_plan_sha256)
        config['plan_reference'] = plan_ref
        scratch = Path(os.environ['CVPR_SCRATCH']).resolve()
        require(args.queue_root.resolve() == scratch / 'shared_gpu_queue' / plan['queue_id'], 'Wrong shared scratch queue')
        parent, payload_ref, receipt_ref = submission_binding(args.name, plan_ref, args.gpus, os.environ,
                                                             actual_arguments, args.timeout_seconds)
        gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        cpus = sorted(os.sched_getaffinity(0))
        require(len(gpus) == len(set(gpus)) == args.gpus and all(UUID.fullmatch(gpu) for gpu in gpus)
                and len(cpus) >= 12 * args.gpus, 'Actual PBS allocation is incomplete')
        config.update(allocated_gpu_uuids=gpus, parent_cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                      cpu_affinity=cpus, parent_payload_reference=payload_ref, parent_receipt_reference=receipt_ref)
        api = queue_module();queue = api.create_shared_queue(args.queue_root, plan, plan_ref)
        directory = scratch / 'gpu_pack' / args.name / config['pbs_jobid'];directory.mkdir(parents=True, exist_ok=False)
        lanes = [{'gpu_uuid': gpu, 'cpu_affinity': cpus[index*12:(index+1)*12], 'lane': index}
                 for index, gpu in enumerate(gpus)]
        # No GPU gate or wait when all approved work was already claimed elsewhere.
        if not queue.has_pending():
            metrics.update(status='queue_empty_on_start', all_owned_units_ended=True)
            code = 0
        else:
            require(time.monotonic() + args.gate_timeout_seconds + 45 <= deadline,
                    'Insufficient parent time for the complete original per-card gate')
            for lane in lanes:
                gate_name = args.name + '_gate_' + str(lane['lane'])
                command = ['python3', '-u', '-m', 'src.tools.render_gate', '--name', gate_name,
                    '--gpus', '1', '--resolution', '255', '--segmentation', '--timeout-seconds', str(args.gate_timeout_seconds)]
                running = api.launch_shared_logged(gate_name, command, lane_environment(os.environ, lane['gpu_uuid']),
                                                  directory, lane['cpu_affinity'])
                running.update(gate_lane=lane)
                active[lane['gpu_uuid']] = running
            while active and not stopping:
                for gpu, running in list(active.items()):
                    if running['process'].poll() is None:
                        continue
                    record, data = finish_logged(running, sources);lane = running['gate_lane'];del active[gpu]
                    row = {'gpu_uuid': gpu, 'cpu_affinity': lane['cpu_affinity'], 'source': record.get('source'),
                           'exit_code': record['exit_code'], 'status': 'failed'}
                    try:
                        require(data is not None and not record.get('provenance_error'), 'Unauthenticated per-card gate')
                        identity = {'pbs_jobid': config['pbs_jobid'], 'node': config['node'],
                                    'cuda_visible_devices': gpu, 'cpu_affinity': lane['cpu_affinity'], 'scratch': str(scratch)}
                        row['status'] = verify_gate(data, identity)
                    except Exception as error:
                        row['error'] = f'{type(error).__name__}: {error}'
                        if data is None or data['metrics'].get('cleanup_errors') != []:
                            metrics['cleanup_errors'].append(row['error'])
                    metrics['gates'].append(row)
                    if row['status'] != 'healthy':metrics['retired_gpu_uuids'].append(gpu)
                if time.monotonic() >= deadline - 45:
                    stopping.append('pack_gate_deadline')
                if active:time.sleep(.1)
            free = [lane for lane in lanes if lane['gpu_uuid'] not in metrics['retired_gpu_uuids']]
            while not stopping and not metrics['cleanup_errors']:
                for gpu, running in list(active.items()):
                    if running['process'].poll() is None:continue
                    ended, data = end_unit(queue, running, sources)
                    metrics['unit_runs'].append(ended);del active[gpu]
                    if data['metrics'].get('black_frame_detected') is True:
                        metrics['retired_gpu_uuids'].append(gpu)
                    else:
                        free.append(running['lane'])
                while free and not stopping:
                    lane = free.pop(0)
                    allocation = {'pbs_jobid': config['pbs_jobid'], 'node': config['node'],
                        'gpu_uuid': lane['gpu_uuid'], 'cpu_affinity': lane['cpu_affinity'],
                        'parent_cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                        'parent_payload_reference': payload_ref, 'parent_receipt_reference': receipt_ref,
                        'owner_identity': api.identity(os.getpid())}
                    claim = queue.claim(allocation, remaining_seconds=deadline-time.monotonic())
                    if claim is None:
                        free.insert(0, lane);break
                    metrics['unit_claims'].append({'unit_id':claim['unit_id'], 'claim_reference':claim['claim_reference']})
                    unit = claim['unit'];identity = {'pbs_jobid': config['pbs_jobid'], 'node': config['node'],
                        'scratch': str(scratch), 'cuda_visible_devices': lane['gpu_uuid'], 'cpu_affinity': lane['cpu_affinity']}
                    environment = lane_environment(os.environ, lane['gpu_uuid'], claim)
                    validate_runtime_claim(environment, identity, unit['producer_module'], unit['name'])
                    try:
                        running = api.launch_shared_logged(unit['name'], unit['command'], environment, directory, lane['cpu_affinity'])
                    except Exception as error:
                        metrics['unit_runs'].append(launch_failure(queue, claim, error))
                        metrics['errors'].append(f'launch:{unit["name"]}:{type(error).__name__}: {error}')
                        stopping.append('unit_launch_failed');break
                    running.update(claim=claim, lane=lane);active[lane['gpu_uuid']] = running
                if not active:
                    if queue.has_pending():
                        stopping.append('insufficient_remaining_time_for_full_original_unit')
                    break  # empty/claimed elsewhere/no full next grant: never idle-wait
                if time.monotonic() >= deadline-45:stopping.append('pack_deadline')
                time.sleep(.1)
            metrics['status'] = 'queue_finished_or_claimed_elsewhere' if not stopping else 'stopped_with_original_work_preserved'
            metrics['stopping_reasons'] = stopping
            code = 0 if not stopping and not metrics['errors'] and not metrics['cleanup_errors'] else 1
            if any(row['exit_code'] != 0 for row in metrics['unit_runs']):code = 1
            if not metrics['unit_runs'] and lanes and len(metrics['retired_gpu_uuids']) == len(lanes):code = 86 if all(row['status']=='black' for row in metrics['gates']) else 1
    except BaseException as error:
        traceback.print_exc();metrics['errors'].append(f'{type(error).__name__}: {error}');code = 1
    finally:
        if active:
            for running in active.values():request_stop_logged(running['process'])
            grace, hard = time.monotonic()+20, time.monotonic()+28
            for running in list(active.values()):
                try:
                    cleanup = stop_logged(running['process'], request=False, grace_deadline=grace, hard_deadline=hard)
                    metrics['cleanup_errors'].extend(cleanup)
                    if 'claim' in running and not cleanup and running['process'].poll() is not None:
                        ended, _ = end_unit(queue, running, sources);metrics['unit_runs'].append(ended)
                    elif running['process'].poll() is not None:
                        record, _ = finish_logged(running, sources)
                        metrics['gates'].append({'status':'stopped','source':record.get('source'),'exit_code':record['exit_code']})
                except Exception as error:
                    metrics['cleanup_errors'].append(f'{type(error).__name__}: {error}')
            active.clear()
        metrics['all_owned_units_ended'] = (not metrics['cleanup_errors']
            and all('release_reference' in row for row in metrics['unit_runs'])
            and sorted(row['unit_id'] for row in metrics['unit_claims']) == sorted(row['unit_id'] for row in metrics['unit_runs']))
        if metrics['cleanup_errors']:code=1
        for sig, handler in old_handlers.items():signal.signal(sig, handler)
    metrics['producer_exit_code'] = code
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:json.dump({'config':config,'metrics':metrics,'sources':sources},stream,indent=2,allow_nan=False);stream.write('\n')
    print('FINAL_METRICS_JSON: '+json.dumps(metrics,sort_keys=True,allow_nan=False),flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
