"""One GPU: fixed-six-scene G01 throughput trials at 2/4/6 scene concurrency.

Each stage is a repeated throughput source, not six additional independent
training scenes. Final pool assembly must deduplicate proposals across stages
and apply the aggregate eight-per-scene/class cap before freezing a pool.
"""
import argparse
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

from src.envs.thor_common import ROOT, write_result
from src.envs import pair_v2_g01_policy as policy
from src.envs import pair_v2_stay_order as stay
from src.tools.g01_admission import (AdmissionPool, capacity, child_assignment, create_plan)
from src.tools.t01_partition import sha_file
from src.tools.t01_parallel_approved import (finish_logged,
                                           request_stop_logged, stop_logged)
from src.tools.t01v2_parallel import shard_configuration, validate_mask, verify_shard_result

SCENES = ('FloorPlan1', 'FloorPlan7', 'FloorPlan203', 'FloorPlan204', 'FloorPlan301', 'FloorPlan401')
CONCURRENCIES = (2, 4, 6)
PARTITION = 'train'  # collection wrapper explicitly binds test; benchmark stays training-only
SCENE_SECONDS = 1200
CHILD_FINALIZATION_SECONDS = 60
GATE_SECONDS = 120
GATE_WRAPPER_SECONDS = 10
STAGE_FINALIZATION_SECONDS = 30
PARENT_CLEANUP_SECONDS = 60


def launch_logged(name, command, environment, directory):
    """Same logged child contract in G01's separate, non-overwriting namespace."""
    if re.fullmatch(r'gen_(?:train|test)_[A-Za-z0-9_]+', name) is None:
        raise ValueError('Require an explicit unique gen_train_/gen_test_ child name')
    paths = [ROOT / 'results' / (name + '.json'), ROOT / 'logs/runs' / name]
    paths += [ROOT / 'logs/runs' / (name + suffix) for suffix in ('.log', '.meta.json', '.diff')]
    if any(path.exists() or path.is_symlink() for path in paths):
        raise FileExistsError('Prior G01 evidence exists; choose a new child name')
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


def required_timeout():
    """Execution caps, not an estimate of observed rendering throughput."""
    waves = sum(math.ceil(len(SCENES) / count) for count in CONCURRENCIES)
    return (GATE_SECONDS + GATE_WRAPPER_SECONDS
            + waves * (SCENE_SECONDS + CHILD_FINALIZATION_SECONDS)
            + len(CONCURRENCIES) * STAGE_FINALIZATION_SECONDS + PARENT_CLEANUP_SECONDS)


def full_scene_grant(deadline, now):
    if deadline - now < SCENE_SECONDS + CHILD_FINALIZATION_SECONDS + PARENT_CLEANUP_SECONDS:
        raise TimeoutError('Insufficient remaining allocation for a full1200-second scene grant')
    return SCENE_SECONDS


def assignments(name, stage_index, concurrency):
    capacity(concurrency)
    if CONCURRENCIES[stage_index] != concurrency:
        raise ValueError('Require the fixed2/4/6 stage order')
    return [{'scene': scene, 'lane_index': index % concurrency,
             'child_run_name': f'{name}_c{concurrency}_{scene.lower()}'}
            for index, scene in enumerate(SCENES)]


def submission_identity(repo, name, job, timeout):
    matches = []
    for path in (repo / 'logs/pbs').glob('*/payload.json'):
        data = json.loads(path.read_text())
        if data.get('name') != name or data.get('dry_run'):
            continue
        receipt = json.loads(path.with_name('submission.json').read_text())
        ids = re.findall(r'\b\d+\.hopper-m-\d+\b', receipt.get('stdout', ''))
        command = data.get('command', [])
        if (ids != [job] or data.get('budget_group') != 'G01' or data.get('task_id') != 'T01'
                or data.get('budget_batch') != 't00b_t01_20261108' or data.get('queue') != 'autox'
                or data.get('requested_gpu_count') != 1
                or type(data.get('requested_wall_seconds')) is not int
                or timeout > data['requested_wall_seconds'] - 30
                or command.count('src.tools.g01_parallel') != 1):
            raise ValueError('G01 allocation differs from its budgeted unpinned single-GPU submission')
        from src.tools.pbs_black_retry import validate_payload_host
        host_authorization = validate_payload_host(repo, data)
        matches.append({'file': str(path.relative_to(repo)), 'sha256': sha_file(path),
            'requested_wall_seconds': data['requested_wall_seconds'], 'budget_group': 'G01'})
    if len(matches) != 1:
        raise ValueError('Expected exactly one immutable G01 reservation for this job/name')
    return matches[0]


def make_shard(name, scene, cpus, gate, record, raw, plan, reference):
    if len(cpus) != 12 or len(set(cpus)) != 12:
        raise ValueError('Every lane must retain all twelve original allocated CPUs')
    shard = shard_configuration(name, scene, 0, list(cpus), gate, record['source'], record['result_path'],
                                list(SCENES), raw, field_of_view=60)
    partition = plan.get('g01_partition', 'train')
    shard.update(policy.declared(scene, partition), **stay.fields(), producer_module=policy.PRODUCER,
                 g01_admission=child_assignment(plan, reference, scene))
    return shard


def verify_child(record, data, expected):
    from src.tools.t01v2_g01_review_checkpoint import verify_admission
    with policy.camera_support():
        verify_shard_result(record, data, expected)
    policy.validate_source(data['config'], data['metrics'])
    if data['config']['shard'].get('g01_admission') != expected['g01_admission']:
        raise ValueError('Child returned a different stage/lane admission assignment')
    verify_admission(data)


def verify_gate(gate, record, config, mask):
    allocation = gate['config']['allocation']
    if (record['exit_code'] or record.get('provenance_error')
            or gate['metrics'].get('passed') is not True or gate['metrics'].get('cleanup_errors') != []
            or allocation.get('pbs_jobid') != config['pbs_jobid']
            or allocation.get('cuda_visible_devices') != mask
            or [row['gpu_uuid'] for row in allocation['devices']] != [mask]
            or gate['config'].get('node') != config['node']
            or gate['config'].get('resolution') != 255 or gate['config'].get('segmentation') is not True
            or len(gate['metrics'].get('devices', [])) != 1
            or any(len(row.get('observation', {}).get('captures', [])) != 2
                   or any(cap.get('segmentation', {}).get('valid') is not True
                          for cap in row.get('observation', {}).get('captures', []))
                   for row in gate['metrics'].get('devices', []))):
        raise ValueError('Gate does not verify this exact single-GPU PBS allocation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=required_timeout())
    args = parser.parse_args()
    if not re.fullmatch(r'gen_' + PARTITION + r'_[A-Za-z0-9_]+', args.name):
        parser.error('Use the unique gen_train_/gen_test_ name for the declared partition')
    if args.timeout_seconds != required_timeout():
        parser.error(f'This fixed benchmark requires its declared {required_timeout()}-second execution cap')
    output = ROOT / 'results' / f'{args.name}.json'
    if output.exists() or output.is_symlink():
        raise FileExistsError('Preserve existing benchmark results; use a unique name')
    config = {'name': args.name, 'pbs_jobid': os.environ.get('PBS_JOBID'), 'node': socket.gethostname(),
        'budget_group': 'G01', 'gpus': 1, 'seed': 0, 'resolution': 255, 'field_of_view': 60,
        'scenes': list(SCENES), 'concurrencies': list(CONCURRENCIES), 'cpu_cores_per_gpu': 12,
        'g01_partition': PARTITION, 'per_scene_class_cap': policy.class_cap(PARTITION),
        'scene_timeout_seconds': SCENE_SECONDS, 'child_finalization_seconds': CHILD_FINALIZATION_SECONDS,
        'gate_timeout_seconds': GATE_SECONDS, 'gate_wrapper_seconds': GATE_WRAPPER_SECONDS,
        'stage_finalization_seconds': STAGE_FINALIZATION_SECONDS, 'parent_cleanup_seconds': PARENT_CLEANUP_SECONDS,
        'timeout_seconds': args.timeout_seconds, 'required_timeout_seconds': required_timeout(),
        'deadline_scope': 'fixed full scene grants plus bounded startup/finalization/cleanup allowances; not measured speed',
        'pool_scope': 'three repeated throughput trials on the same six scenes; deduplicate actual proposals across stages and enforce total8 per scene/class before training-pool freeze',
        'producer_module': policy.PRODUCER, 'stage_plans': []}
    metrics = {'kind': 'g01_parallel_benchmark', 'status': 'failed_before_gate', 'acceptance_passed': False,
        'gate': None, 'stages': [], 'scene_runs': [], 'started_shards': [], 'cleanup_errors': [],
        'black_frame_detected': False}
    sources, active, stopping, pools = [], {}, [], []
    gate_running, code = None, 1
    deadline = time.monotonic() + args.timeout_seconds
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stopping.append(f'signal:{signum}'))

    def finish_child(running, *, stopped=False):
        row, data = finish_logged(running, sources)
        row.update(scene=running['scene'], stage_index=running['stage_index'],
            concurrency=running['concurrency'], lane_index=running['lane_index'], logical_index=0,
            gpu_uuid=config['parent_cuda_visible_devices'], granted_timeout_seconds=SCENE_SECONDS,
            stopped_by_orchestrator=stopped)
        metrics['scene_runs'].append(row)
        try:
            verify_child(row, data, running['shard'])
            row['reported_passed_by_class'] = data['metrics'].get('passed_by_class')
            row['reported_attempts'] = data['metrics'].get('attempts')
        except (ValueError, KeyError, TypeError, OSError) as error:
            row['provenance_error'] = f'{type(error).__name__}: {error}'
            metrics['cleanup_errors'].append('Unverified child identity/cleanup: ' + running['name'])
            stopping.append('child_cleanup_uncertain')
        if row['exit_code'] == 86:
            metrics['black_frame_detected'] = True
            stopping.append('measured_black_frame')
        print('G01_SCENE_FINISHED: ' + json.dumps(row, sort_keys=True), flush=True)

    try:
        if not config['pbs_jobid'] or not os.environ.get('CVPR_SCRATCH'):
            raise RuntimeError('PBS_JOBID and scratch required; no login-node rendering')
        config['submission'] = submission_identity(ROOT, args.name, config['pbs_jobid'], args.timeout_seconds)
        mask = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        validate_mask(mask, 1)
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) != 12:
            raise ValueError('Require the complete original allocation of exactly twelve CPUs')
        config.update(parent_cuda_visible_devices=mask, cpu_affinity=cpus)
        tag = re.sub(r'[^A-Za-z0-9]', '_', config['pbs_jobid'])
        scratch = Path(os.environ['CVPR_SCRATCH']) / 'g01_parallel' / args.name / tag
        scratch.mkdir(parents=True, exist_ok=False)
        gate_name = f'{args.name}_j{tag}_gate'
        gate_running = launch_logged(gate_name, [sys.executable, '-u', '-m', 'src.tools.render_gate',
            '--name', gate_name, '--gpus', '1', '--resolution', '255', '--segmentation',
            '--timeout-seconds', str(GATE_SECONDS)], os.environ, scratch)
        until = time.monotonic() + GATE_SECONDS + GATE_WRAPPER_SECONDS
        while gate_running['process'].poll() is None and not stopping and time.monotonic() < until:
            time.sleep(.1)
        if gate_running['process'].poll() is None:
            metrics['cleanup_errors'].extend(stop_logged(gate_running['process']))
        gate_record, gate = finish_logged(gate_running, sources)
        metrics['gate'], gate_running = gate_record, None
        if gate_record['exit_code'] or gate_record.get('provenance_error') or stopping:
            metrics['status'] = 'blocked_by_render_gate'
            metrics['black_frame_detected'] = (gate_record['exit_code'] == 86
                and not gate_record.get('provenance_error') and not metrics['cleanup_errors'] and not stopping)
        else:
            verify_gate(gate, gate_record, config, mask)
            config['allocation'] = gate['config']['allocation']
            metrics['status'] = 'collecting'
            for stage_index, concurrency in enumerate(CONCURRENCIES):
                if stopping:
                    break
                stage_dir = scratch / f'c{concurrency}'
                stage_dir.mkdir(exist_ok=False)
                rows = assignments(args.name, stage_index, concurrency)
                plan, reference = create_plan(stage_dir, parent_run_name=args.name, stage_index=stage_index,
                    concurrency=concurrency, assignments=rows, pbs_jobid=config['pbs_jobid'],
                    node=config['node'], gpu_uuid=mask, cpu_affinity=cpus, scenes=list(SCENES), partition=PARTITION)
                config['stage_plans'].append({**reference, 'plan': plan})
                pool = AdmissionPool(child_assignment(plan, reference, SCENES[0]))
                pools.append(pool)
                queues = [[row for row in rows if row['lane_index'] == lane] for lane in range(concurrency)]
                positions = [0] * concurrency
                stage_start = time.monotonic()
                stage = {'stage_index': stage_index, 'concurrency': concurrency,
                    'witness_capacity': capacity(concurrency), 'max_unity_instances': concurrency + capacity(concurrency),
                    'fixed_scenes': list(SCENES), 'plan': reference, 'completed': False}
                metrics['stages'].append(stage)
                while True:
                    health = pool.health()
                    if health['poison']:
                        stopping.append('witness_admission_poison:' + health['poison'])
                    if time.monotonic() >= deadline - PARENT_CLEANUP_SECONDS:
                        stopping.append('parent_deadline')
                    for lane in range(concurrency):
                        running = active.get(lane)
                        if running and running['process'].poll() is None and time.monotonic() >= running['child_deadline']:
                            stopping.append('scene_finalization_deadline:' + running['scene'])
                        if running and running['process'].poll() is not None:
                            finish_child(running)
                            active.pop(lane)
                        if lane not in active and not stopping and positions[lane] < len(queues[lane]):
                            assignment = queues[lane][positions[lane]]
                            grant = full_scene_grant(deadline, time.monotonic())
                            scene, name = assignment['scene'], assignment['child_run_name']
                            raw = stage_dir / scene
                            raw.mkdir(exist_ok=False)
                            shard = make_shard(name, scene, cpus, gate, gate_record, raw, plan, reference)
                            path = raw / 'shard_config.json'
                            with path.open('x') as stream:
                                json.dump(shard, stream, indent=2, allow_nan=False)
                            path.chmod(0o444)
                            env = dict(os.environ, CUDA_VISIBLE_DEVICES=mask, T01_PARENT_CUDA_VISIBLE_DEVICES=mask)
                            running = launch_logged(name, [sys.executable, '-u', '-m', policy.PRODUCER,
                                '--name', name, '--scene', scene, '--shard-config', str(path),
                                '--timeout-seconds', str(grant)], env, stage_dir)
                            running.update(scene=scene, shard=shard, stage_index=stage_index, concurrency=concurrency,
                                lane_index=lane, child_deadline=time.monotonic() + grant + CHILD_FINALIZATION_SECONDS)
                            active[lane] = running
                            positions[lane] += 1
                            metrics['started_shards'].append({'scene': scene, 'run_name': name,
                                'stage_index': stage_index, 'concurrency': concurrency, 'lane_index': lane,
                                'logical_index': 0, 'gpu_uuid': mask, 'cpu_affinity': cpus,
                                'wrapper_pid': running['process'].pid, 'shard_config': str(path),
                                'shard_config_sha256': sha_file(path)})
                    if stopping or (not active and all(positions[lane] == len(queues[lane]) for lane in range(concurrency))):
                        break
                    time.sleep(.1)
                stage['wall_seconds_observed'] = time.monotonic() - stage_start
                stage['admission_state_at_stage_end'] = pool.health()
                stage['completed'] = not stopping and not active and not stage['admission_state_at_stage_end']['active_leases']
                if not stage['completed']:
                    stopping.append('stage_not_cleanly_complete')
                    break
            metrics['status'] = 'stopped' if stopping else 'completed'
        code = 86 if metrics['black_frame_detected'] else (0 if metrics['status'] == 'completed'
            and not stopping and not any(row['exit_code'] or row.get('provenance_error') for row in metrics['scene_runs']) else 1)
    except BaseException as error:
        metrics.update(status='orchestration_failed', error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
        code = 1
    finally:
        cleanup_start = time.monotonic()
        pending = list(active.values()) + ([gate_running] if gate_running else [])
        for running in pending:
            try:
                request_stop_logged(running['process'])
            except Exception as error:
                metrics['cleanup_errors'].append(f"request stop {running['name']}: {error}")
        for running in pending:
            metrics['cleanup_errors'].extend(stop_logged(running['process'], request=False,
                grace_deadline=cleanup_start + 20, hard_deadline=cleanup_start + 28))
            if running['process'].poll() is None:
                running['stream'].close()
                metrics['cleanup_errors'].append('Unfinished wrapper: ' + running['name'])
                continue
            if running is gate_running:
                metrics['gate'], _ = finish_logged(running, sources)
            else:
                finish_child(running, stopped=True)
        metrics['admission_final'] = []
        for pool in pools:
            try:
                health = pool.health()
                metrics['admission_final'].append({'stage_index': pool.plan['stage_index'], **health})
                if health['poison'] or health['active_leases']:
                    metrics['cleanup_errors'].append('Witness cleanup not confirmed for stage' + str(pool.plan['stage_index']))
            except Exception as error:
                metrics['cleanup_errors'].append(f'Admission finalization: {error}')
        launched = {row['run_name'] for row in metrics['started_shards']}
        finished = {row['run_name'] for row in metrics['scene_runs']}
        planned = [row for index, count in enumerate(CONCURRENCIES) for row in assignments(args.name, index, count)]
        metrics.update(unstarted_assignments=[row for row in planned if row['child_run_name'] not in launched],
            unfinished_run_names=sorted(launched - finished), stop_reasons=stopping)
        metrics['all_planned_assignments_finished'] = (not metrics['unstarted_assignments']
            and not metrics['unfinished_run_names'] and not stopping)
        if metrics['cleanup_errors']:
            code = 1
        metrics['exit_code'] = code
        write_result(args.name, metrics, config)
        output.write_text(json.dumps({'metrics': metrics, 'config': config, 'sources': sources},
                                     indent=2, allow_nan=False) + '\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
