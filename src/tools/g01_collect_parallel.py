"""Ordinary G01 collection at an explicitly selected, measured concurrency.

Reuse the existing single-GPU scheduler and owned cleanup. This is one stage
over explicit disjoint train/test scenes, not another 2/4/6 throughput benchmark. No speed
estimate or automatic choice of the fastest setting is made here.
"""
import argparse
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import re
import signal
import sys
import traceback

from src.tools import g01_parallel as parent
from src.tools.g01_admission import child_assignment, load_plan, validate_source_admission
from src.tools.t01_partition import audited_result, sha_file
from src.envs.thor_common import ROOT, write_result

MODULE = 'src.tools.g01_collect_parallel'
VERSION = 'g01_single_stage_collection_v1'
BENCHMARK_VALIDATION_SECONDS = 120  # A bounded execution allowance, not observed throughput.
BENCHMARK_SCENES = tuple(parent.SCENES)
BENCHMARK_CONCURRENCIES = tuple(parent.CONCURRENCIES)
SCOPE = ('one collection stage; explicit training scenes once each; aggregate actual proposal '
         'deduplication and total8 per scene/class remain required before training-pool freeze')


def validate_selection(scenes, concurrency, partition='train'):
    parent.capacity(concurrency)
    if (not isinstance(scenes, (list, tuple)) or not scenes or len(set(scenes)) != len(scenes)
            or any(scene not in parent.policy.split_scenes(partition) for scene in scenes)):
        raise ValueError('Require a nonempty unique explicit subset of the declared fixed train/test scene partition')
    if partition == 'test' and concurrency != 6:
        raise ValueError('Evaluation collection uses the frozen six-scene concurrency, never test tuning')
    return tuple(scenes)


def runner_timeout(scenes, concurrency, partition='train'):
    scenes = validate_selection(scenes, concurrency, partition)
    return (parent.GATE_SECONDS + parent.GATE_WRAPPER_SECONDS
            + math.ceil(len(scenes) / concurrency) * (parent.SCENE_SECONDS + parent.CHILD_FINALIZATION_SECONDS)
            + parent.STAGE_FINALIZATION_SECONDS + parent.PARENT_CLEANUP_SECONDS)


def required_timeout(scenes, concurrency, partition='train'):
    return BENCHMARK_VALIDATION_SECONDS + runner_timeout(scenes, concurrency, partition)


def python_arguments(command, module):
    if (not isinstance(command, list) or not command
            or not re.fullmatch(r'python(?:3(?:\.\d+)?)?', Path(command[0]).name)):
        raise ValueError('Require the recorded Python producer command')
    arguments = command[1:]
    if arguments[:1] == ['-u']:
        arguments = arguments[1:]
    if arguments[:2] != ['-m', module]:
        raise ValueError('Unexpected producer module')
    return arguments[2:]


def same_source(actual, saved):
    if not isinstance(saved, dict) or any(actual.get(key) != saved.get(key)
            for key in ('file', 'sha256', 'metrics_sha256', 'git_commit', 'exit_code')):
        raise ValueError('Changed or unbound original logged source')


def benchmark_stage(path, concurrency):
    """Authenticate full work grants and cleanup, including clean-ended partial sources.

    Parent exit1 is retained: each scene must have its full grant and confirmed
    resource cleanup. This does not certify source sealing or scientific acceptance.
    """
    parent.capacity(concurrency)
    data, reference = audited_result(path)
    cfg, metrics, meta = data['config'], data['metrics'], data['_meta']
    name = cfg.get('name')
    caps = {'scene_timeout_seconds': parent.SCENE_SECONDS,
        'child_finalization_seconds': parent.CHILD_FINALIZATION_SECONDS,
        'gate_timeout_seconds': parent.GATE_SECONDS, 'gate_wrapper_seconds': parent.GATE_WRAPPER_SECONDS,
        'stage_finalization_seconds': parent.STAGE_FINALIZATION_SECONDS,
        'parent_cleanup_seconds': parent.PARENT_CLEANUP_SECONDS}
    timeout = (parent.GATE_SECONDS + parent.GATE_WRAPPER_SECONDS
        + sum(math.ceil(len(BENCHMARK_SCENES) / c) for c in BENCHMARK_CONCURRENCIES)
          * (parent.SCENE_SECONDS + parent.CHILD_FINALIZATION_SECONDS)
        + len(BENCHMARK_CONCURRENCIES) * parent.STAGE_FINALIZATION_SECONDS + parent.PARENT_CLEANUP_SECONDS)
    argv = python_arguments(meta.get('command'), 'src.tools.g01_parallel')
    if (not isinstance(name, str) or not re.fullmatch(r'gen_train_[A-Za-z0-9_]+', name)
            or meta.get('name') != name or not meta.get('end') or meta.get('exit_code') not in (0, 1)
            or argv not in (['--name', name], ['--name', name, '--timeout-seconds', str(timeout)])
            or metrics.get('exit_code') != meta['exit_code'] or metrics.get('kind') != 'g01_parallel_benchmark'
            or cfg.get('scenes') != list(BENCHMARK_SCENES)
            or cfg.get('concurrencies') != list(BENCHMARK_CONCURRENCIES)
            or cfg.get('producer_module') != parent.policy.PRODUCER
            or cfg.get('gpus') != 1 or cfg.get('cpu_cores_per_gpu') != 12
            or len(set(cfg.get('cpu_affinity', []))) != 12
            or cfg.get('timeout_seconds') != timeout or cfg.get('required_timeout_seconds') != timeout
            or any(cfg.get(key) != value for key, value in caps.items())
            or metrics.get('cleanup_errors') != [] or metrics.get('black_frame_detected') is not False):
        raise ValueError('Require an ended original benchmark with known full-allocation cleanup')
    reservation = parent.submission_identity(ROOT, name, cfg['pbs_jobid'], timeout)
    if reservation != cfg.get('submission'):
        raise ValueError('Benchmark differs from its original budgeted PBS reservation')
    gate_record = metrics['gate']
    gate, gate_ref = audited_result(ROOT / gate_record['result_path'])
    same_source(gate_ref, gate_record.get('source'))
    def declared_source(ref):
        matches = [row for row in data.get('sources', []) if row.get('file') == ref['file']]
        if len(matches) != 1:
            raise ValueError('Benchmark does not declare this unique original child source')
        same_source(ref, matches[0])
    declared_source(gate_ref)
    parent.verify_gate(gate, gate_record, cfg, cfg['parent_cuda_visible_devices'])
    stage_index = BENCHMARK_CONCURRENCIES.index(concurrency)
    stages = [row for row in metrics.get('stages', []) if row.get('stage_index') == stage_index]
    plans = [row for row in cfg.get('stage_plans', []) if row.get('plan', {}).get('stage_index') == stage_index]
    finals = [row for row in metrics.get('admission_final', []) if row.get('stage_index') == stage_index]
    clean = {'poison': None, 'active_leases': {}}
    if (len(stages) != 1 or len(plans) != 1 or len(finals) != 1
            or finals[0] != {'stage_index': stage_index, **clean}):
        raise ValueError('Selected stage lacks its unique known-clean admission finalization')
    stage, recorded_plan = stages[0], plans[0]
    plan_ref = {key: recorded_plan[key] for key in ('plan_file', 'plan_sha256', 'state_file')}
    if (stage.get('completed') is not True or stage.get('concurrency') != concurrency
            or stage.get('fixed_scenes') != list(BENCHMARK_SCENES) or stage.get('plan') != plan_ref
            or stage.get('admission_state_at_stage_end') != clean
            or type(stage.get('wall_seconds_observed')) not in (int, float)
            or not math.isfinite(stage['wall_seconds_observed']) or stage['wall_seconds_observed'] <= 0):
        raise ValueError('Selected concurrency did not actually complete the same six-scene workload')
    first = child_assignment(recorded_plan['plan'], plan_ref, BENCHMARK_SCENES[0])
    plan = load_plan(first)
    expected = parent.assignments(name, stage_index, concurrency)
    if (plan != recorded_plan['plan'] or plan['assignments'] != expected
            or plan['scenes'] != list(BENCHMARK_SCENES) or plan['parent_run_name'] != name
            or plan['pbs_jobid'] != cfg['pbs_jobid'] or plan['node'] != cfg['node']
            or plan['gpu_uuid'] != cfg['parent_cuda_visible_devices']
            or plan['cpu_affinity'] != cfg['cpu_affinity']):
        raise ValueError('Stage immutable plan differs from the benchmark allocation')
    state = json.loads(Path(plan_ref['state_file']).read_text())
    if (state.get('nonce') != plan['nonce'] or state.get('plan_sha256') != plan_ref['plan_sha256']
            or state.get('version') != plan['version'] or state.get('poison') is not None or state.get('leases') != {}):
        raise ValueError('Original stage admission state is missing or no longer clean')
    rows = [row for row in metrics.get('scene_runs', []) if row.get('stage_index') == stage_index]
    starts = [row for row in metrics.get('started_shards', []) if row.get('stage_index') == stage_index]
    names = {row['child_run_name'] for row in expected}
    if (len(rows) != len(expected) or len(starts) != len(expected)
            or {row.get('run_name') for row in rows} != names or {row.get('run_name') for row in starts} != names):
        raise ValueError('Selected stage lacks exactly one finished actual run per fixed scene')
    children = []
    for assignment in expected:
        row = next(row for row in rows if row['run_name'] == assignment['child_run_name'])
        start = next(row for row in starts if row['run_name'] == assignment['child_run_name'])
        child, child_ref = audited_result(ROOT / row['result_path'])
        same_source(child_ref, row.get('source'))
        declared_source(child_ref)
        if (not child['_meta'].get('end') or row.get('granted_timeout_seconds') != parent.SCENE_SECONDS
                or row.get('stopped_by_orchestrator') is not False
                or row.get('concurrency') != concurrency or row.get('lane_index') != assignment['lane_index']
                or start.get('concurrency') != concurrency or start.get('lane_index') != assignment['lane_index']
                or any(item.get('scene') != assignment['scene'] or item.get('logical_index') != 0
                    or item.get('gpu_uuid') != plan['gpu_uuid'] for item in (row, start))
                or start.get('cpu_affinity') != plan['cpu_affinity']
                or sha_file(Path(start['shard_config'])) != start['shard_config_sha256']):
            raise ValueError('Selected scene was missing, shortened, stopped, or assigned another resource')
        shard_path = Path(start['shard_config'])
        shard = json.loads(shard_path.read_text())
        intended = parent.make_shard(assignment['child_run_name'], assignment['scene'], plan['cpu_affinity'],
            gate, gate_record, shard_path.parent, plan, plan_ref)
        if (shard != intended or child['config']['shard'] != intended
                or python_arguments(child['_meta']['command'], parent.policy.PRODUCER) != [
                    '--name', assignment['child_run_name'], '--scene', assignment['scene'],
                    '--shard-config', str(shard_path), '--timeout-seconds', str(parent.SCENE_SECONDS)]):
            raise ValueError('Child did not execute the immutable full-grant scene assignment')
        parent.verify_child(row, child, intended)
        # Do not accept the ledger-only early-failure exception as cleanup proof.
        validate_source_admission(child['config'], child['metrics'])
        children.append(child_ref)
    return {'source': reference, 'concurrency': concurrency, 'stage_index': stage_index,
        'stage_wall_seconds_observed': stage['wall_seconds_observed'], 'plan': plan_ref,
        'fixed_scenes': list(BENCHMARK_SCENES), 'children': children,
        'parent_exit_code': meta['exit_code'], 'parent_status': metrics.get('status'),
        'selection': 'explicit operator choice; no automatic fastest-setting inference',
        'scope': 'completed workload and known resource cleanup only; no scientific pair acceptance'}


def submission_identity(repo, name, job, timeout, *, scenes, concurrency, benchmark, partition='train', explicit_partition=False):
    expected = ['--name', name, '--scenes', *scenes, '--concurrency', str(concurrency),
                '--benchmark-source', str(benchmark), '--timeout-seconds', str(timeout)]
    validate_selection(scenes, concurrency, partition)
    if not re.fullmatch(r'gen_' + partition + r'_[A-Za-z0-9_]+', name):
        raise ValueError('Collection name differs from its train/test partition')
    if explicit_partition:
        expected += ['--partition', partition]
    elif partition != 'train':
        raise ValueError('Evaluation collection requires explicit --partition test')
    matches = []
    for path in (repo / 'logs/pbs').glob('*/payload.json'):
        data = json.loads(path.read_text())
        if data.get('name') != name or data.get('dry_run'):
            continue
        receipt = json.loads(path.with_name('submission.json').read_text())
        if (re.findall(r'\b\d+\.hopper-m-\d+\b', receipt.get('stdout', '')) != [job]
                or data.get('budget_group') != 'G01' or data.get('task_id') != 'T01'
                or data.get('budget_batch') != 't00b_t01_20261108' or data.get('queue') != 'autox'
                or data.get('requested_host') is not None or data.get('requested_gpu_count') != 1
                or type(data.get('requested_wall_seconds')) is not int
                or timeout > data['requested_wall_seconds'] - 30
                or python_arguments(data.get('command'), MODULE) != expected):
            raise ValueError('Collection differs from its exact budgeted unpinned one-GPU command')
        matches.append({'file': str(path.relative_to(repo)), 'sha256': sha_file(path),
                       'requested_wall_seconds': data['requested_wall_seconds'], 'budget_group': 'G01'})
    if len(matches) != 1:
        raise ValueError('Require one immutable collection reservation for this PBS job/name')
    return matches[0]


@contextmanager
def installed(*, scenes, concurrency, benchmark, proof, partition='train', explicit_partition=False):
    """Reuse the original scheduler; every binding is process-local and restored."""
    scenes = validate_selection(scenes, concurrency, partition)
    previous = {key: getattr(parent, key) for key in
                ('SCENES', 'CONCURRENCIES', 'PARTITION', 'submission_identity', 'write_result', 'finish_logged')}
    def reservation(repo, name, job, timeout):
        if timeout != runner_timeout(scenes, concurrency, partition):
            raise ValueError('Scheduler must retain its complete full-grant deadline')
        return submission_identity(repo, name, job, timeout + BENCHMARK_VALIDATION_SECONDS, scenes=scenes,
                                   concurrency=concurrency, benchmark=benchmark, partition=partition, explicit_partition=explicit_partition)
    def finish(running, sources):
        if proof['source'] not in sources:
            sources.append(copy.deepcopy(proof['source']))
        return previous['finish_logged'](running, sources)
    def write(name, metrics, config):
        metrics['kind'] = 'g01_parallel_collection'
        config.update(collection_version=VERSION,
                      g01_partition=partition, per_scene_class_cap=parent.policy.class_cap(partition),
                      pool_scope=SCOPE if partition == 'train' else 'fixed evaluation scenes only; no training or tuning; aggregate cap6 per scene/class; freeze manifest before evaluation',
                      benchmark_source=str(benchmark), benchmark_selection=copy.deepcopy(proof),
                      runner_timeout_seconds=config['timeout_seconds'],
                      timeout_seconds=required_timeout(scenes, concurrency, partition),
                      required_timeout_seconds=required_timeout(scenes, concurrency, partition),
                      benchmark_validation_timeout_seconds=BENCHMARK_VALIDATION_SECONDS)
        return previous['write_result'](name, metrics, config)
    parent.SCENES, parent.CONCURRENCIES, parent.PARTITION = scenes, (concurrency,), partition
    parent.submission_identity, parent.write_result, parent.finish_logged = reservation, write, finish
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(parent, key, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--partition', choices=('train', 'test'), default=None)
    parser.add_argument('--scenes', nargs='+', required=True)
    parser.add_argument('--concurrency', type=int, choices=BENCHMARK_CONCURRENCIES, required=True)
    parser.add_argument('--benchmark-source', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    args = parser.parse_args(argv)
    partition = args.partition or 'train'
    scenes = validate_selection(args.scenes, args.concurrency, partition)
    if (not re.fullmatch(r'gen_' + partition + r'_[A-Za-z0-9_]+', args.name)
            or args.timeout_seconds != required_timeout(scenes, args.concurrency, partition)):
        parser.error('Require the matching train/test name and exact full-scene-grant deadline')
    output = ROOT / 'results' / f'{args.name}.json'
    if output.exists() or output.is_symlink():
        raise FileExistsError('Existing results are preserved; choose a new collection name')
    try:
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise RuntimeError('Refuse to replace another active process deadline')
        original_alarm = signal.getsignal(signal.SIGALRM)
        def expired(*_):
            raise TimeoutError('Benchmark source authentication exceeded its declared execution cap')
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, BENCHMARK_VALIDATION_SECONDS)
        try:
            proof = benchmark_stage(args.benchmark_source, args.concurrency)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, original_alarm)
    except Exception as error:
        traceback.print_exc()
        metrics = {'kind': 'g01_parallel_collection', 'status': 'benchmark_validation_failed',
                   'acceptance_passed': False, 'exit_code': 1, 'cleanup_errors': [],
                   'error': f'{type(error).__name__}: {error}', 'scene_runs': [], 'started_shards': []}
        write_result(args.name, metrics, {**vars(args), 'benchmark_source': str(args.benchmark_source),
                                         'collection_version': VERSION, 'pool_scope': SCOPE})
        return 1
    original_argv = sys.argv
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with installed(scenes=scenes, concurrency=args.concurrency, benchmark=args.benchmark_source, proof=proof,
                       partition=partition, explicit_partition=args.partition is not None):
            sys.argv = [MODULE, '--name', args.name, '--timeout-seconds', str(runner_timeout(scenes, args.concurrency, partition))]
            code = parent.main()
    finally:
        sys.argv = original_argv
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    # Preserve the benchmark reference even for a pre-gate orchestration error.
    # Metrics/config already carried their truthful collection scope at emission.
    result = json.loads(output.read_text())
    if proof['source'] not in result.setdefault('sources', []):
        result['sources'].append(proof['source'])
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
