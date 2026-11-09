"""One-off PBS preparation for approved G01 test round2; never submits jobs.

Run the default stage through run_logged on one allocated GPU, walltime720s.
The unchanged render gate runs before any inventory metadata scan. Independent
child runs retain the quota receipt inventory and original A/B rejected funnel.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.tools import g01_parallel as runners
from src.tools import g01_collect_parallel as collection
from src.tools import g01_quota_plan as quota
from src.tools import merge_pair_v2 as merge
from src.envs import pair_v2_g01_policy as policy
from src.envs import pair_v2_g01_search as search
from src.envs.thor_common import write_result
from src.tools.t01_parallel_approved import finish_logged, stop_logged

SCENES = tuple(f'FloorPlan{base + n}' for n in range(21, 31) for base in (0, 200, 300, 400))
CUTOFF = ('results/t01v2_checkpoint_test_a_full_01.json', 'results/t01v2_checkpoint_test_b_full_01.json')
COLLECTION_NAME = 'gen_test_config_round2_40_01'
BENCHMARK = 'results/gen_train_concurrency_s15_02.json'
BOUNDS = {'gate': 120, 'inventory': 180, 'funnel': 300, 'parent': 690, 'pbs_wall': 720}
SCRIPT = 'experiments/g01_test_round2_prepare.py'
SCOPE = 'Preparation and construction-receipt counting only; no native scientific revalidation, T02 acceptance, training authorization, or job submission.'


def file_ref(path):
    path = Path(path)
    return {'file': str(path.relative_to(ROOT)), 'sha256': merge.sha_file(path)}


def write_new(path, value, *, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    if immutable:
        path.chmod(0o444)
    return file_ref(path)


def round_robin_plan(proof, name):
    """Fixed scene round robin; global residual tickets are never multiplied."""
    if (set(SCENES) != policy.TEST_SCENES or len(SCENES) != len(set(SCENES))
            or set(SCENES) & policy.TRAIN_SCENES):
        raise ValueError('Require exactly the unchanged forty isolated test scenes')
    remaining = {kind: max(0, 60 - proof['confirmed_by_kind'][kind]) for kind in quota.KINDS}
    assigned = {scene: {kind: 0 for kind in quota.KINDS} for scene in SCENES}
    for kind in search.CONFIG_PRIORITY:
        while remaining[kind]:
            progressed = False
            for scene in SCENES:
                old = proof['confirmed_by_scene'].get(scene, {}).get(kind, 0)
                if assigned[scene][kind] < max(0, 6 - old):
                    assigned[scene][kind] += 1
                    remaining[kind] -= 1
                    progressed = True
                    if not remaining[kind]:
                        break
            if not progressed:
                break
    if any(remaining.values()) or any(not any(value.values()) for value in assigned.values()):
        raise ValueError('Remaining global quota cannot cover all forty scenes with positive bounded tickets')
    plan = {'version': quota.CONFIG_VERSION, 'partition': 'test', 'parent_run_name': COLLECTION_NAME,
        'targets': {kind: 60 for kind in quota.KINDS}, 'count_stage': quota.STAGES['test'],
        'count_basis_authority': quota.RECEIPT_AUTHORITY,
        'count_basis_sources': [{key: item[key] for key in ('file', 'sha256')} for item in proof['checkpoint_references']],
        'confirmed_by_kind': proof['confirmed_by_kind'], 'confirmed_by_scene': proof['confirmed_by_scene'],
        'confirmed_configurations': proof['configuration_selection']['seen_configurations'],
        'outstanding_reservations': [],
        'assignments': [{'scene': scene, 'allowed_kinds': [kind for kind in quota.KINDS if assigned[scene][kind]],
                         'actual_output_caps': assigned[scene]} for scene in SCENES],
        'preparation_run_name': name, 'approved_task_commit': '002d747c', 'seed': 0,
        'allocation_rule': 'Fixed interleaved scene order, one ticket per available scene per round, separately in LS/LF/DF/DS priority; remaining global60 and scene6 use authenticated configuration occupancy.',
        'field_of_view': 60, 'scene_concurrency': 6, 'cpu_cores_per_gpu': 12, 'max_unity_instances': 8,
        'training_authorized': False, 'scope': SCOPE}
    arithmetic = quota.validate_structure(plan)
    return plan, arithmetic


def inventory_stage(name):
    references = [file_ref(ROOT / path) for path in CUTOFF]
    started = time.monotonic()
    proof = quota.reconstruct_test_inventory(references, configuration=True, authority=quota.RECEIPT_AUTHORITY)
    elapsed = time.monotonic() - started
    plan, arithmetic = round_robin_plan(proof, name)
    directory = ROOT / 'logs/review' / name
    inventory = write_new(directory / 'inventory.json', proof)
    plan_ref = write_new(ROOT / 'configs' / (name + '_quota.json'), plan, immutable=True)
    timeout = collection.required_timeout(SCENES, 6, 'test')
    wall = timeout + 30
    walltime = f'{wall // 3600:02d}:{wall % 3600 // 60:02d}:{wall % 60:02d}'
    command = ['python3', '-u', '-m', collection.MODULE, '--name', COLLECTION_NAME,
        '--scenes', *SCENES, '--concurrency', '6', '--benchmark-source', BENCHMARK,
        '--timeout-seconds', str(timeout), '--partition', 'test',
        '--quota-plan', plan_ref['file'], '--quota-plan-sha256', plan_ref['sha256']]
    submission = ['python3', 'scripts/pbs/submit_phase.py', '--repo', str(ROOT),
        '--scratch', os.environ['CVPR_SCRATCH'], '--name', COLLECTION_NAME, '--task', 'T01',
        '--gpus', '1', '--walltime', walltime, '--budget-group', 'G01', '--', *command]
    return {'status': 'inventory_and_plan_ready', 'inventory': inventory, 'plan': plan_ref,
        'receipt_inventory_seconds_observed': elapsed,
        'raw_unique_by_class': proof['raw_unique_proposals_by_kind'], 'configuration_by_class': proof['confirmed_by_kind'],
        'configuration_scenes_by_class': {kind: sum(row[kind] > 0 for row in proof['confirmed_by_scene'].values()) for kind in quota.KINDS},
        'raw_occurrences_retained': True, 'quota_arithmetic': arithmetic,
        'producer_command': command, 'submission_command_for_root_only': submission,
        'required_parent_timeout_seconds': timeout, 'requested_pbs_wall_seconds': wall,
        'requested_pbs_walltime': walltime, 'requested_gpu_hours_upper_bound': wall / 3600,
        'charge_scope': 'one-GPU wall request, not measured usage; original submitter must check actual-rate budget at submission',
        'benchmark_reference': file_ref(ROOT / BENCHMARK), 'benchmark_revalidated_in_this_stage': False,
        '_input_sources': proof['checkpoint_references'], 'scope': SCOPE}


def funnel_stage(name):
    import collections
    import datetime
    from src.envs import pair_table_source as sources
    from src.tools.t01_partition import audited_result
    from src.tools import t01v2_look_funnel as funnel
    NAME = name
    CUTOFF = [('A', 'results/t01v2_checkpoint_test_a_full_01.json'), ('B', 'results/t01v2_checkpoint_test_b_full_01.json')]
    START = time.monotonic()
    MAX_SECONDS = 300  # engineering bound only; incomplete scans fail explicitly
    ORDER = ('physical_both', 'hidden_both', 'prefix_le3', 'first_evidence_ge250', 'solvable_both', 'witness_both')
    report = {'cutoff': [], 'sources': [], 'groups': [], 'errors': [], 'scope':
        'Source-recorded attempts, including rejection and actual-class crossover. Not a new scientific acceptance run.',
        'unknown_policy': 'Unmeasured gates stay unknown. The six requested gates omit target/selection/restore/storage checks. Physics executes first; first-action evidence and own-world solvability share one action search, so their causal ordering is not recoverable.',
        'source_artifacts_modified': False, 'thresholds_changed': False}
    rows = collections.defaultdict(list)

    def deadline():
        if time.monotonic() - START >= MAX_SECONDS:
            raise TimeoutError('Metadata diagnostic hit its explicit CPU/I/O wall bound; no missing rows imputed')

    def bound(a, b):
        return all(a.get(key) == b.get(key) and a.get(key) is not None
                   for key in ('file', 'sha256', 'metrics_sha256', 'git_commit'))

    try:
        seen_sources = set()
        for batch, path in CUTOFF:
            deadline()
            checkpoint, checkpoint_ref, _ = sources._container(merge.ROOT / path)
            if sources._checkpoint_module(checkpoint['_meta']['command']) != sources.G01_CHECKPOINT_MODULE:
                raise ValueError('Wrong independent checkpoint producer')
            report['cutoff'].append({'batch': batch, 'checkpoint': checkpoint_ref})
            runs = {item['source']['file']: item for item in checkpoint['metrics']['source_runs']}
            refs = checkpoint['sources']
            if len(runs) != len(refs) or set(runs) != {ref['file'] for ref in refs}:
                raise ValueError('Checkpoint complete source ledger mismatch')
            for ref in refs:
                deadline()
                if ref['file'] in seen_sources:
                    raise ValueError('Source repeated across original A/B cutoff')
                seen_sources.add(ref['file'])
                data, actual = audited_result(merge.ROOT / ref['file'])
                cfg, metrics = data['config'], data['metrics']
                run = runs[ref['file']]
                if (not bound(ref, actual) or not bound(run['source'], actual)
                        or not data['_meta'].get('end') or cfg.get('g01_partition') != 'test'):
                    raise ValueError('Original source receipt changed or not ended explicit test')
                journal_ref = run.get('attempt_journal')
                unit = {'batch': batch, 'source': actual, 'scene': cfg['scene'], 'journal': journal_ref,
                        'reported_by_proposed_pool_mode': (metrics.get('g01_collection') or {}).get('counted_by_proposed_pool_mode'),
                        'observed_by_proposed_pool_mode': None, 'status': metrics.get('status'),
                        'scene_excluded': metrics.get('scene_excluded'), 'scan_complete': False}
                report['sources'].append(unit)
                if journal_ref is None:
                    unit['unknown_reason'] = 'Original checkpoint has no observed complete journal'
                    continue
                journal_path = merge.reference_path(journal_ref, within=merge.ROOT / 'data/world_pairs/v2')
                raw_bytes = journal_path.read_bytes()
                if raw_bytes and not raw_bytes.endswith(b'\n'):
                    raise ValueError('Complete journal has an unfinished line')
                counts = {kind: 0 for kind in merge.KINDS}
                observed = 0
                for line in raw_bytes.decode('utf-8').splitlines():
                    if not line.strip():
                        continue
                    deadline()
                    journal = json.loads(line)
                    raw = merge.read_reference(journal['raw_attempt'], within=cfg['raw_evidence_dir'])
                    if (raw.get('counts_as_pair_attempt') is not True
                            or any(raw.get(key) != journal.get(key) for key in
                                   ('pair_id', 'scene', 'kind', 'status', 'discard_reasons', 'proposed_pool', 'pair_file', 'pair_sha256'))):
                        raise ValueError('Original raw attempt differs from its authenticated counted journal row')
                    key = f"{raw.get('proposed_pool')}_{raw.get('mode')}"
                    if key not in counts:
                        raise ValueError('Actual proposal class is not recorded; cannot invent mode from final kind')
                    counts[key] += 1
                    observed += 1
                    if key in ('look_stay', 'direct_flip'):
                        gates, values = funnel.measurements(raw)
                        # Keep original measured fields needed by the existing summarizer,
                        # never full token coordinate lists/portable native payloads.
                        row = {field: raw.get(field) for field in ('pair_id', 'scene', 'kind', 'status', 'discard_reasons')}
                        row.update(source_file=actual['file'], raw_attempt=journal['raw_attempt'],
                            prefix_measurement={'prefix_edit_mask_pixels': values['prefix_anchor_pixels'],
                                'prefix_max_abs_diff': values['prefix_max_abs_diff'],
                                'candidates_visible_and_marked': values['candidate_prefix_visibility']},
                            look_solvability={'complete': values['catalog_complete'],
                                'solvable_w0': values['solvable_by_world'][0], 'solvable_w1': values['solvable_by_world'][1],
                                'actions': values['first_actions']},
                            physical_w0=values['physical_by_world']['physical_w0'],
                            physical_w1=values['physical_by_world']['physical_w1'], witnesses=values['witnesses'])
                        if funnel.measurements(row)[0] != gates:
                            raise ValueError('Diagnostic compact view altered an original gate measurement')
                        rows[(batch, key)].append({'row': row, 'gates': gates, 'values': values})
                if observed != run.get('verified_attempts') or observed != metrics.get('attempts'):
                    raise ValueError('Journal count differs from original checkpoint/source terminal count')
                if counts != unit['reported_by_proposed_pool_mode']:
                    raise ValueError('Actual raw proposal modes differ from original scene collection summary')
                unit.update(observed_by_proposed_pool_mode=counts, scan_complete=True)
                print('FUNNEL_SOURCE_SCANNED: ' + json.dumps({'source': actual['file'], 'complete': True}), flush=True)
    except Exception as error:
        report['errors'].append({'error': f'{type(error).__name__}: {error}'})

    for batch, _ in CUTOFF:
        for kind in ('look_stay', 'direct_flip'):
            chosen = rows[(batch, kind)]
            raw = [item['row'] for item in chosen]
            outcomes, reasons, first_failure, uncertain = [collections.Counter() for _ in range(4)]
            examples = {}
            for item in chosen:
                row, gates = item['row'], item['gates']
                reasons.update(row.get('discard_reasons') or [])
                if row.get('status') == 'passed':
                    outcomes['passed_expected_class' if row.get('kind') == kind else 'passed_cross_over_to_' + str(row.get('kind'))] += 1
                    continue
                outcomes['rejected'] += 1
                order = ORDER if kind == 'look_stay' else ('physical_both',)
                failed = next((gate for gate in order if gates[gate] is False), None)
                if failed is None:
                    bucket = 'no_observed_failure_in_reported_gates'
                elif any(gates[gate] is None for gate in order[:order.index(failed)]):
                    bucket = 'failure_with_earlier_unknown_' + failed
                    uncertain[failed] += 1
                else:
                    bucket = 'first_observed_failure_' + failed
                first_failure[bucket] += 1
                if len(examples.setdefault(bucket, [])) < 2:
                    examples[bucket].append({'pair_id': row['pair_id'], 'source_file': row['source_file'],
                        'raw_attempt': row['raw_attempt'], 'discard_reasons': row.get('discard_reasons'),
                        'gates': gates, 'values': item['values']})
            denominator = len(chosen)
            group = {'batch': batch, 'actual_proposed_class': kind, 'observed_attempts': denominator,
                'outcomes': dict(outcomes), 'outcome_proportions_of_observed':
                    {key: value / denominator for key, value in outcomes.items()} if denominator else None,
                'recorded_discard_reasons': dict(reasons), 'first_observed_failure': dict(first_failure),
                'failure_proportions_of_observed': {key: value / denominator for key, value in first_failure.items()} if denominator else None,
                'examples': examples, 'not_causal_first_stop_claim': True,
                'scan_complete_for_batch': all(unit['scan_complete'] for unit in report['sources'] if unit['batch'] == batch)
                    and len([unit for unit in report['sources'] if unit['batch'] == batch]) == 12}
            if kind == 'look_stay':
                group['six_recorded_gates'] = funnel.summarize(raw)
                for cell in group['six_recorded_gates']['stages'].values():
                    cell['proportions_of_observed'] = {label: cell['gate_' + label] / denominator
                        for label in ('pass', 'fail', 'unknown')} if denominator else None
            else:
                group['physical_gate'] = {label: sum(item['gates']['physical_both'] is value for item in chosen)
                                         for label, value in (('pass', True), ('fail', False), ('unknown', None))}
                group['physical_gate_proportions_of_observed'] = {label: value / denominator
                    for label, value in group['physical_gate'].items()} if denominator else None
            report['groups'].append(group)
    report['elapsed_seconds'] = time.monotonic() - START
    report['observed_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['status'] = 'complete_recorded_diagnostic' if not report['errors'] and all(unit['scan_complete'] for unit in report['sources']) and all(sum(unit['batch'] == batch for unit in report['sources']) == 12 for batch, _ in CUTOFF) else 'incomplete_recorded_diagnostic'
    folder = merge.ROOT / 'logs/review' / NAME
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / 'funnel.json').open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    lines = ['只统计完整原始日志的真实提议类型；生成器通过不等于独立验收。物理检查先执行；六关之外的目标/选择/恢复/存储拒绝保留原原因。表中“首个失败”是按物理优先的展示顺序观测到的失败，不推断未记录关卡或证据与可解性的因果先后。跨域成功单列，未知不补零。', '',
             '| 批次 | 实际提议 | 关卡 | 已读尝试 | 扫描完整 | 通过 n(%) | 失败 n(%) | 未知 n(%) | 原结果 / 首个可观察失败 |',
             '|---|---|---|---:|---|---|---|---|---|']
    for group in report['groups']:
        denominator = group['observed_attempts']
        cells = ([(stage, group['six_recorded_gates']['stages'][stage]) for stage in funnel.STAGES]
                 if group['actual_proposed_class'] == 'look_stay' else
                 [('physical_both', {'gate_' + key: value for key, value in group['physical_gate'].items()})])
        for index, (stage, cell) in enumerate(cells):
            counts = [str(cell['gate_' + label]) +
                (f" ({100 * cell['gate_' + label] / denominator:.2f}%)" if denominator else ' (?)')
                for label in ('pass', 'fail', 'unknown')]
            detail = (json.dumps(group['outcomes'], sort_keys=True) + '<br>' +
                json.dumps(group['first_observed_failure'], sort_keys=True) + '<br>占已读尝试比例=' +
                json.dumps(group['failure_proportions_of_observed'], sort_keys=True)) if index == 0 else ''
            lines.append('| ' + ' | '.join([group['batch'], group['actual_proposed_class'], stage,
                str(denominator), str(group['scan_complete_for_batch']), *counts, detail]) + ' |')
    with (folder / 'report.md').open('x') as handle:
        handle.write('\n'.join(lines)+'\n')
    print('TEST_AB_FUNNEL_REPORT: ' + json.dumps({'status': report['status'], 'errors': report['errors'],
        'json': {'file': str((folder/'funnel.json').relative_to(merge.ROOT)), 'sha256': merge.sha_file(folder/'funnel.json')},
        'markdown': {'file': str((folder/'report.md').relative_to(merge.ROOT)), 'sha256': merge.sha_file(folder/'report.md')}}), flush=True)
    return {'status': report['status'], 'errors': report['errors'], 'report_json': file_ref(folder / 'funnel.json'), 'report_markdown': file_ref(folder / 'report.md'), 'elapsed_seconds': report['elapsed_seconds'], 'scope': report['scope'],
        '_input_sources': [item['checkpoint'] for item in report['cutoff']] + [unit['source'] for unit in report['sources']]}


def stage_main(name, stage):
    metrics = {'kind': 'g01_test_round2_' + stage, 'status': 'failed', 'scope': SCOPE}
    config = {'name': name, 'stage': stage, 'timeout_seconds': BOUNDS[stage],
              'pbs_jobid': os.environ.get('PBS_JOBID'), 'fixed_cutoff': list(CUTOFF)}
    previous = signal.getsignal(signal.SIGALRM)
    def expired(*_):
        raise TimeoutError('Explicit preparation stage deadline exceeded')
    code, input_sources = 1, []
    try:
        if not config['pbs_jobid'] or not os.environ.get('CVPR_SCRATCH'):
            raise RuntimeError('Preparation metadata scans are PBS-only')
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise RuntimeError('Refuse to replace an existing deadline')
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, BOUNDS[stage])
        values = inventory_stage(name) if stage == 'inventory' else funnel_stage(name)
        input_sources = values.pop('_input_sources')
        metrics.update(values)
        code = 0 if metrics['status'] in ('inventory_and_plan_ready', 'complete_recorded_diagnostic') else 1
    except BaseException as error:
        traceback.print_exc()
        metrics.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    metrics['exit_code'] = code
    write_result(name, metrics, config)
    output = ROOT / 'results' / (name + '.json')
    payload = json.loads(output.read_text()); payload['sources'] = input_sources
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return code


def prepare(name):
    metrics = {'kind': 'g01_test_round2_preparation', 'status': 'failed_before_gate',
               'gate': None, 'stages': [], 'cleanup_errors': [], 'black_frame_detected': False, 'scope': SCOPE}
    config = {'name': name, 'node': socket.gethostname(), 'pbs_jobid': os.environ.get('PBS_JOBID'),
        'bounds_seconds': BOUNDS, 'parent_cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'fixed_scenes': list(SCENES), 'fixed_cutoff': list(CUTOFF), 'gpus': 1}
    sources, running, stopping = [], None, []
    deadline = time.monotonic() + BOUNDS['parent']
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in old_handlers:
        signal.signal(sig, lambda number, frame: stopping.append(number))
    code = 1
    def run(child, command, seconds):
        nonlocal running
        running = runners.launch_logged(child, command, os.environ, scratch)
        until = min(time.monotonic() + seconds + 10, deadline - 30)
        while running['process'].poll() is None and not stopping and time.monotonic() < until:
            time.sleep(.1)
        if running['process'].poll() is None:
            grace = min(time.monotonic() + 10, deadline - 10)
            metrics['cleanup_errors'].extend(stop_logged(running['process'], grace_deadline=grace,
                                                        hard_deadline=min(grace + 8, deadline - 5)))
        record, data = finish_logged(running, sources)
        running = None
        if stopping or metrics['cleanup_errors'] or record.get('provenance_error'):
            raise RuntimeError('Stopped or unverified preparation child; preserve its failure evidence')
        return record, data
    try:
        if not config['pbs_jobid'] or not os.environ.get('CVPR_SCRATCH'):
            raise RuntimeError('Require actual PBS allocation; no login-node rendering or inventory scan')
        runners.validate_mask(config['parent_cuda_visible_devices'], 1)
        config['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        if len(config['cpu_affinity']) != 12:
            raise ValueError('Require the original twelve allocated CPUs')
        tag = re.sub(r'[^A-Za-z0-9]', '_', config['pbs_jobid'])
        scratch = Path(os.environ['CVPR_SCRATCH']) / 'g01_round2_prepare' / name / tag
        scratch.mkdir(parents=True, exist_ok=False)
        gate_name = name + '_gate'
        gate_record, gate = run(gate_name, [sys.executable, '-u', '-m', 'src.tools.render_gate',
            '--name', gate_name, '--gpus', '1', '--resolution', '255', '--segmentation',
            '--timeout-seconds', '120'], BOUNDS['gate'])
        metrics['gate'] = gate_record
        if gate_record['exit_code'] == 86:
            if gate['metrics'].get('cleanup_errors') != [] or gate['metrics'].get('exit_code') != 86:
                raise ValueError('Native black result lacks completed known cleanup')
            metrics.update(status='blocked_by_render_gate', black_frame_detected=True)
            code = 86
        elif gate_record['exit_code']:
            metrics['status'] = 'blocked_by_render_gate'
        else:
            runners.verify_gate(gate, gate_record, config, config['parent_cuda_visible_devices'])
            config['allocation'] = gate['config']['allocation']
            metrics['status'] = 'preparing'
            failed_stages = []
            for stage in ('inventory', 'funnel'):
                child = name + '_' + stage
                record, data = run(child, [sys.executable, '-u', SCRIPT,
                    '--name', child, '--stage', stage], BOUNDS[stage])
                metrics['stages'].append({'stage': stage, **record})
                if record['exit_code']:
                    failed_stages.append(stage)
                    continue  # independent metadata stages; no successful plan is inferred
                if stage == 'inventory':
                    metrics['inventory_plan'] = {key: copy.deepcopy(data['metrics'][key]) for key in
                        ('inventory', 'plan', 'producer_command', 'submission_command_for_root_only',
                         'required_parent_timeout_seconds', 'requested_pbs_wall_seconds', 'requested_pbs_walltime')}
            if failed_stages:
                metrics.update(status='preparation_stage_failed', failed_stages=failed_stages)
            else:
                metrics['status'], code = 'ready_for_root_review_and_budgeted_submission', 0
    except BaseException as error:
        traceback.print_exc()
        metrics['error'] = f'{type(error).__name__}: {error}'
        code = 1
    finally:
        if running is not None:
            metrics['cleanup_errors'].extend(stop_logged(running['process']))
            try:
                record, _ = finish_logged(running, sources)
                metrics['interrupted_child'] = record
            except Exception as error:
                metrics['cleanup_errors'].append(f'{type(error).__name__}: {error}')
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    if metrics['cleanup_errors']:
        code = 1
    metrics['exit_code'] = code
    write_result(name, metrics, config)
    output = ROOT / 'results' / (name + '.json')
    payload = json.loads(output.read_text()); payload['sources'] = sources
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--stage', choices=('prepare', 'inventory', 'funnel'), default='prepare')
    args = parser.parse_args()
    if re.fullmatch(r'gen_test_[A-Za-z0-9_]+', args.name) is None:
        parser.error('Require a unique gen_test_ preparation name')
    if (ROOT / 'results' / (args.name + '.json')).exists():
        raise FileExistsError('Existing preparation evidence is preserved')
    return prepare(args.name) if args.stage == 'prepare' else stage_main(args.name, args.stage)


if __name__ == '__main__':
    raise SystemExit(main())
