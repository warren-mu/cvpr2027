"""Conservative subbudget reservations and ended shared-GPU allocation splits.

This is scheduler-cost attribution only. No scientific result or cache permission
is inferred from a unit's lifecycle, and an unknown lifecycle never frees budget.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex

VERSION = 'shared_gpu_units_budget_v1'
PRODUCER = 'src.tools.gpu_pack'
ACCOUNT_PRODUCER = 'src.tools.pbs_mixed_accounting'
PRIORITY = ('G01', 'T02', 'E2Epilot')
RELEASE_FIELDS = ('unit_id', 'name', 'budget_group', 'pbs_jobid', 'gpu_uuid', 'cpu_affinity',
                  'claim_reference', 'started_epoch', 'finished_epoch', 'started_monotonic',
                  'finished_monotonic', 'duration_seconds_monotonic', 'result_reference',
                  'exit_code', 'cleanup_errors')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def reference(path, repo):
    path = Path(path).resolve()
    body = path.read_bytes()
    try:
        filename = str(path.relative_to(repo))
    except ValueError:
        filename = str(path)
    return {'file': filename, 'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)}


def read_reference(repo, ref, allowed_root=None):
    require(isinstance(ref, dict) and isinstance(ref.get('file'), str)
            and re.fullmatch(r'[0-9a-f]{64}', ref.get('sha256', '')), 'Exact original file/SHA reference required')
    path = (repo / ref['file']).resolve()
    path.relative_to((allowed_root or repo).resolve())
    body = path.read_bytes()
    require(hashlib.sha256(body).hexdigest() == ref['sha256'], 'Original mixed-pack evidence changed')
    if 'bytes' in ref:
        require(type(ref['bytes']) is int and ref['bytes'] == len(body), 'Original mixed-pack evidence size changed')
    return json.loads(body), path


def logged(repo, ref):
    data, path = read_reference(repo, ref, repo / 'results')
    spec = importlib.util.spec_from_file_location('mixed_original_provenance', repo / 'scripts/audit_provenance.py')
    auditor = importlib.util.module_from_spec(spec); spec.loader.exec_module(auditor)
    require(not auditor.audit_file(str(path))['fail'], 'Mixed-pack source provenance audit failed')
    meta = data['_meta']
    require(isinstance(meta.get('end'), str) and bool(meta['end']), 'Mixed-pack source is not ended')
    require(meta.get('metrics_sha256') == hashlib.sha256(json.dumps(data['metrics'], sort_keys=True).encode()).hexdigest(),
            'Mixed-pack metrics hash differs')
    log = (repo / meta['log_path']).resolve(); log.relative_to(repo / 'logs/runs')
    finals = [line.partition(':')[2].strip() for line in log.read_text().splitlines() if line.startswith('FINAL_METRICS_JSON:')]
    require(finals and json.loads(finals[-1]) == data['metrics'], 'Mixed-pack final metrics are absent from the original log')
    return data, path


def option(command, flag):
    require(command.count(flag) == 1 and command.index(flag) + 1 < len(command), 'Missing/ambiguous pack option: ' + flag)
    return command[command.index(flag) + 1]


def pack_command(command):
    require(isinstance(command, list) and all(isinstance(arg, str) for arg in command), 'Pack command must be an argv list')
    require(command[:4] == ['python3', '-u', '-m', PRODUCER] or command[:3] == ['python3', '-m', PRODUCER],
            'PACK budget requires the explicit shared-GPU producer')
    require(command.count('-m') == 1, 'Ambiguous shared-GPU producer')


def validate_submission(repo, command, name, gpus):
    """Called under the original submit lock, and re-read for budget accounting."""
    repo = Path(repo).resolve(); pack_command(command)
    require(re.fullmatch(r'gpu_pack_[A-Za-z0-9_]+', name) and type(gpus) is int and gpus in (1, 4),
            'Shared queue allocations require an explicit gpu_pack name and one or four GPUs')
    require(option(command, '--name') == name and option(command, '--gpus') == str(gpus), 'Pack CLI/allocation identity differs')
    from src.tools.gpu_shared_claim import load_plan
    plan, ref = load_plan(repo / option(command, '--queue-plan'), option(command, '--queue-plan-sha256'), root=repo)
    groups = [group for group in PRIORITY if any(unit['budget_group'] == group for unit in plan['units'])]
    require(groups and set(unit['budget_group'] for unit in plan['units']) == set(groups), 'Pack plan contains an unknown budget group')
    queue_root = Path(option(command, '--queue-root'))
    require(queue_root.is_absolute() and '..' not in queue_root.parts, 'Pack shared queue root must be an explicit absolute path')
    return {'version': VERSION, 'plan_reference': ref, 'possible_groups': groups,
            'residual_priority': groups, 'reservation_rule': 'entire_allocation_upper_bound_in_each_possible_subbudget',
            'total_rule': 'one_allocation_once_in_T01_and_shared',
            'residual_rule': 'highest_priority_started_group_or_highest_planned_group_if_zero_business'}


def validate_payload(repo, payload):
    require(payload.get('task_id') == 'T01' and payload.get('budget_group') == 'PACK', 'Not an explicit PACK payload')
    actual = validate_submission(repo, payload['command'], payload['name'], payload['requested_gpu_count'])
    require(payload.get('mixed_budget_authorization') == actual, 'Mixed-pack payload differs from its original budget authorization')
    return actual


def apportion(binding, unit_runs, allocated_uuids, allocation_gpu_hours, rate):
    """All unit lifecycles are already authenticated; monotonic intervals own cards."""
    groups = binding['possible_groups']
    require(groups == [group for group in PRIORITY if group in groups] and groups, 'Unknown or unordered pack budget groups')
    require(binding['residual_priority'] == groups, 'Residual priority differs from pre-submission policy')
    require(len(allocated_uuids) == len(set(allocated_uuids)) and allocated_uuids, 'Allocation GPU UUIDs are ambiguous')
    require(finite(allocation_gpu_hours) and allocation_gpu_hours >= 0 and finite(rate) and rate > 0, 'Missing actual PBS cost/rate')
    intervals = defaultdict(list); seconds = defaultdict(list); started = set()
    for row in unit_runs:
        require(row['budget_group'] in groups and row['gpu_uuid'] in allocated_uuids, 'Unit is outside its reserved group/GPU domain')
        start, finish = row.get('started_monotonic'), row.get('finished_monotonic')
        require(finite(start) and finite(finish) and 0 <= start <= finish and row.get('cleanup_errors') == [],
                'Unknown/incomplete unit cleanup cannot release a mixed reservation')
        require(row.get('duration_seconds_monotonic') == finish - start, 'Unit duration differs from its actual recorded interval')
        intervals[row['gpu_uuid']].append((start, finish))
        seconds[row['budget_group']].append(Fraction(str(finish)) - Fraction(str(start)))
        started.add(row['budget_group'])
    for spans in intervals.values():
        ordered = sorted(spans)
        require(all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:])), 'Unit intervals overlap on the same physical GPU')
    if unit_runs:
        span = max(row['finished_monotonic'] for row in unit_runs) - min(row['started_monotonic'] for row in unit_runs)
        require(span <= allocation_gpu_hours * 3600 / len(allocated_uuids), 'Unit timeline exceeds the original PBS walltime')
    total_hours = Fraction(str(allocation_gpu_hours))
    # Exact rational arithmetic preserves the original canonical float total
    # without introducing a fee tolerance or rounding one group independently.
    total_charge = Fraction(str(allocation_gpu_hours * rate))
    hours = {group: sum(seconds[group], Fraction(0)) / 3600 for group in groups}
    occupied = sum(hours.values(), Fraction(0))
    require(occupied <= total_hours, 'Measured unit GPU time exceeds the original PBS allocation')
    owner = next(group for group in groups if group in started) if started else groups[0]
    unit_charge = {group: value * Fraction(str(rate)) for group, value in hours.items()}
    costs = dict(unit_charge)
    costs[owner] = total_charge - sum((costs[group] for group in groups if group != owner), Fraction(0))
    require(costs[owner] >= unit_charge[owner] and sum(costs.values(), Fraction(0)) == total_charge,
            'Mixed allocation cost does not conserve the original PBS total')
    exact = lambda value: {'numerator': str(value.numerator), 'denominator': str(value.denominator)}
    return {'version': VERSION, 'possible_groups': groups, 'residual_group': owner, 'zero_business': not unit_runs,
        'observed_unit_gpu_seconds_by_group': {g: float(sum(seconds[g], Fraction(0))) for g in groups},
        'allocation_gpu_hours': allocation_gpu_hours, 'queue_charge_rate': rate,
        'total_charged_gpu_hours': allocation_gpu_hours * rate,
        'charged_gpu_hours_by_group': {g: float(costs[g]) for g in groups},
        'charged_gpu_hours_by_group_exact': {g: exact(costs[g]) for g in groups},
        'total_charged_gpu_hours_exact': exact(total_charge),
        'unassigned_allocation_gpu_hours': float(total_hours - occupied),
        'residual_charged_gpu_hours': float(costs[owner] - unit_charge[owner]),
        'scope': 'Scheduler cost attribution only; gate, idle, cleanup and timing remainder follow the predeclared priority, not scientific success. Exact fractions conserve the original canonical total; floats are display/budget values.'}



def compute_split(repo, payload_ref, pack_ref, account_ref):
    """Recompute attribution from unchanged original lifecycle and PBS evidence."""
    repo = Path(repo).resolve()
    payload, payload_path = read_reference(repo, payload_ref, repo / 'logs/pbs')
    binding = validate_payload(repo, payload)
    require(payload.get('dry_run') is False and payload_path.stat().st_mode & 0o222 == 0,
            'Mixed accounting requires an original immutable submitted payload')
    receipt_path = payload_path.with_name('submission.json')
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get('returncode') == 0 and not receipt.get('stderr')
            and re.fullmatch(r'\d+\.hopper-m-\d+\s*', receipt.get('stdout', '')), 'Ambiguous original mixed submission receipt')
    job = receipt['stdout'].strip()
    pack, pack_path = logged(repo, pack_ref)
    require(pack_path == repo / 'results' / (payload['name'] + '.json')
            and pack['_meta']['command'] == payload['command'], 'Mixed accounting names another pack producer')
    cfg, metrics = pack['config'], pack['metrics']
    require(cfg.get('pbs_jobid') == job and cfg.get('plan_reference') == binding['plan_reference']
            and metrics.get('kind') == 'gpu_shared_pack' and metrics.get('all_owned_units_ended') is True
            and metrics.get('cleanup_errors') == [], 'Pack lifecycle/identity is incomplete')
    account, account_path = logged(repo, account_ref)
    spec = importlib.util.spec_from_file_location('mixed_original_submit_phase', repo / 'scripts/pbs/submit_phase.py')
    phase = importlib.util.module_from_spec(spec); spec.loader.exec_module(phase)
    observed = phase._final_accounting(repo, {job}, paths=[account_path]).get(job)
    require(observed is not None, 'Original canonical PBS allocation/rate is unavailable')
    rows = [row for row in account['config']['jobs'] if row.get('job_id') == job and row.get('qstat_exit') == 0]
    require(len(rows) == 1, 'Ambiguous original PBS job')
    fields = rows[0]['fields']
    require(fields.get('project') == 'CFP04-SF-125' and fields.get('Job_Owner', '').split('@')[0] == 'e1583364'
            and fields.get('Resource_List.ngpus') == str(payload['requested_gpu_count'])
            and fields.get('Resource_List.ncpus') == str(12 * payload['requested_gpu_count'])
            and fields.get('queue') == ('smallx' if payload['requested_gpu_count'] == 1 else 'mediumx')
            and 'PBS_O_QUEUE=autox' in fields.get('Variable_List', '')
            and receipt.get('command', [None])[0] == 'qsub'
            and shlex.split(fields.get('Submit_arguments', '')) == receipt['command'][1:]
            and fields.get('Output_Path', '').split(':', 1)[-1] == str(payload_path.with_name('pbs.out'))
            and fields.get('job_state') == 'F' and fields.get('run_count') == '1', 'PBS allocation differs from the submitted pack')
    nodes = {item.split('/')[0] for item in fields.get('exec_host', '').split('+')}
    require(nodes == {cfg.get('node')} and pack['_meta'].get('host') == cfg.get('node'), 'Pack unit timeline comes from another node')
    uuids = cfg.get('allocated_gpu_uuids')
    require(isinstance(uuids, list) and len(uuids) == payload['requested_gpu_count']
            and all(isinstance(gpu, str) and re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', gpu) for gpu in uuids),
            'Original physical GPU mask is missing')
    from src.tools.gpu_shared_claim import load_plan
    plan, _ = load_plan(repo / binding['plan_reference']['file'], binding['plan_reference']['sha256'], root=repo)
    units = {unit['unit_id']: unit for unit in plan['units']}
    runs = metrics.get('unit_runs')
    require(isinstance(runs, list), 'Original pack unit ledger is missing')
    queue_root = Path(option(payload['command'], '--queue-root')).resolve()
    seen = set()
    sources = [{**pack_ref, 'metrics_sha256': pack['_meta']['metrics_sha256']},
               {**account_ref, 'metrics_sha256': account['_meta']['metrics_sha256']}]
    artifacts = [payload_ref, reference(receipt_path, repo)]
    for row in runs:
        uid = row.get('unit_id')
        require(uid in units and uid not in seen, 'Unknown/repeated unit lifecycle')
        seen.add(uid); unit = units[uid]
        require(row.get('name') == unit['name'] and row.get('budget_group') == unit['budget_group']
                and row.get('pbs_jobid') == job and row.get('cleanup_errors') == [], 'Unit identity/cleanup differs')
        require(isinstance(row.get('cpu_affinity'), list) and len(row['cpu_affinity']) == 12
                and len(set(row['cpu_affinity'])) == 12 and all(type(cpu) is int and cpu >= 0 for cpu in row['cpu_affinity']),
                'Unit does not have its original twelve CPU cores')
        release, release_path = read_reference(repo, row.get('release_reference'), queue_root)
        require(release_path == queue_root / 'releases' / (uid + '.json') and release_path.stat().st_mode & 0o222 == 0,
                'Unit release is not its original immutable queue record')
        require(all(key in release and release[key] == row.get(key) for key in RELEASE_FIELDS), 'Unit end ledger differs from immutable release evidence')
        claim, claim_path = read_reference(repo, row.get('claim_reference'), queue_root)
        require(claim_path == queue_root / 'claims' / (uid + '.json') and claim_path.stat().st_mode & 0o222 == 0,
                'Unit claim is not its original immutable queue record')
        for key in ('unit_id', 'name', 'budget_group', 'pbs_jobid', 'gpu_uuid', 'cpu_affinity', 'started_epoch', 'started_monotonic'):
            require(claim.get(key) == row.get(key), 'Unit start ledger differs from immutable claim evidence: ' + key)
        require(claim.get('plan_reference') == binding['plan_reference'] and claim.get('command') == unit['command'],
                'Unit claim changed its original plan/command')
        _, claim_payload_path = read_reference(repo, claim.get('parent_payload_reference'), repo / 'logs/pbs')
        _, claim_receipt_path = read_reference(repo, claim.get('parent_receipt_reference'), repo / 'logs/pbs')
        require(claim_payload_path == payload_path and claim_receipt_path == receipt_path
                and claim.get('node') == cfg['node']
                and claim.get('parent_cuda_visible_devices', '').split(',') == uuids,
                'Unit claim is not bound to this original allocation/receipt/GPU mask')
        require(finite(row.get('started_epoch')) and finite(row.get('finished_epoch')),
                'Original unit wall-clock display times are absent')
        if row.get('result_reference') is not None:
            child, child_path = logged(repo, row['result_reference'])
            require(child_path == repo / 'results' / (unit['name'] + '.json')
                    and child['_meta'].get('command') == unit['command']
                    and child['_meta'].get('exit_code') == row['exit_code']
                    and child['config'].get('pbs_jobid') == job, 'Unit ended result belongs to another command/job')
            sources.append({**row['result_reference'], 'metrics_sha256': child['_meta']['metrics_sha256']})
        else:
            require(isinstance(release.get('launch_error'), str) and bool(release['launch_error'])
                    and release.get('process_started') is False, 'A missing unit result is not a substantiated launch failure')
        artifacts.extend((row['claim_reference'], row['release_reference']))
    # The immutable queue has one stable claim path per planned unit. Check its
    # complete ownership set so a failed/unfinished owned claim cannot disappear
    # from the parent unit list and be reclassified as harmless idle overhead.
    owned = set()
    for uid in units:
        claim_path = queue_root / 'claims' / (uid + '.json')
        if not claim_path.is_file():
            continue
        claim, _ = read_reference(repo, reference(claim_path, repo), queue_root)
        if claim.get('pbs_jobid') == job:
            require(claim.get('unit_id') == uid, 'Stable claim path/unit identity differs')
            _, owner_path = read_reference(repo, claim.get('parent_payload_reference'), repo / 'logs/pbs')
            require(owner_path == payload_path, 'Same PBS job has conflicting parent payload ownership')
            owned.add(uid)
    require(owned == seen, 'Parent unit ledger omits or invents a claimed allocation interval')
    allocation = apportion(binding, runs, uuids, observed['gpu_hours'], observed['queue_charge_rate'])
    allocation.update(job_id=job, pack_name=payload['name'], unit_count=len(runs), sources=sources,
                      artifact_references=artifacts,
                      canonical_accounting_result=observed['accounting_result'])
    return allocation


def finalized_split(repo, row, canonical):
    """Missing proof retains reservations; present invalid proof stops accounting."""
    name = row['name'] + '_mixed_accounting'
    path = repo / 'results' / (name + '.json')
    if not path.is_file():
        return None
    report, _ = logged(repo, reference(path, repo))
    require(report['_meta']['exit_code'] == 0 and report['_meta']['name'] == name
            and report['config'].get('name') == name, 'Mixed final accounting did not complete successfully')
    config = report['config']
    expected = ['python3', '-m', ACCOUNT_PRODUCER, '--name', name,
                '--payload', config['payload_reference']['file'], '--payload-sha256', config['payload_reference']['sha256'],
                '--pack-result', config['pack_reference']['file'], '--pack-sha256', config['pack_reference']['sha256'],
                '--accounting-result', config['account_reference']['file'], '--accounting-sha256', config['account_reference']['sha256']]
    require(report['_meta']['command'] == expected, 'Unknown mixed accounting producer/arguments')
    require(config['payload_reference']['file'] == row['payload'], 'Mixed cost report binds another original payload')
    actual = compute_split(repo, config['payload_reference'], config['pack_reference'], config['account_reference'])
    require(report['metrics'] == actual and actual['job_id'] == row['job_id']
            and actual['allocation_gpu_hours'] == canonical['gpu_hours']
            and actual['queue_charge_rate'] == canonical['queue_charge_rate']
            and actual['total_charged_gpu_hours'] == canonical['gpu_hours'] * canonical['queue_charge_rate'],
            'Original mixed allocation no longer matches its final accounting')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    for role in ('payload', 'pack', 'accounting'):
        parser.add_argument('--' + ({'pack': 'pack-result', 'accounting': 'accounting-result'}.get(role, role)), dest=role, required=True)
        parser.add_argument('--' + role + '-sha256', required=True)
    args = parser.parse_args(); repo = Path(__file__).resolve().parents[2]
    refs = [{'file': getattr(args, role), 'sha256': getattr(args, role + '_sha256')} for role in ('payload', 'pack', 'accounting')]
    metrics = compute_split(repo, *refs)
    require(args.name == metrics['pack_name'] + '_mixed_accounting', 'Use the unique original pack cost-report name')
    config = {'name': args.name, 'version': VERSION, 'payload_reference': refs[0], 'pack_reference': refs[1], 'account_reference': refs[2]}
    path = repo / 'results' / (args.name + '.json')
    with path.open('x') as stream:
        json.dump({'config': config, 'metrics': metrics, 'sources': metrics['sources']}, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
