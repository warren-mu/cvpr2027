"""G01-specific shared-queue identity; original generation and checks are reused."""
import argparse
import copy
import hashlib
import json
import os
import socket
from pathlib import Path

from src.tools import g01_ticket_bank as bank
from src.tools import g01_quota_plan as quota

MODULE = 'src.tools.g01_collect_parallel'
ENV_KEYS = ('CVPR_SHARED_CLAIM_FILE', 'CVPR_SHARED_CLAIM_SHA256')


def arguments(command):
    from src.tools.g01_collect_parallel import python_arguments
    return python_arguments(command, MODULE)


def validate_unit(unit, root, *, external=False):
    from src.tools import g01_collect_parallel as collect
    if (unit.get('producer_module') != MODULE or unit.get('budget_group') != 'G01'
            or unit.get('gpus') != 1 or unit.get('cpu_cores') != 12):
        raise ValueError('G01 shared units retain one GPU, twelve CPUs and their original producer/group')
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument('--name', required=True)
    parser.add_argument('--scenes', nargs='+', required=True)
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--benchmark-source', required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    parser.add_argument('--partition', required=True)
    parser.add_argument('--quota-plan', required=True)
    parser.add_argument('--quota-plan-sha256', required=True)
    parser.add_argument('--nearby-physics', action='store_true')
    argv = arguments(unit.get('command'))
    try:
        args = parser.parse_args(argv)
    except (SystemExit, argparse.ArgumentError) as error:
        raise ValueError('Malformed shared G01 original command') from error
    expected = ['--name', args.name, '--scenes', *args.scenes, '--concurrency', '6',
        '--benchmark-source', args.benchmark_source, '--timeout-seconds', str(args.timeout_seconds),
        '--partition', 'test', '--quota-plan', args.quota_plan,
        '--quota-plan-sha256', args.quota_plan_sha256]
    if args.nearby_physics:
        expected.append('--nearby-physics')
    if (argv != expected or args.name != unit.get('name') or args.partition != 'test'
            or args.concurrency != 6 or args.timeout_seconds != collect.required_timeout(args.scenes, 6, 'test')
            or unit.get('timeout_seconds') != args.timeout_seconds):
        raise ValueError('Shared G01 command changed its scene order, full grants, name or test partition')
    refs = unit.get('references')
    if not isinstance(refs, dict):
        raise ValueError('Shared G01 needs original immutable quota/benchmark/bank references')
    quota_ref = {'file': args.quota_plan, 'sha256': args.quota_plan_sha256}
    if refs.get('quota_plan') != quota_ref:
        raise ValueError('Shared unit quota reference differs from its exact command')
    value = bank.read_reference(quota_ref, root)
    bank_ref = quota.reference(refs.get('g01_ticket_bank'))
    ticket_bank = bank.read_reference(bank_ref, root)
    base = bank.validate_bank(ticket_bank, root=root)
    expected_member = bank.expected_member(ticket_bank, bank_ref, refs.get('g01_ticket_slice'), root=root)
    if external:
        legacy = copy.deepcopy(base)
        legacy.update(parent_run_name=expected_member['parent_run_name'], assignments=expected_member['assignments'])
        if value != legacy:
            raise ValueError('External original single-GPU slice changed its original fixed tickets')
    elif value != expected_member:
        raise ValueError('Runnable shared unit lacks exact conservative other-slice reservations')
    if value['parent_run_name'] != args.name or [row['scene'] for row in value['assignments']] != args.scenes:
        raise ValueError('Unit command differs from its immutable slice assignment')
    benchmark = quota.reference(refs.get('benchmark_source'))
    if benchmark['file'] != args.benchmark_source:
        raise ValueError('Shared unit changed the original benchmark source path')
    bank.read_reference(benchmark, root)  # Actual audit/selection remains in the original parent.
    identity, keys = bank.scientific_identity(expected_member, root=root)
    identity.update(benchmark_source=benchmark, timeout_seconds=args.timeout_seconds,
                    producer_module=MODULE, cpu_cores=12, gpus=1)
    if args.nearby_physics:
        from src.envs import pair_v2_nearby_physics
        identity.update(pair_v2_nearby_physics.fields())
    return identity, keys


def runtime_claim(*, root, name, job, command, quota_reference, environment=None):
    environment = os.environ if environment is None else environment
    present = [bool(environment.get(key)) for key in ENV_KEYS]
    value = None if quota_reference is None else bank.read_reference(quota_reference, root)
    banked = isinstance(value, dict) and 'shared_ticket_bank' in value
    if not any(present):
        if banked:
            raise ValueError('Shared ticket-bank execution requires an atomic queue claim')
        return None
    if not all(present) or not banked:
        raise ValueError('Shared G01 requires both claim reference fields and its immutable ticket bank')
    from src.tools import gpu_shared_claim as shared
    reference = {'file': environment[ENV_KEYS[0]], 'sha256': environment[ENV_KEYS[1]]}
    raw = Path(reference['file']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('Shared claim bytes changed before validation')
    claim = json.loads(raw)
    unit = claim['unit']
    identity, keys = validate_unit(unit, root)
    if unit.get('identity') != identity or unit.get('claim_keys') != keys or arguments(unit['command']) != command:
        raise ValueError('Runtime claim changed the unit scientific identity, ticket keys or original command')
    allocation = {'scratch': environment.get('CVPR_SCRATCH'), 'pbs_jobid': job,
        'node': socket.gethostname(), 'cuda_visible_devices': environment.get('CUDA_VISIBLE_DEVICES'),
        'cpu_affinity': sorted(os.sched_getaffinity(0))}
    proof = shared.validate_runtime_claim(environment, allocation, MODULE, name, root=root)
    if proof['pbs_jobid'] != job:
        raise ValueError('Shared claim names another PBS allocation')
    return proof


def validate_admission(plan, *, root):
    """Offline claim authentication supplements every original source/lease check."""
    proof = plan.get('shared_queue_claim')
    if not isinstance(proof, dict):
        raise ValueError('Shared G01 source lacks its immutable claim proof')
    from src.tools import gpu_shared_claim as shared
    unit = proof['unit']
    identity, keys = validate_unit(unit, root)
    if unit.get('identity') != identity or unit.get('claim_keys') != keys:
        raise ValueError('Saved G01 claim altered its scientific ticket identity')
    # validate_unit derived these fields from the exact immutable argv.  Bind
    # the source admission back to that declaration as well as to its allocation.
    from src.envs import pair_v2_nearby_physics as nearby
    if '--nearby-physics' in arguments(unit['command']):
        if plan.get('nearby_physics_enabled') is not True:
            raise ValueError('Shared G01 admission dropped its original nearby-physics opt-in')
        nearby.validate_fields(plan)
        if any(plan[key] != identity[key] for key in nearby.FIELDS):
            raise ValueError('Shared G01 admission physics differs from its original unit')
    elif any(key in plan for key in ('nearby_physics_enabled', *nearby.FIELDS)):
        raise ValueError('Shared G01 legacy unit cannot acquire a different physics declaration')
    if plan.get('g01_engineering_smoke') is not None:
        raise ValueError('A formal shared G01 unit cannot become an engineering smoke source')
    allocation = {key: plan[key] for key in ('scratch', 'pbs_jobid', 'node', 'cpu_affinity')}
    allocation['cuda_visible_devices'] = plan['gpu_uuid']
    verified = shared.validate_saved_claim(proof['claim_reference'], allocation, MODULE,
                                         plan['parent_run_name'], root=root)
    if proof != verified:
        raise ValueError('G01 saved claim differs from its original authenticated receipt')
    for key in ('pbs_jobid', 'node', 'gpu_uuid', 'cpu_affinity'):
        if proof[key] != plan[key]:
            raise ValueError('G01 admission allocation differs from the shared claim: ' + key)
    quota_plan = bank.read_reference(unit['references']['quota_plan'], root)
    if (plan['scenes'] != [row['scene'] for row in quota_plan['assignments']]
            or plan['parent_run_name'] != quota_plan['parent_run_name']
            or plan['stage_index'] != 0 or plan['concurrency'] != 6
            or plan.get('g01_partition') != 'test'):
        raise ValueError('Shared G01 admission changed the original unit scene/lane assignment')
    return verified
