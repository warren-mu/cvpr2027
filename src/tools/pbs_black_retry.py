"""c912aeab: opt-in known-host retry of an authenticated opening black gate.

This only validates submission provenance. It never changes scientific acceptance,
retries a failed experiment, or manages the20-minute queued-job fallback.
"""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shlex

LEGACY_VERSION = 'c912aeab_opening_black_retry_host_v1'
VERSION = 'c912aeab_opening_black_retry_host_v2_exclude22_after709293'
HOSTS = ('hopper-22', 'hopper-28', 'hopper-30')
AVAILABLE_HOSTS = ('hopper-28', 'hopper-30')
PILOT_STOPLOSS_AUTHORIZATION = '16be1c57'
PILOT_STOPLOSS_WALL_SECONDS = 15600
PILOT_STOPLOSS_DRIVER_SECONDS = 15536
DISABLED_HOSTS = {'hopper-22': {
    'reason': 'New pinned submissions excluded after authenticated opening black gate',
    'job_id': '709293.hopper-m-02',
    'gpu_uuid': 'GPU-5ff2e10b-a949-4c82-fa91-0e22f6075d56',
    'strict_guard_result': 'results/diag_g01_test_round2_s01_black_retry_plan_02.json'}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_ref(repo, reference, directory=None):
    require(isinstance(reference, dict), 'Missing black-retry evidence reference')
    filename = reference.get('file', reference.get('path'))
    require(isinstance(filename, str), 'Missing evidence filename')
    path = (repo / filename).resolve()
    path.relative_to(repo if directory is None else repo / directory)
    body = path.read_bytes()
    require(hashlib.sha256(body).hexdigest() == reference.get('sha256'), 'Black-retry evidence SHA differs')
    if 'bytes' in reference:
        require(type(reference['bytes']) is int and reference['bytes'] == len(body), 'Evidence byte count differs')
    return path, body


def logged(repo, reference):
    path, body = file_ref(repo, reference, 'results')
    spec = importlib.util.spec_from_file_location('black_retry_original_auditor', repo / 'scripts/audit_provenance.py')
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    require(not auditor.audit_file(str(path))['fail'], 'Black-retry source failed original provenance audit')
    data = json.loads(body)
    meta = data['_meta']
    require(isinstance(meta.get('end'), str) and bool(meta['end']), 'Black-retry source has not ended')
    log = (repo / meta['log_path']).resolve()
    log.relative_to(repo / 'logs/runs')
    lines = [line.partition(':')[2].strip() for line in log.read_text().splitlines()
             if line.startswith('FINAL_METRICS_JSON:')]
    require(lines and json.loads(lines[-1]) == data['metrics'], 'Exact final black-retry metrics missing')
    if 'log_sha256' in reference:
        require(reference['log_path'] == str(log.relative_to(repo))
                and hashlib.sha256(log.read_bytes()).hexdigest() == reference['log_sha256'], 'Source log changed')
    return data, path


def same_retry_command(repo, old, new, old_name, new_name, *, pilot_stoploss_reduction=0):
    """Only run identity, immutable quota identity, or the explicit pilot stop bound may change."""
    expected = list(old)
    require(expected.count('--name') == new.count('--name') == 1, 'Ambiguous retry name')
    require(expected[expected.index('--name') + 1] == old_name, 'Original command name differs')
    expected[expected.index('--name') + 1] = new_name
    if pilot_stoploss_reduction:
        require(pilot_command(old) and pilot_command(new)
                and type(pilot_stoploss_reduction) is int,
                'Only a native pilot retry may change its explicitly authorized engineering stop bound')
        old_seconds, new_seconds = pilot_timeout(old), pilot_timeout(new)
        require(old_seconds - new_seconds == pilot_stoploss_reduction and new_seconds >= 300
                and (pilot_stoploss_reduction > 0 or (new_seconds == PILOT_STOPLOSS_DRIVER_SECONDS
                    and pilot_option(old, '--arm') in ('T0', 'T4'))),
                'Pilot PBS/driver bounds need the same signed delta; increases are only the explicit16be1c57 T0/T4 bound')
        expected[expected.index('--timeout-seconds') + 1] = new[new.index('--timeout-seconds') + 1]
    if '--quota-plan' in expected:
        require(expected.count('--quota-plan') == new.count('--quota-plan') == 1
                and expected.count('--quota-plan-sha256') == new.count('--quota-plan-sha256') == 1,
                'Quota retry must preserve the exact plan arguments')
        plans = []
        for command, name in ((old, old_name), (new, new_name)):
            ref = {'file': command[command.index('--quota-plan') + 1],
                   'sha256': command[command.index('--quota-plan-sha256') + 1]}
            path, body = file_ref(repo, ref, 'configs')
            require(path.stat().st_mode & 0o222 == 0, 'Retry quota plan must be immutable')
            plan = json.loads(body)
            require(plan.get('parent_run_name') == name, 'Quota parent identity differs')
            plans.append(plan)
        transformed = dict(plans[0], parent_run_name=new_name)
        require(transformed == plans[1], 'Black retry changed quota/scene/seed settings')
        for flag in ('--quota-plan', '--quota-plan-sha256'):
            expected[expected.index(flag) + 1] = new[new.index(flag) + 1]
    require(expected == new, 'Black retry changed its original scientific/resource command')


def pilot_command(command):
    if not isinstance(command, list) or not command or not re.fullmatch(r'python(?:3(?:\.\d+)?)?', Path(command[0]).name):
        return False
    arguments = command[1:]
    if arguments[:1] == ['-u']:
        arguments = arguments[1:]
    return arguments[:3] == ['-m', 'src.tools.smoke_e2e_pilot', 'run']


def pilot_option(command, flag):
    require(command.count(flag) == 1 and command.index(flag) + 1 < len(command), 'Missing or ambiguous pilot option: ' + flag)
    return command[command.index(flag) + 1]


def pilot_timeout(command):
    value = pilot_option(command, '--timeout-seconds')
    require(re.fullmatch(r'[0-9]+', value) is not None, 'Pilot stop bound must be explicit integer seconds')
    return int(value)


def pilot_opening_black(payload, parent, gate, proof):
    """Original formal pilot gate only; no fabricated generic status or job fields."""
    cfg, pm, command = parent['config'], parent['metrics'], payload['command']
    identity, allocation = cfg.get('identity', {}), gate['config']['allocation']
    require(pilot_command(command) and payload.get('budget_group') in ('E2Epilot', 'E2EpilotT2')
            and payload['name'].startswith('smoke_e2e_pilot_')
            and cfg.get('version') == 'smoke_e2e_pilot_driver_v1'
            and proof.get('native_kind') == pm.get('kind') == 'smoke_e2e_pilot_training'
            and parent['_meta']['command'] == command and parent['_meta']['name'] == payload['name']
            and parent['_meta']['exit_code'] == pm.get('producer_exit_code') == 86
            and pm.get('pilot_completed') is False and pm.get('black_frame_detected') is True
            and pm.get('errors') == pm.get('cleanup_errors') == []
            and pm.get('training_logs') == [] and not pm.get('error')
            and not any(key in pm for key in ('updates', 'holdout_evaluations', 'checkpoint_records',
                'actual_completed_updates', 'checkpoint_saved', 'dataset_proof', 'source_authentication_seconds',
                'model_initialization_seconds', 'engineering_smoke_version'))
            and cfg.get('engineering_smoke') is None and 'runtime' not in cfg and 'model_snapshot_reference' not in cfg,
            'Pilot did not stop at its original opening gate before source authentication or training')
    arm = pilot_option(command, '--arm')
    require(arm in ('T0', 'T4', 'T2') and cfg.get('arm') == arm
            and cfg.get('updates') == {'T0': 50, 'T4': 50, 'T2': 5}[arm]
            and cfg.get('gpus') == 1 and cfg.get('budget_group') == payload['budget_group']
            and cfg.get('episodes_per_update') == 64 and cfg.get('seed') == 0
            and cfg.get('name') == pilot_option(command, '--name') == payload['name']
            and cfg.get('timeout_seconds') == pilot_timeout(command)
            and cfg.get('gate_timeout_seconds') == float(pilot_option(command, '--gate-timeout-seconds'))
            and cfg.get('critic_offload') is ('--critic-offload' in command)
            and '--engineering-smoke' not in command,
            'Pilot original arm/method/resource arguments differ')
    if payload['budget_group'] == 'E2EpilotT2':
        require(arm == 'T2' and pilot_option(command, '--budget-group') == 'E2EpilotT2',
                'The separate T2 budget requires its explicit original native T2 command')
    else:
        require('--budget-group' not in command or (arm in ('T0', 'T4')
                and pilot_option(command, '--budget-group') == 'E2Epilot'),
                'Pilot explicit budget flag must preserve its original native route')
    for flag, key in (('--manifest', 'manifest_reference'), ('--holdout-reference', 'holdout_reference')):
        require(json.loads(pilot_option(command, flag)) == cfg.get(key), 'Pilot original immutable input differs: ' + flag)
    require(cfg.get('model_source') == pilot_option(command, '--model-source')
            and cfg.get('pilot_spec') == pilot_option(command, '--pilot-spec')
            and identity.get('pbs_jobid') == proof['job_id'] == allocation['pbs_jobid']
            and identity.get('cuda_visible_devices') == proof['gpu_uuid'] == allocation['cuda_visible_devices']
            and identity.get('cpu_affinity') == allocation['cpu_affinity']
            and identity.get('scratch') == payload.get('environment', {}).get('CVPR_SCRATCH'), 'Pilot original input/allocation identity differs')
    source = pm.get('gate')
    require(isinstance(source, dict) and source.get('exit_code') == 86
            and source.get('file') == 'results/' + payload['name'] + '_gate.json'
            and gate['_meta']['name'] == payload['name'] + '_gate'
            and parent.get('sources') == [source], 'Pilot has business sources or an unrelated gate')
    gate_command = gate['_meta']['command']
    require(isinstance(gate_command, list) and len(gate_command) > 4
            and re.fullmatch(r'python(?:3(?:\.\d+)?)?', Path(gate_command[0]).name) is not None
            and gate_command[1:4] == ['-u', '-m', 'src.tools.render_gate']
            and pilot_option(gate_command, '--name') == payload['name'] + '_gate'
            and pilot_option(gate_command, '--gpus') == '1'
            and pilot_option(gate_command, '--resolution') == '255' and '--segmentation' in gate_command
            and float(pilot_option(gate_command, '--timeout-seconds')) == cfg['gate_timeout_seconds'],
            'Pilot opening gate command differs')
    return source


def next_host(version, index, previous_host):
    require(version in (LEGACY_VERSION, VERSION), 'Unknown pinned retry policy version')
    if version == LEGACY_VERSION:
        return HOSTS[index % len(HOSTS)]
    # Keep the global successful-reservation index, but skip the disabled node.
    start = -1 if previous_host is None else HOSTS.index(previous_host)
    return next(HOSTS[(start + offset) % len(HOSTS)] for offset in range(1, len(HOSTS) + 1)
                if HOSTS[(start + offset) % len(HOSTS)] in AVAILABLE_HOSTS)


def rotation(repo, recorded_index=None):
    """Read immutable successful reservations while submit_phase owns its lock.

    Runtime replays the recorded prefix; later reservations cannot change a job's
    already-bound choice. Unknown/failed receipts do not claim a successful turn.
    """
    successful = {}
    for path in (repo / 'logs/pbs').glob('*/payload.json'):
        payload = json.loads(path.read_text())
        auth = payload.get('black_retry_host_authorization')
        if payload.get('dry_run') or not isinstance(auth, dict):
            continue
        require(auth.get('version') in (LEGACY_VERSION, VERSION), 'Unknown host rotation authorization version')
        receipt_path = path.with_name('submission.json')
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('returncode') != 0:
            continue
        index = auth.get('rotation_index')
        require(type(index) is int and index >= 0 and index not in successful
                and auth.get('requested_host') == payload.get('requested_host') and payload.get('requested_host') in HOSTS
                and re.fullmatch(r'\d+\.hopper-m-\d+\s*', receipt.get('stdout', '')),
                'Successful host rotation history is ambiguous or invalid')
        successful[index] = (path, receipt_path)
    require(sorted(successful) == list(range(len(successful))), 'Successful host rotation history is not contiguous')
    previous_host = None
    v2_seen = False
    for turn in range(len(successful)):
        item = json.loads(successful[turn][0].read_text())
        auth = item['black_retry_host_authorization']
        require(not (v2_seen and auth['version'] == LEGACY_VERSION), 'Legacy rotation cannot resume after v2')
        require(item['requested_host'] == next_host(auth['version'], turn, previous_host),
                'Successful versioned host rotation history has the wrong node')
        if auth['version'] == VERSION:
            require(auth.get('available_hosts') == list(AVAILABLE_HOSTS) and auth.get('disabled_hosts') == DISABLED_HOSTS,
                    'Successful v2 host rotation policy differs')
            v2_seen = True
        previous_host = item['requested_host']
    index = len(successful) if recorded_index is None else recorded_index
    require(type(index) is int and 0 <= index <= len(successful), 'Invalid recorded host rotation index')
    previous = None
    if index:
        previous = {}
        for key, path in zip(('payload', 'receipt'), successful[index - 1]):
            previous[key] = {'file': str(path.relative_to(repo)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return index, previous


def pack_opening_black(repo, payload, payload_path, receipt_path, parent, gate, proof, canonical):
    """One-GPU PACK only: independently replay original zero-business accounting."""
    from src.tools import pbs_mixed_accounting as mixed
    pm, cfg = parent['metrics'], parent['config']
    require(proof.get('native_kind') == pm.get('kind') == 'gpu_shared_pack'
            and parent['_meta']['command'] == payload['command']
            and parent['_meta']['exit_code'] == pm.get('producer_exit_code') == 86
            and cfg.get('pbs_jobid') == proof['job_id'] and cfg.get('gpus') == 1
            and payload['requested_gpu_count'] == 1 and cfg.get('version') == 'shared_gpu_pack_v1'
            and pm.get('unit_claims') == pm.get('unit_runs') == []
            and pm.get('all_owned_units_ended') is True
            and pm.get('cleanup_errors') == pm.get('errors') == [] and not pm.get('error')
            and pm.get('downstream_allowed') is False,
            'PACK retry requires an ended single-GPU opening black gate with no business claims')
    require((pm.get('status'), pm.get('stopping_reasons')) in (
                ('stopped_with_original_work_preserved', ['insufficient_remaining_time_for_full_original_unit']),
                ('queue_finished_or_claimed_elsewhere', [])),
            'PACK did not stop solely after retiring its opening black GPU')
    gpu = proof['gpu_uuid']
    require(cfg.get('allocated_gpu_uuids') == [gpu] and cfg.get('parent_cuda_visible_devices') == gpu
            and pm.get('retired_gpu_uuids') == [gpu]
            and cfg.get('cpu_affinity') == gate['config']['allocation']['cpu_affinity'],
            'PACK original allocation or retired GPU differs')
    gates = pm.get('gates')
    require(isinstance(gates, list) and len(gates) == 1, 'PACK requires exactly one opening gate')
    row = gates[0]
    require(row.get('status') == 'black' and row.get('exit_code') == 86 and not row.get('error')
            and row.get('gpu_uuid') == gpu and row.get('cpu_affinity') == cfg['cpu_affinity'],
            'PACK gate was not an authenticated clean black-card retirement')
    gate_name = payload['name'] + '_gate_0'
    require(gate['_meta']['name'] == gate_name and gate['_meta']['command'] == [
                'python3', '-u', '-m', 'src.tools.render_gate', '--name', gate_name, '--gpus', '1',
                '--resolution', '255', '--segmentation', '--timeout-seconds', str(cfg['gate_timeout_seconds'])],
            'PACK original gate command/name changed')
    for reference, expected in ((cfg.get('parent_payload_reference'), payload_path),
                                (cfg.get('parent_receipt_reference'), receipt_path)):
        path, _ = file_ref(repo, reference, 'logs/pbs')
        require(path == expected, 'PACK source is not bound to its original reservation')
    binding = mixed.validate_payload(repo, payload)
    require(cfg.get('plan_reference') == binding['plan_reference']
            and cfg.get('queue_plan') == mixed.option(payload['command'], '--queue-plan')
            and cfg.get('queue_plan_sha256') == mixed.option(payload['command'], '--queue-plan-sha256')
            and cfg.get('queue_root') == mixed.option(payload['command'], '--queue-root')
            and mixed.option(payload['command'], '--gpus') == '1'
            and cfg.get('timeout_seconds') == float(mixed.option(payload['command'], '--timeout-seconds'))
            and cfg.get('gate_timeout_seconds') == float(mixed.option(payload['command'], '--gate-timeout-seconds')),
            'PACK source changed its exact immutable queue/resource command')
    mixed_reference = proof.get('mixed_accounting')
    cost, cost_path = logged(repo, mixed_reference)
    require(cost_path == repo / 'results' / (payload['name'] + '_mixed_accounting.json')
            and cost['_meta']['exit_code'] == 0,
            'PACK retry lacks its successful original mixed-accounting result')
    original = {'name': payload['name'], 'payload': str(payload_path.relative_to(repo)),
                'job_id': proof['job_id']}
    actual = mixed.finalized_split(repo, original, canonical)
    require(actual is not None and actual == cost['metrics'] and actual.get('unit_count') == 0
            and actual.get('zero_business') is True,
            'PACK original mixed accounting does not prove zero owned business')
    # finalized_split reconstructs the complete immutable claim ownership set;
    # an unreported claim belonging to this PBS allocation must fail there.
    return row['source']


def validate_request_host(repo, *, host, retry_of, evidence_path, evidence_sha256,
                          name, command, task, group, gpus, wall_seconds, _recorded_rotation_index=None,
                          _recorded_version=None):
    repo = Path(repo).resolve()
    if host is None:
        require(evidence_path is None and evidence_sha256 is None, 'Host evidence requires explicit approved retry host')
        return None
    version = VERSION if _recorded_version is None else _recorded_version
    require(version in (LEGACY_VERSION, VERSION) and
            (_recorded_version is None or _recorded_rotation_index is not None), 'Policy version override requires recorded runtime index')
    allowed_hosts = HOSTS if version == LEGACY_VERSION else AVAILABLE_HOSTS
    require(group != 'PACK' or version == VERSION, 'PACK opening-black retries require the current two-host policy')
    require(host in allowed_hosts and gpus == 1 and type(gpus) is int and task == 'T01'
            and group in ('G01', 'T02', 'T01v2', 'E2E', 'E2Epilot', 'E2EpilotT2', 'PACK'), 'Only approved one-GPU black-retry hosts are allowed')
    require(isinstance(retry_of, str) and re.fullmatch(r'\d+\.hopper-m-\d+', retry_of)
            and isinstance(evidence_path, str) and re.fullmatch(r'[0-9a-f]{64}', evidence_sha256 or ''),
            'Pinned first submissions or retries without exact black evidence are forbidden')
    evidence = {'file': evidence_path, 'sha256': evidence_sha256}
    guard, guard_path = logged(repo, evidence)
    proof = guard['metrics'].get('strict_black_gate_proof', guard['metrics'])
    require(guard['_meta']['exit_code'] == 0 and proof.get('eligible_proven_black_gate_retry') is True
            and proof.get('job_id') == retry_of, 'Not a successful strict guard for this original black job')
    payload_path, payload_bytes = file_ref(repo, proof['original_payload'], 'logs/pbs')
    script_path, _ = file_ref(repo, proof['original_job_script'], 'logs/pbs')
    receipt_path, receipt_bytes = file_ref(repo, proof['original_receipt'], 'logs/pbs')
    require(script_path == payload_path.with_name('job.pbs') and receipt_path == payload_path.with_name('submission.json')
            and payload_path.stat().st_mode & 0o222 == script_path.stat().st_mode & 0o222 == 0,
            'Original submission files are not the same immutable reservation')
    payload, receipt = json.loads(payload_bytes), json.loads(receipt_bytes)
    pilot_retry = group in ('E2Epilot', 'E2EpilotT2') and pilot_command(payload.get('command'))
    original_wall = payload.get('requested_wall_seconds')
    stoploss_reduction = 0
    if pilot_retry:
        require(type(original_wall) is int and type(wall_seconds) is int and 0 < wall_seconds
                and (wall_seconds <= original_wall or (group == 'E2Epilot'
                    and pilot_option(payload['command'], '--arm') in ('T0', 'T4')
                    and wall_seconds == PILOT_STOPLOSS_WALL_SECONDS
                    and pilot_timeout(command) == PILOT_STOPLOSS_DRIVER_SECONDS)),
                'Pilot increase requires the exact16be1c57 T0/T4 PBS/driver stop bound')
        stoploss_reduction = original_wall - wall_seconds
        require(original_wall - pilot_timeout(payload['command']) == 64
                and wall_seconds - pilot_timeout(command) == 64,
                'Pilot retry must preserve its original64-second recording/cleanup margin')
    require(payload.get('repo') == str(repo) and payload.get('dry_run') is False
            and payload.get('task_id') == task and payload.get('budget_group') == group
            and payload.get('queue') == 'autox' and payload.get('budget_batch') == 't00b_t01_20261108'
            and payload.get('requested_gpu_count') == gpus and (pilot_retry or payload.get('requested_wall_seconds') == wall_seconds)
            and proof.get('original_command_unchanged') == payload['command']
            and name != payload['name'] and receipt.get('returncode') == 0 and not receipt.get('stderr')
            and receipt.get('stdout', '').strip() == retry_of, 'Original allocation/receipt differs')
    previous_host = payload.get('requested_host')
    require(previous_host is None or previous_host in HOSTS, 'Previous host is outside the approved rotation')
    rotation_index, rotation_previous = rotation(repo, _recorded_rotation_index)
    previous_global_host = None
    if rotation_previous:
        _, previous_bytes = file_ref(repo, rotation_previous['payload'], 'logs/pbs')
        previous_global_host = json.loads(previous_bytes)['requested_host']
    require(host == next_host(version, rotation_index, previous_global_host),
            'Approved hosts must rotate globally, skipping nodes disabled by the recorded policy')
    for candidate in (repo / 'logs/pbs').glob('*/payload.json'):
        prior = json.loads(candidate.read_text())
        receipt_file = candidate.with_name('submission.json')
        if prior.get('dry_run') or prior.get('retry_of') != retry_of or not receipt_file.exists():
            continue
        if json.loads(receipt_file.read_text()).get('returncode') == 0:
            require(_recorded_rotation_index is not None and prior.get('name') == name,
                    'This original black job already has a successful retry submission')
    same_retry_command(repo, payload['command'], command, payload['name'], name,
                       pilot_stoploss_reduction=stoploss_reduction)
    account, account_path = logged(repo, proof['pbs_accounting'])
    phase_spec = importlib.util.spec_from_file_location('black_retry_original_phase', repo / 'scripts/pbs/submit_phase.py')
    phase = importlib.util.module_from_spec(phase_spec)
    phase_spec.loader.exec_module(phase)
    final = phase._final_accounting(repo, {retry_of}, paths=[account_path])
    require(retry_of in final, 'Original canonical final accounting is unavailable')
    jobs = [row for row in account['config']['jobs'] if row.get('job_id') == retry_of]
    require(len(jobs) == 1 and jobs[0]['qstat_exit'] == 0, 'Original canonical job is ambiguous')
    fields = jobs[0]['fields']
    require(fields.get('job_state') == 'F' and fields.get('Exit_status') == '86'
            and fields.get('run_count') == '1' and fields.get('project') == 'CFP04-SF-125'
            and fields.get('Job_Owner', '').split('@')[0] == 'e1583364'
            and fields.get('Resource_List.ngpus') == '1' and fields.get('Resource_List.ncpus') == '12'
            and fields.get('queue') == 'smallx' and 'PBS_O_QUEUE=autox' in fields.get('Variable_List', '')
            and shlex.split(fields['Submit_arguments']) == receipt['command'][1:]
            and fields['Output_Path'].split(':', 1)[1] == str(payload_path.with_name('pbs.out')),
            'Not the original owned final86 one-GPU Express allocation')
    sources = [logged(repo, ref)[0] for ref in proof['source_results']]
    originals = [d for d in sources if d['_meta'].get('name') == payload['name']]
    gates = [d for d in sources if isinstance(d.get('config', {}).get('allocation'), dict)]
    require(len(originals) == len(gates) == 1, 'Original parent/gate chain is missing or ambiguous')
    parent, gate = originals[0], gates[0]
    pm, gc, gm = parent['metrics'], gate['config'], gate['metrics']
    if group == 'PACK':
        parent_gate = pack_opening_black(repo, payload, payload_path, receipt_path, parent, gate, proof,
                                         final[retry_of])
    elif pilot_retry:
        parent_gate = pilot_opening_black(payload, parent, gate, proof)
    else:
        require(parent['_meta']['command'] == payload['command'] and parent['_meta']['exit_code'] == 86
                and parent['config']['pbs_jobid'] == retry_of and pm.get('status') == 'blocked_by_render_gate'
                and pm.get('black_frame_detected') is True and pm.get('cleanup_errors') == []
                and not pm.get('error') and not pm.get('errors'), 'Parent did not stop cleanly at its opening black gate')
        business_fields = {'g01_parallel_collection': ('scene_runs', 'started_shards', 'stages', 'admission_final', 'unfinished_run_names', 'stop_reasons'),
                           't02_parallel_export': ('units',), 't02_table_export': ('pairs',),
                           't02_formal_cache_check': ('worker_runs', 'population_ledger')}
        require(proof.get('native_kind') == pm.get('kind') and pm.get('kind') in business_fields, 'Unsupported opening-black parent kind')
        require(all(pm.get(key) == [] for key in business_fields[pm['kind']]), 'Original black job started business work')
        parent_gate = pm['gate'].get('source', pm['gate'])
    gate_source = next(ref for ref in proof['source_results'] if ref.get('path', ref.get('file')) == 'results/' + gate['_meta']['name'] + '.json')
    require(parent_gate.get('sha256') == gate_source['sha256'], 'Parent does not bind the original gate bytes')
    a = gc['allocation']
    node, gpu = proof['node'], proof['gpu_uuid']
    require({part.split('/')[0] for part in fields['exec_host'].split('+')} == {node}
            and parent['_meta']['host'] == gate['_meta']['host'] == parent['config']['node'] == gc['node'] == node
            and gc['pbs_jobid'] == a['pbs_jobid'] == retry_of
            and a['cuda_visible_devices'] == gpu and a['cuda_driver_uuids'] == [gpu]
            and len(a['cpu_affinity']) == len(set(a['cpu_affinity'])) == 12
            and gc['resolution'] == 255 and gc['segmentation'] is True
            and gate['_meta']['exit_code'] == gm.get('exit_code') == 86
            and gm.get('passed') is False and gm.get('black_frame_detected') is True
            and gm.get('errors') == gm.get('cleanup_errors') == []
            and gm.get('expected_gpu_count') == gm.get('completed_gpu_count') == 1
            and len(gm['devices']) == len(a['devices']) == 1, 'Native gate identity/completion differs')
    observation = gm['devices'][0]['observation']
    require(gm['devices'][0]['worker_error'] is None and not observation.get('error')
            and not observation.get('cleanup_error') and observation['gpu_identity'] == a['devices'][0]
            and observation['gpu_identity']['gpu_uuid'] == gpu and observation['node'] == node
            and [row['stage'] for row in observation['captures']] == ['after_constructor', 'after_Pass']
            and any(row['stage'] == 'Pass' and row['fresh_main_rgb_payloads'] > 0 and row['cached_metadata'] is False
                    for row in observation['wire_packets']), 'Missing successful fresh native black captures')
    from PIL import Image
    require(len(proof['captures']) == 2, 'Strict guard lacks both original black PNGs')
    for capture, ref in zip(observation['captures'], proof['captures']):
        path, body = file_ref(repo, ref, 'logs/runs')
        require(path == Path(capture['png']['path']).resolve()
                and ref['sha256'] == capture['png']['sha256'] and capture['valid_shape'] is True
                and capture['shape'] == [255, 255, 3] and capture['dtype'] == 'uint8'
                and capture['action_success'] is True and not capture['error_message'] and capture['black'] is True
                and capture['nonzero_rgb_values'] == capture['min_rgb'] == capture['max_rgb'] == 0,
                'Native capture is not a successful-action black frame')
        with Image.open(io.BytesIO(body)) as image:
            require(image.mode == 'RGB' and image.size == (255, 255) and image.getextrema() == ((0, 0),) * 3,
                    'Original decoded PNG is not all black')
    binding = {'version': version, 'evidence': {'file': str(guard_path.relative_to(repo)), 'sha256': evidence_sha256},
            'retry_of': retry_of, 'requested_host': host, 'previous_requested_host': previous_host,
            'rotation_index': rotation_index, 'rotation_previous': rotation_previous,
            'original_node': node, 'original_gpu_uuid': gpu,
            'queue_fallback_after_seconds': 1200, 'scope': 'opening black gate retry only; no scientific acceptance'}
    if stoploss_reduction < 0:
        binding['pilot_stoploss_authorization'] = {
            'revision': PILOT_STOPLOSS_AUTHORIZATION, 'original_wall_seconds': original_wall,
            'requested_wall_seconds': wall_seconds, 'driver_seconds': PILOT_STOPLOSS_DRIVER_SECONDS,
            'unchanged_cleanup_recording_margin_seconds': 64,
            'scope': 'Explicit T0/T4 time bound only; original arm,50 updates and all method/input arguments unchanged'}
    if version == VERSION:
        binding.update(available_hosts=list(AVAILABLE_HOSTS), disabled_hosts=json.loads(json.dumps(DISABLED_HOSTS)))
    return binding


def validate_payload_host(repo, payload):
    host = payload.get('requested_host')
    binding = payload.get('black_retry_host_authorization')
    if host is None:
        require(binding is None, 'Unpinned payload must not claim pinned retry authorization')
        return None
    require(isinstance(binding, dict) and binding.get('version') in (LEGACY_VERSION, VERSION), 'Pinned payload lacks explicit black-retry policy')
    evidence = binding.get('evidence', {})
    actual = validate_request_host(repo, host=host, retry_of=payload.get('retry_of'), evidence_path=evidence.get('file'),
        evidence_sha256=evidence.get('sha256'), name=payload['name'], command=payload['command'],
        task=payload['task_id'], group=payload['budget_group'], gpus=payload['requested_gpu_count'],
        wall_seconds=payload['requested_wall_seconds'], _recorded_rotation_index=binding.get('rotation_index'),
        _recorded_version=binding['version'])
    require(actual == binding, 'Pinned runtime payload differs from authenticated retry permission')
    return actual
