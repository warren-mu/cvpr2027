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

VERSION = 'c912aeab_opening_black_retry_host_v1'
HOSTS = ('hopper-22', 'hopper-28', 'hopper-30')


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


def same_retry_command(repo, old, new, old_name, new_name):
    """Only run identity and an otherwise identical immutable quota plan may change."""
    expected = list(old)
    require(expected.count('--name') == new.count('--name') == 1, 'Ambiguous retry name')
    require(expected[expected.index('--name') + 1] == old_name, 'Original command name differs')
    expected[expected.index('--name') + 1] = new_name
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


def rotation(repo, recorded_index=None):
    """Read immutable successful reservations while submit_phase owns its lock.

    Runtime replays the recorded prefix; later reservations cannot change a job's
    already-bound choice. Unknown/failed receipts do not claim a successful turn.
    """
    successful = {}
    for path in (repo / 'logs/pbs').glob('*/payload.json'):
        payload = json.loads(path.read_text())
        auth = payload.get('black_retry_host_authorization')
        if payload.get('dry_run') or not isinstance(auth, dict) or auth.get('version') != VERSION:
            continue
        receipt_path = path.with_name('submission.json')
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('returncode') != 0:
            continue
        index = auth.get('rotation_index')
        require(type(index) is int and index >= 0 and index not in successful
                and auth.get('requested_host') == payload.get('requested_host') == HOSTS[index % len(HOSTS)]
                and re.fullmatch(r'\d+\.hopper-m-\d+\s*', receipt.get('stdout', '')),
                'Successful host rotation history is ambiguous or invalid')
        successful[index] = (path, receipt_path)
    require(sorted(successful) == list(range(len(successful))), 'Successful host rotation history is not contiguous')
    index = len(successful) if recorded_index is None else recorded_index
    require(type(index) is int and 0 <= index <= len(successful), 'Invalid recorded host rotation index')
    previous = None
    if index:
        previous = {}
        for key, path in zip(('payload', 'receipt'), successful[index - 1]):
            previous[key] = {'file': str(path.relative_to(repo)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return index, previous


def validate_request_host(repo, *, host, retry_of, evidence_path, evidence_sha256,
                          name, command, task, group, gpus, wall_seconds, _recorded_rotation_index=None):
    repo = Path(repo).resolve()
    if host is None:
        require(evidence_path is None and evidence_sha256 is None, 'Host evidence requires explicit approved retry host')
        return None
    require(host in HOSTS and gpus == 1 and type(gpus) is int and task == 'T01'
            and group in ('G01', 'T02', 'T01v2', 'E2E', 'E2Epilot'), 'Only approved one-GPU black-retry hosts are allowed')
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
    require(payload.get('repo') == str(repo) and payload.get('dry_run') is False
            and payload.get('task_id') == task and payload.get('budget_group') == group
            and payload.get('queue') == 'autox' and payload.get('budget_batch') == 't00b_t01_20261108'
            and payload.get('requested_gpu_count') == gpus and payload.get('requested_wall_seconds') == wall_seconds
            and proof.get('original_command_unchanged') == payload['command']
            and name != payload['name'] and receipt.get('returncode') == 0 and not receipt.get('stderr')
            and receipt.get('stdout', '').strip() == retry_of, 'Original allocation/receipt differs')
    previous_host = payload.get('requested_host')
    require(previous_host is None or previous_host in HOSTS, 'Previous host is outside the approved rotation')
    rotation_index, rotation_previous = rotation(repo, _recorded_rotation_index)
    require(host == HOSTS[rotation_index % len(HOSTS)], 'Approved hosts must rotate22,28,30 across successful retry submissions')
    for candidate in (repo / 'logs/pbs').glob('*/payload.json'):
        prior = json.loads(candidate.read_text())
        receipt_file = candidate.with_name('submission.json')
        if prior.get('dry_run') or prior.get('retry_of') != retry_of or not receipt_file.exists():
            continue
        if json.loads(receipt_file.read_text()).get('returncode') == 0:
            require(_recorded_rotation_index is not None and prior.get('name') == name,
                    'This original black job already has a successful retry submission')
    same_retry_command(repo, payload['command'], command, payload['name'], name)
    account, account_path = logged(repo, proof['pbs_accounting'])
    from scripts.pbs.submit_phase import _final_accounting
    final = _final_accounting(repo, {retry_of}, paths=[account_path])
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
    return {'version': VERSION, 'evidence': {'file': str(guard_path.relative_to(repo)), 'sha256': evidence_sha256},
            'retry_of': retry_of, 'requested_host': host, 'previous_requested_host': previous_host,
            'rotation_index': rotation_index, 'rotation_previous': rotation_previous,
            'original_node': node, 'original_gpu_uuid': gpu,
            'queue_fallback_after_seconds': 1200, 'scope': 'opening black gate retry only; no scientific acceptance'}


def validate_payload_host(repo, payload):
    host = payload.get('requested_host')
    binding = payload.get('black_retry_host_authorization')
    if host is None:
        require(binding is None, 'Unpinned payload must not claim pinned retry authorization')
        return None
    require(isinstance(binding, dict) and binding.get('version') == VERSION, 'Pinned payload lacks explicit black-retry policy')
    evidence = binding.get('evidence', {})
    actual = validate_request_host(repo, host=host, retry_of=payload.get('retry_of'), evidence_path=evidence.get('file'),
        evidence_sha256=evidence.get('sha256'), name=payload['name'], command=payload['command'],
        task=payload['task_id'], group=payload['budget_group'], gpus=payload['requested_gpu_count'],
        wall_seconds=payload['requested_wall_seconds'], _recorded_rotation_index=binding.get('rotation_index'))
    require(actual == binding, 'Pinned runtime payload differs from authenticated retry permission')
    return actual
