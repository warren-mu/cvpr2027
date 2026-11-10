"""Immutable scientific units and allocation-scoped claims for the shared queue.

The registry stores ownership, never a scientific PASS. Terminal/uncertain
claims are not silently recycled. A new physical attempt needs an explicit
reviewed plan; technical run names cannot make an occupied scientific key free.
"""
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'shared_gpu_units_v1'
CLAIM_VERSION = 'shared_gpu_claim_v1'
PRIORITIES = {'G01': 300, 'T02': 200, 'E2Epilot': 100}
UUID = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z')
SAFE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_]*\Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def canonical_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def reference(path):
    path = Path(path).resolve()
    body = path.read_bytes()
    return {'file': str(path), 'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)}


def read_reference(ref, roots):
    require(isinstance(ref, dict) and isinstance(ref.get('file'), str)
            and re.fullmatch(r'[0-9a-f]{64}', str(ref.get('sha256', ''))), 'Missing immutable reference')
    path = Path(ref['file']).resolve()
    require(any(path.is_relative_to(Path(root).resolve()) for root in roots), 'Reference escaped its evidence roots')
    body = path.read_bytes()
    require(len(body) <= 32 * 1024 * 1024 and hashlib.sha256(body).hexdigest() == ref['sha256'], 'Reference bytes differ')
    if 'bytes' in ref:
        require(type(ref['bytes']) is int and ref['bytes'] == len(body), 'Reference byte count differs')
    return json.loads(body)


def repo_reference(ref, root):
    result = dict(ref)
    result['file'] = str((Path(root) / result['file']).resolve())
    return result


def write_immutable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return reference(path)


def _argument(command, flag):
    require(command.count(flag) == 1 and command.index(flag) + 1 < len(command), 'Missing/ambiguous argument: ' + flag)
    return command[command.index(flag) + 1]


def t02_identity(unit, root):
    """Scheduling identity only. Original exporter still authenticates science."""
    command = unit['command']
    manifest_path = _argument(command, '--manifest')
    manifest_sha = _argument(command, '--manifest-sha256')
    manifest_ref = {'file': manifest_path, 'sha256': manifest_sha}
    manifest = read_reference(repo_reference(manifest_ref, root), [Path(root) / 'configs'])
    require(manifest.get('version') == 't02_explicit_source_units_v1'
            and manifest.get('partition') in ('train', 'test')
            and isinstance(manifest.get('entries'), list) and 1 <= len(manifest['entries']) <= 6,
            'Shared T02 units retain their original at-most-six fixed input manifest')
    keys = []
    low, high = (1, 20) if manifest['partition'] == 'train' else (21, 30)
    scenes = {f'FloorPlan{base + index}' for base in (0, 200, 300, 400) for index in range(low, high + 1)}
    for entry in manifest['entries']:
        require(entry.get('field_of_view') == 60 and isinstance(entry.get('source'), dict)
                and isinstance(entry.get('pair'), dict) and entry.get('scene') in scenes
                and re.fullmatch(r'floorplan[0-9]+_[0-9]+', str(entry.get('pair_id', '')))
                and re.fullmatch(r'[0-9a-f]{64}', str(entry.get('proposal_identity', ''))),
                'Original T02 source identity is incomplete')
        for field, directory in (('source', 'results'), ('pair', 'data/world_pairs/v2')):
            # This is only the immutable manifest's scheduling identity. The
            # unchanged original exporter authenticates every pair/source byte
            # and all science; queue operations must not repeatedly decode the
            # complete native bundles for the entire remaining population.
            ref = entry[field]
            path = (Path(root) / ref.get('file', '')).resolve()
            require(isinstance(ref.get('file'), str) and path.is_file()
                    and path.is_relative_to((Path(root) / directory).resolve())
                    and re.fullmatch(r'[0-9a-f]{64}', str(ref.get('sha256', ''))),
                    'Manifest scheduling reference is missing or escaped its root')
        original_pair = {'file': str((Path(root) / entry['pair']['file']).resolve().relative_to(Path(root).resolve())),
                         'sha256': entry['pair']['sha256']}
        key = canonical_hash({'kind': 't02_original_pair', 'pair': original_pair,
                              'table_protocol': 'paired_view_table_d4_shadow_v2'})
        keys.append(key)
    require(len(keys) == len(set(keys)), 'Repeated original pair in shared T02 unit')
    flags = ('--concurrency', '--timeout-seconds', '--unit-timeout-seconds',
             '--pair-timeout-seconds', '--gate-timeout-seconds')
    params = {key.removeprefix('--'): _argument(command, key) for key in flags}
    expected = [command[0], '-u', '-m', 'src.tools.t02_export_parallel', '--name', unit['name'],
                '--manifest', manifest_path, '--manifest-sha256', manifest_sha]
    for flag in flags:
        expected += [flag, params[flag.removeprefix('--')]]
    sample_rejected_raw = '--sample-rejected-raw' in command
    if sample_rejected_raw:
        expected.append('--sample-rejected-raw')
    compressed = '--compress-test-tables' in command
    selection_ref = None
    if compressed:
        from src.tools import t02_test_compression as compression
        selection_ref = compression.selection_reference(_argument(command, '--test-selection'))
        require(manifest['partition'] == 'test', 'Compressed shared exports require test inputs')
        expected += compression.command_suffix(selection_ref)
    require(command == expected and params['concurrency'] == '6', 'Shared T02 command differs from the original fixed six-way entry')
    require(float(params['timeout-seconds']) == unit['timeout_seconds'], 'Unit time grant differs')
    identity = {'kind': 't02_export_batch', 'manifest': manifest_ref,
                'partition': manifest['partition'], 'parameters': params, 'original_pair_keys': keys}
    if sample_rejected_raw:
        from src.tools.t02_rejected_raw import POLICY
        identity['raw_retention_policy'] = copy.deepcopy(POLICY)
    if compressed:
        identity['test_compression'] = {'policy': copy.deepcopy(compression.codec.POLICY),
                                        'selection_reference': selection_ref}
    return identity, sorted(keys)


def validate_unit(unit, root=ROOT, *, external=False):
    require(isinstance(unit, dict) and SAFE.fullmatch(str(unit.get('name', ''))), 'Unsafe shared unit name')
    group = unit.get('budget_group')
    require(group in PRIORITIES and unit.get('priority') == PRIORITIES[group]
            and unit.get('gpus') == 1 and type(unit.get('gpus')) is int
            and unit.get('cpu_cores') == 12 and type(unit.get('cpu_cores')) is int,
            'Each shared unit needs its original group, one GPU and twelve CPUs')
    command = unit.get('command')
    require(isinstance(command, list) and all(isinstance(x, str) for x in command)
            and len(command) >= 6 and re.fullmatch(r'python(?:3(?:\.\d+)?)?', Path(command[0]).name), 'Invalid original unit command')
    require(command[1:4] == ['-u', '-m', unit.get('producer_module')]
            and _argument(command, '--name') == unit['name'], 'Unit producer/name differs')
    grant = unit.get('timeout_seconds')
    require(isinstance(grant, (int, float)) and not isinstance(grant, bool)
            and math.isfinite(grant) and grant > 0, 'Require complete finite original unit grant')
    if unit['producer_module'] == 'src.tools.t02_export_parallel':
        require(group == 'T02', 'T02 task cannot be charged as another unit group')
        identity, keys = t02_identity(unit, root)
    elif unit['producer_module'] == 'src.tools.g01_collect_parallel':
        require(group == 'G01', 'G01 task cannot be charged as another unit group')
        from src.tools.g01_shared_unit import validate_unit as validate_g01
        identity, keys = validate_g01(unit, Path(root), external=external)
    else:
        raise ValueError('Shared producer is not implemented/authorized; no implicit E2E qualification')
    require(unit.get('identity') == identity and unit.get('claim_keys') == sorted(keys)
            and len(keys) == len(set(keys)) and bool(keys)
            and all(re.fullmatch(r'[0-9a-f]{64}', key) for key in keys)
            and unit.get('unit_id') == canonical_hash(identity), 'Stable scientific unit declaration differs')
    return identity, sorted(keys)


def validate_external(row, root):
    unit = row['unit']
    validate_unit(unit, root, external=True)
    payload = read_reference(repo_reference(row['payload_reference'], root), [Path(root) / 'logs/pbs'])
    receipt = read_reference(repo_reference(row['receipt_reference'], root), [Path(root) / 'logs/pbs'])
    p = (Path(root) / row['payload_reference']['file']).resolve()
    r = (Path(root) / row['receipt_reference']['file']).resolve()
    require(r == p.with_name('submission.json') and payload['name'] == unit['name']
            and payload['command'] == unit['command'] and payload['task_id'] == 'T01'
            and payload['budget_group'] == unit['budget_group'] and payload['requested_gpu_count'] == 1
            and receipt.get('returncode') == 0
            and re.fullmatch(r'\d+\.hopper-m-\d+\s*', receipt.get('stdout', '')),
            'External reservation is not its actual original one-GPU submission')
    return receipt['stdout'].strip()


def validate_plan(plan, root=ROOT):
    require(isinstance(plan, dict) and plan.get('version') == VERSION
            and SAFE.fullmatch(str(plan.get('queue_id', ''))), 'Invalid shared queue plan')
    require(plan.get('priority_order') == list(PRIORITIES), 'Approved shared priorities changed')
    units, external = plan.get('units'), plan.get('external_reservations')
    require(isinstance(units, list) and isinstance(external, list) and units, 'Plan requires explicit units/reservations')
    ids, names, keys = set(), set(), set()
    for unit in units:
        validate_unit(unit, root)
        require(unit['unit_id'] not in ids and unit['name'] not in names
                and not keys.intersection(unit['claim_keys']), 'Overlapping shared scientific units')
        ids.add(unit['unit_id']);names.add(unit['name']);keys.update(unit['claim_keys'])
    for row in external:
        validate_external(row, root)
    return plan


def load_plan(path, expected_sha, *, root=ROOT):
    root = Path(root).resolve();path = (root / path).resolve()
    ref = {'file': str(path), 'sha256': expected_sha}
    plan = read_reference(ref, [root / 'configs'])
    require(path.stat().st_mode & 0o222 == 0, 'Shared plan must be immutable')
    validate_plan(plan, root)
    return plan, reference(path)


def validate_saved_claim(ref, identity, producer_module, name, *, root=ROOT):
    root = Path(root).resolve()
    scratch = Path(identity['scratch']).resolve()
    claim = read_reference(ref, [scratch / 'shared_gpu_queue'])
    require(claim.get('version') == CLAIM_VERSION and Path(ref['file']).stat().st_mode & 0o222 == 0,
            'Missing immutable shared claim')
    plan, plan_ref = load_plan(claim['plan_reference']['file'], claim['plan_reference']['sha256'], root=root)
    require(claim['plan_reference'] == plan_ref, 'Claim changed queue plan bytes')
    candidates = [unit for unit in plan['units'] if unit['unit_id'] == claim['unit_id']]
    require(len(candidates) == 1 and candidates[0] == claim['unit'], 'Claim is not an original planned unit')
    unit = candidates[0]
    require(unit['producer_module'] == producer_module and unit['name'] == name == claim['name']
            and unit['budget_group'] == claim['budget_group'] and unit['command'] == claim['command'],
            'Claim does not authorize this exact producer/name/command')
    payload = read_reference(claim['parent_payload_reference'], [root / 'logs/pbs'])
    receipt = read_reference(claim['parent_receipt_reference'], [root / 'logs/pbs'])
    payload_path = Path(claim['parent_payload_reference']['file'])
    require(Path(claim['parent_receipt_reference']['file']) == payload_path.with_name('submission.json')
            and payload.get('budget_group') == 'PACK' and payload.get('task_id') == 'T01'
            and payload.get('requested_gpu_count') in (1, 4) and payload.get('queue') == 'autox'
            and receipt.get('returncode') == 0 and receipt.get('stdout', '').strip() == identity['pbs_jobid'],
            'Claim lacks its real PACK submission')
    require(Path(payload.get('environment', {}).get('CVPR_SCRATCH', '')).resolve() == scratch,
            'Claim scratch root differs from the original PBS payload')
    binding = payload.get('mixed_budget_authorization', {})
    require(binding.get('plan_reference') == plan_ref and unit['budget_group'] in binding.get('possible_groups', []),
            'Claim is outside the original pack budget reservation')
    cpus = claim.get('cpu_affinity')
    masks = claim.get('parent_cuda_visible_devices', '').split(',')
    require(len(masks) == payload['requested_gpu_count'] and len(set(masks)) == len(masks)
            and all(UUID.fullmatch(x) for x in masks) and claim.get('gpu_uuid') in masks
            and isinstance(cpus, list) and len(cpus) == len(set(cpus)) == 12
            and all(type(x) is int and x >= 0 for x in cpus), 'Claim allocation is invalid')
    require(claim['pbs_jobid'] == identity['pbs_jobid'] and claim['node'] == identity['node']
            and claim['gpu_uuid'] == identity['cuda_visible_devices'] and cpus == identity['cpu_affinity'],
            'Claim GPU/CPU/PBS identity differs from the actual unit')
    require(Path(claim['queue_root']).resolve() == scratch / 'shared_gpu_queue' / plan['queue_id'], 'Claim queue root differs')
    return {**copy.deepcopy(claim), 'claim_reference': copy.deepcopy(ref)}


def validate_runtime_claim(environment, identity, producer_module, name, *, root=ROOT):
    filename, digest = environment.get('CVPR_SHARED_CLAIM_FILE'), environment.get('CVPR_SHARED_CLAIM_SHA256')
    if filename is None and digest is None:
        return None
    require(isinstance(filename, str) and isinstance(digest, str), 'Partial shared claim environment')
    proof = validate_saved_claim({'file': filename, 'sha256': digest}, identity, producer_module, name, root=root)
    require(environment.get('CVPR_QUEUE_PARENT_MASK') == proof['parent_cuda_visible_devices'], 'Parent mask differs')
    return proof


def validate_t02_child_claim(environment, identity, arguments, *, root=ROOT):
    """Bind an original single-pair child to its unchanged six-pair command."""
    filename = environment.get('CVPR_SHARED_CLAIM_FILE')
    if filename is None and environment.get('CVPR_SHARED_CLAIM_SHA256') is None:
        return None
    ref = {'file': filename, 'sha256': environment.get('CVPR_SHARED_CLAIM_SHA256')}
    raw = read_reference(ref, [Path(identity['scratch']) / 'shared_gpu_queue'])
    proof = validate_runtime_claim(environment, identity, 'src.tools.t02_export_parallel', raw['name'], root=root)
    match = re.fullmatch(re.escape(proof['name']) + r'_u([0-9]{4})', arguments.name)
    require(match is not None, 'T02 child name is outside its claimed batch')
    command = proof['command']
    manifest = read_reference(repo_reference({'file':_argument(command,'--manifest'),
        'sha256':_argument(command,'--manifest-sha256')},root),[Path(root)/'configs'])
    index = int(match.group(1));require(index < len(manifest['entries']), 'Unknown T02 child unit')
    entry = manifest['entries'][index]
    require(arguments.pair_id == [entry['pair_id']] and str(arguments.source) == entry['source']['file']
            and arguments.pair_file == entry['pair']['file']
            and arguments.timeout_seconds == float(_argument(command,'--unit-timeout-seconds'))
            and arguments.pair_timeout_seconds == float(_argument(command,'--pair-timeout-seconds'))
            and arguments.gate_timeout_seconds == float(_argument(command,'--gate-timeout-seconds'))
            and getattr(arguments, 'sample_rejected_raw', False) is ('--sample-rejected-raw' in command),
            'T02 child source or complete grant differs from its claimed original batch')
    compressed = '--compress-test-tables' in command
    require(getattr(arguments, 'compress_test_tables', False) is compressed
            and getattr(arguments, 'compression_smoke_dual', False) is False
            and getattr(arguments, 'test_selection', None) == (_argument(command, '--test-selection') if compressed else None),
            'T02 child compression differs from its claimed original batch')
    return {**proof, 't02_child_index':index, 't02_child_entry':copy.deepcopy(entry)}


class SharedQueue:
    """One short flock transaction per claim; no global single-worker lock."""
    def __init__(self, directory, plan, plan_ref, *, root=ROOT):
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
        self.plan, self.plan_ref = copy.deepcopy(plan), dict(plan_ref)
        require(self.directory.name == plan['queue_id'] and self.directory.parent.name == 'shared_gpu_queue', 'Wrong queue root')
        for sub in ('units', 'claims', 'releases'):
            (self.directory / sub).mkdir(parents=True, exist_ok=True)
        with self.lock():
            # Every consumer of this queue sees exactly the same priorities and
            # allowed population. A later plan cannot invisibly insert tasks
            # outside an already submitted allocation's budget authorization.
            binding = self.directory / 'plan_reference.json'
            if binding.exists():
                require(json.loads(binding.read_text()) == self.plan_ref,
                        'An active shared queue cannot change its immutable plan')
            else:
                write_immutable(binding, self.plan_ref)
            state = self.read_state()
            for item in plan['external_reservations']:
                job = validate_external(item, self.root)
                for key in item['unit']['claim_keys']:
                    existing = state['keys'].get(key)
                    if existing is not None:
                        require(existing.get('type') == 'external', 'External reservation overlaps an actual shared claim')
                    state['keys'][key] = {'type': 'external', 'job_id': job, 'reservation': item}
            for unit in plan['units']:
                path = self.directory / 'units' / (unit['unit_id'] + '.json')
                if path.exists():
                    require(json.loads(path.read_text()) == unit, 'A shared unit ID changed its original command/name')
                else:
                    write_immutable(path, unit)
            self.save_state(state)

    @contextmanager
    def lock(self):
        with (self.directory / 'queue.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def read_state(self):
        path = self.directory / 'state.json'
        return json.loads(path.read_text()) if path.exists() else {'version': VERSION, 'keys': {}, 'units': {}}

    def save_state(self, state):
        path = self.directory / 'state.json';temp = path.with_name('state.tmp.' + str(os.getpid()))
        with temp.open('x') as stream:
            json.dump(state, stream, sort_keys=True, allow_nan=False);stream.write('\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(temp, path)

    def available(self, state):
        return sorted([unit for unit in self.plan['units'] if unit['unit_id'] not in state['units']
                       and not any(key in state['keys'] for key in unit['claim_keys'])],
                      key=lambda unit: (-unit['priority'], self.plan['units'].index(unit)))

    def has_pending(self):
        with self.lock():
            return bool(self.available(self.read_state()))

    def claim(self, allocation, *, remaining_seconds):
        with self.lock():
            state = self.read_state();pending = self.available(state)
            if not pending:
                return None
            # Do not bypass a higher-priority full grant to run lower-priority work.
            unit = pending[0]
            if unit['timeout_seconds'] + 45 > remaining_seconds:
                return None
            now, mono = time.time(), time.monotonic()
            body = {**copy.deepcopy(allocation), 'version': CLAIM_VERSION, 'unit_id': unit['unit_id'],
                    'name': unit['name'], 'budget_group': unit['budget_group'], 'command': unit['command'],
                    'unit': unit, 'plan_reference': self.plan_ref, 'queue_root': str(self.directory),
                    'started_epoch': now, 'started_monotonic': mono}
            ref = write_immutable(self.directory / 'claims' / (unit['unit_id'] + '.json'), body)
            for key in unit['claim_keys']:
                state['keys'][key] = {'type': 'shared', 'unit_id': unit['unit_id'], 'claim_reference': ref}
            state['units'][unit['unit_id']] = {'status': 'claimed', 'claim_reference': ref}
            self.save_state(state)
            return {**body, 'claim_reference': ref}

    def finish(self, claim, release):
        require(release.get('cleanup_errors') == [] and release.get('claim_reference') == claim['claim_reference'],
                'Uncertain cleanup cannot release or finish a shared claim')
        with self.lock():
            state = self.read_state();unit_id = claim['unit_id']
            require(state['units'][unit_id] == {'status': 'claimed', 'claim_reference': claim['claim_reference']},
                    'Claim ownership changed')
            ref = write_immutable(self.directory / 'releases' / (unit_id + '.json'), release)
            state['units'][unit_id] = {'status': 'ended', 'claim_reference': claim['claim_reference'], 'release_reference': ref}
            self.save_state(state)
            return ref
