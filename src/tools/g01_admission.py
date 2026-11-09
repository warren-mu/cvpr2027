"""Fail-closed, single-allocation admission for independently owned witnesses.

Persistent leases are never reclaimed because a process dies or a lock expires.
Only an observed successful owned-worker cleanup releases a lease. This keeps
six construction Controllers plus at most two witnesses within eight instances.
"""
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import time
import uuid

from src.envs.thor_common import process_identity

VERSION = 'g01_single_gpu_witness_admission_v1'
ASSIGNMENT_KEYS = ('version', 'plan_file', 'plan_sha256', 'state_file', 'nonce',
                   'parent_run_name', 'stage_index', 'concurrency', 'lane_index', 'scene', 'child_run_name')


class AdmissionStopped(RuntimeError):
    pass


def capacity(concurrency):
    if type(concurrency) is not int or concurrency not in (2, 4, 6):
        raise ValueError('Require the declared 2/4/6 scene concurrency')
    return min(concurrency, 8 - concurrency)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def create_plan(directory, *, parent_run_name, stage_index, concurrency, assignments,
                pbs_jobid, node, gpu_uuid, cpu_affinity, scenes, partition='train', shared_queue_claim=None):
    directory = Path(directory).resolve()
    if len(cpu_affinity) != 12 or len(set(cpu_affinity)) != 12:
        raise ValueError('All scene lanes must share exactly the allocated twelve CPUs')
    if len(assignments) != len(scenes) or {a['scene'] for a in assignments} != set(scenes):
        raise ValueError('Require one explicit run assignment per scene')
    if (len({a['child_run_name'] for a in assignments}) != len(scenes)
            or any(set(a) != {'lane_index', 'scene', 'child_run_name'}
                   or type(a['lane_index']) is not int or not 0 <= a['lane_index'] < concurrency
                   for a in assignments)):
        raise ValueError('Require unique child names and explicit valid scene lanes')
    plan = {'version': VERSION, 'nonce': uuid.uuid4().hex, 'parent_run_name': parent_run_name,
        'stage_index': stage_index, 'concurrency': concurrency, 'witness_capacity': capacity(concurrency),
        'max_unity_instances': concurrency + capacity(concurrency), 'pbs_jobid': pbs_jobid,
        'node': node, 'gpu_uuid': gpu_uuid, 'cpu_affinity': list(cpu_affinity), 'scenes': list(scenes),
        'assignments': copy.deepcopy(assignments)}
    if partition == 'test':
        plan.update(g01_partition='test', per_scene_class_cap=6,
                    collection_purpose='evaluation only; never training or tuning')
    elif partition != 'train':
        raise ValueError('Require the approved train/test partition')
    if shared_queue_claim is not None:
        plan['shared_queue_claim'] = copy.deepcopy(shared_queue_claim)
        plan['scratch'] = os.environ.get('CVPR_SCRATCH')
        from src.tools.g01_shared_unit import validate_admission
        from src.envs.thor_common import ROOT
        validate_admission(plan, root=ROOT)
    validate_partition_plan(plan)
    path = directory / 'admission_plan.json'
    with path.open('x') as stream:
        json.dump(plan, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    path.chmod(0o444)
    reference = {'plan_file': str(path), 'plan_sha256': digest(path.read_bytes()),
                 'state_file': str(directory / 'admission_state.json')}
    state = {'version': VERSION, 'nonce': plan['nonce'], 'plan_sha256': reference['plan_sha256'],
             'leases': {}, 'events': [], 'poison': None}
    with Path(reference['state_file']).open('x') as stream:
        json.dump(state, stream, sort_keys=True, allow_nan=False)
    (directory / 'admission.lock').touch(exist_ok=False)
    return plan, reference


def validate_partition_plan(plan):
    """Old immutable train plans remain valid; test requires its explicit namespace."""
    from src.envs import pair_v2_g01_policy as policy
    partition = plan.get('g01_partition', 'train')
    allowed = policy.split_scenes(partition)
    scenes = plan.get('scenes', [])
    if (not scenes or len(set(scenes)) != len(scenes) or any(scene not in allowed for scene in scenes)
            or len(plan.get('assignments', [])) != len(scenes)
            or {row.get('scene') for row in plan['assignments']} != set(scenes)):
        raise ValueError('Immutable admission plan mixes or changes the declared scene partition')
    if partition == 'test':
        if (plan.get('per_scene_class_cap') != 6 or plan.get('concurrency') != 6
                or plan.get('collection_purpose') != 'evaluation only; never training or tuning'
                or not re.fullmatch(r'gen_test_[A-Za-z0-9_]+', plan.get('parent_run_name', ''))
                or any(not re.fullmatch(r'gen_test_[A-Za-z0-9_]+', row.get('child_run_name', ''))
                       for row in plan['assignments'])):
            raise ValueError('Test collection requires an explicit isolated six-lane evaluation plan')
    elif plan.get('parent_run_name', '').startswith('gen_test_'):
        raise ValueError('A test namespace cannot disguise a training admission plan')
    return partition


def child_assignment(plan, reference, scene):
    matches = [row for row in plan['assignments'] if row['scene'] == scene]
    if len(matches) != 1:
        raise ValueError('Scene is not uniquely assigned')
    return {key: value for key, value in {**reference, **matches[0], 'version': VERSION,
        'nonce': plan['nonce'], 'parent_run_name': plan['parent_run_name'],
        'stage_index': plan['stage_index'], 'concurrency': plan['concurrency']}.items()
        if key in ASSIGNMENT_KEYS}


def load_plan(assignment):
    if set(assignment) != set(ASSIGNMENT_KEYS) or assignment['version'] != VERSION:
        raise ValueError('Missing or altered explicit G01 admission assignment')
    path = Path(assignment['plan_file'])
    if not path.is_absolute() or path.is_symlink() or path.name != 'admission_plan.json':
        raise ValueError('Require the original absolute immutable stage plan')
    raw = path.read_bytes()
    if digest(raw) != assignment['plan_sha256']:
        raise ValueError('G01 admission plan bytes changed')
    plan = json.loads(raw)
    validate_partition_plan(plan)
    if (plan.get('version') != VERSION or plan.get('nonce') != assignment['nonce']
            or plan.get('witness_capacity') != capacity(assignment['concurrency'])
            or plan.get('max_unity_instances') != assignment['concurrency'] + capacity(assignment['concurrency'])
            or Path(assignment['state_file']) != path.parent / 'admission_state.json'
            or any(plan.get(key) != assignment[key] for key in ('parent_run_name', 'stage_index', 'concurrency'))
            or child_assignment(plan, {key: assignment[key] for key in ('plan_file', 'plan_sha256', 'state_file')},
                                assignment['scene']) != assignment):
        raise ValueError('G01 admission header/nonce/lane differs from the immutable plan')
    if not 0 <= assignment['lane_index'] < plan['concurrency']:
        raise ValueError('Invalid scene lane; a lane is not a GPU index')
    return plan


def validate_source_admission(config, metrics=None):
    shard = config['shard']
    assignment = shard['g01_admission']
    plan = load_plan(assignment)
    partition = plan.get('g01_partition', 'train')
    claim = plan.get('shared_queue_claim')
    if claim is not None or shard.get('shared_queue_claim') is not None:
        if claim is None or shard.get('shared_queue_claim') != claim:
            raise ValueError('G01 shard changed or dropped its shared claim binding')
        from src.tools.g01_shared_unit import validate_admission
        from src.envs.thor_common import ROOT
        validate_admission(plan, root=ROOT)
        if plan.get('scratch') != shard.get('allocation', {}).get('scratch'):
            raise ValueError('Shared G01 scratch differs from the original audited render allocation')
    quota_assignment = config.get('g01_quota_assignment')
    if quota_assignment is not None:
        from src.tools import g01_quota_plan as quotas
        quota_value, _, _ = quotas.load_plan(quota_assignment['plan_reference'], authenticate=False)
        if 'shared_ticket_bank' in quota_value and claim is None:
            raise ValueError('A bank-backed G01 source cannot omit its shared claim')
    if (shard.get('g01_partition', 'train') != partition
            or config.get('g01_partition', 'train') != partition):
        raise ValueError('Source partition differs from its immutable admission plan')
    if (assignment['scene'] != config['scene'] or assignment['scene'] != shard['scene']
            or assignment['child_run_name'] != config['name'] or assignment['child_run_name'] != shard['name']
            or shard['logical_index'] != 0 or shard['gpu_identity']['gpu_uuid'] != plan['gpu_uuid']
            or shard['child_cuda_visible_devices'] != plan['gpu_uuid']
            or shard['parent_cuda_visible_devices'] != plan['gpu_uuid']
            or shard['cpu_affinity'] != plan['cpu_affinity'] or len(set(shard['cpu_affinity'])) != 12
            or shard['pbs_jobid'] != plan['pbs_jobid'] or shard['node'] != plan['node']
            or Path(shard['raw_evidence_dir']).resolve().parent != Path(assignment['plan_file']).resolve().parent):
        raise ValueError('G01 source lane does not retain the real one-GPU/twelve-CPU allocation identity')
    if metrics is not None:
        report = metrics.get('g01_witness_admission')
        if (not isinstance(report, dict) or report.get('version') != VERSION
                or report.get('assignment') != assignment or report.get('cleanup_confirmed') is not True
                or report.get('active_leases') != [] or report.get('errors') != []):
            raise ValueError('Missing/uncertain G01 witness admission and cleanup report')
        if (not isinstance(report.get('owner'), dict)
                or any(type(report['owner'].get(key)) is not int
                       for key in ('pid', 'start_ticks', 'pgid', 'sid'))
                or not isinstance(report.get('lease_records'), list)):
            raise ValueError('Missing authenticated witness lease-owner identity')
        tokens = set()
        for record in report['lease_records']:
            if (not isinstance(record.get('token'), str) or not record['token']
                    or record.get('token') in tokens or record.get('owner') != report['owner']
                    or record.get('release_confirmed') is not True
                    or record.get('cleanup_errors') != [] or record.get('worker_exitcode') is None
                    or record.get('remaining_owned_group') is not False):
                raise ValueError('G01 witness lease lacks confirmed owned cleanup')
            identity, scope = record.get('worker_identity'), record.get('cleanup_group_identity')
            if (not isinstance(identity, dict) or not isinstance(scope, dict)
                    or scope.get('pid') != identity.get('pid') or scope.get('start_ticks') != identity.get('start_ticks')
                    or type(scope.get('pid')) is not int or scope['pid'] <= 0
                    or scope.get('pgid') != scope['pid'] or scope.get('sid') != scope['pid']):
                raise ValueError('Witness cleanup lacks an authenticated private process group')
            tokens.add(record['token'])
    return plan


class AdmissionPool:
    def __init__(self, assignment, identity_reader=None):
        self.assignment = copy.deepcopy(assignment)
        self.plan = load_plan(assignment)
        self.path = Path(assignment['state_file'])
        self.identity_reader = identity_reader or process_identity

    @contextmanager
    def transaction(self):
        with self.path.with_name('admission.lock').open('r+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            before = self.path.read_text()
            state = json.loads(before)
            if (state.get('version') != VERSION or state.get('nonce') != self.plan['nonce']
                    or state.get('plan_sha256') != self.assignment['plan_sha256']):
                raise AdmissionStopped('Admission state no longer matches its immutable plan')
            try:
                yield state
            finally:
                after = json.dumps(state, sort_keys=True, allow_nan=False)
                if after != before:
                    temporary = self.path.with_name(f'admission_state.{os.getpid()}.tmp')
                    with temporary.open('w') as stream:
                        stream.write(after)
                        stream.flush(); os.fsync(stream.fileno())
                    os.replace(temporary, self.path)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _health(self, state):
        for lease in state['leases'].values():
            owner = lease['owner']
            current = self.identity_reader(owner['pid'])
            if current is None or current.get('start_ticks') != owner['start_ticks']:
                state['poison'] = 'Lease owner exited/reused before confirmed witness cleanup'
        return state['poison']

    def health(self):
        with self.transaction() as state:
            reason = self._health(state)
            return {'poison': reason, 'active_leases': copy.deepcopy(state['leases'])}

    def acquire(self, owner, deadline):
        while time.time() < deadline:
            with self.transaction() as state:
                reason = self._health(state)
                if reason:
                    raise AdmissionStopped(reason)
                if any(row['owner'] == owner for row in state['leases'].values()):
                    raise AdmissionStopped('A scene attempted overlapping witnesses')
                if len(state['leases']) < self.plan['witness_capacity']:
                    token = uuid.uuid4().hex
                    lease = {'token': token, 'owner': copy.deepcopy(owner),
                        'assignment': copy.deepcopy(self.assignment), 'acquired_at': time.time()}
                    state['leases'][token] = lease
                    state['events'].append({'kind': 'acquire', **lease})
                    return token
            time.sleep(.05)
        raise TimeoutError('No witness admission before the scene cleanup reserve')

    def release(self, token, proof):
        if (proof.get('cleanup_errors') != [] or proof.get('worker_exitcode') is None
                or proof.get('remaining_owned_group') is not False):
            self.poison('Witness cleanup was not confirmed')
            raise AdmissionStopped('Witness cleanup was not confirmed')
        with self.transaction() as state:
            if token not in state['leases']:
                raise AdmissionStopped('Lease was already released or was never acquired')
            if state['leases'][token]['assignment'] != self.assignment:
                state['poison'] = 'A different scene attempted to release a witness lease'
                raise AdmissionStopped(state['poison'])
            state['events'].append({'kind': 'release', 'token': token, 'proof': copy.deepcopy(proof),
                                    'released_at': time.time()})
            del state['leases'][token]

    def poison(self, reason):
        with self.transaction() as state:
            state['poison'] = state['poison'] or str(reason)
            state['events'].append({'kind': 'poison', 'reason': str(reason), 'observed_at': time.time()})


def owned_group_present(worker, identity):
    if not identity or identity.get('pgid') != identity.get('sid') or identity.get('pgid') != worker.process.pid:
        return True  # No scope proof: never test or release a possibly unrelated group.
    try:
        os.killpg(identity['pgid'], 0)
        return True
    except ProcessLookupError:
        return False


@contextmanager
def witness_bridge(config):
    """Install only in a new G01 generator process, never in old producers."""
    from src.envs import pair_v2_stable_shard as stable
    validate_source_admission(config)
    assignment = config['shard']['g01_admission']
    pool = AdmissionPool(assignment)
    owner = process_identity(os.getpid())
    if owner is None:
        raise AdmissionStopped('Cannot authenticate witness lease owner')
    records, active, errors = [], {}, []
    original = stable.Worker

    class AdmittedWitness(original):
        def __init__(self, target, child_config):
            token = pool.acquire(owner, child_config['deadline'] - 10)
            self.g01_token = token
            record = {'token': token, 'owner': copy.deepcopy(owner), 'release_confirmed': False}
            records.append(record); active[token] = record
            try:
                super().__init__(target, child_config)
                record['worker_identity'] = copy.deepcopy(self.identity)
            except BaseException as error:
                message = f'Witness construction failed after admission: {type(error).__name__}: {error}'
                errors.append(message); pool.poison(message)
                raise

        def stop(self):
            if hasattr(self, 'g01_confirmed_cleanup'):
                return list(self.g01_confirmed_cleanup)
            record = active[self.g01_token]
            try:
                # A closed receive pipe must never prevent owned-worker cleanup.
                receive_error, receive_exception = None, None
                try:
                    self.receive()
                except BaseException as error:
                    receive_error = f'{type(error).__name__}: {error}'
                    receive_exception = error
                # Worker.__init__ may observe the child before its setsid().
                # Match stable PID/start ticks, then capture its actual private
                # group immediately before the existing scoped cleanup routine.
                scope = process_identity(self.process.pid)
                if (not self.identity or not scope
                        or scope['start_ticks'] != self.identity['start_ticks']
                        or scope['pgid'] != scope['sid'] or scope['pgid'] != self.process.pid):
                    scope = None
                cleanup = super().stop()
                until = time.monotonic() + 2
                remaining = owned_group_present(self, scope)
                while remaining and time.monotonic() < until:
                    time.sleep(.05); remaining = owned_group_present(self, scope)
                proof = {'cleanup_errors': cleanup, 'worker_exitcode': self.process.exitcode,
                    'worker_identity': copy.deepcopy(self.identity), 'unity_identity': copy.deepcopy(self.unity_identity),
                    'cleanup_group_identity': copy.deepcopy(scope),
                    'remaining_owned_group': remaining, 'receive_error_before_cleanup': receive_error}
                record.update(proof)
                pool.release(self.g01_token, proof)
                record['release_confirmed'] = True
                active.pop(self.g01_token)
                self.g01_confirmed_cleanup = list(cleanup)
                if receive_exception is not None and not isinstance(receive_exception, (EOFError, OSError)):
                    raise receive_exception
                return cleanup
            except BaseException as error:
                message = f'Witness cleanup uncertain: {type(error).__name__}: {error}'
                errors.append(message); pool.poison(message)
                raise

    class Bridge:
        def report(self):
            return {'version': VERSION, 'assignment': copy.deepcopy(assignment), 'owner': copy.deepcopy(owner),
                'lease_records': copy.deepcopy(records), 'active_leases': sorted(active),
                'cleanup_confirmed': not active and not errors, 'errors': list(errors)}

    stable.Worker = AdmittedWitness
    try:
        yield Bridge()
    finally:
        stable.Worker = original
        if active:
            message = 'Generator left witness bridge with unconfirmed active leases'
            errors.append(message); pool.poison(message)
