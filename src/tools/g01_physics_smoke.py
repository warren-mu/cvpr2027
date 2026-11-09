"""Supplement16 engineering smoke: one fixed test scene, full600-second grant.

This is not a formal1200-second collection or a quota contribution. The original
G01 worker, per-card gate, immutable scene assignment, journal and owned witness
cleanup are reused. The revised physics and configuration compaction must be
present together; actual eligibility is reported without rewriting child exits.
"""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import re
import signal
import sys

from src.tools import g01_parallel as parent
from src.tools import g01_quota_plan as quotas
from src.tools.g01_collect_parallel import apply_source_policy, python_arguments
from src.tools.t01_partition import sha_file

MODULE='src.tools.g01_physics_smoke'
VERSION='g01_s16_single_scene_600s_v1'
SCENE='FloorPlan21'
SCENE_SECONDS=600
AUTH_SECONDS=120
RUNNER_SECONDS=1000
PBS_SECONDS=1030
SCOPE='supplement16 engineering smoke only; one test scene; no formal collection, quota occupancy, or training permission'


def require(condition,message):
    if not condition:raise ValueError(message)


def validate_quota(plan,name):
    quotas.validate_structure(plan)
    require(plan['version']==quotas.CONFIG_VERSION and plan['partition']=='test'
            and plan['parent_run_name']==name and [row['scene'] for row in plan['assignments']]==[SCENE]
            and not plan.get('outstanding_reservations') and 'shared_ticket_bank' not in plan,
            'Engineering smoke requires its own immutable one-scene configuration plan')
    return plan


def submission_identity(repo,name,job,timeout,quota_reference):
    expected=['python3','-u','-m',MODULE,'--name',name,'--scene',SCENE,
        '--quota-plan',quota_reference['file'],'--quota-plan-sha256',quota_reference['sha256'],
        '--timeout-seconds',str(RUNNER_SECONDS)]
    require(timeout==RUNNER_SECONDS,'Smoke parent cap changed')
    matches=[]
    for folder in (repo/'logs/pbs').glob(name+'-*'):
        p,r=folder/'payload.json',folder/'submission.json'
        if not p.exists() or not r.exists():continue
        payload,receipt=json.loads(p.read_text()),json.loads(r.read_text())
        if payload.get('name')!=name or receipt.get('returncode')!=0:continue
        require(receipt.get('stdout','').strip()==job and payload.get('command')==expected
                and payload.get('task_id')=='T01' and payload.get('budget_group')=='G01'
                and payload.get('queue')=='autox' and payload.get('requested_gpu_count')==1
                and payload.get('requested_wall_seconds')==PBS_SECONDS
                and payload.get('requested_host') is None,'Smoke differs from its exact nohost1GPU short reservation')
        matches.append({'file':str(p.relative_to(repo)),'sha256':sha_file(p),
                        'requested_wall_seconds':PBS_SECONDS,'budget_group':'G01'})
    require(len(matches)==1,'Require exactly one actual smoke reservation')
    return matches[0]


def summarize_child(data):
    """Reuse final producer counters; never turn science/worker failure into PASS."""
    metrics,config=data['metrics'],data['config']
    argv=python_arguments(data['_meta']['command'],parent.policy.PRODUCER)
    require(config['scene']==SCENE and argv.count('--timeout-seconds')==1
            and float(argv[argv.index('--timeout-seconds')+1])==SCENE_SECONDS,
            'Returned engineering scene/grant changed')
    require(config.get('g01_engineering_smoke',{}).get('version')==VERSION
            and config['g01_engineering_smoke'].get('formal_collection_eligible') is False,
            'Child omitted the explicit smoke-only policy')
    counts=metrics.get('nearby_physics_counts')
    known_counts=(isinstance(counts,dict) and all(type(counts.get(key)) is int and counts[key]>=0
        for key in ('counted_attempts','both_worlds_pass_new_physics','pass_new_but_fail_original_physics')))
    # Native action timeouts and unrelated errors are not engineering success.
    # Only the exact existing bounded-scene stop messages may remain exit1.
    boundary_errors={'TimeoutError: T01v2 simulator deadline reached',
        'TimeoutError: Absolute scene deadline before fresh natural Controller',
        'TimeoutError: Worker deadline expired'}
    error=metrics.get('error')
    expected_deadline=(metrics.get('status')=='incomplete' and error in boundary_errors)
    unexpected_errors={key:copy.deepcopy(metrics[key]) for key in
        ('errors','worker_error','worker_errors','generator_error') if metrics.get(key)}
    if error and not expected_deadline:unexpected_errors['error']=copy.deepcopy(error)
    gate_passed=(metrics.get('status')!='worker_failed' and metrics.get('cleanup_errors')==[]
        and metrics.get('final_metrics_verified') is True
        and metrics.get('terminal_verified_after_cleanup') is True
        and not unexpected_errors and known_counts and counts['both_worlds_pass_new_physics']>0)
    return {'original_status':metrics.get('status'),'original_exit_code':data['_meta']['exit_code'],
            'worker_failed':metrics.get('status')=='worker_failed',
            'cleanup_errors':copy.deepcopy(metrics.get('cleanup_errors')),
            'final_metrics_verified':metrics.get('final_metrics_verified'),
            'terminal_verified_after_cleanup':metrics.get('terminal_verified_after_cleanup'),
            'physics_counts':copy.deepcopy(counts),
            'original_error':copy.deepcopy(error),'expected_scene_deadline':expected_deadline,
            'unexpected_errors':unexpected_errors,
            'engineering_gate_passed':bool(gate_passed),
            'formal_collection_eligible':False,'quota_occupancy_authorized':False,
            'scope':'original final physics counters, not independent pair acceptance'}


@contextmanager
def installed(name,quota_reference):
    """Only parent scheduling hooks; the spawned original worker reads its flags."""
    keys=('SCENES','CONCURRENCIES','PARTITION','SCENE_SECONDS','required_timeout',
          'submission_identity','make_shard','write_result','finish_logged','create_plan')
    old={key:getattr(parent,key) for key in keys}
    state={'quota':None,'proof':None,'observations':[]}
    def reservation(repo,run_name,job,timeout):
        require(run_name==name,'Smoke parent name changed')
        return submission_identity(repo,name,job,timeout,quota_reference)
    def load_quota():
        if state['quota'] is None:
            # The parent has already passed its original render gate. Preserve
            # source authentication and bound it separately from the600s grant.
            previous=signal.getsignal(signal.SIGALRM)
            require(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'Do not replace an active process timer')
            def expired(*_):raise TimeoutError('Smoke quota authentication exceeded120 seconds')
            signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,AUTH_SECONDS)
            try:
                plan,arithmetic,proof=quotas.load_plan(quota_reference)
                validate_quota(plan,name)
                state.update(quota=plan,proof={'arithmetic':arithmetic,'count_basis_proof':proof})
            finally:
                signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        return state['quota']
    def shard(child_name,scene,*args,**kwargs):
        require(scene==SCENE,'No second scene is authorized for engineering smoke')
        plan=load_quota()
        value=old['make_shard'](child_name,scene,*args,**kwargs)
        assignment=quotas.assignment(plan,quota_reference,scene,child_name)
        apply_source_policy(value,scene,'test',assignment,nearby=True,smoke=True)
        require(value['g01_engineering_smoke']['version']==VERSION
                and value['g01_policy']['scene_timeout_seconds']==SCENE_SECONDS,
                'Revised physics+explicit600s smoke policy is not installed')
        return value
    def create_plan(*args,**kwargs):
        return old['create_plan'](*args,**kwargs,nearby_physics=True,engineering_smoke=True)
    def finish(running,sources):
        row,data=old['finish_logged'](running,sources)
        if 'scene' in running and data is not None:
            state['observations'].append(summarize_child(data))
        return row,data
    def write(run_name,metrics,config):
        config.update(smoke_version=VERSION,engineering_smoke=True,formal_collection_eligible=False,
            quota_occupancy_authorized=False,pool_scope=SCOPE,quota_plan_reference=copy.deepcopy(quota_reference),
            quota_authentication=state['proof'],quota_authentication_cap_seconds=AUTH_SECONDS,
            actual_scene_assignments=1,actual_controller_upper_bound=2)
        metrics.update(kind='g01_s16_engineering_smoke',engineering_observations=copy.deepcopy(state['observations']),
            engineering_gate_passed=(len(state['observations'])==1
                and state['observations'][0]['engineering_gate_passed']
                and metrics.get('cleanup_errors')==[] and not metrics.get('error')
                and not metrics.get('errors') and not metrics.get('stop_reasons')
                and not metrics.get('unfinished_run_names')),
            formal_collection_eligible=False,quota_occupancy_authorized=False,scope=SCOPE)
        return old['write_result'](run_name,metrics,config)
    parent.SCENES,parent.CONCURRENCIES,parent.PARTITION=(SCENE,),(6,),'test'
    parent.SCENE_SECONDS=SCENE_SECONDS;parent.required_timeout=lambda:RUNNER_SECONDS
    parent.submission_identity,parent.make_shard,parent.write_result,parent.finish_logged=reservation,shard,write,finish
    parent.create_plan=create_plan
    try:yield state
    finally:
        for key,value in old.items():setattr(parent,key,value)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--scene',choices=[SCENE],required=True)
    parser.add_argument('--quota-plan',required=True)
    parser.add_argument('--quota-plan-sha256',required=True)
    parser.add_argument('--timeout-seconds',type=int,required=True)
    args=parser.parse_args(argv)
    require(re.fullmatch(r'gen_test_s16_smoke_[A-Za-z0-9_]+',args.name) and args.timeout_seconds==RUNNER_SECONDS,
            'Require an explicit smoke name and exact bounded parent grant')
    ref=quotas.reference({'file':args.quota_plan,'sha256':args.quota_plan_sha256})
    previous_argv=sys.argv
    handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        with installed(args.name,ref):
            sys.argv=[MODULE,'--name',args.name,'--timeout-seconds',str(RUNNER_SECONDS)]
            return parent.main()
    finally:
        sys.argv=previous_argv
        for sig,value in handlers.items():signal.signal(sig,value)


if __name__=='__main__':raise SystemExit(main())
