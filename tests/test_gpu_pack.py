"""Real main/claim/release control flow with synthetic subprocess boundaries.

No GPU, renderer, native scientific validation, or usage measurement is mocked
as passing science; these fixtures exercise scheduling and recorded outcomes.
"""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src.tools import gpu_pack as pack
from src.tools import gpu_shared_claim as shared
from test_gpu_shared_claim import Fixture


class Process:
    pid=123
    def __init__(self, code=0):self.returncode=code
    def poll(self):return self.returncode


class PackMainTests(unittest.TestCase):
    def run_main(self, *, gpus=1, units=2, black=(), failed=(), dirty=(), external=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        f=Fixture(temp.name);items=[f.unit(i+1) for i in range(units)]
        ext=[]
        if external:
            for unit in items:
                p=f.artifact('logs/pbs/'+unit['name']+'/payload.json',{'name':unit['name'],'command':unit['command'],
                    'task_id':'T01','budget_group':'T02','requested_gpu_count':1})
                r=f.artifact('logs/pbs/'+unit['name']+'/submission.json',{'returncode':0,'stdout':'9.hopper-m-02\n'})
                ext.append({'unit':unit,'payload_reference':p,'receipt_reference':r})
        plan,ref=f.plan(items,ext);allocation=f.parent(ref)
        gpuids=['GPU-00000000-0000-0000-0000-%012d'%(i+1) for i in range(gpus)]
        pp=Path(allocation['parent_payload_reference']['file']);pp.chmod(0o644)
        payload=json.loads(pp.read_text());payload['requested_gpu_count']=gpus
        pp.write_text(json.dumps(payload));pp.chmod(0o444)
        payload_ref=shared.reference(pp)
        queue=shared.SharedQueue(f.directory,plan,ref,root=f.root)
        launched=[]
        def launch(name,command,environment,directory,cpu_affinity):
            row={'process':Process(),'name':name,'raw_command':command,'environment':dict(environment),
                 'cpu_affinity':list(cpu_affinity)};launched.append(row);return row
        def finish(running,sources):
            gate='gate_lane' in running
            code=(86 if running['environment']['CUDA_VISIBLE_DEVICES'] in black else 0) if gate else (1 if running['name'] in failed else 0)
            source={'file':'results/'+running['name']+'.json','sha256':'b'*64}
            sources.append(source)
            cfg={'pbs_jobid':'1.hopper-m-02','node':'hopper-fixture',
                 'cuda_visible_devices':running['environment']['CUDA_VISIBLE_DEVICES'],'cpu_affinity':running['cpu_affinity']}
            data={'_meta':{'name':running['name'],'command':running['raw_command'],'exit_code':code},'config':cfg,
                  'metrics':{'cleanup_errors':['unknown child'] if running['name'] in dirty else [],
                             'black_frame_detected':code==86,'errors':[]}}
            return {'source':source,'exit_code':code},data
        api=SimpleNamespace(create_shared_queue=lambda *a:queue,launch_shared_logged=launch,
                            identity=lambda pid:{'pid':pid})
        env={'PBS_JOBID':'1.hopper-m-02','CVPR_SCRATCH':str(f.scratch),'CUDA_VISIBLE_DEVICES':','.join(gpuids)}
        args=['--name','gpu_pack_fixture','--queue-plan',ref['file'],'--queue-plan-sha256',ref['sha256'],
              '--queue-root',str(f.directory),'--gpus',str(gpus),'--timeout-seconds','4000','--gate-timeout-seconds','180']
        def live(env,identity,module,name):return shared.validate_runtime_claim(env,identity,module,name,root=f.root)
        with ExitStack() as stack:
            stack.enter_context(patch.object(pack,'ROOT',f.root))
            stack.enter_context(patch.object(pack,'load_plan',return_value=(plan,ref)))
            stack.enter_context(patch.object(pack,'submission_binding',return_value=(payload,payload_ref,allocation['parent_receipt_reference'])))
            stack.enter_context(patch.object(pack,'queue_module',return_value=api))
            stack.enter_context(patch.object(pack,'validate_runtime_claim',side_effect=live))
            stack.enter_context(patch.object(pack,'finish_logged',side_effect=finish))
            stack.enter_context(patch.object(pack,'verify_gate',side_effect=lambda d,i:'black' if d['_meta']['exit_code']==86 else 'healthy'))
            stack.enter_context(patch.object(pack,'request_stop_logged'))
            stack.enter_context(patch.object(pack,'stop_logged',return_value=[]))
            stack.enter_context(patch.object(pack.os,'sched_getaffinity',return_value=set(range(12*gpus)),create=True))
            stack.enter_context(patch.object(pack.socket,'gethostname',return_value='hopper-fixture'))
            stack.enter_context(patch.object(pack.time,'sleep'))
            stack.enter_context(patch.dict(os.environ,env,clear=True))
            stack.enter_context(redirect_stdout(io.StringIO()))
            code=pack.main(args)
        result=json.loads((f.root/'results/gpu_pack_fixture.json').read_text())
        return code,result,launched,queue

    def test_empty_shared_queue_exits_before_any_gpu_gate(self):
        code,result,launched,_=self.run_main(external=True)
        self.assertEqual(code,0);self.assertEqual(launched,[])
        self.assertEqual(result['metrics']['status'],'queue_empty_on_start')
        self.assertTrue(result['metrics']['all_owned_units_ended'])

    def test_four_gpu_lanes_use_disjoint_twelve_cpu_masks_and_original_commands(self):
        code,result,launched,_=self.run_main(gpus=4,units=4)
        self.assertEqual(code,0)
        jobs=[row for row in launched if 'claim' in row]
        self.assertEqual(len(jobs),4)
        self.assertEqual(len({row['environment']['CUDA_VISIBLE_DEVICES'] for row in jobs}),4)
        self.assertEqual(len(set().union(*(set(row['cpu_affinity']) for row in jobs))),48)
        for row in jobs:
            self.assertEqual(len(row['cpu_affinity']),12)
            self.assertEqual(row['raw_command'],row['claim']['command'])
        self.assertEqual(len(result['metrics']['unit_runs']),4)
        self.assertTrue(result['metrics']['all_owned_units_ended'])

    def test_black_lane_is_retired_and_other_lanes_keep_claiming(self):
        gpu='GPU-00000000-0000-0000-0000-000000000001'
        code,result,launched,_=self.run_main(gpus=4,units=5,black=[gpu])
        self.assertEqual(code,0)
        self.assertEqual(result['metrics']['retired_gpu_uuids'],[gpu])
        self.assertEqual(len(result['metrics']['unit_runs']),5)
        self.assertFalse(any('claim' in row and row['environment']['CUDA_VISIBLE_DEVICES']==gpu for row in launched))

    def test_real_original_unit_failure_is_retained_and_next_unit_still_runs(self):
        code,result,_,_=self.run_main(failed=['t02_export_fixture_1'])
        self.assertEqual(code,1)
        self.assertEqual([r['exit_code'] for r in result['metrics']['unit_runs']],[1,0])
        self.assertTrue(result['metrics']['all_owned_units_ended'])

    def test_unknown_cleanup_stops_new_claims_and_never_claims_complete(self):
        code,result,launched,queue=self.run_main(dirty=['t02_export_fixture_1'])
        self.assertEqual(code,1)
        self.assertFalse(result['metrics']['all_owned_units_ended'])
        self.assertTrue(result['metrics']['cleanup_errors'])
        self.assertEqual(len([r for r in launched if 'claim' in r]),1)
        self.assertTrue(queue.has_pending())

    def test_launch_failure_release_explicitly_records_no_started_process(self):
        claim={k:v for k,v in zip(('unit_id','name','budget_group','pbs_jobid','gpu_uuid','cpu_affinity',
            'claim_reference','started_epoch','started_monotonic'),('a','n','T02','1','u',list(range(12)),{},0,0))}
        queue=SimpleNamespace(finish=lambda c,r:{'file':'release','sha256':'a'*64})
        release=pack.launch_failure(queue,claim,RuntimeError('fixture Popen failure'))
        self.assertIs(release['process_started'],False)
        self.assertIsNone(release['result_reference'])
        self.assertIn('RuntimeError',release['launch_error'])


if __name__=='__main__':unittest.main()
