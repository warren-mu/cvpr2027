"""Scheduling/cost integration using real plan, claim, release and budget code.

The fixture's only authenticity boundary is the original provenance auditor;
no native simulator, GPU or scientific PASS is represented by this test.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
import time
import unittest
from unittest.mock import patch

from src.tools import gpu_pack as pack
from src.tools import gpu_shared_claim as shared
from src.tools import pbs_mixed_accounting as mixed
from test_gpu_shared_claim import Fixture


class PackBudgetIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.f = Fixture(self.root)
        self.unit = self.f.unit(1)
        self.plan, self.plan_ref = self.f.plan([self.unit])
        self.job = '123.hopper-m-02'
        self.name = 'gpu_pack_integration'
        self.gpus = [f'GPU-{i:08d}-2222-3333-4444-555555555555' for i in range(1, 5)]
        self.command = ['python3', '-u', '-m', 'src.tools.gpu_pack', '--name', self.name,
            '--queue-plan', self.plan_ref['file'], '--queue-plan-sha256', self.plan_ref['sha256'],
            '--queue-root', str(self.f.directory), '--gpus', '4', '--timeout-seconds', '2000',
            '--gate-timeout-seconds', '180']
        self.binding = mixed.validate_submission(self.root, self.command, self.name, 4)
        self.payload = {'name':self.name, 'repo':str(self.root), 'task_id':'T01', 'budget_group':'PACK',
            'queue':'autox', 'dry_run':False, 'requested_gpu_count':4, 'requested_wall_seconds':2030,
            'command':self.command, 'environment':{'CVPR_SCRATCH':str(self.f.scratch)},
            'mixed_budget_authorization':self.binding}
        self.payload_ref = self.f.artifact('logs/pbs/pack/payload.json', self.payload)
        self.qsub = ['qsub', 'job.pbs']
        self.receipt_ref = self.f.artifact('logs/pbs/pack/submission.json',
            {'returncode':0, 'stderr':'', 'stdout':self.job+'\n', 'command':self.qsub})
        self.allocation = {'pbs_jobid':self.job, 'node':'hopper-28', 'gpu_uuid':self.gpus[0],
            'cpu_affinity':list(range(12)), 'parent_cuda_visible_devices':','.join(self.gpus),
            'parent_payload_reference':shared.repo_reference(self.payload_ref,self.root),
            'parent_receipt_reference':shared.repo_reference(self.receipt_ref,self.root)}
        self.queue = shared.SharedQueue(self.f.directory,self.plan,self.plan_ref,root=self.root)
        self.claim = self.queue.claim(self.allocation, remaining_seconds=2000)
        scripts = self.root/'scripts'
        (scripts/'pbs').mkdir(parents=True)
        (scripts/'audit_provenance.py').write_text('def audit_file(path): return {"fail": []}\n')
        source = Path(mixed.__file__).resolve().parents[2]/'scripts/pbs/submit_phase.py'
        (scripts/'pbs/submit_phase.py').write_bytes(source.read_bytes())

    def logged(self, name, data, command, code):
        data = copy.deepcopy(data)
        data['_meta']={'name':name,'end':'fixture ended','host':'hopper-28','exit_code':code,
            'command':command,'log_path':f'logs/runs/{name}.log',
            'metrics_sha256':hashlib.sha256(json.dumps(data['metrics'],sort_keys=True).encode()).hexdigest()}
        log=self.root/data['_meta']['log_path'];log.parent.mkdir(parents=True,exist_ok=True)
        log.write_text(json.dumps(data['config'],indent=2)+'\nFINAL_METRICS_JSON: '+json.dumps(data['metrics'])+'\n')
        path=self.root/'results'/f'{name}.json';path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data))
        return data,mixed.reference(path,self.root)

    def account(self, runs):
        _, pack_ref=self.logged(self.name,{'config':{'pbs_jobid':self.job,'node':'hopper-28',
            'plan_reference':self.plan_ref,'allocated_gpu_uuids':self.gpus},
            'metrics':{'kind':'gpu_shared_pack','all_owned_units_ended':True,'cleanup_errors':[],
                       'unit_runs':runs}},self.command,1)
        fields={'job_state':'F','run_count':'1','project':'CFP04-SF-125','Job_Owner':'e1583364@node',
            'Resource_List.ngpus':'4','Resource_List.ncpus':'48','Resource_List.queue_charge_rate':'2',
            'resources_used.walltime':'00:01:40','queue':'mediumx','Variable_List':'PBS_O_QUEUE=autox',
            'exec_host':'hopper-28/0*48','Submit_arguments':shlex.join(self.qsub[1:]),
            'Output_Path':'host:'+str(self.root/'logs/pbs/pack/pbs.out')}
        _, account_ref=self.logged('canonical',{'config':{'jobs':[{'job_id':self.job,'qstat_exit':0,'fields':fields}]},
            'metrics':{'complete_allocation_accounting':True,'allocation_gpu_hours_by_job':{self.job:400/3600}}},
            ['python3','src/tools/pbs_usage.py','--name','canonical',self.job],0)
        return mixed.compute_split(self.root,self.payload_ref,pack_ref,account_ref)

    def test_actual_live_claim_end_release_is_accepted_by_original_budget_chain(self):
        identity={'pbs_jobid':self.job,'node':'hopper-28','scratch':str(self.f.scratch),
                  'cuda_visible_devices':self.gpus[0],'cpu_affinity':list(range(12))}
        env={'CVPR_SHARED_CLAIM_FILE':self.claim['claim_reference']['file'],
             'CVPR_SHARED_CLAIM_SHA256':self.claim['claim_reference']['sha256'],
             'CVPR_QUEUE_PARENT_MASK':','.join(self.gpus)}
        proof=shared.validate_runtime_claim(env,identity,self.unit['producer_module'],self.unit['name'],root=self.root)
        self.assertEqual(proof['unit'],self.claim['unit'])
        self.assertEqual({key:proof['claim_reference'][key] for key in ('file','sha256')},
                         {key:self.claim['claim_reference'][key] for key in ('file','sha256')})
        data,ref=self.logged(self.unit['name'],{'config':identity,'metrics':{'cleanup_errors':[],
            'errors':['fixture scientific failure retained']}},self.unit['command'],1)
        record={'source':ref,'exit_code':1}
        with patch.object(pack,'finish_logged',return_value=(record,data)):
            ended,_=pack.end_unit(self.queue,{'claim':self.claim},[])
        result=self.account([ended])
        self.assertEqual(result['unit_count'],1)
        self.assertEqual(result['possible_groups'],['T02'])
        self.assertEqual(result['charged_gpu_hours_by_group'],{'T02':400/3600*2})
        self.assertFalse(self.queue.has_pending())
        self.assertEqual(ended['exit_code'],1)

    def test_actual_failed_popen_release_has_no_invented_child_result_and_still_costs(self):
        ended=pack.launch_failure(self.queue,self.claim,OSError('fixture Popen failed before process'))
        self.assertIsNone(ended['result_reference'])
        self.assertIs(ended['process_started'],False)
        result=self.account([ended])
        self.assertEqual(result['charged_gpu_hours_by_group'],{'T02':400/3600*2})
        self.assertEqual(result['unit_count'],1)
        self.assertFalse(self.queue.has_pending())


if __name__=='__main__':unittest.main()
