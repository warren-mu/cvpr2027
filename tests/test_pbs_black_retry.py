"""Synthetic CPU evidence only; original audit itself is a declared fixture boundary."""
import copy
import hashlib
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from PIL import Image
from src.tools import pbs_black_retry as policy


class Evidence:
    def __init__(self, root, previous=None):
        self.root, self.job, self.name = root, '709207.hopper-m-02', 'gen_test_black_fixture'
        self.new_name = self.name + '_retry1'
        self.node, self.gpu = 'hopper-16', 'GPU-fixture'
        self.directory = root / 'logs/pbs/original'
        self.directory.mkdir(parents=True)
        (root / 'scripts').mkdir()
        (root / 'scripts/audit_provenance.py').write_text('def audit_file(path): return {"fail": []}\n')
        self.command = ['python3', '-u', '-m', 'src.tools.g01_collect_parallel', '--name', self.name,
                        '--scenes', 'FloorPlan21', '--concurrency', '6', '--timeout-seconds', '1800']
        self.payload = {'repo': str(root), 'name': self.name, 'command': self.command,
            'dry_run': False, 'task_id': 'T01', 'budget_group': 'G01', 'queue': 'autox',
            'budget_batch': 't00b_t01_20261108', 'requested_gpu_count': 1,
            'requested_wall_seconds': 1830, 'requested_host': previous}
        self.payload_ref = self.write('logs/pbs/original/payload.json', self.payload, immutable=True)
        script = self.directory / 'job.pbs';script.write_text('original fixture job');script.chmod(0o444)
        receipt = {'returncode': 0, 'stdout': self.job + '\n', 'stderr': '',
            'command': ['qsub', '-N', 'cvpr27_fixture', '-l', 'walltime=00:30:30', '-o', str(self.directory/'pbs.out'), str(script)]}
        self.receipt_ref = self.write('logs/pbs/original/submission.json', receipt)
        fields = {'job_state': 'F', 'Exit_status': '86', 'run_count': '1', 'project': 'CFP04-SF-125',
            'Job_Owner': 'e1583364@login', 'Resource_List.ngpus': '1', 'Resource_List.ncpus': '12',
            'Resource_List.queue_charge_rate': '1', 'resources_used.walltime': '00:00:10',
            'queue': 'smallx', 'Variable_List': 'PBS_O_QUEUE=autox',
            'Submit_arguments': shlex.join(receipt['command'][1:]), 'Output_Path': 'login:'+str(self.directory/'pbs.out'),
            'exec_host': self.node+'/2*12'}
        self.account = {'config': {'jobs': [{'job_id': self.job, 'qstat_exit': 0, 'fields': fields}]},
            'metrics': {'complete_allocation_accounting': True, 'allocation_gpu_hours_by_job': {self.job:10/3600}}}
        self.account_ref = self.logged('account', self.account, ['python3', 'src/tools/pbs_usage.py','--name','account',self.job], 0)
        gate_name = self.name + '_gate';device = {'gpu_uuid': self.gpu, 'logical_index': 0}
        captures=[];self.png_refs=[]
        for index,stage in enumerate(('after_constructor','after_Pass')):
            path=root/'logs/runs'/gate_name/'render_gate'/f'{index}.png';path.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(255,255)).save(path);ref=self.ref(path);self.png_refs.append(ref)
            captures.append({'stage':stage,'png':{'path':str(path),'sha256':ref['sha256']},'valid_shape':True,
                'shape':[255,255,3],'dtype':'uint8','action_success':True,'error_message':'','black':True,
                'nonzero_rgb_values':0,'min_rgb':0,'max_rgb':0})
        self.gate={'config':{'pbs_jobid':self.job,'node':self.node,'resolution':255,'segmentation':True,
            'allocation':{'pbs_jobid':self.job,'cuda_visible_devices':self.gpu,'cuda_driver_uuids':[self.gpu],
                'cpu_affinity':list(range(12)),'devices':[device]}},'metrics':{'exit_code':86,'passed':False,
            'black_frame_detected':True,'errors':[],'cleanup_errors':[],'expected_gpu_count':1,'completed_gpu_count':1,
            'devices':[{'worker_error':None,'observation':{'gpu_identity':device,'node':self.node,'captures':captures,
                'wire_packets':[{'stage':'Pass','fresh_main_rgb_payloads':1,'cached_metadata':False}]}}]}}
        self.gate_ref=self.logged(gate_name,self.gate,['python3','-m','src.tools.render_gate'],86)
        self.parent={'config':{'pbs_jobid':self.job,'node':self.node},'metrics':{'kind':'g01_parallel_collection',
            'status':'blocked_by_render_gate','black_frame_detected':True,'cleanup_errors':[],
            **{key:[] for key in ('scene_runs','started_shards','stages','admission_final','unfinished_run_names','stop_reasons')},
            'gate':{'source':self.gate_ref}}}
        self.parent_ref=self.logged(self.name,self.parent,self.command,86)
        self.proof={'eligible_proven_black_gate_retry':True,'native_kind':'g01_parallel_collection','job_id':self.job,'original_payload':self.payload_ref,
            'original_job_script':self.ref(script),'original_receipt':self.receipt_ref,
            'original_command_unchanged':self.command,'pbs_accounting':self.account_ref,
            'source_results':[self.parent_ref,self.gate_ref],'node':self.node,'gpu_uuid':self.gpu,'captures':self.png_refs}
        self.refresh_guard()

    def ref(self,path):
        body=path.read_bytes();return {'file':str(path.relative_to(self.root)),'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)}
    def write(self,name,data,immutable=False):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
        if immutable:p.chmod(0o444)
        return self.ref(p)
    def logged(self,name,data,command,code):
        metrics=data['metrics'];log=self.root/'logs/runs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
        log.write_text(json.dumps(data['config'],indent=2)+'\nFINAL_METRICS_JSON: '+json.dumps(metrics)+'\n')
        data['_meta']={'name':name,'end':'fixture-ended','host':self.node,'exit_code':code,'command':command,
            'log_path':str(log.relative_to(self.root)), 'metrics_sha256':hashlib.sha256(json.dumps(metrics,sort_keys=True).encode()).hexdigest()}
        return self.write('results/'+name+'.json',data)
    def refresh_guard(self):
        self.guard={'config':{},'metrics':{'strict_black_gate_proof':self.proof}}
        self.guard_ref=self.logged('strict_guard',self.guard,['python3','-c','fixture'],0)
    def arguments(self,host='hopper-22'):
        cmd=list(self.command);cmd[cmd.index('--name')+1]=self.new_name
        return dict(host=host,retry_of=self.job,evidence_path=self.guard_ref['file'],evidence_sha256=self.guard_ref['sha256'],
            name=self.new_name,command=cmd,task='T01',group='G01',gpus=1,wall_seconds=1830)
    def validate(self,host='hopper-22'):
        return policy.validate_request_host(self.root,**self.arguments(host))


class BlackRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.e=Evidence(self.root)
    def tearDown(self):self.temp.cleanup()
    def test_real_black_chain_and_runtime_exact_binding(self):
        binding=self.e.validate();p=dict(self.e.payload,name=self.e.new_name,command=self.e.arguments()['command'],
            requested_host='hopper-22',retry_of=self.e.job,black_retry_host_authorization=binding)
        self.assertEqual(policy.validate_payload_host(self.root,p),binding)
        p['black_retry_host_authorization']['original_gpu_uuid']='different'
        with self.assertRaises(ValueError):policy.validate_payload_host(self.root,p)
    def test_first_nonblack_unknown_host_and_two_gpu_rejected(self):
        for changes in ({'retry_of':None},{'host':'hopper-29'},{'gpus':2},{'evidence_sha256':'0'*64}):
            args=self.e.arguments();args.update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):policy.validate_request_host(self.root,**args)
        self.e.account['config']['jobs'][0]['fields']['Exit_status']='1'
        self.e.proof['pbs_accounting']=self.e.logged('account',self.e.account,['python3','src/tools/pbs_usage.py'],0)
        self.e.refresh_guard()
        with self.assertRaises(ValueError):self.e.validate()
    def test_science_arguments_or_business_started_are_never_retried(self):
        args=self.e.arguments();args['command'][-1]='2000'
        with self.assertRaises(ValueError):policy.validate_request_host(self.root,**args)
        self.e.parent['metrics']['scene_runs']=[{'run_name':'actually_started'}]
        self.e.proof['source_results'][0]=self.e.logged(self.e.name,self.e.parent,self.e.command,86);self.e.refresh_guard()
        with self.assertRaises(ValueError):self.e.validate()
    def test_missing_cleanup_or_nonblack_actual_png_rejected(self):
        self.e.gate['metrics']['cleanup_errors']=['unresolved']
        self.e.proof['source_results'][1]=self.e.logged(self.e.name+'_gate',self.e.gate,['python3'],86);self.e.refresh_guard()
        with self.assertRaises(ValueError):self.e.validate()
        self.e.gate['metrics']['cleanup_errors']=[]
        path=self.root/self.e.png_refs[0]['file'];Image.new('RGB',(255,255),(1,0,0)).save(path)
        ref=self.e.ref(path);self.e.proof['captures'][0]=ref
        self.e.gate['metrics']['devices'][0]['observation']['captures'][0]['png']['sha256']=ref['sha256']
        gref=self.e.logged(self.e.name+'_gate',self.e.gate,['python3'],86);self.e.parent['metrics']['gate']['source']=gref
        pref=self.e.logged(self.e.name,self.e.parent,self.e.command,86);self.e.proof['source_results']=[pref,gref];self.e.refresh_guard()
        with self.assertRaisesRegex(ValueError,'decoded PNG'):self.e.validate()
    def test_original_source_byte_mutation_rejected(self):
        p=self.root/self.e.parent_ref['file'];p.write_text(p.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'SHA'):self.e.validate()
    def test_global_rotation_and_runtime_prefix_are_stable(self):
        first=self.e.validate()
        p=dict(self.e.payload,name=self.e.new_name,command=self.e.arguments()['command'],
            requested_host='hopper-22',retry_of=self.e.job,black_retry_host_authorization=first)
        self.e.write('logs/pbs/first/payload.json',p,True)
        self.e.write('logs/pbs/first/submission.json',{'returncode':0,'stdout':'709300.hopper-m-02\n'})
        with self.assertRaisesRegex(ValueError,'rotate'):self.e.validate('hopper-22')
        with self.assertRaisesRegex(ValueError,'already has'):self.e.validate('hopper-28')
        index, previous=policy.rotation(self.root)
        self.assertEqual(index,1)
        second=dict(first,rotation_index=index,rotation_previous=previous,requested_host=policy.HOSTS[index%3])
        self.assertEqual(policy.validate_payload_host(self.root,p),first)
        q=dict(p,requested_host='hopper-28',black_retry_host_authorization=second)
        self.e.write('logs/pbs/second/payload.json',q,True)
        self.e.write('logs/pbs/second/submission.json',{'returncode':0,'stdout':'709301.hopper-m-02\n'})
        index, previous=policy.rotation(self.root)
        self.assertEqual(index,2)
        self.assertEqual(policy.HOSTS[index%3],'hopper-30')
        self.assertEqual(policy.validate_payload_host(self.root,p),first)
        self.assertIsNone(policy.validate_payload_host(self.root,{'requested_host':None}))

    def test_immutable_quota_retry_only_changes_parent_identity(self):
        old={'parent_run_name':'old','seed':0,'scenes':['FloorPlan21']}
        new=dict(old,parent_run_name='new')
        a=self.e.write('configs/a.json',old,True);b=self.e.write('configs/b.json',new,True)
        argv=lambda n,r:['python3','-m','src.tools.g01_collect_parallel','--name',n,'--quota-plan',r['file'],'--quota-plan-sha256',r['sha256']]
        policy.same_retry_command(self.root,argv('old',a),argv('new',b),'old','new')
        p=self.root/b['file'];p.chmod(0o644);new['seed']=1;b=self.e.write(b['file'],new,True)
        with self.assertRaisesRegex(ValueError,'quota/scene/seed'):
            policy.same_retry_command(self.root,argv('old',a),argv('new',b),'old','new')

if __name__=='__main__':unittest.main()
