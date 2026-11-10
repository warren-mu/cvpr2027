"""Original native pilot shape, real PNG/accounting validation; auditor is synthetic."""
import copy
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from PIL import Image
import test_pbs_black_retry as original
from src.tools import pbs_black_retry as policy


class PilotEvidence(original.Evidence):
    def __init__(self, root, arm='T4', budget_group='E2Epilot'):
        super().__init__(root)
        self.name = 'smoke_e2e_pilot_' + arm + '_s0_01'
        self.new_name = self.name + '_retry1'
        self.manifest = {'file': '/fixed/manifest.json', 'sha256': '1'*64}
        self.holdout = {'file': '/fixed/holdout.json', 'sha256': '2'*64}
        self.command = ['python3','-u','-m','src.tools.smoke_e2e_pilot','run','--name',self.name,
            '--arm',arm,'--manifest',json.dumps(self.manifest),'--holdout-reference',json.dumps(self.holdout),
            '--model-source',json.dumps({'file':'results/model.json','sha256':'3'*64}),
            '--pilot-spec','configs/smoke_e2e_pilot.json','--timeout-seconds','13111','--gate-timeout-seconds','180']
        if budget_group == 'E2EpilotT2':
            self.command += ['--budget-group', 'E2EpilotT2']
        self.payload.update(name=self.name,command=self.command,budget_group=budget_group,
                            requested_wall_seconds=13175,environment={'CVPR_SCRATCH':'/scratch/fixture'})
        (root/self.payload_ref['file']).chmod(0o644)
        self.payload_ref=self.write(self.payload_ref['file'],self.payload,True)
        gate_name=self.name+'_gate'
        self.gate_command=['/scratch/fixture/envs/bin/python3','-u','-m','src.tools.render_gate','--name',gate_name,
            '--gpus','1','--resolution','255','--segmentation','--timeout-seconds','180.0']
        self.gate_ref=self.logged(gate_name,self.gate,self.gate_command,86)
        self.parent={'config':{'name':self.name,'operation':'run','version':'smoke_e2e_pilot_driver_v1',
            'arm':arm,'updates':{'T0':50,'T4':50,'T2':5}[arm],'gpus':1,'budget_group':budget_group,
            'episodes_per_update':64,'seed':0,'engineering_smoke':None,'critic_offload':False,
            'timeout_seconds':13111.,'gate_timeout_seconds':180.,'manifest_reference':self.manifest,
            'holdout_reference':self.holdout,'model_source':self.command[self.command.index('--model-source')+1],
            'pilot_spec':'configs/smoke_e2e_pilot.json','node':self.node,
            'identity':{'pbs_jobid':self.job,'cuda_visible_devices':self.gpu,'cpu_affinity':list(range(12)),
                        'scratch':'/scratch/fixture'}},
            'metrics':{'kind':'smoke_e2e_pilot_training','pilot_completed':False,'black_frame_detected':True,
                'producer_exit_code':86,'errors':[],'cleanup_errors':[],'training_logs':[],
                'gate':dict(self.gate_ref,exit_code=86)},'sources':[dict(self.gate_ref,exit_code=86)]}
        self.proof.update(native_kind='smoke_e2e_pilot_training',original_payload=self.payload_ref,
                          original_command_unchanged=self.command)
        self.refresh_parent()
    def refresh_parent(self):
        self.parent_ref=self.logged(self.name,self.parent,self.command,86)
        self.proof['source_results']=[self.parent_ref,self.gate_ref];self.refresh_guard()
    def arguments(self,host='hopper-28'):
        args=super().arguments(host);args.update(group=self.payload['budget_group'],wall_seconds=self.payload['requested_wall_seconds']);return args


class PilotBlackRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.e=PilotEvidence(self.root)
    def tearDown(self): self.temp.cleanup()
    def test_actual_native_shape_and_absolute_python_gate(self):
        self.assertNotIn('pbs_jobid',self.e.parent['config'])
        self.assertNotIn('status',self.e.parent['metrics'])
        binding=self.e.validate();self.assertEqual(binding['requested_host'],'hopper-28')
        p=dict(self.e.payload,name=self.e.new_name,command=self.e.arguments()['command'],
               requested_host='hopper-28',retry_of=self.e.job,black_retry_host_authorization=binding)
        self.assertEqual(policy.validate_payload_host(self.root,p),binding)
    def test_same_delta_shortening_and_runtime_replay(self):
        args=self.e.arguments();args['wall_seconds']=13164
        args['command'][args['command'].index('--timeout-seconds')+1]='13100'
        binding=policy.validate_request_host(self.root,**args)
        p=dict(self.e.payload,name=self.e.new_name,command=args['command'],requested_wall_seconds=13164,
            requested_host='hopper-28',retry_of=self.e.job,black_retry_host_authorization=binding)
        self.assertEqual(policy.validate_payload_host(self.root,p),binding)
    def test_changed_bounds_must_shorten_together_keep64_and_not_reduce_updates(self):
        for wall,seconds in [(13176,'13112'),(13164,'13111'),(13175,'13100'),(13164,'13099'),(363,'299')]:
            args=self.e.arguments();args['wall_seconds']=wall;args['command'][args['command'].index('--timeout-seconds')+1]=seconds
            with self.subTest(wall=wall,seconds=seconds),self.assertRaises(ValueError):policy.validate_request_host(self.root,**args)
        self.e.parent['config']['updates']=49;self.e.refresh_parent()
        with self.assertRaises(ValueError):self.e.validate()
    def test_every_method_or_input_change_rejected(self):
        for flag,value in [('--arm','T0'),('--manifest','{}'),('--holdout-reference','{}'),
                           ('--model-source','{}'),('--pilot-spec','other.json'),('--gate-timeout-seconds','179')]:
            args=self.e.arguments();args['command'][args['command'].index(flag)+1]=value
            with self.subTest(flag=flag),self.assertRaises(ValueError):policy.validate_request_host(self.root,**args)
        args=self.e.arguments();args['command']+=['--critic-offload']
        with self.assertRaises(ValueError):policy.validate_request_host(self.root,**args)
    def test_no_authentication_training_or_saved_actor_business_allowed(self):
        original_parent=copy.deepcopy(self.e.parent)
        for key,value in [('updates',[]),('holdout_evaluations',[]),('checkpoint_saved',False),
                          ('source_authentication_seconds',0.),('dataset_proof',{}),('actual_completed_updates',0)]:
            self.e.parent=copy.deepcopy(original_parent);self.e.parent['metrics'][key]=value;self.e.refresh_parent()
            with self.subTest(key=key),self.assertRaises(ValueError):self.e.validate()
        self.e.parent=copy.deepcopy(original_parent);self.e.parent['metrics']['training_logs']=[{'file':'started'}];self.e.refresh_parent()
        with self.assertRaises(ValueError):self.e.validate()
    def test_error_cleanup_extra_source_or_identity_mismatch_rejected(self):
        original_parent=copy.deepcopy(self.e.parent)
        for category,key,value in [('metrics','errors',['timeout']),('metrics','cleanup_errors',['unknown']),
                                   ('config','runtime',{}),('config','engineering_smoke','t4_one_update_engineering_v1')]:
            self.e.parent=copy.deepcopy(original_parent);self.e.parent[category][key]=value;self.e.refresh_parent()
            with self.subTest(key=key),self.assertRaises(ValueError):self.e.validate()
        for key,value in [('pbs_jobid','other'),('cuda_visible_devices','other'),('cpu_affinity',[0]),('scratch','/other')]:
            self.e.parent=copy.deepcopy(original_parent);self.e.parent['config']['identity'][key]=value;self.e.refresh_parent()
            with self.subTest(key=key),self.assertRaises(ValueError):self.e.validate()
        self.e.parent=copy.deepcopy(original_parent);self.e.parent['sources'].append({'file':'model'});self.e.refresh_parent()
        with self.assertRaises(ValueError):self.e.validate()
    def test_real_nonblack_png_not_just_saved_flags_rejected(self):
        p=self.root/self.e.png_refs[0]['file'];Image.new('RGB',(255,255),(1,0,0)).save(p);ref=self.e.ref(p)
        self.e.proof['captures'][0]=ref
        self.e.gate['metrics']['devices'][0]['observation']['captures'][0]['png']['sha256']=ref['sha256']
        self.e.gate_ref=self.e.logged(self.e.name+'_gate',self.e.gate,self.e.gate_command,86)
        source=dict(self.e.gate_ref,exit_code=86);self.e.parent['metrics']['gate']=source;self.e.parent['sources']=[source];self.e.refresh_parent()
        with self.assertRaisesRegex(ValueError,'decoded PNG'):self.e.validate()
    def test_original_canonical_must_remain_full_final86(self):
        self.e.account['config']['jobs'][0]['fields']['Exit_status']='1'
        self.e.proof['pbs_accounting']=self.e.logged('account',self.e.account,['python3','src/tools/pbs_usage.py'],0)
        self.e.refresh_guard()
        with self.assertRaises(ValueError):self.e.validate()
    def test_t0_and_t2_use_their_original_full_update_counts(self):
        for arm in ('T0','T2'):
            with tempfile.TemporaryDirectory() as temporary:
                evidence=PilotEvidence(Path(temporary).resolve(),arm)
                self.assertEqual(evidence.validate()['requested_host'],'hopper-28')


    def test_explicit_night_t0_t4_increase_and_runtime_replay(self):
        args=self.e.arguments();args['wall_seconds']=15600
        args['command'][args['command'].index('--timeout-seconds')+1]='15536'
        binding=policy.validate_request_host(self.root,**args)
        self.assertEqual(binding['pilot_stoploss_authorization']['revision'],'16be1c57')
        p=dict(self.e.payload,name=self.e.new_name,command=args['command'],requested_wall_seconds=15600,
            requested_host='hopper-28',retry_of=self.e.job,black_retry_host_authorization=binding)
        self.assertEqual(policy.validate_payload_host(self.root,p),binding)
        for flag,value in [('--arm','T0'),('--manifest','{}'),('--holdout-reference','{}')]:
            changed=copy.deepcopy(args);changed['command'][changed['command'].index(flag)+1]=value
            with self.subTest(flag=flag),self.assertRaises(ValueError):policy.validate_request_host(self.root,**changed)
    def test_other_increases_or_changed_margin_remain_rejected(self):
        for wall,seconds in [(15599,'15535'),(15601,'15537'),(15600,'15535'),(15600,'15537')]:
            args=self.e.arguments();args['wall_seconds']=wall;args['command'][args['command'].index('--timeout-seconds')+1]=seconds
            with self.subTest(wall=wall,seconds=seconds),self.assertRaises(ValueError):policy.validate_request_host(self.root,**args)
    def test_separate_t2_route_requires_exact_original_native_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            e=PilotEvidence(Path(temporary).resolve(),'T2','E2EpilotT2')
            # Real separate T2 allocations are constrained to0.5h by submitter.
            e.command[e.command.index('--timeout-seconds')+1]='1736'
            e.payload['requested_wall_seconds']=1800;e.parent['config']['timeout_seconds']=1736.
            (e.root/e.payload_ref['file']).chmod(0o644)
            e.payload_ref=e.write(e.payload_ref['file'],e.payload,True);e.proof['original_payload']=e.payload_ref
            e.refresh_parent();self.assertEqual(e.validate()['requested_host'],'hopper-28')
            original=copy.deepcopy(e.parent)
            for group in ('E2Epilot','G01'):
                e.parent=copy.deepcopy(original);e.parent['config']['budget_group']=group;e.refresh_parent()
                with self.subTest(group=group),self.assertRaises(ValueError):e.validate()
    def test_t2_never_receives_t0_t4_night_increase(self):
        with tempfile.TemporaryDirectory() as temporary:
            e=PilotEvidence(Path(temporary).resolve(),'T2','E2EpilotT2')
            args=e.arguments();args['wall_seconds']=15600;args['command'][args['command'].index('--timeout-seconds')+1]='15536'
            with self.assertRaises(ValueError):policy.validate_request_host(e.root,**args)
        with tempfile.TemporaryDirectory() as temporary:
            e=PilotEvidence(Path(temporary).resolve(),'T4','E2EpilotT2')
            with self.assertRaises(ValueError):e.validate()

if __name__=='__main__':unittest.main()
