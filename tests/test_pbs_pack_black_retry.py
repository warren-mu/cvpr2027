"""Synthetic scheduler evidence; original auditor/queue-plan science are boundaries."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools import pbs_black_retry as policy
from src.tools import pbs_mixed_accounting as mixed
from src.tools import gpu_shared_claim as shared

_spec = importlib.util.spec_from_file_location('original_black_retry_fixtures',
    Path(__file__).with_name('test_pbs_black_retry.py'))
fixtures = importlib.util.module_from_spec(_spec);_spec.loader.exec_module(fixtures)


class PackEvidence(fixtures.Evidence):
    def __init__(self, root):
        super().__init__(root)
        self.name, self.new_name = 'gpu_pack_fixture', 'gpu_pack_fixture_retry1'
        self.gpu = 'GPU-00000000-0000-0000-0000-000000000001'
        self.queue = root/'scratch/shared_gpu_queue/fixture'
        self.plan = {'units': [{'unit_id':'unit1','name':'gen_test_x','budget_group':'G01','command':['original']} ]}
        self.plan_ref = self.write('configs/shared.json', self.plan, True)
        self.command = ['python3','-u','-m','src.tools.gpu_pack','--name',self.name,
            '--queue-plan','configs/shared.json','--queue-plan-sha256',self.plan_ref['sha256'],
            '--queue-root',str(self.queue),'--gpus','1','--timeout-seconds','1800','--gate-timeout-seconds','180']
        with self.planner_boundary():
            binding = mixed.validate_submission(root, self.command, self.name, 1)
        self.payload.update(name=self.name,command=self.command,budget_group='PACK',mixed_budget_authorization=binding)
        (root/'logs/pbs/original/payload.json').chmod(0o644)  # Synthetic fixture construction only.
        self.payload_ref = self.write('logs/pbs/original/payload.json', self.payload, True)
        self.gate_name = self.name+'_gate_0'
        allocation = self.gate['config']['allocation'];device = allocation['devices'][0]
        device['gpu_uuid'] = self.gpu
        allocation.update(cuda_visible_devices=self.gpu,cuda_driver_uuids=[self.gpu])
        self.gate['metrics']['devices'][0]['observation']['gpu_identity'] = copy.deepcopy(device)
        self.gate_command = ['python3','-u','-m','src.tools.render_gate','--name',self.gate_name,
            '--gpus','1','--resolution','255','--segmentation','--timeout-seconds','180.0']
        self.gate_ref = self.logged(self.gate_name,self.gate,self.gate_command,86)
        self.parent = {'config': {'pbs_jobid':self.job,'node':self.node,'gpus':1,'version':'shared_gpu_pack_v1',
            'allocated_gpu_uuids':[self.gpu],'parent_cuda_visible_devices':self.gpu,'cpu_affinity':list(range(12)),
            'parent_payload_reference':self.payload_ref,'parent_receipt_reference':self.receipt_ref,
            'plan_reference':binding['plan_reference'],'queue_plan':'configs/shared.json',
            'queue_plan_sha256':self.plan_ref['sha256'],'queue_root':str(self.queue),
            'timeout_seconds':1800.0,'gate_timeout_seconds':180.0},
            'metrics': {'kind':'gpu_shared_pack','status':'stopped_with_original_work_preserved',
                'stopping_reasons':['insufficient_remaining_time_for_full_original_unit'],
                'unit_claims':[],'unit_runs':[],'all_owned_units_ended':True,'cleanup_errors':[],'errors':[],
                'producer_exit_code':86,'downstream_allowed':False,'retired_gpu_uuids':[self.gpu],
                'gates':[{'status':'black','exit_code':86,'gpu_uuid':self.gpu,'cpu_affinity':list(range(12)),
                          'source':self.gate_ref}]}}
        self.refresh_parent()
        with self.planner_boundary():
            cost = mixed.compute_split(root,self.payload_ref,self.parent_ref,self.account_ref)
        name = self.name+'_mixed_accounting'
        cfg = {'name':name,'version':mixed.VERSION,'payload_reference':self.payload_ref,
               'pack_reference':self.parent_ref,'account_reference':self.account_ref}
        command = ['python3','-m',mixed.ACCOUNT_PRODUCER,'--name',name,'--payload',self.payload_ref['file'],
            '--payload-sha256',self.payload_ref['sha256'],'--pack-result',self.parent_ref['file'],
            '--pack-sha256',self.parent_ref['sha256'],'--accounting-result',self.account_ref['file'],
            '--accounting-sha256',self.account_ref['sha256']]
        self.mixed_report = {'config':cfg,'metrics':cost};self.mixed_command=command
        self.mixed_ref=self.logged(name,self.mixed_report,command,0)
        self.proof.update(native_kind='gpu_shared_pack',original_payload=self.payload_ref,
            original_command_unchanged=self.command,source_results=[self.parent_ref,self.gate_ref],
            mixed_accounting=self.mixed_ref,gpu_uuid=self.gpu)
        self.refresh_guard()

    def planner_boundary(self):
        # Original immutable queue science is unrelated to this scheduler fixture;
        # the actual mixed-accounting validator, canonical checker and ownset scan run.
        return patch.object(shared,'load_plan',return_value=(self.plan,self.plan_ref))

    def refresh_parent(self):
        self.parent_ref=self.logged(self.name,self.parent,self.command,86)
        self.proof['source_results']=[self.parent_ref,self.gate_ref]
        self.refresh_guard()

    def arguments(self, host='hopper-28'):
        args=super().arguments(host);args['group']='PACK';return args

    def validate(self,host='hopper-28'):
        with self.planner_boundary():return super().validate(host)


class PackBlackRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.e=PackEvidence(self.root)

    def test_real_mixed_zero_business_and_common_native_black_chain(self):
        actual=self.e.validate()
        self.assertEqual(actual['requested_host'],'hopper-28')
        self.assertEqual(actual['version'],policy.VERSION)
        payload=dict(self.e.payload,name=self.e.new_name,command=self.e.arguments()['command'],
            requested_host='hopper-28',retry_of=self.e.job,black_retry_host_authorization=actual)
        with self.e.planner_boundary():self.assertEqual(policy.validate_payload_host(self.root,payload),actual)

    def test_only_name_can_change_and_four_gpu_pack_cannot_pin(self):
        for flag,value in (('--queue-root','/other'),('--queue-plan-sha256','0'*64),('--gpus','4'),('--timeout-seconds','1900')):
            args=self.e.arguments();args['command'][args['command'].index(flag)+1]=value
            with self.subTest(flag=flag),self.e.planner_boundary(),self.assertRaises(ValueError):
                policy.validate_request_host(self.root,**args)
        for extra in ({'gpus':4},{'host':'hopper-22'},{'group':'G01'}):
            args=self.e.arguments();args.update(extra)
            with self.e.planner_boundary(),self.assertRaises(ValueError):policy.validate_request_host(self.root,**args)
        with self.e.planner_boundary(),self.assertRaisesRegex(ValueError,'current two-host'):
            policy.validate_request_host(self.root,**self.e.arguments('hopper-22'),
                _recorded_version=policy.LEGACY_VERSION,_recorded_rotation_index=0)

    def test_any_started_claim_or_unit_cleanup_or_nonblack_gate_rejects(self):
        original=copy.deepcopy(self.e.parent)
        cases=[('unit_claims',[{'unit_id':'unit1'}]),('unit_runs',[{'unit_id':'unit1'}]),
            ('cleanup_errors',['unknown']),('errors',['other']),('all_owned_units_ended',False),
            ('retired_gpu_uuids',[]),('gates',[])]
        for key,value in cases:
            self.e.parent=copy.deepcopy(original);self.e.parent['metrics'][key]=value;self.e.refresh_parent()
            with self.subTest(key=key),self.assertRaises(ValueError):self.e.validate()
        self.e.parent=copy.deepcopy(original);self.e.parent['metrics']['gates'][0]['status']='healthy';self.e.refresh_parent()
        with self.assertRaises(ValueError):self.e.validate()

    def test_hidden_actual_claim_rejected_by_original_mixed_ownership_scan(self):
        self.e.write('scratch/shared_gpu_queue/fixture/claims/unit1.json',
            {'unit_id':'unit1','pbs_jobid':self.e.job,'parent_payload_reference':self.e.payload_ref},True)
        with self.assertRaisesRegex(ValueError,'omits or invents'):self.e.validate()

    def test_missing_or_tampered_mixed_cost_cannot_authorize_retry(self):
        self.e.proof.pop('mixed_accounting');self.e.refresh_guard()
        with self.assertRaises(ValueError):self.e.validate()
        self.e.proof['mixed_accounting']=self.e.mixed_ref
        path=self.root/self.e.mixed_ref['file'];path.write_text(path.read_text()+' ');self.e.refresh_guard()
        with self.assertRaisesRegex(ValueError,'SHA'):self.e.validate()

    def test_unrelated_gate_command_and_allocation_do_not_pass(self):
        self.e.gate_command[-1]='181.0'
        self.e.gate_ref=self.e.logged(self.e.gate_name,self.e.gate,self.e.gate_command,86)
        self.e.parent['metrics']['gates'][0]['source']=self.e.gate_ref;self.e.refresh_parent()
        with self.assertRaisesRegex(ValueError,'gate command'):self.e.validate()


    def test_runtime_revalidates_pinned_payload_but_leaves_unpinned_path_unchanged(self):
        from src.tools import gpu_pack as pack
        def run(payload, directory, job, *, error=None):
            self.e.write(directory+'/payload.json',payload,True)
            self.e.write(directory+'/submission.json',{'returncode':0,'stdout':job+'\n'})
            env={'PBS_JOBID':job,'CVPR_SCRATCH':str(self.root/'scratch')}
            with patch.object(pack,'ROOT',self.root), self.e.planner_boundary(), \
                 patch.object(policy,'validate_payload_host',wraps=policy.validate_payload_host) as check:
                args=(payload['name'],self.e.plan_ref,1,env,payload['command'][4:],1800)
                if error:
                    with self.assertRaises(ValueError):pack.submission_binding(*args)
                else:
                    actual=pack.submission_binding(*args)
                    self.assertEqual(actual[0],payload)
                self.assertEqual(check.call_count,int(payload.get('requested_host') is not None))
        unpinned=copy.deepcopy(self.e.payload)
        unpinned['environment']={'CVPR_SCRATCH':str(self.root/'scratch')}
        run(unpinned,'logs/pbs/'+unpinned['name']+'-runtime',self.e.job)
        authorization=self.e.validate()
        pinned=dict(unpinned,name=self.e.new_name,command=self.e.arguments()['command'],
            requested_host='hopper-28',retry_of=self.e.job,black_retry_host_authorization=authorization)
        location='logs/pbs/'+pinned['name']+'-runtime'
        run(pinned,location,'709900.hopper-m-02')
        # The exact original evidence is checked again at launch, not a boolean.
        target=self.root/location/'payload.json';target.chmod(0o644)
        pinned['black_retry_host_authorization']['original_gpu_uuid']='altered'
        run(pinned,location,'709900.hopper-m-02',error=True)


if __name__=='__main__':unittest.main()
