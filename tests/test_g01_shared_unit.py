"""CPU-only shared ticket/source identity fixtures; no scientific pass fabricated."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from src.tools import g01_ticket_bank as bank
from src.tools import g01_shared_unit as units
from src.tools import g01_quota_plan as quota
from src.tools import g01_collect_parallel as collect
from src.tools import g01_parallel as parent
from test_g01_quota import counts, plan, write


class Fixture:
    def __init__(self, root):
        self.root = root
        scenes = [f'FloorPlan{base+n}' for n in range(21,31) for base in (0,200,300,400)]
        self.base = plan('test', scenes=scenes, caps=counts(look_stay=1, look_flip=1))
        self.base.update(version=quota.CONFIG_VERSION, confirmed_configurations=[])
        self.base_ref = write(root/'configs/base.json', self.base)
        pieces, cursor = [], 0
        for i,size in enumerate(bank.SIZES,1):
            pieces.append({'slice_id':f's{i:02d}', 'parent_run_name':f'gen_test_shared_s{i:02d}',
                           'scenes':scenes[cursor:cursor+size]})
            cursor += size
        self.bank = {'version':bank.VERSION,'base_plan_reference':self.base_ref,'units':pieces}
        self.bank_ref = write(root/'configs/bank.json',self.bank)
        self.value = bank.expected_member(self.bank,self.bank_ref,'s01',root=root)
        self.ref = write(root/'configs/member.json',self.value)
        self.benchmark = write(root/'results/benchmark.json', {'fixture': 'not scientific'})
        name=self.value['parent_run_name']; scenes=pieces[0]['scenes']; timeout=collect.required_timeout(scenes,6,'test')
        identity, keys = bank.scientific_identity(self.value,root=root)
        identity.update(benchmark_source=self.benchmark,timeout_seconds=timeout,
                        producer_module=units.MODULE,cpu_cores=12,gpus=1)
        self.unit = {'name':name,'producer_module':units.MODULE,'budget_group':'G01','gpus':1,'cpu_cores':12,
            'timeout_seconds':timeout,'identity':identity,'claim_keys':keys,
            'command':['python3','-u','-m',units.MODULE,'--name',name,'--scenes',*scenes,
                '--concurrency','6','--benchmark-source',self.benchmark['file'],'--timeout-seconds',str(timeout),
                '--partition','test','--quota-plan',self.ref['file'],'--quota-plan-sha256',self.ref['sha256']],
            'references':{'quota_plan':self.ref,'benchmark_source':self.benchmark,
                          'g01_ticket_bank':self.bank_ref,'g01_ticket_slice':'s01'}}
        self.claim = {'unit':self.unit,'pbs_jobid':'700001.hopper-m-02','node':'fixture-node',
            'gpu_uuid':'GPU-11111111-2222-3333-4444-555555555555','cpu_affinity':list(range(12))}
        self.claim_ref=write(root/'scratch/claim.json', self.claim)
        self.proof={**self.claim,'claim_reference':self.claim_ref}


class TicketBankTests(unittest.TestCase):
    def test_seven_units_share_one_capacity_and_do_not_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder)); keys=[]
            for row in f.bank['units']:
                value=bank.expected_member(f.bank,f.bank_ref,row['slice_id'],root=f.root)
                checked=quota.validate_structure(value)
                self.assertEqual(checked['new_tickets_by_kind']['look_stay']+
                    checked['reserved_upper_bound_by_kind']['look_stay'],40)
                self.assertEqual(value['confirmed_by_kind'],counts())
                keys.extend(bank.scientific_identity(value,root=f.root)[1])
            self.assertEqual(len(keys),40);self.assertEqual(len(set(keys)),40)

    def test_member_cannot_remove_reservations_inflate_caps_change_cutoff_or_scene(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder))
            for edit in ('reservation','caps','cutoff','name','scene'):
                value=copy.deepcopy(f.value)
                if edit=='reservation':value['outstanding_reservations']=[]
                elif edit=='caps':value['assignments'][0]['actual_output_caps']['look_stay']=2
                elif edit=='cutoff':value['count_basis_sources'][0]['sha256']='b'*64
                elif edit=='name':value['parent_run_name']+='retry'
                else:value['assignments'][0]['scene']='FloorPlan22'
                with self.assertRaises(ValueError):bank.verify_member(value,root=f.root)

    def test_count_authority_is_original_base_and_arbitrary_reservation_still_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder))
            proof={'confirmed_by_scene':{},'confirmed_by_kind':counts(),
                   'configuration_selection':{'seen_configurations':[]}}
            with patch.object(quota,'ROOT',f.root), patch.object(quota,'reconstruct_test_inventory',return_value=proof) as native:
                actual=quota.authenticate_count_basis(f.value)
                native.assert_called_once_with(f.base['count_basis_sources'],configuration=True,authority=quota.FULL_AUTHORITY)
                self.assertEqual(actual['shared_ticket_bank']['allocation_id'],bank.base_identity(f.base))
                old=copy.deepcopy(f.value);old.pop('shared_ticket_bank')
                with self.assertRaisesRegex(ValueError,'not implemented'):quota.authenticate_count_basis(old)

    def test_bank_hash_order_duplicates_and_role_names_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder))
            for edit in ('order','duplicate','version','basehash'):
                value=copy.deepcopy(f.bank)
                if edit=='order':value['units'][0]['scenes'].reverse()
                elif edit=='duplicate':value['units'][1]['parent_run_name']=value['units'][0]['parent_run_name']
                elif edit=='version':value['version']='unknown'
                else:value['base_plan_reference']['sha256']='0'*64
                with self.assertRaises(ValueError):bank.validate_bank(value,root=f.root)
            renamed=copy.deepcopy(f.base);renamed['parent_run_name']='gen_test_technical_retry'
            self.assertEqual(bank.base_identity(renamed),bank.base_identity(f.base))


class UnitTests(unittest.TestCase):
    def test_exact_unit_and_external_original_single_share_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder)); identity,keys=units.validate_unit(f.unit,f.root)
            self.assertEqual((identity,keys),(f.unit['identity'],f.unit['claim_keys']))
            external=copy.deepcopy(f.unit);old=copy.deepcopy(f.base)
            old.update(parent_run_name=f.value['parent_run_name'],assignments=f.value['assignments'])
            ref=write(f.root/'configs/original_single.json',old)
            external['references']['quota_plan']=ref
            external['command'][-3]=ref['file'];external['command'][-1]=ref['sha256']
            self.assertEqual(units.validate_unit(external,f.root,external=True),(identity,keys))
            with self.assertRaisesRegex(ValueError,'Runnable'):units.validate_unit(external,f.root)
            with self.assertRaisesRegex(ValueError,'External'):units.validate_unit(f.unit,f.root,external=True)

    def test_changed_command_resource_or_manifest_ref_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder))
            for edit in ('gpu','cpu','timeout','producer','benchmark','quota'):
                unit=copy.deepcopy(f.unit)
                if edit=='gpu':unit['gpus']=2
                elif edit=='cpu':unit['cpu_cores']=24
                elif edit=='timeout':unit['timeout_seconds']-=1
                elif edit=='producer':unit['producer_module']='other'
                elif edit=='benchmark':unit['references']['benchmark_source']['sha256']='0'*64
                else:unit['references']['quota_plan']['sha256']='0'*64
                with self.assertRaises(ValueError):units.validate_unit(unit,f.root)

    def test_shared_bank_requires_claim_legacy_standalone_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder))
            with self.assertRaisesRegex(ValueError,'atomic'):
                units.runtime_claim(root=f.root,name=f.unit['name'],job='job',command=[],quota_reference=f.ref,environment={})
            self.assertIsNone(units.runtime_claim(root=f.root,name='gen_test_old',job='job',command=[],
                                                 quota_reference=f.base_ref,environment={}))

    def test_live_and_offline_claim_keep_original_allocation_and_unit(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder))
            from src.tools import gpu_shared_claim as shared_module
            env={'CVPR_SHARED_CLAIM_FILE':f.claim_ref['file'],'CVPR_SHARED_CLAIM_SHA256':f.claim_ref['sha256']}
            plan={'parent_run_name':f.unit['name'],'pbs_jobid':f.proof['pbs_jobid'],'node':f.proof['node'],
                'gpu_uuid':f.proof['gpu_uuid'],'cpu_affinity':f.proof['cpu_affinity'],
                'scenes':[row['scene'] for row in f.value['assignments']], 'stage_index':0,'concurrency':6,
                'g01_partition':'test','shared_queue_claim':f.proof,'scratch':str(f.root/'scratch')}
            with patch.object(shared_module,'validate_runtime_claim',return_value=f.proof), \
                    patch.object(shared_module,'validate_saved_claim',return_value=f.proof), \
                    patch.object(units.os,'sched_getaffinity',return_value=set(range(12)),create=True):
                self.assertEqual(units.runtime_claim(root=f.root,name=f.unit['name'],job=f.proof['pbs_jobid'],
                    command=units.arguments(f.unit['command']),quota_reference=f.ref,environment=env),f.proof)
                self.assertEqual(units.validate_admission(plan,root=f.root),f.proof)
                for key,new in (('gpu_uuid','wrong'),('cpu_affinity',list(range(1,13))),('pbs_jobid','wrong'),
                                ('scenes',plan['scenes'][::-1]),('stage_index',1)):
                    changed=copy.deepcopy(plan);changed[key]=new
                    with self.assertRaises(ValueError):units.validate_admission(changed,root=f.root)

    def test_actual_shared_helper_binds_original_pack_receipt_mask_cpu_and_scratch(self):
        # Real immutable helper path; only native rendering is outside this CPU fixture.
        from src.tools import gpu_shared_claim as shared
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));unit=copy.deepcopy(f.unit)
            unit.update(priority=300,unit_id=shared.canonical_hash(unit['identity']))
            shared_plan={'version':shared.VERSION,'queue_id':'g01_fixture',
                'priority_order':list(shared.PRIORITIES),'units':[unit],'external_reservations':[]}
            plan_ref=shared.write_immutable(f.root/'configs/shared.json',shared_plan)
            scratch=f.root/'scratch'
            payload={'name':'gpu_pack_fixture','budget_group':'PACK','task_id':'T01','queue':'autox',
                'requested_gpu_count':4,'environment':{'CVPR_SCRATCH':str(scratch)},
                'mixed_budget_authorization':{'plan_reference':plan_ref,'possible_groups':['G01']}}
            payload_ref=shared.write_immutable(f.root/'logs/pbs/pack/payload.json',payload)
            receipt_ref=shared.write_immutable(f.root/'logs/pbs/pack/submission.json',
                {'returncode':0,'stdout':f.proof['pbs_jobid']+'\n'})
            mask=','.join([f.proof['gpu_uuid']]+[f'GPU-{i:08d}-2222-3333-4444-555555555555' for i in (2,3,4)])
            raw={'version':shared.CLAIM_VERSION,'unit':unit,'unit_id':unit['unit_id'],'name':unit['name'],
                'budget_group':'G01','command':unit['command'],'plan_reference':plan_ref,
                'parent_payload_reference':payload_ref,'parent_receipt_reference':receipt_ref,
                'parent_cuda_visible_devices':mask,'queue_root':str(scratch/'shared_gpu_queue/g01_fixture'),
                **{key:f.proof[key] for key in ('pbs_jobid','node','gpu_uuid','cpu_affinity')}}
            ref=shared.write_immutable(scratch/'shared_gpu_queue/g01_fixture/claims'/('unit.json'),raw)
            env={'CVPR_SHARED_CLAIM_FILE':ref['file'],'CVPR_SHARED_CLAIM_SHA256':ref['sha256'],
                'CVPR_SCRATCH':str(scratch),'CUDA_VISIBLE_DEVICES':f.proof['gpu_uuid'],
                'CVPR_QUEUE_PARENT_MASK':mask}
            with patch.object(units.socket,'gethostname',return_value=f.proof['node']), \
                    patch.object(units.os,'sched_getaffinity',return_value=set(range(12)),create=True):
                proof=units.runtime_claim(root=f.root,name=unit['name'],job=f.proof['pbs_jobid'],
                    command=units.arguments(unit['command']),quota_reference=f.ref,environment=env)
                admission={'shared_queue_claim':proof,'scratch':str(scratch),'parent_run_name':unit['name'],
                    'scenes':[row['scene'] for row in f.value['assignments']], 'stage_index':0,'concurrency':6,
                    'g01_partition':'test',**{key:proof[key] for key in ('pbs_jobid','node','gpu_uuid','cpu_affinity')}}
                self.assertEqual(units.validate_admission(admission,root=f.root),proof)
                for key,new in (('CVPR_QUEUE_PARENT_MASK',f.proof['gpu_uuid']),
                                ('CVPR_SCRATCH',str(f.root/'elsewhere')),('CUDA_VISIBLE_DEVICES',mask)):
                    changed={**env,key:new}
                    with self.assertRaises((ValueError,FileNotFoundError)):
                        units.runtime_claim(root=f.root,name=unit['name'],job=f.proof['pbs_jobid'],
                            command=units.arguments(unit['command']),quota_reference=f.ref,environment=changed)
                with patch.object(units.os,'sched_getaffinity',return_value=set(range(1,13)),create=True):
                    with self.assertRaises(ValueError):
                        units.runtime_claim(root=f.root,name=unit['name'],job=f.proof['pbs_jobid'],
                            command=units.arguments(unit['command']),quota_reference=f.ref,environment=env)
                # Exact original receipt bytes are checked again, not a saved PASS boolean.
                Path(receipt_ref['file']).chmod(0o644)
                Path(receipt_ref['file']).write_text('{}')
                with self.assertRaises(ValueError):units.validate_admission(admission,root=f.root)

    def test_original_parent_hooks_restored_and_claim_propagates_to_plan_shard(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder)); proof={'source':{'file':'fixture'}}
            bundle={'plan':f.value,'reference':f.ref,'arithmetic':quota.validate_structure(f.value),'count_basis_proof':{}}
            originals={key:getattr(parent,key) for key in ('create_plan','submission_identity','make_shard')}
            with patch.object(collect,'submission_identity',return_value={'shared_queue_claim':f.proof}), \
                    patch.object(parent,'create_plan',return_value='created') as creator, \
                    patch.object(parent,'make_shard',return_value={}),patch.object(quota,'ROOT',f.root):
                with collect.installed(scenes=f.bank['units'][0]['scenes'],concurrency=6,benchmark='benchmark',
                        proof=proof,partition='test',explicit_partition=True,quota=bundle):
                    parent.submission_identity(f.root,f.unit['name'],f.proof['pbs_jobid'],
                        collect.runner_timeout(f.bank['units'][0]['scenes'],6,'test'))
                    parent.create_plan('directory')
                    self.assertEqual(creator.call_args.kwargs['shared_queue_claim'],f.proof)
                    shard=parent.make_shard(f.unit['name']+'_c6_floorplan21','FloorPlan21')
                    self.assertEqual(shard['shared_queue_claim'],f.proof)
            for key,value in originals.items():self.assertIs(getattr(parent,key),value)


if __name__=='__main__':unittest.main()
