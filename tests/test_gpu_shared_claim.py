"""Scheduling-only fixtures using real immutable JSON and flock processes."""
import copy
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from src.tools import gpu_shared_claim as shared


def contend(directory, plan, ref, allocation, connection):
    queue = shared.SharedQueue(directory, plan, ref, root=Path(ref['file']).parents[1])
    claimed = queue.claim(allocation, remaining_seconds=2000)
    connection.send(None if claimed is None else claimed['unit_id'])
    connection.close()


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.scratch = self.root / 'scratch'
        self.directory = self.scratch / 'shared_gpu_queue' / 'fixture'
        self.gpu = 'GPU-00000000-0000-0000-0000-000000000001'

    def artifact(self, relative, value):
        path = self.root / relative
        ref = shared.write_immutable(path, value)
        return {**ref, 'file':relative}

    def unit(self, index, *, pair_index=None):
        pair_index = index if pair_index is None else pair_index
        pair = self.root / f'data/world_pairs/v2/p{pair_index}.json'
        if not pair.exists():self.artifact(str(pair.relative_to(self.root)), {'pair_id':f'floorplan1_{pair_index:05d}'})
        source = self.root / f'results/source{pair_index}.json'
        if not source.exists():self.artifact(str(source.relative_to(self.root)), {'fixture':'not a science verdict'})
        manifest = self.artifact(f'configs/m{index}.json', {'version':'t02_explicit_source_units_v1','partition':'train',
            'entries':[{'scene':'FloorPlan1','field_of_view':60,'pair_id':f'floorplan1_{pair_index:05d}',
                        'proposal_identity':f'{pair_index:064x}',
                        'pair':{**shared.reference(pair),'file':str(pair.relative_to(self.root))},
                        'source':{**shared.reference(source),'file':str(source.relative_to(self.root))}}]})
        name = f't02_export_fixture_{index}'
        command = ['python3','-u','-m','src.tools.t02_export_parallel','--name',name,'--manifest',manifest['file'],
            '--manifest-sha256',manifest['sha256'],'--concurrency','6','--timeout-seconds','1200',
            '--unit-timeout-seconds','900','--pair-timeout-seconds','660','--gate-timeout-seconds','180']
        unit = {'name':name,'command':command,'producer_module':'src.tools.t02_export_parallel',
                'budget_group':'T02','priority':200,'gpus':1,'cpu_cores':12,'timeout_seconds':1200,'references':{'manifest':manifest}}
        identity, keys = shared.t02_identity(unit,self.root)
        unit.update(identity=identity,claim_keys=keys,unit_id=shared.canonical_hash(identity))
        return unit

    def plan(self, units, external=()):
        plan = {'version':shared.VERSION,'queue_id':'fixture','priority_order':list(shared.PRIORITIES),
                'units':units,'external_reservations':list(external)}
        path = self.root / 'configs/queue.json'
        ref = shared.write_immutable(path,plan)
        loaded, ref = shared.load_plan(path,ref['sha256'],root=self.root)
        return loaded,ref

    def parent(self, plan_ref, *, job='1.hopper-m-02'):
        payload = {'name':'gpu_pack_fixture','repo':str(self.root),'task_id':'T01','budget_group':'PACK','queue':'autox',
            'requested_gpu_count':1,'environment':{'CVPR_SCRATCH':str(self.scratch)},
            'mixed_budget_authorization':{'plan_reference':plan_ref,'possible_groups':['T02']}}
        p=self.artifact('logs/pbs/parent/payload.json',payload)
        r=self.artifact('logs/pbs/parent/submission.json',{'returncode':0,'stdout':job+'\n'})
        return {'pbs_jobid':job,'node':'hopper-fixture','gpu_uuid':self.gpu,'cpu_affinity':list(range(12)),
            'parent_cuda_visible_devices':self.gpu,'parent_payload_reference':shared.repo_reference(p,self.root),
            'parent_receipt_reference':shared.repo_reference(r,self.root),'owner_identity':{'pid':os.getpid()}}


class SharedClaimTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.fixture=Fixture(self.temporary.name)

    def test_actual_live_and_saved_claim_bind_original_bytes_and_allocation(self):
        f=self.fixture;unit=f.unit(1);plan,ref=f.plan([unit]);allocation=f.parent(ref)
        queue=shared.SharedQueue(f.directory,plan,ref,root=f.root)
        claim=queue.claim(allocation,remaining_seconds=2000)
        identity={'pbs_jobid':allocation['pbs_jobid'],'node':allocation['node'],'scratch':str(f.scratch),
                  'cuda_visible_devices':f.gpu,'cpu_affinity':list(range(12))}
        env={'CVPR_SHARED_CLAIM_FILE':claim['claim_reference']['file'],
             'CVPR_SHARED_CLAIM_SHA256':claim['claim_reference']['sha256'],'CVPR_QUEUE_PARENT_MASK':f.gpu}
        actual=shared.validate_runtime_claim(env,identity,unit['producer_module'],unit['name'],root=f.root)
        saved=shared.validate_saved_claim(actual['claim_reference'],identity,unit['producer_module'],unit['name'],root=f.root)
        self.assertEqual(actual,saved)
        for key,value in [('scratch',str(f.root/'elsewhere')),('cuda_visible_devices',f.gpu[:-1]+'2'),('cpu_affinity',list(range(1,13)))]:
            with self.subTest(key=key),self.assertRaises((ValueError,FileNotFoundError)):
                shared.validate_runtime_claim(env,{**identity,key:value},unit['producer_module'],unit['name'],root=f.root)

    def test_original_single_pair_child_is_bound_to_exact_claimed_entry(self):
        f=self.fixture;unit=f.unit(1);plan,ref=f.plan([unit]);allocation=f.parent(ref)
        claim=shared.SharedQueue(f.directory,plan,ref,root=f.root).claim(allocation,remaining_seconds=2000)
        identity={'pbs_jobid':allocation['pbs_jobid'],'node':allocation['node'],'scratch':str(f.scratch),
                  'cuda_visible_devices':f.gpu,'cpu_affinity':list(range(12))}
        env={'CVPR_SHARED_CLAIM_FILE':claim['claim_reference']['file'],
             'CVPR_SHARED_CLAIM_SHA256':claim['claim_reference']['sha256'],'CVPR_QUEUE_PARENT_MASK':f.gpu}
        args=SimpleNamespace(name=unit['name']+'_u0000',pair_id=['floorplan1_00001'],
            source=Path('results/source1.json'),pair_file='data/world_pairs/v2/p1.json',
            timeout_seconds=900.0,pair_timeout_seconds=660.0,gate_timeout_seconds=180.0)
        self.assertEqual(shared.validate_t02_child_claim(env,identity,args,root=f.root)['t02_child_index'],0)
        for key,value in [('name',unit['name']+'_u0001'),('pair_id',['floorplan1_99999']),
                          ('source',Path('results/another.json')),('pair_timeout_seconds',659.0)]:
            bad=copy.deepcopy(args);setattr(bad,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):
                shared.validate_t02_child_claim(env,identity,bad,root=f.root)
        self.assertIsNone(shared.validate_t02_child_claim({},identity,args,root=f.root))

    def test_same_queue_cannot_silently_change_plan_or_priority_population(self):
        f=self.fixture;plan,ref=f.plan([f.unit(1)])
        shared.SharedQueue(f.directory,plan,ref,root=f.root)
        with self.assertRaisesRegex(ValueError,'immutable plan'):
            shared.SharedQueue(f.directory,plan,{**ref,'sha256':'a'*64},root=f.root)

    def test_two_actual_processes_can_claim_one_scientific_unit_only_once(self):
        f=self.fixture;plan,ref=f.plan([f.unit(1)]);allocation=f.parent(ref)
        shared.SharedQueue(f.directory,plan,ref,root=f.root)
        ctx=multiprocessing.get_context('fork');children=[]
        for _ in range(2):
            parent,child=ctx.Pipe(False)
            process=ctx.Process(target=contend,args=(str(f.directory),plan,ref,allocation,child));process.start();child.close()
            children.append((process,parent))
        values=[connection.recv() for _,connection in children]
        for process,connection in children:process.join(5);connection.close();self.assertEqual(process.exitcode,0)
        self.assertEqual(values.count(None),1)
        self.assertEqual(values.count(plan['units'][0]['unit_id']),1)

    def test_new_run_name_or_manifest_cannot_duplicate_an_original_pair(self):
        f=self.fixture;first=f.unit(1);second=f.unit(2,pair_index=1)
        self.assertEqual(first['claim_keys'],second['claim_keys'])
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            f.plan([first,second])

    def test_external_original_single_job_reserves_same_scientific_keys(self):
        f=self.fixture;unit=f.unit(1)
        payload=f.artifact('logs/pbs/old/payload.json',{'name':unit['name'],'command':unit['command'],
            'task_id':'T01','budget_group':'T02','requested_gpu_count':1})
        receipt=f.artifact('logs/pbs/old/submission.json',{'returncode':0,'stdout':'9.hopper-m-02\n'})
        plan,ref=f.plan([unit],[{'unit':unit,'payload_reference':payload,'receipt_reference':receipt}])
        queue=shared.SharedQueue(f.directory,plan,ref,root=f.root)
        self.assertFalse(queue.has_pending());self.assertIsNone(queue.claim({},remaining_seconds=2000))

    def test_short_remaining_time_never_truncates_original_unit_grant(self):
        f=self.fixture;plan,ref=f.plan([f.unit(1)]);queue=shared.SharedQueue(f.directory,plan,ref,root=f.root)
        self.assertIsNone(queue.claim({},remaining_seconds=1244))
        self.assertTrue(queue.has_pending())

    def test_uncertain_cleanup_is_not_recycled(self):
        f=self.fixture;plan,ref=f.plan([f.unit(1)]);queue=shared.SharedQueue(f.directory,plan,ref,root=f.root)
        claim=queue.claim(f.parent(ref),remaining_seconds=2000)
        with self.assertRaisesRegex(ValueError,'Uncertain'):
            queue.finish(claim,{'claim_reference':claim['claim_reference'],'cleanup_errors':['unknown']})
        self.assertFalse(queue.has_pending())
        self.assertIsNone(queue.claim({},remaining_seconds=2000))

    def test_edited_plan_reference_fails_even_with_old_saved_identity(self):
        f=self.fixture;plan,ref=f.plan([f.unit(1)])
        path=Path(ref['file']);path.chmod(0o644);path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'bytes differ'):
            shared.load_plan(path,ref['sha256'],root=f.root)


if __name__=='__main__':unittest.main()
