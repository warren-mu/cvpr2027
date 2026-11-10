"""Synthetic byte/CPU retention tests; no simulator or scientific outcomes invented."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from src.envs import pair_v2_raw_sampling as sample
from src.envs import pair_v2_evidence as evidence
from src.envs import pair_v2_g01_policy as policy
from src.envs import pair_v2_g01_shard as worker
from src.tools import g01_physics_smoke as smoke
from src.tools import g01_parallel as parent
from src.tools import g01_admission as admission
from src.tools import g01_shared_unit as shared

def fixture(name):
    spec=importlib.util.spec_from_file_location('raw_sampling_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def rejected():
    return {'scene':'FloorPlan21','pair_id':'floorplan21_00000','status':'rejected',
            'discard_reasons':['real_reason'],'physical_w0':{'reasons':['actual_object_specific_reason']}}

class SamplingTests(unittest.TestCase):
    def test_exact_seeded_draw_and_reason_independence_without_research_rng(self):
        before=random.getstate();p=sample.SampledRetention('FloorPlan21')
        for ordinal in range(60):
            keep,record=p.decide(rejected())
            original={'seed':0,'scene':'FloorPlan21','attempt_ordinal':ordinal}
            h=hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            self.assertEqual(record['raw_sample'],{'key':original,'sha256':h,'selected':int(h,16)<2**256//10})
            self.assertEqual(keep,record['raw_sample']['selected'])
        self.assertEqual(p.reason_counts,{'actual_object_specific_reason':60})
        self.assertEqual(random.getstate(),before)
        again=sample.SampledRetention('FloorPlan21')
        changed={**rejected(),'physical_w0':None,'discard_reasons':['different']}
        self.assertEqual([again.decide(changed)[0] for _ in range(60)],
                         [sample.draw('FloorPlan21',i)['selected'] for i in range(60)])

    def test_exception_explicit_failure_and_unknown_keep_all_original_bytes(self):
        cases=[(rejected(),RuntimeError('native infrastructure error')),
               ({**rejected(),'status':'error'},None),({**rejected(),'status':'failed'},None),
               ({**rejected(),'status':'unknown'},None),({'scene':'FloorPlan21'},None),
               ({'scene':'FloorPlan21','status':'rejected','discard_reasons':[]},None),
               ({**rejected(),'error':'controller error'},None),
               ({**rejected(),'cleanup_errors':['unknown cleanup']},None)]
        for index,(row,error) in enumerate(cases):
            with self.subTest(case=index),tempfile.TemporaryDirectory() as td:
                p=sample.SampledRetention('FloorPlan21')
                while sample.draw('FloorPlan21',p.ordinal)['selected']:p.decide(rejected())
                session=evidence.SearchEvidenceSession(td,raw_budget_bytes=10000)
                store=session.store(Path(td)/'construction',10000);session.begin()
                reference=store.save('original_failure.bin',b'original failed native bytes')
                keep,record=p.decide(row,error)
                self.assertTrue(keep);self.assertFalse(record['raw_sample']['selected'])
                self.assertFalse(record['sampling_applied_to_rejection'])
                self.assertEqual(record['reason'],'non_scientific_or_unclassified_failure')
                stored=session.finish(keep);self.assertTrue(stored['full_raw_retained'])
                checked={**row,'scene':'FloorPlan21','evidence_retention':{**record,**stored}}
                sample.validate_decision(checked)
                checked['evidence_retention']['retention_requested']=False
                with self.assertRaises(ValueError):sample.validate_decision(checked)
                checked['evidence_retention']['sampling_applied_to_rejection']=True
                checked['evidence_retention']['retention_requested']=record['raw_sample']['selected']
                with self.assertRaisesRegex(ValueError,'scientific rejection'):sample.validate_decision(checked)
                self.assertEqual(Path(reference['file']).read_bytes(),b'original failed native bytes')

    def test_accepted_original_bytes_always_materialize_even_on_unselected_draw(self):
        with tempfile.TemporaryDirectory() as td:
            session=evidence.SearchEvidenceSession(td,raw_budget_bytes=100000)
            store=session.store(Path(td)/'construction',100000)
            common=store.save_json('baseline.json',{'actual':'original baseline'})
            p=sample.SampledRetention('FloorPlan21')
            while sample.draw('FloorPlan21',p.ordinal)['selected']:p.decide(rejected())
            session.begin();ref=store.save('accepted_native.bin',b'original native payload')
            row={**rejected(),'status':'passed','discard_reasons':[]}
            keep,decision=p.decide(row);self.assertTrue(keep);self.assertFalse(decision['raw_sample']['selected'])
            stored=session.finish(keep);row['evidence_retention']={**decision,**stored}
            sample.validate_decision(row)
            self.assertEqual(Path(ref['file']).read_bytes(),b'original native payload')
            self.assertEqual(hashlib.sha256(Path(ref['file']).read_bytes()).hexdigest(),ref['sha256'])
            self.assertTrue(Path(common['file']).exists())
            row['evidence_retention']['full_raw_retained']=False
            with self.assertRaises(ValueError):sample.validate_decision(row)

    def test_unselected_rejection_never_writes_but_its_compact_record_survives(self):
        with tempfile.TemporaryDirectory() as td:
            session=evidence.SearchEvidenceSession(td,raw_budget_bytes=100000)
            store=session.store(Path(td)/'construction',100000)
            retained=store.save('common.bin',b'common')
            session.begin();missing=store.save('rejected.bin',b'rejected native')
            p=sample.SampledRetention('FloorPlan21')
            while sample.draw('FloorPlan21',p.ordinal)['selected']:p.decide(rejected())
            row={**rejected(),**sample.fields()};keep,decision=p.decide(row);self.assertFalse(keep)
            stored=session.finish(keep);row['evidence_retention']={**decision,**stored}
            sample.validate_decision(row)
            with session.reading():
                compact=sample.compact_without_missing_references({'raw':missing,'common':retained,'reasons':row['discard_reasons']})
            self.assertNotIn('file',compact['raw']);self.assertTrue(compact['raw']['raw_payload_omitted'])
            self.assertEqual(compact['common'],retained)
            self.assertFalse(Path(missing['file']).exists());self.assertEqual(stored['disk_files_deleted'],0)
            saved=Path(td)/'attempt.json';saved.write_text(json.dumps(row))
            self.assertEqual(json.loads(saved.read_text())['discard_reasons'],row['discard_reasons'])

    def test_rejection_does_not_destroy_reusable_controller_and_later_acceptance_closure(self):
        with tempfile.TemporaryDirectory() as td:
            session=evidence.SearchEvidenceSession(td,raw_budget_bytes=100000)
            store=session.store(Path(td)/'construction',100000)
            baseline=store.save_json('baseline.json',{'actual_position':1.125})
            pool=store.save_json('native_pool.json',{'baseline':baseline,'native_query':[1,2,3]})
            session.begin();discarded=store.save_json('rejected.json',{'pool':pool,'reason':'actual reject'})
            session.finish(False)
            self.assertFalse(Path(discarded['file']).exists())
            session.begin();accepted=store.save_json('accepted.json',{'baseline':baseline,'pool':pool})
            self.assertTrue(session.finish(True)['evidence_complete_for_acceptance'])
            closure=session._reference_closure([accepted])
            self.assertEqual(set(closure),{baseline['file'],pool['file'],accepted['file']})
            for ref in (baseline,pool,accepted):
                self.assertEqual(hashlib.sha256(Path(ref['file']).read_bytes()).hexdigest(),ref['sha256'])

    def test_selected_rejection_uses_original_cap_and_cannot_upgrade_acceptance(self):
        p=sample.SampledRetention('FloorPlan21')
        while not sample.draw('FloorPlan21',p.ordinal)['selected']:p.decide(rejected())
        with tempfile.TemporaryDirectory() as td:
            session=evidence.SearchEvidenceSession(td,raw_budget_bytes=3)
            store=session.store(Path(td)/'construction',1000);session.begin();ref=store.save('too_big.bin',b'abcd')
            keep,decision=p.decide(rejected());self.assertTrue(keep)
            stored=session.finish(keep)
            self.assertFalse(stored['evidence_complete_for_acceptance']);self.assertTrue(stored['cap_hit'])
            self.assertFalse(Path(ref['file']).exists())

    def test_original_terminal_black_proof_is_retained_outside_random_sample(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            session=evidence.SearchEvidenceSession(td,raw_budget_bytes=10000,terminal_reserve_bytes=1000)
            store=session.store(Path(td)/'construction',10000);session.begin()
            ref=store.save('actual_black.bin',b'original native bytes')
            p=sample.SampledRetention('FloorPlan21')
            while sample.draw('FloorPlan21',p.ordinal)['selected']:p.decide(rejected())
            with patch.object(sample.search.stable,'_SESSION',session):
                keep,record=p.decide(rejected(),SimpleNamespace(reference=ref),black=True)
            self.assertTrue(keep);self.assertFalse(record['raw_sample']['selected'])
            self.assertEqual(record['reason'],'black_frame_stop')
            stored=session.finish(keep)
            self.assertTrue(stored['terminal_evidence']['preserved'])
            self.assertEqual(Path(ref['file']).read_bytes(),b'original native bytes')

    def test_compact_rejection_keeps_all_original_reasons_and_configuration_inputs(self):
        from src.envs import pair_v2_g01_search as search
        row={**rejected(),**sample.fields(),**search.fields(),
             'g01_configuration':{'configuration_id':'fixture'},
             'view':{'position':{'x':0,'y':1,'z':2},'rotation':30},
             'identities':{'anchor_name':'actual-X','candidate_names':['actual-A','actual-B']},
             'evidence_retention':{'full_raw_retained':False},
             'projection_pool_reference':{'file':'/missing-test-native-raw','sha256':'f'*64}}
        compact=worker.compact_attempt(row)
        self.assertEqual(compact['discard_reasons'],row['discard_reasons'])
        self.assertEqual(compact['view'],row['view']);self.assertEqual(compact['identities'],row['identities'])
        sample.validate_fields(compact)
        self.assertIn('actual_object_specific_reason',compact['physical_key_numbers']['physical_w0']['reasons'])

    def test_partial_or_tampered_declarations_fail_and_old_policy_is_identical(self):
        old=policy.declared('FloorPlan21','test')
        self.assertNotIn('raw_sampling_version',old);self.assertEqual(policy.declared_fields({**old,'scene':'FloorPlan21'}),old)
        new=policy.declared('FloorPlan21','test',sampled=True)
        self.assertTrue(new['g01_version'].endswith('_sampled_raw_v1'))
        self.assertEqual(policy.declared_fields({**new,'scene':'FloorPlan21'}),new)
        for key in sample.FIELDS:
            bad={**new,'scene':'FloorPlan21'};bad.pop(key)
            with self.assertRaises(ValueError):policy.declared_fields(bad)
        for seed in (1,False,0.0):
            bad={**new,'scene':'FloorPlan21','raw_sampling_seed':seed}
            with self.assertRaises(ValueError):policy.declared_fields(bad)

    def test_scoped_worker_policy_suppresses_unsampled_thumbnail_and_restores(self):
        original=sample.search.SearchRetention
        cfg={'scene':'FloorPlan21',**sample.fields()}
        with patch.object(sample.search.stable,'thumbnail',return_value={'real':'thumbnail'}) as thumbnail:
            with self.assertRaisesRegex(RuntimeError,'fixture'):
                with sample.installed(cfg):
                    p=sample.search.SearchRetention()
                    while sample.draw('FloorPlan21',p.ordinal)['selected']:p.decide(rejected())
                    p.decide(rejected());self.assertIsNone(sample.search.stable.thumbnail(None,None));thumbnail.assert_not_called()
                    p.decide({**rejected(),'status':'passed'});self.assertEqual(sample.search.stable.thumbnail(None,None),{'real':'thumbnail'})
                    raise RuntimeError('fixture')
            self.assertIs(sample.search.SearchRetention,original)
            self.assertIs(sample.search.stable.thumbnail,thumbnail)

    def test_real_smoke_shard_and_resource_plan_carry_policy_old_path_does_not(self):
        f=fixture('test_g01_physics_smoke');name='gen_test_s16_smoke_rawfixture';value=f.plan(name)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();(root/'configs').mkdir();path=root/'configs/q.json';path.write_text(json.dumps(value));path.chmod(0o444)
            ref={'file':'configs/q.json','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            directory=root/'scratch/c6';directory.mkdir(parents=True)
            with patch.object(smoke.quotas,'ROOT',root),patch.object(smoke.quotas,'authenticate_count_basis',return_value={'fixture':'original authority boundary'}):
                with smoke.installed(name,ref,sampled=True):
                    rows=parent.assignments(name,0,6);gpu='GPU-11111111-2222-3333-4444-555555555555'
                    plan,pref=parent.create_plan(directory,parent_run_name=name,stage_index=0,concurrency=6,assignments=rows,
                        pbs_jobid='123.hopper-m-02',node='fixture',gpu_uuid=gpu,cpu_affinity=list(range(12)),scenes=['FloorPlan21'],partition='test')
                    gate={'config':{'node':'fixture','allocation':{'pbs_jobid':'123.hopper-m-02','cuda_visible_devices':gpu,'devices':[{'gpu_uuid':gpu}]}}}
                    grow={'source':{'file':'results/gate.json','sha256':'b'*64},'result_path':'results/gate.json'}
                    raw=directory/'FloorPlan21';raw.mkdir();child=rows[0]['child_run_name']
                    shard=parent.make_shard(child,'FloorPlan21',list(range(12)),gate,grow,raw,plan,pref)
                    cfg={'name':child,'scene':'FloorPlan21','shard':shard,'field_of_view':60,'raw_evidence_dir':str(raw),
                         'max_attempts':160,'max_raw_bytes':512*1024*1024,'resolution':255,'deadline':10000000000}
                    with worker.installed():worker.enable_config(cfg)
                    policy.validate_source(cfg);admission.validate_source_admission(cfg)
                    self.assertTrue(plan['sample_rejected_raw_enabled']);sample.validate_fields(cfg)
                    self.assertEqual(cfg['raw_evidence_policy']['version'],sample.VERSION)
                    self.assertEqual(worker.tags(cfg)['raw_sampling_version'],sample.VERSION)
                    changed=copy.deepcopy(cfg);changed['shard'].pop('raw_sampling_version')
                    with self.assertRaises(ValueError):admission.validate_source_admission(changed)

    def test_source_decision_cannot_change_ordinal_to_select_another_rejected_sample(self):
        p=sample.SampledRetention('FloorPlan21');row=rejected();keep,record=p.decide(row)
        row['evidence_retention']={**record,'retention_requested':keep}
        sample.validate_decision(row,raw_path='/raw/floorplan21_00000_trial000000/attempt.json')
        row['evidence_retention']['raw_sample']=sample.draw('FloorPlan21',1)
        row['evidence_retention']['retention_requested']=sample.draw('FloorPlan21',1)['selected']
        with self.assertRaisesRegex(ValueError,'trial path'):
            sample.validate_decision(row,raw_path='/raw/floorplan21_00000_trial000000/attempt.json')

    def test_shared_cli_identity_changes_storage_declaration_not_scientific_claim_keys(self):
        f=fixture('test_g01_shared_unit')
        with tempfile.TemporaryDirectory() as td:
            value=f.Fixture(Path(td).resolve());old,oldkeys=shared.validate_unit(value.unit,value.root)
            new=copy.deepcopy(value.unit);new['command'].append('--sample-rejected-raw')
            identity,keys=shared.validate_unit(new,value.root)
            self.assertEqual(identity,{**old,**sample.fields()});self.assertEqual(keys,oldkeys)
            new['command'].append('--sample-rejected-raw')
            with self.assertRaises(ValueError):shared.validate_unit(new,value.root)

    def test_shared_admission_cannot_drop_or_acquire_sampling_policy(self):
        f=fixture('test_g01_shared_unit')
        from src.tools import gpu_shared_claim as claims
        with tempfile.TemporaryDirectory() as td:
            value=f.Fixture(Path(td).resolve())
            value.unit['command'].append('--sample-rejected-raw')
            value.unit['identity'],value.unit['claim_keys']=shared.validate_unit(value.unit,value.root)
            plan={'shared_queue_claim':value.proof,'scratch':str(value.root/'scratch'),
                  'parent_run_name':value.unit['name'],'scenes':[r['scene'] for r in value.value['assignments']],
                  'stage_index':0,'concurrency':6,'g01_partition':'test',
                  **{key:value.proof[key] for key in ('pbs_jobid','node','gpu_uuid','cpu_affinity')},
                  'sample_rejected_raw_enabled':True,**sample.fields()}
            # Original immutable claim/receipt validation is covered by its actual
            # shared-helper suite; this isolates declaration binding to that proof.
            with patch.object(claims,'validate_saved_claim',return_value=value.proof):
                self.assertEqual(shared.validate_admission(plan,root=value.root),value.proof)
                for key in ('sample_rejected_raw_enabled',*sample.FIELDS):
                    bad=copy.deepcopy(plan);bad.pop(key)
                    with self.assertRaises(ValueError):shared.validate_admission(bad,root=value.root)
                for enabled in (False,1):
                    with self.assertRaises(ValueError):shared.validate_admission({**plan,'sample_rejected_raw_enabled':enabled},root=value.root)

if __name__=='__main__':unittest.main()
