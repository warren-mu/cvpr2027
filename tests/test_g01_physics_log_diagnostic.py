"""Small synthetic CPU fixtures; no original log scans or simulator execution."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('physics_diagnostic',Path(__file__).resolve().parents[1]/'experiments/g01_physics_log_diagnostic.py')
diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)


class PhysicsDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.inputs=diag.Inputs(Path(self.temp.name))
    def write(self,filename,value):
        body=json.dumps(value,sort_keys=True).encode();p=self.inputs.path(filename);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
        return {'file':filename,'sha256':diag.digest(body),'bytes':len(body)}
    def reference_fixture(self):
        baseline={'scene':'FloorPlan1','worker_pid':10,'unity_pid':11,'objects':[
            {'name':'X','objectType':'Box','position':{'x':0,'y':0,'z':0}},
            {'name':'A','objectType':'Vase','position':{'x':1,'y':0,'z':0}}]}
        ref=self.write('/scratch/scene/marked_controller_000/construction/baseline.json',baseline)
        p={'valid_physical':False,'edits':{'X':{'position_error_m':.01}},'edited_names':['X'],
            'all_object_changes':{'X':{'position_change_m':.01},'A':{'position_change_m':.03}},
            'common_baseline_check':{'baseline_reference':ref,'measurements':{'inventory_exact':True,'objects':{
                'X':{'position':{'x':0,'y':0,'z':0},'baseline_displacement_m':0},
                'A':{'position':{'x':2,'y':0,'z':0},'baseline_displacement_m':.04}}}}}
        row={'pair_id':'p','scene':'FloorPlan1','search_controller_id':'marked_controller_000',
             'identities':{'anchor_name':'X'},'physical_w0':p,
             'native_retry_records':{'w0':{'X':{'object_name':'X','scene':'FloorPlan1','status':'placed',
                 'action_reference':{'file':'/scratch/scene/marked_controller_000/construction/action.json','sha256':'b'*64},
                 'arguments':{'position':{'x':0,'y':0,'z':0}}}}}}
        cfg={'scene':'FloorPlan1','raw_evidence_dir':'/scratch/scene'}
        metrics={'controller_visits':[{'controller_id':'marked_controller_000','worker_pid':10,'unity_identity':{'pid':11}}]}
        return row,cfg,metrics,baseline,ref
    def prepare_funnel(self,row=None,uncounted=None):
        attempts=0 if row is None else 1
        source_name='results/source.json';journal_ref=None
        if row is not None:
            raw_ref=self.write('/scratch/raw/attempt.json',row)
            journal={k:row[k] for k in ('pair_id','scene','kind','status','discard_reasons','proposed_pool')};journal['raw_attempt']=raw_ref
            body=(json.dumps(journal)+'\n').encode();p=self.inputs.path('data/attempts.jsonl');p.parent.mkdir(parents=True);p.write_bytes(body)
            journal_ref={'file':'data/attempts.jsonl','sha256':diag.digest(body)}
        metrics={'attempts':attempts,'placement_point_failures':[]}
        if uncounted is not None:
            metrics['placement_point_failures']=[{'raw_attempt':self.write('/scratch/raw/uncounted.json',uncounted)}]
        source={'config':{'scene':'FloorPlan1'},'metrics':metrics,'_meta':{'end':'fixture','metrics_sha256':diag.digest(json.dumps(metrics,sort_keys=True).encode())}}
        source_ref=self.write(source_name,source);source_ref['metrics_sha256']=source['_meta']['metrics_sha256']
        funnel={'errors':[],'sources':[{'source':source_ref,'scene':'FloorPlan1','batch':'A','scan_complete':True,'journal':journal_ref,
            'observed_by_proposed_pool_mode':{'look_stay':attempts}}],
            'groups':[{'batch':'A','actual_proposed_class':'look_stay','observed_attempts':attempts}]}
        funnel_ref=self.write(diag.FUNNEL,funnel)
        pm={'report':funnel_ref};name='gen_test_round2_prepare_01_retry3_funnel'
        parent={'config':{'name':name},'metrics':pm,'_meta':{'name':name,'end':'fixture','exit_code':0,
            'metrics_sha256':diag.digest(json.dumps(pm,sort_keys=True).encode())}}
        return self.write('results/'+name+'.json',parent)['sha256']
    def test_unknown_world_and_empty_edits_are_not_negative_measurements(self):
        value=diag.world_details({'physical_w0':{'valid_physical':False,'edits':{}}},'w0',self.inputs)
        self.assertIsNone(value['anchor_above_2cm']);self.assertIsNone(value['other_above_1cm'])
        missing=diag.world_details({},'w1',self.inputs)
        self.assertIsNone(missing['anchor_above_2cm']);self.assertIsNone(missing['other_above_1cm'])
        row={'pair_id':'p','scene':'FloorPlan1','mode':'stay','proposed_pool':'look','kind':None,
            'status':'rejected','discard_reasons':['w0_physical'],'counts_as_pair_attempt':True,
            'physical_w0':{'valid_physical':False,'edits':{}}}
        group=diag.diagnose(self.inputs,self.prepare_funnel(row))['groups'][0]
        self.assertEqual(group['physical_failures'],1)
        self.assertEqual(group['failure_subtypes_nonexclusive']['neither_of_these_measured_displacements'],0)
        self.assertEqual(group['failure_subtypes_unknown'],{'anchor_displacement_unknown':1,'other_displacement_unknown':1})
    def test_same_object_in_two_baselines_is_one_world_object_event(self):
        row,cfg,metrics,_,ref=self.reference_fixture()
        value=diag.world_details(row,'w0',self.inputs,cfg,metrics)
        self.assertEqual(len(value['other_objects']),1)
        other=value['other_objects'][0]
        self.assertEqual((other['local_displacement_m'],other['common_baseline_displacement_m']),(.03,.04))
        self.assertEqual(other['object_type'],'Vase');self.assertEqual(other['type_source'],ref)
        self.assertEqual(other['post_settling_center_distance_to_anchor_m'],2)
        self.assertEqual(other['baseline_object_to_requested_anchor_distance_m'],1)
    def test_foreign_controller_or_scene_metadata_is_rejected(self):
        row,cfg,metrics,baseline,ref=self.reference_fixture()
        row['search_controller_id']='marked_controller_001'
        with self.assertRaisesRegex(AssertionError,'another Controller'):diag.world_details(row,'w0',self.inputs,cfg,metrics)
        row['search_controller_id']='marked_controller_000';baseline['scene']='FloorPlan2'
        new=self.write(ref['file'],baseline);row['physical_w0']['common_baseline_check']['baseline_reference']=new
        # Fresh reader for this independently mutated fixture; original input identity stays immutable.
        with self.assertRaisesRegex(AssertionError,'another scene'):
            diag.world_details(row,'w0',diag.Inputs(Path(self.temp.name)),cfg,metrics)
    def test_missing_original_snapshot_remains_null_and_lists_exact_reference(self):
        row,cfg,metrics,_,ref=self.reference_fixture();self.inputs.path(ref['file']).unlink()
        value=diag.world_details(row,'w0',self.inputs,cfg,metrics)
        other=value['other_objects'][0]
        self.assertIsNone(other['object_type']);self.assertIsNone(other['baseline_object_to_requested_anchor_distance_m'])
        self.assertFalse(other['same_controller_baseline_verified']);self.assertEqual(self.inputs.missing,{ref['file']:ref})
        self.assertEqual(other['post_settling_center_distance_to_anchor_m'],2)
    def test_compact_row_uses_original_path_and_visit_without_new_controller_field(self):
        row,cfg,metrics,_,_=self.reference_fixture();row.pop('search_controller_id');row.pop('identities')
        physical=row.pop('physical_w0')
        physical['common_baseline_key_numbers']={'baseline_reference':physical.pop('common_baseline_check')['baseline_reference'],
                                                'max_other_displacement_m':.04,'violations':{'A':.04}}
        row['physical_key_numbers']={'physical_w0':physical}
        other=diag.world_details(row,'w0',self.inputs,cfg,metrics)['other_objects'][0]
        self.assertTrue(other['same_controller_baseline_verified']);self.assertEqual(other['object_type'],'Vase')
        self.assertIsNone(other['post_settling_center_distance_to_anchor_m'])
        self.assertEqual(other['baseline_object_to_requested_anchor_distance_m'],1)
        unknown=diag.world_details(row,'w0',self.inputs,cfg,{'controller_visits':[]})
        self.assertIsNone(unknown['other_objects'][0]['object_type'])
        self.assertIsNone(unknown['other_objects'][0]['baseline_object_to_requested_anchor_distance_m'])
        self.assertIn('matching source Controller visit absent',unknown['metadata_association_missing'])
    def test_requested_distance_requires_same_controller_action_and_success_metadata(self):
        row,cfg,metrics,_,_=self.reference_fixture()
        ledger=row['native_retry_records']['w0']['X'];ledger.pop('status')
        value=diag.world_details(row,'w0',self.inputs,cfg,metrics)
        self.assertIsNone(value['other_objects'][0]['baseline_object_to_requested_anchor_distance_m'])
        ledger['status']='placed';ledger['action_reference']['file']='/scratch/other/construction/action.json'
        with self.assertRaises(ValueError):diag.world_details(row,'w0',self.inputs,cfg,metrics)
    def test_native_receipt_dedup_and_uncounted_failures_do_not_change_attempt_denominator(self):
        call={'action_reference':{'file':'/scratch/action.json','sha256':'a'*64},'native_success':False,'error':'native_failure'}
        row={'pair_id':'uncounted','scene':'FloorPlan1','mode':'stay','proposed_pool':'look','counts_as_pair_attempt':False,
            'native_retry_records':{'w0':{'X':{'attempts':[call]}}},'native_retry_failure':{'attempts':[copy.deepcopy(call)]}}
        self.assertEqual(len(diag.native_calls(row)),1)
        result=diag.diagnose(self.inputs,self.prepare_funnel(uncounted=row))['groups'][0]
        self.assertEqual(result['counted_proposals'],0);self.assertEqual(result['uncounted_placement_proposals'],1)
        self.assertEqual(result['native_call_status'],{'failed':1});self.assertEqual(result['physical_failures'],0)
        markdown=diag.markdown_report({'status':'complete','groups':[result]})
        self.assertIn('| A/look_stay | 0 | 0 | 0 | 0 | 0 | 0 / 0 | 1 / 1 |',markdown)
        self.assertIn('不能替代实际距离',markdown)
    def test_original_reference_hash_and_entrypoint_hash_are_mandatory(self):
        row,cfg,metrics,_,ref=self.reference_fixture();self.inputs.path(ref['file']).write_text('{}')
        with self.assertRaisesRegex(AssertionError,'SHA differs'):diag.world_details(row,'w0',self.inputs,cfg,metrics)
        self.prepare_funnel()
        with self.assertRaisesRegex(AssertionError,'SHA differs'):diag.diagnose(self.inputs,'0'*64)

if __name__=='__main__':unittest.main()
