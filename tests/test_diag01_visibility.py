"""Synthetic CPU fixtures, never an original-data result or source acceptance."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from src.envs import pair_table
from src.envs.pair_table_export import save_table,load_table
from src.tools.smoke_e2e_pilot_contract import episode_observation


def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


diag=load('diag01_visibility_fixture',Path(__file__).resolve().parents[1]/'experiments/diag01_visibility.py')
fixtures=load('diag01_table_fixture',Path(pair_table.__file__).resolve().parents[2]/'tests/test_pair_table.py')


class VisibilityTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  table=fixtures.table();table.source={'synthetic':True,'t01':{'pair':{'file':'fixture/pair.json','sha256':'a'*64}}}
  ref=save_table(table,self.root/'table');self.table=load_table(ref['file'],expected_sha256=ref['sha256'])
  self.key=self.table.dataset_key
  self.entry={'dataset_key':self.key,'original_pair_id':self.table.pair_id,'scene':self.table.scene,
              'kind':self.table.kind,'world_order':[0,1],'table_manifest':self.table.manifest_reference}
  self.reads=diag.Reads(self.root);self.index=diag.TableIndex([self.entry],self.reads)

 def trajectory(self,actions,*,order=(0,1),world='w0'):
  self.entry['world_order']=list(order)
  backend=pair_table.TableObservationBackend({self.key:self.table},world_orders={self.key:list(order)},allow_synthetic=True)
  env=pair_table.TableEnvironment(backend);env.reset(self.key,world)
  plan={'episode_id':'fixture:episode','dataset_key':self.key,'pair_id':self.key,'world_id':world,'pair_instance_id':'one'}
  traces=[];image_history=[]
  for i,response in enumerate(actions):
   before=copy.deepcopy(env.info)
   try: parsed=diag.parse_action(response)
   except ValueError:parsed=None
   _,reward,done,after=env.step(response)
   image_history.append({'size':[255,255],'mode':'RGB','sha256':before['capture']['marked_rgb_sha256']})
   traces.append({**plan,'decision_index':i,'response':response,'before':before,'after':copy.deepcopy(after),
     'parse_valid':parsed is not None,'action_valid':after['action_valid'],'sample_id':str(i),'reward':reward,
     'action_kind':('select' if parsed.kind=='select' else 'observe') if parsed else 'invalid',
     'input':{'images':copy.deepcopy(image_history)}})
  self.assertTrue(done)
  episode=episode_observation(plan,traces,kind=env.info['kind'],final_info=env.info,selected_name=env.selected_name,success=env.success)
  episode.update(world_order=list(order),original_world_id=f'w{env.backend.world}',original_pair_id=self.table.pair_id,
                 target_name=env.info['target_name'])
  env.close();return json.loads(json.dumps(episode)),json.loads(json.dumps(traces))

 def recalc(self,ep,traces):
  return diag.recompute_episode(ep,traces,self.index,arm='T0',phase='train',update=1,journal={'fixture':True})

 def test_observe_then_select_counts_state_before_select_not_zero_based_index(self):
  ep,traces=self.trajectory(['turn_right',f'select {self.table.labels["A"]}'])
  self.assertFalse(ep['seen_before_select'])
  row=self.recalc(ep,traces)
  self.assertTrue(row['seen_before_select']);self.assertTrue(row['seen_count_mismatch'])
  self.assertEqual((row['independent_first_250_state_index'],row['select_action_number']),(1,2))
  self.assertTrue(row['literal_formula_matches_logged']);self.assertFalse(row['x_seen_step_mismatch'])

 def test_swapped_flip_world_uses_original_world_mask(self):
  ep,traces=self.trajectory(['turn_left',f'select {self.table.labels["B"]}'],order=(1,0),world='w0')
  row=self.recalc(ep,traces)
  self.assertEqual(row['original_world_id'],'w1');self.assertTrue(row['seen_before_select'])
  ep['original_world_id']='w0'
  with self.assertRaisesRegex(ValueError,'mapping'):self.recalc(ep,traces)

 def test_logged_seen_fields_never_drive_raw_mask_answer(self):
  ep,traces=self.trajectory(['turn_left',f'select {self.table.labels["A"]}'])
  ep['x_seen_step']=0;ep['seen_before_select']=True
  row=self.recalc(ep,traces)
  self.assertFalse(row['seen_before_select']);self.assertFalse(row['ever_250'])
  self.assertTrue(row['seen_count_mismatch']);self.assertTrue(row['x_seen_step_mismatch'])

 def test_invalid_parsed_select_still_counts_actual_prior_seen(self):
  ep,traces=self.trajectory(['turn_right','select 999'])
  row=self.recalc(ep,traces)
  self.assertTrue(row['seen_before_select']);self.assertFalse(row['saved_success'])

 def test_no_select_keeps_whole_episode_visibility_and_final_state(self):
  ep,traces=self.trajectory(['turn_right','turn_left','turn_left','turn_right','turn_right'])
  row=self.recalc(ep,traces)
  self.assertFalse(row['seen_before_select']);self.assertTrue(row['ever_250'])
  self.assertEqual(len(row['states']),6)
  self.assertEqual(row['states'][-1]['state_id'],row['states'][-2]['state_id'])
  self.assertFalse(row['actions'][-1]['transition_performed'])

 def test_invalid_actions_consume_turn_without_fake_graph_transition(self):
  ep,traces=self.trajectory(['move 999','not an action',f'select {self.table.labels["A"]}'])
  row=self.recalc(ep,traces)
  self.assertFalse(row['ever_positive']);self.assertTrue(all(not a['transition_performed'] for a in row['actions']))

 def test_original_state_mapping_and_public_image_tampering_rejected(self):
  ep,traces=self.trajectory(['turn_right','select 999'])
  traces[0]['after']['capture']['raw_rgb_sha256']='f'*64
  with self.assertRaisesRegex(ValueError,'public capture'):self.recalc(ep,traces)

 def test_actual_mask_sha_required_not_stored_count(self):
  h=self.index.header(self.key);state=h['states'][h['initial_state']]['transitions']['turn_right']
  ref=self.index.payload(self.key,h['states'][state]['captures'][0]['masks']['X'])
  Path(ref['file']).write_bytes(bytes(255*255))
  with self.assertRaisesRegex(ValueError,'Original bytes changed'):self.index.state(self.key,state,0)

 def test_stored_count_disagreement_recorded_not_used(self):
  h=self.index.header(self.key);state=h['states'][h['initial_state']]['transitions']['turn_right']
  h['states'][state]['anchor_pixels'][0]=0
  actual=self.index.state(self.key,state,0)
  self.assertEqual(actual['own_anchor_pixels'],250);self.assertFalse(actual['stored_anchor_count_matches_bytes'])

 def test_correct_s0_actions_and_marked_input_exact(self):
  self.assertEqual([r['action'] for r in self.index.first_actions(self.key,0) if r['sees_250']],['turn_right'])
  self.assertEqual([r['action'] for r in self.index.first_actions(self.key,1) if r['sees_250']],['turn_left'])
  ep,traces=self.trajectory(['turn_left',f'select {self.table.labels["A"]}'])
  row=self.recalc(ep,traces)
  self.assertFalse(row['s0_first_observation']['sees_250'])
  marked=self.index.marked(self.key,self.table.initial_state,0)
  self.assertEqual(marked.sha256,traces[0]['before']['capture']['marked_rgb_sha256'])
  target=self.root/'examples';target.mkdir()
  outputs=diag.export_examples([row],self.index,target)
  self.assertEqual(len(outputs),1)
  m=json.loads(Path(outputs[0]['file']).read_bytes())
  self.assertEqual(len(m['frames']),3)

 def test_original_preprocessor_only_function_and_exact_input_hash(self):
  upstream=self.root/'upstream';source=upstream/'verl/utils/dataset/rl_dataset.py';source.parent.mkdir(parents=True)
  source.write_text('raise RuntimeError("module top-level must not run")\ndef process_image(image):\n return image\n')
  parents={arm:{'metrics':{'upstream':{'verl':{'path':str(upstream),'commit':'3f55021e08b98315213e10d10b709c7f6bbe84f7'}}}} for arm in ('T0','T4')}
  process,proof=diag.original_image_preprocessor(parents,self.reads)
  ep,traces=self.trajectory(['turn_left',f'select {self.table.labels["A"]}']);row=self.recalc(ep,traces)
  target=self.root/'model_examples';target.mkdir()
  outputs=diag.export_examples([row],self.index,target,process)
  m=json.loads(Path(outputs[0]['file']).read_bytes())
  self.assertIn('actual_model_input',m['frames'][0]);self.assertNotIn('actual_model_input',m['frames'][-1])
  self.assertEqual(len(row['actions'][1]['actual_input_images']),2)
  row['actions'][0]['actual_input_images'][0]['sha256']='f'*64
  target2=self.root/'bad_model_examples';target2.mkdir()
  with self.assertRaisesRegex(ValueError,'actual policy input'):diag.export_examples([row],self.index,target2,process)

 def test_t4_and_pre_evidence_original_equal_boundary_is_different_from_learning(self):
  from src.rl.t4_consistency import descriptors
  from src.tools.smoke_e2e_training_contract import pre_evidence
  members=[]
  for world in ('w0','w1'):
   members.append({'episode_id':world,'pair_instance_id':'same','pair_id':'p','world_id':world,
       'kind':'look_stay','candidate_names':['A','B'],'target_name':'A','rho':.5,
       'last_legal_select':{'action_kind':'select','action_valid':True,'decision_index':1,'x_seen_step':1,
                            'public_select_candidate_names':['A','B']}})
  self.assertTrue(all(r['applicable'] for r in descriptors(members).values()))
  self.assertFalse(pre_evidence(1,1));self.assertTrue(pre_evidence(0,1))

 def test_logged_valid_observations_are_partitioned_by_actual_after_mask(self):
  rows=[]
  for action in ('turn_left','turn_right'):
   ep,traces=self.trajectory([action,'select 999']);rows.append(self.recalc(ep,traces))
  result=diag.summarize(rows)
  self.assertEqual(result['valid_observation_actions'],2)
  self.assertEqual({k:result['observation_after_state_masks'][k] for k in ('zero','positive_below250','at_least250')},
                   {'zero':1,'positive_below250':0,'at_least250':1})

 def test_rehash_rejects_change_after_count_cache(self):
  h=self.index.header(self.key);self.index.state(self.key,h['initial_state'],0)
  ref=self.index.payload(self.key,h['states'][h['initial_state']]['captures'][0]['masks']['X'])
  Path(ref['file']).write_bytes(bytes([1])*(255*255))
  with self.assertRaisesRegex(ValueError,'Original bytes changed'):self.reads.recheck()


if __name__=='__main__':unittest.main()
