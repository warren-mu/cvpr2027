import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('diag01_learning_pool_tested', ROOT/'experiments/diag01_learning_pool.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)


def episode(key='p', world='w0', update=1, kind='look_stay', choice='A', success=True, reward=1., **extra):
    return {'dataset_key':key,'original_pair_id':key,'arm':'T0','phase':'train','update':update,
        'episode_id':f'{key}:{world}:{update}', 'world_id':world,'original_world_id':world,
        'kind':kind,'success':success,'task_reward_sum':reward,'selected_name':choice,
        'valid_terminal_selection':choice is not None, 'sampling_design':'bernoulli_prior',
        'pair_instance_id':f'{key}:{update}', 'valid_observation_actions':1, **extra}


def population(*keys):
    return {key:{'dataset_key':key,'original_pair_id':key,'scene':'FloorPlan1','kind':'look_stay'} for key in keys}


class DiagnosticTests(unittest.TestCase):
    def test_ten_update_window_uses_all_episodes_not_only_endpoint(self):
        rows=[episode(update=1,reward=.8),episode(update=10,success=False,reward=-.05),episode(update=11)]
        windows=d.class_windows(rows)
        first=windows[0]['by_class']['look_stay']
        self.assertEqual((first['episode_count'],first['success_count']),(2,1))
        self.assertAlmostEqual(first['mean_task_reward'],.375)
        self.assertEqual(windows[1]['by_class']['look_stay']['episode_count'],1)
        self.assertIsNone(windows[0]['by_class']['direct_flip']['success_rate'])

    def test_original_world_object_identity_not_screen_label(self):
        rows=[episode(world='w1'),episode(world='w0',update=2)]
        result=d.pair_diagnostics(rows,population('p'))['pairs'][0]['windows'][0]
        self.assertTrue(result['always_same_candidate_in_both_worlds'])
        self.assertEqual(result['original_worlds']['w1']['candidate_object_counts'],{'A':1})
        self.assertEqual(result['actual_paired_draw_count'],0)

    def test_missing_world_and_invalid_select_not_silently_counted_as_same(self):
        p=population('p')
        single=d.pair_diagnostics([episode()],p)['pairs'][0]['windows'][0]
        self.assertIsNone(single['always_same_candidate_in_both_worlds'])
        rows=[episode(),episode(world='w1',choice=None,success=False,reward=0)]
        result=d.pair_diagnostics(rows,p)['pairs'][0]['windows'][0]
        self.assertFalse(result['always_same_candidate_in_both_worlds'])
        self.assertEqual(result['original_worlds']['w1']['invalid_or_no_legal_select_count'],1)
        self.assertEqual(result['episode_count'],2)

    def test_true_paired_draw_has_two_worlds_and_preserves_half_success(self):
        rows=[episode(sampling_design='both_worlds',kind='look_flip'),
              episode(world='w1',sampling_design='both_worlds',kind='look_flip',success=False,reward=0)]
        p=population('p');p['p']['kind']='look_flip'
        result=d.pair_diagnostics(rows,p)['pairs'][0]['windows'][0]
        self.assertEqual(result['success_minus_half'],0)
        self.assertEqual(result['paired_same_candidate_rate'],1)
        with self.assertRaisesRegex(ValueError,'Incomplete actual paired'):
            d.pair_diagnostics(rows[:1],p)

    def test_actual_input_duplicate_not_hidden_by_different_native_RGB(self):
        rows=[{'dataset_key':k,'original_world_id':'w0','raw_rgb_sha256':k*64,
            'marked_rgb_sha256':k*64,'actual_input_rgb_sha256':'f'*64,'actual_input_size':[512,512]} for k in ('a','b')]
        report=d.s0_uniqueness(rows,population('a','b','c'))
        self.assertEqual(len(report['groups']['raw_rgb_sha256']['cross_pair_duplicate_groups']),0)
        self.assertEqual(len(report['groups']['actual_input_rgb_sha256']['cross_pair_duplicate_groups']),1)
        self.assertFalse(report['pairs'][0]['worlds']['w0']['all_observed_actual_inputs_unique_to_pair'])
        self.assertIsNone(report['pairs'][2]['worlds']['w0']['all_observed_actual_inputs_unique_to_pair'])

    def test_shadow_at_tau_and_strictly_after_are_separate(self):
        ep=episode(kind='look_flip')
        states=[{'observation_index':i,'state_id':sid,'own_anchor_pixels':pixels,'shared':shared}
                for i,sid,pixels,shared in [(0,'s0',0,True),(1,'s1',0,False),(2,'s1',0,False),(3,'s2',2,False)]]
        independent={**ep,'states':states,'independent_first_difference_step':1}
        row=d.shadow_diagnostics([ep],[independent])['windows'][0]
        self.assertEqual(row['post_tau_ownX_zero_state_occurrences'],2)
        self.assertEqual(row['strictly_after_tau_ownX_zero_state_occurrences'],1)
        self.assertEqual(row['unique_dataset_world_states'],1)
        self.assertEqual(row['visited_state_occurrences'],4)

    def test_missing_independent_mask_episode_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'D1 omitted'):
            d.shadow_diagnostics([episode(kind='look_flip')],[])

    def test_no_tau_and_no_valid_observation_not_shadow(self):
        ep=episode(kind='look_flip')
        independent={**ep,'states':[{'observation_index':0,'state_id':'s0','own_anchor_pixels':0,'shared':True}],
            'independent_first_difference_step':None}
        self.assertEqual(d.shadow_diagnostics([ep],[independent])['windows'][0]['post_tau_ownX_zero_state_occurrences'],0)
        ep['valid_observation_actions']=0
        self.assertEqual(d.shadow_diagnostics([ep],[])['windows'][0]['observing_LF_episode_count'],0)

    def test_sha_length_and_root_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();p=root/'x';p.write_bytes(b'abc')
            ref={'file':'x','sha256':hashlib.sha256(b'abc').hexdigest(),'bytes':3}
            self.assertEqual(d.read_bytes(ref,root=root),b'abc')
            with self.assertRaises(ValueError):d.read_bytes({**ref,'sha256':'0'*64},root=root)
            with self.assertRaises(ValueError):d.read_bytes({**ref,'bytes':4},root=root)
            with self.assertRaises(ValueError):d.read_bytes({**ref,'file':'../x'},root=root)


class OriginalFilterTests(unittest.TestCase):
    def rows(self):
        rows=[]
        for i in range(80):
            key=f'df{i}'
            rows.append({'dataset_key':key,'pair_id':key,'original_pair_id':key,'scene':'FloorPlan1',
                'kind':'direct_flip','candidate_names':['A','B'],'target_names':['A','B'],
                's0_target_left':[True,False],'synthetic':True,'pair_reference':{'file':key,'sha256':'a'*64}})
        for key,side,distance,scene in [('held','A',.01,'FloorPlan15'),('a','A',.15,'FloorPlan2'),('b','B',.15,'FloorPlan2')]:
            rows.append({'dataset_key':key,'pair_id':key,'original_pair_id':key,'scene':scene,
                'kind':'look_stay','candidate_names':['A','B'],'target_names':[side,side],
                's0_target_left':[True,False],'synthetic':True,'pair_reference':{'file':key,'sha256':'b'*64},
                'stay_displacement':{'distance_m':distance,'method':'synthetic fixture'}})
        return rows

    def inventory(self, rows):
        from src.tools.g01_train_inventory import postfilter
        return {'qualified_rows':rows,'additional_cache_sources':[],'known_rejection_sources':[],
                'input_metadata_references':[],**postfilter(rows,{'FloorPlan15'})}

    def test_holdout_first_exact_threshold_kept_and_original_balance_called(self):
        from src.tools.g01_train_inventory import postfilter,MIN_STAY_METRES
        inv=self.inventory(self.rows())
        result=d.pool_diagnostic(inv,{'selected_scenes':['FloorPlan15']},postfilter_fn=postfilter,minimum=MIN_STAY_METRES)
        self.assertEqual([r['counts_by_kind']['look_stay'] for r in result['stages']],[3,2,2,2])
        self.assertEqual(result['rejected_LS_distribution']['count'],0)
        self.assertFalse(result['formal_T04_pool_authorized'])

    def test_short_stay_rejected_then_opposite_side_balance_drops_rest(self):
        from src.tools.g01_train_inventory import postfilter,MIN_STAY_METRES
        rows=self.rows();rows[-2]['stay_displacement']['distance_m']=.149999
        result=d.pool_diagnostic(self.inventory(rows),{'selected_scenes':['FloorPlan15']},postfilter_fn=postfilter,minimum=MIN_STAY_METRES)
        self.assertEqual([r['counts_by_kind']['look_stay'] for r in result['stages']],[3,2,1,0])
        self.assertEqual(result['rejected_LS_displacements_metres'],[.149999])
        self.assertEqual(result['rejected_LS_pairs'][0]['dataset_key'],'a')

    def test_original_receipt_counts_cannot_be_replaced(self):
        from src.tools.g01_train_inventory import postfilter,MIN_STAY_METRES
        inv=self.inventory(self.rows());inv['confirmed_by_kind']['look_stay']=99
        with self.assertRaisesRegex(ValueError,'postfilter replay differs'):
            d.pool_diagnostic(inv,{'selected_scenes':['FloorPlan15']},postfilter_fn=postfilter,minimum=MIN_STAY_METRES)


class OriginalEpisodeInterfaceTests(unittest.TestCase):
    def fixture(self):
        from src.tools.smoke_e2e_pilot_contract import episode_observation, classified_observations
        episodes=[]; decisions=[]
        for i in range(64):
            plan={'episode_id':f'T0:update001:episode{i:03d}', 'pair_id':'p','dataset_key':'p',
                  'world_id':'w0','pair_instance_id':f'draw{i}','sampling_design':'bernoulli_prior'}
            final={'selected':True,'x_seen_step':None,'tau':None,'success':True,'selected_name':'A'}
            trace={**plan,'kind':'decision','update':1,'decision_index':0,'sample_id':f'sample{i}',
                   'parse_valid':True,'action_valid':True,'action_kind':'select','reward':1.,
                   'input':{'images':[{'mode':'RGB','size':[512,512],'sha256':'1'*64}]},
                   'before':{'candidates':[{'object_name':'A','success_eligible':True}],
                             'capture':{'state_id':'s0','raw_rgb_sha256':'2'*64,'marked_rgb_sha256':'3'*64}},
                   'after':final}
            observed=episode_observation(plan,[trace],kind='look_stay',final_info=final,selected_name='A',success=True)
            observed.update(record_kind='pilot_episode',update=1,original_pair_id='p',world_order=[1,0],original_world_id='w1')
            episodes.append(observed);decisions.append(trace)
        summary=classified_observations(episodes)
        complete={'record_kind':'pilot_update_complete','update':1,'actual_completed_updates':1,'by_class':summary}
        return decisions+episodes+[complete],summary

    def parse(self,records,summary):
        raw=b'\n'.join(json.dumps(r).encode() for r in records)+b'\n'
        return d.extract_journal(raw,arm='T0',update=1,population=population('p'),world_orders={'p':[1,0]},expected_summary=summary)

    def test_original_contract_real_64_episode_journal_to_diagnostic(self):
        records,summary=self.fixture();rows,images=self.parse(records,summary)
        self.assertEqual(len(rows),64);self.assertEqual(len(images),64)
        self.assertEqual(rows[0]['original_world_id'],'w1')
        self.assertEqual(images[0]['actual_input_size'],[512,512])

    def test_original_outcome_reward_world_and_completion_tampering_rejected(self):
        records,summary=self.fixture()
        for mutate in [lambda r:r[64].update(success=False),lambda r:r[64].update(task_reward_sum=0.),
                       lambda r:r[64].update(original_world_id='w0'),lambda r:r.pop()]:
            bad=copy.deepcopy(records);mutate(bad)
            with self.assertRaises(ValueError):self.parse(bad,summary)

    def test_incomplete_or_duplicate_episode_not_dropped(self):
        records,summary=self.fixture()
        for bad in [records[:64]+records[65:],records[:64]+[records[65]]+records[65:]]:
            with self.assertRaises(ValueError):self.parse(bad,summary)


if __name__=='__main__':unittest.main()
