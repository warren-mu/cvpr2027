"""CPU fixtures for existing-input seen regression and exact co-visibility BFS."""
import copy
import unittest

from experiments import diag02_visibility as diag
from src.envs.pair_table import label_assignment
from src.tools import smoke_e2e_pilot_contract as pilot


class Diag02VisibilityTests(unittest.TestCase):
    def history(self):
        plan={'episode_id':'e','pair_id':'key','dataset_key':'key','world_id':'w0'}
        info={'selected':True,'x_seen_step':1,'tau':1}
        traces=[{**plan,'decision_index':0,'action_kind':'observe','parse_valid':True,
                 'action_valid':True,'reward':-.05,'sample_id':'s0'},
                {**plan,'decision_index':1,'action_kind':'select','parse_valid':True,
                 'action_valid':True,'reward':1.,'sample_id':'s1','after':info}]
        episode=pilot.episode_observation(plan,traces,kind='look_flip',final_info=info,
                                         selected_name='A',success=True)
        episode.update(seen_before_select=False,seen_before_select_reason='anchor_not_strictly_before_select')
        independent={'seen_before_select':True,'logged_seen_before_select':False,'logged_x_seen_step':1}
        return episode,traces,independent

    def test_regression_calls_production_preserves_original_and_changes_only_seen(self):
        episode,traces,independent=self.history(); before=copy.deepcopy(episode)
        row=diag.recompute_observation(episode,traces,independent)
        self.assertEqual(episode,before)
        self.assertFalse(row['historical_seen']);self.assertTrue(row['corrected_seen'])
        self.assertTrue(row['all_other_episode_fields_unchanged'])
        self.assertEqual(diag.regression_summary([row])['corrected_vs_independent_mismatches'],0)

    def test_regression_fails_on_non_seen_outcome_or_independent_disagreement(self):
        episode,traces,independent=self.history()
        episode['task_reward_sum']=123
        with self.assertRaisesRegex(ValueError,'non-seen outcome'):
            diag.recompute_observation(episode,traces,independent)
        episode,traces,independent=self.history();independent['seen_before_select']=False
        with self.assertRaisesRegex(ValueError,'independent raw-mask'):
            diag.recompute_observation(episode,traces,independent)

    def header(self):
        return {'initial_state':'s0','states':{
            's0':{'transitions':{'turn right':'s2','turn left':'s1','move 1':'s3'}},
            's1':{'transitions':{'turn right':'s0','move 2':'s4'}},
            's2':{'transitions':{'move 2':'s4'}},
            's3':{'transitions':{'move 1':'s5'}},
            's4':{'transitions':{'move 1':'s6'}},
            's5':{'transitions':{'move 1':'s7'}},
            's6':{'transitions':{'move 1':'s8'}},
            's7':{'transitions':{}},
            's8':{'transitions':{'move 1':'s9'}},
            's9':{'transitions':{}}}}

    def test_shortest_paths_are_directed_bounded_and_lexical_not_success_ranked(self):
        paths=diag.shortest_paths(self.header())
        self.assertEqual(paths['s4'],['turn left','move 2'])
        self.assertEqual(len(paths['s8']),4)
        self.assertNotIn('s9',paths)
        self.assertEqual(paths['s0'],[])

    def test_all_revealing_actions_are_reported_and_shortest_cov_may_require_move(self):
        def measure(state):
            visible=state in ('s1','s2','s7','s8')
            both=state in ('s7','s8')
            return {'state_id':state,'anchor_visible_250':visible,'both_candidates_marked':both,
                    'anchor_and_both_marked':visible and both}
        row=diag.describe_world(self.header(),1,measure)
        self.assertEqual([r['action'] for r in row['one_step_revealing_actions']],['turn left','turn right'])
        self.assertFalse(row['has_one_step_reveal_with_both_marked'])
        self.assertTrue(row['has_cov_state_within_D'])
        self.assertEqual(row['minimum_observation_steps_to_cov'],3)
        self.assertEqual(row['shortest_cov_witness']['actions'],['move 1','move 1','move 1'])

    def test_no_reveal_is_not_vacuously_all_reveals_co_visible(self):
        row=diag.describe_world(self.header(),0,lambda state:{'state_id':state,
            'anchor_visible_250':False,'both_candidates_marked':True,'anchor_and_both_marked':False})
        self.assertFalse(row['has_one_step_reveal'])
        self.assertFalse(row['all_one_step_reveals_have_both_marked'])
        summary=diag.cov_summary([row])
        self.assertEqual(summary['no_cov_state_within_D'],1)
        self.assertIsNone(summary['fraction_revealing_actions_with_both_marked'])

    class Index:
        def __init__(self,counts,distances,*,shared=False,anchor_pixels=250):
            self.counts=counts;self.anchor_pixels=anchor_pixels
            self.h={'pair_id':'p','candidate_names':['A','B'],'states':{'s':{'shared':shared,
                'captures':[{'masks':{n:{'file':f'{w}:{n}'} for n in ('A','B')},
                             'distances':distances[w]} for w in (0,1)]}}}
            self.reads=self
        def header(self,key):return self.h
        def state(self,key,state,world):return {'own_anchor_pixels':self.anchor_pixels,
            'anchor_mask_reference':{'file':f'own{world}:X'}}
        def payload(self,key,ref):return ref
        def mask(self,ref):return self.counts[ref['file']]

    def test_marking_uses_visible_positive_pixels_and_inclusive_distance_limit(self):
        index=self.Index({'0:A':1,'0:B':249,'1:A':0,'1:B':1},
                         [{'A':1.5,'B':1.5000001},{'A':.3,'B':None}])
        row=diag.cov_measurement(index,'key','s',0)
        self.assertTrue(row['anchor_visible_250'])
        self.assertTrue(row['candidate_measurements'][0]['success_eligible'])
        self.assertFalse(row['candidate_measurements'][1]['success_eligible'])
        self.assertFalse(row['anchor_and_both_marked'])
        other=diag.cov_measurement(index,'key','s',1)
        self.assertFalse(any(r['success_eligible'] for r in other['candidate_measurements']))

    def test_public_shared_capture_and_fixed_object_labels_are_used(self):
        index=self.Index({'0:A':1,'0:B':1,'1:A':0,'1:B':0},
                         [{'A':1.5,'B':.2},{'A':1.5,'B':.2}],shared=True,anchor_pixels=0)
        row=diag.cov_measurement(index,'key','s',1)
        self.assertEqual(row['public_capture_original_world_id'],'w0')
        self.assertTrue(row['both_candidates_marked'])
        self.assertFalse(row['anchor_and_both_marked'])
        labels=label_assignment('p',['A','B'])
        self.assertEqual({r['name']:r['label_if_marked'] for r in row['candidate_measurements']},labels)

    def test_anchor_at_249_does_not_count_as_revealed(self):
        index=self.Index({'0:A':1,'0:B':1,'1:A':1,'1:B':1},
                         [{'A':1.,'B':1.},{'A':1.,'B':1.}],anchor_pixels=249)
        row=diag.cov_measurement(index,'key','s',0)
        self.assertTrue(row['both_candidates_marked'])
        self.assertFalse(row['anchor_visible_250'])
        self.assertFalse(row['anchor_and_both_marked'])


if __name__=='__main__':unittest.main()
