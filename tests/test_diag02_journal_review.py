"""Handmade identity, partial-coverage, cov-existence and cache controls."""
import copy
import hashlib
import json
import unittest

from experiments.diag02_journal_review import (STRATEGIES, cache_review,
    cov_crosscheck, direct_review, e2_world_index, expected_forced, inspect_records, uncached_review)


def entry():
    return {'dataset_key':'dataset','original_pair_id':'pair','kind':'direct_flip','world_order':[1,0]}


def model_sample(*, reused=False):
    proof={'public_image_sha256':['b'*64], 'guided_select_choices':['select 1','select 2']}
    descriptor={'input':proof,'model':'public-model-proof','sampling':{'temperature':0}}
    key=hashlib.sha256(json.dumps(descriptor,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return {'forced_observation':False,'response':'select 1','action':'select 1',
        'sample_id':'sample','response_token_ids':[11,12],'response_token_count':2,
        'finish_reason':'stop','stop_reason':None,'input':proof,'request_descriptor':descriptor,
        'exact_input_cache':{'key':key,'origin_sample_id':'sample','reused_generation':reused}}


def direct_records(world='w0', strategy='obs_never'):
    original='w1' if world=='w0' else 'w0';eid=strategy+'-'+world
    state={'state_id':'s0','step':0,'original_world_id':original,
        'anchor_pixels':300,'anchor_mask_sha256':'a'*64,'shared':False}
    after={**state,'step':1}
    decision={'record_kind':'decision','strategy':strategy,'dataset_key':'dataset','world_id':world,
        'episode_id':eid,'decision_index':0,'before_state':state,'after_state':after,
        'sample':model_sample(),'both_candidates_marked_before':True,'public_rgb_sha256':'b'*64,
        'reward':1.,'action_valid':True,'done':True,'selected_object':'A','success':True}
    row={'record_kind':'episode','strategy':strategy,'dataset_key':'dataset','world_id':world,
        'original_world_id':original,'world_order':[1,0],'pair_id':'pair','pair_kind':'direct_flip',
        'episode_id':eid,'states':[state,after],'actions':['select 1'],'steps':1,
        'selected':True,'selected_object':'A','success':True,'independent_first_seen_step':0,
        'seen_before_select':True,'seen_anywhere':True,'positive_anchor_anywhere':True,
        'model_outputs':1,'model_calls':1,'syntax_failures':0,'observation_count':0,
        'observation_attempts':0,'observation_reward_cost':0.,'task_reward':1.,
        'selection_frame_anchor_pixels':300,'selection_frame_both_candidates_marked':True}
    return [decision,row]


def known_cov():
    return {'world':{'has_cov_state_within_D':True,'has_one_step_reveal_with_both_marked':True,
        'minimum_observation_steps_to_cov':1,'all_one_step_actions':[{'state_id':'reveal','anchor_and_both_marked':True}]},
        'states':{'reveal':{'state_id':'reveal','own_anchor_pixels':300,
            'anchor_mask_reference':{'sha256':'a'*64},'both_candidates_marked':True,'anchor_and_both_marked':True},
            's0':{'state_id':'s0','own_anchor_pixels':0,
            'anchor_mask_reference':{'sha256':'b'*64},'both_candidates_marked':True,'anchor_and_both_marked':False}}}


def cov_row(state='reveal'):
    pixels=300 if state=='reveal' else 0;sha='a'*64 if state=='reveal' else 'b'*64
    before={'state_id':state,'anchor_pixels':pixels,'anchor_mask_sha256':sha}
    return {'dataset_key':'dataset','pair_id':'pair','pair_kind':'look_flip','world_id':'w0','original_world_id':'w1',
        'selected':True,'selection_frame_anchor_pixels':pixels,'selection_frame_both_candidates_marked':True,
        'states':[before,before],'observation_count':1,'success':True}


class ReviewTests(unittest.TestCase):
    def test_cov_only_shortcuts_actual_visible_marked_state(self):
        self.assertIsNone(expected_forced('obs_trigger_cov',1,1,250,True))
        self.assertEqual(expected_forced('obs_trigger_cov',1,1,249,True),'turn_right')
        self.assertEqual(expected_forced('obs_trigger_cov',1,1,300,False),'turn_right')
        self.assertIsNone(expected_forced('obs_trigger_cov',2,1,0,True))
        self.assertEqual(expected_forced('obs_trigger',1,1,300,True),'turn_right')
        self.assertIsNone(expected_forced('obs_never',0,None,0,True))

    def test_cov_actual_does_not_equal_graph_existence(self):
        actual=cov_crosscheck(cov_row(),known_cov())
        self.assertTrue(actual['actual_cov_selection'])
        fallback=cov_crosscheck(cov_row('s0'),known_cov())
        self.assertTrue(fallback['e2_has_cov_state_within_D'])
        self.assertFalse(fallback['actual_cov_selection'])

    def test_cov_mask_and_eligibility_are_bound_to_original_world(self):
        row=cov_row();row['states'][-2]['anchor_mask_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'mask count/SHA'):cov_crosscheck(row,known_cov())
        known=known_cov();known['states']['reveal']['both_candidates_marked']=False
        with self.assertRaisesRegex(ValueError,'Selection frame'):cov_crosscheck(cov_row(),known)
        known=known_cov();known['world']['all_one_step_actions']=[]
        with self.assertRaisesRegex(ValueError,'one-step'):cov_crosscheck(cov_row(),known)

    def test_duplicate_e2_world_is_rejected(self):
        world={'dataset_key':'d','original_world_id':'w0','reachable_state_measurements':[]}
        with self.assertRaisesRegex(ValueError,'Duplicate E2 original'):e2_world_index({'E2':{'worlds':[world,world]}})

    def test_empty_source_remains_incomplete_not_zero_accuracy(self):
        out=inspect_records([],'obs_never',{'dataset':entry()},{},require_complete=False)
        self.assertFalse(out['coverage']['complete'])
        self.assertEqual(out['coverage']['expected_episodes'],2)
        self.assertEqual(len(out['coverage']['missing_identities']),2)
        self.assertIsNone(out['summary']['all']['success_rate'])
        with self.assertRaisesRegex(ValueError,'incomplete'):
            inspect_records([],'obs_never',{'dataset':entry()},{},require_complete=True)

    def test_swapped_worlds_and_interleaved_completion_use_identities(self):
        left,right=direct_records('w0'),direct_records('w1')
        out=inspect_records([left[0],right[0],right[1],left[1]],'obs_never',{'dataset':entry()},{},require_complete=True)
        self.assertTrue(out['coverage']['complete'])
        self.assertEqual(out['summary']['all']['success_rate'],1.)
        self.assertEqual(out['episodes'][0]['original_world_id'],'w0')
        wrong=copy.deepcopy(left);wrong[1]['original_world_id']='w0'
        with self.assertRaisesRegex(ValueError,'Original world mapping'):
            inspect_records(wrong,'obs_never',{'dataset':entry()},{},require_complete=False)

    def test_duplicate_terminal_and_unfinished_episode_are_explicit(self):
        rows=direct_records()
        with self.assertRaisesRegex(ValueError,'Duplicate terminal'):
            inspect_records(rows+[rows[-1]],'obs_never',{'dataset':entry()},{},require_complete=False)
        out=inspect_records([rows[0]],'obs_never',{'dataset':entry()},{},require_complete=False)
        self.assertEqual(out['coverage']['unfinished_episode_ids'],['obs_never-w0'])
        self.assertEqual(out['coverage']['observed_completed_episodes'],0)

    def test_cache_reuse_requires_identical_complete_request_and_output(self):
        original=model_sample();reused=model_sample(reused=True)
        out=cache_review([reused,original],require_complete=True)
        self.assertEqual(out['observed_logical_requests'],2)
        self.assertEqual(out['observed_unique_request_keys'],1)
        mutated=copy.deepcopy(reused);mutated['response_token_ids']=[11,13]
        with self.assertRaisesRegex(ValueError,'inconsistent outputs'):
            cache_review([original,mutated],require_complete=True)
        with self.assertRaisesRegex(ValueError,'more than once'):
            cache_review([original,original],require_complete=True)
        with self.assertRaisesRegex(ValueError,'no generation origin'):
            cache_review([reused],require_complete=True)
        cache_review([reused],require_complete=False)

    def test_direct_controls_count_only_observed_three_way_matches(self):
        inspected={s:inspect_records(direct_records(strategy=s),s,{'dataset':entry()},{},require_complete=False) for s in STRATEGIES}
        out=direct_review(inspected,{'dataset':entry()},require_complete=False)
        self.assertFalse(out['full_coverage'])
        self.assertEqual(out['expected_controls'],2)
        self.assertEqual(out['observed_complete_controls'],1)
        with self.assertRaisesRegex(ValueError,'missing Direct'):
            direct_review(inspected,{'dataset':entry()},require_complete=True)
        inspected['obs_trigger']['decisions_by_identity'][('dataset','w0')][0]['sample']['response']='select 2'
        with self.assertRaisesRegex(ValueError,'exactly input/output'):
            direct_review(inspected,{'dataset':entry()},require_complete=False)

    def test_uncached_token_only_difference_is_not_action_difference(self):
        inspected={'obs_never':inspect_records(direct_records(),'obs_never',{'dataset':entry()},{},require_complete=False)}
        original=model_sample();changed=copy.deepcopy(original);changed['response_token_ids']=[11,13]
        def probe(name,sample,changed_mixed,changed_prior):
            return {'name':name,'status':'completed','episodes':1,
                'mixed_batch_origin_differences':int(changed_mixed),'prior_same_order_differences':None if changed_prior is None else int(changed_prior),
                'comparisons':[{'dataset_key':'dataset','world_id':'w0','sample':sample,'request_key':sample['exact_input_cache']['key'],
                    'different_from_mixed_batch_origin':changed_mixed,'different_from_prior_same_order':changed_prior}]}
        report={'uncached_probes':[probe('direct_same_order_1',original,False,None),
            probe('direct_same_order_2',changed,True,True),{'name':'direct_reversed','status':'not_run_deadline_reserve'}]}
        out=uncached_review(report,inspected)
        counts=out['probes'][1]['differences']['first_same_order']
        self.assertEqual(counts['complete_output_identity'],1)
        self.assertEqual(counts['response_token_ids'],1)
        self.assertEqual(counts['action'],0)
        self.assertEqual(counts['response'],0)
        self.assertEqual(out['probes'][2]['status'],'not_run_deadline_reserve')


if __name__=='__main__':unittest.main()
