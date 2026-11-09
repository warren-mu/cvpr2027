"""Regression for real rejected-row compaction before SceneQuota.record.

The fourth same-reason rejection may omit raw images. Its original view and
stable identities remain needed by the unchanged configuration proof.
"""
import copy
from pathlib import Path
import unittest
from src.envs import pair_v2_configuration as configurations
from src.envs import pair_v2_g01_search as search
from src.envs import pair_v2_g01_shard as shard


def rejected():
    row={'pair_id':'floorplan321_00003','scene':'FloorPlan321','kind':None,'mode':'stay',
        'proposed_pool':'look','status':'rejected','counts_as_pair_attempt':True,
        'discard_reasons':['w0_physical'],'pair_file':None,'pair_sha256':None,
        'view':{'position':{'x':0,'y':0.9,'z':1},'yaw':270,'horizon':30},
        'identities':{'anchor_name':'Book_X','candidate_names':['Bowl_A','Vase_B']},
        'evidence_retention':{'full_raw_retained':False,'reason':'compact_rejection',
                              'evidence_complete_for_acceptance':False}}
    row['g01_configuration']=configurations.from_pair(row)
    return row


def quota():
    limits={kind:2 for kind in search.KINDS}
    return search.SceneQuota(6,actual_output_caps=limits,allowed_kinds=list(search.KINDS),
                             scene='FloorPlan321',seen_configurations=[])


class CompactConfigurationTests(unittest.TestCase):
    def test_actual_compact_then_record_keeps_original_inputs_and_records_rejection(self):
        row=rejected();original=copy.deepcopy(row);counter=quota()
        compact=shard.compact_attempt(row)
        self.assertEqual(compact['view'],row['view'])
        self.assertEqual(compact['identities'],row['identities'])
        self.assertEqual(compact['g01_configuration'],configurations.from_pair(compact))
        self.assertEqual(row,original)
        counter.record(Path('raw/attempt.json'),compact)
        self.assertEqual(counter.attempts['look_stay'],1)
        self.assertEqual(counter.passed,{kind:0 for kind in search.KINDS})
        self.assertEqual(counter.configuration_seen,{})
        self.assertFalse(compact['evidence_retention']['evidence_complete_for_acceptance'])

    def test_first_retained_then_first_compact_attempt_use_same_literal_configuration(self):
        counter=quota()
        for index in range(4):
            row=rejected();row['pair_id']=f'floorplan321_{index:05d}'
            row['evidence_retention']['full_raw_retained']=index<3
            stored=row if index<3 else shard.compact_attempt(row)
            counter.record(Path(f'raw/trial{index}/attempt.json'),stored)
        self.assertEqual(counter.attempts['look_stay'],4)
        self.assertFalse(counter.configuration_seen)

    def test_changed_original_input_or_derived_identity_is_still_rejected(self):
        for field in ('view','identities','g01_configuration'):
            row=shard.compact_attempt(rejected())
            if field=='view':row['view']['position']['x']=1e-12
            elif field=='identities':row['identities']['anchor_name']='Other_X'
            else:row['g01_configuration']['configuration_id']='0'*64
            with self.assertRaises(ValueError):quota().record(Path('raw/attempt.json'),row)

    def test_missing_inputs_remain_error_and_legacy_compaction_has_no_new_claim(self):
        row=rejected();row.pop('view')
        with self.assertRaises(KeyError):shard.compact_attempt(row)
        row=rejected();row.pop('g01_configuration')
        compact=shard.compact_attempt(row)
        self.assertNotIn('view',compact);self.assertNotIn('identities',compact)
        self.assertNotIn('g01_configuration',compact)


if __name__=='__main__':unittest.main()
