"""Independent version routing and derived-physics verification fixtures.

These call actual physics measurement functions on synthetic metadata. Native
PNG/action/SHA authentication remains mandatory in the unmodified check_pair
chain and is not asserted by these small measurement fixtures.
"""
import copy
import unittest

from src.envs import pair_v2_policy as old
from src.envs import pair_v2_stable as stable
from src.envs import pair_v2_simulator as engine
from src.envs import pair_v2_nearby_physics as nearby
from src.tools import merge_pair_v2 as merge


def point(x, y=1, z=0):
    return {'x':x,'y':y,'z':z}


def fixture():
    def obj(name, position, *, receptacle=False):
        return {'name':name,'objectId':name+'_id','objectType':name,'position':position,
                'rotation':point(0,0,0),'moveable':not receptacle,'pickupable':not receptacle,
                'receptacle':receptacle,'parentReceptacles':['Table_id'] if name=='X' else []}
    baseline=[obj('X',point(-1)),obj('A',point(.3)),obj('B',point(2.3)),
              obj('Table',point(5),receptacle=True),obj('Far',point(10))]
    reference={'file':'raw/controller/common.json','sha256':'1'*64}
    constructions=[]
    before=copy.deepcopy(baseline)
    for index,destination in enumerate((point(0),point(2))):
        requested=[{'objectName':o['name'],'position':copy.deepcopy(destination if o['name']=='X' else o['position']),
                    'rotation':copy.deepcopy(o['rotation'])} for o in before if o['moveable'] or o['pickupable']]
        after=copy.deepcopy(before)
        for item in after:
            if item['name']=='X':item['position']=copy.deepcopy(destination)
            if item['name']=='Far':item['position']['x']=10+.03*(index+1)
        placement={'X':{'action':'GetSpawnCoordinatesAboveReceptacle','success':True,
            'receptacle_name':'Table','selected_position':copy.deepcopy(destination),
            'coordinates':[copy.deepcopy(destination)]}}
        c={'before_objects':copy.deepcopy(before),'requested_poses':requested,'realized_objects':after,
           'edited_names':['X'],'native_is_scene_at_rest':True,'placement_evidence_by_name':placement,
           'pose_measurements':engine.pose_measurements(before,after),
           'settling':{'settled':True},'agent_unchanged':True,
           'agent_before':{'position':point(0)},'agent_after':{'position':point(0)},
           'edited':{'X':{'parent_membership':True,'position_error_m':0}}}
        c['common_baseline_check']=stable.common_baseline_check(baseline,after,['X'],reference)
        measured=old.physical_check(before,requested,after,['X'],native_is_scene_at_rest=True,
                                    placement_evidence_by_name=placement)
        c['valid_physical']=stable.augment_physical_result(measured,c)['valid_physical']
        constructions.append(c);before=copy.deepcopy(after)
    derived=nearby.measure_pair(constructions,baseline,reference)
    for c,m in zip(constructions,derived['worlds']):c['nearby_physics_measurement']=copy.deepcopy(m)
    row=copy.deepcopy(derived['pair'])
    return row,constructions,baseline,reference


class VersionedPhysicsCheckTests(unittest.TestCase):
    def test_historical_path_never_accepts_a_new_claim_without_source_opt_in(self):
        settings={'marked':True,'frozen':True,'clone_enabled':False,'preflight_version':'settled_baseline_window_v1'}
        self.assertFalse(merge.validate_nearby_physics_opt_in({},None,None,**settings))
        row,_,_,_=fixture()
        with self.assertRaises(ValueError):merge.validate_nearby_physics_opt_in(row,None,None,**settings)
        self.assertTrue(merge.validate_nearby_physics_opt_in(row,nearby.VERSION,nearby.POLICY,**settings))
        for change in ({'clone_enabled':True},{'frozen':False},{'preflight_version':'unknown'}):
            with self.assertRaises(ValueError):
                merge.validate_nearby_physics_opt_in(row,nearby.VERSION,nearby.POLICY,**(settings|change))
        with self.assertRaises(ValueError):
            merge.validate_nearby_physics_opt_in(row,'unknown',nearby.POLICY,**settings)

    def test_far_drift_is_recomputed_and_original_failed_physics_bytes_remain(self):
        row,constructions,baseline,reference=fixture()
        original=copy.deepcopy((row,constructions,baseline,reference))
        measured,proof=merge.verify_nearby_physics_measurements(row,constructions,baseline,reference)
        self.assertTrue(all(m['valid_physical'] for m in measured))
        self.assertFalse(proof['passes_original_physics_rule'])
        self.assertFalse(any(c['valid_physical'] for c in constructions))
        self.assertEqual((row,constructions,baseline,reference),original)
        self.assertEqual(proof['nearby_physics_context']['baseline_reference'],reference)
        self.assertIn('Table',proof['nearby_physics_context']['nearby_object_names'])

    def test_forged_pair_diagnostics_or_world_measurement_do_not_reuse_pass(self):
        for field in ('far_object_drift_max_m','far_drift_objects','passes_original_physics_rule','nearby_physics_context'):
            row,constructions,baseline,reference=fixture()
            row[field]=None
            with self.subTest(field=field),self.assertRaises(ValueError):
                merge.verify_nearby_physics_measurements(row,constructions,baseline,reference)
        row,constructions,baseline,reference=fixture()
        constructions[0]['nearby_physics_measurement']['valid_physical']=False
        with self.assertRaises(ValueError):merge.verify_nearby_physics_measurements(row,constructions,baseline,reference)

    def test_wrong_common_baseline_or_unverified_anchor_rule_is_not_silently_accepted(self):
        row,constructions,baseline,reference=fixture()
        with self.assertRaises(ValueError):
            merge.verify_nearby_physics_measurements(row,constructions,baseline,{**reference,'sha256':'2'*64})
        # Removing a non-collateral original reason is never among the new permissions.
        constructions[0]['native_is_scene_at_rest']=False
        with self.assertRaises(ValueError):merge.verify_nearby_physics_measurements(row,constructions,baseline,reference)



class SourcePhysicsRoutingTests(unittest.TestCase):
    """Use the existing source fixture; native/science boundaries stay explicit."""
    def fixture(self,directory):
        from test_g01_geometry_checkpoint import evidence_fixture
        from src.envs import pair_v2_g01_policy as policy
        f=evidence_fixture(directory,pool='direct',kind='direct_stay')
        f.row['mode']=f.row['proposal']['mode']='stay'
        f.row.update(policy.fields(nearby=True))
        cfg=f.data['config'];cfg.update(policy.declared(f.scene,nearby=True))
        cfg['shard']=copy.deepcopy({k:v for k,v in cfg.items() if k!='shard'})
        f.data['metrics']={k:copy.deepcopy(v) for k,v in cfg.items() if k!='shard'}
        return f

    def test_new_source_explicitly_forwards_scientific_version_and_keeps_old_gate_failure(self):
        import tempfile
        from test_g01_geometry_checkpoint import source_fixture
        from src.tools import t01v2_g01_review_checkpoint as review
        for failed in (False,True):
            with tempfile.TemporaryDirectory() as folder:
                f=self.fixture(folder)
                arguments={'raw_root':f.data['config']['raw_evidence_dir'],'field_of_view':60}
                def science(row,path,**kwargs):
                    self.assertEqual(kwargs,{**arguments,**nearby.fields()})
                    if failed:raise ValueError('original scientific failure retained')
                    return {'original_scientific_gate':True}
                with source_fixture(f,science) as (checked,*_):
                    if failed:
                        with self.assertRaisesRegex(ValueError,'original scientific failure'):
                            review.merge.check_pair(f.row,f.pair_path,**arguments)
                    else:
                        self.assertTrue(review.merge.check_pair(f.row,f.pair_path,**arguments)['original_scientific_gate'])
                    checked.assert_called_once()

    def test_source_version_cannot_be_overridden_by_caller_or_missing_pair_fields(self):
        import tempfile
        from test_g01_geometry_checkpoint import source_fixture
        from src.tools import t01v2_g01_review_checkpoint as review
        for failure in ('caller','row'):
            with tempfile.TemporaryDirectory() as folder:
                f=self.fixture(folder);kwargs={'raw_root':f.data['config']['raw_evidence_dir']}
                if failure=='caller':kwargs['nearby_physics_version']='unknown'
                else:f.row.pop('nearby_physics_version')
                with source_fixture(f,lambda *a,**k:(_ for _ in ()).throw(AssertionError('must stop before science'))):
                    with self.assertRaises(ValueError):review.merge.check_pair(f.row,f.pair_path,**kwargs)

    def test_engineering_smoke_is_rejected_before_independent_collection(self):
        import tempfile
        from unittest.mock import patch
        from src.tools import t01v2_g01_review_checkpoint as review
        from src.envs import pair_v2_g01_policy as policy
        with tempfile.TemporaryDirectory() as folder:
            f=self.fixture(folder);f.data['config']['g01_engineering_smoke']=copy.deepcopy(policy.SMOKE)
            # Source-policy decoding is a declared boundary here; the dispatch
            # must reject smoke even if it was validly decoded as engineering.
            with patch.object(review,'_BASE_LOAD',return_value=(f.data,f.source_ref)), \
                 patch.object(policy,'validate_source',return_value=None), \
                 patch.object(review,'verify_collection',side_effect=AssertionError('smoke is not collection')):
                with review.source_support():
                    with self.assertRaisesRegex(ValueError,'Engineering smoke'):
                        review.merge.load_source('synthetic-smoke')


class AdmissionPhysicsTests(unittest.TestCase):
    def fixture(self,directory,*,nearby_enabled=False,smoke=False):
        from pathlib import Path
        from src.tools import g01_admission as admission
        from src.tools import g01_parallel as parent
        from src.envs import pair_v2_g01_policy as policy
        partition='test' if smoke else 'train'
        scene='FloorPlan21' if smoke else 'FloorPlan1'
        name='gen_test_smoke_fixture' if smoke else 'gen_train_physics_fixture'
        child=name+'_c6_'+scene.lower()
        plan,ref=admission.create_plan(directory,parent_run_name=name,stage_index=0,concurrency=6,
            assignments=[{'lane_index':0,'scene':scene,'child_run_name':child}],
            pbs_jobid='700000.hopper-m-02',node='fixture',gpu_uuid='GPU-fixture',
            cpu_affinity=list(range(12)),scenes=[scene],partition=partition,
            nearby_physics=nearby_enabled,engineering_smoke=smoke)
        gate={'config':{'node':'fixture','allocation':{'pbs_jobid':'700000.hopper-m-02',
            'cuda_visible_devices':'GPU-fixture','devices':[{'gpu_uuid':'GPU-fixture'}]}}}
        gate_ref={'source':{'file':'results/gate.json','sha256':'a'*64},'result_path':'results/gate.json'}
        shard=parent.make_shard(child,scene,list(range(12)),gate,gate_ref,Path(directory)/scene,plan,ref)
        if nearby_enabled:shard.update(nearby.fields())
        if smoke:shard['g01_engineering_smoke']=copy.deepcopy(policy.SMOKE)
        cfg={'name':child,'scene':scene,'g01_partition':partition,'shard':shard}
        if nearby_enabled:cfg.update(nearby.fields())
        if smoke:cfg['g01_engineering_smoke']=copy.deepcopy(policy.SMOKE)
        return plan,cfg

    def test_original_plan_unchanged_and_new_fields_cannot_be_dropped_or_added(self):
        import tempfile
        from src.tools import g01_admission as admission
        for enabled in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                plan,cfg=self.fixture(directory,nearby_enabled=enabled)
                self.assertEqual(admission.validate_source_admission(cfg),plan)
                self.assertEqual('nearby_physics_enabled' in plan,enabled)
                for where in ('source','shard'):
                    changed=copy.deepcopy(cfg);value=changed if where=='source' else changed['shard']
                    if enabled:value.pop('nearby_physics_version')
                    else:value.update(nearby.fields())
                    with self.assertRaises(ValueError):admission.validate_source_admission(changed)

    def test_smoke_declaration_is_bound_without_changing_original_resource_checks(self):
        import tempfile
        from src.tools import g01_admission as admission
        with tempfile.TemporaryDirectory() as directory:
            plan,cfg=self.fixture(directory,nearby_enabled=True,smoke=True)
            self.assertEqual(admission.validate_source_admission(cfg),plan)
            for mutate in ('smoke','cpu'):
                changed=copy.deepcopy(cfg)
                if mutate=='smoke':changed.pop('g01_engineering_smoke')
                else:changed['shard']['cpu_affinity']=[0]
                with self.assertRaises(ValueError):admission.validate_source_admission(changed)


if __name__=='__main__':unittest.main()
