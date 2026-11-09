"""Short-scope scheduling/policy fixtures, not evidence of simulated physics."""
import copy
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch
from src.tools import g01_physics_smoke as smoke
from src.tools import g01_parallel as parent
from src.tools import g01_quota_plan as quotas


def plan(name='gen_test_s16_smoke_fixture'):
    kinds=quotas.KINDS
    return {'version':quotas.CONFIG_VERSION,'partition':'test','parent_run_name':name,
        'count_stage':quotas.STAGES['test'],'targets':dict.fromkeys(kinds,60),
        'confirmed_by_kind':dict.fromkeys(kinds,0),'confirmed_by_scene':{},'confirmed_configurations':[],
        'count_basis_sources':[{'file':'results/fixture.json','sha256':'a'*64}],
        'outstanding_reservations':[],
        'assignments':[{'scene':'FloorPlan21','actual_output_caps':dict.fromkeys(kinds,1),'allowed_kinds':list(kinds)}]}


def child(*,status='incomplete',both=1,rescued=1,code=1):
    return {'config':{'scene':smoke.SCENE,'timeout_seconds':600,'g01_engineering_smoke':{
        'version':smoke.VERSION,'formal_collection_eligible':False}},'_meta':{'exit_code':code,'command':['python3','-u','-m',parent.policy.PRODUCER,'--timeout-seconds','600']},
        'metrics':{'status':status,'cleanup_errors':[],'final_metrics_verified':True,
            'terminal_verified_after_cleanup':True,'nearby_physics_counts':{'counted_attempts':3,
            'both_worlds_pass_new_physics':both,'pass_new_but_fail_original_physics':rescued}}}


class PhysicsSmokeTests(unittest.TestCase):
    def test_exact_one_scene_configuration_plan_without_formal_reservations(self):
        value=plan();self.assertIs(smoke.validate_quota(value,value['parent_run_name']),value)
        for mutate in (lambda p:p['assignments'].append({**p['assignments'][0],'scene':'FloorPlan22'}),
                       lambda p:p.update(version=quotas.VERSION),
                       lambda p:p.update(parent_run_name='gen_test_formal_other')):
            wrong=copy.deepcopy(value);mutate(wrong)
            with self.assertRaises(ValueError):smoke.validate_quota(wrong,value['parent_run_name'])

    def test_context_changes_only_short_parent_and_restores_on_exception(self):
        keys=('SCENES','CONCURRENCIES','PARTITION','SCENE_SECONDS','required_timeout',
              'submission_identity','make_shard','write_result','finish_logged','create_plan')
        before={key:getattr(parent,key) for key in keys}
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with smoke.installed('gen_test_s16_smoke_fixture',{'file':'configs/q.json','sha256':'a'*64}):
                self.assertEqual(parent.SCENES,('FloorPlan21',));self.assertEqual(parent.CONCURRENCIES,(6,))
                self.assertEqual(parent.SCENE_SECONDS,600);self.assertEqual(parent.required_timeout(),1000)
                self.assertEqual(len(parent.assignments('gen_test_s16_smoke_fixture',0,6)),1)
                self.assertEqual(parent.full_scene_grant(720,0),600)
                with self.assertRaises(TimeoutError):parent.full_scene_grant(719,0)
                raise RuntimeError('fixture')
        for key,value in before.items():self.assertEqual(getattr(parent,key),value)

    def test_actual_new_policy_receives_config_quota_nearby_and_smoke_flags(self):
        name='gen_test_s16_smoke_fixture';value=plan(name);ref={'file':'configs/q.json','sha256':'a'*64}
        with patch.object(quotas,'load_plan',return_value=(value,quotas.validate_structure(value),{'fixture':'auth boundary'})) as load, \
             patch.object(parent,'make_shard',return_value={'source_fixture':'original-shard-boundary'}):
            with smoke.installed(name,ref) as state:
                load.assert_not_called()  # Auth happens at original after-gate shard dispatch.
                actual=parent.make_shard(name+'_c6_floorplan21','FloorPlan21')
                self.assertEqual(actual['g01_policy']['scene_timeout_seconds'],600)
                self.assertEqual(actual['g01_engineering_smoke']['version'],smoke.VERSION)
                self.assertFalse(actual['g01_engineering_smoke']['formal_collection_eligible'])
                self.assertEqual(actual['nearby_physics_version'],'nearby_nonanchor_0_5m_v1')
                self.assertEqual(actual['g01_quota_assignment']['plan_reference'],ref)
                self.assertIn('g01_configuration_version',actual)
                self.assertIsNotNone(state['proof'])
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.0,0.0))

    def test_real_short_shard_and_admission_are_bound_and_formal_checker_rejects(self):
        import hashlib
        from src.tools import g01_admission as admission
        from src.envs import pair_v2_g01_shard as worker
        from src.envs import pair_v2_g01_policy as policy
        name='gen_test_s16_smoke_fixture';value=plan(name)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'configs').mkdir()
            qfile=root/'configs/q.json';qfile.write_text(json.dumps(value));qfile.chmod(0o444)
            ref={'file':'configs/q.json','sha256':hashlib.sha256(qfile.read_bytes()).hexdigest()}
            directory=root/'scratch/c6';directory.mkdir(parents=True)
            gpu='GPU-11111111-2222-3333-4444-555555555555'
            with patch.object(quotas,'ROOT',root),patch.object(quotas,'authenticate_count_basis',return_value={'fixture':'count basis boundary'}):
                with smoke.installed(name,ref):
                    assignments=parent.assignments(name,0,6)
                    resource,resource_ref=parent.create_plan(directory,parent_run_name=name,stage_index=0,concurrency=6,
                        assignments=assignments,pbs_jobid='123.hopper-m-02',node='synthetic-node',gpu_uuid=gpu,
                        cpu_affinity=list(range(12)),scenes=['FloorPlan21'],partition='test')
                    gate={'config':{'node':'synthetic-node','allocation':{'pbs_jobid':'123.hopper-m-02',
                        'cuda_visible_devices':gpu,'devices':[{'gpu_uuid':gpu}]}}}
                    gate_row={'source':{'file':'results/fixture_gate.json','sha256':'b'*64},'result_path':'results/fixture_gate.json'}
                    raw=directory/'FloorPlan21';raw.mkdir()
                    child_name=assignments[0]['child_run_name']
                    shard=parent.make_shard(child_name,'FloorPlan21',list(range(12)),gate,gate_row,raw,resource,resource_ref)
                    config={'name':child_name,'scene':'FloorPlan21','shard':shard,'field_of_view':60,
                        'raw_evidence_dir':str(raw),'max_attempts':160,'max_raw_bytes':512*1024*1024,
                        'resolution':255,'deadline':10000000000}
                    with worker.installed():worker.enable_config(config)
                    policy.validate_source(config)
                    admission.validate_source_admission(config)
                    self.assertEqual(resource['g01_engineering_smoke'],config['g01_engineering_smoke'])
                    self.assertEqual(config['g01_policy']['scene_timeout_seconds'],600)
                    bad=copy.deepcopy(config);bad['shard'].pop('g01_engineering_smoke')
                    with self.assertRaises(ValueError):admission.validate_source_admission(bad)
                    from src.tools import t01v2_g01_review_checkpoint as review
                    with review.source_support(),self.assertRaisesRegex(ValueError,'Engineering smoke'):
                        review.merge.shard_producer(['python3','-u','-m',policy.PRODUCER,
                            '--timeout-seconds','600'],config)

    def test_exact_native_submission_rejects_changed_scene_time_gpu_or_host(self):
        name='gen_test_s16_smoke_fixture';ref={'file':'configs/q.json','sha256':'a'*64}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'logs/pbs'/f'{name}-fixture';folder.mkdir(parents=True)
            payload={'name':name,'task_id':'T01','budget_group':'G01','queue':'autox','requested_gpu_count':1,
                'requested_wall_seconds':1030,'requested_host':None,
                'command':['python3','-u','-m',smoke.MODULE,'--name',name,'--scene','FloorPlan21',
                    '--quota-plan',ref['file'],'--quota-plan-sha256',ref['sha256'],'--timeout-seconds','1000']}
            (folder/'submission.json').write_text(json.dumps({'returncode':0,'stdout':'123.hopper-m-02\n'}))
            for changed in [None,'requested_gpu_count','requested_wall_seconds','requested_host','command']:
                candidate=copy.deepcopy(payload)
                if changed=='command':candidate[changed][-1]='999'
                elif changed=='requested_host':candidate[changed]='hopper-28'
                elif changed:candidate[changed]=4
                (folder/'payload.json').write_text(json.dumps(candidate))
                if changed:
                    with self.assertRaises(ValueError):smoke.submission_identity(root,name,'123.hopper-m-02',1000,ref)
                else:self.assertEqual(smoke.submission_identity(root,name,'123.hopper-m-02',1000,ref)['requested_wall_seconds'],1030)

    def test_sealed_deadline_exit_preserved_and_not_formal_accepted(self):
        row=smoke.summarize_child(child())
        self.assertEqual(row['original_exit_code'],1)
        self.assertTrue(row['engineering_gate_passed'])
        self.assertEqual(row['physics_counts']['pass_new_but_fail_original_physics'],1)
        self.assertFalse(row['formal_collection_eligible']);self.assertFalse(row['quota_occupancy_authorized'])
        for data in (child(status='worker_failed'),child(both=0,rescued=0)):
            self.assertFalse(smoke.summarize_child(data)['engineering_gate_passed'])
        for field,value in [('cleanup_errors',['unknown']),('final_metrics_verified',False),('nearby_physics_counts',None),('error','TimeoutError: Native action FIFO timeout'),('errors',['unrelated'])]:
            data=child();data['metrics'][field]=value
            self.assertFalse(smoke.summarize_child(data)['engineering_gate_passed'])

    def test_exact_bounded_deadline_is_allowed_but_other_timeouts_are_not(self):
        data=child();data['metrics']['error']='TimeoutError: T01v2 simulator deadline reached'
        row=smoke.summarize_child(data)
        self.assertTrue(row['expected_scene_deadline']);self.assertTrue(row['engineering_gate_passed'])
        data['metrics']['error']='TimeoutError: Error encountered when running action TeleportFull'
        self.assertFalse(smoke.summarize_child(data)['engineering_gate_passed'])

    def test_formal_opt_in_does_not_change_historical_default_policy(self):
        from src.tools.g01_collect_parallel import apply_source_policy
        from src.envs import pair_v2_g01_policy as policy
        value=plan();ref={'file':'configs/q.json','sha256':'a'*64}
        assignment=quotas.assignment(value,ref,'FloorPlan21',value['parent_run_name']+'_c6_floorplan21')
        old=policy.declared('FloorPlan21','test',assignment)
        self.assertEqual(apply_source_policy({},'FloorPlan21','test',assignment),old)
        future=apply_source_policy({},'FloorPlan21','test',assignment,nearby=True)
        self.assertEqual(future['g01_policy']['scene_timeout_seconds'],1200)
        self.assertNotIn('g01_engineering_smoke',future)
        self.assertEqual(future['nearby_physics_version'],'nearby_nonanchor_0_5m_v1')

    def test_smoke_writer_names_actual_scope_and_does_not_rewrite_exit(self):
        ref={'file':'configs/q.json','sha256':'a'*64};metrics={'exit_code':1,'cleanup_errors':[]};config={}
        with patch.object(parent,'write_result',return_value=None),patch.object(parent,'finish_logged',return_value=({},child())):
            with smoke.installed('gen_test_s16_smoke_fixture',ref):
                parent.finish_logged({'scene':'FloorPlan21'},[])
                parent.write_result('gen_test_s16_smoke_fixture',metrics,config)
        self.assertEqual(metrics['kind'],'g01_s16_engineering_smoke')
        self.assertEqual(metrics['exit_code'],1);self.assertTrue(metrics['engineering_gate_passed'])
        self.assertFalse(config['formal_collection_eligible'])
        self.assertEqual(config['actual_scene_assignments'],1)
        self.assertEqual(config['actual_controller_upper_bound'],2)


if __name__=='__main__':unittest.main()
