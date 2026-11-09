"""Synthetic allocation fixtures; no scheduler, simulator, or source-data scan."""
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
import types
import unittest
from unittest.mock import patch
from src.tools import pbs_mixed_accounting as mixed
SPEC=importlib.util.spec_from_file_location('mixed_phase',Path(__file__).resolve().parents[1]/'scripts/pbs/submit_phase.py')
phase=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(phase)
GROUPS=['G01','T02','E2Epilot']
UUIDS=[f'GPU-0000000{i}-0000-0000-0000-000000000000' for i in range(4)]
def binding(groups=GROUPS):return {'version':mixed.VERSION,'possible_groups':list(groups),'residual_priority':list(groups)}
def run(group='G01',gpu=UUIDS[0],start=10,finish=1010):
 return {'budget_group':group,'gpu_uuid':gpu,'started_monotonic':start,'finished_monotonic':finish,'duration_seconds_monotonic':finish-start,'cleanup_errors':[]}
def state():
 return {'submissions':[],'charge_policy':{'rates':{'smallx':2,'mediumx':2,'largex':2}},'accounted_or_reserved_total':0,'accounted_or_reserved_by_task':{'T00b':0,'T01':0},'accounted_or_reserved_by_subbudget':dict.fromkeys(phase.SUBBUDGET_LIMITS,0)}
class CostTests(unittest.TestCase):
 def test_intervals_actual_rate_priority_remainder_and_exact_total(self):
  result=mixed.apportion(binding(),[run(),run('T02',UUIDS[1],20,520),run('E2Epilot',UUIDS[0],1010,1110)],UUIDS,4,2.5)
  self.assertEqual(sum(Fraction(int(v['numerator']),int(v['denominator'])) for v in result['charged_gpu_hours_by_group_exact'].values()),Fraction(10))
  self.assertEqual(result['residual_group'],'G01');self.assertEqual(result['observed_unit_gpu_seconds_by_group'],{'G01':1000,'T02':500,'E2Epilot':100})
 def test_started_group_or_zero_business_priority(self):
  result=mixed.apportion(binding(),[run('T02')],UUIDS,4,1)
  self.assertEqual(result['residual_group'],'T02');self.assertEqual(result['charged_gpu_hours_by_group']['G01'],0)
  result=mixed.apportion(binding(),[],UUIDS,4,1)
  self.assertTrue(result['zero_business']);self.assertEqual(result['charged_gpu_hours_by_group'],{'G01':4,'T02':0,'E2Epilot':0})
 def test_overlap_unknown_cleanup_wrong_duration_and_excess_time_reject(self):
  cases=[[run(),run('T02',UUIDS[0],500,1200)],[dict(run(),finished_monotonic=None)],[dict(run(),cleanup_errors=['unknown'])],[dict(run(),duration_seconds_monotonic=0)],[run(finish=4000)]]
  for rows in cases:
   with self.subTest(rows=rows),self.assertRaises(ValueError):mixed.apportion(binding(),rows,UUIDS,4,1)
 def test_invalid_rate_group_and_gpu_reject(self):
  for rows,rate in (([run('E2E')],1),([run(gpu='foreign')],1),([],float('nan')),([],True)):
   with self.assertRaises(ValueError):mixed.apportion(binding(),rows,UUIDS,4,rate)
class BudgetTests(unittest.TestCase):
 def test_reserve_each_possible_group_shared_once(self):
  s=state();s['accounted_or_reserved_total']=116
  phase.validate_request('T01',4,1800,'gpu_pack_x',s,'PACK',binding())
  for group in GROUPS:
   changed=copy.deepcopy(s);changed['accounted_or_reserved_by_subbudget'][group]=phase.SUBBUDGET_LIMITS[group]-4+.01
   with self.subTest(group=group),self.assertRaises(ValueError):phase.validate_request('T01',4,1800,'gpu_pack_x',changed,'PACK',binding())
  s['accounted_or_reserved_total']=116.01
  with self.assertRaises(ValueError):phase.validate_request('T01',4,1800,'gpu_pack_x',s,'PACK',binding())
 def test_explicit_route_no_alias_unbound_plan_or_two_gpu(self):
  self.assertEqual(phase.budget_group('T01','gpu_pack_x',['python3','-m','src.tools.gpu_pack'],'PACK'),'PACK')
  for task,name,cmd,group in [('T01','gen_test_x',[],'PACK'),('T01','gpu_pack_x',[],None),('T00b','gpu_pack_x',[],'PACK'),('T01','gpu_pack_x',['src.tools.t02_export'],'PACK')]:
   with self.assertRaises(ValueError):phase.budget_group(task,name,cmd,group)
  for count,auth in ((2,binding()),(3,binding()),(4,None),(4,binding(['E2E']))):
   with self.assertRaises(ValueError):phase.validate_request('T01',count,300,'gpu_pack_x',state(),'PACK',auth)
 def test_final_shared_cost_keeps_full_group_reservations_until_exact_split(self):
  with tempfile.TemporaryDirectory() as tmp:
   repo=Path(tmp);d=repo/'logs/pbs/one';d.mkdir(parents=True)
   p={'budget_batch':phase.BATCH,'task_id':'T01','name':'gpu_pack_x','command':['src.tools.gpu_pack'],'budget_group':'PACK','requested_gpu_count':4,'requested_wall_seconds':1800,'requested_queue_charge_rate':2}
   (d/'payload.json').write_text(json.dumps(p));(d/'submission.json').write_text(json.dumps({'stdout':'123.hopper-m-02','returncode':0}))
   final={'123.hopper-m-02':{'gpu_hours':1,'queue_charge_rate':2,'accounting_result':'results/a.json'}}
   with patch.object(mixed,'validate_payload',return_value=binding()),patch.object(phase,'_final_accounting',return_value=final),patch.object(mixed,'finalized_split',return_value=None):
    value=phase.budget_state(repo,policy=state()['charge_policy'],live_rates={})
   self.assertEqual(value['accounted_or_reserved_total'],2);self.assertEqual({g:value['accounted_or_reserved_by_subbudget'][g] for g in GROUPS},{g:4 for g in GROUPS})
   split={'charged_gpu_hours_by_group':{'G01':1.5,'T02':.5,'E2Epilot':0}}
   with patch.object(mixed,'validate_payload',return_value=binding()),patch.object(phase,'_final_accounting',return_value=final),patch.object(mixed,'finalized_split',return_value=split):
    value=phase.budget_state(repo,policy=state()['charge_policy'],live_rates={})
   self.assertEqual(value['accounted_or_reserved_total'],2);self.assertEqual({g:value['accounted_or_reserved_by_subbudget'][g] for g in GROUPS},split['charged_gpu_hours_by_group'])
class Evidence:
 """Original auditor and planner science are explicit fixture boundaries."""
 def __init__(self,repo):
  self.repo=repo.resolve();self.job='123.hopper-m-02';self.name='gpu_pack_fixture';self.queue=self.repo/'scratch/queue'
  self.write('scripts/audit_provenance.py','def audit_file(path): return {"fail": []}\n',raw=True)
  self.write('scripts/pbs/submit_phase.py',Path(phase.__file__).read_text(),raw=True)
  self.plan={'units':[{'unit_id':'u','name':'gen_test_unit','budget_group':'G01','command':['python3','-m','src.tools.g01_collect_parallel','--name','gen_test_unit']}]}
  self.planref=self.write('configs/plan.json',self.plan,True)
  self.command=['python3','-u','-m','src.tools.gpu_pack','--name',self.name,'--queue-plan','configs/plan.json','--queue-plan-sha256',self.planref['sha256'],'--queue-root',str(self.queue),'--gpus','4','--timeout-seconds','200','--gate-timeout-seconds','180']
  self.module=types.SimpleNamespace(load_plan=lambda *a,**k:(self.plan,self.planref))
  with patch.dict('sys.modules',{'src.tools.gpu_shared_claim':self.module}):auth=mixed.validate_submission(self.repo,self.command,self.name,4)
  self.payload={'name':self.name,'task_id':'T01','budget_group':'PACK','requested_gpu_count':4,'requested_wall_seconds':200,'command':self.command,'mixed_budget_authorization':auth,'dry_run':False}
  self.payloadref=self.write('logs/pbs/reservation/payload.json',self.payload,True)
  qsub=['qsub','job.pbs'];receiptref=self.write('logs/pbs/reservation/submission.json',{'returncode':0,'stderr':'','stdout':self.job,'command':qsub})
  self.child={'config':{'pbs_jobid':self.job},'metrics':{'status':'scientific_failure'}}
  childref=self.logged('gen_test_unit',self.child,self.plan['units'][0]['command'],1)
  self.unit={**run(finish=20),'unit_id':'u','name':'gen_test_unit','pbs_jobid':self.job,'cpu_affinity':list(range(12)),'started_epoch':100,'finished_epoch':110,'result_reference':childref,'exit_code':1}
  claim={k:self.unit[k] for k in ('unit_id','name','budget_group','pbs_jobid','gpu_uuid','cpu_affinity','started_epoch','started_monotonic')}
  claim.update(plan_reference=self.planref,command=self.plan['units'][0]['command'],parent_payload_reference=self.payloadref,
               parent_receipt_reference=receiptref,node='hopper-28',parent_cuda_visible_devices=','.join(UUIDS))
  self.unit['claim_reference']=self.write('scratch/queue/claims/u.json',claim,True)
  self.unit['release_reference']=self.write('scratch/queue/releases/u.json',self.unit,True)
  self.pack={'config':{'pbs_jobid':self.job,'plan_reference':self.planref,'allocated_gpu_uuids':UUIDS,'node':'hopper-28'},'metrics':{'kind':'gpu_shared_pack','all_owned_units_ended':True,'cleanup_errors':[],'unit_runs':[self.unit]}}
  self.packref=self.logged(self.name,self.pack,self.command,1)
  fields={'job_state':'F','run_count':'1','project':'CFP04-SF-125','Job_Owner':'e1583364@node','Resource_List.ngpus':'4','Resource_List.ncpus':'48','Resource_List.queue_charge_rate':'2','resources_used.walltime':'00:01:40','queue':'mediumx','Variable_List':'PBS_O_QUEUE=autox','exec_host':'hopper-28/0*48','Submit_arguments':shlex.join(qsub[1:]),'Output_Path':'host:'+str(self.repo/'logs/pbs/reservation/pbs.out')}
  self.account={'config':{'jobs':[{'job_id':self.job,'qstat_exit':0,'fields':fields}]},'metrics':{'complete_allocation_accounting':True,'allocation_gpu_hours_by_job':{self.job:400/3600}}}
  self.accountref=self.logged('canonical',self.account,['python3','src/tools/pbs_usage.py','--name','canonical',self.job],0)
 def write(self,name,value,immutable=False,raw=False):
  p=self.repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value if raw else json.dumps(value))
  if immutable:p.chmod(0o444)
  return mixed.reference(p,self.repo)
 def logged(self,name,data,cmd,exitcode):
  data['_meta']={'name':name,'end':'ended','host':'hopper-28','exit_code':exitcode,'command':cmd,'log_path':'logs/runs/'+name+'.log','metrics_sha256':hashlib.sha256(json.dumps(data['metrics'],sort_keys=True).encode()).hexdigest()}
  self.write(data['_meta']['log_path'],json.dumps(data['config'],indent=2)+'\nFINAL_METRICS_JSON: '+json.dumps(data['metrics'])+'\n',raw=True)
  return self.write('results/'+name+'.json',data)
 def compute(self):
  with patch.dict('sys.modules',{'src.tools.gpu_shared_claim':self.module}):return mixed.compute_split(self.repo,self.payloadref,self.packref,self.accountref)
class ProvenanceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.f=Evidence(Path(self.tmp.name))
 def test_failed_science_unit_still_has_conserved_allocation_cost(self):
  result=self.f.compute();self.assertEqual(result['unit_count'],1);self.assertEqual(result['queue_charge_rate'],2)
  self.assertEqual(result['charged_gpu_hours_by_group']['G01'],400/3600*2)
  self.assertTrue(all(ref['file'].startswith('results/') and ref.get('metrics_sha256') for ref in result['sources']))
  self.assertTrue(any('claims/' in ref['file'] for ref in result['artifact_references']))
 def test_changed_original_release_and_unknown_cleanup_reject(self):
  ref=self.f.unit['release_reference'];path=self.f.repo/ref['file'];path.chmod(0o644);path.write_text('{}')
  with self.assertRaisesRegex(ValueError,'evidence changed'):self.f.compute()
 def test_unknown_cleanup_rejects_release(self):
  self.f.pack['metrics']['cleanup_errors']=['unknown'];self.f.packref=self.f.logged(self.f.name,self.f.pack,self.f.command,1)
  with self.assertRaisesRegex(ValueError,'incomplete'):self.f.compute()
 def test_owned_claim_cannot_be_omitted_from_parent_unit_runs(self):
  self.f.pack['metrics']['unit_runs']=[];self.f.packref=self.f.logged(self.f.name,self.f.pack,self.f.command,1)
  with self.assertRaisesRegex(ValueError,'omits or invents'):self.f.compute()
 def test_wrong_actual_gpu_count_rejected_by_original_canonical(self):
  self.f.account['config']['jobs'][0]['fields']['Resource_List.ngpus']='1'
  self.f.accountref=self.f.logged('canonical',self.f.account,['python3','src/tools/pbs_usage.py'],0)
  with self.assertRaises((ValueError,RuntimeError)):self.f.compute()
 def test_finalized_split_replays_every_original_reference_and_rejects_claimed_cost_change(self):
  value=self.f.compute();name=self.f.name+'_mixed_accounting'
  cfg={'name':name,'version':mixed.VERSION,'payload_reference':self.f.payloadref,'pack_reference':self.f.packref,'account_reference':self.f.accountref}
  cmd=['python3','-m',mixed.ACCOUNT_PRODUCER,'--name',name,'--payload',self.f.payloadref['file'],'--payload-sha256',self.f.payloadref['sha256'],
       '--pack-result',self.f.packref['file'],'--pack-sha256',self.f.packref['sha256'],
       '--accounting-result',self.f.accountref['file'],'--accounting-sha256',self.f.accountref['sha256']]
  report={'config':cfg,'metrics':value};self.f.logged(name,report,cmd,0)
  row={'name':self.f.name,'payload':self.f.payloadref['file'],'job_id':self.f.job}
  canonical={'gpu_hours':400/3600,'queue_charge_rate':2}
  with patch.dict('sys.modules',{'src.tools.gpu_shared_claim':self.f.module}):
   self.assertEqual(mixed.finalized_split(self.f.repo,row,canonical),value)
   changed=copy.deepcopy(report);changed['metrics']['charged_gpu_hours_by_group']['G01']=0
   self.f.logged(name,changed,cmd,0)
   with self.assertRaisesRegex(ValueError,'no longer matches'):mixed.finalized_split(self.f.repo,row,canonical)
 def test_no_finalized_split_returns_none_and_does_not_invent_zero(self):
  row={'name':self.f.name,'payload':self.f.payloadref['file'],'job_id':self.f.job}
  self.assertIsNone(mixed.finalized_split(self.f.repo,row,{'gpu_hours':400/3600,'queue_charge_rate':2}))
 def test_zero_business_still_charged_to_highest_planned_group(self):
  # Preserve the synthetic claim as evidence outside the stable live namespace;
  # this distinct fixture describes a job that never claimed a unit.
  path=self.f.repo/self.f.unit['claim_reference']['file'];path.rename(path.with_suffix('.unused-fixture'))
  self.f.pack['metrics']['unit_runs']=[];self.f.packref=self.f.logged(self.f.name,self.f.pack,self.f.command,86)
  result=self.f.compute();self.assertTrue(result['zero_business'])
  self.assertEqual(result['charged_gpu_hours_by_group'],{'G01':400/3600*2})
if __name__=='__main__':unittest.main()
