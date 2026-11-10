"""Native DIAG01 route and resource boundaries; no scheduler calls."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('diag01_phase',Path(__file__).resolve().parents[1]/'scripts/pbs/submit_phase.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
NAME='diag_diag01_baselines_s0_01'

def command(timeout='3240',name=NAME):
    return ['python3','-m','src.tools.diag01_baselines','--name',name,
            '--manifest','configs/original.json','--timeout-seconds',timeout]

def state(total=0,rate=1):
    return {'submissions':[],'charge_policy':{'rates':{'smallx':rate,'mediumx':rate}},
            'accounted_or_reserved_by_task':{'T00b':0,'T01':0},'accounted_or_reserved_total':total,
            'accounted_or_reserved_by_subbudget':{k:0 for k in p.SUBBUDGET_LIMITS}}

class Diag01BudgetTests(unittest.TestCase):
    def test_native_only_route_none_and_complete_margin(self):
        self.assertIsNone(p.budget_group('T01',NAME,command()))
        for argv,wall in ((command(),3300),(command('3540'),3600)):
            p.validate_request('T01',1,wall,NAME,state(),command=argv)
        argv=command();argv.insert(1,'-u')
        p.validate_request('T01',1,3300,NAME,state(),command=argv)

    def test_no_alias_producer_task_or_subbudget(self):
        cases=[('T01',NAME,['python3','-m','src.tools.t02_export','--name',NAME],None),
               ('T01','t01_fake',command(name='t01_fake'),None),
               ('T00b',NAME,command(),None),('T01',NAME,[],None)]
        cases += [('T01',NAME,command(),g) for g in [*p.SUBBUDGET_LIMITS,'PACK']]
        for task,name,argv,group in cases:
            with self.subTest(task=task,name=name,group=group),self.assertRaises(ValueError):
                p.budget_group(task,name,argv,group)
        with self.assertRaises(ValueError):p.validate_request('T01',1,3300,NAME,state(),command=None)
        with self.assertRaises(ValueError):p.validate_request('T01',1,3300,'t01_fake',state(),command=command(name='t01_fake'))

    def test_one_gpu_driver_bound_and_explicit_finite_timeout(self):
        for gpus,wall,argv in ((2,3300,command()),(4,3300,command()),(1,3601,command()),
            (1,3299,command()),(1,3600,command('3541')),(1,3300,command('nan')),
            (1,3300,command('inf')),(1,3300,command('0')),(1,3300,command()[:-2]),
            (1,3300,command()+['--timeout-seconds','3240']),
            (1,3300,command()+['--name',NAME]),(1,3300,command(name=NAME+'_other'))):
            with self.subTest(gpus=gpus,wall=wall,argv=argv),self.assertRaises(ValueError):
                p.validate_request('T01',gpus,wall,NAME,state(),command=argv)

    def test_diag01_queue_multiplier_keeps_one_charged_hour_cap(self):
        p.validate_request("T01",1,1800,NAME,state(rate=2),command=command("1740"))
        with self.assertRaisesRegex(ValueError,"one scheduler-charged GPU-hour"):
            p.validate_request("T01",1,1801,NAME,state(rate=2),command=command("1741"))
        p.validate_request("T01",1,1801,"t01_legacy",state(rate=2))

    def test_original_shared_fee_and_legacy_limits_unchanged(self):
        p.validate_request('T01',1,3600,NAME,state(total=119),command=command('3540'))
        for observed in (state(total=119.01),state(total=118.01,rate=2)):
            with self.assertRaises(ValueError):p.validate_request('T01',1,3600,NAME,observed,command=command('3540'))
        p.validate_request('T01',1,14400,'t01_legacy',state())
        with self.assertRaises(ValueError):p.validate_request('T01',1,14401,'t01_legacy',state())
        self.assertEqual(p.TOTAL_LIMIT,120);self.assertEqual(p.LIMITS,{'T00b':60.0,'T01':120.0})
        self.assertEqual(p.SUBBUDGET_LIMITS,{'T01v2':16.0,'E2E':4.0,'E2Epilot':12.0,'E2EpilotT2':.5,'G01':40.0,'T02':40.0})
        self.assertEqual(p.budget_group('T01','gen_test_old',[]),'G01')
        self.assertEqual(p.budget_group('T01','pilot_pairs_v2_old',[]),'T01v2')
        with self.assertRaises(ValueError):p.validate_request('T01',1,3300,NAME,{**state(),'submissions':[{'name':NAME}]},command=command())

if __name__=='__main__':unittest.main()
