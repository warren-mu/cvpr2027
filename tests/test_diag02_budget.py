"""DIAG02 native routes and measured-plus-reserved bounds; no PBS calls."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('diag02_phase', Path(__file__).resolve().parents[1]/'scripts/pbs/submit_phase.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)

def command(size='3b', timeout='1440'):
    name = 'diag_diag02_baselines_' + size + '_s0_01'
    return ['python3', '-u', '-m', 'src.tools.diag02_baselines', '--name', name,
            '--model-size', size, '--model-path', '/scratch/fixture', '--model-source', '{}',
            '--manifest', '{}', '--holdout-reference', '{}', '--timeout-seconds', timeout]

def state(rate=1):
    return {'submissions': [], 'charge_policy': {'rates': {'smallx': rate, 'mediumx': rate}},
            'accounted_or_reserved_by_task': {'T00b': 0, 'T01': 0}, 'accounted_or_reserved_total': 0,
            'accounted_or_reserved_by_subbudget': {g: 0 for g in p.SUBBUDGET_LIMITS}}

def validate(argv, seconds, group, observed=None, gpus=1):
    name = argv[argv.index('--name')+1]
    p.validate_request('T01', gpus, seconds, name, observed or state(), group, command=argv)

class Diag02BudgetTests(unittest.TestCase):
    def test_model_groups_and_cleanup_margin(self):
        for size, bound, wall, group in [('3b','1440',1500,'DIAG02E3'),('7b','2340',2400,'DIAG02E4'),('32b','3540',3600,'DIAG02E4')]:
            c=command(size,bound);name=c[c.index('--name')+1]
            self.assertEqual(p.budget_group('T01',name,c,group),group)
            validate(c,wall,group)
            with self.assertRaises(ValueError):validate(c,int(bound)+59,group)

    def test_alias_wrong_group_duplicate_and_malformed_rejected(self):
        c=command();name=c[c.index('--name')+1]
        for group in (None,'T02','DIAG02E4'):
            with self.assertRaises(ValueError):p.budget_group('T01',name,c,group)
        for mutation in (c+['--model-size','3b'],c+['--name',name],command('3b','nan'),command('3b','1741')):
            with self.assertRaises(ValueError):validate(mutation,1800,'DIAG02E3')
        c[c.index('src.tools.diag02_baselines')]='src.tools.diag01_baselines'
        with self.assertRaises(ValueError):validate(c,1500,'DIAG02E3')

    def test_each_bound_includes_prior_allocations_and_queue_multiplier(self):
        s=state();s['accounted_or_reserved_by_subbudget']['DIAG02E3']=.1
        with self.assertRaises(ValueError):validate(command(),1500,'DIAG02E3',s)
        validate(command('3b','840'),900,'DIAG02E3',state(rate=2))
        with self.assertRaises(ValueError):validate(command('3b','841'),901,'DIAG02E3',state(rate=2))
        s=state();s['accounted_or_reserved_by_subbudget']['DIAG02E4']=1.01
        with self.assertRaises(ValueError):validate(command('32b','3540'),3600,'DIAG02E4',s)

    def test_single_gpu_only_and_model_namespace_bound(self):
        for gpus in (2,3,4):
            with self.assertRaises(ValueError):validate(command(),1500,'DIAG02E3',gpus=gpus)
        c=command();c[c.index('--model-size')+1]='7b'
        with self.assertRaises(ValueError):validate(c,1500,'DIAG02E4')

    def test_combined_diag_only_bound(self):
        s=state();s['accounted_or_reserved_by_subbudget'].update(DIAG02E3=.4,DIAG02E4=2.5)
        with self.assertRaisesRegex(ValueError,'combined three'):validate(command(),1500,'DIAG02E3',s)

if __name__=='__main__': unittest.main()
