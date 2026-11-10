"""Budget boundary fixtures; no scheduler calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('e2e_phase_budget', Path(__file__).resolve().parents[1] / 'scripts/pbs/submit_phase.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


def state(cost=0, total=0, rate=1, g01=0, t02=0, pilot=0, t2=0):
    return {'submissions': [], 'charge_policy': {'rates': {'smallx': rate, 'mediumx': rate}},
            'accounted_or_reserved_by_task': {'T00b': 0, 'T01': 0},
            'accounted_or_reserved_total': total,
            'accounted_or_reserved_by_subbudget': {'T01v2': 0, 'E2E': cost, 'E2Epilot': pilot, 'E2EpilotT2': t2, 'G01': g01, 'T02': t02}}


class E2EBudgetTests(unittest.TestCase):
    def test_four_hour_boundary_and_shared_ceiling(self):
        submit.validate_request('T01', 1, 1800, 'smoke_e2e_export', state(3.5, 119.5))
        for observed in (state(3.501), state(0, 119.501), state(3.1, rate=2)):
            with self.assertRaises(ValueError):
                submit.validate_request('T01', 1, 1800, 'smoke_e2e_export', observed)

    def test_no_multigpu_or_cross_budget_alias(self):
        with self.assertRaises(ValueError):
            submit.validate_request('T01', 2, 1800, 'smoke_e2e_export', state())
        for task, name, command, group in (
            ('T00b', 'smoke_e2e_x', [], None),
            ('T01', 'smoke_e2e_x', [], 'T01v2'),
            ('T01', 'pilot_pairs_v2_x', [], 'E2E'),
            ('T01', 't01_other', ['bash', 'experiments/e2e_export.sh'], None),
            ('T01', 'smoke_e2e_x', ['bash', 'experiments/t01v2_marked_pairs.sh'], None),
        ):
            with self.subTest(name=name, group=group), self.assertRaises(ValueError):
                submit.budget_group(task, name, command, group)
        self.assertEqual(submit.budget_group('T01', 'smoke_e2e_x', []), 'E2E')

    def test_missing_cost_is_not_zero(self):
        observed = state()
        del observed['accounted_or_reserved_by_subbudget']['E2E']
        with self.assertRaises(ValueError):
            submit.validate_request('T01', 1, 300, 'smoke_e2e_x', observed)


class G01BudgetTests(unittest.TestCase):
    def test_forty_charged_hours_and_shared_ceiling(self):
        submit.validate_request('T01', 1, 1800, 'gen_train_probe', state(g01=39.5, total=119.5))
        for observed in (state(g01=39.501), state(g01=39.1, rate=2), state(total=119.501)):
            with self.subTest(state=observed), self.assertRaises(ValueError):
                submit.validate_request('T01', 1, 1800, 'gen_train_probe', observed)

    def test_medium_pack_reserves_every_gpu_at_billing_rate(self):
        submit.validate_request('T01', 4, 1800, 'gen_train_pack', state(g01=36, rate=2))
        with self.assertRaises(ValueError):
            submit.validate_request('T01', 4, 1800, 'gen_train_pack', state(g01=36.001, rate=2))

    def test_cross_budget_aliases_and_disjoint_generation_names(self):
        for task, name, command, group in (
            ('T00b', 'gen_train_x', [], None),
            ('T00b', 'gen_test_x', [], 'G01'),
            ('T01', 'gen_test_x', [], 'T02'),
            ('T01', 'gen_train_x', [], 'E2E'),
            ('T01', 'gen_train_x', [], 'T01v2'),
            ('T01', 'smoke_e2e_x', [], 'G01'),
            ('T01', 'pilot_pairs_v2_x', [], 'G01'),
            ('T01', 'gen_train_x', ['bash', 'experiments/e2e_export.sh'], None),
            ('T01', 'gen_train_x', ['bash', 'experiments/t01v2_marked_pairs.sh'], None),
            ('T01', 'gen_test_x', ['python3', '-m', 'src.tools.t02_export_parallel'], 'G01'),
        ):
            with self.subTest(name=name, group=group), self.assertRaises(ValueError):
                submit.budget_group(task, name, command, group)
        self.assertEqual(submit.budget_group('T01', 'gen_train_x', []), 'G01')
        self.assertEqual(submit.budget_group('T01', 'gen_test_x', []), 'G01')
        submit.validate_request('T01', 1, 1800, 'gen_test_x', state())
        for count in (2, 3, 4):
            with self.assertRaises(ValueError):
                submit.validate_request('T01', count, 1800, 'gen_test_x', state())

    def test_unknown_cost_and_two_gpu_request_rejected(self):
        observed = state()
        del observed['accounted_or_reserved_by_subbudget']['G01']
        with self.assertRaises(ValueError):
            submit.validate_request('T01', 1, 300, 'gen_train_x', observed)
        with self.assertRaises(ValueError):
            submit.validate_request('T01', 2, 300, 'gen_train_x', state())


class E2EPilotBudgetTests(unittest.TestCase):
    def test_twelve_charged_hour_boundary_and_independent_old_e2e_cap(self):
        submit.validate_request('T01', 1, 1800, 'smoke_e2e_pilot_T0_s0',
            state(cost=4, pilot=11.5, total=119.5), 'E2Epilot')
        for observed in (state(pilot=11.501), state(pilot=11.1, rate=2), state(total=119.501)):
            with self.subTest(state=observed), self.assertRaises(ValueError):
                submit.validate_request('T01', 1, 1800, 'smoke_e2e_pilot_T0_s0', observed, 'E2Epilot')
        # Neither independent ledger lends unused budget to the other.
        with self.assertRaises(ValueError):
            submit.validate_request('T01', 1, 300, 'smoke_e2e_old', state(cost=4, pilot=0), 'E2E')
        with self.assertRaises(ValueError):
            submit.validate_request('T01', 1, 300, 'smoke_e2e_pilot_T4_s0', state(cost=0, pilot=12), 'E2Epilot')

    def test_explicit_pilot_namespace_cannot_alias_old_steps_or_other_budgets(self):
        training = ['python3', '-m', 'src.tools.smoke_e2e_train']
        self.assertEqual(submit.budget_group('T01', 'smoke_e2e_pilot_T4_s0', training, 'E2Epilot'), 'E2Epilot')
        for task, name, command, group in (
            ('T01', 'smoke_e2e_pilot_T0_s0', training, None),
            ('T01', 'smoke_e2e_pilot_T0_s0', training, 'E2E'),
            ('T00b', 'smoke_e2e_pilot_T0_s0', training, 'E2Epilot'),
            ('T01', 'smoke_e2e_old_T0', training, 'E2Epilot'),
            ('T01', 'smoke_e2e_look_views', [], 'E2Epilot'),
            ('T01', 'pilot_pairs_v2_old', [], 'E2Epilot'),
            ('T01', 'gen_train_old', [], 'E2Epilot'),
            ('T01', 't02_export_old', [], 'E2Epilot'),
            ('T01', 'smoke_e2e_pilot_T0_s0', ['python3', '-m', 'src.tools.t02_cache_check'], 'E2Epilot'),
            ('T01', 'smoke_e2e_pilot_T0_s0', ['python3', '-m', 'src.tools.smoke_e2e_views'], 'E2Epilot'),
            ('T01', 'smoke_e2e_pilot_T0_s0', ['bash', 'experiments/t01v2_marked_pairs.sh'], 'E2Epilot'),
            ('T01', 'smoke_e2e_pilot_T0_s0', training, 'G01'),
            ('T01', 'smoke_e2e_pilot_T0_s0', training, 'T02'),
        ):
            with self.subTest(task=task, name=name, group=group), self.assertRaises(ValueError):
                submit.budget_group(task, name, command, group)
        for group in (None, 'E2E'):
            self.assertEqual(submit.budget_group('T01', 'smoke_e2e_old_T0', training, group), 'E2E')

    def test_missing_or_nonfinite_pilot_cost_and_multiple_gpus_are_rejected(self):
        for count in (2, 3, 4):
            with self.assertRaises(ValueError):
                submit.validate_request('T01', count, 300, 'smoke_e2e_pilot_T0_s0', state(), 'E2Epilot')
        for cost in (None, True, float('nan'), float('inf'), -1):
            with self.assertRaises(ValueError):
                submit.validate_request('T01', 1, 300, 'smoke_e2e_pilot_T0_s0', state(pilot=cost), 'E2Epilot')
        observed = state()
        del observed['accounted_or_reserved_by_subbudget']['E2Epilot']
        with self.assertRaises(ValueError):
            submit.validate_request('T01', 1, 300, 'smoke_e2e_pilot_T0_s0', observed, 'E2Epilot')

    def test_actual_payload_ledger_keeps_old_final_cost_and_new_full_reservation_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            payloads = [
                {'name': 'smoke_e2e_old_T0', 'budget_group': None, 'budget_revision': 'historical_e2e_4'},
                {'name': 'smoke_e2e_pilot_T4_s0', 'budget_group': 'E2Epilot', 'budget_revision': submit.BUDGET_REVISION},
            ]
            for number, selected in enumerate(payloads, 1):
                directory = repo / 'logs/pbs' / str(number)
                directory.mkdir(parents=True)
                payload = {**selected, 'task_id': 'T01', 'budget_batch': submit.BATCH,
                    'command': ['python3', '-m', 'src.tools.smoke_e2e_train'],
                    'requested_gpu_count': 1, 'requested_wall_seconds': 1800, 'requested_queue_charge_rate': 1}
                (directory / 'payload.json').write_text(json.dumps(payload))
                (directory / 'submission.json').write_text(json.dumps({'stdout': f'{number}.hopper-m-02', 'returncode': 0}))
            final = {'1.hopper-m-02': {'gpu_hours': .25, 'queue_charge_rate': 2,
                                      'accounting_result': 'results/original_accounting.json'}}
            policy = {'rates': {'smallx': 1, 'mediumx': 1}}
            with patch.object(submit, '_final_accounting', return_value=final):
                actual = submit.budget_state(repo, policy=policy, live_rates={'2.hopper-m-02': 2})
            self.assertEqual(actual['accounted_or_reserved_by_subbudget']['E2E'], .5)
            self.assertEqual(actual['accounted_or_reserved_by_subbudget']['E2Epilot'], 1)
            self.assertEqual(actual['accounted_or_reserved_total'], 1.5)
            self.assertEqual([r['basis'] for r in actual['submissions']], ['final_accounting', 'full_request_reservation'])
            # An old generic smoke payload cannot be retroactively routed to pilot.
            old_path = repo / 'logs/pbs/1/payload.json'
            changed = json.loads(old_path.read_text()); changed['budget_group'] = 'E2Epilot'
            old_path.write_text(json.dumps(changed))
            with patch.object(submit, '_final_accounting', return_value=final), self.assertRaises(ValueError):
                submit.budget_state(repo, policy=policy, live_rates={'2.hopper-m-02': 2})


class PilotT2BudgetTests(unittest.TestCase):
    @staticmethod
    def command(name='smoke_e2e_pilot_T2_s0', arm='T2', group='E2EpilotT2'):
        return ['python3', '-u', '-m', 'src.tools.smoke_e2e_pilot', 'run', '--name', name,
                '--arm', arm, '--budget-group', group]

    def test_half_hour_is_separate_from_twelve_and_preserves_shared_ceiling(self):
        name = 'smoke_e2e_pilot_T2_s0'
        submit.validate_request('T01', 1, 900, name, state(pilot=12, t2=.25, total=119.75), 'E2EpilotT2')
        for observed in (state(t2=.251), state(t2=.1, rate=2), state(total=119.751)):
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                submit.validate_request('T01', 1, 900, name, observed, 'E2EpilotT2')
        for bad in (None, True, -1, float('nan'), float('inf')):
            with self.subTest(cost=bad), self.assertRaises(ValueError):
                submit.validate_request('T01', 1, 300, name, state(t2=bad), 'E2EpilotT2')
        missing = state(); del missing['accounted_or_reserved_by_subbudget']['E2EpilotT2']
        with self.assertRaises(ValueError):
            submit.validate_request('T01', 1, 300, name, missing, 'E2EpilotT2')
        for count in (2, 3, 4):
            with self.assertRaises(ValueError):
                submit.validate_request('T01', count, 60, name, state(), 'E2EpilotT2')

    def test_native_t2_and_explicit_name_arm_budget_are_all_required(self):
        name = 'smoke_e2e_pilot_T2_s0'; command = self.command()
        self.assertEqual(submit.budget_group('T01', name, command, 'E2EpilotT2'), 'E2EpilotT2')
        bad_commands = [self.command(arm='T4'), self.command(group='E2Epilot'),
            command[:-2], command + ['--arm', 'T2'], command + ['--engineering-smoke', 'other'],
            ['bash', '-c', 'echo', *command], command[:3]+['src.tools.smoke_e2e_train']+command[4:]]
        for bad in bad_commands:
            with self.subTest(command=bad), self.assertRaises(ValueError):
                submit.budget_group('T01', name, bad, 'E2EpilotT2')
        for task, selected, group in [('T00b', name, 'E2EpilotT2'), ('T01', name, 'E2Epilot'),
                ('T01', name, None), ('T01', 'smoke_e2e_pilot_T0_s0', 'E2EpilotT2'),
                ('T01', 'gen_test_wrong', 'E2EpilotT2')]:
            with self.subTest(name=selected,group=group), self.assertRaises(ValueError):
                submit.budget_group(task, selected, command, group)

    def test_four_hour_twenty_extension_is_native_t0_t4_only(self):
        for arm in ('T0', 'T4'):
            name = 'smoke_e2e_pilot_' + arm + '_s0'
            command = self.command(name, arm, 'E2Epilot')
            submit.validate_request('T01', 1, 15600, name, state(), 'E2Epilot', command=command)
            with self.assertRaises(ValueError):
                submit.validate_request('T01', 1, 15601, name, state(), 'E2Epilot', command=command)
            for wrong in (None, command + ['--engineering-smoke', 'other'], self.command(name, 'T2')):
                with self.assertRaises(ValueError):
                    submit.validate_request('T01', 1, 15600, name, state(), 'E2Epilot', command=wrong)
        for name, group in [('gen_train_fixture','G01'),('t02_export_fixture','T02'),
                            ('smoke_e2e_original','E2E'),('smoke_e2e_pilot_T2_s0','E2EpilotT2')]:
            with self.assertRaises(ValueError):
                submit.validate_request('T01', 1, 14401, name, state(), group, command=self.command())

    def test_actual_old_and_new_payload_ledgers_do_not_move_historical_costs(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            for number, arm, group in [(1,'T0','E2Epilot'), (2,'T4','E2Epilot'), (3,'T2','E2EpilotT2')]:
                directory = repo / 'logs/pbs' / str(number); directory.mkdir(parents=True)
                name = 'smoke_e2e_pilot_' + arm + '_s0'
                command = self.command(name, arm, group)
                if group == 'E2Epilot': command = command[:-2]  # Original pre-change payload.
                payload = {'name':name,'task_id':'T01','budget_group':group,'budget_batch':submit.BATCH,
                    'command':command,'requested_gpu_count':1,'requested_wall_seconds':900,
                    'requested_queue_charge_rate':1}
                (directory / 'payload.json').write_text(json.dumps(payload))
                (directory / 'submission.json').write_text(json.dumps({'stdout':f'{number}.hopper-m-02','returncode':0}))
            final = {'1.hopper-m-02':{'gpu_hours':.25,'queue_charge_rate':2,'accounting_result':'results/old.json'}}
            with patch.object(submit, '_final_accounting', return_value=final):
                actual = submit.budget_state(repo, policy={'rates':{'smallx':1,'mediumx':1}},
                    live_rates={'2.hopper-m-02':2,'3.hopper-m-02':1})
            self.assertEqual(actual['accounted_or_reserved_by_subbudget']['E2Epilot'],1.)
            self.assertEqual(actual['accounted_or_reserved_by_subbudget']['E2EpilotT2'],.25)
            self.assertEqual(actual['accounted_or_reserved_total'],1.25)


class T02BudgetTests(unittest.TestCase):
    def test_export_subbudget_full_reservation_and_shared_120_ceiling(self):
        submit.validate_request('T01', 1, 1800, 't02_export_fixture', state(t02=39.5, total=119.5), 'T02')
        for observed in (state(t02=39.501), state(t02=39.1, rate=2), state(total=119.501)):
            with self.assertRaises(ValueError):
                submit.validate_request('T01', 1, 1800, 't02_export_fixture', observed, 'T02')
        self.assertEqual(submit.LIMITS, {'T00b': 60.0, 'T01': 120.0})
        self.assertEqual(submit.SUBBUDGET_LIMITS, {'T01v2': 16.0, 'E2E': 4.0, 'E2Epilot': 12.0, 'E2EpilotT2': .5, 'G01': 40.0, 'T02': 40.0})

    def test_fresh_cache_validation_uses_t02_budget_and_exact_cache_namespace(self):
        command = ['python3', '-m', 'src.tools.t02_cache_check']
        self.assertEqual(submit.budget_group('T01', 't02_export_cache_fixture', command, 'T02'), 'T02')
        submit.validate_request('T01', 1, 1800, 't02_export_cache_fixture', state(t02=39.5), 'T02')
        for name, group in (('t02_export_fixture', 'T02'), ('gen_test_cache', 'G01'),
                            ('t02_export_cache_fixture', None), ('pilot_pairs_v2_cache', 'T02')):
            with self.assertRaises(ValueError): submit.budget_group('T01', name, command, group)
        with self.assertRaises(ValueError):
            submit.validate_request('T01', 2, 1800, 't02_export_cache_fixture', state(), 'T02')
        with self.assertRaises(ValueError):
            submit.validate_request('T01', 1, 1800, 't02_export_cache_fixture', state(t02=39.501), 'T02')

    def test_explicit_route_cannot_reclassify_old_diagnostics(self):
        for module in ('src.tools.t02_export', 'src.tools.t02_export_parallel'):
            self.assertEqual(submit.budget_group('T01', 't02_export_fixture', ['python3', '-m', module], 'T02'), 'T02')
            self.assertEqual(submit.budget_group('T01', 'pilot_pairs_v2_old', ['python3', '-m', module], 'T01v2'), 'T01v2')
        for task, name, group in (('T01', 't02_export_fixture', None), ('T00b', 't02_export_fixture', 'T02'),
                ('T01', 'gen_test_fixture', 'T02'), ('T01', 'pilot_pairs_v2_old', 'T02'),
                ('T01', 'smoke_e2e_fixture', 'T02'), ('T01', 't02_export_fixture', 'G01')):
            with self.assertRaises(ValueError): submit.budget_group(task, name, [], group)
        with self.assertRaises(ValueError):
            submit.budget_group('T01', 't02_export_fixture', ['python3', '-m', 'src.tools.g01_collect_parallel'], 'T02')
        with self.assertRaises(ValueError):
            submit.validate_request('T01', 2, 1800, 't02_export_fixture', state(), 'T02')
        observed = state(); del observed['accounted_or_reserved_by_subbudget']['T02']
        with self.assertRaises(ValueError):
            submit.validate_request('T01', 1, 1800, 't02_export_fixture', observed, 'T02')


if __name__ == '__main__':
    unittest.main()
