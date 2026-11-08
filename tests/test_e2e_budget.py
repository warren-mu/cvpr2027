"""Budget boundary fixtures; no scheduler calls."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('e2e_phase_budget', Path(__file__).resolve().parents[1] / 'scripts/pbs/submit_phase.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


def state(cost=0, total=0, rate=1):
    return {'submissions': [], 'charge_policy': {'rates': {'smallx': rate}},
            'accounted_or_reserved_by_task': {'T00b': 0, 'T01': 0},
            'accounted_or_reserved_total': total,
            'accounted_or_reserved_by_subbudget': {'T01v2': 0, 'E2E': cost}}


class E2EBudgetTests(unittest.TestCase):
    def test_four_hour_boundary_and_shared_ceiling(self):
        submit.validate_request('T01', 1, 1800, 'smoke_e2e_export', state(3.5, 59.5))
        for observed in (state(3.501), state(0, 59.501), state(3.1, rate=2)):
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


if __name__ == '__main__':
    unittest.main()
