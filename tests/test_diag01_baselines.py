"""Handmade P-table D3 contracts; no model/GPU/rendering and no science result."""
import copy
from dataclasses import replace
import unittest

from src.envs.pair_table import TableEnvironment, TableObservationBackend
from src.tools import diag01_baselines as d3
from test_pair_table import table


class Policy:
    def __init__(self, response='select 1'):
        self.response, self.calls = response, []

    def __call__(self, contexts):
        self.calls.append(copy.deepcopy(contexts))
        return [{'action': self.response, 'response': self.response, 'forced_observation': False,
            'sample_id': f'call{len(self.calls)}_{i}', 'response_token_ids': [1, 2, 3],
            'response_token_count': 3} for i, _ in enumerate(contexts)]


def environments(value=None, *, swapped=False):
    value = value or table()
    return [(value.pair_id, w, TableEnvironment(TableObservationBackend({value.pair_id: value},
        allow_synthetic=True, world_orders={value.pair_id: [1, 0] if swapped else [0, 1]})))
        for w in ('w0', 'w1')]


class DiagnosticTests(unittest.TestCase):
    def test_trigger_uses_fixed_roundtrip_not_privileged_direction(self):
        policy = Policy(); rows = d3.collect('obs_trigger', environments(), policy, lambda row: None)
        by_world = {r['world_id']: r for r in rows}
        self.assertEqual(by_world['w0']['actions'][:-1], list(d3.EXPLORATION))
        self.assertEqual(by_world['w1']['actions'][:-1], ['turn_left', 'turn_right'])
        self.assertEqual(by_world['w0']['independent_first_seen_step'], 3)
        self.assertEqual(by_world['w1']['independent_first_seen_step'], 1)
        for row in rows:
            self.assertTrue(row['seen_before_select'])
            self.assertEqual(row['states'][0]['state_id'], row['states'][-1]['state_id'])
        for contexts in policy.calls:
            for context in contexts:
                self.assertEqual(set(context), {'observations', 'responses', 'select_only'})
                self.assertTrue(context['select_only'])

    def test_actual_mask_recomputes_even_if_stored_seen_metadata_is_zero(self):
        value = table()
        value.states = {key: replace(state, anchor_pixels=(0, 0)) for key, state in value.states.items()}
        rows = d3.collect('obs_trigger', environments(value), Policy(), lambda row: None)
        self.assertTrue(all(r['logged_x_seen_step'] is None for r in rows))
        self.assertTrue(all(r['seen_before_select'] for r in rows))
        self.assertTrue(all(max(s['anchor_pixels'] for s in r['states']) == 250 for r in rows))

    def test_original_world_permutation_is_preserved(self):
        rows = d3.collect('obs_trigger', environments(swapped=True), Policy(), lambda row: None)
        by_world = {r['world_id']: r for r in rows}
        self.assertEqual(by_world['w0']['original_world_id'], 'w1')
        self.assertEqual(by_world['w0']['observation_count'], 2)
        self.assertEqual(by_world['w1']['observation_count'], 4)

    def test_never_and_always_have_declared_observation_costs(self):
        never = d3.collect('obs_never', environments(), Policy(), lambda row: None)
        always = d3.collect('obs_always', environments(), Policy(), lambda row: None)
        self.assertEqual([r['observation_count'] for r in never], [0, 0])
        self.assertFalse(any(r['seen_before_select'] for r in never))
        self.assertEqual([r['observation_count'] for r in always], [4, 4])
        self.assertTrue(all(r['seen_before_select'] for r in always))
        self.assertTrue(all(abs(r['observation_reward_cost'] - .2) < 1e-12 for r in always))

    def test_invalid_diag_output_is_preserved_without_repair(self):
        policy = Policy('I think select 1')
        rows = d3.collect('diag_base', environments(), policy, lambda row: None)
        self.assertTrue(all(r['steps'] == 5 and r['syntax_failures'] == 5 for r in rows))
        self.assertFalse(any(r['success'] or r['selected'] for r in rows))
        self.assertTrue(all(not c['select_only'] for batch in policy.calls for c in batch))

    def test_final_observation_does_not_create_fifth_view(self):
        rows = d3.collect('diag_base', environments(), Policy('turn_left'), lambda row: None)
        self.assertTrue(all(r['states'][-1]['state_id'] == r['states'][-2]['state_id'] for r in rows))
        self.assertTrue(all(r['observation_attempts'] == 5 and r['observation_count'] == 4 for r in rows))
        self.assertFalse(any(r['seen_before_select'] for r in rows))

    def test_summary_uses_complete_world_denominators(self):
        rows = d3.collect('obs_trigger', environments(), Policy(), lambda row: None)
        result = d3.summarize(rows)
        self.assertEqual(result['look_flip']['episodes'], 2)
        self.assertEqual(result['look_all']['success_rate'], .5)
        self.assertEqual(result['look_all']['seen_before_select_rate'], 1.)
        self.assertEqual(result['look_all']['observation_count_mean'], 3.)
        self.assertIsNone(result['look_stay']['success_rate'])

    def test_rejects_missing_outputs_and_nonselect_for_forced_policy(self):
        with self.assertRaisesRegex(ValueError, 'omitted/added'):
            d3.collect('obs_never', environments(), lambda contexts: [], lambda row: None)
        with self.assertRaisesRegex(ValueError, 'Forced-select'):
            d3.collect('obs_never', environments(), Policy('turn_left'), lambda row: None)


if __name__ == '__main__':
    unittest.main()
