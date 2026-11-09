"""Synthetic funnel records; no simulator or measured result claims."""
import copy
import unittest

from src.tools import t01v2_look_funnel as funnel


def action(raw=250, marked=250):
    return {'action': 'turn_left', 'evidence': {'evidence_pixels': raw, 'marked_evidence_pixels': marked,
        'evidence_passed': raw >= 250, 'marked_evidence_passed': marked >= 250, 'resolution': [255, 255]},
        'worlds': [{'world_index': index, 'status': 'observed', 'anchor_pixels': 400, 'anchor_visible': True}
                   for index in (0, 1)]}


def attempt(index=0):
    return {'pair_id': f'fixture{index}', 'scene': 'FloorPlan203', 'mode': 'flip', 'proposed_pool': 'look',
        'status': 'rejected', 'kind': 'look_flip', 'discard_reasons': ['fixture'],
        'prefix_measurement': {'prefix_edit_mask_pixels': [{'w0': 0, 'w1': 0}], 'prefix_max_abs_diff': 3,
                               'candidates_visible_and_marked': True},
        'look_solvability': {'complete': True, 'solvable_w0': True, 'solvable_w1': True, 'actions': [action()]},
        'physical_key_numbers': {world: {'valid_physical': True} for world in ('physical_w0', 'physical_w1')},
        'witnesses': [{'passed': True}, {'passed': True}]}


class LookFunnelTests(unittest.TestCase):
    def test_all_six_gates_require_actual_measurements(self):
        row = attempt(); row.update(status='passed')
        measured, _ = funnel.measurements(row)
        self.assertEqual(measured, dict.fromkeys(funnel.STAGES, True))
        result = funnel.summarize([row])
        self.assertEqual(result['source_reported_look_passed'], 1)
        for value in result['stages'].values():
            self.assertEqual(value['cumulative_pass'], 1)
            self.assertEqual(value['gate_unknown'], 0)

    def test_physics_first_failure_does_not_invent_image_measurements(self):
        row = {'pair_id': 'early', 'scene': 'FloorPlan203', 'status': 'rejected', 'discard_reasons': ['w0_physical'],
               'physical_key_numbers': {'physical_w0': {'valid_physical': False}}}
        measured, values = funnel.measurements(row)
        self.assertFalse(measured['physical_both'])
        self.assertTrue(all(measured[key] is None for key in funnel.STAGES if key != 'physical_both'))
        result = funnel.summarize([row])['stages']
        self.assertEqual(result['hidden_both']['gate_unknown'], 1)
        self.assertEqual(result['hidden_both']['cumulative_unknown'], 1)
        self.assertEqual(result['physical_both']['gate_fail'], 1)
        self.assertIsNone(result['physical_both']['rejected_examples'][0]['earlier_requested_gates_passed'])
        self.assertIsNone(values['prefix_max_abs_diff'])

    def test_exact_existing_pixel_boundaries_and_same_action_rule(self):
        row = attempt(); row['look_solvability']['actions'] = [action(250, 249), action(249, 250)]
        self.assertFalse(funnel.measurements(row)[0]['first_evidence_ge250'])
        row['look_solvability']['actions'].append(action(250, 250))
        self.assertTrue(funnel.measurements(row)[0]['first_evidence_ge250'])
        row['prefix_measurement']['prefix_max_abs_diff'] = 4
        self.assertFalse(funnel.measurements(row)[0]['prefix_le3'])
        row['prefix_measurement']['prefix_edit_mask_pixels'][0]['w1'] = 1
        self.assertFalse(funnel.measurements(row)[0]['hidden_both'])

    def test_incomplete_catalog_can_prove_existence_but_not_absence(self):
        row = attempt(); row['look_solvability'].update(complete=False, solvable_w0=None, solvable_w1=None)
        row['look_solvability']['actions'] = [action(249, 249)]
        self.assertIsNone(funnel.measurements(row)[0]['first_evidence_ge250'])
        self.assertIsNone(funnel.measurements(row)[0]['solvable_both'])
        row['look_solvability']['actions'].append(action())
        self.assertTrue(funnel.measurements(row)[0]['first_evidence_ge250'])

    def test_missing_evidence_never_becomes_zero_or_unavailable(self):
        row = attempt(); row['look_solvability']['actions'][0]['evidence'] = None
        self.assertIsNone(funnel.measurements(row)[0]['first_evidence_ge250'])
        row['look_solvability']['actions'][0]['worlds'][1]['status'] = 'not_in_domain'
        self.assertFalse(funnel.measurements(row)[0]['first_evidence_ge250'])

    def test_evidence_requires_both_observed_worlds_and_consistent_flags(self):
        row = attempt(); row['look_solvability']['actions'][0]['worlds'][1]['status'] = 'failed'
        self.assertIsNone(funnel.measurements(row)[0]['first_evidence_ge250'])
        row = attempt(); row['look_solvability']['actions'][0]['evidence']['evidence_passed'] = False
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            funnel.measurements(row)

    def test_gate_and_cumulative_counts_are_distinct_and_examples_unpadded(self):
        rows = [attempt(index) for index in range(4)]
        for row in rows:
            row['prefix_measurement']['prefix_max_abs_diff'] = 4
        result = funnel.summarize(rows)
        self.assertEqual(result['stages']['prefix_le3']['gate_fail'], 4)
        self.assertEqual(len(result['stages']['prefix_le3']['rejected_examples']), 3)
        self.assertEqual(result['stages']['physical_both']['gate_pass'], 4)
        self.assertEqual(result['stages']['physical_both']['cumulative_pass'], 0)
        self.assertEqual(result['stages']['physical_both']['cumulative_blocked'], 4)
        self.assertEqual(result['stages']['physical_both']['rejected_examples'], [])

    def test_empty_observed_journal_and_unobserved_units_are_different(self):
        unit = {'scene': 'FloorPlan203', 'field_of_view': 100, 'source': {'file': 'fixture'}, 'records': []}
        metrics = funnel.aggregate([unit], 'FloorPlan7')
        self.assertEqual(metrics['groups'][0]['funnel']['observed_attempts'], 0)
        self.assertIsNone(metrics['groups'][2]['funnel'])
        self.assertEqual(len(metrics['unprovided_units']), 5)
        text = funnel.markdown(metrics)
        self.assertIn('尚未提供', text)
        self.assertNotIn('| 60 |', text)
        self.assertNotIn('#', text)
        absent = copy.deepcopy(unit); absent['records'] = None
        self.assertIsNone(funnel.aggregate([absent], 'FloorPlan7')['groups'][0]['funnel'])

    def test_flip_stay_fov_and_unknown_mode_are_not_pooled(self):
        first, second, unknown = attempt(), attempt(1), attempt(2)
        second['mode'] = 'stay'; unknown.pop('mode')
        units = [{'scene': 'FloorPlan203', 'field_of_view': 100, 'source': {'file': 'a'}, 'records': [first, unknown]},
                 {'scene': 'FloorPlan203', 'field_of_view': 60, 'source': {'file': 'b'}, 'records': [second]}]
        groups = funnel.aggregate(units, 'FloorPlan7')['groups']
        self.assertEqual([group['funnel']['observed_attempts'] for group in groups], [1, 0, 0, 1])
        self.assertEqual(groups[0]['known_records_with_unknown_mode'], 1)

    def test_markdown_has_one_table_and_compact_same_action_examples(self):
        row = attempt(); row['look_solvability']['actions'] = [action(219, 218)]
        unit = {'scene': 'FloorPlan203', 'field_of_view': 100, 'source': {'file': 'fixture'}, 'records': [row]}
        text = funnel.markdown(funnel.aggregate([unit], 'FloorPlan7'))
        self.assertEqual(text.count('| FOV |'), 1)
        self.assertIn('turn_left:219/218', text)
        self.assertNotIn('world_index', text)
        self.assertNotIn('##', text)


if __name__ == '__main__':
    unittest.main()
