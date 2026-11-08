"""Synthetic retained/omitted evidence cases, never actual Look measurements."""
from contextlib import redirect_stdout
from collections import Counter
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools import t01v2_look_diagnostic as diagnostic


def prefix(counts=(0, 0), maximum=8):
    return {'prefix_edit_mask_pixels': [{'w0': counts[0], 'w1': counts[1]}],
        'prefix_max_abs_diff': maximum, 'render_qualified_for_shared_prefix': False,
        'candidates_visible_and_marked': True, 'construction_context_verified': True,
        'reasons': ['prefix_rgb_difference_above_3', 'shared_prefix_not_verified'],
        'frames': [{'label_bindings_exact': True, 'public_context_exact': True,
                    'evidence_pixels': 0, 'marked_evidence_pixels': 0, 'resolution': [255, 255]}]}


def fixture(root, raw_rows):
    raw_root, journal_root = root / 'scratch', root / 'data/world_pairs/v2'
    raw_root.mkdir(); journal_root.mkdir(parents=True)
    compact = []
    for index, supplied in enumerate(raw_rows):
        row = {'pair_id': f'floorplan1_{index:05d}', 'scene': 'FloorPlan1', 'kind': 'direct_flip',
               'status': 'rejected', 'discard_reasons': ['neither_hidden_shared_prefix_nor_direct_evidence'],
               'generation_seconds': 1., **copy.deepcopy(supplied)}
        path = raw_root / f'attempt_{index}.json'
        path.write_text(json.dumps(row))
        ref = {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        compact.append({**{key: row[key] for key in diagnostic.MATCH_FIELDS if key in row}, 'raw_attempt': ref})
    journal = journal_root / 'fixture.jsonl'
    journal.write_text(''.join(json.dumps(row) + '\n' for row in compact))
    measured = {'kind': 'pair_v2_shard', 'status': 'incomplete', 'cleanup_errors': [],
        'attempt_journal': str(journal.relative_to(root)), 'attempt_journal_sha256': hashlib.sha256(journal.read_bytes()).hexdigest(),
        'attempts': len(compact), 'passed_by_class': {kind: 0 for kind in diagnostic.merge.KINDS},
        'discard_reason_counts': dict(Counter(reason for row in compact for reason in row['discard_reasons']))}
    data = {'_meta': {'exit_code': 1, 'end': 'synthetic-ended'}, 'metrics': measured,
        'config': {'scene': 'FloorPlan1', 'seed': 0, 'resolution': 255,
            'attempt_journal': measured['attempt_journal'], 'raw_evidence_dir': str(raw_root),
            'shard': {'pbs_jobid': 'synthetic-job'}}}
    ref = {'file': 'results/fixture.json', 'sha256': 'a' * 64, 'metrics_sha256': 'b' * 64}
    return data, ref, compact, journal


class LookDiagnosticTests(unittest.TestCase):
    def test_hidden_rgb_failure_is_distinct_from_anchor_visibility_without_causal_claim(self):
        hidden = diagnostic.prefix_details({'prefix_measurement': prefix()})
        visible = diagnostic.prefix_details({'prefix_measurement': prefix((1, 0))})
        self.assertEqual(hidden['classification'], 'x_hidden_both_rgb_above_3')
        self.assertEqual(visible['classification'], 'x_visible_w0_only')
        self.assertIsNone(hidden['shadow_or_other_render_cause'])
        within = diagnostic.prefix_details({'prefix_measurement': prefix(maximum=3)})
        self.assertEqual(within['classification'], 'x_hidden_both_rgb_within_3')
        unknown = diagnostic.prefix_details({'prefix_measurement': {'prefix_max_abs_diff': 8}})
        self.assertIsNone(unknown['classification'])
        self.assertIsNone(unknown['anchor_pixels_w0_w1'])

    def test_before_search_and_unknown_report_are_not_unsolvability(self):
        self.assertEqual(diagnostic.stage({'discard_reasons': ['target_uniqueness_or_ratio']}), 'before_look_target')
        self.assertEqual(diagnostic.stage({'kind': 'look_flip'}), 'unknown')
        row = {'kind': 'look_flip', 'look_solvability': {'complete': False, 'actions': []}}
        self.assertEqual(diagnostic.stage(row), 'look_search_recorded')
        self.assertIsNone(diagnostic.look_details(row)['solvable_w0'])

    def test_retained_subset_denominators_and_actual_target_subreasons(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(diagnostic.merge, 'ROOT', Path(directory)):
            root = Path(directory)
            data, ref, _, _ = fixture(root, [
                {'prefix_measurement': prefix()}, {'prefix_measurement': prefix((12, 0))},
                {'evidence_policy': 'compact rejection'},
                {'kind': None, 'discard_reasons': ['target_uniqueness_or_ratio'],
                 'target_measurements': [{'reasons': ['closest_ratio_below_1_5']}, {'reasons': ['closest_ratio_below_1_5']}]}])
            result = diagnostic.describe_source(data, ref)
            self.assertTrue(result['counter_consistency'])
            self.assertEqual(result['journal_attempts'], 4)
            self.assertEqual(result['prefix_measurement_attempts'], 2)
            self.assertEqual(result['prefix_classification_unknown_attempts'], 2)
            self.assertEqual(result['neither_hidden_breakdown']['unknown_missing_retained_prefix_measurements'], 1)
            self.assertEqual(result['target_measurement_attempts'], 1)
            self.assertEqual(result['target_reason_counts_in_retained_subset'], {'closest_ratio_below_1_5': 1})
            self.assertIsNone(result['scientific_acceptance'])

    def test_changed_journal_hash_and_raw_hash_are_detected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(diagnostic.merge, 'ROOT', Path(directory)):
            root = Path(directory)
            data, ref, rows, journal = fixture(root, [{'prefix_measurement': prefix()}])
            Path(rows[0]['raw_attempt']['file']).write_text('{}')
            result = diagnostic.describe_source(data, ref)
            self.assertEqual(len(result['raw_errors']), 1)
            self.assertEqual(result['raw_unavailable_attempts'], 1)
            journal.write_text(journal.read_text() + '\n')
            with self.assertRaises(ValueError):
                diagnostic.describe_source(data, ref)

    def test_missing_raw_is_unknown_and_counter_mismatch_remains_explicit(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(diagnostic.merge, 'ROOT', Path(directory)):
            data, ref, _, _ = fixture(Path(directory), [{}])
            data['metrics']['attempts'] = 0
            result = diagnostic.describe_source(data, ref, attempt_reader=lambda *_: None)
            self.assertFalse(result['counter_consistency'])
            self.assertEqual(result['journal_attempts'], 1)
            self.assertEqual(result['raw_unavailable_attempts'], 1)
            self.assertEqual(result['prefix_classification_counts'], {})
            self.assertEqual(result['neither_hidden_breakdown'], {'unknown_missing_retained_prefix_measurements': 1})

    def test_raw_identity_mismatch_and_escaped_path_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(diagnostic.merge, 'ROOT', Path(directory)):
            root = Path(directory)
            data, ref, _, _ = fixture(root, [{}])
            result = diagnostic.describe_source(data, ref, attempt_reader=lambda *_: {'pair_id': 'wrong'})
            self.assertEqual(len(result['raw_errors']), 1)
            with self.assertRaises(ValueError):
                diagnostic.read_attempt({'file': str(root / 'outside.json'), 'sha256': '0' * 64}, root / 'scratch')

    def test_unavailable_journal_does_not_create_zero_attempts(self):
        data = {'_meta': {'exit_code': 1, 'end': 'ended'},
            'metrics': {'kind': 'pair_v2_shard', 'status': 'failed_before_generation', 'cleanup_errors': []},
            'config': {'scene': 'FloorPlan1', 'seed': 0, 'resolution': 255, 'shard': {'pbs_jobid': 'synthetic'}}}
        result = diagnostic.describe_source(data, {'file': 'results/missing.json'})
        self.assertIsNone(result['journal_attempts'])
        self.assertIsNone(result['raw_available_attempts'])
        self.assertIsNone(result['stage_counts'])

    def test_main_emits_exact_final_p8_sources_and_bounded_examples(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(diagnostic.merge, 'ROOT', Path(directory)):
            root = Path(directory)
            data, ref, _, _ = fixture(root, [{'prefix_measurement': prefix()} for _ in range(5)])
            capture = io.StringIO()
            with patch.object(diagnostic.merge, 'load_source', return_value=(data, ref)), redirect_stdout(capture):
                code = diagnostic.main(['--name', 'look_fixture', '--source', ref['file']])
            self.assertEqual(code, 0)
            output = json.loads((root / 'results/look_fixture.json').read_text())
            self.assertEqual(json.loads(capture.getvalue().split('FINAL_METRICS_JSON: ')[-1]), output['metrics'])
            self.assertEqual(output['sources'], [ref])
            self.assertEqual(len(output['metrics']['source_runs'][0]['examples']['hidden_rgb_failure']), 3)
            with self.assertRaises(FileExistsError):
                diagnostic.main(['--name', 'look_fixture', '--source', ref['file']])


if __name__ == '__main__':
    unittest.main()
