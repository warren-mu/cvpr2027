import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.diag02_pool_options import alternatives, combine_rows, scan_cache_receipts, summary
from src.envs.pair_table_balance import balance_pairs


def row(key, kind='look_stay', side='a', scene='FloorPlan1', distance=0.01):
    return {'pair_id': key, 'original_pair_id': key, 'dataset_key': key,
            'proposal_identity': 'proposal_' + key, 'pair_reference': {'file': key, 'sha256': key},
            'candidate_names': ['a', 'b'], 'target_names': ['a', 'b'] if kind.endswith('flip') else [side, side],
            's0_target_left': [True, False], 'scene': scene, 'kind': kind,
            'stay_displacement': {'distance_m': distance}, 'synthetic': True}


class PoolOptions(unittest.TestCase):
    def test_no_displacement_gate_and_holdout_excluded(self):
        data = [row('a'), row('b', side='b'), row('c', scene='FloorPlan15'), row('f', kind='look_flip')]
        report = alternatives(data, ['FloorPlan15'], balance_pairs)
        self.assertEqual(report['no_stay_displacement_no_AB_balance']['pair_count'], 3)
        self.assertEqual(report['only_original_scene_AB_balance_seed0']['pair_count'], 3)
        self.assertEqual(report['holdout_excluded']['pair_count'], 1)
        self.assertFalse(report['stay_displacement_filter_applied'])
        self.assertFalse(report['pool_frozen'])

    def test_original_AB_pools_both_stay_classes_per_scene(self):
        data = [row('a'), row('b'), row('c', kind='direct_stay', side='b'), row('f', kind='look_flip')]
        result = alternatives(data, [], balance_pairs)
        expected = balance_pairs(data, seed=0)
        self.assertEqual(result['original_balance'], expected)
        self.assertEqual(result['only_original_scene_AB_balance_seed0']['pair_count'], 3)
        self.assertEqual(result['only_original_scene_AB_balance_seed0']['by_class']['look_flip']['pair_count'], 1)
        self.assertEqual(result['only_original_scene_AB_balance_seed0']['by_class']['look_stay']['pair_count'], 1)

    def test_duplicate_exact_cache_identity_is_not_counted_twice(self):
        data = row('a')
        self.assertEqual(combine_rows([data], [copy.deepcopy(data)]), [data])

    def test_conflicting_duplicate_fails(self):
        data = row('a'); changed = copy.deepcopy(data); changed['target_names'] = ['b', 'b']
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            combine_rows([data], [changed])

    def test_duplicate_proposal_across_distinct_keys_fails(self):
        a, b = row('a'), row('b'); b['proposal_identity'] = a['proposal_identity']
        with self.assertRaisesRegex(ValueError, 'duplicate one proposal'):
            combine_rows([a], [b])

    def test_zero_stay_ratio_is_explicitly_unknown(self):
        result = summary([row('f', kind='look_flip')])
        self.assertEqual(result['look_flip_to_look_stay'], {'numerator': 1, 'denominator': 0, 'ratio': None})

    def test_scene_counts_are_per_class_distinct_scenes(self):
        result = summary([row('a'), row('b'), row('c', scene='FloorPlan2'), row('f', kind='look_flip')])
        self.assertEqual(result['by_class']['look_stay']['pair_count'], 3)
        self.assertEqual(result['by_class']['look_stay']['scene_count'], 2)
        self.assertEqual(result['by_class']['look_flip']['scene_count'], 1)

    def test_input_unchanged_and_order_deterministic(self):
        data = [row('a'), row('b'), row('c', side='b'), row('f', kind='look_flip')]
        original = copy.deepcopy(data)
        first = alternatives(data, [], balance_pairs)
        second = alternatives(list(reversed(data)), [], balance_pairs)
        self.assertEqual(first['no_stay_displacement_no_AB_balance'], second['no_stay_displacement_no_AB_balance'])
        self.assertEqual(first['only_original_scene_AB_balance_seed0'], second['only_original_scene_AB_balance_seed0'])
        self.assertEqual(data, original)


class CertificateAdmission(unittest.TestCase):
    def fixture(self, root, name, *, kind='t02_formal_cache_check', exit_code=0,
                cache=True, downstream=True, end='2026-11-10T01:02:03'):
        p = root / 'results' / (name + '.json')
        p.parent.mkdir(exist_ok=True)
        data = {'_meta': {'end': end, 'exit_code': exit_code},
                'metrics': {'kind': kind, 'cache_matches_live': cache,
                            'downstream_allowed': downstream}}
        p.write_text(json.dumps(data))
        ref = {'file': str(p.relative_to(root)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        return data, ref

    def test_export_success_and_unsealed_never_admitted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root, 't02_export_cache_export_only', kind='t02_table_export')
            self.fixture(root, 't02_export_cache_running', end=None)
            observed, extra, sources = scan_cache_receipts(root, [])
            self.assertEqual(extra, [])
            self.assertEqual(sources, [])
            self.assertEqual({r['disposition'] for r in observed},
                             {'not_an_original_formal_cache_result', 'unsealed_not_eligible'})

    def test_failed_cache_is_audited_but_not_admitted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data, ref = self.fixture(root, 't02_export_cache_failed', exit_code=1, cache=False, downstream=False)
            with patch('src.tools.t01_partition.audited_result', return_value=(data, ref)):
                observed, extra, sources = scan_cache_receipts(root, [])
            self.assertEqual(extra, [])
            self.assertEqual(sources, [ref])
            self.assertEqual(observed[0]['disposition'], 'failed_or_incomplete_not_eligible')

    def test_covered_cache_not_recounted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, ref = self.fixture(root, 't02_export_cache_covered')
            observed, extra, _ = scan_cache_receipts(root, [ref])
            self.assertEqual(extra, [])
            self.assertEqual(observed[0]['disposition'], 'already_in_fixed83_certificates')

    def test_test_certificate_excluded_and_training_admitted(self):
        for scene, expected in [('FloorPlan21', False), ('FloorPlan1', True)]:
            with self.subTest(scene=scene), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                _, ref = self.fixture(root, 't02_export_cache_new')
                evidence = {'source': ref, 'complete': [{'dataset_key': 'pair_one',
                    'table_manifest': {'file': str(root / 'table.json'), 'sha256': 'not-used-in-fixture'}}]}
                with patch('src.tools.smoke_e2e_pilot_certificates.certificate_metadata', return_value=evidence), \
                     patch('src.tools.g01_train_inventory.read', return_value={'scene': scene}):
                    observed, extra, _ = scan_cache_receipts(root, [])
                self.assertEqual(extra, [ref] if expected else [])
                self.assertEqual(observed[0]['disposition'],
                    'additional_training_cache_certificate' if expected else 'test_certificate_excluded')


if __name__ == '__main__':
    unittest.main()
