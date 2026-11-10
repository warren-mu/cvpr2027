"""Technical CPU fixtures only; no synthetic fixture authorizes a real table."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from src.envs import pair_table
from src.envs.pair_table_checks import check_pair
from src.envs.pair_table_export import save_table


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fig = load('fig01_cpu_export', Path(__file__).resolve().parents[1] / 'experiments' / 'fig01_export.py')
fixtures = load('fig01_original_table_fixtures',
    Path(pair_table.__file__).resolve().parents[2] / 'tests' / 'test_pair_table.py')


def original_checks(table):
    t01 = {'pair_id': table.pair_id, 'selected_action': 'turn_right',
        'anchor_visible_actions_w0': ['turn_right'], 'anchor_visible_actions_w1': ['turn_left'],
        'witnesses': [{'actions': [{'kind': 'turn_right'}, {'kind': 'select', 'target_name': 'A'}]},
                      {'actions': [{'kind': 'turn_left'}, {'kind': 'select', 'target_name': 'B'}]}]}
    checked = check_pair(table, t01)
    assert checked['passed']
    return checked


class FigureTests(unittest.TestCase):
    def test_original_witness_and_normalized_world_order(self):
        table = fixtures.table()
        seen, reason = fig.witness_first_seen(table, table.pair_id, [1, 0], original_checks(table),
                                              allow_synthetic=True)
        self.assertIsNone(reason)
        self.assertEqual(seen['w0']['original_world_id'], 'w1')
        self.assertEqual(seen['w0']['action_prefix'], ['turn_left'])
        self.assertEqual(seen['w1']['action_prefix'], ['turn_right'])
        self.assertEqual((seen['w0']['anchor_pixels'], seen['w0']['tau'], seen['w0']['tau_seen']), (250, 1, 1))

    def test_unavailable_original_witness_is_not_replaced(self):
        table = fixtures.table()
        checked = original_checks(table)
        checked['witness_replay'][0] = {'applicable': False, 'passed': None, 'reason': 'original_witness_exceeds_D_plus_1'}
        seen, reason = fig.witness_first_seen(table, table.pair_id, [0, 1], checked, allow_synthetic=True)
        self.assertIsNone(seen)
        self.assertEqual(reason, 'original_witness_unavailable_or_not_applicable')

    def test_original_trace_tamper_fails(self):
        table = fixtures.table()
        checked = original_checks(table)
        checked['witness_replay'][0]['trace'][0]['state_id'] = table.initial_state
        with self.assertRaisesRegex(ValueError, 'trace changed'):
            fig.witness_first_seen(table, table.pair_id, [0, 1], checked, allow_synthetic=True)

    def test_no_production_synthetic_or_authority_bypass(self):
        table = fixtures.table()
        with self.assertRaisesRegex(ValueError, 'Synthetic'):
            fig.witness_first_seen(table, table.pair_id, [0, 1], original_checks(table))
        self.assertFalse(table.export_verified)
        self.assertFalse(table.cache_verified)

    def test_ranking_min_both_worlds_stable_and_nonholdout_first(self):
        def item(key, pixels, heldout=False):
            return {'dataset_key': key, 'heldout_scene': heldout,
                'ranking_min_anchor_pixels': min(pixels),
                'first_seen': {f'w{i}': {'state_id': f's{i}', 'anchor_pixels': value} for i, value in enumerate(pixels)}}
        rows = [item('b', [900, 250]), item('c', [500, 500]), item('z', [2000, 2000], True), item('a', [250, 900])]
        self.assertEqual([r['dataset_key'] for r in sorted(rows, key=fig.rank_key)], ['c', 'a', 'b', 'z'])

    def test_five_original_environment_images_lossless_masks_and_sources(self):
        table = fixtures.table()
        checked = original_checks(table)
        seen, _ = fig.witness_first_seen(table, table.pair_id, [1, 0], checked, allow_synthetic=True)
        candidate = {'dataset_key': table.pair_id, 'world_order': [1, 0], 'heldout_scene': False,
                     'first_seen': seen, 'ranking_min_anchor_pixels': 250}
        package = {'worlds': [{'objects': [{'name': name, 'objectType': name + '_type'} for name in ('X', 'A', 'B')]}],
                   'source_provenance': {'synthetic_test_only': True}}
        before = [(v.rgb.data, copy.deepcopy(v.masks)) for state in table.states.values() for v in state.captures]
        with tempfile.TemporaryDirectory() as tmp:
            save_table(table, Path(tmp) / 'table')
            ref = fig.export_pair(table, package, candidate, Path(tmp) / 'figure',
                                  code_commit='a' * 40, allow_synthetic=True)
            body = Path(ref['file']).read_bytes()
            self.assertEqual(hashlib.sha256(body).hexdigest(), ref['sha256'])
            manifest = json.loads(body)
            self.assertEqual(len(manifest['frames']), 5)
            self.assertEqual(manifest['correct_targets'], {'w0': 'B', 'w1': 'A'})
            self.assertEqual(manifest['s0_agent_input_sha256']['w0'], manifest['s0_agent_input_sha256']['w1'])
            self.assertNotEqual(manifest['s0_original_raw_rgb_sha256']['w0'], manifest['s0_original_raw_rgb_sha256']['w1'])
            first = manifest['frames'][0]
            self.assertNotEqual(first['agent_input']['decoded_pixels_sha256'], first['raw_rgb']['decoded_pixels_sha256'])
            env = fig.environment(table, table.pair_id, [1, 0], allow_synthetic=True)
            try:
                expected = env.reset(table.pair_id, 'w0')
                with Image.open(first['agent_input']['file']) as image:
                    self.assertEqual(image.tobytes(), expected.image.data)
            finally:
                env.close()
            for row in manifest['frames']:
                for value in row['masks'].values():
                    with Image.open(value['png']['file']) as image:
                        pixels = image.tobytes()
                        self.assertLessEqual(set(pixels), {0, 255})
                        self.assertEqual(pixels.count(255), value['pixels'])
            with self.assertRaises(FileExistsError):
                fig.export_pair(table, package, candidate, Path(tmp) / 'figure',
                                code_commit='a' * 40, allow_synthetic=True)
        self.assertEqual(before, [(v.rgb.data, v.masks) for state in table.states.values() for v in state.captures])


if __name__ == '__main__':
    unittest.main()
