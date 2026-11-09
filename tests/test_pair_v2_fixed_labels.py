"""Synthetic CPU fixed-label regressions; no GPU or scientific result."""
import copy
import hashlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.envs import pair_v2_policy as legacy
from src.envs import pair_v2_fixed_labels as fixed
from src.envs.pair_table import RenderedView, label_assignment, mark_view
from src.envs.paired_env import RGBFrame


PAIR = 'fixed_labels_synthetic'
NAMES = ('candidate-a', 'candidate-b')


def capture(swapped=False, hidden=()):
    masks = {}
    for index, name in enumerate(NAMES):
        mask = np.zeros((255, 255), dtype=bool)
        x = (130 if index == 0 else 20) if swapped else (20 if index == 0 else 130)
        if name not in hidden:
            mask[30:60, x:x + 20] = True
        masks[name] = mask
    return {'rgb': np.full((255, 255, 3), 80, dtype=np.uint8), 'masks_by_name': masks,
            'metadata': {'sequenceId': 7, 'objects': [{'name': name} for name in NAMES]}}


def marked(value, *, pair_id=PAIR):
    with fixed.installed(), fixed.pair_context(pair_id, NAMES, declared=fixed.fields()):
        return legacy.mark_observation(value, NAMES, 'Select the marked object closest to the book.')


def raw_capture(value):
    payloads = {}
    def save(name, content):
        payloads[name] = content
        return {'file': name, 'sha256': hashlib.sha256(content).hexdigest()}
    buffer = io.BytesIO(); Image.fromarray(value['rgb']).save(buffer, format='PNG')
    rgb = save('rgb.png', buffer.getvalue())
    buffer = io.BytesIO()
    np.savez_compressed(buffer, names=np.asarray(list(NAMES)), masks=np.stack([value['masks_by_name'][name] for name in NAMES]))
    masks = save('masks.npz', buffer.getvalue())
    metadata = save('metadata.json', json.dumps(value['metadata']).encode())
    record = {'valid': True, 'black': False, 'rgb_shape': [255, 255, 3],
        'rgb_sha256': hashlib.sha256(value['rgb'].tobytes()).hexdigest(), 'sequence_id': 7,
        'mask_pixels_by_name': {name: int(mask.sum()) for name, mask in value['masks_by_name'].items()},
        'files': {'rgb': rgb, 'masks': masks, 'metadata': metadata}}
    return record, lambda ref: payloads[ref['file']]


class FixedLabelsTests(unittest.TestCase):
    def test_same_objects_keep_numbers_when_screen_order_swaps_between_worlds(self):
        expected = label_assignment(PAIR, NAMES)
        first, second = marked(capture()), marked(capture(swapped=True))
        self.assertEqual(first['mark_labels_by_name'], expected)
        self.assertEqual(second['mark_labels_by_name'], expected)
        self.assertNotEqual(first['marks'][0]['bounding_box'], second['marks'][0]['bounding_box'])

    def test_visible_subset_never_renumbers_second_object_to_one(self):
        assignment = label_assignment(PAIR, NAMES)
        second = next(name for name in NAMES if assignment[name] == 2)
        first = next(name for name in NAMES if assignment[name] == 1)
        observed = marked(capture(hidden=(first,)))
        self.assertEqual(observed['mark_labels_by_name'], {second: 2})
        self.assertEqual([item['label'] for item in observed['marks']], [2])
        self.assertEqual(marked(capture())['mark_labels_by_name'], assignment)
        self.assertFalse(any('name' in item for item in observed['marks']))

    def test_historical_spatial_policy_is_unchanged_after_context_exits(self):
        original = legacy.mark_observation
        a = original(capture(), NAMES, 'instruction')
        b = original(capture(swapped=True), NAMES, 'instruction')
        self.assertNotEqual(a['mark_labels_by_name'], b['mark_labels_by_name'])
        marked(capture())
        self.assertIs(legacy.mark_observation, original)
        self.assertTrue(legacy.public_exact(a, original(capture(), NAMES, 'instruction')))
        self.assertTrue(legacy.public_exact(b, original(capture(swapped=True), NAMES, 'instruction')))

    def test_installed_fixed_renderer_refuses_missing_or_different_pair_context(self):
        with fixed.installed():
            with self.assertRaisesRegex(ValueError, 'matching pair'):
                legacy.mark_observation(capture(), NAMES, 'instruction')
            with fixed.pair_context(PAIR, NAMES, declared=fixed.fields()):
                with self.assertRaisesRegex(ValueError, 'matching pair'):
                    legacy.mark_observation(capture(), ('candidate-a', 'foreign'), 'instruction')
        wrong = fixed.pair_fields(PAIR, NAMES)
        wrong['candidate_label_assignment'] = {name: 3 - value for name, value in wrong['candidate_label_assignment'].items()}
        with self.assertRaisesRegex(ValueError, 'assignment differs'):
            with fixed.pair_context(PAIR, NAMES, declared=wrong):
                self.fail('tampered context was admitted')

    def test_nested_pair_context_restores_outer_assignment(self):
        with fixed.installed(), fixed.pair_context(PAIR, NAMES, declared=fixed.fields()):
            before = legacy.mark_observation(capture(), NAMES, 'instruction')
            with fixed.pair_context('another-pair', NAMES, declared=fixed.fields()):
                self.assertEqual(legacy.mark_observation(capture(), NAMES, 'instruction')['mark_labels_by_name'],
                                 label_assignment('another-pair', NAMES))
            self.assertEqual(legacy.mark_observation(capture(), NAMES, 'instruction')['mark_labels_by_name'], before['mark_labels_by_name'])

    def test_t02_mapping_matches_review_and_witness_visible_subsets(self):
        assignment = label_assignment(PAIR, NAMES)
        for value in (capture(), capture(swapped=True), capture(hidden=(NAMES[0],))):
            observed = marked(value)
            view = RenderedView(rgb=RGBFrame(value['rgb'].tobytes(), 255, 255),
                masks={name: mask.astype(np.uint8).tobytes() for name, mask in value['masks_by_name'].items()},
                distances={name: 1.0 for name in NAMES}, agent={'synthetic': True},
                process_id='synthetic-process', render_id='synthetic-render', source={}, synthetic=True)
            _, table_marks = mark_view(view, assignment, NAMES)
            self.assertEqual({item.label for item in table_marks}, set(observed['mark_labels_by_name'].values()))
            for name, label in observed['mark_labels_by_name'].items():
                self.assertEqual(label, assignment[name])

    def witness_fixture(self):
        assignment = label_assignment(PAIR, NAMES)
        selected = next(name for name in NAMES if assignment[name] == 2)
        hidden = next(name for name in NAMES if name != selected)
        value = capture(swapped=True, hidden=(hidden,))
        frame, read = raw_capture(value)
        row = {**fixed.pair_fields(PAIR, NAMES), 'pair_id': PAIR,
            'identities': {'candidate_names': list(NAMES)}, 'instruction': 'instruction'}
        replay = {**fixed.pair_fields(PAIR, NAMES), 'captures': [frame], 'selected_name': selected,
            'actions': [{'kind': 'select', 'label': 2, 'selected_object_name': selected,
                         'mark_labels_by_name': {selected: 2}}],
            'planned_actions': ['select 2'], 'executed_actions': ['select 2']}
        return row, replay, read

    def test_witness_reconstructs_number_two_from_actual_single_visible_mask(self):
        row, replay, read = self.witness_fixture()
        proof = fixed.verify_witness_labels(row, replay, read)
        self.assertEqual(proof['terminal_visible_assignment'], {replay['selected_name']: 2})

    def test_witness_spatial_renumbering_or_missing_declaration_is_rejected(self):
        row, replay, read = self.witness_fixture()
        replay['actions'][-1]['label'] = 1
        with self.assertRaisesRegex(ValueError, 'fixed physical-object label'):
            fixed.verify_witness_labels(row, replay, read)
        row, replay, read = self.witness_fixture()
        del replay['fixed_label_version']
        with self.assertRaisesRegex(ValueError, 'assignment differs'):
            fixed.verify_witness_labels(row, replay, read)

    def test_table_package_requires_same_declared_mapping_and_preserves_legacy_status(self):
        row, _, _ = self.witness_fixture()
        package = {'pair_id': PAIR, 'candidate_names': list(NAMES), 'row': row, 'source_config': fixed.fields()}
        self.assertTrue(fixed.validate_table_package(package)['t01_t02_assignment_equal'])
        legacy_package = copy.deepcopy(package)
        legacy_package['source_config'] = {}
        for key in fixed.PAIR_FIELDS:
            del legacy_package['row'][key]
        self.assertIsNone(fixed.validate_table_package(legacy_package)['fixed_label_version'])
        del package['source_config']['fixed_label_version']
        with self.assertRaises(ValueError):
            fixed.validate_table_package(package)

    def test_new_pair_checker_does_not_bypass_original_scientific_failure(self):
        row, _, _ = self.witness_fixture()
        def original(value, path, **kwargs):
            self.assertEqual(legacy.mark_observation(capture(), NAMES, 'instruction')['mark_labels_by_name'],
                             label_assignment(PAIR, NAMES))
            raise ValueError('unchanged original physical gate failed')
        with self.assertRaisesRegex(ValueError, 'unchanged original physical gate failed'):
            fixed.checked_pair(original, row, '/synthetic/pair.json', declared_source=fixed.fields())
        self.assertIs(legacy.mark_observation, fixed._LEGACY_MARK)


if __name__ == '__main__':
    unittest.main()
