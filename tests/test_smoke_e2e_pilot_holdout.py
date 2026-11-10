"""Actual freeze-loader integration; auditor is the explicit synthetic boundary."""
import copy
from unittest.mock import patch
import unittest
import test_train_holdout_freeze as freeze_fixture
from src.tools import smoke_e2e_pilot_dataset as dataset
from src.tools import train_holdout_freeze as freeze


class PilotHoldoutTests(unittest.TestCase):
    setUp = freeze_fixture.HoldoutFreezeTests.setUp
    audit_fixture = freeze_fixture.HoldoutFreezeTests.audit_fixture
    receipt = freeze_fixture.HoldoutFreezeTests.receipt

    def prepared(self):
        document, reference, _ = freeze.freeze(self.reference, root=self.root)
        self.receipt(document, reference)
        manifest_ref = self.data['metrics']['manifest']
        proof = copy.deepcopy(self.data['metrics']['dataset_proof'])
        proof['manifest_reference'] = dataset.ref_identity(manifest_ref)
        return reference, manifest_ref, proof

    def test_real_frozen_loader_binds_completed_population_and_exact_split(self):
        ref, manifest_ref, proof = self.prepared()
        with patch.object(dataset, 'ROOT', self.root):
            actual = dataset.authenticate_pilot_holdout(manifest_ref, proof, ref)
        self.assertEqual(actual['reference'], dataset.ref_identity(ref))
        self.assertTrue(actual['pilot_population_and_split_verified'])
        self.assertEqual(actual['selected_scenes'], self.manifest['split']['selected_scenes'])

    def test_missing_or_changed_original_freeze_receipt_is_rejected(self):
        ref, manifest_ref, proof = self.prepared()
        (self.root / 'results/train_holdout_freeze.json').unlink()
        with patch.object(dataset, 'ROOT', self.root), self.assertRaises(FileNotFoundError):
            dataset.authenticate_pilot_holdout(manifest_ref, proof, ref)

    def test_success_flag_cannot_replace_full_manifest_population_binding(self):
        ref, manifest_ref, original = self.prepared()
        for field, value in [('cache_authorized', False), ('common_pool_hash', '0' * 64),
                ('train_keys', []), ('eval_keys', []), ('manifest_reference', None), ('split', {})]:
            proof = copy.deepcopy(original); proof[field] = value
            with self.subTest(field=field), patch.object(dataset, 'ROOT', self.root), self.assertRaises(ValueError):
                dataset.authenticate_pilot_holdout(manifest_ref, proof, ref)

    def test_no_second_scene_selection_or_native_revalidation(self):
        ref, manifest_ref, proof = self.prepared()
        # Original qualification metadata checker recomputes the original split;
        # the runtime assertion does not rebuild the growing-pool/native pipeline.
        with patch.object(dataset, 'ROOT', self.root), \
                patch.object(dataset, 'build_pool', side_effect=AssertionError('native rebuild forbidden')):
            dataset.authenticate_pilot_holdout(manifest_ref, proof, ref)


if __name__ == '__main__': unittest.main()
