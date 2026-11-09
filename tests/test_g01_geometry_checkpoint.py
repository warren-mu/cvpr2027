"""Synthetic G01/source routing over the real indexed supplement15 proof core."""
import copy
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools import t01v2_g01_review_checkpoint as review
from src.tools import t01v2_flip_ranked_review_checkpoint as ranked
from src.envs import pair_v2_g01_policy as policy
from src.envs import pair_v2_g01_search as search
from src.envs import pair_v2_fixed_labels as labels
from src.envs import pair_v2_stay_order as stay
from src.envs import pair_v2_flip_rounds_search as cadence


def evidence_fixture(directory, *, pool='look', kind='look_flip'):
    spec = importlib.util.spec_from_file_location('g01_shared_geometry_fixture',
        Path(__file__).with_name('test_t01v2_flip_geometry_checkpoint.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    f = module.Fixture(directory)
    # This creates a synthetic G01 fixture from identical raw native inputs.
    # No production source is rewritten or made to pretend it used ranked.
    for key in ('flip_unit_version', 'flip_unit_policy', 'flip_search_version', 'flip_search_policy'):
        f.row.pop(key)
    f.row.update(policy.fields(), **search.fields(), **cadence.fields(),
        **labels.pair_fields(f.row['pair_id'], ['A', 'B']), producer_module=policy.PRODUCER,
        proposed_pool=pool, kind=kind)
    catalog = copy.deepcopy(f.read(f.pool_ref))
    catalog.update(search.fields())
    f.pool_ref = f.write(Path(f.pool_ref['file']), catalog)
    f.row['projection_pool_reference'] = f.pool_ref
    f.refresh_index()
    f.row['proposal']['projection_pool']['pool'] = pool
    if pool == 'direct':
        for key in ('flip_geometry_initial_tuple', 'flip_geometry_final_tuple', 'flip_geometry_native_guards'):
            f.row.pop(key)
    cfg = {**policy.declared(f.scene), **stay.fields(), 'scene': f.scene, 'producer_module': policy.PRODUCER,
        'raw_evidence_dir': str(f.root / 'raw'), 'enable_candidate_clone': False, 'scene_pairs_per_class': 8}
    cfg['shard'] = copy.deepcopy(cfg)
    f.data = {'config': cfg, 'metrics': {key: copy.deepcopy(value) for key, value in cfg.items() if key != 'shard'}}
    f.source_ref = {'file': 'results/synthetic_g01.json', 'sha256': 'source-sha'}
    f.association = {'controller_index_reference': f.index, 'baseline_reference': f.baseline_ref,
        'controller_id': 'marked_controller_000', 'source_final_controller_id': 'marked_controller_001',
        'witness_resource_admission': {'fixture': 'independently owned association boundary'}}
    return f


@contextmanager
def source_fixture(f, science, *, admission_error=None):
    """Original source lifecycle and scientific checker are explicit mocks.

    Used-point geometry/native files are NOT mocked, and the admission guard
    must still be called before the pair can reach any scientific checker.
    """
    old_role = review.roles.verify_proposal_role
    with patch.object(review, '_BASE_LOAD', return_value=(f.data, f.source_ref)), \
            patch.object(review, 'verify_collection', return_value={}) as collection, \
            patch.object(review, 'verify_admission', side_effect=admission_error, return_value={}) as admission, \
            patch.object(review, 'verify_pair_source', return_value=f.association) as associated, \
            patch.object(review.labels, 'checked_pair', side_effect=lambda fn, row, path, declared_source, **kw: fn(row, path, **kw)), \
            patch.object(review.merge, 'ROOT', f.root), \
            patch.object(review.merge, 'check_pair', side_effect=science) as original:
        try:
            with review.source_support():
                review.merge.load_source('synthetic-source')
                yield original, associated, collection, admission
        finally:
            if review.roles.verify_proposal_role is not old_role:
                raise AssertionError('G01 source leaked its process-local role verifier')


class G01GeometryCheckpointTests(unittest.TestCase):
    def test_shared_evidence_handles_look_pool_cross_over_without_ranked_identity_spoof(self):
        for kind in ('look_flip', 'direct_flip'):
            with self.subTest(actual_kind=kind), tempfile.TemporaryDirectory() as directory:
                f = evidence_fixture(directory, kind=kind)
                before = copy.deepcopy(f.row)
                proof = ranked.verify_geometry_evidence(f.row, f.pair_path, f.baseline, f.read,
                                                        f.data['config'], root=f.root)
                self.assertTrue(proof['passed'])
                self.assertEqual(proof['verified_native_attempts_by_world'], {'w0': 2, 'w1': 2})
                self.assertEqual(f.row, before)
                with self.assertRaisesRegex(ValueError, 'explicit new v2'):
                    ranked.verify_used_geometry(f.row, f.pair_path, f.baseline, f.read,
                                                f.data['config'], root=f.root)
                self.assertNotIn('flip_unit_version', f.row)

    def test_look_pool_uses_center_proof_and_preserves_full_scientific_kwargs(self):
        for kind in ('look_flip', 'direct_flip'):
            with self.subTest(actual_kind=kind), tempfile.TemporaryDirectory() as directory:
                f = evidence_fixture(directory, kind=kind)
                arguments = {'raw_root': f.data['config']['raw_evidence_dir'], 'field_of_view': 60,
                    'frozen_render_policy': {'fixture': 'original scientific argument'}}
                positions = {obj['name']: obj['position'] for obj in f.baseline['objects']}
                def science(row, path, **kwargs):
                    self.assertIs(row, f.row); self.assertEqual(kwargs, arguments)
                    role = review.roles.verify_proposal_role(row, positions)
                    self.assertTrue(role['passed']); self.assertEqual(role['g01_search_version'], search.VERSION)
                    self.assertNotIn('flip_search_version', role)
                    return {'scientific_fixture': 'original full checker completed'}
                with patch.object(search, 'verify_flip_role', side_effect=AssertionError('Look must not use surface roles')):
                    with source_fixture(f, science) as (original, associated, collection, admission):
                        result = review.merge.check_pair(f.row, f.pair_path, **arguments)
                        original.assert_called_once_with(f.row, f.pair_path, **arguments)
                        associated.assert_called_once(); collection.assert_called_once(); admission.assert_called_once()
                binding = result['g01_proposal_verification']
                self.assertEqual(binding['proposed_pool'], 'look'); self.assertEqual(binding['actual_kind'], kind)
                self.assertEqual(binding['class_cross_over'], kind == 'direct_flip')
                self.assertTrue(result['complete_flip_geometry_verification']['passed'])

    def test_direct_pool_cross_over_keeps_original_surface_role_without_look_claim(self):
        for kind in ('direct_flip', 'look_flip'):
            with self.subTest(actual_kind=kind), tempfile.TemporaryDirectory() as directory:
                f = evidence_fixture(directory, pool='direct', kind=kind)
                positions = {obj['name']: obj['position'] for obj in f.baseline['objects']}
                def science(row, path, **kwargs):
                    self.assertTrue(review.roles.verify_proposal_role(row, positions)['passed'])
                    return {'scientific_fixture': 'unchanged actual rendered class checker'}
                with patch.object(ranked, 'verify_geometry_evidence', side_effect=AssertionError('No Look-pool proof for Direct')):
                    with source_fixture(f, science):
                        result = review.merge.check_pair(f.row, f.pair_path, raw_root=f.data['config']['raw_evidence_dir'])
                self.assertIsNone(result['complete_flip_geometry_verification'])
                self.assertEqual(result['g01_proposal_verification']['class_cross_over'], kind == 'look_flip')
                self.assertEqual(result['g01_proposal_verification']['proposal_verifier'], 'original_native_surface_3d_role')

    def test_stay_delegates_to_original_role_and_science(self):
        with tempfile.TemporaryDirectory() as directory:
            f = evidence_fixture(directory, pool='direct', kind='direct_stay')
            f.row['mode'] = f.row['proposal']['mode'] = 'stay'
            def science(row, path, **kwargs):
                self.assertEqual(review.roles.verify_proposal_role(row, {}), {'original_stay': True})
                return {'science': True}
            with patch.object(review.roles, 'verify_proposal_role', return_value={'original_stay': True}) as role, \
                    patch.object(ranked, 'verify_geometry_evidence', side_effect=AssertionError('Stay geometry changed')):
                with source_fixture(f, science):
                    result = review.merge.check_pair(f.row, f.pair_path, raw_root=f.data['config']['raw_evidence_dir'])
                role.assert_called_once_with(f.row, {})
            self.assertEqual(result['g01_proposal_verification']['proposal_verifier'], 'original_stay_role')

    def test_geometry_native_proof_and_original_scientific_failure_cannot_be_bypassed(self):
        for failure in ('native_membership', 'world1_opposite', 'scientific_gate'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                f = evidence_fixture(directory)
                if failure == 'native_membership': f.row['proposal']['w0']['spawn']['coordinates'] = []
                elif failure == 'world1_opposite':
                    f.row['flip_geometry_native_guards']['w1'][0]['opposite_reference'] = f.row['flip_geometry_native_guards']['w0'][0]['opposite_reference']
                with source_fixture(f, lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('original scientific failure'))):
                    with self.assertRaises(ValueError):
                        review.merge.check_pair(f.row, f.pair_path, raw_root=f.data['config']['raw_evidence_dir'])

    def test_identity_pool_and_admission_errors_fail_closed(self):
        for failure in ('ranked_claim', 'old_g01', 'pool_mismatch', 'missing_source', 'cleanup'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                f = evidence_fixture(directory)
                if failure == 'ranked_claim': f.row['flip_unit_version'] = ranked.policy.VERSION
                elif failure == 'old_g01': f.row['g01_version'] = 'g01_mixed_train80_selected60_v1'
                elif failure == 'pool_mismatch': f.row['proposed_pool'] = 'direct'
                error = ValueError('resource cleanup unknown') if failure == 'cleanup' else None
                with self.assertRaises(ValueError):
                    with source_fixture(f, lambda *a, **k: (_ for _ in ()).throw(AssertionError('must stop before science')),
                                        admission_error=error):
                        review.merge.check_pair(f.row, f.pair_path,
                            raw_root='/unknown' if failure == 'missing_source' else f.data['config']['raw_evidence_dir'])

    def test_authenticated_pair_matching_still_calls_full_controller_and_lease_association(self):
        with tempfile.TemporaryDirectory() as directory:
            f = evidence_fixture(directory)
            pair_ref = f.write(f.pair_path, f.row)
            journal = {'pair_id': f.row['pair_id'], 'raw_attempt': {'file': 'original-attempt'}}
            with patch.object(review, 'verified_passed_rows', return_value=(
                    {'eligible': True}, {}, [(journal, pair_ref, f.pair_path)])), \
                    patch.object(review.association, 'verify_pair_controller', return_value=f.association) as association:
                result = review.verify_pair_source(f.data, f.source_ref, f.row, f.pair_path, root=f.root)
                self.assertEqual(result, f.association)
                association.assert_called_once_with(f.data, f.source_ref, journal, f.row, f.pair_path, root=f.root)
                changed = copy.deepcopy(f.row); changed['kind'] = 'direct_flip'
                with self.assertRaisesRegex(ValueError, 'portable pair'):
                    review.verify_pair_source(f.data, f.source_ref, changed, f.pair_path, root=f.root)


if __name__ == '__main__':
    unittest.main()
