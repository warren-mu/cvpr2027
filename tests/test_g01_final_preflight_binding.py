"""Synthetic native-file provenance fixtures, not simulator measurements."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.envs import pair_v2_g01_policy as policy
from src.envs import pair_v2_g01_search as search
from src.envs import pair_v2_stay_order as stay
from src.tools import t01v2_g01_association as association

_spec = importlib.util.spec_from_file_location('g01_final_preflight_fixture',
    Path(__file__).with_name('test_t01v2_controller_association.py'))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def fixture(root):
    f = _fixture.Fixture(root)
    fields = {**policy.declared('FloorPlan203'), **stay.fields(), 'producer_module': policy.PRODUCER,
        'enable_candidate_clone': False, 'scene_pairs_per_class': 8}
    for value in (f.data['config'], f.data['config']['shard'], f.data['metrics']):
        for key in tuple(value):
            if key.startswith('look_unit_'):
                value.pop(key)
        value.update(fields)
    f.pair.update(kind='direct_stay', **policy.fields(), **search.fields(),
        producer_module=policy.PRODUCER, proposed_pool='direct',
        proposal={'mode': 'stay', 'projection_pool': {'pool': 'direct'}})
    owner = {'pid': 11, 'start_ticks': 90, 'pgid': 11, 'sid': 11}
    leases, portable = [], []
    for world in (0, 1):
        portable.append(f.write(f.directory / f'witness_w{world}.json',
            {'process_id': 100 + world, 'unity_pid': 200 + world}))
        leases.append({'token': f'lease{world}', 'owner': owner,
            'worker_identity': {'pid': 100 + world}, 'unity_identity': {'pid': 200 + world},
            'release_confirmed': True, 'cleanup_errors': [], 'remaining_owned_group': False})
    f.pair['portable_files'] = portable
    f.data['metrics']['g01_witness_admission'] = {'owner': owner, 'lease_records': leases}
    f.refresh_pair()
    return f


def rewrite_preflight(f, index, passed):
    record, visit = f.records[index], f.visits[index]
    path = Path(visit['preflight_reference']['file'])
    record['preflight'].update(passed=passed, error=None,
        reasons=[] if passed else ['initial_inventory_changed_or_displacement_above_0_5m'])
    ref = f.write(path, record['preflight'])
    record['preflight_ref'] = ref
    record['entries'] = [ref if Path(item['file']) == path else item for item in record['entries']]
    visit['preflight_reference'] = ref
    f.refresh_index(index)
    if index == 0:
        f.pair['scene_preflight_reference'] = f.write(f.directory / 'scene_preflight.json', record['preflight'])
        f.refresh_pair()


def final_failure(f):
    rewrite_preflight(f, 0, True)
    rewrite_preflight(f, 1, False)
    f.visits[1]['preflight_passed'] = False
    metrics = f.data['metrics']
    metrics.update(status='completed_search', selected_controller_id='marked_controller_000',
        scene_preflight_reference=f.visits[1]['preflight_reference'],
        scene_excluded={'scene': 'FloorPlan203', 'reason': 'settled_baseline_window_preflight_failed',
            'reasons': f.records[1]['preflight']['reasons'], 'evidence': f.visits[1]['preflight_reference']})


class FinalPreflightBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.f = fixture(Path(temporary.name))
        self.checker = patch.object(association.merge, 'verify_preflight',
            side_effect=_fixture.original_preflight_fixture)
        self.native = self.checker.start()
        self.addCleanup(self.checker.stop)

    def verify(self):
        f = self.f
        return association.verify_pair_controller(f.data, f.source_ref, f.journal, f.pair, f.pair_path, root=f.root)

    def test_original_normal_path_keeps_original_proof_schema(self):
        proof = self.verify()
        self.assertEqual(proof['source_final_controller_id'], 'marked_controller_001')
        self.assertNotIn('source_final_preflight_binding', proof)
        self.assertEqual(proof['policy'], association.POLICY)
        self.native.assert_called_once()

    def test_final_failed_visit_is_authenticated_without_rewriting_original_source(self):
        final_failure(self.f)
        before = copy.deepcopy((self.f.data, self.f.pair, self.f.raw_attempt))
        proof = self.verify()
        binding = proof['source_final_preflight_binding']
        self.assertEqual(binding['selected_controller_id'], 'marked_controller_000')
        self.assertEqual(binding['last_attempted_controller_id'], 'marked_controller_001')
        self.assertEqual(proof['controller_id'], 'marked_controller_000')
        self.assertEqual(proof['source_final_controller_id'], 'marked_controller_000')
        self.assertEqual(proof['original_source_final_preflight_reference'],
                         self.f.visits[1]['preflight_reference'])
        self.assertEqual((self.f.data, self.f.pair, self.f.raw_attempt), before)
        self.native.assert_called_once()
        self.assertEqual(len(proof['witness_resource_admission']['worlds']), 2)

    def test_wrong_reference_sha_is_rejected(self):
        final_failure(self.f)
        self.f.data['metrics']['scene_preflight_reference'] = {
            **self.f.visits[1]['preflight_reference'], 'sha256': '0' * 64}
        with self.assertRaisesRegex(ValueError, 'SHA-mismatched'):
            self.verify()

    def test_repointing_final_and_exclusion_to_previous_success_is_rejected(self):
        final_failure(self.f)
        ref = self.f.visits[0]['preflight_reference']
        self.f.data['metrics']['scene_preflight_reference'] = ref
        self.f.data['metrics']['scene_excluded']['evidence'] = ref
        with self.assertRaisesRegex(ValueError, 'exact original exclusion'):
            self.verify()

    def test_forged_failed_declaration_does_not_override_native_passed_file(self):
        final_failure(self.f)
        rewrite_preflight(self.f, 1, True)
        ref = self.f.visits[1]['preflight_reference']
        self.f.data['metrics']['scene_preflight_reference'] = ref
        self.f.data['metrics']['scene_excluded']['evidence'] = ref
        with self.assertRaisesRegex(ValueError, 'native verdict'):
            self.verify()

    def test_selecting_final_failed_visit_is_rejected(self):
        final_failure(self.f)
        self.f.data['metrics']['selected_controller_id'] = 'marked_controller_001'
        with self.assertRaisesRegex(ValueError, 'last preflight-passed'):
            self.verify()

    def test_selecting_an_earlier_success_is_rejected_even_when_refs_match(self):
        self.f.data['metrics']['selected_controller_id'] = 'marked_controller_000'
        self.f.data['metrics']['scene_preflight_reference'] = self.f.visits[0]['preflight_reference']
        with self.assertRaisesRegex(ValueError, 'last preflight-passed'):
            self.verify()

    def test_exclusion_scene_reason_and_native_reasons_must_all_match(self):
        final_failure(self.f)
        original = copy.deepcopy(self.f.data['metrics']['scene_excluded'])
        for key, value in (('scene', 'FloorPlan204'), ('reason', 'different_failure'),
                           ('reasons', ['fabricated_failure'])):
            with self.subTest(key=key):
                self.f.data['metrics']['scene_excluded'] = {**original, key: value}
                with self.assertRaises(ValueError):
                    self.verify()

    def test_missing_final_index_membership_is_rejected(self):
        final_failure(self.f)
        path = self.f.visits[1]['preflight_reference']['file']
        self.f.records[1]['entries'] = [ref for ref in self.f.records[1]['entries'] if ref['file'] != path]
        self.f.refresh_index(1)
        with self.assertRaisesRegex(ValueError, 'index membership'):
            self.verify()

    def test_tampered_final_preflight_bytes_are_rejected(self):
        final_failure(self.f)
        Path(self.f.visits[1]['preflight_reference']['file']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'SHA-mismatched'):
            self.verify()

    def test_failed_visit_cleanup_or_unordered_ledger_is_rejected(self):
        final_failure(self.f)
        self.f.visits[1]['cleanup_errors'] = ['owned process remains']
        with self.assertRaisesRegex(ValueError, 'ownership/cleanup'):
            self.verify()
        self.f.visits[1]['cleanup_errors'] = []
        self.f.visits.reverse()
        with self.assertRaisesRegex(ValueError, 'final-failed-visit'):
            self.verify()

    def test_original_pair_preflight_and_witness_checks_remain_required(self):
        final_failure(self.f)
        with patch.object(association.merge, 'verify_preflight', side_effect=ValueError('native gate failed')):
            with self.assertRaisesRegex(ValueError, 'native gate failed'):
                self.verify()
        self.f.data['metrics']['g01_witness_admission']['lease_records'] = []
        with self.assertRaisesRegex(ValueError, 'actual admitted resource lease'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
