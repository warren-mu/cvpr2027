"""Synthetic CPU provenance regressions; no simulator or scientific result."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.envs import pair_v2_look_unit_policy as policy
from src.envs import pair_v2_stay_order as stay
from src.tools import t01v2_controller_association as association
from src.tools import t01v2_controller_review_checkpoint as collector


class Fixture:
    def __init__(self, root):
        self.root = root
        self.raw = root / 'scratch'
        self.directory = root / 'data/world_pairs/v2/source/pair'
        self.pair_path = self.directory / 'pair.json'
        self.records, self.visits = [], []
        for index in (0, 1):
            controller = f'marked_controller_{index:03d}'
            directory = self.raw / controller / 'construction'
            snapshot = {'scene': 'FloorPlan203', 'key': 'baseline', 'worker_pid': 11,
                'unity_pid': 22 + index, 'objects': [], 'agent': {}}
            baseline_ref = self.write(directory / 'baseline.json', snapshot)
            setup_ref = self.write(directory / 'initial_canonical_agent_setup.json', {'actual': 'synthetic'})
            preflight = {'preflight_version': 'settled_baseline_window_v1',
                'snapshot': {**snapshot, 'reference': baseline_ref}, 'initial_agent_setup_reference': setup_ref}
            pre_ref = self.write(directory / 'scene_preflight.json', preflight)
            restored_ref = self.write(directory / 'restored.json', {**snapshot, 'key': 'restored'})
            worlds = [self.write(directory / f'world{world}.json', {**snapshot, 'key': f'world{world}'}) for world in (0, 1)]
            attempt_setup = {'pair_id': 'floorplan203_00001', 'scene': 'FloorPlan203', 'status': 'passed',
                'setup_passed': True, 'error': None, 'baseline_reference': baseline_ref,
                'restored_snapshot_reference': restored_ref, 'restoration': {'restored_snapshot_reference': restored_ref}}
            attempt_ref = self.write(directory / 'attempt_setup.json', attempt_setup)
            entries = [baseline_ref, setup_ref, pre_ref, restored_ref, *worlds, attempt_ref]
            index_ref = self.write(directory / 'evidence_index.json', {'files': entries})
            self.records.append({'preflight': preflight, 'preflight_ref': pre_ref, 'baseline_ref': baseline_ref,
                'attempt_setup': attempt_setup, 'attempt_ref': attempt_ref, 'worlds': worlds, 'entries': entries})
            self.visits.append({'controller_id': controller, 'role': 'natural_marked', 'cleanup_errors': [],
                'preflight_passed': True, 'worker_pid': 11, 'unity_identity': {'pid': 22 + index},
                'preflight_reference': pre_ref, 'initial_agent_setup_reference': setup_ref, 'raw_index': index_ref})
        first = self.records[0]
        portable = self.write(self.directory / 'scene_preflight.json', first['preflight'])
        self.pair = {'pair_id': 'floorplan203_00001', 'scene': 'FloorPlan203', 'mode': 'stay',
            'kind': 'look_stay', 'status': 'passed', 'discard_reasons': [], 'search_controller_id': 'marked_controller_000',
            'scene_preflight_reference': portable, 'attempt_setup_reference': first['attempt_ref'],
            'attempt_setup': copy.deepcopy(first['attempt_setup']), 'physical_world_references': first['worlds'],
            'target_w0': 'A', 'target_w1': 'A'}
        fields = {**policy.camera_fields(60, 'FloorPlan10'), **stay.fields(), 'producer_module': policy.PRODUCER}
        self.data = {'_meta': {'exit_code': 1, 'end': 'synthetic-ended'},
            'config': {**fields, 'scene': 'FloorPlan203', 'raw_evidence_dir': str(self.raw),
                'shard': {**fields, 'scene': 'FloorPlan203', 'pbs_jobid': 'synthetic-job'},
                'solvability_version': 'synthetic-solvability'},
            'metrics': {**fields, 'controller_visits': self.visits, 'selected_controller_id': 'marked_controller_001',
                'scene_preflight_reference': self.visits[1]['preflight_reference'], 'cleanup_errors': []}}
        self.source_ref = {'file': 'results/synthetic.json', 'sha256': 'a' * 64, 'metrics_sha256': 'b' * 64}
        self.refresh_pair()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        path.write_bytes(content)
        return {'file': str(path), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}

    def refresh_pair(self):
        self.pair_ref = self.write(self.pair_path, self.pair)
        self.raw_attempt = {**copy.deepcopy(self.pair), 'pair_file': str(self.pair_path), 'pair_sha256': self.pair_ref['sha256']}
        self.refresh_raw()

    def refresh_raw(self):
        raw_ref = self.write(self.raw / 'trial' / 'attempt.json', self.raw_attempt)
        self.journal = {key: copy.deepcopy(self.raw_attempt[key]) for key in
            ('pair_id', 'scene', 'kind', 'status', 'discard_reasons', 'pair_file', 'pair_sha256')}
        self.journal.update(raw_attempt=raw_ref, generation_seconds=1.0)
        path = self.root / 'data/world_pairs/v2/journal.jsonl'
        content = (json.dumps(self.journal, sort_keys=True) + '\n').encode()
        path.write_bytes(content)
        self.journal_ref = {'file': str(path), 'sha256': hashlib.sha256(content).hexdigest()}

    def refresh_index(self, index=0):
        path = Path(self.visits[index]['raw_index']['file'])
        self.visits[index]['raw_index'] = self.write(path, {'files': self.records[index]['entries']})

    def verify(self):
        return association.verify_pair_controller(self.data, self.source_ref, self.journal, self.pair,
                                                  self.pair_path, root=self.root)


def original_preflight_fixture(preflight, read, *, canonical_setup):
    # Confirm association delegates with authenticated indexed reads. The real
    # verify_preflight is not replaced in production and remains in check_pair.
    if canonical_setup is not True:
        raise AssertionError('Canonical setup validation must remain enabled')
    read(preflight['snapshot']['reference'])
    read(preflight['initial_agent_setup_reference'])
    return {'passed': True, 'scope': 'synthetic fixture'}


class ControllerAssociationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = Fixture(Path(self.directory.name))
        self.validation = patch.object(association.merge, 'verify_preflight', side_effect=original_preflight_fixture)
        self.preflight = self.validation.start()
        self.addCleanup(self.validation.stop)

    def test_prior_controller_is_bound_without_rewriting_final_source(self):
        before = copy.deepcopy((self.fixture.data, self.fixture.pair, self.fixture.raw_attempt))
        proof = self.fixture.verify()
        self.assertEqual(proof['controller_id'], 'marked_controller_000')
        self.assertEqual(proof['source_final_controller_id'], 'marked_controller_001')
        self.assertEqual(proof['controller_preflight_reference']['sha256'], self.fixture.visits[0]['preflight_reference']['sha256'])
        self.assertNotEqual(proof['controller_preflight_reference']['sha256'], proof['original_source_final_preflight_reference']['sha256'])
        self.assertFalse(proof['source_record_mutated'])
        self.assertEqual(before, (self.fixture.data, self.fixture.pair, self.fixture.raw_attempt))
        self.preflight.assert_called_once()

    def test_pair_raw_controller_disagreement_never_uses_hash_fallback(self):
        self.fixture.raw_attempt['search_controller_id'] = 'marked_controller_001'
        self.fixture.refresh_raw()
        with self.assertRaisesRegex(ValueError, 'same explicit Controller'):
            self.fixture.verify()

    def test_same_forged_controller_id_does_not_accept_foreign_preflight(self):
        self.fixture.pair['search_controller_id'] = 'marked_controller_001'
        self.fixture.refresh_pair()
        with self.assertRaisesRegex(ValueError, 'exact pair Controller original bytes'):
            self.fixture.verify()

    def test_missing_and_duplicate_visits_fail(self):
        self.fixture.data['metrics']['controller_visits'] = [self.fixture.visits[1]]
        with self.assertRaisesRegex(ValueError, 'Controller identity'):
            self.fixture.verify()
        self.fixture.data['metrics']['controller_visits'] = [*self.fixture.visits, self.fixture.visits[0]]
        with self.assertRaisesRegex(ValueError, 'Controller identity'):
            self.fixture.verify()

    def test_source_final_preflight_must_still_bind_final_selected_visit(self):
        self.fixture.data['metrics']['scene_preflight_reference'] = self.fixture.visits[0]['preflight_reference']
        with self.assertRaisesRegex(ValueError, 'final source preflight'):
            self.fixture.verify()

    def test_missing_index_membership_and_duplicate_files_fail(self):
        first = self.fixture.records[0]
        preflight = first['preflight_ref']
        first['entries'] = [row for row in first['entries'] if row != preflight]
        self.fixture.refresh_index()
        with self.assertRaisesRegex(ValueError, 'absent from its authenticated raw index'):
            self.fixture.verify()
        first['entries'].extend([preflight, preflight]); self.fixture.refresh_index()
        with self.assertRaisesRegex(ValueError, 'Duplicate or foreign file'):
            self.fixture.verify()

    def test_index_size_and_portable_sha_tamper_fail(self):
        self.fixture.records[0]['entries'][2] = {**self.fixture.records[0]['entries'][2], 'bytes': 1}
        self.fixture.refresh_index()
        with self.assertRaisesRegex(ValueError, 'file size'):
            self.fixture.verify()
        self.fixture.records[0]['entries'][2] = self.fixture.records[0]['preflight_ref']
        self.fixture.refresh_index()
        Path(self.fixture.pair['scene_preflight_reference']['file']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'SHA-mismatched'):
            self.fixture.verify()

    def test_worker_identity_and_cleanup_failure_are_rejected(self):
        self.fixture.visits[0]['worker_pid'] = 999
        with self.assertRaisesRegex(ValueError, 'another scene or Controller'):
            self.fixture.verify()
        self.fixture.visits[0]['worker_pid'] = 11
        self.fixture.visits[0]['cleanup_errors'] = ['synthetic unresolved process']
        with self.assertRaisesRegex(ValueError, 'cleanup'):
            self.fixture.verify()

    def test_foreign_baseline_cannot_be_substituted_even_when_both_inline_records_agree(self):
        setup = self.fixture.records[0]['attempt_setup']
        setup['baseline_reference'] = self.fixture.records[1]['baseline_ref']
        new = self.fixture.write(Path(self.fixture.records[0]['attempt_ref']['file']), setup)
        self.fixture.records[0]['entries'][-1] = new; self.fixture.refresh_index()
        self.fixture.pair['attempt_setup_reference'] = new
        self.fixture.pair['attempt_setup'] = copy.deepcopy(setup)
        self.fixture.refresh_pair()
        with self.assertRaisesRegex(ValueError, 'escaped'):
            self.fixture.verify()

    def test_original_preflight_failure_propagates(self):
        self.preflight.side_effect = ValueError('original preflight gate failed')
        with self.assertRaisesRegex(ValueError, 'original preflight gate failed'):
            self.fixture.verify()

    def test_nonlook_source_is_not_admitted(self):
        self.fixture.data['config']['producer_module'] = 'src.envs.pair_v2_marked_shard'
        with self.assertRaises(ValueError):
            self.fixture.verify()

    def test_collection_still_calls_full_pair_validator_and_propagates_failure(self):
        f = self.fixture
        eligible = {'eligible': True, 'partial_source': True}
        with patch.object(collector.merge, 'ROOT', f.root), patch.object(collector.merge, 'load_source', return_value=(f.data, f.source_ref)), \
             patch.object(collector, 'verified_passed_rows', return_value=(eligible, f.journal_ref, [(f.journal, f.pair_ref, f.pair_path)])), \
             patch.object(collector.merge, 'final_pair_eligible', return_value=True), \
             patch.object(collector.merge, 'check_pair', side_effect=ValueError('original pixel/witness gate failed')) as full:
            result = collector.collect_sources([f.root / 'results/synthetic.json'])
        self.assertEqual(result[2], [])
        self.assertEqual(result[5][0]['error'], 'ValueError: original pixel/witness gate failed')
        full.assert_called_once()
        self.assertEqual(set(full.call_args.kwargs), {
            'raw_root', 'clone_enabled', 'preflight_version', 'evidence_version', 'search_version', 'search_policy',
            'search_clone_source', 'solvability_version', 'solvability_policy', 'generation_version', 'generation_policy',
            'attempt_setup_version', 'attempt_setup_policy', 'candidate_policy', 'candidate_policy_spec',
            'projection_pool_version', 'projection_pool_policy', 'frozen_render_version', 'frozen_render_policy', 'field_of_view'})

    def test_zero_pair_nonlook_source_fails_before_eligibility_or_legacy_handling(self):
        f = self.fixture
        f.data['config']['producer_module'] = 'src.envs.pair_v2_marked_shard'
        with patch.object(collector.merge, 'load_source', return_value=(f.data, f.source_ref)), \
             patch.object(collector, 'verified_passed_rows') as rows:
            result = collector.collect_sources([f.root / 'results/synthetic.json'])
        rows.assert_not_called()
        self.assertEqual(result[2], [])
        self.assertIn('explicit Look-only source', result[5][0]['error'])



if __name__ == '__main__':
    unittest.main()
