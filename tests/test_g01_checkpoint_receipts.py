"""Synthetic receipt binding tests; audited science is an explicit boundary.

These fixtures certify no simulator result. They test immutable receipt linkage
without invoking a native source loader or accepting a generator pass flag.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools import g01_checkpoint_receipts as receipts
from src.tools import g01_quota_plan as quota
from src.tools import merge_pair_v2 as merge
from src.envs import pair_table_source as sources


class Fixture:
    def __init__(self, root):
        self.root = root
        self.name = 't01v2_checkpoint_test_receipt_fixture'
        self.source_name = 'gen_test_fixture_c6_floorplan21'
        self.source_path = root / 'results' / (self.source_name + '.json')
        shard_path = root / 'scratch/shard_config.json'
        shard = {'name': self.source_name, 'scene': 'FloorPlan21'}
        shard_ref = self.write(shard_path, shard)
        self.source = {'_meta': {'end': 'ended', 'exit_code': 1, 'git_commit': 'a' * 40,
            'command': ['python3', '-u', '-m', receipts.policy.PRODUCER, '--name', self.source_name,
                        '--scene', 'FloorPlan21', '--shard-config', str(shard_path), '--timeout-seconds', '1200']},
            'config': {'name': self.source_name, 'scene': 'FloorPlan21', 'producer_module': receipts.policy.PRODUCER,
                       'g01_partition': 'test', 'shard': shard}, 'metrics': {'cleanup_errors': []}}
        self.write(self.source_path, self.source)
        _, self.source_ref = self.audit(self.source_path)
        self.source_ref.update(shard_config_file=str(shard_path), shard_config_sha256=shard_ref['sha256'])
        for filename, digest, suffix in (('log_file', 'log_sha256', '.log'), ('meta_file', 'meta_sha256', '.meta.json')):
            ref = self.write(root / 'logs/runs' / (self.source_name + suffix), {'synthetic_original_producer_record': True})
            self.source_ref[filename], self.source_ref[digest] = ref['file'], ref['sha256']
        self.directory = root / 'data/world_pairs/v2' / self.source_name / 'floorplan21_00001'
        view = {'position': {'x': 0, 'y': 0.9, 'z': 1}, 'yaw': 270, 'horizon': 30}
        setup = {'position': view['position'], 'rotation': {'x': 0, 'y': 270, 'z': 0},
                 'horizon': 30, 'standing': True, 'forceAction': False}
        worlds = [{'agent_setup_command': setup,
                   'agent': {'position': view['position'], 'rotation': setup['rotation'],
                             'horizon': 30.000003814697266, 'standing': True}} for _ in (0, 1)]
        self.portable = [self.write(self.directory / f'snapshot_w{i}.json', world) for i, world in enumerate(worlds)]
        image = self.directory / 'review.png'; image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b'synthetic authenticated image bytes; no pixel claim')
        self.image_ref = self.reference(image)
        self.pair = {'pair_id': 'floorplan21_00001', 'scene': 'FloorPlan21', 'kind': 'look_flip',
            'status': 'passed', 'producer_module': receipts.policy.PRODUCER, 'g01_partition': 'test',
            'view': view, 'identities': {'anchor_name': 'X', 'candidate_names': ['A', 'B']},
            'target_w0': 'A', 'target_w1': 'B', 'search_controller_id': 'marked_controller_000',
            'scene_preflight_reference': {'file': 'original_unread_native/preflight', 'sha256': 'a' * 64},
            'attempt_setup_reference': {'file': 'original_unread_native/setup', 'sha256': 'b' * 64},
            'portable_files': self.portable, 'review_image': self.image_ref,
            'proposal': {f'w{i}': {'spawn': {'receptacle_name': 'Table', 'selected_position': {'x': i, 'y': 1, 'z': 0}}} for i in (0, 1)}}
        self.pair_ref = self.write(self.directory / 'pair.json', self.pair)
        identity = merge.proposal_identity(self.pair)
        association = {'version': 'g01_pair_controller_preflight_association_v1',
            'controller_id': 'marked_controller_000', 'source_record_mutated': False,
            'source_reference': self.source_ref, 'pair_preflight_reference': self.pair['scene_preflight_reference'],
            'attempt_setup_reference': self.pair['attempt_setup_reference']}
        checked = {'g01_partition': 'test', 'validity': {'acceptance_passed': True, 'kind': 'look_flip', 'reasons': []},
            'verification_scope': 'construction-stage and applicable supplement 4(a); P-table/BFS/no-leak/cache checks not run',
            'proposal_identity': identity, 'witnesses': [{'passed': True, 'target_name': name} for name in ('A', 'B')],
            'controller_preflight_association': copy.deepcopy(association), 'review_image': self.image_ref}
        self.entry = {'pair_id': self.pair['pair_id'], 'scene': 'FloorPlan21', 'kind': 'look_flip',
            'target_w0': 'A', 'target_w1': 'B', 'source_file': self.source_ref['file'],
            'source_sha256': self.source_ref['sha256'], 'source_metrics_sha256': self.source_ref['metrics_sha256'],
            'pair_reference': self.pair_ref, 'proposal_identity': identity, 'review_image': self.image_ref,
            'controller_preflight_association': association, 'independent_verification': checked}
        self.review_dir = root / 'logs/review' / self.name
        self.review_dir.mkdir(parents=True)
        published = self.review_dir / 'floorplan21_00001.png'; published.write_bytes(image.read_bytes())
        review = {**self.reference(published), 'pair_id': self.pair['pair_id'], 'scene': 'FloorPlan21',
                  'kind': 'look_flip', 'source_file': self.source_ref['file'], 'source_image': self.image_ref}
        counts = {kind: int(kind == 'look_flip') for kind in merge.KINDS}
        self.result_path = root / 'results' / (self.name + '.json')
        self.container = {'_meta': {'name': self.name, 'exit_code': 0, 'end': 'ended', 'git_commit': 'c' * 40,
            'command': ['python3', '-m', sources.G01_CHECKPOINT_MODULE, '--name', self.name,
                        '--source', self.source_ref['file']]},
            'config': {'name': self.name, 'requested_sources': [self.source_ref['file']],
                       'review_directory': self.review_dir.relative_to(root).as_posix()},
            'sources': [self.source_ref], 'metrics': {'kind': 't01v2_review_checkpoint',
                'status': 'checkpoint_ready_for_human_review', 'errors': [], 'final_pilot_acceptance': False,
                'published_pair_count': 1, 'independently_verified_pair_count': 1,
                'published_by_class': counts, 'independently_verified_by_class': counts,
                'candidate_pairs': [self.entry], 'review_files': [review],
                'source_runs': [{'source': self.source_ref, 'eligible': True, 'ledger_only': False, 'cleanup_errors': []}]}}
        self.refresh()

    def reference(self, path):
        return {'file': path.relative_to(self.root).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + '\n')
        return self.reference(path)

    def refresh(self):
        self.expected = self.write(self.result_path, self.container)

    def audit(self, path):
        path = (self.root / path).resolve(); data = json.loads(path.read_text())
        return data, {**self.reference(path), 'metrics_sha256': merge.canonical_hash(data['metrics']),
                      'git_commit': data['_meta']['git_commit'], 'audit': {'warn': [], 'fail': []}}


class CheckpointReceiptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.f = Fixture(Path(temporary.name))
        for target, key, value in ((sources, 'audited_result', self.f.audit),
                                   (receipts, 'audited_result', self.f.audit)):
            active = patch.object(target, key, side_effect=value)
            active.start(); self.addCleanup(active.stop)
        # Policy validation is a separate tested immutable-schema boundary.
        active = patch.object(receipts.policy, 'validate_source')
        self.policy = active.start(); self.addCleanup(active.stop)
        active = patch.object(sources, 'load_reviewed_pair', side_effect=AssertionError('No full native replay here'))
        self.full = active.start(); self.addCleanup(active.stop)

    def load(self):
        return receipts.load_checkpoint(self.f.expected, root=self.f.root)

    def test_successful_independent_receipt_reuses_science_without_native_replay(self):
        _, _, entries, packages = self.load()
        self.assertEqual(len(entries), 1)
        package = next(iter(packages.values()))
        self.assertFalse(package['native_scientific_revalidation_performed'])
        self.assertFalse(package['training_authorized'])
        self.assertEqual(package['independent_verification'], self.f.entry['independent_verification'])
        self.policy.assert_called_once(); self.full.assert_not_called()

    def test_result_and_pair_sha_changes_are_rejected(self):
        self.f.result_path.write_text(self.f.result_path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'exact successful'): self.load()
        self.f.refresh()
        (self.f.directory / 'pair.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'SHA-mismatched'): self.load()

    def test_failed_error_or_unpublished_checkpoint_never_becomes_authority(self):
        original = copy.deepcopy(self.f.container)
        for change in ('exit', 'error', 'status', 'count', 'producer'):
            self.f.container = copy.deepcopy(original)
            if change == 'exit': self.f.container['_meta']['exit_code'] = 1
            elif change == 'error': self.f.container['metrics']['errors'] = ['original native failure']
            elif change == 'status': self.f.container['metrics']['status'] = 'invalid_evidence'
            elif change == 'count': self.f.container['metrics']['published_pair_count'] = 0
            else: self.f.container['_meta']['command'][2] = 'src.tools.t01v2_review_checkpoint'
            self.f.refresh()
            with self.subTest(change=change), self.assertRaises(ValueError): self.load()

    def test_missing_or_forged_saved_independent_verification_is_rejected(self):
        original = copy.deepcopy(self.f.container)
        for change in ('missing', 'false', 'target', 'proposal', 'controller'):
            self.f.container = copy.deepcopy(original)
            entry = self.f.container['metrics']['candidate_pairs'][0]
            if change == 'missing': entry.pop('independent_verification')
            elif change == 'false': entry['independent_verification']['validity']['acceptance_passed'] = False
            elif change == 'target': entry['independent_verification']['witnesses'][0]['target_name'] = 'B'
            elif change == 'proposal': entry['proposal_identity'] = 'f' * 64
            else: entry['independent_verification']['controller_preflight_association']['controller_id'] = 'foreign'
            self.f.refresh()
            with self.subTest(change=change), self.assertRaises(ValueError): self.load()

    def test_original_source_bytes_or_incomplete_cutoff_are_rejected(self):
        self.f.source_path.write_text(self.f.source_path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'changed since'): self.load()
        self.f.write(self.f.source_path, self.f.source)
        self.f.container['metrics']['source_runs'] = []
        self.f.refresh()
        with self.assertRaisesRegex(ValueError, 'incomplete source-run'): self.load()

    def test_world_and_published_review_hash_changes_are_rejected(self):
        world = self.f.root / self.f.portable[0]['file']
        original = world.read_bytes(); world.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'SHA-mismatched'): self.load()
        world.write_bytes(original)
        (self.f.review_dir / 'floorplan21_00001.png').write_bytes(b'altered')
        with self.assertRaisesRegex(ValueError, 'SHA-mismatched'): self.load()

    def test_source_actual_producer_and_immutable_shard_are_bound(self):
        self.f.source['_meta']['command'][3] = 'src.envs.unrelated_generator'
        changed = self.f.write(self.f.source_path, self.f.source)
        self.f.source_ref['sha256'] = changed['sha256']
        self.f.refresh()
        with self.assertRaisesRegex(ValueError, 'required module producer'): self.load()

    def test_source_shard_hash_change_cannot_reuse_the_old_receipt(self):
        Path(self.f.source_ref['shard_config_file']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'SHA-mismatched'): self.load()

    def test_explicit_receipt_inventory_uses_shared_configuration_selection(self):
        with patch.object(quota, 'ROOT', self.f.root):
            proof = quota.reconstruct_test_inventory([self.f.expected], configuration=True,
                                                    authority=quota.RECEIPT_AUTHORITY)
        self.assertEqual(proof['confirmed_by_kind']['look_flip'], 1)
        self.assertFalse(proof['native_scientific_revalidation_performed'])
        self.assertEqual(proof['count_basis_authority'], receipts.VERSION)
        self.full.assert_not_called()
        with self.assertRaises(ValueError):
            quota.reconstruct_test_inventory([self.f.expected], authority=quota.RECEIPT_AUTHORITY)


if __name__ == '__main__':
    unittest.main()
