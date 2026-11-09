"""Synthetic CPU boundaries only; no native/cache certification is manufactured."""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.tools import smoke_e2e_pilot_dataset as dataset
from src.tools import t02_cache_check as cache
from src.envs import pair_table_balance as balance
from src.envs.pair_table_identity import identity_fields


def pool_fixture():
    tables, packages, normalized = {}, {}, {}
    for room in (0, 200, 300, 400):
        for scene_number in range(1, 6):
            scene = f'FloorPlan{room + scene_number}'
            for kind in dataset.KINDS:
                # Stay has both actual sides within every scene. The 160-input
                # fixture remains four-class after the unchanged balance tool.
                for side in (0, 1):
                    original = f'{scene}_{kind}_{side}'
                    ref = {'file': f'data/fixture/{original}.json', 'sha256': hashlib.sha256(original.encode()).hexdigest()}
                    identity = identity_fields(original, ref); key = identity['dataset_key']
                    targets = ['A', 'B'] if kind.endswith('flip') else [['A', 'A'], ['B', 'B']][side]
                    tables[key] = SimpleNamespace(pair_id=original, dataset_key=key, scene=scene, kind=kind,
                        source={'field_of_view': 60}, synthetic=False, export_verified=True, cache_verified=True,
                        manifest_reference={'file': f'/scratch/{original}/table.json', 'sha256': 'a' * 64},
                        shadow_metadata={'shadow_divergent_states': 0, 'non_anchor_divergent_states': 0, 'in_clean_subset': kind.startswith('look_')})
                    packages[key] = {'independent_verification': {'proposal_identity': hashlib.sha256(('proposal:' + original).encode()).hexdigest()},
                                     'source_provenance': {'pair': ref}}
                    normalized[key] = {'pair_id': original, 'original_pair_id': original, 'dataset_key': key,
                        'dataset_identity': identity, 'scene': scene, 'kind': kind, 'candidate_names': ['A', 'B'],
                        'target_names': targets, 's0_target_left': [True, False], 'synthetic': False}
    return tables, packages, normalized


class PilotPoolTests(unittest.TestCase):
    def build(self, tables=None, packages=None):
        original, original_packages, normalized = pool_fixture()
        tables, packages = tables or original, packages or original_packages
        ref = {'file': 'results/cache.json', 'sha256': 'a' * 64}
        proof = {'source': ref}
        with patch.object(dataset, 'authenticate_cache_batch', return_value=(tables, packages, proof)), \
             patch.object(balance, 'normalize_table', side_effect=lambda table: normalized[table.dataset_key]):
            return dataset.build_pool([ref], dataset.SCENE_SPLIT)

    def test_actual_balance_and_scene_holdout_preserve_world_orders_and_no_overlap(self):
        manifest, tables, packages, proof = self.build()
        self.assertEqual(sum(manifest['counts_by_kind'].values()), 160)
        self.assertTrue(manifest['balance']['after']['global_balance_within_0_4_to_0_6'])
        held = set(manifest['split']['selected_scenes'])
        self.assertEqual({dataset.room_type(scene) for scene in held}, {'kitchen', 'living', 'bedroom', 'bathroom'})
        self.assertEqual(len(held), 4)
        self.assertTrue(proof['cache_authorized'])
        self.assertFalse(set(proof['train_keys']) & set(proof['eval_keys']))
        self.assertEqual(set(proof['train_keys']) | set(proof['eval_keys']), set(tables))
        self.assertEqual(set(packages), set(tables))
        for key in proof['eval_keys']:
            self.assertIn(tables[key].scene, held)
        for key in proof['train_keys']:
            self.assertNotIn(tables[key].scene, held)
        self.assertEqual(proof['world_orders'], {row['dataset_key']: row['world_order'] for row in manifest['entries']})
        unhashed = copy.deepcopy(manifest); unhashed.pop('common_pool_hash')
        self.assertEqual(manifest['common_pool_hash'], dataset.canonical_hash(unhashed))

    def test_split_is_stable_under_input_order_and_same_scene_is_indivisible(self):
        entries = self.build()[0]['entries']
        self.assertEqual(dataset.split_entries(entries, dataset.SCENE_SPLIT),
                         dataset.split_entries(list(reversed(entries)), dataset.SCENE_SPLIT))
        for invalid in ('pair_id_hash_seed0_twenty_percent_v1', None):
            with self.assertRaises(ValueError):
                dataset.split_entries(entries, invalid)
        for change in ({'dataset_key': entries[0]['dataset_key']}, {'proposal_identity': entries[0]['proposal_identity']}, {'scene': 'FloorPlan225'}):
            bad = copy.deepcopy(entries); bad[1].update(change)
            with self.assertRaises(ValueError): dataset.split_entries(bad, dataset.SCENE_SPLIT)

    def test_test_scene_nonformal_and_other_camera_cannot_enter(self):
        for field, value in (('scene', 'FloorPlan221'), ('synthetic', True), ('cache_verified', False),
                             ('export_verified', False), ('source', {'field_of_view': 100})):
            tables, packages, _ = pool_fixture(); setattr(next(iter(tables.values())), field, value)
            with self.assertRaises(ValueError): self.build(tables, packages)

    def test_size_gate_applies_after_balance_and_no_class_can_be_missing(self):
        tables, packages, _ = pool_fixture()
        reduced = {key: table for key, table in tables.items() if not table.kind.endswith('stay')}
        with self.assertRaisesRegex(ValueError, '80 unique pairs and all four'):
            self.build(reduced, {key: packages[key] for key in reduced})
        reduced = dict(list(tables.items())[:64])
        with self.assertRaisesRegex(ValueError, '80 unique pairs and all four'):
            self.build(reduced, {key: packages[key] for key in reduced})

    def test_duplicate_proposals_not_dropped_even_across_different_ids(self):
        tables, packages, _ = pool_fixture()
        first, second = list(packages)[:2]
        packages[second]['independent_verification'] = copy.deepcopy(packages[first]['independent_verification'])
        with self.assertRaisesRegex(ValueError, 'Repeated pair source/proposal'):
            self.build(tables, packages)

    def test_manifest_recomputed_not_trusting_saved_authorization_flags(self):
        manifest, tables, packages, proof = self.build()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'pool.json'
            path.write_text(json.dumps(manifest)); ref = {'file': 'pool.json', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            with patch.object(dataset, 'ROOT', root), patch.object(dataset, 'build_pool', return_value=(manifest, tables, packages, proof)):
                actual = dataset.load_pilot_pool(ref)
                self.assertEqual(actual[2]['manifest_reference'], ref)
                changed = copy.deepcopy(manifest); changed['split']['train_keys'].append(changed['split']['eval_keys'][0])
                path.write_text(json.dumps(changed)); badref = {**ref, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                with self.assertRaisesRegex(ValueError, 'pool/split differs'):
                    dataset.load_pilot_pool(badref)
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    dataset.load_pilot_pool(ref)


class FormalBatchTests(unittest.TestCase):
    def test_only_fresh_audit_annotation_is_ignored_not_source_bytes_or_membership(self):
        original = [{'source': {'file': 'results/e.json', 'sha256': 'a' * 64, 'audit': {'warn': ['old']}},
                     'population_membership': 'complete_independently_verified_table'}]
        changed = copy.deepcopy(original); changed[0]['source']['audit'] = {'warn': []}
        self.assertEqual(dataset.immutable_ledger(original), dataset.immutable_ledger(changed))
        changed[0]['source']['sha256'] = 'b' * 64
        self.assertNotEqual(dataset.immutable_ledger(original), dataset.immutable_ledger(changed))
        self.assertEqual(original[0]['source']['audit'], {'warn': ['old']})

    def fixture(self, root):
        name, job = 't02_cache_fixture', '123.hopper-m-02'
        directory = root / 'scratch/t02_cache' / name / job
        directory.mkdir(parents=True)
        cfg = {'name': name, 'version': cache.VERSION, 'source': ['results/export.json'],
            'gpus': 1, 'resolution': 255, 'seed': 0, 'sample_size': 200,
            'pbs_jobid': job, 'scratch': str(root / 'scratch'), 'scratch_directory': str(directory), 'node': 'fixture'}
        plan = {'population_sha256': 'p' * 64, 'selected_keys': [['dataset', 'state']]}
        cfg['population_sha256'] = plan['population_sha256']
        # Scientific sampling/native/authorization are explicit mock boundaries
        # below. This tiny fixture tests that the wrapper cannot skip any proof.
        value = SimpleNamespace(states={'state': SimpleNamespace(captures=[SimpleNamespace(process_id='old')])},
            manifest_reference={'file': '/scratch/table.json', 'sha256': 'b' * 64},
            shadow_metadata={'shadow_divergent_states': 2, 'in_clean_subset': False})
        observed = [SimpleNamespace(fresh=[SimpleNamespace(process_id='new')])]
        pair_dir = directory / 'pair_0000'; pair_dir.mkdir()
        raw = cache.save_json(pair_dir / 'raw_index.json', {})
        # Production requires the exact raw/evidence_index.json path.
        (pair_dir / 'raw').mkdir(); raw = cache.save_json(pair_dir / 'raw/evidence_index.json', {})
        row = {'dataset_key': 'dataset', 'status': 'captured', 'error': None, 'cleanup_errors': [], 'raw_index': raw}
        row['report_reference'] = cache.save_json(pair_dir / 'worker_report.json', copy.deepcopy(row))
        row['scratch_directory'] = str(pair_dir)
        actual = {'cache_matches_live': True}
        recorded = {**actual, 'table_shadow_metadata': {'dataset': value.shadow_metadata},
            'native_authenticated_render_ids': [['authenticated']], 'producer_cleanup_errors': [], 'producer_errors': [],
            'table_native_and_pair_checks_authorized': True}
        metrics = {'kind': 't02_formal_cache_check', 'producer_exit_code': 0, 'errors': [], 'cleanup_errors': [],
            'black_frame_detected': False, 'cache_matches_live': True, 'downstream_allowed': True,
            'stop_downstream': False, 'sample_complete': True, 'gate': {'file': 'results/gate.json', 'sha256': 'c' * 64},
            'population_ledger': [{'fixture': True}], 'sampling_plan': cache.save_json(directory / 'sampling_plan.json', plan),
            'worker_runs': [row], 'validation_report': cache.save_json(directory / 'cache_validation.json', recorded)}
        metrics['authorization_manifest'] = cache.save_json(directory / 'cache_authorization.json', {
            'version': cache.VERSION, 'tables': {'dataset': value.manifest_reference},
            'table_shadow_metadata': recorded['table_shadow_metadata'], 'cache_validation': metrics['validation_report'],
            'sampling_plan': metrics['sampling_plan'], 'full_T02_acceptance': None})
        source = {'file': 'results/cache.json', 'sha256': 'a' * 64}
        data = {'config': cfg, 'metrics': metrics, '_meta': {'name': name, 'end': 'fixture-end', 'exit_code': 0,
            'command': ['python3', '-m', 'src.tools.t02_cache_check', '--name', name, '--source', 'results/export.json']}}
        return data, source, value, observed, plan, actual

    def call(self, root, fixture, *, failure=None):
        data, source, value, observed, plan, actual = fixture
        with ExitStack() as stack:
            stack.enter_context(patch.object(dataset, 'ROOT', root))
            stack.enter_context(patch.object(dataset, 'audited_result', side_effect=lambda path:
                (data, source) if path.name == 'cache.json' else ({}, data['metrics']['gate'])))
            gate = stack.enter_context(patch.object(cache, 'gate_runtime'))
            load = stack.enter_context(patch.object(cache, 'load_exports', return_value=({'dataset': value},
                {'dataset': {'cache_evidence_roots': []}}, {'dataset': {'passed': True}}, [], data['metrics']['population_ledger'])))
            stack.enter_context(patch('src.envs.pair_table_cache.sample_state_keys', return_value=plan))
            stack.enter_context(patch.object(cache, 'load_samples', return_value=observed))
            native = stack.enter_context(patch('src.envs.pair_table_checks.authenticate_cache_sources', return_value={('authenticated',)}, side_effect=failure))
            stack.enter_context(patch('src.envs.pair_table_cache.validate_cache_sample', return_value=copy.deepcopy(actual)))
            authorize = stack.enter_context(patch('src.envs.pair_table_checks.authorize_tables'))
            result = dataset.authenticate_cache_batch(source)
            self.assertEqual((gate.call_count, load.call_count, native.call_count, authorize.call_count), (1, 1, 1, 1))
            return result

    def test_full_chain_and_immutable_reports_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture = self.fixture(root)
            self.assertEqual(set(self.call(root, fixture)[0]), {'dataset'})
            path = Path(fixture[0]['metrics']['validation_report']['file']); path.write_text('{}')
            with self.assertRaises(ValueError): self.call(root, fixture)

    def test_failed_missing_or_unsealed_cache_is_not_training_authorization(self):
        for change in ('exit', 'end', 'sample', 'cleanup', 'workers'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); fixture = self.fixture(root); data = fixture[0]
                if change == 'exit': data['_meta']['exit_code'] = 1
                elif change == 'end': data['_meta']['end'] = None
                elif change == 'sample': data['config']['sample_size'] = 20
                elif change == 'cleanup': data['metrics']['worker_runs'][0]['cleanup_errors'] = ['owner unknown']
                else: data['metrics']['worker_runs'] = []
                with self.assertRaises(ValueError): self.call(root, fixture)

    def test_full_native_failure_and_reused_process_propagate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'native mismatch'):
                self.call(root, fixture, failure=ValueError('native mismatch'))
            fixture[3][0].fresh[0].process_id = 'old'
            with self.assertRaisesRegex(ValueError, 'reused an original'):
                self.call(root, fixture)


if __name__ == '__main__':
    unittest.main()
