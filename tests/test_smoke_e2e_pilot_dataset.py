"""Synthetic CPU boundaries only; no native/cache certification is manufactured."""
from contextlib import ExitStack
import copy
import datetime
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
    def build(self, tables=None, packages=None, normalized=None):
        original, original_packages, original_normalized = pool_fixture()
        tables = original if tables is None else tables
        packages = original_packages if packages is None else packages
        normalized = original_normalized if normalized is None else normalized
        ref = {'file': 'results/cache.json', 'sha256': 'a' * 64}
        proof = {'source': ref}
        with patch.object(dataset, 'authenticate_cache_batch', return_value=(tables, packages, proof)), \
             patch.object(balance, 'normalize_table', side_effect=lambda table: normalized[table.dataset_key]):
            return dataset.build_pool([ref], dataset.SCENE_SPLIT)

    def test_keep_all_and_scene_holdout_preserve_world_orders_and_no_overlap(self):
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

    def test_single_sided_stay_and_global_skew_do_not_remove_any_qualified_pair(self):
        all_tables, all_packages, normalized = pool_fixture()
        tables = {key: table for key, table in all_tables.items() if table.pair_id.endswith('_0')}
        packages = {key: all_packages[key] for key in tables}
        before = copy.deepcopy(normalized)
        original_formal = balance.balance_pairs([normalized[key] for key in tables], seed=0)
        self.assertEqual(len(original_formal['selected']), 40)  # Formal T04 rule remains unchanged.
        manifest, selected, _, proof = self.build(tables, packages, normalized)
        self.assertEqual(set(selected), set(tables))
        self.assertEqual(len(selected), 80)
        self.assertEqual(manifest['counts_by_kind'], {kind: 20 for kind in dataset.KINDS})
        self.assertFalse(manifest['balance']['after']['global_balance_within_0_4_to_0_6'])
        self.assertFalse(manifest['balance']['policy']['global_balance_is_qualification_gate'])
        self.assertFalse(manifest['balance']['policy']['stay_downsampling_applied'])
        self.assertEqual(manifest['balance']['dropped'], [])
        self.assertEqual(normalized, before)
        formal_flip = {row['dataset_key']: row for row in original_formal['selected']}
        for row in manifest['balance']['selected']:
            if row['kind'].endswith('flip'):
                self.assertEqual(row, formal_flip[row['dataset_key']])
            else:
                self.assertEqual(row['world_order'], [0, 1])
                self.assertFalse(row['swapped'])
        self.assertEqual(set(proof['train_keys']) | set(proof['eval_keys']), set(tables))

    def test_unknown_or_skewed_left_diagnostic_is_not_a_new_pilot_gate(self):
        tables, packages, normalized = pool_fixture()
        for row in normalized.values(): row['s0_target_left'] = [None, None]
        manifest, selected, _, _ = self.build(tables, packages, normalized)
        self.assertIsNone(manifest['balance']['after']['global_balance_within_0_4_to_0_6'])
        self.assertEqual(set(selected), set(tables))
        rows = list(normalized.values())
        self.assertEqual(dataset.keep_all_with_flip_order(rows),
            dataset.keep_all_with_flip_order(list(reversed(rows))))

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

    def test_size_gate_applies_to_authenticated_inventory_and_no_class_can_be_missing(self):
        tables, packages, _ = pool_fixture()
        reduced = {key: table for key, table in tables.items() if not table.kind.endswith('stay')}
        with self.assertRaisesRegex(ValueError, '80 unique cache-verified pairs and all four'):
            self.build(reduced, {key: packages[key] for key in reduced})
        reduced = dict(list(tables.items())[:79])
        with self.assertRaisesRegex(ValueError, '80 unique cache-verified pairs and all four'):
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
                legacy = copy.deepcopy(manifest); legacy['version'] = 'smoke_e2e_pilot_formal_dataset_v1'
                path.write_text(json.dumps(legacy)); oldref = {**ref, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                with self.assertRaisesRegex(ValueError, 'Unknown pilot dataset schema'):
                    dataset.load_pilot_pool(oldref)


class DeadlineFallback52Tests(unittest.TestCase):
    """Policy/namespace tests; native/cache science remains an explicit mock boundary."""
    def setUp(self):
        self.policy_bytes=(dataset.ROOT/dataset.FALLBACK52_POLICY['file']).read_bytes()
        self.refs=copy.deepcopy(json.loads(self.policy_bytes)['cache_sources'])
        self.all_tables,self.all_packages,self.normalized=pool_fixture()
        self.selected=dict(list(self.all_tables.items())[:52])
        self.deadline=datetime.datetime(2026,11,10,3,30,tzinfo=datetime.timezone.utc)

    def boundaries(self, *, selected=None, now=None):
        selected=self.selected if selected is None else selected
        keys=list(selected)
        groups=(keys[:5],keys[5:29],keys[29:])
        batches={ref['file']:group for ref,group in zip(self.refs,groups)}
        def authenticate(ref):
            group=batches[ref['file']]
            return ({key:selected[key] for key in group},
                {key:self.all_packages[key] for key in group},{'source':copy.deepcopy(ref)})
        stack=ExitStack()
        stack.enter_context(patch.object(dataset,'utc_now',return_value=self.deadline if now is None else now))
        authenticated=stack.enter_context(patch.object(dataset,'authenticate_cache_batch',side_effect=authenticate))
        stack.enter_context(patch.object(balance,'normalize_table',side_effect=lambda table:self.normalized[table.dataset_key]))
        return stack,authenticated

    def test_early_selection_rejected_before_any_native_authentication(self):
        stack,authenticated=self.boundaries(now=self.deadline-datetime.timedelta(microseconds=1))
        with stack,self.assertRaisesRegex(ValueError,'before 2026-11-10'):
            dataset.build_pool(self.refs,dataset.SCENE_SPLIT,deadline_fallback52=True)
        authenticated.assert_not_called()

    def test_exact52_all_retained_with_original_flip_and_scene_split(self):
        stack,authenticated=self.boundaries()
        with stack:
            manifest,tables,packages,proof=dataset.build_pool(self.refs,dataset.SCENE_SPLIT,deadline_fallback52=True)
        self.assertEqual(authenticated.call_args_list,[unittest.mock.call(ref) for ref in self.refs])
        self.assertEqual(set(tables),set(self.selected));self.assertEqual(set(packages),set(self.selected))
        self.assertEqual(len(manifest['entries']),52)
        self.assertTrue(all(manifest['counts_by_kind'].values()))
        self.assertEqual(manifest['balance']['dropped'],[])
        self.assertEqual(proof['qualification'],manifest['qualification'])
        self.assertEqual(manifest['qualification']['mode'],dataset.FALLBACK52)
        self.assertEqual(manifest['qualification']['policy_reference'],dataset.FALLBACK52_POLICY)
        expected=dataset.keep_all_with_flip_order([self.normalized[key] for key in tables])
        self.assertEqual(manifest['balance'],expected)
        held=set(manifest['split']['selected_scenes'])
        self.assertTrue(all(tables[key].scene in held for key in proof['eval_keys']))
        self.assertTrue(all(tables[key].scene not in held for key in proof['train_keys']))
        unhashed=copy.deepcopy(manifest);unhashed.pop('common_pool_hash')
        self.assertEqual(manifest['common_pool_hash'],dataset.canonical_hash(unhashed))

    def test_51_and_missing_class_rejected_after_complete_authentication(self):
        populations=[dict(list(self.selected.items())[:51]),
                     dict([(key,value) for key,value in self.all_tables.items() if value.kind!='look_stay'][:52])]
        for selected in populations:
            stack,authenticated=self.boundaries(selected=selected)
            with self.subTest(count=len(selected)),stack,self.assertRaisesRegex(ValueError,'exactly52.*all four'):
                dataset.build_pool(self.refs,dataset.SCENE_SPLIT,deadline_fallback52=True)
            self.assertEqual(authenticated.call_count,3)

    def test_source_replacement_or_later_cache_addition_rejected_before_authentication(self):
        changed=copy.deepcopy(self.refs);changed[2]['sha256']='f'*64
        added=[*self.refs,{'file':'results/later31.json','sha256':'f'*64}]
        for refs in (changed,added,self.refs[:2],list(reversed(self.refs))):
            stack,authenticated=self.boundaries()
            with self.subTest(refs=refs),stack,self.assertRaisesRegex(ValueError,'exact original three cache sources'):
                dataset.build_pool(refs,dataset.SCENE_SPLIT,deadline_fallback52=True)
            authenticated.assert_not_called()

    def test_default80_gate_has_no_exception_and_no_qualification_fields(self):
        stack,authenticated=self.boundaries()
        with stack,self.assertRaisesRegex(ValueError,'at least 80'):
            dataset.build_pool(self.refs,dataset.SCENE_SPLIT)
        self.assertEqual(authenticated.call_count,3)
        selected=dict(list(self.all_tables.items())[:80])
        stack,_=self.boundaries(selected=selected,now=self.deadline-datetime.timedelta(days=1))
        with stack:
            manifest,_,_,proof=dataset.build_pool(self.refs,dataset.SCENE_SPLIT)
        self.assertNotIn('qualification',manifest);self.assertNotIn('qualification',proof)
        self.assertEqual(len(manifest['entries']),80)

    def test_frozen_fallback_reload_reauthenticates_exact_sources_and_policy(self):
        stack,_=self.boundaries()
        with stack:
            manifest=dataset.build_pool(self.refs,dataset.SCENE_SPLIT,deadline_fallback52=True)[0]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);policy_path=root/dataset.FALLBACK52_POLICY['file']
            policy_path.parent.mkdir(parents=True);policy_path.write_bytes(self.policy_bytes)
            path=root/'pool.json'
            def write(value):
                path.write_text(json.dumps(value));return {'file':'pool.json','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            ref=write(manifest)
            stack,authenticated=self.boundaries(now=self.deadline+datetime.timedelta(days=2))
            with stack,patch.object(dataset,'ROOT',root):
                tables,_,proof=dataset.load_pilot_pool(ref)
            self.assertEqual(len(tables),52)
            self.assertEqual(authenticated.call_args_list,[unittest.mock.call(ref) for ref in self.refs])
            self.assertEqual(proof['qualification'],manifest['qualification'])
            self.assertEqual(proof['common_pool_hash'],manifest['common_pool_hash'])
            for mutation in ('qualification','source','remove_qualification'):
                changed=copy.deepcopy(manifest)
                if mutation=='qualification':changed['qualification']['policy_reference']['sha256']='0'*64
                elif mutation=='source':changed['cache_sources'][0]['sha256']='0'*64
                else:changed.pop('qualification')
                bad=write(changed);stack,_=self.boundaries()
                with self.subTest(mutation=mutation),stack,patch.object(dataset,'ROOT',root),self.assertRaises(ValueError):
                    dataset.load_pilot_pool(bad)
            ref=write(manifest);policy_path.write_bytes(self.policy_bytes+b' ')
            stack,authenticated=self.boundaries()
            with stack,patch.object(dataset,'ROOT',root),self.assertRaisesRegex(ValueError,'artifact changed'):
                dataset.load_pilot_pool(ref)
            authenticated.assert_not_called()

    def test_reload_before_deadline_still_rejected(self):
        stack,_=self.boundaries()
        with stack:manifest=dataset.build_pool(self.refs,dataset.SCENE_SPLIT,deadline_fallback52=True)[0]
        original_reader=dataset.read_repo_ref
        def reader(ref):
            return manifest if ref.get('file')=='fixture_frozen_manifest' else original_reader(ref)
        stack,authenticated=self.boundaries(now=self.deadline-datetime.timedelta(seconds=1))
        with stack,patch.object(dataset,'read_repo_ref',side_effect=reader),self.assertRaisesRegex(ValueError,'before 2026-11-10'):
            dataset.load_pilot_pool({'file':'fixture_frozen_manifest','sha256':'a'*64})
        authenticated.assert_not_called()


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
