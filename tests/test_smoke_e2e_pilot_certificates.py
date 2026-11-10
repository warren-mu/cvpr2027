"""Synthetic/byte-backed contracts; these fixtures never certify real renders."""
from contextlib import ExitStack
import copy
from dataclasses import replace
import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_smoke_e2e_pilot_dataset as pool_fixtures
import test_pair_table as table_fixtures
from src.tools import smoke_e2e_pilot_certificates as cert
from src.tools import smoke_e2e_pilot_dataset as dataset
from src.tools import t02_cache_check as cache
from src.tools.t02_export import save_json
from src.envs import pair_table_balance as balance, pair_table_checks as checks
from src.envs import pair_table_export as exporter
from src.envs.pair_table_cache import CACHE_VERSION, CACHE_POLICY


def write(path, value, root=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): path.unlink()  # Only this test's disposable synthetic fixture.
    ref = save_json(path, value)
    if root is not None: ref['file'] = str(Path(ref['file']).relative_to(root))
    return ref


def fixture_audit(path):
    path = Path(path).resolve(); body = path.read_bytes(); value = json.loads(body)
    root = next(p for p in path.parents if (p / 'results').is_dir())
    return value, {'file': str(path.relative_to(root)), 'sha256': hashlib.sha256(body).hexdigest(),
        'metrics_sha256': cert.canonical_hash(value['metrics']), 'audit': {'fail': []}}


class ActivationPopulationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        source = cert.ROOT / cert.POLICY_REFERENCE['file']
        target = self.root / cert.POLICY_REFERENCE['file']; target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        self.policy = cert.policy(self.root); self.refs = self.policy['cache_sources']
        self.now = datetime.datetime(2026, 11, 10, 7, tzinfo=datetime.timezone.utc)

    def boundary(self, selected=None):
        tables, packages, normalized = pool_fixtures.pool_fixture()
        selected = dict(list(tables.items())[:83]) if selected is None else selected
        keys = list(selected); groups = [keys[:5], keys[5:29], keys[29:52], keys[52:]]
        mapping = {ref['file']: group for ref, group in zip(self.refs, groups)}
        def load(ref, **_):
            group = mapping[ref['file']]
            metadata = {field: {'file': '/fixture/' + field + str(self.refs.index(ref)), 'sha256': 'e' * 64}
                for field in ('authorization_manifest', 'validation_report', 'sampling_plan')}
            return ({k: selected[k] for k in group}, {k: packages[k] for k in group},
                dict(metadata, source=ref, dataset_keys=sorted(group)))
        stack = ExitStack()
        stack.enter_context(patch.object(dataset, 'ROOT', self.root))
        stack.enter_context(patch.object(cert, 'utc_now', return_value=self.now))
        loader = stack.enter_context(patch.object(cert, 'load_certificate_batch', side_effect=load))
        native = stack.enter_context(patch.object(dataset, 'authenticate_cache_batch', side_effect=AssertionError('No extra native scan')))
        stack.enter_context(patch.object(balance, 'normalize_table', side_effect=lambda table: normalized[table.dataset_key]))
        return stack, loader, native

    def test_exact_deadline_and_missing_extra_result_are_explicit(self):
        with patch.object(cert, 'utc_now', return_value=self.now - datetime.timedelta(microseconds=1)):
            with self.assertRaisesRegex(ValueError, 'before2026|before 2026'): cert.activate(self.refs, root=self.root)
        with patch.object(cert, 'utc_now', return_value=self.now):
            activation = cert.activate(self.refs, root=self.root)
        self.assertFalse(activation['extra_qualification_snapshot']['exists'])
        self.assertEqual(cert.qualification(activation, self.refs, root=self.root)['extra_source_reauthentication'],
            {'status': 'not_awaited', 'passed': None})

    def test_extra_qualification_failed_or_unfinished_is_not_upgraded_but_pass_blocks_new_mode(self):
        path = self.root / self.policy['extra_qualification_result']
        for passed, end, code in ((False, None, None), (False, 'ended', 1), (True, 'ended', 0)):
            write(path, {'metrics': {'passed': passed}, '_meta': {'end': end, 'exit_code': code}})
            with self.subTest(passed=passed, code=code), patch.object(cert, 'utc_now', return_value=self.now):
                if passed:
                    with self.assertRaisesRegex(ValueError, 'already reports PASS'): cert.activate(self.refs, root=self.root)
                else:
                    activation = cert.activate(self.refs, root=self.root)
                    self.assertEqual(activation['extra_qualification_snapshot']['observed_exit_code'], code)
                    self.assertIsNone(cert.qualification(activation, self.refs, root=self.root)['extra_source_reauthentication']['passed'])

    def test_sources_cannot_be_reordered_replaced_or_extended(self):
        changed = copy.deepcopy(self.refs); changed[-1]['sha256'] = 'f' * 64
        for refs in (changed, self.refs[::-1], self.refs[:3], self.refs + [self.refs[0]]):
            with self.subTest(refs=refs), patch.object(cert, 'utc_now', return_value=self.now), self.assertRaises(ValueError):
                cert.activate(refs, root=self.root)

    def test_all83_cached_pairs_remain_and_original_holdout_and_flip_order_are_used(self):
        stack, loader, native = self.boundary()
        with stack:
            manifest, tables, packages, proof = dataset.build_pool(self.refs, dataset.SCENE_SPLIT, certificates_after02=True)
        self.assertEqual(len(tables), 83); self.assertEqual(set(tables), set(packages))
        self.assertEqual(loader.call_count, 4); native.assert_not_called()
        self.assertEqual(manifest['balance']['dropped'], [])
        self.assertEqual(manifest['split'], dataset.split_entries(manifest['entries'], dataset.SCENE_SPLIT))
        self.assertEqual(proof['qualification'], manifest['qualification'])
        self.assertFalse(proof['qualification']['native_source_reauthentication_performed_here'])

    def test_missing_pair_or_class_cannot_become_a_certificate_population(self):
        tables, _, _ = pool_fixtures.pool_fixture()
        populations = [dict(list(tables.items())[:82]), dict([(k, t) for k, t in tables.items() if t.kind != 'look_stay'][:83])]
        for selected in populations:
            stack, loader, native = self.boundary(selected)
            with self.subTest(size=len(selected)), stack, self.assertRaisesRegex(ValueError, 'every original83'):
                dataset.build_pool(self.refs, dataset.SCENE_SPLIT, certificates_after02=True)
            self.assertEqual(loader.call_count, 4); native.assert_not_called()

    def successful_fixture(self):
        stack, loader, native = self.boundary()
        with stack:
            manifest, tables, packages, proof = dataset.build_pool(self.refs, dataset.SCENE_SPLIT, certificates_after02=True)
        manifest_ref = write(self.root / cert.MANIFEST_FILE, manifest)  # Actual producer writes an ABSOLUTE ref.
        argv = ['python3', '-m', 'src.tools.smoke_e2e_pilot_dataset', '--name', cert.NAME]
        for ref in self.refs: argv += ['--cache-source', json.dumps(ref)]
        argv += ['--split-policy', dataset.SCENE_SPLIT, '--certificates-after-02', '--out', cert.MANIFEST_FILE, '--timeout-seconds', '600']
        data = {'_meta': {'command': argv, 'name': cert.NAME, 'end': 'ended', 'exit_code': 0},
            'config': {'name': cert.NAME, 'version': dataset.VERSION, 'certificates_after_02': True,
                'deadline_fallback_52': False, 'cache_source': self.refs, 'out': cert.MANIFEST_FILE, 'split_policy': dataset.SCENE_SPLIT},
            'metrics': {'kind': 'smoke_e2e_pilot_dataset_preflight', 'passed': True, 'producer_exit_code': 0,
                'errors': [], 'manifest': manifest_ref, 'dataset_proof': proof}, 'sources': proof['sources']}
        ref = write(self.root / cert.RESULT_FILE, data, self.root)
        def metadata(expected, **_):
            batch = next(p for p in proof['formal_cache_batches'] if p['source'] == expected)
            refs = {entry['dataset_key']: entry['table_manifest'] for entry in manifest['entries'] if entry['dataset_key'] in batch['dataset_keys']}
            return {'certificate': {'tables': refs}, 'data': {'metrics': batch}}
        return data, ref, manifest, manifest_ref, metadata

    def test_actual_absolute_manifest_ref_and_freeze_use_successful_certificate_receipt(self):
        from src.tools import train_holdout_freeze as holdout
        data, ref, manifest, manifest_ref, metadata = self.successful_fixture()
        with patch.object(cert, 'audited_result', side_effect=fixture_audit), patch.object(cert, 'certificate_metadata', side_effect=metadata):
            actual = cert.authenticate_certificate_qualification(ref, root=self.root)
            self.assertEqual(actual[0], manifest); self.assertEqual(actual[3], {k: manifest_ref[k] for k in ('file', 'sha256')})
            frozen, _, _ = holdout.freeze(ref, root=self.root, certificates_after_02=True)
            self.assertEqual(frozen['selected_scenes'], manifest['split']['selected_scenes'])
            self.assertEqual(frozen['population_authority'], manifest['qualification'])

    def test_failed_or_unfinished_population_receipt_and_changed_holdout_are_rejected(self):
        data, ref, _, _, metadata = self.successful_fixture()
        for label in ('not_ended', 'failed', 'changed_split'):
            bad = copy.deepcopy(data)
            if label == 'not_ended': bad['_meta']['end'] = None
            elif label == 'failed': bad['metrics']['passed'] = False
            else: bad['metrics']['dataset_proof']['split']['selected_scenes'] = []
            changed = write(self.root / cert.RESULT_FILE, bad, self.root)
            with self.subTest(label=label), patch.object(cert, 'audited_result', side_effect=fixture_audit), \
                    patch.object(cert, 'certificate_metadata', side_effect=metadata), self.assertRaises(ValueError):
                cert.authenticate_certificate_qualification(changed, root=self.root)

    def test_reload_keeps_original_mode_after_extra_run_later_passes_and_rechecks_original_manifest(self):
        data, ref, manifest, manifest_ref, metadata = self.successful_fixture()
        write(self.root / self.policy['extra_qualification_result'], {'metrics': {'passed': True}, '_meta': {'end': 'later', 'exit_code': 0}})
        stack, loader, native = self.boundary()
        with stack, patch.object(cert, 'audited_result', side_effect=fixture_audit), \
                patch.object(cert, 'certificate_metadata', side_effect=metadata), \
                patch.object(cert, 'activate', side_effect=AssertionError('Do not redraw activation after freeze')):
            _, _, proof = dataset.load_pilot_pool(manifest_ref)
        self.assertEqual(proof['qualification'], manifest['qualification']); native.assert_not_called()
        self.assertEqual(loader.call_count, 4)


class MetadataCertificateTests(unittest.TestCase):
    def fixture(self, root):
        name = 'cache_fixture'; job = '123.hopper-m-02'; scratch = root / 'scratch'
        directory = scratch / 't02_cache' / name / job
        export_ref = {'file': 'results/export.json', 'sha256': 'a' * 64}
        table_ref = {'file': str(scratch / 'table.json'), 'sha256': 'b' * 64, 'bytes': 20}
        states = [{'dataset_key': 'pair_fixture', 'state_id': f's{i}', 'cache_matches_live': True,
            'worlds': {world: {'fresh_source': {'render_id': f'{i}:{world}'}} for world in ('w0', 'w1')}} for i in range(200)]
        plan = {'seed': 0, 'sample_size': 200, 'population_sha256': 'c' * 64,
            'selected_keys': [['pair_fixture', f's{i}'] for i in range(200)]}
        shadow = {'pair_fixture': {'shadow_divergent_states': 0, 'non_anchor_divergent_states': 0, 'in_clean_subset': True}}
        report = {'version': CACHE_VERSION, 'policy': CACHE_POLICY, 'sample_complete': True, 'checks_passed': True,
            'observed_distinct_states': 200, 'states': states, 'cache_matches_live': True, 'downstream_allowed': True,
            'stop_downstream': False, 'synthetic_fixture': False, 'table_native_and_pair_checks_authorized': True,
            'producer_errors': [], 'producer_cleanup_errors': [], 'sampling_plan': plan, 'table_shadow_metadata': shadow,
            'native_authenticated_render_ids': [[row['dataset_key'], row['state_id'], world,
                row['worlds'][world]['fresh_source']['render_id']] for row in states for world in ('w0', 'w1')]}
        plan_ref = write(directory / 'sampling_plan.json', plan)
        report_ref = write(directory / 'cache_validation.json', report)
        certificate = {'version': cache.VERSION, 'tables': {'pair_fixture': table_ref}, 'table_shadow_metadata': shadow,
            'cache_validation': report_ref, 'sampling_plan': plan_ref, 'full_T02_acceptance': None}
        certificate_ref = write(directory / 'cache_authorization.json', certificate)
        gate_ref = write(root / 'results/gate.json', {'metrics': {}, '_meta': {}}, root)
        data = {'config': {'version': cache.VERSION, 'gpus': 1, 'resolution': 255, 'seed': 0, 'sample_size': 200,
                'name': name, 'source': [export_ref['file']], 'pbs_jobid': job, 'node': 'fixture', 'scratch': str(scratch),
                'scratch_directory': str(directory), 'population_sha256': plan['population_sha256']},
            '_meta': {'name': name, 'end': 'ended', 'exit_code': 0,
                'command': ['python3', '-m', 'src.tools.t02_cache_check', '--name', name, '--source', export_ref['file']]},
            'metrics': {'kind': 't02_formal_cache_check', 'producer_exit_code': 0, 'errors': [], 'cleanup_errors': [],
                'black_frame_detected': False, 'sample_complete': True, 'cache_matches_live': True,
                'downstream_allowed': True, 'stop_downstream': False, 'gate': gate_ref,
                'validation_report': report_ref, 'sampling_plan': plan_ref, 'authorization_manifest': certificate_ref,
                'table_shadow_metadata': shadow, 'population_ledger': [{'dataset_key': 'pair_fixture', 'pair_id': 'fixture',
                    'source': export_ref, 'population_membership': 'complete_independently_verified_table',
                    'table_manifest': dict(table_ref, payload_bytes=123)}],
                # The successful fixed31 source also retains these original
                # initialization-retry diagnostics; they do not alter authority.
                'worker_runs': [{'controller_initialized': True, 'initialization_attempt': 2,
                    'initialization_attempts': [{'status': 'timeout'}, {'status': 'initialized'}],
                    'initialization_retry_policy': {'fixture': True}, 'initialize_timeout': 300}]}}
        return data, write(root / 'results/cache_fixture.json', data, root)

    def test_original_metadata_refs_and_nonredundant_table_ref_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); data, ref = self.fixture(root)
            with patch.object(cert, 'audited_result', side_effect=fixture_audit), patch.object(cache, 'gate_runtime'):
                proof = cert.certificate_metadata(ref, root=root)
                self.assertEqual(len(proof['authenticated_ids']), 400)
                self.assertEqual(set(proof['certificate']['tables']['pair_fixture']), {'file', 'sha256', 'bytes'})
                Path(data['metrics']['authorization_manifest']['file']).write_text('{}')
                with self.assertRaisesRegex(ValueError, 'proof changed'): cert.certificate_metadata(ref, root=root)

    def test_pending_failed_incomplete_or_changed_certificates_fail_closed(self):
        for change in ('no_end', 'exit1', 'partial', 'cleanup', 'missing_certificate', 'missing_render_id'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve(); data, ref = self.fixture(root)
                if change == 'no_end': data['_meta']['end'] = None
                elif change == 'exit1': data['_meta']['exit_code'] = 1
                elif change == 'partial': data['metrics']['sample_complete'] = False
                elif change == 'cleanup': data['metrics']['cleanup_errors'] = ['unknown']
                elif change == 'missing_certificate': Path(data['metrics']['authorization_manifest']['file']).unlink()
                else:
                    path = Path(data['metrics']['validation_report']['file']); report = json.loads(path.read_text())
                    report['native_authenticated_render_ids'].pop(); write(path, report)
                ref = write(root / 'results/cache_fixture.json', data, root)
                with patch.object(cert, 'audited_result', side_effect=fixture_audit), patch.object(cache, 'gate_runtime'), self.assertRaises((ValueError, FileNotFoundError)):
                    cert.certificate_metadata(ref, root=root)


class CachedPayloadTests(unittest.TestCase):
    def test_real_cached_payload_sha_and_pair_check_without_any_native_source_reread(self):
        class Renderer(table_fixtures.FixtureRenderer):
            def render_pair(self, state):
                return tuple(replace(view, synthetic=False, source={'fixture': True}) for view in super().render_pair(state))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); name = 't02_export_fixture'; job = '123.hopper-m-02'; scratch = root / 'scratch'
            pair_dir = scratch / 't02_export' / name / job / 'pair_0000'
            provenance = {'container': {'file': 'results/original.json', 'sha256': 'a' * 64},
                'pair': {'file': 'data/world_pairs/v2/pair.json', 'sha256': 'b' * 64}}
            value = exporter.build_table(Renderer(), pair_id='pair_fixture', scene='FloorPlan1', kind='look_flip',
                instruction='fixture only', anchor_name='X', candidate_names=('A', 'B'), target_names=('A', 'B'),
                initial_position={'x': 0, 'y': 0, 'z': 0}, initial_heading=0,
                source={'t01': provenance, 'field_of_view': 60, 'export_version': cache.EXPORT_VERSION,
                    'frozen_render_version': exporter.FROZEN_RENDER_VERSION})
            table_ref = exporter.save_table(value, pair_dir / 'table')
            key = value.dataset_key; cert_ref = {k: table_ref[k] for k in ('file', 'sha256', 'bytes')}
            t01 = {'pair_id': value.pair_id, 'selected_action': 'turn_right',
                'anchor_visible_actions_w0': ['turn_right'], 'anchor_visible_actions_w1': ['turn_left'],
                'witnesses': [{'actions': [{'kind': 'select', 'target_name': name}]} for name in ('A', 'B')]}
            value.export_verified = True; checked = checks.check_pair(value, t01); self.assertTrue(checked['passed'])
            package = {k: getattr(value, k) for k in ('pair_id', 'scene', 'kind', 'instruction', 'anchor_name')}
            package.update(candidate_names=['A', 'B'], target_names=['A', 'B'], t01=t01,
                source_provenance=provenance, source_config={'field_of_view': 60},
                independent_verification={'validity': {'acceptance_passed': True}, 'proposal_identity': 'd' * 64})
            row = {'pair_id': value.pair_id, 'status': 'exported', 'table_complete': True, 'native_evidence_verified': True,
                'pair_checks_passed': True, 'cleanup_errors': [], 'scratch_directory': str(pair_dir),
                'source_package': write(pair_dir / 'authenticated_source.json', package), 'table_manifest': table_ref,
                'detail': write(pair_dir / 'checks.json', dict(value.shadow_metadata, table_manifest=table_ref,
                    native_checks=dict(value.shadow_metadata, passed=True), pair_checks=checked)), **value.shadow_metadata}
            source_data = {'config': {'name': name, 'version': cache.EXPORT_VERSION, 'gpus': 1, 'depth': 4,
                'pair_id': [value.pair_id], 'scratch': str(scratch), 'pbs_jobid': job, 'scratch_directory': str(pair_dir.parent)},
                '_meta': {'end': 'ended', 'exit_code': 0, 'command': ['python3', '-m', 'src.tools.t02_export', '--name', name]},
                'metrics': {'kind': 't02_table_export', 'producer_exit_code': 0, 'export_checks_complete': True,
                    'errors': [], 'cleanup_errors': [], 'unstarted_pair_ids': [], 'black_frame_detected': False, 'pairs': [row]}}
            source_ref = write(root / 'results/export.json', source_data, root)
            metadata = {'source': {'file': 'results/cache.json', 'sha256': 'c' * 64},
                'complete': [{'source': source_ref, 'dataset_key': key, 'pair_id': value.pair_id, 'table_manifest': table_ref}],
                'certificate': {'tables': {key: cert_ref}, 'table_shadow_metadata': {key: value.shadow_metadata}},
                'report': {}, 'authenticated_ids': set(), 'data': {'metrics': {field: {'file': field, 'sha256': 'e' * 64}
                    for field in ('authorization_manifest', 'validation_report', 'sampling_plan')}}}
            def original_authority(tables, **kwargs):
                for table in tables: table.cache_verified = True
            with patch.object(cert, 'certificate_metadata', return_value=metadata), patch.object(cert, 'audited_result', side_effect=fixture_audit), \
                    patch.object(checks, 'authorize_tables', side_effect=original_authority) as authority, \
                    patch.object(checks, 'verify_render_sources', side_effect=AssertionError('No repeated native scan')), \
                    patch.object(checks, 'authenticate_cache_sources', side_effect=AssertionError('No repeated fresh scan')):
                tables, _, proof = cert.load_certificate_batch(metadata['source'], root=root)
                self.assertTrue(tables[key].cache_verified); self.assertEqual(authority.call_count, 1)
                self.assertEqual(proof['extra_source_reauthentication'], {'status': 'not_awaited', 'passed': None})
                manifest = json.loads(Path(table_ref['file']).read_text())
                payload = Path(table_ref['file']).parent / manifest['files'][0]['file']; payload.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'payload changed'): cert.load_certificate_batch(metadata['source'], root=root)


if __name__ == '__main__': unittest.main()
