"""Technical CPU fixtures only; no synthetic fixture authorizes a real table."""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from src.envs import pair_table
from src.envs.pair_table_checks import check_pair
from src.envs.pair_table_cache import CACHE_POLICY, CACHE_VERSION
from src.envs.pair_table_export import save_table


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fig = load('fig01_cpu_export', Path(__file__).resolve().parents[1] / 'experiments' / 'fig01_export.py')
fixtures = load('fig01_original_table_fixtures',
    Path(pair_table.__file__).resolve().parents[2] / 'tests' / 'test_pair_table.py')


def t01_for(table):
    return {'pair_id': table.pair_id, 'selected_action': 'turn_right',
        'anchor_visible_actions_w0': ['turn_right'], 'anchor_visible_actions_w1': ['turn_left'],
        'witnesses': [{'actions': [{'kind': 'turn_right'}, {'kind': 'select', 'target_name': 'A'}]},
                      {'actions': [{'kind': 'turn_left'}, {'kind': 'select', 'target_name': 'B'}]}]}


def original_checks(table):
    checked = check_pair(table, t01_for(table))
    assert checked['passed']
    return checked


class FigureTests(unittest.TestCase):
    def test_original_first_visible_action_and_normalized_world_order(self):
        table = fixtures.table()
        seen, reason = fig.first_step_seen(table, table.pair_id, [1, 0], original_checks(table),
                                              allow_synthetic=True)
        self.assertIsNone(reason)
        self.assertEqual(seen['w0']['original_world_id'], 'w1')
        self.assertEqual(seen['w0']['action_prefix'], ['turn_left'])
        self.assertEqual(seen['w1']['action_prefix'], ['turn_right'])
        self.assertEqual((seen['w0']['anchor_pixels'], seen['w0']['tau'], seen['w0']['tau_seen']), (250, 1, 1))

    def test_unavailable_selection_witness_does_not_filter_visible_observation(self):
        table = fixtures.table()
        checked = original_checks(table)
        checked['witness_replay'][0] = {'applicable': False, 'passed': None, 'reason': 'original_witness_exceeds_D_plus_1'}
        seen, reason = fig.first_step_seen(table, table.pair_id, [0, 1], checked, allow_synthetic=True)
        self.assertIsNone(reason)
        self.assertEqual(seen['w0']['action_prefix'], ['turn_right'])
        self.assertEqual(seen['w0']['original_selection_witness_replay'], checked['witness_replay'][0])
        self.assertFalse(seen['w0']['selection_witness_used_as_visibility_evidence'])

    def test_invalid_original_visibility_action_fails(self):
        table = fixtures.table()
        checked = original_checks(table)
        checked['first_step_visible_actions']['w0'] = ['turn_left']
        with self.assertRaisesRegex(ValueError, 'no longer sees own X'):
            fig.first_step_seen(table, table.pair_id, [0, 1], checked, allow_synthetic=True)

    def test_no_production_synthetic_or_authority_bypass(self):
        table = fixtures.table()
        with self.assertRaisesRegex(ValueError, 'Synthetic'):
            fig.first_step_seen(table, table.pair_id, [0, 1], original_checks(table))
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
        seen, _ = fig.first_step_seen(table, table.pair_id, [1, 0], checked, allow_synthetic=True)
        candidate = {'dataset_key': table.pair_id, 'world_order': [1, 0], 'heldout_scene': False,
                     'first_seen': seen, 'ranking_min_anchor_pixels': 250}
        package = {'worlds': [{'objects': [{'name': name, 'objectType': name + '_type'} for name in ('X', 'A', 'B')]}],
                   'source_provenance': {'synthetic_test_only': True}, 't01': t01_for(table)}
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


class ScopedFigureTests(unittest.TestCase):
    """Local technical bytes only; mocked receipt authority never runs the CLI."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.table = fixtures.table()
        self.full_ref = save_table(self.table, self.root / 'table')
        self.ref = {k: self.full_ref[k] for k in ('file', 'sha256', 'bytes')}
        self.header = fig.read_reference(self.ref)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return fig.file_ref(path)

    def test_ranking_reads_only_first_step_masks_and_matches_actual_environment(self):
        for order in ([0, 1], [1, 0]):
            checked = original_checks(self.table)
            actual, _ = fig.first_step_seen(self.table, self.table.pair_id, order, checked,
                                               allow_synthetic=True)
            paths = []
            read = Path.read_bytes
            def observed(path):
                paths.append(path)
                return read(path)
            with patch.object(Path, 'read_bytes', observed):
                ranked, reason = fig.first_seen_metadata(self.header, self.ref, order, checked, [])
            self.assertIsNone(reason)
            self.assertEqual(ranked, {world: {k: v for k, v in row.items() if k not in ('tau', 'tau_seen')}
                                      for world, row in actual.items()})
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(path.suffix == '.mask' for path in paths))

    def test_ranking_rejects_mask_bytes_header_count_and_action_tampering(self):
        checked = original_checks(self.table)
        state = checked['witness_replay'][0]['trace'][0]['state_id']
        bad = copy.deepcopy(self.header)
        bad['states'][state]['anchor_pixels'][0] += 1
        with self.assertRaisesRegex(ValueError, 'pixel count'):
            fig.first_seen_metadata(bad, self.ref, [0, 1], checked, [])
        bad_check = copy.deepcopy(checked)
        bad_check['first_step_visible_actions']['w0'] = ['move 999']
        with self.assertRaisesRegex(ValueError, 'transition changed'):
            fig.first_seen_metadata(self.header, self.ref, [0, 1], bad_check, [])
        mask = self.root / 'table' / self.header['states'][state]['captures'][0]['masks']['X']['file']
        mask.write_bytes(bytes(255 * 255))
        with self.assertRaisesRegex(ValueError, 'mask bytes changed'):
            fig.first_seen_metadata(self.header, self.ref, [0, 1], checked, [])

    def test_valid_selection_without_own_seen_remains_rankable_and_distinct(self):
        t01 = t01_for(self.table)
        t01['witnesses'][0]['actions'][0]['kind'] = 'turn_left'
        t01['witnesses'][1]['actions'][0]['kind'] = 'turn_right'
        checked = check_pair(self.table, t01)
        self.assertTrue(checked['passed'])
        for world, replay in enumerate(checked['witness_replay']):
            self.assertTrue(replay['passed'])
            self.assertEqual(self.table.states[replay['trace'][0]['state_id']].anchor_pixels[world], 0)
        seen, reason = fig.first_seen_metadata(self.header, self.ref, [0, 1], checked, [])
        self.assertIsNone(reason)
        self.assertEqual(seen['w0']['action_prefix'], ['turn_right'])
        self.assertEqual(seen['w1']['action_prefix'], ['turn_left'])
        for world in (0, 1):
            self.assertEqual(seen[f'w{world}']['original_selection_witness_replay'], checked['witness_replay'][world])
            self.assertEqual(seen[f'w{world}']['prefix_source'], fig.PREFIX_SOURCE)
            self.assertFalse(seen[f'w{world}']['selection_witness_used_as_visibility_evidence'])

    def test_first_action_is_sorted_original_choice_not_max_pixel_choice(self):
        table = self.table
        initial = table.states[table.initial_state]
        for action, pixels in [('turn_left', 250), ('turn_right', 900)]:
            key = initial.transitions[action]
            state = table.states[key]
            first = replace(state.captures[0], masks={**state.captures[0].masks, 'X': fixtures.mask(pixels, 1000)})
            table.states[key] = replace(state, captures=(first, state.captures[1]),
                                       anchor_pixels=(pixels, state.anchor_pixels[1]))
        t01 = t01_for(table)
        t01['anchor_visible_actions_w0'] = ['turn_left', 'turn_right']
        checked = check_pair(table, t01)
        self.assertTrue(checked['passed'])
        self.assertEqual(checked['first_step_visible_actions']['w0'], ['turn_left', 'turn_right'])
        seen, reason = fig.first_step_seen(table, table.pair_id, [0, 1], checked, allow_synthetic=True)
        self.assertIsNone(reason)
        self.assertEqual((seen['w0']['action_prefix'], seen['w0']['anchor_pixels']), (['turn_left'], 250))
        checked['first_step_visible_actions']['w0'].reverse()
        with self.assertRaisesRegex(ValueError, 'sorted first-step visibility'):
            fig.first_step_seen(table, table.pair_id, [0, 1], checked, allow_synthetic=True)

    def header_population(self):
        # 200 metadata-only state identities exercise the unchanged original sampler.
        pair_ref = {'file': 'fixture/original-pair.json', 'sha256': 'a' * 64}
        header = {'pair_id': 'fixture', 'scene': 'FloorPlan1', 'kind': 'look_flip',
                  'source': {'t01': {'pair': pair_ref}}, 'states': {f's{i}': {} for i in range(200)}}
        key = fig.identity_fields('fixture', pair_ref)['dataset_key']
        ref = self.write(self.root / 'headers' / 'table.json', header)
        entry = {'dataset_key': key, 'pair_id': 'fixture', 'table_manifest': ref}
        entries = {key: {'original_pair_id': 'fixture', 'scene': 'FloorPlan1', 'kind': 'look_flip', 'table_manifest': ref}}
        evidence = {'complete': [entry], 'certificate': {'tables': {key: ref}},
                    'report': {'sampling_plan': fig.sample_state_keys([(key, s) for s in header['states']], seed=0)}}
        return key, header, entries, evidence

    def test_full_header_population_exact_plan_without_payload_reads(self):
        key, header, entries, evidence = self.header_population()
        accessed = []
        self.assertEqual(fig.certified_headers(evidence, entries, accessed), {key: header})
        self.assertEqual(len(accessed), 1)
        # Re-sign a changed header in this fixture: it still cannot alter the old plan.
        header['states']['s200'] = {}
        ref = self.write(self.root / 'headers' / 'table.json', header)
        entries[key]['table_manifest'] = ref
        evidence['certificate']['tables'][key] = ref
        with self.assertRaisesRegex(ValueError, 'sampled population changed'):
            fig.certified_headers(evidence, entries, [])

    def test_header_and_certificate_reference_changes_fail(self):
        key, header, entries, evidence = self.header_population()
        evidence['certificate']['tables'][key] = {**entries[key]['table_manifest'], 'sha256': 'b' * 64}
        with self.assertRaisesRegex(ValueError, 'table reference changed'):
            fig.certified_headers(evidence, entries, [])
        evidence['certificate']['tables'][key] = entries[key]['table_manifest']
        Path(entries[key]['table_manifest']['file']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            fig.certified_headers(evidence, entries, [])

    def test_final_rehash_detects_late_ranking_input_change(self):
        ref = self.write(self.root / 'ranking.json', {'before': True})
        fig.verify_accessed([ref, ref])
        Path(ref['file']).write_text('{"after":true}')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            fig.verify_accessed([ref])

    def metadata_record(self):
        # Receipt boundary is mocked below; all JSON SHA/path and table checks are real.
        from src.tools import t02_export as export
        table = fixtures.table()
        table.source = {'t01': {'pair': {'file': 'fixture/pair.json', 'sha256': 'c' * 64}},
                        'export_version': export.VERSION, 'frozen_render_version': fig.FROZEN_RENDER_VERSION,
                        'field_of_view': 60}
        table.states = {key: replace(state, captures=tuple(replace(c, synthetic=False,
            source={'snapshot_sha256': 'd' * 64}) for c in state.captures)) for key, state in table.states.items()}
        table.synthetic = False
        full = save_table(table, self.root / 'selected')
        ref = {k: full[k] for k in ('file', 'sha256', 'bytes')}
        header = fig.read_reference(ref)
        name, job = 't02_export_fixture', '123.hopper-m-02'
        directory = self.root / 't02_export' / name / job
        pair_dir = directory / 'pair_0000'
        package = {k: getattr(table, k) for k in ('pair_id', 'scene', 'kind', 'instruction', 'anchor_name')}
        package.update(candidate_names=list(table.candidate_names), target_names=list(table.target_names),
            source_provenance=table.source['t01'], source_config={'field_of_view': 60}, t01=t01_for(table),
            independent_verification={'validity': {'acceptance_passed': True}})
        details = {'table_manifest': full, 'native_checks': {'passed': True, **table.shadow_metadata},
                   'pair_checks': check_pair(table, package['t01']), **table.shadow_metadata}
        row = {'pair_id': table.pair_id, 'scratch_directory': str(pair_dir), 'status': 'exported',
               'table_complete': True, 'native_evidence_verified': True, 'pair_checks_passed': True,
               'cleanup_errors': [], 'error': None, 'table_manifest': full, **table.shadow_metadata,
               'source_package': self.write(pair_dir / 'authenticated_source.json', package),
               'detail': self.write(pair_dir / 'checks.json', details)}
        original = {'config': {'name': name, 'version': export.VERSION, 'gpus': 1, 'depth': 4,
            'pair_id': [table.pair_id], 'scratch': str(self.root), 'pbs_jobid': job,
            'scratch_directory': str(directory)}, '_meta': {'end': 'fixture', 'exit_code': 0,
            'command': ['python3', '-m', 'src.tools.t02_export', '--name', name]},
            'metrics': {'kind': 't02_table_export', 'producer_exit_code': 0, 'export_checks_complete': True,
            'errors': [], 'cleanup_errors': [], 'unstarted_pair_ids': [], 'black_frame_detected': False, 'pairs': [row]}}
        source = self.write(self.root / 'export.json', original)
        entry = {'dataset_key': table.dataset_key, 'pair_id': table.pair_id, 'source': source, 'table_manifest': full}
        state = table.states[table.initial_state]
        sampled = {'pair_id': table.pair_id, 'original_pair_id': table.pair_id, 'dataset_key': table.dataset_key,
                   'state_id': table.initial_state, 'kind': table.kind, 'cache_matches_live': True,
                   'cached_classification': pair_table.classify_views(state.captures, table.anchor_name, kind=table.kind),
                   'worlds': {}}
        for world, capture in enumerate(state.captures):
            sampled['worlds'][f'w{world}'] = {'cached_source': {'rgb_sha256': capture.rgb.sha256,
                'render_id': capture.render_id, 'snapshot_sha256': capture.source['snapshot_sha256']},
                'cached_labels': {name: table.labels[name] for name in pair_table.eligible_names(capture, table.candidate_names)},
                'masks': {name: {'cached_mask_sha256': hashlib.sha256(bits).hexdigest()} for name, bits in capture.masks.items()}}
        # Foreign rows stand in for metadata verified by certificate_metadata;
        # this fixture deliberately does not authenticate or certify real captures.
        states = [sampled] + [{'dataset_key': f'other{i}', 'cache_matches_live': True} for i in range(199)]
        report = {'version': CACHE_VERSION, 'policy': CACHE_POLICY, 'sample_complete': True, 'checks_passed': True,
                  'observed_distinct_states': 200, 'states': states, 'cache_matches_live': True,
                  'downstream_allowed': True, 'stop_downstream': False, 'synthetic_fixture': False,
                  'table_shadow_metadata': {table.dataset_key: table.shadow_metadata}}
        evidence = {'source': source, 'report': report, 'certificate': {'tables': {table.dataset_key: ref},
                    'table_shadow_metadata': {table.dataset_key: table.shadow_metadata}}}
        with patch.object(fig.certificates, '_audited', return_value=(original, source)):
            record = fig.accepted_pair_metadata(evidence, entry, header, [])
        return record, original, source

    def test_original_check_receipt_sha_failure_is_not_a_saved_pass(self):
        record, original, source = self.metadata_record()
        Path(record['row']['detail']['file']).write_text('{}')
        with patch.object(fig.certificates, '_audited', return_value=(original, source)):
            with self.assertRaisesRegex(ValueError, 'proof changed'):
                fig.accepted_pair_metadata(record['evidence'], record['entry'], record['header'], [])

    def test_selected_table_original_loader_pair_check_and_cache_binding(self):
        record, _, _ = self.metadata_record()
        table, checked = fig.validate_selected_table(record)
        self.assertTrue(checked['passed'])
        self.assertTrue(table.cache_verified)
        self.assertNotIn('new_native_pass', table.figure_only_authority)
        # A successful receipt cannot override the actual cached bytes/labels.
        record['evidence']['report']['states'][0]['worlds']['w0']['cached_labels'] = {}
        with self.assertRaisesRegex(ValueError, 'different masks or candidate labels'):
            fig.validate_selected_table(record)

    def test_selected_nonranking_payload_tamper_and_original_pair_failure(self):
        record, _, _ = self.metadata_record()
        record['package']['t01']['anchor_visible_actions_w0'] = []
        with self.assertRaisesRegex(ValueError, 'pair checks'):
            fig.validate_selected_table(record)
        record['package']['t01']['anchor_visible_actions_w0'] = ['turn_right']
        header = record['header']
        rgb = self.root / 'selected' / header['states'][header['initial_state']]['captures'][0]['rgb']['file']
        rgb.write_bytes(bytes(255 * 255 * 3))
        with self.assertRaisesRegex(ValueError, 'Table payload changed'):
            fig.validate_selected_table(record)

    def test_failed_whole_batch_certificate_cannot_authorize_selected_pair(self):
        record, _, _ = self.metadata_record()
        record['evidence']['report']['downstream_allowed'] = False
        with self.assertRaisesRegex(ValueError, 'Whole cache validation'):
            fig.validate_selected_table(record)


if __name__ == '__main__':
    unittest.main()
