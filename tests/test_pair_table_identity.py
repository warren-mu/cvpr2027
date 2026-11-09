"""CPU namespace regressions; no synthetic fixture certifies native rendering."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_pair_table import table
from tests.test_pair_table_cache import sample
from src.envs.paired_env import RGBFrame, table_environment
from src.envs.pair_table import TableObservationBackend
from src.envs.pair_table_identity import identity_fields
from src.envs.pair_table_export import save_table, load_table
from src.envs.pair_table_cache import check_cached_state, sample_state_keys, validate_cache_sample
from src.envs.pair_table_checks import verify_cached_binding, authenticate_cache_sources
from src.envs.pair_table_balance import normalize_table, balance_pairs, seeded_digest


def source_table(tag, *, offset=0):
    value = table()
    value.source['t01'] = {'pair': {'file': f'data/world_pairs/{tag}/pair.json',
                                   'sha256': hashlib.sha256(tag.encode()).hexdigest()}}
    if offset:
        value.states = {key: replace(state, captures=tuple(replace(view,
            rgb=RGBFrame(bytes(pixel + offset for pixel in view.rgb.data), 255, 255))
            for view in state.captures)) for key, state in value.states.items()}
    return value


class DatasetIdentityTests(unittest.TestCase):
    def test_exact_source_identity_is_stable_and_keeps_original_label_seed(self):
        first, other = source_table('first'), source_table('other')
        self.assertEqual(first.pair_id, other.pair_id)
        self.assertNotEqual(first.dataset_key, other.dataset_key)
        self.assertEqual(first.labels, other.labels)
        original = copy.deepcopy(first.source)
        first.source['unrelated_export_run'] = 'another_run'
        self.assertEqual(first.dataset_key, source_table('first').dataset_key)
        self.assertEqual(first.source['t01'], original['t01'])
        ref = original['t01']['pair']
        self.assertNotEqual(identity_fields(first.pair_id, ref)['dataset_key'],
                            identity_fields(first.pair_id, {**ref, 'sha256': 'f' * 64})['dataset_key'])

    def test_colliding_ids_have_independent_cache_frames_and_runtime_identity(self):
        first, other = source_table('first'), source_table('other', offset=1)
        env = table_environment({t.dataset_key: t for t in (first, other)}, allow_synthetic=True)
        a = env.reset(first.dataset_key, 'w0')
        self.assertEqual(env.info['pair_id'], first.dataset_key)
        self.assertEqual(env.info['capture']['pair_id'], first.dataset_key)
        self.assertEqual(env.info['original_pair_id'], first.pair_id)
        b = env.reset(other.dataset_key, 'w0')
        self.assertNotEqual(a.image, b.image)
        self.assertEqual(len(env.backend.public_cache), 2)
        self.assertEqual(env.info['dataset_key'], other.dataset_key)
        self.assertNotIn(other.dataset_key, b.text)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            env.reset(first.pair_id, 'w0')
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            TableObservationBackend({first.pair_id: first, other.dataset_key: other}, allow_synthetic=True)

    def test_unique_old_id_alias_preserves_explicit_world_order(self):
        value = source_table('one')
        env = table_environment({value.dataset_key: value}, allow_synthetic=True,
                                world_orders={value.dataset_key: (1, 0)})
        env.reset(value.pair_id, 'w0')
        self.assertEqual(env.info['original_world_id'], 'w1')
        self.assertEqual(env.info['pair_id'], value.pair_id)
        self.assertEqual(env.info['dataset_key'], value.dataset_key)

    def test_manifest_declared_identity_and_legacy_derivation_never_rewrite_source(self):
        value = source_table('saved')
        with tempfile.TemporaryDirectory() as temporary:
            ref = save_table(value, Path(temporary) / 'table')
            path = Path(ref['file']); original = json.loads(path.read_text())
            loaded = load_table(path, expected_sha256=ref['sha256'])
            self.assertTrue(loaded.dataset_identity_serialized)
            self.assertEqual(loaded.dataset_key, value.dataset_key)
            legacy = copy.deepcopy(original); legacy.pop('dataset_identity')
            raw = json.dumps(legacy).encode(); path.write_bytes(raw)
            loaded = load_table(path, expected_sha256=hashlib.sha256(raw).hexdigest())
            self.assertFalse(loaded.dataset_identity_serialized)
            self.assertEqual(loaded.dataset_key, value.dataset_key)
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(loaded.source, value.source)
            original['dataset_identity']['dataset_key'] = source_table('forged').dataset_key
            raw = json.dumps(original).encode(); path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'dataset identity'):
                load_table(path, expected_sha256=hashlib.sha256(raw).hexdigest())

    def test_cache_population_counts_both_sources_without_relabeling_native_ids(self):
        first, other = source_table('first'), source_table('other')
        records = []
        for value in (first, other):
            for index in range(100):
                item = sample(value.pair_id, f's{index}')
                item = replace(item, dataset_key=value.dataset_key,
                    cached=tuple(replace(view, render_id=value.dataset_key + view.render_id) for view in item.cached),
                    fresh=tuple(replace(view, render_id=value.dataset_key + view.render_id) for view in item.fresh))
                records.append(item)
        checked = check_cached_state(records[0])
        self.assertEqual(checked['pair_id'], first.pair_id)
        self.assertEqual(checked['dataset_key'], first.dataset_key)
        self.assertTrue(checked['checks_passed'])
        population = [(item.population_key, item.state_id) for item in records]
        plan = sample_state_keys(population, seed=0)
        def measured(item):
            return {'pair_id': item.pair_id, 'dataset_key': item.dataset_key, 'state_id': item.state_id,
                    'checks_passed': True, 'cache_matches_live': None, 'synthetic': True, 'worlds': {}}
        with patch('src.envs.pair_table_cache.check_cached_state', side_effect=measured):
            report = validate_cache_sample(records, population=population, sampling_plan=plan)
        self.assertEqual(report['observed_distinct_states'], 200)
        self.assertTrue(report['sample_complete'])
        self.assertFalse(report['downstream_allowed'])
        forged = {**checked, 'dataset_key': other.dataset_key}
        with self.assertRaisesRegex(ValueError, 'dataset key'):
            verify_cached_binding(first, forged)
        # A source-free original-ID sample cannot resolve among colliding tables.
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            authenticate_cache_sources([first, other], [replace(records[0], dataset_key=None)], evidence_roots=[])

    def test_balance_keeps_both_identities_and_original_flip_seed(self):
        first, other = source_table('first'), source_table('other')
        rows = [normalize_table(value) for value in (first, other)]
        report = balance_pairs(rows)
        self.assertEqual({row['dataset_key'] for row in report['selected']}, {first.dataset_key, other.dataset_key})
        expected = seeded_digest(first.pair_id, 0, 'flip_world_swap')
        self.assertTrue(all(row['assignment_sha256'] == expected for row in report['selected']))
        self.assertEqual(report['after']['overall']['pair_count'], 2)
        with self.assertRaisesRegex(ValueError, 'unique pair IDs'):
            balance_pairs([rows[0], rows[0]])


if __name__ == '__main__':
    unittest.main()
