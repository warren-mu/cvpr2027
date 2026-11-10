"""DIAG02 handmade strategy and complete-input protocol regressions; CPU only."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from dataclasses import dataclass, replace
from enum import IntEnum
import unittest
from unittest.mock import patch

from src.tools import diag02_baselines as d
from src.tools import diag01_baselines as old
from test_diag01_baselines import environments, Policy
from test_pair_table import table, mask


class CovTests(unittest.TestCase):
    def test_cov_selects_at_reveal_without_return(self):
        rows = d.collect('obs_trigger_cov', environments(), Policy(), lambda row: None)
        self.assertEqual([r['observation_count'] for r in rows], [3, 1])
        self.assertTrue(all(r['selection_frame_anchor_pixels'] >= 250 for r in rows))
        self.assertTrue(all(r['selection_frame_both_candidates_marked'] for r in rows))
        self.assertTrue(all(r['seen_before_select'] for r in rows))
        self.assertTrue(all(r['states'][-1]['state_id'] != r['states'][0]['state_id'] for r in rows))

    def test_original_never_trigger_action_semantics_unchanged(self):
        for strategy in ('obs_never', 'obs_trigger'):
            a = d.collect(strategy, environments(), Policy(), lambda row: None)
            b = old.collect(strategy, environments(), Policy(), lambda row: None)
            for left, right in zip(a, b):
                for field in ('actions', 'observation_count', 'success', 'selected_object', 'task_reward', 'seen_before_select'):
                    self.assertEqual(left[field], right[field])

    def test_cov_falls_back_when_one_candidate_is_out_of_reach(self):
        value = table()
        for key, state in list(value.states.items()):
            if key != value.initial_state:
                captures = tuple(replace(c, distances={**c.distances, 'A': 1.5001}) for c in state.captures)
                value.states[key] = replace(state, captures=captures)
        rows = d.collect('obs_trigger_cov', environments(value), Policy(), lambda row: None)
        self.assertEqual([r['observation_count'] for r in rows], [4, 2])
        self.assertTrue(all(r['states'][-1]['state_id'] == r['states'][0]['state_id'] for r in rows))

    def test_cov_preserves_world_mapping(self):
        rows = d.collect('obs_trigger_cov', environments(swapped=True), Policy(), lambda row: None)
        self.assertEqual([r['original_world_id'] for r in rows], ['w1', 'w0'])
        self.assertEqual([r['observation_count'] for r in rows], [1, 3])

    def test_cov_uses_real_mask_bytes_not_cached_pixel_count(self):
        value = table()
        value.states = {key: replace(state, anchor_pixels=(0, 0)) for key, state in value.states.items()}
        rows = d.collect('obs_trigger_cov', environments(value), Policy(), lambda row: None)
        self.assertEqual([r['observation_count'] for r in rows], [3, 1])
        self.assertTrue(all(r['logged_x_seen_step'] is None for r in rows))

    def test_both_marks_requires_success_eligibility(self):
        env = environments()[0][2]
        env.reset('pair_fixture', 'w0')
        self.assertTrue(d.both_candidates_marked(env))
        env.frame = replace(env.frame, candidates=tuple(replace(row, distance_m=1.5001) for row in env.frame.candidates))
        self.assertFalse(d.both_candidates_marked(env))
        self.assertTrue(all(row.public_selectable for row in env.frame.candidates))


class CacheTests(unittest.TestCase):
    def prepared(self, text='hello', image='0' * 64, model='3b', choice='select 1'):
        return {'descriptor': {'model': model, 'prompt': text,
            'ordered_rgb_images': [{'sha256': image, 'width': 255, 'height': 255}],
            'sampling': {'temperature': 0, 'choice': choice}}}

    def test_single_generation_within_and_across_batches(self):
        cache = d.ExactInputCache(); calls = []
        def infer(rows):
            calls.append(copy.deepcopy(rows))
            return [{'sample_id': 'origin' + str(i), 'response': 'select 1', 'action': 'select 1',
                'response_token_ids': [1742, 1], 'finish_reason': 'stop', 'stop_reason': None} for i, _ in enumerate(rows)]
        first = cache.generate([self.prepared(), self.prepared()], infer)
        again = cache.generate([self.prepared()], infer)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 1)
        self.assertEqual(cache.generated_requests, 1)
        self.assertEqual(cache.reused_requests, 2)
        self.assertEqual(d.output_identity(first[0]), d.output_identity(first[1]))
        self.assertEqual(d.output_identity(first[0]), d.output_identity(again[0]))
        self.assertFalse(first[0]['exact_input_cache']['reused_generation'])
        self.assertTrue(first[1]['exact_input_cache']['reused_generation'])
        first[0]['response_token_ids'].append(999)
        self.assertEqual(again[0]['response_token_ids'], [1742, 1])

    def test_any_public_input_or_model_setting_difference_has_new_key(self):
        rows = [self.prepared(), self.prepared(text='other'), self.prepared(image='1' * 64),
            self.prepared(model='7b'), self.prepared(choice='select 2')]
        self.assertEqual(len({d.request_key(r['descriptor'])[0] for r in rows}), len(rows))

    def test_canonical_order_not_dict_insertion_order(self):
        self.assertEqual(d.request_key({'a': 1, 'b': [2, 3]}), d.request_key({'b': [2, 3], 'a': 1}))
        self.assertNotEqual(d.request_key({'a': [1, 2]}), d.request_key({'a': [2, 1]}))

    def test_resolved_sampling_includes_implicit_and_nested_fields(self):
        class Output(IntEnum):
            FINAL = 2
        @dataclass
        class Guide:
            choice: list
            backend: str = 'xgrammar'
        class Params:
            __struct_fields__ = ('temperature', 'guide', 'output_kind', 'seed', '_all_stop_token_ids')
            temperature = 0.
            guide = Guide(['select 1', 'select 2'])
            output_kind = Output.FINAL
            seed = None
            _all_stop_token_ids = {2, 1}
        actual = d.resolved_parameters(Params())
        self.assertEqual(actual, {'temperature': 0., 'guide': {'choice': ['select 1', 'select 2'], 'backend': 'xgrammar'},
            'output_kind': 2, 'seed': None, '_all_stop_token_ids': {'unordered_set': [1, 2]}})

    def test_cache_rejects_missing_generation(self):
        with self.assertRaisesRegex(ValueError, 'coverage'):
            d.ExactInputCache().generate([self.prepared()], lambda rows: [])

    def test_installed_vllm_sampling_defaults_serialize_without_model(self):
        import importlib.util
        if importlib.util.find_spec('vllm') is None:
            self.skipTest('vLLM package not installed in this CPU-only environment')
        from vllm.sampling_params import SamplingParams, GuidedDecodingParams
        params = SamplingParams(**d.SAMPLING_SETTINGS,
            guided_decoding=GuidedDecodingParams(choice=['select 1', 'select 2']))
        fields = d.resolved_parameters(params)
        self.assertEqual(set(fields), set(params.__struct_fields__))
        self.assertEqual(fields['guided_decoding']['choice'], ['select 1', 'select 2'])
        self.assertEqual(fields['temperature'], 0.)
        self.assertEqual(len(d.request_key(fields)[0]), 64)


class SnapshotTests(unittest.TestCase):
    def test_official_weight_sha_is_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ('a' * 40)
            path.mkdir()
            (path / 'config.json').write_text(json.dumps({'model_type': 'qwen2_5_vl'}))
            payload = b'handmade-weight-check-fixture'
            (path / 'model.safetensors').write_bytes(payload)
            ref = {'file': 'results/fixture.json', 'sha256': '0' * 64}
            data = {'_meta': {'exit_code': 0}, 'metrics': {'completed': True, 'errors': [],
                'model': d.MODELS['7b'], 'snapshot_path': str(path), 'revision': 'a' * 40,
                'model_config_sha256': hashlib.sha256((path / 'config.json').read_bytes()).hexdigest(),
                'files': [{'path': 'model.safetensors', 'bytes': len(payload),
                    'hub_lfs_sha256': hashlib.sha256(payload).hexdigest()}]}}
            cfg = {'model_size': '7b', 'model_path': str(path), 'model_source': ref,
                'identity': {'scratch': folder}}
            with patch('src.tools.t01_partition.audited_result', return_value=(data, ref)):
                good = d.diagnostic_model_reference(cfg)
                self.assertTrue(good['weight_files'][0]['official_lfs_match'])
                data['metrics']['files'][0]['hub_lfs_sha256'] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
                    d.diagnostic_model_reference(cfg)
                data['metrics']['files'][0]['hub_lfs_sha256'] = None
                with self.assertRaisesRegex(ValueError, 'Missing official SHA256'):
                    d.diagnostic_model_reference(cfg)


if __name__ == '__main__':
    unittest.main()
