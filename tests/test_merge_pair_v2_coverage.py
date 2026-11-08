"""Synthetic source-binding regressions; no simulator or scientific verdicts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools import merge_pair_v2 as merge
from src.envs.pair_v2_stable import PREFLIGHT_VERSION, PREFLIGHT_POLICY
from src.envs.pair_v2_search_policy import SEARCH_VERSION, SEARCH_POLICY
from src.envs.pair_v2_solvable_policy import SOLVABILITY_VERSION, SOLVABILITY_POLICY
from src.envs.pair_v2_role_policy import ROLE_VERSION, ROLE_POLICY
from src.envs.pair_v2_isolated_shard import ATTEMPT_SETUP_VERSION, ATTEMPT_SETUP_POLICY
from src.envs.pair_v2_coverage_order import ORDER_VERSION, ORDER_POLICY

MODULE = 'src.envs.pair_v2_isolated_shard'
ORDER_KEYS = ('proposal_order_version', 'proposal_order_policy')


def config(ordered=False):
    # Do not call the current producer's enable_config: historical input must
    # remain independent of a future producer's default activation.
    result = {'enable_candidate_clone': True, 'preflight_version': PREFLIGHT_VERSION,
        'preflight_policy': copy.deepcopy(PREFLIGHT_POLICY), 'evidence_version': 'anchor_union_look_v1',
        'search_version': SEARCH_VERSION, 'search_policy': copy.deepcopy(SEARCH_POLICY),
        'solvability_version': SOLVABILITY_VERSION, 'solvability_policy': copy.deepcopy(SOLVABILITY_POLICY),
        'generation_version': ROLE_VERSION, 'generation_policy': copy.deepcopy(ROLE_POLICY),
        'attempt_setup_version': ATTEMPT_SETUP_VERSION, 'attempt_setup_policy': copy.deepcopy(ATTEMPT_SETUP_POLICY),
        'producer_module': MODULE}
    if ordered:
        result.update(proposal_order_version=ORDER_VERSION, proposal_order_policy=copy.deepcopy(ORDER_POLICY))
    return result


def source_fixture(root, cfg):
    name, gate_name = 'coverage_fixture', 'gate_fixture'
    reference = {'file': f'results/{name}.json', 'sha256': 'a' * 64, 'metrics_sha256': 'b' * 64,
                 'exit_code': 0, 'git_commit': 'c' * 40, 'audit': {'fail': [], 'warn': []}}
    gate_ref = {**copy.deepcopy(reference), 'file': f'results/{gate_name}.json'}
    shard = {'gate_result': gate_ref['file'], 'gate_source': copy.deepcopy(gate_ref)}
    cfg = {**copy.deepcopy(cfg), 'name': name, 'scene': 'FloorPlan1', 'seed': 0, 'resolution': 255,
           'metrics_spec_sha256': 'd' * 64, 'shard': shard}
    arguments = ['--name', name, '--scene', cfg['scene'], '--shard-config', 'fixture_shard.json']
    metrics = {key: copy.deepcopy(value) for key, value in cfg.items()
               if key.endswith('_version') or key.endswith('_policy') or key == 'enable_candidate_clone'}
    metrics['kind'] = 'pair_v2_shard'
    data = {'_meta': {'name': name, 'command': ['python3', '-u', '-m', MODULE, *arguments],
        'end': 'synthetic-ended', 'git_commit': reference['git_commit'], 'log_path': f'logs/runs/{name}.log'},
        'config': cfg, 'metrics': metrics}
    gate = {'_meta': {'name': gate_name, 'command': ['python3', '-m', 'src.tools.render_gate', '--name', gate_name]},
            'config': {'name': gate_name}, 'metrics': {}}
    (root / 'logs/runs').mkdir(parents=True)
    (root / 'fixture_shard.json').write_text(json.dumps(shard))
    (root / f'logs/runs/{name}.log').write_text('synthetic source log\n')
    (root / f'logs/runs/{name}.meta.json').write_text('{}\n')
    (root / f'logs/runs/{gate_name}.log').write_text('ALLOCATION_CONFIG: {}\n')
    return data, reference, gate, gate_ref


class CoverageSourceTests(unittest.TestCase):
    def test_historical_isolated_and_exact_new_order_are_accepted(self):
        for ordered in (False, True):
            cfg = config(ordered)
            args, enabled = merge.shard_producer(['python3', '-u', '-m', MODULE, '--name', 'fixture'], cfg)
            self.assertTrue(enabled)
            self.assertEqual(args, ['--name', 'fixture'])

    def test_partial_unknown_or_wrong_module_order_claim_is_rejected(self):
        mutations = []
        for key in ORDER_KEYS:
            missing = config(True); missing.pop(key); mutations.append(missing)
            unknown = config(True); unknown[key] = None; mutations.append(unknown)
        wrong = config(True); wrong['proposal_order_policy']['scope'] = 'altered'; mutations.append(wrong)
        older = config(True)
        for key in ('attempt_setup_version', 'attempt_setup_policy', 'producer_module'):
            older.pop(key)
        mutations.append(older)
        for cfg in mutations:
            with self.assertRaises(ValueError):
                merge.shard_producer(['python3', '-m', MODULE], cfg)
        with self.assertRaises(ValueError):
            merge.shard_producer(['python3', '-m', 'src.envs.pair_v2_solvable_shard'], config(True))

    def load(self, root, records):
        data, reference, gate, gate_ref = records
        # Audit and runtime identity have independent fixtures; keep this one
        # focused on the real load_source config-to-hashed-metrics boundary.
        with (patch.object(merge, 'ROOT', root),
              patch.object(merge, 'audited_result', side_effect=[(data, copy.deepcopy(reference)), (gate, copy.deepcopy(gate_ref))]),
              patch.object(merge, 'specification_provenance', return_value={'synthetic': True}),
              patch.object(merge, 'verify_source_identity')):
            return merge.load_source(Path(reference['file']))

    def test_actual_load_source_retains_old_and_exact_new_metrics(self):
        for ordered in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                records = source_fixture(root, config(ordered))
                loaded, reference = self.load(root, records)
                self.assertEqual(loaded, records[0])
                self.assertEqual(reference['gate_source']['file'], 'results/gate_fixture.json')
                self.assertEqual('proposal_order_version' in loaded['metrics'], ordered)

    def test_missing_or_different_hashed_order_metadata_fails(self):
        for key in ORDER_KEYS:
            for action in ('missing', 'none', 'mismatch'):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    records = source_fixture(root, config(True))
                    metrics = records[0]['metrics']
                    if action == 'missing':
                        metrics.pop(key)
                    else:
                        metrics[key] = None if action == 'none' else 'changed'
                    with self.assertRaisesRegex(ValueError, 'Hashed metrics coverage order'):
                        self.load(root, records)

    def test_metrics_only_order_key_even_null_cannot_relabel_an_old_source(self):
        for key in ORDER_KEYS:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                records = source_fixture(root, config(False))
                records[0]['metrics'][key] = None
                with self.assertRaisesRegex(ValueError, 'Hashed metrics coverage order'):
                    self.load(root, records)


if __name__ == '__main__':
    unittest.main()
