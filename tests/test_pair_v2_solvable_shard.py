"""Private binding/version and rejection-retention checks, CPU mocks only."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.envs import pair_v2_solvable_shard as shard


def fixture_module():
    path = Path(__file__).with_name('test_pair_v2_solvable_policy.py')
    spec = importlib.util.spec_from_file_location('solvable_cpu_fixtures', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = fixture_module()


def proposal():
    left = [{'position': {'x': 0, 'y': 0, 'z': 0}}]
    right = [{'position': {'x': 1, 'y': 0, 'z': 0}}]
    item, _, domain = next(iter(shard.roles.view_expanded_proposals(left, right, [{}])))
    item['role_domain'] = domain
    return item


class SolvableEntryTests(unittest.TestCase):
    def test_hooks_restore_even_on_worker_failure(self):
        config = shard.enable_config({'name': 'fixture', 'scene': 'FloorPlan1', 'seed': 0})
        old = (shard.base.build_pair, shard.base.export_pair, shard.prior_policy.choose_look,
               shard.prior_policy.classify_pair, shard.stable.compact_attempt,
               shard.search.summary, shard.search.proposal_iterator)

        def worker(cfg, connection):
            self.assertIs(shard.prior_policy.choose_look, shard.policy.choose_look)
            self.assertIs(shard.prior_policy.classify_pair, shard.policy.classify_pair)
            self.assertIs(shard.search.proposal_iterator, shard.proposal_iterator)
            raise RuntimeError('synthetic failure')

        with patch.object(shard, '_SEARCH_GENERATOR', worker):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                shard.generator_worker(config, None)
        self.assertEqual(old, (shard.base.build_pair, shard.base.export_pair, shard.prior_policy.choose_look,
            shard.prior_policy.classify_pair, shard.stable.compact_attempt,
            shard.search.summary, shard.search.proposal_iterator))

    def test_no_choice_retains_specific_world_reason_and_compact_report(self):
        _, report = fixtures.choose(fixtures.Sim({(0, 'turn_left'): 250}))
        config = shard.enable_config({'name': 'fixture', 'scene': 'FloorPlan1', 'seed': 0})
        captured = {}

        def original_build(*args, attempt_row=None):
            shard.policy._LAST_REPORT = copy.deepcopy(report)
            attempt_row.update(kind='look_flip', status='rejected', discard_reasons=['one_action_evidence_unavailable'])
            return attempt_row

        def worker(cfg, connection):
            row = {}
            shard.base.build_pair(None, cfg, {}, fixtures.IDENTITIES, proposal(), {},
                                  'floorplan1_00000', None, [], attempt_row=row)
            captured['row'] = row
            captured['compact'] = shard.stable.compact_attempt(row)
            return {'status': 'completed_search', 'acceptance_passed': False}

        with (patch.object(shard.base, 'build_pair', original_build),
              patch.object(shard, '_SEARCH_GENERATOR', worker)):
            result = shard.generator_worker(config, None)
        self.assertEqual(captured['row']['discard_reasons'], ['look_not_solvable_w1'])
        self.assertEqual(captured['compact']['look_solvability']['anchor_visible_actions_w1'], [])
        self.assertEqual(captured['compact']['proposal_role']['candidate_a_name'], 'A')
        self.assertEqual(result['solvability_version'], shard.policy.SOLVABILITY_VERSION)
        self.assertIsNone(result['p_backend_solvability_verified'])

    def test_plain_direct_build_does_not_require_a_look_report(self):
        config = shard.enable_config({'name': 'fixture', 'scene': 'FloorPlan1', 'seed': 0})
        seen = {}

        def original_build(*args, attempt_row=None):
            seen['proposal'] = copy.deepcopy(args[4])
            attempt_row.update(kind='direct_flip', status='rejected', discard_reasons=['ordinary_measured_reason'])
            return attempt_row

        def worker(cfg, connection):
            row = {}
            shard.base.build_pair(None, cfg, {}, fixtures.IDENTITIES, proposal(), {},
                                  'floorplan1_00001', None, [], attempt_row=row)
            seen['row'] = row
            return {}

        with (patch.object(shard.base, 'build_pair', original_build), patch.object(shard, '_SEARCH_GENERATOR', worker)):
            shard.generator_worker(config, None)
        self.assertEqual(seen['row']['discard_reasons'], ['ordinary_measured_reason'])
        self.assertNotIn('look_solvability', seen['row'])
        self.assertEqual(seen['row']['proposal_role']['attempt_index'], 1)

    def test_exact_portable_capacity_error_is_evidence_incomplete_only(self):
        from src.envs import pair_v2_solvable_export as exporter
        arguments = (None, {}, {}, [], [], [], [], [])
        with patch.object(exporter, 'export_pair', side_effect=ValueError('Portable solvability proof exceeded its declared byte limit')):
            with self.assertRaises(shard.search.EvidenceIncomplete):
                shard.export_pair(*arguments)
        with patch.object(exporter, 'export_pair', side_effect=ValueError('actual corrupt native hash')):
            with self.assertRaisesRegex(ValueError, 'actual corrupt native hash'):
                shard.export_pair(*arguments)

    def test_final_writer_adds_versions_even_before_worker_start(self):
        captured = []

        def main():
            shard.base.write_result('fixture', {}, {})
            return 1

        old = shard.search.generator_worker
        with (patch.object(shard.search, 'main', main),
              patch.object(shard.base, 'write_result', lambda n, m, c: captured.append((m, c)))):
            self.assertEqual(shard.main(), 1)
        self.assertIs(shard.search.generator_worker, old)
        metrics, config = captured[0]
        for key in ('solvability_version', 'generation_version'):
            self.assertEqual(metrics[key], config[key])
        self.assertEqual(config['producer_module'], 'src.envs.pair_v2_solvable_shard')
        self.assertIsNone(metrics['p_backend_solvability_verified'])

    def test_attempt_identity_is_exact_and_not_inferred_from_names(self):
        self.assertEqual(shard.pair_attempt_index('floorplan203_00012', 'FloorPlan203'), 12)
        for pair, scene in (('floorplan2_00012', 'FloorPlan1'), ('floorplan1_12', 'FloorPlan1'),
                            ('floorplan1_na', 'FloorPlan1')):
            with self.assertRaises(ValueError):
                shard.pair_attempt_index(pair, scene)

    def test_new_proposal_iterator_preserves_restore_gate_and_uses_expanded_stream(self):
        objects = [{'name': name, 'position': {'x': index, 'y': 0, 'z': 0}, 'receptacle': name == 'R'}
                   for index, name in enumerate(('A', 'B', 'X', 'R'))]
        events = []

        def restore(value):
            events.append('restore')
            return {}, {'restore_deterministic': True}

        def points(name):
            self.assertEqual(events, ['restore', 'setup'])
            events.append('native')
            return []

        sim = SimpleNamespace(restore_snapshot=restore, action_count=3, spawn_points=points,
                              store=SimpleNamespace(save_json=lambda *args: events.append('setup')))
        left = [{'position': {'x': n, 'y': 0, 'z': 0}} for n in (0, .1)]
        right = [{'position': {'x': n, 'y': 0, 'z': 0}} for n in (1, 1.1)]
        config = {'max_receptacles': 1, 'max_points_per_side': 2, 'max_view_positions': 1}
        with (patch.object(shard.base, 'placement_groups', return_value=(left, right)),
              patch.object(shard.base, 'view_proposals', return_value=[{'v': 0}, {'v': 1}]),
              patch.object(shard, '_ROLE_DOMAINS', {})):
            rows = list(shard.proposal_iterator(sim, {'objects': objects}, fixtures.IDENTITIES, [], config))
            self.assertTrue(shard._ROLE_DOMAINS)
        stay_sides = [row['role_domain']['stay_side'] for row, view in rows if row['mode'] == 'stay']
        self.assertEqual(stay_sides, ['A', 'B', 'A', 'B'])
        self.assertEqual(events, ['restore', 'setup', 'native'])


if __name__ == '__main__':
    unittest.main()
