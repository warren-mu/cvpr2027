"""Synthetic new-dispatch/retry boundaries; no GPU or active-job changes."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from src.tools import phase_watch as watch
from src.tools import t01v2_parallel as original
from src.tools import t01v2_search_parallel as old
from src.tools import t01v2_solvable_parallel as new

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('solvable_search_fixtures', ROOT / 'tests/test_t01v2_search_parallel.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def mark_outer(outer):
    tags = {'solvability_version': new.SOLVABILITY_VERSION, 'scene_producer_module': new.SCENE_MODULE}
    for location in ('config', 'metrics'):
        outer[location].update(tags)
    return outer


def black_fixture(child_failure=True):
    values = fixtures.SearchInfraTests().search_fixture(child_failure)
    payload, fields, outer, gate, gate_name, child, record = values
    mark_outer(outer)
    if child_failure:
        child['_meta']['command'][-1] = new.SCENE_MODULE
        for location in ('config', 'metrics'):
            child[location]['solvability_version'] = new.SOLVABILITY_VERSION
    return values


class SolvableDispatchTests(unittest.TestCase):
    def test_new_scene_replacement_keeps_old_outer_target_and_arguments(self):
        command = ['python3', '-u', '-m', 'src.envs.pair_v2_shard', '--name', 'scene',
                   '--scene', 'FloorPlan1', '--shard-config', 'config.json', '--timeout-seconds', '500']
        changed = new.solvable_scene_command('scene', command)
        self.assertEqual(changed[3], new.SCENE_MODULE)
        self.assertEqual(changed[:3] + changed[4:], command[:3] + command[4:])
        self.assertEqual(old.search_scene_command('scene', command)[3], old.SEARCH_MODULE)
        unrelated = ['python3', '-c', 'pass', new.SCENE_MODULE]
        self.assertIs(new.solvable_scene_command('scene', unrelated), unrelated)
        with self.assertRaises(ValueError):
            new.solvable_scene_command('wrong', command)

    def test_outer_hashes_dispatch_tags_and_restores_private_bindings(self):
        metrics, config = {}, {}
        def run():
            original.write_result('fixture', metrics, config)
            raise RuntimeError('expected fixture')
        with patch.object(original, 'main', side_effect=run), patch.object(original, 'launch_logged') as launch, \
                patch.object(original, 'write_result') as write:
            with self.assertRaises(RuntimeError):
                new.main()
            self.assertIs(original.launch_logged, launch)
            self.assertIs(original.write_result, write)
            self.assertEqual(metrics['solvability_version'], new.SOLVABILITY_VERSION)
            self.assertEqual(config['scene_producer_module'], new.SCENE_MODULE)
            self.assertEqual(metrics, config)
            write.assert_called_once_with('fixture', metrics, config)

    def test_effective_kind_requires_both_hashed_tags_on_shared_shell(self):
        self.assertEqual(watch.effective_kind('t01v2_search', {'config': {}, 'metrics': {}}), 't01v2_search')
        outer = mark_outer({'config': {}, 'metrics': {}})
        self.assertEqual(watch.effective_kind('t01v2_search', outer), 't01v2_solvable')
        self.assertEqual(watch.effective_kind('t01v2_solvable', outer), 't01v2_solvable')
        for location in ('config', 'metrics'):
            for key in ('solvability_version', 'scene_producer_module'):
                bad = copy.deepcopy(outer); del bad[location][key]
                with self.assertRaises(ValueError):
                    watch.effective_kind('t01v2_search', bad)
        with self.assertRaises(ValueError):
            watch.effective_kind('t01v2_solvable', {'config': {}, 'metrics': {}})
        with self.assertRaises(ValueError):
            watch.effective_kind('t01v2', outer)

    def test_exact_new_entry_uses_existing_eight_hour_guard(self):
        name = 'pilot_pairs_v2_solvable_fixture'
        command = ['bash', 'experiments/t01v2_solvable_pairs.sh', '--name', name]
        self.assertEqual(watch.entrypoint(command, name, ROOT), 't01v2_solvable')
        self.assertEqual(fixtures.fixtures.submit.budget_group('T01', name, command), 'T01v2')
        with self.assertRaises(ValueError):
            fixtures.fixtures.submit.budget_group('T01', 'pilot_pairs_unversioned', command)

    def test_new_child_black_proof_matches_outer_and_exact_producer(self):
        payload, fields, outer, gate, gate_name, child, record = black_fixture()
        result = watch.validate_black_retry(payload, fixtures.fixtures.JOB, fields, outer, gate,
            gate_name, 't01v2_search', child, record)
        self.assertEqual(result, 't01v2_solvable_scene_black_capture')
        for location in ('config', 'metrics'):
            bad = copy.deepcopy(child); bad[location]['solvability_version'] = 'wrong'
            with self.assertRaises(ValueError):
                watch.validate_black_retry(payload, fixtures.fixtures.JOB, fields, outer, gate,
                    gate_name, 't01v2_search', bad, record)
        bad = copy.deepcopy(child); bad['_meta']['command'][-1] = old.SEARCH_MODULE
        with self.assertRaises(ValueError):
            watch.validate_black_retry(payload, fixtures.fixtures.JOB, fields, outer, gate,
                gate_name, 't01v2_search', bad, record)
        untagged = copy.deepcopy(outer)
        for location in ('config', 'metrics'):
            for key in ('solvability_version', 'scene_producer_module'):
                del untagged[location][key]
        with self.assertRaises(ValueError):
            watch.validate_black_retry(payload, fixtures.fixtures.JOB, fields, untagged, gate,
                gate_name, 't01v2_search', child, record)
        self.assertFalse(watch.authorized_shard_command(child['_meta']['command'], v2=True, search=True, solvable=True))

    def test_new_gate_black_still_requires_original_per_card_capture(self):
        payload, fields, outer, gate, gate_name, _, _ = black_fixture(False)
        self.assertEqual(watch.validate_black_retry(payload, fixtures.fixtures.JOB, fields, outer,
            gate, gate_name, 't01v2_search'), 'render_gate_black_capture')
        bad = copy.deepcopy(outer); bad['metrics']['solvability_version'] = 'wrong'
        with self.assertRaises(ValueError):
            watch.validate_black_retry(payload, fixtures.fixtures.JOB, fields, bad,
                gate, gate_name, 't01v2_search')


if __name__ == '__main__':
    unittest.main()
