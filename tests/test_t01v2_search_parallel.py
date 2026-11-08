"""CPU-only new search entry, budget and genuine-black retry boundaries."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from src.tools import phase_watch as watch
from src.tools import t01v2_parallel as original
from src.tools import t01v2_search_parallel as search

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('search_infra_fixtures', ROOT / 'tests/test_t01v2_parallel.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class SearchInfraTests(unittest.TestCase):
    def test_only_exact_scene_executable_is_replaced(self):
        command = ['python3', '-u', '-m', 'src.envs.pair_v2_shard', '--name', 'scene',
                   '--scene', 'FloorPlan1', '--shard-config', 'config.json', '--timeout-seconds', '540']
        changed = search.search_scene_command('scene', command)
        self.assertEqual(changed[3], search.SEARCH_MODULE)
        self.assertEqual(changed[:3] + changed[4:], command[:3] + command[4:])
        self.assertEqual(command[3], 'src.envs.pair_v2_shard')
        unrelated = ['python3', '-c', 'pass', 'src.envs.pair_v2_shard']
        self.assertIs(search.search_scene_command('scene', unrelated), unrelated)
        with self.assertRaises(ValueError):
            search.search_scene_command('wrong', command)

    def test_private_launcher_binding_is_restored_after_error(self):
        with patch.object(original, 'launch_logged') as delegate, patch.object(original, 'main', side_effect=RuntimeError('fixture')):
            with self.assertRaises(RuntimeError):
                search.main()
            self.assertIs(original.launch_logged, delegate)

    def test_search_cannot_escape_same_eight_hour_subbudget(self):
        name = 'pilot_pairs_v2_search_fixture'
        command = ['bash', 'experiments/t01v2_search_pairs.sh', '--name', name]
        self.assertEqual(fixtures.submit.budget_group('T01', name, command), 'T01v2')
        self.assertEqual(watch.entrypoint(command, name, ROOT), 't01v2_search')
        with self.assertRaises(ValueError):
            fixtures.submit.budget_group('T01', 'pilot_pairs_unversioned', command)

    def search_fixture(self, child_failure):
        data, fields, outer, gate, gate_name, child, record = fixtures.black_fixture(child_failure)
        data['command'][1] = 'experiments/t01v2_search_pairs.sh'
        outer['_meta']['command'] = list(data['command'])
        if child_failure:
            child['_meta']['command'][-1] = search.SEARCH_MODULE
            versions = {'search_version': search.SEARCH_VERSION, 'preflight_version': 'settled_baseline_window_v1',
                        'evidence_version': 'anchor_union_look_v1'}
            child['config'].update(enable_candidate_clone=True, **versions)
            child['metrics'].update(versions)
        return data, fields, outer, gate, gate_name, child, record

    def test_search_child_requires_all_hashed_versions_and_exact_module(self):
        data, fields, outer, gate, gate_name, child, record = self.search_fixture(True)
        self.assertEqual(watch.validate_black_retry(data, fixtures.JOB, fields, outer, gate, gate_name,
            't01v2_search', child, record), 't01v2_search_scene_black_capture')
        for location, key in (('metrics', 'search_version'), ('config', 'search_version'),
                              ('metrics', 'preflight_version'), ('config', 'evidence_version')):
            bad = copy.deepcopy(child);bad[location][key] = 'wrong'
            with self.assertRaises(ValueError):
                watch.validate_black_retry(data, fixtures.JOB, fields, outer, gate, gate_name,
                    't01v2_search', bad, record)
        for module in ('src.envs.pair_v2_stable_shard', 'src.envs.pair_v2_clone_shard'):
            bad = copy.deepcopy(child);bad['_meta']['command'][-1] = module
            with self.assertRaises(ValueError):
                watch.validate_black_retry(data, fixtures.JOB, fields, outer, gate, gate_name,
                    't01v2_search', bad, record)
        self.assertFalse(watch.authorized_shard_command(child['_meta']['command'], v2=True, search=True, stable=True))

    def test_search_gate_black_keeps_the_original_per_card_proof(self):
        data, fields, outer, gate, gate_name, _, _ = self.search_fixture(False)
        self.assertEqual(watch.validate_black_retry(data, fixtures.JOB, fields, outer, gate, gate_name,
            't01v2_search'), 'render_gate_black_capture')
        data['budget_group'] = None
        with self.assertRaises(ValueError):
            watch.validate_black_retry(data, fixtures.JOB, fields, outer, gate, gate_name, 't01v2_search')


if __name__ == '__main__':
    unittest.main()
