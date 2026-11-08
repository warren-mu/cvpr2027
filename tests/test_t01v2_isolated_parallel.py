"""Versioned dispatch/retry fixtures only; no PBS or simulator calls."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from src.tools import phase_watch as watch
from src.tools import t01v2_parallel as original
from src.tools import t01v2_solvable_parallel as solvable
from src.tools import t01v2_isolated_parallel as isolated
from src.envs.pair_v2_isolated_shard import ATTEMPT_SETUP_POLICY

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('isolated_solvable_fixtures', ROOT / 'tests/test_t01v2_solvable_parallel.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
JOB = fixtures.fixtures.fixtures.JOB
submit = fixtures.fixtures.fixtures.submit


def black_fixture(child_failure=True):
    payload, fields, outer, gate, gate_name, child, record = fixtures.black_fixture(child_failure)
    payload['command'][1] = 'experiments/t01v2_isolated_pairs.sh'
    outer['_meta']['command'] = list(payload['command'])
    for location in ('config', 'metrics'):
        outer[location].update(isolated.isolated_tags())
    if child_failure:
        child['_meta']['command'] = ['python3', '-u', '-m', isolated.SCENE_MODULE]
        child['config']['producer_module'] = isolated.SCENE_MODULE
        for location in ('config', 'metrics'):
            child[location].update(attempt_setup_version=isolated.ATTEMPT_SETUP_VERSION,
                                   attempt_setup_policy=copy.deepcopy(ATTEMPT_SETUP_POLICY))
    return payload, fields, outer, gate, gate_name, child, record


class IsolatedInfraTests(unittest.TestCase):
    def test_scene_module_replacement_preserves_every_other_argument(self):
        command = ['python3', '-u', '-m', 'src.envs.pair_v2_shard', '--name', 'scene',
                   '--scene', 'FloorPlan201', '--shard-config', 'config.json', '--timeout-seconds', '500']
        actual = isolated.isolated_scene_command('scene', command)
        self.assertEqual(actual[3], isolated.SCENE_MODULE)
        self.assertEqual(actual[:3] + actual[4:], command[:3] + command[4:])
        self.assertEqual(command[3], 'src.envs.pair_v2_shard')
        self.assertEqual(solvable.solvable_scene_command('scene', command)[3], solvable.SCENE_MODULE)
        gate = ['python3', '-u', '-m', 'src.tools.render_gate', '--name', 'gate']
        self.assertIs(isolated.isolated_scene_command('gate', gate), gate)
        fake = ['python3', '-c', 'pass', 'src.envs.pair_v2_shard']
        self.assertIs(isolated.isolated_scene_command('scene', fake), fake)
        with self.assertRaises(ValueError):
            isolated.isolated_scene_command('other', command)

    def test_private_dispatch_and_exact_policy_are_restored_after_failure(self):
        metrics, config = {}, {}
        def run():
            original.write_result('fixture', metrics, config)
            raise RuntimeError('fixture failure')
        with patch.object(original, 'main', side_effect=run), patch.object(original, 'launch_logged') as launch, \
                patch.object(original, 'write_result') as write:
            with self.assertRaises(RuntimeError):
                isolated.main()
            self.assertIs(original.launch_logged, launch)
            self.assertIs(original.write_result, write)
            self.assertEqual(metrics, isolated.isolated_tags())
            self.assertEqual(config, isolated.isolated_tags())
            self.assertEqual(config['attempt_setup_policy'], ATTEMPT_SETUP_POLICY)
            write.assert_called_once_with('fixture', metrics, config)
        config['attempt_setup_policy']['timing'] = 'changed fixture'
        self.assertEqual(isolated.isolated_tags()['attempt_setup_policy'], ATTEMPT_SETUP_POLICY)
        self.assertNotEqual(metrics['attempt_setup_policy'], config['attempt_setup_policy'])

    def test_new_entry_still_uses_same_sixteen_and_sixty_hour_budget(self):
        name = 'pilot_pairs_v2_isolated_fixture'
        command = ['bash', 'experiments/t01v2_isolated_pairs.sh', '--name', name]
        self.assertEqual(watch.entrypoint(command, name, ROOT), 't01v2_isolated')
        self.assertEqual(submit.budget_group('T01', name, command), 'T01v2')
        self.assertEqual(submit.SUBBUDGET_LIMITS['T01v2'], 16.0)
        self.assertEqual(submit.TOTAL_LIMIT, 60.0)
        for task, bad_name in (('T00b', name), ('T01', 'pilot_unversioned')):
            with self.assertRaises(ValueError):
                submit.budget_group(task, bad_name, command)
        with self.assertRaises(ValueError):
            submit.budget_group('T01', 'other', ['python3', '-m', 'src.tools.t01v2_isolated_parallel'])

    def test_old_search_activation_is_unchanged_and_never_relabeled_isolated(self):
        empty = {'config': {}, 'metrics': {}}
        self.assertEqual(watch.effective_kind('t01v2_search', empty), 't01v2_search')
        old = fixtures.mark_outer(copy.deepcopy(empty))
        self.assertEqual(watch.effective_kind('t01v2_search', old), 't01v2_solvable')
        self.assertEqual(watch.effective_kind('t01v2_solvable', old), 't01v2_solvable')
        new = {'config': isolated.isolated_tags(), 'metrics': isolated.isolated_tags()}
        self.assertEqual(watch.effective_kind('t01v2_isolated', new), 't01v2_isolated')
        for kind in ('t01v2_search', 't01v2_solvable', 't01v2'):
            with self.assertRaises(ValueError):
                watch.effective_kind(kind, new)
        with self.assertRaises(ValueError):
            watch.effective_kind('t01v2_isolated', old)

    def test_parent_requires_both_exact_hashed_version_and_complete_policy(self):
        outer = {'config': isolated.isolated_tags(), 'metrics': isolated.isolated_tags()}
        for location in ('config', 'metrics'):
            for key in isolated.isolated_tags():
                changed = copy.deepcopy(outer)
                del changed[location][key]
                with self.assertRaises(ValueError):
                    watch.effective_kind('t01v2_isolated', changed)
            changed = copy.deepcopy(outer)
            changed[location]['attempt_setup_policy']['verification'] = 'skip checks'
            with self.assertRaises(ValueError):
                watch.effective_kind('t01v2_isolated', changed)

    def test_scene_black_requires_new_module_policy_and_all_existing_versions(self):
        payload, fields, outer, gate, gate_name, child, record = black_fixture()
        call = lambda changed: watch.validate_black_retry(payload, JOB, fields, outer, gate, gate_name,
                                                          't01v2_isolated', changed, record)
        self.assertEqual(call(child), 't01v2_isolated_scene_black_capture')
        for location in ('config', 'metrics'):
            for key in ('attempt_setup_version', 'attempt_setup_policy', 'solvability_version',
                        'search_version', 'preflight_version', 'evidence_version'):
                changed = copy.deepcopy(child)
                changed[location][key] = 'unknown'
                with self.assertRaises(ValueError):
                    call(changed)
        changed = copy.deepcopy(child)
        changed['config']['attempt_setup_policy']['paired_prefix'] = 'reset between worlds'
        with self.assertRaises(ValueError):
            call(changed)
        for command in (['python3', '-u', '-m', solvable.SCENE_MODULE],
                        ['python3', '-m', 'other', isolated.SCENE_MODULE],
                        ['bash', '-m', isolated.SCENE_MODULE]):
            changed = copy.deepcopy(child)
            changed['_meta']['command'] = command
            with self.assertRaises(ValueError):
                call(changed)
        changed = copy.deepcopy(child)
        changed['config']['producer_module'] = solvable.SCENE_MODULE
        with self.assertRaises(ValueError):
            call(changed)

    def test_no_mixed_module_flags_or_ordinary_failures_are_allowed(self):
        command = ['python3', '-u', '-m', isolated.SCENE_MODULE]
        self.assertTrue(watch.authorized_shard_command(command, v2=True, isolated=True))
        for flag in ('clone', 'stable', 'search', 'solvable'):
            self.assertFalse(watch.authorized_shard_command(command, v2=True, isolated=True, **{flag: True}))
        payload, fields, outer, gate, gate_name, child, record = black_fixture()
        for change in (lambda c: c['_meta'].update(exit_code=1),
                       lambda c: c['metrics'].update(cleanup_errors=['unknown'])):
            changed = copy.deepcopy(child)
            change(changed)
            with self.assertRaises(ValueError):
                watch.validate_black_retry(payload, JOB, fields, outer, gate, gate_name, 't01v2_isolated', changed, record)

    def test_gate_black_uses_original_real_per_card_proof_and_policy(self):
        payload, fields, outer, gate, gate_name, _, _ = black_fixture(False)
        self.assertEqual(watch.validate_black_retry(payload, JOB, fields, outer, gate, gate_name, 't01v2_isolated'),
                         'render_gate_black_capture')
        outer['metrics']['attempt_setup_policy'] = {}
        with self.assertRaises(ValueError):
            watch.validate_black_retry(payload, JOB, fields, outer, gate, gate_name, 't01v2_isolated')


if __name__ == '__main__':
    unittest.main()
