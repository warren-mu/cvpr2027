"""Pilot lifecycle/decision fixtures; no GPU or actual training measurements."""
import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools import smoke_e2e_pilot as pilot


def complete_result(arm='T0', *, late_seen=0, late_reward=0.):
    updates = []
    for index in range(1, pilot.ARM_UPDATES[arm] + 1):
        updates.append({'update': index, 'by_class': {kind: {
            'episode_count': 16, 'seen_before_select_count': late_seen if index >= 41 else 0,
            'task_reward_sum': late_reward * 16 if index >= 41 else 0.}
            for kind in pilot.KINDS}})
    return {'actual_completed_updates': pilot.ARM_UPDATES[arm], 'updates': updates,
        'holdout_evaluations': [{'update': index, 'greedy': True, 'complete': True} for index in pilot.EVAL_UPDATES[arm]],
        'cache_authorized': True, 'dataset_proof': {'cache_authorized': True, 'common_pool_hash': 'a' * 64},
        'common_pool_hash': 'a' * 64, 'common_config_sha256': 'b' * 64, 'pilot_completed': True,
        'errors': [], 'cleanup_errors': [], 'sources': []}


def compared(arm, **kwargs):
    return {'config': {'arm': arm, 'updates': 50, 'critic_offload': True, 'runtime': {'gpu_name': 'fixture-H200'},
        'manifest_reference': {'fixture': 'same'}, 'pilot_spec_reference': {'fixture': 'spec'},
        'common_config_reference': {'fixture': 'common'}, 'model_snapshot_reference': {'fixture': 'base'}},
        'metrics': complete_result(arm, **kwargs)}


class PilotDecisionTests(unittest.TestCase):
    def test_engineering_cannot_enter_comparison_even_with_fabricated_full_length(self):
        for location, key, value in [('config', 'engineering_smoke', pilot.ENGINEERING_SMOKE),
            ('metrics', 'engineering_smoke_version', pilot.ENGINEERING_SMOKE),
            ('metrics', 'engineering_smoke_completed', True),
            ('metrics', 'kind', 'smoke_e2e_pilot_engineering_training')]:
            values = {arm: compared(arm) for arm in ('T0', 'T4')}
            values['T4'][location][key] = value
            with self.subTest(location=location, key=key), self.assertRaisesRegex(ValueError, 'Engineering smoke'):
                pilot.learning_comparison(values)

    def test_both_arms_both_small_changes_stop_without_mutating_configuration(self):
        inputs = {arm: compared(arm) for arm in ('T0', 'T4')}; before = copy.deepcopy(inputs)
        result = pilot.learning_comparison(inputs)
        self.assertTrue(result['stop_and_report'])
        self.assertFalse(result['common_config_frozen'])
        self.assertFalse(result['common_config_mutated'])
        self.assertFalse(result['observation_action_rate_used_for_decision'])
        self.assertEqual(inputs, before)

    def test_either_arm_change_and_exact_threshold_prevent_stop(self):
        for arm in ('T0', 'T4'):
            values = {name: compared(name) for name in ('T0', 'T4')}
            values[arm] = compared(arm, late_reward=.05)
            self.assertFalse(pilot.learning_comparison(values)['stop_and_report'])
            values[arm] = compared(arm, late_seen=1)
            self.assertFalse(pilot.learning_comparison(values)['stop_and_report'])

    def test_pools_and_public_hyperparameters_must_match(self):
        for field in ('manifest_reference', 'pilot_spec_reference', 'common_config_reference', 'critic_offload', 'model_snapshot_reference'):
            values = {arm: compared(arm) for arm in ('T0', 'T4')}
            values['T4']['config'][field] = 'different'
            with self.assertRaisesRegex(ValueError, 'different common inputs'):
                pilot.learning_comparison(values)
        values = {arm: compared(arm) for arm in ('T0', 'T4')}
        values['T4']['metrics']['common_config_sha256'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'different common execution'):
            pilot.learning_comparison(values)

    def test_missing_updates_empty_look_or_incomplete_eval_cannot_mean_no_change(self):
        for mutation in ('update', 'empty', 'eval', 'cache'):
            values = {arm: compared(arm) for arm in ('T0', 'T4')}
            metric = values['T0']['metrics']
            if mutation == 'update': metric['updates'].pop(3)
            elif mutation == 'empty':
                for update in metric['updates']:
                    for row in update['by_class'].values(): row['episode_count'] = 0
            elif mutation == 'eval': metric['holdout_evaluations'][1]['complete'] = False
            else: metric['dataset_proof']['cache_authorized'] = False
            with self.assertRaises(ValueError): pilot.learning_comparison(values)

    def test_window_gives_each_update_equal_weight_with_unequal_episode_counts(self):
        updates = complete_result()['updates']
        updates[0]['by_class']['look_flip'].update(episode_count=1, seen_before_select_count=1, task_reward_sum=1.)
        value = pilot.learning_window(updates, 1, 10)
        self.assertEqual(value['look_episodes'], 305)
        self.assertEqual(value['seen_before_select_rate'], (1 / 17) / 10)
        self.assertEqual(value['task_reward_mean'], (1 / 17) / 10)
        self.assertEqual(value['auxiliary_pooled_seen_rate'], 1 / 305)
        self.assertEqual(value['auxiliary_pooled_task_reward_mean'], 1 / 305)
        for row in updates[4]['by_class'].values(): row['episode_count'] = 0
        with self.assertRaisesRegex(ValueError, 'update contains no actual Look'):
            pilot.learning_window(updates, 1, 10)

    def test_current_spec_and_unknown_or_pending_t4_definition(self):
        source = pilot.ROOT / 'configs/smoke_e2e_pilot.json'
        spec, _ = pilot.load_spec(source)
        self.assertEqual(spec['split_policy'], pilot.SCENE_SPLIT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'configs').mkdir()
            destination = root / 'configs/spec.json'
            spec['t4_consistency']['symmetric_kl_sum_or_half_sum'] = None
            destination.write_text(json.dumps(spec))
            with patch.object(pilot, 'ROOT', root), self.assertRaisesRegex(ValueError, 'section 4b'):
                pilot.load_spec(destination)


class PilotLifecycleTests(unittest.TestCase):
    def run_fixture(self, root, *, gate_code=0, worker_result=None, cleanup=None,
                    arm='T0', engineering_smoke=None, name='smoke_e2e_pilot_fixture'):
        (root / 'configs').mkdir()
        (root / 'configs/rl_common.yaml').write_text('fixture: true\n')
        identity = {'pbs_jobid': '123.hopper-m-02', 'scratch': str(root / 'scratch'),
                    'cpu_affinity': list(range(12)), 'cuda_visible_devices': 'GPU-fixture'}
        spec = {'format_policy': 'explicit_task_zero', 'credit_pipeline': 't05_section1b', 'split_policy': pilot.SCENE_SPLIT}
        args = argparse.Namespace(operation='run', name=name, arm=arm, engineering_smoke=engineering_smoke,
            manifest=json.dumps({'file': 'data/pool.json', 'sha256': 'a' * 64}),
            model_source=json.dumps({'file': 'results/base.json', 'sha256': 'b' * 64}),
            pilot_spec='configs/spec.json', critic_offload=True, timeout_seconds=600., gate_timeout_seconds=30.)
        gate = {'_meta': {'exit_code': gate_code}, 'metrics': {'cleanup_errors': []}}
        observed = []
        class Worker:
            error = None
            def __init__(self, target, config):
                observed.append(config); self.result = copy.deepcopy(worker_result)
            def receive(self): pass
        with ExitStack() as stack, redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()):
            stack.enter_context(patch.object(pilot, 'ROOT', root))
            stack.enter_context(patch.object(pilot, 'load_spec', return_value=(spec, {'file': 'spec', 'sha256': 'c' * 64})))
            stack.enter_context(patch.object(pilot, 'install_signals'))
            stack.enter_context(patch.object(pilot.signal, 'setitimer'))
            stack.enter_context(patch.object(pilot.os, 'sched_getaffinity', create=True, return_value=set(range(12))))
            stack.enter_context(patch.object(pilot, 'execution_environment', return_value=identity))
            stack.enter_context(patch.object(pilot, 'logged_gate', return_value=(gate, {'file': 'gate'}, [])))
            stack.enter_context(patch.object(pilot, 'gate_runtime', return_value=identity))
            model = stack.enter_context(patch.object(pilot, 'model_reference', return_value={'source_result': {'file': 'model'}}))
            manifest = stack.enter_context(patch.object(pilot, 'read_repo_ref', return_value={'version': pilot.DATASET_VERSION, 'split': {'policy': pilot.SCENE_SPLIT}}))
            stack.enter_context(patch.object(pilot, 'Worker', Worker))
            stack.enter_context(patch.object(pilot, 'stop_workers', return_value=cleanup or []))
            code = pilot.run(args)
        saved = json.loads((root / 'results' / (name + '.json')).read_text())
        self.assertEqual(saved['metrics'], json.loads(output.getvalue().split('FINAL_METRICS_JSON: ')[-1]))
        return code, saved, observed, model.call_count, manifest.call_count

    def engineering_result(self):
        result = complete_result('T4')
        result.update(actual_completed_updates=5, updates=result['updates'][:5],
            holdout_evaluations=result['holdout_evaluations'][:1], checkpoint_saved=True,
            engineering_smoke_version=pilot.ENGINEERING_SMOKE)
        return result

    def test_engineering_success_is_five_updates_without_formal_pilot_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            code, saved, workers, model, manifest = self.run_fixture(Path(directory),
                worker_result=self.engineering_result(), arm='T4',
                engineering_smoke=pilot.ENGINEERING_SMOKE, name='smoke_e2e_pilot_engineering_T4_fixture')
        self.assertEqual((code, model, manifest), (0, 1, 1))
        self.assertEqual(workers[0]['updates'], 5)
        self.assertEqual(workers[0]['version'], pilot.ENGINEERING_VERSION)
        self.assertEqual(workers[0]['budget_group'], 'E2Epilot')
        self.assertEqual(workers[0]['format_policy'], 'explicit_task_zero')
        self.assertEqual(workers[0]['credit_pipeline'], 't05_section1b')
        metrics = saved['metrics']
        self.assertTrue(metrics['engineering_smoke_completed'])
        self.assertFalse(metrics['pilot_completed'])
        self.assertFalse(metrics['learning_comparison_eligible'])
        self.assertEqual(metrics['scheduled_holdout_updates'], [0])
        self.assertEqual(metrics['unrun_formal_holdout_updates'], [20, 40, 50])
        self.assertEqual(metrics['kind'], 'smoke_e2e_pilot_engineering_training')

    def test_engineering_failure_and_uncertain_cleanup_preserve_incomplete_status(self):
        for mutation in ('updates', 'cache', 'mode', 'checkpoint', 'holdout', 'cleanup'):
            result = self.engineering_result(); cleanup = []
            if mutation == 'updates': result['actual_completed_updates'] = 4
            elif mutation == 'cache': result['dataset_proof']['cache_authorized'] = False
            elif mutation == 'mode': result.pop('engineering_smoke_version')
            elif mutation == 'checkpoint': result['checkpoint_saved'] = False
            elif mutation == 'holdout': result['holdout_evaluations'] = []
            else: cleanup = ['owned process survived']
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                code, saved, *_ = self.run_fixture(Path(directory), worker_result=result,
                    cleanup=cleanup, arm='T4', engineering_smoke=pilot.ENGINEERING_SMOKE,
                    name='smoke_e2e_pilot_engineering_T4_fixture')
            self.assertEqual(code, 1)
            self.assertFalse(saved['metrics']['pilot_completed'])
            self.assertFalse(saved['metrics']['engineering_smoke_completed'])

    def test_engineering_namespace_and_black_gate_cannot_bypass_authorization(self):
        for arm, mode, name in [('T0', pilot.ENGINEERING_SMOKE, 'smoke_e2e_pilot_engineering_T4_fixture'),
            ('T4', pilot.ENGINEERING_SMOKE, 'smoke_e2e_pilot_fixture'),
            ('T4', None, 'smoke_e2e_pilot_engineering_T4_fixture')]:
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                self.run_fixture(Path(directory), arm=arm, engineering_smoke=mode, name=name)
        with tempfile.TemporaryDirectory() as directory:
            code, saved, workers, model, manifest = self.run_fixture(Path(directory), gate_code=86,
                arm='T4', engineering_smoke=pilot.ENGINEERING_SMOKE, name='smoke_e2e_pilot_engineering_T4_fixture')
        self.assertEqual((code, model, manifest), (86, 0, 0))
        self.assertEqual(workers, [])
        self.assertFalse(saved['metrics']['engineering_smoke_completed'])

    def test_engineering_worker_reauthenticates_original_qualified_pool_before_ray(self):
        from src.tools import smoke_e2e_pilot_training as training
        from src.tools import smoke_e2e_pilot_dataset as dataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'configs').mkdir()
            (root / 'configs/smoke_e2e_pilot.json').write_bytes(
                (pilot.ROOT / 'configs/smoke_e2e_pilot.json').read_bytes())
            common = root / 'configs/rl_common.yaml'; common.write_text('original common fixture\n')
            with patch.object(pilot, 'ROOT', root), patch.object(training, 'ROOT', root), \
                    patch.object(training.os, 'sched_setaffinity', create=True), \
                    patch.object(dataset, 'load_pilot_pool', side_effect=ValueError('original formal cache qualification rejected')) as loader:
                spec, spec_ref = pilot.load_spec('configs/smoke_e2e_pilot.json')
                config = {'arm': 'T4', 'engineering_smoke': pilot.ENGINEERING_SMOKE, 'updates': 5,
                    'identity': {'cpu_affinity': list(range(12))}, 'pilot_spec_reference': spec_ref,
                    'format_policy': spec['format_policy'], 'credit_pipeline': spec['credit_pipeline'],
                    'manifest_reference': {'file': 'data/frozen.json', 'sha256': 'a' * 64},
                    'common_config_reference': training.file_ref(common, root / 'configs')}
                with self.assertRaisesRegex(ValueError, 'original formal cache qualification rejected'):
                    training.pilot_train_worker(config, None)
                loader.assert_called_once_with(config['manifest_reference'])
                self.assertEqual(spec['minimum_balanced_pairs'], 80)
                self.assertEqual(spec['required_classes'], list(pilot.KINDS))

    def test_black_gate_precedes_model_manifest_and_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            code, saved, workers, model, manifest = self.run_fixture(Path(directory), gate_code=86)
        self.assertEqual((code, model, manifest), (86, 0, 0))
        self.assertEqual(workers, [])
        self.assertFalse(saved['metrics']['pilot_completed'])

    def test_complete_worker_and_same_common_inputs_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            code, saved, workers, model, manifest = self.run_fixture(Path(directory), worker_result=complete_result())
        self.assertEqual((code, model, manifest), (0, 1, 1))
        self.assertTrue(saved['metrics']['pilot_completed'])
        self.assertEqual(workers[0]['updates'], 50)
        self.assertEqual(workers[0]['budget_group'], 'E2Epilot')
        self.assertEqual(workers[0]['gpus'], 1)
        self.assertIn('common_config_reference', workers[0])

    def test_failed_update_or_uncertain_cleanup_is_not_completion(self):
        incomplete = complete_result(); incomplete['actual_completed_updates'] = 49
        for result, cleanup in ((incomplete, []), (complete_result(), ['owned process survived'])):
            with tempfile.TemporaryDirectory() as directory:
                code, saved, *_ = self.run_fixture(Path(directory), worker_result=result, cleanup=cleanup)
            self.assertEqual(code, 1)
            self.assertFalse(saved['metrics']['pilot_completed'])


if __name__ == '__main__': unittest.main()
