"""E2E step 6 only: bounded single-GPU pilot and its predeclared comparison.

The dataset must independently reauthenticate complete formal T02 cache batches.
This entry grants no paper-result or main-training approval and never mutates
the shared training configuration based on the observed arm ordering.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time
import traceback

from src.envs.thor_common import ROOT, Worker, deadline_after, install_signals, stop_workers
from src.tools.t01_partition import audited_result, canonical_hash
from src.tools.merge_pair_v2 import producer_arguments
from src.tools.t02_export import save_json, execution_environment, logged_gate, gate_runtime
from src.tools.smoke_e2e_views import file_ref
from src.tools.smoke_e2e_pilot_dataset import (VERSION as DATASET_VERSION, SCENE_SPLIT,
    KINDS, require, ref_identity, read_repo_ref, argv_values)
from src.tools.smoke_e2e_pilot_contract import ENGINEERING_SMOKE, execution_schedule

VERSION = 'smoke_e2e_pilot_driver_v1'
ENGINEERING_VERSION = 'smoke_e2e_pilot_engineering_driver_v2'
SPEC_VERSION = 'smoke_e2e_pilot_spec_v2_keep_all'
MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'
ARM_UPDATES = {'T0': 50, 'T4': 50, 'T2': 5}
EVAL_UPDATES = {'T0': [0, 20, 40, 50], 'T4': [0, 20, 40, 50], 'T2': []}


def load_spec(path):
    path = (ROOT / path).resolve()
    path.relative_to((ROOT / 'configs').resolve())
    spec = json.loads(path.read_text())
    expected = {'version': SPEC_VERSION, 'approved_task_revision': 'fa24c4e8', 'model': MODEL,
        'arm_updates': ARM_UPDATES, 'episodes_per_update': 64, 'turns': 5, 'response_tokens': 256,
        'rho': .5, 'seed': 0, 'format_policy': 'explicit_task_zero', 'credit_pipeline': 't05_section1b',
        'dataset_version': DATASET_VERSION, 'dataset_qualification_revision': '7babc169',
        'minimum_cached_pairs': 80, 'required_classes': list(KINDS),
        'split_policy': SCENE_SPLIT, 'holdout_updates': EVAL_UPDATES,
        'budget_group': 'E2Epilot', 'budget_gpu_hours': 6, 'gpus_per_job': 1,
        'authorized_conditional_max_gpu_hours': 9,
        'budget_extension_condition': 'only after one complete update measured timing predicts above6; above9 stop and report; never reduce formal update counts',
        'paper_result': False, 'automatic_common_config_mutation': False}
    require(all(spec.get(key) == value for key, value in expected.items()), 'Pilot specification differs from the approved task')
    require(all(type(spec.get(key)) is int for key in ('seed', 'episodes_per_update', 'turns',
        'response_tokens', 'minimum_cached_pairs', 'gpus_per_job')),
        'Frozen pilot integer settings cannot be boolean aliases')
    require(spec.get('holdout_eval_decoding_and_budget_contract') == {
        'decoding': 'greedy', 'temperature': 0, 'worlds_per_pair': ['w0', 'w1'],
        'episodes_per_world': 1, 'maximum_rounds': 5, 'maximum_response_tokens': 256,
        'training_overlap_allowed': False}, 'Pilot evaluation must greedily cover both held-out worlds exactly once')
    require(spec.get('t4_consistency') == {
        'beta': .1, 'flip_log_probability_margin': 1.,
        'stay_selection_distribution': 'complete actual selectable candidate scores, normalized over select candidates',
        'symmetric_kl_sum_or_half_sum': 'sum_of_two_per_episode_directional_KLs_to_detached_sampled_other_policy',
        'flip_per_world_to_training_loss_reduction': 'sum_2p_world_c_i_divided_by_all_minibatch_episodes_including_inapplicable_zero',
        'missing_or_invisible_auxiliary_candidate_policy': 'zero_unselectable_Flip_target_or_ineligible_Stay_pair_or_no_legal_select'},
        'T4 consistency convention is not the explicitly approved section 4b')
    diagnostic = spec.get('learning_diagnostic', {})
    require(diagnostic.get('early_updates') == [1, 10] and diagnostic.get('late_updates') == [41, 50]
        and diagnostic.get('seen_absolute_change_threshold') == .05
        and diagnostic.get('task_reward_absolute_change_threshold') == .05
        and diagnostic.get('aggregation') == 'equal-weight mean of per-update Look episode rates and rewards; each update must contain Look episodes'
        and diagnostic.get('stop_if') == 'both T0 and T4 are strictly below both thresholds',
        'Predeclared learning comparison changed')
    return spec, file_ref(path, ROOT / 'configs')


def model_reference(expected, scratch):
    """Only reuse model bytes from step 3, never its five-pair/cache exception."""
    expected = ref_identity(expected)
    data, ref = audited_result(ROOT / expected['file'])
    require(ref_identity(ref) == expected, 'Original model source changed')
    cfg, meta, metrics = data['config'], data['_meta'], data['metrics']
    args = producer_arguments(meta['command'], 'src.tools.smoke_e2e_eval')
    require(meta.get('end') and meta.get('exit_code') == 0 and metrics.get('producer_exit_code') == 0
        and metrics.get('step3_passed') is True and metrics.get('errors') == []
        and metrics.get('cleanup_errors') == [] and metrics.get('model') == MODEL
        and metrics.get('paper_result') is False and argv_values(args, '--name') == [meta['name']]
        and cfg.get('name') == meta['name'], 'Original model source is not a successful audited step 3')
    model = Path(metrics['model_snapshot_directory']).resolve()
    model.relative_to(Path(scratch).resolve())
    sha = hashlib.sha256((model / 'config.json').read_bytes()).hexdigest()
    require(sha == metrics['model_config_sha256'], 'Original model snapshot configuration changed')
    return {'source_result': ref, 'directory': str(model), 'model': MODEL,
        'model_config_sha256': sha, 'scope': 'original model only; no source pool/cache or training authorization inherited'}


def pilot_train_worker(config, connection):
    # Import inside the spawned process: no Ray/model dependencies on the driver.
    from src.tools.smoke_e2e_pilot_training import pilot_train_worker as original
    return original(config, connection)


def completed_worker(result, config):
    mode = config.get('engineering_smoke')
    updates, holdout_updates = execution_schedule(config['arm'], mode)
    require(config['updates'] == updates, 'Run length differs from the explicit pilot execution mode')
    require(result.get('engineering_smoke_version') == mode,
        'Worker engineering mode differs from its original driver')
    require(mode is None or result.get('checkpoint_saved') is True,
        'The engineering run did not save its own completed checkpoint')
    require(result.get('actual_completed_updates') == config['updates'], 'Pilot stopped before all actual updates completed')
    rows = result.get('updates')
    require(isinstance(rows, list) and len(rows) == config['updates']
        and [row.get('update') for row in rows] == list(range(1, config['updates'] + 1)), 'Actual optimizer update coverage differs')
    for row in rows:
        classes = row.get('by_class', {})
        require(set(classes) == set(KINDS) and all(type(value.get('episode_count')) is int
            and value['episode_count'] >= 0 for value in classes.values())
            and sum(value['episode_count'] for value in classes.values()) == 64,
            'Every actual update must account for its complete 64-episode four-class budget')
    evaluations = result.get('holdout_evaluations')
    require(isinstance(evaluations, list)
        and [row.get('update') for row in evaluations] == list(holdout_updates)
        and all(row.get('greedy') is True and row.get('complete') is True for row in evaluations),
        'Pilot holdout evaluation is incomplete')
    require(result.get('cache_authorized') is True and result.get('dataset_proof', {}).get('cache_authorized') is True
        and isinstance(result.get('common_pool_hash'), str) and len(result['common_pool_hash']) == 64
        and isinstance(result.get('common_config_sha256'), str) and len(result['common_config_sha256']) == 64,
        'Pilot runtime has no complete formal dataset/common-configuration proof')
    require(result['dataset_proof'].get('common_pool_hash') == result['common_pool_hash'], 'Runtime dataset identity differs')
    fixed = result['dataset_proof'].get('fixed_holdout', {})
    require(fixed.get('reference') == config.get('holdout_reference')
        and fixed.get('pilot_population_and_split_verified') is True,
        'Runtime has not bound its exact pilot population to the permanent T04 holdout')


def run(args):
    mode = getattr(args, 'engineering_smoke', None)
    updates, holdout_updates = execution_schedule(args.arm, mode)
    group = getattr(args, 'budget_group', 'E2Epilot')
    if args.arm == 'T2' or group == 'E2EpilotT2' or args.name.startswith('smoke_e2e_pilot_T2_'):
        require(group == 'E2EpilotT2' and args.arm == 'T2' and mode is None
            and re.fullmatch(r'smoke_e2e_pilot_T2_[A-Za-z0-9_]+', args.name),
            'T2 requires its separate explicit budget and native five-update namespace')
    else:
        require(group == 'E2Epilot', 'Unknown pilot budget group')
    require(mode is None or re.fullmatch(r'smoke_e2e_pilot_engineering_T4_[A-Za-z0-9_]+', args.name),
        'The engineering prefix needs its separate T4 run namespace')
    require(mode is not None or not args.name.startswith('smoke_e2e_pilot_engineering_'),
        'The engineering namespace requires its explicit versioned mode')
    require(re.fullmatch(r'smoke_e2e_pilot_[A-Za-z0-9_]+', args.name)
        and math.isfinite(args.timeout_seconds) and 300 <= args.timeout_seconds <= 21600
        and math.isfinite(args.gate_timeout_seconds) and 15 <= args.gate_timeout_seconds <= args.timeout_seconds - 60,
        'A unique bounded single-GPU pilot run is required')
    output = ROOT / 'results' / (args.name + '.json')
    require(not output.exists(), 'Preserve previous pilot runs')
    spec, spec_ref = load_spec(args.pilot_spec)
    manifest_ref = ref_identity(json.loads(args.manifest))
    holdout_ref = ref_identity(json.loads(args.holdout_reference))
    common_path = ROOT / 'configs/rl_common.yaml'
    config = {**vars(args), 'version': ENGINEERING_VERSION if mode else VERSION, 'gpus': 1, 'budget_group': group,
        'model': MODEL, 'updates': updates, 'episodes_per_update': 64, 'seed': 0,
        'format_policy': spec['format_policy'], 'credit_pipeline': spec['credit_pipeline'],
        'node': socket.gethostname(), 'manifest_reference': manifest_ref, 'holdout_reference': holdout_ref,
        'pilot_spec_reference': spec_ref, 'common_config_reference': file_ref(common_path, ROOT / 'configs')}
    metrics = {'kind': 'smoke_e2e_pilot_training', 'pilot_completed': False,
        'errors': [], 'cleanup_errors': [], 'black_frame_detected': False,
        'paper_result': False, 'formal_T04_T05_acceptance': None, 'common_config_frozen': False}
    if mode:
        metrics.update(kind='smoke_e2e_pilot_engineering_training', engineering_smoke_version=mode,
            engineering_smoke_completed=False, learning_comparison_eligible=False,
            scheduled_holdout_updates=list(holdout_updates), unrun_formal_holdout_updates=[20, 40, 50],
            scope='T4 first update only; same qualified pool and frozen method; no50-step learning comparison or resume authorization')
    worker, sources, code = None, [], 1
    install_signals()
    try:
        identity = execution_environment(os.environ, os.sched_getaffinity(0)); config['identity'] = identity
        deadline = deadline_after(args.timeout_seconds)
        signal.setitimer(signal.ITIMER_REAL, max(.1, deadline - time.time()))
        scratch = Path(identity['scratch']) / 'smoke_e2e_pilot' / args.name / identity['pbs_jobid']
        scratch.mkdir(parents=True, exist_ok=False)
        logs = ROOT / 'logs/runs' / args.name
        logs.mkdir(parents=True, exist_ok=False)
        config.update(training_directory=str(logs / 'ppo'), checkpoint_directory=str(scratch / 'checkpoint'))
        gate, gate_ref, cleanup = logged_gate(args.name + '_gate', args.gate_timeout_seconds, scratch, deadline, sources)
        metrics.update(gate=gate_ref, cleanup_errors=cleanup)
        if gate['_meta']['exit_code'] != 0:
            metrics['black_frame_detected'] = gate['_meta']['exit_code'] == 86
            code = 86 if metrics['black_frame_detected'] and not cleanup and gate['metrics'].get('cleanup_errors') == [] else 1
        else:
            config['runtime'] = gate_runtime(gate, identity, config['node'], scratch / 'unity_logs')
            require(not cleanup, 'Gate cleanup uncertain')
            # Complete cache/source reauthentication occurs in the owned worker.
            # Even the inexpensive model/source and manifest reads are after gate.
            manifest = read_repo_ref(manifest_ref)
            require(manifest.get('version') == DATASET_VERSION and manifest['split']['policy'] == spec['split_policy'],
                    'Pilot manifest has not frozen the approved scene split')
            config['model_snapshot_reference'] = model_reference(json.loads(args.model_source), identity['scratch'])
            sources.append(config['model_snapshot_reference']['source_result'])
            worker = Worker(pilot_train_worker, {**config, 'deadline': deadline - 20})
            while worker.result is None and worker.error is None and time.time() < deadline - 15:
                worker.receive(); time.sleep(.1)
            worker.receive()
            require(not worker.error and worker.result is not None, 'Pilot worker did not finish: ' + str(worker.error))
            result = copy.deepcopy(worker.result)
            sources.extend(ref for ref in result.pop('sources', []) if ref not in sources)
            metrics.update(result)
            completed_worker(result, config)
            metrics['pilot_completed'] = mode is None
            if mode:
                metrics['engineering_smoke_completed'] = True
            code = 0
    except BaseException as error:
        traceback.print_exc(); metrics['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if worker is not None:
            metrics['cleanup_errors'].extend(stop_workers([worker]))
        if metrics['errors'] or metrics['cleanup_errors']:
            code = 1; metrics['pilot_completed'] = False
            if mode:
                metrics['engineering_smoke_completed'] = False
        if mode:
            metrics['pilot_completed'] = False
        metrics['training_logs'] = [file_ref(path, ROOT / 'logs') for path in
            sorted((ROOT / 'logs/runs' / args.name).rglob('*.json*'))]
        metrics['producer_exit_code'] = code
        output.parent.mkdir(parents=True, exist_ok=True)
        save_json(output, {'metrics': metrics, 'config': config, 'sources': sources})
        print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


def learning_window(updates, first, last):
    selected = [row for row in updates if first <= row['update'] <= last]
    require([row['update'] for row in selected] == list(range(first, last + 1)), 'Learning comparison window is incomplete')
    episodes = seen = 0; reward = 0.; per_update = []
    for update in selected:
        update_episodes = update_seen = 0; update_reward = 0.
        for kind in ('look_flip', 'look_stay'):
            row = update['by_class'][kind]
            n, s, r = row['episode_count'], row['seen_before_select_count'], row['task_reward_sum']
            require(type(n) is int and type(s) is int and 0 <= s <= n
                and isinstance(r, (int, float)) and type(r) is not bool and math.isfinite(r), 'Invalid actual Look window counters')
            update_episodes += n; update_seen += s; update_reward += r
        require(update_episodes > 0, 'Learning comparison update contains no actual Look episodes: ' + str(update['update']))
        per_update.append({'update': update['update'], 'look_episodes': update_episodes,
            'seen_before_select_count': update_seen, 'task_reward_sum': update_reward,
            'seen_before_select_rate': update_seen / update_episodes,
            'task_reward_mean': update_reward / update_episodes})
        episodes += update_episodes; seen += update_seen; reward += update_reward
    return {'first_update': first, 'last_update': last, 'look_episodes': episodes,
        'seen_before_select_count': seen, 'task_reward_sum': reward,
        'per_update': per_update,
        'aggregation': 'equal-weight mean of per-update Look rates/rewards',
        'seen_before_select_rate': math.fsum(row['seen_before_select_rate'] for row in per_update) / len(per_update),
        'task_reward_mean': math.fsum(row['task_reward_mean'] for row in per_update) / len(per_update),
        'auxiliary_pooled_seen_rate': seen / episodes, 'auxiliary_pooled_task_reward_mean': reward / episodes}


def learning_comparison(results):
    require(set(results) == {'T0', 'T4'}, 'The predeclared comparison requires both full arms')
    arm_values = {}
    for arm, data in results.items():
        config, metrics = data['config'], data['metrics']
        require(config.get('engineering_smoke') is None
            and metrics.get('engineering_smoke_version') is None
            and metrics.get('engineering_smoke_completed') is not True
            and metrics.get('kind') != 'smoke_e2e_pilot_engineering_training',
            'Engineering smoke cannot enter the formal50-update learning comparison')
        require(config['arm'] == arm and metrics.get('pilot_completed') is True, 'Only completed matching arms enter comparison')
        completed_worker(metrics, config)
        early, late = learning_window(metrics['updates'], 1, 10), learning_window(metrics['updates'], 41, 50)
        seen = abs(late['seen_before_select_rate'] - early['seen_before_select_rate'])
        reward = abs(late['task_reward_mean'] - early['task_reward_mean'])
        arm_values[arm] = {'early': early, 'late': late, 'absolute_seen_rate_change': seen,
            'absolute_task_reward_change': reward, 'both_below_thresholds': seen < .05 and reward < .05}
    for key in ('manifest_reference', 'pilot_spec_reference', 'common_config_reference', 'critic_offload', 'model_snapshot_reference', 'holdout_reference'):
        require(results['T0']['config'].get(key) == results['T4']['config'].get(key), 'Compared arms have different common inputs: ' + key)
    for key in ('common_pool_hash', 'common_config_sha256'):
        require(results['T0']['metrics'].get(key) == results['T4']['metrics'].get(key), 'Compared arms have different common execution: ' + key)
    hardware = [results[arm]['config'].get('runtime', {}).get('gpu_name') for arm in ('T0', 'T4')]
    require(all(isinstance(name, str) and name for name in hardware) and hardware[0] == hardware[1],
            'Compared arms lack matching authenticated GPU models')
    stop = all(value['both_below_thresholds'] for value in arm_values.values())
    return {'arms': arm_values, 'stop_and_report': stop,
        'decision': 'stop_and_report_before_any_common_change' if stop else 'learning_change_detected_review_common_config_freeze',
        'criterion': 'both arms: absolute Look seen-rate change < 0.05 AND absolute task-return change < 0.05',
        'observation_action_rate_used_for_decision': False, 'paper_result': False,
        'common_config_mutated': False, 'common_config_frozen': False}


def summarize(args):
    require(re.fullmatch(r'smoke_e2e_pilot_[A-Za-z0-9_]+', args.name), 'Unique pilot summary name required')
    output = ROOT / 'results' / (args.name + '.json')
    require(not output.exists(), 'Preserve earlier summaries')
    metrics, sources, values, code = {'kind': 'smoke_e2e_pilot_comparison', 'errors': [], 'paper_result': False}, [], {}, 1
    try:
        for arm, reference in (('T0', args.t0), ('T4', args.t4)):
            expected = ref_identity(json.loads(reference)); data, ref = audited_result(ROOT / expected['file'])
            meta, config, result = data['_meta'], data['config'], data['metrics']
            argv = producer_arguments(meta['command'], 'src.tools.smoke_e2e_pilot')
            require(ref_identity(ref) == expected and meta.get('end') and meta.get('exit_code') == 0
                and result.get('producer_exit_code') == 0 and result.get('errors') == []
                and result.get('cleanup_errors') == [] and result.get('kind') == 'smoke_e2e_pilot_training'
                and argv[:1] == ['run'] and argv_values(argv, '--arm') == [arm]
                and argv_values(argv, '--engineering-smoke') == []
                and [ref_identity(json.loads(v)) for v in argv_values(argv, '--holdout-reference')]
                    == [config.get('holdout_reference')]
                and argv_values(argv, '--name') == [meta['name']] and config.get('version') == VERSION,
                'Require successful audited original pilot T0 and T4 runs')
            values[arm] = data; sources.append(ref)
        metrics.update(learning_comparison(values)); code = 0
    except BaseException as error:
        traceback.print_exc(); metrics['errors'].append(f'{type(error).__name__}: {error}')
    metrics['producer_exit_code'] = code
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, {'metrics': metrics, 'config': {**vars(args), 'version': VERSION, 'gpus': 0}, 'sources': sources})
    print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--name', required=True)
    execute.add_argument('--arm', choices=tuple(ARM_UPDATES), required=True)
    execute.add_argument('--budget-group', choices=('E2Epilot', 'E2EpilotT2'), default='E2Epilot',
        help='T2 requires its separate0.5 GPU-hour ledger; T0/T4 keep the original pilot ledger')
    execute.add_argument('--manifest', required=True, help='JSON {file,sha256} formal pilot pool reference')
    execute.add_argument('--holdout-reference', required=True, help='JSON {file,sha256} permanent T04 holdout freeze')
    execute.add_argument('--model-source', required=True, help='JSON {file,sha256} original model step3 result')
    execute.add_argument('--pilot-spec', default='configs/smoke_e2e_pilot.json')
    execute.add_argument('--critic-offload', action='store_true')
    execute.add_argument('--engineering-smoke', choices=[ENGINEERING_SMOKE], default=None,
        help='T4 engineering check with one complete update; never formal pilot completion')
    execute.add_argument('--timeout-seconds', required=True, type=float)
    execute.add_argument('--gate-timeout-seconds', type=float, default=180.)
    report = commands.add_parser('summarize')
    report.add_argument('--name', required=True); report.add_argument('--t0', required=True); report.add_argument('--t4', required=True)
    args = parser.parse_args()
    return run(args) if args.operation == 'run' else summarize(args)


if __name__ == '__main__':
    sys.exit(main())
