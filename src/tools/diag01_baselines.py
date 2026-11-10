"""DIAG01 D3: four original-model policies on the fixed83 pilot population.

No training, new data, parameter search, or scientific-configuration mutation.
The first fixed Look pair exercises all four policies before the remaining82
pairs within the same single-GPU allocation. Privileged masks control only the
obs_trigger observation schedule; model prompts remain entirely public.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time
import traceback

from src.envs.thor_common import ROOT, Worker, deadline_after, install_signals, stop_workers
from src.envs.paired_env import KINDS, parse_action
from src.envs.pair_table import TableEnvironment, TableObservationBackend
from src.tools.smoke_e2e_eval import VllmPolicy, MODEL, public_actions, EXPLORATION
from src.tools.smoke_e2e_views import file_ref
from src.tools.smoke_e2e_pilot_dataset import load_pilot_pool, authenticate_pilot_holdout, ref_identity
from src.tools.smoke_e2e_pilot import model_reference
from src.tools.t02_export import save_json, execution_environment, logged_gate, gate_runtime

VERSION = 'diag01_fixed83_original_model_v1'
STRATEGIES = ('obs_never', 'obs_always', 'obs_trigger', 'diag_base')


def mask_state(env):
    """Count actual original-world mask bytes, not x_seen_step or cached counts."""
    backend = env.backend
    table, state = backend.table, backend.table.states[backend.state_id]
    mask = state.captures[backend.world].masks[table.anchor_name]
    count = len(mask) - mask.count(0)
    return {'state_id': backend.state_id, 'step': env.steps,
        'original_world_id': f'w{backend.world}', 'anchor_pixels': count,
        'anchor_mask_sha256': hashlib.sha256(mask).hexdigest(), 'shared': state.shared}


def forced_action(strategy, item):
    """Complete a started fixed excursion back to s0 before forced selection.

obs_trigger tests own-mask visibility after each actual observation. Once seen,
it completes the current left/right round trip; it never chooses an observation
direction or the selected target using masks or target identities.
"""
    step = item['env'].steps
    if strategy in ('obs_never', 'diag_base'):
        return None
    if strategy == 'obs_always':
        return EXPLORATION[step] if step < 4 else None
    if strategy != 'obs_trigger':
        raise ValueError('Unknown diagnostic strategy')
    if step >= 4 or (item['first_seen'] is not None and step % 2 == 0):
        return None
    return EXPLORATION[step]


def collect(strategy, environments, policy, journal):
    if strategy not in STRATEGIES:
        raise ValueError('Unknown diagnostic strategy')
    active, episodes = [], []
    for key, world, env in environments:
        observation = env.reset(key, world)
        initial = mask_state(env)
        active.append({'key': key, 'world': world, 'env': env,
            'observations': [observation], 'responses': [], 'states': [initial],
            'first_seen': 0 if initial['anchor_pixels'] >= 250 else None,
            'first_positive': 0 if initial['anchor_pixels'] > 0 else None,
            'task_reward': 0., 'observation_reward_cost': 0., 'observation_attempts': 0,
            'syntax_failures': 0, 'model_outputs': 0})
    for decision in range(5):
        pending, generated = [], {}
        for index, item in enumerate(active):
            if item['env'].done:
                continue
            action = forced_action(strategy, item)
            if action is not None:
                if action not in public_actions(item['observations'][-1]):
                    raise ValueError('Fixed return-to-s0 observation is unavailable')
                generated[index] = {'action': action, 'response': action, 'forced_observation': True,
                    'response_token_ids': [], 'response_token_count': 0, 'sample_id': None}
            else:
                if strategy != 'diag_base' and item['env'].backend.state_id != item['env'].backend.table.initial_state:
                    raise ValueError('Forced-selection policy did not return to s0')
                pending.append((index, {'observations': item['observations'], 'responses': item['responses'],
                    'select_only': strategy != 'diag_base'}))
        if pending:
            values = policy([context for _, context in pending])
            if len(values) != len(pending):
                raise ValueError('Model omitted/added a diagnostic episode')
            for (index, _), sample in zip(pending, values):
                if (not isinstance(sample.get('response'), str) or not isinstance(sample.get('action'), str)
                    or not sample.get('sample_id') or sample.get('forced_observation') is not False
                    or not isinstance(sample.get('response_token_ids'), list)
                    or sample.get('response_token_count') != len(sample['response_token_ids'])
                    or not 0 < sample['response_token_count'] <= 256):
                    raise ValueError('Incomplete actual greedy model output')
                if strategy != 'diag_base' and sample['action'] not in [a for a in
                    public_actions(active[index]['observations'][-1]) if a.startswith('select ')]:
                    raise ValueError('Forced-select output was not a public select action')
                generated[index] = sample
        for index, item in enumerate(active):
            env = item['env']
            if env.done:
                continue
            before, before_mask = copy.deepcopy(env.info), item['states'][-1]
            sample = generated[index]
            if not sample['forced_observation']:
                env.record_model_call(sample['sample_id']); item['model_outputs'] += 1
            try:
                parsed = parse_action(sample['action'])
            except ValueError:
                parsed = None
                item['syntax_failures'] += int(not sample['forced_observation'])
            observation, reward, done, info = env.step(sample['action'])
            after_mask = mask_state(env)
            item['states'].append(after_mask)
            if item['first_seen'] is None and after_mask['anchor_pixels'] >= 250:
                item['first_seen'] = env.steps
            if item['first_positive'] is None and after_mask['anchor_pixels'] > 0:
                item['first_positive'] = env.steps
            item['task_reward'] += reward
            if parsed is not None and parsed.kind in ('turn_left', 'turn_right', 'move'):
                item['observation_attempts'] += 1
                item['observation_reward_cost'] += max(0., -reward)
            journal({'record_kind': 'decision', 'strategy': strategy, 'dataset_key': item['key'],
                'world_id': item['world'], 'episode_id': before['episode_id'], 'decision_index': decision,
                'before_state': before_mask, 'after_state': after_mask, 'sample': sample,
                'public_rgb_sha256': item['observations'][-1].image.sha256,
                'public_text': item['observations'][-1].text, 'reward': reward,
                'action_valid': info.get('action_valid'), 'tau': info['tau'],
                'logged_x_seen_step': info['x_seen_step'], 'done': done,
                'selected_object': info['selected_name'], 'success': info['success']})
            item['responses'].append(sample['response']); item['observations'].append(observation)
            if done:
                row = {'record_kind': 'episode', 'strategy': strategy, 'dataset_key': item['key'],
                    'pair_id': env.backend.table.pair_id, 'world_id': item['world'],
                    'original_world_id': f'w{env.backend.world}', 'world_order': list(env.backend.world_order),
                    'episode_id': info['episode_id'], 'pair_kind': info['kind'],
                    'actions': list(item['responses']), 'states': item['states'],
                    'selected_object': info['selected_name'], 'selected': info['selected'],
                    'success': bool(info['success']), 'steps': env.steps, 'model_calls': env.model_calls,
                    'observation_count': info['obs_count'], 'observation_attempts': item['observation_attempts'],
                    'observation_reward_cost': item['observation_reward_cost'], 'task_reward': item['task_reward'],
                    'tau': info['tau'], 'logged_x_seen_step': info['x_seen_step'],
                    'independent_first_seen_step': item['first_seen'],
                    'seen_before_select': bool(info['selected'] and item['first_seen'] is not None
                        and item['first_seen'] < env.steps),
                    'seen_anywhere': item['first_seen'] is not None,
                    'positive_anchor_anywhere': item['first_positive'] is not None,
                    'syntax_failures': item['syntax_failures'], 'model_outputs': item['model_outputs']}
                journal(row); episodes.append(row)
        if all(item['env'].done for item in active):
            break
    if len(episodes) != len(active) or not all(item['env'].done for item in active):
        raise ValueError('D3 episode did not end within the unchanged D+1 budget')
    return episodes


def summarize(episodes):
    result = {}
    for kind in (*KINDS, 'look_all', 'all'):
        rows = [r for r in episodes if kind == 'all' or r['pair_kind'] == kind
            or (kind == 'look_all' and r['pair_kind'].startswith('look_'))]
        n = len(rows)
        result[kind] = {'episodes': n, 'successes': sum(r['success'] for r in rows),
            'success_rate': sum(r['success'] for r in rows) / n if n else None,
            'observation_count_mean': sum(r['observation_count'] for r in rows) / n if n else None,
            'observation_reward_cost_mean': sum(r['observation_reward_cost'] for r in rows) / n if n else None,
            'task_reward_mean': sum(r['task_reward'] for r in rows) / n if n else None,
            'seen_before_select_count': sum(r['seen_before_select'] for r in rows),
            'seen_before_select_rate': sum(r['seen_before_select'] for r in rows) / n if n else None,
            'seen_anywhere_count': sum(r['seen_anywhere'] for r in rows),
            'syntax_failures': sum(r['syntax_failures'] for r in rows),
            'model_outputs': sum(r['model_outputs'] for r in rows)}
    return result


def eval_worker(config, connection):
    os.sched_setaffinity(0, config['identity']['cpu_affinity'])
    started = time.monotonic()
    tables, packages, proof = load_pilot_pool(config['manifest_reference'])
    holdout = authenticate_pilot_holdout(config['manifest_reference'], proof, config['holdout_reference'])
    if len(tables) != 83 or set(tables) != set(proof['train_keys']) | set(proof['eval_keys']):
        raise ValueError('DIAG01 requires exactly the original83 train+heldout population')
    if any(table.scene not in {f'FloorPlan{x}' for lo in (1, 201, 301, 401) for x in range(lo, lo + 20)} for table in tables.values()):
        raise ValueError('No test scene may enter D3')
    authenticated_seconds = time.monotonic() - started
    model = model_reference(config['model_source'], config['identity']['scratch'])
    started = time.monotonic(); policy = VllmPolicy(model['directory'])
    model_seconds = time.monotonic() - started
    first = sorted(key for key, value in tables.items() if value.kind.startswith('look_'))[0]
    batches = [[first], sorted(set(tables) - {first})]
    records = {strategy: [] for strategy in STRATEGIES}
    logs = Path(config['episode_directory'])
    streams = {s: (logs / (s + '.jsonl')).open('x') for s in STRATEGIES}
    progress = (logs / 'progress.jsonl').open('x')
    try:
        for stage, keys in enumerate(batches):
            for strategy in STRATEGIES:
                if time.time() >= config['deadline'] - 30:
                    raise TimeoutError('D3 allocation deadline; preserve completed episodes')
                backends, environments = [], []
                for key in keys:
                    for world in ('w0', 'w1'):
                        backend = TableObservationBackend({key: tables[key]}, world_orders={key: proof['world_orders'][key]})
                        backends.append(backend)
                        environments.append((key, world, TableEnvironment(backend)))
                def journal(row):
                    streams[strategy].write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                    streams[strategy].flush()
                try:
                    rows = collect(strategy, environments, policy, journal)
                finally:
                    for backend in backends:
                        backend.close()
                if len(rows) != 2 * len(keys):
                    raise ValueError('Missing original pair/world evaluation')
                records[strategy].extend(rows)
                progress.write(json.dumps({'stage': stage, 'strategy': strategy, 'episodes_complete': len(rows),
                    'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}) + '\n'); progress.flush()
            if stage == 0:
                if any(len(records[s]) != 2 for s in STRATEGIES):
                    raise ValueError('First-pair engineering unit did not complete all strategies')
                progress.write(json.dumps({'first_pair_engineering_passed': True, 'dataset_key': first,
                    'check': 'four strategies; both worlds; original D+1; real model outputs; independent masks; complete fields'}) + '\n'); progress.flush()
    finally:
        for stream in streams.values():
            stream.close()
        progress.close()
    expected = {(key, world) for key in tables for world in ('w0', 'w1')}
    for strategy, rows in records.items():
        if len(rows) != 166 or {(r['dataset_key'], r['world_id']) for r in rows} != expected:
            raise ValueError('Full83 strategy population coverage mismatch')
    summary = {s: summarize(rows) for s, rows in records.items()}
    return {'completed': True, 'evaluation_design': 'original83 pairs, both worlds once, greedy; equal world weights; training scenes only',
        'strategies': summary, 'first_pair_engineering_passed': True, 'first_pair_dataset_key': first,
        'trigger_privileged': True, 'trigger_rule': 'fixed L,R,R,L; after own mask >=250 complete current excursion to s0 then select',
        'look_trigger_minus_never_success_rate': summary['obs_trigger']['look_all']['success_rate'] - summary['obs_never']['look_all']['success_rate'],
        'source_authentication_seconds': authenticated_seconds, 'model_loading_seconds': model_seconds,
        'model_reference': model, 'fixed_holdout': holdout, 'common_pool_hash': proof['common_pool_hash'],
        'sources': [*proof['sources'], model['source_result']], 'paper_result': False,
        'formal_test_set_evaluation': False, 'training_performed': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True); p.add_argument('--manifest', required=True)
    p.add_argument('--holdout-reference', required=True); p.add_argument('--model-source', required=True)
    p.add_argument('--timeout-seconds', type=float, default=3240.)
    p.add_argument('--gate-timeout-seconds', type=float, default=180.)
    args = p.parse_args()
    if not re.fullmatch(r'diag_diag01_baselines_[A-Za-z0-9_]+', args.name) or not 300 <= args.timeout_seconds <= 3540:
        p.error('Require a fresh D3 namespace and a sub-hour single-GPU driver')
    output = ROOT / 'results' / (args.name + '.json')
    if output.exists():
        raise FileExistsError('Preserve previous diagnostic results')
    config = {**vars(args), 'version': VERSION, 'gpus': 1, 'node': socket.gethostname(),
        'manifest_reference': ref_identity(json.loads(args.manifest)),
        'holdout_reference': ref_identity(json.loads(args.holdout_reference)),
        'model_source': ref_identity(json.loads(args.model_source)), 'model': MODEL,
        'strategies': list(STRATEGIES), 'maximum_requested_gpu_hours': 1.,
        'training_config_changed': False, 'model_settings_reused_from': 'src.tools.smoke_e2e_eval.VllmPolicy'}
    metrics = {'kind': 'diag01_original_model', 'completed': False, 'errors': [], 'cleanup_errors': [],
        'black_frame_detected': False, 'paper_result': False}
    sources, worker, code = [], None, 1
    install_signals()
    try:
        identity = execution_environment(os.environ, os.sched_getaffinity(0)); config['identity'] = identity
        deadline = deadline_after(args.timeout_seconds)
        signal.setitimer(signal.ITIMER_REAL, max(.1, deadline - time.time()))
        scratch = Path(identity['scratch']) / 'diag01' / args.name / identity['pbs_jobid']
        scratch.mkdir(parents=True, exist_ok=False)
        logs = ROOT / 'logs/runs' / args.name; logs.mkdir(parents=True, exist_ok=False)
        config['episode_directory'] = str(logs)
        gate, gate_ref, cleanup = logged_gate(args.name + '_gate', args.gate_timeout_seconds, scratch, deadline, sources)
        metrics.update(gate=gate_ref, cleanup_errors=cleanup)
        if gate['_meta']['exit_code'] != 0:
            metrics['black_frame_detected'] = gate['_meta']['exit_code'] == 86
            code = 86 if metrics['black_frame_detected'] and not cleanup else 1
        else:
            gate_runtime(gate, identity, config['node'], scratch / 'unity_logs')
            if cleanup:
                raise RuntimeError('Uncertain gate cleanup')
            worker = Worker(eval_worker, {**config, 'deadline': deadline - 20})
            while worker.result is None and worker.error is None and time.time() < deadline - 15:
                worker.receive(); time.sleep(.1)
            worker.receive()
            if worker.error or worker.result is None:
                raise RuntimeError('D3 worker did not finish: ' + str(worker.error))
            result = copy.deepcopy(worker.result)
            sources.extend(ref for ref in result.pop('sources', []) if ref not in sources)
            metrics.update(result); code = 0 if metrics['completed'] else 1
    except BaseException as error:
        traceback.print_exc(); metrics['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if worker is not None:
            metrics['cleanup_errors'].extend(stop_workers([worker]))
        if metrics['errors'] or metrics['cleanup_errors']:
            code = 1; metrics['completed'] = False
        metrics['journals'] = [file_ref(path, ROOT / 'logs') for path in sorted((ROOT / 'logs/runs' / args.name).glob('*.jsonl'))]
        metrics['producer_exit_code'] = code
        save_json(output, {'metrics': metrics, 'config': config, 'sources': sources})
        print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
