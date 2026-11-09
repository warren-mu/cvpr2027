"""E2E steps 4/5: five actual pinned VAGEN PPO updates on reviewed direct pairs.

Run via run_logged only after step 3 passes. Each invocation owns one PBS GPU.
The explicit format-policy flag must reflect the human's reward decision.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

from src.envs.thor_common import ROOT, Worker, deadline_after, install_signals, stop_workers
from src.tools.t01_partition import audited_result
from src.tools.t02_export import save_json, execution_environment, logged_gate, gate_runtime
from src.tools.smoke_e2e_views import file_ref
from src.tools.smoke_e2e_eval import load_replay, MODEL, STRATEGIES
from src.tools.smoke_e2e_replay import python_args
from src.tools.smoke_e2e_training_contract import VERSION, VAGEN_COMMIT, VERL_COMMIT, digest, require_direct
from src.tools.ray_scratch import reserve_directory


def require_eval(path):
    data, ref = audited_result(path)
    m, cfg, meta = data['metrics'], data['config'], data['_meta']
    args = python_args(meta['command'])
    def argument(flag):
        if args.count(flag) != 1 or args.index(flag) + 1 >= len(args):
            raise ValueError('Missing/repeated E2E producer argument: ' + flag)
        return args[args.index(flag) + 1]
    if (args[:2] != ['-m', 'src.tools.smoke_e2e_eval'] or m.get('step3_passed') is not True
            or argument('--name') != meta['name'] or argument('--replay') != cfg['replay']
            or meta['exit_code'] != 0 or m.get('producer_exit_code') != 0 or m.get('errors') != []
            or m.get('cleanup_errors') != [] or m.get('model') != MODEL or m.get('paper_result') is not False):
        raise ValueError('E2E training requires the successful audited original-model step 3')
    loaded = load_replay(ROOT / cfg['replay'])
    worlds = {(pair, world) for pair in loaded[0] for world in ('w0', 'w1')}
    for strategy in STRATEGIES:
        rows = m.get('strategies', {}).get(strategy, [])
        if len(rows) != len(worlds) or {(row.get('pair_id'), row.get('world_id')) for row in rows} != worlds:
            raise ValueError('Step3 must evaluate every reviewed world with all three policies')
    require_direct(loaded[0])
    return data, ref, loaded


def upstream_roots(scratch):
    roots = {'VAGEN': (ROOT / 'third_party/VAGEN', VAGEN_COMMIT),
             'verl': (Path(scratch) / 'sources' / ('verl-' + VERL_COMMIT), VERL_COMMIT)}
    result = {}
    for name, (path, commit) in roots.items():
        if not path.is_dir():
            raise ValueError('Pinned source checkout missing: ' + str(path))
        found = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if found != commit:
            raise ValueError(name + ' differs from the previously reviewed upstream commit')
        result[name] = {'path': str(path), 'commit': found}
        sys.path.insert(0, str(path))
    return result


def build_config(model, checkpoint, *, roots, critic_offload):
    from omegaconf import OmegaConf
    path = Path(roots['VAGEN']['path']) / 'vagen/trainer/config/ppo_trainer.yaml'
    config = OmegaConf.load(path)
    values = {
        'data.train_batch_size': 64, 'data.val_batch_size': 8, 'data.max_prompt_length': 1024,
        'data.max_response_length': 256, 'data.max_trajectory_length': 5000, 'data.truncation': 'error',
        'algorithm.adv_estimator': 'masked_gae', 'algorithm.high_level_gamma': .95,
        'algorithm.kl_ctrl.kl_coef': .001,
        'actor_rollout_ref.model.path': str(model), 'actor_rollout_ref.model.use_remove_padding': True,
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.optim.lr': 1e-6, 'actor_rollout_ref.actor.ppo_mini_batch_size': 32,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.actor.use_kl_loss': False, 'actor_rollout_ref.actor.kl_loss_coef': .001,
        'actor_rollout_ref.actor.kl_loss_type': 'mse',
        'actor_rollout_ref.actor.fsdp_config.param_offload': True,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': True,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'actor_rollout_ref.rollout.gpu_memory_utilization': .1,
        'actor_rollout_ref.rollout.enable_chunked_prefill': False,
        'actor_rollout_ref.rollout.enforce_eager': True,
        'actor_rollout_ref.rollout.free_cache_engine': False,
        'actor_rollout_ref.rollout.n': 1, 'actor_rollout_ref.rollout.temperature': .7,
        'actor_rollout_ref.rollout.seed': 0,
        'actor_rollout_ref.rollout.top_p': .95, 'actor_rollout_ref.rollout.max_num_seqs': 64,
        'actor_rollout_ref.rollout.limit_mm_per_prompt': 5,
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.ref.fsdp_config.param_offload': True,
        'critic.optim.lr': 1e-5, 'critic.model.path': str(model),
        'critic.model.use_remove_padding': True, 'critic.model.enable_gradient_checkpointing': True,
        'critic.ppo_micro_batch_size_per_gpu': 1,
        'critic.model.fsdp_config.param_offload': critic_offload,
        'critic.model.fsdp_config.optimizer_offload': critic_offload,
        'trainer.critic_warmup': 0, 'trainer.logger': ['console'],
        'trainer.project_name': 'cvpr2027_smoke_e2e', 'trainer.experiment_name': 'direct_only_ppo',
        'trainer.n_gpus_per_node': 1, 'trainer.nnodes': 1, 'trainer.total_training_steps': 5,
        'trainer.resume_mode': 'disable', 'trainer.save_freq': -1, 'trainer.test_freq': 20,
        'trainer.val_before_train': False, 'trainer.default_local_dir': str(checkpoint),
        'rollout_manager.max_turns': 5, 'rollout_manager.window_size': 5,
        'rollout_manager.use_multi_turn_reward': True, 'rollout_manager.use_loss_mask': True,
        'rollout_manager.use_gae_mask': True, 'rollout_manager.n_trajectory': 1,
        'rollout_manager.use_service': False}
    for key, value in values.items():
        OmegaConf.update(config, key, value)
    return config, {'base_yaml_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'explicit_overrides': values}


def train_worker(config, connection):
    os.sched_setaffinity(0, config['identity']['cpu_affinity'])
    os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
    source, eval_ref, loaded = require_eval(ROOT / config['eval'])
    roots = upstream_roots(config['identity']['scratch'])
    model = Path(source['metrics']['model_snapshot_directory']).resolve()
    scratch = Path(config['identity']['scratch']).resolve()
    if not model.is_relative_to(scratch) or not (model / 'config.json').is_file():
        raise ValueError('Use the exact step3 model snapshot on scratch')
    if hashlib.sha256((model / 'config.json').read_bytes()).hexdigest() != source['metrics']['model_config_sha256']:
        raise ValueError('Step3 model snapshot changed')
    from omegaconf import OmegaConf
    import ray
    from verl.utils import hf_tokenizer, hf_processor
    from verl.single_controller.ray import RayWorkerGroup
    from vagen.trainer.ppo.ray_trainer import Role, ResourcePoolManager
    from src.tools.smoke_e2e_training_runtime import ObservedActorWorker, ObservedCriticWorker, SmokeTrainer
    smoke_path = ROOT / 'configs/smoke_e2e_training.json'
    smoke_spec = json.loads(smoke_path.read_text())
    if any(smoke_spec.get(key) != expected for key, expected in {
            'version': VERSION, 'updates': 5, 'episodes_per_update': 64, 'turns': 5,
            'response_tokens': 256, 'rho': .5, 'seed': 0, 'model': MODEL}.items()):
        raise ValueError('Published smoke configuration disagrees with the bounded adapter')
    actual, config_proof = build_config(model, config['checkpoint_directory'], roots=roots,
                                        critic_offload=config['critic_offload'])
    stable = OmegaConf.to_container(actual, resolve=True)
    stable['trainer']['default_local_dir'] = '<run_specific_scratch_final_checkpoint>'
    common = {'upstream': {key: value['commit'] for key, value in roots.items()},
        'config': stable, 'format_policy': config['format_policy'], 'eval_source': eval_ref,
        'adapter_version': VERSION, 'published_smoke_spec': file_ref(smoke_path, ROOT / 'configs'),
        'vllm_attention_backend': 'XFORMERS'}
    common_hash = digest(common)
    config_file = Path(config['training_directory']).parent / 'resolved_smoke_config.json'
    with config_file.open('x') as stream:
        json.dump({'common_config': common, 'common_config_sha256': common_hash,
                   'upstream_proof': config_proof}, stream, indent=2, allow_nan=False)
    baseline, baseline_ref = None, None
    if config['arm'] == 'T3':
        baseline, baseline_ref = audited_result(ROOT / config['baseline'])
        if (baseline['_meta']['exit_code'] != 0 or baseline['metrics'].get('step4_passed') is not True
                or python_args(baseline['_meta']['command'])[:2] != ['-m', 'src.tools.smoke_e2e_train']
                or baseline['config'].get('arm') != 'T0'
                or baseline['metrics'].get('producer_exit_code') != 0
                or baseline['metrics'].get('errors') != [] or baseline['metrics'].get('cleanup_errors') != []
                or baseline['metrics'].get('actual_completed_updates') != 5
                or baseline['metrics'].get('common_smoke_config_sha256') != common_hash):
            raise ValueError('T3 requires the successful T0 with exactly the same common smoke config and reviewed pool')
    ray_record = reserve_directory(scratch, config['name'], config['identity']['pbs_jobid'],
        Path(config['checkpoint_directory']).parent, config_file.parent / 'ray_temp.json',
        importlib.metadata.version('ray'))
    os.environ['RAY_TMPDIR'] = ray_record['ray_tmpdir']
    started = time.monotonic()
    try:
        ray.init(num_cpus=len(config['identity']['cpu_affinity']), num_gpus=1,
                 include_dashboard=False, runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true',
                 'NCCL_DEBUG': 'WARN', 'WANDB_MODE': 'offline', 'PYTHONHASHSEED': '0',
                 'VLLM_ATTENTION_BACKEND': 'XFORMERS',
                 'PYTHONPATH': os.pathsep.join([str(ROOT), *(row['path'] for row in roots.values()),
                                               os.environ.get('PYTHONPATH', '')])}})
        tokenizer, processor = hf_tokenizer(str(model)), hf_processor(str(model), use_fast=True)
        workers = {Role.ActorRollout: ray.remote(ObservedActorWorker), Role.Critic: ray.remote(ObservedCriticWorker),
                   Role.RefPolicy: ray.remote(ObservedActorWorker)}
        pool = ResourcePoolManager({'smoke': [1]}, {role: 'smoke' for role in workers})
        trainer = SmokeTrainer(config=actual, tokenizer=tokenizer, processor=processor,
            role_worker_mapping=workers, resource_pool_manager=pool, ray_worker_group_cls=RayWorkerGroup)
        trainer.init_workers()
        loaded_seconds = time.monotonic() - started
        result = trainer.smoke_fit(loaded=loaded, arm=config['arm'], format_policy=config['format_policy'],
            directory=config['training_directory'], checkpoint_directory=config['checkpoint_directory'],
            config_sha256=common_hash)
        if baseline is not None:
            result['per_update_time_ratio_vs_T0'] = [
                row['total_seconds'] / prior['total_seconds']
                for row, prior in zip(result['updates'], baseline['metrics']['updates'])]
        result.update(step4_passed=config['arm'] == 'T0', step5_passed=config['arm'] == 'T3',
            common_smoke_config_sha256=common_hash, upstream=roots,
            model_initialization_seconds=loaded_seconds, resolved_config=file_ref(config_file, ROOT / 'logs'),
            gamma=actual.algorithm.gamma, gae_lambda=actual.algorithm.lam,
            reward_policy=config['format_policy'], checkpoint_saved=True)
        sources = loaded[-1] + [eval_ref]
        if baseline_ref:
            sources.append(baseline_ref)
        result['sources'] = sources
        return result
    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--eval', required=True)
    parser.add_argument('--arm', choices=('T0', 'T3'), required=True)
    parser.add_argument('--baseline')
    parser.add_argument('--format-policy', choices=('explicit_task_zero', 'vagen_valid_bonus'), required=True)
    parser.add_argument('--critic-offload', action='store_true')
    parser.add_argument('--timeout-seconds', type=float, required=True)
    parser.add_argument('--gate-timeout-seconds', type=float, default=180.)
    args = parser.parse_args()
    if (not re.fullmatch(r'smoke_e2e_[A-Za-z0-9_]+', args.name) or not 300 <= args.timeout_seconds <= 10800
            or not 15 <= args.gate_timeout_seconds <= args.timeout_seconds - 60
            or (args.arm == 'T3') != bool(args.baseline)):
        parser.error('Bounded smoke name/deadline required; T3 additionally requires its audited T0 baseline')
    output = ROOT / 'results' / (args.name + '.json')
    if output.exists():
        raise FileExistsError('Preserve previous result; choose a fresh run name')
    config = {**vars(args), 'version': VERSION, 'gpus': 1, 'model': MODEL,
              'node': socket.gethostname(), 'updates': 5, 'episodes_per_update': 64, 'seed': 0}
    metrics = {'kind': 'smoke_e2e_training', 'step4_passed': False, 'step5_passed': False,
               'errors': [], 'cleanup_errors': [], 'black_frame_detected': False,
               'formal_T04_T05_acceptance': None, 'paper_result': False}
    sources, worker, code = [], None, 1
    install_signals()
    try:
        identity = execution_environment(os.environ, os.sched_getaffinity(0)); config['identity'] = identity
        deadline = deadline_after(args.timeout_seconds)
        signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
        scratch = Path(identity['scratch']) / 'smoke_e2e' / args.name / identity['pbs_jobid']
        scratch.mkdir(parents=True, exist_ok=False)
        logs = ROOT / 'logs/runs' / args.name
        logs.mkdir(parents=True, exist_ok=False)
        config.update(training_directory=str(logs / 'ppo'), checkpoint_directory=str(scratch / 'checkpoint'))
        gate, gate_ref, cleanup = logged_gate(args.name + '_gate', args.gate_timeout_seconds, scratch, deadline, sources)
        metrics.update(gate=gate_ref, cleanup_errors=cleanup)
        if gate['_meta']['exit_code'] != 0:
            metrics['black_frame_detected'] = gate['_meta']['exit_code'] == 86
            code = 86 if metrics['black_frame_detected'] and not cleanup else 1
        else:
            gate_runtime(gate, identity, config['node'], scratch / 'unity_logs')
            if cleanup:
                raise RuntimeError('Gate cleanup uncertain')
            worker = Worker(train_worker, {**config, 'deadline': deadline - 20})
            while worker.result is None and worker.error is None and time.time() < deadline - 15:
                worker.receive(); time.sleep(.1)
            worker.receive()
            if worker.error or worker.result is None:
                raise RuntimeError('PPO worker did not finish: ' + str(worker.error))
            result = worker.result
            sources.extend(ref for ref in result.pop('sources', []) if ref not in sources)
            metrics.update(result)
            code = 0 if metrics.get('step4_passed' if args.arm == 'T0' else 'step5_passed') is True else 1
    except BaseException as error:
        traceback.print_exc(); metrics['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if worker is not None:
            metrics['cleanup_errors'].extend(stop_workers([worker]))
        if metrics['errors'] or metrics['cleanup_errors']:
            code = 1; metrics.update(step4_passed=False, step5_passed=False)
        logs = ROOT / 'logs/runs' / args.name
        metrics['training_logs'] = [file_ref(path, ROOT / 'logs') for path in sorted(logs.rglob('*.json*'))]
        metrics['producer_exit_code'] = code
        save_json(output, {'metrics': metrics, 'config': config, 'sources': sources})
        print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
