"""Owned worker entry for the separate approved E2E step6 pilot.

No model or Ray import occurs before the driver render gate and this worker's
formal dataset reauthentication. Original 5-step entry points remain unchanged.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from src.envs.thor_common import ROOT
from src.tools.smoke_e2e_views import file_ref
from src.tools.smoke_e2e_training_contract import FROZEN_PIPELINE, digest
from src.tools.smoke_e2e_train import build_config, upstream_roots
from src.tools.ray_scratch import reserve_directory
from src.tools.smoke_e2e_pilot_contract import execution_schedule


def pilot_train_worker(config, connection):
    from src.tools.smoke_e2e_pilot_dataset import load_pilot_pool
    from src.tools.smoke_e2e_pilot import load_spec, model_reference
    os.sched_setaffinity(0, config['identity']['cpu_affinity'])
    os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
    spec, spec_ref = load_spec(config['pilot_spec_reference']['file'])
    expected_updates, _ = execution_schedule(config['arm'], config.get('engineering_smoke'))
    if (spec_ref != config['pilot_spec_reference']
            or config['updates'] != expected_updates
            or config['credit_pipeline'] != FROZEN_PIPELINE or config['format_policy'] != 'explicit_task_zero'):
        raise ValueError('Pilot worker differs from its approved immutable driver specification')
    common_path = ROOT / 'configs/rl_common.yaml'
    if file_ref(common_path, ROOT / 'configs') != config['common_config_reference']:
        raise ValueError('Common credit/KL configuration changed after submission')
    started_sources = time.monotonic()
    tables, packages, proof = load_pilot_pool(config['manifest_reference'])
    source_seconds = time.monotonic() - started_sources
    if proof['split_policy'] != spec['split_policy']:
        raise ValueError('Pilot holdout differs from the approved scene split')
    model_ref = model_reference(config['model_snapshot_reference']['source_result'], config['identity']['scratch'])
    if model_ref != config['model_snapshot_reference']:
        raise ValueError('Original pretrained model identity changed')
    model, scratch = Path(model_ref['directory']).resolve(), Path(config['identity']['scratch']).resolve()
    roots = upstream_roots(scratch)
    from omegaconf import OmegaConf
    import ray
    from verl.utils import hf_tokenizer, hf_processor
    from verl.single_controller.ray import RayWorkerGroup
    from vagen.trainer.ppo.ray_trainer import Role, ResourcePoolManager
    from src.tools.smoke_e2e_training_runtime import ObservedCriticWorker
    from src.tools.smoke_e2e_pilot_runtime import PilotActorWorker, PilotTrainer
    actual, config_proof = build_config(model, config['checkpoint_directory'], roots=roots,
        critic_offload=config['critic_offload'], pipeline=FROZEN_PIPELINE)
    changes = {'trainer.total_training_steps': 50,
        'trainer.experiment_name': 'e2e_step6_pilot', 'actor_rollout_ref.actor.optim.total_training_steps': 50,
        'critic.optim.total_training_steps': 50, 'actor_rollout_ref.actor.seed': 0, 'critic.seed': 0}
    for key, value in changes.items():
        OmegaConf.update(actual, key, value, force_add=True)
    config_proof['explicit_pilot_overrides'] = changes
    stable = OmegaConf.to_container(actual, resolve=True)
    stable['trainer']['default_local_dir'] = '<run_specific_scratch_final_checkpoint>'
    common = {'upstream': {key: value['commit'] for key, value in roots.items()}, 'config': stable,
        'pilot_spec_reference': spec_ref, 'common_credit_reference': config['common_config_reference'],
        'manifest_reference': config['manifest_reference'], 'common_pool_hash': proof['common_pool_hash'],
        'model_snapshot_reference': model_ref, 'vllm_attention_backend': 'XFORMERS',
        'format_policy': 'explicit_task_zero', 'credit_pipeline': FROZEN_PIPELINE,
        'optimizer_horizon_updates': 50, 'T2_scope': 'first5updates_of_same_common_horizon'}
    common_hash = digest(common)
    config_file = Path(config['training_directory']).parent / 'resolved_pilot_config.json'
    with config_file.open('x') as stream:
        json.dump({'common_config': common, 'common_config_sha256': common_hash,
                   'upstream_proof': config_proof}, stream, indent=2, allow_nan=False)
    ray_record = reserve_directory(scratch, config['name'], config['identity']['pbs_jobid'],
        Path(config['checkpoint_directory']).parent, config_file.parent / 'ray_temp.json',
        importlib.metadata.version('ray'))
    os.environ['RAY_TMPDIR'] = ray_record['ray_tmpdir']
    initialized = time.monotonic()
    try:
        ray.init(num_cpus=len(config['identity']['cpu_affinity']), num_gpus=1,
            include_dashboard=False, runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN', 'WANDB_MODE': 'offline', 'PYTHONHASHSEED': '0',
                'VLLM_ATTENTION_BACKEND': 'XFORMERS',
                'PYTHONPATH': os.pathsep.join([str(ROOT), *(value['path'] for value in roots.values()),
                                              os.environ.get('PYTHONPATH', '')])}})
        tokenizer, processor = hf_tokenizer(str(model)), hf_processor(str(model), use_fast=True)
        workers = {Role.ActorRollout: ray.remote(PilotActorWorker), Role.Critic: ray.remote(ObservedCriticWorker),
                   Role.RefPolicy: ray.remote(PilotActorWorker)}
        pool = ResourcePoolManager({'pilot': [1]}, {role: 'pilot' for role in workers})
        trainer = PilotTrainer(config=actual, tokenizer=tokenizer, processor=processor,
            role_worker_mapping=workers, resource_pool_manager=pool, ray_worker_group_cls=RayWorkerGroup)
        trainer.init_workers()
        initialization_seconds = time.monotonic() - initialized
        result = trainer.pilot_fit(tables=tables, dataset_proof=proof, arm=config['arm'],
            directory=config['training_directory'], checkpoint_directory=config['checkpoint_directory'],
            common_hash=common_hash, engineering_smoke=config.get('engineering_smoke'))
        result.update(cache_authorized=True, dataset_proof=proof, common_pool_hash=proof['common_pool_hash'],
            common_config_sha256=common_hash, resolved_config=file_ref(config_file, ROOT / 'logs'),
            upstream=roots, source_authentication_seconds=source_seconds,
            model_initialization_seconds=initialization_seconds,
            gamma=actual.algorithm.gamma, gae_lambda=actual.algorithm.lam,
            actor_kl_loss=actual.actor_rollout_ref.actor.kl_loss_type,
            actor_kl_loss_coefficient=actual.actor_rollout_ref.actor.kl_loss_coef,
            critic_offload=config['critic_offload'], reward_policy='explicit_task_zero',
            sources=[*proof['sources'], model_ref['source_result']])
        return result
    finally:
        ray.shutdown()
