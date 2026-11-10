"""Final actor weights only for smoke/pilot; never a resumable training checkpoint.

The pinned verl 3f55021 FSDP manager unconditionally saves optimizer and RNG
files. This small writer retains its model state-dict representation and HF
configuration/processor output without asking for any optimizer/critic state.
No prior file is removed. Training and the common optimizer horizon are unchanged.
"""
import copy
from pathlib import Path
import warnings

POLICY = 'smoke_pilot_final_actor_bf16_only_v1'


def _state_backend():
    import torch
    from torch.distributed.fsdp import (FullyShardedDataParallel, StateDictType,
                                       ShardedStateDictConfig)
    return torch, FullyShardedDataParallel, StateDictType, ShardedStateDictConfig


def _offload_backend():
    # Same functions used by the pinned ActorRolloutRefWorker.save_checkpoint.
    from verl.workers.fsdp_workers import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
    return load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu


def actor_bf16_state(state, torch):
    """Cast saved floating tensors only; the live actor/state is never mutated."""
    result = copy.copy(state)
    if hasattr(state, '_metadata'):
        result._metadata = copy.deepcopy(state._metadata)
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError('Unsupported single-GPU actor state; do not silently save non-bf16 weights')
        detached = value.detach()
        result[key] = (detached.to(device='cpu', dtype=torch.bfloat16)
                       if value.is_floating_point() else detached.cpu())
    return result


def save_final_actor_model(worker, local_path, global_step):
    """One-GPU end-of-run RPC; preserve parameter-offload state on save failure."""
    if (getattr(worker, '_is_actor', False) is not True or worker.world_size != 1
            or type(global_step) is not int or global_step < 1):
        raise ValueError('Final actor-only save requires one actual actor and a completed update')
    manager = worker.checkpoint_manager
    if (manager.world_size != 1 or manager.rank != 0
            or manager.model is not worker.actor_module_fsdp):
        raise ValueError('Actor checkpoint manager differs from its original one-GPU model')
    destination = Path(local_path)
    # Never overwrite old weights or create a misleading partial replacement.
    destination.mkdir(parents=True, exist_ok=False)
    loaded = False
    restore = None
    try:
        if worker._is_offload_param:
            load, restore = _offload_backend()
            # The pinned loader moves handles incrementally. Even a partial
            # load failure must restore the original offloaded placement.
            loaded = True
            load(worker.actor_module_fsdp)
        torch, fsdp, state_type, state_config = _state_backend()
        torch.distributed.barrier()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with fsdp.state_dict_type(manager.model, state_type.SHARDED_STATE_DICT,
                                      state_config(offload_to_cpu=True)):
                model_state = actor_bf16_state(manager.model.state_dict(), torch)
                target = destination / 'model_world_size_1_rank_0.pt'
                with target.open('xb') as stream:
                    torch.save(model_state, stream)
        del model_state
        torch.distributed.barrier()
        hf = destination / 'huggingface'
        hf.mkdir(exist_ok=False)
        saved_config = copy.deepcopy(manager.model._fsdp_wrapped_module.config)
        saved_config.torch_dtype = torch.bfloat16
        saved_config.save_pretrained(str(hf))
        manager.processing_class.save_pretrained(str(hf))
        torch.distributed.barrier()
        manager.previous_global_step = global_step
        manager.previous_save_local_path = str(destination)
        return {'policy': POLICY, 'actor_directory': str(destination),
                'model_file': str(target), 'global_step': global_step,
                'saved_floating_dtype': 'bfloat16', 'nonfloating_buffer_dtypes_preserved': True,
                'optimizer_saved': False, 'critic_saved': False, 'extra_state_saved': False,
                'intermediate_checkpoints_saved': False,
                'training_resume_supported': False}
    finally:
        if loaded:
            restore(worker.actor_module_fsdp)
