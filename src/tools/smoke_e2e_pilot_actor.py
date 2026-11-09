"""Add frozen T4 gradients immediately before the pinned PPO optimizer step.

The original update_policy, PPO/actor-KL losses, microbatch accumulation,
gradient clipping, AdamW and scheduler are executed unchanged. This scope is
one GPU, one sequence-parallel rank and the exact fixed pilot minibatch layout.
"""
from contextlib import contextmanager
import time

from src.rl import t4_consistency as consistency


def _device_copy(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _device_copy(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_device_copy(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_device_copy(item, device) for item in value)
    return value


def candidate_scores(actor, bundle, *, temperature):
    """Same pinned full-action forward used by Q_commit, now with gradients."""
    import torch
    from src.tools.smoke_e2e_training_contract import decision_token_spans
    batch = bundle['batch']
    micro = {**_device_copy(dict(batch.batch), torch.cuda.current_device()),
             **{key: [_device_copy(item, torch.cuda.current_device()) for item in values]
                for key, values in batch.non_tensor_batch.items() if key == 'multi_modal_inputs'}}
    _, log_probs = actor._forward_micro_batch(micro_batch=micro, temperature=temperature)
    length = micro['responses'].shape[-1]
    result = {}
    for index, action in enumerate(bundle['actions']):
        spans = decision_token_spans(micro['loss_mask'][index, -length:].tolist(),
            micro['end_of_response_position_mask'][index, -length:].tolist())
        if (len(spans) != bundle['decision_index'] + 1
                or micro['responses'][index, spans[-1]].tolist() != action['expected_complete_token_ids']
                or action['object_name'] in result):
            raise ValueError('T4 action score lacks exact complete select text/EOS/history')
        result[action['object_name']] = log_probs[index, spans[-1]].sum()
    if len(result) != 2:
        raise ValueError('Approved applicable consistency term requires both actual A/B actions')
    return result


def require_completed_optimizer_steps(record):
    """A native skipped-gradient minibatch remains skipped, never a completed update."""
    native, actual = record.get('native_minibatch_steps'), record.get('native_optimizer_step_calls')
    if (type(native) is not int or native <= 0 or type(actual) is not int
            or actual != native):
        raise ValueError('Native PPO did not execute every optimizer minibatch: '
                         f'optimizer_calls={actual}, native_minibatches={native}')


@contextmanager
def update_support(actor, data):
    """Attach one auxiliary accumulation to each actual original minibatch."""
    metadata = data.meta_info.get('pilot_update')
    if not isinstance(metadata, dict) or metadata.get('arm') not in ('T0', 'T2', 'T4'):
        raise ValueError('Pilot worker requires an explicit observed update descriptor')
    episode_ids = list(data.non_tensor_batch['pilot_episode_id'])
    n = int(data.batch.batch_size[0])
    mini, epochs = actor.config.ppo_mini_batch_size, actor.config.ppo_epochs
    if (n != 64 or len(episode_ids) != n or len(set(episode_ids)) != n
            or mini != 32 or type(epochs) is not int or epochs <= 0
            or actor.config.use_dynamic_bsz is not False
            or actor.config.ulysses_sequence_parallel_size != 1):
        raise ValueError('Pilot actor must keep the fixed original 64/32 static minibatch layout')
    descriptors = metadata.get('consistency_descriptors', {})
    if metadata['arm'] == 'T4' and set(descriptors) != set(episode_ids):
        raise ValueError('T4 requires eligible or explicit zero descriptors for all actual episodes')
    record = {'version': consistency.VERSION, 'arm': metadata['arm'],
        'native_minibatch_steps': 0, 'native_optimizer_step_calls': 0,
        'auxiliary_seconds': 0., 'terms': [], 'minibatches': []}
    original_step, original_optimizer = actor._optimizer_step, actor.actor_optimizer.step
    step_had_local = '_optimizer_step' in vars(actor)
    opt_had_local = 'step' in vars(actor.actor_optimizer)
    def optimizer(*args, **kwargs):
        value = original_optimizer(*args, **kwargs)
        record['native_optimizer_step_calls'] += 1
        return value
    def step(*args, **kwargs):
        call = record['native_minibatch_steps']
        if call >= (n // mini) * epochs:
            raise ValueError('Pinned optimizer made an unexpected extra minibatch call')
        offset = (call % (n // mini)) * mini
        members = episode_ids[offset:offset + mini]
        started = time.monotonic()
        values = []
        if metadata['arm'] == 'T4':
            for episode in members:
                descriptor = descriptors[episode]
                if not descriptor['applicable']:
                    continue  # Its zero still belongs to the fixed denominator.
                scores = candidate_scores(actor, metadata['candidate_inputs'][episode],
                                           temperature=data.meta_info['temperature'])
                sampled_other = (metadata['sampled_action_scores'][descriptor['other_episode_id']]
                                 if descriptor['change'] == 'stay' else None)
                value = consistency.term(descriptor, scores, sampled_other_log_probs=sampled_other)
                loss = value * (.1 * descriptor['trajectory_loss_weight'] / mini)
                loss.backward()
                observed = {'episode_id': episode, 'epoch': call // (n // mini),
                    'minibatch_index': call % (n // mini), 'minibatch_N': mini,
                    'change': descriptor['change'], 'c_i': float(value.detach()),
                    'weighted_loss_contribution': float(loss.detach()),
                    'current_action_log_probs': {key: float(item.detach()) for key, item in scores.items()},
                    'sampled_opposite_action_log_probs': sampled_other,
                    'opposite_policy_detached': descriptor['change'] == 'stay'}
                record['terms'].append(observed); values.append(observed['weighted_loss_contribution'])
        record['auxiliary_seconds'] += time.monotonic() - started
        before = record['native_optimizer_step_calls']
        result = original_step(*args, **kwargs)  # Original clip / finite-gradient gate / AdamW.
        record['native_minibatch_steps'] += 1
        record['minibatches'].append({'call_index': call, 'episode_ids': members, 'N': mini,
            'auxiliary_loss': sum(values),
            'optimizer_step_called': record['native_optimizer_step_calls'] > before})
        return result
    actor._optimizer_step, actor.actor_optimizer.step = step, optimizer
    try:
        yield record
        if record['native_minibatch_steps'] != (n // mini) * epochs:
            raise ValueError('Original PPO did not complete its exact expected minibatch calls')
    finally:
        if step_had_local: actor._optimizer_step = original_step
        else: del actor._optimizer_step
        if opt_had_local: actor.actor_optimizer.step = original_optimizer
        else: del actor.actor_optimizer.step
