"""Frozen T05 §1b: raw token GAE, prefix replacement, one batch whitening.

The raw recursion deliberately matches the pinned verl masked-GAE function,
including iteration order and dtype. The upstream function whitens internally,
so it cannot supply the raw tensor by algebraically undoing normalization.
Critic targets stay those of the original recursion. Actor world weights are
returned separately and must multiply PPO token losses, never advantages.

This module consumes authenticated rollout records; it does not infer Q_commit
from labels or fabricate missing scores. T4's auxiliary loss remains separate.
"""
from collections import defaultdict
import math

from src.rl.credit import ARMS, PAIRED_ARMS, CreditContractError, finite, prior, trajectory_loss_weight


VERSION = 't05_section1b_raw_token_gae_prefix_replace_batch_whiten_v1'
VERL_COMMIT = '3f55021e08b98315213e10d10b709c7f6bbe84f7'
VAGEN_COMMIT = '4e2976d87c0f92c97f9a820803e71908cde5fb71'
RETURN_DEFINITION = 'undiscounted_realized_task_reward_suffix_excluding_KL'


def _tensor_inputs(rewards, values, loss_mask, gae_mask):
    import torch
    tensors = (rewards, values, loss_mask, gae_mask)
    if (not all(isinstance(value, torch.Tensor) for value in tensors)
            or rewards.ndim != 2 or any(value.shape != rewards.shape for value in tensors)
            or any(value.device != rewards.device for value in tensors)
            or not rewards.is_floating_point() or not values.is_floating_point()
            or not torch.isfinite(rewards).all() or not torch.isfinite(values).all()):
        raise CreditContractError('Require finite, aligned reward/value tensors and actual response masks')
    for mask in (loss_mask, gae_mask):
        if not torch.all((mask == 0) | (mask == 1)):
            raise CreditContractError('Response masks must be binary')
    # Actual pinned VAGEN assigns the same tensor to both row fields. Reject a
    # changed producer rather than silently changing GAE or its whitening mask.
    if loss_mask.dtype != gae_mask.dtype or not torch.equal(loss_mask, gae_mask):
        raise CreditContractError('Actual VAGEN gae_mask differs from reply loss_mask; frozen equivalence is unverified')
    if int(loss_mask.sum().item()) < 2:
        raise CreditContractError('Upstream unbiased masked whitening needs at least two reply tokens')


def raw_masked_gae(*, token_level_rewards, values, gae_mask, gamma, lam):
    """Pinned masked-GAE arithmetic before whitening, with original returns."""
    import torch
    gamma, lam = finite(gamma, 'gamma'), finite(lam, 'lambda_GAE')
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise CreditContractError('GAE discounts must lie in [0,1]')
    with torch.no_grad():
        batch_size, _ = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        for b in range(batch_size):
            lastgaelam = 0.0
            valid_positions = gae_mask[b].nonzero(as_tuple=True)[0]
            if len(valid_positions) == 0:
                continue
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                if i < len(valid_positions) - 1:
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                else:
                    nextvalue = 0.0
                delta = token_level_rewards[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
            for _, pos in enumerate(valid_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]
    return advantages, returns


def _decisions(records, rewards, loss_mask, arm):
    """Authenticate alignment and recorded task returns without collapsing tokens."""
    import torch
    if not isinstance(records, (list, tuple)):
        raise CreditContractError('Decision records must be an explicit sequence')
    by_row, pair_rows, normalized = defaultdict(list), defaultdict(dict), []
    for original in records:
        if not isinstance(original, dict):
            raise CreditContractError('Decision record must be a mapping')
        row = dict(original)
        for key in ('episode_id', 'pair_id', 'pair_instance_id', 'input_sha256', 'target_name'):
            if not isinstance(row.get(key), str) or not row[key]:
                raise CreditContractError('Missing decision identity: ' + key)
        if (type(row.get('row_index')) is not int or not 0 <= row['row_index'] < rewards.shape[0]
                or type(row.get('decision_index')) is not int or row['decision_index'] < 0
                or row.get('world_id') not in ('w0', 'w1')
                or not isinstance(row.get('action_text'), str) or type(row.get('parse_valid')) is not bool
                or row.get('action_kind') not in ('observe', 'select', 'invalid')
                or (row['parse_valid'] == (row['action_kind'] == 'invalid'))):
            raise CreditContractError('Invalid row/turn/world or syntactic action classification')
        row['rho'] = finite(row.get('rho'), 'rho')
        row['p_world'] = prior(row['rho'])[row['world_id']]
        row['trajectory_loss_weight'] = trajectory_loss_weight(arm, row['world_id'], row['rho'])
        for key in ('tau', 'x_seen_step'):
            if key not in row or (row[key] is not None and (type(row[key]) is not int or row[key] < 0)):
                raise CreditContractError('Require actual T02 temporal boundary: ' + key)
        if (not isinstance(row.get('temporal_proof'), dict) or not row['temporal_proof']
                or (row['x_seen_step'] is not None
                    and (row['tau'] is None or row['x_seen_step'] < row['tau']))):
            raise CreditContractError('Missing/inconsistent T02 temporal proof')
        row['pre_evidence'] = row['tau'] is None or row['decision_index'] < row['tau']
        positions, tokens = row.get('token_indices'), row.get('response_token_ids')
        if (not isinstance(positions, (list, tuple)) or not positions
                or any(type(index) is not int or not 0 <= index < rewards.shape[1] for index in positions)
                or list(positions) != sorted(set(positions))
                or not isinstance(tokens, (list, tuple)) or len(tokens) != len(positions)
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise CreditContractError('Require ordered actual reply token indices and token IDs')
        row['token_indices'], row['response_token_ids'] = list(positions), list(tokens)
        actions = row.get('public_select_action_texts')
        if (not isinstance(actions, (list, tuple)) or len(actions) != len(set(actions))
                or any(not isinstance(action, str) or not action.startswith('select ') for action in actions)):
            raise CreditContractError('Require the complete public selectable action inventory, including an empty set')
        row['public_select_action_texts'] = list(actions)
        row['task_reward'], row['G'] = finite(row.get('task_reward'), 'task_reward'), finite(row.get('G'), 'G')
        if 'Q_commit' not in row:
            raise CreditContractError('Record actual Q_commit or an explicit undefined value')
        if row['Q_commit'] is not None:
            row['Q_commit'] = finite(row['Q_commit'], 'Q_commit')
            if not 0 <= row['Q_commit'] <= 1:
                raise CreditContractError('Q_commit is a success probability on the complete select domain')
        by_row[row['row_index']].append(row)
        normalized.append(row)
    episodes = set()
    for row_index in range(rewards.shape[0]):
        rows = sorted(by_row[row_index], key=lambda item: item['decision_index'])
        active = loss_mask[row_index].nonzero(as_tuple=True)[0].tolist()
        if not rows:
            if active:
                raise CreditContractError('Missing decision records for an active trajectory')
            if torch.any(rewards[row_index] != 0):
                raise CreditContractError('A trajectory without reply tokens contains rewards')
            continue
        first = rows[0]
        if first['episode_id'] in episodes:
            raise CreditContractError('Episode identity reused across tensor rows')
        episodes.add(first['episode_id'])
        identity = ('episode_id', 'pair_id', 'pair_instance_id', 'world_id', 'rho')
        if (any(any(row[key] != first[key] for key in identity) for row in rows)
                or [row['decision_index'] for row in rows] != list(range(len(rows)))
                or [index for row in rows for index in row['token_indices']] != active):
            raise CreditContractError('Decision spans must cover the actual reply mask once in trajectory order')
        expected_rewards = torch.zeros_like(rewards[row_index])
        for index, row in enumerate(rows):
            expected_G = sum(item['task_reward'] for item in rows[index:])
            if not math.isclose(row['G'], expected_G, abs_tol=1e-7, rel_tol=1e-7):
                raise CreditContractError('G differs from the actual undiscounted task-reward suffix')
            expected_rewards[row['token_indices'][-1]] = row['task_reward']
        if not torch.equal(rewards[row_index], expected_rewards):
            raise CreditContractError('Token rewards differ from actual task rewards at reply endpoints; KL/reward extras are forbidden')
        if arm in PAIRED_ARMS:
            paired = pair_rows[first['pair_instance_id']]
            if first['world_id'] in paired:
                raise CreditContractError('Duplicate world in one paired draw')
            paired[first['world_id']] = first
    if arm in PAIRED_ARMS:
        for worlds in pair_rows.values():
            if (set(worlds) != {'w0', 'w1'} or worlds['w0']['pair_id'] != worlds['w1']['pair_id']
                    or worlds['w0']['rho'] != worlds['w1']['rho']):
                raise CreditContractError('A paired draw requires both actual worlds with the same pair and prior')
    return normalized


def _fallback(row):
    return not row['parse_valid'] or not row['public_select_action_texts']


def _decision_credit(row):
    if row['Q_commit'] is None:
        raise CreditContractError('A parseable pre-evidence action with selectable candidates requires actual Q_commit')
    outcome = row['G'] if row['action_kind'] == 'observe' else row['task_reward']
    return outcome - row['Q_commit']


def credit_pipeline(*, arm, token_level_rewards, values, loss_mask, gae_mask, decisions, gamma=1.0, lam=1.0):
    """Return detached token targets and separate per-token actor world weights.

    `decisions` uses tensor-row indices after any batch reordering, reply token
    indices relative to `responses`, and actual response IDs including EOS.
    `parse_valid` means syntactic parseability, not semantic action availability.
    The caller must authenticate public inputs, table temporal proofs and Q
    scores. No mean or last-token aggregation is defined for fallback credit.
    """
    import torch
    from verl.utils import torch_functional as verl_F
    if arm not in ARMS:
        raise CreditContractError('Unknown training arm')
    _tensor_inputs(token_level_rewards, values, loss_mask, gae_mask)
    records = _decisions(decisions, token_level_rewards, loss_mask, arm)
    with torch.no_grad():
        raw, returns = raw_masked_gae(token_level_rewards=token_level_rewards, values=values,
                                      gae_mask=gae_mask, gamma=gamma, lam=lam)
        replaced = raw.clone()
        weights = torch.ones_like(raw)
        groups = defaultdict(list)
        for row in records:
            row['credit_mode'] = 'original_token_gae'
            row['same_world_decision_credit'] = None
            row['decision_advantage'] = None
            weights[row['row_index'], :] = row['trajectory_loss_weight']
            if row['pre_evidence'] and arm in ('T2', 'T3', 'T4'):
                if not _fallback(row):
                    row['same_world_decision_credit'] = _decision_credit(row)
                if arm == 'T2':
                    if _fallback(row):
                        row['credit_mode'] = 'same_world_token_gae_fallback'
                    else:
                        row['credit_mode'] = 'same_world_decision_credit_broadcast'
                        row['decision_advantage'] = row['same_world_decision_credit']
                        replaced[row['row_index'], row['token_indices']] = row['decision_advantage']
                else:
                    groups[(row['pair_instance_id'], row['decision_index'])].append(row)
        for key, members in groups.items():
            worlds = {row['world_id']: row for row in members}
            if len(members) != 2 or set(worlds) != {'w0', 'w1'}:
                raise CreditContractError('Missing matched pre-evidence world at ' + repr(key))
            a, b = worlds['w0'], worlds['w1']
            equal_fields = ('pair_id', 'rho', 'tau', 'input_sha256', 'action_text', 'action_kind',
                            'parse_valid', 'response_token_ids', 'public_select_action_texts')
            if (any(a[field] != b[field] for field in equal_fields)
                    or not a.get('shared_sample_id') or a['shared_sample_id'] != b.get('shared_sample_id')
                    or a.get('shared_input_verified_exact') is not True
                    or b.get('shared_input_verified_exact') is not True):
                raise CreditContractError('Paired prefix lacks exact shared public input and sampled reply token alignment')
            if _fallback(a):
                # One tensor is computed once, then assigned to both copies.
                shared = (a['p_world'] * raw[a['row_index'], a['token_indices']]
                          + b['p_world'] * raw[b['row_index'], b['token_indices']])
                mode = 'prior_weighted_same_position_token_gae_fallback'
            else:
                shared = (a['p_world'] * a['same_world_decision_credit']
                          + b['p_world'] * b['same_world_decision_credit'])
                mode = 'prior_weighted_decision_credit_broadcast'
            for row in members:
                replaced[row['row_index'], row['token_indices']] = shared
                row['credit_mode'] = mode
                if not _fallback(row):
                    row['decision_advantage'] = shared
        # Exactly one whole-batch normalization; no world weights here.
        advantages = verl_F.masked_whiten(replaced, loss_mask)
        mean, variance = verl_F.masked_mean(replaced, loss_mask), verl_F.masked_var(replaced, loss_mask)
        for row in records:
            index = (row['row_index'], row['token_indices'])
            row.update(raw_token_advantages=raw[index].cpu().tolist(),
                       replaced_token_advantages=replaced[index].cpu().tolist(),
                       normalized_token_advantages=advantages[index].cpu().tolist(),
                       critic_returns=returns[index].cpu().tolist(),
                       raw_values=values[index].detach().cpu().tolist(),
                       token_task_rewards=token_level_rewards[index].detach().cpu().tolist(),
                       return_definition=RETURN_DEFINITION)
        normalization = {
            'version': VERSION, 'function': 'verl.utils.torch_functional.masked_whiten',
            'verl_commit': VERL_COMMIT, 'vagen_commit': VAGEN_COMMIT,
            'gamma': gamma, 'lambda_GAE': lam, 'masked_whiten_calls': 1,
            'mask': 'actual_reply_loss_mask_equal_to_actual_pinned_VAGEN_gae_mask',
            'mean': mean.item(), 'unbiased_variance': variance.item(), 'epsilon': 1e-8,
            'effective_reply_tokens': int(loss_mask.sum().item()),
            'shift_mean': True, 'world_weighted_normalization': False,
            'actor_world_weight_semantics': 'paired: 2*p(world); single: 1; multiply PPO token loss, never advantage',
            'world_weights_applied_to_advantages': False,
            'critic_target': 'original_raw_token_GAE_plus_values_at_valid_GAE_positions',
            'KL_in_task_rewards': False, 'KL_location': 'actor_loss_low_var_kl',
            'fallback_decision_aggregation': None,
        }
    return {'raw_advantages': raw, 'replaced_advantages': replaced, 'advantages': advantages,
            'returns': returns, 'world_loss_weights': weights, 'decision_records': records,
            'normalization': normalization}
