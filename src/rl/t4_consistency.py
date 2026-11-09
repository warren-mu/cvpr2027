"""T05 §4b (fa24c4e8): one current-actor term per eligible episode.

The sampled opposite policy is detached. Inapplicable episodes remain explicit
zero terms in the minibatch denominator; no invisible action is synthesized.
"""
import copy
import math

VERSION = 't05_section4b_directional_consistency_v1'
POLICY = {'version': VERSION, 'approval': 'fa24c4e8', 'beta': .1, 'margin': 1.,
    'symmetric_kl_sum_or_half_sum': 'sum_of_two_per_episode_directional_KLs_to_detached_sampled_other_policy',
    'flip_per_world_to_training_loss_reduction': 'sum_2p_world_c_i_divided_by_all_minibatch_episodes_including_inapplicable_zero',
    'missing_or_invisible_auxiliary_candidate_policy': 'zero_unselectable_Flip_target_or_ineligible_Stay_pair_or_no_legal_select'}


def descriptors(episodes):
    """Use actual logical-world target names after the table world permutation."""
    grouped, identifiers = {}, set()
    for episode in episodes:
        if (episode['episode_id'] in identifiers or episode['world_id'] not in ('w0', 'w1')
                or episode['kind'] not in ('direct_flip', 'look_flip', 'direct_stay', 'look_stay')
                or len(episode['candidate_names']) != 2
                or len(set(episode['candidate_names'])) != 2
                or episode['target_name'] not in episode['candidate_names']):
            raise ValueError('Require unique real paired episodes, stable candidates and mapped targets')
        identifiers.add(episode['episode_id'])
        selected = episode['last_legal_select']
        if selected is not None:
            if (selected.get('action_kind') != 'select' or selected.get('action_valid') is not True
                    or type(selected['decision_index']) is not int or selected['decision_index'] < 0
                    or (selected['x_seen_step'] is not None and
                        (type(selected['x_seen_step']) is not int or selected['x_seen_step'] < 0))
                    or len(selected['public_select_candidate_names']) != len(set(selected['public_select_candidate_names']))
                    or not set(selected['public_select_candidate_names']) <= set(episode['candidate_names'])):
                raise ValueError('Consistency select must be an actual legal decision with its full public domain')
        grouped.setdefault(episode['pair_instance_id'], []).append(episode)
    output = {}
    for members in grouped.values():
        if len(members) != 2 or {r['world_id'] for r in members} != {'w0', 'w1'}:
            raise ValueError('Every consistency instance needs its two actual world episodes')
        left, right = sorted(members, key=lambda row: row['world_id'])
        if (left['kind'] != right['kind'] or left['pair_id'] != right['pair_id']
                or set(left['candidate_names']) != set(right['candidate_names'])
                or left['rho'] != right['rho'] or not 0 < left['rho'] < 1):
            raise ValueError('Paired consistency inputs disagree on source/class/candidates/prior')
        change = left['kind'].split('_')[1]
        if (left['target_name'] == right['target_name']) != (change == 'stay'):
            raise ValueError('Actual mapped targets disagree with the unchanged pair class')
        def seen(row):
            select = row['last_legal_select']
            return (select is not None and select['x_seen_step'] is not None
                    and select['decision_index'] >= select['x_seen_step'])
        for own, other in ((left, right), (right, left)):
            select, opposite = own['last_legal_select'], other['last_legal_select']
            reason = None
            if change == 'flip':
                if select is None: reason = 'no_legal_select'
                elif not seen(own): reason = 'own_anchor_not_seen'
                elif not {own['target_name'], other['target_name']} <= set(select['public_select_candidate_names']):
                    reason = 'flip_required_target_unselectable'
            else:
                if select is None or opposite is None: reason = 'stay_pair_missing_legal_select'
                elif not seen(own) or not seen(other): reason = 'stay_pair_anchor_not_seen'
                elif any(set(row['last_legal_select']['public_select_candidate_names']) != set(row['candidate_names'])
                         for row in (own, other)):
                    reason = 'stay_pair_candidate_unselectable'
            output[own['episode_id']] = {**copy.deepcopy(own), 'version': VERSION,
                'other_episode_id': other['episode_id'], 'other_target_name': other['target_name'],
                'applicable': reason is None, 'skip_reason': reason, 'change': change,
                'trajectory_loss_weight': 2 * (own['rho'] if own['world_id'] == 'w1' else 1 - own['rho'])}
    return output


def _ordered_scores(scores, names, *, tensor):
    import torch
    if set(scores) != set(names):
        raise ValueError('Consistency scores must cover exactly the approved stable candidate domain')
    if tensor:
        values = [scores[name] for name in names]
        if any(not isinstance(value, torch.Tensor) or value.ndim != 0 for value in values):
            raise ValueError('Current actor scores must be actual differentiable scalar tensors')
        result = torch.stack(values)
    else:
        # A tensor reference is deliberately detached even if the caller kept
        # its sampling graph. No epsilon, intersection or invented scores.
        values = [float(scores[name].detach()) if isinstance(scores[name], torch.Tensor)
                  else float(scores[name]) for name in names]
        if any(not math.isfinite(value) or value > 0 for value in values):
            raise ValueError('Invalid actual sampled opposite action scores')
        return values
    if not bool(torch.isfinite(result).all()) or bool((result > 0).any()):
        raise ValueError('Invalid actual current actor action log probabilities')
    return result


def term(descriptor, current_log_probs, *, sampled_other_log_probs=None):
    """Return unweighted c_i; None denotes a declared inapplicable zero term."""
    import torch
    if descriptor['applicable'] is not True:
        if descriptor['skip_reason'] is None:
            raise ValueError('Missing reason for an inapplicable zero term')
        return None
    names = sorted(descriptor['candidate_names'])
    current = _ordered_scores(current_log_probs, names, tensor=True)
    if descriptor['change'] == 'flip':
        own = names.index(descriptor['target_name']); other = names.index(descriptor['other_target_name'])
        return (1. - current[own] + current[other]).clamp_min(0.)
    if descriptor['change'] != 'stay' or sampled_other_log_probs is None:
        raise ValueError('Stay needs its opposite episode actual sampling-policy scores')
    sampled = _ordered_scores(sampled_other_log_probs, names, tensor=False)
    frozen = torch.tensor(sampled, dtype=current.dtype, device=current.device)
    log_p, log_q = torch.log_softmax(current, dim=0), torch.log_softmax(frozen, dim=0)
    return torch.sum(log_p.exp() * (log_p - log_q))


def weighted_minibatch_loss(terms, *, minibatch_episode_ids, descriptors_by_episode):
    """β Σ 2p c_i / N; all actual minibatch rows, not only eligible terms, form N."""
    ids = list(minibatch_episode_ids)
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(descriptors_by_episode):
        raise ValueError('Require every actual minibatch episode exactly once')
    expected = {key for key in ids if descriptors_by_episode[key]['applicable']}
    if set(terms) != expected:
        raise ValueError('Missing, extra or cross-minibatch applicable consistency term')
    if not terms:
        return None
    return sum(value * descriptors_by_episode[key]['trajectory_loss_weight']
               for key, value in terms.items()) * (.1 / len(ids))
