"""CPU-only contracts for the five-update, direct-only E2E PPO smoke.

This does not freeze T04's full-training configuration. In particular it never
defines a decision-level reduction of token GAE for an unseen Look trajectory.
"""
import hashlib
import json
import random

VERSION = 'smoke_e2e_direct_vagen_ppo_v1'
VAGEN_COMMIT = '4e2976d87c0f92c97f9a820803e71908cde5fb71'
VERL_COMMIT = '3f55021e08b98315213e10d10b709c7f6bbe84f7'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def available_actions(text):
    lines = [line for line in text.splitlines() if line.startswith('Available actions: ')]
    if len(lines) != 1:
        raise ValueError('Require one complete public action list')
    values = json.loads(lines[0].split(': ', 1)[1])
    if (not isinstance(values, list) or any(not isinstance(item, str) for item in values)
            or len(set(values)) != len(values)):
        raise ValueError('Malformed complete public action list')
    return values  # Empty final-turn mark set is a legitimate Q-undefined state.


def episode_plan(pair_ids, *, arm, update, seed=0):
    pairs = sorted(pair_ids)
    if arm not in ('T0', 'T3') or not pairs or len(set(pairs)) != len(pairs) or not 1 <= update <= 5:
        raise ValueError('Explicit E2E arm, unique pool and update 1..5 required')
    rng = random.Random((seed, update).__repr__())
    plan = []
    for index in range(32 if arm == 'T3' else 64):
        pair = rng.choice(pairs)
        instance = f'{arm}_u{update:02d}_draw{index:03d}'
        worlds = ('w0', 'w1') if arm == 'T3' else (('w1' if rng.random() < .5 else 'w0'),)
        for world in worlds:
            plan.append({'episode_id': instance + '_' + world, 'pair_id': pair,
                         'pair_instance_id': instance, 'world_id': world, 'rho': .5,
                         'trajectory_loss_weight': 1.})
    return plan


def prompt_identity(token_ids, images):
    """Hash AND exact tuple used for paired comparison; includes all history."""
    image_bytes = tuple((image.mode, image.size, image.tobytes()) for image in images)
    exact = (tuple(token_ids), image_bytes)
    serial = {'tokens': list(token_ids), 'images': [
        {'mode': mode, 'size': size, 'sha256': hashlib.sha256(raw).hexdigest()}
        for mode, size, raw in image_bytes]}
    return exact, digest(serial), serial


def generation_groups(plan, identities, *, paired):
    """Only coalesce the two exact prompts from the same sampled world pair."""
    groups, representatives = [], {}
    for index, (row, exact) in enumerate(zip(plan, identities)):
        key = row['pair_instance_id'] if paired else row['episode_id']
        matched = next((group for group in representatives.get(key, [])
                        if identities[group[0]] == exact), None)
        if matched is None:
            matched = [index]
            groups.append(matched)
            representatives.setdefault(key, []).append(matched)
        else:
            if len(matched) != 1 or plan[matched[0]]['world_id'] == row['world_id']:
                raise ValueError('Deduplication may only join one w0 and one w1')
            matched.append(index)
    if len(plan) != len(identities):
        raise ValueError('Prompt identity inventory mismatch')
    return groups


def decision_token_spans(loss_mask, endpoints):
    """Recover every actual assistant action span without an endpoint guess."""
    if len(loss_mask) != len(endpoints):
        raise ValueError('Mask dimensions differ')
    spans, previous = [], -1
    for end, flag in enumerate(endpoints):
        if not flag:
            continue
        tokens = [i for i in range(previous + 1, end + 1) if loss_mask[i]]
        if not tokens or tokens[-1] != end:
            raise ValueError('Empty/misaligned action response')
        spans.append(tokens)
        previous = end
    if [i for span in spans for i in span] != [i for i, flag in enumerate(loss_mask) if flag]:
        raise ValueError('Truncated or unbound assistant tokens')
    return spans


def require_direct(tables):
    if not tables or any(table.states[table.initial_state].shared or not table.kind.startswith('direct_')
                         for table in tables.values()):
        raise ValueError('This smoke preserves token GAE only for the reviewed tau=0 pairs; Look requires the reviewed pre-evidence adapter')


def task_reward(base_reward, *, action_valid, format_policy):
    # Selection of a policy is an explicit CLI requirement, recorded by the
    # launcher; neither option is inferred from the ambiguous T04 wording.
    if format_policy == 'explicit_task_zero':
        return float(base_reward)
    if format_policy == 'vagen_valid_bonus':
        return float(base_reward) + (.5 if action_valid else 0.)
    raise ValueError('Human-confirmed format policy required')
