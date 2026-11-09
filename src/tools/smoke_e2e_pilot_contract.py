"""CPU contracts for the explicitly separate E2E step6 pilot.

The fa24c4e8 learning diagnostic uses all Look episodes and strict seen-before-
parsed-terminal-select timing. These observations do not freeze a method from outcomes.
"""
import math
import random

from src.rl.credit import episode_plan as credit_episode_plan

VERSION = 'smoke_e2e_pilot_training_v1'
ARM_UPDATES = {'T0': 50, 'T4': 50, 'T2': 5}
KINDS = ('direct_flip', 'direct_stay', 'look_flip', 'look_stay')
HOLDOUT_UPDATES = {'T0': (0, 20, 40, 50), 'T4': (0, 20, 40, 50), 'T2': ()}
ENGINEERING_SMOKE = 't4_five_update_engineering_v1'
PAIRED_ARMS = ('T4',)
T4_PENDING_FIELDS = ('symmetric_kl_sum_or_half_sum',
    'flip_per_world_to_training_loss_reduction', 'missing_or_invisible_auxiliary_candidate_policy')


def execution_schedule(arm, engineering_smoke=None):
    """A short prefix of the frozen T4 run, never a replacement pilot arm."""
    if arm not in ARM_UPDATES:
        raise ValueError('Unknown frozen pilot arm')
    if engineering_smoke is None:
        return ARM_UPDATES[arm], HOLDOUT_UPDATES[arm]
    if engineering_smoke != ENGINEERING_SMOKE or arm != 'T4':
        raise ValueError('Engineering smoke must explicitly select the versioned T4 prefix')
    # Keep the original greedy holdout schedule: only update0 is reached.
    # The optimizer still uses the identical50-update horizon in the worker.
    return 5, tuple(update for update in HOLDOUT_UPDATES[arm] if update <= 5)


def training_plan(train_keys, *, arm, update, seed=0):
    keys = sorted(train_keys)
    if (arm not in ARM_UPDATES or type(update) is not int
            or not 1 <= update <= ARM_UPDATES[arm] or seed != 0
            or not keys or len(set(keys)) != len(keys)):
        raise ValueError('Require the explicit pilot arm, unique training keys, seed0 and actual update')
    rng = random.Random(repr((seed, update)))
    draws = [rng.choice(keys) for _ in range(32 if arm in PAIRED_ARMS else 64)]
    rows = credit_episode_plan(arm, draws, rho=.5, rng=rng)
    for index, row in enumerate(rows):
        original = row['pair_instance_id']
        row.update(pair_instance_id=f'{arm}:update{update:03d}:{original}',
                   episode_id=f'{arm}:update{update:03d}:episode{index:03d}',
                   dataset_key=row['pair_id'])
    if len(rows) != 64:
        raise AssertionError('Pilot training must retain the common 64-episode budget')
    return rows


def holdout_plan(eval_keys, *, arm, update):
    keys = sorted(eval_keys)
    if (arm not in HOLDOUT_UPDATES or update not in HOLDOUT_UPDATES[arm]
            or not keys or len(keys) != len(set(keys))):
        raise ValueError('Require an explicitly due pilot holdout endpoint and unique held-out keys')
    return [{'episode_id': f'{arm}:holdout{update:03d}:{index:04d}:{world}',
             'pair_id': key, 'dataset_key': key,
             'pair_instance_id': f'{arm}:holdout{update:03d}:{index:04d}',
             'world_id': world, 'rho': .5, 'trajectory_loss_weight': 1.}
            for index, key in enumerate(keys) for world in ('w0', 'w1')]


def episode_observation(plan, traces, *, kind, final_info, selected_name, success):
    """One actual episode, using syntactic parsing and valid execution separately."""
    if kind not in KINDS or not traces:
        raise ValueError('Need an actual classified episode with its complete decision records')
    if [row['decision_index'] for row in traces] != list(range(len(traces))):
        raise ValueError('Decision records are not the complete contiguous episode')
    for row in traces:
        if (row['episode_id'] != plan['episode_id'] or row['pair_id'] != plan['pair_id']
                or type(row['parse_valid']) is not bool or type(row['action_valid']) is not bool
                or type(row['reward']) not in (int, float) or not math.isfinite(row['reward'])):
            raise ValueError('Episode accounting differs from its actual decision trace')
    observations = [row for row in traces if row['action_kind'] == 'observe' and row['action_valid']]
    # The actual environment terminates on every syntactically parsed select,
    # including a nonexistent/unavailable mark; this is not a parse failure.
    selections = [row for row in traces if row['action_kind'] == 'select' and row['parse_valid']]
    if len(selections) > 1 or bool(selections) != final_info['selected']:
        raise ValueError('Actual terminal selection and episode outcome disagree')
    seen = final_info['x_seen_step']
    premature = bool(selections) and (seen is None or selections[0]['decision_index'] < seen)
    # E2E learning metric includes any parsed terminal select, even an invalid
    # mark. T4 independently requires a legal select for its auxiliary term.
    seen_reason = ('no_parseable_select' if not selections else 'anchor_never_seen' if seen is None
                   else 'anchor_not_strictly_before_select' if seen >= selections[-1]['decision_index'] else None)
    return {**plan, 'kind': kind, 'task_reward_sum': sum(row['reward'] for row in traces),
        'success': bool(success), 'tau': final_info['tau'], 'x_seen_step': seen,
        'selected_name': selected_name,
        'select_decision_index': selections[0]['decision_index'] if selections else None,
        'actual_terminal_selection': bool(selections),
        'valid_terminal_selection': bool(selections and selections[0]['action_valid']),
        'premature_commit': premature if kind.startswith('look_') else None,
        'seen_before_select': (seen_reason is None) if kind.startswith('look_') else None,
        'seen_before_select_reason': seen_reason if kind.startswith('look_') else 'not_Look',
        'valid_action_count': sum(row['action_valid'] for row in traces),
        'valid_observation_actions': len(observations),
        'episode_contains_valid_observation': bool(observations),
        'logical_decisions': len(traces),
        'unparsable_outputs': sum(not row['parse_valid'] for row in traces),
        'unavailable_parsed_actions': sum(row['parse_valid'] and not row['action_valid'] for row in traces),
        'physical_sample_ids': sorted({row['sample_id'] for row in traces})}


def classified_observations(episodes):
    """Retain raw numerators and explicit denominators, including empty classes."""
    if len({row['episode_id'] for row in episodes}) != len(episodes):
        raise ValueError('Duplicate episode in the same reported training/evaluation window')
    rows = {}
    for kind in KINDS:
        selected = [row for row in episodes if row['kind'] == kind]
        n = len(selected)
        decisions = sum(row['logical_decisions'] for row in selected)
        observed = sum(row['episode_contains_valid_observation'] for row in selected)
        actions = sum(row['valid_observation_actions'] for row in selected)
        invalid = sum(row['unparsable_outputs'] for row in selected)
        valid_actions = sum(row['valid_action_count'] for row in selected)
        seen_count = sum(row['seen_before_select'] is True for row in selected) if kind.startswith('look_') else None
        reward = sum(row['task_reward_sum'] for row in selected)
        success = sum(row['success'] for row in selected)
        committed = sum(row['actual_terminal_selection'] for row in selected)
        premature = sum(row['premature_commit'] is True for row in selected) if kind.startswith('look_') else None
        ratio = lambda numerator, denominator: numerator / denominator if denominator else None
        rows[kind] = {'episode_count': n, 'logical_decision_count': decisions,
            'task_reward_sum': reward, 'mean_task_reward': ratio(reward, n),
            'success_count': success, 'success_rate': ratio(success, n),
            'episodes_with_valid_observation': observed,
            'observation_episode_fraction': ratio(observed, n),
            'valid_observation_action_count': actions,
            'observation_action_fraction_of_decisions': ratio(actions, decisions),
            'valid_action_count': valid_actions,
            'observation_action_fraction_of_valid_actions': ratio(actions, valid_actions),
            'mean_valid_observation_actions': ratio(actions, n),
            'actual_terminal_selection_count': committed,
            'valid_terminal_selection_count': sum(row['valid_terminal_selection'] for row in selected),
            'premature_commit_count': premature,
            'seen_before_select_count': seen_count,
            'seen_before_select_fraction_of_episodes': ratio(seen_count, n) if seen_count is not None else None,
            'seen_before_select_skip_reasons': {reason: sum(row['seen_before_select_reason'] == reason for row in selected)
                for reason in ('no_parseable_select', 'anchor_never_seen', 'anchor_not_strictly_before_select')} if kind.startswith('look_') else None,
            'premature_commit_fraction_of_episodes': ratio(premature, n) if premature is not None else None,
            'unparsable_output_count': invalid,
            'unparsable_fraction_of_logical_decisions': ratio(invalid, decisions),
            'unavailable_parsed_action_count': sum(row['unavailable_parsed_actions'] for row in selected),
            'scope': 'observed diagnostic counts; no learning decision or hyperparameter freeze inferred'}
    return rows
