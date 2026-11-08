"""Approved supplement-4 Look observability, separate from frozen evidence rules.

The selected common first action keeps the existing anchor-union criterion.
Each world must additionally have an actually observed, legal one-step view of
X. These two visibility witnesses may use different actions. No old rule changes.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

from . import pair_v2_policy as original
from . import pair_v2_stable_policy as stable

SOLVABILITY_VERSION = 'look_per_world_visible_action_v1'
SOLVABILITY_POLICY = {
    'version': SOLVABILITY_VERSION,
    'source': 'tasks/T01v2.md supplement 4; explicitly human approved',
    'scope': 'Look only; Direct unchanged',
    'visible_pixels_at_255': 250,
    'per_world_condition': 'at least one actual existing observation action sees anchor X',
    'same_visibility_action_required': False,
    'action_domain': 'per-world unchanged observation_actions on the same fixed reachable list; deterministic union',
    'all_domain_actions_assessed': True,
    'selected_common_evidence': 'unchanged raw and marked X-mask union channel difference >10 on >=250 pixels',
    'unavailable': 'authenticated native path failure; not an observation and never X=0',
    'missing_evidence': 'unknown, never treated as zero or accepted',
}
_OLD_CLASSIFY = stable.classify_pair
_LAST_REPORT = None


def reset_report():
    global _LAST_REPORT
    _LAST_REPORT = None


def last_report():
    return copy.deepcopy(_LAST_REPORT)


def reference_present(value):
    return (isinstance(value, dict) and isinstance(value.get('file'), str)
            and isinstance(value.get('sha256'), str) and len(value['sha256']) == 64)


def read_reference(reference):
    if not reference_present(reference):
        raise ValueError('Actual native proof reference is missing')
    data = Path(reference['file']).read_bytes()
    if hashlib.sha256(data).hexdigest() != reference['sha256']:
        raise ValueError('Actual native proof hash changed')
    return json.loads(data)


def action_identity(action):
    from .pair_v2_shard import action_text
    public = action_text(action)
    if action['kind'] in ('turn_left', 'turn_right'):
        return public, {'kind': action['kind']}
    return public, {key: copy.deepcopy(action[key]) for key in ('kind', 'index', 'position')}


def action_union(domains):
    indexed = {}
    for domain in domains:
        for action in domain.get('actions') or []:
            public, identity = action_identity(action)
            if public in indexed and action_identity(indexed[public])[1] != identity:
                raise ValueError('One public action denotes different positions across worlds')
            indexed.setdefault(public, copy.deepcopy(action))
    return sorted(indexed.values(), key=lambda action: (
        0 if action['kind'] == 'turn_left' else 1 if action['kind'] == 'turn_right' else 2,
        action.get('index', 0)))


def _query_outcome(raw):
    return {'action': raw.get('action'), 'arguments': copy.deepcopy(raw.get('arguments')),
        'exception': raw.get('exception'),
        'last_action_success': (raw.get('after_metadata') or {}).get('lastActionSuccess'),
        'action_return': copy.deepcopy((raw.get('after_metadata') or {}).get('actionReturn'))}


def reachable_reference(sim, reachable):
    for item in reversed(sim.query_log):
        if item.get('action') != 'GetReachablePositions':
            continue
        raw = read_reference(item['reference'])
        if (raw.get('exception') is None and (raw.get('after_metadata') or {}).get('lastActionSuccess') is True
                and (raw.get('after_metadata') or {}).get('actionReturn') == reachable):
            return copy.deepcopy(item['reference'])
    raise ValueError('Fixed reachable list lacks its actual native query proof')


def enrich_unavailable(sim, unavailable, queries, source):
    result = copy.deepcopy(unavailable)
    for entry in result:
        reference = (entry.get('path') or {}).get('query_reference')
        if reference is None:
            matches = []
            for item in queries:
                if item.get('action') != 'GetShortestPathToPoint':
                    continue
                raw = read_reference(item['reference'])
                arguments = raw.get('arguments') or {}
                if arguments.get('position') == source and arguments.get('target') == entry['position']:
                    matches.append(item['reference'])
            if len(matches) == 1:
                reference = matches[0]
        if reference is not None:
            entry['query_reference'] = copy.deepcopy(reference)
            entry['query_outcome'] = _query_outcome(read_reference(reference))
    return result


def unavailable_for(domain, action):
    if action['kind'] != 'move':
        return None
    matches = [entry for entry in domain.get('unavailable_moves') or []
               if entry.get('position') == action['position']]
    return copy.deepcopy(matches[0]) if len(matches) == 1 else None


def unavailable_proved(entry):
    if not isinstance(entry, dict) or not reference_present(entry.get('query_reference')):
        return False
    outcome = entry.get('query_outcome') or {}
    if outcome.get('action') != 'GetShortestPathToPoint':
        return False
    if (entry.get('path') or {}).get('complete') is False:
        path = entry['path']
        raw = outcome.get('action_return')
        if not native_path_known(raw) or raw != path.get('raw'):
            return False
        from .pair_simulator import path_complete, point
        try:
            incomplete = not path_complete(raw, point(path['source']), point(path['target']), path['endpoint_tolerance_m'])
        except (KeyError, TypeError, ValueError):
            return False
        return incomplete and outcome.get('last_action_success') is True and outcome.get('exception') is None
    return outcome.get('last_action_success') is False and isinstance(outcome.get('exception'), str)


def native_path_known(payload):
    if not isinstance(payload, dict) or type(payload.get('status')) is bool:
        return False
    status, corners = payload.get('status'), payload.get('corners')
    if status not in (0, 1, 2, 'PathComplete', 'PathPartial', 'PathInvalid') or not isinstance(corners, list):
        return False
    if status in (0, 'PathComplete') and len(corners) < 2:
        return False
    return all(isinstance(item, dict) and all(type(item.get(axis)) in (int, float)
        and math.isfinite(item[axis]) for axis in ('x', 'y', 'z')) for item in corners)


def observation_proved(row):
    capture = row.get('capture_reference') or {}
    files = capture.get('files') or {}
    return (row.get('status') == 'observed' and type(row.get('anchor_pixels')) is int
        and 0 <= row['anchor_pixels'] <= 255 * 255
        and row.get('anchor_visible') is (row['anchor_pixels'] >= 250)
        and capture.get('valid') is True and capture.get('black') is False
        and capture.get('rgb_shape') == [255, 255, 3]
        and all(reference_present(files.get(key)) for key in ('metadata', 'rgb', 'masks'))
        and reference_present(row.get('action_reference'))
        and (row.get('restoration') or {}).get('restore_deterministic') is True
        and reference_present((row.get('restoration') or {}).get('restored_snapshot_reference')))


def solvability_summary(report):
    """Validate recorded structure and derive sets; merger remeasures raw masks."""
    reasons, visible = [], [[], []]
    domains, enumerated, rows = report.get('domains'), report.get('enumerated_actions'), report.get('actions')
    complete = bool(report.get('solvability_version') == SOLVABILITY_VERSION
        and report.get('solvability_policy') == SOLVABILITY_POLICY
        and isinstance(report.get('anchor_name'), str)
        and isinstance(domains, list) and len(domains) == 2
        and isinstance(enumerated, list) and isinstance(rows, list))
    if complete:
        try:
            complete = ([action_identity(action)[1] for action in enumerated]
                        == [action_identity(action)[1] for action in action_union(domains)] and len(rows) == len(enumerated))
            for index, domain in enumerate(domains):
                complete = bool(complete and domain.get('world_index') == index
                    and (domain.get('restoration') or {}).get('restore_deterministic') is True
                    and reference_present((domain.get('restoration') or {}).get('restored_snapshot_reference'))
                    and reference_present(domain.get('reachable_reference'))
                    and isinstance(domain.get('actions'), list)
                    and all(unavailable_proved(entry) for entry in domain.get('unavailable_moves') or []))
            for index, (action, row) in enumerate(zip(enumerated, rows)):
                public, identity = action_identity(action)
                worlds = row.get('worlds')
                if (row.get('action_index') != index or row.get('action') != public
                        or action_identity(row['requested_action'])[1] != identity
                        or not isinstance(worlds, list) or len(worlds) != 2):
                    complete = False
                    continue
                for world_index, observed in enumerate(worlds):
                    own = {action_identity(item)[0] for item in domains[world_index].get('actions') or []}
                    if observed.get('world_index') != world_index:
                        complete = False
                    if public in own:
                        valid = observation_proved(observed)
                        complete = bool(complete and valid)
                        if valid and observed['anchor_visible']:
                            visible[world_index].append(public)
                    else:
                        valid = (observed.get('status') == 'not_in_domain'
                            and observed.get('anchor_pixels') is None and observed.get('anchor_visible') is None
                            and observed.get('unavailable') == unavailable_for(domains[world_index], action)
                            and unavailable_proved(observed.get('unavailable')))
                        complete = bool(complete and valid)
        except (KeyError, TypeError, ValueError):
            complete = False
    if not complete:
        reasons.append('look_solvability_unverified')
    values = []
    for index, actions in enumerate(visible):
        value = True if actions else False if complete else None
        values.append(value)
        if value is False:
            reasons.append(f'look_not_solvable_w{index}')
    return {'complete': complete, 'anchor_visible_actions_w0': visible[0],
        'anchor_visible_actions_w1': visible[1], 'solvable_w0': values[0], 'solvable_w1': values[1],
        'passed': complete and all(value is True for value in values), 'reasons': reasons}


def classify_pair(kind, **kwargs):
    result = _OLD_CLASSIFY(kind, **kwargs)
    if not kind.startswith('look'):
        return result
    report = (kwargs.get('evidence') or {}).get('look_solvability') or {}
    checked = solvability_summary(report)
    reasons = sorted(set(result['reasons']) | set(checked['reasons']))
    if (report.get('anchor_name') != (kwargs.get('target_w0') or {}).get('anchor_name')
            or report.get('selected_action') != kwargs.get('common_observation_action')):
        reasons = sorted(set(reasons) | {'look_solvability_selected_identity_changed'})
    selected = [row for row in report.get('actions', []) if row.get('action') == report.get('selected_action')]
    evidence_without_report = {key: value for key, value in (kwargs.get('evidence') or {}).items()
                               if key != 'look_solvability'}
    if len(selected) != 1 or selected[0].get('evidence') != evidence_without_report:
        reasons = sorted(set(reasons) | {'look_solvability_selected_evidence_changed'})
    for key in ('complete', 'anchor_visible_actions_w0', 'anchor_visible_actions_w1', 'solvable_w0', 'solvable_w1'):
        if report.get(key) != checked[key]:
            reasons = sorted(set(reasons) | {'look_solvability_summary_changed'})
    result.update(reasons=reasons, acceptance_passed=not reasons, d1_candidate_passed=not reasons,
                  solvability_version=SOLVABILITY_VERSION, look_solvability=checked)
    return result


def choose_look(sim, worlds, targets, identities, reachable, prefix):
    """Measure every own-domain action; visibility and continuation are separate."""
    global _LAST_REPORT
    from . import pair_v2_shard as shard
    report = {'solvability_version': SOLVABILITY_VERSION, 'solvability_policy': copy.deepcopy(SOLVABILITY_POLICY),
        'anchor_name': identities['anchor_name'], 'domains': [], 'enumerated_actions': [], 'actions': [],
        'complete': False, 'selected_action': None}
    _LAST_REPORT = report
    reachable_ref = reachable_reference(sim, reachable)
    for index, world in enumerate(worlds):
        _, restored = sim.restore_snapshot(world)
        if restored.get('restore_deterministic') is not True:
            raise ValueError('Observation domain restore failed the existing state-level gate')
        before = len(sim.query_log)
        actions, unavailable = shard.observation_actions(sim, reachable, world)
        report['domains'].append({'world_index': index, 'restoration': copy.deepcopy(restored),
            'reachable_positions': copy.deepcopy(reachable), 'reachable_reference': copy.deepcopy(reachable_ref),
            'actions': copy.deepcopy(actions), 'unavailable_moves': enrich_unavailable(sim, unavailable,
                sim.query_log[before:], world['agent']['position'])})
    actions = action_union(report['domains'])
    report['enumerated_actions'] = copy.deepcopy(actions)
    choices = []
    for action_index, action in enumerate(actions):
        public = shard.action_text(action)
        record = {'action_index': action_index, 'action': public, 'requested_action': copy.deepcopy(action),
                  'worlds': [], 'evidence': None, 'passed': False}
        report['actions'].append(record)
        observed, plans, selections = [None, None], [None, None], [None, None]
        for index, (world, target) in enumerate(zip(worlds, targets)):
            item = {'world_index': index, 'status': 'pending', 'anchor_pixels': None, 'anchor_visible': None,
                'capture_reference': None, 'action_reference': None, 'restoration': None,
                'executed_action': None, 'continuation': {'passed': False, 'plan': None, 'selection': None, 'error': None},
                'error': None}
            record['worlds'].append(item)
            own = {shard.action_text(value) for value in report['domains'][index]['actions']}
            if public not in own:
                item.update(status='not_in_domain', unavailable=unavailable_for(report['domains'][index], action))
                continue
            try:
                _, restoration = sim.restore_snapshot(world)
                item['restoration'] = copy.deepcopy(restoration)
                if restoration.get('restore_deterministic') is not True:
                    raise ValueError('Planner snapshot restoration failed the state-level gate')
                execution = copy.deepcopy(action)
                if action['kind'] == 'move':
                    execution['path'] = sim.shortest_path(world['agent']['position'], action['position'])
                    if execution['path'].get('complete') is not True:
                        raise ValueError('Own-domain public move was not feasible after its fresh restore')
                item['executed_action'] = copy.deepcopy(execution)
                capture = shard.marked(sim, sim.observe(execution), identities)
                observed[index] = capture
                pixels = int(original._mask(capture, identities['anchor_name']).sum())
                item.update(status='observed', anchor_pixels=pixels, anchor_visible=pixels >= 250,
                    capture_reference=copy.deepcopy(capture.get('evidence')),
                    action_reference=copy.deepcopy(sim.last_action_reference))
                # A continuation failure does not erase this actual first view,
                # and cannot prevent the other world from being measured.
                try:
                    possible, label, selected = shard.eligible_select(sim, capture, target)
                    plan = [execution]
                    if not possible:
                        reverse = shard.inverse_action(sim, action, world)
                        returned = shard.marked(sim, sim.observe(reverse), identities)
                        possible, label, selected = shard.eligible_select(sim, returned, target)
                        plan.append(reverse)
                    if not possible:
                        raise ValueError('Inverse observation did not reach a valid select')
                    plan.append({'kind': 'select', 'label': label})
                    plans[index], selections[index] = plan, selected
                    item['continuation'].update(passed=True, plan=copy.deepcopy(plan), selection=copy.deepcopy(selected))
                except (TimeoutError, KeyboardInterrupt, SystemExit):
                    raise
                except Exception as error:
                    item['continuation']['error'] = f'{type(error).__name__}: {error}'
            except (TimeoutError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as error:
                item.update(status='failed', error=f'{type(error).__name__}: {error}',
                            action_reference=copy.deepcopy(getattr(sim, 'last_action_reference', None)))
        if all(value is not None for value in observed):
            evidence = stable.look_evidence(*observed, identities['anchor_name'])
            record['evidence'] = copy.deepcopy(evidence)
            record['passed'] = bool(evidence['evidence_passed'] and evidence['marked_evidence_passed']
                                    and all(plan is not None for plan in plans))
            if record['passed']:
                choices.append({'action': copy.deepcopy(action), 'after': observed, 'plans': plans,
                    'evidence': evidence, 'anchor_pixels': [evidence['anchor_pixels_w0'], evidence['anchor_pixels_w1']],
                    'planning': {'method': 'all existing observation actions measured; shortest valid selected common continuation',
                        'action_counts': [len(plan) for plan in plans], 'selection_measurements': selections,
                        'lower_bound_including_evidence_and_select': 2}})
    report.update(solvability_summary(report))
    if not choices or not report['passed']:
        return None, copy.deepcopy(report)
    choice = min(choices, key=lambda item: (sum(map(len, item['plans'])), max(map(len, item['plans'])),
                                           shard.action_text(item['action'])))
    report['selected_action'] = shard.action_text(choice['action'])
    choice['evidence']['look_solvability'] = copy.deepcopy(report)
    choice['planning'].update(first_action_search=copy.deepcopy(report['actions']),
        global_two_action_lower_bound_attained=all(len(plan) == 2 for plan in choice['plans']))
    return choice, copy.deepcopy(report)


def compact_report(report):
    if not isinstance(report, dict):
        return None
    result = {key: copy.deepcopy(report.get(key)) for key in ('solvability_version', 'anchor_name', 'complete',
        'anchor_visible_actions_w0', 'anchor_visible_actions_w1', 'solvable_w0', 'solvable_w1', 'reasons', 'selected_action')}
    result['actions'] = [{'action': row.get('action'), 'worlds': [
        {key: copy.deepcopy(world.get(key)) for key in ('world_index', 'status', 'anchor_pixels', 'anchor_visible', 'error')}
        for world in row.get('worlds', [])]} for row in report.get('actions', [])]
    result['scope'] = 'compact observed counts/status only; full raw proofs may have been omitted by declared retention policy'
    return result
