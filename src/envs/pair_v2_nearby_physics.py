"""Supplement16 physics scope; original native evidence and old rules stay intact."""
import copy
from contextlib import contextmanager
import math

from . import pair_v2_policy as original
from . import pair_v2_stable as stable

VERSION = 'nearby_nonanchor_0_5m_v1'
POLICY = {
    'version': VERSION, 'approval': '38b15551 supplement16',
    'nearby_radius_m': .5, 'nonanchor_position_limit_m': .01,
    'anchor_position_error_limit_m': .02,
    'distance_origin': 'same settled common baseline object position in both worlds',
    'distance_destination': 'both actual successful native placement requested pivots in complete requested_poses; never original failed spawn points',
    'nearby_scope': 'union of distance<=0.5m to either world request and both declared supporting receptacles',
    'far_objects': 'record original before-to-after and common-baseline displacement; only their nonanchor 1cm rejection is removed',
    'unchanged': 'native legal placement, parent membership, rest, inventories, agent, anchor 2cm, restore, all rendered and witness checks',
    'partial_world': 'nonanchor gate deferred until both actual requests exist; never exported or independently accepted',
}
FIELDS = ('nearby_physics_version', 'nearby_physics_policy')
PAIR_FIELDS = (*FIELDS, 'far_object_drift_max_m', 'far_drift_objects',
               'passes_original_physics_rule', 'nearby_physics_context')
_ORIGINAL_AUGMENT = stable.augment_physical_result


def fields():
    return {'nearby_physics_version': VERSION, 'nearby_physics_policy': copy.deepcopy(POLICY)}


def validate_fields(value):
    if any(value.get(key) != expected for key, expected in fields().items()):
        raise ValueError('Missing or altered supplement16 physics declaration')


def _indexed(objects):
    indexed = original.indexed_objects(objects)
    stable.positions(objects)  # Original finite-position/inventory validation.
    return indexed


def _anchor(constructions):
    edited = constructions[0]['edited_names']
    if len(edited) != 1 or any(c['edited_names'] != edited for c in constructions):
        raise ValueError('Nearby physics requires one identical edited anchor in both worlds')
    return edited[0]


def _context(constructions, baseline, reference):
    anchor = _anchor(constructions)
    indexed = _indexed(baseline)
    if anchor not in indexed:
        raise ValueError('Anchor absent from the common baseline')
    requested, supports = [], []
    for c in constructions:
        poses = {p['objectName']: p for p in c['requested_poses']}
        if len(poses) != len(c['requested_poses']):
            raise ValueError('Duplicate requested native poses')
        point = poses[anchor]['position']
        if any(type(point.get(a)) not in (int, float) or not math.isfinite(point[a]) for a in 'xyz'):
            raise ValueError('Invalid actual requested anchor position')
        requested.append(copy.deepcopy(point))
        support = c['placement_evidence_by_name'][anchor]['receptacle_name']
        if support not in indexed or indexed[support].get('receptacle') is not True:
            raise ValueError('Supporting receptacle absent from common baseline')
        supports.append(support)
    distances = {name: [original.position_distance(obj['position'], point) for point in requested]
                 for name, obj in indexed.items() if name != anchor}
    near = sorted(name for name, ds in distances.items() if min(ds) <= .5 or name in supports)
    return {**fields(), 'finalized': len(constructions) == 2, 'baseline_reference': copy.deepcopy(reference),
            'anchor_name': anchor, 'requested_anchor_positions': requested,
            'supporting_receptacle_names': supports, 'nearby_object_names': near,
            'far_object_names': sorted(set(distances) - set(near)), 'distance_to_requested_X_m': distances}


def _original_measurement(c, baseline, reference):
    measured = original.physical_check(c['before_objects'], c['requested_poses'], c['realized_objects'],
        c['edited_names'], native_is_scene_at_rest=c['native_is_scene_at_rest'],
        placement_evidence_by_name=c['placement_evidence_by_name'])
    checked = stable.common_baseline_check(baseline, c['realized_objects'], c['edited_names'], reference)
    if c.get('common_baseline_check') != checked:
        raise ValueError('Original common-baseline measurement changed')
    return _ORIGINAL_AUGMENT(measured, c)


def _constructor(c):
    """Recompute the original native constructor equation, not a saved PASS."""
    from . import pair_v2_simulator as engine
    changes = engine.pose_measurements(c['before_objects'], c['realized_objects'])
    if c['pose_measurements'] != changes:
        raise ValueError('Original constructor pose measurements changed')
    before_agent, after_agent = c.get('agent_before'), c.get('agent_after')
    if not isinstance(before_agent, dict) or not before_agent or not isinstance(after_agent, dict) or not after_agent:
        raise ValueError('Original constructor agent evidence is missing')
    same_agent = before_agent == after_agent
    if c.get('agent_unchanged') is not same_agent:
        raise ValueError('Original constructor agent verdict changed')
    actual = engine.objects_by_name(c['realized_objects'])
    requested = {p['objectName']: p for p in c['requested_poses']}
    if set(c['edited']) != set(c['edited_names']):
        raise ValueError('Original constructor edited inventory changed')
    edit_ok = True
    for name in c['edited_names']:
        target = actual.get(name)
        proof = c['placement_evidence_by_name'].get(name) or {}
        destination = actual.get(proof.get('receptacle_name'))
        parent = (target is not None and destination is not None
            and isinstance(target.get('parentReceptacles'), list)
            and destination['objectId'] in target['parentReceptacles'])
        error = None if target is None or name not in requested else engine.distance(requested[name]['position'], target['position'])
        stored = c['edited'][name]
        if stored.get('parent_membership') is not parent or stored.get('position_error_m') != error:
            raise ValueError('Original constructor anchor/parent measurements changed')
        edit_ok = edit_ok and error is not None and error <= .02 and parent
    immediate = all(row['position_delta_m'] <= .01 for name, row in changes['objects'].items()
                    if name not in c['edited_names'])
    conditions = {'settling': c['settling'].get('settled') is True,
        'inventory_exact': changes['inventory_exact'], 'editable_inventory_exact': changes['editable_inventory_exact'],
        'agent_unchanged': same_agent, 'anchor_and_parent': edit_ok}
    noncollateral = all(conditions.values())
    verdict = bool(noncollateral and immediate and c['common_baseline_check']['passed'])
    if c.get('valid_physical') is not verdict:
        raise ValueError('Original constructor verdict differs from its measured equation')
    return verdict, ['constructor_' + key + '_failed' for key, passed in conditions.items() if not passed]


def _measure(c, baseline, reference, context, *, pending=False):
    old = _original_measurement(c, baseline, reference)
    far = set(context['far_object_names'])
    if pending:
        far = set(_indexed(baseline)) - {context['anchor_name']}
    ignored = {prefix + name for name in far for prefix in (
        'other_object_position_change_above_0_01m:',
        'common_baseline_other_object_position_change_above_0_01m:')}
    remaining = [reason for reason in old['reasons'] if reason not in ignored]
    constructor_passed, engineering_reasons = _constructor(c)
    remaining = list(dict.fromkeys([*remaining, *engineering_reasons]))
    return {**old, **fields(), 'valid_physical': not remaining, 'reasons': remaining,
            'original_valid_physical': bool(old['valid_physical'] and constructor_passed),
            'original_physics_reasons': copy.deepcopy(old['reasons']),
            'original_constructor_passed': constructor_passed,
            'nearby_physics_context': copy.deepcopy(context),
            'nonanchor_gate_pending': pending,
            'measurement_scope': 'supplement16 same-baseline near/support union; original native physics measurements preserved'}


def measure_pair(constructions, common_objects, common_reference):
    """Pure recomputation from original authenticated physical dictionaries."""
    if len(constructions) != 2:
        raise ValueError('Final nearby physics requires both actual native constructions')
    context = _context(constructions, common_objects, common_reference)
    measured = [_measure(c, common_objects, common_reference, context) for c in constructions]
    baseline = _indexed(common_objects)
    far_rows = []
    for name in context['far_object_names']:
        ds = [original.position_distance(baseline[name]['position'], _indexed(c['realized_objects'])[name]['position'])
              if name in _indexed(c['realized_objects']) else None for c in constructions]
        before = [m['all_object_changes'].get(name, {}).get('position_change_m') for m in measured]
        far_rows.append({'name': name, 'object_type': baseline[name]['objectType'],
            'common_baseline_displacement_m_by_world': ds,
            'before_to_after_displacement_m_by_world': before,
            'displacement_m': max(ds) if all(v is not None for v in ds) else None, 'distance_to_X_m': min(context['distance_to_requested_X_m'][name]),
            'distance_to_requested_X_m_by_world': copy.deepcopy(context['distance_to_requested_X_m'][name])})
    return {'worlds': measured, 'pair': {**fields(),
        'far_object_drift_max_m': (max((r['displacement_m'] for r in far_rows), default=0.0)
            if all(r['displacement_m'] is not None for r in far_rows) else None),
        'far_drift_objects': far_rows,
        'passes_original_physics_rule': all(m['original_valid_physical'] for m in measured),
        'nearby_physics_context': context}}


@contextmanager
def pair_context(sim, row, config):
    """Delay only the nonanchor gate until successful w1 fallback is known.

    The original constructor still writes its unmodified outcome/common-check
    evidence. Additional portable measurements are derived separately. The same
    dict returned to build_pair for w0 is updated before its final classifier.
    """
    validate_fields(config)
    from . import pair_v2_marked_shard as marked
    if getattr(sim, 'clone_context', None) is not None:
        raise ValueError('Supplement16 G01 uses natural candidates only')
    baseline = sim.scene_preflight['snapshot']
    original_construct = sim.construct_world
    had, own = 'construct_world' in vars(sim), vars(sim).get('construct_world')
    previous_augment = marked.augment_physical_result
    results = {}
    row.update(fields(), far_object_drift_max_m=None, far_drift_objects=None,
               passes_original_physics_rule=None, nearby_physics_context=None)
    def augment(measured, physical):
        custom = physical.get('nearby_physics_measurement')
        return previous_augment(measured, physical) if custom is None else custom
    def construct(start, key, placements, require_agent=None):
        world = 'w0' if key.endswith('_w0') else 'w1' if key.endswith('_w1') else None
        if world is None or world in results or (world == 'w1' and 'w0' not in results):
            raise ValueError('Nearby physics requires exactly ordered w0 then w1 constructions')
        result = original_construct(start, key, placements, require_agent=require_agent)
        results[world] = result
        required = ('before_objects', 'requested_poses', 'realized_objects', 'edited_names',
                    'placement_evidence_by_name', 'common_baseline_check', 'pose_measurements',
                    'settling', 'edited', 'native_is_scene_at_rest', 'agent_before', 'agent_after')
        physical = result.get('physical')
        if (not isinstance(physical, dict) or any(key not in physical for key in required)
                or not physical.get('placement_evidence_by_name')
                or any(not isinstance(proof, dict) or not proof.get('receptacle_name')
                       for proof in physical.get('placement_evidence_by_name', {}).values())):
            if result.get('passed') is not False:
                raise ValueError('Passed native construction lacks required original physics evidence')
            row['nearby_physics_context'] = {**fields(), 'finalized': False,
                'reason': 'original_native_construction_failed_before_complete_physics_evidence'}
            return result
        if world == 'w0':
            context = _context([result['physical']], baseline['objects'], baseline['reference'])
            measurement = _measure(result['physical'], baseline['objects'], baseline['reference'], context, pending=True)
            result['physical']['nearby_physics_measurement'] = measurement
            result['passed'] = measurement['valid_physical']
            row['nearby_physics_context'] = copy.deepcopy(context)
        else:
            checked = measure_pair([results[w]['physical'] for w in ('w0', 'w1')], baseline['objects'], baseline['reference'])
            for index, label in enumerate(('w0', 'w1')):
                current = results[label]
                target = current['physical'].setdefault('nearby_physics_measurement', {})
                target.clear(); target.update(checked['worlds'][index])
                current['passed'] = target['valid_physical']
            row.update(checked['pair'])
            # Original build_pair checked w0 before w1 existed. Its w1 stop
            # now also enforces the finalized w0 near/support gate.
            result['passed'] = all(results[w]['passed'] for w in ('w0', 'w1'))
        return result
    try:
        sim.construct_world = construct
        marked.augment_physical_result = augment
        yield
    finally:
        if (row.get('nearby_physics_context') or {}).get('finalized') is not True:
            for result in results.values():
                measured = (result.get('physical') or {}).get('nearby_physics_measurement')
                if measured is not None and measured.get('nonanchor_gate_pending') is True:
                    # True above was an internal permission to construct w1,
                    # not a measured final physical verdict. Do not publish it
                    # as a pass if w1 failed before producing its actual request.
                    if measured.get('valid_physical') is True:
                        measured['valid_physical'] = None
                    measured['reasons'] = list(dict.fromkeys([*measured['reasons'],
                        'nearby_nonanchor_gate_unmeasured_until_actual_w1_request']))
        marked.augment_physical_result = previous_augment
        if had:
            sim.construct_world = own
        else:
            del sim.construct_world


def update_counts(counts, row):
    """Lightweight actual-attempt summary, never an independent acceptance."""
    if row.get('counts_as_pair_attempt') is not True:
        return
    counts['counted_attempts'] += 1
    finalized = (row.get('nearby_physics_context') or {}).get('finalized') is True
    counts['finalized_two_worlds' if finalized else 'pending_or_unmeasured'] += 1
    if not finalized:
        return
    passed = all((row.get('physical_' + w) or {}).get('valid_physical') is True for w in ('w0', 'w1'))
    if passed:
        counts['both_worlds_pass_new_physics'] += 1
        if row.get('passes_original_physics_rule') is False:
            counts['pass_new_but_fail_original_physics'] += 1
    if row.get('status') == 'passed' and row.get('pair_file'):
        old = row.get('passes_original_physics_rule')
        if type(old) is not bool:
            raise ValueError('Exported supplement16 pair lacks original-physics diagnostic')
        counts['exported_original_rule_' + str(old).lower()] += 1
