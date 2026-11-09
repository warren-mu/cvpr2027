"""Supplement15 complete-tuple geometry, before unchanged native/physics checks.

A/B are stable candidate distance roles, never screen halves. The proposed
anchor centre is native surface point plus the original anchor AABB half-height;
it is not a measured placed/settled position. Every geometric verdict is only a
proposal prefilter. Actual masks, evidence, settling, targets and witnesses keep
their original independent acceptance rules.
"""
import copy
from collections import Counter
import itertools
import math

from . import pair_v2_projection_pool as pools
from . import pair_v2_bbox_rank as boxes

VERSION = 'look_flip_complete_tuple_predicted_center_geometry_v1'
POLICY = {
    'version': VERSION, 'approval': 'T01v2 supplement15;5a0a138',
    'input_domain': 'all original native query tokens before horizontal, screen-side, or native-surface-point distance-ratio pruning',
    'candidate_roles': 'A/B are unchanged stable candidate names and metadata.position distance roles; either role may lie on either screen side',
    'anchor_center': 'native selected_position + original anchor axisAlignedBoundingBox.size.y/2 in Y; explicit geometric prediction, not physical or settled measurement',
    'distance': 'full3D Euclidean distance from predicted anchor centre to candidate metadata.position; near<far and far>=1.5*near',
    'prefix': 'all8 projected original anchor AABB corners known in front of camera; whole predicted rectangle outside v0 by at least10 pixels',
    'one_step': 'each world independently has a supplied existing one-action camera with a known bbox and strictly positive clipped rectangle area; actions may differ',
    'agent_range': 'original Look proposal constraint: native selected surface point within3m of original agent position',
    'unknown': 'unknown projection/geometry never counts as a passing prediction',
    'combination': 'two certified opposite-role points at distinct native positions; factorized checks followed by lazy Cartesian enumeration and full-tuple validation',
    'fallback': 'selected first then same certified role domain, maximum40; revalidate every actual native point against the fixed opposite point before placement',
    'seed': 'original seed0 canonical A/B assignment remains; actual w0/w1 source_sides may be A/B or B/A',
    'stay': 'original historical Stay domain, ordering, schedule, retry rules and scientific validation untouched',
    'storage': 'all prefilter input occurrences/reasons retained as compact native query references; complete coordinates stay in original native-query evidence',
    'scope': 'proposal geometry only; no measured mask pixels, physical validity or scientific acceptance claim',
}
FIELDS = ('flip_geometry_version', 'flip_geometry_policy')


def fields():
    return {FIELDS[0]: VERSION, FIELDS[1]: copy.deepcopy(POLICY)}


def validate_fields(value):
    if any(value.get(key) != item for key, item in fields().items()):
        raise ValueError('Missing/altered complete-tuple Flip geometry policy')


class NativePointGroup(list):
    """Carry the entire native input separately from the unchanged Stay list."""
    def __init__(self, all_native_values, historical_stay_values):
        complete = all_native_values if isinstance(all_native_values, tuple) else tuple(all_native_values)
        super().__init__(complete)
        self.all_native_values = complete
        self.stay_values = historical_stay_values


def placement_groups(original_horizontal_groups, points, a, b, anchor, **kwargs):
    complete = tuple(points)
    historical = original_horizontal_groups(complete, a, b, anchor, **kwargs)
    if len(historical) != 2:
        raise ValueError('Original Stay grouping did not return both historical roles')
    return (NativePointGroup(complete, historical[0]), NativePointGroup(complete, historical[1]))


def original_tokens(left, right):
    if (not isinstance(left, NativePointGroup) or not isinstance(right, NativePointGroup)
            or left.all_native_values is not right.all_native_values
            or len(left) != len(left.all_native_values) or len(right) != len(right.all_native_values)
            or any(item is not source for item, source in zip(left, left.all_native_values))
            or any(item is not source for item, source in zip(right, right.all_native_values))):
        raise ValueError('Flip geometry requires both complete, unpruned native carriers')
    return left.all_native_values


def as_token(value):
    """Accept the outer proposal or the actual native spawn-token call argument."""
    if not isinstance(value, dict):
        raise ValueError('Expected an explicit native placement token')
    if 'spawn' in value:
        token = value
    elif 'selected_position' in value:
        token = {'position': value['selected_position'], 'receptacle_name': value['receptacle_name'], 'spawn': value}
    else:
        raise ValueError('Token lacks original native selected-position evidence')
    pools._point_key(token)
    return token


def native_reference(value):
    token = as_token(value)
    return {'position': copy.deepcopy(token['position']), 'receptacle_name': token['receptacle_name'],
            'spawn_reference': pools.native_token_reference(token['spawn'])}


def predicted_role(center, a_position, b_position):
    center, a, b = pools._xyz(center), pools._xyz(a_position), pools._xyz(b_position)
    distance = {role: math.dist([center[key] for key in 'xyz'], [point[key] for key in 'xyz'])
                for role, point in (('A', a), ('B', b))}
    da, db = distance['A'], distance['B']
    role = 'A' if da < db and db >= 1.5 * da else 'B' if db < da and da >= 1.5 * db else None
    return role, distance


def predict(token, a, b, anchor, camera, observation_cameras):
    token = as_token(token)
    result = {'scope': POLICY['scope'], 'native_key': list(pools._point_key(token)),
        'native_reference': native_reference(token), 'predicted_anchor_center': None,
        'center_definition': POLICY['anchor_center'], 'candidate_positions': None,
        'distances_to_candidates_m': None, 'distance_role': None, 'native_agent_distance_m': None,
        'bbox_prediction': None, 'visible_one_step_action_indices': [],
        'proposal_geometry_passed': False, 'reasons': []}
    try:
        size = pools._xyz(anchor['axisAlignedBoundingBox']['size'])
        if any(value < 0 for value in size.values()):
            raise ValueError('Negative original anchor AABB dimension')
        point = pools._xyz(token['position'])
        center = {**point, 'y': point['y'] + size['y'] / 2}
        role, distances = predicted_role(center, a['position'], b['position'])
        agent = pools._xyz(camera['agent_position'])
        agent_distance = math.dist([point[key] for key in 'xyz'], [agent[key] for key in 'xyz'])
        bbox = boxes.predict({'projection_point': center}, anchor, camera, observation_cameras)
        result.update(predicted_anchor_center=center,
            candidate_positions={'A': pools._xyz(a['position']), 'B': pools._xyz(b['position'])},
            distances_to_candidates_m=distances, distance_role=role,
            native_agent_distance_m=agent_distance, bbox_prediction=bbox)
        if role is None:
            result['reasons'].append('predicted_center_3d_ratio_not_satisfied')
        if agent_distance > 3:
            result['reasons'].append('original_native_point_agent_range_exceeds3m')
        start = bbox.get('start')
        if not isinstance(start, dict) or start.get('projectable') is not True:
            result['reasons'].append('start_bbox_projection_unknown')
        elif start['signed_outside_margin_px'] < 10:
            result['reasons'].append('whole_start_bbox_not_outside_by10px')
        visible = [index for index, item in enumerate(bbox['after_actions'])
            if item['projection'].get('projectable') is True
            and item['projection'].get('clipped_rectangle_area_px2') is not None
            and item['projection']['clipped_rectangle_area_px2'] > 0]
        result['visible_one_step_action_indices'] = visible
        if not visible:
            result['reasons'].append('no_known_positive_area_one_step_bbox')
        # A prediction that ended early on malformed camera metadata cannot
        # use an earlier partial action list as a complete valid evaluation.
        if len(bbox['after_actions']) != len(observation_cameras):
            result['reasons'].append('incomplete_one_step_bbox_prediction')
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        result['reasons'].append('unknown_geometric_metadata')
        result['geometry_error'] = f'{type(error).__name__}: {error}'
    result['proposal_geometry_passed'] = not result['reasons']
    return result


def _context(record):
    context = record['flip_geometry_context']
    return context['candidate_a'], context['candidate_b'], context['anchor'], record['camera'], record['observation_cameras']


def compact_prediction(prediction):
    """Per-input evidence; shared camera, box, policies and actions stay once."""
    result = {key: copy.deepcopy(prediction.get(key)) for key in (
        'native_key', 'native_reference', 'predicted_anchor_center', 'distances_to_candidates_m',
        'distance_role', 'native_agent_distance_m', 'visible_one_step_action_indices',
        'proposal_geometry_passed', 'reasons')}
    bbox = prediction.get('bbox_prediction') or {}
    start = bbox.get('start') or {}
    result['start_bbox'] = {key: start.get(key) for key in (
        'projectable', 'signed_outside_margin_px', 'clipped_rectangle_area_px2')}
    result['one_step_boxes'] = [{'action_index': index,
        'projectable': item['projection'].get('projectable'),
        'clipped_rectangle_area_px2': item['projection'].get('clipped_rectangle_area_px2')}
        for index, item in enumerate(bbox.get('after_actions', []))]
    if prediction.get('geometry_error') is not None:
        result['geometry_error'] = prediction['geometry_error']
    return result


def filter_pool(record, a, b, anchor, *, all_native_tokens, check_deadline):
    """Rebuild only Flip Look roles from the ORIGINAL native stream.

    Extra evidence fields deliberately contain compact references, never full
    repeated coordinates. Runtime tokens remain complete and unchanged.
    """
    if not callable(check_deadline):
        raise ValueError('Complete native prefilter requires the original soft deadline callback')
    camera, actions = record['camera'], record['observation_cameras']
    points, diagnostics, seen, reasons = {'A': [], 'B': []}, [], set(), Counter()
    for index, raw in enumerate(all_native_tokens):
        check_deadline()
        token = as_token(raw)
        observation = predict(token, a, b, anchor, camera, actions)
        key = tuple(observation['native_key'])
        duplicate = key in seen
        seen.add(key)
        retained = observation['proposal_geometry_passed'] and not duplicate
        row = {'input_index': index, **compact_prediction(observation), 'retained_in_flip_domain': retained,
            'duplicate_native_key': duplicate}
        if duplicate:
            row['domain_exclusion_reason'] = 'duplicate_native_key_kept_once_in_runtime_domain'
        diagnostics.append(row)
        reasons.update(observation['reasons'])
        if retained:
            role = observation['distance_role']
            center = observation['predicted_anchor_center']
            distances = observation['distances_to_candidates_m']
            near, far = role, 'B' if role == 'A' else 'A'
            start = pools.project_point(center, camera)
            after = [{'action': copy.deepcopy(item['action']), 'projection': pools.project_point(center, item['camera'])}
                     for item in actions]
            points[role].append({'token': token, 'native_key': list(key), 'projection_point': center,
                'side_distance_m': distances[role], 'native_agent_distance_m': observation['native_agent_distance_m'],
                'projected_center_agent_distance_m': math.dist([center[key] for key in 'xyz'],
                    [pools._xyz(camera['agent_position'])[key] for key in 'xyz']),
                'start': start, 'after_actions': after, 'direct_eligible': start['in_frame'], 'look_eligible': True,
                'flip_geometry_input_index': index,
                'flip_proposal_3d_rank': {'near_distance_m': distances[near], 'far_distance_m': distances[far],
                    'ratio_eligible_prediction': True, 'scope': POLICY['scope'], 'position_basis': POLICY['anchor_center']}})
    output = {**record, **fields(), 'points': {**record['points'], 'look': points},
        'flip_geometry_context': {
            'candidate_a': {'name': a.get('name'), 'position': copy.deepcopy(a.get('position'))},
            'candidate_b': {'name': b.get('name'), 'position': copy.deepcopy(b.get('position'))},
            'anchor': {'name': anchor.get('name'), 'axisAlignedBoundingBox': copy.deepcopy(anchor.get('axisAlignedBoundingBox'))},
            'coordinate_evidence': POLICY['storage']},
        'flip_geometry_prefilter': diagnostics,
        'flip_geometry_counts': {'input_occurrences': len(diagnostics), 'unique_native_keys': len(seen),
            'geometry_passed_occurrences': sum(item['proposal_geometry_passed'] for item in diagnostics),
            'retained_by_distance_role': {role: len(items) for role, items in points.items()},
            'rejection_reason_counts': dict(reasons),
            'denominator': 'every original native input occurrence; reasons can overlap; duplicate keys recorded separately',
            'scientific_acceptance': None}}
    counts = copy.deepcopy(record.get('counts', {}))
    for role in ('A', 'B'):
        counts.setdefault(role, {})['look'] = {'before_cap': len(points[role]), 'retained': len(points[role]), 'cap': None}
    output['counts'] = counts
    return output


def validate_token(record, value, role):
    validate_fields(record)
    if role not in ('A', 'B'):
        raise ValueError('Unknown stable candidate distance role')
    token = as_token(value)
    key, reference = pools._point_key(token), native_reference(token)
    matches = [item for item in record['points']['look'][role]
               if tuple(item['native_key']) == key and native_reference(item['token']) == reference]
    if len(matches) != 1:
        raise ValueError('Actual native point is outside its certified original-query role domain')
    prediction = predict(token, *_context(record))
    if prediction['proposal_geometry_passed'] is not True or prediction['distance_role'] != role:
        raise ValueError('Actual native point fails the original complete-tuple geometric predicate')
    return prediction


def complete_pair(record, w0, w1, source_sides=('A', 'B')):
    if tuple(source_sides) not in (('A', 'B'), ('B', 'A')):
        raise ValueError('Flip must retain opposite distance roles, including original seed-swapped worlds')
    first, second = as_token(w0), as_token(w1)
    if first['position'] == second['position']:
        raise ValueError('Complete Flip tuple requires two distinct original native positions')
    predictions = [validate_token(record, value, role) for value, role in zip((first, second), source_sides)]
    return {**fields(), 'scope': POLICY['scope'], 'proposal_tuple_passed': True,
        'source_sides': list(source_sides), 'view': copy.deepcopy(record['view']),
        'candidate_names': [record['flip_geometry_context'][key]['name'] for key in ('candidate_a', 'candidate_b')],
        'worlds': {'w0': predictions[0], 'w1': predictions[1]}}


def complete_pairs(record, source_sides=('A', 'B')):
    """Lazy Cartesian stream; never allocate the product of candidate pools."""
    if tuple(source_sides) not in (('A', 'B'), ('B', 'A')):
        raise ValueError('Unknown complete Flip tuple source roles')
    for first, second in itertools.product(*(record['points']['look'][role] for role in source_sides)):
        try:
            proof = complete_pair(record, first['token'], second['token'], source_sides)
        except ValueError:
            # Do not silently skip a corrupted certified domain. Distinctness
            # is the sole combination exclusion; per-point invalidity raises.
            if first['token']['position'] == second['token']['position']:
                continue
            raise
        yield first['token'], second['token'], proof


def retry_points(record, side, selected, *, exclude_positions=(), max_trials=40):
    if type(max_trials) is not int or not 1 <= max_trials <= 40:
        raise ValueError('Keep the original maximum40 native placement attempts')
    selected = as_token(selected)
    validate_token(record, selected, side)
    selected_key = pools._point_key(selected)
    candidates = [selected] + [item['token'] for item in record['points']['look'][side]
                              if pools._point_key(item['token']) != selected_key]
    # No coordinate changes or inferred points; the actual native-time guard
    # still validates the chosen token against the fixed opposite world.
    return [token for token in candidates if token['position'] not in exclude_positions][:max_trials]


def validate_placement(record, token, role, *, opposite, opposite_role):
    return complete_pair(record, token, opposite, (role, opposite_role))
