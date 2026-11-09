"""Supplement-6 proposal geometry only; no rendered or physical acceptance.

All placement tokens are returned unchanged. The anchor half-height is used
only for a pinhole-camera prefilter, never as a PlaceObjectAtPoint argument.
"""
import copy
import itertools
import math

from . import pair_v2_role_policy as roles

POOL_VERSION = 'native_projection_look_direct_pools_v1'
POOL_POLICY = {
    'version': POOL_VERSION,
    'camera': 'actual metadata cameraPosition, agent yaw and cameraHorizon, vertical fov, screenWidth/screenHeight',
    'projection': 'Unity +Z forward/+X right/+Y up; positive horizon looks down; image edge coordinates [0,width) x [0,height)',
    'proposal_point': 'native selected_position plus anchor AABB half-height in Y; placement input remains native point',
    'direct': 'projected point in the starting frame; original horizontal near-to-far order; at most 40 per side',
    'look': 'positive camera depth, >=10px beyond at least one starting-frame edge; in frame after an existing observation action; native point <=3m from agent; at most 12 per side',
    'schedule': 'Look/Direct alternate globally; each pool independently retains Flip/Stay and expanded Stay A/B alternation; truthful exhaustion fallback',
    'orientation': 'unchanged seed0 SHA256(scene, actual global attempt index); applied by caller after scheduling',
    'retry': 'selected unchanged native token first, then remaining same-side pool tokens in recorded order; at most 40 total',
    'scope': 'proposal prefilter only; does not establish visibility, action feasibility, physics, evidence or acceptance',
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Projection requires finite numeric coordinates')
    return float(value)


def _xyz(value):
    return {axis: _number(value[axis]) for axis in ('x', 'y', 'z')}


def camera_from_metadata(metadata, source_reference):
    """Retain source identity; callers authenticate its bytes outside this helper."""
    agent = metadata['agent']
    rotation = _xyz(agent['rotation'])
    if rotation['x'] != 0 or rotation['z'] != 0:
        raise ValueError('Unsupported non-upright agent camera mount')
    fov = _number(metadata['fov'])
    width, height = metadata['screenWidth'], metadata['screenHeight']
    if not 0 < fov < 180 or any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError('Missing/invalid actual camera FOV or image dimensions')
    if not isinstance(source_reference, dict) or not source_reference:
        raise ValueError('Actual camera needs an explicit metadata source reference')
    return {'camera_position': _xyz(metadata['cameraPosition']),
        'agent_position': _xyz(agent['position']), 'yaw_degrees': rotation['y'],
        'horizon_degrees': _number(agent['cameraHorizon']), 'vertical_fov_degrees': fov,
        'width': width, 'height': height, 'source_reference': copy.deepcopy(source_reference),
        'pose_source': 'actual_metadata', 'scope': POOL_POLICY['scope']}


def camera_for_view(reference, view):
    """Calibrated rigid camera-mount proposal, explicitly not an observed frame."""
    result = copy.deepcopy(reference)
    old_yaw, yaw = math.radians(reference['yaw_degrees']), math.radians(_number(view['yaw']))
    offset = {a: reference['camera_position'][a] - reference['agent_position'][a] for a in ('x', 'y', 'z')}
    local_x = math.cos(old_yaw) * offset['x'] - math.sin(old_yaw) * offset['z']
    local_z = math.sin(old_yaw) * offset['x'] + math.cos(old_yaw) * offset['z']
    position = _xyz(view['position'])
    result.update(agent_position=position, yaw_degrees=_number(view['yaw']),
        horizon_degrees=_number(view['horizon']), camera_position={
            'x': position['x'] + math.cos(yaw) * local_x + math.sin(yaw) * local_z,
            'y': position['y'] + offset['y'],
            'z': position['z'] - math.sin(yaw) * local_x + math.cos(yaw) * local_z},
        pose_source='rigid_mount_geometric_prediction',
        derivation={'reference_camera': copy.deepcopy(reference), 'requested_view': copy.deepcopy(view),
                    'assumption': 'same standing camera mount; prediction only, actual frame remains authoritative'})
    return result


def camera_after_action(camera, action):
    """Predict only an already supplied native-domain action; invent no moves."""
    view = {'position': copy.deepcopy(camera['agent_position']),
            'yaw': camera['yaw_degrees'], 'horizon': camera['horizon_degrees']}
    kind = action.get('kind')
    if kind in ('turn_left', 'turn_right'):
        view['yaw'] += -45 if kind == 'turn_left' else 45
    elif kind == 'move':
        if (type(action.get('index')) is not int or action['index'] < 1
                or (action.get('path') or {}).get('complete') is not True):
            raise ValueError('Move camera prediction requires the supplied existing native-complete action')
        view['position'] = _xyz(action['position'])
    else:
        raise ValueError('Only existing 45-degree turns/native adjacent moves are supported')
    return {'action': copy.deepcopy(action), 'camera': camera_for_view(camera, view),
            'scope': 'geometric prediction from supplied action; not action execution or availability proof'}


def project_point(point, camera):
    point = _xyz(point)
    dx, dy, dz = (point[a] - camera['camera_position'][a] for a in ('x', 'y', 'z'))
    yaw, pitch = math.radians(_number(camera['yaw_degrees'])), math.radians(_number(camera['horizon_degrees']))
    right = math.cos(yaw) * dx - math.sin(yaw) * dz
    horizontal_forward = math.sin(yaw) * dx + math.cos(yaw) * dz
    up = math.cos(pitch) * dy + math.sin(pitch) * horizontal_forward
    depth = -math.sin(pitch) * dy + math.cos(pitch) * horizontal_forward
    result = {'point': point, 'camera_xyz': {'x': right, 'y': up, 'z': depth},
              'projectable': depth > 0, 'u': None, 'v': None, 'in_frame': False,
              'outside_edge_margin_px': None}
    if depth <= 0:
        result['reason'] = 'nonpositive_camera_depth'
        return result
    width, height = camera['width'], camera['height']
    focal = height / (2 * math.tan(math.radians(camera['vertical_fov_degrees']) / 2))
    u, v = width / 2 + focal * right / depth, height / 2 - focal * up / depth
    if not all(math.isfinite(value) for value in (u, v)):
        raise ValueError('Nonfinite camera projection')
    result.update(u=u, v=v, in_frame=0 <= u < width and 0 <= v < height,
        outside_edge_margin_px=max(0., -u, u - width, -v, v - height), focal_pixels=focal)
    return result


def _point_key(token):
    if token['position'] != token['spawn']['selected_position']:
        raise ValueError('Proposal position must equal its unchanged native placement point')
    point = _xyz(token['position'])
    return (token['receptacle_name'], *(point[a] for a in ('x', 'y', 'z')))


def native_token_reference(token):
    """Small diagnostic reference; never a token accepted for placement.

    The complete native coordinate list remains in the authenticated query
    action and the live token. Repeating that list for every selected point
    makes a pool quadratic in the number of native coordinates.
    """
    if token is None:
        return None
    return {key: copy.deepcopy(token[key]) for key in (
        'action', 'success', 'scene', 'receptacle_name', 'receptacle_id_at_query',
        'receptacle_position_at_query', 'selected_position', 'query_reference',
        'source', 'scope') if key in token}


def pool_evidence_record(record):
    """Serialize proposal geometry with native query references, not copies.

    This is a diagnostic storage layout only. Runtime points, point ordering,
    complete placement tokens, and acceptance checks are untouched.
    """
    output = {key: copy.deepcopy(value) for key, value in record.items()
              if key not in ('points', 'diagnostics')}
    def point(item):
        result = {key: copy.deepcopy(value) for key, value in item.items() if key != 'token'}
        token = item['token']
        result['token'] = {key: copy.deepcopy(value) for key, value in token.items() if key != 'spawn'}
        result['token']['spawn_reference'] = native_token_reference(token['spawn'])
        return result
    output['diagnostics'] = {side: [point(item) for item in values]
                             for side, values in record['diagnostics'].items()}
    output['points'] = {pool: {side: [point(item) for item in values] for side, values in sides.items()}
                        for pool, sides in record['points'].items()}
    output['storage_layout'] = 'projection_geometry_native_query_references_v1'
    output['native_coordinate_storage'] = 'complete native coordinates remain in referenced query action bytes; runtime tokens unchanged'
    return output


def build_view_pools(left, right, candidate_a, candidate_b, anchor, *, camera, view, observation_cameras):
    """Input sides are the uncapped existing native-eligible placement groups."""
    half_height = _number(anchor['axisAlignedBoundingBox']['size']['y']) / 2
    if half_height < 0:
        raise ValueError('Negative anchor height')
    output = {'version': POOL_VERSION, 'policy': copy.deepcopy(POOL_POLICY), 'view': copy.deepcopy(view),
        'camera': copy.deepcopy(camera), 'observation_cameras': copy.deepcopy(observation_cameras),
        'anchor_name': anchor['name'], 'anchor_half_height_m': half_height,
        'points': {kind: {'A': [], 'B': []} for kind in ('look', 'direct')}, 'diagnostics': {}, 'counts': {}}
    for side, values, candidate in (('A', left, candidate_a), ('B', right, candidate_b)):
        measured, seen = [], set()
        for token in values:
            key = _point_key(token)
            if key in seen:
                continue
            seen.add(key)
            point = _xyz(token['position'])
            projected = {**point, 'y': point['y'] + half_height}
            start = project_point(projected, camera)
            after = [{'action': copy.deepcopy(item['action']),
                      'projection': project_point(projected, item['camera'])} for item in observation_cameras]
            agent_distance = math.dist(list(point.values()), [_number(camera['agent_position'][a]) for a in ('x', 'y', 'z')])
            near = math.hypot(point['x'] - candidate['position']['x'], point['z'] - candidate['position']['z'])
            look = (start['projectable'] and not start['in_frame']
                    and start['outside_edge_margin_px'] >= 10 and agent_distance <= 3
                    and any(item['projection']['in_frame'] for item in after))
            measured.append({'token': copy.deepcopy(token), 'native_key': list(key),
                'side_distance_m': near, 'native_agent_distance_m': agent_distance,
                'projected_center_agent_distance_m': math.dist(list(projected.values()), list(_xyz(camera['agent_position']).values())),
                'projection_point': projected, 'start': start, 'after_actions': after,
                'direct_eligible': start['in_frame'], 'look_eligible': bool(look)})
        measured.sort(key=lambda item: (item['side_distance_m'], item['native_key']))
        output['diagnostics'][side] = measured
        output['counts'][side] = {'input_native_eligible': len(values), 'unique_native_points': len(measured)}
        for kind, cap in (('look', 12), ('direct', 40)):
            eligible = [item for item in measured if item[kind + '_eligible']]
            output['points'][kind][side] = eligible[:cap]
            output['counts'][side][kind] = {'before_cap': len(eligible), 'retained': min(cap, len(eligible)), 'cap': cap}
    return output


def placement_alternatives(record, pool, side, selected, *, exclude_positions=(), max_trials=40):
    """Original successful token must be written back into all caller proofs."""
    if pool not in ('look', 'direct') or side not in ('A', 'B') or type(max_trials) is not int or not 1 <= max_trials <= 40:
        raise ValueError('Unknown pool/side or retry limit outside 1..40')
    values = [item['token'] for item in record['points'][pool][side]]
    if selected not in values:
        raise ValueError('Selected native token is not in this recorded pool')
    ordered = [selected] + [token for token in values if token != selected]
    return [copy.deepcopy(token) for token in ordered if token['position'] not in exclude_positions][:max_trials]


class _PoolRoles(roles.RoleProposalIterator):
    def __init__(self, records, pool):
        super().__init__([], [], [record['view'] for record in records])
        per_view = []
        for view_index, record in enumerate(records):
            points = {side: [item['token'] for item in record['points'][pool][side]] for side in ('A', 'B')}
            for side in points:
                self.points[side].extend(points[side])
            pairs = {'flip': itertools.product(points['A'], points['B']),
                     'stay_a': itertools.combinations(points['A'], 2), 'stay_b': itertools.combinations(points['B'], 2)}
            per_view.append({key: [(a, b) for a, b in values if a['position'] != b['position']]
                             for key, values in pairs.items()})
        for key in self.rows:
            self.rows[key], self.pairs[key] = [], []
            # Interleave existing views to avoid a single view consuming the prefix.
            for ordinal in range(max((len(item[key]) for item in per_view), default=0)):
                for view_index, item in enumerate(per_view):
                    if ordinal < len(item[key]):
                        first, second = item[key][ordinal]
                        pair_index = len(self.pairs[key])
                        self.pairs[key].append((first, second))
                        self.rows[key].append((first, second, pair_index, view_index))
        self.totals = {key: len(rows) for key, rows in self.rows.items()}


class PoolProposalIterator:
    """Independent original role schedules, interleaved Look/Direct globally."""
    def __init__(self, records):
        self.records = list(records)
        if any(record.get('version') != POOL_VERSION for record in self.records):
            raise ValueError('Unknown projection-pool version')
        self.streams = {pool: _PoolRoles(self.records, pool) for pool in ('look', 'direct')}
        self.next_pool, self.yielded = 'look', {'look': 0, 'direct': 0}

    def __iter__(self):
        return self

    def summary(self):
        return {'projection_pool_version': POOL_VERSION, 'projection_pool_policy': copy.deepcopy(POOL_POLICY),
            'yielded': dict(self.yielded), 'pools': {key: value.summary() for key, value in self.streams.items()},
            'pool_counts_by_view': [copy.deepcopy(record['counts']) for record in self.records],
            'acceptance_passed': None, 'balanced_accepted_data': None}

    def __next__(self):
        wanted = self.next_pool
        available = [pool for pool, stream in self.streams.items() if not stream.summary()['all_exhausted']]
        if not available:
            raise StopIteration
        pool = wanted if wanted in available else available[0]
        proposal, view, domain = next(self.streams[pool])
        self.next_pool = 'direct' if pool == 'look' else 'look'
        self.yielded[pool] += 1
        domain['projection_pool'] = {'version': POOL_VERSION, 'pool': pool, 'requested_pool': wanted,
            'fallback_due_to_exhaustion': pool != wanted, 'ordinal': self.yielded[pool] - 1,
            'view_index': domain['view_index'], 'role_schedule_scope': 'within this projection pool'}
        proposal['role_domain'] = copy.deepcopy(domain)
        proposal['projection_pool'] = copy.deepcopy(domain['projection_pool'])
        return proposal, view, domain
