"""Supplement13 proposal ordering and scene-wide measured-attempt quota.

Geometry ranks native proposals; only unchanged rendered/physical checks accept
worlds. No point or measured result is fabricated by these helpers.
"""
import copy
import math
import re
from . import pair_v2_projection_pool as pools
from . import pair_v2_role_policy as roles
from . import pair_v2_stay_order as stay

VERSION = 'look_flip_full_pool_edge_views_measured_quota_v1'
PRODUCER = 'src.envs.pair_v2_flip_shard'
POLICY = {
    'version': VERSION, 'approved_task_commit': '5a50999',
    'domain': 'all queried native points separated by3D distance ratio1.5,10px-outside,one-step-projection and3m filters; no old horizontal-side, proximity, overlap-proposal or12-point cap for Flip',
    'receptacles': 'all original eligible scene receptacles in deterministic original order',
    'flip_order': 'native3D near/far ratio, then stable native key; actual settled3D acceptance remains mandatory',
    'prediction': 'native selected coordinate is only proposal geometry; PlaceObjectAtPoint resolves its own surface and pivot, so actual settled metadata is authoritative',
    'views': 'existing allowed views reordered toward actual-geometric candidate centres at opposite left/right image edges; actual native visibility still required',
    'identities': 'opposite-edge candidate pairs first, then original same-category/name order',
    'stay': 'original first12 near points per side and supplement11 far-first retries',
    'quota': 'across the entire scene, including Controller restarts: counted Flip attempts >= counted Stay attempts; failed native placement does not earn quota',
    'no_flip_domain': 'continue other combinations/views; never fill a required Flip turn with Stay',
    'unchanged': 'seed0 world-side assignment, StayA/B alternation, physical10-rejection skip, all acceptance thresholds',
}
FIELDS = ('flip_search_version', 'flip_search_policy', 'flip_selected_field_of_view')


def fields(camera):
    if type(camera) is not int or camera not in (60, 100):
        raise ValueError('Explicit selected camera must be60 or100')
    return {FIELDS[0]: VERSION, FIELDS[1]: copy.deepcopy(POLICY), FIELDS[2]: camera}


def validate_fields(value):
    expected = fields(value.get('field_of_view'))
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError('Missing/altered explicit Flip search and selected-camera policy')


def side_for_point(point, positions):
    distances = [math.dist([point[axis] for axis in 'xyz'], [value[axis] for axis in 'xyz'])
                 for value in positions]
    if (any(not math.isfinite(value) for value in distances) or distances[0] == distances[1]
            or max(distances) < 1.5 * min(distances)):
        return None
    return 'A' if distances[0] < distances[1] else 'B'


class PointGroup(list):
    """Keep the historical Stay domain separate from the new full Flip domain."""
    def __init__(self, values, stay_values):
        super().__init__(values)
        self.stay_values = stay_values


def placement_groups(original, points, a, b, anchor, **kwargs):
    old = original(points, a, b, anchor, **kwargs)
    groups, seen = {'A': [], 'B': []}, set()
    for token in points:
        key = pools._point_key(token)
        if key in seen:
            continue
        seen.add(key)
        side = side_for_point(token['position'], [a['position'], b['position']])
        if side is not None:
            groups[side].append(token)
    return tuple(PointGroup(groups[side], old[index]) for index, side in enumerate(('A', 'B')))


class AttemptQuota:
    def __init__(self):
        self.flip, self.stay, self.saved = 0, 0, set()

    def wanted(self):
        return 'flip' if self.flip <= self.stay else 'stay'

    def record(self, path, row):
        if row.get('counts_as_pair_attempt') is not True or str(path) in self.saved:
            return
        mode = row.get('mode')
        if mode not in ('flip', 'stay'):
            raise ValueError('Counted Look attempt lacks its declared Flip/Stay mode')
        if mode == 'stay' and self.flip <= self.stay:
            raise ValueError('Stay cannot consume an unearned measured Flip quota')
        self.saved.add(str(path))
        if mode == 'flip': self.flip += 1
        else: self.stay += 1

    def summary(self):
        return {'counted_flip': self.flip, 'counted_stay': self.stay,
            'next_required_mode': self.wanted(), 'scope': POLICY['quota']}


class ProductRows:
    """Complete native cross-side domain without allocating its Cartesian product."""
    def __init__(self, first, second):
        self.first, self.second = first, second
    def __len__(self):
        return len(self.first) * len(self.second)
    def __getitem__(self, index):
        if type(index) is not int or index < 0 or index >= len(self):
            raise IndexError(index)
        left, right = divmod(index, len(self.second))
        return self.first[left], self.second[right], index, 0


class FlipRoles(roles.RoleProposalIterator):
    def __init__(self, record):
        super().__init__([], [], [record['view']])
        self.points = {side: [item['token'] for item in record['points']['look'][side]] for side in ('A', 'B')}
        self.rows['flip'] = ProductRows(self.points['A'], self.points['B'])
        self.pairs['flip'] = self.rows['flip']
        for side, mode in (('A', 'stay_a'), ('B', 'stay_b')):
            values = [item['token'] for item in record['stay_points'][side]]
            pairs = [(first, second) for index, first in enumerate(values)
                     for second in stay.ordered(values[index + 1:], first['position'])
                     if first['position'] != second['position']]
            self.pairs[mode] = pairs
            self.rows[mode] = [(first, second, index, 0) for index, (first, second) in enumerate(pairs)]
        self.totals = {key: len(value) for key, value in self.rows.items()}


class FlipIterator(pools.PoolProposalIterator):
    def __init__(self, records, quota):
        self.records = list(records)
        if len(self.records) != 1 or self.records[0].get('flip_search_version') != VERSION:
            raise ValueError('Flip search requires one explicitly recorded actual view/identity domain')
        self.quota = quota
        self.streams = {'look': FlipRoles(self.records[0]), 'direct': roles.RoleProposalIterator([], [], [])}
        self.next_pool, self.yielded = 'look', {'look': 0, 'direct': 0}

    def __next__(self):
        mode = self.quota.wanted()
        stream = self.streams['look']
        if mode == 'flip' and stream.summary()['remaining']['flip'] == 0:
            raise StopIteration  # continue other original domains, never substitute Stay
        stream.next_kind = mode
        self.next_pool = 'look'
        return super().__next__()


def full_pool(original, left, right, a, b, anchor, **kwargs):
    # Full tokens are shared read-only in live pools. The complete coordinate
    # array already lives in every native query; never copy it per point.
    record = original([], [], a, b, anchor, **kwargs)
    half = record['anchor_half_height_m']
    camera, actions = kwargs['camera'], kwargs['observation_cameras']
    record['stay_points'] = {}
    for side, values, candidate in (('A', left, a), ('B', right, b)):
        def project(values):
            result = []
            for token in values:
                point = pools._xyz(token['position'])
                center = {**point, 'y': point['y'] + half}
                start = pools.project_point(center, camera)
                after = [{'action': copy.deepcopy(item['action']),
                          'projection': pools.project_point(center, item['camera'])} for item in actions]
                distance = math.dist(list(point.values()), list(pools._xyz(camera['agent_position']).values()))
                result.append({'token': token, 'native_key': list(pools._point_key(token)),
                    'side_distance_m': math.hypot(point['x'] - candidate['position']['x'], point['z'] - candidate['position']['z']),
                    'native_agent_distance_m': distance,
                    'projected_center_agent_distance_m': math.dist(list(center.values()), list(pools._xyz(camera['agent_position']).values())),
                    'projection_point': center, 'start': start, 'after_actions': after,
                    'direct_eligible': start['in_frame'], 'look_eligible': bool(start['projectable'] and not start['in_frame']
                        and start['outside_edge_margin_px'] >= 10 and distance <= 3
                        and any(item['projection']['in_frame'] for item in after))})
            return result
        record['diagnostics'][side] = project(values)
        historical = project(values.stay_values)
        historical.sort(key=lambda item: (item['side_distance_m'], item['native_key']))
        record['stay_points'][side] = [item for item in historical if item['look_eligible']][:12]
        record['counts'][side] = {'input_native_eligible': len(values), 'unique_native_points': len(values)}
    for side, near, far in (('A', a, b), ('B', b, a)):
        eligible = [item for item in record['diagnostics'][side] if item['look_eligible']]
        for item in eligible:
            point = item['token']['position']
            da = math.dist([point[axis] for axis in 'xyz'], [near['position'][axis] for axis in 'xyz'])
            db = math.dist([point[axis] for axis in 'xyz'], [far['position'][axis] for axis in 'xyz'])
            item['flip_proposal_3d_rank'] = {'near_distance_m': da, 'far_distance_m': db,
                'ratio_eligible_prediction': da < db and db >= 1.5 * da,
                'scope': POLICY['prediction']}
        eligible.sort(key=lambda item: (not item['flip_proposal_3d_rank']['ratio_eligible_prediction'],
            item['flip_proposal_3d_rank']['near_distance_m'] / max(item['flip_proposal_3d_rank']['far_distance_m'], 1e-15),
            item['native_key']))
        record['points']['look'][side] = eligible
        record['counts'][side]['look'] = {'before_cap': len(eligible), 'retained': len(eligible), 'cap': None}
        record['counts'][side].update(stay_retained=len(record['stay_points'][side]), direct={'before_cap': 0, 'retained': 0, 'cap': 0})
    record.update(flip_search_version=VERSION, flip_search_policy=copy.deepcopy(POLICY))
    return record


def compact_pool(original, record):
    public = {key: value for key, value in record.items() if key != 'stay_points'}
    value = original(public)
    stay_record = {**public, 'points': {'look': record['stay_points']}, 'diagnostics': {}}
    value['stay_points'] = original(stay_record)['points']['look']
    return value


def verify_flip_role(row, candidate_positions):
    """Same deterministic assignment proof; explicit new native3D proposal domain.

    This never replaces the old checker's actual settled target, physical,
    prefix, evidence, solvability or witness checks.
    """
    names = row['identities']['candidate_names']
    if names != sorted(set(names)) or len(names) != 2 or any(name not in candidate_positions for name in names):
        raise ValueError('Need unchanged stable A/B names and authenticated baseline positions')
    if row.get('flip_search_version') != VERSION or row.get('flip_search_policy') != POLICY:
        raise ValueError('Native3D role verification requires explicit new search policy')
    match = re.fullmatch(re.escape(row['scene'].lower()) + r'_([0-9]{5})', row['pair_id'])
    if match is None:
        raise ValueError('Pair ID lacks its actual numeric attempt index')
    proposal, declared = row['proposal'], row['proposal_role']
    if proposal['mode'] != 'flip' or row.get('mode') != 'flip':
        raise ValueError('Native3D role adapter only verifies Flip')
    canonical = copy.deepcopy(proposal)
    if roles.flip_orientation(row['scene'], int(match.group(1)), 0)['flip_swapped']:
        canonical['w0'], canonical['w1'] = canonical['w1'], canonical['w0']
    expected_proposal, expected = roles.orient_proposal(canonical, row['scene'], int(match.group(1)), 0)
    expected.update(candidate_a_name=names[0], candidate_b_name=names[1])
    if (row.get('generation_version') != roles.ROLE_VERSION or row.get('generation_policy') != roles.ROLE_POLICY
            or expected != declared or expected_proposal != proposal):
        raise ValueError('Recorded Flip assignment differs from unchanged seed0 rule')
    sides = []
    for world in ('w0', 'w1'):
        token = proposal[world]
        if token['position'] != token['spawn']['selected_position']:
            raise ValueError('Role position is not the actual native input coordinate')
        sides.append(side_for_point(token['position'], [candidate_positions[name] for name in names]))
    if None in sides or sides != [declared['w0_source_side'], declared['w1_source_side']]:
        raise ValueError('Native3D proposal side differs from authenticated geometry')
    targets = [row.get('target_w0'), row.get('target_w1')]
    if any(target not in names for target in targets):
        raise ValueError('Accepted row lacks actual target identities')
    actual = ['A' if target == names[0] else 'B' for target in targets]
    return {'passed': True, 'intended_candidate_roles': sides, 'actual_candidate_roles': actual,
        'actual_targets_match_intended': actual == sides, 'assignment_digest': declared['assignment_digest'],
        'scope': POLICY['prediction'], 'flip_search_version': VERSION}


def edge_rank(a, b, camera):
    projections = [pools.project_point(obj.get('axisAlignedBoundingBox', {}).get('center', obj['position']), camera)
                   for obj in (a, b)]
    visible = all(point['in_frame'] for point in projections)
    if not visible:
        return (1, 1, float('inf'))
    left, right = sorted(point['u'] for point in projections)
    opposite = left < camera['width'] / 2 < right
    return (0, int(not opposite), (left + camera['width'] - right) / camera['width'])


def camera_from_snapshot(snapshot):
    agent = snapshot['agent']
    return pools.camera_from_metadata({**snapshot['camera'], 'agent': {
        'position': agent['position'], 'rotation': agent['rotation'], 'cameraHorizon': agent['horizon']}},
        snapshot['reference'])
