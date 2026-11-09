"""G01 mixed proposal domains and bounded collection; scientific gates unchanged."""
import copy
import math
import re

from . import pair_v2_flip_search as flip
from . import pair_v2_projection_pool as pools
from . import pair_v2_role_policy as roles
from . import pair_v2_flip_geometry as geometry
from . import pair_v2_bbox_rank as rank
from . import pair_v2_spawn_sharing as sharing
from . import pair_v2_configuration as configurations

VERSION = 'g01_mixed_complete_look_geometry_four_class_quota_v2'
KINDS = ('direct_flip', 'direct_stay', 'look_flip', 'look_stay')
PER_CLASS = 8
QUOTA_VERSION = 'g01_static_ticket_proposal_selection_v1'
CONFIG_VERSION = 'g01_test_configuration_ticket_proposals_v2'
POLICY = {
    'version': VERSION, 'task': 'G01; approved supplements13/14/15',
    'look_flip': 'supplement15 original complete native domain, predicted-centre3D roles, entire AABB outside10px and each world one-step positive bbox area; actual rendered/physical acceptance unchanged',
    'direct_flip': 'unchanged G01 native-surface3D ratio roles and in-frame centre projection; no proximity or per-side cap',
    'stay': 'original horizontal native domain; original Look12/Direct40 cap and far-first actual-w0 retries',
    'pool_order': 'alternate Look/Direct when an available requested mode exists; explicitly record exhaustion fallback',
    'look_attempt_mix': 'scene-wide counted LookFlip >= counted LookStay, across Controller restarts; uncounted native failures earn nothing',
    'collection': 'at most8 producer-passed outputs per actual class per scene; independent verification remains separate',
    'quota_skip': 'retain original validity verdict and record collection_quota_full; never recast this as physical failure',
    'balance_after_cap': 'additional LookFlip proposals may earn required attempt balance while their full class is not exported',
    'unchanged': 'seed0 role orientation, native token proof, all physical/target/prefix/evidence/solvability/witness thresholds',
}
FIELDS = ('g01_search_version', 'g01_search_policy')


def fields(quota=False, configuration=False):
    spec = copy.deepcopy(POLICY)
    if quota:
        spec.update(version=QUOTA_VERSION,
            pool_order='only explicit allowed proposal kinds with remaining scene tickets; preserve available pool order',
            look_attempt_mix='retain Flip/Stay preference while both are open; never propose a disabled or full class',
            collection='actual output counts bounded by immutable scene/class tickets; scientific acceptance remains separate',
            balance_after_cap='no forced Flip after its tickets are full; Stay-only is allowed by the new task')
    if configuration:
        if not quota: raise ValueError('Configuration proposal policy requires static tickets')
        spec.update(version=CONFIG_VERSION, configuration=configurations.POLICY,
            collection='one exported actual class per canonical configuration across Controller restarts and authenticated historical inventory; static tickets unchanged')
    return {FIELDS[0]: spec['version'], FIELDS[1]: spec}


def validate_fields(value):
    if any(value.get(key) != item for key, item in fields(value.get(FIELDS[0]) in (QUOTA_VERSION, CONFIG_VERSION), value.get(FIELDS[0]) == CONFIG_VERSION).items()):
        raise ValueError('Missing or altered G01 mixed-search policy')


class CollectionQuotaFull(RuntimeError):
    pass


class ConfigurationAlreadyCollected(CollectionQuotaFull):
    pass


class SceneQuota:
    def __init__(self, per_class=PER_CLASS, *, actual_output_caps=None, allowed_kinds=None, scene=None, seen_configurations=None):
        if type(per_class) is not int or per_class not in (6, 8):
            raise ValueError('Require the approved train8/test6 scene-class cap')
        self.per_class = per_class
        self.ticketed = actual_output_caps is not None or allowed_kinds is not None
        self.limits = {kind: per_class for kind in KINDS}
        self.allowed = list(KINDS)
        if self.ticketed:
            from src.tools.g01_quota_plan import counts
            self.limits = counts(actual_output_caps)
            self.allowed = [kind for kind in KINDS if self.limits[kind] > 0]
            if (allowed_kinds != self.allowed or not self.allowed
                    or any(value > per_class for value in self.limits.values())):
                raise ValueError('Invalid bounded scene tickets or allowed proposal classes')
        self.passed = {kind: 0 for kind in KINDS}
        self.attempts = {kind: 0 for kind in KINDS}
        self.saved = set()
        self.configuration_enabled = seen_configurations is not None
        self.scene, self.configuration_seen = scene, {}
        if self.configuration_enabled:
            if not self.ticketed or per_class != 6 or not isinstance(scene, str):
                raise ValueError('Configuration proposal capacity requires explicit test tickets')
            for item in configurations.validate_seen(seen_configurations, scene):
                self.configuration_seen[(item['configuration']['configuration_id'], item['kind'])] = copy.deepcopy(item['configuration'])

    def complete(self):
        return all(self.passed[kind] == self.limits[kind] for kind in KINDS)

    def desired_modes(self, pool, preferred, configuration=None):
        if self.ticketed:
            choices = [mode for mode in ('flip', 'stay')
                       if f'{pool}_{mode}' in self.allowed
                       and self.passed[f'{pool}_{mode}'] < self.limits[f'{pool}_{mode}']
                       and (not self.configuration_enabled or
                            (configurations.validate(configuration)['configuration_id'], f'{pool}_{mode}') not in self.configuration_seen)]
            if pool == 'look' and len(choices) == 2:
                preferred = 'flip' if self.attempts['look_flip'] <= self.attempts['look_stay'] else 'stay'
            return sorted(choices, key=lambda mode: mode != preferred)
        if pool == 'look':
            if self.passed['look_flip'] == self.passed['look_stay'] == self.per_class:
                return []
            required = 'flip' if self.attempts['look_flip'] <= self.attempts['look_stay'] else 'stay'
            if required == 'flip':
                return ['flip']  # required balance never falls back to Stay
            return [mode for mode in ('stay', 'flip') if self.passed[f'look_{mode}'] < self.per_class]
        return [mode for mode in (preferred, 'stay' if preferred == 'flip' else 'flip')
                if self.passed[f'direct_{mode}'] < self.per_class]

    def configuration_of(self, row):
        value = configurations.validate(row.get('g01_configuration'))
        if value != configurations.from_pair(row) or value['key']['scene'] != self.scene:
            raise ValueError('Attempt configuration differs from original scene/view/identities')
        return value

    def before_export(self, row):
        kind = row.get('kind')
        if kind not in self.passed:
            raise ValueError('Unknown actual output class')
        if self.passed[kind] >= self.limits[kind]:
            raise CollectionQuotaFull(kind)
        if self.configuration_enabled:
            value = self.configuration_of(row)
            if (value['configuration_id'], kind) in self.configuration_seen:
                raise ConfigurationAlreadyCollected(kind)

    def record(self, path, row):
        if row.get('counts_as_pair_attempt') is not True or str(path) in self.saved:
            return
        key = f"{row.get('proposed_pool')}_{row.get('mode')}"
        if key not in self.attempts:
            raise ValueError('Counted G01 attempt lacks its actual proposed pool/mode')
        if self.ticketed and (key not in self.allowed or self.passed[key] >= self.limits[key]):
            raise ValueError('Counted proposal used a disabled or already-full class')
        configuration = self.configuration_of(row) if self.configuration_enabled else None
        if configuration is not None and (configuration['configuration_id'], key) in self.configuration_seen:
            raise ValueError('Counted proposal repeated an already represented configuration/class')
        if (not self.ticketed and key == 'look_stay'
                and self.attempts['look_flip'] <= self.attempts['look_stay']):
            raise ValueError('LookStay consumed unearned actual LookFlip attempt balance')
        if row.get('status') == 'passed':
            self.before_export(row)
            self.passed[row['kind']] += 1
            if configuration is not None:
                self.configuration_seen[(configuration['configuration_id'], row['kind'])] = configuration
        self.attempts[key] += 1
        self.saved.add(str(path))

    def summary(self):
        if self.ticketed:
            return {'version': CONFIG_VERSION if self.configuration_enabled else QUOTA_VERSION, 'accepted_by_actual_class': dict(self.passed),
                    **({'configuration_version': configurations.VERSION, 'seen_configurations':
                        [{'configuration': copy.deepcopy(value), 'kind': kind}
                         for (identity, kind), value in sorted(self.configuration_seen.items())]}
                       if self.configuration_enabled else {}),
                    'counted_by_proposed_pool_mode': dict(self.attempts),
                    'per_scene_class_cap': self.per_class, 'actual_output_caps': dict(self.limits),
                    'allowed_kinds': list(self.allowed), 'all_class_caps_reached': self.complete(),
                    'acceptance_scope': 'producer output counts against reserved tickets; not independent acceptance'}
        return {'version': VERSION, 'accepted_by_actual_class': dict(self.passed),
                'counted_by_proposed_pool_mode': dict(self.attempts),
                'per_scene_class_cap': self.per_class, 'all_class_caps_reached': self.complete(),
                'acceptance_scope': 'producer output counts; independent scientific verification still required'}


def _historical_mixed_pool(original, left, right, a, b, anchor, **kwargs):
    record = flip.full_pool(original, left, right, a, b, anchor, **kwargs)
    record['stay_points_by_pool'] = {'look': record.pop('stay_points'), 'direct': {}}
    # The original builder copies each entire native coordinate array into
    # every point. Keep its exact Stay Direct40 geometry/order without that
    # quadratic duplication; source tokens remain shared and unmodified.
    camera, actions = kwargs['camera'], kwargs['observation_cameras']
    for side, near, far, values in (('A', a, b, left), ('B', b, a, right)):
        known = {tuple(item['native_key']): item for item in record['diagnostics'][side]}
        historical, seen = [], set()
        for token in values.stay_values:
            key = pools._point_key(token)
            if key in seen:
                continue
            seen.add(key)
            item = known.get(key)
            if item is None:
                point = pools._xyz(token['position'])
                center = {**point, 'y': point['y'] + record['anchor_half_height_m']}
                start = pools.project_point(center, camera)
                after = [{'action': copy.deepcopy(action['action']),
                    'projection': pools.project_point(center, action['camera'])} for action in actions]
                distance = math.dist(list(point.values()), list(pools._xyz(camera['agent_position']).values()))
                item = {'token': token, 'native_key': list(key),
                    'side_distance_m': math.hypot(point['x'] - near['position']['x'], point['z'] - near['position']['z']),
                    'native_agent_distance_m': distance,
                    'projected_center_agent_distance_m': math.dist(list(center.values()), list(pools._xyz(camera['agent_position']).values())),
                    'projection_point': center, 'start': start, 'after_actions': after,
                    'direct_eligible': start['in_frame'],
                    'look_eligible': bool(start['projectable'] and not start['in_frame']
                        and start['outside_edge_margin_px'] >= 10 and distance <= 3
                        and any(action['projection']['in_frame'] for action in after))}
            if item['direct_eligible']:
                historical.append(item)
        historical.sort(key=lambda item: (item['side_distance_m'], item['native_key']))
        record['stay_points_by_pool']['direct'][side] = historical[:40]
        direct = [item for item in record['diagnostics'][side] if item['direct_eligible']]
        def rank(item):
            point = item['token']['position']
            close = math.dist([point[key] for key in 'xyz'], [near['position'][key] for key in 'xyz'])
            distant = math.dist([point[key] for key in 'xyz'], [far['position'][key] for key in 'xyz'])
            return close / max(distant, 1e-15), item['native_key']
        direct.sort(key=rank)
        record['points']['direct'][side] = direct
        record['counts'][side]['direct'] = {'before_cap': len(direct), 'retained': len(direct), 'cap': None}
        record['counts'][side]['stay_retained_by_pool'] = {
            pool: len(record['stay_points_by_pool'][pool][side]) for pool in ('look', 'direct')}
    # This is an explicit mixed policy, not a claim to have used the old Look-only search.
    record.pop('flip_search_version'); record.pop('flip_search_policy')
    record.update(fields())
    return record


def build_pool(original, left, right, a, b, anchor, *, check_deadline, **kwargs):
    """Rebuild only LookFlip; preserve the original Direct and Stay domains.

    Complete native carriers are mandatory. Surface-point role filtering is
    performed only for the historical Direct branch, never before Look's
    predicted-centre geometry. No native coordinates are copied or invented.
    """
    complete = geometry.original_tokens(left, right)
    direct, seen = {'A': [], 'B': []}, set()
    for token in complete:
        check_deadline()
        key = pools._point_key(token)
        if key in seen:
            continue
        seen.add(key)
        side = flip.side_for_point(token['position'], [a['position'], b['position']])
        if side is not None:
            direct[side].append(token)
    groups = [flip.PointGroup(direct[side], carrier.stay_values)
              for side, carrier in zip(('A', 'B'), (left, right))]
    record = _historical_mixed_pool(original, *groups, a, b, anchor, **kwargs)
    record = geometry.filter_pool(record, a, b, anchor,
        all_native_tokens=complete, check_deadline=check_deadline)
    record = rank.rank_pool(record, anchor, check_deadline=check_deadline)
    return {**record, **sharing.fields()}


def compact_pool(original, record):
    public = {key: value for key, value in record.items() if key != 'stay_points_by_pool'}
    value = original(public)
    stay_record = {**public, 'points': record['stay_points_by_pool'], 'diagnostics': {}}
    value['stay_points_by_pool'] = original(stay_record)['points']
    return value


class MixedIterator(pools.PoolProposalIterator):
    def __init__(self, records, quota):
        self.records = list(records)
        if len(self.records) != 1:
            raise ValueError('One authenticated view/identity catalog required')
        validate_fields(self.records[0])
        if (self.records[0].get(FIELDS[0]) in (QUOTA_VERSION, CONFIG_VERSION)) != quota.ticketed:
            raise ValueError('Proposal pool and scene quota versions differ')
        record = self.records[0]
        if (record.get(FIELDS[0]) == CONFIG_VERSION) != quota.configuration_enabled:
            raise ValueError('Proposal pool and scene configuration policy differ')
        self.configuration = configurations.from_inputs(quota.scene, record['view'], record['identity']) if quota.configuration_enabled else None
        self.streams = {pool: flip.FlipRoles({**record,
            'points': {'look': record['points'][pool]}, 'stay_points': record['stay_points_by_pool'][pool]})
            for pool in ('look', 'direct')}
        self.next_pool, self.yielded, self.quota = 'look', {'look': 0, 'direct': 0}, quota

    def __next__(self):
        if self.quota.complete():
            raise StopIteration
        choices = {}
        for pool, stream in self.streams.items():
            remaining = stream.summary()['remaining']
            for mode in self.quota.desired_modes(pool, stream.next_kind, self.configuration):
                available = remaining['flip'] if mode == 'flip' else remaining['stay_a'] or remaining['stay_b']
                if available:
                    choices[pool] = mode; break
        if not choices:
            raise StopIteration
        wanted = self.next_pool
        pool = wanted if wanted in choices else next(iter(choices))
        stream = self.streams[pool]; stream.next_kind = choices[pool]
        proposal, view, domain = next(stream)
        if proposal['mode'] != choices[pool]:
            raise ValueError('Mixed iterator silently changed its selected mode')
        self.next_pool = 'direct' if pool == 'look' else 'look'
        self.yielded[pool] += 1
        domain['projection_pool'] = {'version': pools.POOL_VERSION, 'pool': pool, 'requested_pool': wanted,
            'fallback_due_to_exhaustion': pool != wanted, 'ordinal': self.yielded[pool] - 1,
            'view_index': domain['view_index'], 'role_schedule_scope': 'within this projection pool'}
        proposal['role_domain'] = copy.deepcopy(domain)
        proposal['projection_pool'] = copy.deepcopy(domain['projection_pool'])
        return proposal, view, domain


def verify_flip_role(row, candidate_positions):
    """Keep Direct's surface-role proof; Look needs its independent bbox proof."""
    validate_fields(row)
    if row.get('proposed_pool') != 'direct':
        raise ValueError('G01 LookFlip requires independent complete-tuple geometry verification')
    names = row['identities']['candidate_names']
    if names != sorted(set(names)) or len(names) != 2 or any(name not in candidate_positions for name in names):
        raise ValueError('Need unchanged stable A/B names and authenticated baseline positions')
    match = re.fullmatch(re.escape(row['scene'].lower()) + r'_([0-9]{5})', row['pair_id'])
    if match is None:
        raise ValueError('Pair ID lacks its actual numeric attempt index')
    proposal, declared = row['proposal'], row['proposal_role']
    if proposal['mode'] != 'flip' or row.get('mode') != 'flip':
        raise ValueError('G01 3D role adapter only verifies Flip')
    canonical = copy.deepcopy(proposal)
    if roles.flip_orientation(row['scene'], int(match.group(1)), 0)['flip_swapped']:
        canonical['w0'], canonical['w1'] = canonical['w1'], canonical['w0']
    expected_proposal, expected = roles.orient_proposal(canonical, row['scene'], int(match.group(1)), 0)
    expected.update(candidate_a_name=names[0], candidate_b_name=names[1])
    if (row.get('generation_version') != roles.ROLE_VERSION or row.get('generation_policy') != roles.ROLE_POLICY
            or expected != declared or expected_proposal != proposal):
        raise ValueError('Recorded G01 Flip assignment differs from unchanged seed0 rule')
    sides = []
    for world in ('w0', 'w1'):
        token = proposal[world]
        if token['position'] != token['spawn']['selected_position']:
            raise ValueError('Role position is not the actual native input coordinate')
        sides.append(flip.side_for_point(token['position'], [candidate_positions[name] for name in names]))
    if None in sides or sides != [declared['w0_source_side'], declared['w1_source_side']]:
        raise ValueError('Native3D proposal side differs from authenticated geometry')
    targets = [row.get('target_w0'), row.get('target_w1')]
    if any(target not in names for target in targets):
        raise ValueError('Accepted row lacks actual target identities')
    actual = ['A' if target == names[0] else 'B' for target in targets]
    return {'passed': True, 'intended_candidate_roles': sides, 'actual_candidate_roles': actual,
        'actual_targets_match_intended': actual == sides, 'assignment_digest': declared['assignment_digest'],
        'scope': 'proposal geometry only; unchanged actual settled target checks remain mandatory',
        'g01_search_version': row['g01_search_version']}
