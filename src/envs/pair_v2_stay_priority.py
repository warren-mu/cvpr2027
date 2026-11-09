"""Choose among existing Stay domains; never change a native point or gate."""
import copy
import math

from . import pair_v2_bbox_rank as boxes
from . import pair_v2_flip_geometry as geometry
from . import pair_v2_flip_search as flip
from . import pair_v2_flip_rounds_search as cadence
from . import pair_v2_projection_pool as pools

VERSION = 'stay_pending_domain_min_world_one_step_area_v1'
POLICY = {
    'version': VERSION,
    'score': 'minimum of the two planned worlds best known one-action projected AABB rectangle areas',
    'scope': 'predicted planned native proposals only; not actual fallback, rendered visibility, mask area or scientific acceptance',
    'selection': 'only when the existing scene-wide counted quota requests Stay; highest known score among this finite round pending Stay domains; stable original-domain ties; unknown retained last',
    'peek': 'read the original next Stay row without advancing any cursor, side alternation or quota',
    'within_domain': 'unchanged original Stay A/B alternation, fixed-w0 far-first combinations and actual native fallback order',
    'round': 'one visit per original round domain; original discovery order restored at each round boundary; Flip selection delegates to original pop',
    'domain': 'no added/deleted points, proposals, combinations or views; no score threshold',
    'deadline': 'original finite scene soft deadline checked before every pending domain and each world prediction',
}
FIELDS = ('stay_priority_version', 'stay_priority_policy')
SELECTION = 'stay_priority_selection'
_ORIGINAL_ROUND = cadence.DiscoveryRound


def fields():
    return {FIELDS[0]: VERSION, FIELDS[1]: copy.deepcopy(POLICY)}


def validate_fields(value):
    if any(value.get(key) != expected for key, expected in fields().items()):
        raise ValueError('Missing/altered explicit Stay priority policy')


def _area(value):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Stay priority area must be finite nonnegative or unknown')
    return value


def validate_selection(row):
    """Check declared structure/internal consistency, not independent science."""
    value = row.get(SELECTION)
    if row.get('mode') == 'flip':
        if value is not None:
            raise ValueError('Flip cannot claim Stay priority selection')
        return
    if row.get('mode') != 'stay' or not isinstance(value, dict):
        raise ValueError('Stay requires its explicit planned-priority declaration')
    if value.get('version') != VERSION or value.get('policy') != POLICY or value.get('scope') != POLICY['scope']:
        raise ValueError('Stay priority selection has another scope/policy')
    if value.get('pool_reference') != row.get('projection_pool_reference'):
        raise ValueError('Stay priority belongs to another original pool')
    side = value.get('scheduled_stay_side')
    if side not in ('A', 'B'):
        raise ValueError('Stay priority lacks its scheduled original side')
    for key in ('stay_yield_index', 'pair_index', 'view_index', 'original_domain_ordinal'):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError('Stay priority has an invalid original stream index')
    areas = value.get('predicted_best_one_step_area_px2')
    planned = value.get('planned_native_references')
    if not isinstance(areas, dict) or set(areas) != {'w0', 'w1'} or not isinstance(planned, dict) or set(planned) != {'w0', 'w1'}:
        raise ValueError('Stay priority requires both planned worlds, including unknowns')
    observed = [_area(areas[world]) for world in ('w0', 'w1')]
    known = all(area is not None for area in observed)
    score = min(observed) if known else None
    if value.get('known_both') is not known or _area(value.get('priority_min_area_px2')) != score:
        raise ValueError('Stay priority score differs from its two planned predictions')
    for world in ('w0', 'w1'):
        token = planned[world]
        point = pools._xyz(token['position'])
        if token.get('spawn') is not None or token.get('spawn_reference', {}).get('selected_position') != point:
            raise ValueError('Stay priority requires unchanged compact native references')
    # Final native proposals can be rewritten by real fallback. Only the
    # unchanged original role-domain indices are compared here.
    domain = row.get('proposal', {}).get('role_domain', {})
    for key, expected in (('stay_side', side), ('stay_yield_index', value['stay_yield_index']),
                          ('pair_index', value['pair_index']), ('view_index', value['view_index'])):
        if domain.get(key) != expected:
            raise ValueError('Stay priority does not bind its original role-domain row')


def peek_stay(item, *, check_deadline):
    """Return a compact score for the next original Stay; no iterator writes."""
    check_deadline()
    stream, record = item['stream'].streams['look'], item['record']
    remaining = stream.summary()['remaining']
    wanted = 'A' if stream.stay_count % 2 == 0 else 'B'
    side = wanted if remaining['stay_' + wanted.lower()] else ('B' if wanted == 'A' else 'A')
    key = 'stay_' + side.lower()
    if not remaining[key]:
        return None
    first, second, pair_index, view_index = stream.rows[key][stream.cursor[key]]
    if stream.views[view_index] != record['view']:
        raise ValueError('Stay priority needs the exact original pool view')
    anchor = record['flip_geometry_context']['anchor']
    areas = {}
    for world, token in (('w0', first), ('w1', second)):
        check_deadline()
        matches = [entry for entry in record['stay_points'][side]
                   if pools._point_key(entry['token']) == pools._point_key(token)]
        if len(matches) != 1:
            raise ValueError('Scheduled Stay point is absent/duplicated in its original pool')
        prediction = boxes.predict(matches[0], anchor, record['camera'], record['observation_cameras'])
        known = [_area(action['projection'].get('clipped_rectangle_area_px2'))
                 for action in prediction['after_actions'] if action['projection'].get('projectable') is True]
        known = [area for area in known if area is not None]
        areas[world] = max(known) if known else None
    known = all(area is not None for area in areas.values())
    return {'version': VERSION, 'policy': copy.deepcopy(POLICY), 'scope': POLICY['scope'],
        'pool_reference': copy.deepcopy(record['reference']), 'scheduled_stay_side': side,
        'stay_yield_index': stream.stay_count, 'pair_index': pair_index, 'view_index': view_index,
        'planned_native_references': {'w0': geometry.native_reference(first), 'w1': geometry.native_reference(second)},
        'predicted_best_one_step_area_px2': areas, 'known_both': known,
        'priority_min_area_px2': min(areas.values()) if known else None,
        'original_domain_ordinal': item['_stay_priority_domain_ordinal']}


class DomainOrder:
    def __init__(self):
        self.next_ordinal = 0

    def ordered(self, active):
        for item in active:
            if '_stay_priority_domain_ordinal' not in item:
                item['_stay_priority_domain_ordinal'] = self.next_ordinal
                self.next_ordinal += 1
        return sorted(active, key=lambda item: item['_stay_priority_domain_ordinal'])


class PriorityRound(_ORIGINAL_ROUND):
    def __init__(self, active, quota, order, *, check_deadline):
        super().__init__(order.ordered(active))
        self.quota, self.check_deadline = quota, check_deadline

    def pop(self, requested_pool):
        if requested_pool not in ('look', 'direct'):
            raise ValueError('Unknown original requested pool')
        self.check_deadline()
        choices = []
        if self.quota.wanted() == 'stay':
            for index, item in enumerate(self.pending):
                proof = peek_stay(item, check_deadline=self.check_deadline)
                if proof is not None:
                    score = proof['priority_min_area_px2']
                    choices.append(((score is None, 0. if score is None else -score, index), index, proof))
        if not choices:
            item = super().pop(requested_pool)
            item['stream']._stay_priority_pending = None
            return item
        _, index, proof = min(choices, key=lambda value: value[0])
        item = self.pending.pop(index)
        item['stream']._stay_priority_pending = proof
        return item


class PriorityIterator(flip.FlipIterator):
    def __next__(self):
        proof = getattr(self, '_stay_priority_pending', None)
        self._stay_priority_pending = None
        proposal, view, domain = super().__next__()
        if proposal['mode'] == 'stay':
            if proof is None:
                raise ValueError('Stay was yielded without the original pending-domain selection')
            expected = {world: geometry.native_reference(proposal[world]) for world in ('w0', 'w1')}
            if expected != proof['planned_native_references']:
                raise ValueError('Stay peek and actual original iterator yield differ')
            for key, source in (('stay_side', 'scheduled_stay_side'), ('stay_yield_index', 'stay_yield_index'),
                                ('pair_index', 'pair_index'), ('view_index', 'view_index')):
                if domain[key] != proof[source]:
                    raise ValueError('Stay peek consumed or changed the original stream state')
            proposal[SELECTION] = proof
        else:
            if proof is not None:
                raise ValueError('A planned Stay silently yielded Flip')
            proposal[SELECTION] = None
        return proposal, view, domain
