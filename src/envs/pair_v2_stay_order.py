"""Supplement 11 search order only; installed by its isolated producer.

The existing physical, pixel, target and solvability checks are not replaced.
"""
import copy
import json
import math

from . import pair_v2_projection_pool as pools

VERSION = 'stay_w1_far_first_without_repeat_v1'
POLICY = {
    'version': VERSION,
    'enumeration': 'same unordered Stay combinations; for each fixed w0, order its remaining w1 suffix far-to-near',
    'distance': '3D Euclidean native position; descending; deterministic ascending native-key ties',
    'native_origin': 'actual physically settled w0 anchor metadata.position, including successful native fallback',
    'native_order': 'scheduled w1 first when w0 did not fall back and that point is untried; remaining retries far-to-near from actual w0',
    'fallback_order': 'when w0 fell back or scheduled w1 was already tried, remaining eligible same-side points far-to-near',
    'exploration': 'per Controller/common baseline/view/identity/pool/side/successful w0 point, skip only ledger-recorded w1 attempts',
    'unchanged': 'native tokens, pool membership/caps, Flip sequence, Stay A/B alternation, Look/Direct alternation, seed0, all acceptance thresholds',
}
FIELDS = ('stay_w1_order_version', 'stay_w1_order_policy')
PRODUCER = 'src.envs.pair_v2_stay_shard'


def fields():
    return {FIELDS[0]: VERSION, FIELDS[1]: copy.deepcopy(POLICY)}


def validate_fields(value):
    if any(value.get(key) != expected for key, expected in fields().items()):
        raise ValueError('Unknown or altered supplement 11 Stay search order')


def validate_source(config, metrics=None):
    validate_fields(config)
    validate_fields(config['shard'])
    if config.get('producer_module') != PRODUCER or config['shard'].get('producer_module') != PRODUCER:
        raise ValueError('Stay producer differs from immutable shard')
    if metrics is not None:
        validate_fields(metrics)
        if metrics.get('producer_module') != PRODUCER:
            raise ValueError('Stay producer metrics differ from source configuration')


def ordered(tokens, origin):
    point = pools._xyz(origin)
    return sorted(tokens, key=lambda token: (-math.dist(
        [pools._number(token['position'][axis]) for axis in ('x', 'y', 'z')],
        [point[axis] for axis in ('x', 'y', 'z')]), pools._point_key(token)))


class FarFirstPoolRoles(pools._PoolRoles):
    """Reorder only the existing Stay lists; keep parent scheduling unchanged."""
    def __init__(self, records, pool):
        super().__init__(records, pool)
        for mode in ('stay_a', 'stay_b'):
            by_view = [[] for _ in records]
            for first, second, _, view_index in self.rows[mode]:
                by_view[view_index].append((first, second))
            for index, pairs in enumerate(by_view):
                groups = {}  # insertion order preserves the original w0 sequence
                for first, second in pairs:
                    groups.setdefault(pools._point_key(first), (first, []))[1].append(second)
                by_view[index] = [(first, second) for first, seconds in groups.values()
                                  for second in ordered(seconds, first['position'])]
            self.rows[mode], self.pairs[mode] = [], []
            for ordinal in range(max((len(rows) for rows in by_view), default=0)):
                for view_index, rows in enumerate(by_view):
                    if ordinal < len(rows):
                        first, second = rows[ordinal]
                        self.rows[mode].append((first, second, len(self.pairs[mode]), view_index))
                        self.pairs[mode].append((first, second))
            if len(self.rows[mode]) != self.totals[mode]:
                raise ValueError('Stay ordering changed the original proposal inventory')


def state_key(baseline, record, role, pool, side, anchor, successful_w0):
    """State is attached to one simulator; no Controller identity guesswork."""
    return json.dumps({'baseline': baseline['reference'], 'view': record['view'],
        'identity': record.get('identity', {'candidate_names': [role.get('candidate_a_name'), role.get('candidate_b_name')]}),
        'pool': pool, 'side': side, 'anchor': anchor, 'successful_w0': pools._point_key(successful_w0)},
        sort_keys=True, separators=(',', ':'), allow_nan=False)


def alternatives(record, pool, side, selected, w0, origin, tried, *, w0_fell_back):
    values = [item['token'] for item in record['points'][pool][side]]
    if selected not in values or w0 not in values:
        raise ValueError('Stay native tokens must belong to the declared same-side pool')
    remaining = [token for token in values if token['position'] != w0['position']
                 and pools._point_key(token) not in tried]
    result = ordered(remaining, origin)
    if not w0_fell_back and selected in result:
        result = [selected] + [token for token in result if token != selected]
    return result[:40]
