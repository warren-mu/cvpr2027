"""Deterministic geometric proposals for T01v2; no physics acceptance by geometry.

These functions only order proposed constructions. Every proposal must still be
rendered, settled, checked and replayed; a geometric pass is never a dataset row.
"""
import itertools
import math


def distance(a, b):
    return math.dist([float(a[k]) for k in ('x', 'y', 'z')],
                     [float(b[k]) for k in ('x', 'y', 'z')])


def identity_sets(objects):
    groups = {}
    for obj in objects:
        groups.setdefault(obj['objectType'], []).append(obj)
    pairs = []
    for category, items in sorted(groups.items()):
        if len(items) != 2 or not all(o.get('pickupable') is True for o in items):
            continue
        candidates = sorted(items, key=lambda o: o['name'])
        for anchor_type, anchors in sorted(groups.items()):
            if anchor_type == category or len(anchors) != 1 or anchors[0].get('pickupable') is not True:
                continue
            pairs.append({'candidate_names': [o['name'] for o in candidates],
                          'anchor_name': anchors[0]['name'], 'target_type': category,
                          'anchor_type': anchor_type})
    return pairs


def view_proposals(reachable, a, b, *, maximum_positions=12):
    """Fixed nearby graph positions and 45-degree headings, not visibility claims."""
    midpoint = {k: (a['position'][k] + b['position'][k]) / 2 for k in ('x', 'y', 'z')}
    unique = {tuple(float(p[k]) for k in ('x', 'y', 'z')): dict(p) for p in reachable}
    ordered = sorted(unique.values(), key=lambda p: (
        max(math.hypot(p['x']-o['position']['x'], p['z']-o['position']['z']) for o in (a, b)),
        distance(p, midpoint), p['x'], p['y'], p['z']))
    result = []
    for p in ordered[:maximum_positions]:
        yaw = math.degrees(math.atan2(midpoint['x']-p['x'], midpoint['z']-p['z'])) % 360
        heading = (int(math.floor(yaw / 45 + .5)) * 45) % 360
        for offset in (0, -45, 45):
            for horizon in (0, 30):
                result.append({'position': p, 'yaw': (heading+offset) % 360, 'horizon': horizon})
    return result


def placement_groups(points, a, b, anchor, *, maximum_per_side=6):
    """Order native placement tokens by horizontal proposal proximity.

    Bounding-box overlap is only a proposal filter. It cannot certify physics,
    and is not used to declare accepted/rejected physical validity.
    Native coordinates are trigger-top points, not the final object pivot;
    the actual settled target is subsequently checked with 3D distances.
    """
    groups = [[], []]
    seen = set()
    for token in points:
        p = token['position']
        key = (token['receptacle_name'], *(float(p[k]) for k in ('x', 'y', 'z')))
        if key in seen:
            continue
        seen.add(key)
        da, db = (math.hypot(p['x']-obj['position']['x'], p['z']-obj['position']['z']) for obj in (a,b))
        if da == db or max(da, db) < 1.5 * min(da, db):
            continue
        side = 0 if da < db else 1
        # Avoid a proposed centre landing within either candidate's box.
        overlap = False
        for obj in (a, b):
            box = obj.get('axisAlignedBoundingBox') or {}
            centre, size = box.get('center'), box.get('size')
            other_size = (anchor.get('axisAlignedBoundingBox') or {}).get('size')
            if centre and size and other_size:
                overlap |= all(abs(p[k]-centre[k]) < (size[k]+other_size[k])/2 for k in ('x', 'y', 'z'))
        if not overlap:
            groups[side].append((min(da, db), -max(da, db), key, token))
    return [[row[-1] for row in sorted(group, key=lambda x: x[:-1])[:maximum_per_side]] for group in groups]


def construction_pairs(left, right):
    """Interleave Flip and Stay proposals; no duplicated world pair is emitted."""
    flips = iter(itertools.product(left, right))
    stays = iter(itertools.chain(itertools.combinations(left, 2), itertools.combinations(right, 2)))
    while True:
        found = False
        for mode, source in (('flip', flips), ('stay', stays)):
            try:
                first, second = next(source)
            except StopIteration:
                continue
            found = True
            if first['position'] == second['position']:
                continue
            yield {'mode': mode, 'w0': first, 'w1': second}
        if not found:
            break


def adjacent_moves(reachable, start, *, maximum_distance=.26):
    """Actual adjacent grid positions; backend must execute and verify the move."""
    candidates = []
    for p in reachable:
        delta = distance(start, p)
        if 1e-6 < delta <= maximum_distance and abs(start['y']-p['y']) <= .01:
            dx, dz = abs(start['x']-p['x']), abs(start['z']-p['z'])
            if min(dx, dz) <= .01:
                candidates.append((delta, p['x'], p['y'], p['z'], dict(p)))
    return [item[-1] for item in sorted(candidates)[:4]]
