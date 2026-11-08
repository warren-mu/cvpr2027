"""Pure T02 section-7 pool balancing; no rendering, files, or acceptance runs.

Rows are an explicit measured projection, not entire mutable view tables:
pair_id, scene, kind, candidate_names (sorted stable A/B names), target_names
(w0,w1), and s0_target_left (two measured bools or unknown None). The returned
world_order maps NEW world labels to original table worlds. Consumers must apply
this mapping to all world-specific fields, keeping original evidence intact.
"""
import copy
import hashlib
import json
from collections import Counter, defaultdict

VERSION = 't02_pair_id_flip_scene_stay_balance_v1'
KINDS = ('direct_flip', 'direct_stay', 'look_flip', 'look_stay')
LEFT_RULE = 'mean(column_index + 0.5) of visible target mask < image_width / 2; exact midline is not left; missing/empty is unknown'
POLICY = {
    'version': VERSION, 'rho': .5,
    'flip': 'SHA256(pair_id, seed, version, purpose) low bit; half-probability label swap',
    'stay': 'within each scene, pool both Stay classes and independently SHA-rank each actual A/B side; keep min side size',
    'world_order': 'new world index -> original table world index, applied to every world-specific field',
    'selection_uses_left_position': False,
    'left_measurement': LEFT_RULE,
    'global_bounds_inclusive': [.4, .6], 'scene_bounds_inclusive': [.3, .7],
    'scope': 'pool composition only; no physical/table/provenance acceptance claim',
}


def left_half_from_mask(mask, width, height):
    """Approved pixel-centre centroid convention; no world-space inference."""
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError('Positive image dimensions are required')
    if mask is None:
        return {'left': None, 'visible_pixels': None, 'centroid_pixel_center_x': None,
                'width': width, 'height': height, 'rule': LEFT_RULE}
    if type(mask) is not bytes or len(mask) != width * height or set(mask) - {0, 1}:
        raise ValueError('Require the complete binary instance mask bytes')
    indices = [index % width for index, bit in enumerate(mask) if bit]
    count = len(indices)
    # Compare doubled integer coordinates to preserve exact-midline semantics.
    twice_sum = sum(2 * column + 1 for column in indices)
    return {'left': twice_sum < count * width if count else None,
            'visible_pixels': count, 'centroid_pixel_center_x': twice_sum / (2 * count) if count else None,
            'width': width, 'height': height, 'rule': LEFT_RULE}


def normalize_table(table):
    """Project actual own-world s0 target masks without changing the table."""
    from .pair_table import PairTable
    if not isinstance(table, PairTable):
        raise ValueError('Require an explicit PairTable')
    views = table.states[table.initial_state].captures
    measurements = [left_half_from_mask(views[index].masks.get(target),
        views[index].rgb.width, views[index].rgb.height) for index, target in enumerate(table.target_names)]
    value = {'pair_id': table.pair_id, 'scene': table.scene, 'kind': table.kind,
        'candidate_names': list(table.candidate_names), 'target_names': list(table.target_names),
        's0_target_left': [item['left'] for item in measurements], 's0_left_measurements': measurements,
        'synthetic': table.synthetic, 'source': copy.deepcopy(table.source)}
    return checked_rows([value])[0]


def seeded_digest(pair_id, seed, purpose, *, scene=None):
    if not isinstance(pair_id, str) or not pair_id or type(seed) is not int:
        raise ValueError('Require a nonempty stable pair_id and integer seed')
    value = {'pair_id': pair_id, 'seed': seed, 'purpose': purpose, 'version': VERSION}
    if scene is not None:
        value['scene'] = scene
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def checked_rows(rows):
    result, seen = [], set()
    for row in rows:
        value = copy.deepcopy(row)
        names, targets, left = value.get('candidate_names'), value.get('target_names'), value.get('s0_target_left')
        if (not isinstance(value.get('pair_id'), str) or not value['pair_id'] or value['pair_id'] in seen
                or not isinstance(value.get('scene'), str) or not value['scene'] or value.get('kind') not in KINDS
                or not isinstance(names, list) or len(names) != 2
                or not all(isinstance(name, str) and name for name in names) or names != sorted(set(names))
                or not isinstance(targets, list) or len(targets) != 2 or any(target not in names for target in targets)
                or not isinstance(left, list) or len(left) != 2
                or any(item is not None and type(item) is not bool for item in left)):
            raise ValueError('Require unique pair IDs and explicit actual A/B targets and nullable left-half measurements')
        if (targets[0] != targets[1]) is not value['kind'].endswith('flip'):
            raise ValueError('Recorded actual targets disagree with Flip/Stay class')
        seen.add(value['pair_id'])
        result.append(value)
    return result


def proportion(rows):
    """Each retained pair contributes exactly two rho=.5 world observations."""
    total, a, left, unknown = len(rows) * 2, 0, 0, 0
    for row in rows:
        a += sum(target == row['candidate_names'][0] for target in row['target_names'])
        left += sum(item is True for item in row['s0_target_left'])
        unknown += sum(item is None for item in row['s0_target_left'])
    observed = total - unknown
    return {'pair_count': len(rows), 'weighted_world_denominator': total,
        'target_a_world_count': a, 'target_a_proportion': a / total if total else None,
        'target_left_world_count': left, 'target_left_unknown_world_count': unknown,
        'target_left_observed_world_denominator': observed,
        'target_left_proportion': left / total if total and not unknown else None,
        'target_left_observed_only_proportion': left / observed if observed else None}


def balance_statistics(rows, scenes):
    overall = proportion(rows)
    scene_rows, scene_class_rows, flagged, unknown = [], [], [], []
    for scene in sorted(set(scenes)):
        selected = [row for row in rows if row['scene'] == scene]
        record = {'scene': scene, **proportion(selected)}
        scene_rows.append(record)
        keys = ('target_a_proportion', 'target_left_proportion')
        outside = [key for key in keys if record[key] is not None and not .3 <= record[key] <= .7]
        missing = [key for key in keys if record[key] is None]
        if outside:
            flagged.append({'scene': scene, 'outside_bounds': outside, **{key: record[key] for key in keys}})
        if missing:
            unknown.append({'scene': scene, 'unknown_proportions': missing})
        for kind in KINDS:
            scene_class_rows.append({'scene': scene, 'kind': kind,
                **proportion([row for row in selected if row['kind'] == kind])})
    known = all(overall[key] is not None for key in ('target_a_proportion', 'target_left_proportion'))
    return {'rho': .5, 'overall': overall, 'by_scene': scene_rows, 'by_scene_class': scene_class_rows,
        'scenes_outside_0_3_to_0_7': flagged, 'scenes_with_unknown_proportions': unknown,
        'global_balance_within_0_4_to_0_6': (all(.4 <= overall[key] <= .6 for key in
            ('target_a_proportion', 'target_left_proportion')) if known else None),
        'full_T02_acceptance': None}


def balance_pairs(rows, *, seed=0):
    """Return a deterministic manifest; never relabel input evidence in place."""
    if type(seed) is not int:
        raise ValueError('Require an explicit integer seed')
    original = checked_rows(rows)
    stay = defaultdict(lambda: {'a': [], 'b': []})
    retain = set()
    for row in original:
        if row['kind'].endswith('flip'):
            retain.add(row['pair_id'])
        else:
            side = 'a' if row['target_names'][0] == row['candidate_names'][0] else 'b'
            stay[row['scene']][side].append(row)
    strata = []
    for scene, sides in sorted(stay.items()):
        limit = min(len(sides['a']), len(sides['b']))
        for side in ('a', 'b'):
            ordered = sorted(sides[side], key=lambda row: (
                seeded_digest(row['pair_id'], seed, 'stay_downsample', scene=scene), row['pair_id']))
            retain.update(row['pair_id'] for row in ordered[:limit])
        strata.append({'scene': scene, 'input_a': len(sides['a']), 'input_b': len(sides['b']),
                       'retained_per_side': limit, 'grouping': 'both Stay classes pooled'})
    selected, dropped, measured = [], [], []
    for row in sorted(original, key=lambda value: (value['scene'], value['kind'], value['pair_id'])):
        if row['pair_id'] not in retain:
            dropped.append({'pair_id': row['pair_id'], 'scene': row['scene'], 'kind': row['kind'],
                            'reason': 'stay_scene_majority_downsample', 'target_name': row['target_names'][0]})
            continue
        assignment_sha = seeded_digest(row['pair_id'], seed, 'flip_world_swap') if row['kind'].endswith('flip') else None
        swapped = bool(int(assignment_sha[:2], 16) & 1) if assignment_sha is not None else False
        order = [1, 0] if swapped else [0, 1]
        mapped = {**copy.deepcopy(row), 'target_names': [row['target_names'][index] for index in order],
                  's0_target_left': [row['s0_target_left'][index] for index in order]}
        measured.append(mapped)
        selected.append({'pair_id': row['pair_id'], 'scene': row['scene'], 'kind': row['kind'],
            'world_order': order, 'swapped': swapped, 'assignment_sha256': assignment_sha,
            'candidate_names': copy.deepcopy(row['candidate_names']),
            'original_target_names': copy.deepcopy(row['target_names']), 'target_names': mapped['target_names'],
            'original_s0_target_left': copy.deepcopy(row['s0_target_left']), 's0_target_left': mapped['s0_target_left']})
    scenes = [row['scene'] for row in original]
    return {'version': VERSION, 'policy': copy.deepcopy(POLICY), 'seed': seed,
        'selected': selected, 'dropped': dropped, 'stay_scene_strata': strata,
        'input_count_by_class': dict(Counter(row['kind'] for row in original)),
        'selected_count_by_class': dict(Counter(row['kind'] for row in measured)),
        'before': balance_statistics(original, scenes), 'after': balance_statistics(measured, scenes),
        'world_labels_have_no_intrinsic_meaning': True, 'source_tables_modified': False,
        'selection_independent_of_left_measurement': True, 'full_T02_acceptance': None}
