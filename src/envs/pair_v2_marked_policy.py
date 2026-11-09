"""Supplement-7 marked-candidate semantics; historical category policy unchanged."""
import copy
import itertools
import math

import numpy as np

from .pair_v2_policy import area_passes, DISTANCE_METHOD
from .pair_v2_search_policy import eligible_moved_object, object_inventory, SEARCH_POLICY
from .pair_approved_policy import position_distance

CANDIDATE_POLICY = 'two_marked_pickupables_v1'
CANDIDATE_POLICY_SPEC = {
    'version': CANDIDATE_POLICY,
    'candidate_domain': 'exactly the two stable-name marked pickupables A/B; other objects are not selectable targets',
    'v0': 'each actual mask >=250 at255 and 3D metadata centre distance from agent <=1.5m',
    'ordering': 'for each actual v0: same-category pairs first, then different-category pairs; stable names within pairs',
    'anchor': 'one object of its category in full inventory; category differs from both candidates; unchanged nonrolling moved-object filter',
    'instruction': 'Select the marked object closest to the <anchor_type>.',
    'marks': 'only A/B receive marks; unchanged actual mask/mark rendering',
    'distance_method': DISTANCE_METHOD, 'closest_separation_ratio': 1.5,
    'default_cloning': False, 'same_class_is_acceptance_filter': False,
    'scope': 'construction-stage qualification; T02 table/BFS/cache checks remain separate',
    'nonrolling_anchor_exclusions': {key: copy.deepcopy(SEARCH_POLICY[key]) for key in
        ('excluded_moved_object_types', 'excluded_native_variant_types', 'excluded_type_suffixes',
         'excluded_state_flags', 'category_matching', 'native_category_source', 'native_state_source')},
}


def all_candidate_pairs(objects):
    inventory = object_inventory(objects)
    candidates = sorted((obj for obj in inventory.values() if obj.get('pickupable') is True), key=lambda obj: obj['name'])
    pairs = list(itertools.combinations(candidates, 2))
    return sorted(pairs, key=lambda pair: (pair[0]['objectType'] != pair[1]['objectType'], pair[0]['name'], pair[1]['name']))


def identity_sets_at_view(capture):
    """Actual masks/metadata only; a geometric projection cannot call this true."""
    inventory = object_inventory(capture['metadata']['objects'])
    rgb = np.asarray(capture['rgb'])
    if rgb.dtype != np.uint8 or rgb.shape != (255, 255, 3):
        raise ValueError('Marked candidate discovery requires actual 255x255 RGB')
    agent = capture['metadata']['agent']['position']
    visible = set()
    for name, obj in inventory.items():
        if obj.get('pickupable') is not True:
            continue
        mask = np.asarray(capture['masks_by_name'].get(name, np.zeros((255, 255), dtype=bool)))
        if mask.shape != (255, 255) or mask.dtype != np.bool_:
            raise ValueError('Malformed actual candidate mask')
        if area_passes(int(mask.sum()), 255, 255) and position_distance(agent, obj['position']) <= 1.5:
            visible.add(name)
    counts = {}
    for obj in inventory.values():
        counts[obj['objectType']] = counts.get(obj['objectType'], 0) + 1
    result = []
    for a, b in all_candidate_pairs(list(inventory.values())):
        if not {a['name'], b['name']} <= visible:
            continue
        for anchor in sorted(inventory.values(), key=lambda obj: (obj['objectType'], obj['name'])):
            if (anchor['objectType'] in (a['objectType'], b['objectType'])
                    or counts[anchor['objectType']] != 1 or not eligible_moved_object(anchor)):
                continue
            result.append({'candidate_names': [a['name'], b['name']], 'anchor_name': anchor['name'],
                'target_type': 'marked object', 'anchor_type': anchor['objectType'],
                'candidate_types': [a['objectType'], b['objectType']],
                'same_class': a['objectType'] == b['objectType'],
                'candidate_policy': CANDIDATE_POLICY, 'search_candidate_origin': 'natural'})
    return result


def closest_target(objects, candidate_names, anchor_name):
    inventory = object_inventory(objects)
    names = list(candidate_names)
    if (len(names) != 2 or names != sorted(set(names)) or anchor_name in names
            or not set((*names, anchor_name)) <= set(inventory)):
        raise ValueError('Need sorted distinct marked A/B and a distinct present anchor')
    a, b, anchor = (inventory[name] for name in (*names, anchor_name))
    reasons = []
    if not all(obj.get('pickupable') is True for obj in (a, b)):
        reasons.append('marked_candidate_not_pickupable')
    if not eligible_moved_object(anchor):
        reasons.append('anchor_not_eligible_nonrolling_pickupable')
    if anchor['objectType'] in (a['objectType'], b['objectType']):
        reasons.append('anchor_category_not_distinct')
    if sum(obj['objectType'] == anchor['objectType'] for obj in inventory.values()) != 1:
        reasons.append('category_only_anchor_not_unique')
    distances = {name: position_distance(inventory[name]['position'], anchor['position']) for name in names}
    ordered = sorted(names, key=lambda name: (distances[name], name))
    near, far = [distances[name] for name in ordered]
    if near == far:
        reasons.append('nearest_target_tie')
    if far < 1.5 * near:
        reasons.append('closest_ratio_below_1_5')
    return {'passed': not reasons, 'target_name': ordered[0] if not reasons else None,
        'candidate_names': names, 'anchor_name': anchor_name, 'distances_m': distances,
        'distance_method': DISTANCE_METHOD, 'candidate_policy': CANDIDATE_POLICY,
        'candidate_types': [a['objectType'], b['objectType']], 'same_class': a['objectType'] == b['objectType'],
        'reasons': reasons}


def stay_horizontal_displacement(worlds, anchor_name):
    """Measured diagnostic only; no newly invented minimum displacement."""
    objects = [object_inventory(world['objects'])[anchor_name] for world in worlds]
    a, b = [obj['position'] for obj in objects]
    return {'distance_m': math.hypot(a['x'] - b['x'], a['z'] - b['z']),
            'positions': [copy.deepcopy(a), copy.deepcopy(b)],
            'method': 'realized metadata anchor positions in XZ', 'minimum_required_m': None}
