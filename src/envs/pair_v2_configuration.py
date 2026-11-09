"""Approved test configuration identity and seed-zero representative selection.

Only canonical input commands define the start. Unity readback and edited anchor
poses are deliberately absent; this is an inventory rule, not a physics gate.
"""
import copy
import hashlib
import json
import math
import random

VERSION = 'test_configuration_canonical_start_seed0_v1'
KINDS = ('direct_flip', 'direct_stay', 'look_flip', 'look_stay')
POLICY = {'version': VERSION, 'approval': '83828652 human approved',
    'key': 'scene; original canonical agent setup command; stable anchor name; unordered two stable candidate names',
    'start': 'literal generator command; no native readback, rounding, or edited object poses',
    'selection': 'one per configuration and actual class; Random(SHA256(version,seed0,configuration_id)).choice over sorted unique proposal identities',
    'seed': 0, 'scope': 'test inventory and proposal capacity only; scientific acceptance unchanged'}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Configuration coordinates must be finite numbers')
    # Numeric equality is exact: int/float and signed zero represent the same
    # command value. No decimal rounding or physical tolerance is introduced.
    result = 0.0 if value == 0 else float(value)
    if type(value) is int and result != value:
        raise ValueError('Integer command coordinate is not exactly representable as a float')
    return result


def from_inputs(scene, view, identities):
    if not isinstance(scene, str) or not scene.startswith('FloorPlan'):
        raise ValueError('Configuration requires an explicit scene')
    if not isinstance(view, dict) or set(view) != {'position', 'yaw', 'horizon'}:
        raise ValueError('Configuration requires the exact original view command')
    if not isinstance(view['position'], dict) or set(view['position']) != {'x', 'y', 'z'}:
        raise ValueError('Configuration requires three explicit start coordinates')
    anchor, candidates = identities.get('anchor_name'), identities.get('candidate_names')
    if (not isinstance(anchor, str) or not anchor or not isinstance(candidates, (list, tuple))
            or len(candidates) != 2 or any(not isinstance(name, str) or not name for name in candidates)
            or len(set(candidates)) != 2 or anchor in candidates):
        raise ValueError('Configuration needs one anchor and two distinct stable candidate names')
    command = {'position': {axis: number(view['position'][axis]) for axis in 'xyz'},
        'rotation': {'x': 0.0, 'y': number(view['yaw']), 'z': 0.0},
        'horizon': number(view['horizon']), 'standing': True, 'forceAction': False}
    key = {'scene': scene, 'start_state': command, 'anchor_name': anchor,
           'candidate_names': sorted(candidates)}
    return {'version': VERSION, 'configuration_id': hashlib.sha256(encoded(key)).hexdigest(), 'key': key}


def from_pair(row, worlds=None):
    value = from_inputs(row['scene'], row['view'], row['identities'])
    if worlds is not None:
        if not isinstance(worlds, list) or len(worlds) != 2:
            raise ValueError('Need both authenticated original worlds')
        command = value['key']['start_state']
        if any(world.get('agent_setup_command') != command for world in worlds):
            raise ValueError('Portable world command differs from the original declared view')
    return value


def validate(value):
    if not isinstance(value, dict) or set(value) != {'version', 'configuration_id', 'key'}:
        raise ValueError('Malformed configuration identity')
    key = value['key']
    if not isinstance(key, dict) or set(key) != {'scene', 'start_state', 'anchor_name', 'candidate_names'}:
        raise ValueError('Malformed configuration key')
    command = key['start_state']
    try:
        expected = from_inputs(key['scene'], {'position': command['position'],
            'yaw': command['rotation']['y'], 'horizon': command['horizon']}, key)
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete canonical configuration command') from error
    if value != expected:
        raise ValueError('Configuration identity differs from its canonical command and hash')
    return copy.deepcopy(expected)


def select(records):
    """Input records must already be independently authenticated unique proposals."""
    groups, identities = {}, set()
    for row in records:
        value = validate(row['configuration'])
        if row.get('kind') not in KINDS or not isinstance(row.get('proposal_identity'), str):
            raise ValueError('Configuration selection requires original class/proposal identity')
        if row['proposal_identity'] in identities:
            raise ValueError('Deduplicate original proposal identities before configuration selection')
        identities.add(row['proposal_identity'])
        groups.setdefault((value['configuration_id'], row['kind']), []).append(row)
    selected, duplicates, seen = [], [], []
    for (identity, kind), values in sorted(groups.items()):
        values = sorted(values, key=lambda row: row['proposal_identity'])
        seed = int(hashlib.sha256(encoded([VERSION, 0, identity])).hexdigest(), 16)
        chosen = random.Random(seed).choice(values)
        selected.append(copy.deepcopy(chosen))
        seen.append({'configuration': copy.deepcopy(chosen['configuration']), 'kind': kind})
        duplicates.extend({'configuration_id': identity, 'kind': kind,
            'selected_proposal_identity': chosen['proposal_identity'], 'excluded': copy.deepcopy(row)}
            for row in values if row is not chosen)
    return {'version': VERSION, 'policy': copy.deepcopy(POLICY), 'selected': selected,
            'duplicates': duplicates, 'seen_configurations': seen}


def validate_seen(rows, scene=None):
    if not isinstance(rows, list):
        raise ValueError('Require explicit confirmed configuration occupancy')
    keys = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'configuration', 'kind'} or row['kind'] not in KINDS:
            raise ValueError('Malformed confirmed configuration/class occupancy')
        value = validate(row['configuration'])
        identity = (value['configuration_id'], row['kind'])
        if identity in keys or (scene is not None and value['key']['scene'] != scene):
            raise ValueError('Duplicate or foreign configuration occupancy')
        keys.add(identity)
    return copy.deepcopy(rows)
