"""Supplement-5 proposal assignment only; realized targets remain authoritative.

No rendering, acceptance threshold, eligible token, or view changes here. A/B
mean the original stable-name candidate order, never the visible mark labels.
"""
import copy
import hashlib
import itertools
import json
import math
import re

ROLE_VERSION = 'candidate_roles_sha256_view_round_robin_v1'
ROLE_POLICY = {
    'version': ROLE_VERSION,
    'seed': 0,
    'candidate_order': 'unchanged lexicographically sorted stable names',
    'flip_assignment': 'SHA256 canonical JSON {attempt_index,scene,seed}; first byte low bit; 0=A,1=B in w0',
    'hash_encoding': 'UTF-8; json sort_keys=True,separators=(comma,colon),ensure_ascii=True',
    'stay_schedule': 'alternate A/B after view expansion within each candidate-anchor combination',
    'empty_or_exhausted_side': 'emit only actual remaining proposals and explicitly record fallback',
    'mode_schedule': 'interleave view-expanded Flip and Stay; continue remaining mode on exhaustion',
    'eligibility': 'original placement tokens, same-position rejection, and original views unchanged',
    'interpretation': 'proposal roles only; no claim of balanced accepted data or physical validity',
}
GENERATION_VERSION = ROLE_VERSION
GENERATION_POLICY = ROLE_POLICY


def _identity(scene, attempt_index, seed):
    if (not isinstance(scene, str) or re.fullmatch(r'FloorPlan[1-9][0-9]*', scene) is None
            or type(attempt_index) is not int or attempt_index < 0 or type(seed) is not int or seed != 0):
        raise ValueError('Require a native scene, nonnegative numeric attempt index, and fixed seed 0')
    return {'attempt_index': attempt_index, 'scene': scene, 'seed': seed}


def flip_orientation(scene, attempt_index, seed=0):
    payload = _identity(scene, attempt_index, seed)
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    digest = hashlib.sha256(raw).hexdigest()
    swapped = bool(int(digest[:2], 16) & 1)
    return {**payload, 'generation_version': ROLE_VERSION, 'assignment_digest': digest,
            'flip_swapped': swapped, 'w0_source_side': 'B' if swapped else 'A',
            'w1_source_side': 'A' if swapped else 'B'}


class RoleProposalIterator:
    """Finite original domains; alternating Stay operates on individual views."""
    def __init__(self, left, right, views):
        self.points = {'A': copy.deepcopy(list(left)), 'B': copy.deepcopy(list(right))}
        self.views = copy.deepcopy(list(views))
        pairs = {
            'flip': list(itertools.product(self.points['A'], self.points['B'])),
            'stay_a': list(itertools.combinations(self.points['A'], 2)),
            'stay_b': list(itertools.combinations(self.points['B'], 2)),
        }
        self.pairs = {key: [(first, second) for first, second in values
                           if first['position'] != second['position']] for key, values in pairs.items()}
        self.rows = {key: [(first, second, pair_index, view_index)
            for pair_index, (first, second) in enumerate(values)
            for view_index in range(len(self.views))] for key, values in self.pairs.items()}
        self.cursor = {key: 0 for key in self.rows}
        self.totals = {key: len(values) for key, values in self.rows.items()}
        self.next_kind, self.stay_count = 'flip', 0

    def __iter__(self):
        return self

    def summary(self):
        remaining = {key: self.totals[key] - self.cursor[key] for key in self.rows}
        return {'generation_version': ROLE_VERSION, 'view_count': len(self.views),
            'point_counts': {key: len(values) for key, values in self.points.items()},
            'placement_pair_counts': {key: len(values) for key, values in self.pairs.items()},
            'domain_totals': dict(self.totals), 'yielded': dict(self.cursor),
            'remaining': remaining, 'unavailable_at_start': [key for key in self.rows if not self.totals[key]],
            'exhausted': [key for key in self.rows if self.totals[key] and not remaining[key]],
            'all_exhausted': not any(remaining.values()), 'accepted_balance': None}

    def __next__(self):
        remaining = self.summary()['remaining']
        if not any(remaining.values()):
            raise StopIteration
        wanted_kind = self.next_kind
        has_stay = bool(remaining['stay_a'] or remaining['stay_b'])
        kind = wanted_kind
        if kind == 'flip' and not remaining['flip']:
            kind = 'stay'
        elif kind == 'stay' and not has_stay:
            kind = 'flip'
        self.next_kind = 'stay' if kind == 'flip' else 'flip'
        stay_side = requested_side = stay_index = None
        if kind == 'flip':
            key, sides = 'flip', ['A', 'B']
        else:
            stay_index = self.stay_count
            requested_side = 'A' if stay_index % 2 == 0 else 'B'
            stay_side = requested_side
            if not remaining['stay_' + stay_side.lower()]:
                stay_side = 'B' if stay_side == 'A' else 'A'
            key, sides = 'stay_' + stay_side.lower(), [stay_side, stay_side]
            self.stay_count += 1
        first, second, pair_index, view_index = self.rows[key][self.cursor[key]]
        self.cursor[key] += 1
        status = self.summary()
        domain = {'generation_version': ROLE_VERSION, 'mode': kind, 'source_sides': sides,
            'stay_side': stay_side, 'stay_yield_index': stay_index, 'requested_stay_side': requested_side,
            'fallback_due_to_exhaustion': kind == 'stay' and stay_side != requested_side,
            'requested_mode': wanted_kind, 'mode_fallback': kind != wanted_kind,
            'pair_index': pair_index, 'view_index': view_index, 'view_count': len(self.views),
            'domain_totals': dict(self.totals), 'remaining_after_yield': status['remaining'],
            'unavailable_at_start': status['unavailable_at_start'], 'exhausted_after_yield': status['exhausted']}
        proposal = {'mode': kind, 'w0': copy.deepcopy(first), 'w1': copy.deepcopy(second)}
        return proposal, copy.deepcopy(self.views[view_index]), domain


def view_expanded_proposals(left, right, views):
    """Yield (proposal, view, role_domain); .summary() also describes empty input."""
    return RoleProposalIterator(left, right, views)


def orient_proposal(proposal, scene, attempt_index, seed=0):
    identity = _identity(scene, attempt_index, seed)
    result = copy.deepcopy(proposal)
    domain = result.get('role_domain')
    mode = result.get('mode')
    if not isinstance(domain, dict) or domain.get('generation_version') != ROLE_VERSION or domain.get('mode') != mode:
        raise ValueError('Proposal requires its explicit original role domain')
    if mode == 'flip':
        if domain.get('source_sides') != ['A', 'B'] or domain.get('stay_side') is not None:
            raise ValueError('Flip source domain must retain original A/B order')
        metadata = flip_orientation(scene, attempt_index, seed)
        if metadata['flip_swapped']:
            result['w0'], result['w1'] = result['w1'], result['w0']
        stay_side = None
    elif mode == 'stay':
        stay_side = domain.get('stay_side')
        if stay_side not in ('A', 'B') or domain.get('source_sides') != [stay_side, stay_side]:
            raise ValueError('Stay source domain must name one actual candidate side')
        metadata = {**identity, 'generation_version': ROLE_VERSION, 'assignment_digest': None,
                    'flip_swapped': None, 'w0_source_side': stay_side, 'w1_source_side': stay_side}
    else:
        raise ValueError('Unknown proposal mode')
    metadata.update(mode=mode, stay_side=stay_side, original_role_domain=copy.deepcopy(domain),
                    scope='intended proposal roles; actual closest targets are independently measured')
    return result, metadata


def verify_proposal_role(row, candidate_positions):
    """Verify assignment using independently authenticated common-baseline positions.

    Actual 3D targets need not equal the horizontal proposal's intended side;
    that diagnostic cannot replace measured targets or become a new threshold.
    """
    names = row['identities']['candidate_names']
    if (not isinstance(names, list) or len(names) != 2 or names != sorted(set(names))
            or any(name not in candidate_positions for name in names)):
        raise ValueError('Need the unchanged stable A/B names and authenticated positions')
    match = re.fullmatch(re.escape(row['scene'].lower()) + r'_([0-9]{5})', row['pair_id'])
    if match is None:
        raise ValueError('Pair ID does not identify its actual numeric scene attempt')
    proposal, declared = row['proposal'], row['proposal_role']
    canonical = copy.deepcopy(proposal)
    assignment = flip_orientation(row['scene'], int(match.group(1)), 0)
    if proposal['mode'] == 'flip' and assignment['flip_swapped']:
        canonical['w0'], canonical['w1'] = canonical['w1'], canonical['w0']
    expected_proposal, expected = orient_proposal(canonical, row['scene'], int(match.group(1)), 0)
    expected.update(candidate_a_name=names[0], candidate_b_name=names[1])
    if (row.get('generation_version') != ROLE_VERSION or row.get('generation_policy') != ROLE_POLICY
            or declared != expected or expected_proposal != proposal or row.get('mode') != proposal['mode']):
        raise ValueError('Recorded proposal assignment differs from its fixed deterministic rule')
    domain = proposal['role_domain']
    if proposal['mode'] == 'stay':
        ordinal = domain.get('stay_yield_index')
        requested = 'A' if type(ordinal) is int and ordinal >= 0 and ordinal % 2 == 0 else 'B'
        if (type(ordinal) is not int or ordinal < 0 or domain.get('requested_stay_side') != requested
                or domain.get('fallback_due_to_exhaustion') is not (requested != domain['stay_side'])
                or (requested != domain['stay_side'] and domain['remaining_after_yield']['stay_' + requested.lower()] != 0)):
            raise ValueError('Stay alternation/fallback is inconsistent with the observed proposal domain')
    sides = []
    for key in ('w0', 'w1'):
        token = proposal[key]
        point = token['spawn']['selected_position']
        if point != token['position']:
            raise ValueError('Role position is not the actual native proposal coordinate')
        distances = [math.hypot(point['x'] - candidate_positions[name]['x'],
                                point['z'] - candidate_positions[name]['z']) for name in names]
        if (any(not math.isfinite(value) for value in distances) or distances[0] == distances[1]
                or max(distances) < 1.5 * min(distances)):
            raise ValueError('Native proposal does not belong to its original separated horizontal side')
        sides.append('A' if distances[0] < distances[1] else 'B')
    if sides != [declared['w0_source_side'], declared['w1_source_side']]:
        raise ValueError('Declared candidate side does not match the authenticated proposal geometry')
    targets = [row.get('target_w0'), row.get('target_w1')]
    if any(target not in names for target in targets):
        raise ValueError('Accepted row lacks actual candidate target identities')
    actual = ['A' if target == names[0] else 'B' for target in targets]
    return {'passed': True, 'intended_candidate_roles': sides, 'actual_candidate_roles': actual,
            'actual_targets_match_intended': actual == sides, 'assignment_digest': declared['assignment_digest'],
            'scope': 'verified proposal assignment; actual targets and accepted distribution remain separate'}
