"""Pure permutation of the existing finite native placement/view proposal set.

This module does not activate itself. A future versioned producer may install
the factory in its private process. Historical/running producer files stay
unchanged. Geometry, visibility, physics and evidence acceptance are untouched.
"""
from contextlib import contextmanager
import copy
import math

from . import pair_v2_role_policy as roles

ORDER_VERSION = 'placement_view_diagonal_coverage_v1'
ORDER_POLICY = {
    'version': ORDER_VERSION,
    'domain': 'exact existing native-filtered placement pairs and existing views; no additions, removals or result-conditioned filtering',
    'view_order': 'positions in original first-occurrence order; within each position horizon-major original heading order; cyclically stagger local slots across positions',
    'product_order': 'gcd(P,V) cyclic diagonals, each of lcm(P,V) steps: pair=step mod P, view_rank=(step+diagonal) mod V',
    'role_schedule': 'unchanged original Flip/Stay and expanded Stay A/B alternation with truthful exhaustion',
    'orientation': 'unchanged original scene plus actual attempt index SHA256 assignment with seed 0',
    'scope': 'search coverage order only; no claim of Look availability, accepted balance or native validity',
}
_ORIGINAL_FACTORY = roles.view_expanded_proposals


def view_index_order(views):
    """Permute indices, preserving even duplicate input records as occurrences."""
    positions = {}
    for index, view in enumerate(views):
        key = tuple(view['position'][axis] for axis in ('x', 'y', 'z'))
        positions.setdefault(key, []).append(index)
    groups = []
    for indices in positions.values():
        horizons = {}
        for index in indices:
            horizons.setdefault(views[index]['horizon'], []).append(index)
        groups.append([index for values in horizons.values() for index in values])
    width = max(map(len, groups), default=0)
    order = []
    for cycle in range(width):
        for position_index, indices in enumerate(groups):
            slot = (position_index + cycle) % width
            if slot < len(indices):
                order.append(indices[slot])
    return order


def product_index_order(pair_count, view_order):
    """Bijection of the finite Cartesian product, including unequal dimensions."""
    if type(pair_count) is not int or pair_count < 0:
        raise ValueError('Require a nonnegative integer placement-pair count')
    count = len(view_order)
    if sorted(view_order) != list(range(count)):
        raise ValueError('View indices must be an exact permutation of the original domain')
    if not pair_count or not count:
        return []
    divisor = math.gcd(pair_count, count)
    period = pair_count * count // divisor
    return [(step % pair_count, view_order[(step + diagonal) % count])
            for diagonal in range(divisor) for step in range(period)]


class CoverageProposalIterator(roles.RoleProposalIterator):
    """Only per-mode rows change; inherited mode/side/orientation rules remain."""
    def __init__(self, left, right, views):
        super().__init__(left, right, views)
        self.view_order = view_index_order(self.views)
        for key, original_rows in self.rows.items():
            indices = product_index_order(len(self.pairs[key]), self.view_order)
            self.rows[key] = [original_rows[pair_index * len(self.views) + view_index]
                              for pair_index, view_index in indices]

    def __next__(self):
        proposal, view, domain = super().__next__()
        key = 'flip' if domain['mode'] == 'flip' else 'stay_' + domain['stay_side'].lower()
        domain['proposal_order'] = {
            'version': ORDER_VERSION, 'mode_ordinal': self.cursor[key] - 1,
            'original_pair_index': domain['pair_index'], 'original_view_index': domain['view_index'],
        }
        return proposal, view, domain

    def summary(self):
        result = super().summary()
        result.update(proposal_order_version=ORDER_VERSION,
                      proposal_order_policy=copy.deepcopy(ORDER_POLICY),
                      original_view_index_order=list(self.view_order))
        return result


def coverage_view_expanded_proposals(left, right, views):
    return CoverageProposalIterator(left, right, views)


def enable_config(config):
    """Future entry must retain these tags in its config and hashed metrics."""
    config.update(proposal_order_version=ORDER_VERSION,
                  proposal_order_policy=copy.deepcopy(ORDER_POLICY))
    return config


@contextmanager
def install_coverage_order():
    """Private process hook only; callers own the new versioned entry/CLI."""
    previous = roles.view_expanded_proposals
    if previous is not _ORIGINAL_FACTORY:
        raise ValueError('Do not compose an unknown or already active proposal-order hook')
    try:
        roles.view_expanded_proposals = coverage_view_expanded_proposals
        yield
    finally:
        roles.view_expanded_proposals = previous
