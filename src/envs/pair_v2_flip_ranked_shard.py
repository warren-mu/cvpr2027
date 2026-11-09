"""Add-only combined producer; every spawned interpreter installs its own hooks."""
import copy
from contextlib import contextmanager
import math
import time
from . import pair_v2_flip_rounds_shard as rounds
from . import pair_v2_flip_ranked_policy as policy
from . import pair_v2_spawn_sharing as sharing
from . import pair_v2_bbox_rank as rank
from . import pair_v2_fixed_labels as labels
from . import pair_v2_flip_rounds_search as cadence
from . import pair_v2_flip_geometry as geometry


def tags():
    return {**rounds.tags(), **sharing.fields(), **rank.fields(), **geometry.fields(),
        'flip_unit_version': policy.VERSION, 'flip_unit_policy': copy.deepcopy(policy.POLICY),
        'producer_module': policy.PRODUCER}


def compact_attempt(row):
    result = rounds.original.compact_attempt(row)
    for key in (*cadence.FIELDS, *sharing.FIELDS, *rank.FIELDS, *geometry.FIELDS,
                'flip_unit_version', 'flip_unit_policy', 'flip_geometry_initial_tuple',
                'flip_geometry_final_tuple', 'flip_geometry_native_guards'):
        result[key] = copy.deepcopy(row.get(key))
    return result


def deadline_check(config):
    if (config is None or type(config.get('deadline')) not in (int, float)
            or not math.isfinite(config['deadline'])):
        raise ValueError('Ranked generation requires the original finite scene soft deadline')
    deadline = config['deadline']
    def check():
        if time.time() >= deadline:
            raise TimeoutError('Original scene soft deadline during predicted-box proposal ordering')
    return check


def guarded_observer(original_observer, sim, proposal, role, record, row, **kwargs):
    """Guard the actual retry validator; retain original observer and physics."""
    if proposal['mode'] != 'flip':
        return original_observer(sim, proposal, role, record, row, **kwargs)
    sides = [role[world + '_source_side'] for world in ('w0', 'w1')]
    row['flip_geometry_initial_tuple'] = geometry.complete_pair(record, proposal['w0'], proposal['w1'], sides)
    original, observed = original_observer(sim, proposal, role, record, row, **kwargs)
    engine = rounds.original.marked.engine
    def construct(baseline, key, placements, require_agent=None):
        world = 'w0' if key.endswith('_w0') else 'w1' if key.endswith('_w1') else None
        if world is None:
            raise ValueError('Complete Flip guard needs an explicit world')
        other = 'w1' if world == 'w0' else 'w0'
        # The original observer rewrites proposal.w0 to the actual successful
        # fallback before the next world is constructed.
        opposite = copy.deepcopy(proposal[other])
        previous = engine.validate_spawn
        def validate(placement, scene):
            token = previous(placement, scene)
            geometry.validate_placement(record, token, role[world + '_source_side'],
                opposite=opposite, opposite_role=role[other + '_source_side'])
            row.setdefault('flip_geometry_native_guards', {}).setdefault(world, []).append({
                'native_reference': geometry.native_reference(token),
                'opposite_reference': geometry.native_reference(opposite),
                'source_side': role[world + '_source_side'], 'opposite_side': role[other + '_source_side']})
            return token
        try:
            engine.validate_spawn = validate
            result = observed(baseline, key, placements, require_agent=require_agent)
        finally:
            engine.validate_spawn = previous
        row['flip_geometry_final_tuple'] = geometry.complete_pair(record, proposal['w0'], proposal['w1'], sides)
        return result
    return original, construct


@contextmanager
def installed(config=None):
    # main installs top-level spawn targets; only the spawned generator can
    # bind a native search deadline and actually build proposal pools.
    check = deadline_check(config) if config is not None else None
    previous_policy = rounds.policy
    rounds.policy = policy
    try:
        with rounds.installed() as quota, sharing.installed():
            marked = rounds.original.marked
            original_full_pool = marked.pools.build_view_pools
            original_observer, original_build = marked.construct_observer, marked.measured_build
            def full_pool(left, right, a, b, anchor, **kwargs):
                if check is None:
                    raise RuntimeError('Proposal ranking is only bound inside the spawned generator')
                complete = geometry.original_tokens(left, right)
                # Only the historical Stay lists enter the old projector. The
                # unpruned native stream is projected once by the new filter.
                groups = [rounds.original.search.PointGroup([], value.stay_values) for value in (left, right)]
                record = original_full_pool(*groups, a, b, anchor, **kwargs)
                record = geometry.filter_pool(record, a, b, anchor,
                    all_native_tokens=complete, check_deadline=check)
                value = rank.rank_pool(record, anchor, check_deadline=check)
                value.update(sharing.fields())
                return value
            def build(sim, cfg, baseline, identities, proposal, view, pair_id, directory, reachable, attempt_row=None):
                if proposal['mode'] == 'flip':
                    geometry.complete_pair(proposal['_private_pool_record'], proposal['w0'], proposal['w1'])
                return original_build(sim, cfg, baseline, identities, proposal, view, pair_id,
                    directory, reachable, attempt_row=attempt_row)
            def observer(*args, **kwargs):
                return guarded_observer(original_observer, *args, **kwargs)
            targets = [(marked, 'tags', tags), (marked, 'compact_attempt', compact_attempt),
                (marked, 'generator_worker', generator_worker), (marked, 'replay_worker', replay_worker),
                (marked.pools, 'build_view_pools', full_pool),
                (marked.base, 'placement_groups', lambda *args, **kwargs:
                    geometry.placement_groups(rounds.original._ORIGINAL_GROUPS, *args, **kwargs)),
                (marked, 'measured_build', build), (marked, 'construct_observer', observer)]
            previous = [(module, key, getattr(module, key)) for module, key, _ in targets]
            try:
                for module, key, value in targets:
                    setattr(module, key, value)
                yield quota
            finally:
                for module, key, value in reversed(previous):
                    setattr(module, key, value)
    finally:
        rounds.policy = previous_policy


def generator_worker(config, connection):
    policy.validate_source(config)
    with installed(config):
        return rounds.original.look.stay._ORIGINAL_GENERATOR(config, connection)


def replay_worker(config, connection):
    policy.validate_source(config)
    with policy.camera_support(), labels.installed(), labels.pair_context(
            config['fixed_label_pair_id'], config['candidate_names'], declared=config):
        # Replay does not generate native coordinates or rank proposal pools.
        result = rounds.original.look._ORIGINAL_REPLAY(config, connection)
    result.update(policy.declared_fields(config), **sharing.fields(), **rank.fields(), **geometry.fields(),
                  **labels.pair_fields(config['fixed_label_pair_id'], config['candidate_names']),
                  producer_module=policy.PRODUCER)
    return result


def main():
    with installed():
        return rounds.original.marked.main()


if __name__ == '__main__':
    raise SystemExit(main())
