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


def tags():
    return {**rounds.tags(), **sharing.fields(), **rank.fields(), 'producer_module': policy.PRODUCER}


def compact_attempt(row):
    result = rounds.original.compact_attempt(row)
    for key in (*cadence.FIELDS, *sharing.FIELDS, *rank.FIELDS):
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
            def full_pool(left, right, a, b, anchor, **kwargs):
                if check is None:
                    raise RuntimeError('Proposal ranking is only bound inside the spawned generator')
                record = original_full_pool(left, right, a, b, anchor, **kwargs)
                value = rank.rank_pool(record, anchor, check_deadline=check)
                value.update(sharing.fields())
                return value
            targets = [(marked, 'tags', tags), (marked, 'compact_attempt', compact_attempt),
                (marked, 'generator_worker', generator_worker), (marked, 'replay_worker', replay_worker),
                (marked.pools, 'build_view_pools', full_pool)]
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
    result.update(policy.declared_fields(config), **sharing.fields(), **rank.fields(),
                  **labels.pair_fields(config['fixed_label_pair_id'], config['candidate_names']),
                  producer_module=policy.PRODUCER)
    return result


def main():
    with installed():
        return rounds.original.marked.main()


if __name__ == '__main__':
    raise SystemExit(main())
