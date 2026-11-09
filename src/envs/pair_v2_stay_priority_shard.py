"""Isolated process-local Stay scheduling; existing deployed producers untouched."""
import copy
from contextlib import contextmanager
from . import pair_v2_flip_ranked_shard as ranked
from . import pair_v2_stay_priority_policy as policy
from . import pair_v2_stay_priority as priority


@contextmanager
def installed(config=None):
    previous_policy = ranked.policy
    ranked.policy = policy
    try:
        with ranked.installed(config) as quota:
            marked = ranked.rounds.original.marked
            order = priority.DomainOrder()
            check = ranked.deadline_check(config) if config is not None else None
            original_build, original_tags, original_compact = marked.measured_build, marked.tags, marked.compact_attempt
            def pending_round(active):
                if check is None:
                    raise RuntimeError('Stay priority requires the original spawned scene deadline')
                return priority.PriorityRound(active, quota, order, check_deadline=check)
            def tags():
                return {**original_tags(), **priority.fields(), 'producer_module': policy.PRODUCER}
            def compact(row):
                result = original_compact(row)
                for key in (*priority.FIELDS, priority.SELECTION):
                    result[key] = copy.deepcopy(row.get(key))
                return result
            def build(sim, cfg, baseline, identities, proposal, view, pair_id, directory, reachable, attempt_row=None):
                row = {} if attempt_row is None else attempt_row
                row[priority.SELECTION] = copy.deepcopy(proposal.get(priority.SELECTION))
                return original_build(sim, cfg, baseline, identities, proposal, view, pair_id,
                                      directory, reachable, attempt_row=row)
            targets = [(ranked.cadence, 'DiscoveryRound', pending_round),
                (marked.pools, 'PoolProposalIterator', lambda records: priority.PriorityIterator(records, quota)),
                (marked, 'tags', tags), (marked, 'compact_attempt', compact),
                (marked, 'measured_build', build), (marked, 'generator_worker', generator_worker),
                (marked, 'replay_worker', replay_worker)]
            previous = [(module, key, getattr(module, key)) for module, key, _ in targets]
            try:
                for module, key, value in targets:
                    setattr(module, key, value)
                yield quota
            finally:
                for module, key, value in reversed(previous):
                    setattr(module, key, value)
    finally:
        ranked.policy = previous_policy


def generator_worker(config, connection):
    policy.validate_source(config)
    with installed(config):
        return ranked.rounds.original.look.stay._ORIGINAL_GENERATOR(config, connection)


def replay_worker(config, connection):
    previous_policy = ranked.policy
    try:
        ranked.policy = policy
        return ranked.replay_worker(config, connection)
    finally:
        ranked.policy = previous_policy


def main():
    with installed():
        return ranked.rounds.original.marked.main()


if __name__ == '__main__':
    raise SystemExit(main())
