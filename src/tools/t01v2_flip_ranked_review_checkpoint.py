"""Require the combined version; reuse all prior independent source/pair checks."""
from contextlib import contextmanager
from . import t01v2_flip_review_checkpoint as original
from src.envs import pair_v2_flip_ranked_policy as policy
from src.envs import pair_v2_spawn_sharing as sharing
from src.envs import pair_v2_bbox_rank as rank


@contextmanager
def source_support():
    previous_policy = original.policy
    try:
        original.policy = policy
        with original.source_support():
            previous_check = original.merge.check_pair
            def check(row, path, **kwargs):
                # The existing checker still authenticates the source, native
                # queries, Controller association, and every scientific gate.
                result = previous_check(row, path, **kwargs)
                if row.get('flip_search_version') is not None or row.get('producer_module') == policy.PRODUCER:
                    if row.get('producer_module') != policy.PRODUCER:
                        raise ValueError('A historical Flip pair cannot claim combined-source provenance')
                    sharing.validate_fields(row); rank.validate_fields(row)
                return result
            original.merge.check_pair = check
            try:
                yield
            finally:
                original.merge.check_pair = previous_check
    finally:
        original.policy = previous_policy


def main(argv=None):
    # Keep the original complete per-Controller collector, with the new source
    # binding scoped to this invocation and rolled back even on failure.
    prior = original.original.collect_sources, original.controller.policy, original.association.policy
    with source_support():
        try:
            original.original.collect_sources = original.controller.collect_sources
            original.controller.policy = original.association.policy = policy
            return original.original.main(argv)
        finally:
            original.original.collect_sources, original.controller.policy, original.association.policy = prior


if __name__ == '__main__':
    raise SystemExit(main())
