"""Explicit Stay-domain ordering source; retain all ranked scientific checks.

Ordering declarations are proposal diagnostics, not measured visibility or a
new acceptance condition. The existing source, per-Controller, native retry,
fixed-label and full scientific validators remain on the same execution path.
"""
from contextlib import contextmanager

from src.envs import pair_v2_stay_priority as priority
from src.envs import pair_v2_stay_priority_policy as policy
from . import t01v2_flip_ranked_review_checkpoint as ranked

merge = ranked.merge


@contextmanager
def source_support():
    previous_policy = ranked.policy
    try:
        ranked.policy = policy
        with ranked.source_support():
            previous_check = merge.check_pair

            def check(row, path, **kwargs):
                if (row.get('producer_module') == policy.PRODUCER
                        or any(key in row for key in priority.FIELDS)):
                    if (row.get('producer_module') != policy.PRODUCER
                            or row.get('flip_unit_version') != policy.VERSION
                            or row.get('flip_unit_policy') != policy.POLICY):
                        raise ValueError('Stay ordering diagnostics require their explicit new source')
                    priority.validate_fields(row)
                    priority.validate_selection(row)
                # Includes the original authenticated source requirement and
                # complete geometry for Flip; Stay retains its original path.
                return previous_check(row, path, **kwargs)

            merge.check_pair = check
            try:
                yield
            finally:
                merge.check_pair = previous_check
    finally:
        ranked.policy = previous_policy


def main(argv=None):
    original = ranked.original
    previous = (original.original.collect_sources,
                original.controller.policy, original.association.policy)
    with source_support():
        try:
            original.original.collect_sources = original.controller.collect_sources
            original.controller.policy = original.association.policy = policy
            return original.original.main(argv)
        finally:
            (original.original.collect_sources,
             original.controller.policy, original.association.policy) = previous


if __name__ == '__main__':
    raise SystemExit(main())
