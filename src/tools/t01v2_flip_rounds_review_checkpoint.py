"""Opt in the new scheduling source; preserve complete prior pair checks."""
from contextlib import contextmanager
from . import t01v2_flip_review_checkpoint as original
from src.envs import pair_v2_flip_rounds_policy as policy


@contextmanager
def source_support():
    previous = original.policy
    try:
        original.policy = policy
        with original.source_support():
            yield
    finally:
        original.policy = previous


def main(argv=None):
    previous = original.policy
    try:
        original.policy = policy
        return original.main(argv)
    finally:
        original.policy = previous


if __name__ == '__main__':
    raise SystemExit(main())
