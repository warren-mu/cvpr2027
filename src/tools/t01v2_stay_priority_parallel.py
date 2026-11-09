"""Single-GPU full1200-second Stay-order trial; all launch guards retained."""
from . import t01v2_flip_parallel as original
from src.envs import pair_v2_stay_priority_policy as policy


def main():
    previous = original.policy
    try:
        original.policy = policy
        return original.main()
    finally:
        original.policy = previous


if __name__ == '__main__':
    raise SystemExit(main())
