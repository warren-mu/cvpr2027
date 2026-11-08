"""Private scene-module adapter for the versioned settled-baseline window.

The frozen orchestrator still owns gate, deadlines, UUID subsets, affinity,
cleanup and evidence. Neither first-round nor earlier clone files are changed.
"""
from pathlib import Path
import re

from src.tools import t01v2_parallel as original_parallel

OLD_MODULE = "src.envs.pair_v2_shard"
STABLE_MODULE = "src.envs.pair_v2_stable_shard"
PREFLIGHT_VERSION = "settled_baseline_window_v1"


def stable_scene_command(name, command):
    """Replace the exact scene executable position, leaving all other argv intact."""
    if (not isinstance(command, list) or len(command) < 3
            or not all(isinstance(arg, str) for arg in command)
            or not re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(command[0]).name)):
        return command
    start = 2 if command[1:2] == ["-u"] else 1
    if command[start:start + 2] != ["-m", OLD_MODULE]:
        return command
    arguments = command[start + 2:]
    for flag in ("--name", "--scene", "--shard-config", "--timeout-seconds"):
        if arguments.count(flag) != 1 or arguments.index(flag) + 1 >= len(arguments):
            raise ValueError("Frozen scene-child CLI shape differs: " + flag)
    if arguments[arguments.index("--name") + 1] != name:
        raise ValueError("Scene child's explicit name differs from run_logged name")
    replaced = list(command)
    replaced[start + 1] = STABLE_MODULE
    return replaced


def main():
    original = original_parallel.launch_logged

    def launch(name, command, environment, directory):
        return original(name, stable_scene_command(name, command), environment, directory)

    original_parallel.launch_logged = launch
    try:
        return original_parallel.main()
    finally:
        original_parallel.launch_logged = original


if __name__ == "__main__":
    raise SystemExit(main())
