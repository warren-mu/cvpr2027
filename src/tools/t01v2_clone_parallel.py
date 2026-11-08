"""Versioned second-round launcher; first-round source and parameters stay frozen.

Only this private Python process temporarily redirects the exact scene-child
module. The all-GPU render gate, assignment, argv values, environment, deadlines,
cleanup and source recording continue through the original orchestrator.
"""
from pathlib import Path
import re

from src.tools import t01v2_parallel as original_parallel

OLD_MODULE = "src.envs.pair_v2_shard"
CLONE_MODULE = "src.envs.pair_v2_clone_shard"


def clone_scene_command(name, command):
    """Replace only a real `python [-u] -m OLD_MODULE` scene invocation."""
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
    replaced[start + 1] = CLONE_MODULE
    return replaced


def main():
    original = original_parallel.launch_logged

    def launch(name, command, environment, directory):
        return original(name, clone_scene_command(name, command), environment, directory)

    original_parallel.launch_logged = launch
    try:
        return original_parallel.main()
    finally:
        original_parallel.launch_logged = original


if __name__ == "__main__":
    raise SystemExit(main())
