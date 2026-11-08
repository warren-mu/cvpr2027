"""Explicit per-attempt common-baseline entry; historical outers stay frozen."""
import copy

from src.tools import t01v2_parallel as original
from src.tools.t01v2_stable_parallel import stable_scene_command
from src.tools.t01v2_solvable_parallel import SOLVABILITY_VERSION
from src.envs.pair_v2_isolated_shard import ATTEMPT_SETUP_VERSION, ATTEMPT_SETUP_POLICY

SCENE_MODULE = 'src.envs.pair_v2_isolated_shard'


def isolated_tags():
    return {'solvability_version': SOLVABILITY_VERSION, 'scene_producer_module': SCENE_MODULE,
            'attempt_setup_version': ATTEMPT_SETUP_VERSION,
            'attempt_setup_policy': copy.deepcopy(ATTEMPT_SETUP_POLICY)}


def isolated_scene_command(name, command):
    replaced = stable_scene_command(name, command)
    if replaced is command:
        return command
    position = 2 if replaced[1:2] == ['-u'] else 1
    replaced[position + 1] = SCENE_MODULE
    return replaced


def main():
    previous_launch, previous_write = original.launch_logged, original.write_result

    def launch(name, command, environment, directory):
        return previous_launch(name, isolated_scene_command(name, command), environment, directory)

    def write(name, metrics, config):
        config.update(isolated_tags())
        metrics.update(isolated_tags())
        return previous_write(name, metrics, config)

    original.launch_logged, original.write_result = launch, write
    try:
        return original.main()
    finally:
        original.launch_logged, original.write_result = previous_launch, previous_write


if __name__ == '__main__':
    raise SystemExit(main())
