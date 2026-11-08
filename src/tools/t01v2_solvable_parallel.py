"""Supplement-4 outer dispatch; already loaded old outers remain unchanged."""
from src.tools import t01v2_parallel as original
from src.tools.t01v2_stable_parallel import stable_scene_command

SCENE_MODULE = 'src.envs.pair_v2_solvable_shard'
SOLVABILITY_VERSION = 'look_per_world_visible_action_v1'


def solvable_scene_command(name, command):
    replaced = stable_scene_command(name, command)
    if replaced is command:
        return command
    position = 2 if replaced[1:2] == ['-u'] else 1
    replaced[position + 1] = SCENE_MODULE
    return replaced


def main():
    previous_launch, previous_write = original.launch_logged, original.write_result

    def launch(name, command, environment, directory):
        return previous_launch(name, solvable_scene_command(name, command), environment, directory)

    def write(name, metrics, config):
        version = {'solvability_version': SOLVABILITY_VERSION, 'scene_producer_module': SCENE_MODULE}
        config.update(version)
        metrics.update(version)
        return previous_write(name, metrics, config)

    original.launch_logged, original.write_result = launch, write
    try:
        return original.main()
    finally:
        original.launch_logged, original.write_result = previous_launch, previous_write


if __name__ == '__main__':
    raise SystemExit(main())
