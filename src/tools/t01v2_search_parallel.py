"""Versioned supplement-3 scene entry; allocation/gate/deadlines stay frozen."""
from src.tools import t01v2_parallel as original
from src.tools.t01v2_stable_parallel import stable_scene_command

SEARCH_MODULE = 'src.envs.pair_v2_search_shard'
SEARCH_VERSION = 'natural_union_clone_search_v1'


def search_scene_command(name, command):
    replaced = stable_scene_command(name, command)
    if replaced is command:
        return command
    position = 2 if replaced[1:2] == ['-u'] else 1
    replaced[position + 1] = SEARCH_MODULE
    return replaced


def main():
    previous = original.launch_logged
    def launch(name, command, environment, directory):
        return previous(name, search_scene_command(name, command), environment, directory)
    original.launch_logged = launch
    try:
        return original.main()
    finally:
        original.launch_logged = previous


if __name__ == '__main__':
    raise SystemExit(main())
