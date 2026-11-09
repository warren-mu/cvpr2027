"""Supplement 6/7 entry using the existing gated one-GPU scene scheduler."""
import copy
import sys

from src.tools import t01v2_parallel as original
from src.tools.t01v2_stable_parallel import stable_scene_command
from src.envs.pair_v2_marked_policy import CANDIDATE_POLICY, CANDIDATE_POLICY_SPEC
from src.envs.pair_v2_projection_pool import POOL_VERSION, POOL_POLICY
from src.envs.pair_v2_isolated_shard import ATTEMPT_SETUP_VERSION, ATTEMPT_SETUP_POLICY
from src.envs.pair_v2_native_retry import FROZEN_RENDER_VERSION, FROZEN_RENDER_POLICY

SCENE_MODULE = 'src.envs.pair_v2_marked_shard'


def marked_tags():
    return {'scene_producer_module': SCENE_MODULE,
            'candidate_policy': CANDIDATE_POLICY, 'candidate_policy_spec': copy.deepcopy(CANDIDATE_POLICY_SPEC),
            'projection_pool_version': POOL_VERSION, 'projection_pool_policy': copy.deepcopy(POOL_POLICY),
            'attempt_setup_version': ATTEMPT_SETUP_VERSION, 'attempt_setup_policy': copy.deepcopy(ATTEMPT_SETUP_POLICY),
            'frozen_render_version': FROZEN_RENDER_VERSION, 'frozen_render_policy': copy.deepcopy(FROZEN_RENDER_POLICY),
            'enable_candidate_clone': False, 'budget_limit_charged_gpu_hours': 16}


def marked_scene_command(name, command):
    replaced = stable_scene_command(name, command)
    if replaced is command:
        return command
    position = 2 if replaced[1:2] == ['-u'] else 1
    replaced[position + 1] = SCENE_MODULE
    return replaced


def validate_plan(argv):
    if argv.count('--gpus') != 1 or argv[argv.index('--gpus') + 1] != '1':
        raise ValueError('Approved marked collection uses one GPU per job')
    if argv.count('--scenes') != 1:
        raise ValueError('Require explicit fixed training scenes')
    tail = argv[argv.index('--scenes') + 1:]
    scenes = tail[:next((i for i, value in enumerate(tail) if value.startswith('--')), len(tail))]
    allowed = {f'FloorPlan{i}' for i in (*range(1, 21), *range(201, 221))}
    if not scenes or len(scenes) != len(set(scenes)) or not set(scenes) <= allowed:
        raise ValueError('Supplement 7 is restricted to the fixed forty training scenes')


def main():
    validate_plan(sys.argv[1:])
    prior_launch, prior_write = original.launch_logged, original.write_result
    def launch(name, command, environment, directory):
        return prior_launch(name, marked_scene_command(name, command), environment, directory)
    def write(name, metrics, config):
        config.update(marked_tags()); metrics.update(marked_tags())
        return prior_write(name, metrics, config)
    original.launch_logged, original.write_result = launch, write
    try:
        return original.main()
    finally:
        original.launch_logged, original.write_result = prior_launch, prior_write


if __name__ == '__main__':
    raise SystemExit(main())
