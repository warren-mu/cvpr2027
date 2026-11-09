"""One explicit20-minute, one-GPU Flip unit at the selected60-degree camera."""
import sys
from src.tools import t01v2_marked_parallel as marked
from src.tools import t01v2_parallel as original
from src.tools.t01v2_look_unit_parallel import single
from src.envs import pair_v2_stay_order as stay
from src.envs import pair_v2_flip_policy as policy


def validate_plan(argv):
    if int(single(argv, '--gpus')) != 1:
        raise ValueError('Flip iterations require exactly one GPU')
    scene = single(argv, '--scenes')
    rest = argv[argv.index('--scenes') + 1:]
    scenes = rest[:next((i for i, value in enumerate(rest) if value.startswith('--')), len(rest))]
    if scenes != [scene]:
        raise ValueError('Require one explicit training scene per Flip iteration')
    fields = policy.camera_fields(int(single(argv, '--field-of-view')), scene)
    for flag, value in (('--timeout-seconds', 1350), ('--scene-timeout-seconds', 1200), ('--gate-timeout-seconds', 120)):
        if float(single(argv, flag)) != value:
            raise ValueError('Require explicit full-unit/gate/finalization timing: ' + flag)
    return fields


def main():
    fields = validate_plan(sys.argv[1:])
    previous = (marked.SCENE_MODULE, marked.marked_tags, marked.validate_plan, marked.declared_fields,
                original.shard_configuration, original.verify_shard_result, original.launch_logged)
    def tags():
        return {**previous[1](), **fields, **stay.fields(), 'scene_producer_module': policy.PRODUCER}
    def shard(*args, **kwargs):
        value = previous[4](*args, **kwargs)
        value.update(stay.fields(), producer_module=policy.PRODUCER)
        return value
    def verify(record, data, expected):
        previous[5](record, data, expected)
        policy.validate_source(data['config'], data['metrics'])
        if any(data['config']['shard'].get(key) != expected.get(key)
               for key in (*stay.FIELDS, *policy.FIELDS, *fields, 'producer_module')):
            raise ValueError('Flip result differs from its immutable planned unit')
    def launch(name, command, environment, directory):
        if policy.PRODUCER in command and float(single(command, '--timeout-seconds')) != 1200:
            raise RuntimeError('Unstarted Flip unit: no full1200-second grant available')
        return previous[6](name, command, environment, directory)
    with policy.camera_support():
        try:
            marked.SCENE_MODULE, marked.marked_tags = policy.PRODUCER, tags
            marked.validate_plan, marked.declared_fields = validate_plan, policy.declared_fields
            original.shard_configuration, original.verify_shard_result, original.launch_logged = shard, verify, launch
            return marked.main()
        finally:
            (marked.SCENE_MODULE, marked.marked_tags, marked.validate_plan, marked.declared_fields,
             original.shard_configuration, original.verify_shard_result, original.launch_logged) = previous


if __name__ == '__main__':
    raise SystemExit(main())
