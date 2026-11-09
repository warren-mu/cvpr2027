"""Three ordered20-minute Look units sequentially on exactly one GPU."""
import sys
from src.tools import t01v2_marked_parallel as marked
from src.tools import t01v2_parallel as original
from src.envs import pair_v2_stay_order as stay
from src.envs import pair_v2_look_unit_policy as policy


def single(argv, flag):
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        raise ValueError('Require one explicit ' + flag)
    return argv[argv.index(flag) + 1]


def validate_plan(argv, kitchen):
    fields = policy.camera_fields(int(single(argv, '--field-of-view')), kitchen)
    gpus = int(single(argv, '--gpus'))
    if gpus != 1:
        raise ValueError('Supplement12 requires one GPU; packing is not authorized')
    single(argv, '--scenes')
    rest = argv[argv.index('--scenes') + 1:]
    scenes = rest[:next((i for i, value in enumerate(rest) if value.startswith('--')), len(rest))]
    if scenes != ['FloorPlan203', 'FloorPlan204', kitchen]:
        raise ValueError('Require exactly the ordered sequential units203,204,kitchen')
    for flag, value in (('--timeout-seconds', 3870), ('--scene-timeout-seconds', 1200), ('--gate-timeout-seconds', 120)):
        if float(single(argv, flag)) != value:
            raise ValueError('Look units require explicit bounded gate/search/cleanup timing: ' + flag)
    policy.declared_fields({**fields, 'pilot_scenes': scenes})
    return fields


def main():
    kitchen = single(sys.argv[1:], '--kitchen')
    index = sys.argv.index('--kitchen')
    argv = sys.argv[:index] + sys.argv[index + 2:]
    validate_plan(argv[1:], kitchen)
    previous = (marked.SCENE_MODULE, marked.marked_tags, marked.validate_plan, marked.declared_fields,
                original.shard_configuration, original.verify_shard_result, original.launch_logged, sys.argv)
    def tags():
        return {**previous[1](), **stay.fields(), **policy.unit_fields(kitchen),
                'scene_producer_module': policy.PRODUCER}
    def shard(*args, **kwargs):
        value = previous[4](*args, **kwargs)
        value.update(stay.fields(), producer_module=policy.PRODUCER)
        return value
    def verify(record, data, expected):
        previous[5](record, data, expected)
        policy.validate_source(data['config'], data['metrics'])
        if any(data['config']['shard'].get(key) != expected.get(key) for key in (*stay.FIELDS, *policy.FIELDS, 'producer_module')):
            raise ValueError('Look-only result differs from planned immutable search policy')
    def launch(name, command, environment, directory):
        # The existing scheduler computes min(requested, remaining minus30s).
        # Reject a truncated unit before launch; completed children remain valid.
        if policy.PRODUCER in command:
            granted = float(single(command, '--timeout-seconds'))
            if granted != policy.POLICY['scene_timeout_seconds']:
                raise RuntimeError('Unstarted Look unit: insufficient time for a full1200-second grant')
        return previous[6](name, command, environment, directory)
    with policy.camera_support():
        try:
            marked.SCENE_MODULE, marked.marked_tags = policy.PRODUCER, tags
            marked.validate_plan = lambda arguments: validate_plan(arguments, kitchen)
            marked.declared_fields = policy.declared_fields
            original.shard_configuration, original.verify_shard_result = shard, verify
            original.launch_logged = launch
            sys.argv = argv
            return marked.main()
        finally:
            (marked.SCENE_MODULE, marked.marked_tags, marked.validate_plan, marked.declared_fields,
             original.shard_configuration, original.verify_shard_result, original.launch_logged, sys.argv) = previous


if __name__ == '__main__':
    raise SystemExit(main())
