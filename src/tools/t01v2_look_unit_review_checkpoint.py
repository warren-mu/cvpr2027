"""Explicit supplement12 source adapter for unchanged independent pair checks."""
import copy
from contextlib import contextmanager
from src.envs import pair_v2_look_unit_policy as policy
from src.tools import merge_pair_v2 as merge
from src.tools import t01v2_review_checkpoint as checkpoint
from src.tools.t01v2_stay_review_checkpoint import source_support as stay_source_support


def is_look_source(config):
    return (config.get('producer_module') == policy.PRODUCER
            or any(key in config for key in policy.FIELDS))


@contextmanager
def source_support():
    with stay_source_support(), policy.camera_support():
        previous_producer, previous_load = merge.shard_producer, merge.load_source
        def producer(command, config):
            if not is_look_source(config):
                return previous_producer(command, config)
            policy.validate_source(config)
            common = copy.deepcopy(config)
            common['producer_module'] = 'src.envs.pair_v2_marked_shard'
            merge.marked_policy_configuration(common)
            arguments = merge.producer_arguments(command, policy.PRODUCER)
            if (arguments.count('--timeout-seconds') != 1
                    or arguments.index('--timeout-seconds') + 1 >= len(arguments)
                    or float(arguments[arguments.index('--timeout-seconds') + 1]) != policy.POLICY['scene_timeout_seconds']):
                raise ValueError('Look-only unit lacks its full1200-second recorded grant')
            return arguments, False
        def load(path):
            # The nested Stay loader sees Stay policy fields too, so call its
            # audited original loader captured before either opt-in wrapper.
            data, reference = _base_load(path)
            if is_look_source(data['config']):
                policy.validate_source(data['config'], data['metrics'])
            elif data['config'].get('producer_module') == 'src.envs.pair_v2_stay_shard':
                from src.envs.pair_v2_stay_order import validate_source
                validate_source(data['config'], data['metrics'])
            return data, reference
        try:
            merge.shard_producer, merge.load_source = producer, load
            yield
        finally:
            merge.shard_producer, merge.load_source = previous_producer, previous_load


_base_load = merge.load_source


def main(argv=None):
    with source_support():
        return checkpoint.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
