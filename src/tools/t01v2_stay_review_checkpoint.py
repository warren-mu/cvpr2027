"""Supplement 11 source opt-in for the unchanged independent checkpoint checks.

Historical source/producer validation stays unchanged. No source config,
metadata, command or pair bytes are rewritten. T02 must explicitly support this
checkpoint module before consuming it; this command only publishes review data.
"""
import copy
from contextlib import contextmanager
from src.envs import pair_v2_stay_order as order
from src.tools import merge_pair_v2 as merge
from src.tools import t01v2_review_checkpoint as checkpoint


def is_stay_source(config):
    return (config.get('producer_module') == order.PRODUCER
            or any(key in config for key in order.FIELDS))


@contextmanager
def source_support():
    original_producer, original_load = merge.shard_producer, merge.load_source
    def producer(command, config):
        if not is_stay_source(config):
            return original_producer(command, config)
        order.validate_source(config)
        common = copy.deepcopy(config)
        common['producer_module'] = 'src.envs.pair_v2_marked_shard'
        merge.marked_policy_configuration(common)
        return merge.producer_arguments(command, order.PRODUCER), False
    def load(path):
        data, reference = original_load(path)
        if is_stay_source(data['config']):
            order.validate_source(data['config'], data['metrics'])
        return data, reference
    try:
        merge.shard_producer, merge.load_source = producer, load
        yield
    finally:
        merge.shard_producer, merge.load_source = original_producer, original_load


def main(argv=None):
    with source_support():
        return checkpoint.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
