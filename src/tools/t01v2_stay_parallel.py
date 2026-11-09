"""One-GPU supplement 11 entry; old producers on running allocations stay old."""
import copy
from src.tools import t01v2_marked_parallel as marked
from src.tools import t01v2_parallel as original
from src.envs import pair_v2_stay_order as order


def main():
    previous = marked.SCENE_MODULE, marked.marked_tags, original.shard_configuration, original.verify_shard_result
    def tags():
        return {**previous[1](), **order.fields(), 'scene_producer_module': order.PRODUCER}
    def shard(*args, **kwargs):
        value = previous[2](*args, **kwargs)
        value.update(order.fields(), producer_module=order.PRODUCER)
        return value
    def verify(record, data, expected):
        previous[3](record, data, expected)
        order.validate_source(data['config'], data['metrics'])
        if any(data['config']['shard'].get(key) != expected.get(key) for key in (*order.FIELDS, 'producer_module')):
            raise ValueError('Stay result differs from planned immutable search policy')
    try:
        marked.SCENE_MODULE, marked.marked_tags = order.PRODUCER, tags
        original.shard_configuration, original.verify_shard_result = shard, verify
        return marked.main()
    finally:
        marked.SCENE_MODULE, marked.marked_tags, original.shard_configuration, original.verify_shard_result = previous


if __name__ == '__main__':
    raise SystemExit(main())
