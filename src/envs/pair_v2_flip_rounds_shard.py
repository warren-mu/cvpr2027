"""Versioned search cadence; existing Flip physics/image/witness code unchanged."""
from contextlib import contextmanager
from . import pair_v2_flip_shard as original
from . import pair_v2_flip_rounds_policy as policy
from . import pair_v2_flip_rounds_search as cadence
from . import pair_v2_fixed_labels as labels


def tags():
    return {**original.tags(), **cadence.fields(), 'producer_module': policy.PRODUCER}


def replay_worker(config, connection):
    labels.validate_fields(config)
    cadence.validate_fields(config)
    with policy.camera_support(), labels.installed(), labels.pair_context(
            config['fixed_label_pair_id'], config['candidate_names'], declared=config):
        result = original.look._ORIGINAL_REPLAY(config, connection)
    result.update(labels.pair_fields(config['fixed_label_pair_id'], config['candidate_names']))
    result.update(cadence.fields())
    return result


@contextmanager
def installed():
    previous_policy = original.policy
    original.policy = policy
    try:
        with original.installed() as quota:
            marked = original.marked
            targets = [(marked, 'run_search', cadence.run_search),
                       (marked, 'tags', tags),
                       (marked, 'generator_worker', generator_worker),
                       (marked, 'replay_worker', replay_worker)]
            previous = [(module, key, getattr(module, key)) for module, key, _ in targets]
            try:
                for module, key, value in targets:
                    setattr(module, key, value)
                yield quota
            finally:
                for module, key, value in reversed(previous):
                    setattr(module, key, value)
    finally:
        original.policy = previous_policy


def generator_worker(config, connection):
    with installed():
        policy.validate_source(config)
        return original.look.stay._ORIGINAL_GENERATOR(config, connection)


def main():
    with installed():
        return original.marked.main()


if __name__ == '__main__':
    raise SystemExit(main())
