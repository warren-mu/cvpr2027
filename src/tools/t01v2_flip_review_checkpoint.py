"""Strict new-source adapter retaining complete independent scientific checks.

Only producer identity, declared camera policy, new Flip proposal geometry and
fixed label drawing are opted in. Original authenticated source/journal,
per-Controller baseline association and full pair validators remain mandatory.
"""
import copy
from contextlib import contextmanager
from src.envs import pair_v2_flip_policy as policy
from src.envs import pair_v2_flip_search as search
from src.envs import pair_v2_fixed_labels as labels
from src.envs import pair_v2_role_policy as roles
from src.tools import merge_pair_v2 as merge
from src.tools import t01v2_controller_review_checkpoint as controller
from src.tools import t01v2_controller_association as association
from src.tools import t01v2_review_checkpoint as original

_BASE_LOAD = merge.load_source


def is_flip_source(config):
    return config.get('producer_module') == policy.PRODUCER or any(key in config for key in policy.FIELDS)


@contextmanager
def source_support():
    with policy.camera_support():
        previous_producer, previous_load = merge.shard_producer, merge.load_source
        previous_check, previous_role = merge.check_pair, roles.verify_proposal_role
        loaded = {}
        def producer(command, config):
            if not is_flip_source(config):
                return previous_producer(command, config)
            policy.validate_source(config)
            common = copy.deepcopy(config)
            common['producer_module'] = 'src.envs.pair_v2_marked_shard'
            merge.marked_policy_configuration(common)
            arguments = merge.producer_arguments(command, policy.PRODUCER)
            if (arguments.count('--timeout-seconds') != 1
                    or arguments.index('--timeout-seconds') + 1 >= len(arguments)
                    or float(arguments[arguments.index('--timeout-seconds') + 1]) != 1200):
                raise ValueError('Flip source lacks its full1200-second logged grant')
            return arguments, False
        def load(path):
            data, reference = _BASE_LOAD(path)
            if is_flip_source(data['config']):
                policy.validate_source(data['config'], data['metrics'])
                raw = data['config']['raw_evidence_dir']
                if raw in loaded and loaded[raw] != data['config']:
                    raise ValueError('Different authenticated configurations claim one raw source directory')
                loaded[raw] = data['config']
            return data, reference
        def role(row, positions):
            if row.get('flip_search_version') == search.VERSION and row.get('mode') == 'flip':
                return search.verify_flip_role(row, positions)
            return previous_role(row, positions)
        def check(row, path, **kwargs):
            config = loaded.get(kwargs.get('raw_root'))
            if config is None:
                if row.get('flip_search_version') is not None or row.get('fixed_label_version') is not None:
                    raise ValueError('New Flip pair has no independently authenticated source configuration')
                return previous_check(row, path, **kwargs)
            if row.get('flip_search_version') != search.VERSION or row.get('flip_search_policy') != search.POLICY:
                raise ValueError('Pair is missing its authenticated new-source search policy')
            return labels.checked_pair(previous_check, row, path, declared_source=config, **kwargs)
        try:
            merge.shard_producer, merge.load_source, merge.check_pair = producer, load, check
            roles.verify_proposal_role = role
            yield
        finally:
            merge.shard_producer, merge.load_source, merge.check_pair = previous_producer, previous_load, previous_check
            roles.verify_proposal_role = previous_role


def main(argv=None):
    # Reuse the complete original per-Controller collector without changing
    # source bytes or the old policy module. These bindings are process-local.
    previous = original.collect_sources, controller.policy, association.policy
    with source_support():
        try:
            original.collect_sources = controller.collect_sources
            controller.policy = association.policy = policy
            return original.main(argv)
        finally:
            original.collect_sources, controller.policy, association.policy = previous


if __name__ == '__main__':
    raise SystemExit(main())
