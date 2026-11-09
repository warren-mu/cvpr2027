"""Add-only supplement12 worker. Existing live marked producers are untouched."""
import copy
from contextlib import contextmanager
from . import pair_v2_marked_shard as marked
from . import pair_v2_stay_shard as stay
from . import pair_v2_stay_order as stay_order
from . import pair_v2_look_unit_policy as policy
from . import pair_v2_projection_pool as pools
from . import pair_v2_role_policy as roles
from . import pair_v2_policy as original_policy

_ORIGINAL_POOL_ITERATOR = pools.PoolProposalIterator
_ORIGINAL_BUILD = marked.measured_build
_ORIGINAL_SUMMARY = marked.summary
_ORIGINAL_REPLAY = marked.replay_worker


class LookOnlyIterator(_ORIGINAL_POOL_ITERATOR):
    def __init__(self, records):
        self.records = list(records)
        if any(record.get('version') != pools.POOL_VERSION for record in self.records):
            raise ValueError('Unknown projection pool')
        self.streams = {'look': stay_order.FarFirstPoolRoles(self.records, 'look'),
            'direct': roles.RoleProposalIterator([], [], [record['view'] for record in self.records])}
        self.next_pool, self.yielded = 'look', {'look': 0, 'direct': 0}

    def __next__(self):
        self.next_pool = 'look'
        return super().__next__()


def enable_config(config):
    fields = policy.declared_fields(config['shard'])
    if config['shard'].get('producer_module') != policy.PRODUCER:
        raise ValueError('Look unit requires its explicit immutable producer assignment')
    stay._ORIGINAL_CONFIG(config)
    config.update(fields, **stay_order.fields(), producer_module=policy.PRODUCER,
        minimum_eligible_attempts=None, max_attempts=1000000,
        proposal_order='Look projection pool only; original Flip/Stay and Stay A/B streams; supplement11 far-first Stay',
        attempt_limit_scope='large infrastructure guard only; this unit is bounded by its recorded walltime/deadline')
    return config


def tags():
    return {**stay.tags(), 'producer_module': policy.PRODUCER,
        'look_unit_version': policy.VERSION, 'look_unit_policy': copy.deepcopy(policy.POLICY)}


class LookOnlyPrefixRejected(RuntimeError):
    pass


def measured_build(*args, **kwargs):
    row = kwargs.get('attempt_row')
    if row is None:
        row = {}; kwargs['attempt_row'] = row
    original = original_policy.prefix_check
    def prefix(*values, **options):
        measured = original(*values, **options)
        row['prefix_measurement'] = copy.deepcopy(measured)
        if measured.get('render_qualified_for_shared_prefix') is not True:
            masks = measured.get('prefix_edit_mask_pixels')
            pixels = ([masks[0].get(key) for key in ('w0', 'w1')]
                      if isinstance(masks, list) and len(masks) == 1 and isinstance(masks[0], dict) else None)
            valid_pixels = pixels is not None and all(type(value) is int and value >= 0 for value in pixels)
            if measured.get('candidates_visible_and_marked') is not True:
                reason = 'v0_candidates_visibility'
            elif valid_pixels and any(pixels):
                reason = 'anchor_visible_at_v0'
            elif valid_pixels and not any(pixels) and type(measured.get('prefix_max_abs_diff')) is int and measured['prefix_max_abs_diff'] > 3:
                reason = 'anchor_hidden_but_prefix_rgb_diff_gt3'
            else:
                reason = 'look_prefix_not_qualified'
            row.update(status='rejected', kind=None, discard_reasons=[reason])
            raise LookOnlyPrefixRejected(reason)
        return measured
    original_policy.prefix_check = prefix
    try:
        try:
            result = _ORIGINAL_BUILD(*args, **kwargs)
        except LookOnlyPrefixRejected:
            result = row
        if isinstance(result.get('kind'), str) and result['kind'].startswith('direct_'):
            raise ValueError('Look-only producer reached an unauthorized Direct branch')
        return result
    finally:
        original_policy.prefix_check = original


def compact_attempt(row):
    result = stay.compact_attempt(row)
    result['mode'] = row.get('mode')
    prefix = row.get('prefix_measurement')
    result['prefix_measurement'] = None if not isinstance(prefix, dict) else {
        key: copy.deepcopy(prefix.get(key)) for key in ('prefix_edit_mask_pixels', 'prefix_max_abs_diff',
            'candidates_visible_and_marked', 'render_qualified_for_shared_prefix', 'reasons')}
    report = row.get('look_solvability')
    if isinstance(report, dict) and isinstance(result.get('look_solvability'), dict):
        for original, compact in zip(report.get('actions', []), result['look_solvability'].get('actions', [])):
            evidence = original.get('evidence')
            compact['evidence'] = None if not isinstance(evidence, dict) else {
                key: copy.deepcopy(evidence.get(key)) for key in ('evidence_pixels', 'marked_evidence_pixels',
                    'evidence_passed', 'marked_evidence_passed', 'resolution', 'relevant_mask_union_pixels',
                    'anchor_pixels_w0', 'anchor_pixels_w1')}
            compact['passed'] = original.get('passed')
    witnesses = row.get('witnesses')
    result['witnesses'] = None if not isinstance(witnesses, list) else [{
        key: copy.deepcopy(witness.get(key, (witness.get('replay') or {}).get(key)))
        for key in ('passed', 'reasons', 'executed_action_count', 'restore_deterministic', 'restore_max_abs_diff')}
        for witness in witnesses]
    for key in ('restore_deterministic', 'restore_max_abs_diff'):
        result[key] = copy.deepcopy(row.get(key))
    return result


def summary(state, config):
    result = _ORIGINAL_SUMMARY(state, config)
    result.update(minimum_eligible_attempts=None, minimum_attempt_count_met=None,
        minimum_count_scope='no minimum-attempt claim; bounded supplement12 Look-only unit')
    return result


def replay_worker(config, connection):
    # Replay is another spawned interpreter; its native FOV guard needs the
    # explicit same-scene policy too. No generator search hooks are installed.
    with policy.camera_support():
        return _ORIGINAL_REPLAY(config, connection)


@contextmanager
def installed():
    with stay.installed(), policy.camera_support():
        previous = (marked.enable_config, marked.tags, marked.compact_attempt, marked.measured_build,
            marked.summary, marked.generator_worker, marked.replay_worker, marked.declared_fields, pools.PoolProposalIterator)
        try:
            marked.enable_config, marked.tags, marked.compact_attempt = enable_config, tags, compact_attempt
            marked.measured_build, marked.summary = measured_build, summary
            marked.generator_worker, marked.replay_worker = generator_worker, replay_worker
            marked.declared_fields, pools.PoolProposalIterator = policy.declared_fields, LookOnlyIterator
            yield
        finally:
            (marked.enable_config, marked.tags, marked.compact_attempt, marked.measured_build,
             marked.summary, marked.generator_worker, marked.replay_worker, marked.declared_fields, pools.PoolProposalIterator) = previous


def generator_worker(config, connection):
    policy.validate_source(config)
    with installed():
        return stay._ORIGINAL_GENERATOR(config, connection)


def main():
    with installed():
        return marked.main()


if __name__ == '__main__':
    raise SystemExit(main())
