"""Add-only Flip iteration; hooks install inside each spawned interpreter."""
import copy
from contextlib import contextmanager
from pathlib import Path
from . import pair_v2_marked_shard as marked
from . import pair_v2_look_unit_shard as look
from . import pair_v2_flip_search as search
from . import pair_v2_flip_policy as policy
from . import pair_v2_fixed_labels as labels

_ORIGINAL_VIEWS = marked.existing_views
_ORIGINAL_IDENTITIES = marked.candidates.identity_sets_at_view
_ORIGINAL_GROUPS = marked.base.placement_groups
_ORIGINAL_POOLS = marked.pools.build_view_pools
_ORIGINAL_POOL_RECORD = marked.pools.pool_evidence_record


def enable_config(config):
    declared = policy.declared_fields(config['shard'])
    if config['shard'].get('producer_module') != policy.PRODUCER:
        raise ValueError('Flip worker requires its explicit immutable producer')
    look.stay._ORIGINAL_CONFIG(config)
    config.update(declared, **look.stay_order.fields(), producer_module=policy.PRODUCER,
        minimum_eligible_attempts=None, max_attempts=1000000,
        max_receptacles=1000000,
        proposal_order='supplement13 full native3D Look Flip domain, opposite-edge views, measured scene-wide Flip quota',
        attempt_limit_scope='infrastructure guard only; full1200-second unit')
    return config


def tags():
    return {**look.stay.tags(), 'producer_module': policy.PRODUCER,
        'flip_search_version': search.VERSION, 'flip_search_policy': copy.deepcopy(search.POLICY),
        **labels.fields()}


def existing_views(baseline, reachable, config):
    views = _ORIGINAL_VIEWS(baseline, reachable, config)
    camera = search.camera_from_snapshot(baseline)
    pairs = marked.candidates.all_candidate_pairs(baseline['objects'])
    pairs = list(pairs)
    return sorted(views, key=lambda view: min((search.edge_rank(a, b, marked.pools.camera_for_view(camera, view))
        for a, b in pairs), default=(1, 1, float('inf'))))


def identity_sets_at_view(capture):
    groups = _ORIGINAL_IDENTITIES(capture)
    objects = marked.engine.objects_by_name(capture['metadata']['objects'])
    camera = marked.pools.camera_from_metadata(capture['metadata'], capture['evidence']['files']['metadata'])
    return sorted(groups, key=lambda identity: search.edge_rank(
        *(objects[name] for name in identity['candidate_names']), camera))


def measured_build(sim, config, baseline, identities, proposal, view, pair_id, directory, reachable, attempt_row=None):
    row = {} if attempt_row is None else attempt_row
    fields = labels.pair_fields(pair_id, identities['candidate_names'])
    row.update(fields, mode=proposal['mode'])
    pair_config = {**config, **fields}
    # Stay uses exactly its historical horizontal12-point domain; the complete
    # new3D domain belongs only to Flip. Both refer to the same saved catalog.
    if proposal['mode'] == 'stay':
        record = proposal['_private_pool_record']
        proposal = {**proposal, '_private_pool_record': {
            **record, 'points': {**record['points'], 'look': record['stay_points']}}}
    with labels.pair_context(pair_id, identities['candidate_names'], declared=pair_config):
        return look.measured_build(sim, pair_config, baseline, identities, proposal, view,
            pair_id, directory, reachable, attempt_row=row)


def compact_attempt(row):
    result = look.compact_attempt(row)
    for key in (*search.FIELDS, *labels.PAIR_FIELDS):
        result[key] = copy.deepcopy(row.get(key))
    return result


def replay_worker(config, connection):
    labels.validate_fields(config)
    with policy.camera_support(), labels.installed(), labels.pair_context(
            config['fixed_label_pair_id'], config['candidate_names'], declared=config):
        result = look._ORIGINAL_REPLAY(config, connection)
    result.update(labels.pair_fields(config['fixed_label_pair_id'], config['candidate_names']))
    return result


@contextmanager
def installed():
    quota = search.AttemptQuota()
    with look.installed(), policy.camera_support(), labels.installed():
        targets = [
            (marked, 'enable_config', enable_config), (marked, 'tags', tags),
            (marked, 'measured_build', measured_build), (marked, 'compact_attempt', compact_attempt),
            (marked, 'existing_views', existing_views), (marked.candidates, 'identity_sets_at_view', identity_sets_at_view),
            (marked.base, 'placement_groups', lambda *args, **kwargs: search.placement_groups(_ORIGINAL_GROUPS, *args, **kwargs)),
            (marked.pools, 'build_view_pools', lambda *args, **kwargs: search.full_pool(_ORIGINAL_POOLS, *args, **kwargs)),
            (marked.pools, 'pool_evidence_record', lambda record: search.compact_pool(_ORIGINAL_POOL_RECORD, record)),
            (marked.pools, 'PoolProposalIterator', lambda records: search.FlipIterator(records, quota)),
            (marked, 'declared_fields', policy.declared_fields),
            (marked, 'generator_worker', generator_worker), (marked, 'replay_worker', replay_worker)]
        original_write = marked._BASE_WRITE
        original_summary = marked.summary
        def write(path, row):
            reference = original_write(path, row)
            if Path(path).name == 'attempt.json':
                quota.record(path, row)
            return reference
        def summary(state, config):
            result = original_summary(state, config)
            result.update(measured_flip_attempt_quota=quota.summary(), minimum_eligible_attempts=None,
                          minimum_attempt_count_met=None, minimum_count_scope='bounded supplement14 unit; no minimum claim')
            return result
        targets.extend([(marked, '_BASE_WRITE', write), (marked, 'summary', summary)])
        previous = [(module, key, getattr(module, key)) for module, key, _ in targets]
        try:
            for module, key, value in targets:
                setattr(module, key, value)
            yield quota
        finally:
            for module, key, value in reversed(previous):
                setattr(module, key, value)


def generator_worker(config, connection):
    policy.validate_source(config)
    with installed():
        return look.stay._ORIGINAL_GENERATOR(config, connection)


def main():
    with installed():
        return marked.main()


if __name__ == '__main__':
    raise SystemExit(main())
