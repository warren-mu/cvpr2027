"""Opt-in single-scene G01 mixed worker; no changes to active/old producers."""
import copy
from contextlib import contextmanager
from pathlib import Path

from . import pair_v2_marked_shard as marked
from . import pair_v2_stay_shard as stay
from . import pair_v2_look_unit_shard as look
from . import pair_v2_flip_shard as flip
from . import pair_v2_fixed_labels as labels
from . import pair_v2_g01_policy as policy
from . import pair_v2_g01_search as search
from . import pair_v2_flip_rounds_search as cadence
from . import pair_v2_spawn_sharing as sharing
from . import pair_v2_bbox_rank as rank
from . import pair_v2_flip_geometry as geometry
from . import pair_v2_flip_ranked_shard as ranked

_ORIGINAL_BUILD = look._ORIGINAL_BUILD
_ORIGINAL_EXPORT = marked.export_pair
_ORIGINAL_DISCOVER = marked.discover_view


def enable_config(config):
    declared = policy.declared_fields(config['shard'])
    if config['shard'].get('producer_module') != policy.PRODUCER:
        raise ValueError('G01 requires its explicit immutable mixed producer')
    stay._ORIGINAL_CONFIG(config)
    config.update(declared, **flip.look.stay_order.fields(), producer_module=policy.PRODUCER,
        minimum_eligible_attempts=None, max_attempts=1000000, max_receptacles=1000000,
        scene_pairs_per_class=policy.class_cap(declared['g01_partition']),
        proposal_order='G01 mixed; complete LookFlip bbox geometry; historical Direct/Stay domains; shared native tokens; full-active-round cadence; fixed labels; actual per-class cap',
        attempt_limit_scope='infrastructure guard; collection caps and explicit scene deadline are authoritative')
    return config


def tags(config=None):
    partition = 'train' if config is None else config['g01_partition']
    return {**stay.tags(), **policy.fields(partition), **search.fields(), **labels.fields(),
            **cadence.fields(), **sharing.fields(), **rank.fields(), **geometry.fields(),
            'producer_module': policy.PRODUCER}


def measured_build(sim, config, baseline, identities, proposal, view, pair_id, directory, reachable, attempt_row=None):
    row = {} if attempt_row is None else attempt_row
    fields = labels.pair_fields(pair_id, identities['candidate_names'])
    row.update(fields, mode=proposal['mode'], proposed_pool=proposal['projection_pool']['pool'])
    if proposal['mode'] == 'flip' and proposal['projection_pool']['pool'] == 'look':
        geometry.complete_pair(proposal['_private_pool_record'], proposal['w0'], proposal['w1'])
    if proposal['mode'] == 'stay':
        record, pool = proposal['_private_pool_record'], proposal['projection_pool']['pool']
        proposal = {**proposal, '_private_pool_record': {**record,
            'points': {**record['points'], pool: record['stay_points_by_pool'][pool]}}}
    with labels.pair_context(pair_id, identities['candidate_names'], declared={**config, **fields}):
        try:
            # The original mixed branch is retained, including Direct acceptance;
            # the supplement12 Look-only prefix rejection hook is not installed.
            return _ORIGINAL_BUILD(sim, {**config, **fields}, baseline, identities, proposal, view,
                                   pair_id, directory, reachable, attempt_row=row)
        except search.CollectionQuotaFull as error:
            if row.get('validity', {}).get('acceptance_passed') is not True or row.get('kind') != str(error):
                raise ValueError('Quota skip lacks an unchanged positive scientific verdict') from error
            row['collection_quota_skip'] = {'kind': row['kind'], 'per_scene_class_cap': policy.class_cap(config['g01_partition']),
                'acceptance_passed_before_collection': True,
                'scope': 'collection capacity only; original scientific validity remains unchanged'}
            row.update(status='rejected', discard_reasons=['collection_quota_full'])
            return row


def compact_attempt(row):
    result = look.compact_attempt(row)
    for key in (*policy.FIELDS, *search.FIELDS, *labels.PAIR_FIELDS,
                *cadence.FIELDS, *sharing.FIELDS, *rank.FIELDS, *geometry.FIELDS,
                'flip_geometry_initial_tuple', 'flip_geometry_final_tuple', 'flip_geometry_native_guards',
                'collection_quota_skip', 'proposed_pool'):
        result[key] = copy.deepcopy(row.get(key))
    if row.get('collection_quota_skip') is not None:
        result['validity'] = copy.deepcopy(row.get('validity'))
    return result


def replay_worker(config, connection):
    # The legacy witness transport sets this internal flag to True for every
    # replay. Preserve marked.replay_worker's existing natural-only semantics,
    # after authenticating the actual snapshot and candidate policy.
    config = copy.deepcopy(config)
    if (config.get('candidate_policy') != marked.candidates.CANDIDATE_POLICY
            or config['snapshot'].get('clone_context') is not None):
        raise ValueError('G01 replay requires its declared natural candidate snapshot')
    config['enable_candidate_clone'] = False
    policy.validate_source(config)
    with policy.camera_support(), labels.installed(), labels.pair_context(
            config['fixed_label_pair_id'], config['candidate_names'], declared=config):
        result = look._ORIGINAL_REPLAY(config, connection)
    result.update(labels.pair_fields(config['fixed_label_pair_id'], config['candidate_names']),
                  **policy.declared_fields(config), producer_module=policy.PRODUCER)
    return result


@contextmanager
def installed(config=None):
    quota = search.SceneQuota(policy.class_cap(config['g01_partition']) if config is not None else 8)
    check = ranked.deadline_check(config) if config is not None else None
    with stay.installed(), policy.camera_support(), labels.installed(), sharing.installed():
        original_write, original_summary = marked._BASE_WRITE, marked.summary
        original_observer = marked.construct_observer
        def full_pool(*args, **kwargs):
            if check is None:
                raise RuntimeError('G01 geometry is only bound inside the spawned generator')
            return search.build_pool(flip._ORIGINAL_POOLS, *args, check_deadline=check, **kwargs)
        def observer(sim, proposal, role, record, row, **kwargs):
            if proposal['mode'] == 'flip' and proposal['projection_pool']['pool'] == 'look':
                return ranked.guarded_observer(original_observer, sim, proposal, role, record, row, **kwargs)
            return original_observer(sim, proposal, role, record, row, **kwargs)
        def write(path, row):
            reference = original_write(path, row)
            if Path(path).name == 'attempt.json':
                quota.record(path, row)
            return reference
        def export(config, row, *args, **kwargs):
            quota.before_export(row)
            return _ORIGINAL_EXPORT(config, row, *args, **kwargs)
        def discover(*args, **kwargs):
            if quota.complete():
                return
            for item in _ORIGINAL_DISCOVER(*args, **kwargs):
                if quota.complete():
                    return
                yield item
                if quota.complete():
                    return
        def summary(state, config):
            result = original_summary(state, config)
            result.update(g01_collection=quota.summary(), minimum_eligible_attempts=None,
                minimum_attempt_count_met=None, minimum_count_scope='no minimum-attempt claim; bounded collection')
            if quota.complete():
                result['search_stop_reason'] = 'g01_all_scene_class_caps_reached'
            return result
        targets = [
            (marked, 'enable_config', enable_config), (marked, 'tags', lambda: tags(config)),
            (marked, 'measured_build', measured_build), (marked, 'compact_attempt', compact_attempt),
            (marked, 'existing_views', flip.existing_views),
            (marked.candidates, 'identity_sets_at_view', flip.identity_sets_at_view),
            (marked.base, 'placement_groups', lambda *args, **kw: geometry.placement_groups(flip._ORIGINAL_GROUPS, *args, **kw)),
            (marked.pools, 'build_view_pools', full_pool),
            (marked.pools, 'pool_evidence_record', lambda record: search.compact_pool(flip._ORIGINAL_POOL_RECORD, record)),
            (marked.pools, 'PoolProposalIterator', lambda records: search.MixedIterator(records, quota)),
            (marked, 'declared_fields', policy.declared_fields),
            (marked, 'generator_worker', generator_worker), (marked, 'replay_worker', replay_worker),
            (marked, '_BASE_WRITE', write), (marked, 'summary', summary),
            (marked, 'export_pair', export), (marked, 'discover_view', discover),
            (marked, 'construct_observer', observer), (marked, 'run_search', cadence.run_search)]
        previous = [(module, key, getattr(module, key)) for module, key, _ in targets]
        try:
            for module, key, value in targets:
                setattr(module, key, value)
            yield quota
        finally:
            for module, key, value in reversed(previous):
                setattr(module, key, value)


def generator_worker(config, connection):
    from src.tools.g01_admission import witness_bridge
    policy.validate_source(config)
    with installed(config), witness_bridge(config) as bridge:
        result = stay._ORIGINAL_GENERATOR(config, connection)
    result['g01_witness_admission'] = bridge.report()
    return result


def main():
    with installed():
        return marked.main()


if __name__ == '__main__':
    raise SystemExit(main())
