"""Opt-in single-scene G01 mixed worker; no changes to active/old producers."""
import copy
from contextlib import contextmanager, nullcontext
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
from . import pair_v2_nearby_physics as nearby_physics
from . import pair_v2_raw_sampling as raw_sampling

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
    if config.get('raw_sampling_version') is not None:
        config['raw_evidence_policy'] = {**config['raw_evidence_policy'], **raw_sampling.fields(),
            'version': raw_sampling.VERSION,
            'attempt_raw': 'all accepted/non-scientific failures; fixed-seed approximately10percent scientific rejected; memory until verdict',
            'other_rejection': 'complete compact counts/reasons; omitted raw and thumbnail not materialized'}
    return config


def tags(config=None):
    partition = 'train' if config is None else config['g01_partition']
    quota_assignment = None if config is None else config.get('g01_quota_assignment')
    extra = {} if quota_assignment is None else policy.quota_plan.declared_fields(quota_assignment)
    configuration = config is not None and config.get('g01_configuration_version') is not None
    training = quota_assignment is not None and quota_assignment.get('version') == policy.quota_plan.TRAIN_VERSION
    return {**stay.tags(), **policy.fields(partition, quota_assignment is not None, configuration,
                config is not None and config.get('nearby_physics_version') is not None,
                config is not None and config.get('g01_engineering_smoke') is not None,
                config is not None and config.get('raw_sampling_version') is not None, training),
            **extra, **search.fields(quota_assignment is not None, configuration, training), **labels.fields(),
            **cadence.fields(), **sharing.fields(), **rank.fields(), **geometry.fields(),
            'producer_module': policy.PRODUCER}


def measured_build(sim, config, baseline, identities, proposal, view, pair_id, directory, reachable, attempt_row=None):
    row = {} if attempt_row is None else attempt_row
    fields = labels.pair_fields(pair_id, identities['candidate_names'])
    row.update(fields, mode=proposal['mode'], proposed_pool=proposal['projection_pool']['pool'])
    if config.get('raw_sampling_version') is not None:
        row.update(raw_sampling.fields())
    if config.get('g01_configuration_version') is not None:
        row['g01_configuration'] = search.configurations.from_inputs(config['scene'], view, identities)
    if proposal['mode'] == 'flip' and proposal['projection_pool']['pool'] == 'look':
        geometry.complete_pair(proposal['_private_pool_record'], proposal['w0'], proposal['w1'])
    if proposal['mode'] == 'stay':
        record, pool = proposal['_private_pool_record'], proposal['projection_pool']['pool']
        proposal = {**proposal, '_private_pool_record': {**record,
            'points': {**record['points'], pool: record['stay_points_by_pool'][pool]}}}
    physics = (nearby_physics.pair_context(sim, row, config)
               if config.get('nearby_physics_version') is not None else nullcontext())
    with physics, labels.pair_context(pair_id, identities['candidate_names'], declared={**config, **fields}):
        try:
            # The original mixed branch is retained, including Direct acceptance;
            # the supplement12 Look-only prefix rejection hook is not installed.
            return _ORIGINAL_BUILD(sim, {**config, **fields}, baseline, identities, proposal, view,
                                   pair_id, directory, reachable, attempt_row=row)
        except search.CollectionQuotaFull as error:
            if row.get('validity', {}).get('acceptance_passed') is not True or row.get('kind') != str(error):
                raise ValueError('Quota skip lacks an unchanged positive scientific verdict') from error
            if isinstance(error, search.ConfigurationAlreadyCollected):
                row['collection_configuration_skip'] = configuration_skip(config, row)
                row.update(status='rejected', discard_reasons=['collection_configuration_already_represented'])
            else:
                row['collection_quota_skip'] = quota_skip(config, row['kind'])
                row.update(status='rejected', discard_reasons=['collection_quota_full'])
            return row


def compact_attempt(row):
    result = look.compact_attempt(row)
    for key in (*policy.FIELDS, *search.FIELDS, *labels.PAIR_FIELDS,
                *cadence.FIELDS, *sharing.FIELDS, *rank.FIELDS, *geometry.FIELDS,
                'flip_geometry_initial_tuple', 'flip_geometry_final_tuple', 'flip_geometry_native_guards',
                'collection_quota_skip', 'proposed_pool') + tuple(
                    key for key in (*nearby_physics.FIELDS, 'far_object_drift_max_m', 'passes_original_physics_rule',
                                    'g01_engineering_smoke') if key in row) + tuple(
                    key for key in (*policy.quota_plan.FIELDS, 'g01_configuration', 'collection_configuration_skip') if key in row):
        result[key] = copy.deepcopy(row.get(key))
    if row.get('nearby_physics_version') is not None:
        context = row.get('nearby_physics_context')
        result['nearby_physics_context'] = (None if context is None else {key: copy.deepcopy(context.get(key))
            for key in ('finalized', 'baseline_reference', 'anchor_name', 'requested_anchor_positions',
                        'supporting_receptacle_names', 'reason')})
        result['nearby_physics_compact_scope'] = 'rejected summary only; per-object far drift and distance maps omitted; accepted pairs retain full measurements'
        for world in ('w0', 'w1'):
            measured = row.get('physical_' + world)
            result['physical_' + world] = (None if measured is None else {key: copy.deepcopy(measured.get(key))
                for key in ('valid_physical', 'original_valid_physical', 'nonanchor_gate_pending', 'reasons')})
    if row.get('g01_configuration') is not None:
        # The immutable configuration validator still checks literal generator
        # inputs on rejected attempts; compact storage must retain those inputs.
        result['view'] = copy.deepcopy(row['view'])
        result['identities'] = copy.deepcopy(row['identities'])
    if row.get('collection_quota_skip') is not None or row.get('collection_configuration_skip') is not None:
        result['validity'] = copy.deepcopy(row.get('validity'))
    if row.get('raw_sampling_version') is not None:
        result.update(raw_sampling.fields())
        if row.get('status') != 'passed' and (row.get('evidence_retention') or {}).get('full_raw_retained') is False:
            result = raw_sampling.compact_without_missing_references(result)
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


def make_quota(config):
    partition = 'train' if config is None else config['g01_partition']
    options = {}
    if config is not None and config.get('g01_quota_version') is not None:
        options = {key: config[key] for key in ('actual_output_caps', 'allowed_kinds')}
    if config is not None and config.get('g01_configuration_version') is not None:
        options.update(scene=config['scene'], seen_configurations=config['g01_seen_configurations'])
    options['training'] = config is not None and config.get('g01_quota_version') == policy.quota_plan.TRAIN_VERSION
    return search.SceneQuota(policy.class_cap(partition), **options)


def quota_skip(config, kind):
    value = {'kind': kind, 'per_scene_class_cap': policy.class_cap(config['g01_partition']),
        'acceptance_passed_before_collection': True,
        'scope': 'collection capacity only; original scientific validity remains unchanged'}
    if config.get('g01_quota_version') is not None:
        value.update(quota_version=config['g01_quota_version'],
                     actual_output_cap=config['actual_output_caps'][kind],
                     plan_reference=copy.deepcopy(config['g01_quota_assignment']['plan_reference']))
    return value


def configuration_skip(config, row):
    return {'version': search.configurations.VERSION, 'configuration': copy.deepcopy(row['g01_configuration']),
        'kind': row['kind'], 'acceptance_passed_before_collection': True,
        'plan_reference': copy.deepcopy(config['g01_quota_assignment']['plan_reference']),
        'scope': 'configuration capacity only; unchanged positive scientific verdict retained'}


@contextmanager
def installed(config=None):
    quota = make_quota(config)
    check = ranked.deadline_check(config) if config is not None else None
    physics_counts = {'counted_attempts': 0, 'finalized_two_worlds': 0, 'pending_or_unmeasured': 0,
        'both_worlds_pass_new_physics': 0, 'pass_new_but_fail_original_physics': 0,
        'exported_original_rule_true': 0, 'exported_original_rule_false': 0}
    with stay.installed(), policy.camera_support(), labels.installed(), sharing.installed():
        original_write, original_summary = marked._BASE_WRITE, marked.summary
        original_observer = marked.construct_observer
        def full_pool(*args, **kwargs):
            if check is None:
                raise RuntimeError('G01 geometry is only bound inside the spawned generator')
            record = search.build_pool(flip._ORIGINAL_POOLS, *args, check_deadline=check, **kwargs)
            if quota.ticketed:
                record.update(search.fields(True, quota.configuration_enabled, quota.training))
            return record
        def observer(sim, proposal, role, record, row, **kwargs):
            if proposal['mode'] == 'flip' and proposal['projection_pool']['pool'] == 'look':
                return ranked.guarded_observer(original_observer, sim, proposal, role, record, row, **kwargs)
            return original_observer(sim, proposal, role, record, row, **kwargs)
        def write(path, row):
            reference = original_write(path, row)
            if Path(path).name == 'attempt.json':
                quota.record(path, row)
                if config is not None and config.get('nearby_physics_version') is not None:
                    nearby_physics.update_counts(physics_counts, row)
            return reference
        def export(config, row, *args, **kwargs):
            if config.get('nearby_physics_version') is not None:
                nearby_physics.validate_fields(row)
                if (row.get('nearby_physics_context', {}).get('finalized') is not True
                        or any(row['physical_' + world].get('nonanchor_gate_pending') is not False
                               for world in ('w0', 'w1'))):
                    raise ValueError('Cannot export a pending supplement16 physical pair')
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
            if config.get('nearby_physics_version') is not None:
                result['nearby_physics_counts'] = copy.deepcopy(physics_counts)
                result['nearby_physics_counts_scope'] = 'actual written counted attempts; physics pass is not full pair acceptance; pending unknown kept separate'
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
        if quota.configuration_enabled or quota.training:
            targets.append((cadence, 'DiscoveryRound', search.ConfigurationPriorityRound))
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
    with installed(config), raw_sampling.installed(config), witness_bridge(config) as bridge:
        result = stay._ORIGINAL_GENERATOR(config, connection)
    result['g01_witness_admission'] = bridge.report()
    return result


def main():
    with installed():
        return marked.main()


if __name__ == '__main__':
    raise SystemExit(main())
