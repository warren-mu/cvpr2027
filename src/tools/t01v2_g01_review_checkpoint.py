"""Strict mixed G01 admission; unchanged scientific and native provenance gates."""
import copy
import json
from pathlib import Path
from contextlib import contextmanager
from src.envs import pair_v2_g01_policy as policy
from src.envs import pair_v2_g01_search as search
from src.envs import pair_v2_fixed_labels as labels
from src.envs import pair_v2_role_policy as roles
from src.envs import pair_v2_flip_geometry as geometry
from src.envs import pair_v2_spawn_sharing as sharing
from src.envs import pair_v2_bbox_rank as rank
from src.envs import pair_v2_flip_rounds_search as cadence
from src.tools import merge_pair_v2 as merge
from src.tools import t01v2_controller_review_checkpoint as controller
from src.tools import t01v2_g01_association as association
from src.tools import t01v2_review_checkpoint as original
from src.tools.t01v2_posthoc_solvability import verified_passed_rows
from src.tools import t01v2_flip_ranked_review_checkpoint as geometry_proof

_BASE_LOAD = merge.load_source


def verify_admission(data):
    from src.tools.g01_admission import validate_source_admission
    config, metrics = data['config'], data['metrics']
    if metrics.get('g01_witness_admission') is None and not merge.source_eligibility(data)['eligible']:
        if (metrics.get('passed_pair_files') or any(value not in (None, 0)
                for value in (metrics.get('passed_by_class') or {}).values())):
            raise ValueError('G01 positive pair claims lack their actual resource cleanup certificate')
        # Runtime/plan identity still needs full authentication. This is a
        # ledger-only failure, never proof of successful witness cleanup.
        validate_source_admission(config)
        return None
    return validate_source_admission(config, metrics)


def is_g01_source(config):
    return config.get('producer_module') == policy.PRODUCER or any(key in config for key in policy.FIELDS)


def proposal_binding(row):
    """Predicted pool does not replace the actual rendered class."""
    pool, kind, mode = row.get('proposed_pool'), row.get('kind'), row.get('mode')
    proposal = row.get('proposal', {})
    if (pool not in ('look', 'direct') or kind not in search.KINDS or mode not in ('flip', 'stay')
            or proposal.get('mode') != mode or proposal.get('projection_pool', {}).get('pool') != pool):
        raise ValueError('G01 row lacks its unchanged explicit proposal pool/mode and actual class')
    return {'proposed_pool': pool, 'actual_kind': kind, 'class_cross_over': kind.split('_', 1)[0] != pool,
        'proposal_verifier': geometry.VERSION if pool == 'look' and mode == 'flip'
            else 'original_native_surface_3d_role' if mode == 'flip' else 'original_stay_role',
        'scope': 'proposal evidence only; actual rendered class requires full original scientific verification'}


def verify_pair_source(data, reference, row, path, *, root):
    eligibility, _, passed = verified_passed_rows(data)
    if not eligibility['eligible']:
        raise ValueError('G01 candidate comes from an ineligible or unfinished source')
    matches = [item for item in passed if Path(item[2]).resolve() == Path(path).resolve()]
    if len(matches) != 1:
        raise ValueError('G01 candidate absent/duplicated in the authenticated passed journal')
    journal, pair_ref, _ = matches[0]
    actual = merge.read_reference(pair_ref, root=root, within=Path(path).resolve().parent)
    if not association.same(row, actual):
        raise ValueError('G01 candidate differs from its authenticated portable pair')
    return association.verify_pair_controller(data, reference, journal, row, path, root=root)


def verify_collection(data):
    """Replay the authenticated actual-attempt journal, never scheduler promises."""
    config, metrics = data['config'], data['metrics']
    _, journal_ref, _ = verified_passed_rows(data)
    measured = metrics.get('g01_collection')
    if journal_ref is None:
        # An audited pre-generation failure has no observations. Do not turn it
        # into a fabricated empty collection, or admit any positive claim.
        if measured is not None:
            raise ValueError('G01 collection claim lacks an authenticated attempt journal')
        return None
    journal = merge.reference_path(journal_ref, within=merge.ROOT / 'data/world_pairs/v2')
    from src.envs.pair_v2_g01_shard import make_quota, quota_skip, configuration_skip
    quota = make_quota(config)
    if quota.ticketed:
        policy.quota_plan.validate_assignment(config, authenticate=True)
    for line in journal.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reference = row['raw_attempt']
        raw = merge.read_reference(reference, within=config['raw_evidence_dir'])
        if raw.get('counts_as_pair_attempt') is not True or raw.get('proposed_pool') != row.get('proposed_pool'):
            raise ValueError('G01 journal contains an uncounted or different-pool raw attempt')
        search.validate_fields(raw)
        if config.get('raw_sampling_version') is not None:
            from src.envs import pair_v2_raw_sampling as sampling
            sampling.validate_fields(raw); sampling.validate_decision(raw, raw_path=merge.reference_path(reference, within=config['raw_evidence_dir']))
        if config.get('nearby_physics_version') is not None:
            from src.envs import pair_v2_nearby_physics as nearby
            nearby.validate_fields(raw)
        if quota.ticketed and any(raw.get(key) != config.get(key) for key in policy.quota_plan.FIELDS):
            raise ValueError('Raw journal attempt changed its immutable scene tickets')
        if (raw.get('g01_search_version') in (search.QUOTA_VERSION, search.CONFIG_VERSION)) != quota.ticketed:
            raise ValueError('Raw journal proposal policy differs from source quota version')
        if raw.get('collection_quota_skip') is not None:
            kind = raw.get('kind')
            expected = quota_skip(config, kind)
            if (raw['collection_quota_skip'] != expected or quota.passed.get(kind) != quota.limits.get(kind)
                    or raw.get('status') != 'rejected' or raw.get('discard_reasons') != ['collection_quota_full']
                    or raw.get('validity', {}).get('acceptance_passed') is not True
                    or raw.get('pair_file') is not None or raw.get('pair_sha256') is not None):
                raise ValueError('G01 quota skip lacks its original positive verdict and actual full class')
        if raw.get('collection_configuration_skip') is not None:
            if not quota.configuration_enabled:
                raise ValueError('Legacy source cannot claim configuration suppression')
            configuration = quota.configuration_of(raw)
            if (raw['collection_configuration_skip'] != configuration_skip(config, raw)
                    or (configuration['configuration_id'], raw.get('kind')) not in quota.configuration_seen
                    or raw.get('collection_quota_skip') is not None or raw.get('status') != 'rejected'
                    or raw.get('discard_reasons') != ['collection_configuration_already_represented']
                    or raw.get('validity', {}).get('acceptance_passed') is not True
                    or raw.get('pair_file') is not None or raw.get('pair_sha256') is not None):
                raise ValueError('Configuration skip lacks its original verdict and authenticated earlier occupancy')
        quota.record(merge.reference_path(reference, within=config['raw_evidence_dir']), raw)
    if measured != quota.summary() or metrics.get('passed_by_class') != quota.passed:
        raise ValueError('G01 collection summary differs from its authenticated actual attempts')
    return quota.summary()


@contextmanager
def source_support():
    with policy.camera_support():
        previous_producer, previous_load = merge.shard_producer, merge.load_source
        previous_check, previous_role = merge.check_pair, roles.verify_proposal_role
        loaded = {}
        loaded_partitions = set()
        def producer(command, config):
            if not is_g01_source(config):
                return previous_producer(command, config)
            policy.validate_source(config)
            if config.get('g01_engineering_smoke') is not None:
                raise ValueError('Engineering smoke is not an independent G01 collection source')
            common = copy.deepcopy(config)
            common['producer_module'] = 'src.envs.pair_v2_marked_shard'
            merge.marked_policy_configuration(common)
            arguments = merge.producer_arguments(command, policy.PRODUCER)
            if (arguments.count('--timeout-seconds') != 1
                    or arguments.index('--timeout-seconds') + 1 >= len(arguments)
                    or float(arguments[arguments.index('--timeout-seconds') + 1]) != 1200):
                raise ValueError('G01 source lacks its full1200-second logged grant')
            return arguments, False
        def load(path):
            data, reference = _BASE_LOAD(path)
            if is_g01_source(data['config']):
                policy.validate_source(data['config'], data['metrics'])
                if data['config'].get('g01_engineering_smoke') is not None:
                    raise ValueError('Engineering smoke cannot enter an independent G01 checkpoint')
                verify_collection(data)
                verify_admission(data)
                partition = data['config']['g01_partition']
                if loaded_partitions and partition not in loaded_partitions:
                    raise ValueError('Training and evaluation sources cannot share a checkpoint/export context')
                loaded_partitions.add(partition)
                raw = data['config']['raw_evidence_dir']
                if raw in loaded and loaded[raw][0]['config'] != data['config']:
                    raise ValueError('Different authenticated configurations claim one raw source directory')
                loaded[raw] = (data, reference)
            return data, reference
        def role(row, positions):
            if row.get('g01_search_version') is not None:
                search.validate_fields(row)
                if row.get('mode') == 'flip':
                    if row.get('proposed_pool') == 'look':
                        raise ValueError('G01 Look-pool Flip lacks independently authenticated geometry evidence')
                    return search.verify_flip_role(row, positions)
            return previous_role(row, positions)
        def check(row, path, **kwargs):
            source = loaded.get(kwargs.get('raw_root'))
            if source is None:
                if any(row.get(key) is not None for key in (*policy.FIELDS, *search.FIELDS, 'fixed_label_version')):
                    raise ValueError('New G01 pair has no independently authenticated source configuration')
                return previous_check(row, path, **kwargs)
            data, reference = source
            config = data['config']
            if (row.get('producer_module') != policy.PRODUCER
                    or any(row.get(key) != item for key, item in policy.fields(config['g01_partition'], config.get('g01_quota_version') is not None,
                    config.get('g01_configuration_version') is not None,
                    config.get('nearby_physics_version') is not None,
                    config.get('g01_engineering_smoke') is not None,
                    config.get('raw_sampling_version') is not None).items())
                    or any(row.get(key) is not None for key in ('flip_unit_version', 'flip_unit_policy',
                        'flip_search_version', 'flip_search_policy', 'look_unit_version', 'look_unit_policy'))):
                raise ValueError('Pair differs from its authenticated G01 source policy')
            if config.get('g01_quota_version') is not None and any(
                    row.get(key) != config.get(key) for key in policy.quota_plan.FIELDS):
                raise ValueError('Pair differs from its authenticated static quota assignment')
            search.validate_fields(row)
            for module in (geometry, sharing, rank, cadence):
                module.validate_fields(row)
            binding = proposal_binding(row)
            root = Path(kwargs.get('root', merge.ROOT)).resolve()
            associated = verify_pair_source(data, reference, row, path, root=root)
            proof = None
            if row['mode'] == 'flip' and binding['proposed_pool'] == 'look':
                read = geometry_proof.IndexedReader(associated, root=root)
                search.validate_fields(read(row['projection_pool_reference']))
                proof = geometry_proof.verify_geometry_evidence(row, path,
                    read(associated['baseline_reference']), read, config, root=root)
            elif any(row.get(key) is not None for key in ('flip_geometry_initial_tuple',
                    'flip_geometry_final_tuple', 'flip_geometry_native_guards')):
                raise ValueError('Direct-pool/Stay source cannot claim a Look-pool Flip geometry proof')
            active_role = roles.verify_proposal_role
            def verified_role(actual, positions):
                if actual is row and proof is not None:
                    result = geometry_proof.verify_role(actual, positions, proof)
                    return {**{key: value for key, value in result.items() if key != 'flip_search_version'},
                            'g01_search_version': row['g01_search_version']}
                return active_role(actual, positions)
            scientific_kwargs = dict(kwargs)
            for key in ('nearby_physics_version', 'nearby_physics_policy'):
                if key in scientific_kwargs and scientific_kwargs[key] != config.get(key):
                    raise ValueError('Caller nearby-physics policy differs from authenticated G01 source')
                if config.get(key) is not None:
                    scientific_kwargs[key] = copy.deepcopy(config[key])
            try:
                roles.verify_proposal_role = verified_role
                checked = labels.checked_pair(previous_check, row, path, declared_source=config, **scientific_kwargs)
            finally:
                roles.verify_proposal_role = active_role
            return {**checked, 'g01_partition': config['g01_partition'], 'g01_proposal_verification': binding,
                'controller_preflight_association': associated, 'complete_flip_geometry_verification': proof}
        try:
            merge.shard_producer, merge.load_source, merge.check_pair = producer, load, check
            roles.verify_proposal_role = role
            yield
        finally:
            merge.shard_producer, merge.load_source, merge.check_pair = previous_producer, previous_load, previous_check
            roles.verify_proposal_role = previous_role


@contextmanager
def collector_support():
    previous = original.collect_sources, controller.policy, controller.verify_pair_controller
    with source_support():
        try:
            original.collect_sources = controller.collect_sources
            controller.policy = policy
            controller.verify_pair_controller = association.verify_pair_controller
            yield
        finally:
            original.collect_sources, controller.policy, controller.verify_pair_controller = previous


def main(argv=None):
    with collector_support():
        return original.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
