"""Training Look ticket counts from original qualification and construction receipts.

Metadata-only reuse of completed original scientific work. This is not another
native/cache check, a final T04 pool, or a claim that unknown exports passed.
"""
import argparse
import copy
import json
import math
import hashlib
from pathlib import Path
import signal
import re
import traceback

from src.envs.thor_common import ROOT
from src.tools.t02_export import save_json
from src.tools.t01_partition import audited_result
from src.tools import g01_quota_plan as quota

MODULE = 'src.tools.g01_train_inventory'
VERSION = 'g01_train_postfilter_inventory_receipt_v1'
AUTHORITY = 'original_qualified_population_metadata_receipt_v1'
MIN_STAY_METRES = .15
SCOPE = 'original successful qualification/cache and published construction receipts; metadata-only recomputation; native science not repeated; not final T04 pool'


def need(value, message):
    if not value: raise ValueError(message)


def read(ref, root=ROOT, within=None):
    from src.tools.merge_pair_v2 import read_reference
    return read_reference(ref, root=root, within=Path(root) if within is None else within)


def counts(rows):
    by_scene = {}
    for row in rows:
        by_scene.setdefault(row['scene'], {kind: 0 for kind in quota.KINDS})[row['kind']] += 1
    return quota.summed(by_scene.values()), by_scene


def stay_metadata(pair, pair_path, root):
    """The original realized cutoff from authenticated portable metadata only."""
    from src.envs.pair_v2_marked_policy import stay_horizontal_displacement
    worlds, references = [], []
    for index in (0, 1):
        matches = [ref for ref in pair['portable_files'] if Path(ref['file']).name == f'snapshot_w{index}.json']
        need(len(matches) == 1, 'Stay needs two unique original world metadata references')
        worlds.append(read(matches[0], root, Path(pair_path).parent))
        references.append(copy.deepcopy(matches[0]))
    measured = stay_horizontal_displacement(worlds, pair['identities']['anchor_name'])
    need(math.isfinite(measured['distance_m']) and measured['distance_m'] >= 0,
         'Invalid realized Stay displacement')
    declared = pair.get('stay_horizontal_displacement')
    need(declared is None or measured == declared,
         'Stay displacement changed from the original independently checked metadata')
    return measured, references


def same_reference(left, right):
    return all(left.get(key) == right.get(key) for key in ('file', 'sha256'))


def postfilter(rows, heldout):
    """Original measured Stay cutoff, then original seed0 A/B balancing.

    Ratio is a readiness diagnostic, never a newly invented selection rule.
    """
    from src.envs.pair_table_balance import balance_pairs
    kept, omitted = [], []
    for row in rows:
        if row['scene'] in heldout:
            omitted.append({'dataset_key': row['dataset_key'], 'reason': 'fixed_T04_holdout_scene'})
        elif row['kind'].endswith('stay') and row['stay_displacement']['distance_m'] < MIN_STAY_METRES:
            omitted.append({'dataset_key': row['dataset_key'], 'reason': 'Stay_horizontal_displacement_below_0_15m',
                            'measurement': copy.deepcopy(row['stay_displacement'])})
        else:
            kept.append(row)
    balanced = balance_pairs(kept, seed=0)
    keys = {row['dataset_key'] for row in balanced['selected']}
    selected = [row for row in rows if row['dataset_key'] in keys]
    by_kind, by_scene = counts(selected)
    flip, stay = by_kind['look_flip'], by_kind['look_stay']
    ratio_ready = stay > 0 and 4 * stay <= 5 * flip and 4 * flip <= 5 * stay
    return {'confirmed_by_kind': by_kind, 'confirmed_by_scene': by_scene,
        'selected_dataset_keys': sorted(keys), 'filter_omissions': omitted,
        'balance': balanced, 'ratio_ready': ratio_ready,
        'look_ratio': flip / stay if stay else None,
        'class_full': {kind: by_kind[kind] >= 40 for kind in ('look_flip', 'look_stay')},
        'global_ready': flip >= 40 and stay >= 40 and ratio_ready
            and balanced['after']['overall']['target_left_unknown_world_count'] == 0,
        'global_ready_scope': 'Look quota target and ratio only; not final T04 or global A/left balance acceptance',
        'global_balance_within_0_4_to_0_6': balanced['after']['global_balance_within_0_4_to_0_6'],
        'formal_T04_ready': None,
        'needs_human_ratio_decision': flip >= 40 and stay >= 40 and not ratio_ready,
        'selection_scope': 'fixed holdout excluded; Stay>=0.15m; original balance_pairs(seed0); ratio does not alter counts'}


def construction_inventory(references, root):
    """Explicit published cutoff; original successful checkpoint producer whitelist."""
    from src.envs import pair_table_source as source
    from src.tools import merge_pair_v2 as merge
    need(isinstance(references, list) and references, 'Complete published construction cutoff is required')
    publications, entries, seen = [], {}, set()
    for expected in references:
        quota.reference(expected)
        need((expected['file'], expected['sha256']) not in seen, 'Duplicate construction checkpoint')
        seen.add((expected['file'], expected['sha256']))
        container, actual, published = source._container(Path(root) / expected['file'])
        need(all(actual.get(k) == expected[k] for k in ('file', 'sha256')),
             'Construction checkpoint bytes changed')
        need(container['_meta'].get('end') and container['_meta'].get('exit_code') == 0
             and container['metrics'].get('errors') == [], 'Failed construction publication cannot free capacity')
        publications.append(actual)
        for entry in published:
            pair_ref = entry['pair_reference']
            pair_path = (Path(root) / pair_ref['file']).resolve()
            row = read(pair_ref, root, Path(root) / 'data/world_pairs/v2')
            need(row.get('status') == 'passed' and quota.scene_allowed(row.get('scene'), 'train')
                and all(row.get(k) == entry.get(k) for k in ('pair_id', 'scene', 'kind')),
                'Construction occupancy must bind original published training pair bytes')
            identity = merge.proposal_identity(row)
            verified = entry.get('independent_verification', {})
            need(entry.get('proposal_identity') == identity and verified.get('proposal_identity') == identity
                and verified.get('validity', {}).get('acceptance_passed') is True,
                'Construction occupancy lacks its successful original independent proposal proof')
            record = {'proposal_identity': identity, 'scene': row['scene'], 'kind': row['kind'],
                'pair_reference': copy.deepcopy(pair_ref), 'checkpoint_reference': expected,
                'target_names': [row['target_w0'], row['target_w1']]}
            binding = {'pair_reference': copy.deepcopy(pair_ref), 'checkpoint_reference': copy.deepcopy(expected)}
            if row['kind'].endswith('stay'):
                measured, metadata_refs = stay_metadata(row, pair_path, root)
                binding.update(stay_displacement=measured, input_metadata_references=metadata_refs)
            if identity in entries:
                need(all(entries[identity][k] == record[k] for k in ('scene', 'kind', 'target_names')),
                    'Historical duplicate proposal conflicts')
                if binding not in entries[identity]['published_bindings']:
                    entries[identity]['published_bindings'].append(binding)
            else:
                entries[identity] = dict(record, published_bindings=[binding])
    return list(entries.values()), publications


def known_rejections(references, constructed, *, root=ROOT):
    """Reuse complete original single-pair T02 verdicts, never infer from errors."""
    from src.tools import t02_cache_check as cache, t02_export_parallel as parallel, t02_export as export
    from src.tools import merge_pair_v2 as merge
    from src.tools.train_holdout_freeze import values
    by_identity = {row['proposal_identity']: row for row in constructed}
    records, sources, seen = [], [], set()
    for expected in references:
        quota.reference(expected)
        token = (expected['file'], expected['sha256'])
        need(token not in seen, 'Duplicate explicit T02 rejection source'); seen.add(token)
        data, actual = audited_result(Path(root) / expected['file'])
        need(all(actual.get(k) == expected[k] for k in ('file', 'sha256')), 'Original T02 rejection source changed')
        cfg, meta, metrics = data['config'], data['_meta'], data['metrics']
        cache.exact_export_command(meta.get('command'), cfg.get('name'))
        need(meta.get('end') and meta.get('exit_code') == 0 and metrics.get('producer_exit_code') == 0
            and metrics.get('export_checks_complete') is True and metrics.get('errors') == []
            and metrics.get('cleanup_errors') == [] and metrics.get('unstarted_pair_ids') == []
            and metrics.get('black_frame_detected') is False and cfg.get('depth') == 4,
            'Only a complete original scientific rejection can release pending capacity')
        rows = metrics.get('pairs')
        need(isinstance(rows, list) and len(rows) == 1 and rows[0].get('status') == 'rejected'
            and rows[0].get('cleanup_errors') == [] and rows[0].get('rejection_reasons'),
            'Require the original single-pair scientific rejection, not a cache/parent failure')
        row = rows[0]
        pair_dir = Path(cfg['scratch']) / 't02_export' / cfg['name'] / cfg['pbs_jobid'] / 'pair_0000'
        need(Path(row['scratch_directory']).resolve() == pair_dir.resolve()
            and Path(cfg['scratch_directory']).resolve() == pair_dir.parent.resolve(), 'Rejected pair private directory changed')
        package = cache.read_json_ref(row['source_package'], pair_dir, expected=pair_dir / 'authenticated_source.json')
        pair_ref = package['source_provenance']['pair']
        pair = read(pair_ref, root, Path(root) / 'data/world_pairs/v2')
        identity = merge.proposal_identity(pair)
        need(identity in by_identity and pair.get('status') == 'passed'
            and all(pair.get(k) == by_identity[identity][k] for k in ('scene', 'kind'))
            and [pair['target_w0'], pair['target_w1']] == by_identity[identity]['target_names'],
            'Rejected source is outside the original construction cutoff')
        binding = next((item for item in by_identity[identity]['published_bindings']
            if same_reference(item['pair_reference'], pair_ref)
            and same_reference(item['checkpoint_reference'], package['source_provenance']['container'])), None)
        need(binding is not None, 'Rejected pair/container is not an authenticated published binding')
        entry = {'source': binding['checkpoint_reference'], 'pair': binding['pair_reference'],
            'pair_id': pair['pair_id'], 'scene': pair['scene'], 'proposal_identity': identity}
        argv = merge.producer_arguments(meta['command'], 'src.tools.t02_export')
        need(values(argv, '--source') == [cfg['source']] and values(argv, '--pair-id') == [pair['pair_id']]
            and values(argv, '--pair-file') == [cfg['pair_file']], 'Original rejected export arguments changed')
        allocation = {key: cfg[key] for key in ('pbs_jobid', 'cuda_visible_devices', 'cpu_affinity', 'scratch', 'node')}
        outcome = parallel.child_outcome({'exit_code': 0}, data, allocation, meta['command'], entry)
        need(outcome['export_checks_complete'], 'Original rejected child is incomplete')
        need(not any(row.get(key) for key in ('raw_index_error', 'worker_error', 'black_frame_reference',
                'portable_black_evidence', 'raw_retention_error'))
            and not row.get('raw_retention', {}).get('retention_error'),
            'Infrastructure/raw/black failures cannot be scientific rejections')
        error = row.get('error')
        if error is not None:
            need(isinstance(error, dict) and error.get('type') == 'ValueError'
                and error.get('message') in export.REJECTIONS
                and row['rejection_reasons'] == [error['message']]
                and row.get('table_complete') is False,
                'Only the original explicit native scientific rejection can release pending capacity')
        else:
            need(row.get('table_complete') is True and row.get('native_evidence_verified') is True
                and row.get('pair_checks_passed') is False,
                'A completed pair-check rejection needs its original table and native verdict')
            detail = cache.read_json_ref(row['detail'], pair_dir, expected=pair_dir / 'checks.json')
            check = detail.get('pair_checks', {})
            need(detail.get('table_manifest') == row.get('table_manifest')
                and detail.get('native_checks', {}).get('passed') is True
                and check.get('passed') is False and check.get('synthetic_only') is False
                and check.get('reasons') == row['rejection_reasons'],
                'Scientific rejection differs from its original complete pair-check receipt')
        sources.append(actual)
        records.append({'proposal_identity': identity, 'source': actual,
            'source_package': row['source_package'], 'pair_reference': pair_ref,
            'rejection_reasons': copy.deepcopy(row['rejection_reasons']), 'native_revalidation_performed': False})
    return records, sources


def additional_cache_rows(references, *, root=ROOT):
    """Count original complete T02/cache certificates using only s0 mask bytes.

    This is a receipt projection, not native or full-table reauthentication.
    Every original batch must be complete; rejected or missing table rows do not
    become accepted through this path. Fixed holdout is applied by the caller.
    """
    from src.tools import smoke_e2e_pilot_certificates as certificates
    from src.tools import t02_cache_check as cache, t02_export as export, t02_export_parallel as parallel
    from src.tools import merge_pair_v2 as merge
    from src.envs.pair_table_export import TABLE_VERSION, FROZEN_RENDER_VERSION
    from src.envs.pair_table_identity import identity_fields
    from src.envs.pair_table_balance import left_half_from_mask, checked_rows
    from src.tools.train_holdout_freeze import values
    rows, publications, proofs, inputs, seen = [], [], [], [], set()
    for expected in references:
        quota.reference(expected)
        token = (expected['file'], expected['sha256'])
        need(token not in seen, 'Duplicate additional cache certificate'); seen.add(token)
        evidence = certificates.certificate_metadata(expected, root=root)
        publications.append(evidence['source']); result_cache = {}
        for item in evidence['complete']:
            source_ref = item['source']; source_key = (source_ref['file'], source_ref['sha256'])
            if source_key not in result_cache:
                data, actual = audited_result(Path(root) / source_ref['file'])
                need(same_reference(actual, source_ref), 'Additional original export bytes changed')
                result_cache[source_key] = data, actual
            data, actual = result_cache[source_key]
            cfg, meta, metrics = data['config'], data['_meta'], data['metrics']
            cache.exact_export_command(meta['command'], cfg['name'])
            need(meta.get('end') and meta.get('exit_code') == 0 and cfg.get('version') == export.VERSION
                and cfg.get('gpus') == 1 and cfg.get('depth') == 4 and metrics.get('kind') == 't02_table_export'
                and metrics.get('producer_exit_code') == 0 and metrics.get('export_checks_complete') is True
                and metrics.get('errors') == [] and metrics.get('cleanup_errors') == []
                and metrics.get('unstarted_pair_ids') == [] and metrics.get('black_frame_detected') is False,
                'Additional original T02 result is incomplete or failed')
            exported = metrics['pairs']
            need([r['pair_id'] for r in exported] == cfg['pair_id'] and len(exported) == 1,
                 'Additional count inputs require original single-pair exports')
            row = exported[0]
            directory = Path(cfg['scratch']).resolve() / 't02_export' / cfg['name'] / cfg['pbs_jobid'] / 'pair_0000'
            need(Path(row['scratch_directory']).resolve() == directory and row['pair_id'] == item['pair_id']
                and row.get('status') == 'exported' and row.get('table_complete') is True
                and row.get('native_evidence_verified') is True and row.get('pair_checks_passed') is True
                and row.get('cleanup_errors') == [] and row.get('error') is None
                and not any(row.get(k) for k in ('raw_index_error', 'worker_error', 'raw_retention_error'))
                and row['table_manifest'] == item['table_manifest'], 'Additional export table was not accepted')
            package = cache.read_json_ref(row['source_package'], directory, expected=directory / 'authenticated_source.json')
            details = cache.read_json_ref(row['detail'], directory, expected=directory / 'checks.json')
            pair_ref = package['source_provenance']['pair']; pair_path = (Path(root) / pair_ref['file']).resolve()
            pair = read(pair_ref, root, Path(root) / 'data/world_pairs/v2')
            proposal = merge.proposal_identity(pair)
            need(pair.get('status') == 'passed' and quota.scene_allowed(pair.get('scene'), 'train')
                and package['independent_verification'].get('proposal_identity') == proposal
                and package['independent_verification'].get('validity', {}).get('acceptance_passed') is True,
                'Additional table lacks its original independent training construction')
            allocation = {key: cfg[key] for key in ('pbs_jobid', 'cuda_visible_devices', 'cpu_affinity', 'scratch', 'node')}
            entry = {'source': package['source_provenance']['container'], 'pair': pair_ref,
                'pair_id': pair['pair_id'], 'scene': pair['scene'], 'proposal_identity': proposal}
            argv = merge.producer_arguments(meta['command'], 'src.tools.t02_export')
            need(values(argv, '--source') == [entry['source']['file']]
                and values(argv, '--pair-id') == [pair['pair_id']]
                and values(argv, '--pair-file') == [pair_ref['file']],
                'Additional original export command changed source identity')
            outcome = parallel.child_outcome({'exit_code': 0}, data, allocation, meta['command'], entry)
            need(outcome['export_checks_complete'] and details['table_manifest'] == row['table_manifest']
                and details['native_checks'].get('passed') is True
                and details['pair_checks'].get('passed') is True and details['pair_checks'].get('synthetic_only') is False,
                'Additional table differs from original native/pair-check receipt')
            key = item['dataset_key']; table_ref = evidence['certificate']['tables'][key]
            need(table_ref == {k: row['table_manifest'][k] for k in ('file', 'sha256', 'bytes')},
                 'Additional cache certificate/table reference changed')
            table = cache.read_json_ref(table_ref, directory)
            identity = identity_fields(pair['pair_id'], pair_ref)
            need(table.get('version') == TABLE_VERSION and table.get('depth') == 4
                and table['source'].get('t01') == package['source_provenance']
                and table['source'].get('export_version') == export.VERSION
                and table['source'].get('frozen_render_version') == FROZEN_RENDER_VERSION
                and table['source'].get('field_of_view') == package['source_config'].get('field_of_view') == 60
                and table.get('dataset_identity', identity) == identity and identity['dataset_key'] == key
                and all(table[k] == package[k] == pair[k] for k in ('pair_id', 'scene', 'kind'))
                and list(table['candidate_names']) == package['candidate_names'] == pair['identities']['candidate_names']
                and list(table['target_names']) == package['target_names'] == [pair['target_w0'], pair['target_w1']],
                'Additional original table/header and published pair disagree')
            # Training tables remain the original raw format; the test-only
            # compressed codec cannot be used to change this count projection.
            need('storage_encoding' not in table, 'Training count requires original uncompressed training table')
            views = table['states'][table['initial_state']]['captures']
            need(len(views) == 2 and all(v.get('synthetic') is False for v in views), 'Synthetic s0 count evidence')
            measurements, mask_refs = [], []
            for world, target in enumerate(table['target_names']):
                ref = views[world]['masks'].get(target); mask = None
                if ref is not None:
                    path = (Path(table_ref['file']).parent / ref['file']).resolve()
                    need(path.parent == Path(table_ref['file']).resolve().parent and ref in table['files'],
                         's0 mask is outside its original table inventory')
                    mask = path.read_bytes()
                    need(len(mask) == ref['bytes'] and hashlib.sha256(mask).hexdigest() == ref['sha256'],
                         'Original s0 target mask bytes changed')
                    mask_refs.append(dict(ref, file=str(path)))
                measurements.append(left_half_from_mask(mask, 255, 255))
            projected = {'pair_id': pair['pair_id'], 'original_pair_id': pair['pair_id'], 'dataset_key': key,
                'dataset_identity': identity, 'scene': pair['scene'], 'kind': pair['kind'],
                'candidate_names': list(table['candidate_names']), 'target_names': list(table['target_names']),
                's0_target_left': [m['left'] for m in measurements], 's0_left_measurements': measurements,
                'proposal_identity': proposal, 'pair_reference': pair_ref, 'checkpoint_reference': entry['source'],
                'table_manifest': table_ref, 'cache_source': evidence['source'], 'synthetic': False}
            inputs.extend([pair_ref, row['source_package'], row['detail'], table_ref, *mask_refs])
            if pair['kind'].endswith('stay'):
                projected['stay_displacement'], refs = stay_metadata(pair, pair_path, root)
                inputs.extend(refs)
            rows.append(checked_rows([projected])[0])
        proofs.append({'source': evidence['source'], 'dataset_keys': [r['dataset_key'] for r in evidence['complete']],
            'source_export_results': [v[1] for v in result_cache.values()],
            **{k: evidence['data']['metrics'][k] for k in ('validation_report', 'authorization_manifest', 'sampling_plan')},
            'native_scientific_revalidation_performed': False,
            'scope': 'Original full T02/cache receipts and SHA-bound s0 masks reused only for generation capacity'})
    return rows, publications, proofs, inputs


def build_inventory(qualification_reference, holdout_reference, construction_sources, *, known_rejection_sources=(),
                    certificate_count_basis=False, additional_cache_sources=(), root=ROOT):
    from src.tools.train_holdout_freeze import authenticate_population, load_frozen_holdout, assert_pilot_split
    from src.tools import merge_pair_v2 as merge
    root = Path(root).resolve()
    need(type(certificate_count_basis) is bool, 'Explicit certificate count mode required')
    manifest, proof, qualified_ref, manifest_ref = authenticate_population(
        qualification_reference, certificates_after_02=certificate_count_basis, root=root)
    if not certificate_count_basis:
        need('qualification' not in manifest and 'qualification' not in proof,
             'Certificate count authority needs its explicit opt-in')
    frozen, freeze_ref = load_frozen_holdout(holdout_reference, root=root)
    assert_pilot_split(manifest['entries'], manifest['split'], frozen)
    normalized = {row['dataset_key']: row for row in manifest['balance']['selected']}
    need(len(normalized) == len(manifest['entries']), 'Qualified normalized population is incomplete')
    rows, input_refs = [], []
    for entry in manifest['entries']:
        key = entry['dataset_key']; original = normalized[key]
        pair_ref = entry['pair_reference']; pair_path = (root / pair_ref['file']).resolve()
        pair = read(pair_ref, root, root / 'data/world_pairs/v2'); input_refs.append(pair_ref)
        need(pair.get('status') == 'passed' and all(pair.get(k) == entry.get(k) for k in ('scene', 'kind'))
            and pair.get('pair_id') == entry['original_pair_id']
            and merge.proposal_identity(pair) == entry['proposal_identity'],
            'Qualified pair differs from its original immutable proposal')
        row = {'pair_id': entry['original_pair_id'], 'original_pair_id': entry['original_pair_id'],
            'dataset_key': key, 'dataset_identity': copy.deepcopy(original['dataset_identity']),
            'proposal_identity': entry['proposal_identity'], 'scene': entry['scene'], 'kind': entry['kind'],
            'candidate_names': copy.deepcopy(original['candidate_names']),
            'target_names': copy.deepcopy(original['original_target_names']),
            's0_target_left': copy.deepcopy(original['original_s0_target_left']),
            'pair_reference': copy.deepcopy(pair_ref), 'synthetic': False}
        need(row['candidate_names'] == pair['identities']['candidate_names']
            and row['target_names'] == [pair['target_w0'], pair['target_w1']],
            'Qualified A/B labels differ from original pair')
        if row['kind'].endswith('stay'):
            measurement, metadata_refs = stay_metadata(pair, pair_path, root)
            input_refs.extend(metadata_refs)
            row['stay_displacement'] = measurement
            row['original_stay_diagnostic_present'] = pair.get('stay_horizontal_displacement') is not None
        rows.append(row)
    extra, extra_sources, extra_proofs, extra_inputs = additional_cache_rows(additional_cache_sources, root=root)
    by_key = {r['dataset_key']: r for r in rows}
    for row in extra:
        if row['dataset_key'] in by_key:
            previous = by_key[row['dataset_key']]
            need(all(previous[k] == row[k] for k in ('pair_reference', 'scene', 'kind', 'candidate_names',
                'target_names', 's0_target_left', 'proposal_identity'))
                and previous.get('stay_displacement') == row.get('stay_displacement'),
                'Repeated cache certificate changes the original count measurement')
        else:
            rows.append(row); by_key[row['dataset_key']] = row
    need(len({row['proposal_identity'] for row in rows}) == len(rows),
         'Distinct dataset keys repeat an original proposal; do not double-count or choose a favorable variant')
    input_refs.extend(extra_inputs)
    heldout = set(frozen['selected_scenes'])
    constructed, publications = construction_inventory(construction_sources, root)
    constructed_by_id = {row['proposal_identity']: row for row in constructed}
    need(all(row['proposal_identity'] in constructed_by_id for row in rows),
        'Construction cutoff omitted a qualified original pair')
    for row in rows:
        prior = constructed_by_id[row['proposal_identity']]
        need(all(prior[k] == row[k] for k in ('scene', 'kind', 'target_names'))
            and any(same_reference(binding['pair_reference'], row['pair_reference'])
                and ('checkpoint_reference' not in row or same_reference(binding['checkpoint_reference'], row['checkpoint_reference']))
                for binding in prior['published_bindings']),
             'Construction and cache-qualified class/targets or published binding conflict')
    occupancy_kind, occupancy_scene = counts(constructed)
    rejected, rejection_publications = known_rejections(known_rejection_sources, constructed, root=root)
    # Only an explicitly supplied, authenticated pair-specific scientific
    # verdict can remove an earlier cache-qualified pair. A failed/timeout
    # whole-population run supplies no such evidence and removes nothing.
    exclusions = [row for row in rows if any(same_reference(row['pair_reference'], r['pair_reference']) for r in rejected)]
    rows = [row for row in rows if row not in exclusions]
    qualified_ids = {row['proposal_identity'] for row in rows}
    filtered = postfilter(rows, heldout)
    # A duplicate proposal can have several published measurements. Release its
    # one global reservation only when every original binding has a known
    # terminal rejection; never select the most convenient duplicate verdict.
    fully_rejected_ids = {row['proposal_identity'] for row in constructed
        if all(any(result['proposal_identity'] == row['proposal_identity']
                   and same_reference(result['pair_reference'], binding['pair_reference'])
                   for result in rejected) for binding in row['published_bindings'])}
    short_stays = [row for row in constructed if row['kind'].endswith('stay')
        and all(binding['stay_displacement']['distance_m'] < MIN_STAY_METRES
                for binding in row['published_bindings'])]
    short_ids = {row['proposal_identity'] for row in short_stays}
    pending = [row for row in constructed if row['proposal_identity'] not in qualified_ids
               and row['proposal_identity'] not in fully_rejected_ids
               and row['proposal_identity'] not in short_ids
               and row['scene'] not in heldout and row['kind'].startswith('look_')]
    # Current filter/balance rejections are known, not pending eligible inventory.
    # Keep their identities and original scene occupancy without blocking repairs
    # through newly generated opposite-side Stay pairs.
    reserved_kind, reserved_scene = counts(pending)
    cached_kind, cached_scene = counts(rows)
    return {'version': VERSION, 'count_stage': quota.STAGES['train'], 'count_basis_authority': AUTHORITY,
        'qualification_reference': qualified_ref, 'manifest_reference': manifest_ref,
        'holdout_reference': quota.reference(holdout_reference), 'holdout_proof': freeze_ref,
        'cache_sources': copy.deepcopy(manifest['cache_sources']) + list(additional_cache_sources),
        'formal_cache_proofs': proof['formal_cache_batches'] + extra_proofs,
        'certificate_count_basis': certificate_count_basis, 'additional_cache_sources': list(additional_cache_sources),
        'additional_cache_publications': extra_sources,
        'population_authority': copy.deepcopy(manifest.get('qualification')),
        'extra_source_reauthentication': copy.deepcopy(manifest.get('qualification', {}).get('extra_source_reauthentication')),
        'formal_pool_authorized': False, 'later_scientific_exclusions': exclusions,
        'construction_sources': copy.deepcopy(construction_sources), 'construction_publications': publications,
        'known_rejection_sources': list(known_rejection_sources), 'known_rejections': rejected,
        'fully_scientifically_rejected_proposal_ids': sorted(fully_rejected_ids),
        'permanent_short_stay_entries': short_stays,
        'permanent_short_stay_scope': 'All published variants have original realized horizontal displacement <0.15m; not counted and not globally reserved; original scene/class occupancy retained',
        'rejection_publications': rejection_publications,
        'construction_occupancy_by_kind': occupancy_kind, 'construction_occupancy_by_scene': occupancy_scene,
        'construction_entries': constructed, 'cached_by_kind': cached_kind, 'cached_by_scene': cached_scene,
        'qualified_rows': rows, 'input_metadata_references': input_refs, **filtered,
        'reserved_upper_bound_by_kind': reserved_kind, 'reserved_upper_bound_by_scene': reserved_scene,
        'reserved_entries': pending, 'native_scientific_revalidation_performed': False,
        'main_T04_pool_frozen': False, 'scope': SCOPE}


def load_receipt(reference, root=ROOT):
    from src.tools.merge_pair_v2 import producer_arguments
    data, actual = audited_result(Path(root) / reference['file'])
    need(all(actual.get(k) == reference[k] for k in ('file', 'sha256')), 'Training count receipt changed')
    cfg, meta, metrics = data['config'], data['_meta'], data['metrics']
    argv = producer_arguments(meta['command'], MODULE)
    need(meta.get('end') and meta.get('exit_code') == 0 and metrics.get('producer_exit_code') == 0 and metrics.get('passed') is True
        and metrics.get('errors') == [] and metrics.get('kind') == VERSION
        and cfg.get('version') == VERSION and cfg.get('name') == meta['name'],
        'Training count receipt must be complete and successfully audited')
    need(argv == cfg['arguments'], 'Count receipt arguments changed')
    from src.tools.train_holdout_freeze import values
    need(values(argv, '--name') == [cfg['name']]
        and [json.loads(v) for v in values(argv, '--qualification')] == [cfg['qualification_reference']]
        and [json.loads(v) for v in values(argv, '--holdout')] == [cfg['holdout_reference']]
        and [json.loads(v) for v in values(argv, '--construction-source')] == cfg['construction_sources'],
        'Count receipt command does not bind its complete input cutoff')
    need([json.loads(v) for v in values(argv, '--known-rejection-source')] == cfg['known_rejection_sources'],
         'Count receipt command does not bind its original rejected-source cutoff')
    need(type(cfg.get('certificate_count_basis', False)) is bool
        and argv.count('--certificate-count-basis') == int(cfg.get('certificate_count_basis', False))
        and [json.loads(v) for v in values(argv, '--additional-cache-source')] == cfg.get('additional_cache_sources', []),
        'Count receipt additional cache authority/cutoff changed')
    proof = build_inventory(cfg['qualification_reference'], cfg['holdout_reference'], cfg['construction_sources'],
        known_rejection_sources=cfg['known_rejection_sources'],
        certificate_count_basis=cfg.get('certificate_count_basis', False),
        additional_cache_sources=cfg.get('additional_cache_sources', []), root=root)
    need(proof == metrics.get('inventory'), 'Count receipt differs from its original qualified metadata inputs')
    expected_sources = [proof['qualification_reference'], *proof['construction_publications'], *proof['rejection_publications'],
        *proof.get('additional_cache_publications', [])]
    freeze_ref = audited_result(Path(root) / 'results/train_holdout_freeze.json')[1]
    expected_sources.append(freeze_ref)
    need(data.get('sources') == expected_sources, 'Count receipt has lost its original source ledger')
    return proof, actual


def validate_plan_bindings(plan):
    need(plan.get('count_basis_authority') == AUTHORITY and plan.get('version') == quota.TRAIN_VERSION,
         'Training tickets need the exact postfilter receipt authority')
    quota.reference(plan.get('holdout_reference'))
    need(len(plan.get('count_basis_sources', [])) == 1, 'Training plan needs one complete fixed inventory cutoff')
    quota.by_scene(plan.get('construction_occupancy_by_scene'), 'train')


def authenticate_plan(plan):
    validate_plan_bindings(plan)
    proof, source = load_receipt(plan['count_basis_sources'][0])
    for key in ('holdout_reference', 'confirmed_by_kind', 'confirmed_by_scene', 'construction_occupancy_by_scene'):
        need(plan.get(key) == proof.get(key), 'Training ticket differs from its original count/holdout proof: ' + key)
    from src.tools.train_holdout_freeze import load_frozen_holdout
    frozen, _ = load_frozen_holdout(plan['holdout_reference'], root=ROOT)
    need(not any(row['scene'] in frozen['selected_scenes'] for row in plan['assignments']),
         'Training generation cannot enter any permanently held-out scene')
    expected = []
    if any(proof['reserved_upper_bound_by_kind'].values()):
        expected = [{'reservation_id': 'original_unfiltered_inventory', 'source': plan['count_basis_sources'][0],
            'scope': 'unverified_inventory_upper_bound', 'by_kind': proof['reserved_upper_bound_by_kind'],
            'by_scene': proof['reserved_upper_bound_by_scene']}]
    need(plan['outstanding_reservations'] == expected, 'Unqualified historical inventory must retain its full conservative reservation')
    need(not proof['needs_human_ratio_decision'], 'Both classes reached40 but ratio is outside bounds; stop and report for human decision')
    return {**proof, 'count_basis_authenticated': True, 'count_receipt_reference': source,
        'parent_native_revalidation_performed': False}


def main():
    import sys
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True); p.add_argument('--qualification', required=True)
    p.add_argument('--holdout', required=True); p.add_argument('--construction-source', action='append', required=True)
    p.add_argument('--known-rejection-source', action='append', default=[],
        help='Exact original complete single-pair T02 scientific rejection result; unknown failures remain reserved')
    p.add_argument('--certificate-count-basis', action='store_true',
        help='Use the approved original83 certificate population for generation counts only, never formal pool authorization')
    p.add_argument('--additional-cache-source', action='append', default=[],
        help='Complete original T02+cache receipt for newly collected pairs; holdout stays fixed')
    p.add_argument('--timeout-seconds', type=int, required=True)
    a = p.parse_args(); need(a.timeout_seconds > 0 and re.fullmatch(r'[A-Za-z0-9_]+', a.name), 'Bounded uniquely named metadata execution required')
    need(not (ROOT / 'results' / (a.name + '.json')).exists(), 'Preserve earlier count receipts')
    cfg = {'name': a.name, 'version': VERSION, 'arguments': sys.argv[1:], 'gpus': 0,
        'qualification_reference': json.loads(a.qualification), 'holdout_reference': json.loads(a.holdout),
        'construction_sources': [json.loads(v) for v in a.construction_source],
        'known_rejection_sources': [json.loads(v) for v in a.known_rejection_source],
        'certificate_count_basis': a.certificate_count_basis,
        'additional_cache_sources': [json.loads(v) for v in a.additional_cache_source]}
    metrics = {'kind': VERSION, 'passed': False, 'errors': [], 'native_scientific_revalidation_performed': False}
    def expired(*_): raise TimeoutError('Training count metadata deadline expired')
    old = signal.signal(signal.SIGALRM, expired)
    sources = []
    try:
        signal.setitimer(signal.ITIMER_REAL, a.timeout_seconds)
        metrics['inventory'] = build_inventory(cfg['qualification_reference'], cfg['holdout_reference'], cfg['construction_sources'],
            known_rejection_sources=cfg['known_rejection_sources'], certificate_count_basis=cfg['certificate_count_basis'],
            additional_cache_sources=cfg['additional_cache_sources'])
        sources = [metrics['inventory']['qualification_reference'], *metrics['inventory']['construction_publications'],
            *metrics['inventory']['rejection_publications'], *metrics['inventory']['additional_cache_publications'],
            audited_result(ROOT / 'results/train_holdout_freeze.json')[1]]
        metrics['passed'] = True
    except BaseException as error:
        traceback.print_exc(); metrics['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)
    metrics['producer_exit_code'] = 0 if metrics['passed'] else 1
    save_json(ROOT / 'results' / (a.name + '.json'), {'metrics': metrics, 'config': cfg, 'sources': sources})
    print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return 0 if metrics['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
