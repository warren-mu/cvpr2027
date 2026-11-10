"""55753931/96fbdd1f pilot exception: reuse original certificates, not repeat native I/O.

This mode never certifies that the extra 709593 source reauthentication passed.
Actual cached-table bytes and all reused certificate identities remain checked.
"""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import re

from src.envs.thor_common import ROOT
from src.tools.t01_partition import audited_result, canonical_hash
from src.tools.merge_pair_v2 import producer_arguments

MODE = 'approved_original_certificates83_after02_v1'
LEGACY_POLICY_REFERENCE = {'file': 'configs/smoke_e2e_pilot_certificates83.json',
    'sha256': '3604c54f84e8f0cb722c93e745f2db4e3422233f28581cad7780ac8812fe5449'}
POLICY_REFERENCE = {'file': 'configs/smoke_e2e_pilot_certificates83_immediate.json',
    'sha256': '68ccb921104bbf20b982a92272e68fba089cb6e39d26c7daa7bd2bdcbd29dc99'}
IMMEDIATE_REVISION = '96fbdd1f'
NAME = 'smoke_e2e_pilot_pool_83_certificates_01'
RESULT_FILE = 'results/' + NAME + '.json'
MANIFEST_FILE = 'data/world_pairs/e2e_pilot_pool_83_certificates_01.json'
EXTRA = {'status': 'not_awaited', 'passed': None}


def need(condition, message):
    if not condition: raise ValueError(message)


def identity(ref):
    need(isinstance(ref, dict) and isinstance(ref.get('file'), str)
        and re.fullmatch(r'[0-9a-f]{64}', str(ref.get('sha256'))), 'Exact original file/SHA required')
    return {key: ref[key] for key in ('file', 'sha256')}


def read(ref, *, root=ROOT, within=None):
    path = (Path(root) / identity(ref)['file']).resolve()
    path.relative_to(Path(within or root).resolve())
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == ref['sha256']
        and ('bytes' not in ref or len(raw) == ref['bytes']), 'Certificate/metadata bytes changed: ' + str(path))
    return json.loads(raw)


def policy(root=ROOT, *, legacy=False):
    value = read(LEGACY_POLICY_REFERENCE if legacy else POLICY_REFERENCE, root=root)
    need(value['version'] == MODE and value['exact_cached_pairs'] == 83
        and value['cache_batch_sizes'] == [5, 24, 23, 31]
        and value['qualification_name'] == NAME and value['manifest_file'] == MANIFEST_FILE,
        'Unknown fixed83 certificate exception')
    if legacy:
        need(value['approved_task_revision'] == '55753931'
            and value['not_before_utc'] == '2026-11-10T07:00:00+00:00', 'Unknown historical authorization')
    else:
        need(value['approved_task_revision'] == IMMEDIATE_REVISION
            and value['activation_condition'] == 'original709593_ended_nonpass_no_clock_wait'
            and value['extra_qualification_job'] == '709593.hopper-m-02'
            and value['extra_qualification_reference']['file'] == value['extra_qualification_result']
            and value['extra_qualification_exit_code'] == 1 and bool(value['extra_qualification_end']),
            'Unknown immediate authorization')
    return value


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def activate(cache_sources, *, root=ROOT):
    """New96 activation binds the already-audited ended failure; no clock wait."""
    p = policy(root); now = utc_now()
    need(now.tzinfo is not None, 'Activation timestamp needs its explicit UTC offset')
    need([identity(ref) for ref in cache_sources] == p['cache_sources'], 'Certificate mode requires the fixed original four cache sources')
    # This pinned result is the existing original709593 evidence, not a new
    # reauthentication. read() checks its exact original SHA before metadata.
    data = read(p['extra_qualification_reference'], root=root)
    meta, cfg, metrics = data['_meta'], data['config'], data['metrics']
    need(meta.get('name') == 'smoke_e2e_pilot_pool_83_01'
        and cfg.get('pbs_jobid') == p['extra_qualification_job']
        and meta.get('end') == p['extra_qualification_end']
        and meta.get('exit_code') == metrics.get('producer_exit_code') == p['extra_qualification_exit_code']
        and metrics.get('passed') is False and metrics.get('manifest') is None,
        'Immediate activation requires original709593 ended/nonPASS; missing, running or PASS cannot activate')
    snapshot = {'file': p['extra_qualification_result'], 'exists': True,
        'sha256': p['extra_qualification_reference']['sha256'], 'observed_passed': False,
        'observed_end': meta['end'], 'observed_exit_code': meta['exit_code'],
        'pbs_jobid': cfg['pbs_jobid'], 'observed_producer_exit_code': metrics['producer_exit_code']}
    return {'checked_at_utc': now.isoformat(), 'extra_qualification_snapshot': snapshot,
        'authorization_revision': IMMEDIATE_REVISION}


def qualification(activation, cache_sources, *, root=ROOT):
    """Validate the frozen initial decision; never consult a later709593 state."""
    revision = activation.get('authorization_revision')
    need(revision in (None, IMMEDIATE_REVISION), 'Unknown activation authorization')
    legacy = revision is None
    p = policy(root, legacy=legacy)
    need([identity(ref) for ref in cache_sources] == p['cache_sources'], 'Frozen certificate source set changed')
    observed = datetime.datetime.fromisoformat(activation['checked_at_utc'])
    snapshot = activation['extra_qualification_snapshot']
    need(observed.tzinfo is not None and snapshot.get('file') == p['extra_qualification_result']
        and type(snapshot.get('exists')) is bool and snapshot.get('observed_passed') is not True,
        'Invalid original certificate-mode activation')
    if legacy:
        need(observed >= datetime.datetime.fromisoformat(p['not_before_utc']), 'Historical activation predates its original07UTC authorization')
        need((snapshot['exists'] and re.fullmatch(r'[0-9a-f]{64}', str(snapshot.get('sha256'))))
            or (not snapshot['exists'] and all(snapshot.get(k) is None for k in
                ('sha256', 'observed_passed', 'observed_end', 'observed_exit_code'))), 'Invalid original extra-run snapshot')
    else:
        need(snapshot['exists'] is True and snapshot.get('sha256') == p['extra_qualification_reference']['sha256']
            and snapshot.get('pbs_jobid') == p['extra_qualification_job']
            and snapshot.get('observed_end') == p['extra_qualification_end']
            and snapshot.get('observed_exit_code') == snapshot.get('observed_producer_exit_code') == p['extra_qualification_exit_code']
            and snapshot.get('observed_passed') is False, 'Original ended709593 activation binding changed')
    result = {'mode': MODE, 'policy_reference': copy.deepcopy(LEGACY_POLICY_REFERENCE if legacy else POLICY_REFERENCE),
        'activation': copy.deepcopy(activation), 'exact_cached_pairs': 83, 'retain_every_authenticated_input_pair': True,
        'extra_source_reauthentication': copy.deepcopy(EXTRA),
        'native_source_reauthentication_performed_here': False,
        'scope': 'Original T02 and formal cache certificates reused under55753931; extra709593 outcome not claimed'}
    if not legacy:
        result['authorization_revision'] = IMMEDIATE_REVISION
    return result


def _audited(ref, root):
    data, actual = audited_result(Path(root) / identity(ref)['file'])
    need(identity(actual) == identity(ref), 'Original logged certificate/result changed')
    return data, actual


def certificate_metadata(expected, *, root=ROOT):
    """Audit original whole-batch certificate metadata, without native/capture I/O."""
    from src.tools import t02_cache_check as cache
    from src.tools.smoke_e2e_pilot_dataset import argv_values
    from src.envs.pair_table_cache import require_cache_validation
    data, source = _audited(expected, root)
    cfg, meta, metrics = data['config'], data['_meta'], data['metrics']
    argv = producer_arguments(meta['command'], 'src.tools.t02_cache_check')
    name = cfg['name']
    need(cfg.get('version') == cache.VERSION and cfg.get('gpus') == 1 and cfg.get('resolution') == 255
        and cfg.get('seed') == 0 and type(cfg.get('sample_size')) is int and cfg['sample_size'] >= 200
        and meta.get('name') == name and meta.get('end') and meta.get('exit_code') == 0
        and argv_values(argv, '--name') == [name] and argv_values(argv, '--source') == cfg.get('source')
        and metrics.get('kind') == 't02_formal_cache_check' and metrics.get('producer_exit_code') == 0
        and metrics.get('errors') == [] and metrics.get('cleanup_errors') == []
        and metrics.get('black_frame_detected') is False and metrics.get('sample_complete') is True
        and metrics.get('cache_matches_live') is True and metrics.get('downstream_allowed') is True
        and metrics.get('stop_downstream') is False, 'Original formal cache certificate is failed or incomplete')
    for flag, key, default in (('--sample-size', 'sample_size', 200), ('--seed', 'seed', 0)):
        values = argv_values(argv, flag)
        need(values == [str(cfg[key])] or (not values and cfg[key] == default), 'Original cache command/sample differs')
    job = cfg.get('pbs_jobid', '')
    need(re.fullmatch(r'\d+\.hopper-m-\d+', job), 'Original cache PBS identity missing')
    directory = Path(cfg['scratch']).resolve() / 't02_cache' / name / job
    need(Path(cfg['scratch_directory']).resolve() == directory, 'Original certificate private directory changed')
    gate, gate_ref = _audited(metrics['gate'], root)
    cache.gate_runtime(gate, cfg, cfg['node'], directory / 'unity_logs')
    report = cache.read_json_ref(metrics['validation_report'], directory, expected=directory / 'cache_validation.json')
    plan = cache.read_json_ref(metrics['sampling_plan'], directory, expected=directory / 'sampling_plan.json')
    cert = cache.read_json_ref(metrics['authorization_manifest'], directory, expected=directory / 'cache_authorization.json')
    require_cache_validation(report)
    need(report.get('table_native_and_pair_checks_authorized') is True
        and report.get('producer_errors') == [] and report.get('producer_cleanup_errors') == []
        and report['sampling_plan'] == plan and cfg['population_sha256'] == plan['population_sha256']
        and plan['seed'] == cfg['seed'] and plan['sample_size'] == cfg['sample_size']
        and cert.get('version') == cache.VERSION and cert.get('cache_validation') == metrics['validation_report']
        and cert.get('sampling_plan') == metrics['sampling_plan'] and cert.get('full_T02_acceptance') is None,
        'Whole-cache report/certificate/sample identities disagree')
    selected = [tuple(key) for key in plan['selected_keys']]
    measured = [(row['dataset_key'], row['state_id']) for row in report['states']]
    need(len(set(selected)) == len(selected) == len(measured) == cfg['sample_size']
        and set(measured) == set(selected), 'Original cache sample is not complete')
    authenticated = {(row['dataset_key'], row['state_id'], world, row['worlds'][world]['fresh_source']['render_id'])
        for row in report['states'] for world in ('w0', 'w1')}
    need(authenticated == {tuple(item) for item in report['native_authenticated_render_ids']},
        'Original native authentication IDs do not bind the full sample')
    ledger = metrics['population_ledger']
    complete = [row for row in ledger if row.get('population_membership') == 'complete_independently_verified_table']
    need(len({row['dataset_key'] for row in complete}) == len(complete)
        and list(dict.fromkeys(row['source']['file'] for row in ledger)) == cfg['source']
        and cert['tables'] == {row['dataset_key']: {k: row['table_manifest'][k] for k in ('file', 'sha256', 'bytes')} for row in complete}
        and all(row in complete or (row.get('population_membership') == 'excluded_before_cache_by_export_checks'
            and row.get('export_status') == 'rejected' and row.get('export_rejection_reasons')) for row in ledger)
        and report['table_shadow_metadata'] == cert['table_shadow_metadata'] == metrics['table_shadow_metadata'],
        'Original complete population is missing, duplicated or changed')
    return {'data': data, 'source': source, 'certificate': cert, 'report': report, 'plan': plan,
        'complete': complete, 'authenticated_ids': authenticated, 'directory': directory}


def load_certificate_batch(expected, *, root=ROOT):
    """Read SHA-verified cached tables and reuse their authentic original checks."""
    from src.tools import t02_cache_check as cache, t02_export as export
    from src.envs.pair_table_export import load_table, FROZEN_RENDER_VERSION
    from src.envs.pair_table_checks import check_pair, authorize_tables, verify_shadow_metadata
    evidence = certificate_metadata(expected, root=root)
    tables, packages, checks, original_results = {}, {}, {}, {}
    for entry in evidence['complete']:
        ref = entry['source']; key = entry['dataset_key']
        if ref['file'] not in original_results:
            original_results[ref['file']] = _audited(ref, root)
        original, actual_ref = original_results[ref['file']]
        need(identity(actual_ref) == identity(ref), 'Same export path has conflicting original bytes')
        cfg, meta, metrics = original['config'], original['_meta'], original['metrics']
        cache.exact_export_command(meta['command'], cfg['name'])
        need(meta.get('end') and meta.get('exit_code') == 0 and cfg.get('version') == export.VERSION
            and cfg.get('gpus') == 1 and cfg.get('depth') == 4 and metrics.get('kind') == 't02_table_export'
            and metrics.get('producer_exit_code') == 0 and metrics.get('export_checks_complete') is True
            and metrics.get('errors') == [] and metrics.get('cleanup_errors') == []
            and metrics.get('unstarted_pair_ids') == [] and metrics.get('black_frame_detected') is False,
            'Original T02 certificate source is failed or incomplete')
        rows = metrics['pairs']
        need([r['pair_id'] for r in rows] == cfg['pair_id'] and len(set(cfg['pair_id'])) == len(rows), 'Original export inventory changed')
        matches = [(i, row) for i, row in enumerate(rows) if row['pair_id'] == entry['pair_id']]
        need(len(matches) == 1, 'Original table source row is ambiguous')
        index, row = matches[0]
        directory = Path(cfg['scratch']).resolve() / 't02_export' / cfg['name'] / cfg['pbs_jobid']
        pair_dir = directory / f'pair_{index:04d}'
        need(Path(cfg['scratch_directory']).resolve() == directory and Path(row['scratch_directory']).resolve() == pair_dir
            and row.get('status') == 'exported' and row.get('table_complete') is True
            and row.get('native_evidence_verified') is True and row.get('pair_checks_passed') is True
            and row.get('cleanup_errors') == [] and row.get('error') is None and not row.get('raw_index_error')
            and row['table_manifest'] == entry['table_manifest'], 'Original T02 table was not fully accepted')
        package = cache.read_json_ref(row['source_package'], pair_dir, expected=pair_dir / 'authenticated_source.json')
        details = cache.read_json_ref(row['detail'], pair_dir, expected=pair_dir / 'checks.json')
        table_ref = evidence['certificate']['tables'][key]
        need(details['table_manifest'] == row['table_manifest'] and details['native_checks'].get('passed') is True
            and details['pair_checks'].get('passed') is True and details['pair_checks'].get('synthetic_only') is False,
            'Original pair/native check receipt is incomplete')
        table = load_table(table_ref['file'], expected_sha256=table_ref['sha256'])
        need(table.manifest_reference == table_ref and table.dataset_key == key and table.pair_id == entry['pair_id']
            and table.source.get('t01') == package['source_provenance']
            and table.source.get('export_version') == export.VERSION and table.source.get('frozen_render_version') == FROZEN_RENDER_VERSION
            and table.source.get('field_of_view') == 60 and package['source_config'].get('field_of_view') == 60
            and not table.synthetic and table.kind in ('direct_flip', 'direct_stay', 'look_flip', 'look_stay')
            and all(getattr(table, k) == package[k] for k in ('pair_id', 'scene', 'kind', 'instruction', 'anchor_name'))
            and list(table.candidate_names) == package['candidate_names'] and list(table.target_names) == package['target_names']
            and package['independent_verification']['validity']['acceptance_passed'] is True,
            'Cached table bytes do not belong to the original certified pair')
        for declared in (row, details, details['native_checks'], details['pair_checks'], evidence['certificate']['table_shadow_metadata'][key]):
            verify_shadow_metadata(table, declared)
        # This flag derives from the original audited T02 receipt, not a new
        # verify_render_sources call. Keep the authority visible to consumers.
        table.export_verified = True
        table.pilot_certificate_authority = {'mode': MODE, 'source': identity(evidence['source']),
            'extra_source_reauthentication': copy.deepcopy(EXTRA)}
        current = check_pair(table, package['t01'])
        need(current.get('passed') is True, 'SHA-verified cached table no longer satisfies unchanged pair checks')
        tables[key], packages[key], checks[key] = table, package, current
    authorize_tables(list(tables.values()), pair_checks=checks, cache_report=evidence['report'],
        authenticated_cache_source_ids=evidence['authenticated_ids'])
    metrics = evidence['data']['metrics']
    proof = {'source': evidence['source'], 'authorization_manifest': copy.deepcopy(metrics['authorization_manifest']),
        'validation_report': copy.deepcopy(metrics['validation_report']), 'sampling_plan': copy.deepcopy(metrics['sampling_plan']),
        'dataset_keys': sorted(tables), 'source_export_results': [r[1] for r in original_results.values()],
        'authority': MODE, 'extra_source_reauthentication': copy.deepcopy(EXTRA)}
    return tables, packages, proof


def authenticate_certificate_qualification(reference, *, root=ROOT):
    """Metadata-only successful population receipt binding for permanent holdout."""
    from src.tools import smoke_e2e_pilot_dataset as dataset
    need(identity(reference)['file'] == RESULT_FILE, 'Use the fixed certificate83 population receipt')
    data, source = _audited(reference, root)
    cfg, meta, metrics = data['config'], data['_meta'], data['metrics']
    argv = producer_arguments(meta['command'], 'src.tools.smoke_e2e_pilot_dataset')
    p = policy(root)
    need(meta.get('end') and meta.get('exit_code') == 0 and meta.get('name') == cfg.get('name') == NAME
        and cfg.get('version') == dataset.VERSION and cfg.get('certificates_after_02') is True
        and cfg.get('deadline_fallback_52') is False and argv.count('--certificates-after-02') == 1
        and '--deadline-fallback-52' not in argv and cfg.get('cache_source') == p['cache_sources']
        and [json.loads(v) for v in dataset.argv_values(argv, '--cache-source')] == p['cache_sources']
        and dataset.argv_values(argv, '--name') == [NAME] and dataset.argv_values(argv, '--out') == [MANIFEST_FILE]
        and cfg.get('out') == MANIFEST_FILE and cfg.get('split_policy') == dataset.SCENE_SPLIT
        and metrics.get('kind') == 'smoke_e2e_pilot_dataset_preflight' and metrics.get('passed') is True
        and metrics.get('producer_exit_code') == 0 and metrics.get('errors') == [],
        'Certificate population is missing, failed, unfinished or belongs to another mode')
    manifest_ref = identity(metrics['manifest'])
    need((Path(root) / manifest_ref['file']).resolve() == (Path(root) / MANIFEST_FILE).resolve(),
         'Fixed certificate83 manifest path changed')
    manifest = read(manifest_ref, root=root); proof = metrics['dataset_proof']
    q = qualification(manifest['qualification']['activation'], p['cache_sources'], root=root)
    unsigned = copy.deepcopy(manifest); unsigned.pop('common_pool_hash', None)
    entries = manifest['entries']; keys = [entry['dataset_key'] for entry in entries]
    split = dataset.split_entries(entries, dataset.SCENE_SPLIT)
    counts = {kind: sum(entry['kind'] == kind for entry in entries) for kind in dataset.KINDS}
    need(len(keys) == len(set(keys)) == 83 and all(counts.values()) and manifest.get('version') == dataset.VERSION
        and manifest.get('partition') == 'train' and manifest.get('field_of_view') == 60 and manifest.get('seed') == 0
        and manifest.get('cache_sources') == p['cache_sources'] and manifest.get('qualification') == proof.get('qualification') == q
        and manifest.get('common_pool_hash') == proof.get('common_pool_hash') == canonical_hash(unsigned)
        and manifest.get('counts_by_kind') == proof.get('counts_by_kind') == counts
        and manifest.get('split') == proof.get('split') == split and proof.get('cache_authorized') is True
        and proof.get('train_keys') == split['train_keys'] and proof.get('eval_keys') == split['eval_keys']
        and proof.get('world_orders') == {row['dataset_key']: row['world_order'] for row in entries},
        'Frozen certificate83 population, qualification or holdout changed')
    actual_keys = []; by_key = {row['dataset_key']: row for row in entries}
    need(len(proof['formal_cache_batches']) == 4, 'Missing original batch certificate')
    for expected, batch_proof, count in zip(p['cache_sources'], proof['formal_cache_batches'], p['cache_batch_sizes']):
        evidence = certificate_metadata(expected, root=root)
        table_refs = evidence['certificate']['tables']; actual_keys.extend(table_refs)
        need(len(table_refs) == count and identity(batch_proof['source']) == expected
            and batch_proof.get('dataset_keys') == sorted(table_refs)
            and all(identity(batch_proof[field]) == identity(evidence['data']['metrics'][field]) for field in
                ('authorization_manifest', 'validation_report', 'sampling_plan'))
            and all(by_key[key]['table_manifest'] == ref for key, ref in table_refs.items()),
            'Certificate population differs from its original exact four successful batches')
    need(len(actual_keys) == len(set(actual_keys)) == 83 and set(actual_keys) == set(keys)
        and [identity(ref) for ref in data['sources']] == [identity(ref) for ref in proof['sources']] == p['cache_sources'],
        'Certificate83 omitted, replaced or duplicated a source identity')
    return manifest, proof, source, manifest_ref
