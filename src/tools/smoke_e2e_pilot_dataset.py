"""Fixed E2E-6 training-only pool; complete original formal-cache proofs required.

No renderer or model is started here. Real source/capture reauthentication can
be substantial CPU work and belongs in an explicitly bounded compute job.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import traceback

from src.envs.thor_common import ROOT
from src.tools.t01_partition import audited_result, canonical_hash
from src.tools.merge_pair_v2 import producer_arguments
from src.tools.t02_export import save_json

VERSION = 'smoke_e2e_pilot_formal_dataset_v2_keep_all'
KINDS = ('direct_flip', 'direct_stay', 'look_flip', 'look_stay')
SCENE_SPLIT = 'room_stratified_scene_hash_seed0_ten_percent_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ref_identity(ref):
    require(isinstance(ref, dict) and isinstance(ref.get('file'), str)
            and re.fullmatch(r'[0-9a-f]{64}', str(ref.get('sha256', ''))), 'Exact file/SHA reference required')
    return {key: ref[key] for key in ('file', 'sha256')}


def read_repo_ref(ref):
    ref = ref_identity(ref)
    path = (ROOT / ref['file']).resolve()
    path.relative_to(ROOT.resolve())
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Pinned repository artifact changed')
    return json.loads(raw)


def argv_values(argv, flag):
    return [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == flag]


def immutable_ledger(rows):
    """Fresh auditor annotations may change; original bytes and science may not."""
    require(isinstance(rows, list), 'Formal cache population ledger is missing')
    result = copy.deepcopy(rows)
    for row in result:
        if isinstance(row.get('source'), dict):
            row['source'].pop('audit', None)
    return result


def authenticate_cache_batch(expected):
    """Rebuild one entire original certificate, never concatenate sample plans."""
    from src.tools import t02_cache_check as cache
    from src.envs.pair_table_cache import sample_state_keys, validate_cache_sample
    from src.envs.pair_table_checks import authorize_tables, authenticate_cache_sources
    expected = ref_identity(expected)
    data, source = audited_result(ROOT / expected['file'])
    require(ref_identity(source) == expected, 'Formal cache result reference changed')
    meta, cfg, metrics = data['_meta'], data['config'], data['metrics']
    argv = producer_arguments(meta['command'], 'src.tools.t02_cache_check')
    name = cfg['name']
    require(cfg.get('version') == cache.VERSION and cfg.get('gpus') == 1
            and cfg.get('resolution') == 255 and cfg.get('seed') == 0
            and type(cfg.get('sample_size')) is int and cfg['sample_size'] >= 200
            and meta.get('name') == name and meta.get('end') and meta.get('exit_code') == 0
            and argv_values(argv, '--name') == [name] and argv_values(argv, '--source') == cfg.get('source')
            and metrics.get('kind') == 't02_formal_cache_check'
            and metrics.get('producer_exit_code') == 0 and metrics.get('errors') == []
            and metrics.get('cleanup_errors') == [] and metrics.get('black_frame_detected') is False
            and metrics.get('cache_matches_live') is True and metrics.get('downstream_allowed') is True
            and metrics.get('stop_downstream') is False and metrics.get('sample_complete') is True,
            'Require a successfully ended complete formal >=200-state cache run')
    for flag, key, default in (('--sample-size', 'sample_size', 200), ('--seed', 'seed', 0)):
        values = argv_values(argv, flag)
        require(values == [str(cfg[key])] or (not values and cfg[key] == default), 'Cache sample command/config differs')
    job = cfg.get('pbs_jobid', '')
    require(re.fullmatch(r'\d+\.hopper-m-\d+', job), 'Formal cache PBS identity missing')
    directory = Path(cfg['scratch']).resolve() / 't02_cache' / name / job
    require(Path(cfg['scratch_directory']).resolve() == directory, 'Formal cache private directory changed')
    gate, gate_ref = audited_result(ROOT / metrics['gate']['file'])
    require(ref_identity(gate_ref) == ref_identity(metrics['gate']), 'Formal gate reference changed')
    cache.gate_runtime(gate, cfg, cfg['node'], directory / 'unity_logs')
    tables, packages, checks, export_refs, ledger = cache.load_exports(cfg['source'], cfg['scratch'])
    require(immutable_ledger(ledger) == immutable_ledger(metrics['population_ledger']),
            'Formal cache population ledger changed')
    population = [(key, state) for key, table in tables.items() for state in table.states]
    plan = sample_state_keys(population, seed=cfg['seed'], sample_size=cfg['sample_size'])
    require(plan == cache.read_json_ref(metrics['sampling_plan'], directory, expected=directory / 'sampling_plan.json')
            and cfg.get('population_sha256') == plan['population_sha256'], 'Original complete sample population differs')
    groups = {}
    for key, state in plan['selected_keys']:
        groups.setdefault(key, []).append(state)
    rows = metrics['worker_runs']
    require(isinstance(rows, list) and [r.get('dataset_key') for r in rows] == list(groups),
            'Formal cache omitted, duplicated or reordered a sampled pair worker')
    samples, authenticated = [], set()
    cached_processes = {capture.process_id for table in tables.values()
                        for state in table.states.values() for capture in state.captures}
    for index, (key, state_ids) in enumerate(groups.items()):
        row, table, package = rows[index], tables[key], packages[key]
        pair_dir = directory / f'pair_{index:04d}'
        require(row.get('status') == 'captured' and row.get('error') is None
                and row.get('cleanup_errors') == [] and not row.get('raw_index_error')
                and not row.get('black_frame_reference') and not row.get('native_authentication_error')
                and Path(row['scratch_directory']).resolve() == pair_dir,
                'Fresh cache worker is incomplete or cleanup is unknown')
        stored = cache.read_json_ref(row['report_reference'], pair_dir, expected=pair_dir / 'worker_report.json')
        require(all(row.get(k) == value for k, value in stored.items()), 'Immutable fresh worker report changed')
        cache.read_json_ref(row['raw_index'], pair_dir, expected=pair_dir / 'raw/evidence_index.json')
        roots = [*package['cache_evidence_roots'], str(pair_dir)]
        observed = cache.load_samples(table, row, pair_dir, roots, state_ids, use_dataset_key=True)
        require(not any(c.process_id in cached_processes for sample in observed for c in sample.fresh),
                'Cache fresh process reused an original export process')
        identities = authenticate_cache_sources([table], observed, evidence_roots=roots)
        samples.extend(observed)
        authenticated.update(identities)
    actual = validate_cache_sample(samples, population=population, sampling_plan=plan)
    actual['table_shadow_metadata'] = {key: table.shadow_metadata for key, table in tables.items()}
    actual['native_authenticated_render_ids'] = [list(key) for key in sorted(authenticated)]
    actual['producer_cleanup_errors'], actual['producer_errors'] = [], []
    authorize_tables(list(tables.values()), pair_checks=checks, cache_report=actual,
                     authenticated_cache_source_ids=authenticated)
    actual['table_native_and_pair_checks_authorized'] = True
    recorded = cache.read_json_ref(metrics['validation_report'], directory, expected=directory / 'cache_validation.json')
    require(canonical_hash(actual) == canonical_hash(recorded), 'Full reconstructed cache report changed')
    certificate = cache.read_json_ref(metrics['authorization_manifest'], directory,
                                     expected=directory / 'cache_authorization.json')
    require(certificate.get('version') == cache.VERSION
            and certificate.get('tables') == {key: table.manifest_reference for key, table in tables.items()}
            and certificate.get('table_shadow_metadata') == actual['table_shadow_metadata']
            and certificate.get('cache_validation') == metrics['validation_report']
            and certificate.get('sampling_plan') == metrics['sampling_plan']
            and certificate.get('full_T02_acceptance') is None,
            'Cache authorization does not cover its exact original table population')
    proof = {'source': source, 'authorization_manifest': copy.deepcopy(metrics['authorization_manifest']),
        'validation_report': copy.deepcopy(metrics['validation_report']), 'sampling_plan': copy.deepcopy(metrics['sampling_plan']),
        'dataset_keys': sorted(tables), 'source_export_results': export_refs}
    return tables, packages, proof


def room_type(scene):
    match = re.fullmatch(r'FloorPlan(\d+)', scene)
    require(match is not None, 'Unknown scene')
    number = int(match[1])
    for base, room in ((0, 'kitchen'), (200, 'living'), (300, 'bedroom'), (400, 'bathroom')):
        if base + 1 <= number <= base + 20:
            return room
    raise ValueError('Pilot cannot use any test/validation scene')


def split_entries(entries, policy):
    """Approved T04 scene split, fixed before any training result is observed."""
    require(policy == SCENE_SPLIT, 'Only the human-approved T04 scene holdout is permitted')
    keys = [entry['dataset_key'] for entry in entries]
    proposals = [entry['proposal_identity'] for entry in entries]
    require(len(keys) == len(set(keys)) and len(proposals) == len(set(proposals)), 'Duplicate source/proposal in pilot split')
    for entry in entries:
        room_type(entry['scene'])
    rooms = defaultdict(set)
    for entry in entries:
        rooms[room_type(entry['scene'])].add(entry['scene'])
    scenes = set()
    for room, choices in sorted(rooms.items()):
        ordered = sorted(choices, key=lambda scene: (canonical_hash({'seed': 0, 'scene': scene, 'room': room}), scene))
        scenes.update(ordered[:max(1, math.ceil(len(ordered) / 10))])
    selected = {row['dataset_key'] for row in entries if row['scene'] in scenes}
    detail = {'holdout_fraction_requested': .1, 'selected_scenes': sorted(scenes),
              'selection': 'per present room type ceil(scene_count/10), at least one whole scene'}
    require(selected and set(keys) - selected, 'Pilot split must have nonempty training and evaluation portions')
    return {'policy': policy, 'seed': 0, **detail,
        'train_keys': sorted(set(keys) - selected), 'eval_keys': sorted(selected),
        'scope': 'fixed T04 scene holdout shared by pilot, every-20-update evaluation and S2; main training pool remains unfrozen'}


def keep_all_with_flip_order(rows):
    """Pilot-only composition: original Flip swap, every Stay retained.

    The approved7babc169 qualification is cache>=80 and all four classes.
    Original T02 balance statistics remain diagnostics; its formal downsampling
    function and all source tables are untouched.
    """
    from src.envs.pair_table_balance import checked_rows, seeded_digest, balance_statistics, VERSION as FLIP_VERSION
    from src.envs.pair_table_identity import population_key
    original = checked_rows(rows)
    selected, measured = [], []
    for row in sorted(original, key=lambda value: (value['scene'], value['kind'], population_key(value))):
        assignment = seeded_digest(row['pair_id'], 0, 'flip_world_swap') if row['kind'].endswith('flip') else None
        swapped = bool(int(assignment[:2], 16) & 1) if assignment is not None else False
        order = [1, 0] if swapped else [0, 1]
        mapped = {**copy.deepcopy(row), 'target_names': [row['target_names'][index] for index in order],
            's0_target_left': [row['s0_target_left'][index] for index in order]}
        measured.append(mapped)
        identity = {key: copy.deepcopy(row[key]) for key in ('dataset_key', 'original_pair_id', 'dataset_identity') if key in row}
        selected.append({'pair_id': row['pair_id'], 'scene': row['scene'], 'kind': row['kind'], **identity,
            'world_order': order, 'swapped': swapped, 'assignment_sha256': assignment,
            'candidate_names': copy.deepcopy(row['candidate_names']),
            'original_target_names': copy.deepcopy(row['target_names']), 'target_names': mapped['target_names'],
            'original_s0_target_left': copy.deepcopy(row['s0_target_left']), 's0_target_left': mapped['s0_target_left']})
    scenes = [row['scene'] for row in original]
    return {'version': 'pilot_keep_all_flip_swap_v1', 'approved_task_revision': '7babc169', 'seed': 0,
        'policy': {'retain_every_authenticated_input_pair': True, 'stay_downsampling_applied': False,
            'flip': 'unchanged original SHA256 low-bit world swap', 'flip_seed_version': FLIP_VERSION,
            'global_balance_is_qualification_gate': False,
            'scope': 'pilot composition only; original T04 balancing remains separate'},
        'selected': selected, 'dropped': [],
        'input_count_by_class': dict(Counter(row['kind'] for row in original)),
        'selected_count_by_class': dict(Counter(row['kind'] for row in measured)),
        'before': balance_statistics(original, scenes), 'after': balance_statistics(measured, scenes),
        'source_tables_modified': False, 'selection_independent_of_left_measurement': True,
        'full_T02_acceptance': None}


def build_pool(cache_sources, split_policy):
    from src.envs.pair_table_balance import normalize_table
    require(isinstance(cache_sources, list) and cache_sources, 'Explicit original cache certificate sources required')
    tables, packages, cache_proofs, seen_proposals = {}, {}, [], {}
    for ref in cache_sources:
        batch, source_packages, proof = authenticate_cache_batch(ref)
        cache_proofs.append(proof)
        for key, table in batch.items():
            package = source_packages[key]
            identity = package['independent_verification']['proposal_identity']
            require(key not in tables and identity not in seen_proposals, 'Repeated pair source/proposal across explicit cache batches')
            room_type(table.scene)
            require(table.kind in KINDS and table.source.get('field_of_view') == 60
                    and not table.synthetic and table.export_verified and table.cache_verified,
                    'Pilot needs real current-camera native/pair/formal-cache-verified training tables')
            tables[key], packages[key], seen_proposals[identity] = table, package, key
    balance = keep_all_with_flip_order([normalize_table(table) for table in tables.values()])
    selected = {row['dataset_key']: row for row in balance['selected']}
    counts = {kind: sum(tables[key].kind == kind for key in selected) for kind in KINDS}
    require(len(selected) >= 80 and all(counts.values()), 'Pilot needs at least 80 unique cache-verified pairs and all four classes')
    entries = []
    for key in sorted(selected):
        table, package = tables[key], packages[key]
        entries.append({'dataset_key': key, 'original_pair_id': table.pair_id,
            'proposal_identity': package['independent_verification']['proposal_identity'],
            'scene': table.scene, 'kind': table.kind,
            'pair_reference': copy.deepcopy(package['source_provenance']['pair']),
            'table_manifest': copy.deepcopy(table.manifest_reference),
            'world_order': selected[key]['world_order'], **table.shadow_metadata})
    split = split_entries(entries, split_policy)
    manifest = {'version': VERSION, 'partition': 'train', 'field_of_view': 60, 'seed': 0,
        'cache_sources': [ref_identity(ref) for ref in cache_sources], 'entries': entries,
        'counts_by_kind': counts, 'balance': balance, 'split': split,
        'paper_result': False, 'main_T04_pool_frozen': False,
        'throughput_20x_model': None, 'formal_full_T02_acceptance': None}
    manifest['common_pool_hash'] = canonical_hash(manifest)
    proof = {'version': VERSION, 'common_pool_hash': manifest['common_pool_hash'],
        'train_keys': split['train_keys'], 'eval_keys': split['eval_keys'], 'split_policy': split_policy,
        'split': copy.deepcopy(split),
        'world_orders': {key: selected[key]['world_order'] for key in selected},
        'counts_by_kind': counts, 'cache_authorized': True, 'formal_cache_batches': cache_proofs,
        'sources': [proof['source'] for proof in cache_proofs], 'paper_result': False,
        'main_T04_pool_frozen': False, 'formal_full_T02_acceptance': None}
    return manifest, {key: tables[key] for key in selected}, {key: packages[key] for key in selected}, proof


def load_pilot_pool(manifest_reference):
    recorded = read_repo_ref(manifest_reference)
    require(recorded.get('version') == VERSION and recorded.get('partition') == 'train', 'Unknown pilot dataset schema')
    manifest, tables, packages, proof = build_pool(recorded['cache_sources'], recorded['split']['policy'])
    require(manifest == recorded, 'Pilot pool/split differs from its fully authenticated frozen manifest')
    proof['manifest_reference'] = ref_identity(manifest_reference)
    return tables, packages, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--cache-source', action='append', required=True, help='JSON {file,sha256} reference')
    parser.add_argument('--split-policy', choices=(SCENE_SPLIT,), default=SCENE_SPLIT)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=float, required=True)
    args = parser.parse_args()
    require(re.fullmatch(r'smoke_e2e_pilot_[A-Za-z0-9_]+', args.name)
            and math.isfinite(args.timeout_seconds) and args.timeout_seconds > 0, 'Unique bounded pilot preflight required')
    output = ROOT / 'results' / (args.name + '.json')
    require(not output.exists(), 'Preserve previous preflights')
    destination = (ROOT / args.out).resolve(); destination.relative_to(ROOT / 'data/world_pairs')
    require(not destination.exists(), 'Pilot dataset manifest must not be overwritten')
    config = {**vars(args), 'out': str(args.out), 'version': VERSION,
              'cache_source': [json.loads(value) for value in args.cache_source], 'gpus': 0}
    metrics = {'kind': 'smoke_e2e_pilot_dataset_preflight', 'passed': False, 'errors': [],
               'manifest': None, 'paper_result': False, 'formal_T04_acceptance': None}
    code, sources = 1, []
    def expired(*_): raise TimeoutError('Pilot dataset authentication deadline reached')
    previous = signal.signal(signal.SIGALRM, expired)
    try:
        require(re.fullmatch(r'\d+\.hopper-m-\d+', os.environ.get('PBS_JOBID', ''))
                and os.environ.get('CVPR_SCRATCH'),
                'Complete multi-table authentication requires a bounded PBS allocation, not the login node')
        config['pbs_jobid'] = os.environ['PBS_JOBID']
        config['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
        manifest, _, _, proof = build_pool(config['cache_source'], args.split_policy)
        destination.parent.mkdir(parents=True, exist_ok=True)
        reference = save_json(destination, manifest)
        metrics.update(passed=True, manifest=reference, dataset_proof=proof)
        sources = proof['sources']; code = 0
    except BaseException as error:
        traceback.print_exc(); metrics['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
        metrics['producer_exit_code'] = code
        output.parent.mkdir(parents=True, exist_ok=True)
        save_json(output, {'metrics': metrics, 'config': config, 'sources': sources})
        print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
