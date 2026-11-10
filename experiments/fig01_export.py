"""FIG01: bounded CPU export of original, cache-authorized LookFlip pixels.

No renderer or native replay. Run only after both pilot arms have started.
This figure selection does not change any research population or acceptance.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback

from src.envs.thor_common import ROOT
from src.envs.pair_table import TableEnvironment, TableObservationBackend
from src.envs.pair_table_checks import check_pair, verify_cached_binding, verify_shadow_metadata
from src.envs.pair_table_cache import sample_state_keys, require_cache_validation
from src.envs.pair_table_export import load_table, FROZEN_RENDER_VERSION
from src.envs.pair_table_identity import identity_fields, population_key
from src.tools import train_holdout_freeze as holdout
from src.tools import smoke_e2e_pilot_certificates as certificates
from src.tools.smoke_e2e_pilot_certificates import authenticate_certificate_qualification
from src.tools.t02_export import save_json

VERSION = 'fig01_original_cache_frames_firststep_v3'
DESTINATION = 'paper/fig_src/lookflip_frames'
PREFIX_SOURCE = 'original first_step_visible_actions sorted first'
RANKING = ('nonholdout first; descending minimum of the two original-first-step own-X>=250 '
           'mask counts; ties by dataset_key then agent-w0/agent-w1 first-seen state_id')


def need(value, message):
    if not value:
        raise ValueError(message)


def file_ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'file': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def read_reference(ref):
    path = (ROOT / ref['file']).resolve()
    data = path.read_bytes()
    need(hashlib.sha256(data).hexdigest() == ref['sha256']
         and ('bytes' not in ref or len(data) == ref['bytes']), 'Original artifact changed')
    return json.loads(data)


def certified_headers(evidence, entries, accessed):
    """Bind the unchanged whole-batch sample plan using all original headers.

    Unselected RGB/mask payloads are not read or newly certified. No subset
    sampling plan or replacement cache certificate is created.
    """
    headers, population = {}, []
    for entry in evidence['complete']:
        key = entry['dataset_key']
        ref = evidence['certificate']['tables'][key]
        need(entries[key]['table_manifest'] == ref, 'Frozen population table reference changed')
        header = read_reference(ref)
        accessed.append(ref)
        derived = identity_fields(header['pair_id'], header['source']['t01']['pair'])
        need(derived['dataset_key'] == key and header.get('dataset_identity', derived) == derived
             and all(header[field] == entries[key][field] for field in ('scene', 'kind'))
             and header['pair_id'] == entries[key]['original_pair_id']
             and header['pair_id'] == entry['pair_id'], 'Certified header identity changed')
        headers[key] = header
        population.extend((key, state) for state in header['states'])
    plan = evidence['report']['sampling_plan']
    need(sample_state_keys(population, seed=plan['seed'], sample_size=plan['sample_size']) == plan,
         'Original full-batch sampled population changed')
    return headers


def accepted_pair_metadata(evidence, entry, header, accessed):
    """The original certificate loader's receipt checks, without table payload I/O."""
    from src.tools import t02_cache_check as cache, t02_export as export
    original, actual = certificates._audited(entry['source'], ROOT)
    cfg, meta, metrics = original['config'], original['_meta'], original['metrics']
    cache.exact_export_command(meta['command'], cfg['name'])
    need(meta.get('end') and meta.get('exit_code') == 0 and cfg.get('version') == export.VERSION
         and cfg.get('gpus') == 1 and cfg.get('depth') == 4 and metrics.get('kind') == 't02_table_export'
         and metrics.get('producer_exit_code') == 0 and metrics.get('export_checks_complete') is True
         and metrics.get('errors') == [] and metrics.get('cleanup_errors') == []
         and metrics.get('unstarted_pair_ids') == [] and metrics.get('black_frame_detected') is False,
         'Original T02 source was not completely accepted')
    rows = metrics['pairs']
    need([row['pair_id'] for row in rows] == cfg['pair_id'] and len(set(cfg['pair_id'])) == len(rows),
         'Original export inventory changed')
    matches = [(index, row) for index, row in enumerate(rows) if row['pair_id'] == entry['pair_id']]
    need(len(matches) == 1, 'Ambiguous original export row')
    index, row = matches[0]
    directory = Path(cfg['scratch']).resolve() / 't02_export' / cfg['name'] / cfg['pbs_jobid']
    pair_dir = directory / f'pair_{index:04d}'
    need(Path(cfg['scratch_directory']).resolve() == directory and Path(row['scratch_directory']).resolve() == pair_dir
         and row.get('status') == 'exported' and row.get('table_complete') is True
         and row.get('native_evidence_verified') is True and row.get('pair_checks_passed') is True
         and row.get('cleanup_errors') == [] and row.get('error') is None and not row.get('raw_index_error')
         and row['table_manifest'] == entry['table_manifest'], 'Original accepted row binding changed')
    package = cache.read_json_ref(row['source_package'], pair_dir, expected=pair_dir / 'authenticated_source.json')
    details = cache.read_json_ref(row['detail'], pair_dir, expected=pair_dir / 'checks.json')
    accessed.extend([actual, row['source_package'], row['detail']])
    need(details['table_manifest'] == row['table_manifest'] and details['native_checks'].get('passed') is True
         and details['pair_checks'].get('passed') is True and details['pair_checks'].get('synthetic_only') is False
         and header['source'].get('t01') == package['source_provenance']
         and header['source'].get('export_version') == export.VERSION
         and header['source'].get('frozen_render_version') == FROZEN_RENDER_VERSION
         and header['source'].get('field_of_view') == package['source_config'].get('field_of_view') == 60
         and all(header[key] == package[key] for key in ('pair_id', 'scene', 'kind', 'instruction', 'anchor_name'))
         and header['candidate_names'] == package['candidate_names'] and header['target_names'] == package['target_names']
         and package['independent_verification']['validity']['acceptance_passed'] is True,
         'Original package/header/check receipt differs')
    return {'entry': entry, 'header': header, 'row': row, 'package': package, 'details': details,
            'export_source': actual, 'evidence': evidence}


def first_step_action(checked, original):
    actions = checked['first_step_visible_actions'][f'w{original}']
    need(isinstance(actions, list) and actions and all(isinstance(a, str) for a in actions)
         and actions == sorted(set(actions)), 'Original sorted first-step visibility is unavailable')
    return actions[0]


def observation_prefix(original, action, state, pixels, checked):
    return {'original_world_id': f'w{original}', 'state_id': state,
            'action_prefix': [action], 'trace_prefix': [{'turn': 1, 'action': action, 'state_id': state}],
            'anchor_pixels': pixels, 'figure_only_observation_prefix': True,
            'prefix_source': PREFIX_SOURCE,
            'original_selection_witness_replay': copy.deepcopy(checked['witness_replay'][original]),
            'selection_witness_used_as_visibility_evidence': False}


def first_seen_metadata(header, table_reference, world_order, checked, accessed):
    """Read own-X masks at the first sorted original visible one-step action.

    Look acceptance already requires a visible first step in both worlds. A
    successful selection witness need not observe own X and is not a filter.
    Selected prefixes are later replayed by the actual environment.
    """
    result = {}
    directory = (ROOT / table_reference['file']).resolve().parent
    for requested, original in enumerate(world_order):
        action = first_step_action(checked, original)
        current = header['states'][header['initial_state']]['transitions'].get(action)
        need(current in header['states'], 'Original first-step action/header transition changed')
        ref = header['states'][current]['captures'][original]['masks'][header['anchor_name']]
        path = (directory / ref['file']).resolve()
        need(path.parent == directory and path.name != 'table.json' and 'encoding' not in ref,
             'Ranking mask escaped the fixed original uncompressed training table')
        body = path.read_bytes()
        need(len(body) == ref['bytes'] == 255 * 255 and hashlib.sha256(body).hexdigest() == ref['sha256']
             and set(body).issubset({0, 1}), 'Original ranking mask bytes changed')
        accessed.append({**ref, 'file': str(path)})
        pixels = body.count(1)
        need(header['states'][current]['anchor_pixels'][original] == pixels, 'Header/mask pixel count differs')
        need(pixels >= 250, 'Original first-step action no longer sees own X')
        result[f'w{requested}'] = observation_prefix(original, action, current, pixels, checked)
    return result, None


def validate_selected_table(record):
    """Full selected-table checks; original full-batch cache proof is reused.

    This is an ephemeral figure authorization, not a new population certificate
    or an authorization for unread tables or future training.
    """
    evidence, entry, package = record['evidence'], record['entry'], record['package']
    key = entry['dataset_key']
    ref = evidence['certificate']['tables'][key]
    table = load_table(ref['file'], expected_sha256=ref['sha256'])
    need(table.manifest_reference == ref and table.dataset_key == key and not table.synthetic
         and table.kind == 'look_flip' and table.pair_id == entry['pair_id']
         and table.source.get('t01') == package['source_provenance'], 'Selected full table identity differs')
    for declared in (record['row'], record['details'], record['details']['native_checks'],
                     record['details']['pair_checks'], evidence['certificate']['table_shadow_metadata'][key]):
        verify_shadow_metadata(table, declared)
    # Same original audited export authority as load_certificate_batch; no new native claim.
    table.export_verified = True
    checked = check_pair(table, package['t01'])
    need(checked.get('passed') is True and checked.get('synthetic_only') is False,
         'Selected complete table failed original pair checks')
    verify_shadow_metadata(table, checked)
    require_cache_validation(evidence['report'])
    for row in evidence['report']['states']:
        if population_key(row) == key:
            verify_cached_binding(table, row)
    verify_shadow_metadata(table, evidence['report']['table_shadow_metadata'][key])
    # Full-batch completeness/native sample IDs were verified by certificate_metadata,
    # and the exact complete population was reconstructed from every SHA-bound header.
    table.cache_verified = True
    table.figure_only_authority = {'source': evidence['source'], 'scope': 'selected complete table; original full-batch cache receipt reused; no new native revalidation'}
    return table, checked


def verify_accessed(accessed):
    """Refuse a changed ranking/header/receipt before publishing any figure."""
    observed = {}
    for ref in accessed:
        path = (ROOT / ref['file']).resolve()
        expected = ref['sha256']
        need(path not in observed or observed[path] == expected, 'Conflicting source SHA at one path')
        if path in observed:
            continue
        body = path.read_bytes()
        need(hashlib.sha256(body).hexdigest() == expected
             and ('bytes' not in ref or len(body) == ref['bytes']), 'Figure input changed during export')
        observed[path] = expected


def environment(table, key, world_order, *, allow_synthetic=False):
    return TableEnvironment(TableObservationBackend({key: table}, world_orders={key: world_order},
                                                    allow_synthetic=allow_synthetic))


def first_step_seen(table, key, world_order, checked, *, allow_synthetic=False):
    """Replay the original first visible action; preserve the selection witness separately."""
    result = {}
    for requested, original in enumerate(world_order):
        action = first_step_action(checked, original)
        env = environment(table, key, world_order, allow_synthetic=allow_synthetic)
        world = f'w{requested}'
        try:
            env.reset(key, world)
            _, _, done, info = env.step(action)
            need(info['action_success'] and not done, 'Original first visible observation action unavailable')
            state = table.states[info['state_id']]
            pixels = state.captures[original].count(table.anchor_name)
            need(pixels >= 250, 'Original first-step action no longer sees own X')
            result[world] = {**observation_prefix(original, action, info['state_id'], pixels, checked),
                             'tau': info['tau'], 'tau_seen': info['x_seen_step']}
        finally:
            env.close()
    return result, None


def rank_key(row):
    seen = row['first_seen']
    return (row['heldout_scene'], -row['ranking_min_anchor_pixels'], row['dataset_key'],
            seen['w0']['state_id'], seen['w1']['state_id'])


def png(path, data, mode):
    """Lossless serialization only; original policy RGB comes from the environment."""
    from PIL import Image
    image = Image.frombytes(mode, (255, 255), data)
    body = io.BytesIO()
    image.save(body, format='PNG')
    with Path(path).open('xb') as stream:
        stream.write(body.getvalue())
    with Image.open(path) as restored:
        need(restored.mode == mode and restored.size == (255, 255) and restored.tobytes() == data,
             'PNG encoding changed original decoded pixels')
    return {**file_ref(path), 'decoded_pixels_sha256': hashlib.sha256(data).hexdigest(),
            'width': 255, 'height': 255, 'mode': mode}


def frame_export(table, key, order, world, action, directory, header, *, allow_synthetic=False):
    env = environment(table, key, order, allow_synthetic=allow_synthetic)
    stem = 's0' if action is None else world + '_' + action
    try:
        observation = env.reset(key, world)
        if action is not None:
            observation, _, done, info = env.step(action)
            need(not done and info['action_success'], 'Required original one-step figure view unavailable')
        info = env.info
        state = table.states[info['state_id']]
        original = env.backend.world
        public = 0 if state.shared else original
        capture = state.captures[public]
        need(env.frame.raw_rgb == capture.rgb, 'Export must use the same public capture as the policy')
        masks = {}
        for role, name in [('X', table.anchor_name), ('A', table.candidate_names[0]), ('B', table.candidate_names[1])]:
            bits = capture.masks[name]
            binary = bytes(255 if bit else 0 for bit in bits)
            masks[role] = {'object_name': name, 'pixels': bits.count(1),
                'original_binary_01_sha256': hashlib.sha256(bits).hexdigest(),
                'png': png(directory / (stem + '_' + role + '_mask.png'), binary, 'L')}
        source = header['states'][info['state_id']]['captures'][public]
        return {'world': world if action else 'shared_w0_w1', 'original_world_id': f'w{original}',
            'public_capture_original_world_id': f'w{public}', 'state_id': info['state_id'],
            'action': action, 'action_prefix': [] if action is None else [action],
            'tau': info['tau'], 'tau_seen': info['x_seen_step'], 'anchor_pixels': capture.count(table.anchor_name),
            'agent_input': png(directory / (stem + '_marked.png'), observation.image.data, 'RGB'),
            'raw_rgb': png(directory / (stem + '_raw.png'), capture.rgb.data, 'RGB'),
            'masks': masks, 'public_text': observation.text,
            'public_marks': [vars(mark) for mark in observation.marks],
            'source_table_capture': copy.deepcopy(source), 'capture_render_id': capture.render_id}
    finally:
        env.close()


def export_pair(table, package, candidate, directory, *, code_commit, allow_synthetic=False):
    key, order = candidate['dataset_key'], candidate['world_order']
    header = read_reference(table.manifest_reference)
    inputs = []
    for world in ('w0', 'w1'):
        env = environment(table, key, order, allow_synthetic=allow_synthetic)
        try:
            observation = env.reset(key, world)
            inputs.append(observation)
        finally:
            env.close()
    need(inputs[0].image.data == inputs[1].image.data and inputs[0] == inputs[1],
         'Actual agent s0 inputs differ between worlds')
    objects = {obj['name']: obj for obj in package['worlds'][0]['objects']}
    names = {'X': table.anchor_name, 'A': table.candidate_names[0], 'B': table.candidate_names[1]}
    object_info = {role: {'name': name, 'object_type': objects[name]['objectType']} for role, name in names.items()}
    directory.mkdir(parents=True, exist_ok=False)
    frames = [frame_export(table, key, order, 'w0', None, directory, header, allow_synthetic=allow_synthetic)]
    for world in ('w0', 'w1'):
        for action in ('turn_left', 'turn_right'):
            frames.append(frame_export(table, key, order, world, action, directory, header,
                                       allow_synthetic=allow_synthetic))
    initial = table.states[table.initial_state]
    manifest = {'version': VERSION, **copy.deepcopy(candidate), 'pair_id': table.pair_id,
        'scene': table.scene, 'room_type': holdout.room_type(table.scene), 'kind': table.kind,
        'objects': object_info, 'candidate_role_policy': 'A/B are original table sorted stable candidate names; not display-label integers',
        'fixed_display_labels': table.labels, 'correct_targets': {f'w{i}': table.target_names[order[i]] for i in (0, 1)},
        'world_order_original_indices': order, 's0_agent_input_sha256': {f'w{i}': inputs[i].image.sha256 for i in (0, 1)},
        's0_original_raw_rgb_sha256': {f'w{i}': initial.captures[i].rgb.sha256 for i in (0, 1)},
        's0_agent_inputs_equal': True, 'frames': frames, 'table_manifest': table.manifest_reference,
        'source_provenance': package['source_provenance'], 'code_commit': code_commit,
        'ranking_policy': RANKING, 'figure_observation_prefix_source': PREFIX_SOURCE,
        'original_selection_witnesses': copy.deepcopy(package['t01']['witnesses']),
        'research_population_changed': False, 'renderer_used': False,
        'authority': 'original successful full-batch T02/cache certificates; selected complete table bytes/pair/cache binding checked; figure-only reuse',
        'new_training_or_cache_certificate': False,
        'extra_source_reauthentication': {'status': 'not_awaited', 'passed': None}}
    save_json(directory / 'manifest.json', manifest)
    return file_ref(directory / 'manifest.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--qualification-reference', required=True)
    parser.add_argument('--holdout-reference', required=True)
    parser.add_argument('--engineering-one-pair', action='store_true',
                        help='CPU smoke of the same top-ranked pair into logs/review/<name>, never the final figure directory')
    args = parser.parse_args()
    need(re.fullmatch(r'diag_fig01_[A-Za-z0-9_]+', args.name), 'Use a unique FIG01 diagnostic run')
    output = ROOT / 'results' / (args.name + '.json')
    need(not output.exists(), 'Preserve prior FIG01 evidence')
    started = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError('FIG01 CPU bound 295 seconds; partial output retained')
    original = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 295.)
    metrics = {'kind': 'fig01_original_cache_frame_export', 'errors': [], 'complete': False,
               'renderer_used': False, 'research_population_changed': False}
    sources, code = [], 1
    try:
        qualification_ref = json.loads(args.qualification_reference)
        frozen_ref = json.loads(args.holdout_reference)
        metadata_started = time.monotonic()
        manifest, proof, source, manifest_ref = authenticate_certificate_qualification(qualification_ref)
        sources.append(source)
        frozen, actual_frozen = holdout.load_frozen_holdout(frozen_ref, root=ROOT)
        need(holdout.ref_identity(actual_frozen) == holdout.ref_identity(frozen_ref), 'Fixed holdout reference changed')
        holdout.assert_pilot_split(manifest['entries'], manifest['split'], frozen)
        source_keys = set(proof['train_keys'])
        entries = {row['dataset_key']: row for row in manifest['entries']}
        accessed = [qualification_ref, manifest_ref, frozen_ref]
        records, all_headers = {}, {}
        for reference in manifest['cache_sources']:
            evidence = certificates.certificate_metadata(reference, root=ROOT)
            sources.append(evidence['source'])
            accessed.extend([evidence['source'], *[evidence['data']['metrics'][field] for field in
                ('authorization_manifest', 'sampling_plan', 'validation_report')]])
            headers = certified_headers(evidence, entries, accessed)
            need(not set(all_headers).intersection(headers), 'Duplicate table across complete original certificates')
            all_headers.update(headers)
            for entry in evidence['complete']:
                key = entry['dataset_key']
                holdout.room_type(entries[key]['scene'])
                if entries[key]['kind'] == 'look_flip':
                    records[key] = accepted_pair_metadata(evidence, entry, headers[key], accessed)
        need(set(all_headers) == set(entries), 'Original full83 header coverage differs')
        metrics['metadata_seconds'] = time.monotonic() - metadata_started
        print('FIG01_PHASE: metadata_complete', flush=True)
        ranking_started = time.monotonic()
        candidates, unrankable = [], []
        for key in sorted(records):
            record = records[key]
            order = entries[key]['world_order']
            seen, reason = first_seen_metadata(record['header'], entries[key]['table_manifest'], order,
                                               record['details']['pair_checks'], accessed)
            if reason:
                unrankable.append({'dataset_key': key, 'reason': reason})
                continue
            candidates.append({'dataset_key': key, 'world_order': order, 'heldout_scene': key not in source_keys,
                'first_seen': seen, 'ranking_min_anchor_pixels': min(v['anchor_pixels'] for v in seen.values())})
        candidates.sort(key=rank_key)
        need(len(candidates) >= 5, 'Fewer than five first-step-rankable LookFlip pairs; no fabricated replacement')
        need(len({records[row['dataset_key']]['header']['pair_id'] for row in candidates[:5]}) == 5,
             'Selected original pair IDs collide; preserve source identities and report')
        selected = copy.deepcopy(candidates[:1 if args.engineering_one_pair else 5])
        metrics['ranking_seconds'] = time.monotonic() - ranking_started
        print('FIG01_PHASE: ranking_complete', flush=True)
        validation_started = time.monotonic()
        tables = {}
        for candidate in selected:
            key = candidate['dataset_key']
            table, checked = validate_selected_table(records[key])
            seen, reason = first_step_seen(table, key, candidate['world_order'], checked)
            need(reason is None and all(all(seen[world].get(field) == value for field, value in row.items())
                 for world, row in candidate['first_seen'].items()), 'Full original environment replay changed metadata ranking')
            candidate['first_seen'] = seen  # Actual environment supplies tau/tau_seen; no hand-set flags.
            tables[key] = table
            sources.append(records[key]['export_source'])
        verify_accessed(accessed)
        metrics['selected_full_validation_seconds'] = time.monotonic() - validation_started
        print('FIG01_PHASE: selected_complete_tables_validated', flush=True)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        need(re.fullmatch(r'[0-9a-f]{40}', commit), 'Exporter Git identity missing')
        destination = ROOT / 'logs/review' / args.name / 'cpu_smoke' if args.engineering_one_pair else ROOT / DESTINATION
        destinations = [destination / tables[row['dataset_key']].pair_id for row in selected]
        need(all(not path.exists() for path in destinations), 'Preserve previously exported figures')
        refs = []
        export_started = time.monotonic()
        for row, directory in zip(selected, destinations):
            key = row['dataset_key']
            refs.append(export_pair(tables[key], records[key]['package'], row, directory, code_commit=commit))
        verify_accessed(accessed)
        metrics['frame_export_seconds'] = time.monotonic() - export_started
        metrics.update(complete=True, selected=selected, ranking=candidates, unrankable=unrankable,
            ranking_policy=RANKING, manifests=refs, qualification_reference=qualification_ref,
            population_manifest_reference=manifest_ref, holdout_reference=frozen_ref, code_commit=commit,
            engineering_one_pair=args.engineering_one_pair, formal_five_export_complete=not args.engineering_one_pair,
            full_population_metadata_verified=True, full_table_payloads_verified=list(tables),
            scope='full83 original certificate/header metadata; all LookFlip original sorted-first-visible-action X masks ranked; only selected tables fully loaded and rechecked; no new native PASS for unread tables',
            new_training_or_cache_certificate=False)
        code = 0
    except BaseException as error:
        traceback.print_exc()
        metrics['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original)
    metrics.update(producer_exit_code=code, elapsed_seconds=time.monotonic() - started)
    save_json(output, {'metrics': metrics, 'config': {**vars(args), 'version': VERSION, 'gpus': 0,
        'cpu_bound_seconds': 295,
        'out': str(Path('logs/review') / args.name / 'cpu_smoke') if args.engineering_one_pair else DESTINATION},
        'sources': sources})
    print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
