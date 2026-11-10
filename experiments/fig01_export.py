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
from src.envs.pair_table_checks import check_pair
from src.tools import smoke_e2e_pilot_dataset as dataset
from src.tools import train_holdout_freeze as holdout
from src.tools.smoke_e2e_pilot_certificates import authenticate_certificate_qualification
from src.tools.t02_export import save_json

VERSION = 'fig01_original_cache_frames_v1'
DESTINATION = 'paper/fig_src/lookflip_frames'
RANKING = ('nonholdout first; descending minimum of the two original-witness first-own-X>=250 '
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
    need(hashlib.sha256(data).hexdigest() == ref['sha256'], 'Original artifact changed')
    return json.loads(data)


def environment(table, key, world_order, *, allow_synthetic=False):
    return TableEnvironment(TableObservationBackend({key: table}, world_orders={key: world_order},
                                                    allow_synthetic=allow_synthetic))


def witness_first_seen(table, key, world_order, checked, *, allow_synthetic=False):
    """Replay only the original accepted witness's literal table actions."""
    result = {}
    for requested, original in enumerate(world_order):
        witness = checked['witness_replay'][original]
        if witness.get('applicable') is not True or witness.get('passed') is not True:
            return None, 'original_witness_unavailable_or_not_applicable'
        env = environment(table, key, world_order, allow_synthetic=allow_synthetic)
        world = f'w{requested}'
        prefix = []
        try:
            env.reset(key, world)
            for row in witness['trace']:
                action = row.get('action')
                if not isinstance(action, str) or action.startswith('select '):
                    break
                _, _, done, info = env.step(action)
                need(info['action_success'] and not done and info['state_id'] == row['state_id'],
                     'Original accepted witness table trace changed')
                prefix.append(copy.deepcopy(row))
                state = table.states[info['state_id']]
                pixels = state.captures[original].count(table.anchor_name)
                if pixels >= 250:
                    result[world] = {'original_world_id': f'w{original}', 'state_id': info['state_id'],
                        'action_prefix': [item['action'] for item in prefix], 'trace_prefix': prefix,
                        'original_complete_trace': copy.deepcopy(witness['trace']),
                        'anchor_pixels': pixels, 'tau': info['tau'], 'tau_seen': info['x_seen_step']}
                    break
        finally:
            env.close()
        if world not in result:
            return None, 'original_witness_has_no_own_X_250_before_select'
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
        'ranking_policy': RANKING, 'research_population_changed': False, 'renderer_used': False,
        'authority': 'original successful T02/cache certificates; unchanged cached-table checks and environment authorization',
        'extra_source_reauthentication': {'status': 'not_awaited', 'passed': None}}
    save_json(directory / 'manifest.json', manifest)
    return file_ref(directory / 'manifest.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--qualification-reference', required=True)
    parser.add_argument('--holdout-reference', required=True)
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
        manifest, _, source, manifest_ref = authenticate_certificate_qualification(qualification_ref)
        sources.append(source)
        # Exactly the production policy's cache-byte loader and authorizer, no synthetic bypass.
        tables, packages, proof = dataset.load_pilot_pool(manifest_ref)
        dataset.authenticate_pilot_holdout(manifest_ref, proof, frozen_ref)
        source_keys = set(proof['train_keys'])
        entries = {row['dataset_key']: row for row in manifest['entries']}
        candidates, unrankable = [], []
        for key in sorted(tables):
            table = tables[key]
            if table.kind != 'look_flip':
                continue
            holdout.room_type(table.scene)
            need(not table.synthetic and table.export_verified and table.cache_verified, 'Original environment authority missing')
            checked = check_pair(table, packages[key]['t01'])  # Pure table replay, no Controller/native imports.
            need(checked['passed'] is True, 'Original cached pair check failed')
            order = entries[key]['world_order']
            seen, reason = witness_first_seen(table, key, order, checked)
            if reason:
                unrankable.append({'dataset_key': key, 'reason': reason})
                continue
            candidates.append({'dataset_key': key, 'world_order': order, 'heldout_scene': key not in source_keys,
                'first_seen': seen, 'ranking_min_anchor_pixels': min(v['anchor_pixels'] for v in seen.values())})
        candidates.sort(key=rank_key)
        need(len(candidates) >= 5, 'Fewer than five original-witness-rankable LookFlip pairs; no fabricated replacement')
        selected = candidates[:5]
        pair_ids = [tables[row['dataset_key']].pair_id for row in selected]
        need(len(set(pair_ids)) == 5, 'Selected original pair IDs collide; preserve source identities and report')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        need(re.fullmatch(r'[0-9a-f]{40}', commit), 'Exporter Git identity missing')
        destinations = [ROOT / DESTINATION / pair_id for pair_id in pair_ids]
        need(all(not path.exists() for path in destinations), 'Preserve previously exported figures')
        refs = []
        for row, directory in zip(selected, destinations):
            key = row['dataset_key']
            refs.append(export_pair(tables[key], packages[key], row, directory, code_commit=commit))
        metrics.update(complete=True, selected=selected, ranking=candidates, unrankable=unrankable,
            ranking_policy=RANKING, manifests=refs, qualification_reference=qualification_ref,
            population_manifest_reference=manifest_ref, holdout_reference=frozen_ref, code_commit=commit)
        code = 0
    except BaseException as error:
        traceback.print_exc()
        metrics['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original)
    metrics.update(producer_exit_code=code, elapsed_seconds=time.monotonic() - started)
    save_json(output, {'metrics': metrics, 'config': {**vars(args), 'version': VERSION, 'gpus': 0,
        'cpu_bound_seconds': 295, 'out': DESTINATION}, 'sources': sources})
    print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
