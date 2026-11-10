"""DIAG02 E3/E4/E5: three original-model policies on the fixed83 pilot population.

No training, new data, parameter search, or scientific-configuration mutation.
The first fixed Look pair exercises all three policies before the remaining82
pairs within the same single-GPU allocation. Privileged masks control only the
obs_trigger observation schedule; model prompts remain entirely public.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time
import traceback

from src.envs.thor_common import ROOT, Worker, deadline_after, install_signals, stop_workers
from src.envs.paired_env import KINDS, parse_action
from src.envs.pair_table import TableEnvironment, TableObservationBackend
from src.tools.smoke_e2e_eval import VllmPolicy, MODEL, public_actions, EXPLORATION
from src.tools.smoke_e2e_views import file_ref
from src.tools.smoke_e2e_pilot_dataset import load_pilot_pool, authenticate_pilot_holdout, ref_identity
from src.tools.smoke_e2e_pilot import model_reference
from src.tools.t02_export import save_json, execution_environment, logged_gate, gate_runtime

VERSION = 'diag02_fixed83_cov_full_input_once_v1'
STRATEGIES = ('obs_never', 'obs_trigger', 'obs_trigger_cov')


def mask_state(env):
    """Count actual original-world mask bytes, not x_seen_step or cached counts."""
    backend = env.backend
    table, state = backend.table, backend.table.states[backend.state_id]
    mask = state.captures[backend.world].masks[table.anchor_name]
    count = len(mask) - mask.count(0)
    return {'state_id': backend.state_id, 'step': env.steps,
        'original_world_id': f'w{backend.world}', 'anchor_pixels': count,
        'anchor_mask_sha256': hashlib.sha256(mask).hexdigest(), 'shared': state.shared}


def forced_action(strategy, item):
    """Original trigger returns to s0; new cov may select at current reveal."""
    step = item['env'].steps
    if strategy == 'obs_never':
        return None
    if strategy not in ('obs_trigger', 'obs_trigger_cov'):
        raise ValueError('Unknown diagnostic strategy')
    if strategy == 'obs_trigger_cov' and cov_select_now(item):
        return None
    if step >= 4 or (item['first_seen'] is not None and step % 2 == 0):
        return None
    return EXPLORATION[step]


def collect(strategy, environments, policy, journal):
    if strategy not in STRATEGIES:
        raise ValueError('Unknown diagnostic strategy')
    active, episodes = [], []
    for key, world, env in environments:
        observation = env.reset(key, world)
        initial = mask_state(env)
        active.append({'key': key, 'world': world, 'env': env,
            'observations': [observation], 'responses': [], 'states': [initial],
            'first_seen': 0 if initial['anchor_pixels'] >= 250 else None,
            'first_positive': 0 if initial['anchor_pixels'] > 0 else None,
            'task_reward': 0., 'observation_reward_cost': 0., 'observation_attempts': 0,
            'syntax_failures': 0, 'model_outputs': 0})
    for decision in range(5):
        pending, generated = [], {}
        for index, item in enumerate(active):
            if item['env'].done:
                continue
            action = forced_action(strategy, item)
            if action is not None:
                if action not in public_actions(item['observations'][-1]):
                    raise ValueError('Fixed return-to-s0 observation is unavailable')
                generated[index] = {'action': action, 'response': action, 'forced_observation': True,
                    'response_token_ids': [], 'response_token_count': 0, 'sample_id': None}
            else:
                if (item['env'].backend.state_id != item['env'].backend.table.initial_state
                        and not (strategy == 'obs_trigger_cov' and cov_select_now(item))):
                    raise ValueError('Forced-selection policy did not return to s0')
                pending.append((index, {'observations': item['observations'], 'responses': item['responses'],
                    'select_only': strategy != 'diag_base'}))
        if pending:
            values = policy([context for _, context in pending])
            if len(values) != len(pending):
                raise ValueError('Model omitted/added a diagnostic episode')
            for (index, _), sample in zip(pending, values):
                if (not isinstance(sample.get('response'), str) or not isinstance(sample.get('action'), str)
                    or not sample.get('sample_id') or sample.get('forced_observation') is not False
                    or not isinstance(sample.get('response_token_ids'), list)
                    or sample.get('response_token_count') != len(sample['response_token_ids'])
                    or not 0 < sample['response_token_count'] <= 256):
                    raise ValueError('Incomplete actual greedy model output')
                if strategy != 'diag_base' and sample['action'] not in [a for a in
                    public_actions(active[index]['observations'][-1]) if a.startswith('select ')]:
                    raise ValueError('Forced-select output was not a public select action')
                generated[index] = sample
        for index, item in enumerate(active):
            env = item['env']
            if env.done:
                continue
            before, before_mask = copy.deepcopy(env.info), item['states'][-1]
            before_cov = both_candidates_marked(env)
            sample = generated[index]
            if not sample['forced_observation']:
                env.record_model_call(sample['sample_id']); item['model_outputs'] += 1
            try:
                parsed = parse_action(sample['action'])
            except ValueError:
                parsed = None
                item['syntax_failures'] += int(not sample['forced_observation'])
            observation, reward, done, info = env.step(sample['action'])
            after_mask = mask_state(env)
            item['states'].append(after_mask)
            if item['first_seen'] is None and after_mask['anchor_pixels'] >= 250:
                item['first_seen'] = env.steps
            if item['first_positive'] is None and after_mask['anchor_pixels'] > 0:
                item['first_positive'] = env.steps
            item['task_reward'] += reward
            if parsed is not None and parsed.kind in ('turn_left', 'turn_right', 'move'):
                item['observation_attempts'] += 1
                item['observation_reward_cost'] += max(0., -reward)
            journal({'record_kind': 'decision', 'strategy': strategy, 'dataset_key': item['key'],
                'world_id': item['world'], 'episode_id': before['episode_id'], 'decision_index': decision,
                'before_state': before_mask, 'after_state': after_mask, 'sample': sample,
                'both_candidates_marked_before': before_cov,
                'public_rgb_sha256': item['observations'][-1].image.sha256,
                'public_text': item['observations'][-1].text, 'reward': reward,
                'action_valid': info.get('action_valid'), 'tau': info['tau'],
                'logged_x_seen_step': info['x_seen_step'], 'done': done,
                'selected_object': info['selected_name'], 'success': info['success']})
            item['responses'].append(sample['response']); item['observations'].append(observation)
            if done:
                row = {'record_kind': 'episode', 'strategy': strategy, 'dataset_key': item['key'],
                    'pair_id': env.backend.table.pair_id, 'world_id': item['world'],
                    'original_world_id': f'w{env.backend.world}', 'world_order': list(env.backend.world_order),
                    'episode_id': info['episode_id'], 'pair_kind': info['kind'],
                    'actions': list(item['responses']), 'states': item['states'],
                    'selected_object': info['selected_name'], 'selected': info['selected'],
                    'selection_frame_anchor_pixels': before_mask['anchor_pixels'] if info['selected'] else None,
                    'selection_frame_both_candidates_marked': before_cov if info['selected'] else None,
                    'success': bool(info['success']), 'steps': env.steps, 'model_calls': env.model_calls,
                    'observation_count': info['obs_count'], 'observation_attempts': item['observation_attempts'],
                    'observation_reward_cost': item['observation_reward_cost'], 'task_reward': item['task_reward'],
                    'tau': info['tau'], 'logged_x_seen_step': info['x_seen_step'],
                    'independent_first_seen_step': item['first_seen'],
                    'seen_before_select': bool(info['selected'] and item['first_seen'] is not None
                        and item['first_seen'] < env.steps),
                    'seen_anywhere': item['first_seen'] is not None,
                    'positive_anchor_anywhere': item['first_positive'] is not None,
                    'syntax_failures': item['syntax_failures'], 'model_outputs': item['model_outputs']}
                journal(row); episodes.append(row)
        if all(item['env'].done for item in active):
            break
    if len(episodes) != len(active) or not all(item['env'].done for item in active):
        raise ValueError('DIAG02 episode did not end within the unchanged D+1 budget')
    return episodes


def summarize(episodes):
    result = {}
    for kind in (*KINDS, 'look_all', 'all'):
        rows = [r for r in episodes if kind == 'all' or r['pair_kind'] == kind
            or (kind == 'look_all' and r['pair_kind'].startswith('look_'))]
        n = len(rows)
        result[kind] = {'episodes': n, 'successes': sum(r['success'] for r in rows),
            'success_rate': sum(r['success'] for r in rows) / n if n else None,
            'observation_count_mean': sum(r['observation_count'] for r in rows) / n if n else None,
            'observation_reward_cost_mean': sum(r['observation_reward_cost'] for r in rows) / n if n else None,
            'task_reward_mean': sum(r['task_reward'] for r in rows) / n if n else None,
            'seen_before_select_count': sum(r['seen_before_select'] for r in rows),
            'seen_before_select_rate': sum(r['seen_before_select'] for r in rows) / n if n else None,
            'seen_anywhere_count': sum(r['seen_anywhere'] for r in rows),
            'syntax_failures': sum(r['syntax_failures'] for r in rows),
            'model_outputs': sum(r['model_outputs'] for r in rows)}
    return result


MODELS = {size: f'Qwen/Qwen2.5-VL-{size.upper()}-Instruct' for size in ('3b', '7b', '32b')}
INFERENCE_SETTINGS = {'tensor_parallel_size': 1, 'dtype': 'bfloat16', 'seed': 0,
    'gpu_memory_utilization': .8, 'max_model_len': 8192, 'max_num_seqs': 10,
    'limit_mm_per_prompt': {'image': 5}, 'enforce_eager': True, 'trust_remote_code': False}
SAMPLING_SETTINGS = {'temperature': 0., 'top_p': 1., 'max_tokens': 256, 'n': 1}


def both_candidates_marked(env):
    """Use the actual frame's marks and success-eligible candidate identities."""
    candidates = {r.object_name: r for r in env.frame.candidates}
    marks = {r.label for r in env.frame.marks}
    names = env.backend.table.candidate_names
    return len(names) == 2 and all(name in candidates and candidates[name].success_eligible
        and candidates[name].mark_label in marks for name in names)


def cov_select_now(item):
    return item['states'][-1]['anchor_pixels'] >= 250 and both_candidates_marked(item['env'])


def request_key(descriptor):
    canonical = json.dumps(descriptor, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest(), canonical


def resolved_parameters(value):
    """Serialize every resolved SamplingParams/GuidedDecodingParams field."""
    from dataclasses import is_dataclass, fields
    from enum import Enum
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return resolved_parameters(value.value)
    if isinstance(value, (tuple, list)):
        return [resolved_parameters(x) for x in value]
    if isinstance(value, (set, frozenset)):
        return {'unordered_set': sorted((resolved_parameters(x) for x in value),
            key=lambda x: json.dumps(x, sort_keys=True, allow_nan=False))}
    if isinstance(value, dict):
        return {str(k): resolved_parameters(v) for k, v in value.items()}
    if is_dataclass(value):
        return {f.name: resolved_parameters(getattr(value, f.name)) for f in fields(value)}
    if hasattr(value, '__struct_fields__'):
        return {name: resolved_parameters(getattr(value, name)) for name in value.__struct_fields__}
    raise TypeError('Unsupported resolved generation setting: ' + type(value).__name__)


class ExactInputCache:
    """One actual generation per complete public request, within one model run.

No pair/world/answer/strategy identity enters the key. This is an explicit
evaluation protocol, not a claim that CUDA greedy generation is deterministic.
"""
    def __init__(self):
        self.entries = {}
        self.logical_requests = self.generated_requests = self.reused_requests = 0

    def generate(self, prepared, infer):
        identities, unique_missing = [], {}
        for row in prepared:
            key, canonical = request_key(row['descriptor'])
            identities.append((key, canonical))
            if key in self.entries:
                if self.entries[key]['canonical'] != canonical:
                    raise ValueError('Canonical input key collision')
            elif key in unique_missing:
                if unique_missing[key][0] != canonical:
                    raise ValueError('Canonical input key collision within batch')
            else:
                unique_missing[key] = (canonical, row)
        ordered = sorted(unique_missing)
        if ordered:
            samples = infer([unique_missing[key][1] for key in ordered])
            if len(samples) != len(ordered):
                raise ValueError('Unique generation output coverage mismatch')
            for key, sample in zip(ordered, samples):
                self.entries[key] = {'canonical': unique_missing[key][0], 'sample': copy.deepcopy(sample)}
        new_left = set(ordered)
        result = []
        for key, canonical in identities:
            entry = self.entries[key]
            if entry['canonical'] != canonical:
                raise ValueError('Stored request descriptor changed')
            sample = copy.deepcopy(entry['sample'])
            actual = key in new_left
            new_left.discard(key)
            sample['exact_input_cache'] = {'version': 'full_public_request_once_v1',
                'key': key, 'reused_generation': not actual,
                'origin_sample_id': entry['sample']['sample_id']}
            result.append(sample)
        self.logical_requests += len(prepared)
        self.generated_requests += len(ordered)
        self.reused_requests += len(prepared) - len(ordered)
        return result


class UniqueInputPolicy:
    def __init__(self, model_ref):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        import vllm, torch, transformers
        path = Path(model_ref['directory'])
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.sample_type, self.guide_type = SamplingParams, GuidedDecodingParams
        self.model = LLM(model=str(path), tokenizer=str(path), **INFERENCE_SETTINGS)
        processor_names = ('config.json', 'generation_config.json', 'preprocessor_config.json', 'processor_config.json',
            'tokenizer_config.json', 'tokenizer.json', 'chat_template.json', 'chat_template.jinja',
            'vocab.json', 'merges.txt', 'special_tokens_map.json')
        self.model_proof = {'model': model_ref['model'], 'snapshot_directory': str(path),
            'model_source_sha256': model_ref['source_result']['sha256'],
            'revision': model_ref.get('revision'), 'model_config_sha256': model_ref['model_config_sha256'],
            'verified_weight_files': model_ref.get('weight_files'),
            'processor_files': {n: hashlib.sha256((path / n).read_bytes()).hexdigest()
                for n in processor_names if (path / n).is_file()},
            'inference_settings': INFERENCE_SETTINGS, 'sampling_settings': SAMPLING_SETTINGS,
            'runtime_versions': {'vllm': vllm.__version__, 'torch': torch.__version__,
                'transformers': transformers.__version__}}
        self.serial = 0
        self.cache = ExactInputCache()

    def prepare(self, contexts):
        from PIL import Image
        from src.tools.smoke_e2e_eval import prompt_messages, VERSION as PROMPT_VERSION
        prepared = []
        for context in contexts:
            observations = context['observations']
            messages = prompt_messages(observations, context['responses'], select_only=context['select_only'])
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            choices = ([a for a in public_actions(observations[-1]) if a.startswith('select ')]
                if context['select_only'] else [])
            if context['select_only'] and not choices:
                raise ValueError('Forced-selection episode has no public selectable mark')
            image_descriptors = [{'width': obs.image.width, 'height': obs.image.height, 'mode': 'RGB',
                'sha256': hashlib.sha256(obs.image.data).hexdigest()} for obs in observations]
            if any(d['sha256'] != obs.image.sha256 for d, obs in zip(image_descriptors, observations)):
                raise ValueError('Actual image bytes disagree with public image SHA')
            proof = {'prompt_utf8_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                'chat_template_sha256': hashlib.sha256(str(self.processor.chat_template).encode()).hexdigest(),
                'public_image_sha256': [d['sha256'] for d in image_descriptors],
                'text_token_ids_before_multimodal_expansion': self.processor.tokenizer.encode(prompt, add_special_tokens=False),
                'guided_select_choices': choices if context['select_only'] else None,
                'prompt_contract': PROMPT_VERSION}
            parameters = self.sample_type(**SAMPLING_SETTINGS,
                guided_decoding=self.guide_type(choice=choices) if context['select_only'] else None)
            descriptor = {'model_processor_runtime': self.model_proof, 'input': proof,
                'ordered_rgb_images': image_descriptors, 'prompt_utf8_bytes': len(prompt.encode('utf-8')),
                'sampling_resolved': resolved_parameters(parameters)}
            images = [Image.frombytes('RGB', (obs.image.width, obs.image.height), obs.image.data) for obs in observations]
            prepared.append({'descriptor': descriptor, 'proof': proof,
                'request': {'prompt': prompt, 'multi_modal_data': {'image': images}},
                'parameters': parameters})
        return prepared

    def infer(self, prepared):
        outputs = self.model.generate([r['request'] for r in prepared],
            [r['parameters'] for r in prepared], use_tqdm=False)
        if len(outputs) != len(prepared):
            raise ValueError('vLLM output batch length mismatch')
        result = []
        for output, row in zip(outputs, prepared):
            if len(output.outputs) != 1:
                raise ValueError('Expected exactly one greedy output')
            sample = output.outputs[0]
            self.serial += 1
            result.append({'response': sample.text, 'action': sample.text.strip(),
                'response_token_ids': list(sample.token_ids), 'response_token_count': len(sample.token_ids),
                'finish_reason': sample.finish_reason, 'stop_reason': sample.stop_reason,
                'sample_id': f'model_sample_{self.serial:06d}', 'forced_observation': False,
                'policy': 'original_Qwen_greedy_vllm_full_input_once', 'input': row['proof'],
                'request_descriptor': row['descriptor']})
        return result

    def __call__(self, contexts):
        return self.cache.generate(self.prepare(contexts), self.infer)

    def uncached(self, contexts):
        return self.infer(self.prepare(contexts))


def diagnostic_model_reference(config):
    expected = config['model_source']
    if config['model_size'] == '3b':
        original = model_reference(expected, config['identity']['scratch'])
        if Path(original['directory']).resolve() != Path(config['model_path']).resolve():
            raise ValueError('3B path differs from authenticated original snapshot')
        original['revision'] = Path(original['directory']).name
        return original
    from src.tools.t01_partition import audited_result
    data, ref = audited_result(ROOT / expected['file'])
    if ref_identity(ref) != expected or data['_meta']['exit_code'] != 0:
        raise ValueError('Downloaded model source was not an unchanged successful audited result')
    m = data['metrics']
    path = Path(config['model_path']).resolve()
    path.relative_to(Path(config['identity']['scratch']).resolve())
    if (m.get('completed') is not True or m.get('errors') != [] or m.get('model') != MODELS[config['model_size']]
            or Path(m['snapshot_path']).resolve() != path or not re.fullmatch(r'[0-9a-f]{40}', str(m.get('revision')))
            or path.name != m['revision']):
        raise ValueError('Downloaded snapshot identity is incomplete or differs')
    sha = hashlib.sha256((path / 'config.json').read_bytes()).hexdigest()
    if sha != m['model_config_sha256'] or json.loads((path / 'config.json').read_text()).get('model_type') != 'qwen2_5_vl':
        raise ValueError('Downloaded model configuration changed')
    started = time.monotonic()
    manifest = m['files']
    weight_rows = [row for row in manifest if row['path'].endswith('.safetensors')]
    if not weight_rows or len({row['path'] for row in weight_rows}) != len(weight_rows):
        raise ValueError('Downloaded weight manifest missing or ambiguous')
    index_path = path / 'model.safetensors.index.json'
    if index_path.exists():
        indexed = set(json.loads(index_path.read_text())['weight_map'].values())
        if indexed != {row['path'] for row in weight_rows}:
            raise ValueError('Weight index shards differ from the pinned download manifest')
    weight_files = []
    for row in weight_rows:
        if Path(row['path']).name != row['path'] or not re.fullmatch(r'[0-9a-f]{64}', str(row.get('hub_lfs_sha256'))):
            raise ValueError('Missing official SHA256 for weight file; not verified: ' + str(row['path']))
        p = path / row['path']
        p.resolve().relative_to(Path(config['identity']['scratch']).resolve())
        if p.stat().st_size != row['bytes']:
            raise ValueError('Weight byte size changed: ' + row['path'])
        digest = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != row['hub_lfs_sha256']:
            raise ValueError('Weight SHA256 differs from official LFS: ' + row['path'])
        weight_files.append({'path': row['path'], 'bytes': row['bytes'], 'sha256': digest.hexdigest(), 'official_lfs_match': True})
    return {'source_result': ref, 'directory': str(path), 'model': m['model'],
        'revision': m['revision'], 'model_config_sha256': sha, 'weight_files': weight_files,
        'weight_verification_seconds': time.monotonic() - started,
        'weight_index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest() if index_path.exists() else None}


def output_identity(sample):
    return {k: sample[k] for k in ('response', 'action', 'response_token_ids', 'finish_reason', 'stop_reason')}


def evaluate_determinism(records, decisions, policy, tables, proof, logs, model_size, deadline):
    """Exact-input protocol verification; optional 3B uncached batch probes."""
    indexed = {s: {(d['dataset_key'], d['world_id']): d for d in decisions[s]
        if not d['sample']['forced_observation']} for s in STRATEGIES}
    direct = [(key, world) for key in sorted(tables) if tables[key].kind.startswith('direct_')
        for world in ('w0', 'w1')]
    if len(direct) != 86:
        raise ValueError('Expected the original86 Direct negative controls')
    comparisons = []
    for identity in direct:
        samples = {s: indexed[s][identity]['sample'] for s in STRATEGIES}
        descriptors = [s['request_descriptor'] for s in samples.values()]
        same_input = all(x == descriptors[0] for x in descriptors)
        same_output = all(output_identity(x) == output_identity(next(iter(samples.values()))) for x in samples.values())
        comparisons.append({'dataset_key': identity[0], 'world_id': identity[1],
            'same_complete_request': same_input, 'same_output': same_output,
            'keys': {s: x['exact_input_cache']['key'] for s, x in samples.items()},
            'samples': samples})
    if not all(r['same_complete_request'] and r['same_output'] for r in comparisons):
        raise ValueError('Exact-input protocol failed its Direct negative control')
    report = {'direct_control_count': len(direct), 'same_complete_request': sum(r['same_complete_request'] for r in comparisons),
        'same_request_different_output': sum(not r['same_output'] for r in comparisons),
        'comparisons': comparisons, 'protocol': 'single generation per full public input, shared across strategies/worlds within model run',
        'does_not_claim_cuda_kernel_determinism': True, 'uncached_probes': []}
    if model_size != '3b':
        return report
    contexts, backends = [], []
    try:
        for key, world in direct:
            backend = TableObservationBackend({key: tables[key]}, world_orders={key: proof['world_orders'][key]})
            backends.append(backend)
            env = TableEnvironment(backend)
            contexts.append({'observations': [env.reset(key, world)], 'responses': [], 'select_only': True})
        # Same requests and settings; only batch membership/order varies. These
        # diagnostic generations do not enter strategy success summaries.
        previous = None
        for name, order in [('direct_same_order_1', list(range(86))),
                ('direct_same_order_2', list(range(86))), ('direct_reversed', list(reversed(range(86))))]:
            if time.time() >= deadline - 90:
                report['uncached_probes'].append({'name': name, 'status': 'not_run_deadline_reserve'})
                continue
            outputs = policy.uncached([contexts[i] for i in order])
            samples = {i: sample for i, sample in zip(order, outputs)}
            comparisons_raw = []
            for i, identity in enumerate(direct):
                baseline = indexed['obs_never'][identity]['sample']
                current = samples[i]
                if baseline['request_descriptor'] != current['request_descriptor']:
                    raise ValueError('Uncached probe input differs from original strategy request')
                comparisons_raw.append({'dataset_key': identity[0], 'world_id': identity[1],
                    'request_key': request_key(current['request_descriptor'])[0], 'sample': current,
                    'different_from_mixed_batch_origin': output_identity(current) != output_identity(baseline),
                    'different_from_prior_same_order': (None if previous is None else
                        output_identity(current) != output_identity(previous[i]))})
            report['uncached_probes'].append({'name': name, 'status': 'completed', 'request_order': order,
                'episodes': len(outputs), 'mixed_batch_origin_differences': sum(r['different_from_mixed_batch_origin'] for r in comparisons_raw),
                'prior_same_order_differences': (None if previous is None else sum(r['different_from_prior_same_order'] for r in comparisons_raw)),
                'comparisons': comparisons_raw})
            if name == 'direct_same_order_1':
                previous = samples
    finally:
        for backend in backends:
            backend.close()
    return report


def eval_worker(config, connection):
    os.sched_setaffinity(0, config['identity']['cpu_affinity'])
    started = time.monotonic()
    tables, packages, proof = load_pilot_pool(config['manifest_reference'])
    holdout = authenticate_pilot_holdout(config['manifest_reference'], proof, config['holdout_reference'])
    if len(tables) != 83 or set(tables) != set(proof['train_keys']) | set(proof['eval_keys']):
        raise ValueError('DIAG02 requires exactly the original83 train+heldout population')
    if any(table.scene not in {f'FloorPlan{x}' for lo in (1, 201, 301, 401) for x in range(lo, lo + 20)} for table in tables.values()):
        raise ValueError('No test scene may enter D3')
    authenticated_seconds = time.monotonic() - started
    model = diagnostic_model_reference(config)
    started = time.monotonic(); policy = UniqueInputPolicy(model)
    model_seconds = time.monotonic() - started
    first = sorted(key for key, value in tables.items() if value.kind.startswith('look_'))[0]
    batches = [[first], sorted(set(tables) - {first})]
    records = {strategy: [] for strategy in STRATEGIES}
    decisions = {strategy: [] for strategy in STRATEGIES}
    logs = Path(config['episode_directory'])
    streams = {s: (logs / (s + '.jsonl')).open('x') for s in STRATEGIES}
    progress = (logs / 'progress.jsonl').open('x')
    try:
        for stage, keys in enumerate(batches):
            for strategy in STRATEGIES:
                if time.time() >= config['deadline'] - 30:
                    raise TimeoutError('DIAG02 allocation deadline; preserve completed episodes')
                backends, environments = [], []
                for key in keys:
                    for world in ('w0', 'w1'):
                        backend = TableObservationBackend({key: tables[key]}, world_orders={key: proof['world_orders'][key]})
                        backends.append(backend)
                        environments.append((key, world, TableEnvironment(backend)))
                def journal(row):
                    if row['record_kind'] == 'decision':
                        decisions[strategy].append(row)
                    streams[strategy].write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                    streams[strategy].flush()
                try:
                    rows = collect(strategy, environments, policy, journal)
                finally:
                    for backend in backends:
                        backend.close()
                if len(rows) != 2 * len(keys):
                    raise ValueError('Missing original pair/world evaluation')
                records[strategy].extend(rows)
                progress.write(json.dumps({'stage': stage, 'strategy': strategy, 'episodes_complete': len(rows),
                    'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}) + '\n'); progress.flush()
            if stage == 0:
                if any(len(records[s]) != 2 for s in STRATEGIES):
                    raise ValueError('First-pair engineering unit did not complete all strategies')
                progress.write(json.dumps({'first_pair_engineering_passed': True, 'dataset_key': first,
                    'check': 'three strategies; both worlds; original D+1; real model outputs; independent masks; complete fields'}) + '\n'); progress.flush()
    finally:
        for stream in streams.values():
            stream.close()
        progress.close()
    expected = {(key, world) for key in tables for world in ('w0', 'w1')}
    for strategy, rows in records.items():
        if len(rows) != 166 or {(r['dataset_key'], r['world_id']) for r in rows} != expected:
            raise ValueError('Full83 strategy population coverage mismatch')
    summary = {s: summarize(rows) for s, rows in records.items()}
    deterministic = evaluate_determinism(records, decisions, policy, tables, proof, logs, config['model_size'], config['deadline'])
    e5_path = logs / 'e5_determinism.json'
    save_json(e5_path, deterministic)
    return {'completed': True, 'evaluation_design': 'original83 pairs, both worlds once, greedy; equal world weights; training scenes only',
        'strategies': summary, 'determinism': {k: v for k, v in deterministic.items() if k not in ('comparisons', 'uncached_probes')},
        'determinism_report': file_ref(e5_path, ROOT / 'logs'),
        'uncached_probe_summary': [{k: v for k, v in row.items() if k not in ('comparisons', 'request_order')} for row in deterministic['uncached_probes']],
        'inference_settings': INFERENCE_SETTINGS, 'sampling_settings': SAMPLING_SETTINGS,
        'unique_input_generation': {'logical_requests': policy.cache.logical_requests, 'actual_generations': policy.cache.generated_requests, 'reused_generations': policy.cache.reused_requests},
        'first_pair_engineering_passed': True, 'first_pair_dataset_key': first,
        'trigger_privileged': True, 'cov_rule': 'same fixed trigger exploration; select immediately only when ownX>=250 and both actual candidate marks are success-eligible, otherwise return as original', 'trigger_rule': 'fixed L,R,R,L; after own mask >=250 complete current excursion to s0 then select',
        'look_trigger_minus_never_success_rate': summary['obs_trigger']['look_all']['success_rate'] - summary['obs_never']['look_all']['success_rate'],
        'look_cov_minus_never_success_rate': summary['obs_trigger_cov']['look_all']['success_rate'] - summary['obs_never']['look_all']['success_rate'],
        'lf_trigger_minus_never_success_rate': summary['obs_trigger']['look_flip']['success_rate'] - summary['obs_never']['look_flip']['success_rate'],
        'lf_cov_minus_never_success_rate': summary['obs_trigger_cov']['look_flip']['success_rate'] - summary['obs_never']['look_flip']['success_rate'],
        'source_authentication_seconds': authenticated_seconds, 'model_loading_seconds': model_seconds,
        'model_reference': model, 'fixed_holdout': holdout, 'common_pool_hash': proof['common_pool_hash'],
        'sources': [*proof['sources'], model['source_result']], 'paper_result': False,
        'formal_test_set_evaluation': False, 'training_performed': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-size', required=True, choices=tuple(MODELS))
    p.add_argument('--model-path', required=True)
    p.add_argument('--name', required=True); p.add_argument('--manifest', required=True)
    p.add_argument('--holdout-reference', required=True); p.add_argument('--model-source', required=True)
    p.add_argument('--timeout-seconds', type=float, default=3240.)
    p.add_argument('--gate-timeout-seconds', type=float, default=180.)
    args = p.parse_args()
    if not re.fullmatch(r'diag_diag02_baselines_(3b|7b|32b)_[A-Za-z0-9_]+', args.name) or not 300 <= args.timeout_seconds <= 3540:
        p.error('Require a fresh DIAG02 namespace and a sub-hour single-GPU driver')
    if not args.name.startswith('diag_diag02_baselines_' + args.model_size + '_'):
        p.error('Model size and namespace differ')
    if args.model_size == '3b' and args.timeout_seconds > 1740:
        p.error('E3 driver must leave PBS room under0.5GPUh')
    output = ROOT / 'results' / (args.name + '.json')
    if output.exists():
        raise FileExistsError('Preserve previous diagnostic results')
    config = {**vars(args), 'version': VERSION, 'gpus': 1, 'node': socket.gethostname(),
        'manifest_reference': ref_identity(json.loads(args.manifest)),
        'holdout_reference': ref_identity(json.loads(args.holdout_reference)),
        'model_source': ref_identity(json.loads(args.model_source)), 'model': MODELS[args.model_size],
        'strategies': list(STRATEGIES), 'maximum_requested_gpu_hours': .5 if args.model_size == '3b' else 1.,
        'training_config_changed': False, 'model_settings_reused_from': 'src.tools.smoke_e2e_eval.VllmPolicy except uniform0.8 memory allocation for32B and exact-input reuse',
        'common_inference_settings': INFERENCE_SETTINGS, 'common_sampling_settings': SAMPLING_SETTINGS}
    metrics = {'kind': 'diag02_original_model', 'completed': False, 'errors': [], 'cleanup_errors': [],
        'black_frame_detected': False, 'paper_result': False}
    sources, worker, code = [], None, 1
    install_signals()
    try:
        identity = execution_environment(os.environ, os.sched_getaffinity(0)); config['identity'] = identity
        deadline = deadline_after(args.timeout_seconds)
        signal.setitimer(signal.ITIMER_REAL, max(.1, deadline - time.time()))
        scratch = Path(identity['scratch']) / 'diag02' / args.name / identity['pbs_jobid']
        scratch.mkdir(parents=True, exist_ok=False)
        logs = ROOT / 'logs/runs' / args.name; logs.mkdir(parents=True, exist_ok=False)
        config['episode_directory'] = str(logs)
        gate, gate_ref, cleanup = logged_gate(args.name + '_gate', args.gate_timeout_seconds, scratch, deadline, sources)
        metrics.update(gate=gate_ref, cleanup_errors=cleanup)
        if gate['_meta']['exit_code'] != 0:
            metrics['black_frame_detected'] = gate['_meta']['exit_code'] == 86
            code = 86 if metrics['black_frame_detected'] and not cleanup else 1
        else:
            gate_runtime(gate, identity, config['node'], scratch / 'unity_logs')
            if cleanup:
                raise RuntimeError('Uncertain gate cleanup')
            worker = Worker(eval_worker, {**config, 'deadline': deadline - 20})
            while worker.result is None and worker.error is None and time.time() < deadline - 15:
                worker.receive(); time.sleep(.1)
            worker.receive()
            if worker.error or worker.result is None:
                raise RuntimeError('DIAG02 worker did not finish: ' + str(worker.error))
            result = copy.deepcopy(worker.result)
            sources.extend(ref for ref in result.pop('sources', []) if ref not in sources)
            metrics.update(result); code = 0 if metrics['completed'] else 1
    except BaseException as error:
        traceback.print_exc(); metrics['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if worker is not None:
            metrics['cleanup_errors'].extend(stop_workers([worker]))
        if metrics['errors'] or metrics['cleanup_errors']:
            code = 1; metrics['completed'] = False
        metrics['journals'] = [file_ref(path, ROOT / 'logs') for path in sorted((ROOT / 'logs/runs' / args.name).glob('*.jsonl'))]
        metrics['producer_exit_code'] = code
        save_json(output, {'metrics': metrics, 'config': config, 'sources': sources})
        print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
