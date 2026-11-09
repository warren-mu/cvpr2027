"""E2E step 3: original Qwen2.5-VL, three policies, fixed approved paired episodes.

All model work occurs in one owned PBS worker after the per-card real-render gate.
Only public marked RGB/text/history enter prompts. This is not a frozen test-set
assessment or a paper metric. CPU fixtures use an injected policy, never a model.
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
from src.tools.t01_partition import audited_result
from src.tools.t02_export import save_json, execution_environment, logged_gate, gate_runtime
from src.tools.smoke_e2e_views import load_success, file_ref
from src.tools.smoke_e2e_replay import python_args
from src.tools.smoke_e2e_scope import SmokeTableBackend, SmokeTableEnvironment, coverage

VERSION = 'smoke_e2e_original_model_eval_v1'
MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'
STRATEGIES = ('obs_never', 'obs_always', 'diag_base')
EXPLORATION = ('turn_left', 'turn_right', 'turn_right', 'turn_left')
SYSTEM = ('You control an embodied agent. Follow the instruction using the numbered marks in the images. '
          'At each turn, reply with exactly one action from Available actions, with no explanation. '
          'Use select <number> to finish. Observation actions return another image. '
          'All earlier images and actions are part of the same episode.')


def load_replay(path):
    data, ref = audited_result(path)
    meta, cfg, m = data['_meta'], data['config'], data['metrics']
    expected = ['-m', 'src.tools.smoke_e2e_replay', '--name', meta['name'],
                '--source', cfg['source'], '--contract-tests', cfg['contract_tests']]
    if (python_args(meta['command']) != expected or meta['exit_code'] != 0
            or m.get('step2_passed') is not True or m.get('producer_exit_code') != 0
            or m.get('errors') != [] or m.get('formal_T02_acceptance') is not None):
        raise ValueError('A successful audited E2E step 2 is required before model evaluation')
    tables, packages, checks, report, identities, certificate, sources = load_success(ROOT / cfg['source'])
    expected_worlds = {(pair, f'w{world}') for pair in tables for world in (0, 1)}
    rows = m.get('world_replays', [])
    if (len(rows) != len(expected_worlds)
            or {(row.get('pair_id'), row.get('world_id')) for row in rows} != expected_worlds
            or any(row.get('success') is not True or not 1 <= row.get('steps', 0) <= 5 for row in rows)):
        raise ValueError('Step 2 does not cover every reviewed pair in both worlds')
    if ref not in sources:
        sources.append(ref)
    return tables, checks, report, identities, certificate, sources


def public_actions(observation):
    lines = [line for line in observation.text.splitlines() if line.startswith('Available actions: ')]
    if len(lines) != 1:
        raise ValueError('Public observation has no unambiguous action list')
    values = json.loads(lines[0].split(': ', 1)[1])
    if (not isinstance(values, list) or not values or len(set(values)) != len(values)
            or any(not isinstance(item, str) for item in values)):
        raise ValueError('Malformed public action list')
    return values


def prompt_messages(observations, responses, *, select_only):
    if len(observations) != len(responses) + 1:
        raise ValueError('Complete public observation/response history required')
    messages = [{'role': 'system', 'content': SYSTEM}]
    for index, observation in enumerate(observations):
        text = observation.text
        if index == len(observations) - 1 and select_only:
            text += '\nChoose one currently marked object now; reply only with select <number>.'
        messages.append({'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]})
        if index < len(responses):
            messages.append({'role': 'assistant', 'content': responses[index]})
    return messages


def collect_strategy(strategy, environments, policy_batch, journal):
    """Batch real active decisions; forced exploration contains no model call."""
    if strategy not in STRATEGIES:
        raise ValueError('Unknown smoke policy')
    active = []
    for pair, world, env in environments:
        observation = env.reset(pair, world)
        active.append({'pair_id': pair, 'world_id': world, 'env': env,
                       'observations': [observation], 'responses': [], 'trace': []})
    episodes = []
    for decision in range(5):
        pending, generated = [], {}
        for index, item in enumerate(active):
            if item['env'].done:
                continue
            forced = strategy == 'obs_always' and decision < 4
            if forced:
                action = EXPLORATION[decision]
                if action not in public_actions(item['observations'][-1]):
                    raise ValueError('The fixed D4 return-to-start exploration action is unavailable')
                generated[index] = {'action': action, 'response': action, 'forced_observation': True,
                                    'response_token_ids': [], 'response_token_count': 0, 'sample_id': None}
            else:
                pending.append((index, {'observations': item['observations'], 'responses': item['responses'],
                                        'select_only': strategy != 'diag_base'}))
        if pending:
            outputs = policy_batch([row for _, row in pending])
            if len(outputs) != len(pending):
                raise ValueError('Model omitted/added an episode output')
            for (index, _), sample in zip(pending, outputs):
                if (not isinstance(sample.get('response'), str) or not isinstance(sample.get('action'), str)
                        or not sample.get('sample_id') or sample.get('forced_observation') is not False
                        or not isinstance(sample.get('response_token_ids'), list)
                        or sample.get('response_token_count') != len(sample['response_token_ids'])
                        or not 0 < sample['response_token_count'] <= 256):
                    raise ValueError('Incomplete actual model sample or token bound violated')
                if strategy != 'diag_base' and sample['action'] not in [
                        action for action in public_actions(active[index]['observations'][-1]) if action.startswith('select ')]:
                    raise ValueError('Forced-select decoding did not return a public select action')
                generated[index] = sample
        for index, item in enumerate(active):
            env = item['env']
            if env.done:
                continue
            before = copy.deepcopy(env.info)
            sample = generated[index]
            if not sample['forced_observation']:
                env.record_model_call(sample['sample_id'])
            observation, reward, done, info = env.step(sample['action'])
            trace = {'kind': 'decision', 'strategy': strategy, 'pair_id': item['pair_id'],
                     'world_id': item['world_id'], 'episode_id': before['episode_id'],
                     'decision_index': decision, 'before_state_id': before['state_id'],
                     'after_state_id': info['state_id'], 'tau_before': before['tau'],
                     'x_seen_step_before': before['x_seen_step'], 'tau': info['tau'],
                     'x_seen_step': info['x_seen_step'], 'sample': sample, 'reward': reward,
                     'selected_object': info['selected_name'], 'success': info['success'], 'done': done,
                     'table_temporal_proof': info['table_temporal_proof'],
                     'public_rgb_sha256': item['observations'][-1].image.sha256,
                     'public_text': item['observations'][-1].text}
            journal(trace)
            item['trace'].append(trace)
            item['responses'].append(sample['response'])
            item['observations'].append(observation)
            if done:
                row = {'kind': 'episode', 'strategy': strategy, 'pair_id': item['pair_id'],
                       'world_id': item['world_id'], 'episode_id': info['episode_id'],
                       'pair_kind': before['kind'], 'tau': info['tau'], 'x_seen_step': info['x_seen_step'],
                       'selected_object': info['selected_name'], 'selected': info['selected'],
                       'success': info['success'], 'steps': env.steps, 'model_calls': env.model_calls,
                       'observation_count': info['obs_count'], 'trace': item['trace']}
                journal(row)
                episodes.append(row)
        if all(item['env'].done for item in active):
            break
    if len(episodes) != len(active) or not all(item['env'].done for item in active):
        raise ValueError('A smoke episode failed to terminate within D+1')
    return episodes


class VllmPolicy:
    def __init__(self, model_path):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.sample_type, self.guide_type = SamplingParams, GuidedDecodingParams
        self.model = LLM(model=str(model_path), tokenizer=str(model_path), tensor_parallel_size=1,
                         dtype='bfloat16', seed=0, gpu_memory_utilization=.4, max_model_len=8192,
                         max_num_seqs=10, limit_mm_per_prompt={'image': 5}, enforce_eager=True,
                         trust_remote_code=False)
        self.serial = 0

    def __call__(self, contexts):
        from PIL import Image
        requests, parameters, proofs = [], [], []
        for context in contexts:
            observations = context['observations']
            messages = prompt_messages(observations, context['responses'], select_only=context['select_only'])
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            choices = [a for a in public_actions(observations[-1]) if a.startswith('select ')]
            if context['select_only'] and not choices:
                raise ValueError('Forced-selection episode has no public selectable mark')
            images = [Image.frombytes('RGB', (obs.image.width, obs.image.height), obs.image.data) for obs in observations]
            requests.append({'prompt': prompt, 'multi_modal_data': {'image': images}})
            parameters.append(self.sample_type(temperature=0., top_p=1., max_tokens=256, n=1,
                guided_decoding=self.guide_type(choice=choices) if context['select_only'] else None))
            proofs.append({'prompt_utf8_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'chat_template_sha256': hashlib.sha256(str(self.processor.chat_template).encode()).hexdigest(),
                'public_image_sha256': [obs.image.sha256 for obs in observations],
                'text_token_ids_before_multimodal_expansion': self.processor.tokenizer.encode(prompt, add_special_tokens=False),
                'guided_select_choices': choices if context['select_only'] else None,
                'prompt_contract': VERSION})
        outputs = self.model.generate(requests, parameters, use_tqdm=False)
        result = []
        if len(outputs) != len(requests):
            raise ValueError('vLLM output batch length mismatch')
        for request, proof in zip(outputs, proofs):
            if len(request.outputs) != 1:
                raise ValueError('Expected exactly one greedy output')
            sample = request.outputs[0]
            self.serial += 1
            result.append({'response': sample.text, 'action': sample.text.strip(),
                'response_token_ids': list(sample.token_ids), 'response_token_count': len(sample.token_ids),
                'finish_reason': sample.finish_reason, 'stop_reason': sample.stop_reason,
                'sample_id': f'model_sample_{self.serial:06d}', 'forced_observation': False,
                'policy': 'original_Qwen_greedy_vllm', 'input': proof})
        return result


def eval_worker(config, connection):
    os.sched_setaffinity(0, config['identity']['cpu_affinity'])
    tables, checks, report, identities, certificate, sources = load_replay(ROOT / config['replay'])
    scratch = Path(config['identity']['scratch']).resolve()
    from huggingface_hub import snapshot_download
    model_path = Path(snapshot_download(MODEL, local_files_only=True)).resolve()
    if not model_path.is_relative_to(scratch):
        raise ValueError('Model weights must already be cached on scratch')
    model_config = json.loads((model_path / 'config.json').read_text())
    if model_config.get('model_type') != 'qwen2_5_vl':
        raise ValueError('Cached model is not Qwen2.5-VL')
    started = time.monotonic()
    policy = VllmPolicy(model_path)
    loaded = time.monotonic() - started
    rows, backends = {}, []
    log = Path(config['episode_log'])
    with log.open('x') as stream:
        def journal(row):
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n'); stream.flush()
        try:
            for strategy in STRATEGIES:
                environments = []
                for pair in tables:
                    for world in ('w0', 'w1'):
                        backend = SmokeTableBackend(tables, certificate=certificate, pair_checks=checks,
                            report=report, authenticated_ids=identities)
                        backends.append(backend)
                        environments.append((pair, world, SmokeTableEnvironment(backend)))
                rows[strategy] = collect_strategy(strategy, environments, policy, journal)
        finally:
            for backend in backends:
                backend.close()
    return {'status': 'step3_passed', 'step3_passed': True, 'model': MODEL,
            'model_snapshot_directory': str(model_path), 'model_snapshot_revision': model_path.name,
            'model_config_sha256': hashlib.sha256((model_path / 'config.json').read_bytes()).hexdigest(),
            'model_loading_seconds': loaded, 'strategies': rows, 'sources': sources,
            'coverage': coverage(list(tables.values())), 'episode_log': file_ref(log, ROOT / 'logs'),
            'evaluation_design': 'exactly one episode for every approved pair/world, equal world weights r=0.5',
            'formal_T03_acceptance': None, 'paper_result': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--replay', required=True)
    parser.add_argument('--timeout-seconds', type=float, required=True)
    parser.add_argument('--gate-timeout-seconds', type=float, default=180.)
    args = parser.parse_args()
    if (not re.fullmatch(r'smoke_e2e_[A-Za-z0-9_]+', args.name)
            or not 300 <= args.timeout_seconds <= 7200
            or not 15 <= args.gate_timeout_seconds <= args.timeout_seconds - 60):
        parser.error('Require a unique smoke name and bounded one-GPU deadline')
    output = ROOT / 'results' / (args.name + '.json')
    if output.exists():
        raise FileExistsError('Preserve previous result; use another run name')
    config = {**vars(args), 'version': VERSION, 'model': MODEL, 'gpus': 1, 'strategies': list(STRATEGIES),
              'episode_rounds': 5, 'max_reply_tokens': 256, 'generation': 'greedy_vllm',
              'prompt_system': SYSTEM, 'vllm_memory_utilization': .4, 'vllm_max_model_len': 8192,
              'vllm_max_num_seqs': 10, 'world_weight': .5, 'node': socket.gethostname()}
    metrics = {'kind': 'smoke_e2e_eval', 'step3_passed': False, 'errors': [], 'cleanup_errors': [],
               'gate': None, 'black_frame_detected': False, 'steps4_5_started': False,
               'formal_T03_acceptance': None, 'paper_result': False}
    sources, worker, code = [], None, 1
    install_signals()
    try:
        identity = execution_environment(os.environ, os.sched_getaffinity(0))
        config['identity'] = identity
        deadline = deadline_after(args.timeout_seconds)
        signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
        directory = Path(identity['scratch']) / 'smoke_e2e' / args.name / identity['pbs_jobid']
        directory.mkdir(parents=True, exist_ok=False)
        logs = ROOT / 'logs/runs' / args.name
        logs.mkdir(parents=True, exist_ok=False)
        config['episode_log'] = str(logs / 'episodes.jsonl')
        gate, gate_ref, cleanup = logged_gate(args.name + '_gate', args.gate_timeout_seconds, directory, deadline, sources)
        metrics.update(gate=gate_ref, cleanup_errors=cleanup)
        if gate['_meta']['exit_code'] != 0:
            metrics['black_frame_detected'] = gate['_meta']['exit_code'] == 86
            code = 86 if metrics['black_frame_detected'] and not cleanup else 1
        else:
            gate_runtime(gate, identity, config['node'], directory / 'unity_logs')
            if cleanup:
                raise RuntimeError('Gate cleanup uncertain')
            worker = Worker(eval_worker, {**config, 'deadline': deadline - 20})
            while worker.result is None and worker.error is None and time.time() < deadline - 15:
                worker.receive(); time.sleep(.1)
            worker.receive()
            if worker.error or worker.result is None:
                raise RuntimeError('Model worker did not finish: ' + str(worker.error))
            result = worker.result
            sources.extend(ref for ref in result.pop('sources', []) if ref not in sources)
            metrics.update(result)
            code = 0 if metrics.get('step3_passed') is True else 1
    except BaseException as error:
        traceback.print_exc()
        metrics['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if worker is not None:
            metrics['cleanup_errors'].extend(stop_workers([worker]))
        if metrics['errors'] or metrics['cleanup_errors']:
            code = 1; metrics['step3_passed'] = False
        log = Path(config.get('episode_log', '/nonexistent'))
        if log.is_file():
            metrics['episode_log'] = file_ref(log, ROOT / 'logs')
        metrics['producer_exit_code'] = code
        output.parent.mkdir(parents=True, exist_ok=True)
        save_json(output, {'metrics': metrics, 'config': config, 'sources': sources})
        print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
