"""Real pinned VAGEN PPO integration, imported only inside the PBS worker.

The smoke uses upstream trajectory tokenization, masked GAE, FSDP workers,
AdamW schedules and actor/critic update methods. No surrogate loss or fabricated
gradient is used. Only the human-approved direct-only scope is implemented.
"""
import copy
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
from verl.utils.seed import seed_everything
from vagen.rollout.qwen_rollout.rollout_manager import QwenVLRolloutManager
from vagen.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage

from src.rl.credit import ActionScores, CandidateScore, q_commit
from src.tools.smoke_e2e_eval import SYSTEM
from src.tools.smoke_e2e_scope import SmokeTableBackend, SmokeTableEnvironment
from src.tools.smoke_e2e_training_contract import (VERSION, episode_plan, prompt_identity,
    generation_groups, decision_token_spans, require_direct, task_reward, digest, available_actions)


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


class ObservedWorkerHooks(Worker):
    """Common direct base required by pinned verl's colocated WorkerDict.

    The upstream actor/critic remain next in each concrete worker's MRO, so
    their constructors and model/update implementations are unchanged.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        seed_everything(0)
        return super().init_model()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def smoke_memory(self, reset=False):
        torch.cuda.synchronize()
        result = {'device': torch.cuda.current_device(),
            'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'torch_allocated_bytes': torch.cuda.memory_allocated(),
            'scope': 'CUDA allocator in actual colocated PPO worker; not a sampled nvidia-smi whole-device peak'}
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return result


class ObservedActorWorker(ObservedWorkerHooks, ActorRolloutRefWorker):
    pass


class ObservedCriticWorker(ObservedWorkerHooks, CriticWorker):
    pass


class EnvConfig:
    def get(self, key, default=None):
        return '<image>' if key == 'image_placeholder' else default

    def config_id(self):
        return VERSION


class TableEpisode:
    config = EnvConfig()

    def __init__(self, env, plan, format_policy):
        self.env, self.plan, self.format_policy = env, plan, format_policy
        self.observation = env.reset(plan['pair_id'], plan['world_id'])
        if env.info['tau'] != 0:
            raise ValueError('Direct smoke must begin with actual tau=0')

    def system_prompt(self):
        return SYSTEM

    def obs(self):
        image = self.observation.image
        return {'obs_str': '<image>\n' + self.observation.text,
                'multi_modal_data': {'<image>': [Image.frombytes('RGB', (image.width, image.height), image.data)]}}

    def step(self, raw):
        self.observation, reward, done, info = self.env.step(raw.strip())
        reward = task_reward(reward, action_valid=info['action_valid'], format_policy=self.format_policy)
        return self.obs(), reward, done, {**info, 'llm_raw_response': raw,
            'metrics': {'turn_metrics': {}, 'traj_metrics': {}}}

    def compute_reward(self):
        return 0.  # select reward already belongs to its actual decision, never add twice.

    def close(self):
        self.env.close()


class TableRollouts(QwenVLRolloutManager):
    def begin(self, *, tables, checks, report, identities, certificate, plan, arm, update,
              format_policy, journal):
        self.envs, self.recorder, self.env_states = {}, defaultdict(list), {}
        self.plan, self.arm, self.update, self.journal = plan, arm, update, journal
        self.decisions = defaultdict(list)
        self.score_requests = []
        self.sample_count = self.generated_tokens = self.logical_tokens = 0
        for index, item in enumerate(plan):
            backend = SmokeTableBackend(tables, certificate=certificate, pair_checks=checks,
                                       report=report, authenticated_ids=identities)
            env = TableEpisode(SmokeTableEnvironment(backend), item, format_policy)
            self.envs[index] = env
            self.env_states[index] = {'step': 0, 'done': False, 'metrics': {'turn_metrics': {}, 'traj_metrics': {}}}
            self.record(index, env.obs(), 0., False, env.env.info)

    def rollout_loop(self):
        for turn in range(5):
            active = [i for i in self.envs if not self.env_states[i]['done']]
            if not active:
                break
            rows, identities, proofs = [], [], []
            for index in active:
                row = self._generate_input_for_rollout(self.recorder[index], turn, None)
                exact, sha, proof = prompt_identity(row['raw_prompt_ids'], row.get('multi_modal_data', {}).get('image', []))
                rows.append(row); identities.append(exact); proofs.append({'input_sha256': sha, **proof})
            groups = generation_groups([self.plan[i] for i in active], identities, paired=self.arm == 'T3')
            selected = [rows[group[0]] for group in groups]
            batch = DataProto.from_single_dict(collate_fn(selected))
            raw = np.empty(len(selected), dtype=object)
            for i, row in enumerate(selected):
                raw[i] = list(row['raw_prompt_ids'])
            batch.non_tensor_batch['raw_prompt_ids'] = raw
            output = self.actor_rollout_wg.generate_sequences(batch)
            response_length = output.batch['responses'].shape[-1]
            mask = output.batch['attention_mask'][:, -response_length:]
            for physical, group in enumerate(groups):
                ids = output.batch['responses'][physical][mask[physical].bool()].tolist()
                if not 0 < len(ids) <= 256:
                    raise ValueError('Actual generated reply escaped the 256-token bound')
                raw_response = self.tokenizer.decode(ids, skip_special_tokens=True)
                # Upstream removes special tokens before inserting responses
                # into subsequent public history. Never repair invalid actions.
                sample_id = f'{self.arm}_u{self.update}_sample{self.sample_count:06d}'
                self.sample_count += 1; self.generated_tokens += len(ids)
                for local in group:
                    index = active[local]
                    env = self.envs[index]
                    before = copy.deepcopy(env.env.info)
                    if before['tau'] != 0:
                        raise ValueError('Pre-evidence token-credit adapter is not authorized by this direct-only smoke')
                    if self.arm == 'T3':
                        choices = [text for text in available_actions(env.observation.text) if text.startswith('select ')]
                        names = {f'select {row.mark_label}': row.object_name for row in env.env.frame.candidates
                                 if row.success_eligible}
                        if any(choice not in names for choice in choices):
                            raise ValueError('Public mark and Q_commit candidate identity disagree')
                        self.score_requests.append({'env_id': index, 'turn': turn, 'recording': copy.deepcopy(self.recorder[index]),
                            'choices': [(text, names[text]) for text in choices], 'input': proofs[local],
                            'target_name': before['target_name']})
                    env.env.record_model_call(sample_id)
                    obs, reward, done, info = env.step(raw_response)
                    self.env_states[index].update(step=turn + 1, done=done)
                    self.record(index, obs, reward, done, info)
                    trace = {'kind': 'decision', 'update': self.update, **self.plan[index],
                        'decision_index': turn, 'sample_id': sample_id, 'shared_sample': len(group) == 2,
                        'response_token_ids': ids, 'response_token_count': len(ids), 'response': raw_response,
                        'input': proofs[local], 'reward': reward, 'before': before, 'after': info}
                    self.decisions[index].append(trace)
                    self.logical_tokens += len(ids)
                    self.journal(trace)
        if len(self.envs) != 64 or any(not state['done'] for state in self.env_states.values()):
            raise ValueError('Every actual batch must finish 64 episodes within five decisions')

    def score_commit(self, policy_id):
        """Teacher-force complete candidate action + EOS under the real actor."""
        calls, candidates, rows, bindings = 0, 0, [], []
        scores_by_request = defaultdict(list)
        call_ids_by_request = defaultdict(set)
        def flush():
            nonlocal calls, candidates
            if not rows:
                return
            batch = DataProto.from_single_dict(collate_fn(rows))
            output = self.actor_rollout_wg.compute_log_prob(batch)
            calls += 1
            length = batch.batch['responses'].shape[-1]
            for index, (request_index, text, name) in enumerate(bindings):
                request = self.score_requests[request_index]
                spans = decision_token_spans(batch.batch['loss_mask'][index, -length:].tolist(),
                    batch.batch['end_of_response_position_mask'][index, -length:].tolist())
                if len(spans) != request['turn'] + 1:
                    raise ValueError('Q_commit action history was truncated')
                token_ids = batch.batch['responses'][index, spans[-1]].tolist()
                expected = self.tokenizer.encode(text + self.tokenizer.eos_token, add_special_tokens=False)
                if token_ids != expected:
                    raise ValueError('Q_commit score does not cover the exact select text plus EOS')
                values = tuple(float(x) for x in output.batch['old_log_probs'][index, spans[-1]].tolist())
                scores_by_request[request_index].append(CandidateScore(name, text, sum(values), values))
                call_ids_by_request[request_index].add(calls)
                candidates += 1
            rows.clear(); bindings.clear()
        for request_index, request in enumerate(self.score_requests):
            for text, name in request['choices']:
                recording = copy.deepcopy(request['recording'])
                recording.append({'env_id': request['env_id'], 'done': True, 'reward': 0.,
                    'info': {'llm_raw_response': text}, 'obs_str': ''})
                rows.append(self._generate_input_for_uptate(recording, request['turn'] + 1, None))
                bindings.append((request_index, text, name))
                if len(rows) == 32:
                    flush()
        flush()
        for request_index, request in enumerate(self.score_requests):
            scores = ActionScores(tuple(scores_by_request[request_index]), request['input']['input_sha256'], policy_id,
                f'{policy_id}:q:{request["env_id"]}:{request["turn"]}',
                'pinned_verl_teacher_forced_complete_select_plus_eos_temperature0.7_untruncated_softmax_not_top_p',
                True, scoring_model_calls=len(call_ids_by_request[request_index]),
                public_select_action_texts=tuple(text for text, _ in request['choices']),
                candidate_domain_source='complete actual public Available actions at the decision')
            scores.validate()
            self.journal({'kind': 'q_commit', 'update': self.update, **self.plan[request['env_id']],
                'decision_index': request['turn'], 'scores': scores.record(),
                'q_commit': q_commit(scores, request['target_name']), 'used_for_pre_evidence_credit': False,
                'physical_scoring_rpc_ids': sorted(call_ids_by_request[request_index]),
                'scoring_calls_scope': 'RPCs are shared by up to32 candidate events; total physical calls counted once per update'})
        return {'teacher_forced_scoring_calls': calls, 'candidate_action_scores': candidates}


class SmokeTrainer(RayPPOTrainer):
    def _create_dataloader(self):
        # Five seeded explicit replay-pool batches replace the Navigation
        # parquet sampler. The real upstream optimizers and schedulers remain.
        self.total_training_steps = 5
        self.config.actor_rollout_ref.actor.optim.total_training_steps = 5
        self.config.critic.optim.total_training_steps = 5

    def smoke_fit(self, *, loaded, arm, format_policy, directory, checkpoint_directory, config_sha256):
        tables, checks, report, identities, certificate, _ = loaded
        require_direct(tables)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        updates = []
        for update in range(1, 6):
            path = directory / f'update_{update:02d}.jsonl'
            with path.open('x') as stream:
                def journal(row):
                    stream.write(json.dumps(json_value(row), sort_keys=True, allow_nan=False) + '\n'); stream.flush()
                started = time.monotonic()
                plan = episode_plan(tables, arm=arm, update=update)
                manager = TableRollouts(self.actor_rollout_wg, self.config.rollout_manager, self.tokenizer, self.processor)
                manager.begin(tables=tables, checks=checks, report=report, identities=identities,
                    certificate=certificate, plan=plan, arm=arm, update=update, format_policy=format_policy, journal=journal)
                policy_id = f'{arm}:seed0:actual_completed_updates{update - 1}'
                journal({'kind': 'update_begin', 'update': update, 'policy_id': policy_id,
                         'plan': plan, 'common_smoke_config_sha256': config_sha256})
                self.actor_rollout_wg.smoke_memory(reset=True)
                try:
                    manager.rollout_loop()
                    batch = manager.generate_batch_for_update()
                    batch.non_tensor_batch['smoke_env_id'] = np.array(list(manager.envs), dtype=object)
                    generated = time.monotonic()
                    scoring = manager.score_commit(policy_id)
                    metrics = {}
                    self._balance_batch(batch, metrics)
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    batch = batch.union(self.actor_rollout_wg.compute_log_prob(batch))
                    batch = batch.union(self.ref_policy_wg.compute_ref_log_prob(batch))
                    batch = batch.union(self.critic_wg.compute_values(batch))
                    length = batch.batch['responses'].shape[-1]
                    batch.batch['token_level_scores'] = batch.batch['multi_turn_token_level_rewards'][:, -length:]
                    batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                    batch = compute_advantage(batch, self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma, lam=self.config.algorithm.lam,
                        num_repeat=1, high_level_gamma=self.config.algorithm.high_level_gamma)
                    # Keep the COMPLETE upstream token advantages and critic
                    # returns. tau=0 means no T3 replacement is applicable.
                    for row_index, env_id in enumerate(batch.non_tensor_batch['smoke_env_id']):
                        spans = decision_token_spans(batch.batch['loss_mask'][row_index, -length:].tolist(),
                            batch.batch['end_of_response_position_mask'][row_index, -length:].tolist())
                        if len(spans) != len(manager.decisions[env_id]):
                            raise ValueError('Actual assistant spans do not cover every decision')
                        for turn, span in enumerate(spans):
                            journal({'kind': 'actual_token_credit', 'update': update, **plan[env_id],
                                'decision_index': turn, 'tau': 0, 'pre_evidence': False,
                                'token_indices': span, 'upstream_masked_gae_advantages': batch.batch['advantages'][row_index, span].tolist(),
                                'critic_returns': batch.batch['returns'][row_index, span].tolist(),
                                'values': batch.batch['values'][row_index, span].tolist(),
                                'advantage_transform': 'identity_all_tokens_post_tau',
                                'upstream_whitening': 'pinned_masked_GAE_whitens_over_all_loss_tokens'})
                    units = {}
                    for index, item in enumerate(plan):
                        unit = item['pair_instance_id'] if arm == 'T3' else item['episode_id']
                        units.setdefault(unit, []).append(item['episode_id'])
                        env = manager.envs[index].env
                        journal({'kind': 'episode', 'update': update, **item, 'tau': env.info['tau'],
                            'x_seen_step': env.info['x_seen_step'], 'selected_name': env.selected_name,
                            'success': env.success, 'steps': env.steps, 'model_calls': env.model_calls})
                    for unit, episodes in units.items():
                        journal({'kind': 's1_independent_decision_unit', 'update': update,
                            'unit_id': unit, 'episodes': episodes, 'observed_pre_evidence_decisions': 0,
                            'c_u': 0., 'basis': 'all actual episode decisions authenticated tau=0; empty pre-evidence sum',
                            'episode_equivalent_divisor': 2 if arm == 'T3' else 1})
                    scored = time.monotonic()
                    critic_output = self.critic_wg.update_critic(batch)
                    critic_finished = time.monotonic()
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                    finished = time.monotonic()
                    row = json_value({'kind': 'update_complete', 'update': update, 'actual_completed_updates': update,
                        'generation_seconds': generated - started, 'scoring_seconds': scored - generated,
                        'update_seconds': finished - scored, 'critic_update_seconds': critic_finished - scored,
                        'actor_update_seconds': finished - critic_finished, 'total_seconds': finished - started,
                        'memory': self.actor_rollout_wg.smoke_memory(reset=False),
                        'physical_policy_sampling_calls': manager.sample_count,
                        'actual_generated_token_count': manager.generated_tokens,
                        'logical_trajectory_token_count': manager.logical_tokens, **scoring,
                        'episodes': len(plan), 'paired_instances': len(units),
                        'pre_evidence_decision_count': 0, 'pre_evidence_logic_covered': False,
                        's2_interval_updates': 20, 's2_due': False,
                        'critic_metrics': critic_output.meta_info['metrics'],
                        'actor_metrics': actor_output.meta_info['metrics'], 'kl_metrics': kl_metrics})
                    journal(row); updates.append(row)
                    print('E2E_PPO_UPDATE: ' + json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
                except BaseException as error:
                    journal({'kind': 'update_failed', 'update': update, 'completed_updates': len(updates),
                             'error': f'{type(error).__name__}: {error}'})
                    raise
                finally:
                    for env in manager.envs.values():
                        env.close()
        # Only one final checkpoint; no previous checkpoint is deleted.
        checkpoint = Path(checkpoint_directory)
        self.actor_rollout_wg.save_checkpoint(str(checkpoint / 'actor'), None, 5, remove_previous_ckpt=False)
        self.critic_wg.save_checkpoint(str(checkpoint / 'critic'), None, 5, remove_previous_ckpt=False)
        return {'actual_completed_updates': len(updates), 'updates': updates,
            'critic_offload': bool(self.config.critic.model.fsdp_config.param_offload),
            'critic_offload_needed': None,
            'critic_offload_interpretation': 'this run records only its explicit setting; necessity requires a retained failed nonoffload run',
            'final_checkpoint_directory': str(checkpoint), 'pre_evidence_logic_covered': False,
            'pre_evidence_coverage_note': '证据前逻辑未覆盖；当前五对全部 tau=0，Look 对需补跑第5步',
            's1_variance': None, 's1_variance_status': 'not_estimated; zero-pre-evidence unit sums are not a variance endpoint',
            's2_due_updates': [], 'formal_T04_T05_acceptance': None, 'paper_result': False}
