"""Real pinned VAGEN PPO integration, imported only inside the PBS worker.

The smoke uses upstream trajectory tokenization, masked GAE, FSDP workers,
AdamW schedules and actor/critic update methods. No surrogate loss or fabricated
gradient is used. The legacy Direct path and the explicitly selected frozen
T05 section1b path are separate; neither certifies full T04/T05 acceptance.
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
from src.tools.smoke_e2e_scope import SmokeTableBackend, SmokeTableEnvironment, coverage
from src.tools.smoke_e2e_training_contract import (VERSION, episode_plan, prompt_identity,
    generation_groups, decision_token_spans, require_training_pool, task_reward, digest, available_actions,
    FROZEN_PIPELINE, FROZEN_VERSION, pre_evidence)
from src.envs.pair_table import parse_action


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
    def __init__(self, pipeline='legacy_direct'):
        self.version = FROZEN_VERSION if pipeline == FROZEN_PIPELINE else VERSION

    def get(self, key, default=None):
        return '<image>' if key == 'image_placeholder' else default

    def config_id(self):
        return self.version


class TableEpisode:
    config = EnvConfig()

    def __init__(self, env, plan, format_policy, pipeline="legacy_direct"):
        self.config = EnvConfig(pipeline)
        self.env, self.plan, self.format_policy = env, plan, format_policy
        self.observation = env.reset(plan['pair_id'], plan['world_id'])
        if pipeline == 'legacy_direct' and env.info['tau'] != 0:
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
              format_policy, journal, pipeline="legacy_direct"):
        self.envs, self.recorder, self.env_states = {}, defaultdict(list), {}
        self.plan, self.arm, self.update, self.journal = plan, arm, update, journal
        self.decisions = defaultdict(list)
        self.score_requests = []
        self.pipeline, self.scores_by_decision = pipeline, {}
        self.sample_count = self.generated_tokens = self.logical_tokens = 0
        for index, item in enumerate(plan):
            backend = SmokeTableBackend(tables, certificate=certificate, pair_checks=checks,
                                       report=report, authenticated_ids=identities)
            env = TableEpisode(SmokeTableEnvironment(backend), item, format_policy, pipeline)
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
                    if self.pipeline == 'legacy_direct' and before['tau'] != 0:
                        raise ValueError('Pre-evidence token-credit adapter is not authorized by this direct-only smoke')
                    try:
                        parsed = parse_action(raw_response.strip())
                    except ValueError:
                        parsed = None
                    is_pre = pre_evidence(turn, before['tau'])
                    if self.arm == 'T3' and is_pre and len(group) != 2:
                        raise ValueError('Paired pre-evidence prompts were not exactly shared')
                    choices = [text for text in available_actions(env.observation.text) if text.startswith('select ')]
                    needs_score = self.arm == 'T3' and (self.pipeline == 'legacy_direct' or (is_pre and parsed is not None and choices))
                    if needs_score:
                        choices = [text for text in available_actions(env.observation.text) if text.startswith('select ')]
                        names = {f'select {row.mark_label}': row.object_name for row in env.env.frame.candidates
                                 if row.success_eligible}
                        if any(choice not in names for choice in choices):
                            raise ValueError('Public mark and Q_commit candidate identity disagree')
                        self.score_requests.append({'env_id': index, 'turn': turn, 'recording': copy.deepcopy(self.recorder[index]),
                            'choices': [(text, names[text]) for text in choices], 'input': proofs[local],
                            'target_name': before['target_name'], 'used_for_pre_evidence_credit': self.pipeline == FROZEN_PIPELINE and is_pre})
                    env.env.record_model_call(sample_id)
                    obs, reward, done, info = env.step(raw_response)
                    self.env_states[index].update(step=turn + 1, done=done)
                    self.record(index, obs, reward, done, info)
                    trace = {'kind': 'decision', 'update': self.update, **self.plan[index],
                        'decision_index': turn, 'sample_id': sample_id, 'shared_sample': len(group) == 2,
                        'response_token_ids': ids, 'response_token_count': len(ids), 'response': raw_response,
                        'input': proofs[local], 'reward': reward, 'before': before, 'after': info,
                        'parse_valid': parsed is not None, 'action_kind': ('select' if parsed.kind == 'select' else 'observe') if parsed is not None else 'invalid',
                        'public_select_action_texts': choices, 'action_valid': info['action_valid']}
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
            q_value = q_commit(scores, request['target_name'])
            self.scores_by_decision[(request['env_id'], request['turn'])] = q_value
            self.journal({'kind': 'q_commit', 'update': self.update, **self.plan[request['env_id']],
                'decision_index': request['turn'], 'scores': scores.record(),
                'q_commit': q_value, 'used_for_pre_evidence_credit': request['used_for_pre_evidence_credit'],
                'physical_scoring_rpc_ids': sorted(call_ids_by_request[request_index]),
                'scoring_calls_scope': 'RPCs are shared by up to32 candidate events; total physical calls counted once per update'})
        return {'teacher_forced_scoring_calls': calls, 'candidate_action_scores': candidates}


def apply_frozen_credit(manager, batch, plan, arm, config, journal, update):
    """Bind the approved token pipeline to actual table trajectories and PPO rows."""
    from src.rl.token_credit import credit_pipeline
    length = batch.batch['responses'].shape[-1]
    decisions = []
    for row_index, env_id in enumerate(batch.non_tensor_batch['smoke_env_id']):
        spans = decision_token_spans(batch.batch['loss_mask'][row_index, -length:].tolist(),
            batch.batch['end_of_response_position_mask'][row_index, -length:].tolist())
        traces = manager.decisions[env_id]
        if len(spans) != len(traces):
            raise ValueError('Actual assistant spans do not cover every table decision')
        final_info = manager.envs[env_id].env.info
        for trace, span in zip(traces, spans):
            turn = trace['decision_index']
            before = trace['before']
            decisions.append({**plan[env_id], 'row_index': row_index, 'decision_index': turn,
                'token_indices': span, 'response_token_ids': batch.batch['responses'][row_index, span].tolist(),
                'tau': final_info['tau'], 'x_seen_step': before['x_seen_step'],
                'input_sha256': trace['input']['input_sha256'], 'shared_sample_id': trace['sample_id'],
                'shared_input_verified_exact': trace['shared_sample'], 'action_text': trace['response'].strip(),
                'parse_valid': trace['parse_valid'], 'action_valid': trace['action_valid'],
                'action_kind': trace['action_kind'], 'task_reward': trace['reward'],
                'G': sum(item['reward'] for item in traces[turn:]),
                'Q_commit': manager.scores_by_decision.get((env_id, turn)),
                'public_select_action_texts': trace['public_select_action_texts'],
                'target_name': before['target_name'], 'temporal_proof': before['table_temporal_proof']})
    result = credit_pipeline(arm=arm, token_level_rewards=batch.batch['token_level_rewards'],
        values=batch.batch['values'], loss_mask=batch.batch['loss_mask'][:, -length:],
        gae_mask=batch.batch['gae_mask'][:, -length:], decisions=decisions,
        gamma=config.algorithm.gamma, lam=config.algorithm.lam)
    # This bounded smoke fixes rho=.5. General unequal weights must also cover
    # the actor KL loss and are not silently approximated by scaling advantages.
    valid_weights = result['world_loss_weights'][batch.batch['loss_mask'][:, -length:].bool()]
    if not torch.equal(valid_weights, torch.ones_like(valid_weights)):
        raise ValueError('This E2E runtime only authorizes rho=.5; unequal actor-loss weights need the full T05 adapter')
    batch.batch['advantages'], batch.batch['returns'] = result['advantages'], result['returns']
    for decision in result['decision_records']:
        row, span = decision['row_index'], decision['token_indices']
        journal({'kind': 'actual_token_credit', 'update': update, 'arm': arm, **decision,
            'pre_evidence': pre_evidence(decision['decision_index'], decision['tau']),
            'raw_token_gae': result['raw_advantages'][row, span].tolist(),
            'replaced_token_advantages': result['replaced_advantages'][row, span].tolist(),
            'applied_token_advantages': result['advantages'][row, span].tolist(),
            'token_world_loss_weights': result['world_loss_weights'][row, span].tolist(),
            'critic_returns': result['returns'][row, span].tolist(),
            'values': batch.batch['values'][row, span].tolist(),
            'normalization': result['normalization'],
            'G_definition': 'actual task reward suffix, gamma1, no KL',
            'advantage_transform': 'raw_token_GAE_then_prefix_replacement_then_one_batch_masked_whiten'})
    return sum(pre_evidence(row['decision_index'], row['tau']) for row in decisions)


class SmokeTrainer(RayPPOTrainer):
    def _create_dataloader(self):
        # Five seeded explicit replay-pool batches replace the Navigation
        # parquet sampler. The real upstream optimizers and schedulers remain.
        self.total_training_steps = 5
        self.config.actor_rollout_ref.actor.optim.total_training_steps = 5
        self.config.critic.optim.total_training_steps = 5

    def smoke_fit(self, *, loaded, arm, format_policy, directory, checkpoint_directory, config_sha256, pipeline="legacy_direct"):
        tables, checks, report, identities, certificate, _ = loaded
        require_training_pool(tables, pipeline)
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
                    certificate=certificate, plan=plan, arm=arm, update=update, format_policy=format_policy, journal=journal, pipeline=pipeline)
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
                    pre_count = 0
                    if pipeline == FROZEN_PIPELINE:
                        if (self.config.algorithm.use_kl_in_reward is not False
                                or self.config.actor_rollout_ref.actor.use_kl_loss is not True
                                or self.config.actor_rollout_ref.actor.kl_loss_type != 'low_var_kl'):
                            raise ValueError('All frozen arms require task-only rewards and the same actor KL loss')
                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        pre_count = apply_frozen_credit(manager, batch, plan, arm, self.config, journal, update)
                        kl_metrics = {'kl_in_task_rewards': False, 'actor_kl_loss': True,
                            'actor_kl_coefficient': self.config.actor_rollout_ref.actor.kl_loss_coef}
                    else:
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
                            'unit_id': unit, 'episodes': episodes, 'observed_pre_evidence_decisions': sum(pre_evidence(d['decision_index'], manager.envs[i].env.info['tau']) for i, item in enumerate(plan) if item['episode_id'] in episodes for d in manager.decisions[i]),
                            'c_u': None if pipeline == FROZEN_PIPELINE and pre_count else 0.,
                            'basis': ('all pre-evidence token vectors and world weights retained in actual_token_credit for analyst S1 reduction; no unapproved fallback token aggregation' if pipeline == FROZEN_PIPELINE and pre_count else 'empty pre-evidence sum'),
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
                        'pre_evidence_decision_count': pre_count, 'pre_evidence_logic_covered': pre_count > 0,
                        'unparsable_output_count': sum(not d['parse_valid'] for ds in manager.decisions.values() for d in ds),
                        'actual_decision_count': sum(len(ds) for ds in manager.decisions.values()),
                        'unparsable_output_ratio': sum(not d['parse_valid'] for ds in manager.decisions.values() for d in ds) / sum(len(ds) for ds in manager.decisions.values()),
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
            'coverage': coverage(list(tables.values())),
            'critic_offload': bool(self.config.critic.model.fsdp_config.param_offload),
            'critic_offload_needed': None,
            'critic_offload_interpretation': 'this run records only its explicit setting; necessity requires a retained failed nonoffload run',
            'final_checkpoint_directory': str(checkpoint), 'pre_evidence_logic_covered': any(row['pre_evidence_logic_covered'] for row in updates),
            'pre_evidence_coverage_note': '实际证据前步有记录' if any(row['pre_evidence_logic_covered'] for row in updates) else '证据前逻辑未覆盖',
            's1_variance': None, 's1_variance_status': 'not_estimated; actual token vectors and independent unit membership retained for analyst',
            's2_due_updates': [], 'formal_T04_T05_acceptance': None, 'paper_result': False}
