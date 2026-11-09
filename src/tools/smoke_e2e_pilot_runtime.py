"""Real one-GPU E2E pilot using formal P tables and pinned VAGEN PPO."""
import copy
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.single_controller.base.decorator import register, Dispatch
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from src.envs.pair_table import TableEnvironment, TableObservationBackend
from src.envs.thor_common import ROOT
from src.tools.smoke_e2e_views import file_ref
from src.rl import t4_consistency as consistency
from src.tools.smoke_e2e_pilot_actor import update_support, require_completed_optimizer_steps
from src.tools.smoke_e2e_pilot_contract import (execution_schedule, training_plan,
    holdout_plan, episode_observation, classified_observations)
from src.tools.smoke_e2e_training_contract import FROZEN_PIPELINE, decision_token_spans, pre_evidence
from src.tools.smoke_e2e_training_runtime import (ObservedWorkerHooks, TableEpisode,
    TableRollouts, SmokeTrainer, apply_frozen_credit, json_value)


class PilotActorWorker(ObservedWorkerHooks, ActorRolloutRefWorker):
    # The immediate common Worker base is required by pinned colocated WorkerDict.
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data):
        if self.world_size != 1:
            raise ValueError('Pilot actor hook is limited to its explicit single GPU')
        with update_support(self.actor, data) as record:
            output = super().update_actor(data)
        output.meta_info['pilot_auxiliary_record'] = record
        return output


class PilotRollouts(TableRollouts):
    paired_arms = ('T4',)
    q_commit_arms = ('T2', 'T4')

    def begin_pilot(self, *, tables, world_orders, plan, arm, update, journal, phase):
        self.envs, self.recorder, self.env_states = {}, defaultdict(list), {}
        self.plan, self.arm, self.update, self.journal = plan, arm, update, journal
        self.decisions, self.score_requests, self.scores_by_decision = defaultdict(list), [], {}
        self.pipeline = FROZEN_PIPELINE
        self.sample_count = self.generated_tokens = self.logical_tokens = 0
        self.expected_episode_count = len(plan)
        self.sample_prefix = f'{arm}_pilot_{phase}'
        self.generation_metadata = {'do_sample': False} if phase == 'holdout' else {}
        if phase == 'holdout':
            self.paired_arms, self.q_commit_arms = (), ()
        for index, item in enumerate(plan):
            backend = TableObservationBackend(tables, world_orders=world_orders)
            env = TableEpisode(TableEnvironment(backend), item, 'explicit_task_zero', FROZEN_PIPELINE)
            self.envs[index] = env
            self.env_states[index] = {'step': 0, 'done': False, 'metrics': {'turn_metrics': {}, 'traj_metrics': {}}}
            self.record(index, env.obs(), 0., False, env.env.info)

    def observed_episodes(self):
        rows = []
        for index, item in enumerate(self.plan):
            env = self.envs[index].env
            observed = episode_observation(item, self.decisions[index], kind=env.info['kind'],
                final_info=env.info, selected_name=env.selected_name, success=env.success)
            observed.update(original_pair_id=env.backend.table.pair_id,
                world_order=list(env.backend.world_order), original_world_id=f'w{env.backend.world}',
                target_name=env.info['target_name'], model_calls=env.model_calls,
                steps=env.steps, temporal_proof=env.info['table_temporal_proof'])
            self.journal({'record_kind': 'pilot_episode', 'update': self.update, **observed})
            rows.append(observed)
        return rows

    def prepare_consistency(self, policy_id):
        if self.arm != 'T4':
            return {'arm': self.arm}, {'applicable_flip_terms': 0, 'applicable_stay_terms': 0,
                'skip_reasons': {}, 'sampled_opposite_scoring_rpcs': 0, 'sampled_candidate_actions': 0}
        episodes, request_by_episode = [], {}
        for index, item in enumerate(self.plan):
            env = self.envs[index].env
            legal = [row for row in self.decisions[index]
                     if row['parse_valid'] and row['action_valid'] and row['action_kind'] == 'select']
            if len(legal) > 1:
                raise ValueError('Table trajectory contains more than one terminal legal select')
            last = None
            if legal:
                trace = legal[-1]
                by_action = {row['action_text']: row['object_name'] for row in trace['before']['candidates']
                             if row['action_text'] is not None and row['action_legal']}
                choices = [(text, by_action[text]) for text in trace['public_select_action_texts']]
                last = {'action_kind': 'select', 'action_valid': True,
                    'decision_index': trace['decision_index'], 'x_seen_step': trace['before']['x_seen_step'],
                    'public_select_candidate_names': [name for _, name in choices]}
                request_by_episode[item['episode_id']] = {'env_id': index, 'trace': trace, 'choices': choices,
                    'recording': copy.deepcopy(self.recorder[index][:trace['decision_index'] + 1])}
            episodes.append({**item, 'kind': env.info['kind'], 'candidate_names': list(env.backend.table.candidate_names),
                'target_name': env.info['target_name'], 'last_legal_select': last})
        descriptors = consistency.descriptors(episodes)
        bundles, scores, queued, binding = {}, {}, [], []
        calls = candidate_count = 0
        def flush():
            nonlocal calls, candidate_count
            if not queued:
                return
            batch = DataProto.from_single_dict(collate_fn(queued))
            output = self.actor_rollout_wg.compute_log_prob(batch)
            calls += 1
            length = batch.batch['responses'].shape[-1]
            for row_index, (episode, action) in enumerate(binding):
                request = request_by_episode[episode]
                spans = decision_token_spans(batch.batch['loss_mask'][row_index, -length:].tolist(),
                    batch.batch['end_of_response_position_mask'][row_index, -length:].tolist())
                if (len(spans) != request['trace']['decision_index'] + 1
                        or batch.batch['responses'][row_index, spans[-1]].tolist() != action['expected_complete_token_ids']):
                    raise ValueError('Sampled opposite score lost its exact candidate/EOS/history')
                value = float(output.batch['old_log_probs'][row_index, spans[-1]].sum())
                scores.setdefault(episode, {})[action['object_name']] = value
                candidate_count += 1
            queued.clear(); binding.clear()
        for episode, descriptor in descriptors.items():
            self.journal({'record_kind': 't4_consistency_eligibility', 'update': self.update,
                          'policy_id': policy_id, **descriptor})
            if not descriptor['applicable']:
                continue
            request = request_by_episode[episode]
            rows, actions = [], []
            for text, name in request['choices']:
                recording = copy.deepcopy(request['recording'])
                recording.append({'env_id': request['env_id'], 'done': True, 'reward': 0.,
                    'info': {'llm_raw_response': text}, 'obs_str': ''})
                row = self._generate_input_for_uptate(recording, request['trace']['decision_index'] + 1, None)
                action = {'text': text, 'object_name': name,
                    'expected_complete_token_ids': self.tokenizer.encode(text + self.tokenizer.eos_token, add_special_tokens=False)}
                rows.append(row); actions.append(action)
                if descriptor['change'] == 'stay':
                    queued.append(row); binding.append((episode, action))
                    if len(queued) == 32:
                        flush()
            bundles[episode] = {'batch': DataProto.from_single_dict(collate_fn(rows)), 'actions': actions,
                'decision_index': request['trace']['decision_index'], 'input_proof': request['trace']['input']}
        flush()
        for episode, values in scores.items():
            self.journal({'record_kind': 't4_sampled_policy_scores', 'update': self.update,
                'episode_id': episode, 'policy_id': policy_id, 'action_log_probs': values,
                'input_proof': request_by_episode[episode]['trace']['input'],
                'scope': 'complete select plus EOS under sampled policy; detached reference for opposite episode'})
        applicable = Counter(row['change'] for row in descriptors.values() if row['applicable'])
        skipped = Counter(row['skip_reason'] for row in descriptors.values() if not row['applicable'])
        return {'arm': self.arm, 'consistency_descriptors': descriptors,
                'candidate_inputs': bundles, 'sampled_action_scores': scores}, {
            'applicable_flip_terms': applicable['flip'], 'applicable_stay_terms': applicable['stay'],
            'skip_reasons': dict(skipped), 'sampled_opposite_scoring_rpcs': calls,
            'sampled_candidate_actions': candidate_count}


class PilotTrainer(SmokeTrainer):
    def _create_dataloader(self):
        # T2 executes only5 updates on the same common50-step optimizer horizon.
        self.total_training_steps = 50
        self.config.actor_rollout_ref.actor.optim.total_training_steps = 50
        self.config.critic.optim.total_training_steps = 50

    def pilot_fit(self, *, tables, dataset_proof, arm, directory, checkpoint_directory, common_hash,
                  engineering_smoke=None):
        update_count, holdout_updates = execution_schedule(arm, engineering_smoke)
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
        train_keys, eval_keys = dataset_proof['train_keys'], dataset_proof['eval_keys']
        if (set(train_keys) & set(eval_keys) or set(train_keys) | set(eval_keys) != set(tables)
                or dataset_proof.get('cache_authorized') is not True):
            raise ValueError('Formal pilot pool must have its complete disjoint authenticated split')
        world_orders = dataset_proof['world_orders']
        updates, evaluations, memory = [], [], []
        def new_manager(plan, update, journal, phase):
            manager = PilotRollouts(self.actor_rollout_wg, self.config.rollout_manager, self.tokenizer, self.processor)
            manager.begin_pilot(tables=tables, world_orders=world_orders, plan=plan,
                arm=arm, update=update, journal=journal, phase=phase)
            return manager
        def evaluate(update):
            plan = holdout_plan(eval_keys, arm=arm, update=update)
            path = directory / f'holdout_{update:03d}.jsonl'
            with path.open('x') as stream:
                def journal(row):
                    stream.write(json.dumps(json_value(row), sort_keys=True, allow_nan=False) + '\n'); stream.flush()
                manager = new_manager(plan, update, journal, 'holdout')
                started = time.monotonic()
                try:
                    manager.rollout_loop()
                    episodes = manager.observed_episodes()
                    if {(row['pair_id'], row['world_id']) for row in episodes} != {(key, world) for key in eval_keys for world in ('w0', 'w1')}:
                        raise ValueError('Greedy holdout did not cover exactly every held-out pair/world')
                    result = {'update': update, 'greedy': True, 'complete': True, 'episodes': len(episodes),
                        'by_class': classified_observations(episodes), 'seconds': time.monotonic() - started,
                        'actual_generated_token_count': manager.generated_tokens,
                        'physical_policy_sampling_calls': manager.sample_count,
                        'scope': 'pilot held-out training scenes only; not frozen test or paper results'}
                    journal({'record_kind': 'pilot_holdout_complete', **result})
                finally:
                    for env in manager.envs.values(): env.close()
            result['episode_log'] = file_ref(path, ROOT / 'logs'); evaluations.append(result)
        if 0 in holdout_updates: evaluate(0)
        for update in range(1, update_count + 1):
            path = directory / f'update_{update:03d}.jsonl'
            with path.open('x') as stream:
                def journal(row):
                    stream.write(json.dumps(json_value(row), sort_keys=True, allow_nan=False) + '\n'); stream.flush()
                plan = training_plan(train_keys, arm=arm, update=update)
                manager = new_manager(plan, update, journal, 'train')
                started = time.monotonic(); policy_id = f'{arm}:pilot:seed0:completed_updates{update - 1}'
                journal({'record_kind': 'pilot_update_begin', 'update': update, 'plan': plan,
                    'policy_id': policy_id, 'common_config_sha256': common_hash})
                self.actor_rollout_wg.smoke_memory(reset=True)
                try:
                    manager.rollout_loop()
                    batch = manager.generate_batch_for_update()
                    batch.non_tensor_batch['smoke_env_id'] = np.array(list(manager.envs), dtype=object)
                    batch.non_tensor_batch['pilot_episode_id'] = np.array([row['episode_id'] for row in plan], dtype=object)
                    generated = time.monotonic()
                    scoring = manager.score_commit(policy_id)
                    auxiliary, aux_counts = manager.prepare_consistency(policy_id)
                    metrics = {}; self._balance_batch(batch, metrics)
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    batch = batch.union(self.actor_rollout_wg.compute_log_prob(batch))
                    batch = batch.union(self.ref_policy_wg.compute_ref_log_prob(batch))
                    batch = batch.union(self.critic_wg.compute_values(batch))
                    length = batch.batch['responses'].shape[-1]
                    batch.batch['token_level_scores'] = batch.batch['multi_turn_token_level_rewards'][:, -length:]
                    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                    if (self.config.algorithm.use_kl_in_reward is not False
                            or self.config.actor_rollout_ref.actor.use_kl_loss is not True
                            or self.config.actor_rollout_ref.actor.kl_loss_type != 'low_var_kl'):
                        raise ValueError('Pilot arms must share the frozen task-only reward/actor-KL pipeline')
                    pre_count = apply_frozen_credit(manager, batch, plan, arm, self.config, journal, update)
                    episodes = manager.observed_episodes()
                    units = {}
                    for index, item in enumerate(plan):
                        unit = item['pair_instance_id'] if arm == 'T4' else item['episode_id']
                        units.setdefault(unit, []).append(index)
                    for unit, indices in units.items():
                        count = sum(pre_evidence(row['decision_index'], manager.envs[i].env.info['tau'])
                                    for i in indices for row in manager.decisions[i])
                        journal({'record_kind': 's1_independent_decision_unit', 'update': update,
                            'unit_id': unit, 'episodes': [plan[i]['episode_id'] for i in indices],
                            'observed_pre_evidence_decisions': count, 'c_u': None if count else 0.,
                            'episode_equivalent_divisor': 2 if arm == 'T4' else 1,
                            'basis': 'complete raw/replaced/applied token vectors and weights retained; no unapproved fallback-token aggregation'})
                    scored = time.monotonic()
                    critic_output = self.critic_wg.update_critic(batch)
                    critic_finished = time.monotonic()
                    batch.meta_info['pilot_update'] = auxiliary
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                    finished = time.monotonic()
                    actual_aux = actor_output.meta_info['pilot_auxiliary_record']
                    journal({'record_kind': 'actual_actor_consistency_update', 'update': update, **actual_aux})
                    require_completed_optimizer_steps(actual_aux)
                    row = json_value({'update': update, 'actual_completed_updates': update,
                        'generation_seconds': generated - started, 'scoring_seconds': scored - generated,
                        'critic_update_seconds': critic_finished - scored, 'actor_update_seconds': finished - critic_finished,
                        'update_seconds': finished - scored, 'total_seconds': finished - started,
                        'episodes': len(plan), 'independent_units': len(units),
                        'by_class': classified_observations(episodes),
                        'physical_policy_sampling_calls': manager.sample_count,
                        'actual_generated_token_count': manager.generated_tokens,
                        'logical_trajectory_token_count': manager.logical_tokens, **scoring,
                        't4_consistency': aux_counts, 'native_actor_optimizer_step_calls': actual_aux['native_optimizer_step_calls'],
                        'native_actor_minibatch_steps': actual_aux['native_minibatch_steps'],
                        'pre_evidence_decision_count': pre_count,
                        'critic_metrics': critic_output.meta_info['metrics'], 'actor_metrics': actor_output.meta_info['metrics'],
                        'memory': self.actor_rollout_wg.smoke_memory(reset=False),
                        'classified_ten_step_checkpoint': update % 10 == 0,
                        's2_due': update in (20, 40), 's2_scope': 'pilot training-scene holdout diagnostics'})
                    journal({'record_kind': 'pilot_update_complete', **row}); updates.append(row)
                    print('E2E_PILOT_UPDATE: ' + json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
                except BaseException as error:
                    journal({'record_kind': 'pilot_update_failed', 'update': update,
                        'completed_updates': len(updates), 'error': f'{type(error).__name__}: {error}'})
                    raise
                finally:
                    for env in manager.envs.values(): env.close()
            row['episode_log'] = file_ref(path, ROOT / 'logs')
            if update in holdout_updates: evaluate(update)
        checkpoint = Path(checkpoint_directory)
        self.actor_rollout_wg.save_checkpoint(str(checkpoint / 'actor'), None, update_count, remove_previous_ckpt=False)
        self.critic_wg.save_checkpoint(str(checkpoint / 'critic'), None, update_count, remove_previous_ckpt=False)
        result = {'actual_completed_updates': len(updates), 'updates': updates, 'holdout_evaluations': evaluations,
            'checkpoint_saved': True, 'final_checkpoint_directory': str(checkpoint),
            'pre_evidence_logic_covered': any(row['pre_evidence_decision_count'] > 0 for row in updates),
            's1_variance': None, 's1_scope': 'actual token vectors and independent units retained for analyst',
            's2_due_updates': [row['update'] for row in evaluations if row['update'] in (20, 40)],
            'formal_T04_T05_acceptance': None, 'paper_result': False, 'common_config_frozen': False}
        if engineering_smoke:
            result['engineering_smoke_version'] = engineering_smoke
        return result
