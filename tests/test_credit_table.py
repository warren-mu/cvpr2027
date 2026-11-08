"""T02 -> T05 integration with actual CPU tables, not simulated EpisodePorts.

The reused fixture is synthetic and cannot certify rendering, cache consistency,
or a production tokenizer/trainer. Run via unittest discovery after commit.
"""
import json
import math
import unittest

from test_pair_table import table
from src.envs.paired_env import table_environment
from src.envs.pair_table import TableEnvironment
from src.rl.credit import ActionScores, CandidateScore, CreditStep, advantages, q_commit
from src.rl.paired_rollout import ModelPrompt, PolicySample, T02EpisodePort, collect_pair


class CreditTableIntegrationTests(unittest.TestCase):
    def ports(self, value):
        environments = {world: table_environment({value.pair_id: value}, allow_synthetic=True)
                        for world in ('w0', 'w1')}
        for environment in environments.values():
            self.assertIsInstance(environment, TableEnvironment)
            self.addCleanup(environment.close)
        return {world: T02EpisodePort(environment, pair_id=value.pair_id, world_id=world,
                                     episode_id='integration_' + world)
                for world, environment in environments.items()}, environments

    def policy(self, actions):
        """One scripted policy sees only the actual public synthetic prompt."""
        calls = []
        def sample(prompt):
            self.assertIsInstance(prompt, ModelPrompt)
            self.assertEqual(set(vars(prompt)), {'images', 'text_token_ids', 'serialization_contract'})
            self.assertEqual(prompt.serialization_contract, 'synthetic_utf8_bytes_not_model_tokenizer')
            self.assertTrue(all(image[1:] == (255, 255, 'uint8', 3) for image in prompt.images))
            public = json.loads(bytes(prompt.text_token_ids).decode())
            self.assertEqual(set(public), {'text', 'marks', 'responses', 'actions', 'contract'})
            round_index = len(prompt.images) - 1
            action = actions[round_index]
            if action == 'select_public':
                current = public['text'][-1]
                allowed = json.loads(current.split('\nAvailable actions: ', 1)[1].split('\nMarks: ', 1)[0])
                action = next(item for item in allowed if item.startswith('select '))
            calls.append(prompt)
            return PolicySample(action, action, 'synthetic_table_policy', f'call_{len(calls)}', 1, (-.2,))
        return sample, calls

    def collect(self, value, actions):
        ports, environments = self.ports(value)
        policy, calls = self.policy(actions)
        result = collect_pair(ports, policy, max_decisions=5, max_response_tokens=256,
            prompt_contract='synthetic_table_full_history', pair_instance_id='integration_draw', allow_synthetic=True)
        return result, calls, environments

    def test_table_fork_return_and_own_seen_times_reach_actual_collector(self):
        result, calls, _ = self.collect(table(), ['turn_right', 'turn_left', 'turn_left', 'select_public'])
        self.assertTrue(result['complete'])
        self.assertEqual(result['environment_steps_by_world'], {'w0': 4, 'w1': 4})
        self.assertEqual(result['physical_policy_sampling_calls'], 7)
        self.assertEqual(len(calls), 7)
        left, right = (result['decisions_by_world'][world] for world in ('w0', 'w1'))
        self.assertEqual(left[0]['shared_sample_id'], right[0]['shared_sample_id'])
        self.assertEqual(left[0]['response_token_log_probabilities'], right[0]['response_token_log_probabilities'])
        self.assertEqual(left[0]['privileged_before']['capture']['source_id'],
                         right[0]['privileged_before']['capture']['source_id'])
        for rows in (left, right):
            self.assertEqual([row['tau'] for row in rows], [None, 1, 1, 1])
            self.assertEqual([row['pre_evidence'] for row in rows], [True, False, False, False])
            self.assertEqual([row['privileged_before']['shared'] for row in rows], [True, False, True, False])
            for row in rows:
                info = row['privileged_before']
                proof = info['table_temporal_proof']
                self.assertEqual(proof['state_id'], info['state_id'])
                self.assertEqual(proof['tau'], row['tau'])
                self.assertEqual(proof['x_seen_step'], row['x_seen_step'])
                self.assertTrue(proof['synthetic'])
                self.assertFalse(info['evidence_provenance_verified'])
                self.assertIsNone(info['cache_matches_live'])
        self.assertEqual([row['x_seen_step'] for row in left], [None, 1, 1, 1])
        self.assertEqual([row['x_seen_step'] for row in right], [None, None, None, 3])
        # Returning to a shared state restores the SAME current public image/text.
        self.assertEqual(left[2]['public_input']['observations'][-1], right[2]['public_input']['observations'][-1])
        # Full history still contains the distinct first observation, so do not
        # mistakenly deduplicate merely because the latest cached frame matches.
        self.assertNotEqual(left[2]['input_sha256'], right[2]['input_sha256'])
        self.assertIsNone(left[2]['shared_sample_id'])
        self.assertIsNone(right[2]['shared_sample_id'])

    def test_actual_selectable_labels_and_temporal_proof_feed_credit_contract(self):
        value = table(distant=True)
        result, calls, _ = self.collect(value, ['select_public'])
        self.assertEqual(len(calls), 1)
        decisions = [result['decisions_by_world'][world][0] for world in ('w0', 'w1')]
        self.assertEqual([row['reward'] for row in decisions], [0.0, 1.0])
        self.assertEqual(result['actual_generated_token_count'], 1)
        self.assertEqual(result['logical_trajectory_token_count'], 2)
        credits = []
        for row in decisions:
            info = row['privileged_before']
            public = row['public_input']['observations'][-1]
            self.assertEqual([mark['label'] for mark in public['marks']], [value.labels['B']])
            self.assertEqual(info['selectable_labels'], [value.labels['B']])
            allowed = json.loads(public['text'].split('\nAvailable actions: ', 1)[1].split('\nMarks: ', 1)[0])
            self.assertNotIn(f'select {value.labels["A"]}', allowed)
            selected = [candidate for candidate in info['candidates'] if candidate['action_legal']]
            self.assertEqual([candidate['object_name'] for candidate in selected], ['B'])
            scores = ActionScores(tuple(CandidateScore(candidate['object_name'], candidate['action_text'],
                math.log(.2), (math.log(.2),)) for candidate in selected),
                row['input_sha256'], row['policy_id'], 'synthetic_table_action_scores',
                'explicit_synthetic_complete_action_event', True,
                public_select_action_texts=tuple(candidate['action_text'] for candidate in selected),
                candidate_domain_source='actual_synthetic_table_public_labels')
            credits.append(CreditStep(episode_id=row['episode_id'], pair_id=row['pair_id'],
                pair_instance_id=row['pair_instance_id'], world_id=row['world_id'], decision_index=row['decision_index'],
                rho=.5, action_text=row['action_text'], action_kind='select', pre_evidence=row['pre_evidence'],
                input_sha256=row['input_sha256'], shared_sample_id=row['shared_sample_id'],
                shared_input_verified_exact=row['synthetic_shared_input_exact'],
                gae_advantage=row['reward'], realized_return=row['reward'], immediate_reward=row['reward'],
                target_name=info['target_name'], scores=scores, gae_source='synthetic_terminal_value_zero',
                return_definition='actual_one_step_fixture_reward', evidence_proof=info['evidence_proof'],
                evidence_status=info['evidence_status'], evidence_provenance_verified=info['evidence_provenance_verified'],
                tau=info['tau'], x_seen_step=info['x_seen_step'], temporal_proof=info['table_temporal_proof'],
                synthetic_fixture=True))
        self.assertEqual([q_commit(row.scores, row.target_name) for row in credits], [0.0, 1.0])
        self.assertEqual(advantages('T3', credits), {row.key: 0.0 for row in credits})

    def test_real_environment_five_round_limit_stops_collect_pair(self):
        result, _, environments = self.collect(table(), ['turn_right'] * 5)
        self.assertTrue(result['complete'])
        self.assertEqual(result['environment_steps_by_world'], {'w0': 5, 'w1': 5})
        self.assertEqual(result['logical_model_calls_by_world'], {'w0': 5, 'w1': 5})
        for world, environment in environments.items():
            rows = result['decisions_by_world'][world]
            self.assertEqual([row['reward'] for row in rows], [-.05] * 5)
            self.assertTrue(rows[-1]['done'])
            self.assertTrue(rows[-1]['privileged_after']['budget_exhausted'])
            self.assertFalse(rows[-1]['privileged_after']['success'])
            self.assertEqual(rows[-1]['privileged_before']['state_id'], rows[-1]['privileged_after']['state_id'])
            self.assertEqual(environment.steps, 5)
            self.assertEqual(environment.model_calls, 5)
            with self.assertRaises(RuntimeError):
                environment.step('select 1')


if __name__ == '__main__':
    unittest.main()
