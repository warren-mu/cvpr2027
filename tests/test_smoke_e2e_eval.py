"""CPU contract fixtures only; no model, torch, vLLM or renderer import."""
import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from src.envs.paired_env import PublicObservation, RGBFrame, PublicMark
from src.tools import smoke_e2e_eval as evaluation


def observation():
    return PublicObservation(RGBFrame(b'\x01\x02\x03' * 16, 4, 4),
        'Select the marked object closest to the Bowl.\nAvailable actions: ["turn_left","turn_right","select 1","select 2"]',
        (PublicMark(1, (0, 0), (0, 0, 1, 1)), PublicMark(2, (2, 2), (2, 2, 3, 3))))


class Environment:
    def reset(self, pair, world):
        self.pair, self.world, self.steps, self.model_calls = pair, world, 0, 0
        self.done, self.success, self.selected = False, False, False
        self.calls = []
        return observation()

    @property
    def info(self):
        return {'episode_id': self.pair + self.world, 'state_id': 's' + str(self.steps),
            'tau': 0, 'x_seen_step': 0, 'kind': 'direct_flip', 'selected_name': 'Object' if self.selected else None,
            'selected': self.selected, 'success': self.success, 'obs_count': self.steps - int(self.selected),
            'table_temporal_proof': {'fixture': True}}

    def record_model_call(self, value):
        self.calls.append(value); self.model_calls += 1

    def step(self, action):
        self.steps += 1
        self.selected = action.startswith('select ')
        self.success = self.selected and action == 'select 1'
        self.done = self.selected or self.steps == 5
        return observation(), float(self.success), self.done, self.info


class Policy:
    def __init__(self, response='select 1'):
        self.response, self.batches = response, []

    def __call__(self, contexts):
        self.batches.append(copy.deepcopy(contexts))
        return [{'response': self.response, 'action': self.response, 'forced_observation': False,
                 'response_token_ids': [1, 2, 3], 'response_token_count': 3,
                 'sample_id': f'b{len(self.batches)}_i{i}'} for i, _ in enumerate(contexts)]


class EvaluationTests(unittest.TestCase):
    def test_unconstrained_diag_accepts_empty_final_turn_action_domain(self):
        empty = PublicObservation(RGBFrame(b'\x01\x02\x03' * 16, 4, 4),
                                  'Instruction\nAvailable actions: []', ())
        policy = object.__new__(evaluation.VllmPolicy)
        policy.processor = SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: 'actual public prompt',
            chat_template='fixture', tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: [7, 8]))
        policy.sample_type = lambda **kwargs: kwargs
        policy.guide_type = lambda **kwargs: self.fail('diag_base must remain unconstrained')
        policy.serial = 0
        generated = []
        def generate(requests, parameters, **kwargs):
            generated.append((requests, parameters))
            return [SimpleNamespace(outputs=[SimpleNamespace(text='invalid', token_ids=[9],
                finish_reason='stop', stop_reason=None)])]
        policy.model = SimpleNamespace(generate=generate)
        rows = policy([{'observations': [empty], 'responses': [], 'select_only': False}])
        self.assertEqual(rows[0]['action'], 'invalid')
        self.assertIsNone(generated[0][1][0]['guided_decoding'])
        self.assertIsNone(rows[0]['input']['guided_select_choices'])
        with self.assertRaises(ValueError):
            policy([{'observations': [empty], 'responses': [], 'select_only': True}])

    def test_import_does_not_load_gpu_dependencies(self):
        import sys
        # This module's own import remains safe in CPU unittest discovery.
        self.assertFalse(hasattr(evaluation, 'torch'))
        self.assertFalse(hasattr(evaluation, 'vllm'))

    def test_public_prompt_contains_complete_history_and_no_privileged_info(self):
        obs = observation()
        messages = evaluation.prompt_messages([obs, obs], ['turn_left'], select_only=True)
        self.assertEqual([row['role'] for row in messages], ['system', 'user', 'assistant', 'user'])
        self.assertEqual(messages[2]['content'], 'turn_left')
        self.assertIn('select <number>', messages[-1]['content'][1]['text'])
        self.assertNotIn('target_name', str(messages))
        with self.assertRaises(ValueError):
            evaluation.prompt_messages([obs], ['turn_left'], select_only=False)

    def test_never_is_one_batched_real_select_per_world(self):
        policy, journal = Policy(), []
        envs = [('p', world, Environment()) for world in ('w0', 'w1')]
        rows = evaluation.collect_strategy('obs_never', envs, policy, journal.append)
        self.assertEqual(len(policy.batches), 1)
        self.assertEqual(len(policy.batches[0]), 2)
        self.assertEqual([(row['steps'], row['model_calls'], row['tau']) for row in rows], [(1, 1, 0)] * 2)
        self.assertTrue(all(row['selected_object'] == 'Object' for row in rows))
        self.assertTrue(all(ctx['select_only'] for ctx in policy.batches[0]))

    def test_always_fixed_roundtrip_uses_model_only_after_four_observations(self):
        policy, journal = Policy(), []
        envs = [('p', world, Environment()) for world in ('w0', 'w1')]
        rows = evaluation.collect_strategy('obs_always', envs, policy, journal.append)
        self.assertEqual(len(policy.batches), 1)
        self.assertEqual(policy.batches[0][0]['responses'], list(evaluation.EXPLORATION))
        self.assertEqual(len(policy.batches[0][0]['observations']), 5)
        self.assertEqual([(row['steps'], row['model_calls']) for row in rows], [(5, 1)] * 2)
        self.assertEqual([row['sample']['action'] for row in rows[0]['trace'][:4]], list(evaluation.EXPLORATION))

    def test_diag_invalid_model_output_is_executed_and_logged_without_repair(self):
        policy, journal = Policy('I think select 1'), []
        rows = evaluation.collect_strategy('diag_base', [('p', 'w0', Environment())], policy, journal.append)
        self.assertEqual(rows[0]['steps'], 5)
        self.assertFalse(rows[0]['success'])
        self.assertIsNone(rows[0]['selected_object'])
        self.assertTrue(all(not ctx['select_only'] for batch in policy.batches for ctx in batch))
        self.assertEqual(rows[0]['trace'][0]['sample']['action'], 'I think select 1')

    def test_incomplete_model_batch_and_bad_forced_select_fail(self):
        with self.assertRaisesRegex(ValueError, 'omitted/added'):
            evaluation.collect_strategy('obs_never', [('p', 'w0', Environment())], lambda _: [], lambda _: None)
        with self.assertRaisesRegex(ValueError, 'Forced-select'):
            evaluation.collect_strategy('obs_never', [('p', 'w0', Environment())], Policy('turn_left'), lambda _: None)

    def test_replay_authentication_rejects_another_producer(self):
        data = {'_meta': {'name': 'smoke_e2e_replay', 'command': ['python3', '-m', 'other'], 'exit_code': 0},
            'config': {'source': 's', 'contract_tests': 't'}, 'metrics': {'step2_passed': True}}
        with patch.object(evaluation, 'audited_result', return_value=(data, {'file': 'replay'})):
            with self.assertRaisesRegex(ValueError, 'step 2'):
                evaluation.load_replay('irrelevant')


if __name__ == '__main__':
    unittest.main()
