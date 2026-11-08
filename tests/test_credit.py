"""Hand-constructed CPU contracts for T05; no model, backend, or GPU runs.

Run only after commit through the project's logged CPU test harness. Synthetic
probabilities and captures below are fixtures, never measured research results.
"""
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

from src.rl.credit import (ActionScores, CandidateScore, CreditContractError, CreditStep,
    UnsupportedCredit, advantages, consistency_components, episode_plan, flip_hinges,
    q_commit, reference_gae, select_distribution, symmetric_kl, torch_flip_hinges,
    torch_symmetric_kl, trajectory_loss_weight)
from src.rl.credit_records import RawTrainingRecorder, TrainingHoldout, independent_credit_units
from src.rl.paired_rollout import (EpisodeState, PolicySample, PublicMark, PublicObservation,
                                 ModelPrompt, PreparedPrompt, Transition, collect_pair, synthetic_prompt)

INPUT = "a" * 64


def score_fixture(probabilities=None, *, input_sha256=INPUT):
    probabilities = {"Cup_A": ("select 1", 0.5), "Cup_B": ("select 2", 0.5)} if probabilities is None else probabilities
    return ActionScores(tuple(CandidateScore(name, text, math.log(p), (math.log(p),))
                              for name, (text, p) in probabilities.items()),
                        input_sha256, "fixture_policy", "fixture_scores", "synthetic_complete_action_including_stop",
                        complete_action_events=True,
                        public_select_action_texts=tuple(text for text, p in probabilities.values()),
                        candidate_domain_source="synthetic_actual_public_select_labels")


def step_fixture(world="w0", *, observe=True, rho=0.5, target=None, scores=None):
    reward = -0.05 if observe else (1.0 if world == "w0" else 0.0)
    total = 0.95 if observe else reward
    gae = reference_gae([-0.05, 1.0] if observe else [reward], [0.0, 0.0] if observe else [0.0],
                        [False, True] if observe else [True], gamma=1.0, gae_lambda=1.0, bootstrap_value=0.0)[0]
    return CreditStep(episode_id="episode_" + world, pair_id="pool_pair", pair_instance_id="draw_0",
        world_id=world, decision_index=0, rho=rho, action_text="turn_left" if observe else "select 1",
        action_kind="observe" if observe else "select", pre_evidence=True, input_sha256=INPUT,
        shared_sample_id="fixture_shared_sample", shared_input_verified_exact=True,
        gae_advantage=gae, realized_return=total, immediate_reward=reward,
        target_name=target or ("Cup_A" if world == "w0" else "Cup_B"), scores=scores or score_fixture(),
        gae_source="synthetic_scalar_GAE_gamma1_lambda1_value0",
        return_definition="undiscounted_sum_of_realized_fixture_rewards_cost_already_included",
        evidence_proof={"synthetic": True, "fixture": "paired_before_observation"},
        evidence_status="not_entered_verified", evidence_provenance_verified=False,
        tau=1, x_seen_step=None, temporal_proof={"synthetic_table_history": True}, synthetic_fixture=True)


def observation(value=7, text="Select the cup by its visible mark.", label=1):
    return PublicObservation(bytes([value] * 12), 2, 2, "uint8", 3, text,
                             (PublicMark(label, (0, 0), (0, 0, 1, 1)),))


class FixturePort:
    def __init__(self, world, frames, evidence=None, seen=None):
        self.world, self.frames = world, frames
        self.evidence = evidence or [False] * len(frames)
        self.seen = self.evidence if seen is None else seen
        self.index, self.calls, self.actions = 0, [], []

    def state(self):
        image = self.frames[self.index]
        entered = self.evidence[self.index]
        return EpisodeState(image, {"world_id": self.world, "pair_id": "pool_pair",
            "episode_id": "episode_" + self.world, "decision_index": self.index, "kind": "look_flip",
            "evidence_entered": entered, "evidence_status": "entered" if entered else "not_entered_verified",
            "tau": next((i for i, flag in enumerate(self.evidence[:self.index + 1]) if flag), None),
            "x_seen_step": next((i for i, flag in enumerate(self.seen[:self.index + 1]) if flag), None),
            "table_temporal_proof": {"synthetic_table": True, "step": self.index},
            "evidence_proof": [{"synthetic": True}], "synthetic_backend": True,
            "capture_provenance_verified": False, "evidence_provenance_verified": False,
            "capture": {"capture_id": f"{self.world}_{self.index}", "source_id": f"raw_{self.world}_{self.index}"},
            "raw_rgb_sha256": hashlib.sha256(image.rgb).hexdigest(), "target_name": "PRIVATE_TARGET_" + self.world})

    def reset(self):
        self.index = 0
        return self.state()

    def record_model_call(self, call_id):
        if call_id in self.calls:
            raise AssertionError("Repeated call ID in one episode")
        self.calls.append(call_id)

    def step(self, action_text):
        self.actions.append(action_text)
        self.index += 1
        done = self.index == len(self.frames) - 1
        return Transition(self.state(), 1.0 if done else -0.05, done)


class CreditMathTests(unittest.TestCase):
    def test_scalar_gae_terminal_and_truncated_bootstrap(self):
        self.assertEqual(reference_gae([1.0], [0.25], [True], gamma=0.9, gae_lambda=0.8, bootstrap_value=9.0), (0.75,))
        self.assertAlmostEqual(reference_gae([1.0], [0.25], [False], gamma=0.9, gae_lambda=0.8, bootstrap_value=2.0)[0], 2.55)

    def test_identical_information_gets_identical_T3_T4_but_independent_T1(self):
        rows = [step_fixture(world, observe=False) for world in ("w0", "w1")]
        for arm in ("T3", "T4"):
            result = advantages(arm, rows)
            self.assertEqual(result[rows[0].key], result[rows[1].key])
        independent = advantages("T1", rows)
        self.assertNotEqual(independent[rows[0].key], independent[rows[1].key])
        for arm in ("T0", "T1"):
            self.assertEqual(advantages(arm, rows), {row.key: row.gae_advantage for row in rows})

    def test_lucky_commit_and_observation_credit(self):
        commits = [step_fixture(world, observe=False) for world in ("w0", "w1")]
        observations = [step_fixture(world) for world in ("w0", "w1")]
        same_world = advantages("T2", commits)
        self.assertAlmostEqual(same_world[commits[0].key], 0.5)
        self.assertAlmostEqual(same_world[commits[1].key], -0.5)
        self.assertGreater(advantages("T1", commits)[commits[0].key], 0)
        paired_observation = advantages("T3", observations)
        self.assertAlmostEqual(paired_observation[observations[0].key], 0.45)  # cost appears once in G=.95
        self.assertEqual(paired_observation[observations[0].key], paired_observation[observations[1].key])
        self.assertEqual(advantages("T3", commits)[commits[0].key], 0.0)

    def test_rho_weights_are_not_hardcoded_half(self):
        rows = [step_fixture(world, observe=False, rho=0.25) for world in ("w0", "w1")]
        result = advantages("T3", rows)
        self.assertAlmostEqual(result[rows[0].key], 0.25)
        self.assertEqual(result[rows[0].key], result[rows[1].key])

    def test_unnecessary_look_stay_is_negative(self):
        scores = score_fixture({"Cup_A": ("select 1", 1.0)})
        rows = [step_fixture(world, target="Cup_A", scores=scores) for world in ("w0", "w1")]
        result = advantages("T3", rows)
        self.assertAlmostEqual(result[rows[0].key], -0.05)
        self.assertEqual(result[rows[0].key], result[rows[1].key])

    def test_post_evidence_steps_keep_upstream_gae_for_every_arm(self):
        row = replace(step_fixture(), pre_evidence=False, evidence_status="entered", tau=0, gae_advantage=-0.123, scores=None)
        for arm in ("T0", "T1", "T2", "T3", "T4"):
            self.assertEqual(advantages(arm, [row])[row.key], -0.123)

    def test_select_conditional_Q_keeps_raw_mass_and_complete_action_scoring(self):
        scores = score_fixture({"Cup_A": ("select 9", 0.2), "Cup_B": ("select 2", 0.3)})
        self.assertAlmostEqual(q_commit(scores, "Cup_A"), 0.4)
        self.assertAlmostEqual(scores.record()["candidate_probability_mass"], 0.5)
        self.assertEqual(q_commit(scores, "currently_unmarked_target"), 0.0)
        with self.assertRaises(CreditContractError):
            q_commit(replace(scores, probability_space="conditional_on_select"), "Cup_A")
        with self.assertRaises(CreditContractError):
            q_commit(replace(scores, complete_action_events=False), "Cup_A")
        bad = replace(scores.candidates[0], token_log_probabilities=(math.log(0.2), math.log(0.2)))
        with self.assertRaises(CreditContractError):
            q_commit(replace(scores, candidates=(bad, scores.candidates[1])), "Cup_A")

    def test_noise_fork_or_unknown_evidence_is_not_an_information_set(self):
        rows = [step_fixture(world) for world in ("w0", "w1")]
        for changed in (replace(rows[1], shared_input_verified_exact=False),
                        replace(rows[1], shared_sample_id="independent"),
                        replace(rows[1], evidence_status="unknown")):
            with self.assertRaises(UnsupportedCredit):
                advantages("T3", [rows[0], changed])

    def test_flip_log_hinges_and_stay_KL_align_stable_names(self):
        logs = {"w0": {"Cup_A": math.log(0.6), "Cup_B": math.log(0.2)},
                "w1": {"Cup_B": math.log(0.2), "Cup_A": math.log(0.6)}}
        hinge = flip_hinges(logs, {"w0": "Cup_A", "w1": "Cup_B"})
        self.assertEqual(hinge["w0"], 0.0)
        self.assertAlmostEqual(hinge["w1"], 1 + math.log(3))
        p, q = {"Cup_A": 0.75, "Cup_B": 0.25}, {"Cup_B": 0.75, "Cup_A": 0.25}
        terms = symmetric_kl(p, q, distribution_source="explicit_fixture_distribution", convention="half_sum")
        self.assertAlmostEqual(terms["forward"], 0.5 * math.log(3))
        self.assertAlmostEqual(terms["symmetric"], 0.5 * math.log(3))
        with self.assertRaises(UnsupportedCredit):
            symmetric_kl({"Cup_A": 0.2}, {"Cup_A": 0.3}, distribution_source="raw_action_mass", convention="half_sum")
        with self.assertRaises(UnsupportedCredit):
            symmetric_kl(p, {"Cup_A": 1.0}, distribution_source="fixture", convention="half_sum")

    def test_T4_auxiliary_only_after_evidence_on_select(self):
        rows = [step_fixture(world, observe=False) for world in ("w0", "w1")]
        self.assertFalse(consistency_components("T4", rows, forked=True, pair_change="flip")["eligible"])
        post = [replace(row, pre_evidence=False, evidence_status="entered", tau=0, x_seen_step=0, input_sha256=str(i) * 64,
                        scores=replace(row.scores, input_sha256=str(i) * 64)) for i, row in enumerate(rows)]
        terms = consistency_components("T4", post, forked=True, pair_change="flip")
        self.assertEqual(terms["weighted_terms"], {"w0": 0.1, "w1": 0.1})
        self.assertFalse(consistency_components("T3", post, forked=True, pair_change="flip")["eligible"])

    def test_all_five_arms_allocate_64_episodes(self):
        for arm in ("T0", "T1", "T2", "T3", "T4"):
            draws = 32 if arm in ("T1", "T3", "T4") else 64
            plan = episode_plan(arm, ["same_pool_pair"] * draws, rho=0.5, rng=random.Random(0))
            self.assertEqual(len(plan), 64)
            self.assertEqual(len({row["episode_id"] for row in plan}), 64)
            self.assertEqual(len({row["pair_instance_id"] for row in plan}), draws)

    def test_no_marks_and_invalid_outputs_use_the_specified_gae_fallback(self):
        rows = [replace(step_fixture(world), scores=score_fixture({}),
                        gae_advantage=2.0 if world == 'w0' else -1.0) for world in ('w0', 'w1')]
        self.assertIsNone(q_commit(rows[0].scores, 'Cup_A'))
        for arm in ('T3', 'T4'):
            measured = advantages(arm, rows)
            self.assertEqual(measured, {row.key: .5 for row in rows})
        self.assertEqual(advantages('T2', rows), {row.key: row.gae_advantage for row in rows})
        invalid = [replace(row, action_kind='invalid', action_text='unparseable', scores=None) for row in rows]
        self.assertEqual(advantages('T3', invalid), {row.key: .5 for row in invalid})
        self.assertEqual(advantages('T2', invalid), {row.key: row.gae_advantage for row in invalid})
        with self.assertRaises(CreditContractError):
            advantages('T2', [replace(rows[0], scores=None)])  # missing is not an observed empty domain

    def test_T1_T3_gamma_lambda_one_differ_only_by_value_versus_Q_baseline(self):
        rows = [step_fixture(world) for world in ('w0', 'w1')]
        value = .2
        rows = [replace(row, gae_advantage=reference_gae([-.05, 1.0], [value, value], [False, True],
                        gamma=1.0, gae_lambda=1.0, bootstrap_value=0)[0]) for row in rows]
        self.assertAlmostEqual(sum(advantages('T1', rows).values()), sum(row.realized_return - value for row in rows))
        self.assertAlmostEqual(sum(advantages('T3', rows).values()),
                               sum(row.realized_return - q_commit(row.scores, row.target_name) for row in rows))

    def test_world_token_weights_are_separate_from_shared_advantages(self):
        for arm in ('T1', 'T3', 'T4'):
            self.assertEqual(trajectory_loss_weight(arm, 'w0', .25), 1.5)
            self.assertEqual(trajectory_loss_weight(arm, 'w1', .25), .5)
        self.assertEqual(trajectory_loss_weight('T2', 'w1', .25), 1.0)

    def test_Q_scores_cover_the_whole_public_domain_and_stay_stable_at_tiny_mass(self):
        scores = score_fixture({'Cup_A': ('select 1', .2), 'Cup_B': ('select 2', .3)})
        with self.assertRaises(CreditContractError):
            q_commit(replace(scores, public_select_action_texts=('select 1',)), 'Cup_A')
        tiny = replace(scores, candidates=tuple(replace(item, log_probability=-1000.0,
                            token_log_probabilities=(-1000.0,)) for item in scores.candidates))
        self.assertEqual(select_distribution(tiny), {'Cup_A': .5, 'Cup_B': .5})

    def test_T4_uses_own_inclusive_seen_boundary_not_tau_or_the_other_world(self):
        rows = [replace(step_fixture(world, observe=False), decision_index=2,
            pre_evidence=False, evidence_status='entered', tau=1, x_seen_step=2 if world == 'w0' else None,
            input_sha256=world, scores=replace(score_fixture(), input_sha256=world)) for world in ('w0', 'w1')]
        flip = consistency_components('T4', rows, pair_change='flip')
        self.assertEqual(flip['weighted_terms'], {'w0': .1})
        self.assertEqual(flip['eligible_by_world'], {'w0': True, 'w1': False})
        stay_rows = [replace(row, target_name='Cup_A') for row in rows]
        self.assertFalse(consistency_components('T4', stay_rows, pair_change='stay')['eligible'])
        future = [replace(row, x_seen_step=3) for row in rows]
        self.assertFalse(consistency_components('T4', future, pair_change='flip')['eligible'])
        with self.assertRaises(UnsupportedCredit):
            advantages('T2', [replace(step_fixture(), tau=0)])  # pre flag contradicts tau

    def test_eligible_Stay_uses_actual_candidate_normalization_and_stable_name_support(self):
        scores = [score_fixture({'Cup_A': ('select 1', .3), 'Cup_B': ('select 2', .1)}),
                  score_fixture({'Cup_B': ('select 2', .3), 'Cup_A': ('select 1', .1)})]
        rows = [replace(step_fixture(world, observe=False, target='Cup_A'), pre_evidence=False,
                        evidence_status='entered', tau=0, x_seen_step=0, input_sha256=world,
                        scores=replace(scores[index], input_sha256=world))
                for index, world in enumerate(('w0', 'w1'))]
        terms = consistency_components('T4', rows, pair_change='stay', kl_convention='half_sum')
        self.assertAlmostEqual(terms['weighted_terms'], .1 * .5 * math.log(3))
        with self.assertRaises(CreditContractError):
            consistency_components('T4', rows, pair_change='stay', kl_convention='half_sum',
                stay_distributions={world: {'Cup_A': .5, 'Cup_B': .5} for world in ('w0', 'w1')})


class TorchGradientTests(unittest.TestCase):
    """CPU tensor backward checks; no model, CUDA query, or trainer integration."""
    @classmethod
    def setUpClass(cls):
        # Missing torch is an error, not a skipped backward-validation claim.
        import torch
        cls.torch = torch

    def tensor(self, values):
        return self.torch.tensor(values, dtype=self.torch.float64, device="cpu", requires_grad=True)

    def flip(self, first, second):
        # Include an observation action in the full softmax. The helper receives
        # genuine select-event logs, never a softmax conditioned on selecting.
        logs = {world: logits.log_softmax(dim=0) for world, logits in (("w0", first), ("w1", second))}
        return torch_flip_hinges({world: {"Cup_A": row[0], "Cup_B": row[1]}
                                  for world, row in logs.items()},
                                 {"w0": "Cup_A", "w1": "Cup_B"})

    def test_flip_violated_margin_gradients_favor_each_world_correct_target(self):
        first, second = self.tensor([-1.0, 1.0, 0.0]), self.tensor([1.0, -1.0, 0.0])
        terms = self.flip(first, second)
        self.assertGreater(float(terms["w0"].detach()), 0.0)
        self.assertGreater(float(terms["w1"].detach()), 0.0)
        (terms["w0"] + terms["w1"]).backward()
        # Gradient descent increases correct logits and decreases the competing
        # target. The common full-action normalization cancels in the log ratio.
        expected = ((first, [-0.1, 0.1, 0.0]), (second, [0.1, -0.1, 0.0]))
        for logits, values in expected:
            self.assertTrue(self.torch.isfinite(logits.grad).all().item())
            self.torch.testing.assert_close(logits.grad, self.torch.tensor(values, dtype=self.torch.float64),
                                            rtol=1e-12, atol=1e-12)

    def test_flip_satisfied_margin_has_exact_zero_loss_and_gradient(self):
        first, second = self.tensor([2.0, 0.0, -1.0]), self.tensor([0.0, 2.0, -1.0])
        terms = self.flip(first, second)
        loss = terms["w0"] + terms["w1"]
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        for logits in (first, second):
            self.assertEqual(self.torch.count_nonzero(logits.grad).item(), 0)

    def test_stay_normalized_logit_gradients_are_finite_and_symmetric(self):
        first, second = self.tensor([0.6, -0.6, 0.2]), self.tensor([-0.2, 0.8, -0.1])
        p, q = first.softmax(dim=0), second.softmax(dim=0)
        names = ("Cup_A", "Cup_B", "Cup_C")
        left = dict(zip(names, p))
        right = {names[index]: q[index] for index in (2, 0, 1)}
        options = {"distribution_source": "explicit_synthetic_normalized_logit_distribution",
                   "convention": "half_sum"}
        loss = torch_symmetric_kl(left, right, **options)
        reverse = torch_symmetric_kl(right, left, **options)
        gradients = self.torch.autograd.grad(loss, (first, second), retain_graph=True)
        reverse_gradients = self.torch.autograd.grad(reverse, (first, second), retain_graph=True)
        summed = torch_symmetric_kl(left, right, **{**options, "convention": "sum"})
        summed_gradients = self.torch.autograd.grad(summed, (first, second))
        scalar = symmetric_kl({name: float(value.detach()) for name, value in left.items()},
                              {name: float(value.detach()) for name, value in right.items()}, **options)
        self.assertAlmostEqual(float(loss.detach()), 0.1 * scalar["symmetric"], places=12)
        self.torch.testing.assert_close(loss, reverse, rtol=1e-12, atol=1e-12)
        for gradient, reversed_gradient, summed_gradient in zip(gradients, reverse_gradients, summed_gradients):
            self.assertTrue(self.torch.isfinite(gradient).all().item())
            self.assertGreater(self.torch.count_nonzero(gradient).item(), 0)
            self.assertAlmostEqual(float(gradient.sum()), 0.0, places=12)
            self.torch.testing.assert_close(gradient, reversed_gradient, rtol=1e-12, atol=1e-12)
            self.torch.testing.assert_close(summed_gradient, 2 * gradient, rtol=1e-12, atol=1e-12)

    def test_stay_equal_distributions_and_zero_support_without_epsilon(self):
        first, second = self.tensor([0.5, -0.5]), self.tensor([0.5, -0.5])
        p, q = first.softmax(dim=0), second.softmax(dim=0)
        options = {"distribution_source": "explicit_synthetic_normalized_distribution", "convention": "half_sum"}
        loss = torch_symmetric_kl(dict(zip(("Cup_A", "Cup_B"), p)),
                                  dict(zip(("Cup_A", "Cup_B"), q)), **options)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        for logits in (first, second):
            self.assertEqual(self.torch.count_nonzero(logits.grad).item(), 0)

        # Mathematical zero/zero terms are omitted; no epsilon turns a true
        # positive-versus-zero support mismatch into a finite surrogate loss.
        left, right = self.tensor([0.0, 1.0]), self.tensor([0.0, 1.0])
        as_map = lambda row: {"Cup_A": row[0], "Cup_B": row[1]}
        zero = torch_symmetric_kl(as_map(left), as_map(right), **options)
        self.assertEqual(float(zero.detach()), 0.0)
        zero.backward()
        for distribution in (left, right):
            self.assertTrue(self.torch.isfinite(distribution.grad).all().item())
            self.assertEqual(self.torch.count_nonzero(distribution.grad).item(), 0)
        with self.assertRaises(UnsupportedCredit):
            torch_symmetric_kl(as_map(self.tensor([0.0, 1.0])), as_map(self.tensor([0.25, 0.75])), **options)
        with self.assertRaises(UnsupportedCredit):
            torch_symmetric_kl(as_map(self.tensor([0.5, 0.5])), {"Cup_A": self.tensor(1.0)}, **options)


class PairedRolloutTests(unittest.TestCase):
    def policy(self):
        calls = []
        def sample(context):
            self.assertFalse(hasattr(context, "privileged"))
            self.assertNotIn("PRIVATE_TARGET", repr(context))
            calls.append(context)
            return PolicySample("turn_left", "public response: turn_left", "fixture_policy", f"sample_{len(calls)}", 4, (-.1,) * 4)
        return sample, calls

    def test_different_full_history_stays_different_despite_equal_current_frame(self):
        ports = {"w0": FixturePort("w0", [observation(7), observation(8), observation(9), observation(9)]),
                 "w1": FixturePort("w1", [observation(7), observation(10), observation(9), observation(9)])}
        policy, calls = self.policy()
        result = collect_pair(ports, policy, max_decisions=3, max_response_tokens=256, prompt_contract="fixture_prompt", pair_instance_id="draw_0", allow_synthetic=True)
        self.assertEqual(len(calls), 5)
        self.assertEqual(ports["w0"].calls[0], ports["w1"].calls[0])
        self.assertNotEqual(ports["w0"].calls[1], ports["w1"].calls[1])
        self.assertEqual(result["first_fork"]["decision_index"], 1)
        self.assertFalse(any(result["first_fork"]["evidence_entered_by_world"].values()))
        self.assertEqual(result["logical_model_calls_by_world"], {"w0": 3, "w1": 3})
        self.assertFalse(result["decisions_by_world"]["w0"][0]["shared_input_verified_exact"])
        self.assertTrue(result["decisions_by_world"]["w0"][0]["synthetic_shared_input_exact"])

    def test_public_text_and_marks_are_part_of_exact_input(self):
        for different in (observation(text="Different public instruction"), observation(label=2)):
            ports = {"w0": FixturePort("w0", [observation(), observation()]),
                     "w1": FixturePort("w1", [different, different])}
            policy, calls = self.policy()
            collect_pair(ports, policy, max_decisions=1, max_response_tokens=256, prompt_contract="fixture", pair_instance_id="draw_0", allow_synthetic=True)
            self.assertEqual(len(calls), 2)

    def test_mock_requires_explicit_opt_in(self):
        ports = {world: FixturePort(world, [observation(), observation()]) for world in ("w0", "w1")}
        with self.assertRaises(CreditContractError):
            collect_pair(ports, self.policy()[0], max_decisions=1, max_response_tokens=256, prompt_contract="fixture", pair_instance_id="draw_0")

    def test_P_cached_source_identity_is_private_and_shared_tokens_count_once(self):
        class SharedCachePort(FixturePort):
            def state(self):
                state = super().state()
                private = dict(state.privileged, capture={'capture_id': 'same_cache', 'source_id': 'w0_cache'},
                               raw_rgb_sha256='different_raw_' + self.world)
                return EpisodeState(state.public, private)
        ports = {world: SharedCachePort(world, [observation(), observation()]) for world in ('w0', 'w1')}
        policy, calls = self.policy()
        result = collect_pair(ports, policy, max_decisions=1, max_response_tokens=256,
                              prompt_contract='fixture', pair_instance_id='draw_0', allow_synthetic=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['actual_generated_token_count'], 4)
        self.assertEqual(result['logical_trajectory_token_count'], 8)
        self.assertEqual(result['decisions_by_world']['w0'][0]['response_token_log_probabilities'],
                         result['decisions_by_world']['w1'][0]['response_token_log_probabilities'])

    def test_exact_actual_prompt_can_reshare_without_a_permanent_fork_latch(self):
        # Scripted actual model-token inputs isolate the dedup state machine.
        # This is not a claim about a real tokenizer or T04 history serialization.
        ports = {'w0': FixturePort('w0', [observation()] * 4),
                 'w1': FixturePort('w1', [observation(), observation(text='alternate'), observation(), observation()])}
        def builder(context):
            base = synthetic_prompt(context)
            tokens = (9,) if len(context.observations) == 2 and context.observations[-1].text == 'alternate' else (1,)
            return PreparedPrompt(ModelPrompt(base.prompt.images, tokens, 'explicit_scripted_fixture_prompt'), context.sha256)
        policy, calls = self.policy()
        result = collect_pair(ports, policy, max_decisions=3, max_response_tokens=256,
                              prompt_contract='fixture', pair_instance_id='draw_0', allow_synthetic=True, prompt_builder=builder)
        self.assertEqual(len(calls), 4)
        self.assertEqual([item['shared'] for item in result['sharing_by_round']], [True, False, True])
        self.assertEqual(ports['w0'].calls[2], ports['w1'].calls[2])

    def test_prompt_tokens_and_all_images_not_just_a_digest_control_sharing(self):
        ports = {'w0': FixturePort('w0', [observation(), observation()]),
                 'w1': FixturePort('w1', [observation(9), observation(9)])}
        def builder(context):
            image = context.observations[0]
            return PreparedPrompt(ModelPrompt(((image.rgb, 2, 2, 'uint8', 3),), (1,), 'fixture'), context.sha256)
        policy, calls = self.policy()
        collect_pair(ports, policy, max_decisions=1, max_response_tokens=256, prompt_contract='fixture',
                     pair_instance_id='draw_0', allow_synthetic=True, prompt_builder=builder)
        self.assertEqual(len(calls), 2)


class RawRecordingTests(unittest.TestCase):
    def holdout(self, directory):
        path = Path(directory) / "holdout.json"
        path.write_text(json.dumps({"role": "train_scene_holdout", "frozen_before_training": True,
            "evaluation_policy_contract": "synthetic_greedy_fixture",
            "episodes": [{"eval_episode_id": "heldout_0", "pair_id": "heldout_pair", "scene": "FloorPlan1", "world_id": "w0", "seed": 0}]}))
        return TrainingHoldout.load(path, train_scenes={"FloorPlan1"}, frozen_test_scenes={"FloorPlan25"},
                                    optimization_pair_ids={"pool_pair"}, frozen_test_pair_ids={"test_pair"})

    def batch(self):
        plan = episode_plan("T3", ["pool_pair"] * 32, rho=0.5, rng=random.Random(0))
        rows = []
        for item in plan:
            first = replace(step_fixture(item["world_id"]), episode_id=item["episode_id"],
                            pair_id=item["pair_id"], pair_instance_id=item["pair_instance_id"],
                            shared_sample_id=item["pair_instance_id"] + "_sample")
            rows.extend([first, replace(first, decision_index=1, pre_evidence=False, evidence_status="entered",
                action_kind="select", action_text="select 1" if item["world_id"] == "w0" else "select 2",
                gae_advantage=1.0, realized_return=1.0, immediate_reward=1.0)])
        return plan, rows, advantages("T3", rows), {item["episode_id"]: {"terminated": True, "truncated": False,
            "environment_steps": 2, "logical_model_calls": 2, "success": True} for item in plan}

    def test_raw_S1_from_first_update_and_S2_only_actual_twentieth_update(self):
        with tempfile.TemporaryDirectory() as directory:
            holdout = self.holdout(directory)
            recorder = RawTrainingRecorder(Path(directory) / "journal", run_id="fixture", arm="T3", seed=0,
                rho=0.5, common_config_sha256="b" * 64, holdout=holdout, allow_synthetic=True, chunk_bytes=128 * 1024)
            plan, rows, applied, summaries = self.batch()
            for update in range(1, 21):
                recorder.begin_update(plan, rows, applied_advantages=applied, advantage_transform="identity",
                                      policy_id="fixture_policy", episode_summaries=summaries)
                due = recorder.finish_update(actual_completed_optimizer_updates=update, policy_checkpoint_id=f"fixture_{update}")
                self.assertEqual(due is not None, update == 20)
            with self.assertRaises(CreditContractError):
                recorder.record_s2([])
            recorder.record_s2([{"eval_episode_id": "heldout_0", "pair_id": "heldout_pair", "scene": "FloorPlan1",
                                 "world_id": "w0", "seed": 0, "success": True}])
            index = recorder.close()
            events = [json.loads(line) for item in index["files"] for line in Path(item["path"]).read_text().splitlines()]
            raw = [row for row in events if row["kind"] == "s1_pre_evidence_decision"]
            self.assertEqual(len(raw), 20 * 64)
            self.assertEqual({row["decision"]["world_id"] for row in raw}, {"w0", "w1"})
            self.assertEqual(raw[0]["completed_optimizer_updates_before"], 0)
            self.assertIn("same_world_unaveraged_credit", raw[0])
            units = [row for row in events if row['kind'] == 's1_independent_decision_unit']
            self.assertEqual(len(units), 20 * 32)
            self.assertAlmostEqual(units[0]['c_u'], .45)
            self.assertEqual(units[0]['episode_equivalent_divisor'], 2)
            evaluation = next(row for row in events if row["kind"] == "s2_evaluation")
            self.assertEqual(evaluation["actual_completed_optimizer_updates"], 20)
            self.assertIsNone(evaluation["auc"])
            self.assertIsNone(evaluation["success_rate"])
            self.assertTrue(index["complete_recording"])

    def test_actual_update_counter_and_shared_applied_advantage_are_guarded(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawTrainingRecorder(Path(directory) / "journal", run_id="fixture", arm="T3", seed=0,
                rho=0.5, common_config_sha256="b" * 64, holdout=self.holdout(directory), allow_synthetic=True)
            plan, rows, applied, summaries = self.batch()
            corrupted = dict(applied)
            corrupted[rows[0].key] += 1
            with self.assertRaises(CreditContractError):
                recorder.begin_update(plan, rows, applied_advantages=corrupted, advantage_transform="unapproved_per_world",
                                      policy_id="fixture_policy", episode_summaries=summaries)
            recorder.begin_update(plan, rows, applied_advantages=applied, advantage_transform="identity",
                                  policy_id="fixture_policy", episode_summaries=summaries)
            with self.assertRaises(CreditContractError):
                recorder.finish_update(actual_completed_optimizer_updates=21, policy_checkpoint_id="fixture")
            recorder.finish_update(actual_completed_optimizer_updates=0, policy_checkpoint_id="fixture", error="synthetic optimizer failure")
            index = recorder.close()
            self.assertEqual(index["actual_completed_optimizer_updates"], 0)

    def test_recorded_episode_and_physical_token_counts_are_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawTrainingRecorder(Path(directory) / 'journal', run_id='fixture', arm='T3', seed=0,
                rho=.5, common_config_sha256='b' * 64, holdout=self.holdout(directory), allow_synthetic=True)
            plan, rows, applied, summaries = self.batch()
            summaries = {key: dict(value, logical_trajectory_token_count=8) for key, value in summaries.items()}
            accounting = {'physical_policy_sampling_calls': 96, 'actual_generated_token_count': 384,
                          'logical_trajectory_token_count': 512, 'teacher_forced_scoring_calls': 32}
            options = dict(applied_advantages=applied, advantage_transform='identity', policy_id='fixture_policy')
            broken = copy.deepcopy(summaries)
            broken[plan[0]['episode_id']]['environment_steps'] = 1
            with self.assertRaises(CreditContractError):
                recorder.begin_update(plan, rows, episode_summaries=broken, rollout_accounting=accounting, **options)
            with self.assertRaises(CreditContractError):
                recorder.begin_update(plan, rows, episode_summaries=summaries,
                    rollout_accounting=dict(accounting, actual_generated_token_count=100), **options)
            recorder.begin_update(plan, rows, episode_summaries=summaries, rollout_accounting=accounting, **options)
            recorder.finish_update(actual_completed_optimizer_updates=1, policy_checkpoint_id='fixture_1')
            self.assertTrue(recorder.close()['complete_recording'])

    def test_S2_cannot_use_frozen_test_or_training_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            holdout = self.holdout(directory)
            with self.assertRaises(CreditContractError):
                TrainingHoldout.load(holdout.manifest_path, train_scenes={"FloorPlan1"}, frozen_test_scenes={"FloorPlan1"},
                                     optimization_pair_ids=set(), frozen_test_pair_ids=set())
            with self.assertRaises(CreditContractError):
                TrainingHoldout.load(holdout.manifest_path, train_scenes={"FloorPlan1"}, frozen_test_scenes={"FloorPlan25"},
                                     optimization_pair_ids={"heldout_pair"}, frozen_test_pair_ids=set())

    def test_S1_independent_units_keep_world_weights_and_observed_zero_pre_steps(self):
        rows = [replace(step_fixture(world, rho=.25), gae_advantage=2 if world == 'w0' else -2)
                for world in ('w0', 'w1')]
        plan = [{'episode_id': row.episode_id, 'pair_id': row.pair_id, 'pair_instance_id': row.pair_instance_id,
                 'world_id': row.world_id, 'rho': row.rho} for row in rows]
        unit = independent_credit_units('T1', plan, rows, advantages('T1', rows))[0]
        self.assertEqual(unit['c_u'], 1.0)  # (1.5*2 + .5*(-2))/2
        self.assertEqual([part['weighted_credit_sum'] for part in unit['world_components']], [3, -1])
        post = [replace(row, pre_evidence=False, tau=0, evidence_status='entered') for row in rows]
        zero = independent_credit_units('T1', plan, post, advantages('T1', post))[0]
        self.assertEqual(zero['c_u'], 0.0)
        self.assertEqual(len(zero['world_components']), 2)
        single = independent_credit_units('T0', plan[:1], rows[:1], advantages('T0', rows[:1]))[0]
        self.assertEqual(single['c_u'], 2)
        with self.assertRaises(CreditContractError):
            independent_credit_units('T1', plan, rows[:1], advantages('T1', rows[:1]))


if __name__ == "__main__":
    unittest.main()
