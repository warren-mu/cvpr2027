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
    q_commit, reference_gae, symmetric_kl)
from src.rl.credit_records import RawTrainingRecorder, TrainingHoldout
from src.rl.paired_rollout import (EpisodeState, PolicySample, PublicMark, PublicObservation,
                                 Transition, collect_pair)

INPUT = "a" * 64


def score_fixture(probabilities=None, *, input_sha256=INPUT):
    probabilities = probabilities or {"Cup_A": ("select 1", 0.5), "Cup_B": ("select 2", 0.5)}
    return ActionScores(tuple(CandidateScore(name, text, math.log(p), (math.log(p),))
                              for name, (text, p) in probabilities.items()),
                        input_sha256, "fixture_policy", "fixture_scores", "synthetic_complete_action_including_stop",
                        complete_action_events=True)


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
        evidence_status="not_entered_verified", evidence_provenance_verified=False, synthetic_fixture=True)


def observation(value=7, text="Select the cup by its visible mark.", label=1):
    return PublicObservation(bytes([value] * 12), 2, 2, "uint8", 3, text,
                             (PublicMark(label, (0, 0), (0, 0, 1, 1)),))


class FixturePort:
    def __init__(self, world, frames, evidence=None):
        self.world, self.frames = world, frames
        self.evidence = evidence or [False] * len(frames)
        self.index, self.calls, self.actions = 0, [], []

    def state(self):
        image = self.frames[self.index]
        entered = self.evidence[self.index]
        return EpisodeState(image, {"world_id": self.world, "pair_id": "pool_pair",
            "episode_id": "episode_" + self.world, "decision_index": self.index, "kind": "look_flip",
            "evidence_entered": entered, "evidence_status": "entered" if entered else "not_entered_verified",
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
        row = replace(step_fixture(), pre_evidence=False, evidence_status="entered", gae_advantage=-0.123, scores=None)
        for arm in ("T0", "T1", "T2", "T3", "T4"):
            self.assertEqual(advantages(arm, [row])[row.key], -0.123)

    def test_unconditional_Q_mass_and_complete_action_scoring(self):
        scores = score_fixture({"Cup_A": ("select 9", 0.2), "Cup_B": ("select 2", 0.3)})
        self.assertAlmostEqual(q_commit(scores, "Cup_A"), 0.2)
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
        post = [replace(row, pre_evidence=False, evidence_status="entered", input_sha256=str(i) * 64,
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


class PairedRolloutTests(unittest.TestCase):
    def policy(self):
        calls = []
        def sample(context):
            self.assertFalse(hasattr(context, "privileged"))
            self.assertNotIn("PRIVATE_TARGET", repr(context))
            calls.append(context)
            return PolicySample("turn_left", "public response: turn_left", "fixture_policy", f"sample_{len(calls)}", 4)
        return sample, calls

    def test_one_shared_sample_then_permanent_fork_without_promoting_noise(self):
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

    def test_S2_cannot_use_frozen_test_or_training_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            holdout = self.holdout(directory)
            with self.assertRaises(CreditContractError):
                TrainingHoldout.load(holdout.manifest_path, train_scenes={"FloorPlan1"}, frozen_test_scenes={"FloorPlan1"},
                                     optimization_pair_ids=set(), frozen_test_pair_ids=set())
            with self.assertRaises(CreditContractError):
                TrainingHoldout.load(holdout.manifest_path, train_scenes={"FloorPlan1"}, frozen_test_scenes={"FloorPlan25"},
                                     optimization_pair_ids={"heldout_pair"}, frozen_test_pair_ids=set())


if __name__ == "__main__":
    unittest.main()
