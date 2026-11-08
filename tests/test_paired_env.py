"""Synthetic CPU-only T02 contracts. These fixtures certify no real rendering."""
from dataclasses import replace
import unittest

from src.envs.pair_core import Action
from src.envs.paired_env import (BackendFrame, Candidate, CaptureEvidence, EpisodeBudget,
                                EvidenceComparison, PairEnvironment, PublicMark, RGBFrame,
                                RewardSpec, compare_prefix, parse_action)
from src.envs.paired_mock import MockObservationBackend, production_backend
from src.envs.paired_vagen import DraftVagenAdapter


def fixture_frame(world="w0", state="start", delta=0, target="private_cup_a", kind="look_flip"):
    raw = RGBFrame(bytes([40 + delta, 50, 60]) * 16, 4, 4)
    # Fixture-only simulated mark paint; no claim that these are THOR pixels.
    marked = RGBFrame(bytes([255, 255, 255]) + raw.data[3:], 4, 4)
    capture = CaptureEvidence("fixture-capture-" + world + "-" + state, "private_pair", world, state,
                              "mock", "fixture-source-" + world + "-" + state, raw.sha256, marked.sha256, True)
    return BackendFrame(raw, marked, "Select the cup beside the kettle.", ((1, "front"),),
                        (Action("move", 1), Action("turn_left"), Action("turn_right"), Action("open", 1)),
                        (PublicMark(1, (0, 0), (0, 0, 1, 1)), PublicMark(2, (2, 2), (2, 2, 3, 3))),
                        (Candidate("private_cup_a", 1, 4, 1.5, True, True), Candidate("private_cup_b", 2, 4, 1.0, True, True)),
                        target, kind, capture)


def paired_frames(state="start", delta=0, kind="look_flip"):
    a = fixture_frame("w0", state, target="private_cup_a", kind=kind)
    b = fixture_frame("w1", state, delta=delta, target="private_cup_b", kind=kind)
    evidence = EvidenceComparison(a.raw_rgb, b.raw_rgb, a.capture, b.capture, tuple(range(16)), a.marked_rgb, b.marked_rgb)
    return replace(a, evidence=evidence), replace(b, evidence=evidence)


def environment(world="w0", start=None, following=None, budget=None):
    a, b = paired_frames()
    start = start or (a if world == "w0" else b)
    if following is None:
        pair = paired_frames("after", 20)
        following = pair[0 if world == "w0" else 1]
    frames = {(row.capture.pair_id, row.capture.world_id, row.capture.state_id): row for row in (start, following)}
    transitions = {("private_pair", world, start.capture.state_id, action.text()): following.capture.state_id
                   for action in start.observation_actions}
    backend = MockObservationBackend(frames, {("private_pair", world): start.capture.state_id}, transitions)
    # -0.25 is explicitly synthetic, NOT a decision on T04/VAGEN format penalty.
    return PairEnvironment(backend, budget or EpisodeBudget(8, 8, 256), RewardSpec(0.05, -0.25))


class PairInterfaceTests(unittest.TestCase):
    def test_same_mock_prefix_is_equal_but_does_not_certify_real_prefix(self):
        left, right = environment(), environment("w1")
        a, b = left.reset("private_pair", "w0"), right.reset("private_pair", "w1")
        self.assertEqual(a, b)
        comparison = compare_prefix(left, right)
        self.assertTrue(comparison["same_policy_input"])
        self.assertTrue(comparison["independent_capture_ids"])
        self.assertFalse(comparison["prefix_verified_identical"])
        self.assertFalse(left.info["capture_provenance_verified"])
        self.assertFalse(left.info["evidence_provenance_verified"])
        self.assertNotEqual(left.info["target_name"], right.info["target_name"])

    def test_policy_input_has_no_privileged_identifiers_or_labels(self):
        env = environment()
        obs = env.reset("private_pair", "w0")
        self.assertEqual(set(obs.policy_dict()), {"image", "text", "marks"})
        for value in ("private_pair", "private_cup_a", "look_flip", "evidence_entered", "target_name", "world_id"):
            self.assertNotIn(value, repr(obs.policy_dict()))
        self.assertIn("select 1", obs.text)
        self.assertIn("Marks:", obs.text)
        self.assertEqual(obs.image, env.frame.marked_rgb)
        self.assertNotEqual(obs.image, env.frame.raw_rgb)

    def test_exact_comparison_includes_image_text_and_marks(self):
        left, right = environment(), environment("w1")
        left.reset("private_pair", "w0")
        right.reset("private_pair", "w1")
        original = right.frame
        right.frame = replace(original, instruction="Other instruction")
        self.assertFalse(compare_prefix(left, right)["same_policy_input"])
        right.frame = replace(original, marks=(replace(original.marks[0], anchor_pixel=(1, 1)), original.marks[1]))
        self.assertFalse(compare_prefix(left, right)["same_policy_input"])
        right.frame = fixture_frame("w1", delta=1)
        self.assertFalse(compare_prefix(left, right)["same_policy_input"])

    def test_select_requires_correct_target_visible_mask_distance_and_category(self):
        base = paired_frames()[0]
        cases = [(base.candidates[0], True),
                 (replace(base.candidates[0], mask_pixel_count=0), False),
                 (replace(base.candidates[0], distance_m=1.5001), False),
                 (replace(base.candidates[0], distance_m=None), False)]
        for candidate, expected in cases:
            with self.subTest(candidate=candidate):
                env = environment(start=replace(base, candidates=(candidate, base.candidates[1])))
                env.reset("private_pair", "w0")
                _, reward, done, info = env.step("select 1")
                self.assertEqual(info["success"], expected)
                self.assertEqual(reward, 1.0 if expected else (0.0 if candidate.public_selectable else -0.25))
                self.assertTrue(done)
        # Private category eligibility changes success, never the public action domain.
        env = environment(start=replace(base, candidates=(base.candidates[0], replace(base.candidates[1], legal_target=False))))
        env.reset("private_pair", "w0")
        _, reward, _, info = env.step("select 2")
        self.assertTrue(info["action_valid"])
        self.assertEqual(reward, 0.0)

    def test_private_distance_and_category_do_not_change_policy_actions(self):
        start = paired_frames()[0]
        first = environment(start=start)
        changed = replace(start, candidates=(replace(start.candidates[0], distance_m=99),
                                             replace(start.candidates[1], legal_target=False)))
        second = environment(start=changed)
        self.assertEqual(first.reset("private_pair", "w0"), second.reset("private_pair", "w0"))
        self.assertEqual(second.step("select 1")[1], 0.0)

    def test_public_non_candidate_mark_is_not_a_select_action(self):
        start = paired_frames()[0]
        env = environment(start=replace(start, candidates=(start.candidates[0], replace(start.candidates[1], public_candidate=False))))
        observation = env.reset("private_pair", "w0")
        self.assertNotIn("select 2", observation.text)
        self.assertFalse(env.step("select 2")[3]["action_valid"])

    def test_wrong_legal_select_is_zero_and_terminal_without_observation(self):
        env = environment()
        env.reset("private_pair", "w0")
        _, reward, done, info = env.step("select 2")
        self.assertEqual(reward, 0.0)
        self.assertTrue(done)
        self.assertTrue(info["action_valid"])
        self.assertFalse(info["success"])
        self.assertEqual(info["obs_count"], 0)
        self.assertEqual(len(env.backend.requests), 1)
        with self.assertRaises(RuntimeError):
            env.step("select 1")

    def test_unknown_select_cannot_alias_another_candidate(self):
        env = environment()
        env.reset("private_pair", "w0")
        _, _, done, info = env.step("select 99")
        self.assertTrue(done)
        self.assertIsNone(info["selected_name"])
        self.assertFalse(info["success"])

    def test_each_observation_action_returns_fresh_frame_and_costs_one(self):
        for action in ("move 1", "turn_left", "turn_right", "open 1"):
            with self.subTest(action=action):
                env = environment()
                env.reset("private_pair", "w0")
                old = env.frame.capture.capture_id
                _, reward, done, info = env.step(action)
                self.assertEqual(reward, -0.05)
                self.assertFalse(done)
                self.assertEqual(info["obs_count"], 1)
                self.assertNotEqual(info["capture"]["capture_id"], old)
                self.assertTrue(info["evidence_entered"])
                self.assertEqual(len(env.backend.requests), 2)

    def test_multiple_actions_never_execute_a_partial_prefix(self):
        env = environment()
        env.reset("private_pair", "w0")
        _, reward, _, info = env.step("move 1, select 1")
        self.assertEqual(reward, -0.25)
        self.assertFalse(info["action_valid"])
        self.assertEqual(info["obs_count"], 0)
        self.assertEqual(len(env.backend.requests), 1)
        for text in ("move 0", "select 01", "open -1", "turn_left turn_right"):
            with self.assertRaises(ValueError):
                parse_action(text)

    def test_common_step_and_model_call_budgets(self):
        env = environment(budget=EpisodeBudget(1, 2, 256))
        env.reset("private_pair", "w0")
        self.assertTrue(env.step("turn_left")[2])
        with self.assertRaises(RuntimeError):
            env.step("select 1")
        env = environment(budget=EpisodeBudget(8, 1, 256))
        env.reset("private_pair", "w0")
        env.record_model_call("actual-policy-sample")
        with self.assertRaises(RuntimeError):
            env.record_model_call("second-sample")
        self.assertTrue(env.step("turn_left")[2])
        self.assertEqual(env.info["model_calls"], 1)

    def test_shared_sample_id_counted_once_per_logical_world(self):
        a, b = environment(), environment("w1")
        a.reset("private_pair", "w0")
        b.reset("private_pair", "w1")
        for env in (a, b):
            env.record_model_call("shared-physical-sample")
            self.assertEqual(env.info["model_calls"], 1)
            with self.assertRaises(ValueError):
                env.record_model_call("shared-physical-sample")

    def test_missing_evidence_is_unknown_not_false_certification(self):
        following = replace(paired_frames("after", 20)[0], evidence=None)
        env = environment(following=following)
        env.reset("private_pair", "w0")
        self.assertEqual(env.info["evidence_status"], "not_entered_verified")
        info = env.step("turn_left")[3]
        self.assertEqual(info["evidence_status"], "unknown")
        self.assertFalse(info["evidence_entered"])

    def test_backend_failure_stops_episode_instead_of_reusing_stale_pixels(self):
        env = environment()
        env.reset("private_pair", "w0")
        env.backend.transitions.clear()
        with self.assertRaises(KeyError):
            env.step("move 1")
        self.assertTrue(env.done)
        with self.assertRaises(RuntimeError):
            env.step("select 1")

    def test_entered_evidence_stays_in_history_after_later_unknown_frame(self):
        start = paired_frames(delta=20)[0]
        following = replace(paired_frames("after", 0)[0], evidence=None)
        env = environment(start=start, following=following)
        env.reset("private_pair", "w0")
        info = env.step("turn_left")[3]
        self.assertTrue(info["evidence_entered"])
        self.assertEqual(info["evidence_status"], "entered")

    def test_noise_and_threshold_evidence_are_distinct(self):
        for delta, expected in ((0, False), (3, False), (10, False), (11, True)):
            with self.subTest(delta=delta):
                evidence = paired_frames("after", delta)[0].evidence
                self.assertEqual(evidence.entered, expected)
        evidence = paired_frames("after", 20)[0].evidence
        self.assertFalse(replace(evidence, relevant_mask_union_pixels=()).entered)
        self.assertEqual(evidence.required_pixels, 1)  # 4x4 synthetic image, area scaled.

    def test_evidence_rejects_same_capture_or_source_relabeling(self):
        proof = paired_frames()[0].evidence
        for bad in (replace(proof.right_capture, capture_id=proof.left_capture.capture_id),
                    replace(proof.right_capture, source_id=proof.left_capture.source_id),
                    replace(proof.right_capture, world_id="w0")):
            with self.assertRaises(ValueError):
                replace(proof, right_capture=bad)

    def test_mark_overlay_cannot_hide_evidence_and_still_claim_entered(self):
        proof = paired_frames("after", 20)[0].evidence
        hidden = replace(proof, marked_right=proof.marked_left,
                         right_capture=replace(proof.right_capture, marked_rgb_sha256=proof.marked_left.sha256))
        self.assertGreaterEqual(hidden.changed_pixels, hidden.required_pixels)
        self.assertFalse(hidden.entered)

    def test_direct_requires_actual_initial_evidence(self):
        with self.assertRaises(ValueError):
            environment(start=paired_frames(kind="direct_flip")[0]).reset("private_pair", "w0")
        env = environment(start=paired_frames(delta=20, kind="direct_flip")[0])
        env.reset("private_pair", "w0")
        self.assertTrue(env.info["evidence_entered"])

    def test_real_backend_unimplemented_and_mock_no_cross_world_fallback(self):
        for kind in ("L", "P"):
            with self.assertRaises(NotImplementedError):
                production_backend(kind)
        env = environment()
        with self.assertRaises(KeyError):
            env.reset("private_pair", "w1")
        env.close()
        with self.assertRaises(RuntimeError):
            env.reset("private_pair", "w0")

    def test_real_identity_cannot_bypass_provenance_verification(self):
        frame = fixture_frame()
        real = replace(frame, capture=replace(frame.capture, backend_kind="L", synthetic=False))
        class UnverifiedBackend:
            def reset(self, pair, world):
                return real
            def verify_capture(self, observation):
                return False
        env = PairEnvironment(UnverifiedBackend(), EpisodeBudget(8, 8, 256), RewardSpec(0.05, -0.25))
        with self.assertRaises(ValueError):
            env.reset("private_pair", "w0")

    def test_backend_capture_hash_and_candidate_labels_are_checked(self):
        frame = fixture_frame()
        with self.assertRaises(ValueError):
            replace(frame, capture=replace(frame.capture, raw_rgb_sha256="0" * 64))
        with self.assertRaises(ValueError):
            replace(frame, candidates=(frame.candidates[0], replace(frame.candidates[1], mark_label=1)))
        with self.assertRaises(ValueError):
            RGBFrame(b"short", 4, 4)

    def test_vagen_scaffold_exposes_public_image_and_separate_info(self):
        env = environment()
        received = []
        def image_adapter(frame):
            received.append(frame)
            return frame.data  # Synthetic serialization only, not a VAGEN PIL integration.
        adapter = DraftVagenAdapter(env, lambda seed: ("private_pair", "w0"), image_adapter, lambda text: text, len)
        obs, info = adapter.reset(0)
        self.assertEqual(set(obs), {"obs_str", "multi_modal_data"})
        self.assertNotIn("private_cup_a", obs["obs_str"])
        self.assertEqual(received[0], env.frame.marked_rgb)
        self.assertEqual(info["target_name"], "private_cup_a")
        _, reward, done, info = adapter.step("select 1")
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)
        self.assertEqual(info["model_calls"], 1)
        self.assertEqual(adapter.compute_reward(), 0.0)

    def test_actual_token_counter_blocks_overlong_response_without_action(self):
        env = environment()
        adapter = DraftVagenAdapter(env, lambda seed: ("private_pair", "w0"), lambda rgb: rgb.data,
                                   lambda text: "move 1", lambda text: 257)
        adapter.reset(0)
        _, _, _, info = adapter.step("fixture tokenizer reports too many tokens")
        self.assertTrue(info["response_token_budget_exceeded"])
        self.assertFalse(info["action_valid"])
        self.assertEqual(info["obs_count"], 0)
        self.assertEqual(len(env.backend.requests), 1)


if __name__ == "__main__":
    unittest.main()
