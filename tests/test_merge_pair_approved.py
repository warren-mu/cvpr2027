"""Synthetic approved-merger boundaries; no simulator/results or GPU execution."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.envs.pair_approved_policy import POLICY
from src.tools.merge_pair_approved import (SelectionChunks, add_candidate, approved_reasons,
                                           measurement_policy, require_source_manifest, verify_raw_attempt)


def fixture():
    point = lambda x: {"x": x, "y": 0.0, "z": 0.0}
    pose = lambda name, x: {"objectName": name, "position": point(x), "rotation": point(0)}
    settling = {"settled": True, "last_observation": {"native_isSceneAtRest": True, "stable": True,
                "inventory_exact": True, "consecutive_stable": 5}}
    common = {"valid_physical": True, "native_collision_clear": True, "errors": [],
              "overlapping_colliders": {"cup": [], "other": []}, "settling": settling}
    edited = {**copy.deepcopy(common), "acceptance_policy": copy.deepcopy(POLICY), "inventory_exact": True,
              "missing": [], "added": [], "position_bounds_passed": True, "legal_receptacle_placement": True,
              "destination_member_after_settling": True, "destination_contains_after_settling": ["cup_new_id"],
              "edited_settling_position_error_m": 0.019, "maximum_other_object_displacement_m": 0.0,
              "all_object_pose_deltas": {
                  "cup": {"before_position": point(0), "after_position": point(2.019), "position_delta_m": 2.019},
                  "other": {"before_position": point(0), "after_position": point(0), "position_delta_m": 0.0}}}
    prefix = {"same_input_at_t_strict": True, "rgb_exact": True, "marked_rgb_exact": True, "text_exact": True,
              "rgb_max_abs_diff": 0, "marked_max_abs_diff": 0, "prefix_max_abs_diff": 0, "different_pixels": 0}
    witnesses = []
    for name in ("cup", "other"):
        actions = [{"action": "turn_left"}, {"action": "select 1", "success": True, "target_name": name,
                    "resolved_target_name": name, "distance_m": 1.5}]
        witnesses.append({"plan": {"found": True}, "replay": {"passed": True, "selection_success": True,
                          "prefix_replay_exact": True, "first_observation_exact": True, "actions": actions}})
    return {"acceptance_policy": copy.deepcopy(POLICY), "kind": "look_flip", "status": "passed", "scene": "FloorPlan3",
        "seed": 0, "discard_reasons": [], "restore_deterministic": True, "valid_physical": True, "targets_separable": True,
        "validity": {key: True for key in ("restore_deterministic", "valid_physical", "unique_targets", "witness_replay", "evidence_available", "targets_separable")},
        "history": "empty before single initial frame", "same_input_at_t": True, "same_input_at_t_strict": True,
        "prefix_max_abs_diff": 0, "prefix_comparison": prefix, "common_observation_action": "turn_left",
        "immediate_action_labels": ["turn_left", "turn_left"], "immediate_target_labels": None,
        "target_name_w0": "cup", "target_name_w1": "other", "evidence_pixels_after_look": 193,
        "evidence_comparison": {"evidence_resolution": [224, 224], "relevant_difference_pixels": 193,
            "relevant_mask_pixels": 500, "acceptance_policy": POLICY["id"], "evidence_channel_difference_exclusive": 10, "evidence_passed": True},
        "physical_checks": [copy.deepcopy(common), edited],
        "edit": {"object_name": "cup", "requested_pose": pose("cup", 2), "native_proposal_verified": True,
                 "proposal_pose_changed": True, "proposal_receptacle_member": True},
        "w0": {"poses": [pose("cup", 0), pose("other", 0)]}, "w1": {"poses": [pose("cup", 2.019), pose("other", 0)]},
        "requested_w1": {"poses": [pose("cup", 2), pose("other", 0)]},
        "restore_checks": [{"passed": True, "comparison": copy.deepcopy(prefix)} for _ in range(2)],
        "witnesses": witnesses, "witness_actions_w0": witnesses[0]["replay"]["actions"], "witness_actions_w1": witnesses[1]["replay"]["actions"]}


class ApprovedMergeTests(unittest.TestCase):
    def test_legacy_collect_only_cannot_be_promoted_even_if_all_flags_claim_pass(self):
        row = fixture()
        policy = measurement_policy({"collect_only": True, "acceptance_policy": POLICY}, {})
        self.assertEqual(policy, "legacy_collect_only")
        self.assertTrue(approved_reasons(row, policy))
        self.assertEqual(approved_reasons(row, POLICY["id"]), [])

    def test_policy_is_exact_and_early_failure_stays_unavailable(self):
        self.assertEqual(measurement_policy({"collect_only": False, "acceptance_policy": POLICY}, {}), POLICY["id"])
        bad = {**POLICY, "edited_settling_position_error_max_m": .03}
        with self.assertRaises(ValueError):
            measurement_policy({"collect_only": False, "acceptance_policy": bad}, {"attempts": 1})
        self.assertEqual(measurement_policy({"collect_only": False}, {"status": "failed_before_generation", "attempts": 0}), "unavailable_before_generation")

    def test_new_measurements_require_v2_and_legacy_only_uses_v1(self):
        approved = {"sha256": "a" * 64, "manifest_id": "approved_v2"}
        legacy = {"sha256": "b" * 64, "manifest_id": "legacy_v1"}
        require_source_manifest(approved, False, approved, legacy)
        require_source_manifest(legacy, True, approved, legacy)
        for ref, mode in ((legacy, False), (approved, True), ({}, False)):
            with self.assertRaises(ValueError):
                require_source_manifest(ref, mode, approved, legacy)

    def test_low_noise_or_text_difference_never_becomes_exact_look(self):
        for change in (lambda r: r.update(prefix_max_abs_diff=1),
                       lambda r: r["prefix_comparison"].update(text_exact=False),
                       lambda r: r.update(same_input_at_t_strict=False)):
            row = fixture()
            change(row)
            self.assertIn("look_prefix_not_exact", approved_reasons(row, POLICY["id"]))
        row = fixture()
        row["status"] = "pending_review"
        self.assertIn("source_did_not_pass", approved_reasons(row, POLICY["id"]))

    def test_strict_area_evidence_needs_recorded_threshold_and_mask(self):
        for mutate in (lambda r: r["evidence_comparison"].update(evidence_channel_difference_exclusive=0),
                       lambda r: r["evidence_comparison"].update(relevant_difference_pixels=192),
                       lambda r: r["evidence_comparison"].pop("relevant_mask_pixels"),
                       lambda r: r["evidence_comparison"].update(relevant_mask_pixels=10)):
            row = fixture()
            mutate(row)
            self.assertIn("missing_or_insufficient_strict_evidence", approved_reasons(row, POLICY["id"]))

    def test_distance_of_edit_is_not_settling_error_and_missing_physics_fails(self):
        self.assertEqual(approved_reasons(fixture(), POLICY["id"]), [])  # Edit moves >2m but settles within2cm.
        for mutate in (lambda r: r["physical_checks"][1].update(edited_settling_position_error_m=.0201),
                       lambda r: r["physical_checks"][1].update(maximum_other_object_displacement_m=.0101),
                       lambda r: r["physical_checks"][1].pop("all_object_pose_deltas"),
                       lambda r: r["physical_checks"][0]["settling"].update(settled=False),
                       lambda r: r["physical_checks"][1].update(legal_receptacle_placement=False)):
            row = fixture()
            mutate(row)
            self.assertIn("missing_or_failed_measured_physics", approved_reasons(row, POLICY["id"]))

    def test_single_edit_and_complete_inventory_are_not_inferred(self):
        row = fixture()
        row["requested_w1"]["poses"][1]["position"]["x"] = 0.1
        self.assertIn("missing_or_failed_measured_physics", approved_reasons(row, POLICY["id"]))
        row = fixture()
        row["w1"]["poses"].pop()
        self.assertIn("missing_or_failed_measured_physics", approved_reasons(row, POLICY["id"]))

    def test_required_validity_and_successful_replay_are_checked(self):
        row = fixture()
        row["validity"]["valid_physical"] = None
        self.assertIn("missing_or_failed_validity", approved_reasons(row, POLICY["id"]))
        row = fixture()
        row["witnesses"][1]["replay"]["actions"][-1]["distance_m"] = 1.5001
        self.assertIn("missing_or_failed_restore_witness", approved_reasons(row, POLICY["id"]))

    def test_cross_policy_old_measurement_does_not_choose_or_block_new_outcome(self):
        groups, old, new = {}, fixture(), fixture()
        old.update(status="collected", same_input_at_t_strict=False)
        add_candidate(groups, "same_physical_pair", old, {"measurement_policy": "legacy_collect_only"})
        self.assertIsNone(groups["same_physical_pair"]["row"])
        add_candidate(groups, "same_physical_pair", new, {"measurement_policy": POLICY["id"]})
        self.assertIs(groups["same_physical_pair"]["row"], new)
        self.assertEqual(groups["same_physical_pair"]["statuses"], {"passed"})
        conflict = copy.deepcopy(new)
        conflict["physical_checks"][1]["edited_settling_position_error_m"] = .03
        add_candidate(groups, "same_physical_pair", conflict, {"measurement_policy": POLICY["id"]})
        self.assertEqual(len(groups["same_physical_pair"]["outcomes"]), 2)

    def test_missing_full_evidence_is_not_synthesized(self):
        with tempfile.TemporaryDirectory() as directory:
            row = {"raw_attempt_evidence": {"path": str(Path(directory) / "absent.json"), "bytes": 0, "sha256": "0" * 64}}
            with self.assertRaises(OSError):
                verify_raw_attempt(row)

    def test_portable_outputs_are_bounded_isolated_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, patch("src.tools.merge_pair_approved.ROOT", Path(directory)):
            path = Path(directory) / "named" / "pilot"
            writer = SelectionChunks(path, "pilot", limit=40)
            writer.write({"x": "a" * 12})
            writer.write({"x": "b" * 12})
            refs = writer.close()
            self.assertEqual(len(refs), 2)
            self.assertTrue(all(ref["bytes"] <= 40 for ref in refs))
            with self.assertRaises(FileExistsError):
                SelectionChunks(path, "pilot")
            empty = SelectionChunks(Path(directory) / "named" / "pending", "pending").close()
            self.assertEqual(empty[0]["bytes"], 0)


if __name__ == "__main__":
    unittest.main()
