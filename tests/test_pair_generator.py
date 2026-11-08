"""Synthetic CPU checks only; no simulator import, no research output files."""
import copy
import unittest

from src.envs.pair_core import Point3
from src.envs.pair_generator import REQUIRED_CHECKS, classify, check_configuration, predefined_split
from src.envs.pair_classify import classify_row
from src.envs.pair_simulator import path_complete, pose_changes


class PairValidationTests(unittest.TestCase):
    def test_low_difference_does_not_change_strict_fields_or_pass(self):
        flags = dict.fromkeys(REQUIRED_CHECKS, True)
        prefix = {"same_input_at_t_strict": False, "text_exact": True, "prefix_max_abs_diff": 3}
        self.assertEqual(classify("look_flip", flags, prefix),
                         ("pending_review", ["same_input_at_t"], True))
        self.assertFalse(prefix["same_input_at_t_strict"])
        flags["restore_deterministic"] = False
        status, reasons, pending = classify("look_stay", flags, prefix)
        self.assertEqual(status, "pending_review")
        self.assertIn("restore_deterministic", reasons)
        self.assertTrue(pending)

    def test_text_mismatch_or_diff_above_three_is_not_low_diff_candidate(self):
        for text, difference in ((False, 1), (True, 4)):
            self.assertEqual(classify("look_flip", {"valid_physical": True},
                {"same_input_at_t_strict": False, "text_exact": text, "prefix_max_abs_diff": difference})[0], "rejected")

    def test_direct_identical_input_is_rejected_even_with_different_targets(self):
        self.assertEqual(classify("direct_flip", {"unique_targets": True},
            {"same_input_at_t_strict": True, "text_exact": True, "prefix_max_abs_diff": 0})[0], "rejected")

    def test_missing_check_is_not_true(self):
        self.assertEqual(classify("look_stay", {"valid_physical": None},
            {"same_input_at_t_strict": True, "text_exact": True, "prefix_max_abs_diff": 0})[0], "rejected")
        self.assertEqual(classify("look_stay", {},
            {"same_input_at_t_strict": True, "text_exact": True, "prefix_max_abs_diff": 0})[0], "rejected")

    def test_path_needs_complete_status_and_both_endpoints(self):
        a, b = Point3(0, 1, 0), Point3(1, 1, 0)
        raw = {"status": "PathComplete", "corners": [{"x": 0, "y": 0, "z": 0}, {"x": 1, "y": 0, "z": 0}]}
        self.assertTrue(path_complete(raw, a, b, 0.01))
        for bad in ({**raw, "status": 1}, {"corners": raw["corners"]},
                    {**raw, "corners": [{"x": 0.1, "y": 0, "z": 0}, raw["corners"][1]]}):
            self.assertFalse(path_complete(bad, a, b, 0.01))

    def test_pose_changes_do_not_hide_tiny_collateral_motion(self):
        before = [{"name": "a", "position": {"x": 0, "y": 0, "z": 0}, "rotation": {"x": 0, "y": 0, "z": 0}}]
        after = copy.deepcopy(before)
        after[0]["position"]["x"] = 1e-12
        result = pose_changes(before, after)
        self.assertFalse(result["all_object_poses_exact"])
        self.assertEqual(result["changed"][0]["position_delta_m"], 1e-12)

    def test_split_is_predeclared_whole_scene_holdout(self):
        split = predefined_split()
        self.assertEqual(split["test_fraction"], 0.2)
        self.assertIn("FloorPlan25", split["test_scenes"])
        self.assertIn("FloorPlan225", split["test_scenes"])
        self.assertNotIn("FloorPlan25", split["train_scenes"])

    def test_thresholds_required_for_formal_mode(self):
        config = {"collect_only": False, "max_displacement_m": None, "min_evidence_pixels": None,
            "viewpoints": 12, "timeout_seconds": 120, "pairs_per_class": 20, "max_attempts": 1,
            "max_search_states": 12, "distance_tie_tolerance_m": 0, "path_endpoint_tolerance_m": 0.01}
        with self.assertRaises(ValueError):
            check_configuration(config)
        config["collect_only"] = True
        check_configuration(config)

    def test_post_classification_preserves_raw_exactness_and_physics_failure(self):
        physical = {"native_collision_clear": True, "actual_edited_displacement_m": 0.2,
                    "pose_changes": {"inventory_exact": True}, "collateral_pose_changes": []}
        row = {"status": "collected", "kind": "look_flip", "same_input_at_t": False,
               "same_input_at_t_strict": False, "physical_checks": [physical, physical],
               "validity": {"restore_deterministic": False, "valid_physical": None, "evidence_available": None},
               "evidence_pixels_after_look": 20,
               "prefix_comparison": {"same_input_at_t_strict": False, "text_exact": True, "prefix_max_abs_diff": 2}}
        classified = classify_row(row, 0.5, 10)
        self.assertEqual(classified["status"], "pending_review")
        self.assertTrue(classified["pending_has_other_failures"])
        self.assertFalse(classified["same_input_at_t"])
        self.assertIsNone(row["validity"]["valid_physical"])


if __name__ == "__main__":
    unittest.main()
