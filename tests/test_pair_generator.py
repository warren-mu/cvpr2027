"""Synthetic CPU checks only; no simulator import, no research output files."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.envs.pair_core import Mark, Point3, RelationalTask, TargetResolution, ViewState
from src.envs.pair_generator import REQUIRED_CHECKS, build_attempt, classify, check_configuration, predefined_split
from src.envs.pair_classify import classify_row
from src.envs.pair_simulator import PairSimulator, path_complete, pose_changes


class PairValidationTests(unittest.TestCase):
    def test_flip_stay_uses_realized_name_despite_regenerated_object_ids(self):
        class PrefixReached(Exception):
            pass

        task = RelationalTask("Cup", "next to", "Kettle")
        config = {"seed": 0, "scene_split": {"test_scenes": []}, "scene_world": {},
                  "collect_only": True, "max_displacement_m": None}
        for second_name, expected in (("Cup_1", "stay"), ("Cup_2", "flip")):
            with self.subTest(second_name=second_name):
                ids = ("Cup|+00.00|+01.00|+00.00", "Cup|+00.50|+01.00|+00.00")
                sim = Mock(scene="FloorPlan1", nodes=())
                sim.world_from_event.return_value = {"key": "w0"}
                sim.relation.side_effect = [
                    (TargetResolution(key, "anchor", (key,), "unique", "fixture"),
                     {"name_by_object_id": {key: name}})
                    for key, name in zip(ids, ("Cup_1", second_name))]
                row = {}
                with patch("src.envs.pair_generator.propose_edit", return_value=({"key": "w1"}, {"object_name": "Cup_1"})), \
                     patch("src.envs.pair_generator.choose_prefix", side_effect=PrefixReached):
                    with self.assertRaises(PrefixReached):
                        build_attempt(sim, config, "pair_00000", "direct_stay", task,
                                      None, Path("unused_no_writes"), row)
                self.assertEqual(row["actual_target_change"], expected)
                self.assertEqual(row["kind"], "direct_" + expected)
                self.assertEqual((row["target_w0"], row["target_w1"]), ids)
                self.assertEqual(row["target_name_w1"], second_name)

    def test_openness_and_common_container_state_survive_id_regeneration(self):
        zero = {"x": 0, "y": 0, "z": 0}

        class ControllerFixture:
            def __init__(self):
                self.calls = []
                self.last_event = SimpleNamespace(metadata={"objects": [{
                    "name": "Box_1", "objectId": "Box|old", "pickupable": True,
                    "openable": True, "openness": 0.0, "isOpen": False,
                    "position": dict(zero), "rotation": dict(zero)}],
                    "agent": {"position": zero, "rotation": zero, "cameraHorizon": 0, "isStanding": True}})

            def step(self, action, **kwargs):
                self.calls.append((action, kwargs))
                obj = self.last_event.metadata["objects"][0]
                if action == "SetObjectPoses":
                    obj["objectId"] = "Box|regenerated"
                else:
                    if kwargs["objectId"] != obj["objectId"]:
                        raise AssertionError("Native action received stale objectId")
                    obj.update(openness=kwargs.get("openness", 0), isOpen=action == "OpenObject")
                self.last_event.metadata.update(lastActionSuccess=True, errorMessage="", actionReturn=None)
                return self.last_event

        controller = ControllerFixture()
        sim = PairSimulator(controller, {"scene": "FloorPlan1", "deadline": float("inf"), "max_open_containers": 1})
        world = sim.world_from_event("w0")
        self.assertEqual(world["openness"], {"Box_1": 0.0})
        sim.apply_world(world, ("Box_1",))
        self.assertEqual(controller.calls[-1][1]["objectId"], "Box|regenerated")
        self.assertTrue(controller.last_event.metadata["objects"][0]["isOpen"])
        observation = {"metadata": {"objects": [{"objectId": "Box|old", "name": "Box_1"}]},
                       "mark_objects": [Mark(1, "Box|old", ("container",), (0, 0, 0, 0), (0, 0), 1)]}
        with patch.object(sim, "graph", return_value=None), patch.object(sim, "teleport"), \
             patch("src.envs.pair_simulator.navigation_transitions", return_value=()):
            transitions = sim.successors(world, ViewState("v0", 0), observation)
        self.assertEqual(transitions[0].next_state.opened_containers, ("Box_1",))
        self.assertEqual(controller.calls[-1][1]["objectId"], "Box|regenerated")

    def test_pose_comparison_matches_name_not_position_derived_id(self):
        obj = {"name": "Cup_1", "objectId": "old", "position": {"x": 0, "y": 1, "z": 0},
               "rotation": {"x": 0, "y": 0, "z": 0}}
        self.assertTrue(pose_changes([obj], [{**obj, "objectId": "regenerated"}])["all_object_poses_exact"])

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
