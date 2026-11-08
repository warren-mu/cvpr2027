"""Synthetic CPU boundary/placement fixtures; no Unity or GPU initialization."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np

from src.envs.pair_approved_policy import compare_inputs, evidence_passes, measure_settled_poses
from src.envs.pair_generator_approved import classify
from src.envs.pair_simulator_approved import PairSimulator, SETTLING


def obj(name, x=0, *, target=False, receptacle=False):
    return {"name": name, "objectId": name + "_id", "objectType": "Mug" if target else "Table",
            "position": {"x": x, "y": 0.0, "z": 0.0}, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "pickupable": target, "moveable": False, "receptacle": receptacle, "isMoving": False}


def world():
    objects = [obj("mug", target=True), obj("table", receptacle=True), obj("other")]
    return {"key": "synthetic", "scene": "fixture", "objects": objects, "openness": {},
            "poses": [{"objectName": "mug", "position": copy.deepcopy(objects[0]["position"]),
                       "rotation": copy.deepcopy(objects[0]["rotation"])}]}


class FakeSimulator(PairSimulator):
    def __init__(self, *, moved=True, member=True, continuous_motion=False):
        self.config = {"scene": "fixture"}
        self.scene = "fixture"
        self.moved, self.member, self.continuous_motion = moved, member, continuous_motion
        self.calls = []
        self.controller = SimpleNamespace(last_event=SimpleNamespace(metadata={"objects": world()["objects"]}))

    def apply_world(self, value, opened=()):
        self.controller.last_event.metadata["objects"] = copy.deepcopy(value["objects"])

    def world_from_event(self, key):
        value = world()
        value.update(key=key, objects=copy.deepcopy(self.controller.last_event.metadata["objects"]))
        value["poses"][0]["position"] = copy.deepcopy(value["objects"][0]["position"])
        return value

    def action(self, name, **kwargs):
        self.calls.append((name, kwargs))
        metadata = copy.deepcopy(self.controller.last_event.metadata)
        self.controller.last_event = SimpleNamespace(metadata=metadata)
        if name == "InitialRandomSpawn" and self.moved:
            metadata["objects"][0]["position"]["x"] = 2.0
            # Side effects are observed but must not become extra requested edits.
            metadata["objects"][2]["position"]["x"] = 5.0
        if name == "Contains":
            metadata["actionReturn"] = ["mug_id"] if self.member else []
        if name == "AdvancePhysicsStep":
            metadata["isSceneAtRest"] = True  # Native acceleration-based flag alone is insufficient.
            if self.continuous_motion:
                metadata["objects"][0]["position"]["x"] += 0.01
        return self.controller.last_event


class ApprovedCriteria(unittest.TestCase):
    def test_evidence_boundary_is_strict_channel_and_area_scaled(self):
        self.assertFalse(evidence_passes(249, 255, 255))
        self.assertTrue(evidence_passes(250, 255, 255))
        self.assertFalse(evidence_passes(192, 224, 224))
        self.assertTrue(evidence_passes(193, 224, 224))
        self.assertFalse(evidence_passes(999, 510, 510))
        self.assertTrue(evidence_passes(1000, 510, 510))

    def test_masks_are_unioned_across_worlds_and_noise_never_is_evidence(self):
        rgb = np.zeros((255, 255, 3), dtype=np.uint8)
        mask = np.zeros((255, 255), dtype=bool)
        mask.flat[:250] = True
        first = {"rgb": rgb, "marked": rgb.copy(), "input_text": "same", "masks": {"old_id": mask}}
        second = copy.deepcopy(first)
        second["masks"] = {"new_id": mask.copy()}
        second["rgb"].reshape(-1, 3)[:250, 0] = 10
        comparison = compare_inputs(first, second, ["old_id", "new_id"])
        self.assertFalse(comparison["evidence_passed"])
        self.assertFalse(comparison["same_input_at_t_strict"])
        second["rgb"].reshape(-1, 3)[:250, 0] = 11
        self.assertTrue(compare_inputs(first, second, ["new_id"])["evidence_passed"])
        self.assertFalse(compare_inputs(first, second, ["irrelevant_id"])["evidence_passed"])

    def test_low_noise_prefix_is_pending_even_with_all_other_checks_true(self):
        validity = {key: True for key in ("restore_deterministic", "valid_physical", "unique_targets",
                                          "witness_replay", "evidence_available", "targets_separable")}
        status, reasons, pending = classify("look_flip", validity, {
            "same_input_at_t_strict": False, "text_exact": True, "prefix_max_abs_diff": 1})
        self.assertEqual(status, "pending_review")
        self.assertTrue(pending)
        self.assertIn("same_input_at_t", reasons)

    def test_edit_travel_is_unbounded_but_settling_error_is_bounded(self):
        before = world()["objects"]
        request = {"objectName": "mug", "position": {"x": 2.0, "y": 0.0, "z": 0.0}, "rotation": before[0]["rotation"]}
        after = copy.deepcopy(before)
        after[0]["position"]["x"] = 2.019
        result = measure_settled_poses(before, request, after, "mug")
        self.assertTrue(result["position_bounds_passed"])
        self.assertGreater(result["edited_travel_distance_m"], 2)
        after[0]["position"]["x"] = 2.021
        self.assertFalse(measure_settled_poses(before, request, after, "mug")["position_bounds_passed"])

    def test_all_other_objects_and_inventory_are_checked(self):
        before = world()["objects"]
        request = world()["poses"][0]
        after = copy.deepcopy(before)
        after[2]["position"]["x"] = 0.01
        self.assertTrue(measure_settled_poses(before, request, after, "mug")["position_bounds_passed"])
        after[2]["position"]["x"] = 0.0101
        self.assertFalse(measure_settled_poses(before, request, after, "mug")["position_bounds_passed"])
        self.assertFalse(measure_settled_poses(before, request, after[:2], "mug")["position_bounds_passed"])
        with self.assertRaises(ValueError):
            measure_settled_poses(before + [before[0]], request, after, "mug")

    def test_native_success_without_move_or_membership_is_not_legal_proof(self):
        rng = SimpleNamespace(choice=lambda values: values[0], randrange=lambda upper: 0)
        task = SimpleNamespace(target_type="Mug", anchor_type="NoSuchType")
        for moved, member in ((False, True), (True, False), (False, False)):
            sim = FakeSimulator(moved=moved, member=member)
            request, edit = sim.placement_proposal(world(), task, rng)
            self.assertFalse(edit["native_proposal_verified"])
        sim = FakeSimulator()
        request, edit = sim.placement_proposal(world(), task, rng)
        self.assertTrue(edit["native_proposal_verified"])
        self.assertEqual(len(request["poses"]), len(world()["poses"]))
        self.assertEqual(request["objects"][2]["position"], world()["objects"][2]["position"])
        native = next(kwargs for name, kwargs in sim.calls if name == "InitialRandomSpawn")
        self.assertEqual(native["objectIds"], ["mug_id"])
        self.assertFalse(native["allowFloor"])

    def test_settling_requires_pose_stability_even_if_native_flag_says_rest(self):
        sim = FakeSimulator(continuous_motion=True)
        realized, evidence = sim.settle_world(world(), "after")
        self.assertFalse(evidence["settled"])
        self.assertEqual(len(evidence["observations"]), SETTLING["maximum_observations"])
        sim = FakeSimulator()
        _, evidence = sim.settle_world(world(), "after")
        self.assertTrue(evidence["settled"])
        self.assertEqual(len(evidence["observations"]), SETTLING["consecutive_stable_observations"])
        requests = [kwargs for name, kwargs in sim.calls if name == "SetObjectPoses"]
        self.assertEqual(requests[0]["objectPoses"], world()["poses"])
        self.assertIs(requests[0]["placeStationary"], False)


if __name__ == "__main__":
    unittest.main()
