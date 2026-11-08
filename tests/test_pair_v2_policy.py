"""Synthetic T01v2 CPU boundaries. Execute only after commit under run_logged."""
import copy
import unittest

import numpy as np

from src.envs.pair_v2_policy import (area_passes, classify_pair, closest_target, compare_view,
    mark_observation, physical_check, prefix_check, public_copy, restore_state_check, single_edit_check, witness_check)


def obj(name, category, x, *, pickup=True, receptacle=False):
    return {"name": name, "objectId": name + "_id", "objectType": category,
            "pickupable": pickup, "moveable": False, "receptacle": receptacle,
            "position": {"x": x, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}, "parentReceptacles": ["table_id"]}


def objects():
    return [obj("A", "Mug", 0.0), obj("B", "Mug", 2.0), obj("X", "Apple", 0.5),
            obj("table", "Table", 0.0, pickup=False, receptacle=True)]


def poses(rows):
    return [{"objectName": row["name"], "position": copy.deepcopy(row["position"]), "rotation": copy.deepcopy(row["rotation"])}
            for row in rows if row.get("pickupable") or row.get("moveable")]


def view(*, noise=0, anchor_pixels=0):
    rgb = np.full((255, 255, 3), 60 + noise, dtype=np.uint8)
    masks = {name: np.zeros((255, 255), dtype=bool) for name in ("A", "B", "X")}
    masks["A"][10:20, 10:35] = True
    masks["B"][10:20, 100:125] = True
    masks["X"].reshape(-1)[30000:30000 + anchor_pixels] = True
    return {"rgb": rgb, "marked": rgb.copy(), "masks_by_name": masks,
            "input_text": "Select the Mug closest to the Apple.", "history": [],
            "marks": [{"label": 1, "bounding_box": [10, 10, 34, 19], "anchor_pixel": [22, 14]},
                      {"label": 2, "bounding_box": [100, 10, 124, 19], "anchor_pixel": [112, 14]}],
            "mark_labels_by_name": {"A": 1, "B": 2}}


def witness(name, actions=None):
    actions = actions or ["turn_right", "select " + ("1" if name == "A" else "2")]
    return {"planned_actions": actions.copy(), "executed_actions": actions.copy(),
            "action_successes": [True] * len(actions), "process_id": 20, "construction_process_id": 10,
            "snapshot_restored": True, "selected_name": name, "selected_mask_pixels": 1,
            "selected_distance_m": 1.5, "success": True, "done": True}


class PairV2PolicyTests(unittest.TestCase):
    def test_closest_boundary_and_zero_distance_tie(self):
        rows = objects()
        rows[2]["position"]["x"] = 0.8  # distances .8 and 1.2, exact float boundary represented below
        rows[0]["position"]["x"] = 0.0
        rows[1]["position"]["x"] = 2.5
        rows[2]["position"]["x"] = 1.0
        self.assertEqual(closest_target(rows, ["A", "B"], "X")["target_name"], "A")
        rows[1]["position"]["x"] = 2.49
        self.assertFalse(closest_target(rows, ["A", "B"], "X")["passed"])
        for row in rows[:3]:
            row["position"]["x"] = 0.0
        self.assertFalse(closest_target(rows, ["A", "B"], "X")["passed"])

    def test_extra_candidates_anchor_ambiguity_and_stable_identity(self):
        rows = objects()
        original = closest_target(rows, ["A", "B"], "X")
        for row in rows:
            row["objectId"] += "_changed"
        self.assertEqual(closest_target(rows, ["A", "B"], "X"), original)
        self.assertFalse(closest_target(rows + [obj("C", "Mug", 4)], ["A", "B"], "X")["passed"])
        self.assertFalse(closest_target(rows + [obj("Y", "Apple", 4)], ["A", "B"], "X")["passed"])

    def test_area_threshold_scales_without_rounding_tolerance(self):
        for count, resolution, expected in ((249, 255, False), (250, 255, True),
                                             (192, 224, False), (193, 224, True), (999, 510, False), (1000, 510, True)):
            self.assertIs(area_passes(count, resolution, resolution), expected)
        with self.assertRaises(ValueError):
            area_passes(True, 255, 255)

    def test_look_keeps_candidates_visible_hides_only_anchor_and_requires_actual_sharing(self):
        first, second = view(), view(noise=3)
        kwargs = dict(candidate_names=["A", "B"], anchor_name="X", same_process=True, agent_unchanged=True)
        unshared = prefix_check([first], [second], **kwargs)
        self.assertTrue(unshared["render_qualified_for_shared_prefix"])
        self.assertFalse(unshared["same_input_at_t"])
        shared = prefix_check([first], [second], shared_prefix_w0=[public_copy(first)], shared_prefix_w1=[public_copy(first)], **kwargs)
        self.assertTrue(shared["same_input_at_t"])
        self.assertFalse(shared["native_public_prefix_exact"])
        self.assertTrue(shared["candidates_visible_and_marked"])
        second["masks_by_name"]["X"][0, 0] = True
        self.assertFalse(prefix_check([first], [second], **kwargs)["render_qualified_for_shared_prefix"])

    def test_prefix_rejects_context_labels_process_and_noncanonical_shared_input(self):
        first, second = view(), view()
        kwargs = dict(candidate_names=["A", "B"], anchor_name="X", same_process=True, agent_unchanged=True)
        for key, value in (("input_text", "different"), ("history", ["turn_left"]), ("mark_labels_by_name", {"A": 2, "B": 1})):
            altered = copy.deepcopy(second)
            altered[key] = value
            self.assertFalse(prefix_check([first], [altered], **kwargs)["render_qualified_for_shared_prefix"])
        kwargs["same_process"] = False
        self.assertFalse(prefix_check([first], [second], **kwargs)["render_qualified_for_shared_prefix"])
        kwargs["same_process"] = True
        fabricated = public_copy(view(noise=2))
        self.assertFalse(prefix_check([first], [second], shared_prefix_w0=[fabricated], shared_prefix_w1=[fabricated], **kwargs)["same_input_at_t"])
        self.assertFalse(prefix_check([first], [view(noise=4)], **kwargs)["render_qualified_for_shared_prefix"])

    def test_evidence_strict_channel_union_and_mask_scope(self):
        first, second = view(anchor_pixels=125), view()
        second["masks_by_name"]["X"].reshape(-1)[30125:30250] = True
        second["rgb"].reshape(-1, 3)[30000:30250, 0] += 10
        second["marked"] = second["rgb"].copy()
        self.assertFalse(compare_view(first, second, ["X"])["evidence_passed"])
        second["rgb"].reshape(-1, 3)[30000:30250, 0] += 1
        second["marked"] = second["rgb"].copy()
        result = compare_view(first, second, ["X"])
        self.assertEqual(result["evidence_pixels"], 250)
        self.assertTrue(result["evidence_passed"])
        self.assertFalse(compare_view(first, second, ["A", "B"])["evidence_passed"])
        second["marked"] = first["marked"].copy()
        self.assertFalse(compare_view(first, second, ["X"])["marked_evidence_passed"])

    def test_candidate_marks_are_spatial_public_and_not_distance_filtered(self):
        original = view()
        a = mark_observation(original, ["A", "B"], "Select a Mug.")
        b = mark_observation(original, ["B", "A"], "Select a Mug.")
        self.assertEqual(a["marks"], b["marks"])
        self.assertEqual(a["mark_labels_by_name"], {"A": 1, "B": 2})
        self.assertNotIn("name", a["marks"][0])
        self.assertNotIn("masks_by_name", public_copy(a))
        original["masks_by_name"]["B"][:] = False
        self.assertEqual(mark_observation(original, ["A", "B"], "Select a Mug.")["mark_labels_by_name"], {"A": 1})

    def physical_fixture(self):
        before = objects()
        requested = poses(before)
        requested[2]["position"]["x"] = 0.0
        after = copy.deepcopy(before)
        after[2]["position"]["x"] = 0.02
        point = copy.deepcopy(requested[2]["position"])
        proof = {"X": {"action": "GetSpawnCoordinatesAboveReceptacle", "success": True,
                        "selected_position": point, "coordinates": [point], "receptacle_name": "table"}}
        return before, requested, after, proof

    def test_physical_position_bounds_allow_rotation_but_require_real_parent_and_sample(self):
        before, request, after, proof = self.physical_fixture()
        after[0]["position"]["x"] = 0.01
        after[2]["rotation"]["x"] = 30
        result = physical_check(before, request, after, ["X"], native_is_scene_at_rest=True, placement_evidence_by_name=proof)
        self.assertTrue(result["valid_physical"])
        after[2]["position"]["x"] = 0.02001
        self.assertFalse(physical_check(before, request, after, ["X"], native_is_scene_at_rest=True, placement_evidence_by_name=proof)["valid_physical"])
        after[2]["position"]["x"] = 0.02
        after[0]["position"]["x"] = 0.01001
        self.assertFalse(physical_check(before, request, after, ["X"], native_is_scene_at_rest=True, placement_evidence_by_name=proof)["valid_physical"])

    def test_physical_unknown_inventory_sampling_and_rest_fail_closed(self):
        before, request, after, proof = self.physical_fixture()
        for state, requests, realized, evidence in ((None, request, after, proof), (True, request[:-1], after, proof),
                (True, request, after[:-1], proof), (True, request, after, {})):
            self.assertFalse(physical_check(before, requests, realized, ["X"], native_is_scene_at_rest=state,
                                            placement_evidence_by_name=evidence)["valid_physical"])
        after[2]["parentReceptacles"] = []
        self.assertFalse(physical_check(before, request, after, ["X"], native_is_scene_at_rest=True, placement_evidence_by_name=proof)["valid_physical"])

    def test_single_edit_is_anchor_position_only(self):
        first = poses(objects())
        second = copy.deepcopy(first)
        second[2]["position"]["x"] = 1.5
        self.assertTrue(single_edit_check(first, second, "X")["passed"])
        second[0]["rotation"]["x"] = 1
        self.assertFalse(single_edit_check(first, second, "X")["passed"])
        self.assertFalse(single_edit_check(first, first, "X")["passed"])

    def test_native_construction_pivot_is_distinct_from_sample_and_not_postsettle_refilled(self):
        before, request, after, proof = self.physical_fixture()
        native_point = {"x": 0.0, "y": -0.2, "z": 0.0}
        proof["X"].update(selected_position=native_point, coordinates=[native_point], requested_position=request[2]["position"],
            geometry={"method": "native_PlaceObjectAtPoint_forceKinematic", "object_name": "X",
                      "placement_action": "PlaceObjectAtPoint", "placement_success": True,
                      "source_rotation": request[2]["rotation"], "placement_reference": "synthetic_construct_action_1",
                      "captured_before_complete_pose_set_and_settling": True,
                      "placement_arguments": {"objectId": "X_at_construct", "position": native_point,
                                              "rotation": request[2]["rotation"], "forceKinematic": True},
                      "returned_position": request[2]["position"], "returned_rotation": request[2]["rotation"]})
        self.assertTrue(physical_check(before, request, after, ["X"], native_is_scene_at_rest=True, placement_evidence_by_name=proof)["valid_physical"])
        proof["X"]["geometry"]["returned_position"] = after[2]["position"]
        self.assertFalse(physical_check(before, request, after, ["X"], native_is_scene_at_rest=True, placement_evidence_by_name=proof)["valid_physical"])

    def test_state_restore_position_boundary_exact_agent_and_rotation_diagnostic(self):
        before, after = objects(), objects()
        agent = {"position": {"x": 0, "y": 1, "z": 0}, "rotation": {"x": 0, "y": 0, "z": 0},
                 "cameraHorizon": 30.0, "isStanding": True}
        after[0]["position"]["x"] = 0.01
        after[0]["rotation"]["z"] = 0.00003
        self.assertTrue(restore_state_check(before, after, agent, agent)["restore_deterministic"])
        after[0]["position"]["x"] = 0.01001
        self.assertFalse(restore_state_check(before, after, agent, agent)["restore_deterministic"])
        after[0]["position"]["x"] = 0.01
        changed_agent = copy.deepcopy(agent)
        changed_agent["cameraHorizon"] = 30.0000038
        self.assertFalse(restore_state_check(before, after, agent, changed_agent)["restore_deterministic"])
        self.assertFalse(restore_state_check(before, after[:-2], agent, agent)["restore_deterministic"])
        public_shape = {"position": agent["position"], "rotation": agent["rotation"], "horizon": 30.0, "standing": True}
        self.assertTrue(restore_state_check(before, after, public_shape, public_shape)["restore_deterministic"])
        self.assertFalse(restore_state_check(before, after, {}, {})["restore_deterministic"])

    def test_witness_requires_real_new_process_select_and_budget_includes_select(self):
        record = witness("A", ["turn_left"] * 11 + ["select 1"])
        self.assertTrue(witness_check(record, "A")["passed"])
        for key, value in (("process_id", 10), ("snapshot_restored", False), ("selected_name", "B"),
                           ("selected_mask_pixels", 0), ("selected_distance_m", 1.50001), ("action_successes", [1] * 12)):
            changed = copy.deepcopy(record)
            changed[key] = value
            self.assertFalse(witness_check(changed, "A")["passed"])
        self.assertFalse(witness_check(witness("A", ["turn_left"] * 12 + ["select 1"]), "A")["passed"])
        self.assertFalse(witness_check({"planned_actions": ["select 1"]}, "A")["passed"])

    def classification(self, kind):
        left, right = view(), view(noise=2)
        look = kind.startswith("look")
        if not look:
            left, right = view(anchor_pixels=250), view(anchor_pixels=250)
            right["rgb"].reshape(-1, 3)[30000:30250, 0] += 11
            right["marked"] = right["rgb"].copy()
        prefix = prefix_check([left], [right], ["A", "B"], "X", same_process=True, agent_unchanged=True,
                              shared_prefix_w0=[public_copy(left)] if look else None,
                              shared_prefix_w1=[public_copy(left)] if look else None)
        after0, after1 = view(anchor_pixels=250), view(anchor_pixels=250)
        after1["rgb"].reshape(-1, 3)[30000:30250, 0] += 11
        after1["marked"] = after1["rgb"].copy()
        w0, w1 = objects(), objects()
        if kind.endswith("flip"):
            w1[2]["position"]["x"] = 1.5
        t0, t1 = closest_target(w0, ["A", "B"], "X"), closest_target(w1, ["A", "B"], "X")
        return dict(target_w0=t0, target_w1=t1, prefix=prefix,
                    evidence=compare_view(after0, after1, ["A", "B", "X"]) if look else compare_view(left, right, ["A", "B", "X"]),
                    physical_w0={"valid_physical": True}, physical_w1={"valid_physical": True}, single_edit={"passed": True},
                    witness_w0=witness_check(witness(t0["target_name"]), t0["target_name"]),
                    witness_w1=witness_check(witness(t1["target_name"]), t1["target_name"]),
                    restore_deterministic=True, restore_max_abs_diff={"w0": 3, "w1": 1}, anchor_pixels_w0=250, anchor_pixels_w1=250,
                    common_observation_action="turn_right" if look else None)

    def test_all_four_classes_state_restore_required_pixel_difference_diagnostic(self):
        for kind in ("look_flip", "look_stay", "direct_flip", "direct_stay"):
            result = classify_pair(kind, **self.classification(kind))
            self.assertTrue(result["acceptance_passed"], result)
            self.assertIs(result["restore_deterministic"], True)
            self.assertEqual(result["restore_max_abs_diff"], {"w0": 3, "w1": 1})
            self.assertEqual(result["same_input_at_t"], kind.startswith("look"))
            missing = self.classification(kind)
            missing["restore_deterministic"] = None
            self.assertFalse(classify_pair(kind, **missing)["acceptance_passed"])

    def test_classification_rejects_wrong_label_or_missing_common_replay_observation(self):
        kwargs = self.classification("look_flip")
        self.assertFalse(classify_pair("look_stay", **kwargs)["acceptance_passed"])
        kwargs["common_observation_action"] = "turn_left"
        self.assertFalse(classify_pair("look_flip", **kwargs)["acceptance_passed"])
        kwargs = self.classification("look_flip")
        kwargs["witness_w0"] = witness_check(witness("B"), "B")
        self.assertFalse(classify_pair("look_flip", **kwargs)["acceptance_passed"])


if __name__ == "__main__":
    unittest.main()
