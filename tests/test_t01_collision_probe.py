"""Synthetic bookkeeping fixtures; no claim about actual Unity API execution.

Root executes these only after commit, through run_logged.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from src.tools import t01_collision_probe as probe


def metadata():
    return {"objects": [
        {"name": "Book_A", "objectId": "Book|0|0|0", "position": {"x": 0.0, "y": 1.0, "z": 0.0},
         "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}, "pickupable": True, "moveable": False,
         "isMoving": False, "unknown_semantic_state": {"original": True}}],
        "agent": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "cameraHorizon": 0.0},
        "inventoryObjects": [], "isSceneAtRest": True, "currentTime": 1.0,
        "sceneName": "FloorPlan203", "lastAction": "Pass", "lastActionSuccess": True,
        "errorMessage": "", "actionReturn": None, "sequenceId": 1}


class FakeController:
    def __init__(self, modifier=None):
        self.last_event = types.SimpleNamespace(metadata=metadata())
        self.calls, self.modifier = [], modifier

    def step(self, action, **kwargs):
        self.calls.append({"action": action, **copy.deepcopy(kwargs)})
        new = copy.deepcopy(self.last_event.metadata)
        new.update(lastAction=action, lastActionSuccess=True, errorMessage="",
                   actionReturn=["Col"] if action == "CollidersObjectCollidingWith" else None,
                   currentTime=new["currentTime"] + 1.0, sequenceId=new["sequenceId"] + 1)
        if self.modifier:
            self.modifier(action, new)
        self.last_event = types.SimpleNamespace(metadata=new)
        return self.last_event


class StateComparisonTests(unittest.TestCase):
    def test_only_explicit_root_bookkeeping_is_excluded(self):
        before, after = metadata(), metadata()
        after.update(lastAction="TeleportObject", lastActionSuccess=False, errorMessage="collision",
                     actionReturn=["Col"], sequenceId=99)
        result = probe.compare_state(before, after)
        self.assertTrue(result["full_state_exact"])
        self.assertTrue(result["safe_to_continue_diagnostic"])
        # An object field with a bookkeeping-like name is still physical data.
        after["objects"][0]["actionReturn"] = "unclassified object property"
        result = probe.compare_state(before, after)
        self.assertFalse(result["full_state_exact"])
        self.assertFalse(result["safe_to_continue_diagnostic"])

    def test_current_time_remains_a_reported_difference_not_fake_equality(self):
        before, after = metadata(), metadata()
        after["currentTime"] = 2.0
        result = probe.compare_state(before, after)
        self.assertFalse(result["full_state_exact"])
        self.assertTrue(result["metadata_side_effect_observed"])
        self.assertEqual(result["changed_paths"], ["/currentTime"])
        self.assertTrue(result["clock_only_difference"])
        self.assertTrue(result["safe_to_continue_diagnostic"])
        del after["currentTime"]
        self.assertFalse(probe.compare_state(before, after)["safe_to_continue_diagnostic"])

    def test_unknown_added_missing_and_changed_fields_fail_closed(self):
        for mutate in (
            lambda x: x.update(new_unclassified_field=False),
            lambda x: x.pop("sceneName"),
            lambda x: x["objects"][0]["unknown_semantic_state"].update(original=False),
            lambda x: x["agent"].update(cameraHorizon=1.0),
            lambda x: x.update(isSceneAtRest=False),
            lambda x: x.update(inventoryObjects=[{"objectId": "Book|0|0|0"}]),
        ):
            before, after = metadata(), metadata()
            mutate(after)
            self.assertFalse(probe.compare_state(before, after)["safe_to_continue_diagnostic"])

    def test_missing_required_or_duplicate_object_identity_is_unknown(self):
        for mutate in (
            lambda x: x.pop("inventoryObjects"),
            lambda x: x["objects"].append(copy.deepcopy(x["objects"][0])),
            lambda x: x["objects"][0].pop("objectId"),
        ):
            before, after = metadata(), metadata()
            mutate(after)
            result = probe.compare_state(before, after)
            self.assertIsNone(result["full_state_exact"])
            self.assertFalse(result["comparison_available"])
            self.assertFalse(result["safe_to_continue_diagnostic"])

    def test_only_object_order_is_normalized_and_exact_floats_are_not_rounded(self):
        before = metadata()
        other = copy.deepcopy(before["objects"][0])
        other.update(name="Book_B", objectId="Book|1|0|0")
        before["objects"].append(other)
        after = copy.deepcopy(before)
        after["objects"].reverse()
        self.assertTrue(probe.compare_state(before, after)["full_state_exact"])
        after["objects"][0]["position"]["x"] = 0.00000000001
        self.assertFalse(probe.compare_state(before, after)["full_state_exact"])


class ActionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = probe.EvidenceStore(Path(temp.name) / "raw")

    def recorder(self, controller, capture=None):
        capture = capture or (lambda event, stage: {"valid": True, "black": False, "rgb_sha256": "synthetic-unchanged-rgb"})
        return probe.ActionRecorder(controller, self.store, capture, time.time() + 60)

    def test_positions_array_false_flags_and_raw_both_actions(self):
        controller = FakeController()
        case = probe.probe_object(self.recorder(controller), metadata()["objects"][0])
        request = controller.calls[1]
        self.assertEqual(request["action"], "TeleportObject")
        self.assertNotIn("position", request)
        self.assertEqual(request["positions"], [metadata()["objects"][0]["position"]])
        for field in ("forceAction", "forceKinematic", "allowTeleportOutOfHand", "makeUnbreakable"):
            self.assertIs(request[field], False)
        self.assertTrue(case["complete"])
        self.assertEqual(case["old_query"]["actionReturn"], ["Col"])
        self.assertIsNone(case["physical_validity"])
        self.assertNotIn("valid_physical", case)
        for key in ("old_query", "identity_teleport"):
            raw = json.loads(Path(case[key]["action_record"]["file"]).read_text())
            self.assertIn("objects", raw["before_metadata"])
            self.assertIn("objects", raw["after_metadata"])
            self.assertTrue(raw["state_comparison"]["clock_only_difference"])

    def test_known_collision_false_is_recorded_but_not_physical_approval(self):
        def reject(action, state):
            if action == "TeleportObject":
                state.update(lastActionSuccess=False, errorMessage="Book|0|0|0 is colliding with Col after teleport.")
        case = probe.probe_object(self.recorder(FakeController(reject)), metadata()["objects"][0])
        self.assertTrue(case["complete"])
        self.assertFalse(case["identity_teleport"]["lastActionSuccess"])
        self.assertTrue(case["identity_teleport"]["native_collision_rejected"])
        self.assertIsNone(case["physical_validity"])

    def test_unknown_api_failure_is_preserved_as_incomplete(self):
        def incompatible(action, state):
            if action == "TeleportObject":
                state.update(lastActionSuccess=False, errorMessage="Invalid arguments: positions")
        case = probe.probe_object(self.recorder(FakeController(incompatible)), metadata()["objects"][0])
        self.assertFalse(case["complete"])
        self.assertFalse(case["identity_teleport"]["api_response_known"])
        self.assertIn("Invalid arguments", case["identity_teleport"]["errorMessage"])

    def test_semantic_side_effect_stops_later_action(self):
        def mutate(action, state):
            if action == "CollidersObjectCollidingWith":
                state["objects"][0]["position"]["x"] = 0.01
        controller = FakeController(mutate)
        case = probe.probe_object(self.recorder(controller), metadata()["objects"][0])
        self.assertFalse(case["complete"])
        self.assertIsNone(case["identity_teleport"])
        self.assertEqual(len(controller.calls), 1)

    def test_black_frame_stops_and_action_exception_keeps_raw_evidence(self):
        controller = FakeController()
        recorder = self.recorder(controller, lambda event, stage: {"valid": False, "black": True})
        with self.assertRaises(probe.BlackFrame):
            recorder.execute("Pass")
        self.assertEqual(controller.calls, [])
        raw = json.loads(Path(recorder.last_record["reference"]["file"]).read_text())
        self.assertIn("BlackFrame", raw["exception"])
        self.assertIn("before_metadata", raw)
        self.assertIsNone(raw["after_metadata"])

    def test_nonempty_hands_refused_before_any_native_action(self):
        controller = FakeController()
        controller.last_event.metadata["inventoryObjects"] = [{"objectId": "held"}]
        with self.assertRaisesRegex(RuntimeError, "empty hands"):
            probe.probe_object(self.recorder(controller), metadata()["objects"][0])
        self.assertEqual(controller.calls, [])

    def test_partial_hash_index_and_size_limit_preserve_earlier_bytes(self):
        reference = self.store.save_json("first.json", {"synthetic": True})
        before = Path(reference["file"]).read_bytes()
        self.assertEqual(hashlib.sha256(before).hexdigest(), reference["sha256"])
        indexed = [json.loads(line) for line in self.store.incremental.read_text().splitlines()]
        self.assertEqual(indexed, [reference])
        self.store.max_bytes = self.store.bytes
        with self.assertRaises(ValueError):
            self.store.save("too_large.bin", b"x")
        self.assertEqual(Path(reference["file"]).read_bytes(), before)
        with self.assertRaises(FileExistsError):
            self.store.save("first.json", b"")


class GateTests(unittest.TestCase):
    def test_fixed_mapping_requires_same_successful_four_gpu_gate_and_mask(self):
        devices = [{"logical_index": index, "gpu_uuid": f"GPU-fixture-{index}"} for index in range(4)]
        environment = {"PBS_JOBID": "fixture.hopper-m-02", "CUDA_VISIBLE_DEVICES": "fixture-mask", "CVPR_SCRATCH": "/tmp/fixture"}
        runtime = {"devices": devices, "pbs_jobid": environment["PBS_JOBID"],
                   "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"], "scratch": str(Path(environment["CVPR_SCRATCH"]).resolve())}
        data = {"_meta": {"exit_code": 0}, "metrics": {"passed": True, "errors": [], "cleanup_errors": [],
                "expected_gpu_count": 4, "completed_gpu_count": 4},
                "config": {"resolution": 224, "segmentation": True, "pbs_jobid": environment["PBS_JOBID"], "allocation": runtime}}
        self.assertEqual(probe.validate_gate(data, environment), runtime)
        for mutate in (lambda d: d["metrics"].update(passed=False),
                       lambda d: d["config"].update(segmentation=False),
                       lambda d: d["config"]["allocation"].update(cuda_visible_devices="different"),
                       lambda d: d["config"]["allocation"]["devices"].reverse()):
            changed = copy.deepcopy(data)
            mutate(changed)
            with self.assertRaises(ValueError):
                probe.validate_gate(changed, environment)
        self.assertEqual(probe.SCENES, ("FloorPlan203", "FloorPlan204", "FloorPlan205", "FloorPlan206"))


if __name__ == "__main__":
    unittest.main()
