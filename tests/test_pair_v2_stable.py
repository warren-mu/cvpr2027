"""Synthetic CPU trajectories only; no Unity, GPU, or research evidence files."""
import copy
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.envs.pair_v2_stable import (StableCloneSimulator, PREFLIGHT_VERSION,
                                   PREFLIGHT_POLICY, position_step, common_baseline_check,
                                   augment_physical_result)
from src.envs.pair_v2_clone import V2CloneSimulator
from src.tools.merge_pair_v2 import shard_producer, verify_preflight


def objects(x=0.0, table=1.0):
    return [{"name": name, "objectId": name, "position": {"x": value, "y": 0.0, "z": 0.0},
             "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}, "pickupable": name == "A", "moveable": False}
            for name, value in (("A", x), ("StaticTable", table))]


class MemoryStore:
    def __init__(self): self.rows = {}
    def save_json(self, name, value):
        path = "/synthetic/raw/" + name
        if path in self.rows: raise FileExistsError(path)
        self.rows[path] = copy.deepcopy(value)
        return {"file": path, "sha256": hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()}
    def read(self, ref):
        value = copy.deepcopy(self.rows[ref["file"]])
        if hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest() != ref["sha256"]:
            raise ValueError("Synthetic reference was changed")
        return value


class SyntheticStable(StableCloneSimulator):
    def __init__(self, initial, window=None):
        self.config = {"preflight_version": PREFLIGHT_VERSION, "preflight_policy": copy.deepcopy(PREFLIGHT_POLICY)}
        self.max_steps, self.time_step, self.step_seconds = 50, .02, .1
        self.action_count = self.snapshot_count = 0
        self.last_action_reference = None
        self.store, self.calls = MemoryStore(), []
        self.scene_preflight = None
        self.clone_context = None
        self.fail_after_advances = None
        self.frames = list(initial) + list(window or [])
        self.advances = 0
        self.controller = SimpleNamespace(last_event=SimpleNamespace(metadata={"objects": objects(), "isSceneAtRest": False}))
    def snapshot(self, key):
        value = {"key": key, "scene": "FloorPlan1", "objects": copy.deepcopy(self.controller.last_event.metadata["objects"]),
                 "native_is_scene_at_rest": self.controller.last_event.metadata["isSceneAtRest"],
                 "last_action_reference": copy.deepcopy(self.last_action_reference)}
        ref = self.store.save_json(f"snapshot_{self.snapshot_count}.json", value)
        self.snapshot_count += 1
        value["reference"] = ref
        return value
    def action(self, name, **arguments):
        self.calls.append(name)
        before = copy.deepcopy(self.controller.last_event.metadata)
        index = self.action_count
        self.action_count += 1
        if name == "AdvancePhysicsStep":
            if self.fail_after_advances == self.advances: raise RuntimeError("synthetic interruption")
            values, rest = self.frames[self.advances]
            self.advances += 1
            self.controller.last_event.metadata.update(objects=copy.deepcopy(values), isSceneAtRest=rest)
        self.controller.last_event.metadata["lastActionSuccess"] = True
        raw = {"index": index, "action": name, "arguments": arguments, "exception": None,
               "before_metadata": before, "after_metadata": copy.deepcopy(self.controller.last_event.metadata)}
        self.last_action_reference = self.store.save_json(f"action_{index}.json", raw)
        return self.controller.last_event


class StablePreflightContracts(unittest.TestCase):
    def test_both_worlds_share_one_collateral_baseline_not_successive_one_cm_allowances(self):
        ref = {"file": "synthetic", "sha256": "fixture"}
        baseline, w0, w1 = objects(table=0), objects(table=.009), objects(table=.018)
        self.assertTrue(common_baseline_check(baseline, w0, ["A"], ref)["passed"])
        self.assertTrue(common_baseline_check(w0, w1, ["A"], ref)["passed"])
        check = common_baseline_check(baseline, w1, ["A"], ref)
        self.assertFalse(check["passed"])
        self.assertEqual(check["violations_above_1cm"], ["StaticTable"])
        self.assertTrue(common_baseline_check(baseline, objects(table=.01), ["A"], ref)["passed"])

    def test_common_gate_failure_propagates_to_world_and_policy_and_preserves_evidence(self):
        sim = SyntheticStable([(objects(), True)], [(objects(), True)] * 50)
        baseline = sim.preflight_scene()["snapshot"]
        native = {"passed": True, "snapshot": {"objects": objects(table=1.018)}, "capture": {},
                  "physical": {"edited_names": ["A"], "valid_physical": True}}
        with patch.object(V2CloneSimulator, "construct_world", return_value=native):
            result = sim.construct_world(baseline, "synthetic_w1", [{"object_name": "A"}])
        self.assertFalse(result["passed"])
        self.assertFalse(result["physical"]["valid_physical"])
        measured = augment_physical_result({"valid_physical": True, "reasons": []}, result["physical"])
        self.assertFalse(measured["valid_physical"])
        self.assertIn("common_baseline_other_object_position_change_above_0_01m:StaticTable", measured["reasons"])
        self.assertTrue(any("stable_construction_" in path for path in sim.store.rows))

    def test_initial_settling_can_exceed_old_1cm_then_full_window_uses_settled_baseline(self):
        sim = SyntheticStable([(objects(.03), True)], [(objects(.031), True)] * 50)
        measured = sim.preflight_scene()
        self.assertTrue(measured["passed"])
        self.assertEqual(measured["snapshot"]["objects"][0]["position"]["x"], .03)
        self.assertEqual(measured["window_end_snapshot"]["objects"][0]["position"]["x"], .031)
        self.assertEqual(sim.advances, 51)
        self.assertTrue(verify_preflight(measured, sim.store.read)["passed"])
        self.assertEqual(set(sim.calls), {"PausePhysicsAutoSim", "AdvancePhysicsStep"})

    def test_ten_strict_small_steps_can_settle_without_native_rest(self):
        initial = [(objects(.0009 if index % 2 == 0 else 0), False) for index in range(10)]
        sim = SyntheticStable(initial, [(objects(), False)] * 50)
        result = sim.preflight_scene()
        self.assertTrue(result["passed"])
        self.assertEqual(result["diagnostics"]["initial_settling"]["criterion"], "ten_small_steps")
        self.assertEqual(verify_preflight(result, sim.store.read)["initial_steps"], 10)

    def test_exact_1mm_is_not_a_small_step_and_N_cannot_be_exceeded(self):
        sim = SyntheticStable([(objects(.001 if i % 2 == 0 else 0), False) for i in range(50)])
        result = sim.preflight_scene()
        self.assertFalse(result["passed"])
        self.assertEqual(sim.advances, 50)
        self.assertIsNone(result["snapshot"])
        self.assertFalse(result["diagnostics"]["stability_window"]["completed"])

    def test_small_step_streak_resets_after_a_large_step(self):
        initial = [(objects(), False)] * 9 + [(objects(.001), False)] * 11
        sim = SyntheticStable(initial, [(objects(.001), False)] * 50)
        result = sim.preflight_scene()
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["diagnostics"]["initial_settling"]["observations"]), 20)

    def test_initial_half_meter_is_inclusive_and_applies_to_static_metadata_objects(self):
        sim = SyntheticStable([(objects(.5), True)], [(objects(.5), True)] * 50)
        self.assertTrue(sim.preflight_scene()["passed"])
        sim = SyntheticStable([(objects(.500001), True)])
        self.assertFalse(sim.preflight_scene()["passed"])
        sim = SyntheticStable([(objects(0, table=1.500001), True)])
        self.assertFalse(sim.preflight_scene()["passed"])

    def test_net_displacement_not_accumulated_path_length_is_the_initial_gate(self):
        initial = [(objects(.4 if i % 2 == 0 else 0), i == 9) for i in range(10)]
        sim = SyntheticStable(initial, [(objects(), False)] * 50)
        result = sim.preflight_scene()
        self.assertTrue(result["passed"])
        last = result["diagnostics"]["initial_settling"]["observations"][-1]
        self.assertGreater(last["measurements"]["objects"]["A"]["observed_path_length_m"], .5)
        self.assertTrue(verify_preflight(result, sim.store.read)["passed"])

    def test_window_checks_every_step_not_just_final_and_does_not_stop_on_native_rest(self):
        sim = SyntheticStable([(objects(), True)], [(objects(.010001), True)] + [(objects(), True)] * 49)
        result = sim.preflight_scene()
        self.assertFalse(result["passed"])
        self.assertEqual(sim.advances, 51)
        self.assertEqual(len(result["diagnostics"]["stability_window"]["observations"]), 50)
        sim = SyntheticStable([(objects(), True)], [(objects(.01), True)] * 50)
        self.assertTrue(sim.preflight_scene()["passed"])

    def test_missing_inventory_or_nonfinite_position_fails_closed(self):
        self.assertFalse(position_step(objects(), objects()[:1], objects())["inventory_exact"])
        with self.assertRaises(ValueError): position_step(objects(), objects(float("nan")), objects())

    def test_interruption_preserves_actual_partial_window_without_claiming_completion(self):
        sim = SyntheticStable([(objects(), True)], [(objects(), True)] * 50)
        sim.fail_after_advances = 4
        with self.assertRaises(RuntimeError): sim.preflight_scene()
        saved = sim.store.rows["/synthetic/raw/scene_preflight.json"]
        self.assertFalse(saved["passed"])
        self.assertFalse(saved["diagnostics"]["stability_window"]["completed"])
        self.assertEqual(len(saved["diagnostics"]["stability_window"]["observations"]), 3)

    def test_merge_rejects_short_window_changed_raw_reference_and_baseline_substitution(self):
        sim = SyntheticStable([(objects(.03), True)], [(objects(.031), True)] * 50)
        original = sim.preflight_scene()
        short = copy.deepcopy(original)
        short["diagnostics"]["stability_window"]["observations"].pop()
        with self.assertRaises(ValueError): verify_preflight(short, sim.store.read)
        swapped = copy.deepcopy(original)
        swapped["snapshot"] = swapped["window_end_snapshot"]
        with self.assertRaises(ValueError): verify_preflight(swapped, sim.store.read)
        changed = copy.deepcopy(original)
        changed["diagnostics"]["initial_settling"]["observations"][0]["action_reference"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError): verify_preflight(changed, sim.store.read)

    def test_stable_source_requires_exact_version_policy_and_new_module(self):
        config = {"enable_candidate_clone": True, "preflight_version": PREFLIGHT_VERSION,
                  "preflight_policy": copy.deepcopy(PREFLIGHT_POLICY), "evidence_version": "anchor_union_look_v1"}
        command = ["python", "-u", "-m", "src.envs.pair_v2_stable_shard"]
        self.assertTrue(shard_producer(command, config)[1])
        for replacement in (None, "future-version"):
            with self.assertRaises(ValueError): shard_producer(command, {**config, "preflight_version": replacement})
        with self.assertRaises(ValueError): shard_producer(command, {**config, "preflight_policy": {}})


if __name__ == "__main__":
    unittest.main()
