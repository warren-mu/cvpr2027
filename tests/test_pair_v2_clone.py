"""CPU-only synthetic control flow; no Unity, GPU, or research output."""
import copy
import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.envs import pair_v2_clone as backend
from src.envs.pair_v2_clone import (V2CloneSimulator, prepare_clone_baseline,
                                  restore_clone_inventory, clone_replay_worker)
from src.envs.pair_v2_clone_policy import make_clone_recipe, verify_clone_creation
from src.envs.pair_v2_simulator import encoded


def point(x=0.0, y=0.0, z=0.0):
    return {"x": x, "y": y, "z": z}


def objects():
    return [{"name": name, "objectType": kind, "objectId": name + "_native",
             "position": point(x, 0.5), "rotation": point(), "pickupable": pickup,
             "moveable": movable, "receptacle": receptacle, "parentReceptacles": ["Table_native"]}
            for name, kind, x, pickup, movable, receptacle in (
                ("A", "Mug", 1.0, True, False, False), ("X", "Apple", 2.0, True, False, False),
                ("Chair", "Chair", 3.0, False, True, False), ("Table", "Table", 4.0, False, False, True))]


def poses(rows):
    return [{"objectName": row["name"], "position": copy.deepcopy(row["position"]),
             "rotation": copy.deepcopy(row["rotation"])} for row in rows if row["moveable"] or row["pickupable"]]


def spawn():
    return {"action": "GetSpawnCoordinatesAboveReceptacle", "success": True, "scene": "FloorPlan1",
            "receptacle_name": "Table", "selected_position": point(0, 0.7),
            "coordinates": [point(0, 0.7)], "query_reference": {"file": "synthetic-query", "sha256": "fixture"}}


class MemoryStore:
    def __init__(self):
        self.rows = []

    def save_json(self, name, value):
        self.rows.append((name, copy.deepcopy(value)))
        return {"file": name, "sha256": hashlib.sha256(encoded(value)).hexdigest()}


class SyntheticCloneSimulator(V2CloneSimulator):
    def __init__(self):
        self.scene = "FloorPlan1"
        self.clone_context = None
        self.clone_preparation_attempted = False
        self.clone_recreation_reference = None
        self.scene_preflight = {"passed": True}
        self.spawn_tokens = {hashlib.sha256(encoded(spawn())).hexdigest(): spawn()}
        self.store = MemoryStore()
        self.action_count = self.snapshot_count = 0
        self.last_action_reference = None
        self.last_capture = {"synthetic": True}
        self.max_steps, self.time_step, self.step_seconds = 2, 0.02, 0.1
        self.calls = []
        self.omit_created_clone = False
        self.controller = SimpleNamespace(last_event=SimpleNamespace(metadata={
            "objects": objects(), "inventoryObjects": [], "isSceneAtRest": True,
            "agent": {"position": point(), "rotation": point(), "cameraHorizon": 0.0, "isStanding": True}}))

    def snapshot(self, key):
        from src.envs.thor_checks import agent_pose
        metadata = self.controller.last_event.metadata
        value = {"key": key, "scene": self.scene, "objects": copy.deepcopy(metadata["objects"]),
                 "poses": poses(metadata["objects"]), "agent": agent_pose(self.controller.last_event),
                 "native_is_scene_at_rest": metadata["isSceneAtRest"]}
        if self.clone_context is not None:
            value["clone_context"] = copy.deepcopy(self.clone_context)
        value["reference"] = self.store.save_json(f"snapshot_{self.snapshot_count}.json", value)
        self.snapshot_count += 1
        return value

    def action(self, action, **arguments):
        self.calls.append((action, copy.deepcopy(arguments)))
        self.action_count += 1
        metadata = self.controller.last_event.metadata
        if action == "SetObjectPoses":
            seen = set()
            for pose in arguments["objectPoses"]:
                obj = next(row for row in metadata["objects"] if row["name"] == pose["objectName"])
                if pose["objectName"] in seen:
                    if self.omit_created_clone:
                        continue
                    obj = copy.deepcopy(obj)
                    obj.update(name="actual_copy_name", objectId="actual_new_native_id")
                    metadata["objects"].append(obj)
                seen.add(pose["objectName"])
                obj.update(position=copy.deepcopy(pose["position"]), rotation=copy.deepcopy(pose["rotation"]))
        elif action == "PlaceObjectAtPoint":
            obj = next(row for row in metadata["objects"] if row["objectId"] == arguments["objectId"])
            obj["position"] = {**arguments["position"], "y": 0.55}
            obj["rotation"] = copy.deepcopy(arguments["rotation"])
            obj["parentReceptacles"] = ["Table_native"]
        elif action == "AdvancePhysicsStep":
            metadata["isSceneAtRest"] = True
        metadata["lastActionSuccess"] = True
        self.last_action_reference = self.store.save_json(f"action_{self.action_count}.json", {"action": action})
        return self.controller.last_event


class CloneBackendContracts(unittest.TestCase):
    def test_one_duplicate_creation_then_unique_complete_physical_request(self):
        sim = SyntheticCloneSimulator()
        result = prepare_clone_baseline(sim, sim.snapshot("original"), "A", spawn())
        self.assertTrue(result["passed"], result["record"])
        self.assertEqual(result["lineage"]["clone_name"], "actual_copy_name")
        sets = [args["objectPoses"] for action, args in sim.calls if action == "SetObjectPoses"]
        self.assertEqual([p["objectName"] for p in sets[0]], ["A", "Chair", "X", "A"])
        self.assertEqual(set(p["objectName"] for p in sets[1]), {"A", "Chair", "X", "actual_copy_name"})
        self.assertEqual(len(sets[1]), 4)
        self.assertFalse(any(action in ("Reset", "InitialRandomSpawn", "TeleportFull") for action, _ in sim.calls))
        requested = result["record"]["construction_physical"]["requested_poses"]
        self.assertEqual(next(p for p in requested if p["objectName"] == "actual_copy_name")["position"]["y"], 0.55)
        self.assertTrue(result["snapshot"]["clone_context"]["setup_passed"])

    def test_original_preflight_and_owned_native_coordinate_are_required(self):
        sim = SyntheticCloneSimulator()
        sim.scene_preflight["passed"] = False
        with self.assertRaises(RuntimeError):
            prepare_clone_baseline(sim, sim.snapshot("original"), "A", spawn())
        sim.scene_preflight["passed"] = True
        sim.spawn_tokens = {}
        with self.assertRaises(ValueError):
            prepare_clone_baseline(sim, sim.snapshot("original"), "A", spawn())
        self.assertEqual(sim.calls, [])

    def test_failed_creation_without_context_blocks_further_world_construction(self):
        sim = SyntheticCloneSimulator()
        sim.omit_created_clone = True
        with self.assertRaises(RuntimeError):
            prepare_clone_baseline(sim, sim.snapshot("original"), "A", spawn())
        self.assertIsNone(sim.clone_context)
        self.assertTrue(sim.clone_preparation_attempted)
        with self.assertRaises(RuntimeError):
            sim.construct_world(sim.snapshot("failed"), "must-not-run", [])
        self.assertTrue(any(name.startswith("clone_baseline_attempt_") for name, _ in sim.store.rows))

    def test_clone_setup_is_once_per_controller_and_candidates_stay_fixed(self):
        sim = SyntheticCloneSimulator()
        result = prepare_clone_baseline(sim, sim.snapshot("original"), "A", spawn())
        with self.assertRaises(RuntimeError):
            prepare_clone_baseline(sim, result["snapshot"], "A", spawn())
        for name in ("A", "actual_copy_name"):
            with self.assertRaises(ValueError):
                sim.construct_world(result["snapshot"], "world", [{"object_name": name, "spawn": spawn()}])
        with patch.object(backend._BaseSimulator, "construct_world", return_value="delegated") as method:
            self.assertEqual(sim.construct_world(result["snapshot"], "world", [{"object_name": "X", "spawn": spawn()}]), "delegated")
            method.assert_called_once()

    def test_fresh_inventory_recreation_binds_actual_lineage_once(self):
        source = SyntheticCloneSimulator()
        result = prepare_clone_baseline(source, source.snapshot("original"), "A", spawn())
        replay = SyntheticCloneSimulator()
        check = restore_clone_inventory(replay, result["snapshot"])
        self.assertTrue(check["passed"])
        self.assertEqual(sum(action == "SetObjectPoses" for action, _ in replay.calls), 1)
        again = restore_clone_inventory(replay, result["snapshot"])
        self.assertTrue(again["already_recreated"])
        self.assertEqual(sum(action == "SetObjectPoses" for action, _ in replay.calls), 1)
        replay.clone_context = None
        with self.assertRaises(ValueError):
            restore_clone_inventory(replay, result["snapshot"])

    def test_extended_snapshot_preserves_old_and_hashes_full_clone_context(self):
        sim = SyntheticCloneSimulator()
        sim.clone_context = {"setup_passed": True, "synthetic": True}
        old = {"key": "original", "reference": {"file": "old-snapshot", "sha256": "old"}}
        with patch.object(backend._BaseSimulator, "snapshot", return_value=copy.deepcopy(old)):
            value = V2CloneSimulator.snapshot(sim, "copy")
        stored = sim.store.rows[-1][1]
        self.assertEqual(stored["clone_context"], sim.clone_context)
        self.assertEqual(stored["unextended_snapshot_reference"], old["reference"])
        self.assertNotIn("reference", stored)
        self.assertEqual(value["reference"]["sha256"], hashlib.sha256(encoded(stored)).hexdigest())

    def test_private_replay_binding_is_restored_even_after_failure(self):
        original = backend.engine.V2Simulator
        config = {"enable_candidate_clone": True, "snapshot": {"clone_context": {"setup_passed": True}}}
        def fail(*args):
            self.assertIs(backend.engine.V2Simulator, V2CloneSimulator)
            raise RuntimeError("synthetic replay failure")
        with patch.object(backend.engine, "replay_worker", side_effect=fail):
            with self.assertRaises(RuntimeError):
                clone_replay_worker(config, None)
        self.assertIs(backend.engine.V2Simulator, original)
        with self.assertRaises(ValueError):
            clone_replay_worker({"snapshot": config["snapshot"]}, None)


if __name__ == "__main__":
    unittest.main()
