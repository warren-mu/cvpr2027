"""Synthetic CPU contracts only; no Unity, CUDA, measurements or research output."""
import copy
import hashlib
from types import SimpleNamespace
import unittest

from src.envs.pair_v2_simulator import (V2Simulator, complete_pose_list, encoded,
    pose_measurements, restoration_state, validate_spawn)


def xyz(x=0.0, y=0.0, z=0.0):
    return {"x": x, "y": y, "z": z}


def object_row(name, *, pickupable=False, moveable=False, receptacle=False):
    return {"name": name, "objectId": name + "_id", "objectType": name,
            "position": xyz(), "rotation": xyz(), "pickupable": pickupable,
            "moveable": moveable, "receptacle": receptacle, "parentReceptacles": [],
            "axisAlignedBoundingBox": {"center": xyz(), "size": xyz(1.0, 1.0, 1.0)}}


def fixture():
    objects = [object_row("Furniture", moveable=True), object_row("X", pickupable=True),
               object_row("Table", receptacle=True)]
    poses = [{"objectName": row["name"], "position": copy.deepcopy(row["position"]),
              "rotation": copy.deepcopy(row["rotation"])} for row in objects if row["moveable"] or row["pickupable"]]
    agent = {"position": xyz(), "rotation": xyz(), "horizon": 30.0000038, "standing": True}
    return {"key": "base", "scene": "FloorPlan1", "objects": objects, "poses": poses, "agent": agent}


def spawn():
    return {"action": "GetSpawnCoordinatesAboveReceptacle", "success": True, "scene": "FloorPlan1",
            "receptacle_name": "Table", "selected_position": xyz(0.2, 0.8, 0.2),
            "coordinates": [xyz(0.2, 0.8, 0.2)], "receptacle_position_at_query": xyz(),
            "query_reference": {"file": "synthetic_only", "sha256": "fixture"}}


class MemoryStore:
    def __init__(self):
        self.rows = []

    def save_json(self, name, value):
        self.rows.append((name, copy.deepcopy(value)))
        return {"file": name, "sha256": hashlib.sha256(encoded(value)).hexdigest()}


class SyntheticSimulator(V2Simulator):
    """Tests the real construction control flow with explicit synthetic actions."""
    def __init__(self):
        base = fixture()
        agent = base["agent"]
        self.controller = SimpleNamespace(last_event=SimpleNamespace(metadata={
            "objects": copy.deepcopy(base["objects"]), "inventoryObjects": [], "isSceneAtRest": False,
            "agent": {"position": agent["position"], "rotation": agent["rotation"],
                      "cameraHorizon": agent["horizon"], "isStanding": agent["standing"]}}), unity_pid=123)
        self.scene, self.max_steps = base["scene"], 3
        self.time_step, self.step_seconds = 0.02, 0.1
        self.store, self.calls = MemoryStore(), []
        self.action_count = 0
        self.last_action_reference = None
        self.last_capture = {"evidence": {"synthetic": True}}
        self.scene_preflight = {"passed": True}
        self.spawn_tokens = {hashlib.sha256(encoded(spawn())).hexdigest(): spawn()}
        self.collateral = 0.0
        self.rotated = False
        self.agent_drift = False
        self.scene_rests = True

    def action(self, action, **arguments):
        self.calls.append((action, copy.deepcopy(arguments)))
        self.action_count += 1
        self.last_action_reference = {"file": f"synthetic_action_{self.action_count}"}
        metadata = self.controller.last_event.metadata
        if action == "PlaceObjectAtPoint":
            obj = next(o for o in metadata["objects"] if o["objectId"] == arguments["objectId"])
            # Synthetic native result is deliberately different from the native
            # grid point. Requested pivot must come from this pre-settle result.
            obj["position"] = {**arguments["position"], "y": 0.55}
            obj["rotation"] = copy.deepcopy(arguments["rotation"])
            obj["parentReceptacles"] = ["Table_id"]
        elif action == "SetObjectPoses":
            for pose in arguments["objectPoses"]:
                obj = next(o for o in metadata["objects"] if o["name"] == pose["objectName"])
                obj["position"] = copy.deepcopy(pose["position"])
                obj["rotation"] = copy.deepcopy(pose["rotation"])
        elif action == "AdvancePhysicsStep":
            metadata["isSceneAtRest"] = self.scene_rests
            metadata["objects"][0]["position"]["x"] = self.collateral
            if self.rotated:
                metadata["objects"][0]["rotation"]["x"] = 0.00003
            if self.agent_drift:
                metadata["agent"]["cameraHorizon"] = 30.0000076
        metadata["lastActionSuccess"] = True
        self.last_capture = {"evidence": {"source_action": action, "synthetic": True}}
        return self.controller.last_event

    def snapshot(self, key):
        metadata = self.controller.last_event.metadata
        from src.envs.thor_checks import full_poses, agent_pose
        return {"key": key, "scene": self.scene, "objects": copy.deepcopy(metadata["objects"]),
                "poses": full_poses(self.controller.last_event), "agent": agent_pose(self.controller.last_event),
                "native_is_scene_at_rest": metadata["isSceneAtRest"]}


class PairV2SimulatorContracts(unittest.TestCase):
    def test_complete_inventory_uses_moveable_or_pickupable(self):
        base = fixture()
        self.assertEqual(len(complete_pose_list(base["objects"], base["poses"])), 2)
        with self.assertRaises(ValueError):
            complete_pose_list(base["objects"], base["poses"][1:])
        with self.assertRaises(ValueError):
            complete_pose_list(base["objects"], base["poses"] + [base["poses"][0]])

    def test_native_spawn_requires_selected_raw_coordinate(self):
        token = spawn()
        self.assertEqual(validate_spawn({"spawn": token}, "FloorPlan1"), token)
        token["selected_position"]["y"] += 0.01
        with self.assertRaises(ValueError):
            validate_spawn({"spawn": token}, "FloorPlan1")

    def test_restoration_agent_is_exact_and_object_rotation_is_diagnostic(self):
        base = fixture()
        actual = copy.deepcopy(base["objects"])
        actual[0]["position"]["x"] = 0.01
        actual[0]["rotation"]["z"] = 0.00003
        self.assertTrue(restoration_state(base, actual, base["agent"])["restore_deterministic"])
        agent = copy.deepcopy(base["agent"])
        agent["horizon"] = 30.0000076
        self.assertFalse(restoration_state(base, actual, agent)["restore_deterministic"])
        actual[0]["position"]["x"] = 0.010001
        self.assertFalse(restoration_state(base, actual, base["agent"])["restore_deterministic"])

    def test_inventory_loss_never_passes_restoration(self):
        base = fixture()
        self.assertFalse(restoration_state(base, base["objects"][1:], base["agent"])["restore_deterministic"])

    def test_original_scene_preflight_never_rewrites_poses(self):
        sim = SyntheticSimulator()
        result = sim.preflight_scene()
        self.assertTrue(result["passed"])
        self.assertEqual([action for action, _ in sim.calls], ["PausePhysicsAutoSim", "AdvancePhysicsStep"])

    def test_unsettled_or_displaced_scene_is_excluded(self):
        sim = SyntheticSimulator()
        sim.scene_rests = False
        self.assertFalse(sim.preflight_scene()["passed"])
        self.assertEqual(sum(a == "AdvancePhysicsStep" for a, _ in sim.calls), sim.max_steps)
        sim = SyntheticSimulator()
        sim.collateral = 0.010001
        self.assertFalse(sim.preflight_scene()["passed"])

    def test_proposal_pivot_is_frozen_before_complete_set_and_settle(self):
        sim = SyntheticSimulator()
        result = sim.construct_world(fixture(), "w0", [{"object_name": "X", "spawn": spawn()}])
        self.assertTrue(result["passed"])
        self.assertEqual([a for a, _ in sim.calls],
                         ["PausePhysicsAutoSim", "PlaceObjectAtPoint", "SetObjectPoses", "AdvancePhysicsStep"])
        self.assertTrue(sim.calls[1][1]["forceKinematic"])
        self.assertFalse(sim.calls[2][1]["placeStationary"])
        self.assertEqual(len(sim.calls[2][1]["objectPoses"]), 2)
        self.assertEqual(result["physical"]["requested_poses"][1]["position"]["y"], 0.55)
        self.assertEqual(result["capture"]["evidence"]["source_action"], "AdvancePhysicsStep")

    def test_collateral_position_and_agent_drift_fail_but_rotation_does_not(self):
        for field, expected in (("rotated", True), ("agent_drift", False), ("collateral", False)):
            sim = SyntheticSimulator()
            setattr(sim, field, 0.010001 if field == "collateral" else True)
            result = sim.construct_world(fixture(), "w0", [{"object_name": "X", "spawn": spawn()}])
            self.assertEqual(result["passed"], expected)

    def test_unowned_spawn_and_cloning_are_not_silently_accepted(self):
        sim = SyntheticSimulator()
        sim.spawn_tokens = {}
        with self.assertRaises(ValueError):
            sim.construct_world(fixture(), "w0", [{"object_name": "X", "spawn": spawn()}])
        sim = SyntheticSimulator()
        with self.assertRaises(NotImplementedError):
            sim.construct_world(fixture(), "w0", [{"object_name": "X", "spawn": spawn(), "clone": True}])


if __name__ == "__main__":
    unittest.main()
