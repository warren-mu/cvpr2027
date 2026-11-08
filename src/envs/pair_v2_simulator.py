"""Constructive T01v2 backend; import is CPU-only and never launches Unity.

The caller owns allocation validation, deadlines and the parent Worker cleanup.
Construction never resets the scene or teleports the agent. Metadata reads and
capture() issue no simulator actions. A capture after construction is the fresh
event returned by its final AdvancePhysicsStep, not an unlabelled cached render.
Snapshots serialize reported poses/state, not Unity's complete hidden state.
"""
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

from .thor_common import controller_session
from .thor_checks import agent_pose, full_poses
from ..tools.t01_collision_probe import EvidenceStore

BUILD = "f0825767cd50d69f666c7f282e54abfe58f1e917"
SOURCE = "https://github.com/allenai/ai2thor/blob/" + BUILD + "/unity/Assets/Scripts/"
FORBIDDEN = frozenset(("CollidersObjectCollidingWith", "TeleportObject", "DistanceToObject",
                       "BBoxDistance", "CheckWhatObjectOn", "InitialRandomSpawn", "Reset"))


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def distance(a, b):
    values = [float(a[k]) - float(b[k]) for k in ("x", "y", "z")]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite position")
    return math.sqrt(sum(x * x for x in values))


def objects_by_name(objects):
    result = {}
    for obj in objects:
        name = obj.get("name")
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("Missing/ambiguous stable object name")
        result[name] = obj
    return result


def editable_names(objects):
    return {name for name, obj in objects_by_name(objects).items()
            if obj.get("moveable") or obj.get("pickupable")}


def complete_pose_list(objects, poses):
    """Reject partial lists before SetObjectPoses can disable an omitted object."""
    names = [p["objectName"] for p in poses]
    if not names or len(set(names)) != len(names) or set(names) != editable_names(objects):
        raise ValueError("SetObjectPoses must cover each movable OR pickupable name exactly once")
    for pose in poses:
        distance(pose["position"], pose["position"])
        distance(pose["rotation"], pose["rotation"])
    return copy.deepcopy(poses)


def pose_measurements(before_objects, after_objects):
    before, after = objects_by_name(before_objects), objects_by_name(after_objects)
    names = editable_names(before_objects)
    rows = {}
    for name in sorted(names & after.keys()):
        old, new = before[name], after[name]
        rows[name] = {"position_delta_m": distance(old["position"], new["position"]),
                      "rotation_before": copy.deepcopy(old["rotation"]),
                      "rotation_after": copy.deepcopy(new["rotation"]),
                      "rotation_exact": old["rotation"] == new["rotation"],
                      "isMoving": new.get("isMoving")}
    return {"inventory_exact": set(before) == set(after),
            "editable_inventory_exact": names == editable_names(after_objects),
            "missing": sorted(set(before) - set(after)), "added": sorted(set(after) - set(before)),
            "objects": rows}


def restoration_state(snap, objects, agent):
    changes = pose_measurements(snap["objects"], objects)
    positions_ok = (changes["inventory_exact"] and changes["editable_inventory_exact"]
                    and all(row["position_delta_m"] <= 0.01 for row in changes["objects"].values()))
    agent_exact = encoded(snap["agent"]) == encoded(agent)
    return {"restored_positions_within_1cm": positions_ok, "agent_pose_exact": agent_exact,
            "restore_deterministic": bool(positions_ok and agent_exact),
            "pose_measurements": changes, "agent_requested": snap["agent"], "agent_realized": agent,
            "scope": "state restore: all movable/pickupable positions <=1cm and exact full agent pose; RGB separate"}


def validate_spawn(placement, scene):
    """Validate the selected literal native coordinate, without inventing offsets."""
    token = placement["spawn"]
    if (token.get("action") != "GetSpawnCoordinatesAboveReceptacle"
            or token.get("success") is not True or token.get("scene") != scene
            or not isinstance(token.get("query_reference"), dict)
            or not isinstance(token.get("receptacle_name"), str)):
        raise ValueError("Missing native receptacle placement evidence")
    selected = token["selected_position"]
    distance(selected, selected)
    if not any(encoded(selected) == encoded(point) for point in token.get("coordinates", [])):
        raise ValueError("Selected position is not one of the recorded native spawn coordinates")
    return copy.deepcopy(token)


def placement_geometry(obj, receptacle, token, clearance_m=0.005):
    """Pre-physics proposal for a flat top; never infer a contact from an AABB.

    The native API samples the *top of the receptacle trigger*, not a pivot or
    raycast surface. Keep its x/z, use the receptacle AABB top as an explicit
    surface proposal and preserve the object's current rotation. Native settling
    and parentReceptacles, not these boxes, decide whether this proposal works.
    """
    if not math.isfinite(clearance_m) or clearance_m != 0.005:
        raise ValueError("The predeclared geometric proposal uses 0.005m clearance")
    object_box = copy.deepcopy(obj["axisAlignedBoundingBox"])
    receptor_box = copy.deepcopy(receptacle["axisAlignedBoundingBox"])
    for box in (object_box, receptor_box):
        distance(box["center"], box["center"])
        distance(box["size"], box["size"])
        if any(float(box["size"][axis]) <= 0 for axis in ("x", "y", "z")):
            raise ValueError("Missing or degenerate AABB for constructive proposal")
    offset = obj["position"]["y"] - (object_box["center"]["y"] - object_box["size"]["y"] / 2)
    if offset < 0 or not math.isfinite(offset):
        raise ValueError("Object pivot is below its reported bounding-box bottom")
    requested = copy.deepcopy(token["selected_position"])
    requested["y"] = receptor_box["center"]["y"] + receptor_box["size"]["y"] / 2 + offset + clearance_m
    # Reject points outside the destination's footprint before any physics work.
    for axis in ("x", "z"):
        if abs(requested[axis] - receptor_box["center"][axis]) > receptor_box["size"][axis] / 2:
            raise ValueError("Native trigger point lies outside the receptor AABB footprint")
    proof = copy.deepcopy(token)
    proof.update(requested_position=requested, geometry={
        "method": "receptacle_aabb_top_plus_object_pivot_bottom_offset",
        "object_name": obj["name"], "object_position": copy.deepcopy(obj["position"]),
        "object_rotation": copy.deepcopy(obj["rotation"]), "object_aabb": object_box,
        "receptacle_aabb": receptor_box, "clearance_m": clearance_m,
        "object_pivot_bottom_offset_m": offset,
        "limitation": "flat-top surface proposal, not a raycast or collision/penetration certificate",
        "native_sample_source": SOURCE + "Contains.cs#L154",
        "official_pivot_offset_reference": SOURCE + "PhysicsRemoteFPSAgentController.cs#L3244"})
    return proof


class BlackFrame(SystemExit):
    """Stop a shard with the established black-frame-only exit code."""
    def __init__(self, reference):
        self.reference = reference
        super().__init__(86)


class V2Simulator:
    def __init__(self, controller, config, raw_dir):
        self.controller, self.config = controller, config
        self.scene = config["scene"]
        self.deadline = float(config["deadline"])
        self.resolution = int(config["resolution"])
        self.max_steps = int(config.get("settle_max_steps", 50))
        self.time_step = float(config.get("physics_time_step", 0.02))
        self.step_seconds = float(config.get("physics_seconds_per_step", 0.1))
        if (self.max_steps < 1 or not 0 < self.time_step <= 0.05
                or not math.isfinite(self.step_seconds) or self.step_seconds < self.time_step):
            raise ValueError("Invalid bounded settling configuration")
        scratch = Path(config["runtime"]["scratch"]).resolve()
        directory = Path(raw_dir).resolve()
        if not directory.is_relative_to(scratch) or directory == scratch:
            raise ValueError("Raw simulator evidence requires a private scratch subdirectory")
        self.store = EvidenceStore(directory, max_bytes=int(config.get("max_raw_bytes", 512 * 1024 * 1024)))
        self.action_count, self.frame_count, self.snapshot_count = 0, 0, 0
        self.last_action = "constructor"
        self.last_action_reference = None
        self.query_log = []
        self.last_capture = None
        self.scene_preflight = None
        self.spawn_tokens = {}
        self.agent_setup_command = None
        self.agent_setup_actual = None
        self.constructor = self.capture("constructor")

    def check_deadline(self):
        if time.time() >= self.deadline:
            raise TimeoutError("T01v2 simulator deadline reached")

    def _capture_event(self, event, stem):
        import numpy as np
        from PIL import Image
        self.check_deadline()
        prefix = f"frame_{self.frame_count:06d}"
        self.frame_count += 1
        metadata = copy.deepcopy(event.metadata)
        reference = self.store.save_json(prefix + "_metadata.json", metadata)
        frame = getattr(event, "frame", None)
        rgb = None if frame is None else np.asarray(frame).copy()
        files = {"metadata": reference}
        valid = rgb is not None and rgb.shape == (self.resolution, self.resolution, 3) and rgb.dtype == np.uint8
        if rgb is not None and rgb.ndim == 3 and rgb.shape[-1] == 3 and rgb.dtype == np.uint8:
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="PNG")
            files["rgb"] = self.store.save(prefix + "_rgb.png", buffer.getvalue())
        black = bool(valid and not np.any(rgb))
        evidence = {"stem": stem, "files": files, "valid": bool(valid), "black": black,
                    "rgb_shape": None if rgb is None else list(rgb.shape),
                    "rgb_sha256": None if rgb is None else hashlib.sha256(rgb.tobytes()).hexdigest(),
                    "source_action": self.last_action, "action_index": self.action_count - 1,
                    "sequence_id": metadata.get("sequenceId")}
        if not valid or black:
            failure = self.store.save_json(prefix + "_invalid_rgb.json", evidence)
            if black:
                raise BlackFrame(failure)
            raise RuntimeError("Missing/invalid RGB; evidence=" + str(failure["file"]))
        by_id = {o["objectId"]: o["name"] for o in objects_by_name(metadata["objects"]).values()}
        masks, structural = {}, {}
        for object_id, raw in event.instance_masks.items():
            mask = np.asarray(raw, dtype=bool).copy()
            if mask.shape != rgb.shape[:2]:
                raise ValueError("Instance mask dimensions do not match RGB")
            if object_id in by_id:
                name = by_id[object_id]
                if name in masks:
                    raise ValueError("Two mask IDs map to one stable object name")
                masks[name] = mask
            else:
                structural[object_id] = mask
        buffer = io.BytesIO()
        names = sorted(masks)
        np.savez_compressed(buffer, names=np.asarray(names, dtype=str),
            masks=np.stack([masks[n] for n in names]) if names else np.zeros((0, self.resolution, self.resolution), dtype=bool),
            structural_ids=np.asarray(sorted(structural), dtype=str),
            structural_masks=np.stack([structural[n] for n in sorted(structural)]) if structural else np.zeros((0, self.resolution, self.resolution), dtype=bool))
        files["masks"] = self.store.save(prefix + "_masks.npz", buffer.getvalue())
        evidence["mask_pixels_by_name"] = {name: int(mask.sum()) for name, mask in masks.items()}
        observation = {"rgb": rgb, "masks_by_name": masks, "metadata": metadata,
                       "agent": copy.deepcopy(metadata["agent"]),
                       "camera": {k: copy.deepcopy(metadata.get(k)) for k in
                                  ("cameraPosition", "fov", "screenWidth", "screenHeight")},
                       "evidence": evidence, "rgb_sha256": evidence["rgb_sha256"],
                       "worker_pid": os.getpid(), "unity_pid": self.controller.unity_pid}
        self.last_capture = observation
        return observation

    def capture(self, stem="capture"):
        """Read/copy the current event; never claim this operation rendered anew."""
        return self._capture_event(self.controller.last_event, stem)

    def action(self, action, **arguments):
        self.check_deadline()
        if action in FORBIDDEN:
            raise ValueError("Forbidden v2 action: " + action)
        index = self.action_count
        self.action_count += 1
        row = {"index": index, "action": action, "arguments": copy.deepcopy(arguments),
               "before_metadata": copy.deepcopy(self.controller.last_event.metadata),
               "after_metadata": None, "capture": None, "exception": None}
        self.last_action = action
        try:
            event = self.controller.step(action=action, **arguments)
            row["after_metadata"] = copy.deepcopy(event.metadata)
            observed = self._capture_event(event, f"action_{index}_{action}")
            row["capture"] = observed["evidence"]
            if event.metadata.get("lastActionSuccess") is not True:
                raise RuntimeError(f"{action}: {event.metadata.get('errorMessage')}")
            return event
        except BaseException as error:
            row["exception"] = f"{type(error).__name__}: {error}"
            if isinstance(error, BlackFrame):
                row["black_frame_reference"] = error.reference
            raise
        finally:
            self.last_action_reference = self.store.save_json(f"action_{index:06d}.json", row)
            self.query_log.append({"action": action, "reference": self.last_action_reference,
                                   "exception": row["exception"]})

    def snapshot(self, key):
        event = self.controller.last_event
        poses = complete_pose_list(event.metadata["objects"], full_poses(event))
        snap = {"key": key, "scene": self.scene, "poses": poses,
                "objects": copy.deepcopy(event.metadata["objects"]), "agent": agent_pose(event),
                "camera": {k: copy.deepcopy(event.metadata.get(k)) for k in
                           ("cameraPosition", "fov", "screenWidth", "screenHeight")},
                "native_is_scene_at_rest": event.metadata.get("isSceneAtRest"),
                "agent_setup_command": (copy.deepcopy(self.agent_setup_command)
                    if self.agent_setup_actual == agent_pose(event) else None),
                "openness": {o["name"]: o.get("openness", float(bool(o.get("isOpen"))))
                             for o in event.metadata["objects"] if o.get("openable")},
                "worker_pid": os.getpid(), "unity_pid": self.controller.unity_pid,
                "last_action": self.last_action, "last_action_reference": self.last_action_reference,
                "capture_reference": None if self.last_capture is None else copy.deepcopy(self.last_capture["evidence"]),
                "serialization_scope": "reported poses/agent/openness; velocities and full renderer state unavailable"}
        reference = self.store.save_json(f"snapshot_{self.snapshot_count:06d}.json", snap)
        self.snapshot_count += 1
        snap["reference"] = reference
        return snap

    def _settle(self, before_objects):
        """Dynamic bodies + explicit simulation, with autosimulation paused throughout."""
        observations = []
        settled = False
        for index in range(self.max_steps):
            event = self.action("AdvancePhysicsStep", timeStep=self.time_step,
                                simSeconds=self.step_seconds, allowAutoSimulation=False)
            deltas = pose_measurements(before_objects, event.metadata["objects"])
            settled = event.metadata.get("isSceneAtRest") is True
            observations.append({"index": index, "native_is_scene_at_rest": event.metadata.get("isSceneAtRest"),
                                 "pose_measurements": deltas, "action_reference": self.last_action_reference})
            if settled:
                break
        return {"settled": settled, "observations": observations,
                "maximum_steps": self.max_steps, "time_step_seconds": self.time_step,
                "seconds_per_step": self.step_seconds,
                "rotation_is_diagnostic_only": True,
                "simulation": "PausePhysicsAutoSim then explicit AdvancePhysicsStep; no collision query"}

    def preflight_scene(self):
        """The unedited scene is tested before any SetObjectPoses/randomization."""
        if self.action_count:
            raise RuntimeError("Original-scene preflight must be the first backend operation")
        baseline = self.snapshot("original_scene")
        self.action("PausePhysicsAutoSim")
        settling = self._settle(baseline["objects"])
        realized = self.snapshot("preflight_settled")
        changes = pose_measurements(baseline["objects"], realized["objects"])
        within = all(observation["pose_measurements"]["inventory_exact"]
            and observation["pose_measurements"]["editable_inventory_exact"]
            and all(row["position_delta_m"] <= 0.01
                    for row in observation["pose_measurements"]["objects"].values())
            for observation in settling["observations"])
        self.scene_preflight = {"passed": bool(settling["settled"] and within),
            "snapshot": realized, "diagnostics": {"settling": settling, "pose_measurements": changes,
            "position_max_m": 0.01, "source_snapshot": baseline}}
        self.store.save_json("scene_preflight.json", self.scene_preflight)
        return copy.deepcopy(self.scene_preflight)

    def reachable_positions(self):
        event = self.action("GetReachablePositions")
        values = event.metadata.get("actionReturn")
        if not isinstance(values, list) or not values:
            raise RuntimeError("No native reachable positions")
        for point in values:
            distance(point, point)
        return {"positions": copy.deepcopy(values), "query_reference": self.last_action_reference}

    def position_agent(self, position, yaw, horizon=0):
        """Candidate setup only; never called inside construct_world/prefix capture."""
        arguments = {"position": copy.deepcopy(position), "rotation": {"x": 0, "y": yaw, "z": 0},
                     "horizon": horizon, "standing": True, "forceAction": False}
        self.action("TeleportFull", **arguments)
        self.agent_setup_command = arguments
        self.agent_setup_actual = agent_pose(self.controller.last_event)
        return self.last_capture

    def spawn_points(self, receptacle_name):
        obj = objects_by_name(self.controller.last_event.metadata["objects"])[receptacle_name]
        if obj.get("receptacle") is not True:
            raise ValueError("Destination is not a native receptacle")
        before_position = copy.deepcopy(obj["position"])
        event = self.action("GetSpawnCoordinatesAboveReceptacle", objectId=obj["objectId"], anywhere=True)
        values = event.metadata.get("actionReturn")
        if not isinstance(values, list):
            raise RuntimeError("Unknown native spawn-coordinate schema")
        rows = []
        for point in values:
            distance(point, point)
            token = {"action": "GetSpawnCoordinatesAboveReceptacle", "success": True,
                "scene": self.scene, "receptacle_name": receptacle_name, "receptacle_id_at_query": obj["objectId"],
                "receptacle_position_at_query": before_position,
                "selected_position": copy.deepcopy(point), "coordinates": copy.deepcopy(values),
                "query_reference": copy.deepcopy(self.last_action_reference),
                "source": SOURCE + "PhysicsRemoteFPSAgentController.cs#L3423",
                "scope": "native receptacle samples; placement requires subsequent settling/membership checks"}
            self.spawn_tokens[hashlib.sha256(encoded(token)).hexdigest()] = copy.deepcopy(token)
            rows.append(token)
        return rows

    def construct_world(self, baseline, key, placements, require_agent=None):
        """Write complete poses, settle dynamic bodies; do not move the camera/agent."""
        if not self.scene_preflight or not self.scene_preflight["passed"]:
            raise RuntimeError("Construction requires a passed original-scene preflight")
        if baseline["scene"] != self.scene:
            raise ValueError("Snapshot scene mismatch")
        current = self.controller.last_event
        fixed_agent = copy.deepcopy(require_agent if require_agent is not None else agent_pose(current))
        initial_agent = agent_pose(current)
        if initial_agent != fixed_agent:
            raise ValueError("Agent already differs from required prefix pose")
        current_objects = current.metadata["objects"]
        poses = complete_pose_list(current_objects, baseline["poses"])
        if editable_names(current_objects) != editable_names(baseline["objects"]):
            raise ValueError("Construction inventory differs from baseline")
        requested = {p["objectName"]: p for p in poses}
        evidence = {}
        proposals = []
        for placement in placements:
            if placement.get("clone"):
                raise NotImplementedError("First v2 backend supports natural candidates only; cloning not measured")
            name = placement["object_name"]
            if name in evidence or name not in requested:
                raise ValueError("Duplicate or noneditable placement")
            token = validate_spawn(placement, self.scene)
            if hashlib.sha256(encoded(token)).hexdigest() not in self.spawn_tokens:
                raise ValueError("Spawn coordinate was not sampled by this backend instance")
            destination = objects_by_name(current_objects).get(token["receptacle_name"])
            if destination is None or destination.get("receptacle") is not True:
                raise ValueError("Placement destination disappeared")
            base_object = objects_by_name(baseline["objects"])[name]
            if "rotation" in placement and placement["rotation"] != base_object["rotation"]:
                raise ValueError("v2 single-position edit preserves the source object's rotation")
            evidence[name] = token
            proposals.append((name, base_object, token))
        self.action("PausePhysicsAutoSim")
        if self.controller.last_event.metadata.get("inventoryObjects") != []:
            raise RuntimeError("Constructive placement requires explicitly empty hands")
        for name, base_object, token in proposals:
            obj = objects_by_name(self.controller.last_event.metadata["objects"])[name]
            args = {"objectId": obj["objectId"], "position": copy.deepcopy(token["selected_position"]),
                    "rotation": copy.deepcopy(base_object["rotation"]), "forceKinematic": True}
            # This is a CONSTRUCTION proposal. The official forceKinematic=True
            # branch skips the settling coroutine, so this returned pivot is
            # captured BEFORE the later complete-pose request and physics steps.
            event = self.action("PlaceObjectAtPoint", **args)
            returned = objects_by_name(event.metadata["objects"])[name]
            proof = copy.deepcopy(token)
            proof.update(requested_position=copy.deepcopy(returned["position"]), geometry={
                "method": "native_PlaceObjectAtPoint_forceKinematic", "object_name": name,
                "source_rotation": copy.deepcopy(base_object["rotation"]),
                "placement_action": "PlaceObjectAtPoint", "placement_arguments": args,
                "placement_success": event.metadata.get("lastActionSuccess"),
                "returned_position": copy.deepcopy(returned["position"]),
                "returned_rotation": copy.deepcopy(returned["rotation"]),
                "placement_reference": copy.deepcopy(self.last_action_reference),
                "captured_before_complete_pose_set_and_settling": True,
                "source": SOURCE + "PhysicsRemoteFPSAgentController.cs#L3178"})
            requested[name]["position"] = copy.deepcopy(returned["position"])
            # Preserve the baseline rotation in the actual requested world. The
            # native proposal's returned Euler representation remains evidence.
            evidence[name] = proof
        poses = complete_pose_list(current_objects, poses)
        before = copy.deepcopy(baseline["objects"])
        request_record = {"key": key, "baseline_key": baseline["key"], "complete_requested_poses": poses,
                          "placement_evidence_by_name": evidence, "required_agent": fixed_agent}
        request_ref = self.store.save_json(f"construction_{self.action_count:06d}_request.json", request_record)
        # No preceding placeStationary=True: preserve native collision mode.
        self.action("SetObjectPoses", objectPoses=poses, placeStationary=False)
        settling = self._settle(before)
        observed = self.last_capture  # final AdvancePhysicsStep really rendered this event
        snap = self.snapshot(key)
        changes = pose_measurements(before, snap["objects"])
        actual = objects_by_name(snap["objects"])
        rows = {}
        for name in sorted(evidence):
            obj, token = actual[name], evidence[name]
            destination = actual.get(token["receptacle_name"])
            rows[name] = {"position_error_m": distance(requested[name]["position"], obj["position"]),
                "requested_position": requested[name]["position"], "actual_position": obj["position"],
                "requested_rotation": requested[name]["rotation"], "actual_rotation": obj["rotation"],
                "parentReceptacles": copy.deepcopy(obj.get("parentReceptacles")),
                "target_receptacle_name": token["receptacle_name"],
                "target_receptacle_id": None if destination is None else destination["objectId"],
                "parent_membership": destination is not None and isinstance(obj.get("parentReceptacles"), list)
                    and destination["objectId"] in obj["parentReceptacles"]}
        unchanged = all(row["position_delta_m"] <= 0.01 for name, row in changes["objects"].items() if name not in evidence)
        agent_unchanged = initial_agent == snap["agent"] == fixed_agent
        passed = (settling["settled"] and changes["inventory_exact"] and changes["editable_inventory_exact"]
                  and unchanged and agent_unchanged
                  and all(row["position_error_m"] <= 0.02 and row["parent_membership"] for row in rows.values()))
        physical = {"valid_physical": bool(passed), "settling": settling, "edited": rows,
            "collateral_position_within_1cm": unchanged, "pose_measurements": changes,
            "agent_unchanged": agent_unchanged, "agent_before": initial_agent, "agent_after": snap["agent"],
            "native_is_scene_at_rest": snap["native_is_scene_at_rest"],
            "before_objects": before, "requested_poses": poses, "realized_objects": snap["objects"],
            "edited_names": sorted(evidence), "placement_evidence_by_name": evidence,
            "request_reference": request_ref,
            "physical_validity_method": "approved receptacle construction + native rest + position bounds + parentReceptacles; no penetration-depth measurement"}
        result = {"passed": bool(passed), "snapshot": snap, "physical": physical,
                  "capture": observed, "placement_evidence_by_name": evidence}
        self.store.save_json(f"construction_{self.action_count:06d}_outcome.json",
                            {k: v for k, v in result.items() if k != "capture"})
        return result

    def shortest_path(self, source, target):
        from .pair_simulator import path_complete, point
        tolerance = float(self.config.get("path_endpoint_tolerance_m", 0.05))
        event = self.action("GetShortestPathToPoint", position=copy.deepcopy(source),
                            target=copy.deepcopy(target), allowedError=tolerance)
        raw = copy.deepcopy(event.metadata.get("actionReturn"))
        return {"source": copy.deepcopy(source), "target": copy.deepcopy(target), "raw": raw,
                "complete": path_complete(raw, point(source), point(target), tolerance),
                "query_reference": self.last_action_reference, "endpoint_tolerance_m": tolerance}

    def observe(self, action):
        """One actual observation action. Moves need a saved native path certificate."""
        kind = action["kind"]
        if kind in ("turn_left", "turn_right"):
            self.action("RotateLeft" if kind == "turn_left" else "RotateRight", degrees=45)
        elif kind == "move":
            path = action["path"]
            current = agent_pose(self.controller.last_event)
            target = action["position"]
            if (path.get("complete") is not True or encoded(target) != encoded(path["target"])
                    or distance(current["position"], path["source"]) > path["endpoint_tolerance_m"]
                    or not 0 < distance(current["position"], target) <= 1.5):
                raise ValueError("Move requires a complete adjacent-viewpoint path from the current state")
            self.action("TeleportFull", position=copy.deepcopy(target), rotation=current["rotation"],
                        horizon=current["horizon"], standing=current["standing"], forceAction=False)
        else:
            raise ValueError("v2 observation must turn 45 degrees or move to a verified adjacent viewpoint")
        return self.last_capture

    def restore_snapshot(self, snap):
        """Explicit replay/setup operation; never used between paired prefixes."""
        if snap["scene"] != self.scene:
            raise ValueError("Replay scene mismatch")
        poses = complete_pose_list(self.controller.last_event.metadata["objects"], snap["poses"])
        self.action("PausePhysicsAutoSim")
        self.action("SetObjectPoses", objectPoses=poses, placeStationary=True)
        for name, desired in sorted(snap.get("openness", {}).items()):
            obj = objects_by_name(self.controller.last_event.metadata["objects"])[name]
            actual = obj.get("openness", float(bool(obj.get("isOpen"))))
            if actual != desired:
                self.action("OpenObject" if desired > 0 else "CloseObject", objectId=obj["objectId"],
                            forceAction=True, **({"openness": desired} if desired > 0 else {}))
        # Reuse the originally EXECUTED setup arguments (e.g. horizon=30), not
        # its float32 metadata result (30.0000038), which compounds Euler drift.
        setup = copy.deepcopy(snap.get("agent_setup_command"))
        if setup is None:
            setup = {**copy.deepcopy(snap["agent"]), "forceAction": False}
        self.action("TeleportFull", **setup)
        self.agent_setup_command = setup
        self.agent_setup_actual = agent_pose(self.controller.last_event)
        measurement = restoration_state(snap, self.controller.last_event.metadata["objects"],
                                        agent_pose(self.controller.last_event))
        restored = self.snapshot(snap["key"] + "_restored")
        measurement["restored_snapshot_reference"] = restored["reference"]
        measurement["executed_agent_setup_arguments"] = setup
        measurement["restore_max_abs_diff"] = None
        measurement["restore_rgb_exact"] = None
        source_ref = (snap.get("capture_reference") or {}).get("files", {}).get("rgb")
        if source_ref:
            import numpy as np
            from PIL import Image
            content = Path(source_ref["file"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != source_ref["sha256"]:
                raise ValueError("Construction prefix PNG hash changed before replay")
            original = np.asarray(Image.open(io.BytesIO(content)).convert("RGB"))
            actual = self.last_capture["rgb"]
            if original.shape != actual.shape:
                raise ValueError("Snapshot/replay RGB shape mismatch")
            measurement["restore_max_abs_diff"] = int(np.abs(original.astype(np.int16) - actual.astype(np.int16)).max())
            measurement["restore_rgb_exact"] = bool(np.array_equal(original, actual))
        self.store.save_json(f"restore_{self.action_count:06d}.json", measurement)
        return self.last_capture, measurement

    def select(self, target_name):
        """Read-only native metadata + actual mask; no DistanceToObject action."""
        import numpy as np
        event = self.controller.last_event
        target = objects_by_name(event.metadata["objects"])[target_name]
        mask = event.instance_masks.get(target["objectId"])
        pixels = 0 if mask is None else int(np.asarray(mask, dtype=bool).sum())
        measured = distance(event.metadata["agent"]["position"], target["position"])
        return {"kind": "select", "target_name": target_name, "selected_object_name": target_name,
                "selected_object_id": target["objectId"], "visible_mask_pixels": pixels,
                "distance_m": measured, "metadata_distance": target.get("distance"),
                "distance_method": "3D Euclidean agent metadata.position to object metadata.position",
                "success": bool(pixels > 0 and measured <= 1.5), "measurement_action_sent": False}

    def finish(self):
        return self.store.index()


def replay_worker(config, connection):
    """Use common.Worker to run this in a distinct process and own Unity lifetime."""
    config = copy.deepcopy(config)
    replay_raw = Path(config['raw_dir'])
    config['runtime']['log_dir'] = str(replay_raw.with_name(replay_raw.name + '_unity_logs'))
    from .pair_v2_policy import closest_target, mark_observation
    result = {"status": "running", "passed": False, "actions": [], "captures": [],
              "cleanup_errors": [], "worker_pid": os.getpid(), "process_id": os.getpid(),
              "executed_actions": [], "action_successes": [], "success": False, "done": False,
              "error": None}
    sim = None
    try:
        with controller_session(config, "v2_replay", segmentation=True, connection=connection) as controller:
            sim = V2Simulator(controller, config, config["raw_dir"])
            snap, actions = config["snapshot"], config["actions"]
            result["unity_pid"] = controller.unity_pid
            if snap.get("worker_pid") == os.getpid() or snap.get("unity_pid") == controller.unity_pid:
                raise RuntimeError("Witness must run in a fresh worker and Unity process")
            if not 1 <= len(actions) <= 12 or actions[-1].get("kind") != "select":
                raise ValueError("Witness requires <=12 actions including a final select")
            if any(a.get("kind") == "select" for a in actions[:-1]):
                raise ValueError("Only the terminal witness action may commit")
            def public_action(action):
                kind = action["kind"]
                if kind in ("turn_left", "turn_right"):
                    return kind
                number = action.get("label") if kind == "select" else action.get("index")
                if kind not in ("select", "move") or type(number) is not int or number < 1:
                    raise ValueError("Witness requires explicit public select label / move index")
                return f"{kind} {number}"
            result["planned_actions"] = [public_action(a) for a in actions]
            result["construction_process_id"] = snap["worker_pid"]
            observation, restoration = sim.restore_snapshot(snap)
            result["restoration"] = restoration
            result["snapshot_restored"] = restoration["restore_deterministic"]
            result["restore_deterministic"] = restoration["restore_deterministic"]
            result["restore_max_abs_diff"] = restoration["restore_max_abs_diff"]
            result["captures"].append(observation["evidence"])
            for index, action in enumerate(actions):
                if action["kind"] == "select":
                    marked = mark_observation(observation, config["candidate_names"], config["instruction"])
                    selected = [name for name, label in marked["mark_labels_by_name"].items() if label == action["label"]]
                    resolved = closest_target(observation["metadata"]["objects"],
                                              config["candidate_names"], config["anchor_name"])
                    if len(selected) != 1:
                        row = {"kind": "select", "success": False, "selected_object_name": None,
                               "distance_m": None, "visible_mask_pixels": 0}
                    else:
                        row = sim.select(selected[0])
                        row["success"] = (row["success"] and selected[0] == config["target_name"]
                            and resolved["passed"] and resolved["target_name"] == selected[0])
                    row.update(label=action["label"], realized_relation=resolved,
                               mark_labels_by_name=marked["mark_labels_by_name"])
                    result["actions"].append(row)
                    result.update(selected_name=row["selected_object_name"], selected_distance_m=row["distance_m"],
                        selected_mask_pixels=row["visible_mask_pixels"], done=True, success=row["success"],
                        passed=row["success"] and result["snapshot_restored"])
                    result["action_successes"].append(bool(row["success"]))
                else:
                    execution = copy.deepcopy(action)
                    if execution['kind'] == 'move':
                        current_position = agent_pose(sim.controller.last_event)['position']
                        execution['path'] = sim.shortest_path(current_position, execution['position'])
                    observation = sim.observe(execution)
                    result["actions"].append({"kind": action["kind"], "success": True,
                                               "action_reference": sim.last_action_reference})
                    result["captures"].append(observation["evidence"])
                    result["action_successes"].append(True)
                result["executed_actions"].append(public_action(action))
                connection.send(("progress", copy.deepcopy(result)))
            result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", passed=False, success=False, error=f"{type(error).__name__}: {error}")
        if isinstance(error, BlackFrame):
            result.update(exit_code=86, black_frame_reference=error.reference)
        # Controller.stop exceptions are also fatal, not successful witnesses.
    finally:
        if sim is not None:
            try:
                result["raw_evidence"] = sim.finish()
            except BaseException as error:
                result.update(status="failed", passed=False)
                result["cleanup_errors"].append(f"evidence finalization: {type(error).__name__}: {error}")
    return result
