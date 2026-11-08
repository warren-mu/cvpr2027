"""T01 real-THOR adapter. Imports do not initialize Unity or a GPU.

Pinned query semantics: AI2-THOR build f0825767cd50d69f666c7f282e54abfe58f1e917.
Native primitive-collider queries are operational checks, not a proof about
unreported meshes. Bounding boxes are used only for the explicitly named
nearest-relation metric; they never establish non-penetration.
"""
from collections import deque
from dataclasses import asdict
import copy
import hashlib
import json
import math
from pathlib import Path
import time

from .pair_core import (Action, ObjectState, PathEvidence, Point3, RelationalTask,
                        RelationEvidence, Viewpoint, ViewState, WorldSnapshot,
                        build_navigation_graph, farthest_point_sample,
                        navigation_transitions, neighbor_descriptions,
                        resolve_target, selection_action, visible_marks)
from .thor_common import step, valid_frame
from .thor_checks import full_poses, agent_pose

BUILD = "f0825767cd50d69f666c7f282e54abfe58f1e917"
SOURCE = "https://github.com/allenai/ai2thor/blob/" + BUILD + "/unity/Assets/Scripts/"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def point(value):
    return Point3(**{axis: value[axis] for axis in ("x", "y", "z")})


def object_by_identity(objects, field, value):
    """Resolve one realized object; names persist when SetObjectPoses rewrites IDs."""
    matches = [obj for obj in objects if obj.get(field) == value]
    if len(matches) != 1:
        raise ValueError(f"Expected one object with {field}={value!r}, found {len(matches)}")
    return matches[0]


def snapshot(event, scene, key):
    objects = []
    for obj in event.metadata["objects"]:
        objects.append(ObjectState(obj["objectId"], obj["name"], obj["objectType"],
            point(obj["position"]), point(obj["rotation"]),
            **{field: bool(obj.get(field, False)) for field in
               ("moveable", "pickupable", "openable", "receptacle")}))
    return WorldSnapshot(scene, key, tuple(objects), True)


def pose_changes(before, after):
    """Exact float comparisons are retained; no implicit acceptance epsilon."""
    a = {obj["name"]: obj for obj in before}
    b = {obj["name"]: obj for obj in after}
    changed = []
    for name in sorted(a.keys() & b.keys()):
        if a[name]["position"] != b[name]["position"] or a[name]["rotation"] != b[name]["rotation"]:
            changed.append({"name": name,
                "position_delta_m": point(a[name]["position"]).distance(point(b[name]["position"])),
                "rotation_exact": a[name]["rotation"] == b[name]["rotation"],
                "before": {k: a[name][k] for k in ("position", "rotation")},
                "after": {k: b[name][k] for k in ("position", "rotation")}})
    return {"inventory_exact": set(a) == set(b), "missing": sorted(a.keys() - b.keys()),
            "added": sorted(b.keys() - a.keys()), "changed": changed,
            "all_object_poses_exact": set(a) == set(b) and not changed}


def path_complete(payload, source, target, endpoint_tolerance):
    """Require explicit complete status and both endpoints, not just corners."""
    if not isinstance(payload, dict) or type(payload.get("status")) is bool or payload.get("status") not in (0, "PathComplete"):
        return False
    corners = payload.get("corners")
    if not isinstance(corners, list) or len(corners) < 2:
        return False
    try:
        parsed = [point(item) for item in corners]
    except (TypeError, ValueError, KeyError):
        return False
    # NavMesh corners are floor-height points; agent Y is body height.
    xz = lambda a, b: math.hypot(a.x - b.x, a.z - b.z)
    return xz(parsed[0], source) <= endpoint_tolerance and xz(parsed[-1], target) <= endpoint_tolerance


def compare_inputs(first, second, relevant_ids):
    import numpy as np
    if first["rgb"].shape != second["rgb"].shape:
        raise ValueError("RGB shape changed")
    delta = np.abs(first["rgb"].astype(np.int16) - second["rgb"].astype(np.int16))
    marked = np.abs(first["marked"].astype(np.int16) - second["marked"].astype(np.int16))
    union = np.zeros(delta.shape[:2], dtype=bool)
    for obs in (first, second):
        for object_id in relevant_ids:
            if object_id in obs["masks"]:
                union |= obs["masks"][object_id]
    changed = np.any(delta != 0, axis=2)
    return {"rgb_exact": bool(np.array_equal(first["rgb"], second["rgb"])),
            "marked_rgb_exact": bool(np.array_equal(first["marked"], second["marked"])),
            "text_exact": first["input_text"] == second["input_text"],
            "rgb_max_abs_diff": int(delta.max()), "marked_max_abs_diff": int(marked.max()),
            "prefix_max_abs_diff": int(max(delta.max(), marked.max())),
            "different_pixels": int(changed.sum()),
            "relevant_mask_pixels": int(union.sum()),
            "relevant_difference_pixels": int((changed & union).sum()),
            "same_input_at_t_strict": bool(np.array_equal(first["rgb"], second["rgb"])
                and np.array_equal(first["marked"], second["marked"])
                and first["input_text"] == second["input_text"])}


def save_observation(directory, stem, observation):
    import numpy as np
    from PIL import Image
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for field in ("rgb", "marked"):
        p = directory / f"{stem}_{field}.png"
        Image.fromarray(observation[field]).save(p)
        paths[field] = str(p)
    masks = sorted(observation["masks"])
    p = directory / f"{stem}_masks.npz"
    np.savez_compressed(p, ids=np.asarray(masks, dtype=str),
                        masks=np.stack([observation["masks"][key] for key in masks])
                        if masks else np.zeros((0, *observation["rgb"].shape[:2]), dtype=bool))
    paths["masks"] = str(p)
    p = directory / f"{stem}_metadata.json"
    p.write_text(json.dumps({key: observation[key] for key in
        ("metadata", "marks", "input_text", "state", "world_key", "rgb_sha256", "marked_sha256")},
        indent=2, allow_nan=False) + "\n")
    paths["metadata"] = str(p)
    return paths


class PairSimulator:
    def __init__(self, controller, config):
        self.controller, self.config = controller, config
        self.scene = config["scene"]
        self.query_log = []
        self.world = None
        self.graph_cache = {}
        self.observation_cache = {}
        self.nodes = ()
        self.reset_count_at_pair_start = None

    def check_deadline(self):
        if time.time() >= self.config["deadline"]:
            raise TimeoutError("T01 generation deadline expired")

    def action(self, action, **kwargs):
        self.check_deadline()
        event = self.controller.step(action=action, **kwargs)
        self.query_log.append({"action": action, "arguments": copy.deepcopy(kwargs),
            "success": event.metadata.get("lastActionSuccess"),
            "error": event.metadata.get("errorMessage"),
            "actionReturn": copy.deepcopy(event.metadata.get("actionReturn"))})
        if event.metadata.get("lastActionSuccess") is not True:
            raise RuntimeError(f"{action}: {event.metadata.get('errorMessage')}")
        return event

    def world_from_event(self, key):
        event = self.controller.last_event
        return {"key": key, "scene": self.scene, "poses": full_poses(event),
                "objects": copy.deepcopy(event.metadata["objects"]),
                "agent": agent_pose(event),
                "openness": {obj["name"]: obj.get("openness", 1.0 if obj.get("isOpen") else 0.0)
                             for obj in event.metadata["objects"] if obj.get("openable")}}

    def apply_world(self, world, opened=()):
        """No reset here: every switch uses the complete supplied pose list."""
        expected = {o["name"] for o in world["objects"] if o.get("moveable") or o.get("pickupable")}
        names = [p["objectName"] for p in world["poses"]]
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("Incomplete or duplicated SetObjectPoses list")
        self.action("SetObjectPoses", objectPoses=world["poses"], placeStationary=True)
        # f082 SetObjectPoses regenerates objectIds from positions. Both saved
        # openness and opened-container state therefore use stable unique names.
        for name, baseline in sorted(world["openness"].items()):
            openness = 1.0 if name in opened else baseline
            obj = object_by_identity(self.controller.last_event.metadata["objects"], "name", name)
            if obj.get("openness", 1.0 if obj.get("isOpen") else 0.0) != openness:
                self.action("OpenObject" if openness > 0 else "CloseObject", objectId=obj["objectId"],
                            forceAction=True, **({"openness": openness} if openness > 0 else {}))
        self.world = world
        return self.controller.last_event

    def teleport(self, state):
        node = next(node for node in self.nodes if node.viewpoint_id == state.viewpoint_id)
        return self.action("TeleportFull", position=node.position.as_dict(),
            rotation={"x": 0, "y": state.heading_index * 45, "z": 0}, horizon=0,
            standing=True, forceAction=False)

    def sample_nodes(self, seed):
        event = self.action("GetReachablePositions")
        raw = event.metadata.get("actionReturn")
        if not isinstance(raw, list):
            raise RuntimeError("GetReachablePositions did not return positions")
        sample = farthest_point_sample([point(item) for item in raw], count=self.config["viewpoints"], seed=seed)
        if not sample.complete:
            raise RuntimeError(sample.reason)
        self.nodes = tuple(Viewpoint(f"v{i:02d}", p) for i, p in enumerate(sample.points))
        return {"sample": asdict(sample), "reachable_positions": raw}

    def graph(self, world, opened=()):
        key = world["key"] + ":" + digest(list(opened))
        if key in self.graph_cache:
            return self.graph_cache[key]
        self.apply_world(world, opened)
        evidence = {}
        for source in self.nodes:
            for target in self.nodes:
                if source == target or source.position.distance(target.position) > 1.5:
                    continue
                try:
                    event = self.action("GetShortestPathToPoint", position=source.position.as_dict(),
                        target=target.position.as_dict(), allowedError=self.config["path_endpoint_tolerance_m"])
                    raw = event.metadata.get("actionReturn")
                    feasible = path_complete(raw, source.position, target.position, self.config["path_endpoint_tolerance_m"])
                    # The native API rejects incomplete NavMesh paths. Keep its
                    # exact response; physical execution is additionally checked
                    # for every move in the final witness replay.
                except RuntimeError:
                    feasible = False
                evidence[(source.viewpoint_id, target.viewpoint_id)] = PathEvidence(
                    source.viewpoint_id, target.viewpoint_id, key, feasible, f"query:{len(self.query_log)-1}")
        graph = build_navigation_graph(self.nodes, evidence, context_key=key, opened_containers=opened)
        self.graph_cache[key] = graph
        return graph

    def capture(self, world, state, task, graph):
        import numpy as np
        from PIL import Image, ImageDraw
        event = self.controller.last_event
        rgb = valid_frame(event, self.config["resolution"]).copy()
        masks = {key: np.asarray(mask, dtype=bool).copy() for key, mask in event.instance_masks.items()
                 if np.any(mask)}
        snap = snapshot(event, self.scene, world["key"])
        mask_pixels = {key: [(int(x), int(y)) for y, x in np.argwhere(mask)] for key, mask in masks.items()
                       if any(o.object_id == key and (o.object_type == task.target_type or (o.openable and o.receptacle))
                              for o in snap.objects)}
        marks = visible_marks(snap, mask_pixels, target_type=task.target_type,
                              width=rgb.shape[1], height=rgb.shape[0])
        marked = Image.fromarray(rgb)
        draw = ImageDraw.Draw(marked)
        for mark in marks:
            x, y = mark.anchor_pixel
            label = str(mark.label)
            box = draw.textbbox((x, y), label)
            draw.rectangle(box, fill="white")
            draw.text((x, y), label, fill="black")
        marked = np.asarray(marked).copy()
        # Public input never includes the privileged target or object IDs.
        neighbors = neighbor_descriptions(graph, state)
        public_neighbors = [{"index": n["index"], "direction": n["direction"]} for n in neighbors]
        text = task.instruction({task.target_type: task.target_type, task.anchor_type: task.anchor_type})
        text += "\nMoves: " + json.dumps(public_neighbors, sort_keys=True)
        text += "\nActions: move <index>; turn_left; turn_right; open <mark>; select <mark>."
        return {"rgb": rgb, "marked": marked, "masks": masks,
                "metadata": copy.deepcopy(event.metadata), "marks": [asdict(mark) for mark in marks],
                "mark_objects": marks, "input_text": text, "state": asdict(state), "world_key": world["key"],
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "marked_sha256": hashlib.sha256(marked.tobytes()).hexdigest()}

    def observe(self, world, state, task):
        graph = self.graph(world, state.opened_containers)
        self.apply_world(world, state.opened_containers)
        self.teleport(state)
        return self.capture(world, state, task, graph)

    def relation(self, world, task, apply=True):
        if apply:
            self.apply_world(world)
        snap = snapshot(self.controller.last_event, self.scene, world["key"])
        anchors = [obj for obj in snap.objects if obj.object_type == task.anchor_type]
        candidates = [obj for obj in snap.objects if obj.object_type == task.target_type]
        values = {}
        method = {"next to": "official BBoxDistance divisions=3 (sampled oriented bounding-box surface distance)",
                  "on": "official CheckWhatObjectOn divisions=3 belowDistance=0.01m (downward surface rays)",
                  "in": "official receptacle trigger membership restricted to explicitly declared enclosed types"}[task.relation]
        if len(anchors) == 1:
            anchor = anchors[0]
            for candidate in candidates:
                if candidate.object_id == anchor.object_id:
                    continue
                value = None
                if task.relation == "next to":
                    value = self.action("BBoxDistance", objectId0=candidate.object_id,
                                        objectId1=anchor.object_id, divisions=3).metadata.get("actionReturn")
                elif task.relation == "on":
                    on_id = self.action("CheckWhatObjectOn", objectId=candidate.object_id,
                        divisions=3, belowDistance=0.01).metadata.get("actionReturn")
                    value = None if on_id is None else on_id == anchor.object_id
                elif anchor.object_type in self.config["enclosed_receptacle_types"]:
                    raw = self.action("Contains", objectId=anchor.object_id, forceAction=True).metadata.get("actionReturn")
                    value = candidate.object_id in raw if isinstance(raw, list) else None
                values[(candidate.object_id, anchor.object_id)] = value
        evidence = RelationEvidence(world["key"], task.relation, method, values)
        result = resolve_target(task, snap, evidence, distance_tie_tolerance_m=self.config["distance_tie_tolerance_m"])
        return result, {"method": method,
                       "name_by_object_id": {obj.object_id: obj.name for obj in snap.objects},
                       "values": [{"candidate": c, "anchor": a, "value": v}
                                                    for (c, a), v in sorted(values.items())]}

    def physical_check(self, world, baseline, edited_name):
        self.apply_world(world)
        objects = copy.deepcopy(self.controller.last_event.metadata["objects"])
        changes = pose_changes(baseline["objects"], objects)
        edited = [c for c in changes["changed"] if c["name"] == edited_name]
        collateral = [c for c in changes["changed"] if c["name"] != edited_name]
        overlaps, errors = {}, []
        # Query every editable object, not only the intentionally edited one.
        # This uses Unity primitive overlap tests; never AABB overlap.
        for obj in sorted(objects, key=lambda item: item["name"]):
            if not (obj.get("moveable") or obj.get("pickupable")):
                continue
            try:
                raw = self.action("CollidersObjectCollidingWith", objectId=obj["objectId"]).metadata.get("actionReturn")
                if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
                    raise RuntimeError("Native collider query returned an unknown schema")
                overlaps[obj["name"]] = raw
            except RuntimeError as error:
                errors.append(str(error))
        displacement = edited[0]["position_delta_m"] if edited else 0.0
        native_clear = None if errors or not overlaps else all(not values for values in overlaps.values())
        # All collateral differences remain explicit. A single-object claim
        # fails if another object's reported pose changed, even very slightly.
        threshold = None if self.config.get("collect_only") else self.config.get("max_displacement_m")
        passed = (native_clear is True and changes["inventory_exact"] and not collateral
                  and displacement <= threshold) if threshold is not None else None
        return {"valid_physical": passed if native_clear is not None else None,
            "native_collision_clear": native_clear, "overlapping_colliders": overlaps,
            "errors": errors, "actual_edited_displacement_m": displacement,
            "realized_objects": objects,
            "pose_changes": changes, "collateral_pose_changes": collateral,
            "method": "CollidersObjectCollidingWith for every movable OR pickupable object; complete pose inventory and measured displacement",
            "scope": "Native Capsule/Box/Sphere overlap, excluding named BoundingBox, SimObjVisible/Agent layers; not a universal mesh penetration proof",
            "source": SOURCE + "UtilityFunctions.cs#L137"}

    def goal_action(self, observation, target_name):
        # DistanceToObject is the simulator's measured horizontal collider-bound
        # distance. Keep this explicit; metadata.visible is never consulted.
        import numpy as np
        target = object_by_identity(self.controller.last_event.metadata["objects"], "name", target_name)["objectId"]
        event = self.action("DistanceToObject", objectId=target)
        distance = event.metadata.get("actionReturn")
        # This upstream query uses actionFinished (not fast metadata emit).
        # If querying range advances the observed scene, do not select using
        # stale marks or pretend the new image was already the decision input.
        now_masks = {key: mask for key, mask in event.instance_masks.items() if np.any(mask)}
        same = (np.array_equal(observation["rgb"], event.frame)
                and observation["metadata"].get("agent") == event.metadata.get("agent")
                and pose_changes(observation["metadata"]["objects"], event.metadata["objects"])["all_object_poses_exact"]
                and set(observation["masks"]) == set(now_masks)
                and all(np.array_equal(mask, now_masks[key]) for key, mask in observation["masks"].items()))
        self.query_log[-1]["range_query_preserved_decision_image_agent_and_masks"] = bool(same)
        if not same:
            return None, distance
        action = selection_action(target, observation["mark_objects"], measured_distance_m=distance)
        return action, distance

    def successors(self, world, state, observation):
        graph = self.graph(world, state.opened_containers)
        result = list(navigation_transitions(graph, state, destination_heading=lambda old, _: old.heading_index))
        # Opening is verified by an actual non-forced action before admitting an
        # edge. Only one newly opened container is allowed in a pilot witness.
        if len(state.opened_containers) < self.config["max_open_containers"]:
            for mark in observation["mark_objects"]:
                if "container" not in mark.roles:
                    continue
                name = object_by_identity(observation["metadata"]["objects"], "objectId", mark.object_id)["name"]
                if name in state.opened_containers:
                    continue
                self.apply_world(world, state.opened_containers)
                self.teleport(state)
                try:
                    current_id = object_by_identity(self.controller.last_event.metadata["objects"], "name", name)["objectId"]
                    event = self.action("OpenObject", objectId=current_id, openness=1.0, forceAction=False)
                    obj = object_by_identity(event.metadata["objects"], "name", name)
                    if obj.get("isOpen") is not True:
                        continue
                    from .pair_core import Transition
                    result.append(Transition(Action("open", mark.label), ViewState(state.viewpoint_id,
                        state.heading_index, tuple(sorted((*state.opened_containers, name))))))
                except RuntimeError:
                    continue
        return result

    def plan(self, world, start, task, target_name):
        queue, seen, expanded = deque([(start, [], [start])]), {start}, 0
        while queue and expanded < self.config["max_search_states"]:
            state, actions, states = queue.popleft()
            expanded += 1
            observation = self.observe(world, state, task)
            select, distance = self.goal_action(observation, target_name)
            if select is not None:
                return {"found": True, "actions": actions + [select], "states": states,
                        "expanded": expanded, "terminal_distance_m": distance}
            if len(actions) >= 11:
                continue
            for transition in sorted(self.successors(world, state, observation), key=lambda x: x.action.text()):
                if transition.next_state not in seen:
                    seen.add(transition.next_state)
                    queue.append((transition.next_state, actions + [transition.action], states + [transition.next_state]))
        return {"found": False, "actions": [], "states": [], "expanded": expanded,
                "reason": "search_limit_or_no_executable_goal_within_12_actions"}

    def replay(self, world, start, task, target_name, plan, directory, expected_prefix=None, expected_after=None):
        if not plan["found"] or not 1 <= len(plan["actions"]) <= 12:
            return {"passed": False, "reason": "no_bounded_plan", "actions": []}
        observation = self.observe(world, start, task)
        frames = [save_observation(directory, "replay_00", observation)]
        same_prefix = (compare_inputs(expected_prefix, observation, []) ["same_input_at_t_strict"]
                       if expected_prefix is not None else None)
        state, records, first_after_exact = start, [], None
        for index, action in enumerate(plan["actions"]):
            graph_key = world["key"] + ":" + digest(list(state.opened_containers))
            if graph_key not in self.graph_cache:
                raise RuntimeError("Replay graph was not checked during planning; refusing to reinitialize a witness")
            graph = self.graph_cache[graph_key]
            # graph is already cached by planning; never reinitialize during a
            # witness. Each observation action yields one actual returned frame.
            if action.kind == "select":
                # Opening a container can physically move other objects.
                # Re-resolve in the actually replayed state, without restoring
                # poses or closing the container. Initial truth alone is not
                # evidence of the unique target after an observation action.
                current_target, current_evidence = self.relation(world, task, apply=False)
                actual, distance = self.goal_action(observation, target_name)
                selected = [mark.object_id for mark in observation["mark_objects"] if mark.label == action.argument]
                resolved_name = current_evidence["name_by_object_id"].get(current_target.target_id)
                passed = (actual == action and resolved_name == target_name
                          and index == len(plan["actions"]) - 1)
                records.append({"action": action.text(), "selected_object_id": selected[0] if len(selected) == 1 else None,
                                "target_name": target_name, "resolved_target_name": resolved_name,
                                "distance_m": distance, "distance_target_id": self.query_log[-1]["arguments"]["objectId"], "success": passed,
                                "realized_target_resolution": asdict(current_target),
                                "realized_relation_evidence": current_evidence})
                return {"passed": passed and same_prefix is not False and first_after_exact is not False,
                        "selection_success": passed, "prefix_replay_exact": same_prefix,
                        "first_observation_exact": first_after_exact, "actions": records, "frames": frames}
            if action.kind == "open":
                matches = [m for m in observation["mark_objects"] if m.label == action.argument and "container" in m.roles]
                if len(matches) != 1:
                    raise RuntimeError("Replay open label no longer refers to one visible container")
                name = object_by_identity(observation["metadata"]["objects"], "objectId", matches[0].object_id)["name"]
                self.action("OpenObject", objectId=matches[0].object_id, openness=1.0, forceAction=False)
                state = ViewState(state.viewpoint_id, state.heading_index,
                                  tuple(sorted((*state.opened_containers, name))))
            else:
                transitions = navigation_transitions(graph, state, destination_heading=lambda old, _: old.heading_index)
                following = next(t.next_state for t in transitions if t.action == action)
                if action.kind == "move":
                    # Non-forced Teleport validates the actual destination. The
                    # directed complete NavMesh path was recorded separately.
                    self.teleport(following)
                else:
                    self.action("RotateLeft" if action.kind == "turn_left" else "RotateRight", degrees=45)
                state = following
            graph_key = world["key"] + ":" + digest(list(state.opened_containers))
            if graph_key not in self.graph_cache:
                raise RuntimeError("Missing verified graph for replay container state")
            graph = self.graph_cache[graph_key]
            observation = self.capture(world, state, task, graph)
            if index == 0 and expected_after is not None:
                first_after_exact = compare_inputs(expected_after, observation, [])["same_input_at_t_strict"]
            records.append({"action": action.text(), "success": True, "state": asdict(state)})
            frames.append(save_observation(directory, f"replay_{index+1:02d}", observation))
        return {"passed": False, "reason": "missing_terminal_select", "actions": records, "frames": frames}
