"""Versioned T01 native placement/settling checks for the approved criteria.

Old workers retain their original adapter. Native random placement is only a
proposal: its success flag alone is insufficient (the pinned implementation may
return success when no object moved). Record actual pose, receptacle membership,
bounded physical settling and all collateral movements before accepting.
"""
import copy

from .pair_simulator import PairSimulator as OriginalSimulator, SOURCE, object_by_identity
from .pair_approved_policy import (POLICY, indexed_objects, measure_settled_poses,
                                   position_distance, rotation_delta_degrees)


SETTLING = {
    "time_step_seconds": 0.02, "seconds_per_observation": 0.1,
    "maximum_observations": 50, "consecutive_stable_observations": 5,
    "position_stability_max_m": 0.00001,
    "rotation_component_stability_max_degrees": 0.0001,
    "requires_native_isSceneAtRest": True,
    "purpose": "bounded conservative settling detection; not a relaxation of 2cm/1cm or pixel equality",
}


class PairSimulator(OriginalSimulator):
    def apply_world(self, world, opened=()):
        self.action("PausePhysicsAutoSim")
        event = super().apply_world(world, opened)
        self.action("PausePhysicsAutoSim")
        return self.controller.last_event

    def settle_world(self, world, key):
        """Release the COMPLETE editable inventory and measure real physics."""
        self.apply_world(world)
        self.action("SetObjectPoses", objectPoses=world["poses"], placeStationary=False)
        self.action("PausePhysicsAutoSim")
        previous = indexed_objects(self.controller.last_event.metadata["objects"])
        trace, consecutive = [], 0
        for index in range(SETTLING["maximum_observations"]):
            # Explicit pause also covers upstream automatic resumption at rest.
            self.action("PausePhysicsAutoSim")
            event = self.action("AdvancePhysicsStep", timeStep=SETTLING["time_step_seconds"],
                                simSeconds=SETTLING["seconds_per_observation"], allowAutoSimulation=False)
            current = indexed_objects(event.metadata["objects"])
            inventory = set(current) == set(previous)
            deltas = {name: {
                "position_m": position_distance(previous[name]["position"], current[name]["position"]),
                "rotation_components_degrees": rotation_delta_degrees(previous[name]["rotation"], current[name]["rotation"]),
                "isMoving": current[name].get("isMoving"),
            } for name in sorted(set(previous) & set(current))}
            native_rest = event.metadata.get("isSceneAtRest") is True
            stable = inventory and native_rest and all(
                row["position_m"] <= SETTLING["position_stability_max_m"]
                and max(row["rotation_components_degrees"].values()) <= SETTLING["rotation_component_stability_max_degrees"]
                and row["isMoving"] is not True for row in deltas.values())
            consecutive = consecutive + 1 if stable else 0
            trace.append({"index": index, "native_isSceneAtRest": event.metadata.get("isSceneAtRest"),
                          "inventory_exact": inventory, "objects": deltas,
                          "stable": stable, "consecutive_stable": consecutive})
            previous = current
            if consecutive >= SETTLING["consecutive_stable_observations"]:
                break
        realized = self.world_from_event(key)
        self.action("PausePhysicsAutoSim")
        return realized, {"settled": consecutive >= SETTLING["consecutive_stable_observations"],
                          "policy": SETTLING, "observations": trace,
                          "complete_requested_poses": copy.deepcopy(world["poses"]),
                          "source": SOURCE + "PhysicsSceneManager.cs#L334"}

    def native_collisions(self):
        overlaps, errors = {}, []
        objects = copy.deepcopy(self.controller.last_event.metadata["objects"])
        for obj in sorted(objects, key=lambda item: item["name"]):
            if not (obj.get("moveable") or obj.get("pickupable")):
                continue
            try:
                raw = self.action("CollidersObjectCollidingWith", objectId=obj["objectId"]).metadata.get("actionReturn")
                if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
                    raise RuntimeError("Unknown native collision response")
                overlaps[obj["name"]] = raw
            except RuntimeError as error:
                errors.append(str(error))
        return {"native_collision_clear": bool(overlaps) and not errors and all(not values for values in overlaps.values()),
                "overlapping_colliders": overlaps, "errors": errors,
                "scope": "Native Capsule/Box/Sphere overlap query; no AABB or unreported-mesh nonpenetration claim",
                "source": SOURCE + "UtilityFunctions.cs#L137"}

    def placement_proposal(self, baseline, task, rng):
        self.apply_world(baseline)
        objects = self.controller.last_event.metadata["objects"]
        editable = sorted((o for o in objects if (o.get("pickupable") or o.get("moveable"))
                           and o["objectType"] in (task.target_type, task.anchor_type)), key=lambda o: o["name"])
        if not editable:
            raise RuntimeError("No editable candidate or anchor")
        obj = rng.choice(editable)
        receptacles = sorted((o for o in objects if o.get("receptacle") and o["name"] != obj["name"]
                              and o["objectType"] != "Floor"), key=lambda o: o["name"])
        if not receptacles:
            raise RuntimeError("No nonfloor destination receptacle")
        destination = rng.choice(receptacles)
        seed = rng.randrange(2**31)
        event = self.action("InitialRandomSpawn", randomSeed=seed, forceVisible=False,
            placeStationary=True, numPlacementAttempts=5, objectIds=[obj["objectId"]],
            receptacleObjectIds=[destination["objectId"]], allowFloor=False, allowMoveable=True)
        proposed = object_by_identity(event.metadata["objects"], "name", obj["name"])
        destination_now = object_by_identity(event.metadata["objects"], "name", destination["name"])
        # Restrict the requested edit to this one measured native placement even
        # if upstream changed some other field as a side effect. The complete
        # baseline pose list is copied; all other requested poses are unchanged.
        request = copy.deepcopy(baseline)
        requested_pose = next(p for p in request["poses"] if p["objectName"] == obj["name"])
        requested_pose.update(position=copy.deepcopy(proposed["position"]), rotation=copy.deepcopy(proposed["rotation"]))
        request["key"] = baseline["key"] + "_requested_edit"
        contains = self.action("Contains", objectId=destination_now["objectId"], forceAction=True).metadata.get("actionReturn")
        changed = proposed["position"] != obj["position"] or proposed["rotation"] != obj["rotation"]
        member = isinstance(contains, list) and proposed["objectId"] in contains
        edit = {"object_name": obj["name"], "object_id": obj["objectId"],
                "requested_pose": copy.deepcopy(requested_pose),
                "requested_displacement_m": position_distance(obj["position"], proposed["position"]),
                "kind": "native_single_object_receptacle_placement",
                "destination_receptacle_name": destination["name"],
                "destination_receptacle_type": destination["objectType"],
                "proposal_seed": seed, "proposal_pose_changed": changed,
                "proposal_contains_return": contains, "proposal_receptacle_member": member,
                "native_proposal_verified": bool(changed and member),
                "native_placement_source": SOURCE + "InstantiatePrefabTest.cs#L137",
                "native_success_flag_is_not_placement_proof": True}
        return request, edit

    def edited_physical_check(self, baseline, request, edit):
        realized, settling = self.settle_world(request, baseline["key"] + "_w1_settled")
        measurements = measure_settled_poses(baseline["objects"], edit["requested_pose"],
                                             realized["objects"], edit["object_name"])
        obj = object_by_identity(realized["objects"], "name", edit["object_name"])
        dest = object_by_identity(realized["objects"], "name", edit["destination_receptacle_name"])
        contains = self.action("Contains", objectId=dest["objectId"], forceAction=True).metadata.get("actionReturn")
        member = isinstance(contains, list) and obj["objectId"] in contains
        collisions = self.native_collisions()
        check = {**collisions, **measurements, "settling": settling,
                 "destination_contains_after_settling": contains, "destination_member_after_settling": member,
                 "legal_receptacle_placement": edit["native_proposal_verified"] and member,
                 "acceptance_policy": POLICY, "realized_objects": realized["objects"]}
        check["valid_physical"] = (settling["settled"] and measurements["position_bounds_passed"]
            and check["legal_receptacle_placement"] and collisions["native_collision_clear"])
        return realized, check
