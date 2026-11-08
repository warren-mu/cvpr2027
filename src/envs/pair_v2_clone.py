"""Opt-in T01v2 common-baseline cloning; first-round files stay unchanged.

Use V2CloneSimulator only from a separately committed second-round entry point.
Call preflight_scene first, then prepare_clone_baseline once. A failed creation
or settling attempt requires a new Controller; this module never deletes a clone
or silently disables old objects to retry. Subsequent w0/w1 changes use the
ordinary complete unique-name list and only move the distinct anchor X.
"""
import copy
import hashlib
from pathlib import Path

from . import pair_v2_simulator as engine
from .pair_v2_simulator import (V2Simulator as _BaseSimulator, complete_pose_list,
                               distance, encoded, objects_by_name, validate_spawn)
from .pair_v2_clone_policy import (make_clone_recipe, verify_clone_creation,
    clone_baseline_check, clone_pair_check, clone_replay_check)


def _reference_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _current_original_inventory(sim, original):
    before = objects_by_name(original)
    actual = objects_by_name(sim.controller.last_event.metadata["objects"])
    if set(before) != set(actual):
        raise ValueError("Clone setup requires the complete original-name inventory")
    for name, obj in before.items():
        if any(obj.get(key) != actual[name].get(key) for key in
               ("objectType", "pickupable", "moveable", "receptacle")):
            raise ValueError("Original object roles differ before clone setup")


def prepare_clone_baseline(sim, baseline, source_name, spawn, key="clone_baseline"):
    """Create one actual copy, then legally place/settle it BEFORE the world pair.

    The extra entry is appended after all original movable/pickupable entries.
    Its temporary creation coordinate is the native sample, not a settled pivot;
    physics remains paused. The new object's own PlaceObjectAtPoint proposal then
    fixes the requested pivot before a complete unique-name SetObjectPoses and
    explicit settling. No clone creation or cleanup is hidden in w0 -> w1.
    """
    if not isinstance(sim, V2CloneSimulator):
        raise TypeError("Use the explicitly enabled V2CloneSimulator for clone preparation")
    if not sim.scene_preflight or sim.scene_preflight.get("passed") is not True:
        raise RuntimeError("Original-scene settling preflight must pass before cloning")
    if sim.clone_context is not None or sim.clone_preparation_attempted:
        raise RuntimeError("Clone setup may run once per Controller; retain failure and start a new one")
    if baseline["scene"] != sim.scene:
        raise ValueError("Common baseline scene mismatch")
    original = copy.deepcopy(baseline["objects"])
    _current_original_inventory(sim, original)
    complete_pose_list(original, baseline["poses"])
    token = validate_spawn({"spawn": spawn}, sim.scene)
    if hashlib.sha256(encoded(token)).hexdigest() not in sim.spawn_tokens:
        raise ValueError("Clone placement coordinate must come from this Controller's native query")
    indexed = objects_by_name(original)
    destination = indexed.get(token["receptacle_name"])
    if destination is None or destination.get("receptacle") is not True or destination["name"] == source_name:
        raise ValueError("Clone requires a distinct actual destination receptacle")
    # Position-generated native IDs round to centimetres. This is only a
    # proposal prefilter; actual full IDs/names are independently checked below.
    if distance(indexed[source_name]["position"], token["selected_position"]) <= 0.02:
        raise ValueError("Clone creation proposal is too close to its source for distinct position IDs")
    recipe = make_clone_recipe(original, source_name, token["selected_position"])
    sim.clone_preparation_attempted = True
    record = {"key": key, "scene": sim.scene, "passed": False,
              "original_snapshot_reference": baseline.get("reference"),
              "recipe": recipe, "spawn_evidence": token, "creation": None,
              "clone_baseline_check": None, "error": None,
              "world_pair_edit": False, "requires_new_controller_after_failure": True}
    try:
        sim.action("PausePhysicsAutoSim")
        if sim.controller.last_event.metadata.get("inventoryObjects") != []:
            raise RuntimeError("Clone construction requires empty hands")
        # The sole deliberately duplicated pose list: complete original list
        # plus one source copy. No physics is advanced at this intermediate pose.
        event = sim.action("SetObjectPoses", objectPoses=copy.deepcopy(recipe["objectPoses"]), placeStationary=False)
        created = copy.deepcopy(event.metadata["objects"])
        action_success = event.metadata.get("lastActionSuccess")
        creation_reference = copy.deepcopy(sim.last_action_reference)
        lineage = verify_clone_creation(original, created, recipe, action_success=action_success)
        record["creation"] = {"lineage": lineage, "created_objects": created,
                              "action_success": action_success, "action_reference": creation_reference}
        if not lineage["passed"]:
            raise RuntimeError("Native clone identity creation failed: " + repr(lineage["reasons"]))
        clone_name = lineage["clone_name"]
        sim.clone_context = {"schema_version": 1, "enabled": True, "setup_passed": False,
            "original_objects": original, "recipe": recipe, "lineage": lineage,
            "creation_action_reference": creation_reference,
            "source_baseline_reference": baseline.get("reference"),
            "scope": "one clone shared by both worlds before counterfactual anchor-only editing"}
        created_snapshot = sim.snapshot(key + "_created_unvalidated")
        # Call the frozen base construction deliberately: the subclass prevents
        # ordinary pair construction while clone setup remains unvalidated.
        placed = _BaseSimulator.construct_world(sim, created_snapshot, key + "_placement",
            [{"object_name": clone_name, "spawn": token}], require_agent=created_snapshot["agent"])
        physical = placed["physical"]
        requested_pose = next(p for p in physical["requested_poses"] if p["objectName"] == clone_name)
        check = clone_baseline_check(original, created, placed["snapshot"]["objects"], recipe,
            creation_action_success=action_success, requested_clone_pose=requested_pose,
            placement_evidence=physical["placement_evidence_by_name"][clone_name],
            native_is_scene_at_rest=physical["native_is_scene_at_rest"])
        record.update(clone_baseline_check=check, construction_physical=physical,
                      passed=placed["passed"] is True and check["passed"] is True)
        if not record["passed"]:
            return {"passed": False, "record": record, "snapshot": placed["snapshot"],
                    "capture": placed["capture"], "requires_new_controller": True}
        validation_ref = sim.store.save_json(f"clone_baseline_validation_{sim.action_count:06d}.json", record)
        sim.clone_context.update(setup_passed=True, baseline_validation_reference=validation_ref)
        validated = sim.snapshot(key + "_validated")
        return {"passed": True, "snapshot": validated, "capture": placed["capture"],
                "lineage": copy.deepcopy(lineage), "record": record,
                "record_reference": validation_ref, "requires_new_controller": False}
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        sim.store.save_json(f"clone_baseline_attempt_{sim.action_count:06d}.json", record)


def restore_clone_inventory(sim, snapshot):
    """Recreate the saved name inventory, without treating creation as validity."""
    context = snapshot.get("clone_context")
    if not isinstance(context, dict) or context.get("setup_passed") is not True:
        raise ValueError("Snapshot lacks a validated common clone baseline")
    lineage, recipe = context["lineage"], context["recipe"]
    current = sim.controller.last_event.metadata["objects"]
    names = set(objects_by_name(current))
    if names == set(lineage["baseline_names"]):
        # Reusing an already recreated inventory is safe only with exactly the
        # same saved recipe/lineage, never an unrelated coincidentally named copy.
        if (sim.clone_context is None or sim.clone_context.get("recipe") != recipe
                or sim.clone_context.get("lineage") != lineage):
            raise ValueError("Existing clone inventory has no matching verified lineage")
        return {"passed": True, "already_recreated": True, "lineage": copy.deepcopy(lineage)}
    _current_original_inventory(sim, context["original_objects"])
    if sim.clone_preparation_attempted:
        raise RuntimeError("Cannot repair an incomplete clone recreation within the same Controller")
    sim.clone_preparation_attempted = True
    sim.action("PausePhysicsAutoSim")
    event = sim.action("SetObjectPoses", objectPoses=copy.deepcopy(recipe["objectPoses"]), placeStationary=False)
    measured = clone_replay_check(context["original_objects"], copy.deepcopy(event.metadata["objects"]),
        recipe, lineage, action_success=event.metadata.get("lastActionSuccess"))
    measured["action_reference"] = copy.deepcopy(sim.last_action_reference)
    reference = sim.store.save_json(f"clone_recreation_{sim.action_count:06d}.json", measured)
    if not measured["passed"]:
        raise RuntimeError("Fresh process clone lineage mismatch: " + repr(measured["reasons"]))
    sim.clone_context = copy.deepcopy(context)
    sim.clone_recreation_reference = reference
    return measured


class V2CloneSimulator(_BaseSimulator):
    """Opt-in extension; the first-round V2Simulator implementation is untouched."""
    def __init__(self, *args, **kwargs):
        self.clone_context = None
        self.clone_preparation_attempted = False
        self.clone_recreation_reference = None
        super().__init__(*args, **kwargs)

    def snapshot(self, key):
        snapshot = super().snapshot(key)
        if self.clone_context is None:
            return snapshot
        old_reference = snapshot.pop("reference")
        snapshot.update(clone_context=copy.deepcopy(self.clone_context),
                        unextended_snapshot_reference=old_reference,
                        clone_recreation_reference=copy.deepcopy(self.clone_recreation_reference))
        # Preserve the old snapshot; add a separately hashed complete version
        # whose JSON includes the lineage required by a fresh Unity process.
        reference = self.store.save_json(f"clone_snapshot_{self.snapshot_count:06d}.json", snapshot)
        self.snapshot_count += 1
        snapshot["reference"] = reference
        return snapshot

    def construct_world(self, baseline, key, placements, require_agent=None):
        if ((self.clone_preparation_attempted or self.clone_context is not None)
                and (self.clone_context or {}).get("setup_passed") is not True):
            raise RuntimeError("Clone common baseline has not passed physical validation")
        if self.clone_context is not None:
            clones = set(self.clone_context["lineage"]["candidate_names"])
            if any(p["object_name"] in clones for p in placements):
                raise ValueError("Both cloned candidates are fixed in the common baseline; only a distinct X may move")
        return super().construct_world(baseline, key, placements, require_agent=require_agent)

    def restore_snapshot(self, snapshot):
        if snapshot.get("clone_context") is not None:
            recreation = restore_clone_inventory(self, snapshot)
        elif self.clone_context is not None:
            raise ValueError("Cannot silently remove the prepared clone to restore an original scene")
        else:
            recreation = None
        capture, measurement = super().restore_snapshot(snapshot)
        measurement["clone_recreation"] = recreation
        return capture, measurement


def clone_replay_worker(config, connection):
    """Private-worker adapter; only an explicit new entry point enables it.

    The frozen replay implementation is reused unchanged. Its simulator binding
    is replaced only for this invocation in its private spawned worker process,
    then restored; no installed/source file or other running job is changed.
    """
    if config.get("enable_candidate_clone") is not True:
        raise ValueError("Clone replay requires explicit second-round opt-in")
    if config.get("snapshot", {}).get("clone_context", {}).get("setup_passed") is not True:
        raise ValueError("Clone replay requires an authenticated validated clone snapshot")
    previous = engine.V2Simulator
    try:
        engine.V2Simulator = V2CloneSimulator
        result = engine.replay_worker(config, connection)
    finally:
        engine.V2Simulator = previous
    result["clone_backend"] = {"source_file": __file__, "source_sha256": _reference_digest(__file__),
                               "opt_in": True, "original_backend_file_unmodified": True}
    return result
