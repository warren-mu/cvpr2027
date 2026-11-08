"""Opt-in settled-baseline preflight; frozen earlier simulators are unchanged."""
import copy
import hashlib
import json
import math
from pathlib import Path

from .pair_v2_clone import V2CloneSimulator
from .pair_v2_simulator import objects_by_name

PREFLIGHT_VERSION = "settled_baseline_window_v1"
PREFLIGHT_POLICY = {"initial_max_steps": 50, "small_step_streak_required": 10,
    "small_step_exclusive_m": 0.001, "initial_displacement_max_m": 0.5,
    "stability_window_steps": 50, "window_displacement_max_m": 0.01,
    "object_scope": "all objects reported in metadata, matched by stable name",
    "initial_displacement": "Euclidean initial-to-current displacement; path length is diagnostic only"}


def positions(objects):
    indexed = objects_by_name(objects)
    if not indexed:
        raise ValueError("Preflight requires a nonempty reported object inventory")
    result = {}
    for name, obj in indexed.items():
        point = obj["position"]
        if any(type(point.get(axis)) not in (int, float) or not math.isfinite(point[axis]) for axis in "xyz"):
            raise ValueError("Nonfinite or missing object position: " + name)
        result[name] = copy.deepcopy(point)
    return result


def position_step(previous, current, baseline, path_lengths=None):
    """Measure every reported object; rotations do not introduce a new gate."""
    old, new, origin = positions(previous), positions(current), positions(baseline)
    path_lengths = path_lengths or {}
    rows = {}
    for name in sorted(set(old) & set(new) & set(origin)):
        step = math.dist([old[name][a] for a in "xyz"], [new[name][a] for a in "xyz"])
        delta = math.dist([origin[name][a] for a in "xyz"], [new[name][a] for a in "xyz"])
        rows[name] = {"previous_position": old[name], "position": new[name], "baseline_position": origin[name],
            "step_displacement_m": step, "baseline_displacement_m": delta,
            "observed_path_length_m": path_lengths.get(name, 0.0) + step}
    exact = set(old) == set(new) == set(origin)
    return {"inventory_exact": exact, "missing_names": sorted(set(origin) - set(new)),
            "added_names": sorted(set(new) - set(origin)), "objects": rows,
            "all_steps_below_1mm": exact and all(row["step_displacement_m"] < .001 for row in rows.values())}


def exceeded(measured, limit):
    return [name for name, row in measured["objects"].items() if row["baseline_displacement_m"] > limit]


def common_baseline_check(baseline_objects, realized_objects, edited_names, reference):
    """Both worlds' non-edited objects use one settled/common-clone baseline."""
    measured = position_step(baseline_objects, realized_objects, baseline_objects)
    edited = set(edited_names)
    if not edited or not edited <= set(positions(baseline_objects)):
        raise ValueError("Missing edited identity in the common baseline")
    violations = [name for name in exceeded(measured, .01) if name not in edited]
    reasons = (["common_baseline_inventory_changed"] if not measured["inventory_exact"] else [])
    reasons += ["common_baseline_other_object_position_change_above_0_01m:" + name for name in violations]
    return {"passed": not reasons, "baseline_reference": copy.deepcopy(reference), "edited_names": sorted(edited),
            "measurements": measured, "violations_above_1cm": violations, "reasons": reasons,
            "scope": "all non-edited metadata objects relative to the same settled/clone common baseline in both worlds"}


def augment_physical_result(result, physical):
    """New entry may apply this to its unchanged first-round measurement helper."""
    result = copy.deepcopy(result)
    check = physical.get("common_baseline_check")
    if not isinstance(check, dict):
        raise ValueError("Stable construction lacks its common-baseline position check")
    result["common_baseline_check"] = copy.deepcopy(check)
    result["valid_physical"] = bool(result.get("valid_physical") and check.get("passed") is True)
    result["reasons"] = list(dict.fromkeys([*(result.get("reasons") or []), *check["reasons"]]))
    return result


class StableCloneSimulator(V2CloneSimulator):
    """Composable after clone-shard PreparationSimulator in the new entry's MRO."""
    def preflight_scene(self):
        if self.action_count or self.config.get("preflight_version") != PREFLIGHT_VERSION:
            raise RuntimeError("Versioned stable preflight must be the first backend operation")
        if (self.max_steps != PREFLIGHT_POLICY["initial_max_steps"]
                or self.config.get("preflight_policy") != PREFLIGHT_POLICY
                or (self.time_step, self.step_seconds) != (.02, .1)):
            raise ValueError("Require the explicit approved N/M policy and unchanged physics step durations")
        self._stable_common = None
        record = {"preflight_version": PREFLIGHT_VERSION, "policy": copy.deepcopy(PREFLIGHT_POLICY),
            "passed": False, "snapshot": None, "window_end_snapshot": None, "reasons": [], "error": None,
            "physics": {"time_step_seconds": self.time_step, "seconds_per_step": self.step_seconds},
            "diagnostics": {"source_snapshot": None, "pause_action_reference": None, "pause_measurements": None,
                "initial_settling": {"settled": False, "criterion": None, "observations": []},
                "stability_window": {"completed": False, "observations": []}}}
        initial = record["diagnostics"]["initial_settling"]
        window = record["diagnostics"]["stability_window"]
        try:
            source = self.snapshot("original_scene")
            record["diagnostics"]["source_snapshot"] = source
            event = self.action("PausePhysicsAutoSim")
            measured = position_step(source["objects"], event.metadata["objects"], source["objects"])
            record["diagnostics"].update(pause_action_reference=copy.deepcopy(self.last_action_reference),
                                         pause_measurements=measured)
            previous = copy.deepcopy(event.metadata["objects"])
            travelled = {name: row["observed_path_length_m"] for name, row in measured["objects"].items()}
            if not measured["inventory_exact"] or exceeded(measured, .5):
                record["reasons"].append("initial_inventory_changed_or_displacement_above_0_5m")
                return copy.deepcopy(record)
            streak = 0
            for index in range(50):
                event = self.action("AdvancePhysicsStep", timeStep=self.time_step,
                                    simSeconds=self.step_seconds, allowAutoSimulation=False)
                measured = position_step(previous, event.metadata["objects"], source["objects"], travelled)
                streak = streak + 1 if measured["all_steps_below_1mm"] else 0
                row = {"index": index, "native_is_scene_at_rest": event.metadata.get("isSceneAtRest"),
                    "small_step_streak": streak, "measurements": measured,
                    "action_reference": copy.deepcopy(self.last_action_reference)}
                initial["observations"].append(row)
                previous = copy.deepcopy(event.metadata["objects"])
                travelled = {name: item["observed_path_length_m"] for name, item in measured["objects"].items()}
                if not measured["inventory_exact"] or exceeded(measured, .5):
                    record["reasons"].append("initial_inventory_changed_or_displacement_above_0_5m")
                    break
                if row["native_is_scene_at_rest"] is True or streak >= 10:
                    initial.update(settled=True, criterion="native_rest" if row["native_is_scene_at_rest"] is True else "ten_small_steps")
                    record["snapshot"] = self.snapshot("preflight_settled_baseline")
                    break
            if not initial["settled"]:
                if not record["reasons"]:
                    record["reasons"].append("initial_not_settled_within_50_steps")
                return copy.deepcopy(record)
            baseline = record["snapshot"]["objects"]
            previous, travelled = copy.deepcopy(baseline), {}
            for index in range(50):
                event = self.action("AdvancePhysicsStep", timeStep=self.time_step,
                                    simSeconds=self.step_seconds, allowAutoSimulation=False)
                measured = position_step(previous, event.metadata["objects"], baseline, travelled)
                window["observations"].append({"index": index, "native_is_scene_at_rest": event.metadata.get("isSceneAtRest"),
                    "measurements": measured, "action_reference": copy.deepcopy(self.last_action_reference)})
                previous = copy.deepcopy(event.metadata["objects"])
                travelled = {name: item["observed_path_length_m"] for name, item in measured["objects"].items()}
            window["completed"] = True
            record["window_end_snapshot"] = self.snapshot("preflight_window_end")
            if any(not row["measurements"]["inventory_exact"] or exceeded(row["measurements"], .01)
                   for row in window["observations"]):
                record["reasons"].append("stability_window_inventory_changed_or_displacement_above_0_01m")
            record["passed"] = not record["reasons"]
            return copy.deepcopy(record)
        except BaseException as error:
            record["error"] = f"{type(error).__name__}: {error}"
            record["reasons"].append("preflight_interrupted_or_action_failed")
            raise
        finally:
            self.scene_preflight = copy.deepcopy(record)
            self.store.save_json("scene_preflight.json", record)

    def construct_world(self, baseline, key, placements, require_agent=None):
        context = self.clone_context
        if context is not None:
            if context.get("setup_passed") is not True:
                raise RuntimeError("Clone preparation has not passed")
            reference = context["baseline_validation_reference"]
            data = Path(reference["file"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != reference["sha256"]:
                raise ValueError("Common clone baseline reference changed")
            common = json.loads(data)["construction_physical"]["realized_objects"]
        else:
            common = self.scene_preflight["snapshot"]["objects"]
            reference = self.scene_preflight["snapshot"]["reference"]
        cached = getattr(self, "_stable_common", None)
        if cached is not None and cached != (reference, common):
            raise ValueError("Common baseline changed within one scene controller")
        self._stable_common = copy.deepcopy((reference, common))
        result = super().construct_world(baseline, key, placements, require_agent=require_agent)
        check = common_baseline_check(common, result["snapshot"]["objects"], result["physical"]["edited_names"], reference)
        result["physical"]["common_baseline_check"] = check
        result["physical"]["common_baseline_check_reference"] = self.store.save_json(
            f"common_baseline_check_{self.action_count:06d}.json", check)
        result["passed"] = bool(result["passed"] and check["passed"])
        result["physical"]["valid_physical"] = result["passed"]
        self.store.save_json(f"stable_construction_{self.action_count:06d}_outcome.json",
                             {k: v for k, v in result.items() if k != "capture"})
        return result
