#!/usr/bin/env python3
"""T00b D/E/F/G isolation diagnostics, separately wrapped by run_logged.

Completion means observations were collected, never that exact RGB equality
passed. Imports do not initialize a simulator or GPU. No Unity modifications.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

try:
    from .thor_common import (ROOT, Worker, controller_session, deadline_after,
                              install_signals, prepare_runtime, step, stop_workers, write_result)
    from .thor_checks import agent_pose, full_poses
    from .thor_restore_diagnostics import (capture, compare_images, compare_snapshots,
                                           save_evidence, save_partial, snapshot)
except ImportError:
    from thor_common import (ROOT, Worker, controller_session, deadline_after,
                             install_signals, prepare_runtime, step, stop_workers, write_result)
    from thor_checks import agent_pose, full_poses
    from thor_restore_diagnostics import (capture, compare_images, compare_snapshots,
                                          save_evidence, save_partial, snapshot)

NAMES = {mode: "t00b_" + mode for mode in
         ("no_reset_identity", "no_reset_edit_undo", "no_reset_revisit", "cross_process")}
BUILD = "f0825767cd50d69f666c7f282e54abfe58f1e917"


def artifact(path):
    return {"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save_capture(directory, label, item):
    """Persist a capture before the next action; payload enables cross-process comparison."""
    import numpy as np
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    ids = sorted(item["masks"])
    shape = item["frame"].shape[:2]
    arrays = directory / f"{label}_payload.npz"
    np.savez_compressed(arrays, frame=item["frame"],
                        segmentation=item["segmentation"] if item["segmentation"] is not None else np.empty((0,), dtype=np.uint8),
                        object_ids=np.asarray(ids, dtype=str),
                        masks=np.stack([item["masks"][key] for key in ids]) if ids else np.empty((0, *shape), dtype=bool))
    metadata = directory / f"{label}_metadata.json"
    metadata.write_text(json.dumps({"snapshot": item["snapshot"], "captured_monotonic": item["captured_monotonic"]},
                                   indent=2, allow_nan=False) + "\n")
    rgb = directory / f"{label}_rgb.png"
    Image.fromarray(item["frame"]).save(rgb)
    return {"arrays": artifact(arrays), "metadata": artifact(metadata), "rgb": artifact(rgb)}


def load_capture(saved):
    import numpy as np
    for record in saved.values():
        if artifact(ROOT / record["file"])["sha256"] != record["sha256"]:
            raise RuntimeError("Capture artifact hash changed before comparison")
    metadata = json.loads((ROOT / saved["metadata"]["file"]).read_text())
    with np.load(ROOT / saved["arrays"]["file"], allow_pickle=False) as arrays:
        segmentation = arrays["segmentation"]
        return {"frame": arrays["frame"].copy(),
                "segmentation": segmentation.copy() if segmentation.size else None,
                "masks": {str(key): mask.copy() for key, mask in zip(arrays["object_ids"], arrays["masks"])},
                **metadata}


def refresh(metrics, final=False):
    cases = metrics["cases"]
    compared = [case for case in cases if "rgb" in case]
    completed = [case for case in cases if case["status"] == "completed"]
    metrics.update(cases_attempted=len(cases), cases_completed=len(completed),
                   rgb_comparisons_completed=len(compared),
                   exact_zero_failed_cases=sum(not case["rgb"]["rgb_exact_zero"] for case in compared),
                   max_pixel_difference=max((case["rgb"]["max_pixel_difference"] for case in compared), default=None),
                   exact_zero_passed=all(case["rgb"]["rgb_exact_zero"] for case in compared) if len(compared) == 20 else None)
    if final:
        metrics["diagnostic_completed"] = len(completed) == 20
        metrics["status"] = "completed" if metrics["diagnostic_completed"] else "incomplete"


def action_plan(mode, poses, agent, destination=None):
    """No reset/physics-policy changes between the before and after captures."""
    if mode == "no_reset_identity":
        return [("identity", "SetObjectPoses", {"objectPoses": copy.deepcopy(poses)}),
                ("after", "TeleportFull", copy.deepcopy(agent))], {}
    if mode == "no_reset_edit_undo":
        if not poses:
            raise RuntimeError("No movable or pickupable object for the fixed edit")
        edited = copy.deepcopy(poses)
        chosen = min(edited, key=lambda pose: pose["objectName"])
        original = copy.deepcopy(chosen)
        chosen["position"]["x"] += 0.5
        return [("edited", "SetObjectPoses", {"objectPoses": edited}),
                ("after", "SetObjectPoses", {"objectPoses": copy.deepcopy(poses)})], {
                    "edited_object_name": chosen["objectName"], "requested_displacement_m": {"x": 0.5, "y": 0.0, "z": 0.0},
                    "original_object_pose": original, "requested_object_pose": copy.deepcopy(chosen),
                    "physical_validity": None,
                    "note": "Fixed +x edit, no collision-free claim and no selection by RGB outcome."}
    if mode == "no_reset_revisit":
        if destination is None:
            raise ValueError("Revisit requires a verified navigation destination")
        away = copy.deepcopy(agent)
        away["position"] = copy.deepcopy(destination)
        return [("away", "TeleportFull", away), ("after", "TeleportFull", copy.deepcopy(agent))], {}
    raise ValueError("Unknown local mode")


def xz_distance(first, second):
    values = [point[axis] for point in (first, second) for axis in ("x", "z")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("Path endpoints must have finite X/Z coordinates")
    return math.hypot(first["x"] - second["x"], first["z"] - second["z"])


def path_endpoints_match(returned, start, target, allowed_error):
    """Action success establishes PathComplete in the pinned Unity implementation.

    Additionally check both endpoints, because upstream's allowedError rejection
    combines the endpoint errors with AND. No RGB tolerance is involved.
    """
    if not isinstance(returned, dict) or not isinstance(returned.get("corners"), list):
        return False
    if "status" in returned and (isinstance(returned["status"], bool) or returned["status"] not in (0, "PathComplete")):
        return False
    corners = returned["corners"]
    if len(corners) < 2:
        return False
    return xz_distance(corners[0], start) <= allowed_error and xz_distance(corners[-1], target) <= allowed_error


def navigation_destination(controller, reachable, agent, raw, allowed_error):
    start = agent["position"]
    candidates = []
    unique = {tuple(point[axis] for axis in ("x", "y", "z")): point for point in reachable}
    for point in unique.values():
        distance = math.dist([point[axis] for axis in ("x", "y", "z")],
                             [start[axis] for axis in ("x", "y", "z")])
        if 1.0 <= distance <= 1.5:
            candidates.append((distance, point["x"], point["y"], point["z"], point))
    raw["navigation"] = {"kind": "two-node diagnostic local navigation graph; not a T01 full-scene graph",
                         "source": copy.deepcopy(start), "reachable_positions": copy.deepcopy(reachable),
                         "minimum_viewpoint_spacing_m": 1.0, "maximum_edge_distance_m": 1.5,
                         "path_allowed_error_m": allowed_error, "attempts": [],
                         "path_complete_source": f"https://github.com/allenai/ai2thor/blob/{BUILD}/unity/Assets/Scripts/BaseFPSAgentController.cs",
                         "path_complete_source_function": "SafelyComputeNavMeshPath; GetShortestPathToPoint"}
    for _, _, _, _, target in sorted(candidates):
        accepted = True
        for direction, origin, destination in (("outward", start, target), ("return", target, start)):
            # Match the actual pinned Unity build, whose Vector3 target differs
            # from the older x/y/z form still present in Python util.metrics.
            event = controller.step(action="GetShortestPathToPoint", position=origin,
                                    target=destination, allowedError=allowed_error)
            attempt = {"direction": direction, "source": copy.deepcopy(origin), "target": copy.deepcopy(destination),
                       "lastActionSuccess": event.metadata.get("lastActionSuccess"),
                       "errorCode": event.metadata.get("errorCode"),
                       "errorMessage": event.metadata.get("errorMessage"),
                       "actionReturn": copy.deepcopy(event.metadata.get("actionReturn")), "snapshot": snapshot(event)}
            raw["navigation"]["attempts"].append(attempt)
            accepted = attempt["lastActionSuccess"] is True and path_endpoints_match(attempt["actionReturn"], origin, destination, allowed_error)
            attempt["accepted_complete_path_with_matching_endpoints"] = accepted
            if not accepted:
                break
        if accepted:
            raw["navigation"]["destination"] = copy.deepcopy(target)
            return copy.deepcopy(target)
    raise RuntimeError("No verified complete path to a reachable diagnostic viewpoint 1.0–1.5m away")


def source_state(controller, number, rng):
    controller.reset(scene="FloorPlan1")  # permitted only before the case comparison interval
    step(controller, "InitialRandomSpawn", randomSeed=number, forceVisible=False,
         placeStationary=True, numPlacementAttempts=5)
    reachable = step(controller, "GetReachablePositions").metadata["actionReturn"]
    if not reachable:
        raise RuntimeError("Randomized state has no reachable positions")
    event = step(controller, "TeleportFull", position=rng.choice(reachable),
                 rotation={"x": 0, "y": rng.choice((0, 90, 180, 270)), "z": 0},
                 horizon=rng.choice((0, 30)), standing=True)
    return event, reachable


def verify_build(controller):
    actual = getattr(getattr(controller, "_build", None), "commit_id", None)
    if actual != BUILD:
        raise RuntimeError(f"Expected reviewed Unity build {BUILD}, received {actual!r}")
    return actual


def retain_failure(directory, raw, captures, row, error):
    row.update(status="failed", error=f"{type(error).__name__}: {error}")
    try:
        row["partial_artifacts"] = save_partial(directory, raw, captures)
    except Exception as evidence_error:
        row["evidence_error"] = f"{type(evidence_error).__name__}: {evidence_error}"
    traceback.print_exc()


def local_worker(config, connection):
    metrics = {"status": "running", "diagnostic_completed": False, "cases": []}
    rng = random.Random(0)
    with controller_session(config, config["mode"], segmentation=True, connection=connection) as controller:
        metrics["unity_build_commit_id"] = verify_build(controller)
        for number in range(20):
            row = {"state_index": number, "random_seed": number, "status": "running"}
            metrics["cases"].append(row)
            refresh(metrics)
            connection.send(("progress", copy.deepcopy(metrics)))
            directory = Path(config["runtime"]["log_dir"]) / f"case_{number:02d}"
            raw, captures = {"mode": config["mode"], "state_index": number, "action_trace": []}, []
            try:
                event, reachable = source_state(controller, number, rng)
                poses, agent = full_poses(event), agent_pose(event)
                raw.update(input_object_poses=poses, input_agent_pose=agent, source_snapshot=snapshot(event))
                destination = None
                if config["mode"] == "no_reset_revisit":
                    destination = navigation_destination(controller, reachable, agent, raw, config["path_allowed_error_m"])
                    # Navigation queries precede the comparison; Pass is a new rendered frame.
                    event = step(controller, "Pass")
                    agent = agent_pose(event)
                    raw["revisit_return_agent_pose"] = copy.deepcopy(agent)
                    if agent["position"] != raw["navigation"]["source"]:
                        raise RuntimeError("Agent position changed during navigation queries; path evidence does not match baseline")
                first = capture(event, config["resolution"])
                captures.append(first)
                row["before_capture"] = save_capture(directory, "before", first)
                actions, edit = action_plan(config["mode"], poses, agent, destination)
                raw["edit"] = edit
                row["complete_pose_count"] = len(poses)
                row["intermediate_captures"] = []
                for label, action, kwargs in actions:
                    trace = {"label": label, "action": action, "arguments": copy.deepcopy(kwargs)}
                    raw["action_trace"].append(trace)
                    event = step(controller, action, **kwargs)
                    item = capture(event, config["resolution"])
                    captures.append(item)
                    trace["snapshot"] = item["snapshot"]
                    saved = save_capture(directory, label, item)
                    row["intermediate_captures"].append({"label": label, "capture": saved,
                        "transforms_from_before": compare_snapshots(first["snapshot"], item["snapshot"])})
                    if label == "edited":
                        changes = row["intermediate_captures"][-1]["transforms_from_before"]
                        row["edit_observation"] = {
                            **copy.deepcopy(edit),
                            "edited_object_changes": [change for change in changes["changed_object_transforms"]
                                                      if change["name"] == edit["edited_object_name"]],
                            "collateral_object_changes": [change for change in changes["changed_object_transforms"]
                                                          if change["name"] != edit["edited_object_name"]],
                            "object_inventory_equal": changes["object_inventory_equal"]}
                second = captures[-1]
                row["transforms"] = compare_snapshots(first["snapshot"], second["snapshot"])
                row["rgb"], images = compare_images(first, second, config["edge_band_pixels"])
                row["artifacts"] = save_evidence(directory, first, second, images, raw)
                row["status"] = "completed"
            except BaseException as error:
                retain_failure(directory, raw, captures, row, error)
                if isinstance(error, (TimeoutError, KeyboardInterrupt, SystemExit)):
                    refresh(metrics)
                    connection.send(("progress", copy.deepcopy(metrics)))
                    raise
            refresh(metrics)
            connection.send(("progress", copy.deepcopy(metrics)))
        refresh(metrics, final=True)
    return metrics


def cross_worker(config, connection):
    """Source generation or one independent Unity process; never compare shared buffers."""
    role = config["role"]
    metrics = {"role": role, "cases": [], "worker_pid": os.getpid()}
    rng = random.Random(0)
    with controller_session(config, role, segmentation=True, connection=connection) as controller:
        metrics["unity_build_commit_id"] = verify_build(controller)
        metrics["unity_pid"] = controller.unity_pid
        for number in range(20):
            row = {"state_index": number, "random_seed": number, "status": "running"}
            metrics["cases"].append(row)
            connection.send(("progress", copy.deepcopy(metrics)))
            directory = Path(config["runtime"]["log_dir"]) / f"case_{number:02d}" / role
            raw, captures = {"role": role, "state_index": number, "action_trace": []}, []
            try:
                if role == "source":
                    event, _ = source_state(controller, number, rng)
                    row["plan"] = {"object_poses": full_poses(event), "agent_pose": agent_pose(event)}
                    raw["input"] = copy.deepcopy(row["plan"])
                else:
                    plan = config["plans"][number]
                    if plan is None:
                        raise RuntimeError("Source input generation failed; no substitute case allowed")
                    raw["input"] = copy.deepcopy(plan)
                    row["input_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
                    controller.reset(scene=config["scene"])
                    event = step(controller, "SetObjectPoses", objectPoses=plan["object_poses"])
                    raw["action_trace"].append({"action": "SetObjectPoses", "snapshot": snapshot(event)})
                    event = step(controller, "TeleportFull", **plan["agent_pose"])
                    raw["action_trace"].append({"action": "TeleportFull", "snapshot": snapshot(event)})
                item = capture(event, config["resolution"])
                captures.append(item)
                row["capture"] = save_capture(directory, "captured", item)
                raw["capture_snapshot"] = item["snapshot"]
                path = directory / "action_trace.json"
                path.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
                row["action_trace"] = artifact(path)
                row["status"] = "completed"
            except BaseException as error:
                retain_failure(directory, raw, captures, row, error)
                if isinstance(error, (TimeoutError, KeyboardInterrupt, SystemExit)):
                    connection.send(("progress", copy.deepcopy(metrics)))
                    raise
            connection.send(("progress", copy.deepcopy(metrics)))
    return metrics


def collect(fn, configurations, deadline):
    workers = []
    error = None
    try:
        # Creation is inside the cleanup scope, including partial startup of G.
        for config in configurations:
            workers.append(Worker(fn, config))
        while True:
            for worker in workers:
                worker.receive()
            if any(worker.error for worker in workers):
                raise RuntimeError(json.dumps([worker.error for worker in workers]))
            if all(worker.result is not None for worker in workers):
                break
            if time.time() >= deadline:
                raise TimeoutError("Diagnostic mode deadline exceeded")
            time.sleep(0.1)
    except BaseException as caught:
        error = f"{type(caught).__name__}: {caught}"
        traceback.print_exc()
    finally:
        cleanup_errors = stop_workers(workers)
    return [{"observations": copy.deepcopy(worker.latest), "worker_error": worker.error,
             "worker_identity": worker.identity, "unity_identity": worker.unity_identity} for worker in workers], error, cleanup_errors


def cross_process(config):
    metrics = {"status": "incomplete", "diagnostic_completed": False, "cases": [], "phases": {}}
    outputs, error, cleanup = collect(cross_worker, [{**config, "role": "source"}], config["deadline"])
    metrics["phases"]["source"] = outputs[0] if outputs else {"observations": {}}
    if cleanup:
        metrics["cleanup_errors"] = cleanup
    if error or cleanup:
        metrics["error"] = error or "Source cleanup failed; independent captures not started"
        refresh(metrics)
        return metrics
    rows = {row["state_index"]: row for row in outputs[0]["observations"].get("cases", [])}
    plans = [rows.get(number, {}).get("plan") if rows.get(number, {}).get("status") == "completed" else None for number in range(20)]
    outputs, error, cleanup = collect(cross_worker,
        [{**config, "role": role, "plans": plans} for role in ("process_a", "process_b")], config["deadline"])
    for role, output in zip(("process_a", "process_b"), outputs):
        metrics["phases"][role] = output
    if error:
        metrics["error"] = error
    if cleanup:
        metrics["cleanup_errors"] = cleanup
    identities = [output["unity_identity"] for output in outputs]
    worker_identities = [output["worker_identity"] for output in outputs]
    independent = (len(outputs) == 2 and all(identities) and all(worker_identities)
                   and identities[0]["pid"] != identities[1]["pid"]
                   and worker_identities[0]["pid"] != worker_identities[1]["pid"])
    metrics["independent_unity_processes_verified"] = bool(independent)
    phase_cases = [{row["state_index"]: row for row in output["observations"].get("cases", [])} for output in outputs]
    for number in range(20):
        row = {"state_index": number, "random_seed": number, "status": "running"}
        metrics["cases"].append(row)
        captures = []
        raw = {"mode": "cross_process", "state_index": number, "input": plans[number], "process_identities": identities}
        directory = Path(config["runtime"]["log_dir"]) / f"case_{number:02d}"
        try:
            if time.time() >= config["deadline"]:
                raise TimeoutError("Cross-process comparison deadline exceeded")
            if not independent:
                raise RuntimeError("Independent Unity/worker identities could not be established")
            for cases in phase_cases:
                case = cases.get(number, {})
                if case.get("status") != "completed":
                    raise RuntimeError("Independent capture missing or failed: " + str(case.get("error")))
                expected_hash = hashlib.sha256(json.dumps(plans[number], sort_keys=True, allow_nan=False).encode()).hexdigest()
                if case.get("input_sha256") != expected_hash:
                    raise RuntimeError("Independent process input hash differs from the shared source plan")
                captures.append(load_capture(case["capture"]))
            first, second = captures
            row["complete_pose_count"] = len(plans[number]["object_poses"])
            row["transforms"] = compare_snapshots(first["snapshot"], second["snapshot"])
            row["rgb"], images = compare_images(first, second, config["edge_band_pixels"])
            row["artifacts"] = save_evidence(directory, first, second, images, raw)
            row["status"] = "completed"
        except BaseException as caught:
            retain_failure(directory, raw, captures, row, caught)
            if isinstance(caught, (TimeoutError, KeyboardInterrupt, SystemExit)):
                break
    refresh(metrics, final=True)
    if error or cleanup:
        metrics.update(status="incomplete", diagnostic_completed=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(NAMES), required=True)
    parser.add_argument("--name")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args()
    name = args.name or NAMES[args.mode]
    if name != NAMES[args.mode]:
        parser.error("Each mode must use its corresponding t00b_<mode> run name")
    config = {"name": name, "mode": args.mode, "scene": "FloorPlan1", "seed": 0,
              "cases": 20, "resolution": 224, "timeout_seconds": args.timeout_seconds,
              "grid_size": 0.1, "visibility_distance": 10, "field_of_view": 100,
              "quality": "Ultra (ai2thor 5.0.0 default)", "render_instance_segmentation": True,
              "edge_band_pixels": 1, "path_allowed_error_m": 0.01,
              "physics_policy": "default auto simulation; no PausePhysicsAutoSim actions",
              "server_timeout_seconds": 45, "server_start_timeout_seconds": 90,
              "notes": ["Original source setup: FloorPlan1, InitialRandomSpawn seeds 0..19, Python Random(0) position/yaw/horizon draws.",
                        "D/E/F reset only to generate each source; comparison interval never resets or pauses physics.",
                        "F queries complete outward AND return NavMesh paths before baseline Pass; deterministic nearest verified point at 1–1.5m, not a full T01 graph.",
                        "E requests one fixed name-sorted movable/pickupable object's +x 0.5m edit; realized transforms are measured, physical validity is not asserted.",
                        "G uses two independent worker/Unity processes with identical full pose and agent inputs, after an independent source generator exits.",
                        "RGB equality requires exact zero; completion/exit zero only means all diagnostics completed.",
                        "Equal reported transforms do not serialize hidden Unity/rendering state; shadow attribution remains unknown.",
                        "Instance boundary band is descriptive: union of four-neighbor inner boundaries, Chebyshev dilation radius 1 pixel."]}
    metrics = {"status": "incomplete", "diagnostic_completed": False, "cases": []}
    install_signals()
    try:
        if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 300:
            raise ValueError("Each mode has a positive timeout capped at 300 seconds")
        config["deadline"] = deadline_after(args.timeout_seconds)
        config["runtime"] = prepare_runtime(name)
        if args.mode == "cross_process":
            metrics = cross_process(config)
        else:
            outputs, error, cleanup = collect(local_worker, [config], config["deadline"])
            if outputs:
                metrics = outputs[0]["observations"] or metrics
                metrics["process_identity"] = {key: outputs[0][key] for key in ("worker_identity", "unity_identity")}
            if error or cleanup:
                metrics.update(status="incomplete", diagnostic_completed=False)
                if error:
                    metrics["error"] = error
                if cleanup:
                    metrics["cleanup_errors"] = cleanup
    except BaseException as error:
        metrics.update(status="incomplete", diagnostic_completed=False, error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        write_result(name, metrics, config)
    return 0 if metrics.get("diagnostic_completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
