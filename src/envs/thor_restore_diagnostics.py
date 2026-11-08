#!/usr/bin/env python3
"""Three separately logged T00 restore diagnostics; no metric tolerance changes.

Run each mode through scripts/run_logged.py with its matching name:
  --mode no_reset  --name t00_restore_no_reset
  --mode metadata  --name t00_restore_metadata
  --mode paused    --name t00_restore_paused

Exit zero means all diagnostic observations completed, not that RGB equality
passed. Exact-zero outcomes are separate fields. No GPU initialization on import.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback

try:
    from .thor_common import (ROOT, Worker, controller_session, deadline_after,
                              install_signals, prepare_runtime, step, valid_frame, write_result)
    from .thor_checks import full_poses, agent_pose
except ImportError:
    from thor_common import (ROOT, Worker, controller_session, deadline_after,
                            install_signals, prepare_runtime, step, valid_frame, write_result)
    from thor_checks import full_poses, agent_pose

NAMES = {"no_reset": "t00_restore_no_reset", "metadata": "t00_restore_metadata",
         "paused": "t00_restore_paused"}


def snapshot(event):
    """Retain actual reported transforms for ALL metadata objects, not just movers."""
    metadata = event.metadata
    objects = []
    for obj in metadata["objects"]:
        record = {key: copy.deepcopy(obj[key]) for key in
                  ("name", "objectId", "objectType", "position", "rotation", "moveable", "pickupable",
                   "isMoving", "velocity", "angularVelocity", "isOpen", "openness", "isToggled") if key in obj}
        objects.append(record)
    if len({obj["name"] for obj in objects}) != len(objects):
        raise RuntimeError("Object names are not unique; cannot match ALL realized transforms safely")
    frame = getattr(event, "frame", None)
    return {"objects": sorted(objects, key=lambda obj: obj["name"]),
            "agent": copy.deepcopy(metadata["agent"]),
            "camera": {key: copy.deepcopy(value) for key, value in metadata.items()
                       if "camera" in key.lower() or key.lower() in ("fov", "fieldofview")},
            "isSceneAtRest": metadata.get("isSceneAtRest"),
            "observed_rgb_shape": list(frame.shape) if frame is not None else None,
            "reported_render_metadata": {key: copy.deepcopy(metadata[key]) for key in
                                         ("screenWidth", "screenHeight", "screen_width", "screen_height", "fov", "fieldOfView") if key in metadata},
            "physics_auto_simulation_reported": metadata.get("autoSimulation"),
            "physics_simulation_paused_reported": metadata.get("physicsSimulationPaused"),
            "metadata_keys": sorted(metadata), "lastAction": metadata.get("lastAction"),
            "lastActionSuccess": metadata.get("lastActionSuccess")}


def vector_delta(first, second):
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise RuntimeError("Missing vector metadata")
    values = []
    for axis in ("x", "y", "z"):
        a, b = first.get(axis), second.get(axis)
        if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float))
                or not isinstance(b, (int, float)) or not math.isfinite(a) or not math.isfinite(b)):
            raise RuntimeError("Missing or non-finite transform component")
        values.append(b - a)
    return {"components": dict(zip(("x", "y", "z"), values)),
            "l2": math.sqrt(sum(value * value for value in values)),
            "max_abs_component": max(abs(value) for value in values)}


def field_changes(first, second, prefix=""):
    """Exact reported-value comparison, without an acceptance epsilon."""
    changes = []
    if isinstance(first, dict) and isinstance(second, dict):
        for key in sorted(set(first) | set(second)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in first or key not in second:
                changes.append({"field": path, "first_present": key in first, "second_present": key in second,
                                "first": first.get(key), "second": second.get(key)})
            else:
                changes.extend(field_changes(first[key], second[key], path))
    elif first != second:
        changes.append({"field": prefix, "first": first, "second": second})
    return changes


def compare_snapshots(first, second):
    a = {obj["name"]: obj for obj in first["objects"]}
    b = {obj["name"]: obj for obj in second["objects"]}
    common = sorted(set(a) & set(b))
    changed, position_deltas, rotation_deltas, wrapped_rotation_deltas, auxiliary_changes = [], [], [], [], []
    for name in common:
        position = vector_delta(a[name].get("position"), b[name].get("position"))
        rotation = vector_delta(a[name].get("rotation"), b[name].get("rotation"))
        wrapped_rotation = {axis: (value + 180) % 360 - 180 for axis, value in rotation["components"].items()}
        position_deltas.append(position["l2"])
        rotation_deltas.append(rotation["max_abs_component"])
        wrapped_rotation_deltas.append(max(abs(value) for value in wrapped_rotation.values()))
        auxiliary_changes.extend(field_changes({key: value for key, value in a[name].items() if key not in ("position", "rotation")},
                                               {key: value for key, value in b[name].items() if key not in ("position", "rotation")}, name))
        if a[name]["position"] != b[name]["position"] or a[name]["rotation"] != b[name]["rotation"]:
            changed.append({"name": name, "position_delta_m": position,
                            "raw_euler_delta_degrees": rotation,
                            "wrapped_euler_component_deltas_degrees": wrapped_rotation})
    inventory_equal = set(a) == set(b)
    agent_changes = field_changes(first["agent"], second["agent"], "agent")
    camera_changes = field_changes(first["camera"], second["camera"], "camera")
    render_changes = field_changes({key: first[key] for key in ("observed_rgb_shape", "reported_render_metadata")},
                                  {key: second[key] for key in ("observed_rgb_shape", "reported_render_metadata")})
    camera_available = "cameraPosition" in first["camera"] and "cameraPosition" in second["camera"]
    objects_equal = inventory_equal and not changed
    known_transform_change = not objects_equal or bool(agent_changes) or bool(camera_changes)
    observed_equal = False if known_transform_change else (True if camera_available else None)
    known_state_change = known_transform_change or bool(auxiliary_changes) or bool(render_changes) or first["isSceneAtRest"] != second["isSceneAtRest"]
    observed_state_equal = False if known_state_change else (True if camera_available else None)
    return {"object_inventory_equal": inventory_equal, "objects_compared": len(common),
            "missing_in_second": sorted(set(a) - set(b)), "new_in_second": sorted(set(b) - set(a)),
            "all_object_position_rotation_exact": objects_equal,
            "changed_object_transforms": changed,
            "max_object_position_delta_m": max(position_deltas) if position_deltas else None,
            "max_object_raw_euler_component_delta_degrees": max(rotation_deltas) if rotation_deltas else None,
            "max_object_wrapped_euler_component_delta_degrees": max(wrapped_rotation_deltas) if wrapped_rotation_deltas else None,
            "object_auxiliary_field_changes": auxiliary_changes,
            "agent_fields_exact": not agent_changes, "agent_field_changes": agent_changes,
            "agent_position_delta_m": vector_delta(first["agent"].get("position"), second["agent"].get("position")),
            "camera_reported_fields_exact": not camera_changes, "camera_field_changes": camera_changes,
            "camera_position_delta_m": vector_delta(first["camera"]["cameraPosition"], second["camera"]["cameraPosition"]) if camera_available else None,
            "render_field_changes": render_changes,
            "isSceneAtRest_values": [first["isSceneAtRest"], second["isSceneAtRest"]],
            "camera_world_position_available_both": camera_available,
            "observed_transforms_exact": observed_equal,
            "observed_transform_change": None if observed_equal is None else not observed_equal,
            "observed_state_fields_exact": observed_state_equal,
            "observed_state_changed": None if observed_state_equal is None else not observed_state_equal}


def capture(event, resolution):
    import numpy as np
    frame = valid_frame(event, resolution).copy()
    masks = {}
    for object_id, mask in event.instance_masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (resolution, resolution):
            raise RuntimeError("Instance mask dimensions do not match RGB")
        if np.any(mask):
            masks[object_id] = mask.copy()
    segmentation = event.instance_segmentation_frame
    return {"frame": frame, "masks": masks,
            "segmentation": None if segmentation is None else segmentation.copy(),
            "snapshot": snapshot(event), "captured_monotonic": time.monotonic()}


def instance_boundaries(masks, shape):
    """Union of inner four-neighbor mask boundaries; outside image is background."""
    import numpy as np
    if not masks:
        return None
    boundary = np.zeros(shape, dtype=bool)
    for mask in masks.values():
        padded = np.pad(mask, 1, constant_values=False)
        interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        boundary |= mask & ~interior
    return boundary


def compare_images(first, second, radius):
    import numpy as np
    difference = np.abs(first["frame"].astype(np.int16) - second["frame"].astype(np.int16))
    changed = np.any(difference != 0, axis=2)
    ys, xs = np.nonzero(changed)
    count = int(changed.sum())
    row = {"rgb_exact_zero": count == 0, "max_pixel_difference": int(difference.max()),
           "different_pixels": count, "different_channel_values": int(np.count_nonzero(difference)),
           "changed_bbox_xyxy_inclusive": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if count else None,
           "absolute_channel_difference_histogram": np.bincount(difference.ravel(), minlength=256).tolist(),
           "shadow_attribution": "unknown; spatial overlays require image inspection",
           "edge_band_radius_pixels": radius}
    boundaries = [instance_boundaries(item["masks"], changed.shape) for item in (first, second)]
    boundary, band = None, None
    if all(item is not None for item in boundaries):
        boundary = boundaries[0] | boundaries[1]
        band = boundary.copy()
        for _ in range(radius):
            padded = np.pad(band, 1, constant_values=False)
            band = np.logical_or.reduce([padded[dy:dy + changed.shape[0], dx:dx + changed.shape[1]]
                                         for dy in range(3) for dx in range(3)])
        near = int(np.count_nonzero(changed & band))
        row.update(edge_analysis_available=True, boundary_pixels=int(boundary.sum()), edge_band_pixels=int(band.sum()),
                   changed_pixels_on_instance_boundary=int(np.count_nonzero(changed & boundary)),
                   changed_pixels_in_edge_band=near, changed_pixels_outside_edge_band=count - near,
                   changed_pixel_fraction_in_edge_band=near / count if count else None)
    else:
        row.update(edge_analysis_available=False, boundary_pixels=None, edge_band_pixels=None,
                   changed_pixels_on_instance_boundary=None, changed_pixels_in_edge_band=None,
                   changed_pixels_outside_edge_band=None, changed_pixel_fraction_in_edge_band=None)
    return row, {"difference": difference, "changed": changed, "boundary": boundary, "band": band}


def save_evidence(directory, first, second, images, raw):
    import numpy as np
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    paths = []

    def save_image(name, array):
        path = directory / name
        Image.fromarray(array).save(path)
        paths.append(path)

    for index, item in enumerate((first, second), 1):
        save_image(f"capture_{index}_rgb.png", item["frame"])
        if item["segmentation"] is not None:
            save_image(f"capture_{index}_instances.png", item["segmentation"])
        capture_boundary = instance_boundaries(item["masks"], item["frame"].shape[:2])
        if capture_boundary is not None:
            save_image(f"capture_{index}_instance_boundaries.png", capture_boundary.astype(np.uint8) * 255)
        ids = sorted(item["masks"])
        if ids:
            path = directory / f"capture_{index}_masks.npz"
            np.savez_compressed(path, object_ids=np.asarray(ids), masks=np.stack([item["masks"][key] for key in ids]))
            paths.append(path)
    save_image("rgb_difference_mask.png", images["changed"].astype(np.uint8) * 255)
    save_image("rgb_max_channel_difference.png", images["difference"].max(axis=2).astype(np.uint8))
    ys, xs = np.nonzero(images["changed"])
    path = directory / "exact_rgb_differences.npz"
    np.savez_compressed(path, changed_xy=np.column_stack((xs, ys)),
                        signed_rgb_difference=second["frame"].astype(np.int16) - first["frame"].astype(np.int16))
    paths.append(path)
    overlay = first["frame"].copy()
    if images["boundary"] is not None:
        save_image("instance_boundary_union.png", images["boundary"].astype(np.uint8) * 255)
        save_image("instance_boundary_band.png", images["band"].astype(np.uint8) * 255)
        overlay[images["boundary"]] = [0, 255, 255]
    overlay[images["changed"]] = [255, 0, 0]
    save_image("difference_red_boundary_cyan_overlay.png", overlay)
    raw["capture_snapshots"] = [item["snapshot"] for item in (first, second)]
    raw["capture_monotonic"] = [item["captured_monotonic"] for item in (first, second)]
    path = directory / "raw_metadata.json"
    path.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
    paths.append(path)
    return [{"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]


def save_partial(directory, raw, captures):
    """Best-effort retention on an action/comparison failure, including one frame."""
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, item in enumerate(captures, 1):
        for suffix, array in (("rgb", item["frame"]), ("instances", item["segmentation"])):
            if array is not None:
                path = directory / f"partial_capture_{index}_{suffix}.png"
                Image.fromarray(array).save(path)
                paths.append(path)
    raw["capture_snapshots"] = [item["snapshot"] for item in captures]
    raw["capture_monotonic"] = [item["captured_monotonic"] for item in captures]
    path = directory / "partial_metadata.json"
    path.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
    paths.append(path)
    return [{"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]


def diagnostics_worker(config, connection):
    metrics = {"status": "running", "diagnostic_completed": False, "cases": []}

    def publish():
        compared = [row for row in metrics["cases"] if "rgb" in row]
        completed = [row for row in metrics["cases"] if row["status"] == "completed"]
        metrics.update(cases_attempted=len(metrics["cases"]), cases_completed=len(completed), rgb_comparisons_completed=len(compared),
                       exact_zero_failed_cases=sum(not row["rgb"]["rgb_exact_zero"] for row in compared),
                       max_pixel_difference=max((row["rgb"]["max_pixel_difference"] for row in compared), default=None),
                       exact_zero_passed=all(row["rgb"]["rgb_exact_zero"] for row in compared) if len(compared) == 20 else None)
        connection.send(("progress", copy.deepcopy(metrics)))

    rng = random.Random(config["seed"])
    with controller_session(config, config["mode"], segmentation=True, connection=connection) as controller:
        build = getattr(controller, "_build", None)
        metrics["unity_build_commit_id"] = getattr(build, "commit_id", None)
        publish()
        for number in range(20):
            row = {"state_index": number, "random_seed": config["seed"] + number, "status": "running"}
            metrics["cases"].append(row)
            raw = {"mode": config["mode"], "state_index": number, "random_seed": row["random_seed"]}
            captures = []
            directory = Path(config["runtime"]["log_dir"]) / f"case_{number:02d}"
            publish()
            try:
                # Exactly the baseline seed/draw order; pausing happens only
                # AFTER this source state is generated.
                controller.reset(scene=config["scene"])
                step(controller, "InitialRandomSpawn", randomSeed=row["random_seed"],
                     forceVisible=False, placeStationary=True, numPlacementAttempts=5)
                reachable = step(controller, "GetReachablePositions").metadata["actionReturn"]
                if not reachable:
                    raise RuntimeError("Randomized state has no reachable agent positions")
                event = step(controller, "TeleportFull", position=rng.choice(reachable),
                             rotation={"x": 0, "y": rng.choice((0, 90, 180, 270)), "z": 0},
                             horizon=rng.choice((0, 30)), standing=True)
                poses, agent = full_poses(event), agent_pose(event)
                raw.update(input_object_poses=poses, input_agent_pose=agent, source_snapshot=snapshot(event))
                row["complete_pose_count"] = len(poses)
                if config["mode"] == "no_reset":
                    paused = step(controller, "PausePhysicsAutoSim")
                    raw["pause_action_success"] = paused.metadata["lastActionSuccess"]
                    raw["before_captures_snapshot"] = snapshot(paused)
                    for _ in range(2):
                        captures.append(capture(step(controller, "Pass"), config["resolution"]))
                    row["before_first_vs_first"] = compare_snapshots(raw["before_captures_snapshot"], captures[0]["snapshot"])
                else:
                    raw["restore_action_trace"] = []
                    for repetition in (1, 2):
                        controller.reset(scene=config["scene"])
                        trace = {"repetition": repetition, "pause_action_success": None}
                        raw["restore_action_trace"].append(trace)
                        if config["mode"] == "paused":
                            paused = step(controller, "PausePhysicsAutoSim")
                            trace["pause_action_success"] = paused.metadata["lastActionSuccess"]
                            trace["after_pause_snapshot"] = snapshot(paused)
                        placed = step(controller, "SetObjectPoses", objectPoses=poses)
                        trace["after_set_object_poses_snapshot"] = snapshot(placed)
                        restored = step(controller, "TeleportFull", **agent)
                        captures.append(capture(restored, config["resolution"]))
                first, second = captures
                raw["capture_snapshots"] = [item["snapshot"] for item in captures]
                raw["capture_monotonic"] = [item["captured_monotonic"] for item in captures]
                row["transforms"] = compare_snapshots(first["snapshot"], second["snapshot"])
                row["rgb"], images = compare_images(first, second, config["edge_band_pixels"])
                row["source_to_capture_transforms"] = [compare_snapshots(raw["source_snapshot"], item["snapshot"]) for item in captures]
                row["artifacts"] = save_evidence(directory, first, second, images, raw)
                row["status"] = "completed"
            except (TimeoutError, KeyboardInterrupt, SystemExit) as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
                try:
                    row["partial_artifacts"] = save_partial(directory, raw, captures)
                except Exception as error:
                    row["evidence_error"] = f"{type(error).__name__}: {error}"
                publish()
                raise
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
                traceback.print_exc()
                # Retain any realized metadata even if capture/comparison failed.
                try:
                    row["partial_artifacts"] = save_partial(directory, raw, captures)
                except Exception as evidence_error:
                    row["evidence_error"] = f"{type(evidence_error).__name__}: {evidence_error}"
            print("RESTORE_DIAGNOSTIC_CASE: " + json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
            publish()
        metrics["diagnostic_completed"] = all(row["status"] == "completed" for row in metrics["cases"]) and len(metrics["cases"]) == 20
        metrics["status"] = "completed" if metrics["diagnostic_completed"] else "incomplete"
        publish()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(NAMES), required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--edge-band-pixels", type=int, default=1)
    args = parser.parse_args()
    name = args.name or NAMES[args.mode]
    if name != NAMES[args.mode]:
        parser.error("Each mode must use its matching t00_restore_* name")
    config = {"name": name, "mode": args.mode, "scene": "FloorPlan1", "seed": 0,
              "resolution": 224, "cases": 20, "timeout_seconds": args.timeout_seconds,
              "edge_band_pixels": args.edge_band_pixels, "grid_size": 0.1, "visibility_distance": 10,
              "field_of_view": 100, "quality": "Ultra (ai2thor 5.0.0 default)",
              "render_instance_segmentation": True, "server_timeout_seconds": 45, "server_start_timeout_seconds": 90,
              "notes": [
                  "Seeds/draw order: original T00 FloorPlan1 InitialRandomSpawn seeds 0..19; Python Random(0) position/yaw/horizon draws.",
                  "no_reset: original setup, PausePhysicsAutoSim, then two full-render Pass actions without a reset or restore between them.",
                  "metadata: reset + full SetObjectPoses + original TeleportFull twice with default physics, exactly as the original restore test.",
                  "paused: pause after EACH restore reset, before SetObjectPoses and TeleportFull; no added Pass after the restore.",
                  "TeleportFull keeps forceAction=False in every mode; its manual CharacterController gravity may adjust the agent even when auto physics is paused.",
                  "Pause action success is recorded; AI2THOR5.0 does not expose the autoSimulation/physicsSimulationPaused flags in standard metadata, so absent values remain null.",
                  "Every metadata object position/rotation, full agent metadata, and available camera metadata are saved; unreported camera fields are not invented.",
                  "Equal reported transforms do not prove equality of all hidden Unity/lighting state. isSceneAtRest is not proof that physics is paused.",
                  "RGB equality remains exact zero. Diagnostic completion and process exit zero do not imply RGB-equality acceptance.",
                  "Raw Euler component deltas are descriptive metadata differences, not quaternion geodesic rotation errors.",
                  "Boundary = union of four-neighbor inner instance-mask boundaries in both captures, including image-border contact; band uses Chebyshev dilation by the stated pixel radius.",
                  "Boundary overlap, difference histogram, and spatial masks are descriptive analysis only; the radius is not an acceptance threshold.",
                  "No automated shadow attribution is made. Inspect RGB pairs, difference masks, and red-difference/cyan-boundary overlays.",
              ]}
    metrics = {"status": "incomplete", "diagnostic_completed": False, "cases": []}
    worker = None
    install_signals()
    try:
        if not 0 < args.timeout_seconds <= 300 or not 0 <= args.edge_band_pixels <= 224:
            raise ValueError("Each mode is capped at 300 seconds; edge-band radius must be 0..224 pixels")
        config["deadline"] = deadline_after(args.timeout_seconds)
        config["runtime"] = prepare_runtime(name)
        worker = Worker(diagnostics_worker, config)
        while True:
            worker.receive()
            metrics = copy.deepcopy(worker.latest) or metrics
            if worker.result is not None:
                break
            if worker.error:
                raise RuntimeError(json.dumps(worker.error))
            if time.time() >= config["deadline"]:
                raise TimeoutError("Diagnostic mode deadline exceeded")
            time.sleep(0.1)
    except BaseException as error:
        metrics.update(status="incomplete", diagnostic_completed=False, error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        if worker:
            cleanup_errors = worker.stop()
            if cleanup_errors:
                metrics.update(status="incomplete", diagnostic_completed=False, cleanup_errors=cleanup_errors)
        write_result(name, metrics, config)
    return 0 if metrics.get("diagnostic_completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
