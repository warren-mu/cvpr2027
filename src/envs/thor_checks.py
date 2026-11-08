#!/usr/bin/env python3
"""T00: real RGB/masks, 20 full-pose restores, and 10 single-object moves."""
import argparse
import copy
import json
import math
import random
import statistics
import time
import traceback

try:
    from .thor_common import (ROOT, Worker, controller_session, deadline_after,
                              install_signals, prepare_runtime, step, valid_frame, write_result)
except ImportError:
    from thor_common import (ROOT, Worker, controller_session, deadline_after,
                            install_signals, prepare_runtime, step, valid_frame, write_result)


def full_poses(event):
    """Every moveable OR pickupable object is required by SetObjectPoses."""
    objects = event.metadata["objects"]
    poses = [{"objectName": o["name"], "position": copy.deepcopy(o["position"]),
              "rotation": copy.deepcopy(o["rotation"])}
             for o in objects if o.get("moveable") or o.get("pickupable")]
    if not poses or len({p["objectName"] for p in poses}) != len(poses):
        raise RuntimeError("Empty or ambiguous movable-object pose inventory")
    return sorted(poses, key=lambda p: p["objectName"])


def agent_pose(event):
    agent = event.metadata["agent"]
    return {"position": copy.deepcopy(agent["position"]),
            "rotation": copy.deepcopy(agent["rotation"]),
            "horizon": agent["cameraHorizon"], "standing": agent["isStanding"]}


def distance(a, b):
    return math.sqrt(sum((a[axis] - b[axis]) ** 2 for axis in ("x", "y", "z")))


def check_worker(config, connection):
    import numpy as np
    from PIL import Image

    metrics = {"status": "running", "restore_states": [], "object_moves": []}

    def publish():
        differences = [r["max_pixel_difference"] for r in metrics["restore_states"]
                       if r["max_pixel_difference"] is not None]
        errors = [r["position_error_m"] for r in metrics["object_moves"] if r["position_error_m"] is not None]
        counts = [r["collateral_moved_over_1cm_count"] for r in metrics["object_moves"]
                  if r["collateral_moved_over_1cm_count"] is not None]
        metrics.update(restore_states_attempted=len(metrics["restore_states"]),
                       restore_comparisons_completed=len(differences),
                       restore_max_pixel_difference=max(differences) if differences else None,
                       restore_failed_states=sum(r["failed"] for r in metrics["restore_states"]),
                       object_moves_attempted=len(metrics["object_moves"]),
                       object_position_errors_measured=len(errors),
                       object_position_error_median_m=statistics.median(errors) if errors else None,
                       object_position_error_max_m=max(errors) if errors else None,
                       collateral_moves_measured=len(counts),
                       collateral_moved_over_1cm_total=sum(counts) if counts else None)
        connection.send(("progress", copy.deepcopy(metrics)))

    with controller_session(config, "checks", segmentation=True, connection=connection) as controller:
        event = controller.last_event
        frame = valid_frame(event, config["resolution"])
        smoke_path = ROOT / "logs" / "thor_smoke.png"
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(smoke_path)
        masks = {key: value for key, value in event.instance_masks.items() if np.any(value)}
        ids = {obj["objectId"] for obj in event.metadata["objects"]}
        matches = sum(object_id in ids for object_id in masks)
        metrics.update({
            "smoke_frame_path": str(smoke_path.relative_to(ROOT)),
            "smoke_frame_shape": list(frame.shape),
            "instance_count": len(masks), "instance_ids_matching_metadata": matches,
            "instance_id_match_ratio": matches / len(masks) if masks else None,
            "unmatched_instance_ids": sorted(set(masks) - ids),
        })
        if not masks:
            publish()
            raise RuntimeError("No nonempty instance masks in the first frame")
        publish()
        rng = random.Random(config["seed"])
        failure_dir = ROOT / "logs" / "runs" / config["name"] / "restore_failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        for number in range(20):
            row = {"state_index": number, "random_seed": config["seed"] + number,
                   "max_pixel_difference": None, "failed": True}
            metrics["restore_states"].append(row)
            try:
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
                inventory = sorted(obj["name"] for obj in event.metadata["objects"])
                # Preserve exact realized state for reproducing failures, rather
                # than treating the random seed as a state serialization.
                row.update(object_poses=poses, agent_pose=agent,
                           object_inventory=inventory, complete_pose_count=len(poses))
                frames = []
                inventory_preserved = []
                for repetition in (1, 2):
                    controller.reset(scene=config["scene"])
                    step(controller, "SetObjectPoses", objectPoses=poses)
                    restored = step(controller, "TeleportFull", **agent)
                    frames.append(valid_frame(restored, config["resolution"]).copy())
                    inventory_preserved.append(
                        inventory == sorted(o["name"] for o in restored.metadata["objects"]))
                diff = np.abs(frames[0].astype(np.int16) - frames[1].astype(np.int16))
                row.update(max_pixel_difference=int(diff.max()),
                           different_pixels=int(np.count_nonzero(np.any(diff != 0, axis=2))),
                           inventory_preserved=inventory_preserved,
                           failed=bool(np.any(diff) or not all(inventory_preserved)))
                if row["failed"]:
                    for repetition, image in enumerate(frames, 1):
                        Image.fromarray(image).save(failure_dir / f"state_{number:02d}_restore_{repetition}.png")
            except TimeoutError:
                publish()
                raise
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                traceback.print_exc()
            print("RESTORE_STATE: " + json.dumps(row, sort_keys=True), flush=True)
            publish()
        differences = [row["max_pixel_difference"] for row in metrics["restore_states"]
                       if row["max_pixel_difference"] is not None]
        metrics.update(restore_comparisons_completed=len(differences),
                       restore_max_pixel_difference=max(differences) if differences else None,
                       restore_failed_states=sum(row["failed"] for row in metrics["restore_states"]))
        publish()

        controller.reset(scene=config["scene"])
        baseline = copy.deepcopy(controller.last_event.metadata["objects"])
        baseline_poses = full_poses(controller.last_event)
        candidates = sorted(o["name"] for o in baseline if o.get("pickupable"))
        rng.shuffle(candidates)
        selected = candidates[:10]
        metrics["pickupable_candidates"] = len(candidates)
        for number, name in enumerate(selected):
            row = {"move_index": number, "object_name": name, "position_error_m": None,
                   "collateral_moved_over_1cm_count": None, "action_success": False}
            metrics["object_moves"].append(row)
            try:
                controller.reset(scene=config["scene"])
                before_event = step(controller, "SetObjectPoses", objectPoses=baseline_poses)
                before = {o["name"]: o for o in before_event.metadata["objects"]}
                poses = full_poses(before_event)
                target = next(p for p in poses if p["objectName"] == name)
                target["position"]["x"] += config["object_shift_m"]
                row["target_position"] = copy.deepcopy(target["position"])
                # All other movable objects remain explicitly present.
                after_event = step(controller, "SetObjectPoses", objectPoses=poses)
                after = {o["name"]: o for o in after_event.metadata["objects"]}
                row["action_success"] = True
                row["missing_object_names"] = sorted(set(before) - set(after))
                if name not in after:
                    raise RuntimeError("Target disappeared after SetObjectPoses")
                row["actual_position"] = after[name]["position"]
                row["position_error_m"] = distance(row["target_position"], after[name]["position"])
                collateral = [{"object_name": other,
                               "displacement_m": distance(before[other]["position"], after[other]["position"])}
                              for other in sorted(set(before) & set(after)) if other != name]
                row["collateral_moved_over_1cm"] = [item for item in collateral if item["displacement_m"] > 0.01]
                row["collateral_moved_over_1cm_count"] = len(row["collateral_moved_over_1cm"])
                if row["missing_object_names"]:
                    raise RuntimeError("Object inventory changed during a single-object edit")
            except TimeoutError:
                publish()
                raise
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                traceback.print_exc()
            print("OBJECT_MOVE: " + json.dumps(row, sort_keys=True), flush=True)
            publish()
        errors = [r["position_error_m"] for r in metrics["object_moves"] if r["position_error_m"] is not None]
        collateral_counts = [r["collateral_moved_over_1cm_count"] for r in metrics["object_moves"]
                             if r["collateral_moved_over_1cm_count"] is not None]
        metrics.update(object_moves_attempted=len(selected), object_position_errors_measured=len(errors),
                       object_position_error_median_m=statistics.median(errors) if errors else None,
                       object_position_error_max_m=max(errors) if errors else None,
                       collateral_moved_over_1cm_total=sum(collateral_counts) if collateral_counts else None,
                       collateral_moves_measured=len(collateral_counts))
        complete = (len(selected) == 10 and len(errors) == 10 and len(collateral_counts) == 10
                    and not any("error" in row for row in metrics["object_moves"])
                    and metrics["restore_failed_states"] == 0 and len(differences) == 20
                    and metrics["instance_id_match_ratio"] == 1.0)
        metrics["status"] = "passed" if complete else "failed"
        publish()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="t00_thor_checks")
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    args = parser.parse_args()
    config = vars(args).copy()
    config.update(object_shift_m=0.05, random_states=20, object_moves=10,
                  grid_size=0.1, visibility_distance=10, field_of_view=100,
                  quality="Ultra (ai2thor 5.0.0 default)",
                  server_timeout_seconds=45, server_start_timeout_seconds=90,
                  restore_scope="reset scene, restore complete movable/pickupable poses and full agent pose twice",
                  physical_validity_claim=False)
    metrics = {"status": "failed"}
    worker = None
    install_signals()
    try:
        if args.resolution < 224 or args.timeout_seconds <= 0:
            raise ValueError("Resolution must be >=224 and timeout positive")
        config["deadline"] = deadline_after(args.timeout_seconds)
        config["runtime"] = prepare_runtime(args.name)
        worker = Worker(check_worker, config)
        while True:
            worker.receive()
            metrics = copy.deepcopy(worker.latest) or metrics
            if worker.result is not None:
                break
            if worker.error:
                raise RuntimeError(json.dumps(worker.error))
            if time.time() >= config["deadline"]:
                raise TimeoutError("T00 checks deadline exceeded")
            time.sleep(0.1)
    except BaseException as error:
        metrics.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        if worker:
            cleanup_errors = worker.stop()
            if cleanup_errors:
                metrics.update(status="failed", cleanup_errors=cleanup_errors)
        write_result(args.name, metrics, config)
    return 0 if metrics.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
