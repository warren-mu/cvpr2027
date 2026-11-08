"""T00b receive-timing diagnostic, with four predeclared independent processes.

Only a logged, committed PBS run may produce observations. Every arm uses the
unchanged thor_common.controller_session and identical actions. Black frames
are evidence, never a reason to skip the remaining stages or replace a trial.
"""
import argparse
from contextlib import nullcontext
import copy
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import time
import traceback

from .thor_common import (ROOT, Worker, controller_session, deadline_after,
                          install_signals, prepare_runtime, step, stop_workers, write_result)
from .thor_checks import agent_pose
from .thor_isolation_diagnostics import BUILD, source_state
from .thor_render_probe import capture_event, file_hash, filtered_env, observe_fifo_receives, write_json

ARMS = ("baseline_without_receive_observer", "with_existing_probe_capture_observer",
        "fixed_250ms_delay_after_receive", "baseline_repeat")
STAGES = ("after_constructor", "after_InitialRandomSpawn_seed0",
          "after_GetReachablePositions_and_Teleport_current_pose", "after_reset_FloorPlan1",
          "after_isolation_source_state_seed0_random0")


def snapshot_event(event):
    """Quick copies only: no mask expansion, image encoding, file or Pipe I/O."""
    return {"metadata": copy.deepcopy(event.metadata),
            "rgb": None if event.frame is None else event.frame.copy(),
            "segmentation": None if event.instance_segmentation_frame is None
                else event.instance_segmentation_frame.copy()}


def save_stage(snapshot, directory, label, resolution):
    """Called only after this trial's Unity has stopped; preserve black images."""
    import numpy as np
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / f"{label}_metadata.json"
    write_json(metadata, snapshot["metadata"])
    row = {"stage": label, "metadata_path": str(metadata), "metadata_sha256": file_hash(metadata),
           "lastAction": snapshot["metadata"].get("lastAction"),
           "lastActionSuccess": snapshot["metadata"].get("lastActionSuccess"),
           "rgb": None, "segmentation": None}
    arrays = {}
    for key in ("rgb", "segmentation"):
        frame = snapshot[key]
        if frame is None:
            continue
        arrays[key] = frame
        stats = {"shape": list(frame.shape), "dtype": str(frame.dtype),
                 "min": int(frame.min()) if frame.size else None,
                 "max": int(frame.max()) if frame.size else None,
                 "nonzero_values": int(np.count_nonzero(frame)),
                 "nonzero_pixels": int(np.count_nonzero(np.any(frame != 0, axis=2)))
                    if frame.ndim == 3 else None}
        if frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[-1] in (3, 4):
            path = directory / f"{label}_{key}.png"
            Image.fromarray(frame).save(path)
            stats.update(png_path=str(path), png_sha256=file_hash(path))
        row[key] = stats
    rgb = snapshot["rgb"]
    row["matches_existing_nonblack_frame_guard"] = (
        bool(rgb.shape == (resolution, resolution, 3) and np.any(rgb)) if rgb is not None else False)
    raw = directory / f"{label}_arrays.npz"
    np.savez_compressed(raw, **arrays)
    row.update(raw_arrays_path=str(raw), raw_arrays_sha256=file_hash(raw))
    return row


def timing_worker(config, connection):
    from ai2thor.fifo_server import FifoServer
    directory = Path(config["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    metrics = {"arm": config["arm"], "status": "running", "completed": False,
               "stages": [], "receive_observations": [], "cleanup_errors": [],
               "environment_before_controller": filtered_env(os.environ),
               "stage_buffers_copied": [], "completed_receive_delays": 0}
    snapshots = []
    current_stage = "constructor"

    def observe(server, event, previous_files):
        action = json.loads(server._last_action_message) if server._last_action_message else None
        observation = capture_event(event, directory / "receive_observer",
            len(metrics["receive_observations"]), current_stage, action)
        observation["cached_raw_files_mapping_reused"] = server.raw_files is previous_files
        metrics["receive_observations"].append(observation)
        # Match the successful probe's per-receive file/Pipe publication too;
        # neither baseline nor the fixed-delay arm performs this extra I/O.
        write_json(directory / "receive_progress.json", metrics)
        connection.send(("progress", copy.deepcopy(metrics)))

    def delay(server, event, previous_files):
        time.sleep(0.25)
        metrics["completed_receive_delays"] += 1

    if config["arm"] == "with_existing_probe_capture_observer":
        (directory / "receive_observer").mkdir()
        receive_context = observe_fifo_receives(FifoServer, observe)
    elif config["arm"] == "fixed_250ms_delay_after_receive":
        receive_context = observe_fifo_receives(FifoServer, delay)
    else:
        receive_context = nullcontext()

    def remember(label, event):
        # All arms use exactly this same stage-copy work. Baseline does not
        # publish or encode anything between individual receive calls.
        snapshot = snapshot_event(event)
        snapshots.append((label, snapshot))
        metrics["stage_buffers_copied"].append(label)
        if event.metadata.get("lastActionSuccess") is not True:
            raise RuntimeError(f"{label}: {event.metadata.get('errorMessage')}")

    try:
        # Reserve up to 15s for the existing helper's scoped stop plus 5s to
        # persist copied stages. The parent enforces the whole 75s worker cap.
        signal.setitimer(signal.ITIMER_REAL, max(0.1, config["deadline"] - time.time() - 20))
        with receive_context:
            with controller_session(config, config["arm"], segmentation=True, connection=connection) as controller:
                if controller._build.commit_id != BUILD:
                    raise RuntimeError("Unexpected Unity build")
                metrics["controller"] = {"quality": controller.quality, "headless": controller.headless,
                    "width": controller.width, "height": controller.height, "gpu_device": controller.gpu_device,
                    "initialization_parameters": copy.deepcopy(controller.initialization_parameters),
                    "build": controller._build.commit_id, "unity_command": list(controller.server.unity_proc.args)}
                metrics["environment_during_controller"] = filtered_env(os.environ)
                remember(STAGES[0], controller.last_event)
                current_stage = STAGES[1]
                remember(current_stage, step(controller, "InitialRandomSpawn", randomSeed=0,
                    forceVisible=False, placeStationary=True, numPlacementAttempts=5))
                current_stage = STAGES[2]
                event = step(controller, "GetReachablePositions")
                remember(current_stage, step(controller, "TeleportFull", **agent_pose(event)))
                current_stage = STAGES[3]
                remember(current_stage, controller.reset(scene="FloorPlan1"))
                current_stage = STAGES[4]
                event, _ = source_state(controller, 0, random.Random(0))
                remember(current_stage, event)
        metrics.update(status="completed", completed=True)
    except BaseException as error:
        metrics.update(status="failed", completed=False, error=f"{type(error).__name__}: {error}",
                       last_stage_started=current_stage)
        traceback.print_exc()
    finally:
        # controller_session has already stopped Unity (or propagated its stop
        # failure). No stage output is encoded during the baseline action trace.
        signal.setitimer(signal.ITIMER_REAL, max(0.1, config["deadline"] - time.time()))
        try:
            for label, snapshot in snapshots:
                metrics["stages"].append(save_stage(snapshot, directory / "stages", label, config["resolution"]))
            metrics["stages_observed"] = len(metrics["stages"])
            metrics["all_stages_match_nonblack_frame_guard"] = (
                all(row["matches_existing_nonblack_frame_guard"] for row in metrics["stages"])
                if len(metrics["stages"]) == len(STAGES) else None)
            write_json(directory / "observations.json", metrics)
            connection.send(("progress", copy.deepcopy(metrics)))
        except BaseException as error:
            metrics.update(status="failed", completed=False,
                persistence_error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
            connection.send(("progress", copy.deepcopy(metrics)))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="t00b_render_timing_probe")
    parser.add_argument("--timeout-seconds", type=float, default=360)
    args = parser.parse_args()
    if not re.fullmatch(r"t00b_[A-Za-z0-9_]+", args.name):
        parser.error("Use a unique t00b_ name containing only letters, digits and underscores")
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 360:
        parser.error("Overall timeout must be positive and at most 360 seconds")
    directory = ROOT / "logs" / "runs" / args.name
    if (ROOT / "results" / f"{args.name}.json").exists():
        parser.error("Existing result is preserved; use a new name")
    directory.mkdir(parents=True, exist_ok=False)
    config = {"name": args.name, "scene": "FloorPlan1", "resolution": 224,
        "grid_size": 0.1, "visibility_distance": 10, "field_of_view": 100,
        "quality": "Ultra (ai2thor 5.0.0 default)", "render_instance_segmentation": True,
        "arms": list(ARMS), "stages": list(STAGES), "timeout_seconds": args.timeout_seconds,
        "worker_timeout_seconds": 75, "worker_cleanup_and_save_reserve_seconds": 20,
        "delay_after_receive_seconds": 0.25, "seed": 0,
        "notes": ["Same unchanged controller_session, GPU allocation, scene, settings and action sequence in every arm.",
                  "Each arm starts one new worker/Unity; immutable releases and existing common cache policy are retained.",
                  "Baselines have no receive wrapper. All arms copy five stage RGB/segmentation/metadata snapshots; save them only after Unity stops.",
                  "Observer arm saves every receive via existing capture_event and publishes progress JSON/Pipe, matching the earlier probe's I/O; delay arm only waits 0.25s after each original receive.",
                  "Final stage calls the unmodified isolation.source_state(controller,0,Random(0)), including its own reset.",
                  "Black frames are retained and do not abort a trial; the existing nonblack RGB shape/any-value guard is reported unchanged.",
                  "Fixed sequential order and shared cache history mean differences establish timing association at most, not a causal mechanism.",
                  "No action, renderImage flag, quality, backend, random seed, or research exact-zero criterion is changed.",
                  "Partial in-memory snapshots are saved best-effort after failures; a hard kill can prevent persistence, never imply an observed frame."]}
    metrics = {"status": "incomplete", "diagnostic_completed": False, "arms": [], "cleanup_errors": []}
    install_signals()
    try:
        config["deadline"] = deadline_after(args.timeout_seconds)
        config["runtime"] = prepare_runtime(args.name)
        for arm in ARMS:
            if time.time() >= config["deadline"] - 24:
                raise TimeoutError("Overall deadline cannot accommodate another trial and cleanup")
            settings = {**config, "arm": arm, "directory": str(directory / arm),
                        "deadline": min(time.time() + 75, config["deadline"] - 4)}
            worker = None
            observation = {"arm": arm, "status": "not_started", "completed": False}
            try:
                worker = Worker(timing_worker, settings)
                while time.time() < settings["deadline"] and worker.result is None and worker.error is None:
                    worker.receive()
                    time.sleep(0.03)
                worker.receive()
                observation = worker.result or worker.latest or observation
                if worker.error or worker.result is None:
                    observation.update(status="failed", completed=False,
                        worker_error=worker.error or {"type": "TimeoutError", "message": "75-second worker deadline"})
            finally:
                cleanup = stop_workers([worker]) if worker else []
                if cleanup:
                    observation.update(status="failed", completed=False)
                    metrics["cleanup_errors"].extend(cleanup)
                metrics["arms"].append(observation)
            if cleanup or observation.get("cleanup_errors"):
                raise RuntimeError("Cleanup ownership uncertain; refusing another Unity process")
        metrics["diagnostic_completed"] = len(metrics["arms"]) == len(ARMS) and all(row.get("completed") for row in metrics["arms"])
        metrics["status"] = "completed" if metrics["diagnostic_completed"] else "incomplete"
    except BaseException as error:
        metrics["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        write_result(args.name, metrics, config)
    return 0 if metrics["diagnostic_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
