"""Logged, bounded RGB gate for every GPU in an existing PBS allocation.

Exit 86 means completed, correctly shaped captures contained black RGB. Other
errors return 1. No HOME or CUDA mask is changed; private workers select only
validated logical slots through the allocation Controller adapter.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import time
import traceback

from src.envs.thor_common import Worker, install_signals, process_identity, stop_workers

ROOT = Path(__file__).resolve().parents[2]
BLACK_FRAME_EXIT = 86
FIFO_SHA256 = "434d4eac1647c2acb32542c7c513a66fca40022fcdc1d23d119f2fd4ffa6ecc3"


def frame_observation(frame, resolution):
    import numpy as np
    if frame is None:
        return {"shape": None, "dtype": None, "valid_shape": False,
                "nonzero_rgb_values": None, "black": False, "valid": False}
    data = np.asarray(frame)
    shape = data.shape == (resolution, resolution, 3) and data.dtype == np.uint8
    nonzero = int(np.count_nonzero(data)) if shape else None
    return {"shape": list(data.shape), "dtype": str(data.dtype), "valid_shape": shape,
            "nonzero_rgb_values": nonzero, "black": shape and nonzero == 0,
            "min_rgb": int(data.min()) if shape else None,
            "max_rgb": int(data.max()) if shape else None,
            "valid": shape and nonzero > 0}


def gate_exit_code(rows, expected_gpus, errors):
    """Black is retryable only when no unrelated error/cleanup ambiguity exists."""
    if errors or len(rows) != expected_gpus:
        return 1
    observations = [row.get("observation", {}) for row in rows]
    if any(row.get("worker_error") or item.get("error") or item.get("cleanup_error")
           or not item.get("captures") or any(not capture.get("valid_shape")
           or not capture.get("action_success") for capture in item.get("captures", []))
           for row, item in zip(rows, observations)):
        return 1
    if any(capture.get("black") for item in observations for capture in item["captures"]):
        return BLACK_FRAME_EXIT
    return 0 if all(item.get("valid") is True for item in observations) else 1


def wire_observation(previous_files, metadata, files, stage, sequence_id):
    """Pure observation of hash-checked FIFO's cache identity contract."""
    fresh = files is not previous_files
    agents = metadata.get("agents", []) if isinstance(metadata, dict) else []
    packets = files.get("image", []) if fresh else []
    return {"stage": stage, "sequence_id": sequence_id,
            "fresh_main_rgb_payloads": len(packets), "cached_metadata": not fresh,
            "payload_sha256": [hashlib.sha256(body).hexdigest() for body in packets],
            "payload_bytes": [len(body) for body in packets],
            "screen_width": agents[0].get("screenWidth") if len(agents) == 1 else None,
            "screen_height": agents[0].get("screenHeight") if len(agents) == 1 else None}


def worker(config, connection):
    import ai2thor.fifo_server as fifo
    from ai2thor.platform import CloudRendering
    from PIL import Image
    from src.envs.thor_allocation import make_controller_class
    runtime, logical = config["runtime"], config["logical"]
    directory, resolution = Path(config["directory"]), config["resolution"]
    row = {"logical_index": logical, "gpu_identity": runtime["devices"][logical],
           "node": socket.gethostname(), "valid": False, "captures": [], "wire_packets": []}
    connection.send(("progress", row))
    controller = None
    original_receive = fifo.FifoServer._recv_message
    stage = "constructor"
    try:
        if hashlib.sha256(Path(fifo.__file__).read_bytes()).hexdigest() != FIFO_SHA256:
            raise RuntimeError("RGB gate requires the pinned unmodified AI2THOR 5 FIFO source")

        def receive(server, timeout):
            previous_files = server.raw_files
            metadata, files = original_receive(server, timeout)
            # In this hash-checked version METADATA_PATCH returns the identical
            # previous raw_files object. Normal wire messages create a new dict.
            row["wire_packets"].append(wire_observation(previous_files, metadata, files, stage, server.sequence_id))
            return metadata, files

        fifo.FifoServer._recv_message = receive
        cls = make_controller_class(runtime, role="render_gate")
        controller = cls.__new__(cls)
        controller.__init__(scene="FloorPlan1", agentMode="default", gridSize=0.1,
                            visibilityDistance=10, renderDepthImage=False,
                            renderInstanceSegmentation=config["segmentation"],
                            width=resolution, height=resolution, fieldOfView=100,
                            platform=CloudRendering, gpu_device=logical,
                            server_timeout=30, server_start_timeout=60)
        connection.send(("unity_identity", process_identity(controller.unity_pid)))
        for label in ("after_constructor", "after_Pass"):
            if label == "after_Pass":
                stage = "Pass"
                event = controller.step(action="Pass")
            else:
                event = controller.last_event
            observation = frame_observation(event.frame, resolution)
            observation.update(stage=label, action_success=event.metadata.get("lastActionSuccess") is True,
                               action=event.metadata.get("lastAction"),
                               error_message=event.metadata.get("errorMessage"))
            if observation["valid_shape"]:
                path = directory / f"gpu_{logical}_{label}.png"
                Image.fromarray(event.frame).save(path)
                observation["png"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            if config["segmentation"]:
                segmentation = frame_observation(event.instance_segmentation_frame, resolution)
                observation["segmentation"] = segmentation
            row["captures"].append(observation)
            connection.send(("progress", row))
        row["unity_build_commit"] = getattr(getattr(controller, "_build", None), "commit_id", None)
        row["actual_fresh_rgb_payload_count"] = sum(packet["fresh_main_rgb_payloads"] for packet in row["wire_packets"])
        row["valid"] = all(capture["valid"] and capture["action_success"] for capture in row["captures"])
        if not any(packet["stage"] == "Pass" and packet["fresh_main_rgb_payloads"] > 0
                   for packet in row["wire_packets"]):
            raise RuntimeError("Pass did not produce an observed fresh RGB wire payload")
    except BaseException as error:
        row["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        fifo.FifoServer._recv_message = original_receive
        if controller is not None:
            try:
                controller.stop()
            except BaseException as error:
                row.update(valid=False, cleanup_error=f"{type(error).__name__}: {error}")
        (directory / f"gpu_{logical}.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpus", required=True, type=int, choices=range(1, 9))
    parser.add_argument("--resolution", type=int, required=True, choices=(224, 255))
    parser.add_argument("--segmentation", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=180)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.name) or not 15 <= args.timeout_seconds <= 600:
        parser.error("Require a safe unique name and a 15..600 second deadline")
    output = ROOT / "results" / f"{args.name}.json"
    directory = ROOT / "logs/runs" / args.name / "render_gate"
    if output.exists() or directory.exists():
        raise RuntimeError("Refusing to overwrite an existing gate run; use a new run name")
    directory.mkdir(parents=True, exist_ok=False)
    workers, errors, runtime, rows = [], [], None, []
    deadline = time.time() + args.timeout_seconds
    install_signals()
    signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
    try:
        from src.envs.thor_allocation import prepare_allocation
        runtime = prepare_allocation(args.name, expected_gpus=args.gpus)
        for logical in range(args.gpus):
            workers.append(Worker(worker, {"runtime": runtime, "logical": logical,
                "directory": str(directory), "resolution": args.resolution,
                "segmentation": args.segmentation, "deadline": deadline}))
        while True:
            for item in workers:
                item.receive()
            if all(item.result is not None or item.error is not None for item in workers):
                break
            if time.time() >= deadline:
                raise TimeoutError("All-GPU render gate deadline exceeded")
            time.sleep(0.05)
    except BaseException as error:
        errors.append(f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        cleanup_errors = stop_workers(workers)
        errors.extend(cleanup_errors)
        rows = [{"observation": item.latest, "worker_error": item.error,
                 "worker_identity": item.identity, "unity_identity": item.unity_identity} for item in workers]
    code = gate_exit_code(rows, args.gpus, errors)
    metrics = {"passed": code == 0, "exit_code": code, "black_frame_detected": any(
        capture.get("black") for row in rows for capture in row["observation"].get("captures", [])),
        "devices": rows, "errors": errors, "cleanup_errors": cleanup_errors,
        "expected_gpu_count": args.gpus, "completed_gpu_count": sum(item.result is not None for item in workers)}
    result = {"metrics": metrics, "config": {"name": args.name, "allocation": runtime,
        "node": socket.gethostname(), "pbs_jobid": os.environ.get("PBS_JOBID"),
        "resolution": args.resolution, "segmentation": args.segmentation,
        "timeout_seconds": args.timeout_seconds, "fifo_source_sha256": FIFO_SHA256,
        "camera": {"scene": "FloorPlan1", "fieldOfView": 100, "gridSize": 0.1,
                   "visibilityDistance": 10, "agentMode": "default", "platform": "CloudRendering"},
        "scope": "render_validity_preflight_not_training_frames",
        "notes": ["Nonzero RGB is necessary, not proof of semantic image correctness.",
                  "Wire counts include constructor frames; cached metadata responses are excluded.",
                  "Startup wire frames may be 300px; returned captures must match requested resolution.",
                  "Parent and worker CUDA masks remain exactly the original PBS allocation."]}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
