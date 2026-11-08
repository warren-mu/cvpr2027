"""Actual two-GPU Navigation-camera RGB gate before model initialization.

Executed only inside the training run_logged command. No training frame counter
is installed in this separate preflight process; it never changes HOME or CVD.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

from src.envs.thor_common import Worker, install_signals, process_identity, stop_workers


def frame_observation(frame):
    import numpy as np
    if frame is None:
        return {"shape": None, "dtype": None, "nonzero_rgb_values": None, "valid": False}
    data = np.asarray(frame)
    valid_shape = data.shape == (255, 255, 3) and data.dtype == np.uint8
    nonzero = int(np.count_nonzero(data)) if valid_shape else None
    return {"shape": list(data.shape), "dtype": str(data.dtype), "nonzero_rgb_values": nonzero,
            "min_rgb": int(data.min()) if valid_shape else None,
            "max_rgb": int(data.max()) if valid_shape else None,
            "valid": valid_shape and nonzero > 0}


def worker(config, connection):
    from ai2thor.platform import CloudRendering
    from PIL import Image
    from src.envs.thor_allocation import make_controller_class
    runtime, logical = config["runtime"], config["logical"]
    directory = Path(config["directory"])
    row = {"scope": "pretraining_rgb_smoke", "logical_index": logical,
           "gpu_identity": runtime["devices"][logical], "valid": False}
    connection.send(("progress", row))
    controller = None
    try:
        cls = make_controller_class(runtime, role="training_rgb_smoke")
        controller = cls.__new__(cls)
        controller.__init__(scene="FloorPlan1", agentMode="default", gridSize=0.1,
                            visibilityDistance=10, renderDepthImage=False,
                            renderInstanceSegmentation=False, width=255, height=255,
                            fieldOfView=100, platform=CloudRendering, gpu_device=logical,
                            server_timeout=30, server_start_timeout=60)
        connection.send(("unity_identity", process_identity(controller.unity_pid)))
        row["initial_frame"] = frame_observation(controller.last_event.frame)
        event = controller.step(action="Pass")
        row["pass_action_success"] = event.metadata.get("lastActionSuccess")
        row["pass_frame"] = frame_observation(event.frame)
        row["metadata"] = {key: event.metadata.get(key) for key in
                           ("sceneName", "agent", "screenWidth", "screenHeight", "lastAction", "lastActionSuccess", "errorMessage")}
        row["unity_build_commit"] = getattr(getattr(controller, "_build", None), "commit_id", None)
        if row["pass_frame"]["shape"] == [255, 255, 3] and row["pass_frame"]["dtype"] == "uint8":
            image = directory / f"gpu_{logical}.png"
            Image.fromarray(event.frame).save(image)
            row["png"] = {"path": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
        row["valid"] = (row["initial_frame"]["valid"] and row["pass_frame"]["valid"]
                        and row["pass_action_success"] is True)
    except BaseException as error:
        row["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        if controller is not None:
            try:
                controller.stop()
            except BaseException as error:
                row.update(valid=False, cleanup_error=f"{type(error).__name__}: {error}")
        (directory / f"gpu_{logical}.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    runtime = manifest["allocation"]
    directory = Path(runtime["log_dir"]) / "rgb_smoke"
    directory.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 150
    result = {"scope": "pretraining_rgb_smoke", "passed": False, "devices": [],
              "camera": {"width": 255, "height": 255, "fieldOfView": 100, "gridSize": 0.1,
                         "visibilityDistance": 10, "agentMode": "default", "platform": "CloudRendering",
                         "renderDepthImage": False, "renderInstanceSegmentation": False},
              "camera_source": "vagen/env/navigation/env.py:thor_config and NavigationEnvConfig defaults",
              "source_hashes": manifest["instrumentation_sources"],
              "scene": "FloorPlan1", "actions": ["Controller initialization", "Pass"],
              "operational_timeout_overrides": {"server_timeout": 30, "server_start_timeout": 60,
                                                "overall_seconds": 150},
              "note": "Nonzero RGB is a necessary render-validity gate, not proof of image correctness or training success."}
    workers = []
    install_signals()
    try:
        for logical in (0, 1):
            workers.append(Worker(worker, {"runtime": runtime, "logical": logical,
                                           "directory": str(directory), "deadline": deadline}))
        while True:
            for item in workers:
                item.receive()
            if all(item.result is not None or item.error is not None for item in workers):
                break
            if time.time() >= deadline:
                raise TimeoutError("Two-GPU RGB preflight deadline exceeded")
            time.sleep(0.05)
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        result["cleanup_errors"] = stop_workers(workers)
        result["devices"] = [{"observation": item.latest, "worker_error": item.error,
                               "worker_identity": item.identity, "unity_identity": item.unity_identity}
                              for item in workers]
        result["passed"] = (len(workers) == 2 and not result.get("error") and not result["cleanup_errors"]
                            and all(item.result is not None and item.result.get("valid") is True and item.error is None
                                    for item in workers))
        output = directory.parent / "rgb_smoke.json"
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("T00B_RGB_SMOKE: " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
