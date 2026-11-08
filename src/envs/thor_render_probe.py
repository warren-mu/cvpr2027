"""Bounded T00b black-frame investigation; keeps every black image as evidence.

Four predeclared independent Unity processes: 224/current cache environment,
224/fresh cache environment, 255/current, 300/current. This is instrumentation,
not a replacement benchmark or a change to the exact-zero research criterion.
Run only inside PBS through run_logged, after committing this file.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import time
import traceback

from .thor_common import (ROOT, Worker, deadline_after, install_signals,
                          prepare_runtime, process_identity, stop_workers, write_result)
from .thor_checks import agent_pose

BUILD = "f0825767cd50d69f666c7f282e54abfe58f1e917"
TRIALS = (("224_default", 224, False), ("224_fresh", 224, True),
          ("255_default", 255, False), ("300_default", 300, False))
ENV_KEYS = ("HOME", "TMPDIR", "TEMP", "TMP", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
            "XDG_DATA_HOME", "XDG_RUNTIME_DIR", "CUDA_CACHE_PATH", "CUDA_VISIBLE_DEVICES",
            "DISPLAY", "LD_LIBRARY_PATH", "PYTHONPATH", "VK_ICD_FILENAMES", "VK_DRIVER_FILES",
            "VK_LAYER_PATH", "VK_INSTANCE_LAYERS", "__GL_SHADER_DISK_CACHE",
            "__GL_SHADER_DISK_CACHE_PATH", "__GL_SHADER_DISK_CACHE_SIZE",
            "AI2THOR_SERVER_SIDE_SCREENSHOT", "AI2THOR_SERVER_TYPE")


def filtered_env(environment):
    # Never record the complete environment or AI2THOR_CLIENT_TOKEN.
    return {key: environment[key] for key in ENV_KEYS if key in environment}


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


@contextmanager
def observe_fifo_receives(server_class, observer):
    """Wrap one private worker's receive method while preserving class identity.

    AI2-THOR 5.0 _build_server compares classes by equality, so a FifoServer
    subclass is not accepted. Only this process's in-memory method is changed;
    no installed file or class reference is replaced, and all exits restore it.
    """
    original = server_class.receive

    def receive(server, timeout=None):
        previous_files = getattr(server, "raw_files", None)
        event = original(server, timeout=timeout)
        observer(server, event, previous_files)
        return event

    server_class.receive = receive
    try:
        yield
    finally:
        server_class.receive = original


def capture_event(event, directory, index, stage, last_action):
    """Save raw observations BEFORE deciding whether an RGB frame is usable."""
    import numpy as np
    from PIL import Image
    stem = f"receive_{index:03d}"
    metadata_path = directory / f"{stem}_metadata.json"
    write_json(metadata_path, copy.deepcopy(event.metadata))
    row = {"index": index, "stage": stage, "last_action_sent": last_action,
           "metadata_path": str(metadata_path), "metadata_sha256": file_hash(metadata_path),
           "lastAction": event.metadata.get("lastAction"),
           "lastActionSuccess": event.metadata.get("lastActionSuccess"),
           "errorMessage": event.metadata.get("errorMessage"), "rgb": None,
           "instance_segmentation": None, "instance_masks": None}
    arrays = {}
    for field, frame in (("rgb", event.frame),
                         ("instance_segmentation", event.instance_segmentation_frame)):
        if frame is None:
            continue
        frame = np.asarray(frame).copy()
        arrays[field] = frame
        stats = {"shape": list(frame.shape), "dtype": str(frame.dtype),
                 "min": int(frame.min()) if frame.size else None,
                 "max": int(frame.max()) if frame.size else None,
                 "nonzero_values": int(np.count_nonzero(frame)),
                 "nonzero_pixels": int(np.count_nonzero(np.any(frame != 0, axis=-1)))
                    if frame.ndim == 3 else None,
                 "array_sha256": hashlib.sha256(frame.tobytes()).hexdigest()}
        if frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[-1] in (3, 4):
            path = directory / f"{stem}_{field}.png"
            Image.fromarray(frame).save(path)
            stats.update(png_path=str(path), png_sha256=file_hash(path))
        row[field] = stats
    try:
        masks = {key: np.asarray(mask, dtype=bool).copy() for key, mask in event.instance_masks.items()}
        ids = sorted(masks)
        row["instance_masks"] = {"count": len(ids), "pixels_by_id":
                                 {key: int(np.count_nonzero(masks[key])) for key in ids}}
        if ids:
            arrays["mask_ids"] = np.asarray(ids, dtype=str)
            arrays["masks"] = np.stack([masks[key] for key in ids])
    except Exception as error:
        row["instance_masks_error"] = f"{type(error).__name__}: {error}"
    raw_path = directory / f"{stem}_arrays.npz"
    np.savez_compressed(raw_path, **arrays)
    row.update(raw_arrays_path=str(raw_path), raw_arrays_sha256=file_hash(raw_path))
    return row


def probe_worker(config, connection):
    # Importing is safe here: every worker has its own process/environment.
    import ai2thor.build
    from ai2thor.controller import Controller
    from ai2thor.fifo_server import FifoServer
    from ai2thor.platform import CloudRendering

    runtime = config["runtime"]
    directory = Path(config["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    session = Path(runtime["scratch"]) / "render_probe" / config["name"] / config["trial"]
    session.mkdir(parents=True, exist_ok=False)
    metrics = {"trial": config["trial"], "status": "running", "completed": False,
               "worker_identity": process_identity(os.getpid()), "captures": [], "stages": [],
               "cleanup_errors": [], "environment_before": filtered_env(os.environ)}
    stage = "constructor"

    def publish():
        write_json(directory / "progress.json", metrics)
        connection.send(("progress", copy.deepcopy(metrics)))

    # Match thor_common's current environment for the default condition.
    os.environ["TMPDIR"] = str(session)
    os.environ["XDG_CACHE_HOME"] = str(Path(runtime["scratch"]) / "cache")
    os.environ["XDG_CONFIG_HOME"] = str(session / "config")
    changes = {}
    if config["fresh_cache"]:
        for key, child in (("TMPDIR", "tmp"), ("TEMP", "tmp"), ("TMP", "tmp"),
                           ("XDG_CACHE_HOME", "xdg_cache"), ("XDG_CONFIG_HOME", "xdg_config"),
                           ("XDG_DATA_HOME", "xdg_data"), ("CUDA_CACHE_PATH", "cuda_cache"),
                           ("__GL_SHADER_DISK_CACHE_PATH", "nvidia_shader_cache")):
            destination = session / "fresh" / child
            destination.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(destination)
            changes[key] = str(destination)
    metrics.update(environment_configured=filtered_env(os.environ), fresh_environment_changes=changes)
    (session / "cuda-vulkan-mapping.json").write_text(json.dumps({
        str(runtime["gpu_index"]): runtime["vulkan_device_index"]}) + "\n")
    metrics["source_files"] = {}
    for module_name in ("ai2thor.controller", "ai2thor.fifo_server", "ai2thor.server",
                        "ai2thor.platform", "ai2thor._quality_settings", "ai2thor.build"):
        path = Path(importlib.import_module(module_name).__file__).resolve()
        metrics["source_files"][module_name] = {"path": str(path), "sha256": file_hash(path)}
    metrics["package_versions"] = {name: importlib.metadata.version(name)
                                   for name in ("ai2thor", "numpy", "Pillow")}
    publish()

    def record_receive(server, event, previous_files):
        action = json.loads(server._last_action_message) if server._last_action_message else None
        observation = capture_event(event, directory, len(metrics["captures"]), stage, action)
        observation["cached_raw_files_mapping_reused"] = server.raw_files is previous_files
        observation["rgb_wire_payloads"] = [
            {"bytes": len(body), "nonzero_bytes_including_any_alpha": len(body) - body.count(b"\0"),
             "sha256": hashlib.sha256(body).hexdigest()}
            for body in server.raw_files.get("image", [])]
        metrics["captures"].append(observation)
        publish()

    class ProbeController(Controller):
        @property
        def base_dir(self):
            return str(session)

        @property
        def releases_dir(self):
            return str(Path(runtime["scratch"]) / "ai2thor" / "releases")

        @property
        def log_dir(self):
            return str(directory)

        def prune_releases(self):
            return None

        def unity_command(self, width, height, headless):
            return super().unity_command(width, height, headless) + ["-logFile", str(directory / "Player.log")]

        def _start_unity_thread(self, env, width, height, server_params, image_name):
            metrics["environment_before_spawn"] = filtered_env(env)
            try:
                return super()._start_unity_thread(env, width, height, server_params, image_name)
            finally:
                # Upstream mutates env with the ACTUAL Popen flags before spawn.
                metrics["environment_passed_to_unity"] = filtered_env(env)
                process = getattr(self.server, "unity_proc", None)
                if process is not None:
                    identity = process_identity(process.pid)
                    metrics["unity_identity"] = identity
                    metrics["unity_command"] = list(process.args)
                    if identity:
                        connection.send(("unity_identity", identity))
                    try:
                        proc_env = Path(f"/proc/{process.pid}/environ").read_bytes().split(b"\0")
                        values = dict(item.decode(errors="replace").split("=", 1) for item in proc_env if b"=" in item)
                        metrics["environment_observed_in_proc"] = filtered_env(values)
                        metrics["command_observed_in_proc"] = [item.decode(errors="replace") for item in
                            Path(f"/proc/{process.pid}/cmdline").read_bytes().split(b"\0") if item]
                    except OSError as error:
                        metrics["proc_observation_error"] = f"{type(error).__name__}: {error}"
                publish()

    ai2thor.build.base_url = ai2thor.build.base_url.replace("http://", "https://", 1)
    controller = ProbeController.__new__(ProbeController)
    with observe_fifo_receives(FifoServer, record_receive):
        try:
            # Keep the exact upstream FifoServer class selected by default.
            controller.__init__(scene="FloorPlan1", platform=CloudRendering,
                gpu_device=runtime["controller_gpu_device"], width=config["resolution"], height=config["resolution"],
                agentMode="default", renderDepthImage=False, renderInstanceSegmentation=True,
                gridSize=0.1, visibilityDistance=10, fieldOfView=100,
                server_timeout=15, server_start_timeout=35)
            metrics["controller"] = {"quality": controller.quality, "headless": controller.headless,
                "width": controller.width, "height": controller.height, "gpu_device": controller.gpu_device,
                "initialization_parameters": copy.deepcopy(controller.initialization_parameters),
                "unity_build_commit_id": controller._build.commit_id,
                "executable": controller._build.executable_path,
                "executable_sha256": file_hash(controller._build.executable_path),
                "fifo_server_pipe_path": controller.server.server_pipe_path,
                "fifo_client_pipe_path": controller.server.client_pipe_path}
            if controller._build.commit_id != BUILD:
                raise RuntimeError("Unexpected Unity build")
            metrics["stages"].append({"stage": "initial", "capture_index": len(metrics["captures"]) - 1})

            def act(label, action, **kwargs):
                nonlocal stage
                stage = label
                event = controller.step(action=action, **kwargs)
                metrics["stages"].append({"stage": label, "action": action, "arguments": copy.deepcopy(kwargs),
                                          "capture_index": len(metrics["captures"]) - 1})
                publish()
                if not event.metadata.get("lastActionSuccess"):
                    raise RuntimeError(f"{action}: {event.metadata.get('errorMessage')}")
                return event

            act("after_InitialRandomSpawn_seed0", "InitialRandomSpawn", randomSeed=0,
                forceVisible=False, placeStationary=True, numPlacementAttempts=5)
            event = act("after_GetReachablePositions", "GetReachablePositions")
            # Query can reuse cached pixels; its evidence is labeled accordingly.
            act("after_TeleportFull_same_pose", "TeleportFull", **agent_pose(event))
            for number in range(1, 6):
                act(f"after_Pass_{number}", "Pass")
            metrics.update(status="completed", completed=True)
        except BaseException as error:
            metrics.update(status="failed", completed=False, error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
        finally:
            signal.setitimer(signal.ITIMER_REAL, max(0.1, min(8, config["deadline"] - time.time())))
            try:
                if getattr(controller, "server", None) is not None:
                    controller.stop()
            except BaseException as error:
                metrics["cleanup_errors"].append(f"{type(error).__name__}: {error}")
                metrics.update(status="failed", completed=False)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                publish()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="t00b_render_probe")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args()
    if not re.fullmatch(r"t00b_[A-Za-z0-9_]+", args.name):
        parser.error("Use a new t00b_ name containing only letters, digits and underscores")
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 300:
        parser.error("Overall deadline must be positive and at most 300 seconds")
    directory = ROOT / "logs" / "runs" / args.name
    # Reject existing raw evidence rather than silently overwriting a failure.
    directory.mkdir(parents=True, exist_ok=False)
    config = {"name": args.name, "timeout_seconds": args.timeout_seconds, "worker_timeout_seconds": 60,
        "trials": [{"trial": trial, "resolution": resolution, "fresh_cache": fresh}
                   for trial, resolution, fresh in TRIALS],
        "notes": ["Each trial is a fresh independent worker and Unity process; shared immutable build only.",
                  "Default condition reproduces current thor_common temp/cache settings; fresh condition sets new temp/XDG/CUDA/NVIDIA-cache paths without deleting existing files.",
                  "Fresh selected directories do not prove that every internal driver cache is redirected; inspect actual Player.log paths.",
                  "All RGB bytes including black startup/cached query frames are retained; nonzero is a diagnostic observation, not a revised exact-zero criterion.",
                  "Initial means constructor completion; every preceding FIFO receive is also retained with stage=constructor.",
                  "GetReachablePositions may reuse cached images; only later rendered actions can establish a change.",
                  "Multiple environment paths change together in the fresh arm; an improvement would not identify one individual cache as the cause.",
                  "Completion means stages were observed, not that RGB is nonblack; no causal claim is made automatically."]}
    metrics = {"status": "incomplete", "diagnostic_completed": False, "trials": [], "cleanup_errors": []}
    install_signals()
    try:
        config["deadline"] = deadline_after(args.timeout_seconds)
        config["runtime"] = prepare_runtime(args.name)
        for trial, resolution, fresh in TRIALS:
            if time.time() >= config["deadline"] - 8:
                raise TimeoutError("Overall probe deadline expired before the next predeclared trial")
            settings = {**config, "trial": trial, "resolution": resolution, "fresh_cache": fresh,
                "directory": str(directory / trial), "deadline": min(time.time() + 60, config["deadline"] - 8)}
            worker = None
            observation = {"trial": trial, "status": "not_started", "completed": False}
            try:
                worker = Worker(probe_worker, settings)
                while time.time() < settings["deadline"] and worker.result is None and worker.error is None:
                    worker.receive()
                    time.sleep(0.03)
                worker.receive()
                observation = worker.result or worker.latest or observation
                if worker.error or worker.result is None:
                    observation.update(status="failed", completed=False,
                        worker_error=worker.error or {"type": "TimeoutError", "message": "60-second worker deadline"})
            finally:
                cleanup = stop_workers([worker]) if worker else []
                if cleanup:
                    observation.update(status="failed", completed=False)
                    metrics["cleanup_errors"].extend(cleanup)
                metrics["trials"].append(observation)
            if cleanup or observation.get("cleanup_errors"):
                raise RuntimeError("Cleanup failed; refusing to start another Unity process")
        metrics["diagnostic_completed"] = all(row.get("completed") for row in metrics["trials"]) and len(metrics["trials"]) == 4
        metrics["status"] = "completed" if metrics["diagnostic_completed"] else "incomplete"
    except BaseException as error:
        metrics["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        write_result(args.name, metrics, config)
    return 0 if metrics["diagnostic_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
