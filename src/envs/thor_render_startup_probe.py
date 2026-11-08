"""Same-allocation T00b startup comparison, reusing both existing probe workers.

No shared source or installed package is edited. The original successful probe
is called directly, including its spawn/proc observer and receive instrumentation.
Only the named delay arm temporarily delays return from Controller startup.
"""
import argparse
import copy
import math
from pathlib import Path
import re
import time
import traceback

from .thor_common import (ROOT, Worker, deadline_after, install_signals,
                          prepare_runtime, stop_workers, write_result)
from .thor_render_probe import file_hash, probe_worker
from .thor_render_timing_probe import timing_worker

ARMS = ("common_baseline", "original_successful_probe", "common_pre_receive_2s", "common_repeat")


def startup_worker(config, connection):
    arm = config["startup_arm"]
    if arm == "original_successful_probe":
        # Do not reimplement or omit any part of the previously successful arm.
        return probe_worker(config, connection)
    if arm == "common_pre_receive_2s":
        from ai2thor.controller import Controller
        original = Controller._start_unity_thread

        def delayed_start(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            # Only after normal upstream return; no receive/action/reset added.
            time.sleep(2.0)
            return result

        Controller._start_unity_thread = delayed_start
        try:
            return timing_worker(config, connection)
        finally:
            Controller._start_unity_thread = original
    if arm in ("common_baseline", "common_repeat"):
        return timing_worker(config, connection)
    raise ValueError(f"Unknown predeclared startup arm: {arm}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="t00b_render_startup_probe")
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
        "arms": list(ARMS), "timeout_seconds": args.timeout_seconds,
        "worker_timeout_seconds": 75, "pre_receive_delay_seconds": 2.0, "seed": 0,
        "reused_sources": {name: {"path": str(Path(__file__).with_name(name)),
                                   "sha256": file_hash(Path(__file__).with_name(name))}
                           for name in ("thor_common.py", "thor_render_probe.py", "thor_render_timing_probe.py")},
        "notes": ["Fixed order, independent worker and Unity per arm, one unchanged PBS GPU allocation.",
                  "common_baseline/common_repeat directly reuse timing_worker with its original baseline arm settings.",
                  "original_successful_probe directly calls probe_worker with trial=224_default and fresh_cache=False; its full original spawn/proc observation, receive observer and action sequence remain intact.",
                  "common_pre_receive_2s only wraps Controller._start_unity_thread in this private process, sleeps two seconds after normal original return, and otherwise calls the same timing_worker baseline.",
                  "The three common arms use the timing probe's five stages; the original-probe arm retains its distinct existing stages/captures schema and sequence.",
                  "Each arms[].observations dictionary preserves the original worker output schema, including partial observations and black frames; completion is separate from nonblack status.",
                  "Per-arm log roots keep common_baseline and the delayed baseline Unity logs separate.",
                  "The fixed sequential order and shared cache history limit causal conclusions; nonblack/black differences cannot by themselves identify a node, cache, driver or startup mechanism.",
                  "If the original probe also returns black on this allocation, previous receive-observer I/O is not sufficient; this does not establish a node root cause.",
                  "No shared helper, action flag, rendering backend, quality setting, seed, or exact-zero research criterion is changed."]}
    metrics = {"status": "incomplete", "diagnostic_completed": False, "arms": [], "cleanup_errors": []}
    install_signals()
    try:
        config["deadline"] = deadline_after(args.timeout_seconds)
        config["runtime"] = prepare_runtime(args.name)
        for arm in ARMS:
            if time.time() >= config["deadline"] - 24:
                raise TimeoutError("Overall deadline cannot accommodate the next trial and cleanup")
            arm_directory = directory / arm
            runtime = {**config["runtime"], "log_dir": str(arm_directory)}
            settings = {**config, "startup_arm": arm, "runtime": runtime,
                        "directory": str(arm_directory),
                        "deadline": min(time.time() + 75, config["deadline"] - 4),
                        "arm": "baseline_repeat" if arm == "common_repeat" else "baseline_without_receive_observer",
                        "trial": "224_default", "fresh_cache": False}
            worker = None
            record = {"arm": arm, "status": "not_started", "completed": False,
                      "worker_function": "probe_worker" if arm == "original_successful_probe" else "timing_worker",
                      "worker_deadline": settings["deadline"], "directory": str(arm_directory),
                      "unity_log_root": runtime["log_dir"], "observations": {}}
            try:
                worker = Worker(startup_worker, settings)
                while time.time() < settings["deadline"] and worker.result is None and worker.error is None:
                    worker.receive()
                    time.sleep(0.03)
                worker.receive()
                record["observations"] = copy.deepcopy(worker.result if worker.result is not None else worker.latest)
                record["worker_result_received"] = worker.result is not None
                record["completed"] = worker.result is not None and bool(record["observations"].get("completed"))
                if worker.error or worker.result is None:
                    record["completed"] = False
                    record["worker_error"] = worker.error or {"type": "TimeoutError", "message": "75-second worker deadline"}
            finally:
                if worker is not None and not record["observations"] and worker.latest:
                    # Preserve already received progress even if the parent was
                    # interrupted before normal result collection completed.
                    record["observations"] = copy.deepcopy(worker.latest)
                cleanup = stop_workers([worker]) if worker else []
                if cleanup:
                    record["completed"] = False
                    record["cleanup_errors"] = cleanup
                    metrics["cleanup_errors"].extend(cleanup)
                record["status"] = "completed" if record["completed"] else "incomplete"
                metrics["arms"].append(record)
            if cleanup or record["observations"].get("cleanup_errors"):
                raise RuntimeError("Cleanup ownership uncertain; refusing another Unity process")
        metrics["diagnostic_completed"] = len(metrics["arms"]) == len(ARMS) and all(row["completed"] for row in metrics["arms"])
        metrics["status"] = "completed" if metrics["diagnostic_completed"] else "incomplete"
    except BaseException as error:
        metrics["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        write_result(args.name, metrics, config)
    return 0 if metrics["diagnostic_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
