#!/usr/bin/env python3
"""T00 movement/render timings at 224 and VAGEN's pinned 255 pixel default."""
import argparse
import copy
import json
import math
import os
import statistics
import time
import traceback

try:
    from .thor_common import (Worker, controller_session, deadline_after, install_signals,
                              prepare_runtime, step, stop_workers, valid_frame, write_result)
except ImportError:
    from thor_common import (Worker, controller_session, deadline_after, install_signals,
                            prepare_runtime, step, stop_workers, valid_frame, write_result)


def tree_cpu_seconds():
    import psutil
    process = psutil.Process()
    total = 0.0
    for item in [process] + process.children(recursive=True):
        try:
            cpu = item.cpu_times()
            total += cpu.user + cpu.system
        except psutil.NoSuchProcess:
            pass
    return total


def prepare_movement(controller, move_magnitude):
    """Find the first legal adjacent grid edge, then validate both directions.

    Setup/warm-up is excluded from throughput. The timed actions are real
    MoveAhead/MoveBack translations, not Pass/rotation-only rendering.
    """
    points = step(controller, "GetReachablePositions").metadata["actionReturn"]
    indexed = {(round(p["x"], 3), round(p["z"], 3)): p for p in points}
    failures = []
    for key in sorted(indexed):
        for dx, dz in ((0, move_magnitude), (move_magnitude, 0)):
            neighbor = (round(key[0] + dx, 3), round(key[1] + dz, 3))
            if neighbor not in indexed:
                continue
            origin = indexed[key]
            try:
                step(controller, "TeleportFull", position=origin,
                     rotation={"x": 0, "y": math.degrees(math.atan2(dx, dz)), "z": 0},
                     horizon=0, standing=True)
                start = controller.last_event.metadata["agent"]["position"]
                forward = step(controller, "MoveAhead", moveMagnitude=move_magnitude)
                end = forward.metadata["agent"]["position"]
                if math.hypot(end["x"] - start["x"], end["z"] - start["z"]) < move_magnitude - 0.01:
                    raise RuntimeError("Warm-up did not translate the agent")
                back = step(controller, "MoveBack", moveMagnitude=move_magnitude)
                actual = back.metadata["agent"]["position"]
                if math.hypot(actual["x"] - start["x"], actual["z"] - start["z"]) > 0.01:
                    raise RuntimeError("Warm-up did not return to its starting grid point")
                return {"start_position": start, "forward_position": end,
                        "setup_failed_edges": failures}
            except TimeoutError:
                raise
            except Exception as error:
                failures.append(str(error))
                if len(failures) >= 20:
                    raise RuntimeError(f"No legal movement edge in first 20 attempts: {failures}")
    raise RuntimeError("No adjacent reachable grid points for movement benchmark")


def throughput_worker(config, connection):
    with controller_session(config, config["worker_id"], connection=connection) as controller:
        valid_frame(controller.last_event, config["resolution"])
        movement = prepare_movement(controller, config["move_magnitude"])
        connection.send(("ready", {"movement": movement, "worker_pid": os.getpid(),
                                   "unity_pid": controller.unity_pid}))
        token = connection.recv()
        if token != "go":
            raise RuntimeError("Invalid start barrier message")
        durations = []
        started = time.monotonic()
        cpu_start = tree_cpu_seconds()
        for number in range(200):
            action = "MoveAhead" if number % 2 == 0 else "MoveBack"
            before = controller.last_event.metadata["agent"]["position"]
            frame_start = time.perf_counter()
            event = step(controller, action, moveMagnitude=config["move_magnitude"])
            valid_frame(event, config["resolution"])
            after = event.metadata["agent"]["position"]
            if math.hypot(after["x"] - before["x"], after["z"] - before["z"]) < config["move_magnitude"] - 0.01:
                raise RuntimeError(f"Timed action {number} succeeded without the required translation")
            durations.append(time.perf_counter() - frame_start)
            # Preserve all measured timings if a later action hangs or crashes.
            connection.send(("progress", {"worker_pid": os.getpid(),
                "unity_pid": controller.unity_pid, "movement": movement,
                "completed_frames": len(durations), "frame_seconds": list(durations),
                "started_monotonic": started}))
        cpu_seconds = tree_cpu_seconds() - cpu_start
        ended = time.monotonic()
        elapsed = ended - started
        result = {
            "worker_pid": os.getpid(), "unity_pid": controller.unity_pid,
            "movement": movement, "completed_frames": len(durations),
            "frame_seconds": durations,
            "median_frame_seconds": statistics.median(durations),
            "median_fps": statistics.median(1.0 / duration for duration in durations),
            "wall_fps": len(durations) / elapsed,
            "started_monotonic": started, "ended_monotonic": ended,
            "measurement_wall_seconds": elapsed, "cpu_seconds": cpu_seconds,
            "cpu_average_cores": cpu_seconds / elapsed,
            "cpu_average_percent_one_core_100": 100 * cpu_seconds / elapsed,
        }
        # Publish before controller cleanup, so cleanup failure cannot erase data.
        connection.send(("progress", result))
    return result


class ResourceSampler:
    """Direct NVML graphics-process memory plus psutil process-tree RSS."""
    def __init__(self, uuid, workers):
        import psutil
        import pynvml
        self.psutil = psutil
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        self.workers = workers
        self.observations = [{"gpu_memory_peak_bytes": None, "rss_peak_bytes": None,
                              "gpu_memory_samples": 0, "rss_samples": 0} for _ in workers]
        self.samples = 0
        self.device_memory_peak_bytes = None
        self.errors = set()

    def sample(self):
        nvml = self.nvml
        memory = nvml.nvmlDeviceGetMemoryInfo(self.handle).used
        self.device_memory_peak_bytes = max(self.device_memory_peak_bytes or 0, memory)
        by_pid = {}
        query_succeeded = False
        for getter in (nvml.nvmlDeviceGetGraphicsRunningProcesses, nvml.nvmlDeviceGetComputeRunningProcesses):
            try:
                entries = getter(self.handle)
                query_succeeded = True
                for entry in entries:
                    used = entry.usedGpuMemory
                    if used is None or not isinstance(used, int) or used < 0 or used >= 2**63:
                        continue
                    by_pid[entry.pid] = max(by_pid.get(entry.pid, 0), used)
            except nvml.NVMLError as error:
                self.errors.add(f"{getter.__name__}: {error}")
        for worker, observation in zip(self.workers, self.observations):
            try:
                process = self.psutil.Process(worker.process.pid)
                tree = [process] + process.children(recursive=True)
                pids = {p.pid for p in tree}
                rss = 0
                for process in tree:
                    try:
                        rss += process.memory_info().rss
                    except self.psutil.NoSuchProcess:
                        pass
                observation["rss_peak_bytes"] = max(observation["rss_peak_bytes"] or 0, rss)
                observation["rss_samples"] += 1
                seen = pids & set(by_pid)
                if query_succeeded and seen:
                    used = sum(by_pid[pid] for pid in seen)
                    observation["gpu_memory_peak_bytes"] = max(observation["gpu_memory_peak_bytes"] or 0, used)
                    observation["gpu_memory_samples"] += 1
            except self.psutil.NoSuchProcess:
                pass
        self.samples += 1

    def close(self):
        self.nvml.nvmlShutdown()


def run_trial(config, k, trial_kind, trial_deadline, publish):
    workers = []
    sampler = None
    row = {"resolution": config["resolution"], "concurrency": k, "kind": trial_kind,
           "requested_frames_per_instance": 200, "status": "running", "instances": []}
    try:
        for index in range(k):
            worker_config = dict(config, deadline=trial_deadline,
                                 worker_id=f"r{config['resolution']}_{trial_kind}_k{k}_i{index}")
            workers.append(Worker(throughput_worker, worker_config))
        while not all(worker.ready for worker in workers):
            for worker in workers:
                worker.receive()
                if worker.error:
                    raise RuntimeError(json.dumps(worker.error))
            if time.time() >= trial_deadline:
                raise TimeoutError("Trial initialization deadline exceeded")
            time.sleep(0.1)
        sampler = ResourceSampler(config["runtime"]["gpu_uuid"], workers)
        sampler.sample()
        start = time.monotonic()
        for worker in workers:
            worker.connection.send("go")
        next_sample = start
        while True:
            for worker in workers:
                worker.receive()
            row["instances"] = [copy.deepcopy(worker.latest) for worker in workers]
            if time.monotonic() >= next_sample:
                sampler.sample()
                next_sample = time.monotonic() + 0.5
            publish(row)
            if any(worker.error for worker in workers):
                raise RuntimeError(json.dumps([worker.error for worker in workers if worker.error]))
            if all(worker.result is not None for worker in workers):
                break
            if time.time() >= trial_deadline:
                raise TimeoutError("Trial measurement deadline exceeded")
            time.sleep(0.05)
        elapsed = max(worker.result["ended_monotonic"] for worker in workers) - start
        total_frames = sum(worker.result["completed_frames"] for worker in workers)
        row.update(completed_frames=total_frames, concurrent_wall_seconds=elapsed,
                   total_fps=total_frames / elapsed, status="passed")
    except (KeyboardInterrupt, SystemExit) as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    except Exception as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        for worker in workers:
            worker.receive()
        row["instances"] = [copy.deepcopy(worker.latest) for worker in workers]
        if sampler:
            for instance, observation in zip(row["instances"], sampler.observations):
                instance.update(observation)
            row.update(resource_sampling_interval_seconds=0.5, resource_samples=sampler.samples,
                       device_memory_peak_bytes=sampler.device_memory_peak_bytes,
                       resource_query_errors=sorted(sampler.errors))
            if any(item["gpu_memory_peak_bytes"] is None for item in sampler.observations):
                row["status"] = "failed"
                row["resource_error"] = "No measured NVIDIA process-memory value for one or more instances"
            try:
                sampler.close()
            except Exception as error:
                row["status"] = "failed"
                row["resource_cleanup_error"] = str(error)
        cleanup_errors = stop_workers(workers)
        if cleanup_errors:
            row.update(status="failed", cleanup_errors=cleanup_errors)
        publish(row)
        print("THROUGHPUT_TRIAL: " + json.dumps(row, sort_keys=True), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="t00_thor_throughput")
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--trial-timeout-seconds", type=float, default=360)
    args = parser.parse_args()
    config = vars(args).copy()
    config.update(resolutions=[224, 255], concurrency_levels=[1, 4, 8, 11], frames_per_instance=200,
                  vagen_commit="4e2976d87c0f92c97f9a820803e71908cde5fb71",
                  official_navigation_resolution=255,
                  quality="Ultra (ai2thor 5.0.0 default)", segmentation=False,
                  grid_size=0.1, visibility_distance=10, field_of_view=100, move_magnitude=0.5,
                  server_timeout_seconds=45, server_start_timeout_seconds=90,
                  timing_scope="movement action through synchronous RGB receipt and frame validation",
                  cpu_scope="Python worker plus recursive Unity descendants, user+system CPU seconds / measurement walltime",
                  gpu_memory_scope="sampled peak NVML graphics/compute process-memory bytes per worker tree; device total also recorded")
    metrics = {"status": "failed", "trials": []}
    install_signals()
    try:
        if args.timeout_seconds <= 0 or args.trial_timeout_seconds <= 0:
            raise ValueError("Timeouts must be positive")
        config["deadline"] = deadline_after(args.timeout_seconds)
        config["runtime"] = prepare_runtime(args.name)
        for resolution in config["resolutions"]:
            for kind, k in [("single", 1)] + [("concurrent", k) for k in config["concurrency_levels"]]:
                if time.time() >= config["deadline"]:
                    raise TimeoutError("Overall throughput deadline exceeded")
                metrics["trials"].append({})

                def publish(row):
                    metrics["trials"][-1] = copy.deepcopy(row)

                trial_config = dict(config, resolution=resolution)
                trial_deadline = min(config["deadline"], time.time() + args.trial_timeout_seconds)
                row = run_trial(trial_config, k, kind, trial_deadline, publish)
                if row.get("cleanup_errors"):
                    raise RuntimeError("Worker cleanup was incomplete or ownership uncertain; stopping further trials")
        metrics["status"] = "passed" if all(row["status"] == "passed" for row in metrics["trials"]) else "failed"
    except BaseException as error:
        metrics.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        write_result(args.name, metrics, config)
    return 0 if metrics["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
