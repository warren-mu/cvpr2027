"""One logged PBS job: gate every allocated GPU, then collect disjoint T01 scenes.

Queue, billing multiplier and submission are deliberately outside this program.
The explicit total deadline includes the gate; allow additional PBS cleanup time.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

from src.envs.thor_common import ROOT, write_result
from src.tools.t01_partition import (DEFAULT_MANIFEST, audited_result, cpu_shards,
                                    ensure_unused, load_manifest, scene_shards, sha_file)


def request_stop_logged(process):
    import psutil
    if process.poll() is None:
        try:
            for child in psutil.Process(process.pid).children():
                child.terminate()
        except psutil.NoSuchProcess:
            pass


def stop_logged(process, request=True, grace_deadline=None, hard_deadline=None):
    """Ask the experiment to clean up while its run_logged parent remains alive."""
    import psutil
    errors = []
    if process.poll() is not None:
        return errors
    try:
        wrapper = psutil.Process(process.pid)
        if request:
            request_stop_logged(process)
        until = grace_deadline if grace_deadline is not None else time.monotonic() + 20
        hard = hard_deadline if hard_deadline is not None else until + 8
        remaining = lambda maximum: max(0.0, min(maximum, hard - time.monotonic()))
        while process.poll() is None and time.monotonic() < until:
            time.sleep(0.1)
        if process.poll() is None:
            # Last resort is confined to this observed owned process tree.
            owned = wrapper.children(recursive=True)
            for child in reversed(owned):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(owned, timeout=remaining(2))
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=remaining(2))
            if alive:
                errors.append("Owned processes survived scoped termination")
            try:
                process.wait(timeout=remaining(2))
            except subprocess.TimeoutExpired:
                wrapper.terminate()
                try:
                    process.wait(timeout=remaining(2))
                except subprocess.TimeoutExpired:
                    wrapper.kill()
                errors.append("run_logged could not finalize after child termination")
    except (psutil.Error, subprocess.TimeoutExpired) as error:
        errors.append(f"{type(error).__name__}: {error}")
    return errors


def launch_logged(name, command, environment, directory):
    ensure_unused(name)
    path = directory / f"{name}.launcher.log"
    stream = path.open("x")
    env = dict(environment, CVPR_RUN_NAME=name, CVPR_RESULTS_PATH=str(ROOT / "results" / f"{name}.json"))
    argv = [sys.executable, "-u", str(ROOT / "scripts/run_logged.py"), "--name", name, "--seed", "0", "--", *command]
    try:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    except BaseException:
        stream.close()
        raise
    return {"process": process, "stream": stream, "launcher_log": path, "name": name,
            "start_monotonic": time.monotonic(), "command": argv}


def finish_logged(running, sources):
    process = running["process"]
    if process.poll() is None:
        raise RuntimeError("Cannot summarize an unfinished logged process")
    running["stream"].close()
    record = {"run_name": running["name"], "exit_code": process.returncode,
              "wall_seconds_observed": time.monotonic() - running["start_monotonic"],
              "result_path": f"results/{running['name']}.json",
              "launcher_log": str(running["launcher_log"]),
              "launcher_log_sha256": sha_file(running["launcher_log"])}
    data = None
    try:
        data, reference = audited_result(record["result_path"])
        if data["_meta"]["exit_code"] != process.returncode:
            raise ValueError("Recorded source exit differs from observed child exit")
        sources.append(reference)
        record["source"] = reference
        record["source_status"] = data["metrics"].get("status")
    except (OSError, ValueError, KeyError, TypeError) as error:
        record["provenance_error"] = f"{type(error).__name__}: {error}"
    return record, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--partition", choices=("a", "b"), required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--scene-timeout-seconds", type=float, required=True)
    parser.add_argument("--gate-timeout-seconds", type=float, default=180)
    args = parser.parse_args()
    if not re.fullmatch(r"pilot_pairs_[A-Za-z0-9_]+", args.name):
        parser.error("Use a new pilot_pairs_ run name")
    if (not all(math.isfinite(value) and value > 0 for value in
                (args.timeout_seconds, args.scene_timeout_seconds, args.gate_timeout_seconds))
            or not 15 <= args.gate_timeout_seconds <= 600
            or args.scene_timeout_seconds > 4 * 3600 - 120
            or args.timeout_seconds <= args.gate_timeout_seconds + 60):
        parser.error("Require finite explicit deadlines and time for gate plus cleanup")
    output = ROOT / "results" / f"{args.name}.json"
    if output.exists():
        raise FileExistsError("Existing collection evidence is preserved")
    metrics = {"kind": "t01_parallel_collect", "status": "failed_before_gate", "acceptance_passed": False,
               "scene_runs": [], "started_shards": [], "gate": None, "cleanup_errors": []}
    config = {"name": args.name, "partition": args.partition, "timeout_seconds": args.timeout_seconds,
              "scene_timeout_seconds": args.scene_timeout_seconds, "gate_timeout_seconds": args.gate_timeout_seconds,
              "node": socket.gethostname(), "pbs_jobid": os.environ.get("PBS_JOBID"),
              "collect_only": False, "acceptance_policy": "t01_human_20261108", "seed": 0, "instances_per_gpu": 1, "cpu_cores_per_gpu": 12,
              "notes": ["Fixed disjoint scene shards; no scene replacement or result-based reassignment.",
                        "One Unity per GPU is deliberate; multi-instance contention is not assumed beneficial.",
                        "Source failures and unstarted scenes remain visible; collection completion is not scientific acceptance.",
                        "Timeouts are execution caps; queue, billing and resubmission belong to the submitter.",
                        "Parent PBS CUDA mask is never changed; each scene child receives one original allocated UUID."]}
    sources, active, gate_running = [], {}, None
    stopping = []
    code = 1
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stopping.append(f"signal:{signum}"))
    deadline = time.monotonic() + args.timeout_seconds
    try:
        manifest, identity = load_manifest(args.manifest)
        config["manifest"] = identity
        if not config["pbs_jobid"] or not os.environ.get("CVPR_SCRATCH"):
            raise RuntimeError("PBS_JOBID and CVPR_SCRATCH are required")
        parent_mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        parent_uuids = parent_mask.split(",")
        queues = scene_shards(manifest, args.partition)
        if (len(parent_uuids) != len(queues) or len(set(parent_uuids)) != len(queues)
                or any(not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid) for uuid in parent_uuids)):
            raise ValueError("Require the original complete PBS full-UUID mask with the declared GPU count")
        cpus = cpu_shards(os.sched_getaffinity(0), len(queues))
        config.update(parent_cuda_visible_devices=parent_mask, cpu_shards=cpus,
                      scene_shards=queues, pilot_scenes=manifest["partitions"][args.partition]["scenes"])
        job_tag = re.sub(r"[^A-Za-z0-9]", "_", config["pbs_jobid"])
        scratch = Path(os.environ["CVPR_SCRATCH"]) / "t01_parallel" / args.name / job_tag
        scratch.mkdir(parents=True, exist_ok=False)
        gate_name = f"{args.name}_j{job_tag}_gate"
        gate_running = launch_logged(gate_name, [sys.executable, "-u", "-m", "src.tools.render_gate",
            "--name", gate_name, "--gpus", str(len(queues)), "--resolution", "224", "--segmentation",
            "--timeout-seconds", str(args.gate_timeout_seconds)], os.environ, scratch)
        gate_deadline = min(deadline, time.monotonic() + args.gate_timeout_seconds + 30)
        while gate_running["process"].poll() is None and not stopping and time.monotonic() < gate_deadline:
            time.sleep(0.1)
        if gate_running["process"].poll() is None:
            metrics["cleanup_errors"].extend(stop_logged(gate_running["process"]))
        gate_record, gate = finish_logged(gate_running, sources)
        gate_running = None
        metrics["gate"] = gate_record
        if gate_record["exit_code"] != 0 or gate_record.get("provenance_error") or stopping:
            code = 86 if gate_record["exit_code"] == 86 and not metrics["cleanup_errors"] and not stopping else 1
            metrics["status"] = "blocked_by_render_gate"
        else:
            devices = gate["config"]["allocation"]["devices"]
            if ({item["gpu_uuid"] for item in devices} != set(parent_uuids)
                    or gate["metrics"].get("passed") is not True):
                raise ValueError("Gate identities do not exactly match the original allocation")
            config["allocation"] = gate["config"]["allocation"]
            positions = [0] * len(queues)
            black_detected = False
            metrics["status"] = "collecting"
            while True:
                if time.monotonic() >= deadline - 30:
                    stopping.append("collection_deadline")
                for logical in range(len(queues)):
                    if logical in active and active[logical]["process"].poll() is not None:
                        running = active.pop(logical)
                        record, data = finish_logged(running, sources)
                        record.update(scene=running["scene"], logical_index=logical, gpu_uuid=parent_uuids[logical],
                                      granted_timeout_seconds=running["granted_timeout_seconds"])
                        metrics["scene_runs"].append(record)
                        print("T01_SHARD_FINISHED: " + json.dumps(record, sort_keys=True), flush=True)
                        if record["exit_code"] == 86:
                            black_detected = True
                            stopping.append("measured_black_frame")
                    if logical not in active and not stopping and positions[logical] < len(queues[logical]):
                        scene = queues[logical][positions[logical]]
                        positions[logical] += 1
                        name = f"{args.name}_j{job_tag}_g{logical}_{scene.lower()}"
                        granted = min(args.scene_timeout_seconds, deadline - time.monotonic() - 30)
                        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=parent_uuids[logical],
                                           T01_PARENT_CUDA_VISIBLE_DEVICES=parent_mask)
                        running = launch_logged(name, [sys.executable, "-u", "-m", "src.envs.pair_shard_approved",
                            "--name", name, "--manifest", str(args.manifest.resolve()), "--partition", args.partition,
                            "--scene", scene, "--gpu-uuid", parent_uuids[logical], "--cpu-list", ",".join(map(str, cpus[logical])),
                            "--gate-result", gate_record["result_path"], "--timeout-seconds", str(granted)], environment, scratch)
                        running.update(scene=scene, granted_timeout_seconds=granted)
                        active[logical] = running
                        metrics["started_shards"].append({"scene": scene, "run_name": name,
                            "logical_index": logical, "gpu_uuid": parent_uuids[logical], "wrapper_pid": running["process"].pid})
                if stopping or (not active and all(positions[i] == len(queues[i]) for i in range(len(queues)))):
                    break
                time.sleep(0.1)
            code = 86 if black_detected else (1 if stopping or any(row["exit_code"] or row.get("provenance_error") for row in metrics["scene_runs"]) else 0)
            metrics["status"] = "stopped" if stopping else ("completed" if code == 0 else "completed_with_failures")
    except BaseException as error:
        metrics.update(status="orchestration_failed", error=f"{type(error).__name__}: {error}")
        code = 1
        traceback.print_exc()
    finally:
        cleanup_started = time.monotonic()
        grace_deadline, hard_deadline = cleanup_started + 20, cleanup_started + 28
        to_stop = list(active.values()) + ([gate_running] if gate_running is not None else [])
        for running in to_stop:
            try:
                request_stop_logged(running["process"])
            except Exception as error:
                metrics["cleanup_errors"].append(f"request stop {running['name']}: {error}")
        if gate_running is not None:
            metrics["cleanup_errors"].extend(stop_logged(gate_running["process"], request=False,
                grace_deadline=grace_deadline, hard_deadline=hard_deadline))
            if gate_running["process"].poll() is not None:
                metrics["gate"], _ = finish_logged(gate_running, sources)
        for logical, running in active.items():
            metrics["cleanup_errors"].extend(stop_logged(running["process"], request=False,
                grace_deadline=grace_deadline, hard_deadline=hard_deadline))
            if running["process"].poll() is not None:
                record, _ = finish_logged(running, sources)
                record.update(scene=running["scene"], logical_index=logical,
                              gpu_uuid=config["parent_cuda_visible_devices"].split(",")[logical],
                              stopped_by_orchestrator=True, granted_timeout_seconds=running["granted_timeout_seconds"])
                metrics["scene_runs"].append(record)
            else:
                running["stream"].close()
                metrics["cleanup_errors"].append(f"Unfinished run_logged wrapper: {running['name']}")
        observed = {row["scene"] for row in metrics["scene_runs"]}
        started = {row["scene"] for row in metrics["started_shards"]}
        metrics["unstarted_scenes"] = [scene for scene in config.get("pilot_scenes", []) if scene not in started]
        metrics["unfinished_scenes"] = sorted(started - observed)
        metrics["stop_reasons"] = stopping
        metrics["all_planned_scenes_finished"] = bool(config.get("pilot_scenes")) and not metrics["unstarted_scenes"] and not metrics["unfinished_scenes"] and not stopping
        metrics["per_gpu_started_scene_counts"] = {str(index): sum(row.get("logical_index") == index for row in metrics["started_shards"])
                                                  for index in range(len(config.get("scene_shards", [])))}
        if metrics["cleanup_errors"]:
            code = 1
        metrics["exit_code"] = code
        write_result(args.name, metrics, config)
        output.write_text(json.dumps({"metrics": metrics, "config": config, "sources": sources}, indent=2, allow_nan=False) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
