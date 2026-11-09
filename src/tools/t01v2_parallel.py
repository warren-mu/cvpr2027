"""Explicit T01v2 scene shards within one budgeted PBS allocation.

A successful orchestration is not pilot acceptance. Sources and unstarted scenes
remain explicit; no result-based scene substitution, collision-rule change, or
GPU accounting calculation belongs here.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time
import traceback

from src.envs.thor_common import ROOT, write_result
from src.tools.t01_partition import cpu_shards, sha_file
from src.tools.t01_parallel_approved import (finish_logged, launch_logged,
                                            request_stop_logged, stop_logged)


def scene_plan(scenes, gpus):
    permitted = {f"FloorPlan{i}" for i in (*range(1, 31), *range(201, 231))}
    if (type(gpus) is not int or gpus not in (1, 2, 3, 4) or not isinstance(scenes, list)
            or len(scenes) < gpus or any(not isinstance(scene, str) or scene not in permitted for scene in scenes)
            or len(set(scenes)) != len(scenes)):
        raise ValueError("Require a fixed nonduplicate list of pilot scenes, at least one per allocated GPU")
    return [scenes[index::gpus] for index in range(gpus)]


def validate_mask(mask, gpus):
    uuids = mask.split(",")
    if (len(uuids) != gpus or len(set(uuids)) != gpus
            or any(not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", item) for item in uuids)):
        raise ValueError("Require the original PBS full-UUID mask with exactly the requested GPUs")
    return uuids


def submission_identity(repo, name, job, gpus, timeout):
    """Require the same immutable submitter reservation; never infer free work."""
    matches = []
    for path in (repo / "logs/pbs").glob("*/payload.json"):
        data = json.loads(path.read_text())
        if data.get("name") != name or data.get("dry_run"):
            continue
        receipt = json.loads(path.with_name("submission.json").read_text())
        ids = re.findall(r"\b\d+\.hopper-m-\d+\b", receipt.get("stdout", ""))
        if (ids != [job] or data.get("budget_group") != "T01v2" or data.get("task_id") != "T01"
                or data.get("budget_batch") != "t00b_t01_20261108" or data.get("queue") != "autox"
                or data.get("requested_host") is not None or data.get("requested_gpu_count") != gpus
                or not isinstance(data.get("requested_wall_seconds"), int)
                or timeout > data["requested_wall_seconds"] - 30):
            raise ValueError("T01v2 allocation differs from its budgeted unpinned submission/deadline")
        matches.append({"file": str(path.relative_to(repo)), "sha256": sha_file(path),
                        "requested_wall_seconds": data["requested_wall_seconds"], "budget_group": "T01v2"})
    if len(matches) != 1:
        raise ValueError("Expected exactly one immutable T01v2 reservation for this PBS job/name")
    return matches[0]


def shard_configuration(name, scene, logical, cpu_set, gate, gate_ref, gate_path, scenes, raw, *, field_of_view=100):
    if type(field_of_view) is not int or field_of_view not in (60, 100):
        raise ValueError('Require an explicit 60/100 camera field of view')
    allocation = gate["config"]["allocation"]
    device = allocation["devices"][logical]
    return {"schema_version": 1, "name": name, "scene": scene, "seed": 0, "resolution": 255,
            "field_of_view": field_of_view,
            "logical_index": logical, "gpu_identity": device, "pbs_jobid": allocation["pbs_jobid"],
            "node": gate["config"]["node"], "parent_cuda_visible_devices": allocation["cuda_visible_devices"],
            "child_cuda_visible_devices": device["gpu_uuid"], "cpu_affinity": cpu_set,
            "gate_source": gate_ref, "gate_result": gate_path, "allocation": allocation,
            "scene_plan": scenes, "raw_evidence_dir": str(raw)}


def verify_shard_result(record, data, expected):
    if data is None or record.get("provenance_error"):
        raise ValueError("Shard result has no complete audited source")
    meta, metrics, shard = data.get("_meta", {}), data.get("metrics", {}), data.get("config", {}).get("shard", {})
    from src.envs.pair_v2_fov import bind_source_fields, declared_fields
    actual_fov = bind_source_fields(data.get('config', {}), shard)
    if actual_fov != declared_fields(expected):
        raise ValueError('Shard camera differs from the planned FOV assignment')
    keys = ("name", "scene", "seed", "resolution", "logical_index", "gpu_identity", "pbs_jobid", "node",
            "parent_cuda_visible_devices", "child_cuda_visible_devices", "cpu_affinity", "gate_source")
    if (meta.get("name") != expected["name"] or meta.get("exit_code") != record["exit_code"]
            or metrics.get("kind") != "pair_v2_shard" or any(shard.get(key) != expected[key] for key in keys)):
        raise ValueError("Shard result does not match the fixed scene/GPU/gate assignment")
    if metrics.get("cleanup_errors") != []:
        raise ValueError("Shard cleanup is missing or failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpus", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--scene-timeout-seconds", type=float, required=True)
    parser.add_argument("--gate-timeout-seconds", type=float, default=180)
    parser.add_argument("--field-of-view", type=int, choices=(60, 100), default=100)
    args = parser.parse_args()
    if not re.fullmatch(r"pilot_pairs_v2_[A-Za-z0-9_]+", args.name):
        parser.error("Use a new pilot_pairs_v2_ name; final pilot aggregation has its own name")
    values = (args.timeout_seconds, args.scene_timeout_seconds, args.gate_timeout_seconds)
    if (not all(math.isfinite(value) and value > 0 for value in values)
            or not 15 <= args.gate_timeout_seconds <= 600
            or args.timeout_seconds <= args.gate_timeout_seconds + 60
            or args.scene_timeout_seconds > args.timeout_seconds - 30):
        parser.error("Require finite deadlines with time for the gate and bounded cleanup")
    queues = scene_plan(args.scenes, args.gpus)
    output = ROOT / "results" / f"{args.name}.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserve existing v2 result; use a unique name")
    config = {"name": args.name, "pbs_jobid": os.environ.get("PBS_JOBID"), "node": socket.gethostname(),
              "budget_group": "T01v2", "budget_limit_charged_gpu_hours": 8, "phase_limit_charged_gpu_hours": 60,
              "resolution": 255, "seed": 0, "pilot_scenes": args.scenes, "scene_shards": queues,
              "field_of_view": args.field_of_view,
              "gpus": args.gpus, "cpu_cores_per_gpu": 12, "scene_shard_processes_per_gpu": 1,
              "instances_per_gpu": 2, "instance_scope": "one construction controller plus one sequential independent replay; never two replay workers concurrently",
              "timeout_seconds": args.timeout_seconds, "scene_timeout_seconds": args.scene_timeout_seconds,
              "gate_timeout_seconds": args.gate_timeout_seconds}
    metrics = {"kind": "pair_v2_parallel", "status": "failed_before_gate", "acceptance_passed": False,
               "gate": None, "scene_runs": [], "started_shards": [], "cleanup_errors": [],
               "black_frame_detected": False}
    sources, active, stopping = [], {}, []
    gate_running, code = None, 1
    deadline = time.monotonic() + args.timeout_seconds
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stopping.append(f"signal:{signum}"))
    try:
        if not config["pbs_jobid"] or not os.environ.get("CVPR_SCRATCH"):
            raise RuntimeError("PBS_JOBID and scratch are mandatory; no login-node work")
        config["submission"] = submission_identity(ROOT, args.name, config["pbs_jobid"], args.gpus, args.timeout_seconds)
        parent_mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        uuids = validate_mask(parent_mask, args.gpus)
        cpus = cpu_shards(os.sched_getaffinity(0), args.gpus)
        config.update(parent_cuda_visible_devices=parent_mask, cpu_shards=cpus)
        tag = re.sub(r"[^A-Za-z0-9]", "_", config["pbs_jobid"])
        scratch = Path(os.environ["CVPR_SCRATCH"]) / "t01v2_parallel" / args.name / tag
        scratch.mkdir(parents=True, exist_ok=False)
        gate_name = f"{args.name}_j{tag}_gate"
        gate_running = launch_logged(gate_name, [sys.executable, "-u", "-m", "src.tools.render_gate",
            "--name", gate_name, "--gpus", str(args.gpus), "--resolution", "255", "--segmentation",
            "--timeout-seconds", str(args.gate_timeout_seconds)], os.environ, scratch)
        gate_deadline = min(deadline - 30, time.monotonic() + args.gate_timeout_seconds + 10)
        while gate_running["process"].poll() is None and not stopping and time.monotonic() < gate_deadline:
            time.sleep(0.1)
        if gate_running["process"].poll() is None:
            metrics["cleanup_errors"].extend(stop_logged(gate_running["process"]))
        record, gate = finish_logged(gate_running, sources)
        metrics["gate"], gate_running = record, None
        if record["exit_code"] or record.get("provenance_error") or stopping:
            code = 86 if record["exit_code"] == 86 and not record.get("provenance_error") and not metrics["cleanup_errors"] and not stopping else 1
            metrics.update(status="blocked_by_render_gate", black_frame_detected=code == 86)
        else:
            allocation = gate["config"]["allocation"]
            if (gate["metrics"].get("passed") is not True or gate["metrics"].get("cleanup_errors") != []
                    or allocation.get("pbs_jobid") != config["pbs_jobid"]
                    or allocation.get("cuda_visible_devices") != parent_mask
                    or [row["gpu_uuid"] for row in allocation["devices"]] != uuids
                    or len(gate["metrics"].get("devices", [])) != args.gpus
                    or any(len(row.get("observation", {}).get("captures", [])) != 2
                           or any(capture.get("segmentation", {}).get("valid") is not True
                                  for capture in row.get("observation", {}).get("captures", []))
                           for row in gate["metrics"].get("devices", []))
                    or gate["config"].get("node") != config["node"]
                    or gate["config"].get("resolution") != 255 or gate["config"].get("segmentation") is not True):
                raise ValueError("Gate does not verify this exact PBS allocation and v2 render configuration")
            config["allocation"] = allocation
            positions = [0] * args.gpus
            metrics["status"] = "collecting"
            while True:
                if time.monotonic() >= deadline - 30:
                    stopping.append("collection_deadline")
                for logical in range(args.gpus):
                    running = active.get(logical)
                    if running and running["process"].poll() is None and time.monotonic() >= running["child_deadline"]:
                        stopping.append("scene_deadline:" + running["scene"])
                    if running and running["process"].poll() is not None:
                        active.pop(logical)
                        child_record, data = finish_logged(running, sources)
                        child_record.update(scene=running["scene"], logical_index=logical, gpu_uuid=uuids[logical],
                                            granted_timeout_seconds=running["granted_timeout_seconds"])
                        metrics["scene_runs"].append(child_record)
                        print("T01V2_SHARD_FINISHED: " + json.dumps(child_record, sort_keys=True), flush=True)
                        try:
                            verify_shard_result(child_record, data, running["shard"])
                        except ValueError as error:
                            child_record["provenance_error"] = str(error)
                            metrics["cleanup_errors"].append("Unverified shard identity/cleanup: " + running["name"])
                            stopping.append("shard_cleanup_uncertain")
                        if child_record["exit_code"] == 86:
                            metrics["black_frame_detected"] = True
                            stopping.append("measured_black_frame")
                    if logical not in active and not stopping and positions[logical] < len(queues[logical]):
                        scene = queues[logical][positions[logical]]
                        positions[logical] += 1
                        name = f"{args.name}_j{tag}_g{logical}_{scene.lower()}"
                        granted = min(args.scene_timeout_seconds, deadline - time.monotonic() - 30)
                        if granted <= 0:
                            stopping.append("collection_deadline")
                            break
                        raw = scratch / scene
                        raw.mkdir(exist_ok=False)
                        shard = shard_configuration(name, scene, logical, cpus[logical], gate, record["source"],
                                                    record["result_path"], args.scenes, raw, field_of_view=args.field_of_view)
                        path = raw / "shard_config.json"
                        with path.open("x") as stream:
                            json.dump(shard, stream, indent=2, allow_nan=False)
                        path.chmod(0o444)
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuids[logical], T01_PARENT_CUDA_VISIBLE_DEVICES=parent_mask)
                        running = launch_logged(name, [sys.executable, "-u", "-m", "src.envs.pair_v2_shard",
                            "--name", name, "--scene", scene, "--shard-config", str(path),
                            "--timeout-seconds", str(granted)], env, scratch)
                        running.update(scene=scene, shard=shard, granted_timeout_seconds=granted, child_deadline=time.monotonic() + granted + 5)
                        active[logical] = running
                        metrics["started_shards"].append({"scene": scene, "run_name": name, "logical_index": logical,
                            "gpu_uuid": uuids[logical], "wrapper_pid": running["process"].pid,
                            "shard_config": str(path), "shard_config_sha256": sha_file(path)})
                if stopping or (not active and all(positions[i] == len(queues[i]) for i in range(args.gpus))):
                    break
                time.sleep(0.1)
            code = 86 if metrics["black_frame_detected"] else (1 if stopping or any(row["exit_code"] or row.get("provenance_error") for row in metrics["scene_runs"]) else 0)
            metrics["status"] = "stopped" if stopping else ("completed" if code == 0 else "completed_with_failures")
    except BaseException as error:
        metrics.update(status="orchestration_failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        code = 1
    finally:
        started = time.monotonic()
        pending = list(active.values()) + ([gate_running] if gate_running else [])
        for running in pending:
            try:
                request_stop_logged(running["process"])
            except Exception as error:
                metrics["cleanup_errors"].append(f"request stop {running['name']}: {error}")
        for running in pending:
            metrics["cleanup_errors"].extend(stop_logged(running["process"], request=False,
                grace_deadline=started + 20, hard_deadline=started + 28))
            if running["process"].poll() is None:
                running["stream"].close()
                metrics["cleanup_errors"].append("Unfinished wrapper: " + running["name"])
                continue
            row, data = finish_logged(running, sources)
            if running is gate_running:
                metrics["gate"] = row
            else:
                logical = next(index for index, item in active.items() if item is running)
                row.update(scene=running["scene"], logical_index=logical,
                           gpu_uuid=config["parent_cuda_visible_devices"].split(",")[logical],
                           stopped_by_orchestrator=True, granted_timeout_seconds=running["granted_timeout_seconds"])
                metrics["scene_runs"].append(row)
                try:
                    verify_shard_result(row, data, running["shard"])
                except ValueError as error:
                    row["provenance_error"] = str(error)
                    metrics["cleanup_errors"].append("Unverified shard identity/cleanup: " + running["name"])
        finished = {row["scene"] for row in metrics["scene_runs"]}
        launched = {row["scene"] for row in metrics["started_shards"]}
        metrics.update(unstarted_scenes=[scene for scene in args.scenes if scene not in launched],
                       unfinished_scenes=sorted(launched - finished), stop_reasons=stopping)
        metrics["all_planned_scenes_finished"] = not metrics["unstarted_scenes"] and not metrics["unfinished_scenes"] and not stopping
        if metrics["cleanup_errors"]:
            code = 1
        metrics["exit_code"] = code
        write_result(args.name, metrics, config)
        output.write_text(json.dumps({"metrics": metrics, "config": config, "sources": sources}, indent=2, allow_nan=False) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
