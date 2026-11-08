"""Single-GPU, bounded T02 table export; run only through run_logged in PBS.

This producer prepares exports and independently checks their native evidence.
It does not run the >=200-state fresh-process cache check, authorize training,
infer human review from a checkpoint, or implement the E2E small-sample exception.
All large tables, Unity logs and native evidence remain on scratch.
"""
import argparse
import copy
import hashlib
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

from src.envs.thor_common import ROOT, Worker, controller_session, deadline_after, stop_workers
from src.envs.pair_table_export import build_table, load_table, save_table, SimulatorTableRenderer
from src.envs.pair_table_checks import check_pair, verify_render_sources
from src.tools.t01_partition import audited_result, cpu_shards, sha_file
from src.tools.t01_parallel_approved import stop_logged


VERSION = "t02_single_gpu_export_v1"
REJECTIONS = frozenset(("graph_mismatch", "non_anchor_divergent", "source_world_restore_failed"))


def save_json(path, value):
    path = Path(path)
    body = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(body)
    return {"file": str(path.resolve()), "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}


def execution_environment(environment, affinity):
    """Reject login execution, expanded masks, and insufficient allocated CPUs."""
    job, mask, scratch = (environment.get(k, "") for k in ("PBS_JOBID", "CUDA_VISIBLE_DEVICES", "CVPR_SCRATCH"))
    if (not re.fullmatch(r"\d+\.hopper-m-\d+", job)
            or not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", mask)
            or not scratch or not Path(scratch).is_absolute()):
        raise ValueError("Require a real PBS job, its unchanged single full-UUID mask, and absolute scratch")
    return {"pbs_jobid": job, "cuda_visible_devices": mask,
            "scratch": str(Path(scratch).resolve()), "cpu_affinity": cpu_shards(affinity, 1)[0]}


def gate_runtime(data, identity, node, log_dir):
    config, metrics = data["config"], data["metrics"]
    allocation = config["allocation"]
    devices = allocation.get("devices", [])
    rows = metrics.get("devices", [])
    if (data["_meta"].get("exit_code") != 0 or metrics.get("passed") is not True
            or metrics.get("cleanup_errors") != [] or metrics.get("errors") != []
            or metrics.get("expected_gpu_count") != 1 or metrics.get("completed_gpu_count") != 1
            or config.get("node") != node
            or config.get("resolution") != 255 or config.get("segmentation") is not True
            or allocation.get("pbs_jobid") != identity["pbs_jobid"]
            or allocation.get("cuda_visible_devices") != identity["cuda_visible_devices"]
            or allocation.get("cuda_driver_uuids") != [identity["cuda_visible_devices"]]
            or len(devices) != 1 or devices[0].get("logical_index") != 0
            or devices[0].get("gpu_uuid") != identity["cuda_visible_devices"]
            or allocation.get("scratch") != identity["scratch"]
            or len(rows) != 1 or rows[0].get("worker_error") is not None
            or rows[0].get("observation", {}).get("gpu_identity") != devices[0]):
        raise ValueError("Gate does not authenticate this exact single-GPU PBS allocation")
    observation = rows[0]["observation"]
    captures = observation.get("captures", [])
    if (observation.get("valid") is not True or observation.get("error")
            or observation.get("cleanup_error") or len(captures) != 2
            or any(c.get("valid") is not True or c.get("action_success") is not True
                   or c.get("segmentation", {}).get("valid") is not True for c in captures)):
        raise ValueError("Actual RGB and segmentation gate captures are incomplete")
    device = devices[0]
    return {**identity, **device, "controller_gpu_device": device["gpu_index"],
            "log_dir": str(log_dir), "ai2thor_version": allocation["ai2thor_version"]}


def export_checked(sim, package, directory, evidence_roots, max_table_bytes):
    """One complete graph only; serialized flags are never trusted on reload."""
    worlds = package["worlds"]
    if (len(worlds) != 2 or worlds[0].get("clone_context") != worlds[1].get("clone_context")
            or worlds[0].get("agent_setup_command") != worlds[1].get("agent_setup_command")):
        raise ValueError("Worlds do not share the authenticated common inventory/canonical camera")
    restoration = []
    # Recreate a clone inventory once in this owned controller, then verify both
    # original worlds. Never clone/delete an object while switching w0 -> w1.
    for world in worlds:
        _, measured = sim.restore_snapshot(world)
        restoration.append(measured)
        if measured.get("restore_deterministic") is not True:
            raise ValueError("source_world_restore_failed")
    reachable, reference = package["reachable_positions"], package["reachable_reference"]
    if package["kind"].startswith("look_") and (reachable is None or reference is None):
        raise ValueError("Look requires its original authenticated native action catalog")
    if (reachable is None) != (reference is None):
        raise ValueError("Reachable catalog and its native reference must be present together")
    catalog_scope = package["reachable_scope"]
    if reachable is None:
        # Direct has no all-action T01 catalog. This real query defines an
        # explicit fixed domain; both worlds still get native per-state paths.
        _, restored = sim.restore_snapshot(worlds[0])
        if restored.get("restore_deterministic") is not True:
            raise ValueError("source_world_restore_failed")
        actual = sim.reachable_positions()
        reachable, reference = actual["positions"], actual["query_reference"]
        catalog_scope = "fresh native GetReachablePositions in restored direct w0; original order retained"
    renderer = SimulatorTableRenderer(sim, worlds, reachable, package["initial_position"],
        package["anchor_name"], package["candidate_names"], pair_id=package["pair_id"], reachable_reference=reference)
    table = build_table(renderer, **{key: package[key] for key in (
        "pair_id", "scene", "kind", "instruction", "anchor_name", "candidate_names", "target_names",
        "initial_position", "initial_heading")}, source={"t01": package["source_provenance"],
            "export_version": VERSION, "reachable_scope": catalog_scope, "reachable_reference": reference})
    manifest = save_table(table, directory, max_total_bytes=max_table_bytes)
    # Check the bytes that downstream consumers will actually load, not only
    # the in-memory object that produced them. load_table clears both flags.
    loaded = load_table(manifest["file"], expected_sha256=manifest["sha256"])
    native = verify_render_sources(loaded, evidence_roots=evidence_roots)
    if native.get("passed") is not True or loaded.synthetic or loaded.export_verified is not True:
        raise ValueError("native_evidence_verification_failed")
    checks = check_pair(loaded, package["t01"])
    return {"table_manifest": manifest, "table_complete": True, "native_checks": native,
            "pair_checks": checks, "restoration": restoration, "reachable_scope": catalog_scope,
            "reachable_reference": reference, "export_measurements": table.export_measurements,
            "cache_matches_live": None, "cache_status": "not_run_requires_at_least_200_paired_states",
            "downstream_allowed": False, "task_acceptance": None}


def pair_worker(config, connection):
    from src.envs.pair_v2_simulator import BlackFrame, V2Simulator
    from src.envs.pair_v2_clone import V2CloneSimulator

    os.sched_setaffinity(0, config["runtime"]["cpu_affinity"])
    package = config["package"]
    directory = Path(config["directory"])
    sim, body_finished = None, False
    report = {"pair_id": package["pair_id"], "scene": package["scene"], "kind": package["kind"],
              "status": "error", "table_complete": False, "rejection_reasons": [], "error": None,
              "black_frame_reference": None, "cleanup_errors": [], "cache_matches_live": None,
              "task_acceptance": None, "downstream_allowed": False}
    start = time.monotonic()
    try:
        with controller_session(config, "controller", segmentation=True, connection=connection) as controller:
            try:
                cls = V2CloneSimulator if package["clone_context"] is not None else V2Simulator
                sim = cls(controller, config, directory / "raw")
                detail = export_checked(sim, package, directory / "table",
                    [*package["evidence_roots"], str(directory)], config["max_table_bytes"])
                report["detail"] = save_json(directory / "checks.json", detail)
                report.update(table_complete=True, table_manifest=detail["table_manifest"],
                    native_evidence_verified=detail["native_checks"]["passed"],
                    pair_checks_passed=detail["pair_checks"]["passed"],
                    rejection_reasons=detail["pair_checks"]["reasons"],
                    export_measurements=detail["export_measurements"],
                    status="exported" if detail["pair_checks"]["passed"] else "rejected")
            except BlackFrame as error:
                report.update(status="black_frame", black_frame_reference=error.reference)
            except BaseException as error:
                message = str(error)
                report.update(status="rejected" if type(error) is ValueError and message in REJECTIONS else "error",
                    rejection_reasons=[message] if type(error) is ValueError and message in REJECTIONS else [],
                    error={"type": type(error).__name__, "message": message})
                traceback.print_exc()
            finally:
                body_finished = True
                if sim is not None:
                    try:
                        report["raw_index"] = sim.finish()
                    except BaseException as error:
                        report["raw_index_error"] = f"{type(error).__name__}: {error}"
                        report["status"] = "error"
                    report["actual_frame_records"] = sim.frame_count
                    report["actual_action_records"] = sim.action_count
    except BaseException as error:
        report.update(status="error", error={"type": type(error).__name__, "message": str(error)})
        if body_finished:
            report["cleanup_errors"].append("Controller shutdown: " + str(error))
        traceback.print_exc()
    report["worker_elapsed_seconds"] = time.monotonic() - start
    # The append-only raw index survives a forced deadline/native crash even if
    # no final index or complete table can be written.
    incremental = directory / "raw/evidence_index.jsonl"
    if incremental.is_file():
        report["raw_incremental_index"] = {"file": str(incremental), "sha256": sha_file(incremental),
                                            "bytes": incremental.stat().st_size}
    report["report_reference"] = save_json(directory / "pair_report.json", report)
    return report


def preserve_black(reference, directory, runtime, scene, output):
    """Small portable black proof; a missing/invalid image is never exit 86."""
    from src.tools.t01v2_black_evidence import checked_reference
    from PIL import Image
    import io

    raw = (directory / "raw").resolve()
    path, body = checked_reference(reference, raw, 1024 * 1024)
    record = json.loads(body)
    if record.get("black") is not True or record.get("valid") is not True or record.get("rgb_shape") != [255, 255, 3]:
        raise ValueError("No complete measured black RGB record")
    image_path, image_bytes = checked_reference(record["files"]["rgb"], raw, 2 * 1024 * 1024)
    meta_path, meta_bytes = checked_reference(record["files"]["metadata"], raw, 8 * 1024 * 1024)
    metadata = json.loads(meta_bytes)
    if metadata.get("sceneName") not in (scene, scene + "_physics") or metadata.get("sequenceId") != record.get("sequence_id"):
        raise ValueError("Black capture scene/sequence mismatch")
    with Image.open(io.BytesIO(image_bytes)) as image:
        if (image.mode != "RGB" or image.size != (255, 255) or image.getextrema() != ((0, 0),) * 3
                or hashlib.sha256(image.tobytes()).hexdigest() != record.get("rgb_sha256")):
            raise ValueError("Recorded black PNG is not the exact full all-zero RGB")
    output.mkdir(parents=True, exist_ok=False)
    copied = []
    for filename, original, content in (("failure.json", path, body), ("black.png", image_path, image_bytes),
                                        ("metadata.json", meta_path, meta_bytes)):
        target = output / filename
        with target.open("xb") as stream:
            stream.write(content)
        digest = hashlib.sha256(content).hexdigest()
        if sha_file(target) != digest:
            raise ValueError("Portable black copy differs")
        copied.append({"file": str(target.relative_to(ROOT)), "original_file": str(original),
                       "sha256": digest, "bytes": len(content)})
    return {"files": copied, "pbs_jobid": runtime["pbs_jobid"], "gpu_uuid": runtime["gpu_uuid"],
            "scene": scene, "node": socket.gethostname(), "rgb_shape": [255, 255, 3], "black": True}


def black_only_failure(row):
    return (row.get("status") == "black_frame" and bool(row.get("black_frame_reference"))
            and row.get("cleanup_errors") == [] and row.get("error") is None
            and not row.get("raw_index_error"))


def logged_gate(name, seconds, directory, deadline, sources):
    output = ROOT / "results" / f"{name}.json"
    if output.exists() or (ROOT / "logs/runs" / name).exists():
        raise FileExistsError("Gate name already used")
    command = [sys.executable, "-u", str(ROOT / "scripts/run_logged.py"), "--name", name, "--seed", "0", "--",
               sys.executable, "-u", "-m", "src.tools.render_gate", "--name", name, "--gpus", "1",
               "--resolution", "255", "--segmentation", "--timeout-seconds", str(seconds)]
    process, cleanup = None, []
    with (directory / "gate_launcher.log").open("x") as stream:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CVPR_RUN_NAME": name,
                "CVPR_RESULTS_PATH": str(output)}, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=max(.1, min(seconds + 15, deadline - time.time() - 30)))
            except subprocess.TimeoutExpired:
                cleanup.extend(stop_logged(process))
        finally:
            if process is not None and process.poll() is None:
                cleanup.extend(stop_logged(process))
    data, reference = audited_result(output)
    sources.append(reference)
    if data["_meta"]["exit_code"] != process.returncode:
        raise ValueError("Gate process/result exit mismatch")
    return data, reference, cleanup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pair-id", action="append", required=True)
    parser.add_argument("--pair-file", help="Exact published pair path, only for one ambiguous pair ID")
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--pair-timeout-seconds", type=float, required=True)
    parser.add_argument("--gate-timeout-seconds", type=float, default=180)
    parser.add_argument("--max-raw-mib", type=int, default=2048)
    parser.add_argument("--max-table-mib", type=int, default=512)
    args = parser.parse_args()
    if (not re.fullmatch(r"t02_[A-Za-z0-9_]+", args.name)
            or len(set(args.pair_id)) != len(args.pair_id) or (args.pair_file and len(args.pair_id) != 1)
            or not all(math.isfinite(x) and x > 0 for x in (args.timeout_seconds, args.pair_timeout_seconds))
            or not 15 <= args.gate_timeout_seconds <= 600 or args.timeout_seconds <= args.gate_timeout_seconds + 60
            or args.pair_timeout_seconds > args.timeout_seconds - 30
            or not 1 <= args.max_raw_mib <= 2048 or not 1 <= args.max_table_mib <= 2048):
        parser.error("Require a unique t02_ name, explicit distinct pair IDs and finite bounded deadlines/caps")
    output = ROOT / "results" / f"{args.name}.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserve the previous result; use a unique name")
    config = {**vars(args), "source": str(args.source), "version": VERSION, "gpus": 1, "resolution": 255,
              "depth": 4, "node": socket.gethostname(), "human_review_status": "not_inferred_from_source",
              "execution_scope": "T02 export CPU-prepared; actual GPU invocation requires prior human/Claude T01 review"}
    metrics = {"kind": "t02_table_export", "status": "failed", "pairs": [], "unstarted_pair_ids": list(args.pair_id),
               "gate": None, "cleanup_errors": [], "errors": [], "black_frame_detected": False,
               "export_checks_complete": False, "cache_matches_live": None,
               "cache_status": "not_run_requires_at_least_200_paired_states", "downstream_allowed": False,
               "task_acceptance": None}
    sources, worker, code = [], None, 1
    from src.envs.thor_common import install_signals
    install_signals()
    try:
        identity = execution_environment(os.environ, os.sched_getaffinity(0))
        config.update(identity)
        deadline = deadline_after(args.timeout_seconds)
        signal.setitimer(signal.ITIMER_REAL, max(.1, deadline - time.time()))
        config["absolute_deadline_epoch"] = deadline
        directory = Path(identity["scratch"]) / "t02_export" / args.name / identity["pbs_jobid"]
        directory.mkdir(parents=True, exist_ok=False)
        config["scratch_directory"] = str(directory)
        from src.envs.pair_table_source import load_reviewed_pair
        packages = []
        for pair_id in args.pair_id:
            package = load_reviewed_pair(args.source, pair_id, pair_file=args.pair_file)
            for reference in package["sources"]:
                if reference not in sources:
                    sources.append(reference)
            packages.append(package)
        gate, gate_ref, cleanup = logged_gate(args.name + "_gate", args.gate_timeout_seconds, directory, deadline, sources)
        metrics["gate"], metrics["cleanup_errors"] = gate_ref, cleanup
        if gate["_meta"]["exit_code"] != 0:
            metrics.update(status="blocked_by_render_gate", black_frame_detected=gate["_meta"]["exit_code"] == 86)
            code = 86 if metrics["black_frame_detected"] and not cleanup and gate["metrics"].get("cleanup_errors") == [] else 1
        else:
            runtime = gate_runtime(gate, identity, config["node"], directory / "unity_logs")
            config["runtime"] = runtime
            if cleanup:
                raise RuntimeError("Gate cleanup is uncertain")
            for index, package in enumerate(packages):
                if time.time() >= deadline - 30:
                    metrics["errors"].append("export_deadline_before_next_pair")
                    break
                pair_dir = directory / f"pair_{index:04d}"
                pair_dir.mkdir(exist_ok=False)
                source_ref = save_json(pair_dir / "authenticated_source.json", package)
                child = {"name": args.name, "scene": package["scene"], "resolution": 255,
                    "runtime": runtime, "package": package, "directory": str(pair_dir),
                    "deadline": min(deadline - 15, time.time() + args.pair_timeout_seconds),
                    "max_raw_bytes": args.max_raw_mib * 1024 * 1024, "max_table_bytes": args.max_table_mib * 1024 * 1024}
                for key in ("grid_size", "visibility_distance", "field_of_view"):
                    child[key] = package["source_config"][key]
                worker = Worker(pair_worker, child)
                metrics["unstarted_pair_ids"].remove(package["pair_id"])
                while worker.result is None and worker.error is None and time.time() < child["deadline"] + 1:
                    worker.receive()
                    time.sleep(.05)
                worker.receive()
                row = worker.result or {"pair_id": package["pair_id"], "scene": package["scene"],
                    "status": "error", "table_complete": False, "error": worker.error or {"type": "TimeoutError",
                    "message": "Owned worker deadline reached without a complete result"}, "cleanup_errors": []}
                row.update(worker_identity=worker.identity, unity_identity=worker.unity_identity)
                cleanup = stop_workers([worker])
                worker = None
                row.update(source_package=source_ref, scratch_directory=str(pair_dir))
                row["cleanup_errors"].extend(cleanup)
                metrics["cleanup_errors"].extend(row["cleanup_errors"])
                metrics["pairs"].append(row)
                if row["cleanup_errors"]:
                    break
                if row.get("black_frame_reference"):
                    row["portable_black_evidence"] = preserve_black(row["black_frame_reference"], pair_dir,
                        runtime, package["scene"], ROOT / "logs/runs" / args.name / f"black_pair_{index:04d}")
                    only_black = black_only_failure(row)
                    metrics.update(black_frame_detected=True, status="black_frame" if only_black else "error_with_black_evidence")
                    code = 86 if only_black else 1
                    break
            if not metrics["black_frame_detected"]:
                metrics["export_checks_complete"] = (not metrics["unstarted_pair_ids"] and not metrics["cleanup_errors"]
                    and not metrics["errors"] and all(row["status"] in ("exported", "rejected") for row in metrics["pairs"]))
                code = 0 if metrics["export_checks_complete"] else 1
                metrics["status"] = "export_checks_completed_cache_pending" if code == 0 else "incomplete"
    except BaseException as error:
        traceback.print_exc()
        metrics["errors"].append(f"{type(error).__name__}: {error}")
        code = 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if worker is not None:
            metrics["cleanup_errors"].extend(stop_workers([worker]))
        if metrics["cleanup_errors"]:
            code = 1
        metrics["producer_exit_code"] = code
        output.parent.mkdir(parents=True, exist_ok=True)
        save_json(output, {"metrics": metrics, "config": config, "sources": sources})
        print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
