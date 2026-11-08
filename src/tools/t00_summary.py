#!/usr/bin/env python3
"""Aggregate existing T00 evidence; run through scripts/run_logged.py.

Example (supply every T00 accounting result and every training attempt):
  python3 scripts/run_logged.py --name t00_summary -- \
    python3 src/tools/t00_summary.py --accounting-results results/t00_pbs_usage.json \
    --training-results results/t00_vagen_smoke.json

This script performs no simulator, GPU, scheduler, or training operations.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

from pbs_usage import queue_wait_record

ROOT = Path(__file__).resolve().parents[2]
CHECK_FIELDS = (
    "status", "smoke_frame_path", "smoke_frame_shape", "instance_count",
    "instance_ids_matching_metadata", "instance_id_match_ratio", "unmatched_instance_ids",
    "restore_states_attempted", "restore_comparisons_completed", "restore_failed_states",
    "restore_max_pixel_difference", "object_moves_attempted", "object_position_errors_measured",
    "object_position_error_median_m", "object_position_error_max_m",
    "collateral_moves_measured", "collateral_moved_over_1cm_total", "error", "cleanup_errors",
)


def metrics_hash(metrics):
    # Same canonical encoding as the unchanged run_logged/audit_provenance tools.
    return hashlib.sha256(json.dumps(metrics, sort_keys=True).encode()).hexdigest()


def number(value, field):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field}: expected a finite nonnegative measurement or null")
    return value


def measured_range(values, scale=1):
    observed = [value / scale for value in values if value is not None]
    return {"measured_instances": len(observed),
            "mean": statistics.fmean(observed) if observed else None,
            "min": min(observed) if observed else None,
            "max": max(observed) if observed else None}


def load_source(path, output_path, auditor, sources, audits):
    path = (ROOT / path).resolve()
    try:
        relative = path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"Source must be inside this repository: {path}") from error
    if relative.parts[0] != "results" or path.suffix != ".json" or path == output_path:
        raise ValueError(f"Expected a source results JSON distinct from output: {relative}")
    raw = path.read_bytes()  # Explicitly supplied missing sources are errors.
    data = json.loads(raw)
    if not isinstance(data.get("metrics"), dict) or not isinstance(data.get("config"), dict):
        raise ValueError(f"{relative}: missing metrics/config objects")
    audit = auditor.audit_file(str(path))
    audits.append(audit)
    if audit["fail"]:
        raise ValueError(f"{relative}: provenance failed: {'; '.join(audit['fail'])}")
    if not data.get("_meta", {}).get("metrics_sha256") == metrics_hash(data["metrics"]):
        raise ValueError(f"{relative}: metrics hash does not match _meta")
    source = {"file": relative.as_posix(), "metrics_sha256": metrics_hash(data["metrics"]),
              "sha256": hashlib.sha256(raw).hexdigest(),
              "exit_code": data["_meta"].get("exit_code"),
              "git_commit": data["_meta"].get("git_commit")}
    sources.append(source)
    return data, source


def checks_summary(data, source):
    original = data["metrics"]
    summary = {key: copy.deepcopy(original[key]) for key in CHECK_FIELDS if key in original}
    summary.update(source_file=source["file"], source_exit_code=source["exit_code"])
    summary["restore_state_observations"] = [
        {key: copy.deepcopy(row[key]) for key in
         ("state_index", "random_seed", "max_pixel_difference", "different_pixels", "failed",
          "complete_pose_count", "inventory_preserved", "error") if key in row}
        for row in original.get("restore_states", [])
    ]
    summary["object_move_observations"] = [
        {key: copy.deepcopy(row[key]) for key in
         ("move_index", "object_name", "action_success", "position_error_m",
          "collateral_moved_over_1cm_count", "missing_object_names", "error") if key in row}
        for row in original.get("object_moves", [])
    ]
    return summary


def throughput_summary(data, source):
    rows = []
    for trial_index, trial in enumerate(data["metrics"].get("trials", [])):
        instances = trial.get("instances", [])
        row = {key: copy.deepcopy(trial[key]) for key in
               ("resolution", "kind", "concurrency", "status", "completed_frames",
                "requested_frames_per_instance", "concurrent_wall_seconds", "total_fps",
                "resource_samples", "resource_query_errors", "error", "resource_error",
                "resource_cleanup_error", "cleanup_errors") if key in trial}
        row.update(trial_index=trial_index, observed_instance_count=len(instances),
                   completed_frames_per_instance=[instance.get("completed_frames") for instance in instances])
        cpu = [number(instance.get("cpu_average_cores"), "cpu_average_cores") for instance in instances]
        memory = [number(instance.get("gpu_memory_peak_bytes"), "gpu_memory_peak_bytes") for instance in instances]
        row["cpu_average_cores_per_instance"] = measured_range(cpu)
        row["gpu_memory_peak_mib_per_instance"] = measured_range(memory, 1024 ** 2)
        device = number(trial.get("device_memory_peak_bytes"), "device_memory_peak_bytes")
        row["device_memory_peak_mib_sampled"] = device / (1024 ** 2) if device is not None else None
        if trial.get("kind") == "single":
            if trial.get("concurrency") != 1 or len(instances) > 1:
                raise ValueError(f"Single-instance trial {trial_index} has inconsistent concurrency")
            row["single_instance_median_fps"] = (
                number(instances[0].get("median_fps"), "median_fps") if instances else None
            )
        rows.append(row)
    summary = {"source_file": source["file"], "source_exit_code": source["exit_code"],
               "source_status": data["metrics"].get("status"), "rows": rows}
    for key in ("error", "cleanup_errors"):
        if key in data["metrics"]:
            summary[key] = copy.deepcopy(data["metrics"][key])
    return summary


def allocation_record(job, recorded_hours):
    fields = job["fields"]
    ngpus_text = fields.get("Resource_List.ngpus")
    ngpus = None
    if ngpus_text is not None:
        if not re.fullmatch(r"\d+", str(ngpus_text)):
            raise ValueError(f"{job['job_id']}: invalid allocated GPU count")
        ngpus = int(ngpus_text)
    walltime = fields.get("resources_used.walltime")
    seconds = None
    if walltime is not None:
        match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)", walltime)
        if not match:
            raise ValueError(f"{job['job_id']}: invalid measured walltime")
        hours, minutes, remainder = map(int, match.groups())
        seconds = hours * 3600 + minutes * 60 + remainder
    if seconds is not None and ngpus is not None:
        expected = ngpus * seconds / 3600
        basis = "allocated_gpus_times_measured_walltime"
    elif fields.get("run_count") == "0" and not fields.get("exec_host"):
        expected = 0.0
        basis = "scheduler_records_job_never_started"
    else:
        raise ValueError(f"{job['job_id']}: cannot substantiate recorded GPU-hours")
    hours = number(recorded_hours, "allocation_gpu_hours")
    if hours is None or not math.isclose(hours, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{job['job_id']}: GPU-hours disagree with scheduler fields")
    return {"job_id": job["job_id"], "allocated_ngpus": ngpus,
            "measured_walltime": walltime, "measured_walltime_seconds": seconds,
            "allocation_gpu_hours": hours, "basis": basis,
            "pbs_exit_status": fields.get("Exit_status"), "run_count": fields.get("run_count"),
            "exec_host": fields.get("exec_host"), "queue": fields.get("queue"),
            "project": fields.get("project"), **queue_wait_record(fields)}


def accounting_summary(inputs):
    records, required_jobs, unavailable = {}, set(), []
    for data, source in inputs:
        jobs = data["config"].get("jobs")
        hours = data["metrics"].get("allocation_gpu_hours_by_job")
        if not isinstance(jobs, list) or not jobs or not isinstance(hours, dict):
            raise ValueError(f"{source['file']}: no PBS accounting observations")
        source_jobs = set()
        for job in jobs:
            job_id = job.get("job_id")
            if not isinstance(job_id, str) or not re.fullmatch(r"\d+\.hopper-m-\d+", job_id):
                raise ValueError(f"{source['file']}: invalid full Hopper job ID {job_id!r}")
            source_jobs.add(job_id)
            required_jobs.add(job_id)
            if (job.get("qstat_exit") != 0 or job.get("fields", {}).get("job_state") != "F"
                    or job_id not in hours):
                unavailable.append({"job_id": job_id, "source_file": source["file"],
                                    "reason": "observation not final or GPU-hours unavailable"})
                continue
            record = allocation_record(job, hours[job_id])
            if job_id in records:
                previous = {key: value for key, value in records[job_id].items() if key != "source_files"}
                if record != previous:
                    raise ValueError(f"{job_id}: inconsistent final accounting records in {records[job_id]['source_files']} and {source['file']}")
                records[job_id]["source_files"].append(source["file"])
            else:
                records[job_id] = dict(record, source_files=[source["file"]])
        if set(hours) - source_jobs:
            raise ValueError(f"{source['file']}: GPU-hours have no corresponding scheduler observations")
    missing = sorted(required_jobs - set(records))
    subtotal = math.fsum(row["allocation_gpu_hours"] for row in records.values())
    return {"complete_for_supplied_jobs": not missing, "supplied_job_ids": sorted(required_jobs),
            "unique_accounted_job_count": len(records), "jobs": [records[key] for key in sorted(records)],
            "total_allocation_gpu_hours": subtotal if not missing else None,
            "known_allocation_gpu_hours_subtotal": subtotal,
            "unresolved_job_ids": missing, "unavailable_source_observations": unavailable}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="t00_summary")
    parser.add_argument("--checks-result", default="results/t00_thor_checks.json")
    parser.add_argument("--throughput-result", default="results/t00_thor_throughput.json")
    parser.add_argument("--accounting-results", required=True, nargs="+")
    parser.add_argument("--training-results", nargs="+", default=[])
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        parser.error("Invalid run name")
    output = ROOT / "results" / f"{args.name}.json"
    sources, audits, errors = [], [], []
    metrics = {"aggregation_status": "failed"}
    config = {
        "name": args.name, "requested_sources": vars(args), "source_provenance_checks": audits,
        "notes": [
            "Aggregation success validates evidence processing, not T00 experiment acceptance; source failures remain visible.",
            "Accounting covers the supplied job IDs only; the caller must supply all T00 allocations, including failed attempts.",
            "GPU-hours = allocated GPUs * scheduler measured walltime seconds / 3600; overlapping final job records count once.",
            "GPU-hours are allocation usage, not independently reconciled project credit charges.",
            "Queue waiting is recomputed from saved final PBS fields, without scheduler queries: queue_wait_seconds=stime-qtime; eligible_wait_seconds=stime-etime.",
            "PBS timestamp strings use server Asia/Singapore (+08:00); raw and offset-bearing timestamps are retained per job.",
            "Queue waits are recorded single-run intervals, not cumulative transfer/requeue history; run_count other than 1, missing/invalid timestamps, or negative intervals remain null with reasons.",
            "Single median FPS and total FPS are copied from recorded measurements; single K=1 and concurrent K=1 stay distinct.",
            "CPU cores are measured worker-plus-Unity CPU seconds / measurement walltime; each instance's 1.0 means one core.",
            "Ranges/means use available instance measurements only, with counts; absent measurements remain null, never zero-filled.",
            "GPU memory is sampled per-instance peak bytes / 1048576; these peaks are not a simultaneous aggregate peak.",
            "Training metrics are copied as recorded, including nulls, failure status, and explicitly labeled estimates.",
            "Source metrics hashes use the audit format; full-file SHA-256 also protects the accounting/config evidence used here.",
        ],
    }
    try:
        spec = importlib.util.spec_from_file_location("t00_existing_auditor", ROOT / "scripts" / "audit_provenance.py")
        auditor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auditor)
        cache = {}

        def load(path):
            key = str((ROOT / path).resolve())
            if key not in cache:
                cache[key] = load_source(path, output, auditor, sources, audits)
            return cache[key]

        # Validate all specified sources before performing report arithmetic.
        requested = [args.checks_result, args.throughput_result] + args.accounting_results + args.training_results
        for path in requested:
            try:
                load(path)
            except (OSError, ValueError, TypeError, KeyError) as error:
                errors.append(f"{path}: {type(error).__name__}: {error}")
        if errors:
            raise ValueError("One or more required source files failed validation")
        metrics["render_checks"] = checks_summary(*load(args.checks_result))
        metrics["throughput"] = throughput_summary(*load(args.throughput_result))
        # Repeated paths, like repeated jobs, must not increase accounting.
        accounting_paths = list(dict.fromkeys(str((ROOT / path).resolve()) for path in args.accounting_results))
        metrics["accounting"] = accounting_summary([load(path) for path in accounting_paths])
        if args.training_results:
            metrics["training"] = []
            for path in dict.fromkeys(args.training_results):
                data, source = load(path)
                row = {"source_file": source["file"], "source_exit_code": source["exit_code"],
                       "metrics": copy.deepcopy(data["metrics"])}
                for key in ("acceptance_passed", "last_stage", "warnings", "measurement_notes"):
                    if key in data:
                        row[key] = copy.deepcopy(data[key])
                metrics["training"].append(row)
        if not metrics["accounting"]["complete_for_supplied_jobs"]:
            raise ValueError("Some supplied jobs lack final accounting; total GPU-hours is unknown")
        metrics["aggregation_status"] = "passed"
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    if errors:
        metrics["errors"] = errors
    payload = {"metrics": metrics, "config": config, "sources": sources}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return 0 if metrics["aggregation_status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
