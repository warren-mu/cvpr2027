#!/usr/bin/env python3
"""Summarize explicit completed evidence; invoke through run_logged after commit.

Example: python3 scripts/run_logged.py --name t00b_progress -- \
  python3 src/tools/phase_progress.py --name t00b_progress --budget-gpu-hours 8 \
  --source results/diagnostic_attempt.json results/training_attempt.json results/pbs_usage.json

No simulator, scheduler query, training, or source/result replacement is performed.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.envs.thor_common import write_result
from restore_diagnostic_summary import comparison_summary, stats, truth_counts
from t00_summary import accounting_summary, metrics_hash


def pick(mapping, keys):
    """Preserve explicit nulls and represent missing requested fields as null."""
    return {key: mapping.get(key) for key in keys}


def load_source(path, output, auditor):
    path = (ROOT / path).resolve()
    relative = path.relative_to(ROOT)
    if relative.parts[0] != "results" or path.suffix != ".json" or path == output:
        raise ValueError(f"Expected a source results JSON distinct from output: {path}")
    raw = path.read_bytes()
    data = json.loads(raw)
    if (not isinstance(data, dict) or not isinstance(data.get("metrics"), dict)
            or not isinstance(data.get("config"), dict)):
        raise ValueError(f"Missing metrics/config objects: {relative}")
    meta = data.get("_meta", {})
    if not isinstance(meta, dict) or type(meta.get("exit_code")) is not int:
        raise ValueError(f"Source run has no final integer exit code: {relative}")
    audit = auditor.audit_file(str(path))
    if audit["fail"]:
        raise ValueError(f"{relative}: {'; '.join(audit['fail'])}")
    digest = metrics_hash(data["metrics"])
    if meta.get("metrics_sha256") != digest:
        raise ValueError(f"Metrics hash mismatch: {relative}")
    log = (ROOT / (meta.get("log_path") or f"logs/runs/{meta['name']}.log")).resolve()
    lines = re.findall(r"^FINAL_METRICS_JSON:\s*(.+?)\s*$", log.read_text(errors="replace"), re.M)
    if not lines or metrics_hash(json.loads(lines[-1])) != digest:
        raise ValueError(f"Missing or unequal final metrics in source log: {relative}")
    reference = {"file": relative.as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
                 "metrics_sha256": digest, "exit_code": meta["exit_code"],
                 "run_name": meta.get("name"), "start": meta.get("start"),
                 "git_commit": meta.get("git_commit"), "log_path": str(log.relative_to(ROOT))}
    return data, reference, audit


def diagnostic_summary(metrics):
    cases = metrics.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Diagnostic cases must be a list")
    if any(not isinstance(row, dict) for row in cases):
        raise ValueError("Diagnostic case observations must be objects")
    indices = [row.get("state_index") for row in cases]
    if any(type(index) is not int for index in indices) or len(set(indices)) != len(indices):
        raise ValueError("Diagnostic state_index values must be unique integers")
    comparisons = [row.get("transforms") for row in cases]
    rgb = [row.get("rgb") or {} for row in cases]
    return {
        **pick(metrics, ("cases_attempted", "cases_completed", "rgb_comparisons_completed",
                         "diagnostic_completed", "exact_zero_passed", "exact_zero_failed_cases",
                         "max_pixel_difference", "independent_unity_processes_verified")),
        "case_status_distribution": dict(Counter(row.get("status", "missing") for row in cases)),
        "rgb_exact_zero_distribution": truth_counts([row.get("rgb_exact_zero") for row in rgb]),
        "max_pixel_difference_distribution": stats([row.get("max_pixel_difference") for row in rgb]),
        "different_pixels_distribution": stats([row.get("different_pixels") for row in rgb]),
        "paired_reported_transforms": comparison_summary(comparisons),
        "cases": [{**pick(row, ("state_index", "status", "error")),
                   "rgb": pick(row.get("rgb") or {}, ("rgb_exact_zero", "max_pixel_difference", "different_pixels"))}
                  for row in cases],
    }


def source_summary(data, reference):
    metrics, config = data["metrics"], data["config"]
    row = {"source_file": reference["file"], "source_exit_code": reference["exit_code"],
           **pick(metrics, ("status", "error", "cleanup_errors")),
           "acceptance_passed": data.get("acceptance_passed", metrics.get("acceptance_passed")),
           "last_stage": data.get("last_stage"), "warnings": data.get("warnings"),
           "pbs_jobid": config.get("pbs_jobid") or (config.get("runtime") or {}).get("pbs_jobid"),
           "config": pick(config, ("mode", "seed", "resolution", "cases", "variant", "memory_variant",
                                   "training_config_changes", "required_completed_updates", "collect_only"))}
    if "allocation_gpu_hours_by_job" in metrics:
        row.update(kind="pbs_accounting", metrics=pick(metrics, (
            "complete_allocation_accounting", "allocation_gpu_hours_by_job")))
    elif "completed_training_steps" in metrics:
        # Keep per-step timing/frame evidence separate across attempts and variants.
        row.update(kind="training", metrics={key: value for key, value in metrics.items()
            if key.startswith(("median_", "gpu_", "process_tree_", "actual_new_rgb_", "required_render_fps"))
            or key in ("completed_training_steps", "resource_sample_count", "training_command_wall_seconds",
                       "training_command_exit_code", "wrapper_exit_code", "timed_out", "cleanup_remaining",
                       "pretraining_rgb_smoke_passed", "render_frame_accounting", "full_training_estimates",
                       "estimated_full_training_hours", "official_config_total_training_steps")},
                   completed_step_timings=data.get("completed_step_timings"),
                   measurement_notes=data.get("measurement_notes"))
    elif "counts_by_class" in metrics:
        row.update(kind="world_pairs", metrics=pick(metrics, (
            "attempts", "counts_by_class", "discard_reason_counts", "pending_prefix_only_by_class",
            "mean_pair_generation_seconds", "generation_time_denominator", "generation_time_observed_attempts",
            "look_prefix_comparisons", "look_strict_prefix_count", "look_strict_prefix_fraction",
            "look_passed_flip_stay_ratio", "collect_only", "scene_failures", "output_files")))
    elif isinstance(metrics.get("cases"), list):
        row.update(kind="diagnostic", metrics=diagnostic_summary(metrics))
    else:
        # Infrastructure failures/probes stay visible; no attempt to reinterpret their schema.
        row.update(kind="other", metrics=pick(metrics, (
            "status", "passed", "completed", "error", "cleanup_errors", "trials", "results")))
    return row


def memory_value(raw):
    match = re.fullmatch(r"(\d+)(b|kb|mb|gb|tb)", str(raw).strip().lower())
    kib = None
    if match:
        amount, unit = match.groups()
        kib = int(amount) * {"b": 1 / 1024, "kb": 1, "mb": 1024,
                             "gb": 1024 ** 2, "tb": 1024 ** 3}[unit]
    return {"raw": raw, "kib": kib}


def account(inputs, expected_jobs):
    sources = [(data, ref) for data, ref in inputs if "allocation_gpu_hours_by_job" in data["metrics"]]
    summary = accounting_summary(sources)
    memory, observations = {}, []
    for data, ref in sources:
        for job in data["config"]["jobs"]:
            fields, job_id = job.get("fields", {}), job["job_id"]
            values = {"requested_mem": memory_value(fields.get("Resource_List.mem")),
                      "actual_mem": memory_value(fields.get("resources_used.mem")),
                      "actual_vmem": memory_value(fields.get("resources_used.vmem")),
                      "exec_vnode_raw": fields.get("exec_vnode"),
                      "requested_select_raw": fields.get("Resource_List.select")}
            observations.append({"source_file": ref["file"], "job_id": job_id,
                                 "qstat_exit": job.get("qstat_exit"), "job_state": fields.get("job_state"), **values})
            if job.get("qstat_exit") == 0 and fields.get("job_state") == "F":
                if job_id in memory and memory[job_id] != values:
                    raise ValueError(f"{job_id}: inconsistent final memory/resource fields")
                memory[job_id] = values
    for row in summary["jobs"]:
        row.update(memory.get(row["job_id"], {}))
    summary["memory_observations"] = observations
    summary["accounting_sources_supplied"] = bool(sources)
    missing = set(summary["unresolved_job_ids"]) | (expected_jobs - set(summary["supplied_job_ids"]))
    summary["experiment_job_ids_without_final_accounting"] = sorted(missing)
    summary["complete_for_supplied_evidence"] = bool(sources) and not missing
    if not summary["complete_for_supplied_evidence"]:
        summary["total_allocation_gpu_hours"] = None
    if not sources:
        summary["known_allocation_gpu_hours_subtotal"] = None
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", nargs="+", action="extend", required=True)
    parser.add_argument("--budget-gpu-hours", type=float)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        parser.error("Invalid run name")
    if args.budget_gpu_hours is not None and (not math.isfinite(args.budget_gpu_hours) or args.budget_gpu_hours <= 0):
        parser.error("Budget must be positive and finite")
    output = ROOT / "results" / f"{args.name}.json"
    if output.exists():
        parser.error("Output already exists; use a new run name")
    references, audits, inputs, errors = [], [], [], []
    metrics = {"aggregation_status": "failed", "runs": [], "accounting": None}
    config = {"name": args.name, "requested_sources": args.source, "budget_gpu_hours": args.budget_gpu_hours,
              "source_provenance_checks": audits,
              "notes": ["Only explicitly supplied runs/jobs are included; no latest/best selection or completeness claim for the entire project.",
                        "Aggregation success is not experiment acceptance. Nonzero source exits and strict diagnostic failures remain visible.",
                        "Missing measurements remain null. Transform distributions concern reported metadata, not all Unity state.",
                        "Training variants, timings and candidate counts stay separate; no pooled median or duplicate-candidate acceptance total.",
                        "GPU-hours count each consistent full final PBS job ID once, including failed jobs; usage is not a reconciled credit charge.",
                        "PBS memory units b/kb/mb/gb/tb use powers of 1024; unknown units stay null. Process-tree RSS can double-count shared pages.",
                        "PBS waits are recorded single-run queue-entry/eligibility-to-start intervals, not cumulative requeue history."]}
    try:
        spec = importlib.util.spec_from_file_location("phase_existing_auditor", ROOT / "scripts/audit_provenance.py")
        auditor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auditor)
        paths, identities = set(), set()
        for requested in args.source:
            path = (ROOT / requested).resolve()
            if path in paths:
                raise ValueError(f"Duplicate source path: {requested}")
            paths.add(path)
            data, reference, audit = load_source(path, output, auditor)
            identity = (reference["run_name"], reference["start"], reference["git_commit"])
            if identity in identities:
                raise ValueError(f"Duplicate source run identity: {requested}")
            identities.add(identity)
            references.append(reference)
            audits.append(audit)
            inputs.append((data, reference))
        # Validate every source before computing any report totals.
        metrics["runs"] = [source_summary(data, reference) for data, reference in inputs]
        expected = {row["pbs_jobid"] for row in metrics["runs"] if row["pbs_jobid"]}
        metrics["accounting"] = account(inputs, expected)
        total = metrics["accounting"]["total_allocation_gpu_hours"]
        metrics["budget"] = {"limit_gpu_hours": args.budget_gpu_hours, "used_gpu_hours": total,
            "remaining_gpu_hours": args.budget_gpu_hours - total if args.budget_gpu_hours is not None and total is not None else None,
            "within_budget": total <= args.budget_gpu_hours if args.budget_gpu_hours is not None and total is not None else None}
        metrics["source_counts"] = {"total": len(inputs),
            "by_kind": dict(Counter(row["kind"] for row in metrics["runs"])),
            "by_exit_code": dict(Counter(str(ref["exit_code"]) for ref in references)),
            "acceptance": truth_counts([row["acceptance_passed"] for row in metrics["runs"]])}
        metrics["aggregation_status"] = "completed" if metrics["accounting"]["complete_for_supplied_evidence"] else "incomplete_accounting"
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    if errors:
        metrics["errors"] = errors
    if output.exists():
        raise FileExistsError(f"Output appeared during aggregation; refusing to replace {output}")
    write_result(args.name, metrics, config)
    # Add source identities for the existing auditor's P8 check; metrics/FINAL are unchanged.
    output.write_text(json.dumps({"metrics": metrics, "config": config, "sources": references},
                                 indent=2, allow_nan=False) + "\n")
    return 0 if metrics["aggregation_status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
