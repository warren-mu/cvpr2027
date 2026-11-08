"""Format explicit, audited final PBS accounting into JSON and a queue table.

No scheduler query, estimation, source rewrite or selection of a favorable retry.
Queue intervals are recorded single-run intervals, not a queue-time distribution.
Charges use Resource_List.queue_charge_rate, not an independent credit ledger.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import traceback

from src.tools.t01_partition import ROOT, audited_result
from src.tools.pbs_usage import queue_wait_record
from src.tools.pbs_cancel_evidence import match_terminal, not_started, verify_evidence

JOB = re.compile(r"\d+\.hopper-m-\d+\Z")


def producer_arguments(command, root):
    if (not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command)
            or not re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(command[0]).name)):
        raise ValueError("Require the actual Python PBS accounting command")
    argv = command[1:]
    if argv[:1] == ["-u"]:
        argv = argv[1:]
    if not argv or (root / argv[0]).resolve() != (root / "src/tools/pbs_usage.py").resolve():
        raise ValueError("Only the canonical src/tools/pbs_usage.py producer is accepted")
    name, cancelled, jobs, seen = "t00_pbs_usage", None, [], set()
    index = 1
    while index < len(argv):
        token = argv[index]
        if token in ("--name", "--cancelled-evidence"):
            count = 1 if token == "--name" else 3
            if token in seen or index + count >= len(argv):
                raise ValueError("Duplicate/incomplete accounting argument")
            seen.add(token)
            values = argv[index + 1:index + 1 + count]
            if token == "--name":
                name = values[0]
            else:
                cancelled = values
            index += count + 1
        elif JOB.fullmatch(token):
            jobs.append(token)
            index += 1
        else:
            raise ValueError("Unknown accounting argument; do not infer producer identity")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not jobs or len(set(jobs)) != len(jobs):
        raise ValueError("Require one exact run name and distinct explicit full PBS job IDs")
    return name, jobs, cancelled


def validate_source(data, reference, root):
    """Bind config fields (outside the metrics hash) to canonical printed config."""
    meta, config, metrics = data["_meta"], data["config"], data["metrics"]
    name, ids, cancelled_paths = producer_arguments(meta.get("command"), root)
    if (meta.get("name") != name or Path(reference["file"]).name != name + ".json"
            or meta.get("exit_code") != 0 or metrics.get("complete_allocation_accounting") is not True
            or config.get("errors") != [] or config.get("cancelled_evidence") != cancelled_paths):
        raise ValueError("Accounting source identity/completion differs from its logged command")
    rows, hours = config.get("jobs"), metrics.get("allocation_gpu_hours_by_job")
    if (not isinstance(rows, list) or [row.get("job_id") for row in rows] != ids
            or not isinstance(hours, dict) or set(hours) != set(ids)):
        raise ValueError("Accounting job observations/measurements differ from explicit command IDs")
    log = (root / meta.get("log_path", f"logs/runs/{name}.log")).resolve()
    log.relative_to(root)
    if json.dumps(config, indent=2) not in log.read_text(errors="replace"):
        raise ValueError("Saved PBS fields differ from the canonical source log")
    declared = metrics.get("verified_never_started_cancelled_jobs", [])
    if not isinstance(declared, list) or len(set(declared)) != len(declared) or not set(declared) <= set(ids):
        raise ValueError("Invalid declared never-started cancellation set")
    cancellations = {}
    if declared:
        cancellations, cancellation_refs = verify_evidence(root, cancelled_paths or [])
        if not set(declared) <= set(cancellations) or data.get("sources") != cancellation_refs:
            raise ValueError("Cancellation claim differs from its original audited three-part chain")
    for job in rows:
        wait = queue_wait_record(job["fields"])
        for key, actual in wait.items():
            if key in job and job[key] != actual:
                raise ValueError("Stored queue interval disagrees with raw final PBS timestamps")
        for key, value in (("queue_wait_seconds_by_job", wait["queue_wait_seconds"]),
                           ("eligible_wait_seconds_by_job", wait["eligible_wait_seconds"])):
            if key in metrics and (set(metrics[key]) != set(ids) or metrics[key].get(job["job_id"]) != value):
                raise ValueError("Queue interval metrics differ from their actual PBS records")
    if "total_allocation_gpu_hours" in metrics:
        total = finite_number(metrics["total_allocation_gpu_hours"], "source total")
        expected = math.fsum(finite_number(hours[job], "source job hours") for job in ids)
        if not math.isclose(total, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Source accounting total differs from its explicit jobs")
    return rows, hours, {job: cancellations[job] for job in declared}


def finite_number(value, field):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid nonnegative measured number: " + field)
    return value


def job_record(job, recorded_hours, cancellation=None):
    fields, job_id = job["fields"], job["job_id"]
    if (not JOB.fullmatch(job_id) or job.get("qstat_exit") != 0 or fields.get("job_state") != "F"
            or fields.get("project") != "CFP04-SF-125"):
        raise ValueError("Queue tables require an explicit verified final job in this project")
    raw_gpu = fields.get("Resource_List.ngpus")
    if raw_gpu is not None and not re.fullmatch(r"\d+", str(raw_gpu)):
        raise ValueError("Malformed PBS GPU count")
    gpus = int(raw_gpu) if raw_gpu is not None else None
    measured = finite_number(recorded_hours, "allocation_gpu_hours")
    raw_wall = fields.get("resources_used.walltime")
    seconds, never_started = None, False
    if raw_wall is not None:
        match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)", str(raw_wall))
        if not match or gpus is None:
            raise ValueError("Missing or malformed actual allocation duration/GPU count")
        h, m, s = map(int, match.groups())
        seconds = h * 3600 + m * 60 + s
        expected = gpus * seconds / 3600
        basis = "Resource_List.ngpus * resources_used.walltime_seconds / 3600"
    elif fields.get("run_count") == "0" and not_started(fields, "F"):
        expected, never_started, basis = 0.0, True, "scheduler explicitly records no run or allocation"
    elif cancellation is not None:
        match_terminal(fields, cancellation["postflight_fields"])
        expected, never_started, basis = 0.0, True, "audited Q/no-allocation -> qdel exit0 -> F/no-allocation chain"
    else:
        raise ValueError("Missing usage is not proof of zero allocation cost")
    if not math.isclose(measured, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Recorded GPU-hours disagree with the authenticated scheduler measurement")
    raw_rate = fields.get("Resource_List.queue_charge_rate")
    rate, rate_reason = None, None
    if raw_rate is None or raw_rate == "":
        rate_reason = "missing_recorded_Resource_List.queue_charge_rate"
    else:
        try:
            rate = float(raw_rate)
        except (ValueError, TypeError) as error:
            raise ValueError("Malformed recorded queue charge rate") from error
        if isinstance(raw_rate, bool) or not math.isfinite(rate) or rate < 0:
            raise ValueError("Recorded queue charge rate must be finite and nonnegative")
    charged = None if rate is None else finite_number(measured * rate, "recorded-rate charged GPU-hours")
    nodes = []
    if fields.get("exec_host"):
        parts = fields["exec_host"].split("+")
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+/\d+(?:\*\d+)?", part) for part in parts):
            raise ValueError("Malformed executed-host field; no node guess")
        nodes = sorted({part.split("/")[0] for part in parts})
    wait = queue_wait_record(fields)
    # A never-started job has no entry-to-start interval, including when some
    # unrelated source field might look like a zero-duration timestamp pair.
    if never_started and (wait["queue_wait_seconds"] is not None or wait["eligible_wait_seconds"] is not None):
        raise ValueError("Never-started cancellation unexpectedly has a measured start interval")
    return {"job_id": job_id, "job_state": "F", "gpu_count": gpus,
            "gpu_count_scope": "requested_but_not_allocated" if never_started else "PBS_recorded_allocation",
            "nodes": nodes, "node": ",".join(nodes) if nodes else None,
            "exec_host_raw": fields.get("exec_host"), "exec_vnode_raw": fields.get("exec_vnode"),
            "queue": fields.get("queue"), "project": fields.get("project"),
            "pbs_exit_status": fields.get("Exit_status"), "run_count_raw": fields.get("run_count"),
            "never_started": never_started, "requested_walltime": fields.get("Resource_List.walltime"),
            "actual_walltime": raw_wall, "actual_walltime_seconds": seconds,
            "allocation_gpu_hours": measured, "allocation_accounting_basis": basis,
            "queue_charge_rate_raw": raw_rate, "queue_charge_rate": rate,
            "scheduler_charged_gpu_hours": charged,
            "charged_hours_unavailable_reason": rate_reason, **wait,
            "queue_wait_minutes": None if wait["queue_wait_seconds"] is None else wait["queue_wait_seconds"] / 60,
            "eligible_wait_minutes": None if wait["eligible_wait_seconds"] is None else wait["eligible_wait_seconds"] / 60}


def aggregate(inputs, root):
    records, sources, seen = {}, [], set()
    for data, reference in inputs:
        if reference["file"] in seen:
            raise ValueError("Duplicate explicit accounting source")
        seen.add(reference["file"])
        rows, hours, cancellations = validate_source(data, reference, root)
        sources.append(reference)
        for job in rows:
            job_id = job["job_id"]
            record = job_record(job, hours[job_id], cancellations.get(job_id))
            if job_id in records:
                previous = {key: value for key, value in records[job_id].items() if key != "source_files"}
                if record != previous:
                    raise ValueError("Conflicting final cost/wait/GPU/rate or raw identity: " + job_id)
                records[job_id]["source_files"].append(reference["file"])
            else:
                records[job_id] = {**record, "source_files": [reference["file"]]}
    if not records:
        raise ValueError("No explicitly supplied final accounting jobs")
    table = [records[key] for key in sorted(records, key=lambda key: (int(key.split(".")[0]), key))]
    charge_complete = all(row["scheduler_charged_gpu_hours"] is not None for row in table)
    known = math.fsum(row["scheduler_charged_gpu_hours"] for row in table if row["scheduler_charged_gpu_hours"] is not None)
    metrics = {"aggregation_status": "passed", "jobs": table, "unique_job_count": len(table),
        "complete_allocation_accounting": True, "complete_recorded_rate_charges": charge_complete,
        "total_allocation_gpu_hours": math.fsum(row["allocation_gpu_hours"] for row in table),
        "total_scheduler_charged_gpu_hours": known if charge_complete else None,
        "known_scheduler_charged_gpu_hours_subtotal": known,
        "charge_unavailable_job_ids": [row["job_id"] for row in table if row["scheduler_charged_gpu_hours"] is None],
        "queue_wait_unavailable_job_ids": [row["job_id"] for row in table if row["queue_wait_seconds"] is None],
        "scope": "explicit unique final jobs only; no pending reservations or credit-ledger reconciliation",
        "queue_scope": "recorded queue-entry/start interval for final run_count=1 only; no earlier transfer/requeue history inferred"}
    return metrics, sources


def markdown(metrics):
    def cell(value):
        if value is None:
            return "unknown"
        if isinstance(value, float):
            return format(value, ".12g")
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# PBS queue wait and recorded charges", "", "Times are PBS server Asia/Singapore (+08:00). Exact values and raw timestamp strings are retained in the JSON.", "",
        "| Job | GPUs | Node | qtime (raw) | stime (raw) | Wait seconds | Wait minutes | Actual walltime | Allocation GPUh | Recorded rate | Charged GPUh |",
        "|---|---:|---|---|---|---:|---:|---|---:|---:|---:|"]
    for row in metrics["jobs"]:
        values = [row["job_id"], row["gpu_count"], row["node"], row["pbs_timestamps_raw"]["qtime"],
            row["pbs_timestamps_raw"]["stime"], row["queue_wait_seconds"], row["queue_wait_minutes"],
            row["actual_walltime"], row["allocation_gpu_hours"], row["queue_charge_rate"], row["scheduler_charged_gpu_hours"]]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines += ["", "Unique-job allocation GPUh: " + cell(metrics["total_allocation_gpu_hours"]) + ".",
        "Recorded-rate charged GPUh: " + cell(metrics["total_scheduler_charged_gpu_hours"]) +
        "; known subtotal: " + cell(metrics["known_scheduler_charged_gpu_hours_subtotal"]) + ".", "",
        "Cancelled never-started jobs retain a null wait; their GPU count is requested, not allocated. Missing rates stay unknown. Repeated final records are counted once only when they agree.",
        "These are supplied completed-job costs, not a remaining-budget calculation. Pending reservations and actual project-credit reconciliation are outside this table.", "",
        "## Accounting sources", ""]
    for row in metrics["jobs"]:
        links = ", ".join(f"[{Path(path).stem}](../../{path})" for path in row["source_files"])
        reasons = row["queue_wait_unavailable_reason"]
        lines.append(f"- {row['job_id']}: {links}" + (f"; wait unavailable: `{reasons}`" if reasons else ""))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.name):
        parser.error("Use a unique safe run name")
    output, report = ROOT / "results" / f"{args.name}.json", ROOT / "logs/reports" / f"{args.name}.md"
    if any(path.exists() or path.is_symlink() for path in (output, report)):
        raise FileExistsError("Preserve existing report evidence; use a unique run name")
    sources, code = [], 1
    metrics = {"aggregation_status": "failed", "jobs": [], "error": None}
    try:
        inputs = []
        for path in args.source:
            data, reference = audited_result(path)
            if (ROOT / reference["file"]).resolve() == output:
                raise ValueError("Output cannot be its own source")
            inputs.append((data, reference))
            sources.append(reference)
        metrics, sources = aggregate(inputs, ROOT)
        text = markdown(metrics)
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("x") as stream:
            stream.write(text)
        metrics["markdown_report"] = {"file": str(report.relative_to(ROOT)),
                                       "sha256": hashlib.sha256(text.encode()).hexdigest()}
        code = 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        traceback.print_exc()
        metrics = {"aggregation_status": "failed", "jobs": [], "error": f"{type(error).__name__}: {error}"}
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"metrics": metrics, "config": {"source_files": [str(p) for p in args.source],
        "allocation_formula": "PBS Resource_List.ngpus * resources_used.walltime seconds / 3600",
        "charged_formula": "recorded allocation GPUh * recorded Resource_List.queue_charge_rate",
        "queue_wait_formula": "(stime - qtime) seconds; divide by 60 for minutes; final run_count=1 only",
        "eligible_wait_formula": "(stime - etime) seconds; divide by 60 for minutes; final run_count=1 only",
        "timestamp_timezone": "Asia/Singapore (+08:00)", "duplicate_policy": "exact agreeing final records once; conflicts fail"},
        "sources": sources}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
