#!/usr/bin/env python3
"""Submit T00b/T01 work with a conservative shared allocation budget.

Unfinished or unaccounted jobs reserve their entire requested GPU walltime.
Only final, provenance-checked PBS accounting can replace that reservation.
Use this command through run_logged + record_command; never cancel peers.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

BATCH = "t00b_t01_20261108"
LIMITS = {"T00b": 8.0, "T01": 4.0}
TOTAL_LIMIT = 12.0
JOB_ID = re.compile(r"\b\d+\.hopper-m-\d+\b")


def validate_host(value):
    if value is not None and (not isinstance(value, str) or not re.fullmatch(r"hopper-[0-9]+", value)):
        raise ValueError("host must be a Hopper compute hostname of the form hopper-<digits>")
    return value


def render_job_script(template, gpus, payload_path, host=None):
    """Add only the optional compute-host constraint to the normal select."""
    validate_host(host)
    if type(gpus) is not int or gpus not in (1, 2):
        raise ValueError("Select requires one or two GPUs")
    anchor = "#PBS -l select=1:ngpus=1"
    if (template.count("@@CVPR_PAYLOAD@@") != 1 or template.count(anchor) != 1
            or template.splitlines().count(anchor) != 1):
        raise ValueError("Unexpected PBS template")
    resource = f"#PBS -l select=1:ngpus={gpus}"
    if host is not None:
        resource += f":host={host}"
    return template.replace(anchor, resource).replace("@@CVPR_PAYLOAD@@", shlex.quote(str(payload_path)))


def wall_seconds(value):
    match = re.fullmatch(r"(\d{2}):([0-5]\d):([0-5]\d)", value)
    if not match:
        raise ValueError("walltime must be HH:MM:SS")
    h, m, s = map(int, match.groups())
    return h * 3600 + m * 60 + s


def recorded_final_hours(fields, hours, job_id):
    if fields.get("job_state") != "F":
        return None
    if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not math.isfinite(hours) or hours < 0:
        return None
    if fields.get("run_count") == "0" and not fields.get("exec_host") and not fields.get("resources_used.walltime"):
        if hours != 0:
            raise RuntimeError(f"Never-started job has nonzero recorded cost: {job_id}")
        return 0.0
    if not fields.get("resources_used.walltime") or not fields.get("Resource_List.ngpus"):
        return None  # No substantiated measurement: retain the full reservation.
    seconds = wall_seconds(fields["resources_used.walltime"])
    ngpus = int(fields["Resource_List.ngpus"])
    if ngpus <= 0 or not math.isclose(hours, ngpus * seconds / 3600, abs_tol=1e-12):
        raise RuntimeError(f"Inconsistent final GPU cost: {job_id}")
    return hours


def _final_accounting(repo, wanted):
    if not wanted:
        return {}
    spec = importlib.util.spec_from_file_location("phase_provenance", repo / "scripts/audit_provenance.py")
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    records = {}
    for path in sorted((repo / "results").glob("*.json")):
        data = json.loads(path.read_text())
        jobs = data.get("config", {}).get("jobs", [])
        if not isinstance(jobs, list):
            continue
        relevant = [job for job in jobs if isinstance(job, dict) and job.get("job_id") in wanted
                    and job.get("fields", {}).get("job_state") == "F" and job.get("qstat_exit") == 0]
        if not relevant:
            continue
        meta = data.get("_meta", {})
        metrics = data.get("metrics", {})
        if meta.get("exit_code") != 0 or auditor.audit_file(str(path))["fail"]:
            continue
        producer_command = meta.get("command", [])
        if (len(producer_command) < 2 or
                (repo / producer_command[1]).resolve() != (repo / "src/tools/pbs_usage.py").resolve() or
                metrics.get("complete_allocation_accounting") is not True):
            continue
        log_path = (repo / meta["log_path"]).resolve()
        log_path.relative_to(repo)
        if json.dumps(data["config"], indent=2) not in log_path.read_text(errors="replace"):
            raise RuntimeError(f"PBS fields differ from the recorded accounting log: {path}")
        expected_hash = hashlib.sha256(json.dumps(metrics, sort_keys=True).encode()).hexdigest()
        if meta.get("metrics_sha256") != expected_hash:
            raise RuntimeError(f"Accounting metrics hash mismatch: {path}")
        for job in relevant:
            fields = job["fields"]
            hours = metrics.get("allocation_gpu_hours_by_job", {}).get(job["job_id"])
            hours = recorded_final_hours(fields, hours, job["job_id"])
            if hours is None:
                continue
            if job["job_id"] in records and records[job["job_id"]]["gpu_hours"] != hours:
                raise RuntimeError(f"Conflicting final accounting: {job['job_id']}")
            records[job["job_id"]] = {"gpu_hours": hours, "accounting_result": str(path.relative_to(repo))}
    return records


def budget_state(repo):
    submissions, seen = [], set()
    for path in sorted((repo / "logs/pbs").glob("*/payload.json")):
        payload = json.loads(path.read_text())
        if payload.get("budget_batch") != BATCH or payload.get("dry_run"):
            continue
        receipt_path = path.parent / "submission.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        ids = JOB_ID.findall(receipt.get("stdout", ""))
        if len(ids) > 1:
            raise RuntimeError(f"Ambiguous submission receipt: {receipt_path}")
        job_id = ids[0] if ids else None
        if job_id and job_id in seen:
            raise RuntimeError(f"Duplicate submission accounting for {job_id}")
        if job_id:
            seen.add(job_id)
        task = payload["task_id"]
        if task not in LIMITS:
            raise RuntimeError(f"Unknown task in this budget batch: {task}")
        requested = payload["requested_gpu_count"] * payload["requested_wall_seconds"] / 3600
        submissions.append({"task": task, "name": payload["name"], "job_id": job_id, "payload": str(path.relative_to(repo)),
                            "requested_gpu_hours": requested,
                            "submission_returncode": receipt.get("returncode")})
    final = _final_accounting(repo, seen)
    used = {task: 0.0 for task in LIMITS}
    for row in submissions:
        observed = final.get(row["job_id"])
        # A nonzero qsub return can also mean a lost acknowledgement. It does
        # not independently prove that the scheduler never accepted the job.
        cost = observed["gpu_hours"] if observed else row["requested_gpu_hours"]
        row.update(accounted_or_reserved_gpu_hours=cost,
                   basis="final_accounting" if observed else "full_request_reservation")
        if observed:
            row["accounting_result"] = observed["accounting_result"]
        used[row["task"]] += cost
    return {"batch": BATCH, "limits_gpu_hours": LIMITS, "total_limit_gpu_hours": TOTAL_LIMIT,
            "accounted_or_reserved_by_task": used, "accounted_or_reserved_total": math.fsum(used.values()),
            "submissions": submissions}


def validate_request(task, gpus, seconds, name, state):
    if task not in LIMITS or gpus not in (1, 2) or (task == "T01" and gpus != 1):
        raise ValueError("T01 requires one GPU; T00b permits one or two")
    allowed_name = r"t00b_[A-Za-z0-9_]+" if task == "T00b" else r"(?:t01_[A-Za-z0-9_]+|pilot_pairs(?:_[A-Za-z0-9_]+)?)"
    if not re.fullmatch(allowed_name, name):
        raise ValueError("Run name must identify the selected task")
    if any(row["name"] == name for row in state["submissions"]):
        raise ValueError("Run name already has a submission; use a distinct evidence name")
    if not 0 < seconds <= (3 * 3600 if task == "T00b" else 4 * 3600):
        raise ValueError("Request exceeds this task's per-job walltime limit")
    proposed = gpus * seconds / 3600
    if state["accounted_or_reserved_by_task"][task] + proposed > LIMITS[task] + 1e-12:
        raise ValueError("Request would exceed the task's measured-plus-reserved GPU budget")
    if state["accounted_or_reserved_total"] + proposed > TOTAL_LIMIT + 1e-12:
        raise ValueError("Request would exceed the combined 12 GPU-hour budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", required=True, choices=sorted(LIMITS))
    parser.add_argument("--gpus", type=int, required=True, choices=(1, 2))
    parser.add_argument("--walltime", required=True)
    parser.add_argument("--host", type=validate_host, help="Optional compute host, e.g. hopper-29")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--scratch", default=os.environ.get("CVPR_SCRATCH", f"/scratch/{os.environ.get('USER', '')}/cvpr2027"))
    parser.add_argument("--dry-run", action="store_true")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    command = argv[split + 1:]
    if not command:
        parser.error("Supply an explicit command after --")
    repo = args.repo.resolve()
    if not (repo / "scripts/run_logged.py").is_file() or not Path(args.scratch).is_absolute():
        parser.error("Require the project repository and an absolute scratch path")
    lock_path = repo / "logs/pbs/.t00b_t01_submission.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    budget_lock = lock_path.open("a")
    try:
        fcntl.flock(budget_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another submission is reserving this budget; retry after it finishes")
    try:
        seconds = wall_seconds(args.walltime)
        state = budget_state(repo)
        validate_request(args.task, args.gpus, seconds, args.name, state)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    for suffix in (repo / "results" / f"{args.name}.json", repo / "logs/runs" / f"{args.name}.log",
                   repo / "logs/runs" / f"{args.name}.meta.json"):
        if suffix.exists():
            parser.error(f"Refusing to overwrite existing evidence: {suffix}")
    token = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = repo / "logs/pbs" / f"{args.name}-{token}"
    directory.mkdir(parents=True, exist_ok=False)
    environment = {"CVPR_SCRATCH": args.scratch}
    for key in ("CVPR_CUDA_MODULE", "CVPR_VULKAN_MODULE", "CVPR_RENDER_DEADLINE_EPOCH"):
        if key in os.environ:
            environment[key] = os.environ[key]
    payload = {"repo": str(repo), "name": args.name, "command": command, "environment": environment,
               "task_id": args.task, "budget_batch": BATCH, "requested_gpu_count": args.gpus,
               "requested_wall_seconds": seconds, "requested_host": args.host, "dry_run": args.dry_run}
    payload_path = directory / "payload.json"
    payload_path.write_text(json.dumps(payload, indent=2) + "\n")
    payload_path.chmod(0o444)
    (directory / "budget_before_submission.json").write_text(json.dumps(state, indent=2) + "\n")
    template = (repo / "scripts/pbs/job_1gpu.pbs").read_text()
    try:
        rendered = render_job_script(template, args.gpus, payload_path, args.host)
    except ValueError as error:
        parser.error(str(error))
    script = directory / "job.pbs"
    script.write_text(rendered)
    script.chmod(0o444)
    qsub = ["qsub", "-N", "cvpr27_" + token[-8:], "-l", f"walltime={args.walltime}",
            "-o", str(directory / "pbs.out"), str(script)]
    print("BUDGET_BEFORE_SUBMISSION: " + json.dumps(state, sort_keys=True), flush=True)
    print("Submit: " + shlex.join(qsub), flush=True)
    if args.dry_run:
        print(script.read_text(), end="")
        return 0
    try:
        completed = subprocess.run(qsub, capture_output=True, text=True, check=False)
        receipt = {"command": qsub, "stdout": completed.stdout, "stderr": completed.stderr,
                   "returncode": completed.returncode}
    except OSError as error:
        receipt = {"command": qsub, "stdout": "", "stderr": str(error), "returncode": None}
    (directory / "submission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    sys.stdout.write(receipt["stdout"])
    sys.stderr.write(receipt["stderr"])
    return receipt["returncode"] if receipt["returncode"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
