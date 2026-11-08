"""Read accounting for explicitly named T00 jobs. Invoke through run_logged."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess
import sys


PBS_TIMEZONE = timezone(timedelta(hours=8), "Asia/Singapore")


def queue_wait_record(fields):
    """Read recorded queue/start intervals without inferring missing history.

    qstat -fx timestamps have no offset; Hopper displays Asia/Singapore time.
    qtime can describe a later queue entry after transfer, so this interval is
    not a claim about cumulative waiting across every queue/requeue episode.
    Multiple runs cannot recover total waiting from one final timestamp set.
    """
    raw = {key: fields.get(key) for key in ("qtime", "etime", "stime")}
    parsed, problems = {}, {}
    months = {name: index for index, name in enumerate(
        ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}
    for key, value in raw.items():
        parsed[key] = None
        if value is None or value == "":
            problems[key] = "missing_" + key
            continue
        match = re.fullmatch(r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{2}):(\d{2}):(\d{2})\s+(\d{4})", str(value))
        try:
            if match is None:
                raise ValueError("unsupported PBS timestamp format")
            month, day, hour, minute, second, year = match.groups()
            parsed[key] = datetime(int(year), months[month], int(day), int(hour),
                                   int(minute), int(second), tzinfo=PBS_TIMEZONE)
        except (ValueError, KeyError):
            problems[key] = "invalid_" + key

    def interval(begin):
        if fields.get("job_state") != "F":
            return None, "job_not_final"
        if str(fields.get("run_count")) != "1":
            return None, "run_count_not_one_total_wait_unknown"
        if begin in problems or "stime" in problems:
            return None, problems.get(begin, problems.get("stime"))
        seconds = int((parsed["stime"] - parsed[begin]).total_seconds())
        if seconds < 0:
            return None, "stime_precedes_" + begin
        return seconds, None

    queue_seconds, queue_reason = interval("qtime")
    eligible_seconds, eligible_reason = interval("etime")
    return {"queue_wait_seconds": queue_seconds,
            "eligible_wait_seconds": eligible_seconds,
            "queue_wait_unavailable_reason": queue_reason,
            "eligible_wait_unavailable_reason": eligible_reason,
            "pbs_timestamp_timezone": "Asia/Singapore",
            "pbs_timestamp_utc_offset": "+08:00",
            "pbs_timestamps_raw": raw,
            "pbs_timestamps_iso8601": {
                key: value.isoformat() if value is not None else None for key, value in parsed.items()},
            "queue_wait_scope": "recorded_queue_entry_to_start_for_single_run_job_not_cumulative_queue_history"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="t00_pbs_usage")
    parser.add_argument("jobs", nargs="+")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    observations = []
    hours = {}
    queue_waits, eligible_waits = {}, {}
    errors = []
    for job in args.jobs:
        if not re.fullmatch(r"\d+\.hopper-m-\d+", job):
            parser.error("Only explicit Hopper job IDs are accepted")
        proc = subprocess.run(["qstat", "-fx", job], capture_output=True, text=True, timeout=120)
        log = root / "logs" / "pbs" / ("accounting_" + job + ".txt")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(proc.stdout + proc.stderr)
        fields = {}
        key = None
        for line in proc.stdout.splitlines():
            match = re.match(r"^    (\S+) = (.*)$", line)
            if match:
                key, value = match.groups()
                fields[key] = value
            elif line.startswith("\t") and key:
                fields[key] += line.strip()
        wait = queue_wait_record(fields)
        row = {"job_id": job, "qstat_exit": proc.returncode, "fields": fields, **wait}
        queue_waits[job] = wait["queue_wait_seconds"]
        eligible_waits[job] = wait["eligible_wait_seconds"]
        observations.append(row)
        if proc.returncode or fields.get("job_state") != "F":
            errors.append(job + ": accounting unavailable or job not finished")
            continue
        walltime = fields.get("resources_used.walltime")
        ngpus = fields.get("Resource_List.ngpus")
        if walltime and ngpus and re.fullmatch(r"\d+:\d{2}:\d{2}", walltime):
            h, m, s = map(int, walltime.split(":"))
            hours[job] = int(ngpus) * (h * 3600 + m * 60 + s) / 3600
        elif fields.get("run_count") == "0" and not fields.get("exec_host"):
            hours[job] = 0.0
        else:
            errors.append(job + ": missing measured allocation duration; not inferred")
    metrics = {"allocation_gpu_hours_by_job": hours,
               "complete_allocation_accounting": not errors,
               "queue_wait_seconds_by_job": queue_waits,
               "eligible_wait_seconds_by_job": eligible_waits}
    if not errors:
        metrics["total_allocation_gpu_hours"] = sum(hours.values())
    result = {"metrics": metrics, "config": {
        "formula": "allocated Resource_List.ngpus * resources_used.walltime_seconds / 3600",
        "queue_wait_formula": "stime - qtime, in seconds; only final jobs with run_count=1",
        "eligible_wait_formula": "stime - etime, in seconds; only final jobs with run_count=1",
        "timestamp_timezone": "Asia/Singapore (+08:00)",
        "queue_wait_scope": "Recorded intervals only; transfers/requeues may hide earlier waiting. Missing, invalid, or multi-run history remains null.",
        "scope": "PBS allocation GPU-hours, not an independently reconciled project credit charge",
        "jobs": observations, "errors": errors}}
    out = root / "results" / (args.name + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["config"], indent=2), flush=True)
    print("FINAL_METRICS_JSON: " + json.dumps(metrics), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
