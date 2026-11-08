"""Read accounting for explicitly named T00 jobs. Invoke through run_logged."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="t00_pbs_usage")
    parser.add_argument("jobs", nargs="+")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    observations = []
    hours = {}
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
        row = {"job_id": job, "qstat_exit": proc.returncode, "fields": fields}
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
               "complete_allocation_accounting": not errors}
    if not errors:
        metrics["total_allocation_gpu_hours"] = sum(hours.values())
    result = {"metrics": metrics, "config": {
        "formula": "allocated Resource_List.ngpus * resources_used.walltime_seconds / 3600",
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
