"""Read-only, logged Hopper queue/capacity snapshot with explicit human alias."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.envs.thor_common import write_result


def queue_blocks(output):
    queues, current, key = {}, None, None
    for line in output.splitlines():
        match = re.match(r"^Queue:\s*(\S+)", line)
        if match:
            current, key = queues.setdefault(match[1], {}), None
            continue
        match = re.match(r"^\s{4}([^\s=]+)\s*=\s*(.*)$", line)
        if current is not None and match:
            key, value = match.groups()
            current[key] = value
        elif current is not None and key and line.startswith("\t"):
            current[key] += line.strip()
    return queues


def run_query(item):
    label, command = item
    started = time.monotonic()
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=180)
        return label, {"command": command, "exit_code": proc.returncode,
                       "stdout": proc.stdout, "stderr": proc.stderr,
                       "elapsed_seconds": time.monotonic() - started}
    except (OSError, subprocess.TimeoutExpired) as error:
        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        return label, {"command": command, "exit_code": None,
                       "stdout": decode(getattr(error, "stdout", "")),
                       "stderr": decode(getattr(error, "stderr", "")),
                       "error": f"{type(error).__name__}: {error}",
                       "elapsed_seconds": time.monotonic() - started}


def summarize_nodes(document):
    nodes = document.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError("pbsnodes JSON has no nonempty nodes mapping")
    rows = []
    for name, record in sorted(nodes.items()):
        available, assigned = record.get("resources_available", {}), record.get("resources_assigned", {})
        pool = available.get("node_pool", "")
        if not isinstance(pool, str) or not pool.startswith("nus_hpc"):
            continue
        total, used = available.get("ngpus"), assigned.get("ngpus")
        valid_counts = type(total) is int and type(used) is int and 0 <= used <= total
        state = str(record.get("state", ""))
        unavailable = any(word in state.split(",") for word in ("down", "offline", "unknown", "stale", "maintenance"))
        rows.append({"node": name, "pool": pool, "state": state,
                     "resources_available": available, "resources_assigned": assigned,
                     "reported_unassigned_gpus": total - used if valid_counts else None,
                     "unassigned_on_non_offline_node": total - used if valid_counts and not unavailable else
                        (0 if valid_counts and unavailable else None)})
    if not rows:
        raise ValueError("No nus_hpc node-pool records found")
    complete = all(row["unassigned_on_non_offline_node"] is not None for row in rows)
    return {"nodes": rows, "node_count": len(rows),
            "gpu_capacity_distribution": dict(Counter(str(row["resources_available"].get("ngpus")) for row in rows)),
            "unassigned_gpus_on_non_offline_nodes": sum(row["unassigned_on_non_offline_node"] for row in rows) if complete else None,
            "complete_node_counts": complete,
            "interpretation": "Snapshot of reported unassigned GPU resources, not a guarantee a job is schedulable; CPU, RAM, reservations, node-pool and queue constraints also apply."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--human-premium-alias", choices=("autox",), required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.name):
        parser.error("Invalid unique run name")
    directory = ROOT / "logs" / "runs" / args.name
    directory.mkdir(parents=True, exist_ok=False)
    queries = [("hpc_autox", ["hpc", "autox"]),
               ("queue_configuration", ["qstat", "-Qf"]),
               ("own_jobs", ["qstat", "-u", os.environ["USER"]]),
               ("node_capacity", ["pbsnodes", "-a", "-F", "json"]),
               ("node_summary", ["pbsnodes", "-aSj"])]
    config = {"name": args.name, "queries": [], "observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "read_only": True, "human_premium_alias": args.human_premium_alias,
              "human_confirmation": "2026-11-08: premium 指 Express/autox",
              "notes": ["Premium alias is explicitly confirmed by the human, not inferred from a queue name.",
                  "Existing queue_charge_rate fields are configuration evidence, not an independent project billing statement.",
                  "Node job-ID lists are omitted from the saved JSON capacity snapshot; all fields used for GPU counts are preserved."]}
    metrics = {"status": "incomplete", "premium_verified": False, "premium_queue_candidates": [],
               "queues": {}, "capacity": None, "errors": []}
    with ThreadPoolExecutor(max_workers=4) as executor:
        observations = dict(executor.map(run_query, queries))
    for label, _ in queries:
        result = observations[label]
        output = result.pop("stdout")
        result["original_stdout_sha256"] = hashlib.sha256(output.encode()).hexdigest()
        if result["exit_code"] != 0:
            metrics["errors"].append({"query": label, "error": result.get("error", result["stderr"]), "exit_code": result["exit_code"]})
        try:
            if label == "queue_configuration" and result["exit_code"] == 0:
                metrics["queues"] = queue_blocks(output)
                if not metrics["queues"]:
                    raise ValueError("No queue blocks parsed")
                metrics["premium_queue_candidates"] = [q for q in metrics["queues"] if "premium" in q.lower()]
                limits = {}
                for name in ("smallx", "mediumx", "largex"):
                    fields = metrics["queues"].get(name, {})
                    match = re.fullmatch(r"\[u:PBS_GENERIC=(\d+)\]", fields.get("max_run", ""))
                    limits[name] = int(match[1]) if match else None
                metrics["express_generic_per_user_running_limits"] = limits
                metrics["express_sum_of_queue_limits"] = sum(limits.values()) if all(v is not None for v in limits.values()) else None
            elif label == "node_capacity" and result["exit_code"] == 0:
                document = json.loads(output)
                metrics["capacity"] = summarize_nodes(document)
                for row in document.get("nodes", {}).values():
                    row.pop("jobs", None)
                output = json.dumps(document, indent=2) + "\n"
                result["saved_stdout_transformation"] = "Remove nodes.*.jobs only; preserve capacity and assignment fields"
        except (ValueError, TypeError, KeyError) as error:
            metrics["errors"].append({"query": label, "error": f"{type(error).__name__}: {error}"})
        path = directory / f"{label}.txt"
        path.write_text(output)
        result.update(label=label, saved_stdout=str(path.relative_to(ROOT)),
                      saved_stdout_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        config["queries"].append(result)
    metrics["status"] = "observed" if not metrics["errors"] else "incomplete"
    queues = metrics["queues"]
    rates = {q: queues.get(q, {}).get("resources_default.queue_charge_rate") for q in ("smallx", "mediumx", "largex")}
    destinations = queues.get("autox", {}).get("route_destinations", "").split(",")
    verified = (args.human_premium_alias == "autox" and set(destinations) == {"smallx", "mediumx", "largex"}
                and all(v is not None for v in rates.values()))
    metrics["premium_verified"] = verified
    metrics["premium_queue"] = "autox"
    metrics["configured_queue_charge_rates"] = rates
    metrics["premium_status"] = "Human-confirmed Express/autox alias; scheduler-configured rates below, not an independently reconciled credit statement" if verified else "Queue configuration could not verify the human-confirmed route"
    write_result(args.name, metrics, config)
    return 0 if not metrics["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
