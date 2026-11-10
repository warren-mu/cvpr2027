#!/usr/bin/env python3
"""Submit T00b/T01 work with a conservative shared scheduler-charged budget.

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
LIMITS = {"T00b": 60.0, "T01": 120.0}
TOTAL_LIMIT = 120.0
BUDGET_REVISION = "20261110_human16be1c57_premium_120_t01v2_16_e2e_4_e2epilot_12_e2epilotT2_0p5_g01_40_t02_40_v1"
SUBBUDGET_LIMITS = {"T01v2": 16.0, "E2E": 4.0, "E2Epilot": 12.0, "E2EpilotT2": 0.5, "G01": 40.0, "T02": 40.0}
PACK_GROUP = "PACK"  # A reservation route, never an additional budget.
JOB_ID = re.compile(r"\b\d+\.hopper-m-\d+\b")
EXTERNAL_FINAL_CONFIG = "configs/phase_external_final_jobs.json"
EXTERNAL_FINAL_VERSION = "g01_direct_qsub_final_cost_v1"


def native_pilot_budget_command(command, name, arm, group):
    """Exact native entry and budget identity; training settings stay in the driver."""
    if not isinstance(command, list) or not command or not re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(command[0]).name):
        return False
    args = command[1:]
    if args[:1] == ["-u"]:
        args = args[1:]
    if args[:3] != ["-m", "src.tools.smoke_e2e_pilot", "run"]:
        return False
    def option(flag):
        if args.count(flag) != 1 or args.index(flag) + 1 >= len(args):
            return None
        return args[args.index(flag) + 1]
    budget = option("--budget-group") if "--budget-group" in args else "E2Epilot"
    return (option("--name") == name and option("--arm") == arm and budget == group
            and "--engineering-smoke" not in args)



def diag01_timeout(command, name):
    """Only the explicitly approved native one-GPU diagnostic, never a subbudget alias."""
    if (not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command)
            or not re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(command[0]).name)):
        raise ValueError("DIAG01 requires its native Python module command")
    args = command[1:]
    if args[:1] == ["-u"]:
        args = args[1:]
    if args[:2] != ["-m", "src.tools.diag01_baselines"] or args.count("-m") != 1:
        raise ValueError("DIAG01 requires its exact native producer")
    def option(flag):
        if args.count(flag) != 1 or args.index(flag) + 1 >= len(args):
            raise ValueError("Missing/ambiguous DIAG01 argument: " + flag)
        return args[args.index(flag) + 1]
    if option("--name") != name:
        raise ValueError("DIAG01 command name differs")
    try:
        timeout = float(option("--timeout-seconds"))
    except (TypeError, ValueError) as error:
        raise ValueError("DIAG01 needs an explicit finite driver timeout") from error
    if not math.isfinite(timeout) or not 300 <= timeout <= 3540:
        raise ValueError("DIAG01 driver timeout is outside its original 300..3540 second bound")
    return timeout


def budget_group(task, name, command, declared=None):
    """Subbudgets share the approved120 total; historical diagnostic routes are retained."""
    diag01 = isinstance(name, str) and re.fullmatch(r"diag_diag01_baselines_[A-Za-z0-9_]+", name)
    if diag01 or "src.tools.diag01_baselines" in (command or []):
        if task != "T01" or not diag01 or declared is not None:
            raise ValueError("DIAG01 requires task T01, its own name and no subbudget declaration")
        diag01_timeout(command, name)
        return None
    pack = isinstance(name, str) and name.startswith("gpu_pack_")
    pack_launch = "src.tools.gpu_pack" in (command or [])
    if pack or pack_launch or declared == PACK_GROUP:
        if task != "T01" or not pack or declared != PACK_GROUP or (command and not pack_launch):
            raise ValueError("Shared GPU queue requires explicit PACK, task T01 and its gpu_pack_ producer/name")
        return PACK_GROUP
    named = isinstance(name, str) and (name == "pilot_pairs_v2" or name.startswith("pilot_pairs_v2_"))
    launched = any(isinstance(arg, str) and
                   (arg.endswith(("/t01v2_pairs.sh", "/t01v2_clone_pairs.sh", "/t01v2_stable_pairs.sh", "/t01v2_search_pairs.sh", "/t01v2_solvable_pairs.sh", "/t01v2_isolated_pairs.sh", "/t01v2_marked_pairs.sh"))
                    or arg in ("src.tools.t01v2_parallel", "src.tools.t01v2_clone_parallel", "src.tools.t01v2_stable_parallel", "src.tools.t01v2_search_parallel", "src.tools.t01v2_solvable_parallel", "src.tools.t01v2_isolated_parallel", "src.tools.t01v2_marked_parallel"))
                   for arg in (command or []))
    e2e = isinstance(name, str) and name.startswith("smoke_e2e_")
    e2e_pilot = isinstance(name, str) and name.startswith("smoke_e2e_pilot_")
    e2e_launch = any(isinstance(arg, str) and (arg.endswith("/e2e_export.sh")
        or arg in ("src.tools.e2e_export", "src.tools.smoke_e2e_views")) for arg in (command or []))
    g01 = isinstance(name, str) and name.startswith(("gen_train_", "gen_test_"))
    t02_launch = any(arg in ("src.tools.t02_export", "src.tools.t02_export_parallel", "src.tools.t02_cache_check") for arg in (command or []))
    t02 = isinstance(name, str) and name.startswith("t02_export_")
    if (isinstance(name, str) and name.startswith("smoke_e2e_pilot_T2_")) or declared == "E2EpilotT2":
        if (task != "T01" or not isinstance(name, str) or not name.startswith("smoke_e2e_pilot_T2_")
                or declared != "E2EpilotT2"
                or (command and not native_pilot_budget_command(command, name, "T2", "E2EpilotT2"))):
            raise ValueError("T2 requires its explicit E2EpilotT2 budget, T2 name and native five-update arm")
        return "E2EpilotT2"
    # Step 6 is a new explicitly declared ledger. Never infer it from a generic
    # smoke command or move historical steps 1--5 out of their original E2E cap.
    if e2e_pilot or declared == "E2Epilot":
        if (task != "T01" or not e2e_pilot or declared != "E2Epilot"
                or named or launched or e2e_launch or g01 or t02_launch):
            raise ValueError("E2E pilot requires task T01, explicit E2Epilot budget and its own smoke_e2e_pilot_ name")
        return "E2Epilot"
    if t02 or declared == "T02":
        if task != "T01" or not t02 or declared != "T02" or named or launched or e2e or e2e_launch or g01:
            raise ValueError("T02 export requires task T01, explicit T02 budget and its own t02_export_ name")
        if command and not t02_launch:
            raise ValueError("T02 export must use its declared single or parallel exporter")
        if "src.tools.t02_cache_check" in (command or []) and not name.startswith("t02_export_cache_"):
            raise ValueError("Fresh T02 cache validation requires its isolated t02_export_cache_ name")
        return "T02"
    if g01 or declared == "G01":
        if task != "T01" or not g01 or named or launched or e2e or e2e_launch or t02_launch or declared not in (None, "G01"):
            raise ValueError("G01 overnight expansion requires task T01 and a unique gen_train_/gen_test_ name")
        return "G01"
    if e2e or e2e_launch or declared == "E2E":
        if task != "T01" or not e2e or named or launched or declared not in (None, "E2E"):
            raise ValueError("E2E requires task T01 and a unique smoke_e2e name")
        return "E2E"
    if declared not in (None, "T01v2"):
        raise ValueError("Unknown phase subbudget")
    group = "T01v2" if named or launched or declared == "T01v2" else None
    if group and (task != "T01" or not named):
        raise ValueError("T01v2 requires task T01 and a unique pilot_pairs_v2 name")
    return group


def validate_host(value):
    if value is not None and (not isinstance(value, str) or not re.fullmatch(r"hopper-[0-9]+", value)):
        raise ValueError("host must be a Hopper compute hostname of the form hopper-<digits>")
    return value


def render_job_script(template, gpus, payload_path, host=None):
    """Add only the optional compute-host constraint to the normal select."""
    validate_host(host)
    if type(gpus) is not int or not 1 <= gpus <= 8:
        raise ValueError("Select requires one to eight GPUs on one node")
    anchor = "#PBS -l select=1:ngpus=1"
    if (template.count("@@CVPR_PAYLOAD@@") != 1 or template.count(anchor) != 1
            or template.splitlines().count(anchor) != 1
            or [line for line in template.splitlines() if line.startswith("#PBS -q")] != ["#PBS -q autox"]):
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


def positive_rate(value):
    try:
        rate = float(value)
    except (TypeError, ValueError):
        raise RuntimeError("Missing scheduler queue_charge_rate; refuse an inferred billing rate")
    if isinstance(value, bool) or not math.isfinite(rate) or rate <= 0:
        raise RuntimeError("queue_charge_rate must be finite and positive")
    return rate


def pbs_blocks(output, heading):
    blocks, current, key = {}, None, None
    for line in output.splitlines():
        match = re.match(r"^" + re.escape(heading) + r":\s*(\S+)", line)
        if match:
            current, key = blocks.setdefault(match[1], {}), None
            continue
        match = re.match(r"^    (\S+) = (.*)$", line)
        if current is not None and match:
            key, current_value = match.groups()
            current[key] = current_value
        elif current is not None and key and line.startswith("\t"):
            current[key] += line.strip()
    return blocks


def charge_policy():
    command = ["qstat", "-Qf"]
    query = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if query.returncode:
        raise RuntimeError("Cannot verify current Express queue rates: " + query.stderr)
    queues = pbs_blocks(query.stdout, "Queue")
    if set(queues.get("autox", {}).get("route_destinations", "").split(",")) != {"smallx", "mediumx", "largex"}:
        raise RuntimeError("Express/autox routing differs from the human-approved service")
    rates = {q: positive_rate(queues.get(q, {}).get("resources_default.queue_charge_rate"))
             for q in ("smallx", "mediumx", "largex")}
    return {"human_premium_alias": "Express/autox", "queue": "autox", "rates": rates,
            "queried_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": command, "raw_queue_configuration": query.stdout,
            "charge_basis": "PBS allocation GPU-hours multiplied by scheduler queue_charge_rate; not an independent credit-ledger reconciliation"}


def queue_for_gpus(gpus):
    return "smallx" if gpus <= 2 else "mediumx" if gpus <= 7 else "largex"


def live_job_rates(job_ids):
    if not job_ids:
        return {}
    proc = subprocess.run(["qstat", "-fx", *sorted(job_ids)], capture_output=True, text=True, timeout=180)
    if proc.returncode:
        raise RuntimeError("Cannot verify unfinished job billing rates: " + proc.stderr)
    blocks = pbs_blocks(proc.stdout, "Job Id")
    rates = {}
    for job_id in job_ids:
        fields = blocks.get(job_id, {})
        if fields.get("project") != "CFP04-SF-125":
            raise RuntimeError("Unexpected/missing project for budgeted job: " + job_id)
        rates[job_id] = positive_rate(fields.get("Resource_List.queue_charge_rate"))
    return rates


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


def _final_accounting(repo, wanted, paths=None):
    if not wanted:
        return {}
    spec = importlib.util.spec_from_file_location("phase_provenance", repo / "scripts/audit_provenance.py")
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    records = {}
    for path in sorted(paths if paths is not None else (repo / "results").glob("*.json")):
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
        cancelled = {}
        declared_cancelled = metrics.get("verified_never_started_cancelled_jobs", [])
        if declared_cancelled:
            sys.path.insert(0, str(repo))
            from src.tools.pbs_cancel_evidence import match_terminal, verify_evidence
            cancelled, references = verify_evidence(repo, data["config"].get("cancelled_evidence") or [], auditor)
            if (not isinstance(declared_cancelled, list) or len(set(declared_cancelled)) != len(declared_cancelled)
                    or not set(declared_cancelled) <= set(cancelled) or data.get("sources") != references):
                raise RuntimeError("Cancellation zero-cost claim differs from its authenticated source chain")
        for job in relevant:
            fields = job["fields"]
            hours = metrics.get("allocation_gpu_hours_by_job", {}).get(job["job_id"])
            if job["job_id"] in declared_cancelled:
                if isinstance(hours, bool) or hours != 0.0:
                    raise RuntimeError("Verified never-started cancellation must have exactly zero allocation GPU-hours")
                match_terminal(fields, cancelled[job["job_id"]]["postflight_fields"])
            else:
                hours = recorded_final_hours(fields, hours, job["job_id"])
            if hours is None:
                continue
            if job["job_id"] in records and records[job["job_id"]]["gpu_hours"] != hours:
                raise RuntimeError(f"Conflicting final accounting: {job['job_id']}")
            rate = positive_rate(fields.get("Resource_List.queue_charge_rate"))
            if job["job_id"] in records and records[job["job_id"]]["queue_charge_rate"] != rate:
                raise RuntimeError("Conflicting final billing rate: " + job["job_id"])
            records[job["job_id"]] = {"gpu_hours": hours, "queue_charge_rate": rate,
                                      "charged_gpu_hours": hours * rate,
                                      "accounting_result": str(path.relative_to(repo))}
    return records


def _external_final_jobs(repo, normal_jobs):
    """Register the one preserved CPU request that PBS actually charged a GPU.

    This is an explicit closed-job exception, not a submission transport, a
    synthetic submit_phase payload or permission for further direct qsub jobs.
    """
    config_path = repo / EXTERNAL_FINAL_CONFIG
    original_submit = repo / "results/diag_submit_test_round2_cpu_01.json"
    if not config_path.exists():
        if original_submit.exists():
            raise RuntimeError("Known external PBS allocation has no final-cost registration")
        return [], {}
    registry = json.loads(config_path.read_text())
    entries = registry.get("entries")
    if (registry.get("version") != EXTERNAL_FINAL_VERSION or registry.get("budget_batch") != BATCH
            or not isinstance(entries, list)):
        raise RuntimeError("Invalid external final-cost registry")
    if original_submit.exists() and not entries:
        raise RuntimeError("Known external PBS allocation cannot be omitted from its registry")
    spec = importlib.util.spec_from_file_location("external_phase_provenance", repo / "scripts/audit_provenance.py")
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)

    def read_ref(reference):
        if (not isinstance(reference, dict) or not isinstance(reference.get("file"), str)
                or re.fullmatch(r"[0-9a-f]{64}", str(reference.get("sha256"))) is None):
            raise RuntimeError("External evidence needs an exact original file/SHA")
        path = (repo / reference["file"]).resolve()
        path.relative_to(repo.resolve())
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != reference["sha256"]:
            raise RuntimeError("External original evidence changed: " + reference["file"])
        return path, body.decode("utf-8")

    def logged(reference):
        path, body = read_ref(reference)
        data = json.loads(body)
        meta = data.get("_meta", {})
        digest = hashlib.sha256(json.dumps(data.get("metrics"), sort_keys=True).encode()).hexdigest()
        if (path.parent != (repo / "results").resolve() or meta.get("exit_code") != 0
                or not isinstance(meta.get("end"), str) or not meta["end"]
                or meta.get("metrics_sha256") != digest or auditor.audit_file(str(path))["fail"]):
            raise RuntimeError("External source is not an authenticated completed logged result")
        log_path = (repo / meta["log_path"]).resolve()
        log_path.relative_to(repo.resolve())
        return path, data, log_path.read_text()

    submissions, final, seen = [], {}, set(normal_jobs)
    for entry in entries:
        job = entry.get("job_id")
        if (job != "709137.hopper-m-02" or entry.get("name") != "test_round2_cpu_01"
                or entry.get("task_id") != "T01" or entry.get("budget_group") != "G01"
                or entry.get("submission_transport") != "direct_qsub"
                or type(entry.get("requested_gpu_count")) is not int or entry["requested_gpu_count"] != 0):
            raise RuntimeError("Only the explicit ended G01 CPU-request/hook allocation is registered")
        if job in seen:
            raise RuntimeError("External final cost duplicates a normal payload or registry job: " + job)
        for payload_path in (repo / "logs/pbs").glob("*/payload.json"):
            receipt_path = payload_path.parent / "submission.json"
            if receipt_path.exists() and job in JOB_ID.findall(json.loads(receipt_path.read_text()).get("stdout", "")):
                raise RuntimeError("External job already has a normal payload receipt: " + job)
        seen.add(job)
        submission_path, submission, submit_log = logged(entry["submission_result"])
        transcript_path, transcript = read_ref(entry["submission_transcript"])
        script_path, script = read_ref(entry["request_script"])
        output_path, _ = read_ref(entry["pbs_output"])
        accounting_path, accounting, _ = logged(entry["accounting_result"])
        if (submission_path != original_submit.resolve()
                or transcript_path != (repo / "logs/pbs/test_round2_cpu_01/submission_transcript.txt").resolve()
                or script_path != (repo / "experiments/g01_test_round2_cpu_prepare.sh").resolve()
                or output_path != (repo / "logs/pbs/test_round2_cpu_01/pbs.out").resolve()
                or accounting_path != (repo / "results/g01_round2_cpu_attempt_accounting_01.json").resolve()):
            raise RuntimeError("External registry names different original artifacts")
        command = ["qsub", "-N", "cvpr27_cupre01", "-o", str(output_path),
                   "experiments/g01_test_round2_cpu_prepare.sh"]
        wrapper = ["python3", "-m", "src.tools.record_command", "--name", "diag_submit_test_round2_cpu_01", "--", *command]
        if (submission.get("config", {}).get("command") != command or submission["_meta"].get("command") != wrapper
                or submission["metrics"].get("command_exit_code") != 0
                or [line for line in submit_log.splitlines() if JOB_ID.fullmatch(line)] != [job]
                or [line for line in transcript.splitlines() if JOB_ID.fullmatch(line)] != [job]
                or [line for line in script.splitlines() if line.startswith("#PBS -P")] != ["#PBS -P CFP04-SF-125"]
                or [line for line in script.splitlines() if line.startswith("#PBS -q")] != ["#PBS -q interactive"]
                or [line for line in script.splitlines() if line.startswith("#PBS -l")] != [
                    "#PBS -l select=1:ncpus=1:mem=8gb:ngpus=0", "#PBS -l walltime=00:10:00"]):
            raise RuntimeError("Original direct-qsub request/receipt no longer proves the zero-GPU request")
        expected_account = ["python3", "src/tools/pbs_usage.py", "--name", "g01_round2_cpu_attempt_accounting_01", job]
        jobs = accounting.get("config", {}).get("jobs")
        if (accounting["_meta"].get("command") != expected_account or not isinstance(jobs, list)
                or len(jobs) != 1 or jobs[0].get("job_id") != job or jobs[0].get("qstat_exit") != 0
                or accounting["config"].get("errors") != []):
            raise RuntimeError("Exact original final accounting is missing")
        fields = jobs[0]["fields"]
        if (fields.get("job_state") != "F" or fields.get("project") != "CFP04-SF-125"
                or fields.get("Job_Owner", "").split("@")[0] != "e1583364"
                or fields.get("Job_Name") != "cvpr27_cupre01" or fields.get("queue") != "interactive"
                or fields.get("run_count") != "1" or fields.get("Resource_List.ngpus") != "1"
                or fields.get("Resource_List.walltime") != "00:10:00"
                or shlex.split(fields.get("Submit_arguments", "")) != command[1:]
                or fields.get("Output_Path", "").split(":", 1)[-1] != str(output_path)):
            raise RuntimeError("Final PBS allocation is not the authenticated original one-GPU hook result")
        observed = _final_accounting(repo, {job}, paths=[accounting_path]).get(job)
        if observed is None:
            raise RuntimeError("External cost has no verified final allocation; never reserve zero")
        final[job] = observed
        submissions.append({"task": "T01", "budget_group": "G01", "name": entry["name"],
            "job_id": job, "payload": None, "submission_transport": "direct_qsub",
            "requested_gpu_count": 0, "actual_gpu_count": 1, "gpus": 1,
            "requested_gpu_hours": 0.0, "recorded_reservation_rate": None, "submission_returncode": 0,
            "actual_queue": fields["queue"], "original_requested_queue": "interactive",
            "external_final_registration": {"file": EXTERNAL_FINAL_CONFIG,
                "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(), "entry": entry},
            "scope": "Original CPU request; actual PBS GPU allocation cost only. Not premium or submit_phase submission."})
    return submissions, final


def budget_state(repo, policy=None, live_rates=None):
    policy = policy if policy is not None else charge_policy()
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
        group = budget_group(task, payload["name"], payload.get("command"), payload.get("budget_group"))
        mixed = None
        if group == PACK_GROUP:
            from src.tools.pbs_mixed_accounting import validate_payload
            mixed = validate_payload(repo, payload)
        submissions.append({"task": task, "budget_group": group, "name": payload["name"], "job_id": job_id, "payload": str(path.relative_to(repo)),
                            "requested_gpu_hours": requested, "gpus": payload["requested_gpu_count"],
                            "recorded_reservation_rate": payload.get("requested_queue_charge_rate"),
                            "submission_returncode": receipt.get("returncode"),
                            **({"mixed_budget_authorization": mixed} if mixed is not None else {})})
    external, external_final = _external_final_jobs(repo, seen)
    final = _final_accounting(repo, seen)
    submissions.extend(external)
    final.update(external_final)
    live_rates = live_rates if live_rates is not None else live_job_rates(seen - set(final))
    used = {task: 0.0 for task in LIMITS}
    subgroup_costs = {group: [] for group in SUBBUDGET_LIMITS}
    for row in submissions:
        observed = final.get(row["job_id"])
        # A nonzero qsub return can also mean a lost acknowledgement. It does
        # not independently prove that the scheduler never accepted the job.
        if observed:
            rate = observed["queue_charge_rate"]
        elif row["job_id"]:
            rate = positive_rate(live_rates.get(row["job_id"]))
        else:
            # Lost acknowledgement is still fully reserved, at the greater
            # of recorded and currently configured route rates.
            rate = max(positive_rate(policy["rates"][queue_for_gpus(row["gpus"])]),
                       positive_rate(row["recorded_reservation_rate"]) if row["recorded_reservation_rate"] is not None else 0)
        allocation = observed["gpu_hours"] if observed else row["requested_gpu_hours"]
        cost = allocation * rate
        row.update(accounted_or_reserved_gpu_hours=cost, queue_charge_rate=rate,
                   accounted_or_reserved_allocation_gpu_hours=allocation,
                   basis="final_accounting" if observed else "full_request_reservation")
        if observed:
            row["accounting_result"] = observed["accounting_result"]
        used[row["task"]] += cost
        if row["budget_group"] == PACK_GROUP:
            from src.tools.pbs_mixed_accounting import finalized_split
            split = finalized_split(repo, row, observed) if observed else None
            groups = row["mixed_budget_authorization"]["possible_groups"]
            if observed and observed["gpu_hours"] == 0:
                group_costs = {group: 0.0 for group in groups}
                row["mixed_subbudget_basis"] = "canonical_zero_allocation"
            elif split is not None:
                group_costs = split["charged_gpu_hours_by_group"]
                row["mixed_subbudget_basis"] = "verified_final_unit_intervals_and_priority_remainder"
                row["mixed_final_accounting"] = split
            else:
                # Shared/T01 still counts this physical allocation exactly once.
                # Until an authenticated unit split exists, every possible group
                # conservatively retains the entire original requested bound.
                reserve_rate = max(rate, positive_rate(row["recorded_reservation_rate"]))
                reserve = row["requested_gpu_hours"] * reserve_rate
                group_costs = {group: max(reserve, cost) for group in groups}
                row["mixed_subbudget_basis"] = "full_allocation_reservation_per_possible_group"
            row["accounted_or_reserved_by_subbudget"] = group_costs
            for group, value in group_costs.items():
                subgroup_costs[group].append(value)
        elif row["budget_group"]:
            subgroup_costs[row["budget_group"]].append(cost)
    return {"batch": BATCH, "charge_policy": policy, "units": "scheduler_charged_gpu_hours", "limits_gpu_hours": LIMITS, "total_limit_gpu_hours": TOTAL_LIMIT,
            "accounted_or_reserved_by_task": used, "accounted_or_reserved_total": math.fsum(used.values()),
            "subbudget_limits_gpu_hours": SUBBUDGET_LIMITS,
            "accounted_or_reserved_by_subbudget": {group: math.fsum(costs) for group, costs in subgroup_costs.items()},
            "submissions": submissions}


def validate_request(task, gpus, seconds, name, state, group=None, mixed_authorization=None, *, command=None):
    permitted = {"T00b": (1, 2, 4, 8), "T01": (1, 2, 3, 4)}
    if task not in permitted or type(gpus) is not int or gpus not in permitted[task]:
        raise ValueError("GPU count is outside the authorized single-node task plan")
    allowed_name = r"t00b_[A-Za-z0-9_]+" if task == "T00b" else r"(?:diag_diag01_baselines_[A-Za-z0-9_]+|t01_[A-Za-z0-9_]+|pilot_pairs(?:_[A-Za-z0-9_]+)?|smoke_e2e_[A-Za-z0-9_]+|gen_(?:train|test)_[A-Za-z0-9_]+|t02_export_[A-Za-z0-9_]+|gpu_pack_[A-Za-z0-9_]+)"
    if not re.fullmatch(allowed_name, name):
        raise ValueError("Run name must identify the selected task")
    diag01 = bool(re.fullmatch(r"diag_diag01_baselines_[A-Za-z0-9_]+", name))
    if diag01 or "src.tools.diag01_baselines" in (command or []):
        budget_group(task, name, command, group)
        timeout = diag01_timeout(command, name)
        if gpus != 1 or not 0 < seconds <= 3600 or timeout > seconds - 60:
            raise ValueError("DIAG01 requires one GPU, PBS <=3600 seconds and >=60 seconds driver cleanup margin")
    if any(row["name"] == name for row in state["submissions"]):
        raise ValueError("Run name already has a submission; use a distinct evidence name")
    long_pilot = (task == "T01" and group == "E2Epilot" and any(
        name.startswith("smoke_e2e_pilot_" + arm + "_")
        and native_pilot_budget_command(command, name, arm, "E2Epilot") for arm in ("T0", "T4")))
    maximum_seconds = 15600 if long_pilot else (3 * 3600 if task == "T00b" else 4 * 3600)
    if not 0 < seconds <= maximum_seconds:
        raise ValueError("Request exceeds this task's per-job walltime limit")
    proposed = gpus * seconds / 3600 * positive_rate(state["charge_policy"]["rates"][queue_for_gpus(gpus)])
    if diag01 and proposed > 1.0 + 1e-12:
        raise ValueError("DIAG01 request exceeds its one scheduler-charged GPU-hour cap")
    if gpus >= 8 and seconds < 7201:
        raise ValueError("largex requires walltime of at least 02:00:01")
    if gpus * seconds / 3600 > 20:
        raise ValueError("This request exceeds the existing per-job 20 GPU-hour approval boundary")
    if state["accounted_or_reserved_by_task"][task] + proposed > LIMITS[task] + 1e-12:
        raise ValueError("Request would exceed the task's measured-plus-reserved GPU budget")
    if state["accounted_or_reserved_total"] + proposed > TOTAL_LIMIT + 1e-12:
        raise ValueError("Request would exceed the combined 120 GPU-hour budget")
    group = budget_group(task, name, command if diag01 else [], group)
    if group == PACK_GROUP:
        from src.tools.pbs_mixed_accounting import VERSION, PRIORITY
        groups = (mixed_authorization or {}).get("possible_groups")
        if (gpus not in (1, 4) or (mixed_authorization or {}).get("version") != VERSION
                or not isinstance(groups, list) or not groups or groups != [g for g in PRIORITY if g in groups]
                or mixed_authorization.get("residual_priority") != groups):
            raise ValueError("PACK needs its authenticated immutable plan and one/four GPUs")
        for possible in groups:
            recorded = state.get("accounted_or_reserved_by_subbudget", {}).get(possible)
            if (isinstance(recorded, bool) or not isinstance(recorded, (int, float))
                    or not math.isfinite(recorded) or recorded < 0
                    or recorded + proposed > SUBBUDGET_LIMITS[possible] + 1e-12):
                raise ValueError(f"Mixed pack full allocation reservation would exceed or lacks the {possible} subbudget")
        return
    if mixed_authorization is not None:
        raise ValueError("Ordinary single-group jobs cannot claim a mixed reservation")
    if group in ("E2E", "E2Epilot", "E2EpilotT2", "T02") and gpus != 1:
        raise ValueError("E2E/E2Epilot/E2EpilotT2/T02 use exactly one GPU per job")
    if group == "G01" and name.startswith("gen_test_") and gpus != 1:
        raise ValueError("Evaluation generation retains the frozen single-GPU collection plan")
    if group == "G01" and gpus == 2:
        raise ValueError("G01 follows the approved one-GPU or packed mediumx queue plan, never two GPUs")
    if group:
        recorded = state.get("accounted_or_reserved_by_subbudget", {}).get(group)
        if isinstance(recorded, bool) or not isinstance(recorded, (int, float)) or not math.isfinite(recorded) or recorded < 0:
            raise ValueError(f"Missing or invalid {group} measured-plus-reserved subbudget")
        if recorded + proposed > SUBBUDGET_LIMITS[group] + 1e-12:
            raise ValueError(f"Request would exceed the {group} {SUBBUDGET_LIMITS[group]:g} charged GPU-hour subbudget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", required=True, choices=sorted(LIMITS))
    parser.add_argument("--gpus", type=int, required=True, choices=(1, 2, 3, 4, 8))
    parser.add_argument("--walltime", required=True)
    parser.add_argument("--host", type=validate_host, help="Only a proven opening-black retry on hopper-22/28/30")
    parser.add_argument("--black-retry-evidence", help="Original logged strict black-gate guard result")
    parser.add_argument("--black-retry-evidence-sha256", help="Exact guard result SHA256")
    parser.add_argument("--retry-of", help="Original explicit PBS job ID; provenance metadata only")
    parser.add_argument("--budget-group", choices=sorted([*SUBBUDGET_LIMITS, PACK_GROUP]), help="Additional limit within the existing phase budget; PACK reserves each plan group")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--scratch", default=os.environ.get("CVPR_SCRATCH", f"/scratch/{os.environ.get('USER', '')}/cvpr2027"))
    parser.add_argument("--dry-run", action="store_true")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    if args.retry_of is not None and not re.fullmatch(r"\d+\.hopper-m-\d+", args.retry_of):
        parser.error("--retry-of requires an explicit Hopper PBS job ID")
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
        group = budget_group(args.task, args.name, command, args.budget_group)
        sys.path.insert(0, str(repo))
        mixed_authorization = None
        if group == PACK_GROUP:
            from src.tools.pbs_mixed_accounting import validate_submission
            mixed_authorization = validate_submission(repo, command, args.name, args.gpus)
        from src.tools.pbs_black_retry import validate_request_host
        host_authorization = validate_request_host(repo, host=args.host, retry_of=args.retry_of,
            evidence_path=args.black_retry_evidence, evidence_sha256=args.black_retry_evidence_sha256,
            name=args.name, command=command, task=args.task, group=group, gpus=args.gpus, wall_seconds=seconds)
        state = budget_state(repo)
        validate_request(args.task, args.gpus, seconds, args.name, state, group, mixed_authorization, command=command)
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
               "requested_wall_seconds": seconds, "requested_host": args.host, "dry_run": args.dry_run,
               "budget_revision": BUDGET_REVISION, "queue": "autox",
               "retry_of": args.retry_of, "budget_group": group,
               "requested_queue_charge_rate": state["charge_policy"]["rates"][queue_for_gpus(args.gpus)]}
    if host_authorization is not None:
        payload["black_retry_host_authorization"] = host_authorization
    if mixed_authorization is not None:
        payload["mixed_budget_authorization"] = mixed_authorization
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
