"""Verify explicit logged Q -> successful qdel -> F, without inventing PBS fields.

Only zero allocation accounting for that proven never-started cancellation is
supported. Missing usage by itself is never a zero measurement. Raw fields and
all source/log hashes remain available for an independent audit.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re

from src.tools.phase_watch import load_logged, load_module

JOB = re.compile(r"\d+\.hopper-m-\d+\Z")
IDENTITY = ("Job_Name", "Job_Owner", "project", "Resource_List.ngpus", "Resource_List.select", "Resource_List.walltime")


def flatten_fields(fields):
    result = {}
    for key, value in fields.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                result[key + "." + subkey] = str(subvalue) if subvalue is not None else None
        else:
            result[key] = str(value) if value is not None else None
    return result


def not_started(fields, state):
    return (fields.get("job_state") == state and fields.get("run_count") in (None, "", "0")
            and not any(fields.get(key) for key in ("stime", "exec_host", "exec_vnode"))
            and fields.get("resources_used") in (None, "")
            and not any(key.startswith("resources_used.") and value not in (None, "") for key, value in fields.items()))


def match_terminal(fields, proof):
    if (not not_started(fields, "F") or not not_started(proof, "F")
            or any(not proof.get(key) or fields.get(key) != proof[key] for key in IDENTITY)):
        raise ValueError("Current terminal PBS fields do not match the never-started cancellation evidence")
    return True


def pbs_json(log):
    decoder, matches = json.JSONDecoder(), []
    for match in re.finditer(r"(?m)^\s*\{", log):
        try:
            value, _ = decoder.raw_decode(log[match.start():].lstrip())
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("Jobs"), dict):
            matches.append(value["Jobs"])
    if len(matches) != 1:
        raise ValueError("Expected one complete PBS JSON observation in the saved command log")
    return {job: flatten_fields(fields) for job, fields in matches[0].items()}


def validate_chain(records, logs, user):
    """Pure semantic verifier; caller first authenticates all three run logs."""
    if len(records) != 3 or len(logs) != 3 or not user:
        raise ValueError("Require preflight, cancellation and postflight from the submitting owner")
    times, commands, hosts = [], [], []
    for record in records:
        meta, config = record.get("_meta", {}), record.get("config", {})
        command = config.get("command")
        if (meta.get("exit_code") != 0 or record.get("metrics", {}).get("command_exit_code") != 0
                or record.get("error") is not None or not isinstance(command, list)
                or not all(isinstance(arg, str) for arg in command)):
            raise ValueError("Cancellation evidence command did not complete successfully")
        commands.append(command)
        times.append((datetime.fromisoformat(meta["start"]), datetime.fromisoformat(meta["end"])))
        hosts.append(meta.get("host"))
    if (not hosts[0] or len(set(hosts)) != 1 or any(start > end for start, end in times)
            or not times[0][1] <= times[1][0] <= times[1][1] <= times[2][0]):
        raise ValueError("Cancellation evidence has inconsistent host or observation ordering")
    pre, delete, post = commands
    if (not delete or Path(delete[0]).name != "qdel" or not delete[1:]
            or any(not JOB.fullmatch(job) for job in delete[1:]) or len(set(delete[1:])) != len(delete[1:])):
        raise ValueError("Cancellation must be one explicit qdel argument vector without flags")
    snapshots = []
    for command, log in ((pre, logs[0]), (post, logs[2])):
        if (len(command) < 5 or Path(command[0]).name != "qstat" or command[1:4] != ["-xf", "-F", "json"]
                or any(not JOB.fullmatch(job) for job in command[4:]) or len(set(command[4:])) != len(command[4:])):
            raise ValueError("Pre/post evidence must query explicit IDs through qstat -xf -F json")
        rows = pbs_json(log)
        if set(rows) != set(command[4:]) or not set(delete[1:]) <= set(rows):
            raise ValueError("PBS JSON inventory differs from its logged explicit query")
        snapshots.append(rows)
    verified = {}
    for job in delete[1:]:
        before, after = snapshots[0][job], snapshots[1][job]
        if (not not_started(before, "Q") or not not_started(after, "F")
                or before.get("project") != "CFP04-SF-125"
                or not str(before.get("Job_Owner", "")).startswith(user + "@")
                or any(not before.get(key) or after.get(key) != before[key] for key in IDENTITY)):
            raise ValueError("A cancelled job started, changed identity/resources, or lacks complete Q/F evidence: " + job)
        verified[job] = {"basis": "verified_logged_queued_cancellation", "allocation_gpu_hours": 0.0,
                         "preflight_fields": before, "postflight_fields": after,
                         "scope": "explicit Q/no allocation -> successful qdel -> F/no allocation; no run_count value is synthesized"}
    return verified


def verify_evidence(repo, paths, auditor=None, user=None):
    repo = Path(repo).resolve()
    if len(paths) != 3 or len(set(map(str, paths))) != 3:
        raise ValueError("Supply three distinct ordered cancellation result paths")
    if auditor is None:
        auditor = load_module("cancel_evidence_auditor", repo / "scripts/audit_provenance.py")
    records, sources, logs = [], [], []
    for supplied in paths:
        path = (repo / supplied).resolve()
        data, reference = load_logged(repo, path, auditor)
        meta, command = data["_meta"], data.get("config", {}).get("command")
        argv = meta.get("command", [])
        # Both exact Python entry forms execute this same tracked recorder.
        # Never rewrite the stored command or accept a module name in argv data.
        arguments = None
        if (isinstance(argv, list) and len(argv) >= 6 and all(isinstance(arg, str) for arg in argv)
                and re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(argv[0]).name)):
            if argv[1:3] == ["-m", "src.tools.record_command"]:
                arguments = argv[3:]
            elif (repo / argv[1]).resolve() == repo / "src/tools/record_command.py":
                arguments = argv[2:]
        if (arguments is None or arguments[:3] != ["--name", meta.get("name"), "--"]
                or arguments[3:] != command
                or meta.get("metrics_sha256") != hashlib.sha256(json.dumps(data["metrics"], sort_keys=True).encode()).hexdigest()):
            raise ValueError("Source config command is not its audited record_command invocation")
        reference.update(file=reference["path"], metrics_sha256=meta["metrics_sha256"])
        records.append(data)
        sources.append(reference)
        logs.append((repo / reference["log_path"]).read_text(errors="replace"))
    verified = validate_chain(records, logs, user or os.environ.get("USER"))
    return verified, sources
