"""One logged phase-job observation; account finals and retry proven black gates.

No daemon, cancellation, qalter, or inferred result. Every retry goes through the
unchanged charged-budget submitter. An immutable intent prevents duplicate
submissions after interruption or an uncertain acknowledgement; root reviews a
blocked intent before any manual retry, including a definite budget rejection.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
QSTAT = "/cm/shared/apps/pbspro/current/bin/qstat"
JOB_ID = re.compile(r"\d+\.hopper-m-\d+\Z")
ENTRYPOINTS = {"experiments/t00b_scale_smoke.sh": "scale",
               "experiments/t01_parallel_pairs.sh": "t01",
               "experiments/t00b_pack_smoke.sh": "pack",
               "experiments/t00b_gated_diagnostics.sh": "diagnostics",
               "experiments/t01_collision_pack.sh": "probe_pack"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require_unused_run(repo, name):
    paths = [repo / "results" / f"{name}.json", repo / "logs/runs" / name,
             *[repo / "logs/runs" / f"{name}{suffix}" for suffix in (".log", ".meta.json", ".diff")]]
    if any(path.exists() for path in paths):
        raise FileExistsError("Preserve existing child evidence: " + name)


def submissions(repo, batch, pattern):
    records, seen = [], set()
    for path in sorted((repo / "logs/pbs").glob("*/payload.json")):
        payload = json.loads(path.read_text())
        if payload.get("budget_batch") != batch or payload.get("dry_run"):
            continue
        receipt_path = path.with_name("submission.json")
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        ids = pattern.findall(receipt.get("stdout", ""))
        if len(ids) > 1 or (ids and ids[0] in seen):
            raise ValueError("Ambiguous/duplicate phase job receipt: " + str(receipt_path))
        job_id = ids[0] if ids else None
        if job_id:
            seen.add(job_id)
        records.append({"payload": payload, "path": str(path.relative_to(repo)),
                        "payload_sha256": digest(path), "job_id": job_id,
                        "submission_returncode": receipt.get("returncode")})
    return records


def entrypoint(command, name, repo):
    """Only the explicit new argv launchers are eligible; no shell strings."""
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise ValueError("Expected an explicit argument vector")
    argv = list(command)
    if (len(argv) >= 6 and Path(argv[0]).name in ("python", "python3", "python3.10")
            and (repo / argv[1]).resolve() == repo / "src/tools/record_command.py"):
        if argv[2:5] != ["--name", name, "--"]:
            raise ValueError("Outer record_command name/shape differs")
        argv = argv[5:]
    if len(argv) < 4 or argv[0] != "bash":
        raise ValueError("Not an authorized new phase entrypoint")
    script = str((repo / argv[1]).resolve().relative_to(repo))
    if script not in ENTRYPOINTS or argv.count("--name") != 1:
        raise ValueError("Legacy/unknown launcher is not automatically retried")
    if argv[argv.index("--name") + 1] != name:
        raise ValueError("Launcher name differs from payload")
    if any(part == name and (i == 0 or command[i - 1] != "--name") for i, part in enumerate(command)):
        raise ValueError("Run name also occurs outside a --name argument")
    return ENTRYPOINTS[script]


def load_logged(repo, path, auditor):
    path = path.resolve()
    path.relative_to(repo / "results")
    data = json.loads(path.read_text())
    meta, metrics = data.get("_meta", {}), data.get("metrics", {})
    if auditor.audit_file(str(path))["fail"]:
        raise ValueError("Result failed provenance audit: " + str(path))
    log = (repo / meta["log_path"]).resolve()
    log.relative_to(repo)
    finals = [line.split("FINAL_METRICS_JSON:", 1)[1].strip()
              for line in log.read_text(errors="replace").splitlines() if line.startswith("FINAL_METRICS_JSON:")]
    if not finals or json.loads(finals[-1]) != metrics:
        raise ValueError("Missing/mismatched exact FINAL_METRICS_JSON")
    return data, {"path": str(path.relative_to(repo)), "sha256": digest(path),
                  "log_path": str(log.relative_to(repo)), "log_sha256": digest(log)}


def final_observations(repo, final, auditor):
    """Recover exact terminal fields from the already verified accounting file.

    These are historical terminal observations, never presented as live qstat.
    A missing/duplicate job row cannot silently fall back to a live query.
    """
    sources, observed = {}, {}
    for job_id, record in sorted(final.items()):
        path = (repo / record["accounting_result"]).resolve()
        if path not in sources:
            data, reference = load_logged(repo, path, auditor)
            command = data.get("_meta", {}).get("command", [])
            if (data.get("_meta", {}).get("exit_code") != 0 or len(command) < 2
                    or (repo / command[1]).resolve() != repo / "src/tools/pbs_usage.py"
                    or data.get("metrics", {}).get("complete_allocation_accounting") is not True):
                raise ValueError("Cached terminal fields require successful official PBS accounting")
            # Config is outside metrics_sha256; require the exact config printed
            # by pbs_usage in its audited log, as the budget verifier also does.
            if json.dumps(data["config"], indent=2) not in (repo / reference["log_path"]).read_text(errors="replace"):
                raise ValueError("Cached PBS fields differ from their original logged configuration")
            sources[path] = data, reference
        data, reference = sources[path]
        rows = data.get("config", {}).get("jobs", [])
        matches = [row for row in rows if isinstance(row, dict) and row.get("job_id") == job_id]
        if (len(matches) != 1 or matches[0].get("qstat_exit") != 0
                or not isinstance(matches[0].get("fields"), dict)
                or matches[0]["fields"].get("job_state") != "F"
                or data.get("metrics", {}).get("allocation_gpu_hours_by_job", {}).get(job_id) != record["gpu_hours"]):
            raise ValueError("Missing, ambiguous, nonfinal, or inconsistent cached PBS job: " + job_id)
        observed[job_id] = {"job_id": job_id, "fields": matches[0]["fields"],
                            "observation_source": "final_accounting", "source": reference}
    return observed


def query_unaccounted(job_ids, cached):
    """One explicit live query only for IDs without a verified terminal row."""
    if set(cached) - set(job_ids) or any(row.get("job_id") != job_id or row.get("fields", {}).get("job_state") != "F"
                                       for job_id, row in cached.items()):
        raise ValueError("Cached observations do not match this submitted-job inventory")
    live_ids = sorted(set(job_ids) - set(cached))
    command = [QSTAT, "-fx", *live_ids] if live_ids else None
    result = {"command": command, "queried_job_ids": live_ids, "returncode": None,
              "stdout": "", "stderr": "", "timeout_seconds": 180, "timed_out": False,
              "skipped_reason": None if live_ids else "all_jobs_have_verified_final_accounting"}
    if command:
        try:
            query = subprocess.run(command, capture_output=True, text=True, timeout=180)
            result.update(returncode=query.returncode, stdout=query.stdout, stderr=query.stderr)
        except subprocess.TimeoutExpired as error:
            text = lambda value: value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            result.update(timed_out=True, stdout=text(error.stdout), stderr=text(error.stderr))
    return result


def combine_observations(job_ids, cached, raw, source, submit):
    expected = sorted(set(job_ids) - set(cached))
    headers = re.findall(r"^Job Id:\s*(\S+)", raw["stdout"], re.M)
    if (raw.get("queried_job_ids") != expected or sorted(headers) != expected
            or len(headers) != len(set(headers))):
        raise ValueError("Live PBS observations are missing, duplicated, or not explicitly requested")
    live = submit.pbs_blocks(raw["stdout"], "Job Id")
    return {**cached, **{job_id: {"job_id": job_id, "fields": fields,
                               "observation_source": "live_qstat", "source": source}
                        for job_id, fields in live.items()}}


def authorized_shard_command(command):
    """Allow the real unbuffered Python launcher, not a module name in argv data."""
    if (not isinstance(command, list) or len(command) < 3
            or not all(isinstance(part, str) for part in command)
            or not re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(command[0]).name)):
        return False
    arguments = command[1:]
    if arguments[:1] == ["-u"]:
        arguments = arguments[1:]
    return arguments[:2] in (["-m", "src.envs.pair_shard"],
                             ["-m", "src.envs.pair_shard_approved"])


def validate_black_retry(payload, job_id, fields, outer, gate, gate_name, kind, child=None, child_record=None):
    """Pure eligibility check after both result provenance chains are audited."""
    if fields.get("project") != "CFP04-SF-125" or fields.get("job_state") != "F" or fields.get("Exit_status") != "86":
        raise ValueError("PBS did not finish this project job with exit 86")
    if (outer.get("_meta", {}).get("exit_code") != 86 or outer["_meta"].get("command") != payload["command"]
            or outer["_meta"].get("name") != payload["name"]):
        raise ValueError("Outer result is not this exact submitted command with exit 86")
    config = outer.get("config", {})
    outer_job = config.get("pbs_job_id", config.get("pbs_jobid", (config.get("allocation") or {}).get("pbs_jobid")))
    if outer_job != job_id:
        raise ValueError("Outer result belongs to another PBS job")
    expected = (payload["name"] + "_j" + re.sub(r"[^A-Za-z0-9]", "_", job_id) + "_gate"
                if kind == "t01" else payload["name"] + "_gate")
    metrics, gate_config, meta = gate.get("metrics", {}), gate.get("config", {}), gate.get("_meta", {})
    allocation = gate_config.get("allocation") or {}
    child_failure = kind == "t01" and child is not None
    gate_code = 0 if child_failure else 86
    count = payload["requested_gpu_count"]
    devices = allocation.get("devices", [])
    nodes = {part.split("/")[0] for part in fields.get("exec_host", "").split("+") if part}
    if (gate_name != expected or meta.get("name") != expected or meta.get("exit_code") != gate_code
            or metrics.get("exit_code") != gate_code or metrics.get("passed") is not child_failure
            or metrics.get("errors") or metrics.get("cleanup_errors")
            or gate_config.get("pbs_jobid") != job_id or allocation.get("pbs_jobid") != job_id
            or metrics.get("completed_gpu_count") != count or metrics.get("expected_gpu_count") != count
            or [row.get("logical_index") for row in devices] != list(range(count))
            or len({row.get("gpu_uuid") for row in devices}) != count
            or nodes != {gate_config.get("node")}):
        raise ValueError("Gate black-frame/allocation evidence is incomplete or inconsistent")
    command = meta.get("command", [])
    if ("src.tools.render_gate" not in command or "--name" not in command
            or command[command.index("--name") + 1] != expected):
        raise ValueError("Gate was not produced by the explicit shared render gate")
    observed = metrics.get("devices", [])
    if len(observed) != count:
        raise ValueError("Gate is missing per-device capture evidence")
    for expected_device, row in zip(devices, observed):
        observation = row.get("observation", {})
        captures = observation.get("captures", [])
        if (row.get("worker_error") or observation.get("error") or observation.get("cleanup_error")
                or observation.get("gpu_identity") != expected_device or len(captures) != 2
                or any(capture.get("valid_shape") is not True or capture.get("action_success") is not True for capture in captures)
                or (child_failure and any(capture.get("valid") is not True for capture in captures))):
            raise ValueError("Gate per-device captures or cleanup are incomplete")
    if child_failure:
        cm, cc, meta = child.get("metrics", {}), child.get("config", {}), child.get("_meta", {})
        shard, record = cc.get("shard", {}), child_record or {}
        identity = shard.get("gpu_identity", {})
        stop = cm.get("black_frame_stop", {})
        measurement = stop.get("measurement", {})
        if (outer.get("metrics", {}).get("cleanup_errors") or cm.get("cleanup_errors")
                or cm.get("kind") != "t01_scene_collect" or meta.get("exit_code") != 86
                or record.get("exit_code") != 86 or record.get("provenance_error")
                or meta.get("name") != record.get("run_name") or cm.get("black_frame_detected") is not True
                or not (cm.get("source_black_frame_detected") is True or cm.get("capture_black_frame_detected") is True)
                or shard.get("pbs_jobid") != job_id or identity not in devices
                or shard.get("child_cuda_visible_devices") != identity.get("gpu_uuid")
                or shard.get("parent_cuda_visible_devices") != allocation.get("cuda_visible_devices")
                or record.get("gpu_uuid") != identity.get("gpu_uuid")
                or record.get("logical_index") != identity.get("logical_index")
                or record.get("scene") != shard.get("scene") or stop.get("scene") != shard.get("scene")
                or stop.get("stage") not in ("source", "capture")
                or measurement.get("shape") != [224, 224, 3] or measurement.get("dtype") != "uint8"
                or measurement.get("nonzero_values") != 0 or measurement.get("min") != 0 or measurement.get("max") != 0
                or not authorized_shard_command(meta.get("command"))):
            raise ValueError("T01 child black capture, cleanup, or allocation evidence is inconsistent")
        return "t01_scene_black_capture"
    captures = [capture for row in metrics.get("devices", [])
                for capture in row.get("observation", {}).get("captures", [])]
    if metrics.get("black_frame_detected") is not True or not any(capture.get("valid_shape") is True and capture.get("black") is True
               and capture.get("nonzero_rgb_values") == 0 for capture in captures):
        raise ValueError("No correctly shaped, actually measured black capture")
    return "render_gate_black_capture"


def child_black_evidence(repo, payload, outer, gate_evidence, auditor):
    """Read one explicitly linked failed shard, including its raw PNG/hash proof."""
    failures = []
    for record in outer.get("metrics", {}).get("scene_runs", []):
        if record.get("exit_code") != 86 or record.get("provenance_error"):
            continue
        try:
            child, reference = load_logged(repo, repo / record["result_path"], auditor)
            if (record.get("source", {}).get("sha256") != reference["sha256"]
                    or child.get("config", {}).get("shard", {}).get("gate_source", {}).get("sha256") != gate_evidence["sha256"]):
                raise ValueError("Outer/shard/gate source hashes differ")
            stop = child.get("metrics", {}).get("black_frame_stop", {})
            scratch = Path(payload["environment"]["CVPR_SCRATCH"]).resolve()
            portable = bool(stop.get("portable_file") and stop.get("portable_image"))
            if portable:
                raw, image = (repo / stop["portable_file"]).resolve(), (repo / stop["portable_image"]).resolve()
                raw.relative_to(repo / "logs/runs")
                image.relative_to(repo / "logs/runs")
                if stop.get("portable_file_sha256") != stop["sha256"] or stop.get("portable_image_sha256") != stop["image_sha256"]:
                    raise ValueError("Portable copy hashes differ from original black capture")
            else:
                raw, image = Path(stop["file"]).resolve(), Path(stop["image"]).resolve()
                raw.relative_to(scratch)
                image.relative_to(scratch)
            if digest(raw) != stop["sha256"] or digest(image) != stop["image_sha256"]:
                raise ValueError("Raw black capture evidence changed")
            supplemental = {"file", "sha256", "portable_file", "portable_file_sha256", "portable_image", "portable_image_sha256"}
            if json.loads(raw.read_text()) != {key: value for key, value in stop.items() if key not in supplemental}:
                raise ValueError("Raw black capture metadata differs from logged measurements")
            reference["black_capture"] = {"file": str(raw), "sha256": digest(raw),
                                          "image": str(image), "image_sha256": digest(image), "portable_copy": portable}
            return child, record, reference
        except (OSError, ValueError, KeyError) as error:
            failures.append(str(error))
    raise ValueError("No complete audited T01 black-capture shard: " + "; ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--once", required=True, action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.name):
        parser.error("Require a unique safe observation name")
    output = ROOT / "results" / f"{args.name}.json"
    directory = ROOT / "logs/runs" / args.name
    if output.exists() or directory.exists():
        raise FileExistsError("Preserve prior observation; use a unique --name")
    directory.mkdir(parents=True, exist_ok=False)
    metrics = {"observations": [], "accounting_runs": [], "retries": [], "errors": []}
    config = {"name": args.name, "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "scope": "single read/account/retry iteration; no cancellation or job alteration"}
    lock = None
    try:
        submit = load_module("phase_submit_watch", ROOT / "scripts/pbs/submit_phase.py")
        auditor = load_module("phase_audit_watch", ROOT / "scripts/audit_provenance.py")
        lock = (ROOT / "logs/pbs/.phase_watch.lock").open("a")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = submissions(ROOT, submit.BATCH, submit.JOB_ID)
        by_id = {row["job_id"]: row for row in records if row["job_id"]}
        already_retried = {row["payload"].get("retry_of") for row in records}
        final = submit._final_accounting(ROOT, set(by_id))
        cached = final_observations(ROOT, final, auditor)
        raw = query_unaccounted(by_id, cached)
        save_new(directory / "qstat.json", raw)
        raw_reference = {"path": str((directory / "qstat.json").relative_to(ROOT)), "sha256": digest(directory / "qstat.json")}
        config.update(submissions=records, raw_qstat=raw_reference,
                      cached_final_job_ids=sorted(cached), live_queried_job_ids=raw["queried_job_ids"])
        if raw["timed_out"] or (raw["command"] and raw["returncode"] != 0):
            raise RuntimeError("Live qstat failed/timed out; partial raw observation retained, no automated action")
        observations = combine_observations(by_id, cached, raw, raw_reference, submit)
        fields_by_id = {job_id: row["fields"] for job_id, row in observations.items()}
        for job_id, row in observations.items():
            fields = row["fields"]
            if fields.get("project") != "CFP04-SF-125" or fields.get("Job_Owner", "").split("@")[0] != os.environ.get("USER"):
                raise RuntimeError("Missing/unexpected project or owner: " + job_id)
            metrics["observations"].append(row)
        pending = [job for job, fields in fields_by_id.items() if job in by_id and fields.get("job_state") == "F" and job not in final]
        if pending:
            name = args.name + "_accounting"
            require_unused_run(ROOT, name)
            argv = [sys.executable, "scripts/run_logged.py", "--name", name, "--", sys.executable,
                    "src/tools/pbs_usage.py", "--name", name, *sorted(pending)]
            code = subprocess.run(argv, cwd=ROOT).returncode
            metrics["accounting_runs"].append({"name": name, "command": argv, "exit_code": code})
            final = submit._final_accounting(ROOT, set(by_id))
        for job_id, record in by_id.items():
            payload, fields = record["payload"], fields_by_id[job_id]
            if fields.get("job_state") != "F":
                continue
            decision = {"old_job_id": job_id, "old_name": payload["name"], "status": "not_retried"}
            metrics["retries"].append(decision)
            try:
                intent = ROOT / "logs/pbs" / f"phase_retry_{job_id}.json"
                if job_id in already_retried or intent.exists():
                    raise ValueError("Existing retry payload/intent; uncertain acknowledgements require root review")
                if job_id not in final:
                    raise ValueError("Final accounting is not yet independently logged and verified")
                if Path(payload["repo"]).resolve() != ROOT or payload.get("queue") != "autox":
                    raise ValueError("Submission is not this repository's approved autox entry")
                kind = entrypoint(payload["command"], payload["name"], ROOT)
                outer, outer_evidence = load_logged(ROOT, ROOT / "results" / f"{payload['name']}.json", auditor)
                gate_name = (payload["name"] + "_j" + re.sub(r"[^A-Za-z0-9]", "_", job_id) + "_gate"
                             if kind == "t01" else payload["name"] + "_gate")
                gate, gate_evidence = load_logged(ROOT, ROOT / "results" / f"{gate_name}.json", auditor)
                child = child_record = child_evidence = None
                if kind == "t01" and gate.get("_meta", {}).get("exit_code") == 0:
                    child, child_record, child_evidence = child_black_evidence(ROOT, payload, outer, gate_evidence, auditor)
                if kind == "probe_pack":
                    from src.tools.probe_pack_retry import validate_probe_pack_retry
                    reason, child_evidence = validate_probe_pack_retry(
                        ROOT, payload, job_id, fields, outer, gate, gate_evidence, auditor)
                else:
                    reason = validate_black_retry(payload, job_id, fields, outer, gate, gate_name, kind, child, child_record)
                if int(fields.get("Resource_List.ngpus", "0")) != payload["requested_gpu_count"] or submit.wall_seconds(fields.get("Resource_List.walltime", "")) != payload["requested_wall_seconds"]:
                    raise ValueError("Scheduler resources differ from original immutable request")
                environment = payload.get("environment", {})
                if set(environment) - {"CVPR_SCRATCH", "CVPR_CUDA_MODULE", "CVPR_VULKAN_MODULE"}:
                    raise ValueError("Absolute deadlines or unknown environment need root review")
                new_name = payload["name"] + "_r" + uuid.uuid4().hex[:8]
                command = [new_name if part == payload["name"] else part for part in payload["command"]]
                seconds = payload["requested_wall_seconds"]
                walltime = f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
                submit_name = args.name + "_retry_" + job_id.split(".")[0]
                require_unused_run(ROOT, submit_name)
                argv = [sys.executable, "scripts/run_logged.py", "--name", submit_name, "--", sys.executable,
                        "src/tools/record_command.py", "--name", submit_name, "--", sys.executable,
                        "scripts/pbs/submit_phase.py", "--name", new_name, "--task", payload["task_id"],
                        "--gpus", str(payload["requested_gpu_count"]), "--walltime", walltime,
                        "--scratch", environment["CVPR_SCRATCH"], "--retry-of", job_id, "--", *command]
                decision.update(new_name=new_name, submission_run=submit_name, command=argv,
                                evidence={"outer": outer_evidence, "gate": gate_evidence, "child": child_evidence},
                                retry_reason=reason, status="intent_recorded")
                save_new(intent, {"retry_of": job_id, "watch_name": args.name, **decision})
                env = dict(os.environ)
                for key in ("CVPR_RENDER_DEADLINE_EPOCH", "CVPR_CUDA_MODULE", "CVPR_VULKAN_MODULE"):
                    env.pop(key, None)
                env.update(environment)
                code = subprocess.run(argv, cwd=ROOT, env=env).returncode
                after = submissions(ROOT, submit.BATCH, submit.JOB_ID)
                matches = [row for row in after if row["payload"].get("retry_of") == job_id]
                decision.update(submission_exit_code=code, new_submissions=matches,
                                status="submitted" if code == 0 and len(matches) == 1 and matches[0]["job_id"] else "blocked_or_uncertain_root_review")
                if decision["status"] != "submitted":
                    decision["reason"] = "Submission failed, budget rejected, or acknowledgement uncertain; immutable intent blocks automatic repeats pending root review"
                already_retried.add(job_id)
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                decision["reason"] = f"{type(error).__name__}: {error}"
    except BaseException as error:
        metrics["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if lock is not None:
            lock.close()
        save_new(output, {"metrics": metrics, "config": config})
        print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return 1 if metrics["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
