#!/usr/bin/env python3
"""Allocation-scoped GPU queue; each child runs through run_logged.

PBS workers require --queue-root under CVPR_SCRATCH and full GPU UUIDs from the
inherited CUDA_VISIBLE_DEVICES. --gpus all never enumerates outside that mask.
One worker owns a queue root. Completed names cannot be reused: retry requires
--as-name NEW and an explicit new -- command with isolated outputs. State
transitions rename records and retain their event history.
"""
import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "jobs"
STATES = ("queue", "running", "done", "failed", "cancelled")
UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def jpath(state, name):
    if not NAME.fullmatch(name):
        raise ValueError("Unsafe queue name")
    return JOBS / state / f"{name}.json"


def ensure_dirs():
    for state in STATES:
        (JOBS / state).mkdir(parents=True, exist_ok=True)


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, path)


@contextmanager
def lock(name, blocking=True):
    ensure_dirs()
    with (JOBS / name).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def find(name):
    found = [state for state in STATES if jpath(state, name).exists()]
    if len(found) > 1:
        raise RuntimeError(f"Ambiguous preserved records for {name}: {found}")
    return found[0] if found else None


def transition(name, source, target, **fields):
    """Caller holds queue lock. Never overwrite another state's evidence."""
    src, dst = jpath(source, name), jpath(target, name)
    if dst.exists():
        raise FileExistsError(dst)
    job = load(src)
    job.update(fields)
    job.setdefault("history", []).append({"from": source, "to": target, "time": now(), **fields})
    save(src, job)
    os.rename(src, dst)
    return job


def select_gpu_ids(requested, environment):
    """Pure PBS mask policy, used by CPU fixtures without querying any device."""
    inherited = environment.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [part.strip() for part in inherited.split(",")]
    if environment.get("PBS_JOBID"):
        if not tokens or any(not UUID.fullmatch(token) for token in tokens) or len(set(tokens)) != len(tokens):
            raise ValueError("PBS CUDA_VISIBLE_DEVICES must contain unique full GPU UUIDs; numeric/global-index guesses are forbidden")
        selected = tokens if requested == "all" else [part.strip() for part in requested.split(",")]
        if (not selected or any(not UUID.fullmatch(token) for token in selected)
                or len(set(selected)) != len(selected) or not set(selected).issubset(tokens)):
            raise ValueError("Requested GPUs are not a unique UUID subset of the original PBS mask")
        return selected
    if requested == "all":
        raise ValueError("Outside PBS, pass explicit GPU UUIDs or indices")
    selected = [part.strip() for part in requested.split(",")]
    if not selected or len(set(selected)) != len(selected) or any(not (UUID.fullmatch(gpu) or gpu.isdecimal()) for gpu in selected):
        raise ValueError("Invalid GPU list")
    return selected


def verify_devices(gpus):
    observed = subprocess.run(["nvidia-smi", "-i", ",".join(gpus), "--query-gpu=uuid", "--format=csv,noheader"],
                              text=True, capture_output=True, timeout=20, check=True)
    uuids = [line.strip() for line in observed.stdout.splitlines() if line.strip()]
    if len(uuids) != len(gpus) or len(set(uuids)) != len(gpus):
        raise RuntimeError("nvidia-smi did not confirm every selected GPU")
    if all(UUID.fullmatch(gpu) for gpu in gpus) and set(uuids) != set(gpus):
        raise RuntimeError("Selected GPU UUIDs differ from nvidia-smi")


def artifact_paths(name, results=None):
    return [ROOT / (results or f"results/{name}.json"),
            *[ROOT / "logs" / "runs" / f"{name}{suffix}" for suffix in (".log", ".meta.json", ".diff", ".worker.out")]]


def require_unused(name, results=None):
    if find(name) is not None or any(path.exists() for path in artifact_paths(name, results)):
        raise FileExistsError(f"Preserved queue/result/log evidence exists for {name}; use a new name")


def cmd_submit(args, command):
    if not command or args.gpus < 1:
        raise ValueError("A command and positive GPU count are required")
    with lock("queue.lock"):
        require_unused(args.name, args.results)
        job = {"name": args.name, "cmd": command, "gpus": args.gpus, "priority": args.priority,
               "results": args.results, "submitted": now(), "submitted_by": os.environ.get("CVPR_ROLE", "lead"),
               "note": args.note, "history": [{"from": None, "to": "queue", "time": now()}]}
        save(jpath("queue", args.name), job)
    print(f"queued {args.name} (gpus={args.gpus})", flush=True)
    return 0


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": int(fields[19]), "pgid": int(fields[2]), "sid": int(fields[3])}
    except (OSError, ValueError, IndexError):
        return None


def exited_without_reaping(process):
    # Keep the child PID reserved until group cleanup; Popen.poll() would reap it.
    event = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    return event is not None


def stop_processes(records):
    """Shared bounded TERM/KILL; never signal an unverified or reaped group."""
    errors = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for record in records:
            current = identity(record["process"].pid)
            expected = record["identity"]
            if current and expected and current == expected and current["pgid"] == current["sid"] == current["pid"]:
                try:
                    os.killpg(current["pid"], sig)
                except ProcessLookupError:
                    pass
            elif sig == signal.SIGTERM:
                errors.append(f"Uncertain private process identity {record['process'].pid}; refusing group signal")
        if sig == signal.SIGTERM and records:
            time.sleep(0.25 if all(exited_without_reaping(r["process"]) for r in records) else 2)
    deadline = time.monotonic() + 2
    for record in records:
        try:
            record["exit_code"] = record["process"].wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            record["exit_code"] = None
            errors.append(f"Child {record['process'].pid} did not exit within cleanup deadline")
        record["output"].close()
    return errors


def cmd_worker(args):
    gpus = select_gpu_ids(args.gpus, os.environ)
    verify_devices(gpus)
    if (not all(math.isfinite(value) for value in (args.poll, args.exit_when_idle, args.timeout_seconds))
            or args.poll <= 0 or args.exit_when_idle < 0 or args.timeout_seconds < 0):
        raise ValueError("Invalid worker time bounds")
    free, active, failed = list(gpus), {}, False
    started = idle_since = time.monotonic()
    stop = {"signal": None}
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in old_handlers:
        signal.signal(sig, lambda number, _frame: stop.update(signal=number))
    outcome = {"pbs_jobid": os.environ.get("PBS_JOBID"), "host": socket.gethostname(),
               "parent_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "selected_gpu_uuids": gpus,
               "started": now(), "jobs": [], "cleanup_errors": [], "exit_code": None}
    claimed = False
    try:
        with lock("worker.lock", blocking=False):
            if (JOBS / "worker_summary.json").exists() or (JOBS / "worker_claim.json").exists():
                raise FileExistsError("This queue root already has worker evidence; use a new root")
            save(JOBS / "worker_claim.json", {"pid": os.getpid(), "identity": identity(os.getpid()),
                 "host": socket.gethostname(), "pbs_jobid": os.environ.get("PBS_JOBID"), "started": now()})
            claimed = True
            while True:
                finished = [name for name, record in active.items() if exited_without_reaping(record["process"])]
                for name in finished:
                    record = active.pop(name)
                    cleanup = stop_processes([record])
                    outcome["cleanup_errors"].extend(cleanup)
                    code = record["exit_code"]
                    success = code == 0 and not cleanup
                    with lock("queue.lock"):
                        transition(name, "running", "done" if success else "failed", finished=now(),
                                   exit_code=code, cleanup_errors=cleanup)
                    outcome["jobs"].append({"name": name, "exit_code": code, "gpu_ids": record["gpus"]})
                    failed |= not success
                    free.extend(record["gpus"])
                    if cleanup:
                        raise RuntimeError("Uncertain cleanup: stop scheduling further jobs")
                if stop["signal"] is not None:
                    raise InterruptedError(f"Worker received signal {stop['signal']}")
                if args.timeout_seconds and time.monotonic() - started >= args.timeout_seconds:
                    raise TimeoutError("Queue worker deadline reached")
                launched = False
                with lock("queue.lock"):
                    pending = [load(path) for path in sorted((JOBS / "queue").glob("*.json"))]
                    pending.sort(key=lambda job: (-job.get("priority", 0), job["submitted"], job["name"]))
                    for job in pending:
                        if type(job.get("gpus")) is not int or job["gpus"] < 1:
                            raise ValueError("Invalid queued GPU count")
                        if job["gpus"] > len(free):
                            continue
                        if any(path.exists() for path in artifact_paths(job["name"], job.get("results"))):
                            raise FileExistsError(f"Refusing to overwrite artifacts for {job['name']}")
                        used = free[:job["gpus"]]
                        transition(job["name"], "queue", "running", gpu_ids=used, pbs_jobid=os.environ.get("PBS_JOBID"),
                                   started=now(), host=socket.gethostname())
                        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(used),
                                           CVPR_QUEUE_PARENT_MASK=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                                           CVPR_QUEUE_ROOT=str(JOBS), CVPR_QUEUE_JOB=job["name"])
                        wrapped = [sys.executable, str(ROOT / "scripts/run_logged.py"), "--name", job["name"]]
                        if job.get("results"):
                            wrapped += ["--results", job["results"]]
                        wrapped += ["--", *job["cmd"]]
                        output = (ROOT / "logs/runs" / f"{job['name']}.worker.out").open("x")
                        try:
                            process = subprocess.Popen(wrapped, cwd=ROOT, env=environment, stdout=output,
                                                       stderr=subprocess.STDOUT, start_new_session=True)
                        except BaseException as error:
                            output.close()
                            transition(job["name"], "running", "failed", error=f"Launch failed: {type(error).__name__}: {error}")
                            raise
                        active[job["name"]] = {"process": process, "identity": identity(process.pid),
                                                "output": output, "gpus": used}
                        running = load(jpath("running", job["name"]))
                        running.update(pid=process.pid, process_identity=active[job["name"]]["identity"])
                        save(jpath("running", job["name"]), running)
                        free = free[job["gpus"]:]
                        launched = True
                        print(f"[worker] {job['name']} GPUs={used} pid={process.pid}", flush=True)
                        break
                if active or launched:
                    idle_since = time.monotonic()
                elif pending:
                    raise RuntimeError("Pending jobs do not fit this allocation")
                elif args.exit_when_idle and time.monotonic() - idle_since >= args.exit_when_idle:
                    break
                time.sleep(min(args.poll, 0.05) if launched else args.poll)
    except BaseException as error:
        failed = True
        outcome["error"] = f"{type(error).__name__}: {error}"
        print(outcome["error"], file=sys.stderr, flush=True)
    finally:
        if active:
            cleanup = stop_processes(list(active.values()))
            outcome["cleanup_errors"].extend(cleanup)
            with lock("queue.lock"):
                for name, record in active.items():
                    transition(name, "running", "failed", finished=now(), exit_code=record.get("exit_code"),
                               error=outcome.get("error", "Worker stopped"), cleanup_errors=cleanup)
                    outcome["jobs"].append({"name": name, "exit_code": record.get("exit_code"), "gpu_ids": record["gpus"]})
        outcome.update(finished=now(), exit_code=1 if failed else 0)
        # A second worker must not replace the first worker's summary.
        if claimed and not (JOBS / "worker_summary.json").exists():
            save(JOBS / "worker_summary.json", outcome)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return outcome["exit_code"]


def cmd_status(_args):
    ensure_dirs()
    for state in STATES:
        for path in sorted((JOBS / state).glob("*.json")):
            job = load(path)
            print(json.dumps({"state": state, **job}, sort_keys=True))
    return 0


def cmd_cancel(args):
    with lock("queue.lock"):
        state = find(args.name)
        if state != "queue":
            raise ValueError("Only queued work can be cancelled here; running work is owned by its worker, never signal a saved historical PID")
        transition(args.name, "queue", "cancelled", cancelled=now())
    return 0


def cmd_retry(args, command):
    if not command:
        raise ValueError("Retry requires an explicit new command after --; an old command may still write old result paths")
    with lock("queue.lock"):
        state = find(args.name)
        if state not in ("done", "failed", "cancelled"):
            raise ValueError("Retry source must be a terminal record")
        require_unused(args.as_name)
        previous = load(jpath(state, args.name))
        job = {key: previous.get(key) for key in ("gpus", "priority", "note")}
        job.update(name=args.as_name, cmd=command, results=None, submitted=now(), retry_of={"name": args.name, "state": state},
                   history=[{"from": None, "to": "queue", "time": now()}])
        save(jpath("queue", args.as_name), job)
    return 0


def main():
    global JOBS
    argv, command = sys.argv[1:], []
    if "--" in argv:
        split = argv.index("--")
        argv, command = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--name", required=True)
    submit.add_argument("--gpus", type=int, default=1)
    submit.add_argument("--priority", type=int, default=0)
    submit.add_argument("--results")
    submit.add_argument("--note", default="")
    worker = sub.add_parser("worker")
    worker.add_argument("--gpus", default="all")
    worker.add_argument("--poll", type=float, default=1)
    worker.add_argument("--exit-when-idle", type=float, default=0)
    worker.add_argument("--timeout-seconds", type=float, default=0)
    status = sub.add_parser("status")
    cancel = sub.add_parser("cancel")
    cancel.add_argument("name")
    retry = sub.add_parser("retry")
    retry.add_argument("name")
    retry.add_argument("--as-name", required=True)
    for child in (submit, worker, status, cancel, retry):
        child.add_argument("--queue-root", type=Path)
    args = parser.parse_args(argv)
    JOBS = (args.queue_root or JOBS).resolve()
    if os.environ.get("PBS_JOBID"):
        scratch = os.environ.get("CVPR_SCRATCH")
        if not args.queue_root or not scratch or Path(scratch).resolve() not in JOBS.parents:
            parser.error("PBS queue-root must be an explicit private subdirectory of CVPR_SCRATCH")
    (ROOT / "logs/runs").mkdir(parents=True, exist_ok=True)
    try:
        return {"submit": lambda a: cmd_submit(a, command), "worker": cmd_worker, "status": cmd_status,
                "cancel": cmd_cancel, "retry": lambda a: cmd_retry(a, command)}[args.action](args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
