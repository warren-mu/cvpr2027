#!/usr/bin/env python3
"""File-based GPU job queue. Every job runs through scripts/run_logged.py (full provenance).

Submit (any window):
    python3 scripts/gpu_queue.py submit --name main_nav_ours_s0 --gpus 8 --priority 2 \
        -- bash experiments/train.sh configs/nav_ours.yaml 0
Run a worker (one per node; shared filesystem => several nodes can share one queue):
    python3 scripts/gpu_queue.py worker --gpus 0,1,2,3,4,5,6,7
    python3 scripts/gpu_queue.py worker --gpus all
Inspect / control:
    python3 scripts/gpu_queue.py status
    python3 scripts/gpu_queue.py cancel main_nav_ours_s0
    python3 scripts/gpu_queue.py retry  main_nav_ours_s0

Jobs live in jobs/{queue,running,done,failed}/<name>.json. Claiming a job is an atomic
os.rename, so two workers never take the same job. Higher priority runs first, then FIFO.
"""
import argparse
import datetime as dt
import json
import os
import signal
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = os.path.join(ROOT, "jobs")
STATES = ("queue", "running", "done", "failed")


def jpath(state, name):
    return os.path.join(JOBS, state, f"{name}.json")


def ensure_dirs():
    for s in STATES:
        os.makedirs(os.path.join(JOBS, s), exist_ok=True)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(path, job):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def find(name):
    for s in STATES:
        if os.path.exists(jpath(s, name)):
            return s
    return None


# ------------------------------------------------------------------ commands
def cmd_submit(args, cmd):
    ensure_dirs()
    if not cmd:
        sys.exit("no command after --")
    state = find(args.name)
    if state in ("queue", "running"):
        sys.exit(f"job '{args.name}' is already {state}")
    job = {"name": args.name, "cmd": cmd, "gpus": args.gpus, "priority": args.priority,
           "results": args.results, "submitted": now(), "submitted_by": os.environ.get("CVPR_ROLE", "lead"),
           "note": args.note}
    save(jpath("queue", args.name), job)
    print(f"queued {args.name} (gpus={args.gpus}, priority={args.priority})")


def all_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
        ids = [l.strip() for l in out.splitlines() if l.strip()]
        if ids:
            return ids
    except (OSError, subprocess.TimeoutExpired):
        pass
    sys.exit("could not list GPUs; pass --gpus 0,1,...")


def pick(free_count):
    cands = []
    for fn in os.listdir(os.path.join(JOBS, "queue")):
        if fn.endswith(".json"):
            try:
                job = load(os.path.join(JOBS, "queue", fn))
            except (OSError, json.JSONDecodeError):
                continue
            if job["gpus"] <= free_count:
                cands.append(job)
    cands.sort(key=lambda j: (-j.get("priority", 0), j.get("submitted", "")))
    return cands


def cmd_worker(args):
    ensure_dirs()
    gpus = all_gpus() if args.gpus == "all" else [g.strip() for g in args.gpus.split(",") if g.strip()]
    free = list(gpus)
    procs = {}  # name -> (Popen, gpu list)
    idle_since = time.time()
    host = socket.gethostname()
    print(f"[worker {host}] gpus={gpus}", flush=True)
    stop = {"flag": False}

    def handle(sig, _frm):
        stop["flag"] = True
    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)

    while True:
        # reap finished
        for name, (p, used) in list(procs.items()):
            code = p.poll()
            if code is None:
                continue
            src = jpath("running", name)
            job = load(src) if os.path.exists(src) else {"name": name}
            job.update({"finished": now(), "exit_code": code})
            dst = jpath("done" if code == 0 else "failed", name)
            save(dst, job)
            if os.path.exists(src):
                os.remove(src)
            free.extend(used)
            del procs[name]
            print(f"[worker] {name} -> {'done' if code == 0 else 'FAILED'} (exit {code})", flush=True)

        if stop["flag"]:
            for name, (p, _) in procs.items():
                print(f"[worker] stopping {name}", flush=True)
                os.killpg(p.pid, signal.SIGTERM)
            break

        # launch new
        launched = False
        for job in pick(len(free)):
            src, dst = jpath("queue", job["name"]), jpath("running", job["name"])
            try:
                os.rename(src, dst)  # atomic claim
            except OSError:
                continue
            used, free = free[:job["gpus"]], free[job["gpus"]:]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(used))
            wrapped = [sys.executable, os.path.join(ROOT, "scripts", "run_logged.py"), "--name", job["name"]]
            if job.get("results"):
                wrapped += ["--results", job["results"]]
            wrapped += ["--"] + job["cmd"]
            out = open(os.path.join(ROOT, "logs", "runs", f"{job['name']}.worker.out"), "w")
            p = subprocess.Popen(wrapped, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            job.update({"started": now(), "host": host, "gpu_ids": used, "pid": p.pid})
            save(dst, job)
            procs[job["name"]] = (p, used)
            print(f"[worker] started {job['name']} on GPUs {used} (pid {p.pid})", flush=True)
            launched = True
            break  # re-evaluate free GPUs before the next pick

        if procs or launched:
            idle_since = time.time()
        elif args.exit_when_idle and time.time() - idle_since > args.exit_when_idle:
            print("[worker] queue empty, exiting", flush=True)
            break
        time.sleep(0.2 if launched else args.poll)


def cmd_status(_args):
    ensure_dirs()
    rows = []
    for s in STATES:
        for fn in sorted(os.listdir(os.path.join(JOBS, s))):
            if fn.endswith(".json"):
                try:
                    j = load(os.path.join(JOBS, s, fn))
                except (OSError, json.JSONDecodeError):
                    continue
                rows.append((s, j.get("name"), j.get("gpus"), j.get("priority", 0),
                             j.get("gpu_ids", ""), j.get("started") or j.get("submitted"),
                             j.get("exit_code", "")))
    if not rows:
        print("no jobs")
        return
    print(f"{'state':<8} {'name':<34} {'gpus':>4} {'pri':>3} {'gpu_ids':<16} {'time':<20} exit")
    for r in rows:
        print(f"{r[0]:<8} {str(r[1]):<34} {str(r[2]):>4} {str(r[3]):>3} {','.join(r[4]) if r[4] else '':<16} {str(r[5]):<20} {r[6]}")
    counts = {s: sum(1 for r in rows if r[0] == s) for s in STATES}
    print("\n" + "  ".join(f"{k}={v}" for k, v in counts.items()))


def cmd_cancel(args):
    state = find(args.name)
    if state == "queue":
        os.remove(jpath("queue", args.name))
        print(f"removed {args.name} from queue")
    elif state == "running":
        job = load(jpath("running", args.name))
        try:
            os.killpg(job["pid"], signal.SIGTERM)  # only works on the worker's host
            print(f"sent SIGTERM to {args.name} (pid {job['pid']} on {job.get('host')})")
        except (ProcessLookupError, PermissionError, KeyError) as e:
            print(f"could not signal {args.name}: {e}. Stop it on host {job.get('host')}.")
    else:
        print(f"{args.name}: not queued or running (state={state})")


def cmd_retry(args):
    state = find(args.name)
    if state not in ("failed", "done"):
        sys.exit(f"{args.name}: state={state}, nothing to retry")
    job = load(jpath(state, args.name))
    for k in ("started", "finished", "exit_code", "pid", "gpu_ids", "host"):
        job.pop(k, None)
    job["submitted"] = now()
    save(jpath("queue", args.name), job)
    os.remove(jpath(state, args.name))
    print(f"requeued {args.name} (previous state {state}; old results are archived by run_logged)")


def main():
    argv = sys.argv[1:]
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--name", required=True)
    s.add_argument("--gpus", type=int, default=1)
    s.add_argument("--priority", type=int, default=0)
    s.add_argument("--results", default=None)
    s.add_argument("--note", default="")
    w = sub.add_parser("worker")
    w.add_argument("--gpus", default="all")
    w.add_argument("--poll", type=float, default=10.0)
    w.add_argument("--exit-when-idle", type=float, default=0, help="seconds; 0 = run forever")
    sub.add_parser("status")
    c = sub.add_parser("cancel")
    c.add_argument("name")
    r = sub.add_parser("retry")
    r.add_argument("name")
    args = ap.parse_args(argv)
    os.makedirs(os.path.join(ROOT, "logs", "runs"), exist_ok=True)
    {"submit": lambda a: cmd_submit(a, cmd), "worker": cmd_worker, "status": cmd_status,
     "cancel": cmd_cancel, "retry": cmd_retry}[args.action](args)


if __name__ == "__main__":
    main()
