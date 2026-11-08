#!/usr/bin/env python3
"""Run an experiment with full provenance.

    python3 scripts/run_logged.py --name exp01_baseline --results results/exp01_baseline.json \
        -- python src/train.py --cfg configs/base.yaml

What it does
  * tees stdout/stderr to logs/runs/<name>.log
  * records command, git commit (+ diff if dirty), host, GPU, start/end, exit code
    in logs/runs/<name>.meta.json
  * if the command wrote the results JSON, stamps it with a `_meta` block
  * an older results file with the same name is archived to results/_archive/
    (history is never silently overwritten -- negative results stay on disk)
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd):
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def git_info(name):
    commit = sh(["git", "rev-parse", "HEAD"])
    if not commit:
        return {"git_commit": None, "git_dirty": None}
    dirty = bool(sh(["git", "status", "--porcelain", "--untracked-files=no"]))
    info = {"git_commit": commit, "git_dirty": dirty}
    if dirty:
        diff = sh(["git", "diff", "HEAD"])
        path = os.path.join("logs", "runs", f"{name}.diff")
        with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
            f.write(diff)
        info["git_diff_path"] = path
    return info


def gpu_info():
    out = sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    return [l.strip() for l in out.splitlines() if l.strip()] or None


def main():
    if "--" not in sys.argv:
        sys.exit("usage: run_logged.py --name NAME [--results PATH] -- <command ...>")
    split = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="experiment id, e.g. exp03_ablation_noattn")
    ap.add_argument("--results", default=None, help="results JSON the command writes (default results/<name>.json)")
    ap.add_argument("--seed", default=None)
    args = ap.parse_args(sys.argv[1:split])
    cmd = sys.argv[split + 1:]
    if not cmd:
        sys.exit("no command given after --")

    results_rel = args.results or os.path.join("results", f"{args.name}.json")
    results_abs = os.path.join(ROOT, results_rel)
    os.makedirs(os.path.join(ROOT, "logs", "runs"), exist_ok=True)
    os.makedirs(os.path.dirname(results_abs), exist_ok=True)

    if os.path.exists(results_abs):
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        arch = os.path.join(ROOT, "results", "_archive")
        os.makedirs(arch, exist_ok=True)
        shutil.move(results_abs, os.path.join(arch, f"{args.name}.{stamp}.json"))
        print(f"[run_logged] archived previous {results_rel} -> results/_archive/", file=sys.stderr)

    log_rel = os.path.join("logs", "runs", f"{args.name}.log")
    meta = {
        "name": args.name,
        "command": cmd,
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "gpus": gpu_info(),
        "seed": args.seed,
        "start": dt.datetime.now().isoformat(timespec="seconds"),
        "log_path": log_rel,
        "producer": "scripts/run_logged.py",
        "role": os.environ.get("CVPR_ROLE"),
    }
    meta.update(git_info(args.name))
    start_ts = dt.datetime.now().timestamp()

    with open(os.path.join(ROOT, log_rel), "w", encoding="utf-8") as log:
        log.write(f"# run_logged: {' '.join(cmd)}\n# start: {meta['start']}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
        log.write(f"# exit_code: {code}\n")

    meta["end"] = dt.datetime.now().isoformat(timespec="seconds")
    meta["duration_sec"] = round(dt.datetime.now().timestamp() - start_ts, 1)
    meta["exit_code"] = code

    if os.path.exists(results_abs) and os.path.getmtime(results_abs) >= start_ts - 1:
        with open(results_abs, encoding="utf-8") as f:
            res = json.load(f)
        metrics_blob = json.dumps(res.get("metrics", {}), sort_keys=True).encode()
        meta["metrics_sha256"] = hashlib.sha256(metrics_blob).hexdigest()
        res["_meta"] = meta
        with open(results_abs, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"[run_logged] stamped {results_rel} (exit_code={code})", file=sys.stderr)
    else:
        meta["results_missing"] = True
        print(f"[run_logged] WARNING: command did not write {results_rel}", file=sys.stderr)

    with open(os.path.join(ROOT, "logs", "runs", f"{args.name}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    sys.exit(code)


if __name__ == "__main__":
    main()
