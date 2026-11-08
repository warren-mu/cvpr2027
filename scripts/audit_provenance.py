#!/usr/bin/env python3
"""Audit the provenance of every results JSON, whichever agent produced it.

    python3 scripts/audit_provenance.py              # all results/*.json (skips results/_archive/)
    python3 scripts/audit_provenance.py results/t00_thor_checks.json
    python3 scripts/audit_provenance.py --json logs/audit.json

Checks per results file (FAIL = must be fixed before review; WARN = reviewer looks closer):
  P1 FAIL  has `_meta` written by scripts/run_logged.py
  P2 FAIL  `_meta.metrics_sha256` matches the current `metrics` (detects edits after the run)
  P3 FAIL  logs/runs/<name>.log exists and its `# exit_code:` line equals `_meta.exit_code`
  P4 FAIL  logs/runs/<name>.meta.json exists and agrees with `_meta`
  P5 FAIL  `_meta.git_commit` exists in this repo; a dirty run has its diff file
  P6 FAIL  the log's last `FINAL_METRICS_JSON:` line equals `metrics` (WARN if the line is missing)
  P7 WARN  exit_code != 0 (failed runs are kept, but must not be cited)
  P8 FAIL  aggregate files (with `sources`): every source still exists and its metrics hash is unchanged
Exit code 1 if any FAIL.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_KEYS = ("name", "command", "start", "git_commit", "metrics_sha256", "exit_code")


def metrics_hash(metrics):
    return hashlib.sha256(json.dumps(metrics, sort_keys=True).encode()).hexdigest()


def close(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b


def git_has_commit(commit):
    try:
        r = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT,
                           capture_output=True, timeout=20)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return None


def audit_file(path):
    rel = os.path.relpath(path, ROOT)
    out = {"file": rel, "fail": [], "warn": []}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        out["fail"].append(f"P1 unreadable JSON: {e}")
        return out

    meta = data.get("_meta")
    if not isinstance(meta, dict) or meta.get("producer") != "scripts/run_logged.py":
        out["fail"].append("P1 no _meta from scripts/run_logged.py (file was not produced by a logged run)")
        return out
    name = meta.get("name") or os.path.splitext(os.path.basename(path))[0]
    metrics = data.get("metrics", {})

    if meta.get("metrics_sha256") != metrics_hash(metrics):
        out["fail"].append("P2 metrics changed after the run (sha256 mismatch)")

    log_path = os.path.join(ROOT, meta.get("log_path") or os.path.join("logs", "runs", f"{name}.log"))
    log_text = None
    if not os.path.exists(log_path):
        out["fail"].append(f"P3 missing log {os.path.relpath(log_path, ROOT)}")
    else:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            log_text = f.read()
        codes = re.findall(r"^# exit_code: (-?\d+)\s*$", log_text, flags=re.M)
        if not codes:
            out["fail"].append("P3 log has no '# exit_code:' line (run incomplete or log edited)")
        elif int(codes[-1]) != meta.get("exit_code"):
            out["fail"].append(f"P3 log exit_code {codes[-1]} != _meta.exit_code {meta.get('exit_code')}")

    side = os.path.join(ROOT, "logs", "runs", f"{name}.meta.json")
    if not os.path.exists(side):
        out["fail"].append(f"P4 missing {os.path.relpath(side, ROOT)}")
    else:
        with open(side, encoding="utf-8") as f:
            side_meta = json.load(f)
        bad = [k for k in META_KEYS if side_meta.get(k) != meta.get(k)]
        if bad:
            out["fail"].append(f"P4 meta.json disagrees with _meta on {bad}")

    commit = meta.get("git_commit")
    if not commit:
        out["fail"].append("P5 run was not made from a git commit")
    else:
        has = git_has_commit(commit)
        if has is False:
            out["fail"].append(f"P5 git commit {commit[:10]} not in this repo (push the code that produced it)")
        elif has is None:
            out["warn"].append("P5 could not run git to check the commit")
    if meta.get("git_dirty"):
        diff = meta.get("git_diff_path")
        if not diff or not os.path.exists(os.path.join(ROOT, diff)):
            out["fail"].append("P5 dirty run without its diff file")
        else:
            out["warn"].append(f"P5 run used uncommitted changes; see {diff}")

    if log_text is not None and "sources" not in data:
        lines = re.findall(r"^FINAL_METRICS_JSON:\s*(.+?)\s*$", log_text, flags=re.M)
        if not lines:
            out["warn"].append("P6 log has no FINAL_METRICS_JSON line; metrics cannot be cross-checked")
        else:
            try:
                printed = json.loads(lines[-1])
            except ValueError:
                out["fail"].append("P6 FINAL_METRICS_JSON line is not valid JSON")
            else:
                if not close(printed, metrics):
                    out["fail"].append("P6 metrics in results file differ from FINAL_METRICS_JSON in the log")

    if meta.get("exit_code") not in (0, None):
        out["warn"].append(f"P7 run failed (exit_code {meta.get('exit_code')}); keep it, do not cite it")

    for src in data.get("sources", []) or []:
        sp = os.path.join(ROOT, src.get("file", ""))
        if not os.path.exists(sp):
            out["fail"].append(f"P8 source missing: {src.get('file')}")
            continue
        with open(sp, encoding="utf-8") as f:
            sdata = json.load(f)
        if metrics_hash(sdata.get("metrics", {})) != src.get("metrics_sha256"):
            out["fail"].append(f"P8 source changed since aggregation: {src.get('file')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--json", default=None, help="also write the report as JSON")
    args = ap.parse_args()
    files = args.files or sorted(p for p in glob.glob(os.path.join(ROOT, "results", "*.json")))
    files = [os.path.abspath(p) for p in files]
    if not files:
        print("no results files to audit")
        return 0
    reports = [audit_file(p) for p in files]
    n_fail = sum(1 for r in reports if r["fail"])
    for r in reports:
        status = "FAIL" if r["fail"] else ("WARN" if r["warn"] else "PASS")
        print(f"{status:4}  {r['file']}")
        for m in r["fail"]:
            print(f"      FAIL {m}")
        for m in r["warn"]:
            print(f"      warn {m}")
    print(f"\n{len(reports)} files, {n_fail} with FAIL")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
