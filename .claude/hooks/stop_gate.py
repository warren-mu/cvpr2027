#!/usr/bin/env python3
"""Stop hook: before the writer / lead window ends a turn, run the fast paper gate
if any paper source changed since the last clean gate.

Exit 2 + stderr  -> Claude keeps working and sees the problems.
stop_hook_active -> we already blocked once this turn; let it stop (no infinite loop),
                    but leave the failure in logs/gate_failures.log for the auditor.
"""
import json
import os
import subprocess
import sys
import time

GATED_ROLES = {"lead", "pi-lead", "writer"}


def newest_mtime(root):
    newest = 0.0
    paper = os.path.join(root, "paper")
    for dirpath, _, files in os.walk(paper):
        for f in files:
            if f.endswith((".tex", ".bib")):
                newest = max(newest, os.path.getmtime(os.path.join(dirpath, f)))
    claims = os.path.join(root, "CLAIMS.md")
    if os.path.exists(claims):
        newest = max(newest, os.path.getmtime(claims))
    return newest


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)
    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    role = (data.get("agent_type") or os.environ.get("CVPR_ROLE") or "lead").lower()
    if role not in GATED_ROLES:
        sys.exit(0)
    checker = os.path.join(root, "scripts", "check_paper.py")
    if not os.path.exists(checker) or not os.path.isdir(os.path.join(root, "paper")):
        sys.exit(0)

    stamp = os.path.join(root, "logs", ".gate_ok_stamp")
    last_ok = os.path.getmtime(stamp) if os.path.exists(stamp) else 0.0
    if newest_mtime(root) <= last_ok:
        sys.exit(0)  # nothing changed since the last clean gate

    try:
        proc = subprocess.run([sys.executable, checker, "--fast", "--quiet"], cwd=root,
                              capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        sys.exit(0)

    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    if proc.returncode == 0:
        with open(stamp, "w") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))
        sys.exit(0)

    report = (proc.stdout + proc.stderr).strip()[-3500:]
    with open(os.path.join(root, "logs", "gate_failures.log"), "a", encoding="utf-8") as f:
        f.write(f"\n--- {time.strftime('%Y-%m-%dT%H:%M:%S')} role={role} "
                f"stop_hook_active={data.get('stop_hook_active')} ---\n{report}\n")

    if data.get("stop_hook_active"):
        sys.exit(0)  # already blocked once; don't loop. Failure stays logged.

    sys.stderr.write(
        "论文快速检查未通过（scripts/check_paper.py --fast）。请修复下列 ERROR；"
        "如果某项需要别的角色处理（例如缺实验数字、缺引用校验），在 STATUS.md 开任务并说明，然后再结束。\n\n"
        + report + "\n")
    sys.exit(2)


if __name__ == "__main__":
    main()
