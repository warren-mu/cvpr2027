#!/usr/bin/env python3
"""TaskCompleted hook (agent-teams mode): a task can't be marked done without evidence.

- [EXP]   tasks: description must contain `evidence: <path>` and that file must exist.
- [WRITE] tasks: scripts/check_paper.py --fast must pass.
Exit 2 + stderr -> completion is refused and the teammate sees why.
"""
import json
import os
import re
import subprocess
import sys

EVIDENCE = re.compile(r"evidence\s*[:：]\s*(\S+)", re.I)


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)
    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    subject = data.get("task_subject", "") or ""
    desc = data.get("task_description", "") or ""
    text = subject + "\n" + desc

    if "[EXP]" in subject.upper():
        paths = EVIDENCE.findall(text)
        if not paths:
            sys.stderr.write("[EXP] 任务完成前必须在任务描述里写 `evidence: results/<exp_id>.json`。\n")
            sys.exit(2)
        missing = [p for p in paths if not os.path.exists(os.path.join(root, p))]
        if missing:
            sys.stderr.write(f"证据文件不存在：{', '.join(missing)}。请用 scripts/run_logged.py 产生结果后再标记完成。\n")
            sys.exit(2)

    if "[WRITE]" in subject.upper():
        checker = os.path.join(root, "scripts", "check_paper.py")
        if os.path.exists(checker):
            proc = subprocess.run([sys.executable, checker, "--fast", "--quiet"], cwd=root,
                                  capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                sys.stderr.write("check_paper.py --fast 未通过，不能标记 [WRITE] 任务完成：\n"
                                 + (proc.stdout + proc.stderr)[-3000:])
                sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
