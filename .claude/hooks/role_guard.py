#!/usr/bin/env python3
"""PreToolUse hook: enforce file ownership per role and block result tampering.

Role is taken from (in order):
  1. hook input `agent_type` (set for subagents, in-process teammates, `claude --agent X`)
  2. env CVPR_ROLE (set by launch_tmux.sh for each window)
  3. "lead"

Decisions are printed as hookSpecificOutput JSON (deny / ask). Exit 0 with no
output means "no opinion" -> normal permission flow.
"""
import json
import os
import re
import sys
import time

ROLE_ALIASES = {
    "pi-lead": "lead", "lead": "lead", "pi": "lead",
    "scout": "scout",
    "experimenter": "experimenter", "exp": "experimenter",
    "writer": "writer",
    "skeptic-reviewer": "reviewer", "reviewer": "reviewer",
    "auditor": "auditor",
    "analyst": "analyst",
}

# Files whose modification always needs a human click ("ask").
PROTECTED = (
    ".claude/", "CLAUDE.md", "docs/RESEARCH_CONTRACT.md",
    "scripts/check_paper.py", "scripts/verify_bib.py",
    "scripts/export_numbers.py", "scripts/run_logged.py",
    "scripts/summarize_results.py", "scripts/gpu_queue.py", "analysis/plotlib.py",
    "docs/METRICS_SPEC.md", "experiments/REGISTRY.md",
)
# Generated-only paths: nobody edits these by hand.
GENERATED = ("results/", "paper/generated/", "logs/", "figures/", "jobs/")

# (path prefix, roles allowed). First match wins. Unlisted paths: allowed.
OWNERSHIP = [
    ("paper/refs.bib", {"lead", "scout"}),
    ("paper/", {"lead", "writer"}),
    ("reviews/", {"lead", "reviewer", "auditor"}),
    ("CLAIMS.md", {"lead", "writer", "auditor"}),
    ("lit/", {"lead", "scout"}),
    ("analysis/", {"lead", "analyst"}),
    ("src/", {"lead", "experimenter"}),
    ("configs/", {"lead", "experimenter"}),
    ("experiments/", {"lead", "experimenter"}),
    ("scripts/", {"lead", "experimenter"}),
]

OWNER_HINT = {
    "paper/refs.bib": "给 scout 开 [LIT] 任务来新增/修正引用",
    "paper/": "给 writer 开 [WRITE] 任务",
    "reviews/": "评审文件只由 skeptic-reviewer / auditor 写",
    "CLAIMS.md": "CLAIMS.md 由 writer 登记、auditor 核验",
    "lit/": "文献笔记由 scout 维护",
    "analysis/": "统计与作图脚本由 analyst 维护",
    "src/": "给 experimenter 开 [EXP] 任务",
    "configs/": "给 experimenter 开 [EXP] 任务",
    "experiments/": "给 experimenter 开 [EXP] 任务",
    "scripts/": "给 experimenter 开 [EXP] 任务",
}

DESTRUCTIVE = re.compile(
    r"\brm\s+-[a-zA-Z]*[rf]|\bgit\s+reset\s+--hard|\bgit\s+clean\s+-[a-zA-Z]*f"
    r"|\bgit\s+push\b[^;&|]*\s(--force|-f)\b|\bgit\s+checkout\s+--\s|\bgit\s+restore\s+\."
)
SHELL_WRITE_TO_GENERATED = re.compile(
    r"(>{1,2}\s*|\btee\s+(-a\s+)?|\bsed\s+-i\S*\s+[^|;&]*?)\S*(results/|paper/generated/|figures/|jobs/)"
)
VERIFIED_CELL = re.compile(r"\|\s*verified\s*\|", re.I)


def project_root(data):
    return os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()


def get_role(data):
    raw = (data.get("agent_type") or os.environ.get("CVPR_ROLE") or "lead").strip().lower()
    return ROLE_ALIASES.get(raw, raw)


def log(root, entry):
    try:
        os.makedirs(os.path.join(root, "logs"), exist_ok=True)
        with open(os.path.join(root, "logs", "guard.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def decide(decision, reason, root, role, target):
    log(root, {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role,
               "target": target, "decision": decision, "reason": reason})
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
        "permissionDecisionReason": reason,
    }}, ensure_ascii=False))
    sys.exit(0)


def rel_path(root, path):
    if not path:
        return None
    full = path if os.path.isabs(path) else os.path.join(root, path)
    rel = os.path.relpath(os.path.realpath(full), os.path.realpath(root))
    if rel.startswith(".."):
        return None  # outside the project: no opinion
    return rel.replace(os.sep, "/")


def count_verified(text):
    return len(VERIFIED_CELL.findall(text or ""))


def check_claims_edit(data, root, role, rel):
    """Only the auditor may add `| verified |` cells to CLAIMS.md."""
    if role == "auditor":
        return
    ti = data.get("tool_input", {}) or {}
    tool = data.get("tool_name", "")
    if tool == "Write":
        try:
            with open(os.path.join(root, rel), encoding="utf-8") as f:
                old = f.read()
        except OSError:
            old = ""
        new = ti.get("content", "")
    else:
        old = ti.get("old_string", "")
        new = ti.get("new_string", "")
        for e in ti.get("edits", []) or []:  # MultiEdit-style payloads
            old += e.get("old_string", "")
            new += e.get("new_string", "")
    if count_verified(new) > count_verified(old):
        decide("deny", f"角色 {role} 不能把主张标为 verified；只有 auditor 能做核验。"
               "请把状态写成 pending，并给 auditor 开 [AUDIT] 任务。", root, role, rel)


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)
    root = project_root(data)
    role = get_role(data)
    tool = data.get("tool_name", "")
    ti = data.get("tool_input", {}) or {}

    if tool == "Bash":
        cmd = ti.get("command", "") or ""
        if SHELL_WRITE_TO_GENERATED.search(cmd) and "run_logged.py" not in cmd:
            decide("deny", "results/ 和 paper/generated/ 只能由 scripts/run_logged.py 与 "
                   "scripts/export_numbers.py 生成，不能用 shell 重定向/sed 写入。", root, role, cmd[:200])
        if DESTRUCTIVE.search(cmd):
            decide("ask", "破坏性命令（删除 / 强推 / 硬重置），需要人类确认。", root, role, cmd[:200])
        sys.exit(0)

    path = ti.get("file_path") or ti.get("notebook_path") or ti.get("path")
    rel = rel_path(root, path)
    if rel is None:
        sys.exit(0)

    for p in GENERATED:
        if rel.startswith(p):
            decide("deny", f"{rel} 是脚本生成的文件，禁止手工编辑。实验结果用 scripts/run_logged.py "
                   "（或 scripts/gpu_queue.py）重新运行产生；论文数字运行 scripts/export_numbers.py；"
                   "图表用 analysis/ 下的脚本经 plotlib.savefig_with_provenance 生成。",
                   root, role, rel)

    for p in PROTECTED:
        if rel == p.rstrip("/") or rel.startswith(p):
            decide("ask", f"{rel} 是研究合同 / 协作规则 / 校验脚本。修改它会改变全队的约束，"
                   "需要人类确认。", root, role, rel)

    if rel == "CLAIMS.md":
        check_claims_edit(data, root, role, rel)

    for prefix, allowed in OWNERSHIP:
        if rel == prefix or rel.startswith(prefix):
            if role not in allowed:
                hint = OWNER_HINT.get(prefix, "")
                decide("deny", f"角色 {role} 不负责 {prefix}（允许：{', '.join(sorted(allowed))}）。"
                       f"{hint}，并在 STATUS.md 里登记。", root, role, rel)
            break
    sys.exit(0)


if __name__ == "__main__":
    main()
