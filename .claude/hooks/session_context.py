#!/usr/bin/env python3
"""SessionStart hook: give every new / resumed / compacted window the same shared state.

Plain-text stdout from a SessionStart hook is added to Claude's context.
Kept well under the 10,000-character cap.
"""
import datetime as dt
import json
import os
import sys

DEADLINES = [  # AoE = UTC-12, so 23:59 AoE == next day 11:59 UTC
    ("论文注册（标题/摘要/作者，之后不能增删作者）", dt.datetime(2026, 11, 11, 11, 59, tzinfo=dt.timezone.utc)),
    ("正文提交", dt.datetime(2026, 11, 17, 11, 59, tzinfo=dt.timezone.utc)),
    ("补充材料", dt.datetime(2026, 11, 24, 11, 59, tzinfo=dt.timezone.utc)),
]
ROLE_NAMES = {"lead": "总控 lead", "pi-lead": "总控 lead", "scout": "文献 scout",
              "experimenter": "实验 experimenter", "writer": "写作 writer",
              "skeptic-reviewer": "审稿 skeptic-reviewer", "reviewer": "审稿 skeptic-reviewer",
              "auditor": "审计 auditor", "analyst": "分析 analyst"}


def head(path, max_lines, max_chars):
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()[:max_lines]
    except OSError:
        return None
    text = "".join(lines)
    return text[:max_chars]


def latest_review(root):
    d = os.path.join(root, "reviews")
    try:
        files = [f for f in os.listdir(d) if f.startswith("ROUND_")]
    except OSError:
        return None
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    return files[:3]


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        data = {}
    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    role = (data.get("agent_type") or os.environ.get("CVPR_ROLE") or "lead").lower()
    now = dt.datetime.now(dt.timezone.utc)

    out = [f"[CVPR 协作框架] 本窗口角色：{ROLE_NAMES.get(role, role)}。严格遵守 CLAUDE.md 的角色边界与防幻觉铁律。"]
    out.append("截止倒计时：")
    for name, t in DEADLINES:
        delta = t - now
        if delta.total_seconds() > 0:
            h = int(delta.total_seconds() // 3600)
            out.append(f"  - {name}：还剩 {h // 24} 天 {h % 24} 小时")
        else:
            out.append(f"  - {name}：已过")

    contract = head(os.path.join(root, "docs", "RESEARCH_CONTRACT.md"), 45, 3000)
    if contract:
        out.append("\n=== 研究合同（docs/RESEARCH_CONTRACT.md 摘要）===\n" + contract)
    status = head(os.path.join(root, "STATUS.md"), 70, 4000)
    if status:
        out.append("\n=== 任务板（STATUS.md 摘要）===\n" + status)
    rv = latest_review(root)
    if rv:
        out.append("\n最新评审文件：" + ", ".join("reviews/" + f for f in rv))
    out.append("作战手册：docs/PLAYBOOK.md（窗口分工、交接协议、每日节奏）；实验登记：experiments/REGISTRY.md。")
    out.append("\n先确认你在 STATUS.md 里属于自己角色的任务，再动手；完成后更新 STATUS.md 并停下汇报。")
    print("\n".join(out)[:9000])


if __name__ == "__main__":
    main()
