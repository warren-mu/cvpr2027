#!/usr/bin/env python3
"""PostToolUse hook (Bash, async): append every shell command to logs/commands.jsonl.

This is the provenance trail the auditor uses to check how results were produced.
"""
import json
import os
import sys
import time


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return
    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    cmd = (data.get("tool_input") or {}).get("command", "")
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "role": data.get("agent_type") or os.environ.get("CVPR_ROLE") or "lead",
        "session": data.get("session_id", "")[:8],
        "cwd": data.get("cwd", ""),
        "command": cmd[:2000],
    }
    try:
        os.makedirs(os.path.join(root, "logs"), exist_ok=True)
        with open(os.path.join(root, "logs", "commands.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


if __name__ == "__main__":
    main()
