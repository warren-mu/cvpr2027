#!/usr/bin/env bash
# 以某个角色启动一个 Claude Code 窗口（保留 Claude Code 默认系统提示，追加角色说明）。
#   bash scripts/claude_as.sh <role> [其他 claude 参数]
#   role: lead | scout | exp-env | exp-train | experimenter | analyst | writer | reviewer | auditor
set -euo pipefail
cd "$(dirname "$0")/.."

ROLE="${1:-lead}"; shift || true
FOCUS=""
case "$ROLE" in
  lead)       FILE=pi-lead;          CVPR_ROLE=lead ;;
  reviewer)   FILE=skeptic-reviewer; CVPR_ROLE=reviewer ;;
  exp-env)    FILE=experimenter;     CVPR_ROLE=experimenter
              FOCUS="【本窗口分工：仿真侧】你只负责 src/envs/（编辑 API、可解性神谕、孪生生成、Twin-Probe 诊断）。不要改 src/train/。" ;;
  exp-train)  FILE=experimenter;     CVPR_ROLE=experimenter
              FOCUS="【本窗口分工：训练侧】你只负责 src/train/ 与 configs/（奖励、阶段1/2 训练、提交 GPU 队列任务）。不要改 src/envs/。" ;;
  scout|experimenter|writer|auditor|analyst) FILE="$ROLE"; CVPR_ROLE="$ROLE" ;;
  *) echo "unknown role: $ROLE" >&2; exit 1 ;;
esac

# 去掉 YAML frontmatter，只取正文作为追加的系统提示
PROMPT="$(awk 'BEGIN{n=0} /^---[[:space:]]*$/ && n<2 {n++; next} n>=2 {print}' ".claude/agents/${FILE}.md")"
[[ -n "$FOCUS" ]] && PROMPT="${PROMPT}

${FOCUS}"

export CVPR_ROLE
printf '\033]0;CVPR:%s\007' "$ROLE"   # 终端标题
exec claude --append-system-prompt "$PROMPT" "$@"
