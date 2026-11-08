#!/usr/bin/env bash
# 模式 B（实验性）：Claude Code 原生 agent teams。一个 lead 会话自动生成 teammates、
# 共享任务列表、互相发消息；TaskCompleted hook 会拦截没有证据的"完成"。
#   bash launch_team.sh          # in-process 模式（任何终端可用，Shift/方向键切换 teammate）
#   bash launch_team.sh panes    # 每个 teammate 一个 tmux 分屏（需要 tmux 或 iTerm2）
# 注意：agent teams 是实验功能，/resume 不会恢复 in-process teammates，token 消耗显著更高。
set -euo pipefail
cd "$(dirname "$0")"
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1

if [[ "${1:-}" == "panes" ]]; then
  if [[ -z "${TMUX:-}" ]]; then
    exec tmux new-session -s cvpr-team -c "$PWD" "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 claude --teammate-mode tmux"
  fi
  exec claude --teammate-mode tmux
fi
exec claude
