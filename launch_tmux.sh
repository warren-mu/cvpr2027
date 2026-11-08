#!/usr/bin/env bash
# 模式 A（推荐）：多窗口协作。每个 tmux 窗口一个角色，共享 STATUS.md / CLAIMS.md / reviews/。
#   bash launch_tmux.sh                 # 核心 5 窗：lead exp-env exp-train writer analyst
#   bash launch_tmux.sh full            # 全 8 窗：再加 scout reviewer auditor
#   GPUS=0,1,2,3,4,5,6,7 bash launch_tmux.sh full   # 同时启动本机 GPU 队列 worker
# 另有两个监控窗：gate（论文门禁）和 queue（GPU 队列）。
# 切换窗口：Ctrl-b 然后数字键 / Ctrl-b w 列表；退出整个会话：tmux kill-session -t cvpr
set -euo pipefail
cd "$(dirname "$0")"
SESSION=cvpr

command -v tmux >/dev/null || { echo "需要 tmux"; exit 1; }
command -v claude >/dev/null || { echo "需要 Claude Code（claude 命令）"; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "会话 $SESSION 已存在：tmux attach -t $SESSION"; exit 0; }

if [[ "${1:-}" == "full" ]]; then
  ROLES=(lead scout exp-env exp-train analyst writer reviewer auditor)
else
  ROLES=(lead exp-env exp-train writer analyst)
fi

first=1
for r in "${ROLES[@]}"; do
  cmd="bash scripts/claude_as.sh $r; exec bash"
  if [[ $first -eq 1 ]]; then
    tmux new-session -d -s "$SESSION" -n "$r" -c "$PWD" "$cmd"; first=0
  else
    tmux new-window -t "$SESSION" -n "$r" -c "$PWD" "$cmd"
  fi
done

tmux new-window -t "$SESSION" -n gate -c "$PWD" \
  "watch -n 120 'python3 scripts/check_paper.py --fast 2>&1 | tail -n 25; echo; tail -n 8 logs/guard.jsonl 2>/dev/null'"

if [[ -n "${GPUS:-}" ]]; then
  tmux new-window -t "$SESSION" -n queue -c "$PWD" "python3 scripts/gpu_queue.py worker --gpus $GPUS; exec bash"
  tmux split-window -t "$SESSION:queue" -v -c "$PWD" "watch -n 30 python3 scripts/gpu_queue.py status"
else
  tmux new-window -t "$SESSION" -n queue -c "$PWD" "watch -n 30 python3 scripts/gpu_queue.py status"
fi

tmux select-window -t "$SESSION:lead"
tmux attach -t "$SESSION"
