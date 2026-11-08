#!/usr/bin/env bash
# 只读的服务器体检：不安装、不修改任何东西。报告写到 logs/preflight_<时间>.txt
#   bash scripts/server_preflight.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
OUT="logs/preflight_$(date +%Y%m%d-%H%M%S).txt"

section() { printf '\n===== %s =====\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

{
section "主机"
hostname; uname -srm; (cat /etc/os-release 2>/dev/null | grep -E '^(PRETTY_NAME)=') || true
echo "CPU 核数: $(nproc 2>/dev/null)"; free -h 2>/dev/null | head -2

section "GPU 总览"
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  nvidia-smi | grep -E "Driver Version|CUDA Version" | head -1
  section "GPU 上正在运行的进程（谁在用）"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null | while IFS=, read -r uuid pid mem; do
    pid="$(echo "$pid" | xargs)"; [[ -z "$pid" ]] && continue
    idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$(echo "$uuid" | xargs)" | cut -d, -f1)
    user=$(ps -o user= -p "$pid" 2>/dev/null || echo "?"); cmd=$(ps -o comm= -p "$pid" 2>/dev/null || echo "?")
    echo "GPU $idx  pid=$pid  user=$user  cmd=$cmd  mem=$(echo "$mem" | xargs)"
  done
else
  echo "未找到 nvidia-smi"
fi

section "磁盘"
df -h /mnt/data2 2>/dev/null || true
df -h "$HOME" 2>/dev/null | tail -1
echo "项目目录占用: $(du -sh . 2>/dev/null | cut -f1)"

section "软件"
for t in conda python3 pip git tmux unzip curl node npm gcc nvcc claude codex; do
  if have "$t"; then printf '%-8s %s\n' "$t" "$("$t" --version 2>&1 | head -1)"; else printf '%-8s (未安装)\n' "$t"; fi
done
have conda && conda env list 2>/dev/null | head -20

section "AI2-THOR 渲染前提（无显示器服务器需 Vulkan / CloudRendering）"
echo "DISPLAY=${DISPLAY:-<空>}"
ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null || echo "未找到 Vulkan ICD 目录"
if have vulkaninfo; then vulkaninfo --summary 2>/dev/null | grep -E "deviceName|apiVersion" | head -4; else echo "vulkaninfo 未安装（不一定是问题，安装 AI2-THOR 后用 CloudRendering 实测）"; fi
have Xvfb && echo "Xvfb 可用" || echo "Xvfb 未安装"

section "外网连通性（安装依赖需要）"
for u in https://pypi.org https://github.com https://huggingface.co https://claude.ai; do
  code=$(curl -s -o /dev/null -m 6 -w '%{http_code}' "$u" 2>/dev/null) || true
  [[ -z "$code" || "$code" == "000" ]] && code="不可达"
  echo "$u -> $code"
done

section "限制"
echo "ulimit -n: $(ulimit -n)   ulimit -u: $(ulimit -u)"
} 2>&1 | tee "$OUT"

echo
echo "报告已保存：$OUT"
