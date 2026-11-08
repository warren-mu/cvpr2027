#!/usr/bin/env bash
# Explicit second round only; same T01v2 8-GPUh and combined 60-GPUh guards.
set -euo pipefail
t01v2_clone_repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$t01v2_clone_repo"
source scripts/scratch_env.sh
: "${PBS_JOBID:?A budgeted PBS allocation is required}"
: "${CUDA_VISIBLE_DEVICES:?Preserve the original PBS mask}"
export PYTHONUNBUFFERED=1
exec python3 -u -m src.tools.t01v2_clone_parallel "$@"
