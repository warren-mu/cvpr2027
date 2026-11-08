#!/usr/bin/env bash
# Versioned settled-baseline window; same T01v2 8-GPUh and total 60-GPUh guards.
set -euo pipefail
t01v2_stable_repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$t01v2_stable_repo"
source scripts/scratch_env.sh
: "${PBS_JOBID:?A budgeted PBS allocation is required}"
: "${CUDA_VISIBLE_DEVICES:?Preserve the original PBS mask}"
export PYTHONUNBUFFERED=1
exec python3 -u -m src.tools.t01v2_stable_parallel "$@"
