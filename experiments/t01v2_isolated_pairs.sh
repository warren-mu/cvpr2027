#!/usr/bin/env bash
# Explicit baseline-isolated generation; unchanged allocation and 8/60 budget.
set -euo pipefail
isolated_repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$isolated_repo"
source scripts/scratch_env.sh
: "${PBS_JOBID:?A budgeted PBS allocation is required}"
: "${CUDA_VISIBLE_DEVICES:?Preserve the original PBS mask}"
export PYTHONUNBUFFERED=1
exec python3 -u -m src.tools.t01v2_isolated_parallel "$@"
