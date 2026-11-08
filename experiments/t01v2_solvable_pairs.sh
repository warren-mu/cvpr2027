#!/usr/bin/env bash
# Explicit supplement-4 entry for later submissions; same pilot budget.
set -euo pipefail
solvable_repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$solvable_repo"
source scripts/scratch_env.sh
: "${PBS_JOBID:?A budgeted PBS allocation is required}"
: "${CUDA_VISIBLE_DEVICES:?Preserve the original PBS mask}"
export PYTHONUNBUFFERED=1
exec python3 -u -m src.tools.t01v2_solvable_parallel "$@"
