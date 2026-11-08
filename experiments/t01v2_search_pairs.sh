#!/usr/bin/env bash
# Supplements 3-5; future outer startups only, same T01v2 8 / combined 60 GPU-hour budget.
set -euo pipefail
search_repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$search_repo"
source scripts/scratch_env.sh
: "${PBS_JOBID:?A budgeted PBS allocation is required}"
: "${CUDA_VISIBLE_DEVICES:?Preserve the original PBS mask}"
export PYTHONUNBUFFERED=1
exec python3 -u -m src.tools.t01v2_solvable_parallel "$@"
