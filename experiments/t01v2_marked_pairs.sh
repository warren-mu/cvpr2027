#!/usr/bin/env bash
set -euo pipefail
marked_repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$marked_repo"
source scripts/scratch_env.sh
: "${PBS_JOBID:?A budgeted PBS allocation is required}"
: "${CUDA_VISIBLE_DEVICES:?Preserve the original PBS mask}"
export PYTHONUNBUFFERED=1
exec python3 -u -m src.tools.t01v2_marked_parallel "$@"
