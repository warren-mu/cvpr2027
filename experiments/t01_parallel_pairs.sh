#!/usr/bin/env bash
# The caller supplies run_logged --name matching the orchestrator --name.
set -euo pipefail
t01_parallel_repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$t01_parallel_repo"
source scripts/scratch_env.sh
: "${PBS_JOBID:?Submit through the authorized PBS allocation}"
: "${CUDA_VISIBLE_DEVICES:?Keep the original PBS GPU mask}"
export PYTHONUNBUFFERED=1
exec python3 -u -m src.tools.t01_parallel_approved "$@"
