#!/usr/bin/env bash
# Invoke this through scripts/run_logged.py inside the assigned PBS GPU.
set -euo pipefail
t01_repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$t01_repo"
source scripts/scratch_env.sh
: "${PBS_JOBID:?T01 simulation requires a PBS allocation}"
: "${CUDA_VISIBLE_DEVICES:?Use only the scheduler-assigned GPU}"
export PYTHONUNBUFFERED=1
# Runtime validates UUID and Vulkan identity. Never rewrite the scheduler mask.
exec python3 -u -m src.envs.pair_generator "$@"
