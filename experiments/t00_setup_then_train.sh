#!/usr/bin/env bash
# A bounded dependency repair followed by a separately logged training attempt.
# The enclosing PBS job also uses run_logged + record_command for its exit status.
set -euo pipefail
: "${PBS_JOBID:?Run only inside a Hopper PBS allocation}"
[[ $# == 2 ]] || { echo 'Usage: t00_setup_then_train.sh SETUP_RUN TRAIN_RUN' >&2; exit 2; }
t00_setup_name=$1
t00_train_name=$2
t00_repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$t00_repo"
source scripts/scratch_env.sh
for t00_name in "$t00_setup_name" "$t00_train_name"; do
  [[ "$t00_name" =~ ^t00_[A-Za-z0-9_]+$ ]] || { echo 'Invalid run name' >&2; exit 2; }
  for t00_path in "results/$t00_name.json" "logs/runs/$t00_name.log" "logs/runs/$t00_name.meta.json"; do
    [[ ! -e "$t00_path" ]] || { echo "Refusing to reuse existing evidence: $t00_path" >&2; exit 2; }
  done
done
[[ "$t00_setup_name" != "$t00_train_name" ]] || { echo 'Run names must be distinct' >&2; exit 2; }

python3 scripts/run_logged.py --name "$t00_setup_name" -- \
  python3 src/tools/record_command.py --name "$t00_setup_name" -- \
  timeout --signal=TERM --kill-after=30s 1200s bash scripts/setup_env.sh --full

CVPR_RUN_NAME="$t00_train_name" CVPR_RESULTS_PATH="$t00_repo/results/$t00_train_name.json" \
  python3 scripts/run_logged.py --name "$t00_train_name" -- bash experiments/t00_vagen_smoke.sh
