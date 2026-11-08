#!/usr/bin/env bash
# Optional restore diagnostics, a bounded repair, then separately logged training.
# The enclosing PBS job also uses run_logged + record_command for its exit status.
set -euo pipefail
: "${PBS_JOBID:?Run only inside a Hopper PBS allocation}"
[[ $# == 2 || $# == 3 ]] || { echo 'Usage: t00_setup_then_train.sh SETUP_RUN TRAIN_RUN [RESTORE_PREFIX]' >&2; exit 2; }
t00_setup_name=$1
t00_train_name=$2
t00_repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$t00_repo"
source scripts/scratch_env.sh
t00_names=("$t00_setup_name" "$t00_train_name")
if [[ $# == 3 ]]; then
  t00_restore_prefix=$3
  for t00_mode in no_reset metadata paused; do t00_names+=("${t00_restore_prefix}_${t00_mode}"); done
fi
for t00_name in "${t00_names[@]}"; do
  [[ "$t00_name" =~ ^t00_[A-Za-z0-9_]+$ ]] || { echo 'Invalid run name' >&2; exit 2; }
  for t00_path in "results/$t00_name.json" "logs/runs/$t00_name.log" "logs/runs/$t00_name.meta.json"; do
    [[ ! -e "$t00_path" ]] || { echo "Refusing to reuse existing evidence: $t00_path" >&2; exit 2; }
  done
done
[[ "$t00_setup_name" != "$t00_train_name" ]] || { echo 'Run names must be distinct' >&2; exit 2; }

if [[ $# == 3 ]]; then
  t00_diagnostics_complete=1
  export CVPR_RENDER_DEADLINE_EPOCH=$(( $(date +%s) + 1200 ))
  for t00_mode in no_reset metadata paused; do
    t00_name="${t00_restore_prefix}_${t00_mode}"
    # Exact-zero failure is an observation, not a reason to suppress other tests.
    # A partial/unsafe run is preserved and prevents the later training stage.
    t00_diag_exit=0
    python3 scripts/run_logged.py --name "$t00_name" -- \
      python3 src/envs/thor_restore_diagnostics.py --mode "$t00_mode" \
        --name "$t00_name" --timeout-seconds 300 || t00_diag_exit=$?
    t00_check_exit=0
    python3 - "results/$t00_name.json" "$t00_diag_exit" <<'PY' || t00_check_exit=$?
import json
from pathlib import Path
import sys
p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(2)
try:
    m = json.loads(p.read_text())["metrics"]
    if not isinstance(m, dict):
        raise ValueError("metrics is not an object")
except (OSError, ValueError, KeyError) as error:
    print("Cannot verify simulator cleanup from the diagnostic result:", error)
    raise SystemExit(2)
if m.get("cleanup_errors"):
    print("Cannot continue after unverified simulator cleanup:", m["cleanup_errors"])
    raise SystemExit(2)
if int(sys.argv[2]) or not m.get("diagnostic_completed") or m.get("cases_completed") != 20:
    print("Diagnostic incomplete; preserve evidence and do not start training:", p)
    raise SystemExit(1)
print("Diagnostic completed; exact_zero_passed =", m.get("exact_zero_passed"))
PY
    if [[ "$t00_check_exit" == 2 ]]; then exit 2; fi
    if [[ "$t00_check_exit" != 0 ]]; then t00_diagnostics_complete=0; fi
  done
  [[ "$t00_diagnostics_complete" == 1 ]] || exit 1
  unset CVPR_RENDER_DEADLINE_EPOCH
fi

python3 scripts/run_logged.py --name "$t00_setup_name" -- \
  python3 src/tools/record_command.py --name "$t00_setup_name" -- \
  timeout --signal=TERM --kill-after=30s 1200s bash scripts/setup_env.sh --full

CVPR_RUN_NAME="$t00_train_name" CVPR_RESULTS_PATH="$t00_repo/results/$t00_train_name.json" \
  python3 scripts/run_logged.py --name "$t00_train_name" -- bash experiments/t00_vagen_smoke.sh
