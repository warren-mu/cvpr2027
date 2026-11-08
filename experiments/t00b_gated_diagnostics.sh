#!/usr/bin/env bash
# New gate contract; existing queued diagnostic scripts remain untouched.
set -euo pipefail
[[ $# == 2 && $1 == --name && $2 =~ ^t00b_[A-Za-z0-9_]+$ ]] || {
  echo 'Usage: t00b_gated_diagnostics.sh --name UNIQUE_PREFIX' >&2; exit 2;
}
prefix=$2
: "${PBS_JOBID:?Run through a PBS allocation}"
: "${CUDA_VISIBLE_DEVICES:?Preserve the PBS GPU allocation}"
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo"
source scripts/scratch_env.sh
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
python3 - "$prefix" <<'PY'
from pathlib import Path
import sys
for suffix in ('gate', 'no_reset_identity', 'no_reset_edit_undo', 'no_reset_revisit', 'cross_process'):
    name = sys.argv[1] + '_' + suffix
    paths = [Path('results') / (name + '.json'), Path('logs/runs') / name]
    paths += [Path('logs/runs') / (name + ending) for ending in ('.log', '.meta.json', '.diff')]
    if any(path.exists() for path in paths):
        raise SystemExit('Prior child evidence exists; use a new prefix: ' + name)
PY
env CVPR_RUN_NAME="${prefix}_gate" CVPR_RESULTS_PATH="$repo/results/${prefix}_gate.json" \
  python3 scripts/run_logged.py --name "${prefix}_gate" -- \
  python3 -m src.tools.render_gate --name "${prefix}_gate" --gpus 1 \
  --resolution 224 --segmentation --timeout-seconds 180
exec bash experiments/t00b_diagnostics.sh "$prefix"
