#!/usr/bin/env bash
# Four independent logged diagnostics within one 30-minute, one-GPU PBS job.
set -euo pipefail
: "${PBS_JOBID:?Run T00b diagnostics inside a PBS allocation}"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo"
source scripts/scratch_env.sh
python="$CVPR_CONDA_PREFIX/bin/python"
test -x "$python"
export CVPR_RENDER_DEADLINE_EPOCH
# Preserve an earlier shared deadline supplied by the allocation wrapper.
CVPR_RENDER_DEADLINE_EPOCH=$("$python" -c 'import os,time; print(min(time.time()+1500, float(os.environ.get("CVPR_RENDER_DEADLINE_EPOCH", "inf"))))')
overall=0
for mode in no_reset_identity no_reset_edit_undo no_reset_revisit cross_process; do
  name="t00b_$mode"
  set +e
  "$python" scripts/run_logged.py --name "$name" --seed 0 -- \
    "$python" src/envs/thor_isolation_diagnostics.py --mode "$mode" --name "$name" --timeout-seconds 300
  code=$?
  set -e
  if (( code != 0 )); then overall=1; fi
  # Incomplete observations do not hide later modes, but uncertain cleanup does.
  "$python" - "$name" <<'PY'
import json, os, pathlib, sys
path = pathlib.Path('results') / (sys.argv[1] + '.json')
if not path.is_file():
    raise SystemExit('Stopping: diagnostic result was not written')
result = json.loads(path.read_text())
if result.get('metrics', {}).get('cleanup_errors'):
    raise SystemExit('Stopping: simulator cleanup could not be verified')
if result.get('config', {}).get('runtime', {}).get('pbs_jobid') != os.environ['PBS_JOBID']:
    raise SystemExit('Stopping: result has no verified runtime for this allocation')
PY
done
exit "$overall"
