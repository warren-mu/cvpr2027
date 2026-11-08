#!/usr/bin/env bash
# PBS run_logged -> record_command owns the outer result. Child names are distinct.
# A diagnostic failure is retained while an independent CUDA packing check may
# still run, but any black frame, signal, or uncertain cleanup stops this job.
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo"
source scripts/scratch_env.sh
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
name=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name=$2; shift 2;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ "$name" =~ ^t01_[A-Za-z0-9_]+$ ]] || exit 2
: "${PBS_JOBID:?PBS allocation required}"
gate=${name}_gate
probe=${name}_probe
pack=t00b_${name}_pack
for child in "$gate" "$probe" "$pack" "${pack}_gate" "${pack}_a" "${pack}_b"; do
  for path in "results/$child.json" "logs/runs/$child" "logs/runs/$child.log" \
      "logs/runs/$child.meta.json" "logs/runs/$child.diff"; do
    [[ ! -e "$path" && ! -L "$path" ]] || { echo "Preserve existing evidence: $path" >&2; exit 2; }
  done
done
trap 'exit 130' INT
trap 'exit 143' TERM
env CVPR_RUN_NAME="$gate" CVPR_RESULTS_PATH="$repo/results/$gate.json" \
  python3 scripts/run_logged.py --name "$gate" -- python3 -m src.tools.render_gate \
  --name "$gate" --gpus 4 --resolution 224 --segmentation --timeout-seconds 180

set +e
env CVPR_RUN_NAME="$probe" CVPR_RESULTS_PATH="$repo/results/$probe.json" \
  python3 scripts/run_logged.py --name "$probe" -- python3 -m src.tools.t01_collision_probe \
  --name "$probe" --gate-result "results/$gate.json" --timeout-seconds 180
probe_code=$?
set -e
case "$probe_code" in 86|130|143) exit "$probe_code";; esac

# Do not share the allocated cards with the next stage if cleanup or provenance
# of the completed probe is unknown. This is a continuation guard, not acceptance.
python3 - "$probe" "$probe_code" "$gate" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

name, code, gate = sys.argv[1], int(sys.argv[2]), sys.argv[3]
data = json.loads(Path('results', name + '.json').read_text())
m, c, meta = data['metrics'], data['config'], data['_meta']
workers = m.get('workers')
gate_hash = hashlib.sha256(Path('results', gate + '.json').read_bytes()).hexdigest()
if (meta.get('name') != name or meta.get('exit_code') != code or not meta.get('end')
        or m.get('kind') != 't01_collision_probe' or m.get('exit_code') != code
        or c.get('pbs_jobid') != os.environ['PBS_JOBID']
        or c.get('gate_source', {}).get('sha256') != gate_hash
        or m.get('cleanup_errors') != [] or m.get('black_frame_detected') is not False
        or not isinstance(workers, list) or len(workers) != 4
        or any(row.get('cleanup_errors') != [] or row.get('black_frame_detected') is not False for row in workers)):
    raise SystemExit('Probe cleanup/identity is uncertain or a black frame was recorded; pack stage refused')
PY

set +e
env CVPR_RUN_NAME="$pack" CVPR_RESULTS_PATH="$repo/results/$pack.json" \
  python3 scripts/run_logged.py --name "$pack" -- bash experiments/t00b_pack_smoke.sh \
  --name "$pack" --seconds 120 --task-timeout 240 --worker-timeout 360 --gate-timeout 180
pack_code=$?
set -e
case "$pack_code" in 86|130|143) exit "$pack_code";; esac
[[ "$probe_code" == 0 ]] || exit "$probe_code"
exit "$pack_code"
