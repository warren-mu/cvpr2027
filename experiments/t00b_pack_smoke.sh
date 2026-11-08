#!/usr/bin/env bash
# Invoke via run_logged --name NAME -- bash experiments/t00b_pack_smoke.sh --name NAME.
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo"
source scripts/scratch_env.sh
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
name= seconds=120 task_timeout=240 worker_timeout=360 gate_timeout=180
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name=$2; shift 2;;
    --seconds) seconds=$2; shift 2;;
    --task-timeout) task_timeout=$2; shift 2;;
    --worker-timeout) worker_timeout=$2; shift 2;;
    --gate-timeout) gate_timeout=$2; shift 2;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ "$name" =~ ^t00b_[A-Za-z0-9_]+$ ]] || { echo 'A unique --name t00b_* is required' >&2; exit 2; }
: "${PBS_JOBID:?PBS allocation required}"
queue_root="$CVPR_SCRATCH/gpu_queue/${PBS_JOBID//[^A-Za-z0-9_.-]/_}/$name"
gate_code=1 worker_code=1
finish() {
  local code=$?
  trap - EXIT INT TERM
  set +e
  python3 -m src.tools.pack_smoke summarize --name "$name" --queue-root "$queue_root" \
    --gate-exit "$gate_code" --worker-exit "$worker_code"
  local summary_code=$?
  if [[ $code == 0 && $summary_code != 0 ]]; then code=$summary_code; fi
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python3 -m src.tools.pack_smoke prepare --name "$name" --queue-root "$queue_root" \
  --seconds "$seconds" --task-timeout "$task_timeout" --worker-timeout "$worker_timeout"
set +e
python3 scripts/run_logged.py --name "${name}_gate" -- \
  python3 -m src.tools.render_gate --name "${name}_gate" --gpus 4 --resolution 255 --timeout-seconds "$gate_timeout"
gate_code=$?
set -e
[[ $gate_code == 0 ]] || exit "$gate_code"
python3 scripts/gpu_queue.py submit --queue-root "$queue_root" --name "${name}_a" --gpus 2 -- \
  python3 -m src.tools.pack_smoke train --name "${name}_a" --seconds "$seconds" --timeout-seconds "$task_timeout" --seed 1701
python3 scripts/gpu_queue.py submit --queue-root "$queue_root" --name "${name}_b" --gpus 2 -- \
  python3 -m src.tools.pack_smoke train --name "${name}_b" --seconds "$seconds" --timeout-seconds "$task_timeout" --seed 1702
set +e
python3 scripts/gpu_queue.py worker --queue-root "$queue_root" --gpus all --poll 0.1 --exit-when-idle 1 \
  --timeout-seconds "$worker_timeout"
worker_code=$?
set -e
exit "$worker_code"
