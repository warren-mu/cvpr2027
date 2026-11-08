#!/usr/bin/env bash
# Scale entry; 2/4/8 GPU wrappers share only the short Ray scratch-path helper.
set -euo pipefail
t00b_repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$t00b_repo"
source scripts/scratch_env.sh
export PYTHONPATH="$t00b_repo${PYTHONPATH:+:$PYTHONPATH}"
t00b_base= t00b_gpus= t00b_wall= t00b_variant=both
while (( $# )); do
  case "$1" in
    --name) t00b_base=${2:?Missing name}; shift 2 ;;
    --gpus) t00b_gpus=${2:?Missing GPU count}; shift 2 ;;
    --wall-seconds) t00b_wall=${2:?Missing requested walltime seconds}; shift 2 ;;
    --both) t00b_variant=both; shift ;;
    --variant) t00b_variant=${2:?Missing variant}; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$t00b_base" =~ ^[A-Za-z0-9_]+$ && "$t00b_gpus" =~ ^(2|4|8)$ && "$t00b_wall" =~ ^[0-9]+$ ]] || {
  echo 'Usage: t00b_scale_smoke.sh --name UNIQUE --gpus 2|4|8 --wall-seconds SECONDS --both' >&2; exit 2;
}
[[ "$t00b_variant" == both || "$t00b_variant" == baseline || "$t00b_variant" == vllm04 ]] || exit 2
: "${PBS_JOBID:?Run through PBS}"
: "${CUDA_VISIBLE_DEVICES:?Preserve the PBS GPU allocation}"
# These are walltime bounds, never accounting/queue price assumptions.
(( t00b_wall >= 3600 && t00b_wall <= 10800 )) || { echo 'Walltime must be 3600..10800 seconds' >&2; exit 2; }
export T00B_DEADLINE_EPOCH=${T00B_DEADLINE_EPOCH:-$(python3 - "$t00b_wall" <<'PYDEADLINE'
import sys, time
print(time.time() + int(sys.argv[1]) - 120)
PYDEADLINE
)}
t00b_devices=$(python3 - "$t00b_gpus" <<'PYDEVICES'
import json, sys
print(json.dumps(list(range(int(sys.argv[1]))), separators=(',', ':')))
PYDEVICES
)
if [[ "$t00b_variant" == both ]]; then
  # Reject reuse before run_logged can archive or truncate prior evidence.
  for suffix in gate baseline vllm04; do
    child=${t00b_base}_$suffix
    [[ ! -e "results/$child.json" && ! -e "logs/runs/$child.log" && ! -e "logs/runs/$child" ]] || {
      echo "Run name already used: $child" >&2; exit 2;
    }
  done
  env CVPR_RUN_NAME="${t00b_base}_gate" CVPR_RESULTS_PATH="$t00b_repo/results/${t00b_base}_gate.json" \
    python3 scripts/run_logged.py --name "${t00b_base}_gate" -- \
    python3 -m src.tools.render_gate --name "${t00b_base}_gate" --gpus "$t00b_gpus" \
    --resolution 255 --timeout-seconds 180
  env CVPR_RUN_NAME="${t00b_base}_baseline" CVPR_RESULTS_PATH="$t00b_repo/results/${t00b_base}_baseline.json" \
    python3 scripts/run_logged.py --name "${t00b_base}_baseline" -- \
    bash experiments/t00b_scale_smoke.sh --name "$t00b_base" --gpus "$t00b_gpus" \
    --wall-seconds "$t00b_wall" --variant baseline
  env CVPR_RUN_NAME="${t00b_base}_vllm04" CVPR_RESULTS_PATH="$t00b_repo/results/${t00b_base}_vllm04.json" \
    python3 scripts/run_logged.py --name "${t00b_base}_vllm04" -- \
    bash experiments/t00b_scale_smoke.sh --name "$t00b_base" --gpus "$t00b_gpus" \
    --wall-seconds "$t00b_wall" --variant vllm04
  exit 0
fi
t00b_expected=${t00b_base}_baseline
[[ "$t00b_variant" != vllm04 ]] || t00b_expected=${t00b_base}_vllm04
t00b_name=${CVPR_RUN_NAME:-$t00b_expected}
[[ "$t00b_name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid CVPR_RUN_NAME' >&2; exit 2; }
[[ "$t00b_name" == "$t00b_expected" ]] || { echo 'Run name must match its baseline/variant name' >&2; exit 2; }
t00b_results=${CVPR_RESULTS_PATH:-$t00b_repo/results/$t00b_name.json}
[[ "$t00b_results" == /* ]] || t00b_results=$t00b_repo/$t00b_results
t00b_logs=$t00b_repo/logs/runs/$t00b_name
mkdir -p "$t00b_logs"
t00b_pbs_id=${PBS_JOBID:-unallocated}
t00b_work=${CVPR_SCRATCH:?}/t00b_scale/$t00b_name-${t00b_pbs_id//[^A-Za-z0-9_.-]/_}
t00b_server_pid=
t00b_server_birth=
t00b_stage=preflight

t00b_finish() {
  local code=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$t00b_server_pid" && -n "$t00b_server_birth" ]]; then
    python3 - "$t00b_server_pid" "$t00b_server_birth" <<'PY'
import os, signal, sys
import psutil
try:
    root = psutil.Process(int(sys.argv[1]))
    if root.create_time() != float(sys.argv[2]):
        raise SystemExit('Server PID identity changed; refusing cleanup')
    owned = [root, *root.children(recursive=True)]
    if os.getpgid(root.pid) == root.pid:
        os.killpg(root.pid, signal.SIGTERM)
    else:
        root.terminate()
    _, alive = psutil.wait_procs(owned, timeout=5)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
except psutil.NoSuchProcess:
    pass
PY
    wait "$t00b_server_pid" 2>/dev/null
  fi
  python3 "$t00b_repo/src/tools/t00b_archive.py" --raw-dir "$t00b_work" \
    --archive-dir "$t00b_logs/evidence" --index "$t00b_logs/evidence_index.json"
  local archive_code=$?
  if [[ $code == 0 && $archive_code != 0 ]]; then code=$archive_code; fi
  python3 "$t00b_repo/src/tools/parse_t00b_scale_log.py" \
    --train-log "$t00b_work/train.log" --resource-log "$t00b_work/resources.jsonl" \
    --frame-events "$t00b_work/frame_events.jsonl" --manifest "$t00b_logs/config.json" \
    --archive-index "$t00b_logs/evidence_index.json" \
    --exit-code "$code" --stage "$t00b_stage" --output "$t00b_results"
  local parse_code=$?
  if [[ $code == 0 && $parse_code != 0 ]]; then code=$parse_code; fi
  exit "$code"
}
trap t00b_finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
: "${PBS_JOBID:?Run through Hopper PBS}"
: "${CUDA_VISIBLE_DEVICES:?Preserve PBS GPU allocation}"
mkdir -p "$(dirname "$t00b_work")"
mkdir "$t00b_work"
t00b_limit=7800
[[ "$t00b_variant" != vllm04 ]] || t00b_limit=2100
t00b_timeout=$(python3 - "$t00b_limit" "$t00b_variant" <<'PY'
import os, sys, time
remaining = float(os.environ['T00B_DEADLINE_EPOCH']) - time.time() - 180
if sys.argv[2] == 'baseline':
    remaining -= 2100  # Preserve an explicit variant window within this allocation.
if remaining < 300:
    raise SystemExit('Insufficient job budget for another training phase')
print(min(float(sys.argv[1]), remaining))
PY
)
export WANDB_MODE=offline WANDB_DIR="$t00b_work/wandb" PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
t00b_ray_tmp=$(python3 "$t00b_repo/src/tools/ray_scratch.py" --name "$t00b_name" \
  --work-dir "$t00b_work" --record "$t00b_logs/ray_temp.json")
export RAY_TMPDIR="$t00b_ray_tmp"
echo "T00B_RAY_TMPDIR: $RAY_TMPDIR (provenance: $t00b_logs/ray_temp.json)"
export T00B_FRAME_EVENTS="$t00b_work/frame_events.jsonl" T00B_RUNTIME_JSON="$t00b_work/runtime.json"
t00b_port=$(python3 - <<'PYPORT'
import socket
with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    print(probe.getsockname()[1])
PYPORT
)
export T00B_SERVER_URL=http://127.0.0.1:$t00b_port
unset T00B_PHASE_TOKEN
export T00B_PHASE_TOKEN_FILE="$t00b_work/phase_token"
python3 - "$T00B_PHASE_TOKEN_FILE" <<'PY'
import os, secrets, sys
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as stream:
    stream.write(secrets.token_hex(24) + '\n')
PY
mkdir -p "$WANDB_DIR" "$RAY_TMPDIR"
t00b_stage=prepare_overlay
python3 src/tools/t00b_scale_prepare.py --name "$t00b_name" --variant "$t00b_variant" \
  --gpus "$t00b_gpus" --wall-seconds "$t00b_wall" --server-port "$t00b_port" --gate-result "$t00b_repo/results/${t00b_base}_gate.json" \
  --work-dir "$t00b_work" --log-dir "$t00b_logs" --timeout "$t00b_timeout" \
  --baseline-result "$t00b_repo/results/${t00b_base}_baseline.json"
export PYTHONPATH="$t00b_work/overlay:$t00b_repo${PYTHONPATH:+:$PYTHONPATH}"
t00b_stage=start_navigation_server
cd "$t00b_work"
setsid python3 -m vagen.server.server server.host=127.0.0.1 "server.port=$t00b_port" \
  "navigation.devices=$t00b_devices" > "$t00b_work/server.log" 2>&1 &
t00b_server_pid=$!
t00b_server_birth=$(python3 -c 'import psutil,sys; print(psutil.Process(int(sys.argv[1])).create_time())' "$t00b_server_pid")
python3 - "$t00b_server_pid" "$t00b_logs/config.json" <<'PY'
import json, os, sys, time, urllib.request
from t00b_runtime_hooks import phase_token
pid = int(sys.argv[1])
expected = json.load(open(sys.argv[2]))['hook_sha256']
for _ in range(90):
    os.kill(pid, 0)
    try:
        with urllib.request.urlopen(os.environ['T00B_SERVER_URL'] + '/health', timeout=1) as response:
            health = json.load(response)
        request = urllib.request.Request(os.environ['T00B_SERVER_URL'] + '/t00b/status',
            headers={'X-T00b-Token': phase_token()})
        with urllib.request.urlopen(request, timeout=1) as response:
            instrument = json.load(response)
        if 'navigation' not in health.get('registered_envs', []) or instrument.get('hook_sha256') != expected:
            raise RuntimeError('Navigation registration or instrumentation identity mismatch')
        print('T00B_SERVER_READY: ' + json.dumps({'health': health, 'instrumentation': instrument}), flush=True)
        break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(1)
else:
    raise RuntimeError('Instrumented Navigation server did not become ready; see server.log')
PY
t00b_stage=training
t00b_timeout=$(python3 - "$t00b_timeout" "$t00b_variant" <<'PY'
import os, sys, time
remaining = float(os.environ['T00B_DEADLINE_EPOCH']) - time.time() - 180
if sys.argv[2] == 'baseline':
    remaining -= 2100
if remaining < 300:
    raise SystemExit('Insufficient job budget after render preflight')
print(min(float(sys.argv[1]), remaining))
PY
)
python3 "$t00b_repo/src/tools/t00b_scale_monitor.py" \
  --train-log "$t00b_work/train.log" --resource-log "$t00b_work/resources.jsonl" \
  --manifest "$t00b_logs/config.json" --server-pid "$t00b_server_pid" \
  --server-create-time "$t00b_server_birth" --timeout "$t00b_timeout" -- \
  bash -e -o pipefail "$t00b_work/official/scripts/examples/vagen_base/navigation/run.sh"
t00b_stage=completed
