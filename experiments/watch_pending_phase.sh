#!/usr/bin/env bash
# Foreground, bounded monitoring for the active execution session. No daemon,
# cancellation, queue movement, or new work selection. The reviewed phase_watch
# alone may retry a fully evidenced black-frame failure after its budget check.
# Invoke through run_logged -> record_command; each observation is also logged.
set -euo pipefail
set -o noclobber
[[ $# == 2 ]] || { echo 'usage: watch_pending_phase.sh UNIQUE_PREFIX BASELINE_RESULT'; exit 2; }
watch_prefix=$1
watch_baseline=$2
[[ "$watch_prefix" =~ ^t00b_[A-Za-z0-9_]+$ ]] || exit 2
[[ "$watch_baseline" =~ ^results/[A-Za-z0-9_]+\.json$ && -f "$watch_baseline" ]] || exit 2
: "${CVPR_SCRATCH:?Source scripts/scratch_env.sh before monitoring}"
mkdir -p "$CVPR_SCRATCH/monitor_logs"
# Reserve this observation namespace once, including concurrent invocations.
# The marker is deliberately retained on failure; choose a fresh prefix.
watch_console_dir="$CVPR_SCRATCH/monitor_logs/$watch_prefix"
mkdir "$watch_console_dir"
for watch_iteration in 1 2 3 4 5 6 7 8 9 10 11 12; do
  watch_run="${watch_prefix}_p${watch_iteration}"
  for watch_path in "results/$watch_run.json" "logs/runs/$watch_run" \
      "logs/runs/$watch_run.log" "logs/runs/$watch_run.meta.json" \
      "logs/runs/$watch_run.diff" "$watch_console_dir/$watch_run.console.txt"; do
    [[ ! -e "$watch_path" && ! -L "$watch_path" ]] || {
      echo "MONITOR_REFUSES_EXISTING_EVIDENCE $watch_path"; exit 2;
    }
  done
done

for watch_iteration in 1 2 3 4 5 6 7 8 9 10 11 12; do
  # Three bounded sleeps keep query traffic low; the caller remains attached.
  for watch_pause in 1 2 3; do sleep 60; done
  watch_run="${watch_prefix}_p${watch_iteration}"
  watch_exit=0
  python3 scripts/run_logged.py --name "$watch_run" -- python3 -m src.tools.phase_watch \
    --name "$watch_run" --once > "$watch_console_dir/$watch_run.console.txt" 2>&1 || watch_exit=$?
  if [[ "$watch_exit" != 0 ]]; then
    echo "MONITOR_NEEDS_REVIEW $watch_run exit=$watch_exit"
    exit "$watch_exit"
  fi
  watch_comparison=0
  python3 - "$watch_baseline" "results/$watch_run.json" <<'PY' || watch_comparison=$?
import json
import sys

def read(path):
    with open(path) as stream:
        data = json.load(stream)
    if data.get('_meta', {}).get('exit_code') != 0 or data['metrics']['errors']:
        raise ValueError('Only successful logged observations may control this wait')
    if any(row.get('exit_code') != 0 for row in data['metrics']['accounting_runs']):
        raise ValueError('An accounting child failed; return control for review')
    return data

baseline, current = map(read, sys.argv[1:])
def states(data):
    return {r['job_id']: {key: r['fields'].get(key) for key in
            ('job_state', 'Exit_status', 'exec_host', 'queue')}
            for r in data['metrics']['observations']}
changed = states(baseline) != states(current)
action = any(r['status'] != 'not_retried' for r in current['metrics']['retries'])
print(json.dumps({'observation': sys.argv[2],
                  'observed_at_utc': current['config']['observed_at_utc'],
                  'state_changed': changed, 'retry_action': action,
                  'states': states(current) if changed or action else 'unchanged'}, sort_keys=True), flush=True)
raise SystemExit(10 if changed or action else 0)
PY
  if [[ "$watch_comparison" == 10 ]]; then
    echo 'MONITOR_RETURNED_FOR_STATE_CHANGE; task completion is not implied'
    exit 0
  elif [[ "$watch_comparison" != 0 ]]; then
    exit "$watch_comparison"
  fi
done
echo 'MONITOR_OBSERVATION_BOUND_REACHED; jobs remain untouched and task completion is not implied'
