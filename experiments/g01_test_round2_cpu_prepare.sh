#!/bin/bash
# One authorized CPU-only attempt; a queue/hook refusal is evidence, not retried here.
#PBS -P CFP04-SF-125
#PBS -q interactive
#PBS -l select=1:ncpus=1:mem=8gb:ngpus=0
#PBS -l walltime=00:10:00
#PBS -j oe
set -euo pipefail
: "${PBS_JOBID:?Submit through PBS; no login-node inventory scans}"
repo=/home/svu/e1583364/CVPR2027/cvpr2027
cd -- "$repo"
source scripts/scratch_env.sh
export PATH=/cm/shared/apps/pbspro/current/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
prefix=gen_test_round2_prepare_cpu_01
# Refuse to archive or overwrite any previous stage evidence.
for stage in allocation inventory funnel; do
    name="${prefix}_${stage}"
    for path in "results/${name}.json" "logs/runs/${name}" "logs/runs/${name}.log" "logs/runs/${name}.meta.json" "logs/runs/${name}.diff"; do
        if [[ -e "$path" || -L "$path" ]]; then
            printf 'Existing preparation evidence: %s\n' "$path" >&2
            exit 2
        fi
    done
done
# This preflight is itself logged. It does not import a renderer or infer a GPU
# count from CUDA environment variables: PBS must explicitly report ngpus=0.
python3 scripts/run_logged.py --name "${prefix}_allocation" --seed 0 -- \
    python3 -m src.tools.record_command --name "${prefix}_allocation" -- \
    python3 - "$PBS_JOBID" <<'PY'
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
job = sys.argv[1]
if re.fullmatch(r'[0-9]+\.hopper-m-[0-9]+', job) is None:
    raise ValueError('Require the actual Hopper PBS job ID')
observed = subprocess.run(['/cm/shared/apps/pbspro/current/bin/qstat', '-f', job],
                          capture_output=True, text=True, timeout=45)
path = Path('logs/pbs') / ('cpu_allocation_' + job + '.txt')
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('x') as handle:
    handle.write(observed.stdout + observed.stderr)
print(observed.stdout, end='')
print(observed.stderr, end='', file=sys.stderr)
if observed.returncode != 0:
    raise ValueError('Original PBS resource query failed; no inventory stages may run')
fields, key = {}, None
for line in observed.stdout.splitlines():
    match = re.match(r'^    (\S+) = (.*)$', line)
    if match:
        key, value = match.groups(); fields[key] = value
    elif line.startswith('\t') and key:
        fields[key] += line.strip()
if (fields.get('project') != 'CFP04-SF-125' or fields.get('job_state') != 'R'
        or fields.get('queue') != 'interactive' or fields.get('Resource_List.ngpus') != '0'
        or fields.get('Resource_List.ncpus') != '1' or fields.get('Resource_List.mem', '').lower() != '8gb'
        or fields.get('Resource_List.walltime') != '00:10:00'
        or fields.get('Job_Owner', '').split('@')[0] != 'e1583364'):
    raise ValueError('PBS did not grant the exact authorized CPU-only project/allocation; stop immediately')
select = fields.get('Resource_List.select', '').split(':')
chunks = dict(item.split('=', 1) for item in select[1:])
if (select[0] != '1' or chunks.get('ngpus') != '0' or chunks.get('ncpus') != '1'
        or chunks.get('mem', '').lower() != '8gb' or 'host' in chunks):
    raise ValueError('PBS select differs from the unpinned one-CPU, zero-GPU request')
print('CPU_ALLOCATION_CHECK_JSON: ' + json.dumps({'job_id': job, 'project': fields['project'],
    'queue': fields['queue'], 'job_state': fields['job_state'], 'ngpus': fields['Resource_List.ngpus'],
    'ncpus': fields['Resource_List.ncpus'], 'mem': fields['Resource_List.mem'],
    'walltime': fields['Resource_List.walltime'], 'exec_host': fields.get('exec_host'),
    'original_qstat': {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
    'scope': 'PBS allocation authentication only; no rendering or scientific acceptance'}, sort_keys=True), flush=True)
PY
# The already committed script enforces separate SIGALRM bounds:180s/300s.
# A normal inventory failure must not prevent the independent funnel diagnostic.
status=0
if ! python3 scripts/run_logged.py --name "${prefix}_inventory" --seed 0 -- \
    python3 -u experiments/g01_test_round2_prepare.py --name "${prefix}_inventory" --stage inventory; then
    status=1
fi
if ! python3 scripts/run_logged.py --name "${prefix}_funnel" --seed 0 -- \
    python3 -u experiments/g01_test_round2_prepare.py --name "${prefix}_funnel" --stage funnel; then
    status=1
fi
exit "$status"
