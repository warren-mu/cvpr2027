#!/usr/bin/env bash
# Invoke through the PBS template, itself wrapped by run_logged as t00_render_job.
set -euo pipefail
: "${PBS_JOBID:?Rendering requires a PBS allocation}"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo"
source scripts/scratch_env.sh
python="$CVPR_CONDA_PREFIX/bin/python"
test -x "$python"
export CVPR_RENDER_DEADLINE_EPOCH
CVPR_RENDER_DEADLINE_EPOCH=$("$python" -c 'import time; print(time.time()+5100)')
mkdir -p logs
set +e
nvidia-smi > logs/nvidia_smi.txt 2>&1
echo "nvidia-smi exit=$?"
vulkaninfo --summary > logs/vulkaninfo.txt 2>&1
echo "vulkaninfo exit=$?"
"$python" -m pip freeze > logs/env_freeze.txt 2>&1
echo "pip freeze exit=$?"
for url in https://github.com https://pypi.org/simple/ https://huggingface.co; do
  curl -L -s -o /dev/null --max-time 10 -w '%{http_code} %{url_effective}\n' "$url"
done > logs/compute_network.txt 2>&1
cat logs/compute_network.txt
"$python" scripts/run_logged.py --name t00_thor_checks -- "$python" src/envs/thor_checks.py
checks_code=$?
"$python" - <<'PY'
import json
from pathlib import Path
p = Path('results/t00_thor_checks.json')
if not p.exists() or json.loads(p.read_text()).get('metrics', {}).get('cleanup_errors'):
    raise SystemExit('Throughput not started: checks cleanup could not be verified')
PY
if (( $? != 0 )); then exit 1; fi
"$python" scripts/run_logged.py --name t00_thor_throughput -- "$python" src/envs/thor_throughput.py
throughput_code=$?
printf 'THOR_CHECKS_EXIT=%s\nTHOR_THROUGHPUT_EXIT=%s\n' "$checks_code" "$throughput_code"
if (( checks_code != 0 || throughput_code != 0 )); then exit 1; fi
