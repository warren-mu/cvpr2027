#!/usr/bin/env bash
# Run through scripts/run_logged.py inside one Hopper PBS GPU allocation.
set -euo pipefail
t00_repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$t00_repo"
if [[ -f scripts/scratch_env.sh ]]; then source scripts/scratch_env.sh; fi
t00_name=${CVPR_RUN_NAME:-t00_vagen_smoke}
[[ "$t00_name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid CVPR_RUN_NAME' >&2; exit 2; }
t00_results=${CVPR_RESULTS_PATH:-$t00_repo/results/$t00_name.json}
if [[ "$t00_results" != /* ]]; then t00_results=$t00_repo/$t00_results; fi
t00_logs=$t00_repo/logs/runs/$t00_name
mkdir -p "$t00_logs"
t00_train_log=$t00_logs/train.log
t00_resources=$t00_logs/resources.jsonl
t00_manifest=$t00_logs/config.json
t00_server_pid=
t00_server_started=
t00_stage=preflight

t00_finish() {
  local code=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$t00_server_pid" && -n "$t00_server_started" ]]; then
    # Check PID birth time: a dead server's PID must never target another job.
    python3 - "$t00_server_pid" "$t00_server_started" <<'PY'
import os, signal, sys
import psutil
try:
    root = psutil.Process(int(sys.argv[1]))
    if root.create_time() != float(sys.argv[2]):
        raise SystemExit('Server PID changed; refusing to signal it')
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
    wait "$t00_server_pid" 2>/dev/null
  fi
  python3 "$t00_repo/src/tools/parse_train_log.py" \
    --train-log "$t00_train_log" --resource-log "$t00_resources" \
    --manifest "$t00_manifest" --exit-code "$code" --stage "$t00_stage" \
    --output "$t00_results"
  local parse_code=$?
  if [[ "$code" == 0 && "$parse_code" != 0 ]]; then code=$parse_code; fi
  exit "$code"
}
trap t00_finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
: "${PBS_JOBID:?Run through Hopper PBS}"
: "${CUDA_VISIBLE_DEVICES:?PBS must assign one GPU}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo 'T00 requires one allocated GPU' >&2; exit 2; }
t00_vagen=${VAGEN_DIR:-$t00_repo/third_party/VAGEN}
t00_scratch=${T00_SCRATCH_ROOT:-${CVPR_SCRATCH:-/scratch/$USER/cvpr2027}/t00}
t00_work=$t00_scratch/$t00_name-${PBS_JOBID//[^A-Za-z0-9_.-]/_}
mkdir -p "$t00_work"
export WANDB_MODE=offline
export WANDB_DIR=$t00_work/wandb
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
mkdir -p "$WANDB_DIR"
export T00_VAGEN_SOURCE=$t00_vagen T00_WORK_DIR=$t00_work T00_MANIFEST=$t00_manifest T00_RUN_NAME=$t00_name
export T00_TOTAL_STEPS=${T00_TOTAL_STEPS:-21}
export T00_MEMORY_VARIANT=${T00_MEMORY_VARIANT:-none}
t00_stage=prepare_official_launcher
python3 - <<'PY'
import hashlib, importlib.metadata, importlib.util, json, os, pathlib, shutil, socket, subprocess
from src.envs.thor_common import prepare_runtime
memory_variant = os.environ['T00_MEMORY_VARIANT']
if memory_variant not in ('none', 'critic_offload'):
    raise RuntimeError('T00_MEMORY_VARIANT must be none or critic_offload')
source = pathlib.Path(os.environ['T00_VAGEN_SOURCE']).resolve()
work = pathlib.Path(os.environ['T00_WORK_DIR']).resolve()
commit = '4e2976d87c0f92c97f9a820803e71908cde5fb71'
head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
if head != commit:
    raise RuntimeError(f'Expected VAGEN {commit}; found {head}')
verl_spec = importlib.util.find_spec('verl')
if verl_spec is None or verl_spec.origin is None:
    raise RuntimeError('The pinned verl dependency is not installed in the active Python')
verl_source = pathlib.Path(verl_spec.origin).resolve().parent.parent
verl_commit = subprocess.check_output(['git', '-C', str(verl_source), 'rev-parse', 'HEAD'], text=True).strip()
if verl_commit != '3f55021e08b98315213e10d10b709c7f6bbe84f7':
    raise RuntimeError('The active verl package is not the pinned vagen-legacy revision')
relative = pathlib.Path('scripts/examples/vagen_base/navigation/run.sh')
original = (source / relative).read_bytes()
sha = hashlib.sha256(original).hexdigest()
expected = '6465e1ca0ec653dad41471864b59f4d1c2568ebe1c2d121e1fefb20ea1dac3c2'
if sha != expected:
    raise RuntimeError('Official Navigation launcher differs from the inspected pinned source')
steps = int(os.environ['T00_TOTAL_STEPS'])
if not 21 <= steps <= 300:
    raise RuntimeError('T00_TOTAL_STEPS must be 21..300; default 21 yields 20 updates in this legacy loop')
gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True, timeout=30).strip().splitlines()
if len(gpu_lines) != 1:
    raise RuntimeError('Expected exactly one nvidia-smi GPU inside the PBS allocation')
gpu_index, gpu_uuid, gpu_name = [part.strip() for part in gpu_lines[0].split(',', 2)]
visible = os.environ['CUDA_VISIBLE_DEVICES']
if visible.startswith('GPU-') and not gpu_uuid.startswith(visible):
    raise RuntimeError('Allocated GPU UUID differs from nvidia-smi')
runtime = prepare_runtime(os.environ['T00_RUN_NAME'])
if runtime['gpu_uuid'] != gpu_uuid or runtime['gpu_index'] != int(gpu_index):
    raise RuntimeError('GPU enumeration changed during allocation preflight')
device = runtime['controller_gpu_device']
# AI2-THOR 5 accepts a CUDA index and translates it to Vulkan. Its shared
# mapping file is otherwise reused without checking the current node's UUIDs.
# Keep the unmodified official Controller, and never overwrite a stale cache.
from ai2thor.util.lock import LockEx
cache_dir = pathlib.Path.home() / '.ai2thor'
expected_cache_dir = pathlib.Path(runtime['scratch']) / 'ai2thor'
if not cache_dir.is_dir() or cache_dir.resolve() != expected_cache_dir.resolve():
    raise RuntimeError('Expected ~/.ai2thor to point to the setup-created scratch ai2thor directory')
release_dir = cache_dir / 'releases'
release_dirs = sorted(path.name for path in release_dir.iterdir() if path.is_dir()) if release_dir.exists() else []
if len(release_dirs) > 2:
    raise RuntimeError(
        f'AI2-THOR has {len(release_dirs)} cached release directories; training was not started. '
        'Upstream retains only three releases, so this guard reserves one slot for a new build. '
        'Existing releases are preserved; broader cache support requires review.'
    )
release_guard = {
    'path': str(release_dir.resolve()), 'existing_directories': release_dirs,
    'maximum_existing_directories': 2, 'reserved_new_build_slots': 1,
    'upstream_retained_release_count': 3,
}
cache_path = cache_dir / 'cuda-vulkan-mapping.json'
cache_key = str(runtime['gpu_index'])
expected_vulkan = runtime['vulkan_device_index']
cache_created = False
try:
    # Match upstream's .lock filename and lockf convention; do not wait on
    # another process or silently replace its mapping.
    with LockEx(str(cache_path), blocking=False):
        try:
            with cache_path.open('x') as handle:
                json.dump({cache_key: expected_vulkan}, handle)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            cache_created = True
        except FileExistsError:
            pass
        cache_bytes = cache_path.read_bytes()
        try:
            mapping = json.loads(cache_bytes)
        except (ValueError, UnicodeError) as error:
            raise RuntimeError('Existing AI2-THOR GPU mapping is unreadable; preserved without changes') from error
        if not isinstance(mapping, dict) or type(mapping.get(cache_key)) is not int or mapping[cache_key] != expected_vulkan:
            raise RuntimeError(
                f'Existing AI2-THOR GPU mapping does not match allocated CUDA index {cache_key} '
                f'and verified Vulkan index {expected_vulkan}; preserved without changes. '
                'A per-job cache adapter requires review before training.'
            )
except BlockingIOError as error:
    raise RuntimeError('AI2-THOR mapping cache is locked by another process; training was not started') from error
mapping_cache = {
    'path': str(cache_path.resolve()), 'created_exclusively': cache_created,
    'sha256': hashlib.sha256(cache_bytes).hexdigest(),
    'verified_cuda_index': int(cache_key), 'verified_vulkan_index': expected_vulkan,
    'shared_cache': True,
}
text = original.decode()
changes = {
    'trainer.n_gpus_per_node=4': 'trainer.n_gpus_per_node=1',
    'actor_rollout_ref.rollout.tensor_model_parallel_size=4': 'actor_rollout_ref.rollout.tensor_model_parallel_size=1',
    'trainer.total_training_steps=300': f'trainer.total_training_steps={steps}',
}
# Separately named opt-in repair after the observed baseline OOM. These are
# existing official launcher options; keep every other setting unchanged.
memory_changes = {}
if memory_variant == 'critic_offload':
    memory_changes = {
        'critic.model.fsdp_config.param_offload=False': 'critic.model.fsdp_config.param_offload=True',
        'critic.model.fsdp_config.optimizer_offload=False': 'critic.model.fsdp_config.optimizer_offload=True',
    }
changes.update(memory_changes)
for old, new in changes.items():
    if text.count(old) != 1:
        raise RuntimeError(f'Expected exactly one {old}')
    text = text.replace(old, new)
target = work / 'official' / relative
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(text)
env_source = source / relative.parent / 'env_config.yaml'
env_bytes = env_source.read_bytes()
committed_env = subprocess.check_output(['git', '-C', str(source), 'show', f'{commit}:{relative.parent}/env_config.yaml'])
if env_bytes != committed_env:
    raise RuntimeError('Official Navigation env_config.yaml has local edits')
shutil.copyfile(env_source, target.parent / 'env_config.yaml')
# Keep the official port and fail rather than connecting to another job's server.
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 5000))
config = {
    'baseline': 'official_vagen_base_navigation_not_vagen_full',
    'vagen_commit': commit, 'official_launcher': str(relative), 'official_launcher_sha256': sha,
    'verl_commit': verl_commit,
    'installed_versions': {name: importlib.metadata.version(name) for name in
                           ('torch', 'vllm', 'transformers', 'ai2thor', 'numpy', 'psutil')},
    'vllm_cuda_source_sha256': hashlib.sha256(pathlib.Path(
        importlib.metadata.distribution('vllm').locate_file('vllm/platforms/cuda.py')).read_bytes()).hexdigest(),
    'official_env_config_sha256': hashlib.sha256(env_bytes).hexdigest(),
    'official_total_training_steps': 300, 'configured_total_training_steps': steps,
    'required_completed_updates': 20, 'default_resolution': 255,
    'model': 'Qwen/Qwen2.5-VL-3B-Instruct', 'training_config_changes': changes,
    'memory_variant': memory_variant,
    'memory_variant_changes_relative_to_official': memory_changes,
    'server_gpu_topology_change': {'navigation.devices': [device]},
    'server_host': '127.0.0.1', 'server_port': 5000, 'watchdog_seconds': 3600,
    'use_state_reward': False, 'judge_credentials_required': False,
    'wandb_mode': 'offline', 'working_directory': str(work),
    'generated_launcher': str(target), 'pbs_jobid': os.environ['PBS_JOBID'],
    'cuda_visible_devices': visible, 'gpu_uuid': gpu_uuid, 'gpu_name': gpu_name,
    'gpu_device_for_ai2thor': device,
    'gpu_preflight': runtime, 'ai2thor_mapping_cache': mapping_cache,
    'ai2thor_release_prune_guard': release_guard,
    'notes': ['Trainer count, rollout tensor parallelism, and simulator device list are one-GPU topology adaptations.',
              'Legacy loop increments before >= stop: configured 21 corresponds to 20 completed updates.',
              'Batch sizes, model, rewards, validation, checkpoint behavior, and optimizer hyperparameters are unchanged.',
              'The optional critic_offload variant changes only critic parameter and optimizer offload from False to True; none preserves the baseline.',
              'Shared AI2-THOR mapping is verified against this allocation before startup; concurrent jobs sharing or rewriting it are unsupported.',
              'At most two existing release directories are allowed, leaving one build slot below the upstream pruning threshold.',
              'Official shell is executed with -e and pipefail so tee cannot hide a training failure.'],
}
if config['vllm_cuda_source_sha256'] != '1d27d2b5d25dbf8c511f39b7031f2d90c30fd3b5c7f052ea17a7c1138bc57c2b':
    raise RuntimeError('Expected the exact logged vLLM 0.8.2 UUID compatibility repair')
pathlib.Path(os.environ['T00_MANIFEST']).write_text(json.dumps(config, indent=2) + '\n')
print('T00_CONFIG_JSON: ' + json.dumps(config, sort_keys=True), flush=True)
PY
t00_device=$(python3 -c 'import json,os; print(json.load(open(os.environ["T00_MANIFEST"]))["gpu_device_for_ai2thor"])')
t00_copy=$t00_work/official/scripts/examples/vagen_base/navigation/run.sh
t00_stage=start_navigation_server
cd "$t00_work"
setsid python3 -m vagen.server.server server.host=127.0.0.1 server.port=5000 \
  "navigation.devices=[$t00_device]" > "$t00_logs/server.log" 2>&1 &
t00_server_pid=$!
t00_server_started=$(python3 -c 'import psutil,sys; print(psutil.Process(int(sys.argv[1])).create_time())' "$t00_server_pid")
export T00_SERVER_PID=$t00_server_pid
python3 - <<'PY'
import json, os, time, urllib.request
pid = int(os.environ['T00_SERVER_PID'])
for _ in range(90):
    os.kill(pid, 0)
    try:
        with urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=1) as response:
            data = json.load(response)
        if data.get('status') != 'ok' or 'navigation' not in data.get('registered_envs', []):
            raise RuntimeError('Server is up but Navigation was not registered: ' + json.dumps(data))
        print('T00_SERVER_READY: ' + json.dumps(data), flush=True)
        break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(1)
else:
    raise RuntimeError('Navigation server did not become ready within 90 seconds; see server.log')
PY
t00_stage=training
python3 "$t00_repo/src/tools/t00_monitor.py" \
  --train-log "$t00_train_log" --resource-log "$t00_resources" \
  --server-pid "$t00_server_pid" --server-create-time "$t00_server_started" --timeout 3600 -- \
  bash -e -o pipefail "$t00_copy"
t00_stage=completed
