#!/usr/bin/env bash
# Run only in an allocated PBS job. --render is an intentional partial environment;
# --full follows the pinned legacy README and its install.sh, then Navigation deps.
set -euo pipefail
stage="${1:---full}"
case "$stage" in --bootstrap|--render|--full) ;; *) echo 'Usage: setup_env.sh [--bootstrap|--render|--full]' >&2; exit 2 ;; esac
[[ $# -le 1 ]] || { echo 'Unexpected arguments' >&2; exit 2; }
: "${PBS_JOBID:?Environment installation must run in an allocated CPU or GPU PBS job, never on a login node}"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$repo/scripts/scratch_env.sh"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'This pinned bootstrap targets Hopper Linux x86_64' >&2; exit 2; }

MINIFORGE_VERSION=24.11.3-2
VAGEN_COMMIT=4e2976d87c0f92c97f9a820803e71908cde5fb71
VERL_COMMIT=3f55021e08b98315213e10d10b709c7f6bbe84f7
XGRAMMAR_COMMIT=989222175c2a30fb7987d8bcce35bec1bf6817f2
vagen_source="$CVPR_SCRATCH/sources/VAGEN-$VAGEN_COMMIT"
verl_source="$CVPR_SCRATCH/sources/verl-$VERL_COMMIT"
xgrammar_source="$CVPR_SCRATCH/sources/xgrammar-$XGRAMMAR_COMMIT"
export MAX_JOBS=2  # Keep CPU-only builds within an ordinary small allocation.
export TORCH_CUDA_ARCH_LIST=9.0  # H200; permits explicit architecture during CPU-only builds.

# No existing directory, old environment, runtime, or symlink is ever removed.
safe_link() {
  local target="$1" link="$2"
  mkdir -p "$target" "$(dirname -- "$link")"
  if [[ -L "$link" ]]; then
    [[ "$(readlink -f -- "$link")" == "$(readlink -f -- "$target")" ]] || {
      echo "Refusing to replace existing symlink: $link" >&2; return 1;
    }
  elif [[ -e "$link" ]]; then
    echo "Refusing to overwrite $link; ask the user before moving or deleting existing data." >&2
    return 1
  else
    ln -s -- "$target" "$link"
  fi
}
safe_link "$CVPR_SCRATCH/ai2thor" "$HOME/.ai2thor"
# Conda records environment registrations here even with CONDA_ENVS_PATH set.
safe_link "$CVPR_SCRATCH/conda/user-data" "$HOME/.conda"
if [[ ! -e "$CONDARC" ]]; then
  (set -o noclobber; printf 'channels:\n  - conda-forge\nchannel_priority: strict\n' > "$CONDARC")
fi

download_once() {
  local url="$1" target="$2"
  if [[ ! -s "$target" ]]; then
    curl --fail --location --retry 3 --output "$target.part" "$url"
    mv -- "$target.part" "$target"
  fi
}
if [[ ! -x "$CVPR_MINIFORGE/bin/conda" ]]; then
  [[ ! -e "$CVPR_MINIFORGE" ]] || { echo "Incomplete existing Miniforge: $CVPR_MINIFORGE; refusing to overwrite" >&2; exit 1; }
  installer="Miniforge3-$MINIFORGE_VERSION-Linux-x86_64.sh"
  installer_dir="$CVPR_SCRATCH/installers"
  mkdir -p "$installer_dir"
  release="https://github.com/conda-forge/miniforge/releases/download/$MINIFORGE_VERSION"
  download_once "$release/$installer" "$installer_dir/$installer"
  download_once "$release/$installer.sha256" "$installer_dir/$installer.sha256"
  (cd "$installer_dir" && sha256sum --check "$installer.sha256")
  # The constructor's bootstrap uses --no-rc. libmamba rejects an inherited
  # CONDARC/MAMBARC together with --no-rc, even when the file itself is valid.
  # Clear those only for the installer; keep scratch cache/env paths and TMPDIR.
  # Also avoid inheriting an unrelated active conda/mamba environment.
  env -u CONDARC -u MAMBARC -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
    -u CONDA_SHLVL -u CONDA_PROMPT_MODIFIER -u CONDA_EXE -u CONDA_PYTHON_EXE \
    -u MAMBA_ROOT_PREFIX -u MAMBA_EXE \
    bash "$installer_dir/$installer" -b -p "$CVPR_MINIFORGE"
fi
source "$CVPR_MINIFORGE/etc/profile.d/conda.sh"
if [[ ! -x "$CVPR_CONDA_PREFIX/bin/python" ]]; then
  [[ ! -e "$CVPR_CONDA_PREFIX" ]] || { echo "Incomplete existing environment: $CVPR_CONDA_PREFIX; refusing to overwrite" >&2; exit 1; }
  conda create --yes --prefix "$CVPR_CONDA_PREFIX" --override-channels -c conda-forge python=3.10 pip
fi
conda activate "$CVPR_CONDA_PREFIX"
python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 10), f"README requires Python 3.10, found {sys.version}"
print("Environment:", sys.executable, sys.version)
PY
# Observed PBS 706152 startup failure: gym_sokoban imports pkg_resources,
# which the installed setuptools 84 no longer provides. Version 80.9.0
# retains that module and supports Python >=3.9, including this Python 3.10.
# https://github.com/pypa/setuptools/blob/v80.9.0/pyproject.toml
# https://github.com/pypa/setuptools/blob/v80.9.0/pkg_resources/__init__.py
python -m pip install setuptools==80.9.0
[[ "$stage" != --bootstrap ]] || exit 0

init_modules() {
  if ! type module >/dev/null 2>&1; then
    for script in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
      if [[ -r "$script" ]]; then
        set +u
        source "$script"
        set -u
        break
      fi
    done
  fi
}
check_vulkan() {
  python -c 'import ctypes; ctypes.CDLL("libvulkan.so.1")' && command -v vulkaninfo >/dev/null 2>&1
}
ensure_vulkan() {
  echo 'Checking the available Vulkan loader and tools before installing anything.'
  if check_vulkan; then return; fi
  init_modules
  if type module >/dev/null 2>&1; then
    module -t avail 2>&1 | grep -i vulkan || true
    if [[ -n "${CVPR_VULKAN_MODULE:-}" ]]; then module load "$CVPR_VULKAN_MODULE"; fi
  fi
  if check_vulkan; then return; fi
  echo 'No usable system/module Vulkan loader + tools; trying conda-forge (no sudo).'
  conda install --yes --prefix "$CVPR_CONDA_PREFIX" --override-channels -c conda-forge libvulkan-loader vulkan-tools
  source "$repo/scripts/scratch_env.sh"
  check_vulkan || { echo 'Vulkan loader/tools remain unavailable; stop and report this limitation.' >&2; return 1; }
  echo 'A loader is not a GPU driver: actual H200 Vulkan rendering must still be tested in the GPU job.'
}

# Navigation README pins. These are sufficient for the separate rendering stage;
# --render does not claim that the VAGEN reinforcement-learning stack is installed.
python -m pip install ai2thor==5.0.0 numpy==1.25.1 psutil==6.1.1 nvidia-ml-py==12.560.30
ensure_vulkan
if [[ "$stage" == --render ]]; then
  python -m pip check
  python -m pip freeze
  exit 0
fi

checkout_pinned() {
  local url="$1" directory="$2" commit="$3" ref="${4:-vagen-legacy}"
  if [[ ! -d "$directory/.git" ]]; then
    [[ ! -e "$directory" ]] || { echo "Existing non-repository source path: $directory" >&2; return 1; }
    git clone --filter=blob:none --no-checkout --branch "$ref" "$url" "$directory"
    git -C "$directory" checkout --detach "$commit"
  fi
  [[ "$(git -C "$directory" rev-parse HEAD)" == "$commit" ]] || { echo "Source commit mismatch: $directory" >&2; return 1; }
  git -C "$directory" diff --quiet
  git -C "$directory" diff --cached --quiet
  printf '%s %s\n' "$directory" "$commit"
}
mkdir -p "$CVPR_SCRATCH/sources"
checkout_pinned https://github.com/JamesKrW/verl.git "$verl_source" "$VERL_COMMIT"
checkout_pinned https://github.com/mll-lab-nu/VAGEN.git "$vagen_source" "$VAGEN_COMMIT"
safe_link "$vagen_source" "$repo/third_party/VAGEN"

# Observed repair for PBS 706124: vllm==0.8.2 requires xgrammar==0.1.16,
# whose PyPI release is unavailable. Preserve that exact version from its
# official tag/commit instead of changing the README's vllm dependency.
# Release: https://github.com/mlc-ai/xgrammar/releases/tag/v0.1.16
# Source-build instructions linked by that revision's README:
# https://github.com/mlc-ai/xgrammar/blob/v0.1.16/docs/start/install.rst
# Its pyproject builds C++17 bindings with scikit-build-core/pybind11; torch
# is a runtime dependency, not a build requirement. Let verl resolve it below.
checkout_pinned https://github.com/mlc-ai/xgrammar.git "$xgrammar_source" "$XGRAMMAR_COMMIT" v0.1.16
if python - "$xgrammar_source" <<'PY'
import importlib.metadata
import json
from pathlib import Path
import sys

try:
    dist = importlib.metadata.distribution("xgrammar")
    origin = json.loads(dist.read_text("direct_url.json") or "{}")
    expected = Path(sys.argv[1]).resolve().as_uri()
    reusable = (
        dist.version == "0.1.16"
        and origin.get("url") == expected
        and isinstance(origin.get("dir_info"), dict)
        and not origin["dir_info"].get("editable", False)
    )
except (importlib.metadata.PackageNotFoundError, ValueError, OSError):
    reusable = False
sys.exit(0 if reusable else 1)
PY
then
  echo 'Reusing exact xgrammar 0.1.16 installed from the verified pinned source path.'
else
  git -C "$xgrammar_source" submodule update --init --recursive
  git -C "$xgrammar_source" submodule status --recursive
  export CMAKE_BUILD_PARALLEL_LEVEL=2
  python -m pip install --no-deps "$xgrammar_source"
fi
python - <<'PY'
import importlib.metadata
version = importlib.metadata.version("xgrammar")
assert version == "0.1.16", f"Expected exact xgrammar 0.1.16, found {version}"
print("Pinned source xgrammar metadata version:", version)
PY

# README: install the pinned legacy verl editable first, then VAGEN install.sh.
# vllm==0.8.2 in this verl revision resolves torch==2.6.0. CUDA is not README-pinned.
python -m pip install -e "$verl_source"
init_modules
if ! command -v nvcc >/dev/null 2>&1; then
  if type module >/dev/null 2>&1; then
    module load "${CVPR_CUDA_MODULE:-cuda12.4/toolkit/12.4.1}"
  fi
fi
command -v nvcc >/dev/null 2>&1 || { echo 'flash-attn may require nvcc; no usable CUDA toolkit. Set CVPR_CUDA_MODULE to an available module and retry.' >&2; exit 1; }
nvcc --version
# Observed PBS 706127 failure: the upstream flash-attn command's isolated
# metadata build could not import torch, although verl installed torch 2.6.0.
# Install the same pinned version using that environment, then run upstream
# install.sh unchanged (its identical requirement is already satisfied).
python -m pip install --no-build-isolation flash-attn==2.7.4.post1
(cd "$vagen_source" && bash -e scripts/install.sh)
# Navigation README dependencies are separate from the top-level installation.
python -m pip install ai2thor==5.0.0 numpy==1.25.1 psutil==6.1.1 nvidia-ml-py==12.560.30
python -m pip check
# Metadata consistency alone missed gym_sokoban's legacy pkg_resources import.
# Validate the actual server and training imports inside this PBS allocation.
python "$repo/src/tools/patch_vllm_uuid.py"
python - <<'PY'
import ctypes
import importlib
import json
import os
import uuid
from vagen.env import REGISTERED_ENV
if "navigation" not in REGISTERED_ENV:
    # The registry suppresses optional ImportError; expose its original cause.
    importlib.import_module("vagen.env.navigation")
    raise RuntimeError("Navigation was not registered")
for name in ("vagen.server.server", "torch", "flash_attn", "vllm"):
    importlib.import_module(name)
from vllm.platforms.cuda import device_id_to_physical_device_id
from vllm.model_executor.models import ModelRegistry
import torch
visible = os.environ["CUDA_VISIBLE_DEVICES"]
if not visible.startswith("GPU-") or "," in visible:
    raise RuntimeError("This T00 validation requires the single complete PBS GPU UUID")
cuda = ctypes.CDLL("libcuda.so.1")
def checked(name, *args):
    code = getattr(cuda, name)(*args)
    if code:
        raise RuntimeError(f"{name} returned CUDA error {code}")
checked("cuInit", 0)
device = ctypes.c_int()
checked("cuDeviceGet", ctypes.byref(device), 0)
raw_uuid = (ctypes.c_ubyte * 16)()
checked("cuDeviceGetUuid_v2", ctypes.byref(raw_uuid), device)
actual_uuid = "GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid)))
if actual_uuid != visible or torch.cuda.device_count() != 1:
    raise RuntimeError("CUDA driver device 0 differs from the single PBS UUID")
physical_index = device_id_to_physical_device_id(0)
info, architecture = ModelRegistry.inspect_model_cls(["Qwen2_5_VLForConditionalGeneration"])
if architecture != "Qwen2_5_VLForConditionalGeneration" or info is None:
    raise RuntimeError("Qwen2.5-VL model inspection did not complete")
print("SETUP_IMPORT_CHECK: Navigation registered; server, torch, flash_attn, vllm imported")
print("SETUP_GPU_IDENTITY_CHECK: " + json.dumps({"pbs_uuid": visible,
    "cuda_device_0_uuid": actual_uuid, "nvml_physical_index": physical_index,
    "qwen_architecture_inspected": architecture, "cuda_visible_devices_unchanged": True}))
PY
python -m pip freeze
conda list --explicit
printf 'SETUP_COMPLETE stage=%s VAGEN=%s VERL=%s XGRAMMAR=%s\n' "$stage" "$VAGEN_COMMIT" "$VERL_COMMIT" "$XGRAMMAR_COMMIT"
