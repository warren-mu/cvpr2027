#!/usr/bin/env bash
# Source this file; all large state and caches belong on scratch, never in the repo.
export CVPR_SCRATCH="${CVPR_SCRATCH:-/scratch/${USER:?USER must be set}/cvpr2027}"
case "$CVPR_SCRATCH" in /*) ;; *) echo 'CVPR_SCRATCH must be absolute' >&2; return 2 2>/dev/null || exit 2 ;; esac
export CVPR_MINIFORGE="$CVPR_SCRATCH/miniforge3"
export CVPR_CONDA_PREFIX="$CVPR_SCRATCH/envs/cvpr2027"
export CONDA_PKGS_DIRS="$CVPR_SCRATCH/conda/pkgs"
export CONDA_ENVS_PATH="$CVPR_SCRATCH/envs"
export CONDARC="$CVPR_SCRATCH/config/condarc"
export PIP_CACHE_DIR="$CVPR_SCRATCH/cache/pip"
export PYTHONPYCACHEPREFIX="$CVPR_SCRATCH/cache/pycache"
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$CVPR_SCRATCH/cache/xdg"
export XDG_CONFIG_HOME="$CVPR_SCRATCH/config/xdg"
export XDG_DATA_HOME="$CVPR_SCRATCH/data/xdg"
export HF_HOME="$CVPR_SCRATCH/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$CVPR_SCRATCH/cache/torch"
export TORCH_EXTENSIONS_DIR="$CVPR_SCRATCH/cache/torch_extensions"
export TRITON_CACHE_DIR="$CVPR_SCRATCH/cache/triton"
export CUDA_CACHE_PATH="$CVPR_SCRATCH/cache/cuda"
export RAY_TMPDIR="$CVPR_SCRATCH/ray"
export TMPDIR="$CVPR_SCRATCH/tmp"
export TEMP="$TMPDIR" TMP="$TMPDIR"
export T00_SCRATCH_ROOT="$CVPR_SCRATCH/t00"
export CVPR_MODEL_DIR="$CVPR_SCRATCH/models"
export CVPR_TRAIN_OUTPUT_DIR="$T00_SCRATCH_ROOT/training"
export CVPR_CHECKPOINT_DIR="$T00_SCRATCH_ROOT/checkpoints"
export WANDB_DIR="$CVPR_SCRATCH/wandb"
export WANDB_CACHE_DIR="$CVPR_SCRATCH/cache/wandb"
export WANDB_CONFIG_DIR="$CVPR_SCRATCH/config/wandb"
export WANDB_MODE=offline
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" "$(dirname "$CONDARC")" \
  "$PIP_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" \
  "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$RAY_TMPDIR" \
  "$TMPDIR" "$CVPR_MODEL_DIR" "$CVPR_TRAIN_OUTPUT_DIR" "$CVPR_CHECKPOINT_DIR" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$CVPR_SCRATCH/ai2thor"
if [[ -d "$CVPR_CONDA_PREFIX/bin" ]]; then
  export PATH="$CVPR_CONDA_PREFIX/bin:$PATH"
fi
if [[ -f "$CVPR_CONDA_PREFIX/lib/libvulkan.so.1" ]]; then
  export LD_LIBRARY_PATH="$CVPR_CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# Deliberately never assign CUDA_VISIBLE_DEVICES: PBS owns GPU visibility.
