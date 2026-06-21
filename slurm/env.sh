#!/bin/bash
# ── Cluster environment setup — edit for your HPC system ─────────────────────
# Sourced by every SLURM job script before running Python.

# -- modules (uncomment / adjust for your cluster) --
# module purge
# module load python/3.11
# module load cuda/12.1
# module load gcc/12

# -- project root (resolved relative to this file) --
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# -- virtual environment --
source "$PROJECT_ROOT/.venv/bin/activate"

# -- add code/ to PYTHONPATH so `import pipeline.xxx` works --
export PYTHONPATH="$PROJECT_ROOT/code:${PYTHONPATH:-}"

# ── Config overrides via env vars (all optional) ──────────────────────────────
# export SAE_MODEL_NAME="google/gemma-3-4b-pt"
# export SAE_LAYERS="4,8,13,17,22,25"
# export SAE_N_PER_CELL=150
# export SAE_EPOCHS=30
# export SAE_OUT_DIR="results"
# export SAE_DEVICE="cuda"            # force device (auto-detected if unset)
# export SAE_DTYPE=bfloat16          # default on CUDA; use float32 if bfloat16 also NaNs
# export SAE_ACT_BATCH=8             # reduce if deeper layers produce NaN
# export SAE_MEASURE_ACCURACY=1       # enable stage5

# HuggingFace
# export HF_TOKEN="hf_..."                             # set your token here
# export HF_HOME=/scratch/$USER/.cache/huggingface    # point to fast scratch if available

echo "ENV  project=$PROJECT_ROOT  python=$(which python)  device=${SAE_DEVICE:-auto}"
