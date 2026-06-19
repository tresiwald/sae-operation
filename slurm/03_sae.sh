#!/bin/bash
#SBATCH --job-name=sae_train
#SBATCH --output=logs/03_sae_%A_%a.out
#SBATCH --error=logs/03_sae_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-5        # one task per layer index; adjust to len(LAYERS)-1

# Each array task trains one layer.  SLURM_ARRAY_TASK_ID is the index into
# the layers list stored in acts_checkpoint.pt (e.g. index 0 → layer 4).
#
# To train only specific layers: --array=0,2,4
# To limit concurrency:         --array=0-5%3

set -euo pipefail
source "$(dirname "$0")/../slurm/env.sh"

echo "Array task ${SLURM_ARRAY_TASK_ID} — training SAE for layer index ${SLURM_ARRAY_TASK_ID}"
python -m pipeline.stage3_sae
# stage3_sae.py reads SLURM_ARRAY_TASK_ID automatically when --layer is not passed
