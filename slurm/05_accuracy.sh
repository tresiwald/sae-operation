#!/bin/bash
#SBATCH --job-name=sae_accuracy
#SBATCH --output=logs/05_accuracy_%j.out
#SBATCH --error=logs/05_accuracy_%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Optional.  SAE_ACC_SAMPLE controls how many records per op to evaluate.
# Default 200 per op × 4 ops = 800 records → ~15 min on an A100.

set -euo pipefail
source "$(dirname "$0")/../slurm/env.sh"

python -m pipeline.stage5_accuracy
