#!/bin/bash
#SBATCH --job-name=sae_acts
#SBATCH --output=logs/02_acts_%j.out
#SBATCH --error=logs/02_acts_%j.err
#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:A40:1
#SBATCH --time=02:00:00

# Gemma-3-1B  @ 1 record/step: ~25 min
# Gemma-3-4B                  : ~1.5 hr  → bump --time to 03:00:00
# Gemma-3-12B                 : ~4 hr    → bump to 06:00:00

set -euo pipefail
source "$(dirname "$0")/../slurm/env.sh"

python -m pipeline.stage2_acts
