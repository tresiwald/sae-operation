#!/bin/bash
#SBATCH --job-name=sae_analysis
#SBATCH --output=logs/04_analysis_%j.out
#SBATCH --error=logs/04_analysis_%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Needs GPU only for the cheat-holdout inference pass.
# If you skip cheat scoring, --partition=cpu is enough.

set -euo pipefail
source "$(dirname "$0")/../slurm/env.sh"

python -m pipeline.stage4_analysis
