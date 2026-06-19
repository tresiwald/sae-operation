#!/bin/bash
#SBATCH --job-name=sae_data
#SBATCH --output=logs/01_data_%j.out
#SBATCH --error=logs/01_data_%j.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail
source "$(dirname "$0")/../slurm/env.sh"

python -m pipeline.stage1_data
