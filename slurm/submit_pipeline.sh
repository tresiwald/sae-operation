#!/bin/bash
# Submit the full SAE fingerprint pipeline as a SLURM dependency chain.
#
# Usage:
#   bash slurm/submit_pipeline.sh                 # full pipeline
#   bash slurm/submit_pipeline.sh --skip-data     # skip stage1 (data already exists)
#   bash slurm/submit_pipeline.sh --skip-acts     # skip stage1+2 (acts cached)
#   bash slurm/submit_pipeline.sh --skip-sae      # skip stage1-3 (SAEs cached)
#   bash slurm/submit_pipeline.sh --accuracy      # also submit stage5
#
# The script prints each job ID and the full dependency chain.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

SKIP_DATA=0; SKIP_ACTS=0; SKIP_SAE=0; RUN_ACC=0
N_LAYERS=6   # length of LAYERS list; must match --array in 03_sae.sh

for arg in "$@"; do
    case $arg in
        --skip-data)  SKIP_DATA=1 ;;
        --skip-acts)  SKIP_DATA=1; SKIP_ACTS=1 ;;
        --skip-sae)   SKIP_DATA=1; SKIP_ACTS=1; SKIP_SAE=1 ;;
        --accuracy)   RUN_ACC=1 ;;
        --n-layers=*) N_LAYERS="${arg#*=}" ;;
    esac
done

# ── Stage 1: data generation ──────────────────────────────────────────────────
if [ $SKIP_DATA -eq 0 ]; then
    JID1=$(sbatch --parsable slurm/01_data.sh)
    echo "Stage 1 (data):     job $JID1"
    DEP_ACTS="--dependency=afterok:$JID1"
else
    echo "Stage 1 (data):     skipped"
    DEP_ACTS=""
fi

# ── Stage 2: activation collection ───────────────────────────────────────────
if [ $SKIP_ACTS -eq 0 ]; then
    JID2=$(sbatch --parsable $DEP_ACTS slurm/02_acts.sh)
    echo "Stage 2 (acts):     job $JID2"
    DEP_SAE="--dependency=afterok:$JID2"
else
    echo "Stage 2 (acts):     skipped"
    DEP_SAE=""
fi

# ── Stage 3: SAE training (array job, one task per layer) ─────────────────────
if [ $SKIP_SAE -eq 0 ]; then
    ARRAY_END=$((N_LAYERS - 1))
    JID3=$(sbatch --parsable $DEP_SAE \
           --array="0-${ARRAY_END}" \
           slurm/03_sae.sh)
    echo "Stage 3 (sae):      job $JID3  (array 0-${ARRAY_END})"
    DEP_ANALYSIS="--dependency=afterok:$JID3"
else
    echo "Stage 3 (sae):      skipped"
    DEP_ANALYSIS=""
fi

# ── Stage 4: analysis + plots ────────────────────────────────────────────────
JID4=$(sbatch --parsable $DEP_ANALYSIS slurm/04_analysis.sh)
echo "Stage 4 (analysis): job $JID4"

# ── Stage 5: accuracy (optional) ─────────────────────────────────────────────
if [ $RUN_ACC -eq 1 ]; then
    # Can run in parallel with stage 4 (only needs data.pkl)
    JID5=$(sbatch --parsable ${DEP_SAE:-""} slurm/05_accuracy.sh)
    echo "Stage 5 (accuracy): job $JID5"
fi

echo ""
echo "Pipeline submitted.  Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/04_analysis_\${JID4}.out"
