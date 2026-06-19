#!/bin/bash
# Submit the SAE fingerprint pipeline for 11 sample-size variants.
#
# Scales: 150 → 153600  (current × 2^0 … × 2^10)
#
# Each scale is fully independent; all pipelines run in parallel.
# Within each scale: stage1 → stage2 → stage3[] → stage4 are chained.
#
# Usage:
#   bash slurm/submit_scaling.sh                     # submit all 11 scales
#   bash slurm/submit_scaling.sh --dry-run           # print commands only
#   bash slurm/submit_scaling.sh --scales "150 300"  # subset
#   bash slurm/submit_scaling.sh --accuracy          # also submit stage5
#   bash slurm/submit_scaling.sh --n-layers 6        # default 6
#
# After all runs finish:
#   python -m pipeline.stage6_scaling_analysis

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# ── defaults ──────────────────────────────────────────────────────────────────
SCALES="150 300 600 1200 2400 4800 9600 19200 38400 76800 153600"
N_LAYERS=6
DRY_RUN=0
RUN_ACC=0

for arg in "$@"; do
    case $arg in
        --dry-run)        DRY_RUN=1 ;;
        --accuracy)       RUN_ACC=1 ;;
        --n-layers=*)     N_LAYERS="${arg#*=}" ;;
        --scales=*)       SCALES="${arg#*=}" ;;
    esac
done

ARRAY_END=$((N_LAYERS - 1))

# ── time-limit table (seconds) ────────────────────────────────────────────────
# Activation collection is the bottleneck; scales linearly with n_train.
# Base (n=150, ~8k records): ~30 min on single GPU with batching.
# Each doubling adds the same time.  Add 50% buffer.  Cap at 48 hr.
base_acts_sec=1800  # 30 min for n=150
base_n=150

acts_time_hms() {
    local n=$1
    local secs
    secs=$(echo "scale=0; ($base_acts_sec * $n / $base_n * 3 + 1) / 2" | bc)
    secs=$(( secs < 600    ? 600    : secs ))   # minimum 10 min
    secs=$(( secs > 172800 ? 172800 : secs ))   # cap 48 hr
    printf "%02d:%02d:%02d" $((secs/3600)) $(( (secs%3600)/60 )) $((secs%60))
}

# SAE training: ~7 min per layer per epoch set; roughly constant in n
sae_time="01:30:00"
# Analysis: reloads model for cheat holdout; ~30 min + inference
ana_time="02:00:00"

# ── sbatch wrapper ─────────────────────────────────────────────────────────────
sbatch_or_dry() {
    if [ $DRY_RUN -eq 1 ]; then
        echo "  [DRY] sbatch $*"
        echo "FAKE_JID_${RANDOM}"
    else
        sbatch --parsable "$@"
    fi
}

# ── submit loop ───────────────────────────────────────────────────────────────
echo "Submitting scaling pipeline for scales: $SCALES"
echo ""

for N in $SCALES; do
    OUT="results/scale_${N}"
    ACTS_T=$(acts_time_hms "$N")
    echo "── scale n_per_cell=${N}  out=${OUT}  acts_walltime=${ACTS_T} ──"

    # Common env overrides for this scale
    ENV_ARGS=(
        --export=ALL
        --export="SAE_N_PER_CELL=${N},SAE_OUT_DIR=${OUT}"
    )

    # Stage 1: data generation (fast, CPU)
    JID1=$(sbatch_or_dry \
        --job-name="sae_data_${N}" \
        --output="logs/01_data_${N}_%j.out" \
        --error="logs/01_data_${N}_%j.err" \
        --partition=cpu \
        --ntasks=1 --cpus-per-task=4 --mem=8G --time=00:15:00 \
        "${ENV_ARGS[@]}" \
        slurm/01_data.sh)
    echo "  Stage 1 (data):     job $JID1"

    # Stage 2: activation collection (GPU, time scales with N)
    JID2=$(sbatch_or_dry \
        --job-name="sae_acts_${N}" \
        --output="logs/02_acts_${N}_%j.out" \
        --error="logs/02_acts_${N}_%j.err" \
        --partition=gpu --ntasks=1 --cpus-per-task=8 --mem=64G \
        --gres=gpu:1 --time="${ACTS_T}" \
        --dependency=afterok:${JID1} \
        "${ENV_ARGS[@]}" \
        slurm/02_acts.sh)
    echo "  Stage 2 (acts):     job $JID2  [wall=${ACTS_T}]"

    # Stage 3: SAE training — array job, one task per layer
    JID3=$(sbatch_or_dry \
        --job-name="sae_train_${N}" \
        --output="logs/03_sae_${N}_%A_%a.out" \
        --error="logs/03_sae_${N}_%A_%a.err" \
        --partition=gpu --ntasks=1 --cpus-per-task=4 --mem=32G \
        --gres=gpu:1 --time="${sae_time}" \
        --array="0-${ARRAY_END}" \
        --dependency=afterok:${JID2} \
        "${ENV_ARGS[@]}" \
        slurm/03_sae.sh)
    echo "  Stage 3 (sae):      job $JID3  [array 0-${ARRAY_END}]"

    # Stage 4: analysis + plots
    JID4=$(sbatch_or_dry \
        --job-name="sae_analysis_${N}" \
        --output="logs/04_analysis_${N}_%j.out" \
        --error="logs/04_analysis_${N}_%j.err" \
        --partition=gpu --ntasks=1 --cpus-per-task=8 --mem=48G \
        --gres=gpu:1 --time="${ana_time}" \
        --dependency=afterok:${JID3} \
        "${ENV_ARGS[@]}" \
        slurm/04_analysis.sh)
    echo "  Stage 4 (analysis): job $JID4"

    # Stage 5: accuracy (optional)
    if [ $RUN_ACC -eq 1 ]; then
        JID5=$(sbatch_or_dry \
            --job-name="sae_acc_${N}" \
            --output="logs/05_acc_${N}_%j.out" \
            --error="logs/05_acc_${N}_%j.err" \
            --partition=gpu --ntasks=1 --cpus-per-task=4 --mem=48G \
            --gres=gpu:1 --time="03:00:00" \
            --dependency=afterok:${JID2} \
            "${ENV_ARGS[@]}" \
            slurm/05_accuracy.sh)
        echo "  Stage 5 (accuracy): job $JID5"
    fi

    echo ""
done

# Stage 6: scaling aggregation (runs after all stage4 jobs finish)
echo "── Stage 6: scaling aggregation ──"
echo "(Submit manually after all scale runs complete, or add --dependency=afterok:<all JID4s>)"
echo "  python -m pipeline.stage6_scaling_analysis"
echo ""
echo "Monitor:  squeue -u \$USER | grep sae_"
echo "Results:  ls results/scale_*/results.json"
