#!/bin/bash
# Run the full SAE fingerprint pipeline locally (no SLURM).
#
# Usage:
#   bash run_pipeline.sh                 # full pipeline
#   bash run_pipeline.sh --skip-data     # skip stage 1 (data already exists)
#   bash run_pipeline.sh --skip-acts     # skip stages 1-2 (acts cached)
#   bash run_pipeline.sh --skip-sae      # skip stages 1-3 (SAEs cached)
#   bash run_pipeline.sh --accuracy      # also run stage 5
#
# All SAE_* env vars from slurm/env.sh apply here too.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

# ── environment (mirrors slurm/env.sh) ────────────────────────────────────────
source .venv/bin/activate
export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"

# ── flags ─────────────────────────────────────────────────────────────────────
SKIP_DATA=0; SKIP_ACTS=0; SKIP_SAE=0; RUN_ACC=0

for arg in "$@"; do
    case $arg in
        --skip-data)  SKIP_DATA=1 ;;
        --skip-acts)  SKIP_DATA=1; SKIP_ACTS=1 ;;
        --skip-sae)   SKIP_DATA=1; SKIP_ACTS=1; SKIP_SAE=1 ;;
        --accuracy)   RUN_ACC=1 ;;
    esac
done

LOG_TS=$(date +%Y%m%d_%H%M%S)

run_stage() {
    local name="$1"; local module="$2"; shift 2
    local log="logs/${name}_${LOG_TS}.log"
    echo "==> Stage $name  →  $log"
    python -m "$module" "$@" 2>&1 | tee "$log"
}

# ── stages ────────────────────────────────────────────────────────────────────
[ $SKIP_DATA -eq 0 ] && run_stage "01_data"     pipeline.stage1_data
[ $SKIP_ACTS -eq 0 ] && run_stage "02_acts"     pipeline.stage2_acts

if [ $SKIP_SAE -eq 0 ]; then
    # Read layers from the acts checkpoint so we don't hard-code them here
    LAYERS=$(python - <<'EOF'
import torch, sys, os
sys.path.insert(0, "code")
from pipeline.config import ckpt_acts
ck = torch.load(ckpt_acts(), weights_only=True)
print(" ".join(str(l) for l in ck["layers"]))
EOF
)
    echo "==> Layers to train: $LAYERS"
    for L in $LAYERS; do
        run_stage "03_sae_L${L}" pipeline.stage3_sae --layer "$L"
    done
fi

run_stage "04_analysis"  pipeline.stage4_analysis
[ $RUN_ACC -eq 1 ] && run_stage "05_accuracy" pipeline.stage5_accuracy

echo ""
echo "Pipeline complete.  Logs in logs/  Results in results/"
