#!/bin/bash
# Run the full pipeline + experiments for ONE model into its own results dir.
#
# Usage:
#   bash run_model.sh qwen-base                 # pipeline + experiments + gen
#   bash run_model.sh qwen-math pipeline        # just the SAE pipeline
#   bash run_model.sh qwen-base experiments     # just exp1-5 (+ gen)
#
# Each model writes to results/<key>/ so models can be compared side by side.

set -euo pipefail
cd "$(dirname "$0")"

KEY="${1:?usage: run_model.sh <gemma|qwen-base|qwen-math> [pipeline|experiments|all]}"
STAGE="${2:-all}"

source models.sh
select_model "$KEY"
export SAE_MODEL_NAME SAE_LAYERS SAE_OUT_DIR SAE_CKPT_DIR SAE_DATA_MODE

echo "════════════════════════════════════════════════════════════════"
echo " Model : $SAE_MODEL_NAME"
echo " Out   : $SAE_OUT_DIR"
echo " Layers: $SAE_LAYERS   Data: $SAE_DATA_MODE"
echo "════════════════════════════════════════════════════════════════"

if [ "$STAGE" = pipeline ] || [ "$STAGE" = all ]; then
    bash run_pipeline.sh
fi
if [ "$STAGE" = experiments ] || [ "$STAGE" = all ]; then
    bash run_experiments.sh all
    bash run_experiments.sh gen
fi

echo "Done: $KEY → $SAE_OUT_DIR"
