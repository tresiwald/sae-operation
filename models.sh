#!/bin/bash
# Model profiles for the cross-model comparison.
#
# Qwen2.5-1.5B (base) vs Qwen2.5-Math-1.5B is a near-controlled pair: same
# family, tokenizer and architecture, differing mainly in math capability.
# Gemma is kept for continuity. Sweep layers are placed at matched fractional
# depths so layers are comparable across the different layer counts.
#
# select_model <key> sets, for the calling shell:
#   SAE_MODEL_NAME, SAE_LAYERS, SAE_OUT_DIR, SAE_CKPT_DIR, SAE_DATA_MODE

select_model() {
    case "$1" in
        gemma)                       # 26 layers, d=1152
            SAE_MODEL_NAME="google/gemma-3-1b-pt"
            SAE_LAYERS="4,8,13,17,22,25" ;;

        # ── 1.5B base / math pair (28 layers, d=1536) ──────────────────────────
        qwen-base|qwen-base-1.5b)
            SAE_MODEL_NAME="Qwen/Qwen2.5-1.5B"
            SAE_LAYERS="4,9,14,18,24,27" ;;
        qwen-math|qwen-math-1.5b)
            SAE_MODEL_NAME="Qwen/Qwen2.5-Math-1.5B"
            SAE_LAYERS="4,9,14,18,24,27" ;;

        # ── 7B base / math pair (28 layers, d=3584) ────────────────────────────
        # Heavier: ~14 GB model in bf16 + a d_sae≈28.7k SAE per layer. Consider
        # SAE_RATIO=4 and a smaller SAE_GEN_BATCH on memory-tight nodes.
        qwen-base-7b)
            SAE_MODEL_NAME="Qwen/Qwen2.5-7B"
            SAE_LAYERS="4,9,14,18,24,27" ;;
        qwen-math-7b)
            SAE_MODEL_NAME="Qwen/Qwen2.5-Math-7B"
            SAE_LAYERS="4,9,14,18,24,27" ;;

        *)
            echo "unknown model key: $1" >&2
            echo "  use: gemma | qwen-base[-1.5b] | qwen-math[-1.5b]" >&2
            echo "       qwen-base-7b | qwen-math-7b" >&2
            return 1 ;;
    esac
    SAE_OUT_DIR="results/$1"
    SAE_CKPT_DIR="results/$1/checkpoints"
    # magnitude-stratified data by default so the comparison is on clean data
    SAE_DATA_MODE="${SAE_DATA_MODE:-loguniform}"
}

# Two controlled capability pairs at two scales — the comparison grid.
SCALE_PAIRS="qwen-base-1.5b qwen-math-1.5b qwen-base-7b qwen-math-7b"
ALL_MODELS="gemma $SCALE_PAIRS"
