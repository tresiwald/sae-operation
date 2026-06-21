#!/bin/bash
# Run the follow-up experiments (exp1-exp3) locally, reusing the trained SAEs.
#
# Usage:
#   bash run_experiments.sh            # run exp1 (decision gate) only
#   bash run_experiments.sh 1 2        # run exp1 and exp2
#   bash run_experiments.sh all        # run all three
#
# exp1 is the decision gate: if operations are not encoded under matched
# operands, exp2/exp3 are not worth running. Check exp1's verdict first.

set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate
export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"

WHICH="${*:-1}"
[ "$WHICH" = "all" ] && WHICH="1 2 3"

LOG_TS=$(date +%Y%m%d_%H%M%S)
run() {
    local n="$1"; local mod="$2"
    echo "==> exp${n}  →  logs/exp${n}_${LOG_TS}.log"
    python -m "experiments.$mod" 2>&1 | tee "logs/exp${n}_${LOG_TS}.log"
}

mkdir -p logs
for n in $WHICH; do
    case $n in
        1) run 1 exp1_hard_negatives ;;
        2) run 2 exp2_error_mechanism ;;
        3) run 3 exp3_ablation ;;
        *) echo "unknown experiment: $n" ;;
    esac
done

echo ""
echo "Done. Results + plots in results/ (exp1_*, exp2_*, exp3_*)."
