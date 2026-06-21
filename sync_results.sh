#!/bin/bash
# Commit and push results (JSON + plots) to GitHub after a pipeline run.
#
# Usage:
#   bash sync_results.sh                      # auto-message from results.json
#   bash sync_results.sh "custom commit msg"  # explicit message
#
# Safe to run repeatedly — skips push if nothing changed.

set -euo pipefail
cd "$(dirname "$0")"

# ── build commit message ──────────────────────────────────────────────────────
if [ -n "${1:-}" ]; then
    MSG="$1"
else
    # pull key fields from results.json if available
    if [ -f results/results.json ] && command -v python3 &>/dev/null; then
        MSG=$(python3 - <<'EOF'
import json, pathlib
r = json.loads(pathlib.Path("results/results.json").read_text())
model  = r.get("model", "?").split("/")[-1]
best   = r.get("best_layer", "?")
layers = r.get("layers", [])
print(f"results: {model}  best_layer={best}  layers={layers}")
EOF
)
    else
        MSG="results: pipeline run $(date +%Y-%m-%d)"
    fi
fi

# ── stage results files ───────────────────────────────────────────────────────
git add results/*.json results/*.png 2>/dev/null || true
git add logs/                        2>/dev/null || true

if git diff --cached --quiet; then
    echo "Nothing to commit — results already up to date."
    exit 0
fi

git commit -m "$MSG"
git push
echo "Pushed: $MSG"
