#!/bin/bash
# Create a virtual environment and install all dependencies.
#
# Usage:
#   bash setup.sh            # create .venv and install
#   bash setup.sh --force    # delete existing .venv and reinstall from scratch

set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"

if [ "${1:-}" = "--force" ] && [ -d "$VENV" ]; then
    echo "==> Removing existing $VENV"
    rm -rf "$VENV"
fi

if [ ! -d "$VENV" ]; then
    echo "==> Creating virtual environment in $VENV"
    python3 -m venv "$VENV"
else
    echo "==> Virtual environment already exists — skipping creation (use --force to recreate)"
fi

echo "==> Activating $VENV"
source "$VENV/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip --quiet

echo "==> Installing requirements"
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "To activate:  source $VENV/bin/activate"
echo "To run:       bash run_pipeline.sh"
