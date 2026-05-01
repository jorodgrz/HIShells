#!/usr/bin/env bash
set -euo pipefail

# Quiet TensorFlow logs
export TF_CPP_MIN_LOG_LEVEL=3

# Tweaks (can override via env: RUN=..., EVERY=...)
RUN="${RUN:-pv_unet_$(date +%Y%m%d_%H%M)}"
EVERY="${EVERY:-5}"

echo "== dataset selftest =="
python -m src.pv.dataset --config pv_config.yaml --selftest

echo "== training =="
python -m src.train.train_pv_unet --config pv_config.yaml --run "$RUN" --quiet --every "$EVERY"

echo "== outputs =="
echo "  - best model: runs/$RUN/best_model.keras"
echo "  - final model: runs/$RUN/final_model.keras"
echo "  - logs: runs/$RUN/history.csv"