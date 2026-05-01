#!/usr/bin/env bash
set -euo pipefail

echo "== Resolve config =="
python -m src.qa.print_resolved_config --config pv_config.yaml

echo "== Infer PV probabilities with latest best checkpoint =="
python -m src.infer.infer_pv --config pv_config.yaml

echo "== Calibrate global threshold on labels =="
python -m src.infer.calibrate_threshold --config pv_config.yaml || {
  echo "Calibration failed (do you have labels?). Skipping."; }

echo "== Aggregate PV hits to sky candidates =="
python -m src.infer.aggregate_candidates --config pv_config.yaml

echo "== Peek at calibration & candidates =="
test -f data/calib/threshold.txt && cat data/calib/threshold.txt || echo "(no threshold.txt)"
test -f data/candidates.json && head -n 50 data/candidates.json || echo "(no candidates.json)"

echo "✅ Inference + aggregation complete."