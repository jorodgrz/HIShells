#!/usr/bin/env bash
set -euo pipefail

# --- locate repo root ---
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export MPLBACKEND=Agg

echo "== Resolve config =="
python -m src.qa.print_resolved_config --config pv_config.yaml

echo "== FITS header check =="
python -m src.qa.check_headers --config pv_config.yaml

echo "== Inspect catalog (.dat) =="
python scripts/inspect_catalog.py

echo "== Make PV slices =="
python -m src.pv.make_pv --config pv_config.yaml

echo "== Label PV slices (from two .dat tables) =="
python -m src.pv.label_pv --config pv_config.yaml

echo "== Generate/train/val/test splits =="
python scripts/gen_splits.py --config pv_config.yaml --seed 123

echo "== Overlay PV + labels for spot-checks =="
python -m src.qa.overlay_pv_debug --config pv_config.yaml --max 8

echo "✅ PV generation complete. Manifests in data/splits/"