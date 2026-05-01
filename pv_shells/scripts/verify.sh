#!/usr/bin/env bash
set -euo pipefail

# --- locate repo root (directory containing pv_config.yaml and src/) ---
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# assume layout .../pv_shells/scripts/verify.sh -> repo root is parent
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# If invoked from anywhere else, still hop to the root
cd "$ROOT_DIR"

# Double-check we’re in the right place
if [[ ! -f "pv_config.yaml" || ! -d "src" ]]; then
  echo "❌ Could not find pv_config.yaml or src/ in: $ROOT_DIR"
  echo "   Make sure you're in the project root or script can locate it."
  exit 1
fi

# Make local package discoverable
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# Headless plots (matplotlib)
export MPLBACKEND=Agg

echo "== Repo root: $ROOT_DIR =="
echo "== Python: $(python --version) =="
echo "== PYTHONPATH includes repo root =="

echo "==[0/7] Resolve dynamic config =="
python -m src.qa.print_resolved_config --config pv_config.yaml --set optim.batch_size=2

echo "==[1/7] Environment report =="
python -m src.qa.verify_env

echo "==[2/7] FITS header sanity =="
python -m src.qa.check_headers --config pv_config.yaml

echo "==[3/7] Synthetic e2e injection test =="
python -m src.qa.run_injection_test --config pv_config.yaml

echo "==[4/7] Generate PV + labels (stubs OK) =="
python -m src.pv.make_pv --config pv_config.yaml
python -m src.pv.label_pv --config pv_config.yaml

echo "==[5/7] Quick PV overlays (png outputs) =="
python -m src.qa.overlay_pv_debug --config pv_config.yaml --max 6

echo "==[6/7] Pytest smoke tests =="
pytest -q --maxfail=1 -x tests

echo "✅ All verification steps passed."