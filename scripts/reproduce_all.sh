#!/usr/bin/env bash
# Master reproduction script (run from warp-benchmark/).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"

echo "== WARP benchmark reproduction pipeline =="

echo "[1/7] Dependencies"
pip install -r requirements.txt

echo "[2/7] Download OPFDataset (Case 118)"
python -c "from torch_geometric.datasets import OPFDataset; OPFDataset(root='data', case_name='pglib_opf_case118_ieee')"

echo "[3/7] Extract dual labels (long-running; set env SKIP_DUAL_EXTRACT=1 to skip)"
if [[ "${SKIP_DUAL_EXTRACT:-0}" != "1" ]]; then
  python data/extract_duals.py --case pglib_opf_case118_ieee --data-root data \
    --output-root data/duals/pglib_opf_case118_ieee --n-train "${N_TRAIN:-5000}" --n-val "${N_VAL:-500}" --n-test "${N_TEST:-50}"
else
  echo "Skipping dual extraction."
fi

echo "[4/7] Normalisation stats"
python data/compute_norm_stats.py --data-dir data/duals/pglib_opf_case118_ieee/train --output data/case118/norm_stats.pt

echo "[5/7] Train (skip if SKIP_TRAIN=1)"
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  python train/train_warp.py --epochs "${EPOCHS:-200}" --hidden-dim "${HIDDEN:-128}" --k-steps "${K_STEPS:-15}"
  python train/train_lstm.py --epochs "${EPOCHS:-200}" || true
else
  echo "Skipping training."
fi

echo "[6/7] Evaluate oracle protocol"
python eval/evaluate.py suite-oracle --duals-dir data/duals/pglib_opf_case118_ieee --n-test "${N_TEST:-50}" || true

echo "[7/7] Figures"
python scripts/generate_figures.py || true

echo "Done."
