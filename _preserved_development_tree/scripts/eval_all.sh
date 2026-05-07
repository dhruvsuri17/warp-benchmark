#!/usr/bin/env bash
# Run full benchmark evaluation for a given case.
#
# Usage:
#   bash scripts/eval_all.sh case118

set -euo pipefail

CASE="${1:-case118}"

echo "=== Running full evaluation for $CASE ==="

echo "--- Flat start baseline ---"
python -c "
from eval.benchmark import run_benchmark
run_benchmark('$CASE', 'fulltop', ['flat_start'])
"

echo "--- DC warm-start baseline ---"
python -c "
from eval.benchmark import run_benchmark
run_benchmark('$CASE', 'fulltop', ['dc_warmstart'])
"

echo "--- Det-GNN baseline ---"
python -c "
from eval.benchmark import run_benchmark
run_benchmark('$CASE', 'fulltop', ['det_gnn'])
"

echo "=== Evaluation complete for $CASE ==="
echo "Results in: results/tables/"
