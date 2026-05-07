"""
Full evaluation loop across baselines and WARP variants.

Run this to reproduce Table 1 and Table 2.

Baseline execution order:
1. flat_start
2. dc_warmstart
3. det_gnn
4. diffopf
5. warp_k1
6. warp_k5
"""

import logging
import csv
from pathlib import Path
from typing import List

import numpy as np

from eval.metrics import compute_metrics, EvalMetrics

logger = logging.getLogger(__name__)


def run_benchmark(case: str, split: str, methods: List[str],
                  results_dir: str = "results/tables",
                  **kwargs) -> dict:
    """
    Run full benchmark for a given case and split.

    Args:
        case: grid case name (e.g., "case118")
        split: "fulltop" or "n-1"
        methods: list of method names to evaluate
        results_dir: where to save CSV results

    Returns:
        dict of {method: EvalMetrics}
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    for method in methods:
        logger.info(f"Evaluating {method} on {case} {split}...")

        if method == "flat_start":
            from baselines.flat_start import evaluate_flat_start
            results = evaluate_flat_start(case, split, **kwargs)
        elif method == "dc_warmstart":
            from baselines.dc_warmstart import evaluate_dc_warmstart
            results = evaluate_dc_warmstart(case, split, **kwargs)
        elif method == "det_gnn":
            from baselines.det_gnn_baseline import evaluate_det_gnn
            results = evaluate_det_gnn(case, split, **kwargs)
        else:
            logger.warning(f"Unknown method: {method}, skipping")
            continue

        metrics = compute_metrics(results, method, case, split)
        all_metrics[method] = metrics

        _save_results_csv(results, case, split, method, results_dir)

        logger.info(
            f"  {method}: IPM mean={metrics.ipm_mean:.1f} "
            f"(std={metrics.ipm_std:.1f}), "
            f"conv={metrics.conv_rate:.1%}"
        )

    return all_metrics


def _save_results_csv(results: List[dict], case: str, split: str,
                      method: str, results_dir: Path):
    """Save per-instance results to CSV."""
    filename = results_dir / f"{case}_{split}_{method}.csv"

    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"  Saved {len(results)} results to {filename}")
