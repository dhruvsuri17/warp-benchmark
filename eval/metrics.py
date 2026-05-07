"""
Evaluation metrics: IPM iteration counter, convergence rate, WS-RMSE.

Primary metric: IPOPT interior-point iteration count.
"""

import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class EvalMetrics:
    method: str
    case: str
    split: str
    n_instances: int
    ipm_mean: float
    ipm_std: float
    ipm_median: float
    ipm_p90: float
    conv_rate: float
    ws_rmse_mean: float
    obj_gap_mean: float


def compute_metrics(results: List[dict], method: str, case: str,
                    split: str) -> EvalMetrics:
    """
    Aggregate per-instance results into summary metrics.

    Args:
        results: list of dicts with keys from benchmark evaluation
        method: method name
        case: grid case name
        split: "fulltop" or "n-1"

    Returns:
        EvalMetrics summary
    """
    iters = [r["n_ipm_iters"] for r in results if r["n_ipm_iters"] is not None]
    converged = [r["converged"] for r in results]
    ws_rmse = [r.get("ws_rmse", 0) for r in results]
    obj_gaps = [r.get("obj_gap_pct", 0) for r in results]

    iters_arr = np.array(iters) if iters else np.array([0])

    return EvalMetrics(
        method=method,
        case=case,
        split=split,
        n_instances=len(results),
        ipm_mean=float(iters_arr.mean()),
        ipm_std=float(iters_arr.std()),
        ipm_median=float(np.median(iters_arr)),
        ipm_p90=float(np.percentile(iters_arr, 90)),
        conv_rate=float(np.mean(converged)),
        ws_rmse_mean=float(np.mean(ws_rmse)),
        obj_gap_mean=float(np.mean(obj_gaps)),
    )


def ws_rmse(x_ws: np.ndarray, x_opt: np.ndarray,
            x_range: np.ndarray = None) -> float:
    """
    Warm-start RMSE normalised by variable range.

    Args:
        x_ws: warm-start solution vector
        x_opt: converged optimal solution
        x_range: per-variable range for normalisation (max - min from training set)
    """
    diff = x_ws - x_opt
    if x_range is not None:
        diff = diff / np.maximum(x_range, 1e-6)
    return float(np.sqrt(np.mean(diff ** 2)))
