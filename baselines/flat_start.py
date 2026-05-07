"""
Flat start baseline: V=1 p.u., theta=0, Pg at midpoint of limits.

This is the ceiling baseline — WARP should always beat this.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def make_flat_start(n_bus: int, n_gen: int,
                    Pg_min: np.ndarray = None, Pg_max: np.ndarray = None) -> dict:
    """
    Construct a flat start warm-start vector.

    Returns:
        dict with Vm, Va, Pg, Qg arrays
    """
    Vm = np.ones(n_bus)
    Va = np.zeros(n_bus)

    if Pg_min is not None and Pg_max is not None:
        Pg = (Pg_min + Pg_max) / 2.0
    else:
        Pg = np.zeros(n_gen)

    Qg = np.zeros(n_gen)

    return {"Vm": Vm, "Va": Va, "Pg": Pg, "Qg": Qg}


def evaluate_flat_start(case: str, split: str, test_loader=None,
                        nets=None, **kwargs) -> List[dict]:
    """
    Run flat start evaluation across test set.

    Returns:
        list of per-instance result dicts
    """
    from eval.ipopt_wrapper import run_flat_start

    results = []

    if nets is None:
        logger.warning("No pandapower nets provided, returning empty results")
        return results

    for i, net in enumerate(nets):
        result = run_flat_start(net, **kwargs)
        results.append({
            "case": case,
            "split": split,
            "method": "flat_start",
            "instance_id": i,
            "n_ipm_iters": result.n_iterations,
            "converged": result.converged,
            "obj_value": result.obj_value,
            "obj_gap_pct": 0.0,
            "ws_rmse": None,
            "ipopt_time_s": result.solve_time_s,
            "total_time_s": result.solve_time_s,
        })

    return results
