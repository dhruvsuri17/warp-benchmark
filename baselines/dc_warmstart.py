"""
DC-OPF warm-start baseline.

Solves the DC-OPF (linear approximation) via pandapower,
then lifts the solution to AC variables for warm-starting IPOPT.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def solve_dc_opf(net) -> dict:
    """
    Solve DC-OPF and extract warm-start variables.

    DC-OPF gives voltage angles and generator active power.
    Voltage magnitudes are set to 1 p.u. (flat), reactive power to 0.
    """
    import pandapower as pp

    try:
        pp.rundcopp(net)
    except Exception as e:
        logger.warning(f"DC-OPF failed: {e}")
        return None

    Vm = np.ones(len(net.bus))
    Va = np.radians(net.res_bus["va_degree"].values)
    Pg = net.res_gen["p_mw"].values.copy()
    Qg = np.zeros(len(net.gen))

    return {"Vm": Vm, "Va": Va, "Pg": Pg, "Qg": Qg}


def evaluate_dc_warmstart(case: str, split: str, nets=None,
                          **kwargs) -> List[dict]:
    """
    Run DC warm-start evaluation across test set.
    """
    from eval.ipopt_wrapper import run_ipopt

    results = []

    if nets is None:
        logger.warning("No pandapower nets provided, returning empty results")
        return results

    for i, net in enumerate(nets):
        x_ws = solve_dc_opf(net)
        result = run_ipopt(net, x_ws=x_ws, **kwargs)

        results.append({
            "case": case,
            "split": split,
            "method": "dc_warmstart",
            "instance_id": i,
            "n_ipm_iters": result.n_iterations,
            "converged": result.converged,
            "obj_value": result.obj_value,
            "obj_gap_pct": None,
            "ws_rmse": None,
            "ipopt_time_s": result.solve_time_s,
            "total_time_s": result.solve_time_s,
        })

    return results
