"""
IPOPT warm-start harness via pandapower.

Accepts a warm-start vector x_ws = (Vm, Va, Pg, Qg),
sets it as IPOPT's initial point, runs to convergence,
and returns iteration count + solution quality metrics.

This is the most operationally critical file in WARP.
"""

import re
import time
import io
import sys
import logging
from typing import Optional
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IPOPTResult:
    x_opt: Optional[dict]
    n_iterations: Optional[int]
    converged: bool
    obj_value: Optional[float]
    constraint_violations: Optional[float]
    solve_time_s: float


def set_warmstart(net, x_ws: dict):
    """
    Set warm-start point in pandapower net object.

    Args:
        net: pandapower network
        x_ws: dict with keys "Vm", "Va" (radians), "Pg", "Qg"
    """
    if "Vm" in x_ws:
        net.res_bus.loc[:, "vm_pu"] = x_ws["Vm"]
    if "Va" in x_ws:
        net.res_bus.loc[:, "va_degree"] = np.degrees(x_ws["Va"])
    if "Pg" in x_ws:
        net.res_gen.loc[:, "p_mw"] = x_ws["Pg"]
    if "Qg" in x_ws:
        net.res_gen.loc[:, "q_mvar"] = x_ws["Qg"]


def parse_ipopt_iterations(output: str) -> Optional[int]:
    """
    Parse IPOPT iteration count from stdout output.

    Looks for: "Number of Iterations....: N"
    """
    match = re.search(r"Number of Iterations\.*:\s+(\d+)", output)
    if match:
        return int(match.group(1))
    return None


def run_ipopt(net, x_ws: Optional[dict] = None,
              max_iter: int = 300, tol: float = 1e-6,
              linear_solver: str = "mumps") -> IPOPTResult:
    """
    Run IPOPT OPF via pandapower with optional warm-start.

    Args:
        net: pandapower network (will be modified in-place)
        x_ws: warm-start dict, or None for flat start
        max_iter: maximum IPOPT iterations
        tol: convergence tolerance
        linear_solver: "ma57" (HSL) or "mumps" (default)

    Returns:
        IPOPTResult with solution, iteration count, and diagnostics
    """
    import pandapower as pp

    if x_ws is not None:
        set_warmstart(net, x_ws)
        init = "results"
    else:
        init = "flat"

    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    t0 = time.time()
    try:
        pp.runopp(
            net,
            init=init,
            verbose=True,
            SOLVER="ipopt",
            OPF_VIOLATION=tol,
            MAX_ITERS=max_iter,
        )
        converged = True
    except Exception as e:
        logger.warning(f"IPOPT failed: {e}")
        converged = False
    finally:
        solve_time = time.time() - t0
        sys.stdout = old_stdout

    output = captured.getvalue()
    n_iter = parse_ipopt_iterations(output)

    if converged:
        x_opt = {
            "Vm": net.res_bus["vm_pu"].values.copy(),
            "Va": np.radians(net.res_bus["va_degree"].values.copy()),
            "Pg": net.res_gen["p_mw"].values.copy(),
            "Qg": net.res_gen["q_mvar"].values.copy(),
        }
        obj_value = net.res_cost if hasattr(net, "res_cost") else None
        max_violation = _compute_max_violation(net) if converged else None
    else:
        x_opt = None
        obj_value = None
        max_violation = None

    return IPOPTResult(
        x_opt=x_opt,
        n_iterations=n_iter,
        converged=converged,
        obj_value=obj_value,
        constraint_violations=max_violation,
        solve_time_s=solve_time,
    )


def _compute_max_violation(net) -> float:
    """Compute max constraint violation from solved pandapower network."""
    violations = []

    if hasattr(net, "res_bus") and "vm_pu" in net.res_bus.columns:
        vm = net.res_bus["vm_pu"].values
        if "max_vm_pu" in net.bus.columns:
            v_over = np.maximum(0, vm - net.bus["max_vm_pu"].values)
            violations.extend(v_over.tolist())
        if "min_vm_pu" in net.bus.columns:
            v_under = np.maximum(0, net.bus["min_vm_pu"].values - vm)
            violations.extend(v_under.tolist())

    return max(violations) if violations else 0.0


def run_flat_start(net, **kwargs) -> IPOPTResult:
    """Run IPOPT with flat start (V=1, theta=0, Pg=midpoint)."""
    return run_ipopt(net, x_ws=None, **kwargs)
