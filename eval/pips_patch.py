"""Pandapower PIPS solver monkey-patch utilities (baseline comparisons).

Some experiments patch `pandapower.pypower.opf_execute.pipsopf_solver` to adjust
INIT handling. Keep patches scoped and restore the original reference afterwards.
"""

from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Callable


@contextlib.contextmanager
def patched_pips_solver(patch_fn: Callable):
    """Temporarily replace pandapower's PIPS OPF solver.

    `patch_fn` receives `(original_solver, om, ppopt, out_opt)` and must call the
    original or delegate as needed.
    """
    import pandapower.pypower.opf_execute as _opf_exec

    _orig = _opf_exec.pipsopf_solver

    def _wrapped(om, ppopt, out_opt=None):
        return patch_fn(_orig, om, ppopt, out_opt)

    _opf_exec.pipsopf_solver = _wrapped
    try:
        yield
    finally:
        _opf_exec.pipsopf_solver = _orig


def run_ppopp_capture_stdout(net, init: str):
    """Run `pp.runopp` and return (converged: bool, stdout: str)."""
    import pandapower as pp

    old = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        pp.runopp(net, init=init, verbose=2, numba=False)
        conv = bool(net.OPF_converged)
    except Exception:
        conv = False
    finally:
        sys.stdout = old
    return conv, buf.getvalue()


def strip_init_results_patch(orig_solver, om, ppopt, out_opt=None):
    """Force INIT away from 'results' when warm-start not desired (legacy benchmarks)."""
    if ppopt.get("INIT") == "results":
        ppopt = dict(ppopt)
        ppopt["INIT"] = "pf"
    return orig_solver(om, ppopt, out_opt)
