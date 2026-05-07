"""Direct IPOPT OPF solver via cyipopt, using pandapower's internal PPC.

Bypasses PIPS entirely. Provides proper warm-start API:
  - warm_start_init_point = yes
  - bound_mult_init_method = mu-based
  - Accepts primal x0, and optionally dual lam_g0 / lam_x0

Uses pandapower's pypower functions for objective, constraints, and Jacobians
so results are directly comparable to PIPS benchmarks.
"""
import numpy as np
import cyipopt
from numpy import inf

from pandapower.pypower.opf_costfcn import opf_costfcn
from pandapower.pypower.opf_consfcn import opf_consfcn
from pandapower.pypower.makeYbus import makeYbus
from pandapower.pypower.idx_brch import RATE_A
from numpy import flatnonzero as find


class PyPowerOPF(cyipopt.Problem):
    """Wraps pandapower's pypower OPF as a cyipopt Problem."""

    def __init__(self, om, ppopt):
        self.om = om
        self.ppopt = ppopt

        baseMVA, bus, gen, branch, gencost, Au, lbu, ubu, ppopt_, \
            N, fparm, H, Cw, z0, zl, zu, userfcn, _ = self._opf_args(om, ppopt)

        self.baseMVA = baseMVA
        self.bus = bus
        self.gen = gen
        self.branch = branch
        self.gencost = gencost

        Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
        self.Ybus = Ybus
        self.Yf = Yf
        self.Yt = Yt

        self.il = find((branch[:, RATE_A] != 0) & (branch[:, RATE_A] < 1e10))
        self.Yf_il = Yf[self.il, :]
        self.Yt_il = Yt[self.il, :]

        x0, xmin, xmax = om.getv()
        self.n = len(x0)
        self.x0_default = x0.copy()

        try:
            A, lA, uA = om.linear_constraints()
            self.A_lin = A
            self.lA = lA
            self.uA = uA
        except Exception:
            self.A_lin = None
            self.lA = None
            self.uA = None

        self._eval_initial_constraints(x0, xmin, xmax)
        self._compute_jacobian_structure(x0)

        super().__init__(
            n=self.n,
            m=self.m,
            lb=xmin,
            ub=xmax,
            cl=self.cl,
            cu=self.cu,
        )

    def _opf_args(self, om, ppopt):
        from pandapower.pypower.opf_args import opf_args
        return opf_args(om.get_ppc(), ppopt)

    def _eval_initial_constraints(self, x0, xmin, xmax):
        """Figure out constraint dimensions and bounds."""
        hn, gn, _, _ = opf_consfcn(x0, self.om, self.Ybus,
                                    self.Yf_il, self.Yt_il,
                                    self.ppopt, self.il)

        n_eq = len(gn)
        n_ineq = len(hn)
        n_lin = self.A_lin.shape[0] if self.A_lin is not None and self.A_lin.shape[0] > 0 else 0

        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.n_lin = n_lin
        self.m = n_eq + n_ineq + n_lin

        cl = np.zeros(self.m)
        cu = np.zeros(self.m)

        cl[:n_eq] = 0.0
        cu[:n_eq] = 0.0

        cl[n_eq:n_eq+n_ineq] = -inf
        cu[n_eq:n_eq+n_ineq] = 0.0

        if n_lin > 0:
            cl[n_eq+n_ineq:] = self.lA
            cu[n_eq+n_ineq:] = self.uA

        self.cl = cl
        self.cu = cu

    def objective(self, x):
        f, _ = opf_costfcn(x, self.om)
        return f

    def gradient(self, x):
        _, df = opf_costfcn(x, self.om)
        return df

    def constraints(self, x):
        hn, gn, _, _ = opf_consfcn(x, self.om, self.Ybus,
                                    self.Yf_il, self.Yt_il,
                                    self.ppopt, self.il)
        c = np.concatenate([gn, hn])
        if self.n_lin > 0 and self.A_lin is not None:
            c = np.concatenate([c, self.A_lin @ x])
        return c

    def jacobian(self, x):
        _, _, dhn, dgn = opf_consfcn(x, self.om, self.Ybus,
                                      self.Yf_il, self.Yt_il,
                                      self.ppopt, self.il)
        dgn_a = np.asarray(dgn.todense() if hasattr(dgn, 'todense') else dgn)
        dhn_a = np.asarray(dhn.todense() if hasattr(dhn, 'todense') else dhn)
        J = np.vstack([dgn_a.T, dhn_a.T])
        if self.n_lin > 0 and self.A_lin is not None:
            J = np.vstack([J, np.asarray(self.A_lin.todense() if hasattr(self.A_lin, 'todense') else self.A_lin)])
        return J.flatten()

    def _compute_jacobian_structure(self, x0):
        pass

    def jacobianstructure(self):
        return (np.repeat(np.arange(self.m), self.n),
                np.tile(np.arange(self.n), self.m))


def solve_opf_ipopt(om, ppopt, x0=None, lam_g0=None, lam_x0=None,
                    warm_start=False, mu_init=1e-1, max_iter=300,
                    print_level=0):
    """Solve OPF via IPOPT with optional warm-start.

    Args:
        om: pandapower OpfModel
        ppopt: pypower options
        x0: initial primal variables (None = flat start midpoint)
        lam_g0: initial constraint multipliers (None = zero)
        lam_x0: initial bound multipliers (None = IPOPT default)
        warm_start: if True, enable IPOPT warm-start mode
        mu_init: initial barrier parameter
        max_iter: maximum iterations
        print_level: IPOPT verbosity (5 for iteration output)

    Returns:
        dict with keys: x, obj, n_iters, converged, lam_g, lam_x
    """
    nlp = PyPowerOPF(om, ppopt)

    if x0 is None:
        _, xmin, xmax = om.getv()
        ll, uu = xmin.copy(), xmax.copy()
        ll[xmin == -inf] = -1e10
        uu[xmax == inf] = 1e10
        x0 = (ll + uu) / 2.0

    nlp.add_option("print_level", print_level)
    nlp.add_option("max_iter", max_iter)
    nlp.add_option("tol", 1e-6)
    nlp.add_option("acceptable_tol", 1e-4)
    nlp.add_option("hessian_approximation", "limited-memory")

    if warm_start:
        nlp.add_option("warm_start_init_point", "yes")
        nlp.add_option("warm_start_bound_push", 1e-8)
        nlp.add_option("warm_start_bound_frac", 1e-8)
        nlp.add_option("warm_start_mult_bound_push", 1e-8)
        nlp.add_option("warm_start_slack_bound_push", 1e-8)
        nlp.add_option("warm_start_slack_bound_frac", 1e-8)
        nlp.add_option("bound_mult_init_method", "mu-based")
        nlp.add_option("mu_init", mu_init)

    solve_kwargs = {"x": x0}
    if lam_g0 is not None:
        nlp.add_option("warm_start_init_point", "yes")
        solve_kwargs["lagrange"] = lam_g0
    if lam_x0 is not None:
        zl = np.maximum(lam_x0, 0)
        zu = np.maximum(-lam_x0, 0)
        solve_kwargs["zl"] = zl
        solve_kwargs["zu"] = zu

    x_sol, info = nlp.solve(**solve_kwargs)

    n_iters = -1
    for key in ["iter_count", "niter", "num_iters"]:
        if key in info:
            n_iters = info[key]; break
    if n_iters == -1 and "status_msg" in info:
        import re
        m = re.search(r"Number of Iterations.*?:\s*(\d+)", str(info))
        if m:
            n_iters = int(m.group(1))

    return {
        "x": x_sol,
        "obj": info.get("obj_val", float("nan")),
        "n_iters": n_iters,
        "converged": info.get("status", -1) == 0,
        "status_msg": info.get("status_msg", ""),
        "status": info.get("status", -1),
        "lam_g": info.get("mult_g", None),
        "lam_x_lower": info.get("mult_x_L", None),
        "lam_x_upper": info.get("mult_x_U", None),
        "info": info,
    }


def build_om_from_net(net, init="flat"):
    """Build pypower OpfModel from pandapower network."""
    from pandapower.converter.pypower.to_ppc import _pd2ppc
    from pandapower.auxiliary import _init_runopp_options
    from pandapower.pypower.opf_setup import opf_setup
    from pandapower.pypower.ppoption import ppoption

    _init_runopp_options(net, calculate_voltage_angles=True, check_connectivity=True,
                         switch_rx_ratio=2, delta=1e-10, init=init, numba=False,
                         trafo3w_losses="hv")

    ppc, ppci = _pd2ppc(net)
    ppopt = ppoption(VERBOSE=0, OPF_ALG=560, INIT=init)
    om = opf_setup(ppci, ppopt)
    om.build_cost_params()
    return om, ppopt
