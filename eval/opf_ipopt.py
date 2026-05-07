"""OPF via cyipopt — following IPM-LSTM's pattern.

Key differences from our ipopt_opf_v2.py:
- intermediate() callback for iteration counting + mu tracking
- IPM-LSTM's exact warm-start settings (bound_push=1e-20, monotone mu)
- Extracts full (x*, lam_g*, zl*, zu*, mu*) from solves for dual labels
- Uses pypower's opf_consfcn/opf_costfcn/opf_hessfcn for AC-OPF
"""
import numpy as np
import cyipopt
from numpy import inf, flatnonzero as find
from scipy import sparse

import scipy.sparse
if not hasattr(scipy.sparse.csr_matrix, 'H'):
    scipy.sparse.csr_matrix.H = property(lambda self: self.conjugate().T)
    scipy.sparse.csc_matrix.H = property(lambda self: self.conjugate().T)
    scipy.sparse.coo_matrix.H = property(lambda self: self.conjugate().T)

from pandapower.pypower.opf_costfcn import opf_costfcn
from pandapower.pypower.opf_consfcn import opf_consfcn
from pandapower.pypower.opf_hessfcn import opf_hessfcn
from pandapower.pypower.makeYbus import makeYbus
from pandapower.pypower.idx_brch import RATE_A


class ACOPF(cyipopt.Problem):
    """AC-OPF as cyipopt Problem with intermediate callback."""

    def __init__(self, om, ppopt):
        self.om = om
        self.ppopt = ppopt

        ppc = om.get_ppc()
        self.baseMVA = ppc["baseMVA"]
        self.bus = ppc["bus"]
        self.gen = ppc["gen"]
        self.branch = ppc["branch"]

        Ybus, Yf, Yt = makeYbus(self.baseMVA, self.bus, self.branch)
        self.Ybus = Ybus
        self.Yf = Yf
        self.Yt = Yt

        self.il = find((self.branch[:, RATE_A] != 0) & (self.branch[:, RATE_A] < 1e10))
        self.Yf_il = Yf[self.il, :]
        self.Yt_il = Yt[self.il, :]

        x0, xmin, xmax = om.getv()
        self.n = len(x0)

        hn, gn, dhn, dgn = opf_consfcn(x0, om, Ybus, self.Yf_il, self.Yt_il, ppopt, self.il)
        self.n_eq = len(gn)
        self.n_ineq = len(hn)
        self.m = self.n_eq + self.n_ineq

        cl = np.zeros(self.m)
        cu = np.zeros(self.m)
        cl[:self.n_eq] = 0.0; cu[:self.n_eq] = 0.0
        cl[self.n_eq:] = -inf; cu[self.n_eq:] = 0.0

        # Sparsity from union of sample points
        patterns = set()
        for x_s in [x0, (xmin + xmax) / 2]:
            x_safe = np.clip(x_s, np.where(xmin > -1e10, xmin, -1e4) + 1e-8,
                             np.where(xmax < 1e10, xmax, 1e4) - 1e-8)
            try:
                _, _, dh, dg = opf_consfcn(x_safe, om, Ybus, self.Yf_il, self.Yt_il, ppopt, self.il)
                dg_t = dg.T.tocoo() if sparse.issparse(dg) else sparse.coo_matrix(np.asarray(dg).T)
                dh_t = dh.T.tocoo() if sparse.issparse(dh) else sparse.coo_matrix(np.asarray(dh).T)
                J = sparse.vstack([dg_t, dh_t]).tocoo()
                patterns |= set(zip(J.row.tolist(), J.col.tolist()))
            except Exception:
                pass

        all_entries = sorted(patterns)
        self._jac_rows = np.array([e[0] for e in all_entries], dtype=np.intc)
        self._jac_cols = np.array([e[1] for e in all_entries], dtype=np.intc)

        # Hessian sparsity
        lmbda = {"eqnonlin": np.ones(self.n_eq), "ineqnonlin": np.ones(self.n_ineq)}
        x_safe = np.clip(x0, np.where(xmin > -1e10, xmin, -1e4) + 1e-6,
                         np.where(xmax < 1e10, xmax, 1e4) - 1e-6)
        H = opf_hessfcn(x_safe, lmbda, om, Ybus, self.Yf_il, self.Yt_il, ppopt, self.il)
        H_tril = sparse.tril(H).tocoo() if sparse.issparse(H) else sparse.tril(sparse.csr_matrix(H)).tocoo()
        self._hess_rows = H_tril.row.astype(np.intc)
        self._hess_cols = H_tril.col.astype(np.intc)

        # Tracking
        self.iter_objectives = []
        self.iter_mus = []

        super().__init__(
            n=self.n, m=self.m,
            lb=xmin, ub=xmax, cl=cl, cu=cu,
        )

    def objective(self, x):
        f, _ = opf_costfcn(x, self.om)
        return f

    def gradient(self, x):
        _, df = opf_costfcn(x, self.om)
        return df

    def constraints(self, x):
        hn, gn, _, _ = opf_consfcn(x, self.om, self.Ybus,
                                    self.Yf_il, self.Yt_il, self.ppopt, self.il)
        return np.concatenate([gn, hn])

    def jacobian(self, x):
        _, _, dhn, dgn = opf_consfcn(x, self.om, self.Ybus,
                                      self.Yf_il, self.Yt_il, self.ppopt, self.il)
        dgn_t = dgn.T.tocsc() if sparse.issparse(dgn) else sparse.csc_matrix(np.asarray(dgn).T)
        dhn_t = dhn.T.tocsc() if sparse.issparse(dhn) else sparse.csc_matrix(np.asarray(dhn).T)
        J = sparse.vstack([dgn_t, dhn_t]).tocsc()
        return np.array(J[self._jac_rows, self._jac_cols]).flatten()

    def jacobianstructure(self):
        return (self._jac_rows, self._jac_cols)

    def hessian(self, x, lagrange, obj_factor):
        lmbda = {
            "eqnonlin": lagrange[:self.n_eq],
            "ineqnonlin": lagrange[self.n_eq:],
        }
        H = opf_hessfcn(x, lmbda, self.om, self.Ybus,
                        self.Yf_il, self.Yt_il, self.ppopt, self.il,
                        cost_mult=obj_factor)
        H_tril = sparse.tril(H).tocsc() if sparse.issparse(H) else sparse.tril(sparse.csr_matrix(H)).tocsc()
        return np.array(H_tril[self._hess_rows, self._hess_cols]).flatten()

    def hessianstructure(self):
        return (self._hess_rows, self._hess_cols)

    def intermediate(self, alg_mod, iter_count, obj_value,
                     inf_pr, inf_du, mu, d_norm, regularization_size,
                     alpha_du, alpha_pr, ls_trials):
        self.iter_objectives.append(obj_value)
        self.iter_mus.append(mu)


def build_om(net, init="flat"):
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


def solve_opf(om, ppopt, x0=None, lam_g0=None, zl0=None, zu0=None,
              mu_init=None, warm_start=False, max_iter=200, tol=1e-6):
    """Solve OPF via IPOPT with IPM-LSTM-style warm-start.

    Returns dict with x, lam_g, zl, zu, mu_final, n_iters, obj, converged.
    """
    nlp = ACOPF(om, ppopt)

    if x0 is None:
        x0_v, xmin, xmax = om.getv()
        ll, uu = xmin.copy(), xmax.copy()
        ll[xmin == -inf] = -1e10
        uu[xmax == inf] = 1e10
        x0 = (ll + uu) / 2.0

    nlp.add_option("max_iter", max_iter)
    nlp.add_option("tol", tol)
    nlp.add_option("acceptable_tol", 1e-4)
    nlp.add_option("acceptable_iter", 10)
    nlp.add_option("print_level", 0)

    if warm_start and mu_init is not None:
        nlp.add_option("warm_start_init_point", "yes")
        nlp.add_option("warm_start_bound_push", 1e-20)
        nlp.add_option("warm_start_bound_frac", 1e-20)
        nlp.add_option("warm_start_slack_bound_push", 1e-20)
        nlp.add_option("warm_start_slack_bound_frac", 1e-20)
        nlp.add_option("warm_start_mult_bound_push", 1e-20)
        nlp.add_option("mu_strategy", "monotone")
        nlp.add_option("mu_init", float(mu_init))
    elif warm_start:
        nlp.add_option("warm_start_init_point", "yes")
        nlp.add_option("warm_start_bound_push", 1e-8)
        nlp.add_option("warm_start_bound_frac", 1e-8)
        nlp.add_option("warm_start_slack_bound_push", 1e-8)
        nlp.add_option("warm_start_slack_bound_frac", 1e-8)
        nlp.add_option("warm_start_mult_bound_push", 1e-8)
        nlp.add_option("mu_strategy", "adaptive")

    solve_kw = {"x": x0}
    if lam_g0 is not None:
        solve_kw["lagrange"] = lam_g0
    if zl0 is not None:
        solve_kw["zl"] = zl0
    if zu0 is not None:
        solve_kw["zu"] = zu0

    nlp.add_option("sb", "yes")  # suppress IPOPT banner

    x_sol, info = nlp.solve(**solve_kw)

    n_iters = len(nlp.iter_objectives)
    mu_final = nlp.iter_mus[-1] if nlp.iter_mus else None

    return {
        "x": x_sol,
        "obj": info.get("obj_val", float("nan")),
        "n_iters": n_iters,
        "converged": info.get("status", -1) == 0,
        "status": info.get("status", -1),
        "lam_g": info.get("mult_g", None),
        "zl": info.get("mult_x_L", None),
        "zu": info.get("mult_x_U", None),
        "mu_final": mu_final,
        "mu_history": nlp.iter_mus,
    }
