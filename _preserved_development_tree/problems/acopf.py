"""AC-OPF problem class following IPM-LSTM's interface.

Implements the same methods as IPM-LSTM's QP class:
- obj_fn, obj_grad: objective and gradient
- ineq_resid, eq_resid: constraint residuals
- cal_kkt: KKT system (Jacobian J, residual F, centrality mu)
- F0: KKT residual without centering (for convergence check)
- opt_solve: IPOPT solver with warm-start

All methods operate on batched tensors [batch_size, dim, 1] for
compatibility with IPM-LSTM's training loop.

Uses precomputed per-instance data (Q_cost, G_bus, B_bus, etc.)
stored as tensors for GPU-accelerated KKT computation.
"""
import torch
import torch.nn as nn
import numpy as np
import cyipopt
from eval.opf_ipopt import ACOPF as ACOPF_IPOPT, build_om, solve_opf
import pandapower.networks as pn
from torch_geometric.datasets import OPFDataset
from scipy import sparse


class ACOPFProblem:
    """AC-OPF problem in IPM-LSTM format.

    For case118: num_var=344, num_eq=236, num_ineq=372.
    Variables: [Va(118), Vm(118), Pg(54), Qg(54)]
    Eq constraints: [Pmis(118), Qmis(118)]
    Ineq constraints: [Sf(186), St(186)]
    Bounds: Vm has [0.94, 1.06], Pg/Qg have generator limits
    """

    def __init__(self, case_name="pglib_opf_case118_ieee", split="train",
                 num_groups=1, device="cuda:0", max_instances=None):
        self.device = device
        self.case_name = case_name

        ds = OPFDataset(root="data", case_name=case_name, split=split,
                        num_groups=num_groups)
        n = len(ds) if max_instances is None else min(len(ds), max_instances)
        self.data_size = n

        # Build one reference network to get dimensions
        net = pn.case118()
        om, ppopt = build_om(net)
        vv = om.get_idx()[0]

        self.num_var = vv['iN']['Qg']  # total variables
        self.n_bus = vv['N']['Va']
        self.n_gen = vv['N']['Pg']

        x0, xmin, xmax = om.getv()
        self.num_eq = 236    # Pmis + Qmis = 2 * n_bus
        self.num_ineq = 372  # Sf + St = 2 * n_constrained_lines

        self.num_lb = self.num_var  # all variables have bounds
        self.num_ub = self.num_var

        # Store bounds as tensors
        lb = np.where(xmin > -1e10, xmin, -1e4)
        ub = np.where(xmax < 1e10, xmax, 1e4)
        self.lb = torch.tensor(lb, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        self.ub = torch.tensor(ub, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        # Store dataset graphs for access during training
        self.graphs = [ds[i] for i in range(n)]

        # Precompute IPOPT solutions for warm-start benchmarking
        self._om_cache = {}

    def get_graph(self, idx):
        """Get the PyG HeteroData graph for instance idx."""
        return self.graphs[idx]

    def get_batch_graphs(self, indices):
        """Get a list of graphs for batch indices."""
        return [self.graphs[i] for i in indices]

    def build_ipopt_model(self, idx):
        """Build IPOPT model for instance idx."""
        if idx in self._om_cache:
            return self._om_cache[idx]

        data = self.graphs[idx]
        net = pn.case118()
        Pd = data["load"].x[:, 0].numpy() * 100
        Qd = data["load"].x[:, 1].numpy() * 100
        for i in range(min(len(net.load), len(Pd))):
            net.load.at[i, "p_mw"] = Pd[i]
            net.load.at[i, "q_mvar"] = Qd[i]

        om, ppopt = build_om(net)
        self._om_cache[idx] = (om, ppopt)
        return om, ppopt

    def opt_solve(self, indices=None, initial_y=None, init_mu=None,
                  init_g=None, init_zl=None, init_zu=None):
        """Solve via IPOPT with optional warm-start (IPM-LSTM interface)."""
        if indices is None:
            indices = range(self.data_size)

        results = []
        total_iters = 0

        for i, idx in enumerate(indices):
            om, ppopt = self.build_ipopt_model(idx)

            x0 = None
            lam_g0 = None
            zl0 = None
            zu0 = None
            mu = None
            ws = False

            if initial_y is not None:
                x0 = initial_y[i].cpu().numpy().flatten()
                x0_v, xmin, xmax = om.getv()
                x0 = np.clip(x0, xmin + 1e-10, xmax - 1e-10)
                ws = True

            if init_mu is not None:
                mu = init_mu[i].cpu().item()

            if init_g is not None:
                lam_g0 = init_g[i].cpu().numpy().flatten()

            if init_zl is not None:
                zl0 = init_zl[i].cpu().numpy().flatten()

            if init_zu is not None:
                zu0 = init_zu[i].cpu().numpy().flatten()

            r = solve_opf(om, ppopt, x0=x0, lam_g0=lam_g0, zl0=zl0, zu0=zu0,
                          warm_start=ws, mu_init=mu)

            results.append(r)
            total_iters += r["n_iters"]

        mean_iters = total_iters / len(indices)
        return results, mean_iters
