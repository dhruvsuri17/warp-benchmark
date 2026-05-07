"""KKT-based dual completion: compute (lambda, z) from primal prediction x_hat.

Given a primal prediction x̂ close to x*, analytically compute consistent
dual variables using the KKT stationarity condition:
    ∇f(x̂) + J_eq(x̂)ᵀ λ + J_ineq(x̂)ᵀ μ - z = 0

Uses PyTorch autograd for the Jacobian computation (leveraging the existing
differentiable AC power flow implementation).
"""
import numpy as np
import torch
from scipy.sparse.linalg import lsqr


def compute_duals_from_primal(x_hat_dict, data, net, G, B, device="cpu"):
    """Compute dual variables from primal prediction using KKT stationarity.

    Args:
        x_hat_dict: dict with keys 'Vm', 'Va', 'Pg', 'Qg' as numpy arrays
        data: PyG HeteroData graph
        net: pandapower network (for bounds and cost coefficients)
        G, B: admittance matrices (numpy)

    Returns:
        lam_eq: equality constraint multipliers [2*n_bus]
        z_lower: lower bound multipliers [n_vars]
        z_upper: upper bound multipliers [n_vars]
        mu: centrality measure
    """
    Vm = torch.tensor(x_hat_dict["Vm"], dtype=torch.float64, requires_grad=True)
    Va = torch.tensor(x_hat_dict["Va"], dtype=torch.float64, requires_grad=True)
    Pg = torch.tensor(x_hat_dict["Pg"], dtype=torch.float64, requires_grad=True)
    Qg = torch.tensor(x_hat_dict["Qg"], dtype=torch.float64, requires_grad=True)

    n_bus = len(Vm)
    n_gen = len(Pg)
    G_t = torch.tensor(G, dtype=torch.float64)
    B_t = torch.tensor(B, dtype=torch.float64)

    # Power balance residuals g(x) = [dP; dQ]
    Vd = Va.unsqueeze(1) - Va.unsqueeze(0)
    Vo = Vm.unsqueeze(1) * Vm.unsqueeze(0)
    Pc = (Vo * (G_t * torch.cos(Vd) + B_t * torch.sin(Vd))).sum(1)
    Qc = (Vo * (G_t * torch.sin(Vd) - B_t * torch.cos(Vd))).sum(1)

    P_inj = torch.zeros(n_bus, dtype=torch.float64)
    Q_inj = torch.zeros(n_bus, dtype=torch.float64)

    gb = data["generator", "generator_link", "bus"].edge_index[1].numpy()
    for g in range(n_gen):
        P_inj[gb[g]] += Pg[g]
        Q_inj[gb[g]] += Qg[g]

    lb_idx = data["load", "load_link", "bus"].edge_index[1].numpy()
    Pd = data["load"].x[:, 0].numpy()
    Qd = data["load"].x[:, 1].numpy()
    for l in range(len(Pd)):
        P_inj[lb_idx[l]] -= Pd[l]
        Q_inj[lb_idx[l]] -= Qd[l]

    dP = Pc - P_inj
    dQ = Qc - Q_inj
    g_eq = torch.cat([dP, dQ])  # [2*n_bus]

    # Objective gradient: f = sum_g (c2*Pg^2 + c1*Pg)
    grad_f = torch.zeros(2 * n_bus + 2 * n_gen, dtype=torch.float64)
    if hasattr(net, 'poly_cost') and len(net.poly_cost) > 0:
        baseMVA = net.sn_mva if hasattr(net, 'sn_mva') else 100.0
        c2 = net.poly_cost["cp2_eur_per_mw2"].values
        c1 = net.poly_cost["cp1_eur_per_mw"].values
        for g in range(min(n_gen, len(c2))):
            grad_f[2*n_bus + g] = (2 * c2[g] * Pg[g].detach().item() + c1[g]) / baseMVA
    else:
        for g in range(n_gen):
            grad_f[2*n_bus + g] = 2 * Pg[g].detach().item()

    # Jacobian of g_eq w.r.t. x = [Vm, Va, Pg, Qg] via autograd
    x_all = torch.cat([Vm, Va, Pg, Qg])
    n_vars = len(x_all)
    n_eq = len(g_eq)

    J_rows = []
    for i in range(n_eq):
        if x_all.grad is not None:
            x_all.grad = None
        Vm.grad = None; Va.grad = None; Pg.grad = None; Qg.grad = None
        g_eq[i].backward(retain_graph=True)
        row = torch.zeros(n_vars, dtype=torch.float64)
        if Vm.grad is not None:
            row[:n_bus] = Vm.grad.double()
        if Va.grad is not None:
            row[n_bus:2*n_bus] = Va.grad.double()
        if Pg.grad is not None:
            row[2*n_bus:2*n_bus+n_gen] = Pg.grad.double()
        if Qg.grad is not None:
            row[2*n_bus+n_gen:] = Qg.grad.double()
        J_rows.append(row.numpy())

    J_eq = np.array(J_rows)  # [2*n_bus, n_vars]

    # Solve: J_eq^T λ ≈ -grad_f  (KKT stationarity, ignoring bounds)
    result = lsqr(J_eq.T, -grad_f.numpy())
    lam_eq = result[0]

    # Compute bound multipliers from KKT residual
    residual = grad_f.numpy() + J_eq.T @ lam_eq
    z_lower = np.maximum(0, -residual)
    z_upper = np.maximum(0, residual)

    # Centrality estimate
    from inference.warmstart import get_variable_bounds
    lb, ub = get_variable_bounds(net)
    x_flat = np.concatenate([
        x_hat_dict["Vm"],
        np.degrees(x_hat_dict["Va"]) if x_hat_dict["Va"].max() < 10 else x_hat_dict["Va"],
        x_hat_dict["Pg"],
        x_hat_dict["Qg"],
    ])
    s_lower = np.maximum(x_flat - lb, 1e-10)
    s_upper = np.maximum(ub - x_flat, 1e-10)
    comp_lower = s_lower * z_lower
    comp_upper = s_upper * z_upper
    mu = (comp_lower.sum() + comp_upper.sum()) / (2 * n_vars)

    return lam_eq, z_lower, z_upper, mu
