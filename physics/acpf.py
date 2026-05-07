"""
Differentiable AC power flow residuals.

Used in two places:
1. Auxiliary training loss on the Tweedie x0_hat estimate
2. Inference-time scoring function for multi-sample selection

MUST be fully differentiable (PyTorch ops only, no pandapower at train time).

OPFDataset solution ordering:
    bus.y  = [Va, Vm]
    gen.y  = [Pg, Qg]
"""

import torch
import torch.nn.functional as F


def ac_power_balance(Vm, Va, P_inj, Q_inj, G, B):
    """
    Compute AC power balance residuals.

    Args:
        Vm:    [n_bus] voltage magnitudes (per-unit)
        Va:    [n_bus] voltage angles (radians)
        P_inj: [n_bus] net active power injection at each bus (gen - load)
        Q_inj: [n_bus] net reactive power injection at each bus (gen - load)
        G:     [n_bus, n_bus] conductance matrix
        B:     [n_bus, n_bus] susceptance matrix

    Returns:
        dP: [n_bus] active power mismatch
        dQ: [n_bus] reactive power mismatch
    """
    Va_diff = Va.unsqueeze(1) - Va.unsqueeze(0)  # [n, n]
    Vm_outer = Vm.unsqueeze(1) * Vm.unsqueeze(0)  # [n, n]

    cos_diff = torch.cos(Va_diff)
    sin_diff = torch.sin(Va_diff)

    P_calc = (Vm_outer * (G * cos_diff + B * sin_diff)).sum(dim=1)
    Q_calc = (Vm_outer * (G * sin_diff - B * cos_diff)).sum(dim=1)

    dP = P_calc - P_inj
    dQ = Q_calc - Q_inj

    return dP, dQ


def compute_injections(data):
    """
    Compute net power injection at each bus from HeteroData.

    P_inj[bus] = sum(Pg at bus) - sum(Pd at bus) + Gs[bus] * Vm[bus]^2
    Q_inj[bus] = sum(Qg at bus) - sum(Qd at bus) - Bs[bus] * Vm[bus]^2

    Note: Shunt contributions are already in Y-bus diagonal,
    so we DON'T add them here — they're captured by the P_calc/Q_calc
    power flow equations via the G and B matrices.

    Returns:
        P_inj: [n_bus]
        Q_inj: [n_bus]
    """
    n_bus = data["bus"].x.shape[0]
    P_inj = torch.zeros(n_bus)
    Q_inj = torch.zeros(n_bus)

    Pg = data["generator"].y[:, 0]
    Qg = data["generator"].y[:, 1]
    gen_bus = data["generator", "generator_link", "bus"].edge_index[1]

    P_inj.scatter_add_(0, gen_bus, Pg)
    Q_inj.scatter_add_(0, gen_bus, Qg)

    Pd = data["load"].x[:, 0]
    Qd = data["load"].x[:, 1]
    load_bus = data["load", "load_link", "bus"].edge_index[1]

    P_inj.scatter_add_(0, load_bus, -Pd)
    Q_inj.scatter_add_(0, load_bus, -Qd)

    return P_inj, Q_inj


def compute_residuals(data, G, B):
    """
    Compute AC power flow residuals for a single HeteroData instance
    using its ground-truth solution.

    Returns:
        dP: [n_bus] active power mismatch
        dQ: [n_bus] reactive power mismatch
    """
    Va = data["bus"].y[:, 0]
    Vm = data["bus"].y[:, 1]

    P_inj, Q_inj = compute_injections(data)

    return ac_power_balance(Vm, Va, P_inj, Q_inj, G, B)


def compute_line_flows(Vm, Va, edge_index, G, B):
    """
    Compute apparent power flow magnitude on each line.

    Returns:
        S_flow: [n_edges] apparent power
    """
    src, dst = edge_index[0], edge_index[1]
    Vm_i, Vm_j = Vm[src], Vm[dst]
    Va_diff = Va[src] - Va[dst]

    g_ij = G[src, dst]
    b_ij = B[src, dst]

    P_flow = Vm_i ** 2 * g_ij - Vm_i * Vm_j * (
        g_ij * torch.cos(Va_diff) + b_ij * torch.sin(Va_diff)
    )
    Q_flow = -Vm_i ** 2 * b_ij - Vm_i * Vm_j * (
        g_ij * torch.sin(Va_diff) - b_ij * torch.cos(Va_diff)
    )

    return torch.sqrt(P_flow ** 2 + Q_flow ** 2 + 1e-12)


def physics_loss(Vm, Va, Pg, Qg, data, G, B,
                 gen_bus_idx, load_bus_idx,
                 Pd, Qd, Vm_min, Vm_max, S_max,
                 edge_index,
                 lambda_V=1.0, lambda_S=1.0):
    """
    Combined physics loss for training: power balance + voltage + thermal.

    All inputs are differentiable tensors (from Tweedie estimate or prediction).
    """
    n_bus = Vm.shape[0]
    P_inj = torch.zeros(n_bus, device=Vm.device)
    Q_inj = torch.zeros(n_bus, device=Vm.device)

    P_inj.scatter_add_(0, gen_bus_idx, Pg)
    Q_inj.scatter_add_(0, gen_bus_idx, Qg)
    P_inj.scatter_add_(0, load_bus_idx, -Pd)
    Q_inj.scatter_add_(0, load_bus_idx, -Qd)

    dP, dQ = ac_power_balance(Vm, Va, P_inj, Q_inj, G, B)
    L_balance = (dP ** 2 + dQ ** 2).sum()

    L_voltage = (
        F.relu(Vm - Vm_max) ** 2 + F.relu(Vm_min - Vm) ** 2
    ).sum()

    S_flow = compute_line_flows(Vm, Va, edge_index, G, B)
    L_thermal = F.relu(S_flow - S_max).pow(2).sum()

    return L_balance + lambda_V * L_voltage + lambda_S * L_thermal


def residual_score(Vm, Va, Pg, Qg, gen_bus_idx, load_bus_idx,
                   Pd, Qd, G, B):
    """
    Cheap scoring function for multi-sample selection at inference.
    Returns scalar: sum of squared power balance residuals. Lower = better.
    """
    n_bus = Vm.shape[0]
    P_inj = torch.zeros(n_bus, device=Vm.device)
    Q_inj = torch.zeros(n_bus, device=Vm.device)

    P_inj.scatter_add_(0, gen_bus_idx, Pg)
    Q_inj.scatter_add_(0, gen_bus_idx, Qg)
    P_inj.scatter_add_(0, load_bus_idx, -Pd)
    Q_inj.scatter_add_(0, load_bus_idx, -Qd)

    dP, dQ = ac_power_balance(Vm, Va, P_inj, Q_inj, G, B)
    return (dP ** 2 + dQ ** 2).sum()
