"""
AC residual scoring for multi-sample warm-start selection.

Given K samples, score each by power balance residual and pick the best.
"""

import torch

from physics.admittance import build_ybus_from_heterodata
from physics.acpf import ac_power_balance


@torch.no_grad()
def score_and_select(bus_samples, gen_samples, data, G=None, B=None):
    """
    Score K warm-start candidates by AC power balance residual and select best.

    Args:
        bus_samples: [K, n_bus, 2] (Va, Vm) candidates
        gen_samples: [K, n_gen, 2] (Pg, Qg) candidates
        data: HeteroData graph
        G, B: precomputed Y-bus matrices (optional, computed if None)

    Returns:
        best_bus: [n_bus, 2] best (Va, Vm)
        best_gen: [n_gen, 2] best (Pg, Qg)
        best_idx: int index of best sample
        scores: [K] residual scores (lower = better)
    """
    if G is None or B is None:
        G, B = build_ybus_from_heterodata(data)

    K = bus_samples.shape[0]
    scores = torch.zeros(K, device=bus_samples.device)

    gen_bus = data["generator", "generator_link", "bus"].edge_index[1]
    load_bus = data["load", "load_link", "bus"].edge_index[1]
    Pd = data["load"].x[:, 0]
    Qd = data["load"].x[:, 1]
    n_bus = bus_samples.shape[1]

    for k in range(K):
        Va = bus_samples[k, :, 0]
        Vm = bus_samples[k, :, 1]
        Pg = gen_samples[k, :, 0]
        Qg = gen_samples[k, :, 1]

        P_inj = torch.zeros(n_bus, device=Va.device)
        Q_inj = torch.zeros(n_bus, device=Va.device)
        P_inj.scatter_add_(0, gen_bus, Pg)
        Q_inj.scatter_add_(0, gen_bus, Qg)
        P_inj.scatter_add_(0, load_bus, -Pd)
        Q_inj.scatter_add_(0, load_bus, -Qd)

        dP, dQ = ac_power_balance(Vm, Va, P_inj, Q_inj, G, B)
        scores[k] = (dP ** 2 + dQ ** 2).sum()

    best_idx = scores.argmin().item()
    return bus_samples[best_idx], gen_samples[best_idx], best_idx, scores
