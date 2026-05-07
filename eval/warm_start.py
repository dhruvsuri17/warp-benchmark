"""Warm-start strategies for IPOPT/PIPS injection."""
import numpy as np


def get_variable_bounds(net):
    """Extract variable bounds [Vm, Va, Pg, Qg] from pandapower network."""
    n_bus = len(net.bus)
    n_gen = len(net.gen)

    lb = np.zeros(2 * n_bus + 2 * n_gen)
    ub = np.zeros(2 * n_bus + 2 * n_gen)

    # Vm bounds
    lb[:n_bus] = net.bus["min_vm_pu"].values
    ub[:n_bus] = net.bus["max_vm_pu"].values

    # Va bounds (typically unbounded, use wide range)
    lb[n_bus:2*n_bus] = -np.pi
    ub[n_bus:2*n_bus] = np.pi

    # Pg bounds (MW, convert from net which is in MW)
    lb[2*n_bus:2*n_bus+n_gen] = net.gen["min_p_mw"].values
    ub[2*n_bus:2*n_bus+n_gen] = net.gen["max_p_mw"].values

    # Qg bounds
    lb[2*n_bus+n_gen:] = net.gen["min_q_mvar"].values
    ub[2*n_bus+n_gen:] = net.gen["max_q_mvar"].values

    return lb, ub


def pack_prediction(bus_pred, gen_pred, net):
    """Pack model predictions into canonical variable vector [Vm, Va, Pg, Qg].

    bus_pred: [n_bus, 2] — (Va, Vm) in per-unit/radians
    gen_pred: [n_gen, 2] — (Pg, Qg) in per-unit
    """
    Vm = bus_pred[:, 1].cpu().numpy()
    Va = bus_pred[:, 0].cpu().numpy()
    Pg = gen_pred[:, 0].cpu().numpy() * 100  # pu → MW (baseMVA=100)
    Qg = gen_pred[:, 1].cpu().numpy() * 100

    n_bus = len(net.bus)
    n_gen = len(net.gen)
    x = np.zeros(2 * n_bus + 2 * n_gen)
    x[:n_bus] = Vm[:n_bus]
    x[n_bus:2*n_bus] = np.degrees(Va[:n_bus])  # PIPS uses degrees internally
    x[2*n_bus:2*n_bus+min(n_gen, len(Pg))] = Pg[:n_gen]
    x[2*n_bus+n_gen:2*n_bus+n_gen+min(n_gen, len(Qg))] = Qg[:n_gen]
    return x


def centred_warmstart(x_hat, net, alpha=0.5):
    """Blend GNN prediction with IPM midpoint to restore centrality.

    alpha=0.0 → pure midpoint (flat start)
    alpha=1.0 → pure GNN prediction
    """
    lb, ub = get_variable_bounds(net)
    x_mid = (lb + ub) / 2.0
    x_blend = alpha * x_hat + (1 - alpha) * x_mid
    x_blend = np.clip(x_blend, lb + 1e-6 * (ub - lb), ub - 1e-6 * (ub - lb))
    return x_blend


def inject_warmstart(net, x_ws):
    """Inject variable vector [Vm, Va, Pg, Qg] into pandapower res_* tables."""
    n_bus = len(net.bus)
    n_gen = len(net.gen)

    Vm = x_ws[:n_bus]
    Va = x_ws[n_bus:2*n_bus]
    Pg = x_ws[2*n_bus:2*n_bus+n_gen]
    Qg = x_ws[2*n_bus+n_gen:]

    net.res_bus["vm_pu"] = Vm
    net.res_bus["va_degree"] = Va
    net.res_bus["p_mw"] = 0.0
    net.res_bus["q_mvar"] = 0.0
    for i in range(n_gen):
        net.res_gen.at[i, "p_mw"] = Pg[i]
        net.res_gen.at[i, "q_mvar"] = Qg[i]
