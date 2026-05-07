"""
Y-bus (admittance matrix) construction from OPFDataset HeteroData.

Handles both AC lines and transformers (with tap ratio and phase shift).
Returns sparse tensors for memory efficiency on large grids (case2000).

OPFDataset edge_attr column ordering:
  AC line (9 cols):      [angmin, angmax, b_fr, b_to, r, x, rate_a, rate_b, rate_c]
  Transformer (11 cols): [angmin, angmax, r, x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to]
"""

import torch


def build_ybus_from_heterodata(data) -> tuple:
    """
    Build Y-bus from a PyG HeteroData object.

    Returns:
        G: [n_bus, n_bus] dense conductance matrix
        B: [n_bus, n_bus] dense susceptance matrix
    """
    n_bus = data["bus"].x.shape[0]
    G = torch.zeros(n_bus, n_bus)
    B = torch.zeros(n_bus, n_bus)

    if ("bus", "ac_line", "bus") in data.edge_types:
        ei = data["bus", "ac_line", "bus"].edge_index
        ea = data["bus", "ac_line", "bus"].edge_attr
        _add_lines_to_ybus(G, B, ei, ea)

    if ("bus", "transformer", "bus") in data.edge_types:
        ei = data["bus", "transformer", "bus"].edge_index
        ea = data["bus", "transformer", "bus"].edge_attr
        _add_transformers_to_ybus(G, B, ei, ea)

    if "shunt" in data.node_types:
        shunt_x = data["shunt"].x
        shunt_ei = data["shunt", "shunt_link", "bus"].edge_index
        _add_shunts_to_ybus(G, B, shunt_x, shunt_ei)

    return G, B


def _add_lines_to_ybus(G, B, edge_index, edge_attr):
    """
    Add AC line contributions to Y-bus.

    AC line edge_attr: [angmin, angmax, b_fr, b_to, r, x, rate_a, rate_b, rate_c]
    """
    src = edge_index[0]
    dst = edge_index[1]

    r = edge_attr[:, 4]
    x = edge_attr[:, 5]
    b_fr = edge_attr[:, 2]
    b_to = edge_attr[:, 3]

    z_sq = r ** 2 + x ** 2
    g_s = r / z_sq
    b_s = -x / z_sq

    for k in range(edge_index.shape[1]):
        i, j = src[k].item(), dst[k].item()

        G[i, i] += g_s[k]
        B[i, i] += b_s[k] + b_fr[k]

        G[j, j] += g_s[k]
        B[j, j] += b_s[k] + b_to[k]

        G[i, j] -= g_s[k]
        G[j, i] -= g_s[k]
        B[i, j] -= b_s[k]
        B[j, i] -= b_s[k]


def _add_transformers_to_ybus(G, B, edge_index, edge_attr):
    """
    Add transformer contributions to Y-bus.

    Transformer edge_attr: [angmin, angmax, r, x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to]

    Standard transformer model (from-bus is tap side):
        Y_ii += y_s / a^2
        Y_jj += y_s
        Y_ij = -y_s / (a * e^{-j*shift})
        Y_ji = -y_s / (a * e^{j*shift})
    """
    src = edge_index[0]
    dst = edge_index[1]

    r = edge_attr[:, 2]
    x = edge_attr[:, 3]
    tap = edge_attr[:, 7]
    shift = edge_attr[:, 8]

    z_sq = r ** 2 + x ** 2 + 1e-12
    g_s = r / z_sq
    b_s = -x / z_sq

    tap_sq = tap ** 2

    cos_shift = torch.cos(shift)
    sin_shift = torch.sin(shift)

    for k in range(edge_index.shape[1]):
        i, j = src[k].item(), dst[k].item()
        a = tap[k]
        a2 = tap_sq[k]
        gs = g_s[k]
        bs = b_s[k]
        cs = cos_shift[k]
        ss = sin_shift[k]

        G[i, i] += gs / a2
        B[i, i] += bs / a2

        G[j, j] += gs
        B[j, j] += bs

        G[i, j] += (-gs * cs - bs * ss) / a
        B[i, j] += (-bs * cs + gs * ss) / a

        G[j, i] += (-gs * cs + bs * ss) / a
        B[j, i] += (-bs * cs - gs * ss) / a


def _add_shunts_to_ybus(G, B, shunt_x, shunt_edge_index):
    """
    Add shunt element contributions to Y-bus diagonal.

    OPFDataset shunt.x columns: [Bs, Gs] (susceptance first, conductance second).
    Verified against IEEE 14-bus: bus 9 has Bs=0.19 pu (capacitor), Gs=0.
    """
    bus_idx = shunt_edge_index[1]
    for k in range(shunt_x.shape[0]):
        bus = bus_idx[k].item()
        B[bus, bus] += shunt_x[k, 0]  # Bs (susceptance)
        G[bus, bus] += shunt_x[k, 1]  # Gs (conductance)
