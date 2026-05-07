"""HetGNN-IPM: Heterogeneous GNN as Newton step solver inside IPM loop.

Instead of a flat LSTM over the 1,640-dim KKT vector, the GNN operates
on the power grid graph. Each bus node carries (Va, Vm, lam_P, lam_Q, zl_Va,
zl_Vm, zu_Va, zu_Vm) and each generator node carries (Pg, Qg, zl_Pg, zl_Qg,
zu_Pg, zu_Qg). Message passing exploits grid topology to predict Newton steps.

This is the novel contribution: topology-aware IPM step prediction for AC-OPF.
"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _mlp(in_dim, hidden_dim, out_dim, n_layers=2):
    layers = []
    for i in range(n_layers):
        d_in = in_dim if i == 0 else hidden_dim
        d_out = out_dim if i == n_layers - 1 else hidden_dim
        layers.append(nn.Linear(d_in, d_out))
        if i < n_layers - 1:
            layers += [nn.LayerNorm(d_out), nn.SiLU()]
    return nn.Sequential(*layers)


class HetGNNIPMLayer(nn.Module):
    """One layer of heterogeneous message passing for IPM state."""

    def __init__(self, h, edge_ac_dim=9, edge_xfmr_dim=11):
        super().__init__()
        self.msg_ac = _mlp(2*h + edge_ac_dim, h, h)
        self.msg_xfmr = _mlp(2*h + edge_xfmr_dim, h, h)
        self.msg_g2b = _mlp(2*h, h, h)
        self.msg_b2g = _mlp(2*h, h, h)
        self.msg_l2b = _mlp(2*h, h, h)
        self.upd_bus = _mlp(2*h, h, h)
        self.upd_gen = _mlp(2*h, h, h)
        self.norm_bus = nn.LayerNorm(h)
        self.norm_gen = nn.LayerNorm(h)

    def forward(self, hb, hg, hl, data):
        nb, h = hb.shape
        mb = torch.zeros(nb, h, device=hb.device)

        if ("bus", "ac_line", "bus") in data.edge_types:
            ei = data["bus", "ac_line", "bus"].edge_index
            ea = data["bus", "ac_line", "bus"].edge_attr
            s, d = ei[0], ei[1]
            m = self.msg_ac(torch.cat([hb[d], hb[s], ea], -1))
            mb.scatter_add_(0, d.unsqueeze(1).expand_as(m), m)
            m2 = self.msg_ac(torch.cat([hb[s], hb[d], ea], -1))
            mb.scatter_add_(0, s.unsqueeze(1).expand_as(m2), m2)

        if ("bus", "transformer", "bus") in data.edge_types:
            ei = data["bus", "transformer", "bus"].edge_index
            ea = data["bus", "transformer", "bus"].edge_attr
            s, d = ei[0], ei[1]
            m = self.msg_xfmr(torch.cat([hb[d], hb[s], ea], -1))
            mb.scatter_add_(0, d.unsqueeze(1).expand_as(m), m)
            m2 = self.msg_xfmr(torch.cat([hb[s], hb[d], ea], -1))
            mb.scatter_add_(0, s.unsqueeze(1).expand_as(m2), m2)

        ei = data["generator", "generator_link", "bus"].edge_index
        m = self.msg_g2b(torch.cat([hb[ei[1]], hg[ei[0]]], -1))
        mb.scatter_add_(0, ei[1].unsqueeze(1).expand_as(m), m)

        ei = data["load", "load_link", "bus"].edge_index
        m = self.msg_l2b(torch.cat([hb[ei[1]], hl[ei[0]]], -1))
        mb.scatter_add_(0, ei[1].unsqueeze(1).expand_as(m), m)

        mg = torch.zeros(hg.shape[0], h, device=hg.device)
        ei = data["bus", "generator_link", "generator"].edge_index
        m = self.msg_b2g(torch.cat([hg[ei[1]], hb[ei[0]]], -1))
        mg.scatter_add_(0, ei[1].unsqueeze(1).expand_as(m), m)

        hb = hb + self.upd_bus(torch.cat([self.norm_bus(hb), mb], -1))
        hg = hg + self.upd_gen(torch.cat([self.norm_gen(hg), mg], -1))
        return hb, hg, hl


class HetGNNIPM(nn.Module):
    """Heterogeneous GNN for predicting IPM Newton steps on AC-OPF.

    Architecture:
    1. Encode current primal-dual state per node
    2. N layers of heterogeneous message passing
    3. Decode per-node Newton step deltas

    Per-bus state: [Va, Vm, lam_P, lam_Q, zl_Va, zl_Vm, zu_Va, zu_Vm] = 8 dims
    Per-gen state: [Pg, Qg, zl_Pg, zl_Qg, zu_Pg, zu_Qg] = 6 dims
    Per-bus features: bus.x [4 dims]
    Per-gen features: gen.x [11 dims]
    Per-load features: load.x [2 dims]

    Output per-bus: delta [Va, Vm, lam_P, lam_Q, zl_Va, zl_Vm, zu_Va, zu_Vm]
    Output per-gen: delta [Pg, Qg, zl_Pg, zl_Qg, zu_Pg, zu_Qg]
    """

    BUS_STATE_DIM = 8   # Va, Vm, lam_P, lam_Q, zl_Va, zl_Vm, zu_Va, zu_Vm
    GEN_STATE_DIM = 6   # Pg, Qg, zl_Pg, zl_Qg, zu_Pg, zu_Qg
    BUS_FEAT_DIM = 4
    GEN_FEAT_DIM = 11
    LOAD_FEAT_DIM = 2

    def __init__(self, hidden_dim=128, num_layers=6):
        super().__init__()
        h = hidden_dim

        # Encode features + IPM state into hidden dim
        self.bus_enc = nn.Linear(self.BUS_FEAT_DIM + self.BUS_STATE_DIM, h)
        self.gen_enc = nn.Linear(self.GEN_FEAT_DIM + self.GEN_STATE_DIM, h)
        self.load_enc = nn.Linear(self.LOAD_FEAT_DIM, h)

        # Message passing layers
        self.layers = nn.ModuleList([HetGNNIPMLayer(h) for _ in range(num_layers)])

        # Decode Newton step per node
        self.bus_head = nn.Sequential(
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, self.BUS_STATE_DIM))
        self.gen_head = nn.Sequential(
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, self.GEN_STATE_DIM))

    def forward(self, data, bus_state, gen_state):
        """Predict Newton step direction.

        Args:
            data: PyG HeteroData graph (topology + features)
            bus_state: [n_bus, 8] current primal-dual state per bus
            gen_state: [n_gen, 6] current primal-dual state per gen

        Returns:
            delta_bus: [n_bus, 8] step direction for bus variables
            delta_gen: [n_gen, 6] step direction for gen variables
        """
        hb = self.bus_enc(torch.cat([data["bus"].x, bus_state], dim=-1))
        hg = self.gen_enc(torch.cat([data["generator"].x, gen_state], dim=-1))
        hl = self.load_enc(data["load"].x)

        for layer in self.layers:
            hb, hg, hl = layer(hb, hg, hl, data)

        delta_bus = self.bus_head(hb)
        delta_gen = self.gen_head(hg)
        return delta_bus, delta_gen

    def name(self):
        return 'hetgnn_ipm'
