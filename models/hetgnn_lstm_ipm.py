"""HetGNN-LSTM-IPM: GNN encoder + LSTM inner optimizer.

The GNN encodes grid topology into per-node embeddings (once per outer step).
The LSTM iteratively refines the Newton step using those embeddings.

Architecture:
1. HetGNN encodes (graph features + current IPM state) → per-node embeddings
2. Embeddings are flattened and concatenated into the LSTM's input
3. LSTM runs inner_T steps to predict the Newton direction
4. Newton step is unpacked back onto bus/gen nodes

This combines:
- Topology awareness (GNN understands which buses connect)
- Recurrent optimization (LSTM learns step-refinement strategy)
- IPM-LSTM's proven training recipe (KKT sub-objective loss)
"""
import torch
import torch.nn as nn


def _mlp(in_dim, hidden_dim, out_dim, n_layers=2):
    layers = []
    for i in range(n_layers):
        d_in = in_dim if i == 0 else hidden_dim
        d_out = out_dim if i == n_layers - 1 else hidden_dim
        layers.append(nn.Linear(d_in, d_out))
        if i < n_layers - 1:
            layers += [nn.LayerNorm(d_out), nn.SiLU()]
    return nn.Sequential(*layers)


class HetGNNEncoder(nn.Module):
    """Lightweight HetGNN that encodes graph + IPM state into node embeddings."""

    def __init__(self, bus_in=12, gen_in=17, load_in=2, hidden=64, num_layers=3):
        super().__init__()
        h = hidden
        self.bus_enc = nn.Linear(bus_in, h)
        self.gen_enc = nn.Linear(gen_in, h)
        self.load_enc = nn.Linear(load_in, h)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "msg_ac": _mlp(2*h + 9, h, h),
                "msg_xfmr": _mlp(2*h + 11, h, h),
                "msg_g2b": _mlp(2*h, h, h),
                "msg_b2g": _mlp(2*h, h, h),
                "msg_l2b": _mlp(2*h, h, h),
                "upd_bus": _mlp(h, h, h),
                "upd_gen": _mlp(h, h, h),
                "norm_bus": nn.LayerNorm(h),
                "norm_gen": nn.LayerNorm(h),
            }))

    def forward(self, data, bus_state, gen_state):
        hb = self.bus_enc(torch.cat([data["bus"].x, bus_state], -1))
        hg = self.gen_enc(torch.cat([data["generator"].x, gen_state], -1))
        hl = self.load_enc(data["load"].x)

        for layer in self.layers:
            nb, h = hb.shape
            mb = torch.zeros(nb, h, device=hb.device)

            if ("bus", "ac_line", "bus") in data.edge_types:
                ei = data["bus", "ac_line", "bus"].edge_index
                ea = data["bus", "ac_line", "bus"].edge_attr
                s, d = ei[0], ei[1]
                m = layer["msg_ac"](torch.cat([hb[d], hb[s], ea], -1))
                mb.scatter_add_(0, d.unsqueeze(1).expand_as(m), m)
                m2 = layer["msg_ac"](torch.cat([hb[s], hb[d], ea], -1))
                mb.scatter_add_(0, s.unsqueeze(1).expand_as(m2), m2)

            if ("bus", "transformer", "bus") in data.edge_types:
                ei = data["bus", "transformer", "bus"].edge_index
                ea = data["bus", "transformer", "bus"].edge_attr
                s, d = ei[0], ei[1]
                m = layer["msg_xfmr"](torch.cat([hb[d], hb[s], ea], -1))
                mb.scatter_add_(0, d.unsqueeze(1).expand_as(m), m)
                m2 = layer["msg_xfmr"](torch.cat([hb[s], hb[d], ea], -1))
                mb.scatter_add_(0, s.unsqueeze(1).expand_as(m2), m2)

            ei_g = data["generator", "generator_link", "bus"].edge_index
            mg2b = layer["msg_g2b"](torch.cat([hb[ei_g[1]], hg[ei_g[0]]], -1))
            mb.scatter_add_(0, ei_g[1].unsqueeze(1).expand_as(mg2b), mg2b)

            ei_l = data["load", "load_link", "bus"].edge_index
            ml2b = layer["msg_l2b"](torch.cat([hb[ei_l[1]], hl[ei_l[0]]], -1))
            mb.scatter_add_(0, ei_l[1].unsqueeze(1).expand_as(ml2b), ml2b)

            mg = torch.zeros(hg.shape[0], h, device=hg.device)
            ei_b2g = data["bus", "generator_link", "generator"].edge_index
            mb2g = layer["msg_b2g"](torch.cat([hg[ei_b2g[1]], hb[ei_b2g[0]]], -1))
            mg.scatter_add_(0, ei_b2g[1].unsqueeze(1).expand_as(mb2g), mb2g)

            hb = hb + layer["upd_bus"](layer["norm_bus"](hb + mb))
            hg = hg + layer["upd_gen"](layer["norm_gen"](hg + mg))

        return hb, hg


class HetGNNLSTMIPM(nn.Module):
    """GNN encoder + LSTM optimizer for IPM Newton steps.

    The GNN runs once per outer IPM step to encode topology.
    The LSTM runs inner_T steps to refine the Newton direction.
    """

    BUS_STATE_DIM = 8
    GEN_STATE_DIM = 6

    def __init__(self, gnn_hidden=64, gnn_layers=3, lstm_hidden=64, inner_T=5):
        super().__init__()
        self.gnn = HetGNNEncoder(
            bus_in=4 + self.BUS_STATE_DIM,
            gen_in=11 + self.GEN_STATE_DIM,
            hidden=gnn_hidden,
            num_layers=gnn_layers,
        )

        # Project GNN embeddings to per-variable features
        self.bus_proj = nn.Linear(gnn_hidden, self.BUS_STATE_DIM)
        self.gen_proj = nn.Linear(gnn_hidden, self.GEN_STATE_DIM)

        # LSTM for iterative refinement (operates per-variable)
        self.inner_T = inner_T
        lstm_in = 2  # [current_value, gnn_guidance]
        self.W_i = nn.Parameter(torch.randn(lstm_in, lstm_hidden) * 0.01)
        self.U_i = nn.Parameter(torch.randn(lstm_hidden, lstm_hidden) * 0.01)
        self.b_i = nn.Parameter(torch.zeros(lstm_hidden))
        self.W_f = nn.Parameter(torch.randn(lstm_in, lstm_hidden) * 0.01)
        self.U_f = nn.Parameter(torch.randn(lstm_hidden, lstm_hidden) * 0.01)
        self.b_f = nn.Parameter(torch.zeros(lstm_hidden))
        self.W_o = nn.Parameter(torch.randn(lstm_in, lstm_hidden) * 0.01)
        self.U_o = nn.Parameter(torch.randn(lstm_hidden, lstm_hidden) * 0.01)
        self.b_o = nn.Parameter(torch.zeros(lstm_hidden))
        self.W_u = nn.Parameter(torch.randn(lstm_in, lstm_hidden) * 0.01)
        self.U_u = nn.Parameter(torch.randn(lstm_hidden, lstm_hidden) * 0.01)
        self.b_u = nn.Parameter(torch.zeros(lstm_hidden))
        self.W_out = nn.Parameter(torch.randn(lstm_hidden, 1) * 0.01)
        self.b_out = nn.Parameter(torch.zeros(1))

    def forward(self, data, bus_state, gen_state):
        """Predict Newton step using GNN encoding + LSTM refinement.

        Args:
            data: PyG HeteroData
            bus_state: [n_bus, 8] current primal-dual state
            gen_state: [n_gen, 6] current primal-dual state

        Returns:
            delta_bus: [n_bus, 8] step direction
            delta_gen: [n_gen, 6] step direction
        """
        # GNN encodes topology + state
        hb, hg = self.gnn(data, bus_state, gen_state)

        # Project to per-variable guidance signal
        bus_guide = self.bus_proj(hb)  # [n_bus, 8]
        gen_guide = self.gen_proj(hg)  # [n_gen, 6]

        # Flatten all variables for LSTM processing
        # Bus: [n_bus * 8] variables, Gen: [n_gen * 6] variables
        n_bus, n_bus_vars = bus_state.shape
        n_gen, n_gen_vars = gen_state.shape
        total_vars = n_bus * n_bus_vars + n_gen * n_gen_vars

        cur_val = torch.cat([bus_state.reshape(-1), gen_state.reshape(-1)])  # [total_vars]
        guide = torch.cat([bus_guide.reshape(-1), gen_guide.reshape(-1)])    # [total_vars]

        # LSTM iterative refinement (per-variable, shared weights)
        cur_val = cur_val.unsqueeze(0).unsqueeze(-1)  # [1, total_vars, 1]
        guide = guide.unsqueeze(0).unsqueeze(-1)       # [1, total_vars, 1]

        H = torch.zeros(1, total_vars, self.W_i.shape[1], device=bus_state.device)
        C = torch.zeros_like(H)

        delta = torch.zeros_like(cur_val)

        for t in range(self.inner_T):
            inp = torch.cat([delta, guide], dim=-1)  # [1, total_vars, 2]
            I = torch.sigmoid(inp @ self.W_i + H @ self.U_i + self.b_i)
            F = torch.sigmoid(inp @ self.W_f + H @ self.U_f + self.b_f)
            O = torch.sigmoid(inp @ self.W_o + H @ self.U_o + self.b_o)
            U = torch.tanh(inp @ self.W_u + H @ self.U_u + self.b_u)
            C = I * U + F * C
            H = O * torch.tanh(C)
            step = H @ self.W_out + self.b_out
            delta = delta + step

        delta = delta.squeeze(0).squeeze(-1)  # [total_vars]

        delta_bus = delta[:n_bus * n_bus_vars].reshape(n_bus, n_bus_vars)
        delta_gen = delta[n_bus * n_bus_vars:].reshape(n_gen, n_gen_vars)
        return delta_bus, delta_gen

    def name(self):
        return 'hetgnn_lstm_ipm'
