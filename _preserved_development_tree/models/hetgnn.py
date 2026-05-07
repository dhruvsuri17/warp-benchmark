"""
HetGNN: Heterogeneous Graph Neural Network denoiser for WARP.

Works directly with PyG HeteroData from OPFDataset.

Node types: bus (14 nodes, 4 feat), generator (5, 11), load (11, 2), shunt (1, 2)
Edge types: ac_line (bus->bus, 9 feat), transformer (bus->bus, 11 feat),
            generator_link (gen<->bus), load_link (load<->bus), shunt_link (shunt<->bus)

Output: predicted [Va, Vm] for each bus and [Pg, Qg] for each generator.

Architecture constants:
    HIDDEN_DIM    = 256
    NUM_LAYERS    = 8
    PE_DIM        = 16
    TIMESTEP_DIM  = 128
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from models.embeddings import SinusoidalTimestepEmbedding, AdaLN, LaplacianPE

HIDDEN_DIM = 256
NUM_LAYERS = 8
PE_DIM = 16
TIMESTEP_DIM = 128

NODE_FEAT_DIMS = {"bus": 4, "generator": 11, "load": 2, "shunt": 2}

EDGE_FEAT_DIMS = {"ac_line": 9, "transformer": 11}


def _mlp(in_dim, hidden_dim, out_dim, n_layers=2):
    layers = []
    for i in range(n_layers):
        d_in = in_dim if i == 0 else hidden_dim
        d_out = out_dim if i == n_layers - 1 else hidden_dim
        layers.append(nn.Linear(d_in, d_out))
        if i < n_layers - 1:
            layers.append(nn.LayerNorm(d_out))
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class HetGNNLayer(nn.Module):
    """
    One layer of heterogeneous message passing.

    Messages flow along all edge types. Each edge type has its own message MLP.
    Each node type has its own update MLP with adaLN conditioning.
    Residual connections at every layer.
    """

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message MLPs for bus-bus edges (ac_line, transformer)
        self.msg_ac_line = _mlp(2 * hidden_dim + EDGE_FEAT_DIMS["ac_line"], hidden_dim, hidden_dim)
        self.msg_transformer = _mlp(2 * hidden_dim + EDGE_FEAT_DIMS["transformer"], hidden_dim, hidden_dim)

        # Message MLPs for bipartite edges (no edge features)
        self.msg_gen_to_bus = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.msg_bus_to_gen = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.msg_load_to_bus = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.msg_shunt_to_bus = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)

        # adaLN + update per node type
        self.adaln_bus = AdaLN(hidden_dim, cond_dim)
        self.adaln_gen = AdaLN(hidden_dim, cond_dim)
        self.adaln_load = AdaLN(hidden_dim, cond_dim)

        self.update_bus = _mlp(hidden_dim, hidden_dim, hidden_dim)
        self.update_gen = _mlp(hidden_dim, hidden_dim, hidden_dim)
        self.update_load = _mlp(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, h_bus, h_gen, h_load, h_shunt, data, cond_bus, cond_gen, cond_load):
        n_bus = h_bus.shape[0]
        n_gen = h_gen.shape[0]
        n_load = h_load.shape[0]

        # --- Aggregate messages into bus nodes ---
        msg_bus = torch.zeros(n_bus, self.hidden_dim, device=h_bus.device)

        # ac_line: bus -> bus
        if ("bus", "ac_line", "bus") in data.edge_types:
            ei = data["bus", "ac_line", "bus"].edge_index
            ea = data["bus", "ac_line", "bus"].edge_attr
            src, dst = ei[0], ei[1]
            inp = torch.cat([h_bus[dst], h_bus[src], ea], dim=-1)
            m = self.msg_ac_line(inp)
            msg_bus.scatter_add_(0, dst.unsqueeze(1).expand_as(m), m)
            # reverse direction too
            inp_rev = torch.cat([h_bus[src], h_bus[dst], ea], dim=-1)
            m_rev = self.msg_ac_line(inp_rev)
            msg_bus.scatter_add_(0, src.unsqueeze(1).expand_as(m_rev), m_rev)

        # transformer: bus -> bus
        if ("bus", "transformer", "bus") in data.edge_types:
            ei = data["bus", "transformer", "bus"].edge_index
            ea = data["bus", "transformer", "bus"].edge_attr
            src, dst = ei[0], ei[1]
            inp = torch.cat([h_bus[dst], h_bus[src], ea], dim=-1)
            m = self.msg_transformer(inp)
            msg_bus.scatter_add_(0, dst.unsqueeze(1).expand_as(m), m)
            inp_rev = torch.cat([h_bus[src], h_bus[dst], ea], dim=-1)
            m_rev = self.msg_transformer(inp_rev)
            msg_bus.scatter_add_(0, src.unsqueeze(1).expand_as(m_rev), m_rev)

        # generator -> bus
        ei = data["generator", "generator_link", "bus"].edge_index
        src_g, dst_b = ei[0], ei[1]
        inp = torch.cat([h_bus[dst_b], h_gen[src_g]], dim=-1)
        m = self.msg_gen_to_bus(inp)
        msg_bus.scatter_add_(0, dst_b.unsqueeze(1).expand_as(m), m)

        # load -> bus
        ei = data["load", "load_link", "bus"].edge_index
        src_l, dst_b = ei[0], ei[1]
        inp = torch.cat([h_bus[dst_b], h_load[src_l]], dim=-1)
        m = self.msg_load_to_bus(inp)
        msg_bus.scatter_add_(0, dst_b.unsqueeze(1).expand_as(m), m)

        # shunt -> bus
        if "shunt" in data.node_types and h_shunt is not None and h_shunt.shape[0] > 0:
            ei = data["shunt", "shunt_link", "bus"].edge_index
            src_s, dst_b = ei[0], ei[1]
            inp = torch.cat([h_bus[dst_b], h_shunt[src_s]], dim=-1)
            m = self.msg_shunt_to_bus(inp)
            msg_bus.scatter_add_(0, dst_b.unsqueeze(1).expand_as(m), m)

        # --- Aggregate messages into generator nodes ---
        msg_gen = torch.zeros(n_gen, self.hidden_dim, device=h_gen.device)
        ei = data["bus", "generator_link", "generator"].edge_index
        src_b, dst_g = ei[0], ei[1]
        inp = torch.cat([h_gen[dst_g], h_bus[src_b]], dim=-1)
        m = self.msg_bus_to_gen(inp)
        msg_gen.scatter_add_(0, dst_g.unsqueeze(1).expand_as(m), m)

        # --- Update bus ---
        h_bus_cond = self.adaln_bus(h_bus + msg_bus, cond_bus)
        h_bus_new = h_bus + self.update_bus(h_bus_cond)

        # --- Update generator ---
        h_gen_cond = self.adaln_gen(h_gen + msg_gen, cond_gen)
        h_gen_new = h_gen + self.update_gen(h_gen_cond)

        # --- Update load (receives no messages back, but condition on timestep) ---
        h_load_cond = self.adaln_load(h_load, cond_load)
        h_load_new = h_load + self.update_load(h_load_cond)

        return h_bus_new, h_gen_new, h_load_new, h_shunt


class HetGNN(nn.Module):
    """
    Full HetGNN denoiser for WARP.

    Takes HeteroData + timestep, outputs predicted noise for [Va, Vm, Pg, Qg].
    """

    def __init__(self, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 pe_dim=PE_DIM, timestep_dim=TIMESTEP_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projections per node type
        # In diffusion mode, bus gets +2 (noisy Va, Vm) and gen gets +2 (noisy Pg, Qg)
        self.bus_proj = nn.Linear(NODE_FEAT_DIMS["bus"] + 2, hidden_dim)
        self.gen_proj = nn.Linear(NODE_FEAT_DIMS["generator"] + 2, hidden_dim)
        self.load_proj = nn.Linear(NODE_FEAT_DIMS["load"], hidden_dim)
        self.shunt_proj = nn.Linear(NODE_FEAT_DIMS["shunt"], hidden_dim)
        # Separate projections for deterministic mode (no noisy state)
        self.bus_proj_det = nn.Linear(NODE_FEAT_DIMS["bus"], hidden_dim)
        self.gen_proj_det = nn.Linear(NODE_FEAT_DIMS["generator"], hidden_dim)

        # Laplacian PE encoder
        self.pe_encoder = LaplacianPE(pe_dim, hidden_dim)

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(timestep_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(timestep_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message passing layers
        self.layers = nn.ModuleList([
            HetGNNLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Output heads
        self.bus_head = nn.Linear(hidden_dim, 2)  # (Va, Vm)
        self.gen_head = nn.Linear(hidden_dim, 2)  # (Pg, Qg)

    def forward(self, data, t, bus_pe=None, bus_noisy=None, gen_noisy=None):
        """
        Args:
            data: PyG HeteroData with node/edge types
            t: [1] or [batch_size] integer timestep
            bus_pe: [n_bus, pe_dim] precomputed Laplacian PE (optional)
            bus_noisy: [n_bus, 2] noisy (Va, Vm) for diffusion mode (None for det)
            gen_noisy: [n_gen, 2] noisy (Pg, Qg) for diffusion mode (None for det)

        Returns:
            bus_out: [n_bus, 2] predicted noise/values for (Va, Vm)
            gen_out: [n_gen, 2] predicted noise/values for (Pg, Qg)
        """
        # Input projections — concat noisy state if in diffusion mode
        if bus_noisy is not None:
            h_bus = self.bus_proj(torch.cat([data["bus"].x, bus_noisy], dim=-1))
            h_gen = self.gen_proj(torch.cat([data["generator"].x, gen_noisy], dim=-1))
        else:
            h_bus = self.bus_proj_det(data["bus"].x)
            h_gen = self.gen_proj_det(data["generator"].x)
        h_load = self.load_proj(data["load"].x)

        if "shunt" in data.node_types and data["shunt"].x.shape[0] > 0:
            h_shunt = self.shunt_proj(data["shunt"].x)
        else:
            h_shunt = torch.zeros(0, self.hidden_dim, device=h_bus.device)

        # Add Laplacian PE to bus embeddings
        if bus_pe is not None:
            h_bus = h_bus + self.pe_encoder(bus_pe)

        # Timestep conditioning
        t_emb = self.time_proj(self.time_embed(t))  # [1, hidden] or [B, hidden]
        if t_emb.shape[0] == 1:
            cond_bus = t_emb.expand(h_bus.shape[0], -1)
            cond_gen = t_emb.expand(h_gen.shape[0], -1)
            cond_load = t_emb.expand(h_load.shape[0], -1)
        else:
            cond_bus = t_emb
            cond_gen = t_emb
            cond_load = t_emb

        # Message passing
        for layer in self.layers:
            h_bus, h_gen, h_load, h_shunt = layer(
                h_bus, h_gen, h_load, h_shunt, data,
                cond_bus, cond_gen, cond_load,
            )

        # Output heads
        bus_out = self.bus_head(h_bus)   # [n_bus, 2] = (Va, Vm)
        gen_out = self.gen_head(h_gen)   # [n_gen, 2] = (Pg, Qg)

        return bus_out, gen_out
