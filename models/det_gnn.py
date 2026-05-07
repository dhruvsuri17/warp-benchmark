"""
Deterministic GNN baseline — same HetGNN backbone, no diffusion.

Trained with MSE loss: (load, graph) -> (Va, Vm, Pg, Qg).
Uses the deterministic input projections (no noisy state concatenation).
"""

import torch
import torch.nn as nn

from models.hetgnn import HetGNN, HIDDEN_DIM, NUM_LAYERS, PE_DIM, TIMESTEP_DIM


class DetGNN(nn.Module):

    def __init__(self, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 pe_dim=PE_DIM, timestep_dim=TIMESTEP_DIM):
        super().__init__()
        self.backbone = HetGNN(hidden_dim, num_layers, pe_dim, timestep_dim)

    def forward(self, data, bus_pe=None):
        """
        Predict OPF solution directly (t=0, no noisy state).
        """
        t = torch.zeros(1, dtype=torch.long, device=data["bus"].x.device)
        return self.backbone(data, t, bus_pe=bus_pe,
                             bus_noisy=None, gen_noisy=None)
