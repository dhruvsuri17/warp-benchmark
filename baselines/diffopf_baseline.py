"""
DiffOPF reimplementation baseline.

DiffOPF (Sayed et al.) uses an MLP denoiser instead of a GNN.
Must retrain on OPFData case118/500/2000 for fair comparison.
Their paper only shows case14/30/57.
"""

import torch
import torch.nn as nn

from models.diffusion import GaussianDiffusion
from models.embeddings import SinusoidalTimestepEmbedding


class DiffOPFDenoiser(nn.Module):
    """
    MLP-based denoiser (DiffOPF architecture).
    No graph structure — takes flattened features as input.
    """

    def __init__(self, var_dim: int, load_dim: int, hidden_dim: int = 512,
                 n_layers: int = 6, timestep_dim: int = 128):
        super().__init__()

        self.timestep_encoder = SinusoidalTimestepEmbedding(timestep_dim)

        layers = []
        in_dim = var_dim + load_dim + timestep_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else var_dim
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.SiLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                load: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: [batch, var_dim] noisy variables
            t: [batch] timesteps
            load: [batch, load_dim] load features

        Returns:
            eps_pred: [batch, var_dim] predicted noise
        """
        t_emb = self.timestep_encoder(t)
        inp = torch.cat([x_t, load, t_emb], dim=-1)
        return self.net(inp)
