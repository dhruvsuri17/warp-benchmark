"""
Timestep sinusoidal embeddings and Laplacian positional encoding.

- Sinusoidal timestep embedding (for diffusion conditioning via adaLN)
- Laplacian PE (computed once per graph, cached)
"""

import math

import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding as in DDPM / DiT.
    Maps integer timestep to a TIMESTEP_DIM-dimensional vector.
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch_size] integer timesteps

        Returns:
            [batch_size, dim] embedding
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalisation conditioned on timestep embedding.

    gamma, beta = MLP(t_emb)
    h = gamma * LayerNorm(h) + beta

    Applied after message aggregation, before update MLP at every layer.
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [n_nodes, hidden_dim] node features
            cond: [batch_size, cond_dim] conditioning (timestep embedding)
                  or [n_nodes, cond_dim] if already expanded

        Returns:
            [n_nodes, hidden_dim] conditioned features
        """
        params = self.proj(cond)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma * self.norm(h) + beta


class LaplacianPE(nn.Module):
    """
    Learnable projection of precomputed Laplacian positional encodings.
    Sign-invariant via absolute value before projection.
    """

    def __init__(self, pe_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(pe_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, pe: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pe: [n_nodes, pe_dim] Laplacian eigenvectors

        Returns:
            [n_nodes, hidden_dim] projected PE
        """
        return self.proj(pe.abs())
