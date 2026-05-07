"""
WARP training losses: L_ddpm + weighted L_physics.

The physics loss is applied on the Tweedie x0 estimate, NOT on the noisy x_t.
A time-dependent weighting schedule downweights the physics loss at extreme t values:
- High t: x0_hat is garbage (too noisy), physics gradients are noise
- Low t: x0_hat is near-clean but denoising is almost trivial
- Peak around t ~ 0.3*T
"""

import math

import torch
import torch.nn.functional as F

from physics.acpf import physics_loss as _physics_loss


def physics_schedule(t: torch.Tensor, T: int = 1000) -> torch.Tensor:
    """
    Time-dependent weight for physics loss. Peaks around t ~ 0.3*T.
    Uses a sin curve that rises quickly and tails off.

    Args:
        t: [batch] integer timesteps
        T: total diffusion steps

    Returns:
        [batch] weights in [0, 1]
    """
    t_frac = t.float() / T
    return torch.sin(math.pi * t_frac).clamp(min=0)


def warp_loss(eps_pred: torch.Tensor, eps_true: torch.Tensor,
              x0_hat: torch.Tensor, graph_data: dict, t: torch.Tensor,
              T: int = 1000,
              lambda_phy: float = 0.1, lambda_V: float = 1.0,
              lambda_S: float = 1.0):
    """
    Combined WARP training loss.

    Args:
        eps_pred: predicted noise
        eps_true: ground truth noise
        x0_hat: Tweedie estimate of clean sample
        graph_data: dict with keys needed by physics_loss
            (n_bus, n_gen, Pd, Qd, G, B, edge_index, Vm_min, Vm_max,
             S_max, gen_bus_idx)
        t: [batch] integer timesteps
        T: total diffusion steps
        lambda_phy: physics loss weight
        lambda_V: voltage violation weight
        lambda_S: thermal violation weight

    Returns:
        total_loss: scalar
        loss_dict: dict of component losses for logging
    """
    L_ddpm = F.mse_loss(eps_pred, eps_true)

    phy_weight = physics_schedule(t, T).mean()

    L_phy = _physics_loss(
        x0_hat=x0_hat,
        n_bus=graph_data["n_bus"],
        n_gen=graph_data["n_gen"],
        Pd=graph_data["Pd"],
        Qd=graph_data["Qd"],
        G=graph_data["G"],
        B=graph_data["B"],
        edge_index=graph_data["edge_index"],
        Vm_min=graph_data["Vm_min"],
        Vm_max=graph_data["Vm_max"],
        S_max=graph_data["S_max"],
        gen_bus_idx=graph_data["gen_bus_idx"],
        lambda_V=lambda_V,
        lambda_S=lambda_S,
    )

    total = L_ddpm + lambda_phy * phy_weight * L_phy

    return total, {
        "L_ddpm": L_ddpm.item(),
        "L_phy": L_phy.item(),
        "phy_weight": phy_weight.item(),
    }


def det_gnn_loss(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """
    Simple MSE loss for deterministic GNN baseline.
    """
    return F.mse_loss(x_pred, x_true)
