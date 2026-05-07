"""
DDPM forward/reverse process and DDIM sampler.

Noise schedule: cosine (Nichol & Dhariwal 2021)
Training: forward process + noise prediction
Inference: DDIM (Song et al. 2021), 50 steps, deterministic (eta=0)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_schedule(T: int = 1000, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule (Nichol & Dhariwal 2021).

    Returns:
        betas: [T] noise schedule
    """
    t = torch.linspace(0, T, T + 1)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0, 0.999)


class GaussianDiffusion(nn.Module):
    """
    DDPM / DDIM diffusion process for OPF variable vectors.

    Training: add noise, predict it
    Inference: DDIM reverse process with K parallel samples
    """

    def __init__(self, T: int = 1000, schedule: str = "cosine"):
        super().__init__()
        self.T = T

        if schedule == "cosine":
            betas = cosine_schedule(T)
        else:
            betas = torch.linspace(1e-4, 0.02, T)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> tuple:
        """
        Forward process: add noise to clean data.

        Args:
            x0: [batch, dim] clean OPF solutions
            t: [batch] integer timesteps
            noise: [batch, dim] optional pre-sampled noise

        Returns:
            x_t: [batch, dim] noisy data
            noise: [batch, dim] the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]

        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_t, noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor,
                   eps_pred: torch.Tensor) -> torch.Tensor:
        """
        Tweedie estimate of x0 from noisy x_t and predicted noise.
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]

        while sqrt_alpha.dim() < x_t.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        x0_hat = (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha.clamp(min=1e-8)
        return x0_hat

    @torch.no_grad()
    def ddim_sample(self, model, shape, n_steps: int = 50, eta: float = 0.0,
                    model_kwargs: dict = None, device: str = "cpu") -> torch.Tensor:
        """
        DDIM reverse sampling (deterministic when eta=0).

        Args:
            model: noise prediction network (HetGNN)
            shape: output shape [batch, dim]
            n_steps: number of DDIM steps (50 recommended)
            eta: stochasticity (0 = deterministic DDIM)
            model_kwargs: additional kwargs passed to model
            device: device to sample on

        Returns:
            x0: [batch, dim] denoised samples
        """
        if model_kwargs is None:
            model_kwargs = {}

        step_indices = torch.linspace(self.T - 1, 0, n_steps + 1).long()
        x = torch.randn(shape, device=device)

        for i in range(n_steps):
            t_curr = step_indices[i]
            t_prev = step_indices[i + 1]

            t_batch = torch.full((shape[0],), t_curr, device=device, dtype=torch.long)
            eps_pred = model(t=t_batch, **model_kwargs)

            alpha = self.alphas_cumprod[t_curr]
            alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

            x0_hat = self.predict_x0(x, t_batch, eps_pred)

            sigma = eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)
            )

            dir_xt = torch.sqrt((1 - alpha_prev - sigma ** 2).clamp(min=0)) * eps_pred
            x = torch.sqrt(alpha_prev) * x0_hat + dir_xt

            if sigma > 0 and i < n_steps - 1:
                x = x + sigma * torch.randn_like(x)

        return x
