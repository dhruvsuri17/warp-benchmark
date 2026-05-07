"""
DDIM/DDPM sampler for WARP inference.

Generates K warm-start candidates from the learned diffusion prior.
Uses DDIM with 50 steps (deterministic, eta=0).
"""

import torch

from models.diffusion import GaussianDiffusion


@torch.no_grad()
def sample_warmstarts(model, diffusion, data, K=5, n_steps=50, eta=0.0, device="cpu"):
    """
    Generate K warm-start samples via DDIM reverse process.

    Args:
        model: trained HetGNN denoiser
        diffusion: GaussianDiffusion instance
        data: HeteroData graph (single instance, used for structure/features)
        K: number of parallel samples
        n_steps: DDIM steps (default 50)
        eta: stochasticity (0 = deterministic DDIM)
        device: device for computation

    Returns:
        bus_samples: [K, n_bus, 2] (Va, Vm) candidates
        gen_samples: [K, n_gen, 2] (Pg, Qg) candidates
    """
    model.eval()
    data = data.to(device)

    n_bus = data["bus"].x.shape[0]
    n_gen = data["generator"].x.shape[0]

    T = diffusion.T
    step_indices = torch.linspace(T - 1, 0, n_steps + 1).long().to(device)

    bus_samples = torch.randn(K, n_bus, 2, device=device)
    gen_samples = torch.randn(K, n_gen, 2, device=device)

    for i in range(n_steps):
        t_curr = step_indices[i]
        t_prev = step_indices[i + 1]

        alpha = diffusion.alphas_cumprod[t_curr]
        alpha_prev = diffusion.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

        sigma = eta * torch.sqrt(
            (1 - alpha_prev) / (1 - alpha).clamp(min=1e-12) * (1 - alpha / alpha_prev).clamp(min=0)
        )

        t_batch = torch.full((1,), t_curr, device=device, dtype=torch.long)

        bus_avg = torch.zeros(n_bus, 2, device=device)
        gen_avg = torch.zeros(n_gen, 2, device=device)

        for k in range(K):
            bus_pred, gen_pred = model(data, t_batch)
            bus_avg += bus_pred
            gen_avg += gen_pred

        bus_eps = bus_avg / K
        gen_eps = gen_avg / K

        sqrt_a = diffusion.sqrt_alphas_cumprod[t_curr]
        sqrt_1ma = diffusion.sqrt_one_minus_alphas_cumprod[t_curr]

        for k in range(K):
            bus_x0 = (bus_samples[k] - sqrt_1ma * bus_eps) / sqrt_a.clamp(min=1e-6)
            gen_x0 = (gen_samples[k] - sqrt_1ma * gen_eps) / sqrt_a.clamp(min=1e-6)

            dir_bus = torch.sqrt((1 - alpha_prev - sigma ** 2).clamp(min=0)) * bus_eps
            dir_gen = torch.sqrt((1 - alpha_prev - sigma ** 2).clamp(min=0)) * gen_eps

            bus_samples[k] = torch.sqrt(alpha_prev) * bus_x0 + dir_bus
            gen_samples[k] = torch.sqrt(alpha_prev) * gen_x0 + dir_gen

            if sigma > 0 and i < n_steps - 1:
                bus_samples[k] += sigma * torch.randn_like(bus_samples[k])
                gen_samples[k] += sigma * torch.randn_like(gen_samples[k])

    return bus_samples, gen_samples
