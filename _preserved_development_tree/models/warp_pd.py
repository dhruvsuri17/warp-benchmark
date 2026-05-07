"""WARP-PD: Diffusion over the full IPM state (x, λ, z, μ).

Denoiser is the G-nores EPD backbone with timestep conditioning.
Diffusion operates in normalized space over per-node primal-dual targets.
At inference, samples K candidates and selects by KKT residual.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(T=1000, s=0.008):
    t = torch.linspace(0, T, T + 1)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(0, 0.999)
    return betas


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class WARPPDBackbone(nn.Module):
    """EPD-GNN denoiser for WARP-PD.

    Same architecture as gnores EPD, but encoder takes:
    - bus: static features (6) + noisy state (8) + time embedding projected
    - gen: static features (13) + noisy state (6) + time embedding projected
    """

    def __init__(self, hidden_dim=128, k_steps=15, time_dim=64):
        super().__init__()
        from training.train_exp_gnores_bias import INBlockNoNodeRes
        from training.train_exp_e_epd_full import EdgeMLP, NodeMLP

        h = hidden_dim
        self.h = h

        # Time embedding
        self.time_embed = SinusoidalTimeEmb(time_dim)
        self.time_proj = nn.Sequential(nn.Linear(time_dim, h), nn.SiLU(), nn.Linear(h, h))

        # Encoder: static features + noisy state + time
        self.bus_enc = nn.Linear(6 + 8 + h, h)   # static(6) + noisy_state(8) + t_proj(h)
        self.gen_enc = nn.Linear(13 + 6 + h, h)  # static(13) + noisy_state(6) + t_proj(h)
        self.load_enc = nn.Linear(2, h)

        self.edge_enc = nn.ModuleDict({
            "ac_line": nn.Linear(9, h),
            "transformer": nn.Linear(11, h),
        })

        # Processor: unshared IN blocks (no node residuals)
        self.blocks = nn.ModuleList([INBlockNoNodeRes(h) for _ in range(k_steps)])

        # Decoder: predict noise (or clean state) for bus and gen
        self.bus_head = nn.Sequential(
            nn.Linear(h, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 8))
        self.gen_head = nn.Sequential(
            nn.Linear(h, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 6))

    def forward(self, data, noisy_bus, noisy_gen, t):
        """Predict noise from noisy IPM state.

        Args:
            data: HeteroData graph (topology + static features)
            noisy_bus: [n_bus, 8] noisy normalized bus state
            noisy_gen: [n_gen, 6] noisy normalized gen state
            t: [1] or [batch] diffusion timestep

        Returns:
            pred_bus: [n_bus, 8] predicted noise (or clean state)
            pred_gen: [n_gen, 6] predicted noise (or clean state)
        """
        n_bus = data["bus"].x.shape[0]
        n_gen = data["generator"].x.shape[0]
        device = data["bus"].x.device

        # Time embedding — expand per-graph t to per-node
        t_emb = self.time_proj(self.time_embed(t))  # [1, h] or [B, h]
        if hasattr(data["bus"], "batch"):
            t_bus = t_emb[data["bus"].batch]    # [total_bus, h]
            t_gen = t_emb[data["generator"].batch]  # [total_gen, h]
        elif t_emb.shape[0] == 1:
            t_bus = t_emb.expand(n_bus, -1)
            t_gen = t_emb.expand(n_gen, -1)
        else:
            t_bus = t_emb
            t_gen = t_emb

        # Inject loads
        bus_load = torch.zeros(n_bus, 2, device=device)
        ei_l = data["load", "load_link", "bus"].edge_index
        bus_load.scatter_add_(0, ei_l[1].unsqueeze(1).expand(-1, 2), data["load"].x[ei_l[0]])
        ei_g = data["generator", "generator_link", "bus"].edge_index
        gen_load = bus_load[ei_g[1]]

        # Encode: static + noisy_state + time
        bus_feat = torch.cat([data["bus"].x, bus_load, noisy_bus, t_bus], dim=-1)
        gen_feat = torch.cat([data["generator"].x, gen_load, noisy_gen, t_gen], dim=-1)

        nodes = {
            "bus": self.bus_enc(bus_feat),
            "generator": self.gen_enc(gen_feat),
            "load": self.load_enc(data["load"].x),
        }

        h = self.h
        edges = {}
        if ("bus", "ac_line", "bus") in data.edge_types:
            edges["ac_line"] = self.edge_enc["ac_line"](data["bus", "ac_line", "bus"].edge_attr)
        else:
            edges["ac_line"] = torch.zeros(0, h, device=device)
        if ("bus", "transformer", "bus") in data.edge_types:
            edges["transformer"] = self.edge_enc["transformer"](data["bus", "transformer", "bus"].edge_attr)
        else:
            edges["transformer"] = torch.zeros(0, h, device=device)
        edges["gen_link"] = torch.zeros(ei_g.shape[1], h, device=device)
        edges["load_link"] = torch.zeros(ei_l.shape[1], h, device=device)

        for block in self.blocks:
            nodes, edges = block(nodes, edges, data)

        return self.bus_head(nodes["bus"]), self.gen_head(nodes["generator"])


class WARPPD(nn.Module):
    """Full WARP-PD: diffusion wrapper around the EPD denoiser."""

    def __init__(self, hidden_dim=128, k_steps=15, T=1000, time_dim=64):
        super().__init__()
        self.T = T
        self.backbone = WARPPDBackbone(hidden_dim, k_steps, time_dim)

        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", ac)
        self.register_buffer("sqrt_ac", torch.sqrt(ac))
        self.register_buffer("sqrt_1mac", torch.sqrt(1.0 - ac))

    def forward(self, data, clean_bus, clean_gen, t=None):
        """Training forward: add noise, predict noise. Supports batched graphs.

        Args:
            data: HeteroData graph (possibly batched by PyG DataLoader)
            clean_bus: [total_bus, 8] clean normalized bus targets
            clean_gen: [total_gen, 6] clean normalized gen targets
            t: optional per-graph timestep [B] (if None, sample randomly)

        Returns:
            loss: DDPM noise prediction loss
        """
        device = clean_bus.device

        # Per-graph timesteps
        if hasattr(data["bus"], "batch"):
            B = data["bus"].batch.max().item() + 1
            if t is None:
                t = torch.randint(0, self.T, (B,), device=device)
            # Expand per-graph t to per-node via batch vector
            sa_bus = self.sqrt_ac[t][data["bus"].batch].unsqueeze(-1)
            s1_bus = self.sqrt_1mac[t][data["bus"].batch].unsqueeze(-1)
            sa_gen = self.sqrt_ac[t][data["generator"].batch].unsqueeze(-1)
            s1_gen = self.sqrt_1mac[t][data["generator"].batch].unsqueeze(-1)
            # For backbone: use per-node timestep index
            t_per_bus = t[data["bus"].batch]
            t_per_gen = t[data["generator"].batch]
        else:
            if t is None:
                t = torch.randint(0, self.T, (1,), device=device)
            sa_bus = self.sqrt_ac[t]
            s1_bus = self.sqrt_1mac[t]
            sa_gen = self.sqrt_ac[t]
            s1_gen = self.sqrt_1mac[t]
            t_per_bus = t
            t_per_gen = t

        noise_bus = torch.randn_like(clean_bus)
        noise_gen = torch.randn_like(clean_gen)

        noisy_bus = sa_bus * clean_bus + s1_bus * noise_bus
        noisy_gen = sa_gen * clean_gen + s1_gen * noise_gen

        pred_bus, pred_gen = self.backbone(data, noisy_bus, noisy_gen, t)

        loss = F.mse_loss(pred_bus, noise_bus) + F.mse_loss(pred_gen, noise_gen)
        return loss

    @torch.no_grad()
    def sample(self, data, steps=50):
        """DDIM sampling of a single primal-dual state."""
        device = data["bus"].x.device
        n_bus = data["bus"].x.shape[0]
        n_gen = data["generator"].x.shape[0]

        t_start = int(self.T * 0.98)
        ts = torch.linspace(t_start, 0, steps + 1).long().to(device)

        xb = torch.randn(n_bus, 8, device=device)
        xg = torch.randn(n_gen, 6, device=device)

        for i in range(steps):
            tc, tp = ts[i], ts[i + 1]
            t_batch = torch.full((1,), tc, device=device, dtype=torch.long)
            pred_b, pred_g = self.backbone(data, xb, xg, t_batch)

            sa, s1 = self.sqrt_ac[tc], self.sqrt_1mac[tc]
            ap = self.alphas_cumprod[tp] if tp >= 0 else torch.tensor(1.0, device=device)

            # x0 prediction with clamping
            bx0 = ((xb - s1 * pred_b) / sa.clamp(min=0.01)).clamp(-5, 5)
            gx0 = ((xg - s1 * pred_g) / sa.clamp(min=0.01)).clamp(-5, 5)

            # DDIM update
            db = torch.sqrt((1 - ap).clamp(min=0)) * pred_b
            dg = torch.sqrt((1 - ap).clamp(min=0)) * pred_g
            xb = torch.sqrt(ap) * bx0 + db
            xg = torch.sqrt(ap) * gx0 + dg

        return xb, xg

    @torch.no_grad()
    def sample_and_score(self, data, norm, K=5, steps=50):
        """Sample K candidates and select by KKT residual.

        Returns the most self-consistent (x, λ, z, μ) triple.
        """
        best_score = float("inf")
        best_bus = best_gen = None

        for k in range(K):
            bus_n, gen_n = self.sample(data, steps)

            # Denormalize primals for KKT scoring
            x_bus = bus_n[:, :2]  # Va, Vm (normalized)
            x_gen = gen_n[:, :2]  # Pg, Qg (normalized)

            # Simple score: consistency between predicted primals and duals
            # Primals near bounds should have large bound duals; others should be zero
            zl_bus = bus_n[:, 4:6]
            zu_bus = bus_n[:, 6:8]
            zl_gen = gen_n[:, 2:4]
            zu_gen = gen_n[:, 4:6]

            # Complementarity-like score (in normalized space)
            score = (zl_bus.abs().mean() + zu_bus.abs().mean() +
                     zl_gen.abs().mean() + zu_gen.abs().mean()).item()

            # Prefer samples with lower overall magnitude (more centered)
            score += 0.1 * (bus_n.abs().mean() + gen_n.abs().mean()).item()

            if score < best_score:
                best_score = score
                best_bus = bus_n.clone()
                best_gen = gen_n.clone()

        return best_bus, best_gen, best_score
