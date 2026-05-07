"""
WARP training: HetGNN denoiser with DDPM + physics-informed auxiliary loss.

The model sees: clean graph features + noisy solution + timestep -> predicts noise.
Physics loss is applied on the clamped Tweedie x0_hat estimate with hard cutoff.

Usage:
    PYTHONPATH=. python training/train_warp.py --case case14 --epochs 5 --hidden-dim 64
"""

import argparse
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader

from models.hetgnn import HetGNN
from models.diffusion import GaussianDiffusion
from training.scheduler import get_cosine_schedule_with_warmup
from physics.admittance import build_ybus_from_heterodata
from physics.acpf import ac_power_balance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CASE_NAMES = {
    "case14": "pglib_opf_case14_ieee",
    "case57": "pglib_opf_case57_ieee",
    "case118": "pglib_opf_case118_ieee",
}


def physics_schedule(t, T):
    """Weight for physics loss. Hard cutoff above 0.7*T, sin shape below."""
    t_frac = t.float().mean() / T
    if t_frac > 0.7:
        return 0.0
    return math.sin(math.pi * t_frac / 0.7)


def clamp_solution(bus_x0, gen_x0):
    """Clamp Tweedie estimate to physically plausible ranges."""
    Va = bus_x0[:, 0].clamp(-1.0, 1.0)     # angles rarely exceed ±60°
    Vm = bus_x0[:, 1].clamp(0.8, 1.2)       # voltage magnitudes in [0.8, 1.2] pu
    Pg = gen_x0[:, 0].clamp(-0.5, 10.0)     # active power per-unit
    Qg = gen_x0[:, 1].clamp(-5.0, 5.0)      # reactive power per-unit
    return torch.stack([Va, Vm], dim=-1), torch.stack([Pg, Qg], dim=-1)


def compute_physics_loss(bus_pred, gen_pred, data, G, B):
    """AC power balance residual on predicted solution."""
    Va = bus_pred[:, 0]
    Vm = bus_pred[:, 1]
    Pg = gen_pred[:, 0]
    Qg = gen_pred[:, 1]

    n_bus = Va.shape[0]
    P_inj = torch.zeros(n_bus, device=Va.device)
    Q_inj = torch.zeros(n_bus, device=Va.device)

    gen_bus = data["generator", "generator_link", "bus"].edge_index[1]
    P_inj.scatter_add_(0, gen_bus, Pg)
    Q_inj.scatter_add_(0, gen_bus, Qg)

    load_bus = data["load", "load_link", "bus"].edge_index[1]
    Pd = data["load"].x[:, 0].to(Va.device)
    Qd = data["load"].x[:, 1].to(Va.device)
    P_inj.scatter_add_(0, load_bus, -Pd)
    Q_inj.scatter_add_(0, load_bus, -Qd)

    dP, dQ = ac_power_balance(Vm, Va, P_inj, Q_inj, G.to(Va.device), B.to(Va.device))
    return (dP ** 2 + dQ ** 2).mean()


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    case_name = CASE_NAMES.get(args.case, args.case)

    logger.info(f"Loading {case_name}...")
    train_ds = OPFDataset(root=args.data_root, case_name=case_name, split="train",
                          num_groups=args.num_groups)
    val_ds = OPFDataset(root=args.data_root, case_name=case_name, split="val",
                        num_groups=args.num_groups)
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = HetGNN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pe_dim=args.pe_dim,
    ).to(device)

    diffusion = GaussianDiffusion(T=args.T, schedule="cosine").to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} params, T={args.T}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(500, total_steps // 20),
        num_training_steps=total_steps,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_ddpm = float("inf")

    for epoch in range(args.epochs):
        model.train()
        sum_ddpm = 0
        sum_phy = 0
        n = 0
        n_phy_applied = 0
        t0 = time.time()

        for data in train_loader:
            data = data.to(device)

            bus_y = data["bus"].y       # [n_bus, 2] = (Va, Vm)
            gen_y = data["generator"].y  # [n_gen, 2] = (Pg, Qg)

            noise_bus = torch.randn_like(bus_y)
            noise_gen = torch.randn_like(gen_y)

            t = torch.randint(0, args.T, (1,), device=device)

            sqrt_a = diffusion.sqrt_alphas_cumprod[t]
            sqrt_1ma = diffusion.sqrt_one_minus_alphas_cumprod[t]

            bus_noisy = sqrt_a * bus_y + sqrt_1ma * noise_bus
            gen_noisy = sqrt_a * gen_y + sqrt_1ma * noise_gen

            # Model sees clean features + noisy solution + timestep
            bus_pred, gen_pred = model(data, t,
                                       bus_noisy=bus_noisy,
                                       gen_noisy=gen_noisy)

            # Noise prediction MSE
            L_ddpm = F.mse_loss(bus_pred, noise_bus) + F.mse_loss(gen_pred, noise_gen)

            # Physics loss with safeguards
            phy_w = physics_schedule(t, args.T)
            if phy_w > 0.01:
                bus_x0_hat = (bus_noisy - sqrt_1ma * bus_pred) / sqrt_a.clamp(min=1e-4)
                gen_x0_hat = (gen_noisy - sqrt_1ma * gen_pred) / sqrt_a.clamp(min=1e-4)
                bus_x0_hat, gen_x0_hat = clamp_solution(bus_x0_hat, gen_x0_hat)

                G, B = build_ybus_from_heterodata(data)
                L_phy = compute_physics_loss(bus_x0_hat, gen_x0_hat, data, G, B)
                L_phy = L_phy.clamp(max=100.0)

                loss = L_ddpm + args.lambda_phy * phy_w * L_phy
                sum_phy += L_phy.item()
                n_phy_applied += 1
            else:
                loss = L_ddpm

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            sum_ddpm += L_ddpm.item()
            n += 1

            if n % 1000 == 0:
                avg_d = sum_ddpm / n
                avg_p = sum_phy / max(n_phy_applied, 1)
                logger.info(f"  [{n}/{len(train_loader)}] L_ddpm={avg_d:.4f} "
                            f"L_phy={avg_p:.4f} (applied {n_phy_applied}/{n})")

        elapsed = time.time() - t0
        avg_ddpm = sum_ddpm / max(n, 1)
        avg_phy = sum_phy / max(n_phy_applied, 1)

        # Validation: compute L_ddpm on val set
        val_ddpm = evaluate(model, diffusion, val_loader, device, args.T)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train L_ddpm: {avg_ddpm:.4f} | Val L_ddpm: {val_ddpm:.4f} | "
            f"L_phy: {avg_phy:.4f} | Phy applied: {n_phy_applied}/{n} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {elapsed:.1f}s"
        )

        if val_ddpm < best_ddpm:
            best_ddpm = val_ddpm
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "diffusion_state_dict": diffusion.state_dict(),
                "val_ddpm": val_ddpm,
                "args": vars(args),
            }, ckpt_dir / "best.pt")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "diffusion_state_dict": diffusion.state_dict(),
            "args": vars(args),
        }, ckpt_dir / "latest.pt")

    logger.info(f"Best val L_ddpm: {best_ddpm:.4f}")
    return best_ddpm


@torch.no_grad()
def evaluate(model, diffusion, loader, device, T, max_batches=300):
    model.eval()
    total = 0
    n = 0
    for data in loader:
        data = data.to(device)
        bus_y = data["bus"].y
        gen_y = data["generator"].y

        noise_bus = torch.randn_like(bus_y)
        noise_gen = torch.randn_like(gen_y)
        t = torch.randint(0, T, (1,), device=device)

        sqrt_a = diffusion.sqrt_alphas_cumprod[t]
        sqrt_1ma = diffusion.sqrt_one_minus_alphas_cumprod[t]

        bus_noisy = sqrt_a * bus_y + sqrt_1ma * noise_bus
        gen_noisy = sqrt_a * gen_y + sqrt_1ma * noise_gen

        bus_pred, gen_pred = model(data, t, bus_noisy=bus_noisy, gen_noisy=gen_noisy)
        L = F.mse_loss(bus_pred, noise_bus) + F.mse_loss(gen_pred, noise_gen)
        total += L.item()
        n += 1
        if n >= max_batches:
            break
    return total / max(n, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14")
    parser.add_argument("--data-root", default="data/opfdata")
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--pe-dim", type=int, default=16)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-phy", type=float, default=0.1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="results/checkpoints/warp_case14")
    args = parser.parse_args()
    train(args)
