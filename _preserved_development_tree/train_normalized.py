"""WARP training with per-variable normalization.

Trains in normalized space where all target variables are ~N(0,1).
This fixes the scale mismatch that causes DDIM sampling to struggle
with generator variables (Pg range [0, 9.2] vs Va range [-0.5, 0.2]).

Usage:
    python train_normalized.py [--epochs 30] [--hidden-dim 128] [--lam-phy 0.1]
"""
import os, sys, argparse, math, time, logging
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("WARP-NORM")

from colab_warp import (
    HetGNN, GaussianDiffusion, DetGNN,
    build_ybus, physics_loss, ac_power_balance,
    cosine_lr, phy_sched, clamp_sol,
    DEVICE,
)
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from normalizer import VariableNormalizer


def train_det_normalized(norm, case, epochs, hidden_dim, num_layers, num_groups, lr):
    """Train DetGNN in normalized target space."""
    log.info(f"=== Training DetGNN (normalized) on {case} ===")
    train_ds = OPFDataset(root="data", case_name=case, split="train", num_groups=num_groups)
    val_ds = OPFDataset(root="data", case_name=case, split="val", num_groups=num_groups)
    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    model = DetGNN(hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    total_steps = epochs * len(train_loader)
    sched = cosine_lr(opt, min(500, total_steps // 20), total_steps)

    best_val = float("inf")
    Path("ckpt").mkdir(exist_ok=True)

    for ep in range(epochs):
        model.train(); s_loss = 0; n = 0; t0 = time.time()
        for data in train_loader:
            data = data.to(DEVICE)
            bp, gp = model(data)
            by_n = norm.normalize_bus(data["bus"].y)
            gy_n = norm.normalize_gen(data["generator"].y)
            loss = F.mse_loss(bp, by_n) + F.mse_loss(gp, gy_n)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            s_loss += loss.item(); n += 1

        model.eval(); v_loss = 0; v_rmse_raw = 0; vn = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                bp, gp = model(data)
                by_n = norm.normalize_bus(data["bus"].y)
                gy_n = norm.normalize_gen(data["generator"].y)
                v_loss += (F.mse_loss(bp, by_n) + F.mse_loss(gp, gy_n)).item()
                bp_raw = norm.denormalize_bus(bp)
                gp_raw = norm.denormalize_gen(gp)
                v_rmse_raw += (
                    (bp_raw - data["bus"].y).pow(2).mean().sqrt()
                    + (gp_raw - data["generator"].y).pow(2).mean().sqrt()
                ).item() / 2
                vn += 1
                if vn >= 500:
                    break

        vl = v_loss / vn; vr = v_rmse_raw / vn
        log.info(f"DetGNN Ep {ep+1}/{epochs} | Train: {s_loss/n:.6f} | "
                 f"Val: {vl:.6f} | Raw-RMSE: {vr:.4f} | {time.time()-t0:.0f}s")
        if vl < best_val:
            best_val = vl
            torch.save({"model": model.state_dict(), "val_loss": vl, "val_rmse": vr},
                       "ckpt/det_norm_best.pt")

    log.info(f"DetGNN best val: {best_val:.6f}")
    return model


def train_warp_normalized(norm, case, epochs, hidden_dim, num_layers, num_groups,
                          lr, lam_phy, T):
    """Train WARP diffusion in normalized target space."""
    log.info(f"=== Training WARP (normalized) on {case} ===")
    train_ds = OPFDataset(root="data", case_name=case, split="train", num_groups=num_groups)
    val_ds = OPFDataset(root="data", case_name=case, split="val", num_groups=num_groups)
    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    model = HetGNN(hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    diff = GaussianDiffusion(T).to(DEVICE)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    total_steps = epochs * len(train_loader)
    sched = cosine_lr(opt, min(500, total_steps // 20), total_steps)

    best = float("inf")
    Path("ckpt").mkdir(exist_ok=True)

    for ep in range(epochs):
        model.train(); sd = 0; sp = 0; n = 0; np_ = 0; t0 = time.time()
        for data in train_loader:
            data = data.to(DEVICE)
            by_n = norm.normalize_bus(data["bus"].y)
            gy_n = norm.normalize_gen(data["generator"].y)
            nb, ng = torch.randn_like(by_n), torch.randn_like(gy_n)
            t = torch.randint(0, T, (1,), device=DEVICE)
            sa, s1 = diff.sqrt_ac[t], diff.sqrt_1mac[t]
            bn, gn = sa * by_n + s1 * nb, sa * gy_n + s1 * ng

            bp, gp = model(data, t, bus_noisy=bn, gen_noisy=gn)
            Ld = F.mse_loss(bp, nb) + F.mse_loss(gp, ng)

            pw = phy_sched(t, T)
            if pw > 0.01 and lam_phy > 0:
                bx_n = (bn - s1 * bp) / sa.clamp(min=1e-4)
                gx_n = (gn - s1 * gp) / sa.clamp(min=1e-4)
                bx = norm.denormalize_bus(bx_n)
                gx = norm.denormalize_gen(gx_n)
                bx, gx = clamp_sol(bx, gx)
                if hasattr(data["bus"], "batch"):
                    graphs = data.to_data_list()
                    gi = torch.randint(0, len(graphs), (1,)).item()
                    bm = data["bus"].batch == gi
                    gm = data["generator"].batch == gi
                    G, B = build_ybus(graphs[gi])
                    Lp = physics_loss(bx[bm], gx[gm], graphs[gi], G, B).clamp(max=100.0)
                else:
                    G, B = build_ybus(data)
                    Lp = physics_loss(bx, gx, data, G, B).clamp(max=100.0)
                loss = Ld + lam_phy * pw * Lp
                sp += Lp.item(); np_ += 1
            else:
                loss = Ld

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            sd += Ld.item(); n += 1

        model.eval(); vd = 0; vn = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                by_n = norm.normalize_bus(data["bus"].y)
                gy_n = norm.normalize_gen(data["generator"].y)
                nb, ng = torch.randn_like(by_n), torch.randn_like(gy_n)
                t = torch.randint(0, T, (1,), device=DEVICE)
                sa, s1 = diff.sqrt_ac[t], diff.sqrt_1mac[t]
                bp, gp = model(data, t, bus_noisy=sa*by_n+s1*nb, gen_noisy=sa*gy_n+s1*ng)
                vd += (F.mse_loss(bp, nb) + F.mse_loss(gp, ng)).item()
                vn += 1
                if vn >= 500:
                    break

        log.info(f"WARP  Ep {ep+1}/{epochs} | L_ddpm: {sd/n:.4f} | "
                 f"Val: {vd/vn:.4f} | L_phy: {sp/max(np_,1):.4f} | {time.time()-t0:.0f}s")
        if vd / vn < best:
            best = vd / vn
            torch.save({"model": model.state_dict(), "diff": diff.state_dict(),
                        "val": best}, "ckpt/warp_norm_best.pt")

    log.info(f"WARP best val L_ddpm: {best:.4f}")
    return model, diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--det-epochs", type=int, default=50)
    parser.add_argument("--warp-epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-groups", type=int, default=5)
    parser.add_argument("--lr-det", type=float, default=3e-4)
    parser.add_argument("--lr-warp", type=float, default=1e-4)
    parser.add_argument("--lam-phy", type=float, default=0.1)
    parser.add_argument("--T", type=int, default=1000)
    args = parser.parse_args()

    torch.cuda.set_per_process_memory_fraction(0.4)

    log.info("Fitting normalizer on training set...")
    train_ds = OPFDataset(root="data", case_name=args.case, split="train",
                          num_groups=args.num_groups)
    norm = VariableNormalizer().fit(train_ds)
    norm.save("ckpt/normalizer_stats.json")
    log.info(f"Normalizer stats: {norm.stats}")

    det_model = train_det_normalized(
        norm, args.case, args.det_epochs, args.hidden_dim,
        args.num_layers, args.num_groups, args.lr_det)

    warp_model, warp_diff = train_warp_normalized(
        norm, args.case, args.warp_epochs, args.hidden_dim,
        args.num_layers, args.num_groups, args.lr_warp, args.lam_phy, args.T)

    log.info("Training complete. Run benchmark_v2.py with --ckpt-dir ckpt to evaluate.")
