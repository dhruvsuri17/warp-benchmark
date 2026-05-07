"""Train DetGNN with dual prediction heads.

Uses extracted (x*, lam_g*, zl*, zu*, mu*) labels from IPOPT solves.
Loss = primal_MSE + w_dual * dual_MSE + w_mu * mu_MSE
"""
import os, sys, argparse, time, logging, math
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("TRAIN-DUAL")

from models.det_gnn_dual import DetGNNDual
from normalizer import VariableNormalizer
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DualDataset(torch.utils.data.Dataset):
    """Wraps OPFDataset with dual labels from extracted files."""

    def __init__(self, opf_dataset, duals_dir):
        self.opf = opf_dataset
        self.duals_dir = Path(duals_dir)
        self.valid_indices = []
        for i in range(len(opf_dataset)):
            if (self.duals_dir / f"duals_{i:06d}.pt").exists():
                self.valid_indices.append(i)
        log.info(f"DualDataset: {len(self.valid_indices)}/{len(opf_dataset)} "
                 f"have dual labels in {duals_dir}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        data = self.opf[real_idx]
        duals = torch.load(self.duals_dir / f"duals_{real_idx:06d}.pt",
                           weights_only=True)
        return data, duals


def train_dual_model(args):
    log.info(f"Device: {DEVICE}")
    torch.cuda.set_per_process_memory_fraction(0.4)

    norm = VariableNormalizer().load(f"{args.ckpt_dir}/normalizer_stats.json")

    duals_base = Path(args.duals_dir) / args.case

    train_opf = OPFDataset(root="data", case_name=args.case, split="train",
                           num_groups=args.num_groups)
    val_opf = OPFDataset(root="data", case_name=args.case, split="val",
                         num_groups=args.num_groups)

    train_ds = DualDataset(train_opf, duals_base / "train")
    val_ds = DualDataset(val_opf, duals_base / "val")

    if len(train_ds) == 0:
        log.error("No training instances with dual labels found!")
        return

    # Use regular DataLoader (not PyG) since we return (data, duals) tuples
    # Process one at a time for simplicity
    model = DetGNNDual(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(DEVICE)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def cosine_lr(optimizer, warmup, total):
        def f(step):
            if step < warmup:
                return step / max(1, warmup)
            p = (step - warmup) / max(1, total - warmup)
            return max(0, 0.5 * (1 + math.cos(math.pi * p)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

    total_steps = args.epochs * len(train_ds)
    sched = cosine_lr(opt, min(200, total_steps // 20), total_steps)

    best_val = float("inf")
    Path(args.ckpt_dir).mkdir(exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        s_primal, s_dual, s_mu, n = 0, 0, 0, 0
        t0 = time.time()

        indices = np.random.permutation(len(train_ds))
        for ii in indices:
            data, duals = train_ds[ii]
            data = data.to(DEVICE)

            output = model(data)

            # Primal loss (normalized space)
            bp_n = norm.normalize_bus(output["bus_pred"])
            gp_n = norm.normalize_gen(output["gen_pred"])
            by_n = norm.normalize_bus(data["bus"].y)
            gy_n = norm.normalize_gen(data["generator"].y)
            L_primal = F.mse_loss(bp_n, by_n) + F.mse_loss(gp_n, gy_n)

            # Dual loss
            lam_eq_gt = duals["lam_g"][:output["lam_eq"].shape[0] * 2].to(DEVICE)
            lam_eq_pred = output["lam_eq"].flatten()
            L_lam = F.mse_loss(lam_eq_pred, lam_eq_gt[:len(lam_eq_pred)])

            zl_gt = duals["zl"].to(DEVICE)
            zu_gt = duals["zu"].to(DEVICE)
            n_bus = output["zl_bus"].shape[0]
            n_gen = output["zl_gen"].shape[0]

            zl_bus_pred = output["zl_bus"].flatten()
            zu_bus_pred = output["zu_bus"].flatten()
            zl_gen_pred = output["zl_gen"].flatten()
            zu_gen_pred = output["zu_gen"].flatten()

            # Match predictions to ground truth (Va,Vm for bus, Pg,Qg for gen)
            zl_bus_gt = zl_gt[:2*n_bus] if len(zl_gt) >= 2*n_bus else zl_gt
            zu_bus_gt = zu_gt[:2*n_bus] if len(zu_gt) >= 2*n_bus else zu_gt
            zl_gen_gt = zl_gt[2*n_bus:2*n_bus+2*n_gen] if len(zl_gt) >= 2*n_bus+2*n_gen else torch.zeros_like(zl_gen_pred)
            zu_gen_gt = zu_gt[2*n_bus:2*n_bus+2*n_gen] if len(zu_gt) >= 2*n_bus+2*n_gen else torch.zeros_like(zu_gen_pred)

            L_z = (F.mse_loss(zl_bus_pred, zl_bus_gt[:len(zl_bus_pred)])
                   + F.mse_loss(zu_bus_pred, zu_bus_gt[:len(zu_bus_pred)])
                   + F.mse_loss(zl_gen_pred, zl_gen_gt[:len(zl_gen_pred)])
                   + F.mse_loss(zu_gen_pred, zu_gen_gt[:len(zu_gen_pred)]))

            L_dual = L_lam + L_z

            # Mu loss (log scale for positive values)
            mu_gt = duals["mu"].to(DEVICE)
            mu_pred = output["mu"].squeeze()
            L_mu = F.mse_loss(torch.log(mu_pred + 1e-12), torch.log(mu_gt + 1e-12))

            loss = L_primal + args.w_dual * L_dual + args.w_mu * L_mu

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            s_primal += L_primal.item()
            s_dual += L_dual.item()
            s_mu += L_mu.item()
            n += 1

        # Validation
        model.eval()
        v_primal, v_dual, v_mu, vn = 0, 0, 0, 0
        with torch.no_grad():
            for ii in range(min(200, len(val_ds))):
                data, duals = val_ds[ii]
                data = data.to(DEVICE)
                output = model(data)

                bp_n = norm.normalize_bus(output["bus_pred"])
                gp_n = norm.normalize_gen(output["gen_pred"])
                by_n = norm.normalize_bus(data["bus"].y)
                gy_n = norm.normalize_gen(data["generator"].y)
                L_p = F.mse_loss(bp_n, by_n) + F.mse_loss(gp_n, gy_n)

                mu_gt = duals["mu"].to(DEVICE)
                mu_pred = output["mu"].squeeze()
                L_m = F.mse_loss(torch.log(mu_pred + 1e-12), torch.log(mu_gt + 1e-12))

                v_primal += L_p.item()
                v_mu += L_m.item()
                vn += 1

        vl = v_primal / max(vn, 1)
        log.info(f"Ep {ep+1}/{args.epochs} | "
                 f"Train: primal={s_primal/n:.4f} dual={s_dual/n:.4f} mu={s_mu/n:.4f} | "
                 f"Val: primal={vl:.4f} mu={v_mu/max(vn,1):.4f} | "
                 f"{time.time()-t0:.0f}s")

        if vl < best_val:
            best_val = vl
            torch.save({
                "model": model.state_dict(),
                "val_primal": vl,
                "epoch": ep,
            }, f"{args.ckpt_dir}/det_dual_best.pt")

    log.info(f"Best val primal loss: {best_val:.6f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--ckpt-dir", default="ckpt")
    parser.add_argument("--duals-dir", default="data/duals")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--w-dual", type=float, default=0.1)
    parser.add_argument("--w-mu", type=float, default=0.01)
    args = parser.parse_args()

    train_dual_model(args)
