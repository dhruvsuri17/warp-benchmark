"""
Deterministic GNN baseline training on OPFDataset HeteroData.

Usage:
    python training/train_det.py --case case14 --epochs 2 --hidden-dim 64 --num-layers 4
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader

from models.det_gnn import DetGNN
from training.scheduler import get_cosine_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CASE_NAMES = {
    "case14": "pglib_opf_case14_ieee",
    "case57": "pglib_opf_case57_ieee",
    "case118": "pglib_opf_case118_ieee",
}


def compute_loss(bus_pred, gen_pred, data):
    """MSE loss against ground-truth solution."""
    bus_target = data["bus"].y    # [n_bus, 2] = (Va, Vm)
    gen_target = data["generator"].y  # [n_gen, 2] = (Pg, Qg)

    loss_bus = F.mse_loss(bus_pred, bus_target)
    loss_gen = F.mse_loss(gen_pred, gen_target)

    return loss_bus + loss_gen, {"bus": loss_bus.item(), "gen": loss_gen.item()}


def compute_ws_rmse(bus_pred, gen_pred, data):
    """Warm-start RMSE: ||pred - target|| / sqrt(dim)."""
    bus_err = (bus_pred - data["bus"].y).pow(2).mean().sqrt()
    gen_err = (gen_pred - data["generator"].y).pow(2).mean().sqrt()
    return (bus_err + gen_err).item() / 2


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

    # Single-graph dataloader (no batching across graphs for HeteroData)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = DetGNN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pe_dim=args.pe_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n = 0
        t0 = time.time()

        for data in train_loader:
            data = data.to(device)

            bus_pred, gen_pred = model(data)
            loss, _ = compute_loss(bus_pred, gen_pred, data)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n += 1

            if n % 500 == 0:
                logger.info(f"  [{n}/{len(train_loader)}] loss={loss.item():.6f}")

        train_loss = total_loss / max(n, 1)
        elapsed = time.time() - t0

        # Validation
        val_loss, val_rmse = evaluate(model, val_loader, device)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"WS-RMSE: {val_rmse:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "args": vars(args),
            }, ckpt_dir / "best.pt")

    logger.info(f"Best val loss: {best_val:.6f}")
    return best_val


@torch.no_grad()
def evaluate(model, loader, device, max_batches=500):
    model.eval()
    total_loss = 0
    total_rmse = 0
    n = 0

    for data in loader:
        data = data.to(device)
        bus_pred, gen_pred = model(data)
        loss, _ = compute_loss(bus_pred, gen_pred, data)
        rmse = compute_ws_rmse(bus_pred, gen_pred, data)
        total_loss += loss.item()
        total_rmse += rmse
        n += 1
        if n >= max_batches:
            break

    return total_loss / max(n, 1), total_rmse / max(n, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14")
    parser.add_argument("--data-root", default="data/opfdata")
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--pe-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="results/checkpoints/det_gnn_case14")
    args = parser.parse_args()
    train(args)
