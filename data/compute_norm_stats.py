"""Compute per-dimension mean/std over extracted dual-label tensors (training set)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def stack_vectors(duals_dir: Path) -> np.ndarray:
    rows = []
    paths = sorted(duals_dir.glob("duals_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No duals_*.pt in {duals_dir}")
    for p in paths:
        d = torch.load(p, map_location="cpu")
        parts = [
            d["x"].numpy().reshape(-1),
            d["lam_g"].numpy().reshape(-1),
            d["zl"].numpy().reshape(-1),
            d["zu"].numpy().reshape(-1),
            np.array([float(d["mu"].item() if hasattr(d["mu"], "item") else d["mu"])], dtype=np.float32),
        ]
        rows.append(np.concatenate(parts))
    return np.stack(rows, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="e.g. data/case118/train")
    parser.add_argument("--output", type=str, default="data/case118/norm_stats.pt")
    args = parser.parse_args()

    ddir = Path(args.data_dir)
    mat = stack_vectors(ddir)
    mean = torch.tensor(mat.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(mat.std(axis=0).clip(min=1e-8), dtype=torch.float32)
    out = {"mean": mean, "std": std}
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, outp)
    print(f"Saved norm stats shape={mean.shape} to {outp}")


if __name__ == "__main__":
    main()
