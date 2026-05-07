"""Compare raw DetGNN vs WARP predictions to understand why benchmark results are identical."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from colab_warp import (
    DetGNN, HetGNN, GaussianDiffusion, sample_and_score,
    DEVICE, log
)
from torch_geometric.datasets import OPFDataset

HIDDEN_DIM = 128
NUM_LAYERS = 6

det_model = DetGNN(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
det_ckpt = torch.load("ckpt/det_best.pt", map_location=DEVICE, weights_only=True)
det_model.load_state_dict(det_ckpt["model"])
det_model.eval()

warp_model = HetGNN(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
warp_diff = GaussianDiffusion().to(DEVICE)
warp_ckpt = torch.load("ckpt/warp_best.pt", map_location=DEVICE, weights_only=True)
warp_model.load_state_dict(warp_ckpt["model"])
warp_diff.load_state_dict(warp_ckpt["diff"])
warp_model.eval()

test_ds = OPFDataset(root="data", case_name="pglib_opf_case118_ieee", split="test", num_groups=1)

print(f"\n{'='*80}")
print(f"  Comparing raw predictions: DetGNN vs WARP (before runpp)")
print(f"{'='*80}\n")
print(f"{'Inst':>4}  {'Det bus RMSE':>12}  {'WARP bus RMSE':>13}  {'Det gen RMSE':>12}  {'WARP gen RMSE':>13}  {'bus diff':>10}  {'gen diff':>10}")
print("-" * 90)

for idx in range(20):
    data = test_ds[idx]

    with torch.no_grad():
        bp_det, gp_det = det_model(data.to(DEVICE))

    bp_warp, gp_warp, score = sample_and_score(warp_model, warp_diff, data, K=3, steps=30)

    bus_y = data["bus"].y.to(DEVICE)
    gen_y = data["generator"].y.to(DEVICE)

    det_bus_rmse = (bp_det - bus_y).pow(2).mean().sqrt().item()
    warp_bus_rmse = (bp_warp - bus_y).pow(2).mean().sqrt().item()
    det_gen_rmse = (gp_det - gen_y).pow(2).mean().sqrt().item()
    warp_gen_rmse = (gp_warp - gen_y).pow(2).mean().sqrt().item()

    bus_diff = (bp_det - bp_warp).pow(2).mean().sqrt().item()
    gen_diff = (gp_det - gp_warp).pow(2).mean().sqrt().item()

    print(f"{idx:4d}  {det_bus_rmse:12.6f}  {warp_bus_rmse:13.6f}  {det_gen_rmse:12.6f}  {warp_gen_rmse:13.6f}  {bus_diff:10.6f}  {gen_diff:10.6f}")

print(f"\n{'='*80}")
print("If bus_diff and gen_diff are large but iteration counts are identical,")
print("it means runpp is converging both to the same PF fixed point.")
print(f"{'='*80}")
