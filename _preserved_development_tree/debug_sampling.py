"""Debug DDIM sampling step by step to find where it breaks."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import math
from colab_warp import (
    HetGNN, GaussianDiffusion, build_ybus,
    DEVICE, log
)
from torch_geometric.datasets import OPFDataset

HIDDEN_DIM = 128
NUM_LAYERS = 6

model = HetGNN(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
diff = GaussianDiffusion().to(DEVICE)
ckpt = torch.load("ckpt/warp_best.pt", map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt["model"])
diff.load_state_dict(ckpt["diff"])
model.eval()

test_ds = OPFDataset(root="data", case_name="pglib_opf_case118_ieee", split="test", num_groups=1)
data = test_ds[0].to(DEVICE)

bus_y = data["bus"].y
gen_y = data["generator"].y
nb_ = bus_y.shape[0]
ng_ = gen_y.shape[0]

print(f"Bus target range: [{bus_y.min():.4f}, {bus_y.max():.4f}], shape={bus_y.shape}")
print(f"Gen target range: [{gen_y.min():.4f}, {gen_y.max():.4f}], shape={gen_y.shape}")

# Test 1: Check if model can predict noise at a low-noise timestep
print("\n=== Test 1: Noise prediction quality at various timesteps ===")
for t_val in [0, 10, 50, 100, 500, 999]:
    t = torch.tensor([t_val], device=DEVICE)
    sa, s1 = diff.sqrt_ac[t], diff.sqrt_1mac[t]
    noise_b = torch.randn_like(bus_y)
    noise_g = torch.randn_like(gen_y)
    bn = sa * bus_y + s1 * noise_b
    gn = sa * gen_y + s1 * noise_g
    with torch.no_grad():
        bp, gp = model(data, t, bus_noisy=bn, gen_noisy=gn)
    noise_err_b = (bp - noise_b).pow(2).mean().sqrt().item()
    noise_err_g = (gp - noise_g).pow(2).mean().sqrt().item()
    x0_b = (bn - s1 * bp) / sa.clamp(min=1e-6)
    x0_err_b = (x0_b - bus_y).pow(2).mean().sqrt().item()
    print(f"  t={t_val:4d} | sa={sa.item():.4f} s1={s1.item():.4f} | "
          f"noise_err_bus={noise_err_b:.4f} noise_err_gen={noise_err_g:.4f} | "
          f"x0_bus_rmse={x0_err_b:.4f}")

# Test 2: DDIM sampling with step-by-step tracking
print("\n=== Test 2: DDIM sampling trajectory ===")
T = diff.T
steps = 30
t_start = int(T * 0.98)
ts = torch.linspace(t_start, 0, steps+1).long().to(DEVICE)

xb = torch.randn(nb_, 2, device=DEVICE)
xg = torch.randn(ng_, 2, device=DEVICE)

print(f"{'Step':>4} {'tc':>4} {'tp':>4} {'sa':>8} {'s1':>8} {'|xb|':>10} {'|bx0|':>10} {'bx0_rmse':>10} {'|xg|':>10} {'gx0_rmse':>10}")
print("-" * 100)

for i in range(steps):
    tc, tp = ts[i], ts[i+1]
    t_batch = torch.full((1,), tc, device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        bp, gp = model(data, t_batch, bus_noisy=xb, gen_noisy=xg)
    sa, s1 = diff.sqrt_ac[tc], diff.sqrt_1mac[tc]
    ap = diff.alphas_cumprod[tp] if tp >= 0 else torch.tensor(1.0, device=DEVICE)
    bx0 = (xb - s1*bp) / sa.clamp(min=0.01)
    gx0 = (xg - s1*gp) / sa.clamp(min=0.01)
    bx0 = torch.stack([bx0[:,0].clamp(-1, 0.5), bx0[:,1].clamp(0.9, 1.1)], -1)
    gx0 = torch.stack([gx0[:,0].clamp(-0.5, 10), gx0[:,1].clamp(-4, 5)], -1)
    db = torch.sqrt((1-ap).clamp(min=0)) * bp
    dg = torch.sqrt((1-ap).clamp(min=0)) * gp
    xb_new = torch.sqrt(ap) * bx0 + db
    xg_new = torch.sqrt(ap) * gx0 + dg

    bx0_rmse = (bx0 - bus_y).pow(2).mean().sqrt().item()
    gx0_rmse = (gx0 - gen_y).pow(2).mean().sqrt().item()
    print(f"{i:4d} {tc.item():4d} {tp.item():4d} {sa.item():8.4f} {s1.item():8.4f} "
          f"{xb.abs().mean().item():10.4f} {bx0.abs().mean().item():10.4f} {bx0_rmse:10.4f} "
          f"{xg.abs().mean().item():10.4f} {gx0_rmse:10.4f}")

    xb, xg = xb_new, xg_new

final_rmse_b = (xb - bus_y).pow(2).mean().sqrt().item()
final_rmse_g = (xg - gen_y).pow(2).mean().sqrt().item()
print(f"\nFinal xb RMSE: {final_rmse_b:.4f}")
print(f"Final xg RMSE: {final_rmse_g:.4f}")
