"""Benchmark for normalized-space models. Denormalizes predictions before injection."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from colab_warp import (
    DetGNN, HetGNN, GaussianDiffusion, DEVICE, log,
    build_ybus, ac_power_balance,
)
from normalizer import VariableNormalizer
from benchmark_v2 import (
    run_opf, set_ws_projected, set_ws_raw, _project,
)
from torch_geometric.datasets import OPFDataset
import time, io

torch.cuda.set_per_process_memory_fraction(0.3)

HIDDEN_DIM = 128
NUM_LAYERS = 6

log.info("Loading normalized models...")
norm = VariableNormalizer().load("ckpt/normalizer_stats.json")
log.info(f"Normalizer stats: {norm.stats}")

det_model = DetGNN(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
det_ckpt = torch.load("ckpt/det_norm_best.pt", map_location=DEVICE, weights_only=True)
det_model.load_state_dict(det_ckpt["model"])
det_model.eval()
log.info(f"DetGNN val_loss={det_ckpt['val_loss']:.6f}")

warp_model = HetGNN(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
warp_diff = GaussianDiffusion().to(DEVICE)
warp_ckpt = torch.load("ckpt/warp_norm_best.pt", map_location=DEVICE, weights_only=True)
warp_model.load_state_dict(warp_ckpt["model"])
warp_diff.load_state_dict(warp_ckpt["diff"])
warp_model.eval()
log.info(f"WARP val_ddpm={warp_ckpt.get('val', '?')}")


@torch.no_grad()
def sample_and_score_norm(model, diff, norm, data, K=5, steps=30):
    """DDIM sampling in normalized space, denormalize for scoring."""
    model.eval()
    data = data.to(DEVICE)
    nb_ = data["bus"].x.shape[0]
    ng_ = data["generator"].x.shape[0]
    T = diff.T
    t_start = int(T * 0.98)
    ts = torch.linspace(t_start, 0, steps + 1).long().to(DEVICE)

    best_score, best_bus, best_gen = float("inf"), None, None
    gb = data["generator", "generator_link", "bus"].edge_index[1]
    lb = data["load", "load_link", "bus"].edge_index[1]
    Pd, Qd = data["load"].x[:, 0], data["load"].x[:, 1]
    G, B = build_ybus(data)

    for k in range(K):
        xb = torch.randn(nb_, 2, device=DEVICE)
        xg = torch.randn(ng_, 2, device=DEVICE)

        for i in range(steps):
            tc, tp = ts[i], ts[i + 1]
            t_batch = torch.full((1,), tc, device=DEVICE, dtype=torch.long)
            bp, gp = model(data, t_batch, bus_noisy=xb, gen_noisy=xg)
            sa, s1 = diff.sqrt_ac[tc], diff.sqrt_1mac[tc]
            ap = diff.alphas_cumprod[tp] if tp >= 0 else torch.tensor(1.0, device=DEVICE)
            bx0 = ((xb - s1 * bp) / sa.clamp(min=0.01)).clamp(-3, 3)
            gx0 = ((xg - s1 * gp) / sa.clamp(min=0.01)).clamp(-3, 3)
            db = torch.sqrt((1 - ap).clamp(min=0)) * bp
            dg = torch.sqrt((1 - ap).clamp(min=0)) * gp
            xb = torch.sqrt(ap) * bx0 + db
            xg = torch.sqrt(ap) * gx0 + dg

        bx_raw = norm.denormalize_bus(xb)
        gx_raw = norm.denormalize_gen(xg)

        Pi = torch.zeros(nb_, device=DEVICE)
        Qi = torch.zeros(nb_, device=DEVICE)
        Pi.scatter_add_(0, gb, gx_raw[:, 0])
        Qi.scatter_add_(0, gb, gx_raw[:, 1])
        Pi.scatter_add_(0, lb, -Pd)
        Qi.scatter_add_(0, lb, -Qd)
        dP, dQ = ac_power_balance(bx_raw[:, 1], bx_raw[:, 0], Pi, Qi, G, B)
        score = (dP ** 2 + dQ ** 2).sum().item()

        if score < best_score:
            best_score = score
            best_bus, best_gen = bx_raw.clone(), gx_raw.clone()

    return best_bus, best_gen, best_score


import pandapower as pp
import pandapower.networks as pn
import pandapower.pypower.opf_execute as _opf_exec

_orig_solver = _opf_exec.pipsopf_solver
def _patched_solver(om, ppopt, out_opt=None):
    if ppopt.get('INIT') == 'results':
        ppopt = dict(ppopt)
        ppopt['INIT'] = 'pf'
    return _orig_solver(om, ppopt, out_opt)
_opf_exec.pipsopf_solver = _patched_solver

test_ds = OPFDataset(root="data", case_name="pglib_opf_case118_ieee", split="test", num_groups=1)
make_net = pn.case118
N_TEST = 50

def set_loads(net, data):
    Pd = data["load"].x[:, 0].cpu().numpy() * 100
    Qd = data["load"].x[:, 1].cpu().numpy() * 100
    for i in range(min(len(net.load), len(Pd))):
        net.load.at[i, "p_mw"] = Pd[i]
        net.load.at[i, "q_mvar"] = Qd[i]

log.info(f"=== BENCHMARK (normalized models, eps=0.02) ===")
flat_i, det_raw_i, det_proj_i, warp_raw_i, warp_proj_i = [], [], [], [], []

for idx in range(min(N_TEST, len(test_ds))):
    data = test_ds[idx]

    net = make_net(); set_loads(net, data)
    _, nf, _ = run_opf(net, pp, "flat")

    with torch.no_grad():
        bp_n, gp_n = det_model(data.to(DEVICE))
    bp = norm.denormalize_bus(bp_n)
    gp = norm.denormalize_gen(gp_n)

    net = make_net(); set_loads(net, data); set_ws_raw(net, bp, gp)
    _, nd_raw, _ = run_opf(net, pp, "results")

    net = make_net(); set_loads(net, data); set_ws_projected(net, bp, gp, eps=0.02)
    _, nd_proj, _ = run_opf(net, pp, "results")

    bb, bg, _ = sample_and_score_norm(warp_model, warp_diff, norm, data, K=3, steps=30)

    net = make_net(); set_loads(net, data); set_ws_raw(net, bb, bg)
    _, nw_raw, _ = run_opf(net, pp, "results")

    net = make_net(); set_loads(net, data); set_ws_projected(net, bb, bg, eps=0.02)
    _, nw_proj, _ = run_opf(net, pp, "results")

    flat_i.append(nf); det_raw_i.append(nd_raw); det_proj_i.append(nd_proj)
    warp_raw_i.append(nw_raw); warp_proj_i.append(nw_proj)

    bus_rmse = (bp - data["bus"].y.to(DEVICE)).pow(2).mean().sqrt().item()
    gen_rmse = (gp - data["generator"].y.to(DEVICE)).pow(2).mean().sqrt().item()
    log.info(f"  #{idx:3d}  Flat={nf}  Det_raw={nd_raw}  Det_proj={nd_proj}  "
             f"WARP_raw={nw_raw}  WARP_proj={nw_proj}  rmse=({bus_rmse:.3f},{gen_rmse:.3f})")

def _s(lst):
    v = [x for x in lst if x is not None]
    return np.mean(v), np.median(v) if v else (float('nan'), float('nan'))

fm = _s(flat_i)[0]
print("\n" + "=" * 70)
print("  NORMALIZED MODEL BENCHMARK RESULTS")
print("=" * 70)
for label, vals in [("Flat", flat_i), ("Det raw", det_raw_i), ("Det proj", det_proj_i),
                    ("WARP raw", warp_raw_i), ("WARP proj", warp_proj_i)]:
    m, md = _s(vals)
    print(f"  {label:12s}  mean={m:5.1f}  median={md:4.0f}  vs Flat: {(1-m/fm)*100:+.1f}%")
print("=" * 70)
