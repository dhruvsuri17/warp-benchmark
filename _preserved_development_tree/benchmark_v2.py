"""Benchmark v2: with feasibility projection onto strict interior.

The key insight: PIPS's midpoint (lb+ub)/2 is a well-centered feasible
interior point by design. Model predictions that land near constraint
boundaries are poison for interior-point methods. This benchmark projects
all predictions onto the strict feasible interior before injecting into PIPS.

Usage:
    python benchmark_v2.py [--ckpt-dir ckpt] [--eps 0.02] [--n-test 50]
"""
import os, sys, argparse, time, io, logging
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("BENCH-V2")

from colab_warp import (
    DetGNN, HetGNN, GaussianDiffusion, sample_and_score,
    build_ybus, ac_power_balance,
    DEVICE,
)
from torch_geometric.datasets import OPFDataset


def load_models(ckpt_dir, hidden_dim=128, num_layers=6):
    det_model = DetGNN(hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    det_ckpt = torch.load(f"{ckpt_dir}/det_best.pt", map_location=DEVICE, weights_only=True)
    det_model.load_state_dict(det_ckpt["model"])
    det_model.eval()
    log.info(f"DetGNN val_loss={det_ckpt['val_loss']:.6f} rmse={det_ckpt.get('val_rmse', '?')}")

    warp_model = HetGNN(hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    warp_diff = GaussianDiffusion().to(DEVICE)
    warp_ckpt = torch.load(f"{ckpt_dir}/warp_best.pt", map_location=DEVICE, weights_only=True)
    warp_model.load_state_dict(warp_ckpt["model"])
    warp_diff.load_state_dict(warp_ckpt["diff"])
    warp_model.eval()
    log.info(f"WARP val_ddpm={warp_ckpt.get('val', '?')}")

    return det_model, warp_model, warp_diff


def project_to_feasible_interior(net, bus_pred, gen_pred, eps=0.02):
    """Project predictions onto strict feasible interior [lb+eps*(ub-lb), ub-eps*(ub-lb)].

    This prevents IPM from starting near/outside constraint boundaries.
    eps=0.02 means 2% margin from each bound.
    """
    Vm = bus_pred[:, 1].cpu().numpy()
    Va = bus_pred[:, 0].cpu().numpy()
    Pg = gen_pred[:, 0].cpu().numpy() * 100
    Qg = gen_pred[:, 1].cpu().numpy() * 100

    Vm_min = net.bus["min_vm_pu"].values
    Vm_max = net.bus["max_vm_pu"].values
    Vm = _project(Vm, Vm_min, Vm_max, eps)

    Pg_min = net.gen["min_p_mw"].values
    Pg_max = net.gen["max_p_mw"].values
    Qg_min = net.gen["min_q_mvar"].values
    Qg_max = net.gen["max_q_mvar"].values

    n_gen = min(len(net.gen), len(Pg))
    Pg[:n_gen] = _project(Pg[:n_gen], Pg_min[:n_gen], Pg_max[:n_gen], eps)
    Qg[:n_gen] = _project(Qg[:n_gen], Qg_min[:n_gen], Qg_max[:n_gen], eps)

    return Va, Vm, Pg, Qg


def _project(val, lo, hi, eps):
    margin = eps * (hi - lo)
    lo_inner = lo + margin
    hi_inner = hi - margin
    valid = lo_inner < hi_inner
    lo_c = np.where(valid, lo_inner, (lo + hi) / 2 - 1e-6)
    hi_c = np.where(valid, hi_inner, (lo + hi) / 2 + 1e-6)
    return np.clip(val, lo_c, hi_c)


def set_ws_projected(net, bus_pred, gen_pred, eps=0.02):
    """Set warm-start with feasibility projection."""
    Va, Vm, Pg, Qg = project_to_feasible_interior(net, bus_pred, gen_pred, eps)

    net.res_bus["vm_pu"] = Vm
    net.res_bus["va_degree"] = np.degrees(Va)
    net.res_bus["p_mw"] = 0.0
    net.res_bus["q_mvar"] = 0.0
    for i in range(min(len(net.gen), len(Pg))):
        net.res_gen.at[i, "p_mw"] = Pg[i]
        net.res_gen.at[i, "q_mvar"] = Qg[i]


def set_ws_raw(net, bus_pred, gen_pred):
    """Set warm-start without projection (for comparison)."""
    net.res_bus["vm_pu"] = bus_pred[:, 1].cpu().numpy()
    net.res_bus["va_degree"] = np.degrees(bus_pred[:, 0].cpu().numpy())
    net.res_bus["p_mw"] = 0.0
    net.res_bus["q_mvar"] = 0.0
    Pg = gen_pred[:, 0].cpu().numpy() * 100
    Qg = gen_pred[:, 1].cpu().numpy() * 100
    for i in range(min(len(net.gen), len(Pg))):
        net.res_gen.at[i, "p_mw"] = Pg[i]
        net.res_gen.at[i, "q_mvar"] = Qg[i]


def run_opf(net, pp, init="flat"):
    old = sys.stdout
    sys.stdout = c = io.StringIO()
    t0 = time.time()
    try:
        pp.runopp(net, init=init, verbose=2)
        conv = net.OPF_converged
    except Exception:
        conv = False
    el = time.time() - t0
    sys.stdout = old
    out = c.getvalue()
    ni = None
    for line in out.split("\n"):
        s = line.strip()
        if s and s[0].isdigit():
            parts = s.split()
            if len(parts) >= 3:
                try:
                    int(parts[0]); float(parts[1]); ni = int(parts[0])
                except Exception:
                    pass
    return conv, ni, el


def run_benchmark(det_model, warp_model, warp_diff, case, n_test, eps):
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

    test_ds = OPFDataset(root="data", case_name=case, split="test", num_groups=1)
    case_map = {
        "pglib_opf_case14_ieee": pn.case14,
        "pglib_opf_case57_ieee": pn.case57,
        "pglib_opf_case118_ieee": pn.case118,
    }
    make_net = case_map.get(case, pn.case118)

    def set_loads(net, data):
        Pd = data["load"].x[:, 0].cpu().numpy() * 100
        Qd = data["load"].x[:, 1].cpu().numpy() * 100
        for i in range(min(len(net.load), len(Pd))):
            net.load.at[i, "p_mw"] = Pd[i]
            net.load.at[i, "q_mvar"] = Qd[i]

    results = []

    for idx in range(min(n_test, len(test_ds))):
        data = test_ds[idx]
        row = {"idx": idx}

        # Flat
        net = make_net(); set_loads(net, data)
        _, nf, _ = run_opf(net, pp, "flat")
        row["flat"] = nf

        # DetGNN raw (no projection)
        with torch.no_grad():
            bp, gp = det_model(data.to(DEVICE))
        bus_y = data["bus"].y.to(DEVICE)
        gen_y = data["generator"].y.to(DEVICE)
        row["det_bus_rmse"] = (bp - bus_y).pow(2).mean().sqrt().item()
        row["det_gen_rmse"] = (gp - gen_y).pow(2).mean().sqrt().item()

        net = make_net(); set_loads(net, data); set_ws_raw(net, bp, gp)
        _, nd_raw, _ = run_opf(net, pp, "results")
        row["det_raw"] = nd_raw

        # DetGNN projected
        net = make_net(); set_loads(net, data); set_ws_projected(net, bp, gp, eps)
        _, nd_proj, _ = run_opf(net, pp, "results")
        row["det_proj"] = nd_proj

        # WARP raw
        bb, bg, sc = sample_and_score(warp_model, warp_diff, data, K=3, steps=30)
        row["warp_bus_rmse"] = (bb - bus_y).pow(2).mean().sqrt().item()
        row["warp_gen_rmse"] = (bg - gen_y).pow(2).mean().sqrt().item()

        net = make_net(); set_loads(net, data); set_ws_raw(net, bb, bg)
        _, nw_raw, _ = run_opf(net, pp, "results")
        row["warp_raw"] = nw_raw

        # WARP projected
        net = make_net(); set_loads(net, data); set_ws_projected(net, bb, bg, eps)
        _, nw_proj, _ = run_opf(net, pp, "results")
        row["warp_proj"] = nw_proj

        results.append(row)
        log.info(f"  #{idx:3d}  Flat={nf}  Det_raw={nd_raw}  Det_proj={nd_proj}  "
                 f"WARP_raw={nw_raw}  WARP_proj={nw_proj}  "
                 f"det_rmse=({row['det_bus_rmse']:.3f},{row['det_gen_rmse']:.3f})  "
                 f"warp_rmse=({row['warp_bus_rmse']:.3f},{row['warp_gen_rmse']:.3f})")

    # Summary
    def _stats(key):
        vals = [r[key] for r in results if r[key] is not None]
        return np.mean(vals), np.median(vals) if vals else (float('nan'), float('nan'))

    print("\n" + "=" * 80)
    print("  BENCHMARK V2 RESULTS — with feasibility projection")
    print("=" * 80)
    for label, key in [("Flat start", "flat"),
                       ("DetGNN raw", "det_raw"), ("DetGNN proj", "det_proj"),
                       ("WARP raw", "warp_raw"), ("WARP proj", "warp_proj")]:
        m, md = _stats(key)
        print(f"  {label:15s}  mean={m:5.1f}  median={md:4.0f}")

    fm, _ = _stats("flat")
    for label, key in [("DetGNN raw", "det_raw"), ("DetGNN proj", "det_proj"),
                       ("WARP raw", "warp_raw"), ("WARP proj", "warp_proj")]:
        m, _ = _stats(key)
        print(f"  {label:15s} vs Flat:  {(1 - m / fm) * 100:+.1f}%")
    print("=" * 80)

    # Save per-instance CSV
    import csv
    csv_path = f"logs/benchmark_v2_{case}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    log.info(f"Per-instance results saved to {csv_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="ckpt")
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--n-test", type=int, default=50)
    args = parser.parse_args()

    det_model, warp_model, warp_diff = load_models(
        args.ckpt_dir, args.hidden_dim, args.num_layers)

    run_benchmark(det_model, warp_model, warp_diff,
                  args.case, args.n_test, args.eps)
