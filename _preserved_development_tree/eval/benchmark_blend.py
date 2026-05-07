"""Experiment N1: Centrality blend sweep.

Blends GNN prediction with IPM midpoint at various alpha values.
alpha=0 → pure midpoint (flat start), alpha=1 → pure GNN prediction.

If there exists an alpha* where iterations < 19.6 → centrality hypothesis confirmed.
Expected: U-shaped curve in alpha — minimum at some alpha < 1.

Usage:
    python eval/benchmark_blend.py [--alphas 0.0 0.1 0.2 0.3 0.5 0.7 1.0]
"""
import os, sys, argparse, time, io, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("N1-BLEND")

from colab_warp import DetGNN, HetGNN, GaussianDiffusion, DEVICE
from normalizer import VariableNormalizer
from inference.warmstart import (
    get_variable_bounds, pack_prediction, centred_warmstart, inject_warmstart,
)
from torch_geometric.datasets import OPFDataset

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


def run_opf(net, init="flat"):
    old = sys.stdout
    sys.stdout = c = io.StringIO()
    try:
        pp.runopp(net, init=init, verbose=2, numba=False)
        conv = net.OPF_converged
    except Exception:
        conv = False
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
    return conv, ni


def set_loads(net, data):
    Pd = data["load"].x[:, 0].cpu().numpy() * 100
    Qd = data["load"].x[:, 1].cpu().numpy() * 100
    for i in range(min(len(net.load), len(Pd))):
        net.load.at[i, "p_mw"] = Pd[i]
        net.load.at[i, "q_mvar"] = Qd[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--ckpt-dir", default="ckpt")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--use-norm", action="store_true", default=True,
                        help="Use normalized model checkpoints")
    args = parser.parse_args()

    norm = None
    if args.use_norm:
        norm = VariableNormalizer().load(f"{args.ckpt_dir}/normalizer_stats.json")
        log.info(f"Using normalizer: {norm.stats}")

    det_model = DetGNN(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(DEVICE)
    ckpt_name = "det_norm_best.pt" if args.use_norm else "det_best.pt"
    det_ckpt = torch.load(f"{args.ckpt_dir}/{ckpt_name}", map_location=DEVICE, weights_only=True)
    det_model.load_state_dict(det_ckpt["model"])
    det_model.eval()
    log.info(f"Loaded DetGNN from {ckpt_name}, val_loss={det_ckpt.get('val_loss', '?')}")

    case_map = {
        "pglib_opf_case14_ieee": pn.case14,
        "pglib_opf_case57_ieee": pn.case57,
        "pglib_opf_case118_ieee": pn.case118,
    }
    make_net = case_map.get(args.case, pn.case118)
    test_ds = OPFDataset(root="data", case_name=args.case, split="test", num_groups=1)

    alphas = args.alphas
    results = {a: [] for a in alphas}

    log.info(f"=== N1: CENTRALITY BLEND SWEEP ===")
    log.info(f"Alphas: {alphas}")
    log.info(f"Testing {min(args.n_test, len(test_ds))} instances")

    for idx in range(min(args.n_test, len(test_ds))):
        data = test_ds[idx]

        with torch.no_grad():
            bp, gp = det_model(data.to(DEVICE))

        if norm is not None:
            bp = norm.denormalize_bus(bp)
            gp = norm.denormalize_gen(gp)

        x_hat = pack_prediction(bp, gp, make_net())

        row_str = f"  #{idx:3d}"

        for alpha in alphas:
            net = make_net()
            set_loads(net, data)

            if alpha == 0.0:
                _, ni = run_opf(net, "flat")
            else:
                x_ws = centred_warmstart(x_hat, net, alpha=alpha)
                inject_warmstart(net, x_ws)
                _, ni = run_opf(net, "results")

            results[alpha].append(ni)
            row_str += f"  a={alpha:.2f}:{ni}"

        log.info(row_str)

    print("\n" + "=" * 80)
    print("  EXPERIMENT N1: CENTRALITY BLEND SWEEP")
    print("=" * 80)
    print(f"  {'Alpha':>6}  {'Mean':>6}  {'Median':>7}  {'vs Flat':>8}")
    print("-" * 40)

    flat_mean = np.mean([x for x in results[0.0] if x is not None])
    for alpha in alphas:
        vals = [x for x in results[alpha] if x is not None]
        if vals:
            m = np.mean(vals)
            md = np.median(vals)
            delta = (1 - m / flat_mean) * 100 if flat_mean > 0 else 0
            marker = " <-- BEST" if m == min(
                np.mean([x for x in results[a] if x is not None]) for a in alphas
                if any(x is not None for x in results[a])) else ""
            print(f"  {alpha:6.2f}  {m:6.1f}  {md:7.0f}  {delta:+7.1f}%{marker}")

    print("=" * 80)

    best_alpha = min(alphas, key=lambda a: np.mean([x for x in results[a] if x]))
    best_mean = np.mean([x for x in results[best_alpha] if x])
    print(f"\n  Best alpha: {best_alpha:.2f} (mean={best_mean:.1f})")

    if best_mean < flat_mean:
        print(f"  ✓ CENTRALITY HYPOTHESIS CONFIRMED — alpha={best_alpha:.2f} beats flat-start!")
        print(f"    Reduction: {(1-best_mean/flat_mean)*100:.1f}%")
    elif best_alpha < 1.0 and best_mean < np.mean([x for x in results[1.0] if x]):
        print(f"  ~ Partial confirmation — blending helps vs pure GNN, but doesn't beat flat-start")
    else:
        print(f"  ✗ Centrality hypothesis NOT confirmed — minimum at alpha={best_alpha:.2f}")

    print("=" * 80)

    import csv
    csv_path = "logs/n1_blend_sweep.csv"
    os.makedirs("logs", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx"] + [f"alpha_{a}" for a in alphas])
        for i in range(len(results[alphas[0]])):
            writer.writerow([i] + [results[a][i] for a in alphas])
    log.info(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
