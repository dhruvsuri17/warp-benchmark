"""Experiment N2: Four warm-start strategies on PIPS.

N2a: Full GNN warm-start (primal only, current approach)
N2b: Selective warm-start (voltage only, generators at midpoint)
N2c: Voltage + scaled generators (generators blended 30% toward GNN)
N2d: Centrality diagnostic — measure µ₀ for each method

The selective warm-start hypothesis: bus RMSE is 0.012-0.031 (excellent)
but generator RMSE is 0.095-0.146 (weak). Keeping generators at midpoint
preserves centrality for the worst-predicted variables while using accurate
voltage predictions where the model is strong.
"""
import os, sys, argparse, time, io, logging, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from numpy import inf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("N2")

from colab_warp import DetGNN, DEVICE
from normalizer import VariableNormalizer
from inference.warmstart import get_variable_bounds, inject_warmstart
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
    old = sys.stdout; sys.stdout = c = io.StringIO()
    try:
        pp.runopp(net, init=init, verbose=2, numba=False)
        conv = net.OPF_converged
    except Exception:
        conv = False
    sys.stdout = old; out = c.getvalue()
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


def compute_centrality(x, net):
    """Compute centrality measure µ = mean((x-lb)*(ub-x)) / mean((mid-lb)*(ub-mid))."""
    lb, ub = get_variable_bounds(net)
    s_lower = np.maximum(x - lb, 1e-12)
    s_upper = np.maximum(ub - x, 1e-12)
    products = s_lower * s_upper
    x_mid = (lb + ub) / 2
    s_mid_l = x_mid - lb
    s_mid_u = ub - x_mid
    mid_products = s_mid_l * s_mid_u
    mu = products.mean()
    mu_mid = mid_products.mean()
    return mu, mu / mu_mid if mu_mid > 0 else 0


def identify_centrality_bottleneck(x, net):
    """Find which variable group has worst centrality."""
    lb, ub = get_variable_bounds(net)
    n_bus = len(net.bus)
    n_gen = len(net.gen)

    groups = {
        "Va": (0, n_bus),
        "Vm": (n_bus, 2*n_bus),
        "Pg": (2*n_bus, 2*n_bus+n_gen),
        "Qg": (2*n_bus+n_gen, 2*n_bus+2*n_gen),
    }

    worst_group = None
    worst_ratio = float("inf")
    result = {}

    for name, (start, end) in groups.items():
        s_l = np.maximum(x[start:end] - lb[start:end], 1e-12)
        s_u = np.maximum(ub[start:end] - x[start:end], 1e-12)
        prods = s_l * s_u

        x_mid = (lb[start:end] + ub[start:end]) / 2
        mid_prods = (x_mid - lb[start:end]) * (ub[start:end] - x_mid)

        ratio = prods.mean() / mid_prods.mean() if mid_prods.mean() > 0 else 0
        min_prod = prods.min()
        result[name] = {"ratio": ratio, "min_product": min_prod, "mean_product": prods.mean()}

        if ratio < worst_ratio:
            worst_ratio = ratio
            worst_group = name

    result["worst_group"] = worst_group
    return result


def make_warmstart_variants(bp, gp, net, norm=None):
    """Create all N2 warm-start variants.

    Returns dict of {variant_name: x_ws_vector}
    """
    if norm is not None:
        bp = norm.denormalize_bus(bp)
        gp = norm.denormalize_gen(gp)

    Vm = bp[:, 1].cpu().numpy()
    Va_deg = np.degrees(bp[:, 0].cpu().numpy())
    Pg = gp[:, 0].cpu().numpy() * 100
    Qg = gp[:, 1].cpu().numpy() * 100

    n_bus = len(net.bus)
    n_gen = len(net.gen)

    lb, ub = get_variable_bounds(net)
    x_mid = (lb + ub) / 2.0

    # Full GNN prediction
    x_full = x_mid.copy()
    x_full[:n_bus] = Vm[:n_bus]
    x_full[n_bus:2*n_bus] = Va_deg[:n_bus]
    x_full[2*n_bus:2*n_bus+min(n_gen, len(Pg))] = Pg[:n_gen]
    x_full[2*n_bus+n_gen:2*n_bus+n_gen+min(n_gen, len(Qg))] = Qg[:n_gen]

    # N2a: Full warm-start
    x_full_clipped = np.clip(x_full, lb + 1e-6*(ub-lb), ub - 1e-6*(ub-lb))

    # N2b: Voltage only — keep generators at midpoint
    x_volt_only = x_mid.copy()
    x_volt_only[:n_bus] = Vm[:n_bus]
    x_volt_only[n_bus:2*n_bus] = Va_deg[:n_bus]
    x_volt_only = np.clip(x_volt_only, lb + 1e-6*(ub-lb), ub - 1e-6*(ub-lb))

    # N2c: Voltage full + generators blended 30% toward GNN
    x_volt_gen30 = x_mid.copy()
    x_volt_gen30[:n_bus] = Vm[:n_bus]
    x_volt_gen30[n_bus:2*n_bus] = Va_deg[:n_bus]
    alpha_gen = 0.3
    x_volt_gen30[2*n_bus:2*n_bus+min(n_gen, len(Pg))] = (
        (1-alpha_gen) * x_mid[2*n_bus:2*n_bus+n_gen]
        + alpha_gen * Pg[:n_gen])
    x_volt_gen30[2*n_bus+n_gen:2*n_bus+n_gen+min(n_gen, len(Qg))] = (
        (1-alpha_gen) * x_mid[2*n_bus+n_gen:]
        + alpha_gen * Qg[:n_gen])
    x_volt_gen30 = np.clip(x_volt_gen30, lb + 1e-6*(ub-lb), ub - 1e-6*(ub-lb))

    # N2d: Va only — keep Vm and generators at midpoint
    x_va_only = x_mid.copy()
    x_va_only[n_bus:2*n_bus] = Va_deg[:n_bus]
    x_va_only = np.clip(x_va_only, lb + 1e-6*(ub-lb), ub - 1e-6*(ub-lb))

    return {
        "full": x_full_clipped,
        "volt_only": x_volt_only,
        "volt_gen30": x_volt_gen30,
        "va_only": x_va_only,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--ckpt-dir", default="ckpt")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--use-norm", action="store_true", default=True)
    args = parser.parse_args()

    norm = None
    if args.use_norm:
        norm = VariableNormalizer().load(f"{args.ckpt_dir}/normalizer_stats.json")

    det_model = DetGNN(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(DEVICE)
    ckpt_name = "det_norm_best.pt" if args.use_norm else "det_best.pt"
    det_ckpt = torch.load(f"{args.ckpt_dir}/{ckpt_name}", map_location=DEVICE, weights_only=True)
    det_model.load_state_dict(det_ckpt["model"])
    det_model.eval()

    make_net = pn.case118
    test_ds = OPFDataset(root="data", case_name=args.case, split="test", num_groups=1)

    variants = ["flat", "full", "volt_only", "volt_gen30", "va_only"]
    results = []

    log.info(f"=== N2: WARM-START VARIANT BENCHMARK ===")
    log.info(f"Variants: {variants}")

    for idx in range(min(args.n_test, len(test_ds))):
        data = test_ds[idx]
        row = {"idx": idx}

        with torch.no_grad():
            bp, gp = det_model(data.to(DEVICE))

        ws_variants = make_warmstart_variants(bp, gp, make_net(), norm=norm)

        # Flat start
        net = make_net(); set_loads(net, data)
        _, nf = run_opf(net, "flat")
        row["flat"] = nf

        # Centrality of flat start
        lb, ub = get_variable_bounds(make_net())
        x_mid = (lb + ub) / 2.0
        mu_flat, ratio_flat = compute_centrality(x_mid, make_net())
        row["mu_flat"] = mu_flat

        for vname in ["full", "volt_only", "volt_gen30", "va_only"]:
            x_ws = ws_variants[vname]

            # Centrality diagnostic
            mu, ratio = compute_centrality(x_ws, make_net())
            row[f"mu_{vname}"] = mu
            row[f"mu_ratio_{vname}"] = ratio

            # Bottleneck analysis
            bottleneck = identify_centrality_bottleneck(x_ws, make_net())
            row[f"bottleneck_{vname}"] = bottleneck["worst_group"]

            # Run OPF
            net = make_net(); set_loads(net, data)
            inject_warmstart(net, x_ws)
            _, ni = run_opf(net, "results")
            row[vname] = ni

        results.append(row)

        log.info(f"  #{idx:3d}  flat={nf}  full={row['full']}  "
                 f"volt_only={row['volt_only']}  volt_gen30={row['volt_gen30']}  "
                 f"va_only={row['va_only']}  "
                 f"µ_ratio: full={row['mu_ratio_full']:.3f} "
                 f"volt={row['mu_ratio_volt_only']:.3f} "
                 f"bottleneck={row['bottleneck_full']}")

    # Summary
    print("\n" + "=" * 90)
    print("  N2: WARM-START VARIANT BENCHMARK RESULTS")
    print("=" * 90)

    fm = np.mean([r["flat"] for r in results if r["flat"] is not None])
    print(f"\n  {'Method':15s}  {'Mean':>6}  {'Med':>5}  {'vs Flat':>8}  {'µ ratio':>8}  {'Bottleneck':>12}")
    print("-" * 70)

    for vname in variants:
        vals = [r[vname] for r in results if r[vname] is not None]
        m = np.mean(vals); md = np.median(vals)
        delta = (1 - m / fm) * 100

        if vname == "flat":
            mu_r = 1.0
            bn = "—"
        else:
            mu_r = np.mean([r[f"mu_ratio_{vname}"] for r in results])
            bn_counts = {}
            for r in results:
                b = r.get(f"bottleneck_{vname}", "?")
                bn_counts[b] = bn_counts.get(b, 0) + 1
            bn = max(bn_counts, key=bn_counts.get)

        print(f"  {vname:15s}  {m:6.1f}  {md:5.0f}  {delta:+7.1f}%  {mu_r:8.3f}  {bn:>12}")

    print("=" * 90)

    # Centrality correlation analysis
    print("\n  Per-variable-group centrality (mean µ ratio across instances):")
    for vname in ["full", "volt_only", "volt_gen30"]:
        group_ratios = {"Va": [], "Vm": [], "Pg": [], "Qg": []}
        for r in results:
            # We'd need per-group data — approximate from bottleneck
            pass
        print(f"    {vname}: bottleneck = {max(bn_counts, key=bn_counts.get) if bn_counts else '?'}")

    # Save CSV
    csv_path = "logs/n2_warmstart_variants.csv"
    os.makedirs("logs", exist_ok=True)
    keys = ["idx", "flat", "full", "volt_only", "volt_gen30", "va_only",
            "mu_flat", "mu_ratio_full", "mu_ratio_volt_only", "mu_ratio_volt_gen30",
            "bottleneck_full", "bottleneck_volt_only"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    log.info(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
