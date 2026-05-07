"""Direction 1: Active constraint screening using DetGNN predictions.

Instead of warm-starting x₀, use DetGNN's predictions to identify and remove
non-binding constraints before flat-starting the reduced problem. This
sidesteps the IPM centrality problem entirely.

Strategy: if DetGNN predicts a variable is far from its bound, that constraint
is unlikely to be binding at the optimum. Remove it → fewer constraints →
fewer IPM iterations.

Based on Pineda-Morales 2020, Park-Van Hentenryck 2022.
Literature: ~20% iteration reduction, 50-80% constraint removal on case118.
"""
import os, sys, argparse, time, io, logging, csv, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("CS")

from colab_warp import DetGNN, DEVICE
from normalizer import VariableNormalizer
from torch_geometric.datasets import OPFDataset

import pandapower as pp
import pandapower.networks as pn


def run_opf(net, init="flat"):
    old = sys.stdout; sys.stdout = c = io.StringIO()
    try:
        pp.runopp(net, init=init, verbose=2, numba=False)
        conv = net.OPF_converged
    except Exception as e:
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
    obj = None
    if conv:
        try:
            obj = net.res_cost
        except Exception:
            pass
    return conv, ni, obj


def set_loads(net, data):
    Pd = data["load"].x[:, 0].cpu().numpy() * 100
    Qd = data["load"].x[:, 1].cpu().numpy() * 100
    for i in range(min(len(net.load), len(Pd))):
        net.load.at[i, "p_mw"] = Pd[i]
        net.load.at[i, "q_mvar"] = Qd[i]


def compute_line_loading(Vm, Va, net):
    """Estimate line apparent power flow from voltage predictions."""
    n_bus = len(net.bus)
    S_from = np.zeros(len(net.line))

    for idx, line in net.line.iterrows():
        fb = int(line["from_bus"])
        tb = int(line["to_bus"])
        if fb >= n_bus or tb >= n_bus:
            continue

        r = line["r_ohm_per_km"] * line["length_km"]
        x = line["x_ohm_per_km"] * line["length_km"]
        vn = net.bus.at[fb, "vn_kv"]
        if vn == 0:
            continue
        z_base = vn ** 2 / (net.sn_mva if hasattr(net, 'sn_mva') else 100.0)
        r_pu = r / z_base if z_base > 0 else 0
        x_pu = x / z_base if z_base > 0 else 0

        z_sq = r_pu**2 + x_pu**2
        if z_sq < 1e-12:
            continue
        g = r_pu / z_sq
        b = -x_pu / z_sq

        Vi = Vm[fb]; Vj = Vm[tb]
        theta = Va[fb] - Va[tb]
        P_ij = Vi**2 * g - Vi * Vj * (g * np.cos(theta) + b * np.sin(theta))
        Q_ij = -Vi**2 * (b + line.get("c_nf_per_km", 0) * line["length_km"] * 1e-9 * 2 * np.pi * 50 * z_base / 2) \
               - Vi * Vj * (-b * np.cos(theta) + g * np.sin(theta))
        S_from[idx] = np.sqrt(P_ij**2 + Q_ij**2) * (net.sn_mva if hasattr(net, 'sn_mva') else 100.0)

    return S_from


def screen_constraints(net, Vm, Va, Pg, Qg, vm_margin=0.02, line_threshold=0.7,
                       gen_margin_frac=0.1):
    """Remove predicted-non-binding constraints from net.

    Returns: (net_reduced, stats_dict)
    """
    n_bus = len(net.bus)
    n_gen = len(net.gen)
    n_line = len(net.line)
    stats = {"vm_upper_removed": 0, "vm_lower_removed": 0,
             "line_removed": 0, "pg_upper_removed": 0, "pg_lower_removed": 0,
             "qg_upper_removed": 0, "qg_lower_removed": 0,
             "total_original": 0, "total_removed": 0}

    # Voltage magnitude screening
    for i in range(min(n_bus, len(Vm))):
        vm_pred = Vm[i]
        vm_max = net.bus.at[i, "max_vm_pu"]
        vm_min = net.bus.at[i, "min_vm_pu"]
        stats["total_original"] += 2

        if vm_pred < vm_max - vm_margin:
            net.bus.at[i, "max_vm_pu"] = 2.0
            stats["vm_upper_removed"] += 1
            stats["total_removed"] += 1

        if vm_pred > vm_min + vm_margin:
            net.bus.at[i, "min_vm_pu"] = 0.0
            stats["vm_lower_removed"] += 1
            stats["total_removed"] += 1

    # Line thermal screening
    S_pred = compute_line_loading(Vm, Va, net)
    for i in range(n_line):
        stats["total_original"] += 1
        max_load = net.line.at[i, "max_loading_percent"]
        if max_load <= 0 or max_load >= 999:
            continue
        sn_mva = net.sn_mva if hasattr(net, 'sn_mva') else 100.0
        max_s = max_load / 100.0 * sn_mva
        if max_s > 0 and S_pred[i] / max_s < line_threshold:
            net.line.at[i, "max_loading_percent"] = 1000.0
            stats["line_removed"] += 1
            stats["total_removed"] += 1

    # Generator Pg/Qg screening
    for g in range(min(n_gen, len(Pg))):
        stats["total_original"] += 4
        pg_pred = Pg[g]
        pg_max = net.gen.at[g, "max_p_mw"]
        pg_min = net.gen.at[g, "min_p_mw"]
        pg_range = pg_max - pg_min
        if pg_range > 0:
            if pg_pred < pg_max - gen_margin_frac * pg_range:
                net.gen.at[g, "max_p_mw"] = pg_max * 10
                stats["pg_upper_removed"] += 1
                stats["total_removed"] += 1
            if pg_pred > pg_min + gen_margin_frac * pg_range:
                net.gen.at[g, "min_p_mw"] = pg_min * 10 if pg_min < 0 else -pg_max
                stats["pg_lower_removed"] += 1
                stats["total_removed"] += 1

        qg_pred = Qg[g]
        qg_max = net.gen.at[g, "max_q_mvar"]
        qg_min = net.gen.at[g, "min_q_mvar"]
        qg_range = qg_max - qg_min
        if qg_range > 0:
            if qg_pred < qg_max - gen_margin_frac * qg_range:
                net.gen.at[g, "max_q_mvar"] = qg_max * 10
                stats["qg_upper_removed"] += 1
                stats["total_removed"] += 1
            if qg_pred > qg_min + gen_margin_frac * qg_range:
                net.gen.at[g, "min_q_mvar"] = qg_min * 10 if qg_min < 0 else -qg_max
                stats["qg_lower_removed"] += 1
                stats["total_removed"] += 1

    pct = stats["total_removed"] / max(stats["total_original"], 1) * 100
    stats["removal_pct"] = pct
    return net, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--ckpt-dir", default="ckpt")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--use-norm", action="store_true", default=True)
    parser.add_argument("--vm-margins", nargs="+", type=float,
                        default=[0.005, 0.01, 0.02, 0.03])
    parser.add_argument("--line-threshold", type=float, default=0.7)
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

    all_results = []

    for vm_margin in args.vm_margins:
        log.info(f"=== CONSTRAINT SCREENING: vm_margin={vm_margin} ===")
        results = {"vm_margin": vm_margin, "flat": [], "screened": [],
                   "removal_pcts": [], "converged": 0, "total": 0,
                   "false_neg": 0}

        for idx in range(min(args.n_test, len(test_ds))):
            data = test_ds[idx]

            with torch.no_grad():
                bp, gp = det_model(data.to(DEVICE))
            if norm:
                bp = norm.denormalize_bus(bp)
                gp = norm.denormalize_gen(gp)

            Vm = bp[:, 1].cpu().numpy()
            Va = bp[:, 0].cpu().numpy()
            Pg = gp[:, 0].cpu().numpy() * 100
            Qg = gp[:, 1].cpu().numpy() * 100

            # Baseline: flat start on original problem
            net_orig = make_net(); set_loads(net_orig, data)
            conv_orig, ni_orig, obj_orig = run_opf(net_orig, "flat")
            results["flat"].append(ni_orig)

            # Screened: remove constraints, then flat start
            net_screened = make_net(); set_loads(net_screened, data)
            net_screened, stats = screen_constraints(
                net_screened, Vm, Va, Pg, Qg,
                vm_margin=vm_margin, line_threshold=args.line_threshold)
            conv_sc, ni_sc, obj_sc = run_opf(net_screened, "flat")
            results["screened"].append(ni_sc)
            results["removal_pcts"].append(stats["removal_pct"])
            results["total"] += 1
            if conv_sc:
                results["converged"] += 1

            # Check for false negatives: did we remove a binding constraint?
            # (approximate: if screened solution has worse objective than original)
            if conv_sc and conv_orig and obj_sc is not None and obj_orig is not None:
                try:
                    sc_cost = float(net_screened.res_cost) if hasattr(net_screened.res_cost, '__float__') else 0
                    orig_cost = float(net_orig.res_cost) if hasattr(net_orig.res_cost, '__float__') else 0
                    if sc_cost > 0 and orig_cost > 0 and abs(sc_cost - orig_cost) / orig_cost > 0.001:
                        results["false_neg"] += 1
                except Exception:
                    pass

            log.info(f"  #{idx:3d}  flat={ni_orig}  screened={ni_sc}  "
                     f"removed={stats['removal_pct']:.0f}%  "
                     f"vm_up={stats['vm_upper_removed']} vm_lo={stats['vm_lower_removed']} "
                     f"line={stats['line_removed']} "
                     f"pg={stats['pg_upper_removed']+stats['pg_lower_removed']} "
                     f"qg={stats['qg_upper_removed']+stats['qg_lower_removed']}")

        flat_vals = [x for x in results["flat"] if x is not None]
        sc_vals = [x for x in results["screened"] if x is not None]
        fm = np.mean(flat_vals); sm = np.mean(sc_vals)
        rm = np.mean(results["removal_pcts"])

        all_results.append({
            "vm_margin": vm_margin,
            "flat_mean": fm, "flat_median": np.median(flat_vals),
            "screened_mean": sm, "screened_median": np.median(sc_vals),
            "delta_pct": (1 - sm/fm) * 100 if fm > 0 else 0,
            "removal_pct": rm,
            "conv_rate": results["converged"] / results["total"] * 100,
            "false_neg_rate": results["false_neg"] / results["total"] * 100,
        })

    # Summary
    print("\n" + "=" * 90)
    print("  DIRECTION 1: ACTIVE CONSTRAINT SCREENING RESULTS")
    print("=" * 90)
    print(f"  {'VM margin':>10}  {'Flat':>6}  {'Screen':>7}  {'Δ iters':>8}  "
          f"{'Removed':>8}  {'Conv%':>6}  {'FalseNeg%':>10}")
    print("-" * 75)
    for r in all_results:
        print(f"  {r['vm_margin']:10.3f}  {r['flat_mean']:6.1f}  {r['screened_mean']:7.1f}  "
              f"{r['delta_pct']:+7.1f}%  {r['removal_pct']:7.0f}%  "
              f"{r['conv_rate']:5.0f}%  {r['false_neg_rate']:9.1f}%")
    print("=" * 90)

    best = max(all_results, key=lambda r: r["delta_pct"])
    if best["delta_pct"] > 0:
        print(f"\n  ✓ CONSTRAINT SCREENING WORKS! Best: vm_margin={best['vm_margin']}")
        print(f"    {best['delta_pct']:+.1f}% iteration reduction "
              f"({best['flat_mean']:.1f} → {best['screened_mean']:.1f})")
        print(f"    {best['removal_pct']:.0f}% constraints removed, "
              f"{best['false_neg_rate']:.1f}% false negative rate")
    else:
        print(f"\n  ✗ Constraint screening did not reduce iterations.")
    print("=" * 90)

    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/constraint_screening.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader()
        w.writerows(all_results)
    log.info(f"Summary saved to {csv_path}")


if __name__ == "__main__":
    main()
