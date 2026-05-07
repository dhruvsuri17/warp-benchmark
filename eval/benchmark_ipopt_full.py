"""Full IPOPT benchmark: flat vs primal-only vs primal+KKT-duals.

Uses the working ipopt_opf_v2 solver with exact Hessian.
Computes KKT-based dual estimates from DetGNN's primal prediction.
"""
import os, sys, argparse, time, logging, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from numpy import inf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("IPOPT-FULL")

from colab_warp import DetGNN, DEVICE
from normalizer import VariableNormalizer
from eval.ipopt_opf_v2 import build_om, solve_opf
from torch_geometric.datasets import OPFDataset
import pandapower.networks as pn


def set_loads_on_net(net, data):
    Pd = data["load"].x[:, 0].cpu().numpy() * 100
    Qd = data["load"].x[:, 1].cpu().numpy() * 100
    for i in range(min(len(net.load), len(Pd))):
        net.load.at[i, "p_mw"] = Pd[i]
        net.load.at[i, "q_mvar"] = Qd[i]


def prediction_to_x(bp, gp, om):
    """Convert DetGNN prediction to IPOPT variable vector [Va, Vm, Pg, Qg].

    Uses OPF model variable indices (which may differ from net.gen count
    because ext_grid is treated as a generator in pypower).
    """
    vv = om.get_idx()[0]
    n_va = vv['N']['Va']
    n_vm = vv['N']['Vm']
    n_pg = vv['N']['Pg']
    n_qg = vv['N']['Qg']
    n_total = vv['iN']['Qg']

    Vm = bp[:, 1].cpu().numpy()
    Va_deg = np.degrees(bp[:, 0].cpu().numpy())
    Pg = gp[:, 0].cpu().numpy()  # per-unit (pypower internal)
    Qg = gp[:, 1].cpu().numpy()  # per-unit

    x0_default, _, _ = om.getv()
    x = x0_default.copy()
    x[vv['i1']['Va']:vv['i1']['Va']+min(n_va, len(Va_deg))] = Va_deg[:n_va]
    x[vv['i1']['Vm']:vv['i1']['Vm']+min(n_vm, len(Vm))] = Vm[:n_vm]
    x[vv['i1']['Pg']:vv['i1']['Pg']+min(n_pg, len(Pg))] = Pg[:n_pg]
    x[vv['i1']['Qg']:vv['i1']['Qg']+min(n_qg, len(Qg))] = Qg[:n_qg]
    return x


def compute_kkt_duals(om, ppopt, x_pred):
    """Compute approximate dual variables from primal prediction via KKT.

    Solves IPOPT from x_pred with very few iterations to get dual estimates.
    """
    r = solve_opf(om, ppopt, x0=x_pred, warm_start=True,
                  mu_init=1e-1, max_iter=5, print_level=0)
    return r.get("lam_g"), r.get("zl"), r.get("zu")


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

    results = []

    log.info("=== IPOPT FULL BENCHMARK ===")
    log.info("Methods: flat, primal-only, primal+3-step-duals, primal+full-duals")

    for idx in range(min(args.n_test, len(test_ds))):
        data = test_ds[idx]
        row = {"idx": idx}

        # Build OPF model for this instance
        net = make_net(); set_loads_on_net(net, data)
        om, ppopt = build_om(net)

        x0, xmin, xmax = om.getv()
        ll, uu = xmin.copy(), xmax.copy()
        ll[xmin == -inf] = -1e10; uu[xmax == inf] = 1e10
        x_mid = (ll + uu) / 2.0

        # Get DetGNN prediction
        with torch.no_grad():
            bp, gp = det_model(data.to(DEVICE))
        if norm:
            bp = norm.denormalize_bus(bp)
            gp = norm.denormalize_gen(gp)
        x_pred = prediction_to_x(bp, gp, om)
        x_pred = np.clip(x_pred, xmin + 1e-8, xmax - 1e-8)

        # Method 1: IPOPT flat start
        r_flat = solve_opf(om, ppopt, x0=x_mid, warm_start=False, print_level=0)
        row["flat"] = r_flat["n_iters"]

        # Method 2a: IPOPT with GNN x0, no warm-start flags (default IPM init)
        r_cold = solve_opf(om, ppopt, x0=x_pred, warm_start=False, print_level=0)
        row["cold_gnn"] = r_cold["n_iters"]

        # Method 2b: IPOPT primal-only warm-start
        r_primal = solve_opf(om, ppopt, x0=x_pred, warm_start=True,
                             mu_init=1e-1, print_level=0)
        row["primal_ws"] = r_primal["n_iters"]

        # Method 3: IPOPT primal + quick-dual (3 IPOPT steps to estimate duals)
        lam_g, zl, zu = compute_kkt_duals(om, ppopt, x_pred)
        if lam_g is not None:
            r_dual3 = solve_opf(om, ppopt, x0=x_pred, lam_g0=lam_g,
                                zl0=zl, zu0=zu,
                                warm_start=True, mu_init=1e-1, print_level=0)
            row["primal_dual3"] = r_dual3["n_iters"]
        else:
            row["primal_dual3"] = None

        # Method 4: IPOPT primal + oracle duals (from flat-start solution)
        if r_flat["lam_g"] is not None:
            r_oracle = solve_opf(om, ppopt, x0=x_pred,
                                 lam_g0=r_flat["lam_g"],
                                 zl0=r_flat["zl"], zu0=r_flat["zu"],
                                 warm_start=True, mu_init=1e-6, print_level=0)
            row["primal_oracle_duals"] = r_oracle["n_iters"]
        else:
            row["primal_oracle_duals"] = None

        results.append(row)
        log.info(f"  #{idx:3d}  flat={row['flat']}  cold_gnn={row['cold_gnn']}  "
                 f"primal_ws={row['primal_ws']}  dual3={row['primal_dual3']}  "
                 f"oracle_duals={row['primal_oracle_duals']}")

    def _s(key):
        v = [r[key] for r in results if r[key] is not None]
        return (np.mean(v), np.median(v)) if v else (float('nan'), float('nan'))

    fm = _s("flat")[0]
    print("\n" + "=" * 80)
    print("  IPOPT FULL BENCHMARK RESULTS")
    print("=" * 80)
    for label, key in [("IPOPT flat", "flat"),
                       ("IPOPT cold GNN x0", "cold_gnn"),
                       ("IPOPT primal WS", "primal_ws"),
                       ("IPOPT primal + 5-step duals", "primal_dual3"),
                       ("IPOPT primal + oracle duals", "primal_oracle_duals")]:
        m, md = _s(key)
        delta = (1 - m / fm) * 100 if fm > 0 else 0
        print(f"  {label:32s}  mean={m:5.1f}  median={md:4.0f}  vs flat: {delta:+.1f}%")
    print("=" * 80)

    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/ipopt_full_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    log.info(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
