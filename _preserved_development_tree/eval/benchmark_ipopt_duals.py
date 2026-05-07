"""Benchmark: IPOPT with oracle duals from extracted labels.

Tests the ceiling: what iteration count is achievable if the model
perfectly predicts (x, lam_g, zl, zu, mu)?
"""
import os, sys, argparse, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("IPOPT-DUAL")

from eval.opf_ipopt import build_om, solve_opf
from torch_geometric.datasets import OPFDataset
import pandapower.networks as pn
from numpy import inf


def set_loads(net, data):
    Pd = data["load"].x[:, 0].cpu().numpy() * 100
    Qd = data["load"].x[:, 1].cpu().numpy() * 100
    for i in range(min(len(net.load), len(Pd))):
        net.load.at[i, "p_mw"] = Pd[i]
        net.load.at[i, "q_mvar"] = Qd[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--duals-dir", default="data/duals")
    parser.add_argument("--n-test", type=int, default=50)
    args = parser.parse_args()

    duals_dir = Path(args.duals_dir) / args.case / "test"
    test_ds = OPFDataset(root="data", case_name=args.case, split="test", num_groups=1)
    make_net = pn.case118

    cold_iters, primal_iters, dual_iters, dual_mu_iters = [], [], [], []

    log.info(f"=== IPOPT DUAL BENCHMARK ({args.case}) ===")

    for idx in range(min(args.n_test, len(test_ds))):
        dual_path = duals_dir / f"duals_{idx:06d}.pt"
        if not dual_path.exists():
            log.info(f"  #{idx}: no duals file, skipping")
            continue

        data = test_ds[idx]
        duals = torch.load(dual_path, weights_only=True)

        net = make_net(); set_loads(net, data)
        om, ppopt = build_om(net)
        x0_v, xmin, xmax = om.getv()
        ll, uu = xmin.copy(), xmax.copy()
        ll[xmin == -inf] = -1e10; uu[xmax == inf] = 1e10
        x_mid = (ll + uu) / 2.0

        x_opt = duals["x"].numpy()
        lam_g = duals["lam_g"].numpy()
        zl = duals["zl"].numpy()
        zu = duals["zu"].numpy()
        mu = duals["mu"].item()

        # Cold start
        r_cold = solve_opf(om, ppopt, x0=x_mid, warm_start=False)
        cold_iters.append(r_cold["n_iters"])

        # Primal only
        x_clip = np.clip(x_opt, xmin + 1e-10, xmax - 1e-10)
        r_primal = solve_opf(om, ppopt, x0=x_clip, warm_start=True)
        primal_iters.append(r_primal["n_iters"])

        # Primal + duals (adaptive mu)
        r_dual = solve_opf(om, ppopt, x0=x_clip,
                           lam_g0=lam_g, zl0=zl, zu0=zu,
                           warm_start=True)
        dual_iters.append(r_dual["n_iters"])

        # Primal + duals + mu (full IPM-LSTM style)
        r_dual_mu = solve_opf(om, ppopt, x0=x_clip,
                              lam_g0=lam_g, zl0=zl, zu0=zu,
                              warm_start=True, mu_init=mu)
        dual_mu_iters.append(r_dual_mu["n_iters"])

        log.info(f"  #{idx:3d}  cold={r_cold['n_iters']}  primal={r_primal['n_iters']}  "
                 f"dual={r_dual['n_iters']}  dual+mu={r_dual_mu['n_iters']}")

    def _s(v):
        return np.mean(v), np.median(v) if v else (float('nan'), float('nan'))

    fm = _s(cold_iters)[0]
    print(f"\n{'='*70}", flush=True)
    print(f"  IPOPT BENCHMARK WITH ORACLE DUALS ({len(cold_iters)} instances)", flush=True)
    print(f"{'='*70}", flush=True)
    for label, vals in [("Cold (midpoint)", cold_iters),
                        ("Primal only", primal_iters),
                        ("Primal + duals", dual_iters),
                        ("Primal + duals + mu*", dual_mu_iters)]:
        m, md = _s(vals)
        delta = (1 - m/fm)*100 if fm > 0 else 0
        print(f"  {label:25s}  mean={m:5.1f}  median={md:4.0f}  vs cold: {delta:+.1f}%", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
