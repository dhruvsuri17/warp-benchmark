"""Extract (x*, lam_g*, zl*, zu*, mu*) from IPOPT solves for training labels.

Runs IPOPT on each training/val instance and saves the full primal-dual solution.
These become labels for training the dual prediction head.
"""
import os, sys, time, logging, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("EXTRACT")

from eval.opf_ipopt import build_om, solve_opf
from torch_geometric.datasets import OPFDataset
import pandapower.networks as pn


def set_loads(net, data):
    Pd = data["load"].x[:, 0].cpu().numpy() * 100
    Qd = data["load"].x[:, 1].cpu().numpy() * 100
    for i in range(min(len(net.load), len(Pd))):
        net.load.at[i, "p_mw"] = Pd[i]
        net.load.at[i, "q_mvar"] = Qd[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--save-dir", default="data/duals")
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()

    save_dir = Path(args.save_dir) / args.case / args.split
    save_dir.mkdir(parents=True, exist_ok=True)

    ds = OPFDataset(root="data", case_name=args.case, split=args.split,
                    num_groups=args.num_groups)
    n = len(ds) if args.max_instances is None else min(len(ds), args.max_instances)

    case_map = {
        "pglib_opf_case14_ieee": pn.case14,
        "pglib_opf_case30_ieee": pn.case30,
        "pglib_opf_case57_ieee": pn.case57,
        "pglib_opf_case118_ieee": pn.case118,
        "pglib_opf_case6470_rte": pn.case6470rte,
    }
    if args.case not in case_map:
        raise ValueError(f"No pandapower network for {args.case}. Available: {list(case_map.keys())}")
    make_net = case_map[args.case]

    log.info(f"Extracting duals for {args.case}/{args.split}, {n} instances")
    log.info(f"Saving to {save_dir}")

    success = 0
    t0 = time.time()

    for idx in range(n):
        data = ds[idx]
        net = make_net()
        set_loads(net, data)

        om, ppopt = build_om(net)
        r = solve_opf(om, ppopt, warm_start=False, max_iter=200)

        if r["converged"] and r["lam_g"] is not None:
            torch.save({
                "x": torch.tensor(r["x"], dtype=torch.float32),
                "lam_g": torch.tensor(r["lam_g"], dtype=torch.float32),
                "zl": torch.tensor(r["zl"], dtype=torch.float32),
                "zu": torch.tensor(r["zu"], dtype=torch.float32),
                "mu": torch.tensor(r["mu_final"], dtype=torch.float32),
                "obj": torch.tensor(r["obj"], dtype=torch.float32),
                "n_iters": r["n_iters"],
            }, save_dir / f"duals_{idx:06d}.pt")
            success += 1

        if idx % 50 == 0 or idx == n - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (n - idx - 1) / rate if rate > 0 else 0
            log.info(f"  [{idx+1}/{n}] success={success}, "
                     f"rate={rate:.1f} inst/s, ETA={eta/60:.0f}min")

    log.info(f"Done. {success}/{n} converged. Saved to {save_dir}")


if __name__ == "__main__":
    main()
