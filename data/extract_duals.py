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


def extract_split(case_name, split, num_groups, data_root, save_dir, max_instances, make_net):
    """Run extraction for one split; returns (success_count, attempted_count)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    ds = OPFDataset(root=data_root, case_name=case_name, split=split,
                    num_groups=num_groups)
    n = len(ds) if max_instances is None else min(len(ds), max_instances)

    log.info(f"Extracting duals for {case_name}/{split}, {n} instances → {save_dir}")

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

    log.info(f"Done {split}. {success}/{n} converged. Saved to {save_dir}")
    return success, n


def main():
    parser = argparse.ArgumentParser(
        description="Extract IPOPT primal-dual-barrier labels for OPFDataset instances.",
    )
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--data-root", default="data", help="Root passed to OPFDataset (contains raw downloads).")
    parser.add_argument("--save-dir", default=None, help="Legacy: save under save-dir/case/split")
    parser.add_argument("--output-root", default=None,
                        help="Release layout: write to output-root/<split>/duals_*.pt")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    args = parser.parse_args()

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

    if args.n_train is not None or args.n_val is not None or args.n_test is not None:
        out = Path(args.output_root or "data/case118")
        specs = [
            ("train", args.n_train),
            ("val", args.n_val),
            ("test", args.n_test),
        ]
        for split, lim in specs:
            if lim is None:
                continue
            extract_split(
                args.case, split, args.num_groups, args.data_root,
                out / split, lim, make_net,
            )
        return

    if args.output_root:
        save_dir = Path(args.output_root) / args.split
    elif args.save_dir:
        save_dir = Path(args.save_dir) / args.case / args.split
    else:
        save_dir = Path("data/duals") / args.case / args.split

    extract_split(
        args.case, args.split, args.num_groups, args.data_root,
        save_dir, args.max_instances, make_net,
    )


if __name__ == "__main__":
    main()
