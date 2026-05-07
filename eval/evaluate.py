"""
Main evaluation entrypoint for the public release.

Delegates to existing benchmark modules so the IPOPT configuration stays in one place.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    root = _root()
    os.chdir(root)
    sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser(description="WARP benchmark evaluation")
    sub = parser.add_subparsers(dest="suite", required=True)

    p_oracle = sub.add_parser(
        "suite-oracle",
        help="Oracle primal/dual warm-start ceiling (requires extracted test duals)",
    )
    p_oracle.add_argument("--case", default="pglib_opf_case118_ieee")
    p_oracle.add_argument("--duals-dir", default="data/duals")
    p_oracle.add_argument("--n-test", type=int, default=50)

    p_dual = sub.add_parser(
        "suite-dual-model",
        help="DetGNN-dual checkpoint benchmark vs cold / primal / oracle",
    )
    p_dual.add_argument("--case", default="pglib_opf_case118_ieee")
    p_dual.add_argument("--ckpt-dir", default=None, help="Directory with normalizer_stats.json + checkpoint")
    p_dual.add_argument("--checkpoint", default=None, help="Explicit .pt weights (DetGNNDual)")
    p_dual.add_argument("--duals-dir", default="data/duals")
    p_dual.add_argument("--n-test", type=int, default=50)

    args, forwarded = parser.parse_known_args(argv)

    if args.suite == "suite-oracle":
        sys.argv = [
            "benchmark_ipopt_duals.py",
            "--case",
            args.case,
            "--duals-dir",
            args.duals_dir,
            "--n-test",
            str(args.n_test),
        ] + forwarded
        from eval.benchmark_ipopt_duals import main as _main

        _main()
        return

    if args.suite == "suite-dual-model":
        ckpt_dir = args.ckpt_dir
        if args.checkpoint:
            ckpt_dir = str(Path(args.checkpoint).parent)
        if ckpt_dir is None:
            ckpt_dir = "ckpt"
        sys.argv = [
            "benchmark_dual_model.py",
            "--case",
            args.case,
            "--ckpt-dir",
            ckpt_dir,
            "--duals-dir",
            args.duals_dir,
            "--n-test",
            str(args.n_test),
        ] + forwarded
        from eval.benchmark_dual_model import main as _main

        _main()
        return


if __name__ == "__main__":
    main()
