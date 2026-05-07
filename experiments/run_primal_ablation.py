"""Primal-only strategy exploration — uses eval/ablation config generation + legacy sweep."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wrapper around eval.ablation for primal-focused sweeps.",
    )
    parser.add_argument(
        "--base-config",
        default=str(ROOT / "experiments" / "configs" / "ablation_case118.yaml"),
        help="Usually experiments/configs/ablation_case118.yaml",
    )
    parser.add_argument("--output-dir", default="results/ablation_primal")
    args, rest = parser.parse_known_args()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    from eval.ablation import generate_ablation_configs

    os.makedirs(args.output_dir, exist_ok=True)
    cfgs = generate_ablation_configs(args.base_config, args.output_dir)
    print(f"Generated {len(cfgs)} configs under {args.output_dir}")
    print("Run training via experiments/run_experiment.py --config <yaml> for each.")


if __name__ == "__main__":
    main()
