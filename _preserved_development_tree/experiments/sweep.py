"""
Hyperparameter sweep runner (K, lambda_phy, layers).

Usage:
    python experiments/sweep.py --base-config experiments/configs/warp_case118.yaml
"""

import argparse
import logging
import subprocess
from pathlib import Path

from eval.ablation import generate_ablation_configs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--axes", nargs="+", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path("experiments/configs/sweep")
    configs = generate_ablation_configs(args.base_config, str(output_dir), args.axes)

    for cfg_path in configs:
        logger.info(f"Running sweep config: {cfg_path}")
        subprocess.run([
            "python", "experiments/run_experiment.py",
            "--config", cfg_path,
            "--gpu", str(args.gpu),
        ], check=True)


if __name__ == "__main__":
    main()
