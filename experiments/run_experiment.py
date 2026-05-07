"""
Experiment dispatcher: reads config, launches training + evaluation.

Usage:
    python experiments/run_experiment.py --config experiments/configs/warp_case118.yaml
"""

import argparse
import logging
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["gpu"] = args.gpu

    model_type = config["model"]["type"]

    if not args.eval_only:
        logger.info(f"Starting training: {config['experiment']['name']}")
        if model_type == "warp":
            from training.train_warp import train
            train(config)
        elif model_type == "det_gnn":
            from training.train_det import train
            train(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    logger.info("Training complete. Run eval/benchmark.py for evaluation.")


if __name__ == "__main__":
    main()
