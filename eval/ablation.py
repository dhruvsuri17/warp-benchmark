"""
Ablation sweep runner.

Runs ablation experiments varying:
- K (number of samples): 1, 2, 5, 10, 20
- lambda_phy: 0, 0.01, 0.1, 1.0
- num_layers: 2, 4, 6, 8
- PE: with/without Laplacian PE
- physics loss schedule: uniform vs sin-weighted
"""

import logging
from pathlib import Path
from itertools import product

import yaml

logger = logging.getLogger(__name__)


ABLATION_AXES = {
    "K": [1, 2, 5, 10, 20],
    "lambda_phy": [0.0, 0.01, 0.1, 1.0],
    "num_layers": [2, 4, 6, 8],
    "pe_dim": [0, 16],
    "physics_schedule": ["uniform", "sin"],
}


def generate_ablation_configs(base_config_path: str, output_dir: str,
                              axes: list = None) -> list:
    """
    Generate YAML configs for ablation sweep.

    Args:
        base_config_path: path to base experiment config
        output_dir: where to write ablation configs
        axes: which ablation axes to sweep (default: all)

    Returns:
        list of config file paths
    """
    with open(base_config_path) as f:
        base = yaml.safe_load(f)

    if axes is None:
        axes = list(ABLATION_AXES.keys())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = []

    for axis in axes:
        values = ABLATION_AXES[axis]
        for val in values:
            cfg = _deep_copy_config(base)
            cfg["experiment"]["name"] = f"ablation_{axis}_{val}"

            if axis == "K":
                cfg["diffusion"]["K"] = val
            elif axis == "lambda_phy":
                cfg["training"]["lambda_phy"] = val
            elif axis == "num_layers":
                cfg["model"]["num_layers"] = val
            elif axis == "pe_dim":
                cfg["model"]["pe_dim"] = val
            elif axis == "physics_schedule":
                cfg["training"]["physics_schedule"] = val

            path = output_dir / f"ablation_{axis}_{val}.yaml"
            with open(path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            configs.append(str(path))

    logger.info(f"Generated {len(configs)} ablation configs in {output_dir}")
    return configs


def _deep_copy_config(cfg: dict) -> dict:
    import copy
    return copy.deepcopy(cfg)
