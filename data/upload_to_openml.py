"""
Upload WARP dual-labeled AC-OPF dataset to OpenML (tabular ARFF export).

Requires:
    export OPENML_API_KEY="..."   # never commit this value

The canonical training artifact remains the `.pt` files; this script flattens them
for OpenML discoverability.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import openml
import pandas as pd
import torch


def pt_files_to_dataframe(data_dir: Path, split: str) -> pd.DataFrame:
    """Convert .pt dual label files to a pandas DataFrame."""
    records: list[dict] = []
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    for pt_file in sorted(split_dir.glob("duals_*.pt")):
        data = torch.load(pt_file, map_location="cpu")
        record: dict = {}
        for key in ["x", "lam_g", "zl", "zu"]:
            tensor = data[key].numpy().flatten()
            prefix = {
                "x": "x_star",
                "lam_g": "lambda_star",
                "zl": "z_l_star",
                "zu": "z_u_star",
            }[key]
            for i, val in enumerate(tensor):
                record[f"{prefix}_{i}"] = float(val)
        record["mu_star"] = float(data["mu"].item() if hasattr(data["mu"], "item") else data["mu"])
        record["obj"] = float(data["obj"].item() if hasattr(data["obj"], "item") else data["obj"])
        record["split"] = split
        stem = pt_file.stem  # duals_000001
        record["instance_idx"] = int(stem.split("_")[-1])
        records.append(record)
    return pd.DataFrame(records)


def upload_to_openml(
    data_dir: str | Path,
    github_url: str = "https://github.com/anonymous/warp-benchmark",
) -> int:
    """Upload the combined dataset to OpenML; returns new dataset id."""
    api_key = os.environ.get("OPENML_API_KEY")
    if not api_key:
        raise ValueError("Set OPENML_API_KEY environment variable (do not commit API keys).")
    openml.config.apikey = api_key

    root = Path(data_dir)
    dfs = []
    for split in ["train", "val", "test"]:
        if (root / split).is_dir():
            dfs.append(pt_files_to_dataframe(root, split))
    if not dfs:
        raise FileNotFoundError(f"No train/val/test under {root}")

    full_df = pd.concat(dfs, ignore_index=True)

    oml_dataset = openml.datasets.functions.create_dataset(
        name="WARP-OPF-Case118-DualLabeled",
        description=(
            "Dual-labeled AC Optimal Power Flow dataset for interior-point method "
            "warm-start benchmarking. Each instance contains the IPOPT-converged "
            "primal-dual-barrier state (x*, lambda*, z_l*, z_u*, mu*) for "
            "pglib_opf_case118_ieee from OPFDataset. "
            "See code release for tensor layout and graph structure. "
            f"{github_url}"
        ),
        creator="Anonymous (NeurIPS 2026 submission)",
        licence="MIT",
        data=full_df,
        default_target_attribute="mu_star",
        attributes="auto",
        collection_date="2026",
        language="English",
        citation=(
            "Why Primal-Only Warm-Starts Fail for Interior-Point Solvers: "
            "A Corrected Benchmark and Primal-Dual Method for AC Optimal Power Flow. "
            "NeurIPS 2026 Evaluations & Datasets Track."
        ),
    )

    dataset_id = oml_dataset.publish()
    print(f"Dataset uploaded to OpenML with ID: {dataset_id}")
    print(f"URL: https://www.openml.org/d/{dataset_id}")
    return int(dataset_id)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "data_dir",
        nargs="?",
        default="data/case118",
        help="Directory containing train/, val/, test/ with duals_*.pt",
    )
    p.add_argument(
        "--github-url",
        default="https://github.com/anonymous/warp-benchmark",
        help="Link printed in the OpenML description (use anonymised URL for review).",
    )
    args = p.parse_args()
    upload_to_openml(args.data_dir, github_url=args.github_url)
