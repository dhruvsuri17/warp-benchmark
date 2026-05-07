"""Regenerate paper figures from logged results (extend with your plotting code)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    placeholder = out / "README.txt"
    placeholder.write_text(
        "Place plotting scripts here (matplotlib/seaborn). "
        "Wire them to tables exported from eval/*.py runs.\n"
    )
    print(f"Placeholder written to {placeholder}")


if __name__ == "__main__":
    main()
