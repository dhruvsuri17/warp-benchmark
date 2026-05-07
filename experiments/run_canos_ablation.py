"""Independent CANOS validation — placeholder entry that forwards to legacy benchmark scripts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-benchmark",
        default="eval/benchmark_barrier_loss.py",
        help="Path relative to warp-benchmark root",
    )
    args, forwarded = parser.parse_known_args()

    target = ROOT / args.legacy_benchmark
    if not target.is_file():
        print(
            "Configure --legacy-benchmark to a CANOS-related script under eval/. "
            f"Missing: {target}",
            file=sys.stderr,
        )
        sys.exit(1)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    # Execute as __main__
    import runpy

    sys.argv = [str(target.name)] + forwarded
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
