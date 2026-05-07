"""N-1 / topology robustness experiments — thin wrapper over eval/benchmark_n2.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    rest = sys.argv[1:]
    sys.argv = ["benchmark_n2.py"] + rest
    from eval.benchmark_n2 import main as _main

    _main()


if __name__ == "__main__":
    main()
