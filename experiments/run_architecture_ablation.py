"""Architecture evolution experiments — dispatch to legacy training sweep."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sweep = ROOT / "experiments" / "sweep.py"
    if not sweep.is_file():
        print("experiments/sweep.py not found in release tree.", file=sys.stderr)
        sys.exit(1)
    os.chdir(ROOT)
    subprocess.run([sys.executable, str(sweep)] + sys.argv[1:], check=False)


if __name__ == "__main__":
    main()
