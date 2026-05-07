"""Public release entry: IPM-LSTM-style training (`training.train_ipm`)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from training.train_ipm import main as _main

    _main()


if __name__ == "__main__":
    main()
