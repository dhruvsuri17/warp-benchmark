#!/usr/bin/env python3
"""Validate croissant.json using mlcroissant (same checks as the HF validator core)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("croissant", nargs="?", default="croissant.json", type=Path)
    args = parser.parse_args()

    try:
        import mlcroissant as mc
    except ImportError:
        print("pip install mlcroissant", file=sys.stderr)
        return 1

    raw = json.loads(args.croissant.read_text())
    try:
        mc.Dataset(raw)
    except Exception as e:
        print("FAIL:", e, file=sys.stderr)
        return 1
    print("OK:", args.croissant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
