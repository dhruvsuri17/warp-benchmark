#!/usr/bin/env python3
"""
Upload model checkpoints (.pt) to a Hugging Face *model* repo.

Authentication: HF_TOKEN or `huggingface-cli login`.

Example:
  python scripts/hf_publish_models.py \\
    --repo-id dhruvsuri17/warp-benchmark-warp-pd \\
    --checkpoint-dir models/checkpoints
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="dhruvsuri17/warp-benchmark-warp-pd")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/checkpoints"),
        help="Directory containing .pt files",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = _token()
    if not args.dry_run and not token:
        print("ERROR: Set HF_TOKEN or run huggingface-cli login.", file=sys.stderr)
        return 1

    pts = sorted(args.checkpoint_dir.glob("*.pt"))
    if not pts:
        print(f"No .pt files under {args.checkpoint_dir}", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi(token=token) if token else HfApi()

    if args.dry_run:
        print(f"[dry-run] create_repo({args.repo_id}, repo_type=model)")
        for p in pts:
            print(f"[dry-run] upload {p.name}")
        return 0

    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    for p in pts:
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=p.name,
            repo_id=args.repo_id,
            repo_type="model",
        )
        print(f"Uploaded {p.name}")

    print(f"https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
