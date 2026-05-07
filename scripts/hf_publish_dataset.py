#!/usr/bin/env python3
"""
Create/update a Hugging Face *dataset* repo and upload dual-label splits + metadata.

Authentication (pick one):
  export HF_TOKEN="hf_..."           # recommended in CI
  huggingface-cli login              # interactive

Nothing is uploaded unless you run this script with network access and a valid token.

Example:
  cd warp-benchmark
  export PYTHONPATH=.
  python scripts/hf_publish_dataset.py \\
    --repo-id dhruvsuri17/warp-opf-case118-dual \\
    --data-root data/duals/pglib_opf_case118_ieee \\
    --card huggingface/DATASET_CARD_TEMPLATE.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload WARP dual-label data to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        default="dhruvsuri17/warp-opf-case118-dual",
        help="Dataset repo id: user-or-org/name",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing train/, val/, test/ with duals_*.pt",
    )
    parser.add_argument(
        "--card",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "huggingface" / "DATASET_CARD_TEMPLATE.md",
        help="Dataset card source (will be uploaded as README.md)",
    )
    parser.add_argument(
        "--croissant",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "croissant.json",
        help="Optional Croissant file to attach at repo root",
    )
    parser.add_argument("--private", action="store_true", help="Create/use a private dataset repo")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not call the Hub",
    )
    parser.add_argument(
        "--tarballs-dir",
        type=Path,
        default=None,
        help="If set (default: <repo>/submission/tarballs when present), upload gzip tars for Croissant URLs.",
    )
    parser.add_argument(
        "--license-file",
        type=Path,
        default=None,
        help="Upload as LICENSE (default: ../LICENSE)",
    )
    args = parser.parse_args()

    token = _token()
    if not args.dry_run and not token:
        print(
            "ERROR: No Hugging Face token. Set HF_TOKEN or run `huggingface-cli login`.",
            file=sys.stderr,
        )
        return 1

    splits = ["train", "val", "test"]
    for sp in splits:
        p = args.data_root / sp
        if not p.is_dir():
            print(f"WARNING: missing split directory: {p}", file=sys.stderr)

    root = Path(__file__).resolve().parents[1]
    tarballs_dir = args.tarballs_dir
    if tarballs_dir is None:
        default_tars = root / "submission" / "tarballs"
        if default_tars.is_dir() and any(default_tars.glob("*.tar.gz")):
            tarballs_dir = default_tars
    lic = args.license_file or (root / "LICENSE")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi(token=token) if token else HfApi()

    if args.dry_run:
        print(f"[dry-run] create_repo(repo_id={args.repo_id!r}, repo_type=dataset, private={args.private})")
        if args.card.is_file():
            print(f"[dry-run] upload README from {args.card}")
        if args.croissant.is_file():
            print(f"[dry-run] upload {args.croissant} -> croissant.json")
        if lic.is_file():
            print(f"[dry-run] upload {lic} -> LICENSE")
        if tarballs_dir and tarballs_dir.is_dir():
            print(f"[dry-run] upload_folder({tarballs_dir} -> submission/tarballs/)")
        for sp in splits:
            d = args.data_root / sp
            if d.is_dir():
                print(f"[dry-run] upload_folder({d} -> {sp}/)")
        return 0

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    if args.card.is_file():
        api.upload_file(
            path_or_fileobj=str(args.card),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded dataset card -> README.md ({args.card})")

    if args.croissant.is_file():
        api.upload_file(
            path_or_fileobj=str(args.croissant),
            path_in_repo="croissant.json",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded {args.croissant.name}")

    if lic.is_file():
        api.upload_file(
            path_or_fileobj=str(lic),
            path_in_repo="LICENSE",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded LICENSE ({lic})")

    if tarballs_dir and tarballs_dir.is_dir():
        api.upload_folder(
            folder_path=str(tarballs_dir),
            path_in_repo="submission/tarballs",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded tarball folder -> submission/tarballs/ ({tarballs_dir})")

    for sp in splits:
        d = args.data_root / sp
        if not d.is_dir():
            continue
        n = len(list(d.glob("duals_*.pt")))
        api.upload_folder(
            folder_path=str(d),
            path_in_repo=sp,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded split {sp}/ ({n} duals_*.pt files)")

    url = f"https://huggingface.co/datasets/{args.repo_id}"
    print(f"Done. Dataset URL: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
