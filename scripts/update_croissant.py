"""Fill croissant.json after OpenML upload (dataset id + optional file hashes)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--tar-dir", type=str, default=None)
    parser.add_argument("--croissant", type=str, default="croissant.json")
    args = parser.parse_args()

    did = args.dataset_id or os.environ.get("OPENML_DATASET_ID")
    croissant_path = Path(args.croissant)
    meta = json.loads(croissant_path.read_text())

    if did:
        did_s = str(did)
        meta["url"] = meta["url"].replace("DATASET_ID", did_s)
        for item in meta.get("distribution", []):
            if "contentUrl" in item:
                item["contentUrl"] = item["contentUrl"].replace("DATASET_ID", did_s)

    if args.tar_dir:
        tdir = Path(args.tar_dir)
        name_to_path = {p.name: p for p in tdir.glob("*.tar.gz")}
        for item in meta.get("distribution", []):
            name = item.get("name")
            if name in name_to_path:
                item["sha256"] = sha256_file(name_to_path[name])

    croissant_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {croissant_path}")


if __name__ == "__main__":
    main()
