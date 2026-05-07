#!/usr/bin/env python3
"""
Build train/val/test tarballs (for Croissant checksums + optional HF attachments),
then write croissant.json in Croissant 1.0 shape with RAI fields.

Usage:
  cd warp-benchmark
  python scripts/build_release_tarballs_and_croissant.py \\
    --duals-root data/duals/pglib_opf_case118_ieee \\
    --hf-repo dhruvsuri17/warp-opf-case118-dual \\
    --openml-id PLACEHOLDER

Validate:
  python scripts/validate_croissant.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

# Croissant 1.0 @context (subset aligned with ML Commons examples; includes RAI prefix).
CROISSANT_CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "wd": "https://www.wikidata.org/wiki/",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tar_split(split_dir: Path, out_tar: Path) -> None:
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, "w:gz") as tf:
        for p in sorted(split_dir.glob("duals_*.pt")):
            tf.add(p, arcname=p.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duals-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("submission/tarballs"))
    ap.add_argument("--croissant-out", type=Path, default=Path("croissant.json"))
    ap.add_argument("--hf-repo", default="dhruvsuri17/warp-opf-case118-dual")
    ap.add_argument("--openml-id", default="DATASET_ID", help="OpenML numeric id after upload")
    args = ap.parse_args()

    splits = ["train", "val", "test"]
    urls_hf = []
    dist = []
    out_dir = args.out_dir

    for sp in splits:
        sd = args.duals_root / sp
        if not sd.is_dir():
            raise SystemExit(f"Missing split directory: {sd}")
        tar_path = out_dir / f"{sp}.tar.gz"
        tar_split(sd, tar_path)
        digest = sha256_file(tar_path)
        size_b = tar_path.stat().st_size
        name = f"{sp}.tar.gz"
        hf_url = f"https://huggingface.co/datasets/{args.hf_repo}/resolve/main/submission/tarballs/{name}"
        oml_url = f"https://www.openml.org/d/{args.openml_id}/{sp}.tar.gz"
        urls_hf.append(hf_url)
        dist.append(
            {
                "@type": "cr:FileObject",
                "@id": name,
                "name": name,
                "description": f"Dual-label tensors for OPFDataset split `{sp}` (PyTorch .pt files, gzip tar).",
                "contentSize": f"{size_b} B",
                "contentUrl": hf_url,
                "encodingFormat": "application/gzip",
                "sha256": digest,
            }
        )

    meta = {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "name": "WARP-OPF-Case118-DualLabeled",
        "description": (
            "Dual-labeled AC Optimal Power Flow dataset with IPOPT-converged "
            "primal-dual-barrier tensors for interior-point warm-start benchmarking "
            "on pglib_opf_case118_ieee (PyG OPFDataset). Each split archive contains "
            "duals_XXXXXX.pt files with keys x, lam_g, zl, zu, mu, obj."
        ),
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "citeAs": (
            "NeurIPS 2026 Evaluations & Datasets Track — why primal-only warm-starts "
            "fail for interior-point solvers on AC-OPF (anonymous submission)."
        ),
        "license": "MIT",
        "url": f"https://huggingface.co/datasets/{args.hf_repo}",
        "sameAs": [
            f"https://www.openml.org/d/{args.openml_id}",
            f"https://huggingface.co/datasets/{args.hf_repo}",
        ],
        "version": "1.0.0",
        "datePublished": "2026-05-06",
        "creator": {"@type": "Organization", "name": "Anonymous (NeurIPS 2026 submission)"},
        "distribution": dist,
        "rai:dataCollection": (
            "Primal scenarios from OPFDataset; dual states from IPOPT (cyipopt, exact Hessian, tol=1e-4)."
        ),
        "rai:dataCollectionType": "Synthetic load perturbations on pglib_opf_case118_ieee.",
        "rai:personalSensitiveInformation": "None (synthetic).",
        "rai:dataSocialImpact": (
            "Potential faster OPF solves may improve grid dispatch and renewables integration; "
            "misuse could inform unrealistic market assumptions if deployed naively."
        ),
        "rai:dataLimitations": (
            "Single network; extraction cost scales poorly; multiplier sparsity patterns may differ on other cases."
        ),
        "rai:dataBiases": "Synthetic loads may not match real operating distributions.",
        "rai:dataUseCases": (
            "Training/evaluating ML warm-start models for interior-point AC-OPF solvers."
        ),
    }

    args.croissant_out.parent.mkdir(parents=True, exist_ok=True)
    args.croissant_out.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.croissant_out}")
    print("Tarballs:", out_dir)
    for u in urls_hf:
        print(" ", u)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
