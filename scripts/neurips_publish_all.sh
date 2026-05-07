#!/usr/bin/env bash
# One-shot: tarballs + Croissant + Hugging Face dataset upload (requires HF_TOKEN).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"

echo "== [1/3] Build tarballs + croissant.json (Croissant 1.0 + RAI) =="
python scripts/build_release_tarballs_and_croissant.py \
  --duals-root data/duals/pglib_opf_case118_ieee \
  --hf-repo "${HF_REPO:-dhruvsuri17/warp-opf-case118-dual}"

python scripts/validate_croissant.py croissant.json

echo "== [2/3] Validate Hub auth =="
if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: Set HF_TOKEN (https://huggingface.co/settings/tokens) to upload."
  exit 1
fi

echo "== [3/3] Upload dataset files =="
python scripts/hf_publish_dataset.py \
  --repo-id "${HF_REPO:-dhruvsuri17/warp-opf-case118-dual}" \
  --data-root data/duals/pglib_opf_case118_ieee \
  --card huggingface/DATASET_CARD_TEMPLATE.md \
  --croissant croissant.json

echo "Done. Attach croissant.json to OpenReview and cite:"
echo "  https://huggingface.co/datasets/${HF_REPO:-dhruvsuri17/warp-opf-case118-dual}"
