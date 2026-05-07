# Publishing checklist (nothing runs automatically)

These steps require **your** machine or CI with secrets — nothing was pushed from the agent environment.

## 1. Hugging Face dataset mirror

**Automated (recommended before NeurIPS upload):** builds Croissant 1.0 + tarballs + uploads splits:

```bash
cd warp-benchmark
pip install -r requirements.txt
export HF_TOKEN="hf_..."   # from https://huggingface.co/settings/tokens

bash scripts/neurips_publish_all.sh
```

Manual steps equivalent:

```bash
python scripts/build_release_tarballs_and_croissant.py --duals-root data/duals/pglib_opf_case118_ieee
python scripts/validate_croissant.py
python scripts/hf_publish_dataset.py \
  --repo-id dhruvsuri17/warp-opf-case118-dual \
  --data-root data/duals/pglib_opf_case118_ieee \
  --card huggingface/DATASET_CARD_TEMPLATE.md
```

Dry run (no token needed for logic check):

```bash
python scripts/hf_publish_dataset.py --dry-run \
  --data-root data/duals/pglib_opf_case118_ieee \
  --repo-id dhruvsuri17/warp-opf-case118-dual
```

## 2. Hugging Face model weights

Place `warp_gnores_case118.pt` and `lstm_case118.pt` in `models/checkpoints/`, then:

```bash
export HF_TOKEN="hf_..."
python scripts/hf_publish_models.py --repo-id dhruvsuri17/warp-benchmark-warp-pd
```

## 3. OpenML (NeurIPS primary registry)

```bash
export OPENML_API_KEY="..."   # never commit this
python data/upload_to_openml.py data/duals/pglib_opf_case118_ieee
python scripts/update_croissant.py --dataset-id <ID_FROM_UPLOAD>
```

## 4. Git remote

Push your git repo to GitHub (anonymous account for review) as usual; the agent did not create or push remotes.
