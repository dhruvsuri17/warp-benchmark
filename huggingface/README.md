# Hugging Face Hub (mirror)

NeurIPS primary registry is **OpenML**; HF is optional for accessibility under [**`dhruvsuri17`**](https://huggingface.co/dhruvsuri17).

## Publish (you run locally)

See **[`../scripts/PUBLISH.md`](../scripts/PUBLISH.md)** for token setup and commands.

Quick publish after dual labels exist:

```bash
cd warp-benchmark
export HF_TOKEN="hf_..."   # https://huggingface.co/settings/tokens
python scripts/hf_publish_dataset.py \
  --repo-id dhruvsuri17/warp-opf-case118-dual \
  --data-root data/duals/pglib_opf_case118_ieee \
  --card huggingface/DATASET_CARD_TEMPLATE.md
```

Model checkpoints:

```bash
python scripts/hf_publish_models.py --repo-id dhruvsuri17/warp-benchmark-warp-pd
```

## Files here

| File | Purpose |
|------|---------|
| `DATASET_CARD_TEMPLATE.md` | Uploaded as `README.md` on the Hub dataset repo (edit OpenML URL / repo links before publishing). |

## Croissant

Upload `croissant.json` from the repo root via `hf_publish_dataset.py` (included automatically) or attach manually on the Hub.
