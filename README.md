# WARP Benchmark: Primal-Dual Warm-Starting for Interior-Point Solvers

Public release layout for the WARP benchmark suite (interior-point warm starts on AC optimal power flow).

**Artifacts**

| | Link |
|---|------|
| **Code** | [github.com/dhruvsuri17/warp-benchmark](https://github.com/dhruvsuri17/warp-benchmark) |
| **Dataset (HF Hub)** | [huggingface.co/datasets/dhruvsuri17/warp-opf-case118-dual](https://huggingface.co/datasets/dhruvsuri17/warp-opf-case118-dual) — dual-label `.pt` splits (`train` / `val` / `test`), gzip tarballs under `submission/tarballs/`, `croissant.json`, and `LICENSE` |
| **Croissant (copy in repo)** | [`croissant.json`](croissant.json) — NeurIPS E&D metadata (core + RAI); validate with `python scripts/validate_croissant.py` or the [online validator](https://huggingface.co/spaces/MLCommons/croissant-validator) |

After cloning, run all commands with working directory `warp-benchmark/` and `PYTHONPATH` set to `.` (or `pip install -e .`). Local symlinks under `data/duals` and `data/pglib_opf_case118_ieee` point into a full checkout of the development repo; if those links are broken on your machine, **use the Hugging Face dataset** above or regenerate tensors with `data/extract_duals.py`.

## Layout

| Path | Role |
|------|------|
| `data/` | Dual-label extraction, OpenML upload script, Case 118 splits (populate `.pt` files locally or use OpenML). |
| `eval/` | IPOPT protocol (`opf_ipopt.py`), metrics, warm start helpers, **main entry** `evaluate.py`. |
| `models/` | WARP-PD (`warp.py` / `warp_pd.py`), DetGNN, LSTM-style IPM baseline, checkpoints directory. |
| `train/` | Thin CLI entrypoints + YAML defaults; full implementations live in `training/`. |
| `experiments/` | Paper experiment runners (delegate to `eval/` where possible). |
| `scripts/` | `reproduce_all.sh`, Croissant updater, figure reproduction. |
| `huggingface/` | Dataset card template + Hub docs; publishing scripts are under `scripts/hf_publish_*.py`. |
| `_preserved_development_tree/` | Full snapshot of the pre-release repository layout (nothing removed from development history). |
| `croissant.json` | Croissant 1.0 + RAI fields; mirrors the file published on the Hugging Face dataset repo. |

## Quick start

```bash
cd warp-benchmark
pip install -r requirements.txt
export PYTHONPATH="${PWD}"

# Download OPF graphs (PyG OPFDataset)
python -c "from torch_geometric.datasets import OPFDataset; OPFDataset(root='data', case_name='pglib_opf_case118_ieee')"

# Extract dual labels (long-running; adjust counts as needed). The path must match
# `eval/benchmark_*.py` expectations: <duals-root>/<case_name>/<split>/duals_*.pt
python data/extract_duals.py --case pglib_opf_case118_ieee --data-root data \
  --output-root data/duals/pglib_opf_case118_ieee --n-train 5000 --n-val 500 --n-test 50

# Normalisation stats for training (matches extraction layout above)
python data/compute_norm_stats.py --data-dir data/duals/pglib_opf_case118_ieee/train \
  --output data/case118/norm_stats.pt

# Train WARP-PD (see also train/configs/warp_default.yaml)
python train/train_warp.py --epochs 200 --hidden-dim 128 --k-steps 15

# Evaluate (oracle ceiling protocol — requires extracted test duals)
python eval/evaluate.py suite-oracle --duals-dir data/duals

# Full pipeline (after checkpoints & data exist)
bash scripts/reproduce_all.sh
```

## Dataset

- **Hosted copy (recommended)**: Download from **[Hugging Face](https://huggingface.co/datasets/dhruvsuri17/warp-opf-case118-dual)** — per-split folders of `duals_*.pt`, tarball exports, dataset card (`README.md`), `croissant.json`, and `LICENSE`.
- **Git / local checkout**: Large `.pt` tensors are **not** committed to GitHub; this repo holds code and `croissant.json`. After clone, pull data from HF or run `data/extract_duals.py`.
- **Monorepo layout**: If you also have the parent `warp` repo, **`data/duals` → `../../data/duals`** and **`data/pglib_opf_case118_ieee`** (PyG cache) may resolve without downloading.
- **Layout**: Evaluation expects `data/duals/<case_name>/{train,val,test}/duals_*.pt`.
- **OpenML** (optional second registry): `data/upload_to_openml.py` with `OPENML_API_KEY` in the environment only — never commit keys. After upload, refresh Croissant URLs if you dual-publish (`scripts/update_croissant.py`, `scripts/build_release_tarballs_and_croissant.py`).

## Evaluation protocol

Aligned with `eval/opf_ipopt.py`: IPOPT via cyipopt, exact Hessian, warm-start settings documented in code (`solve_opf`). Iteration counts come from the IPOPT `intermediate()` callback.

## Publishing (updates)

Initial dataset + metadata were published to **[HF](https://huggingface.co/datasets/dhruvsuri17/warp-opf-case118-dual)** and code to **[GitHub](https://github.com/dhruvsuri17/warp-benchmark)**. To refresh uploads or add OpenML, see [`scripts/PUBLISH.md`](scripts/PUBLISH.md) (`scripts/hf_publish_dataset.py`, `scripts/hf_publish_models.py`, `data/upload_to_openml.py`).

Requires `HF_TOKEN` (or `huggingface-cli login`) and `OPENML_API_KEY` only in your environment — never commit tokens.

## Croissant

The repo root [`croissant.json`](croissant.json) matches the Hub copy. After OpenML upload, regenerate with `scripts/build_release_tarballs_and_croissant.py --openml-id <id>` if you need OpenML URLs in metadata. Validate with `python scripts/validate_croissant.py` or the [online validator](https://huggingface.co/spaces/MLCommons/croissant-validator).

## License

MIT — see `LICENSE`.

## Citation

```bibtex
@inproceedings{anonymous2026warp,
  title={Why Primal-Only Warm-Starts Fail for Interior-Point Solvers},
  author={Anonymous},
  booktitle={NeurIPS 2026 Evaluations \& Datasets Track},
  year={2026}
}
```

Replace `anonymous2026warp` with the de-anonymized citation after review.
