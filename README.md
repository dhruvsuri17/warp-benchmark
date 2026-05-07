# WARP Benchmark: Primal-Dual Warm-Starting for Interior-Point Solvers

Public release layout for the WARP benchmark suite (interior-point warm starts on AC optimal power flow). This folder is self-contained: run all commands with working directory `warp-benchmark/` and `PYTHONPATH` set to `.` (or install with `pip install -e .`).

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
| `croissant.json` | Machine-readable dataset metadata (NeurIPS E&D); fill OpenML ID and hashes before submission. |

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

- **Why “data” looked empty**: Large `.pt` tensors are **gitignored** on purpose (NeurIPS supplementary zip / HF/OpenML carry the bytes). In this checkout, `warp-benchmark/data/` only shipped scripts and placeholders until you generate or copy artifacts.
- **Your machine**: Dual labels already live under the parent repo at `warp/data/duals/` (thousands of `.pt` files). This release wires them in via **`data/duals` → `../../data/duals`** so paths like `data/duals/pglib_opf_case118_ieee/train/` resolve without duplicating disk. **`data/pglib_opf_case118_ieee`** symlinks to the existing PyG **OPFDataset** cache under `warp/data/dataset_release_1/pglib_opf_case118_ieee/` so `OPFDataset(root="data", …)` works when you run from `warp-benchmark/`.
- **Primary artifact**: PyTorch `.pt` files per instance (dual labels). Evaluation expects `data/duals/<case_name>/{train,val,test}/duals_*.pt`. For OpenML tabular export, point `upload_to_openml.py` at the folder that contains `train/`, `val/`, `test/`.
- **Discoverability**: Upload tabular export to **OpenML** with `data/upload_to_openml.py` (`OPENML_API_KEY` **must** be supplied via environment — never commit secrets).
- **Optional mirror**: Publish dataset card + files on Hugging Face Hub (`huggingface/README.md`).

## Evaluation protocol

Aligned with `eval/opf_ipopt.py`: IPOPT via cyipopt, exact Hessian, warm-start settings documented in code (`solve_opf`). Iteration counts come from the IPOPT `intermediate()` callback.

## Publishing (HF / OpenML / git)

**Nothing is uploaded automatically.** Use [`scripts/PUBLISH.md`](scripts/PUBLISH.md): Hugging Face dataset (`scripts/hf_publish_dataset.py`), checkpoints (`scripts/hf_publish_models.py`), OpenML (`data/upload_to_openml.py`), then git push to your remote.

Requires `HF_TOKEN` (or `huggingface-cli login`) and `OPENML_API_KEY` as env vars — never commit tokens.

## Croissant

Edit `croissant.json`: set `DATASET_ID` from OpenML after upload, run `python scripts/update_croissant.py`, then validate with `mlcroissant` or the [online validator](https://huggingface.co/spaces/MLCommons/croissant-validator).

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
