# WARP Case 118 — Dataset datasheet (Gebru et al. style)

## Motivation

Dual-labeled AC-OPF instances enable benchmarking learned warm starts for interior-point solvers under the same IPOPT configuration used in the paper.

## Composition

- **Grid**: `pglib_opf_case118_ieee` via PyTorch Geometric `OPFDataset`.
- **Splits**: `train` / `val` / `test` stored as `duals_XXXXXX.pt` files (see extraction script).
- **Fields per file** (IPOPT-converged): primal vector `x`, equality/inequality multipliers `lam_g`, bound multipliers `zl`, `zu`, barrier scalar `mu`, objective `obj`, iteration count metadata.

## Collection process

1. Sample loads from OPFDataset instances.
2. Build pandapower network; solve AC-OPF with IPOPT (cyipopt) using exact Hessian and documented tolerances.
3. Export full primal-dual-barrier state upon convergence.

## Preprocessing

- Optional flattening / normalisation: `compute_norm_stats.py` writes `norm_stats.pt` for training.

## Uses

Training and evaluation of models that predict interior-point state for warm starts (WARP-PD, baselines). Not intended for unit commitment or security-constrained OPF without adaptation.

## Distribution

- Primary: `.pt` files (this repository / supplementary zip).
- Discoverability: OpenML ARFF export via `upload_to_openml.py`.
- Optional Hub mirror: see `../huggingface/README.md`.

## Limitations & biases

See `rai:*` fields in `croissant.json`. Single-network study; synthetic load variation; computational cost scales poorly to large cases.

## Maintenance

Release artifacts are versioned with git tags. Issues and updates should track OpenML dataset revision IDs when republishing.

## License

MIT (dataset labels and code).
