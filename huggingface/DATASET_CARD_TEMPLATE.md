---
license: mit
language:
- en
tags:
- optimization
- optimal-power-flow
- physics-informed
- pytorch
pretty_name: WARP OPF Case118 Dual-Labeled
size_categories:
- 10K<n<100K
---

# WARP-OPF Case118 DualLabeled

Dual-labeled AC optimal power flow instances for benchmarking interior-point warm starts on `pglib_opf_case118_ieee` (via PyTorch Geometric `OPFDataset`).

## Dataset summary

Each training example is an IPOPT-converged primal–dual–barrier snapshot saved as `duals_XXXXXX.pt` with tensors compatible with the code release (`warp-benchmark`).

## Primary registry

NeurIPS / ML discovery: also published on **OpenML** (see paper supplementary). Replace this sentence with the OpenML URL after upload.

## Source code

Anonymous review: link the anonymous GitHub ZIP from OpenReview. Camera-ready: link the public `warp-benchmark` repository.

## Citation

```bibtex
@inproceedings{anonymous2026warp,
  title={Why Primal-Only Warm-Starts Fail for Interior-Point Solvers},
  booktitle={NeurIPS 2026 Evaluations \& Datasets Track},
  year={2026}
}
```

## Croissant

See `croissant.json` in this dataset repository root.
