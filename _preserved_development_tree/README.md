# WARP: Warm-start via Adversarial Residual Prior

**Reducing IPOPT iterations for AC Optimal Power Flow using diffusion-based warm-starts.**

WARP learns a conditional distribution over AC-OPF solutions using a heterogeneous GNN as the denoiser backbone of a DDPM. At inference, it samples K warm-start candidates, scores each via a cheap AC power balance residual, and passes the best to IPOPT. The solver still runs — so feasibility is exact. We just make it converge in far fewer iterations.

## Key Idea

```
Load scenario → HetGNN Denoiser (DDPM) → K warm-start samples
    → AC residual scoring → best sample → IPOPT → exact solution
```

The primary metric is **IPOPT interior-point iteration count**, not MSE or optimality gap. IPOPT still runs and guarantees feasibility — WARP just gives it a better starting point.

## Architecture

- **HetGNN denoiser**: Typed message passing over bus, generator, load, and shunt nodes with per-relation MLPs, adaptive LayerNorm (adaLN) for timestep conditioning, and Laplacian positional encoding
- **DDPM with cosine schedule**: 1000 training steps, DDIM sampling with 50 steps at inference
- **Physics-informed loss**: Differentiable AC power balance residuals as auxiliary training signal on the Tweedie x0 estimate, with time-weighted scheduling and clamping for stability
- **Multi-sample scoring**: Generate K candidates, rank by AC residual, pass best to IPOPT

## Repository Structure

```
warp/
├── models/
│   ├── hetgnn.py          # HetGNN denoiser (core architecture)
│   ├── diffusion.py       # DDPM/DDIM, cosine noise schedule
│   ├── det_gnn.py         # Deterministic GNN baseline
│   └── embeddings.py      # Sinusoidal timestep, adaLN, Laplacian PE
│
├── physics/
│   ├── acpf.py            # Differentiable AC power flow residuals
│   ├── admittance.py      # Y-bus construction from HeteroData
│   └── constraints.py     # Voltage/thermal penalty losses
│
├── training/
│   ├── train_warp.py      # WARP diffusion training
│   ├── train_det.py       # DetGNN baseline training
│   ├── losses.py          # L_ddpm + weighted L_physics
│   └── scheduler.py       # Cosine LR with warmup
│
├── inference/
│   ├── sample.py          # DDIM sampler, multi-sample generation
│   ├── score.py           # AC residual scoring for sample selection
│   └── warmstart.py       # End-to-end: sample → score → IPOPT
│
├── eval/
│   ├── ipopt_wrapper.py   # Pandapower/IPOPT interface
│   ├── metrics.py         # IPM iteration counter, WS-RMSE
│   ├── benchmark.py       # Full evaluation loop
│   └── ablation.py        # Ablation sweep runner
│
├── baselines/
│   ├── flat_start.py      # V=1, θ=0 baseline
│   ├── dc_warmstart.py    # DC-OPF warm-start
│   ├── det_gnn_baseline.py
│   └── diffopf_baseline.py
│
├── data/
│   ├── opfdata.py         # OPFDataset loader wrapper
│   ├── transforms.py      # Normalisation, typed graphs, PE
│   └── splits.py          # Train/val/test split logic
│
├── experiments/
│   ├── configs/           # YAML configs per experiment
│   ├── run_experiment.py  # Config-driven training
│   └── sweep.py           # Hyperparameter sweep
│
├── colab_warp.py          # Single-cell Colab runner (A100)
├── requirements.txt
└── setup.py
```

## Quick Start

### Local development (CPU, case14/case57)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torch-geometric pandapower scipy pyyaml tqdm

# Validate physics module (Gate 1)
PYTHONPATH=. python -c "
from torch_geometric.datasets import OPFDataset
from physics.admittance import build_ybus_from_heterodata
from physics.acpf import compute_residuals
ds = OPFDataset(root='data/opfdata', case_name='pglib_opf_case14_ieee', split='train')
d = ds[0]; G, B = build_ybus_from_heterodata(d)
dP, dQ = compute_residuals(d, G, B)
print(f'Max residual: {max(dP.abs().max(), dQ.abs().max()):.2e}')  # should be < 1e-4
"

# Train DetGNN baseline (2-epoch smoke test)
PYTHONPATH=. python training/train_det.py --case case14 --epochs 2 --hidden-dim 64 --num-layers 4

# Train WARP diffusion model
PYTHONPATH=. python training/train_warp.py --case case14 --epochs 5 --hidden-dim 64 --num-layers 4
```

### GPU training (Colab A100, case118)

Paste `colab_warp.py` into a single Colab cell and run. It handles everything: installs deps, downloads data, trains DetGNN + WARP, and benchmarks against flat-start. ~20-30 min on A100.

## Data

Uses [OPFDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.OPFDataset.html) from PyG — heterogeneous graph data derived from [pglib-opf](https://github.com/power-grid-lib/pglib-opf).

Supported grids: `case14`, `case30`, `case57`, `case118`, `case500`, `case2000`

### Data format (HeteroData)

| Field | Shape | Description |
|-------|-------|-------------|
| `bus.x` | `[n_bus, 4]` | Bus features (base_kv, type, Vm_min, Vm_max) |
| `bus.y` | `[n_bus, 2]` | Solution: (Va, Vm) |
| `generator.x` | `[n_gen, 11]` | Generator features (limits, costs) |
| `generator.y` | `[n_gen, 2]` | Solution: (Pg, Qg) |
| `load.x` | `[n_load, 2]` | Load: (Pd, Qd) in per-unit |
| `shunt.x` | `[n_shunt, 2]` | Shunt: (Bs, Gs) — susceptance first |
| `ac_line.edge_attr` | `[n_line, 9]` | [angmin, angmax, b_fr, b_to, r, x, rate_a/b/c] |
| `transformer.edge_attr` | `[n_xfmr, 11]` | [angmin, angmax, r, x, rate_a/b/c, tap, shift, ...] |

## Experimental Results (case118, A100)

### Training (batch_size=64, 15 epochs each)

| Phase | Time | Final Loss |
|-------|------|------------|
| DetGNN | ~30 min (2 min/epoch) | val MSE 0.045, WS-RMSE 0.149 |
| WARP | ~53 min (3.5 min/epoch) | val L_ddpm 0.46, L_phy 9.0 |

### Warm-start Quality (RMSE vs ground-truth OPF solution)

| Method | Bus (Va, Vm) RMSE | Gen (Pg, Qg) RMSE |
|--------|-------------------|-------------------|
| DetGNN | 0.12 | 0.15 |
| WARP-K3 | 0.17 | 1.24 |

DetGNN's deterministic predictions are significantly more accurate than WARP's diffusion samples, particularly for generator variables (Pg ranges [0, 9.2] per-unit which is hard for the diffusion model).

### Benchmark (IPOPT iteration count, 50 test instances)

With warm-starts properly injected into PIPS (see insights below):

| Method | Mean Iters | Median | vs Flat |
|--------|-----------|--------|---------|
| Flat start | 19.6 | 18 | — |
| DetGNN WS | 31.0 | 27 | -58% (worse) |
| WARP-K3 WS | 31.0 | 30 | -58% (worse) |

Both warm-starts currently **hurt** convergence. See insights below for analysis.

## Key Insights and Known Issues

### 1. Pandapower's PIPS solver ignores `init="results"`

The PIPS interior-point solver (`pipsopf_solver.py`) always overwrites x0 with `(lower_bound + upper_bound) / 2` unless `init == "pf"`. This means `runopp(init="results")` loads bus voltages from `res_bus` into the PPC but the solver then discards them. The benchmark monkey-patches `opf_execute.pipsopf_solver` to treat `"results"` like `"pf"` so x0 is preserved.

### 2. Running `runpp` before `runopp` washes out warm-start differences

If you run `pp.runpp(init="results")` to seed a power flow solution and then `pp.runopp(init="pf")`, Newton-Raphson converges to the same unique PF fixed point regardless of the initial warm-start (as long as both are in the basin of attraction). This makes all warm-starts appear identical.

### 3. DDIM sampling explodes at high noise timesteps

The cosine schedule gives `sqrt(alpha_cumprod[999]) ≈ 0.00005`. The DDIM x0 prediction divides by this value: `x0 = (x_t - sqrt(1-alpha)*eps) / sqrt(alpha)`, amplifying errors by ~20,000x at t=999. Fix: start sampling from `t_start = 0.98*T` and clamp x0 predictions to physically meaningful ranges per variable (Va ∈ [-1, 0.5], Vm ∈ [0.9, 1.1], Pg ∈ [-0.5, 10], Qg ∈ [-4, 5]).

### 4. Interior-point methods need feasible-interior starting points

PIPS's flat start `(lb + ub) / 2` is designed to be a well-centered interior point. Model predictions that violate variable bounds or are near constraint boundaries can push x0 outside the feasible region, causing the IPM to take more steps recovering. Future work: project predictions onto the feasible interior before passing to IPOPT.

### 5. PyG batching with HeteroData and physics loss

PyG's `DataLoader` handles HeteroData batching automatically (offsets edge indices, creates batch vectors). The GNN's message passing and supervised losses work with batched graphs. However, the physics loss (`build_ybus`) constructs a dense N×N admittance matrix and cannot operate on merged graphs. Fix: extract one random graph per batch via `data.to_data_list()` for physics loss computation.

## Validation Gates

The project follows a strict gate system to catch bugs early:

| Gate | Criterion | What it validates |
|------|-----------|-------------------|
| **Gate 1** | AC power balance residuals < 1e-4 on ground-truth | Y-bus + physics module correct |
| **Gate 2** | DetGNN WS-RMSE < 0.05 on case118 | GNN architecture can learn OPF |
| **Gate 3** | WARP-K1 reduces IPOPT iters vs flat-start | Diffusion warm-start actually helps |

## Dependencies

```
torch>=2.1.0
torch-geometric>=2.4.0
pandapower>=2.13.0
numpy, scipy, pyyaml, tqdm, matplotlib
```

## Citation

```bibtex
@article{warp2025,
  title={WARP: Warm-start via Adversarial Residual Prior for AC Optimal Power Flow},
  year={2025}
}
```
