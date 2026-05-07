# WARP: Warm-start via Adversarial Residual Prior
## Claude Code Project Configuration

This file gives Claude Code full context on the WARP research project.
Read this entirely before touching any file. Every architectural decision here
was made deliberately — don't second-guess the structure without asking.

---

## 1. What This Project Is

WARP is a NeurIPS 2026 submission. The core idea:

> AC Optimal Power Flow (AC-OPF) solvers like IPOPT spend most of their time
> in Newton iterations. The number of iterations is highly sensitive to the
> initial point. WARP learns a conditional distribution over OPF solutions using
> a **heterogeneous GNN as the denoiser backbone of a DDPM**, samples K warm-
> starts at inference, scores each via a cheap AC power balance residual, and
> passes the best to IPOPT. The solver still runs — so feasibility is exact.
> We just make it converge in far fewer iterations.

**The primary metric is IPOPT interior-point iteration count, not MSE or
optimality gap.** This is the key differentiator from every prior ML-OPF paper.

**Do not suggest replacing IPOPT.** That is a different paper (CANOS, DeepOPF).
This paper's entire value proposition is that IPOPT still runs.

---

## 2. Repo Structure

```
warp/
├── CLAUDE.md                  ← this file
├── README.md
├── requirements.txt
├── setup.py
│
├── data/
│   ├── opfdata.py             ← OPFDataset loader (PyG), all preprocessing
│   ├── transforms.py          ← feature normalisation, PE computation
│   └── splits.py              ← train/val/test split logic
│
├── models/
│   ├── hetgnn.py              ← HetGNN denoiser (THE core architecture)
│   ├── diffusion.py           ← DDPM forward/reverse process, DDIM sampler
│   ├── det_gnn.py             ← deterministic baseline (same backbone, no diff)
│   └── embeddings.py          ← timestep sinusoidal, Laplacian PE
│
├── physics/
│   ├── acpf.py                ← AC power flow residuals (differentiable)
│   ├── constraints.py         ← voltage/thermal penalty losses
│   └── admittance.py          ← Y-bus construction from PyG edge features
│
├── training/
│   ├── train_warp.py          ← main WARP training script
│   ├── train_det.py           ← deterministic GNN baseline training
│   ├── losses.py              ← L_ddpm + L_physics combined loss
│   └── scheduler.py           ← cosine LR schedule, noise schedule
│
├── inference/
│   ├── sample.py              ← DDIM/DDPM sampler, multi-sample scoring
│   ├── warmstart.py           ← IPOPT warm-start harness
│   └── score.py               ← AC residual scoring for K samples
│
├── eval/
│   ├── metrics.py             ← IPM iteration counter, conv rate, WS RMSE
│   ├── benchmark.py           ← full evaluation loop across baselines
│   ├── ablation.py            ← ablation sweep runner
│   └── ipopt_wrapper.py       ← pandapower/pyipopt interface, iter extraction
│
├── baselines/
│   ├── flat_start.py          ← V=1, theta=0, Pg midpoint
│   ├── dc_warmstart.py        ← DC-OPF via pandapower, lifted to AC
│   ├── det_gnn_baseline.py    ← wrapper around det_gnn.py for eval
│   └── diffopf_baseline.py    ← DiffOPF reimplementation (MLP denoiser)
│
├── experiments/
│   ├── configs/               ← YAML configs per experiment
│   │   ├── warp_case118.yaml
│   │   ├── warp_case500.yaml
│   │   ├── warp_case2000.yaml
│   │   ├── det_gnn_case118.yaml
│   │   └── ablation_case118.yaml
│   ├── run_experiment.py      ← reads config, dispatches train + eval
│   └── sweep.py               ← hyperparameter sweep (K, lambda_phy, layers)
│
├── scripts/
│   ├── download_opfdata.py    ← downloads OPFData via PyG
│   ├── install_ipopt.sh       ← IPOPT + HSL MA57 setup (Linux)
│   ├── train_gpu.sh           ← GPU training launcher (tmux + logging)
│   └── eval_all.sh            ← runs full benchmark table
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_hetgnn_architecture.ipynb
│   ├── 03_results_tables.ipynb
│   └── 04_n1_analysis.ipynb
│
├── results/
│   ├── checkpoints/           ← model weights (git-ignored, >100MB)
│   ├── logs/                  ← MLflow or W&B run logs
│   └── tables/                ← CSV outputs for paper tables
│
└── paper/
    ├── WARPSTART.tex          ← main LaTeX source (deepmind.cls style)
    ├── warp_refs.bib          ← bibliography
    └── figs/                  ← generated figures (PDF)
```

**Never put results directly in the repo root. Everything goes in `results/`.**
**Never hardcode paths. Use `pathlib.Path` and config YAMLs everywhere.**

---

## 3. Core Architecture: HetGNN Denoiser

This is `models/hetgnn.py`. Get this right before anything else.

### Node types and feature dims

```python
NODE_TYPES = {
    "pq_bus":    4,   # (Pd, Qd, Vm_noisy, Va_noisy)
    "pv_bus":    6,   # (Pd, Qd, Pg_max, Pg_min, Vm_noisy, Va_noisy)
    "slack_bus": 4,   # (Vm_set, 0, Vm_noisy, Va_noisy)
    "generator": 8,   # (Pg_noisy, Qg_noisy, Pg_max, Pg_min,
                      #  Qg_max, Qg_min, cost_c1, cost_c2)
}

EDGE_TYPES = {
    "line":        5,  # (g_ij, b_ij, b_sh_ij, S_max, contingency_flag)
    "transformer": 5,  # (g_ij, b_ij, tap_ratio, phase_shift, S_max)
    "gen_bus":     2,  # (unit_type_onehot, commitment_status)
}
```

### Architecture constants (do not change without explicit instruction)

```python
HIDDEN_DIM    = 256
NUM_LAYERS    = 8
PE_DIM        = 16     # Laplacian eigenvector dim
TIMESTEP_DIM  = 128    # sinusoidal timestep embedding
MLP_LAYERS    = 2      # depth of message and update MLPs
```

### Message passing (typed, per relation)

For each layer `l` and each edge type `r`:
```
m_v = CONCAT over r [ SUM over u in N^r(v) [ psi^r(h_v, h_u, e_vu) ] ]
h_v = phi^{tau_v}( LN( h_v + W^{tau_v} @ m_v ) )
```

- `psi^r`: 2-layer MLP with LayerNorm, one per edge type
- `phi^{tau_v}`: 2-layer MLP with LayerNorm, one per node type
- Use **LayerNorm not BatchNorm** — graph batch sizes vary
- Use **residual connections** at every layer

### Timestep and contingency conditioning (adaLN)

```python
# timestep -> gamma, beta for adaLN
t_emb = sinusoidal_embedding(t, dim=TIMESTEP_DIM)
gamma, beta = MLP_2layer(t_emb).chunk(2, dim=-1)
h = gamma * LayerNorm(h) + beta
```

Apply adaLN **after** message aggregation, **before** the update MLP at every
layer. This is how DiT/Stormer do it — don't use cross-attention for timestep.

### Laplacian PE

```python
# Compute once per graph, cache it
L = compute_normalised_laplacian(edge_index, edge_weight=conductance)
eigvals, eigvecs = torch.linalg.eigh(L.to_dense())
pe = eigvecs[:, 1:PE_DIM+1]  # skip trivial eigenvector
# Concatenate pe to initial node features before layer 0
```

Under N-1 (line outage), `L` changes by rank-1 update — recompute PE.
**Cache PE per (graph_id, contingency_mask) tuple** to avoid recomputation
during multi-sample inference.

### Output head

```python
# Per bus: predict noise for (Vm, Va)
noise_bus = Linear(HIDDEN_DIM, 2)(h_bus)

# Per generator: predict noise for (Pg, Qg)  
noise_gen = Linear(HIDDEN_DIM, 2)(h_gen)

# Concatenate in canonical variable order: [Vm, Va, Pg, Qg]
noise = cat([noise_bus_Vm, noise_bus_Va, noise_gen_Pg, noise_gen_Qg])
```

---

## 4. Diffusion Process

This is `models/diffusion.py`.

### Noise schedule

```python
# Cosine schedule (Nichol & Dhariwal 2021)
def cosine_schedule(T=1000, s=0.008):
    t = torch.linspace(0, T, T+1)
    f = torch.cos((t/T + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0, 0.999)
```

### Training (forward process + noise prediction)

```python
# Sample random timestep
t = torch.randint(0, T, (batch_size,))

# Add noise
noise = torch.randn_like(x0)
x_t = sqrt_alphas_cumprod[t] * x0 + sqrt_one_minus_alphas_cumprod[t] * noise

# Predict noise
eps_pred = hetgnn(x_t, t, load, graph, contingency)

# Tweedie estimate of x0 for physics loss
x0_hat = (x_t - sqrt_one_minus_alphas_cumprod[t] * eps_pred) \
         / sqrt_alphas_cumprod[t]
```

### DDIM inference (50 steps)

Use DDIM (Song et al. 2021) at inference — 50 steps, deterministic.
Do not use stochastic DDPM at inference; it's slower and noisier for this task.

```python
# DDIM step
eta = 0.0  # deterministic
sigma = eta * sqrt((1 - alpha_prev) / (1 - alpha)) * sqrt(1 - alpha / alpha_prev)
x_prev = sqrt(alpha_prev) * x0_hat + sqrt(1 - alpha_prev - sigma**2) * eps_pred
```

---

## 5. Physics: AC Power Flow Residuals

This is `physics/acpf.py`. This module is used in **two places**:
1. As the auxiliary training loss on `x0_hat` (Tweedie estimate)
2. As the inference-time scoring function for multi-sample selection

**It must be fully differentiable (PyTorch ops only, no pandapower at train
time).** Pandapower is only used in the IPOPT evaluation harness.

### Y-bus construction

```python
def build_ybus(edge_index, r, x, b_sh, tap, shift, n_bus):
    """
    Returns Y_real (G matrix) and Y_imag (B matrix) as sparse tensors.
    Handles transformers via tap ratio and phase shift.
    """
    y_series = 1 / (r + 1j * x)
    # ... standard Y-bus construction
    # Return as torch.sparse_coo_tensor for efficient matmul
```

### Power balance residuals

```python
def ac_power_balance(Vm, Va, Pg, Qd, G, B):
    """
    Args:
        Vm:  [n_bus]  voltage magnitudes
        Va:  [n_bus]  voltage angles (radians)
        Pg:  [n_bus]  net active injection (gen - load), MW
        Qd:  [n_bus]  reactive load, MVAR
        G, B: [n_bus, n_bus] conductance and susceptance matrices
    Returns:
        dP: [n_bus]  active power mismatch
        dQ: [n_bus]  reactive power mismatch (PQ buses only)
    """
    Vcos = Vm.unsqueeze(1) * Vm.unsqueeze(0) * torch.cos(Va.unsqueeze(1) - Va.unsqueeze(0))
    Vsin = Vm.unsqueeze(1) * Vm.unsqueeze(0) * torch.sin(Va.unsqueeze(1) - Va.unsqueeze(0))
    P_calc = (G * Vcos + B * Vsin).sum(dim=1)
    Q_calc = (G * Vsin - B * Vcos).sum(dim=1)
    dP = P_calc - Pg
    dQ = Q_calc - Qd  # only at PQ buses in full formulation
    return dP, dQ
```

### Physics loss (used in training + inference scoring)

```python
def physics_loss(x0_hat, graph, lambda_V=1.0, lambda_S=1.0):
    Vm, Va, Pg, Qg = unpack_variables(x0_hat, graph)
    dP, dQ = ac_power_balance(Vm, Va, Pg - graph.Pd, graph.Qg - graph.Qd, G, B)

    L_balance = (dP**2 + dQ**2).sum()

    L_voltage = (F.relu(Vm - graph.Vm_max)**2 + F.relu(graph.Vm_min - Vm)**2).sum()

    S_flow = compute_line_flows(Vm, Va, graph)
    L_thermal = F.relu(S_flow.abs() - graph.S_max).pow(2).sum()

    return L_balance + lambda_V * L_voltage + lambda_S * L_thermal
```

---

## 6. Training Loss

This is `training/losses.py`.

```python
def warp_loss(eps_pred, eps_true, x0_hat, graph, t,
              lambda_phy=0.1, lambda_V=1.0, lambda_S=1.0):

    # Primary: DDPM noise prediction MSE
    L_ddpm = F.mse_loss(eps_pred, eps_true)

    # Auxiliary: physics on Tweedie x0 estimate
    # Downweight at low t (x0_hat is noisy) and high t (too early)
    # Weight peaks around t ~ 0.3*T
    phy_weight = physics_schedule(t)  # e.g. sin(pi * t/T) curve
    L_phy = physics_loss(x0_hat, graph, lambda_V, lambda_S)

    return L_ddpm + lambda_phy * phy_weight * L_phy, {
        "L_ddpm": L_ddpm.item(),
        "L_phy":  L_phy.item(),
    }
```

**Do not apply physics loss at all timesteps equally.** At high noise levels
(large `t`), the Tweedie estimate `x0_hat` is garbage and the physics loss
adds noise to the gradient. Use a weighting schedule that peaks at mid-t.

---

## 7. IPOPT Interface

This is `eval/ipopt_wrapper.py`. This is the most operationally critical file.

### What it must do

1. Accept a warm-start vector `x_ws = (Vm, Va, Pg, Qg)`
2. Set it as IPOPT's initial point via pandapower's `net` object
3. Run IPOPT to convergence
4. Return: `(x_opt, n_iterations, converged, obj_value, constraint_violations)`

### Getting iteration count out of IPOPT

IPOPT does not expose iteration count cleanly through pandapower's default
interface. Use one of these approaches (in order of preference):

**Option A (preferred): Parse IPOPT stdout**
```python
import subprocess, re

def run_ipopt_with_iter_count(net, x_ws, timeout=300):
    # Set warm start in pandapower net
    net.res_bus["vm_pu"] = x_ws["Vm"]
    net.res_bus["va_degree"] = np.degrees(x_ws["Va"])
    # ... set gen setpoints

    # Run with stdout capture
    result = pandapower.runopf(net, verbose=True, capture_output=True)
    
    # Parse iter count from IPOPT output
    match = re.search(r"Number of Iterations\.*:\s+(\d+)", result.stdout)
    n_iter = int(match.group(1)) if match else None
    return result, n_iter
```

**Option B: Use pyipopt directly**
```python
import pyipopt
# Build NLP from scratch using pandapower's internal formulation
# More work but gives clean programmatic access to iter count
```

**Option C (fallback): HSL iteration log**
HSL MA57 writes iteration logs to `/tmp/ipopt_iter.log` if you set
`option: output_file`. Parse that.

### HSL MA57 setup

IPOPT needs HSL MA57 for performance on large cases. Without it, case2000
will be very slow and baselines will be noisy.

```bash
# Get HSL license at: https://licences.stfc.ac.uk/product/coin-hsl
# After getting coin-hsl-archive.tar.gz:
cd /opt && tar xzf coin-hsl-archive.tar.gz
cd ThirdParty-HSL && ./configure --prefix=/usr/local && make install
# Recompile IPOPT with --with-hsl=/usr/local
```

If HSL is unavailable, use MUMPS (default) but note in the paper that
baselines were run with MUMPS for reproducibility.

---

## 8. Data Pipeline

This is `data/opfdata.py`.

### Loading OPFData via PyG

```python
from torch_geometric.datasets import OPFDataset

# Available grids: 14, 57, 118, 500, 2000, and others
# Available cases: 'fulltop', 'n-1'
dataset = OPFDataset(
    root="data/opfdata",
    case_name="pglib_opf_case118_ieee",
    split="train",
    topological_perturbations=False,  # False = FullTop, True = N-1
)
```

### What OPFData provides per instance

```
data.x          # node features (load, gen limits, etc.)
data.edge_index # connectivity
data.edge_attr  # line parameters (r, x, b, S_max)
data.y          # optimal solution (Vm, Va, Pg, Qg) — this is our x^*
data.obj        # optimal objective value
```

### Preprocessing in `data/transforms.py`

1. **Normalise all features** to zero mean, unit variance using train-set stats
   — save normalisation stats as `data/opfdata/{case}/norm_stats.pt`
2. **Build typed graph**: split `data.x` into PQ/PV/slack/generator node types
   based on bus type flags in OPFData
3. **Compute Laplacian PE**: top-16 eigenvectors of normalised Laplacian
   weighted by conductance. Cache as `data/opfdata/{case}/pe_cache.pt`
4. **N-1 contingency**: for N-1 split, the contingency mask is in
   `data.contingency_mask` — a binary vector over lines. Set the corresponding
   edge features' `contingency_flag` to 1 and zero out the line's admittance
   contribution to Y-bus

### Variable ordering (canonical, never deviate)

```
x = [Vm_1, ..., Vm_N, Va_1, ..., Va_N, Pg_1, ..., Pg_G, Qg_1, ..., Qg_G]
```

This ordering is used everywhere: diffusion noise, physics residuals, IPOPT
warm-start setter, and metrics. If you change it, everything breaks silently.

---

## 9. Evaluation Harness

This is `eval/benchmark.py`. Run this to reproduce Table 1 and Table 2.

### Baseline execution order (run in this order, each depends on prior)

```
1. flat_start        → get baseline iteration counts (ceiling)
2. dc_warmstart      → intermediate baseline
3. det_gnn           → train first (see §11), then eval
4. diffopf           → retrain on OPFData, then eval
5. warp_k1           → WARP with K=1 (isolates prior quality)
6. warp_k5           → WARP with K=5 (full method)
```

### Metrics to collect per instance

```python
{
    "case": "case118",
    "split": "fulltop",          # or "n-1"
    "method": "warp_k5",
    "instance_id": int,
    "n_ipm_iters": int,          # PRIMARY METRIC
    "converged": bool,
    "obj_value": float,
    "obj_gap_pct": float,        # vs flat-start converged obj
    "ws_rmse": float,            # ||x_ws - x_opt|| / range
    "max_violation": float,      # max constraint violation at convergence
    "diffusion_time_s": float,   # inference time (WARP only)
    "ipopt_time_s": float,
    "total_time_s": float,
}
```

Save everything to `results/tables/{case}_{split}_{method}.csv`.
The paper table generator in `notebooks/03_results_tables.ipynb` reads these.

### Statistical reporting

Always report: mean, std, median, 90th percentile of IPM iterations.
For convergence rate: fraction converging within 300 iterations.
For N-1: break down by line-loading quartile (Q1=lightly loaded outages,
Q4=heavily loaded outages — this is where WARP's topology-awareness matters
most and should be highlighted).

---

## 10. Experiment Configs (YAML)

Example `experiments/configs/warp_case118.yaml`:

```yaml
experiment:
  name: "warp_case118_fulltop"
  case: "pglib_opf_case118_ieee"
  split: "fulltop"
  seed: 42

model:
  type: "warp"
  hidden_dim: 256
  num_layers: 8
  pe_dim: 16
  timestep_dim: 128

diffusion:
  T: 1000
  schedule: "cosine"
  ddim_steps: 50   # inference only
  K: 5             # samples at inference

training:
  epochs: 200
  batch_size: 64
  lr: 1.0e-4
  lr_schedule: "cosine"
  weight_decay: 1.0e-5
  lambda_phy: 0.1
  lambda_V: 1.0
  lambda_S: 1.0
  grad_clip: 1.0
  mixed_precision: "bf16"

eval:
  ipopt_max_iter: 300
  ipopt_tol: 1.0e-6
  ipopt_linear_solver: "ma57"  # or "mumps" if HSL unavailable

paths:
  data_root: "data/opfdata"
  checkpoint_dir: "results/checkpoints/warp_case118"
  log_dir: "results/logs/warp_case118"
  results_dir: "results/tables"
```

---

## 11. Execution Order: What to Build First

**Do not start with diffusion. Build the deterministic GNN baseline first.**
The Det-GNN is 80% of WARP's code and validates the architecture before you
add the diffusion complexity. If Det-GNN is broken, WARP will be broken.

### Phase 1: Infrastructure (Day 1–2)

```
□ data/opfdata.py        — OPFDataset loader, verify shapes
□ data/transforms.py     — normalisation, typed graph construction
□ physics/admittance.py  — Y-bus from edge features
□ physics/acpf.py        — power balance residuals (test against pandapower)
□ eval/ipopt_wrapper.py  — IPOPT harness, verify iter count extraction
```

**Validation gate**: Run `python scripts/download_opfdata.py --case case118`
and verify that `ac_power_balance()` on a ground-truth solution gives residuals
< 1e-4. If it doesn't, the physics module is wrong.

### Phase 2: Det-GNN Baseline (Day 3–4)

```
□ models/embeddings.py    — Laplacian PE, sinusoidal timestep embed
□ models/hetgnn.py        — typed MPNN, adaLN, output head
□ training/train_det.py   — MSE training on (load, graph) -> x*
□ eval/benchmark.py       — partial: flat_start, dc_ws, det_gnn
```

**Validation gate**: Det-GNN on case118 should achieve WS-RMSE < 0.05 on
the test set after 50 epochs. If not, debug the architecture before continuing.

### Phase 3: WARP (Day 5–7)

```
□ models/diffusion.py     — DDPM, cosine schedule, Tweedie formula
□ training/losses.py      — L_ddpm + weighted L_phy
□ training/train_warp.py  — full WARP training loop
□ inference/sample.py     — DDIM sampler
□ inference/score.py      — multi-sample AC residual scoring
□ inference/warmstart.py  — end-to-end: sample -> score -> IPOPT
```

**Validation gate**: WARP-K1 on case118 FullTop should reduce mean IPM
iterations vs flat-start. If K1 is worse than Det-GNN warm-start, the
diffusion training is broken (likely physics loss weight or Tweedie formula).

### Phase 4: Full Experiments (Day 8–12)

```
□ Train on case500, case2000
□ Run N-1 evaluation (zero-shot — no retraining)
□ K sweep (K=1,2,5,10,20) on case2000
□ Ablation table (case118)
□ DiffOPF baseline (retrain on OPFData)
```

### Phase 5: Paper (Day 13–14)

```
□ Fill results tables in WARPSTART.tex
□ Generate figures: IPM distribution plots, K sweep, N-1 breakdown
□ Ablation table
□ Write discussion of N-1 results
```

---

## 12. GPU Training Commands

```bash
# Single GPU (development)
python training/train_warp.py \
    --config experiments/configs/warp_case118.yaml \
    --gpu 0

# Multi-GPU (case2000, use all available A100s)
torchrun --nproc_per_node=4 training/train_warp.py \
    --config experiments/configs/warp_case2000.yaml

# Background training with logging (cloud VM)
tmux new -s warp_case2000
python training/train_warp.py \
    --config experiments/configs/warp_case2000.yaml \
    2>&1 | tee results/logs/warp_case2000.log
# Ctrl+B D to detach

# Check training progress
tail -f results/logs/warp_case2000.log

# Full eval (after training)
bash scripts/eval_all.sh case2000
```

---

## 13. Dependencies

```
# requirements.txt
torch>=2.1.0
torch-geometric>=2.4.0
torch-scatter
torch-sparse
pandapower>=2.13.0
numpy>=1.24.0
scipy>=1.10.0
pyyaml>=6.0
mlflow>=2.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pytest>=7.4.0
```

Install:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install -r requirements.txt
```

IPOPT:
```bash
bash scripts/install_ipopt.sh
# Verify:
python -c "import pandapower; import pandapower.converter; print('pandapower OK')"
python -c "from pyipopt import pyipopt; print('pyipopt OK')"
```

---

## 14. Known Issues and Gotchas

**OPFData N-1 split size**: The N-1 split has `n_instances × n_lines` rows.
For case2000 (3206 lines × 300k instances) this is enormous — do not try to
load it all into memory. Use `torch.utils.data.DataLoader` with `num_workers=4`
and stream from disk.

**Laplacian PE under N-1**: When a line is outaged, recompute PE from the
modified Laplacian. Do not reuse the FullTop PE — the eigenvectors change
non-trivially under even a single line removal, especially for lines with high
betweenness centrality.

**IPOPT initial point format**: pandapower's `runopf()` does not accept a
warm-start directly in all versions. You may need to set:
```python
net.res_bus.loc[:, "vm_pu"] = Vm_warmstart
net.res_bus.loc[:, "va_degree"] = Va_warmstart_degrees
# And for generators:
net.res_gen.loc[:, "p_mw"] = Pg_warmstart
net.res_gen.loc[:, "q_mvar"] = Qg_warmstart
```
Then call `pandapower.runopf(net, init="results")`.

**Variable ordering in OPFData vs WARP**: OPFData stores solution in a
different ordering than WARP's canonical order. The transform in
`data/transforms.py` must reorder to `[Vm, Va, Pg, Qg]` before training.
Verify with an assertion after loading.

**Physics loss scale**: The AC power balance residuals are in per-unit (MW/MVA
base). At the start of training, `L_phy` may be orders of magnitude larger
than `L_ddpm`. Use `lambda_phy=0.1` and the time-weighted schedule to prevent
physics loss from dominating early in training.

**Batch normalisation**: Do NOT use BatchNorm anywhere in the GNN. Graph
batch sizes vary and BatchNorm statistics will be wrong on small batches or
single-graph inference. Use LayerNorm only.

**case2000 memory**: The Y-bus for case2000 is 2000×2000. Keep it sparse
(`torch.sparse_coo_tensor`). Dense matmul on 2000×2000 per batch will OOM
even on A100-80GB.

**Gradient clipping**: Use `grad_clip=1.0` (global norm). Diffusion models
on structured data can have gradient spikes around t≈0 (denoising near the
data manifold). Without clipping, training will diverge intermittently.

---

## 15. Paper-Specific Notes

**The paper is WARPSTART.tex in `paper/`.** Use deepmind.cls (same as GridGen).
All figures go in `paper/figs/` as PDFs. Generate them from
`notebooks/03_results_tables.ipynb` and `notebooks/04_n1_analysis.ipynb`.

**Table placeholders**: Results tables in the LaTeX have `--` placeholders.
Fill them from `results/tables/*.csv`. Do not manually type numbers — generate
the LaTeX table rows from a script to avoid transcription errors.

**The key claim to protect**: "WARP reduces mean IPOPT interior-point
iterations by X× on case2000 N-1 vs flat-start, zero-shot, while preserving
exact feasibility." Every experiment should be designed to make this claim
stronger or to identify its failure modes.

**DiffOPF comparison**: Must retrain DiffOPF on OPFData case118/500/2000.
Their paper only shows case14/30/57. If DiffOPF fails to scale to case2000,
that is a result worth reporting (not just ignoring the comparison).

**NeurIPS submission checklist** (from reviewer guidelines):
- [ ] Theoretical claim or strong empirical claim, not both weakly
- [ ] Reproducibility: all hyperparams in paper + appendix
- [ ] Baselines run under identical conditions (same IPOPT version, same HSL)
- [ ] Error bars on main table (std over 3 seeds)
- [ ] Ablation covers every architectural decision
- [ ] Limitations section is honest (no global optimality, no sub-second for K=5)
- [ ] Code released with submission (anonymised)

---

## 16. Contact and Review

This project is research-only. Do not deploy any part of WARP to production
grid operations without extensive validation by licensed power systems engineers.

IPOPT solutions are local optima only. WARP does not affect which local optimum
IPOPT finds — only how fast it gets there from the warm-start.

**Questions about power systems physics**: consult `physics/` module docstrings
and the AC-OPF formulation in `paper/WARPSTART.tex` Section 2.1.
**Questions about diffusion**: consult `models/diffusion.py` and
Nichol & Dhariwal (2021), Song et al. DDIM (2021).
**Questions about GNN architecture**: consult `models/hetgnn.py` and
OPF-HGNN (Ghamizi et al. 2024), CANOS (Piloto et al. 2024).
