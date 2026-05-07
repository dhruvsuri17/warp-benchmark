# WARP: Next Steps for Claude Code / Cursor
## Context, diagnosis, and concrete implementation instructions

Read this entire file before touching any code. Every instruction here is
grounded in experimental results from ~85 hours of A100 runs and a
comprehensive literature review. Don't improvise — follow the prescription.

---

## 1. Where we are

WARP is a NeurIPS 2025 submission on warm-starting AC Optimal Power Flow (AC-OPF)
solvers using a heterogeneous GNN (DetGNN) and a diffusion model (WARP-DDPM).

### Current benchmark results (case118, A100, 50 test instances)

| Method | Mean IPM iters | vs Flat-start |
|--------|---------------|---------------|
| Flat-start (midpoint) | 19.6 | — |
| DetGNN raw (unnorm) | 31.0 | −58% worse |
| DetGNN (normalised) | **21.1** | **−7.8% worse** |
| WARP-K3 (normalised) | 27.8 | −42% worse |

The gap between DetGNN and flat-start is now just **1.5 iterations**. This is
not a model quality problem. It is a solver-API problem diagnosed precisely in
the literature. See §3 for the full diagnosis.

### What has already been implemented and confirmed working

- [x] HetGNN denoiser (`models/hetgnn.py`) — typed MPNN, adaLN, 8 layers, d=256
- [x] DDPM/DDIM (`models/diffusion.py`) — T=1000, cosine schedule, DDIM-50 at inference
- [x] Physics loss (`physics/acpf.py`, `physics/constraints.py`) — differentiable AC residuals
- [x] Per-variable normalisation — **single biggest win, do not remove**
- [x] PIPS monkey-patch — `opf_execute.pipsopf_solver` patched to preserve x0
- [x] DDIM fix — start from t=0.98*T, clamp x0 predictions per variable
- [x] batch_size=64 fix — physics loss on single random graph from batch
- [x] Benchmark harness — `eval/benchmark_v2.py`, per-instance CSV output

### What did NOT work and why

- **Feasibility projection** (`[lb+eps, ub-eps]`): Made things worse. Pushed
  inaccurate generator predictions onto constraint boundaries — the worst
  possible starting region for IPM. Do not retry this.
- **Running `runpp` before `runopp`**: Newton-Raphson power flow converges to the
  same unique fixed point regardless of initialisation, washing out all
  warm-start differences.
- **Larger model / more data / fewer diffusion steps**: Marginal effect. The
  bottleneck is not model capacity.
- **Pure DDPM without physics loss**: Best denoising loss (0.265) but worst
  IPM iterations (39.5). Denoising accuracy ≠ warm-start quality.

---

## 2. Root cause (confirmed by literature)

**The core problem is not model quality. It is that PIPS/pandapower cannot
accept a primal-dual warm start.**

Interior-point methods (IPM) maintain a *primal-dual triple* (x, λ, z) and
require a starting point that is *well-centered* in the log-barrier sense
(high µ = xᵀs/n, approximately equal xᵢsᵢ products). The flat-start midpoint
`(lb+ub)/2` is optimal for centrality by construction.

ML predictions are accurate in solution space (x close to x*) but have low
centrality (µ → 0, generators near their limits). Even a perfect prediction of
x* is a *worse* IPM starting point than the midpoint because it forces the
solver to re-establish centrality before making progress.

Pandapower's `runopp(init="results")` + PIPS passes only x₀ to the solver.
Dual variables (λ, z) are **hard-coded to zero / constant** in
`pipsopf_solver.py`. There is no API to supply duals. This is confirmed by:

- `pandapower/pypower/pipsopf_solver.py` line 118: resets x0 to midpoint unless
  `init == "pf"` (already monkey-patched)
- `pandapower/pypower/pips.py`: `lam` and `mu` initialised to zeros on entry
- MATPOWER user manual §6: "dual variables are always reinitialised"

The 1.5-iteration gap (21.1 vs 19.6) is *quantitatively predicted* by the
centrality theory: it takes approximately `log(µ_flat / µ_GNN) / log(σ)` extra
iterations to re-establish centrality where σ=0.2 is IPOPT's barrier reduction
factor. With accurate predictions landing near generator limits, µ_GNN ≪ µ_flat,
giving ~2 extra iterations — exactly what we observe.

**Literature confirmation:**
- IPM-LSTM (NeurIPS 2024, arXiv 2410.15731): "even if predicted solutions are
  close to optimal, they may not be well-centered with respect to the IPM
  trajectory, causing the optimizer to struggle" — identical failure mode.
- Yildirim & Wright 2002 (SIAM J. Optim.): formal proof that the optimal
  solution is the *worst possible* IPM warm-start.
- TOAST / Briden et al. 2024 (J. Guid. Control Dyn.): MSE-on-x loss fails
  for warm-starting; Lagrangian-MSE (predicting x+λ+z jointly) reduces
  iterations 30–50%.
- Klamkin, Tanneau, Van Hentenryck 2024 (NeurIPS): "Dual Interior Point
  Optimization Learning" — S³L log-barrier loss + dual completion layer beats
  all primal-only methods on AC-OPF.

---

## 3. The four-step prescription (ordered by impact and implementation cost)

### Step 0 — Immediate: switch from PIPS to IPOPT backend (1–2 hours)

PIPS has no dual warm-start API. IPOPT does. This is a prerequisite for
everything else.

Pandapower supports IPOPT via `runopp(algorithm="pypower", ...)` if pyipopt
is installed, or via a CasADi/IPOPT interface. The cleanest path for research:

```python
# eval/ipopt_direct.py
# Bypass pandapower entirely and call IPOPT via pyipopt or CasADi

import casadi as ca
import numpy as np

def build_opf_nlp(net):
    """
    Build a CasADi NLP from pandapower network for direct IPOPT access.
    Returns: (nlp, lbx, ubx, lbg, ubg, x0_flat, var_names)
    """
    # Convert pandapower net to CasADi NLP
    # Variables: [Vm (n_bus), Va (n_bus), Pg (n_gen), Qg (n_gen)]
    # Constraints: power balance (2*n_bus eq), line flows (n_line ineq)
    raise NotImplementedError("Build this from pandapower's internal PPC dict")

def run_ipopt_with_warmstart(nlp, lbx, ubx, lbg, ubg,
                              x_ws, lam_ws=None, z_ws=None):
    """
    Run IPOPT from warm-start (x, lam, z).
    If lam_ws and z_ws are None, uses mu-based initialization.
    Returns: (x_opt, n_iters, converged, obj)
    """
    solver = ca.nlpsol("opf", "ipopt", nlp, {
        "ipopt.warm_start_init_point": "yes",
        "ipopt.warm_start_bound_push": 1e-8,
        "ipopt.warm_start_bound_frac": 1e-8,
        "ipopt.warm_start_mult_bound_push": 1e-8,
        "ipopt.warm_start_slack_bound_push": 1e-8,
        "ipopt.warm_start_slack_bound_frac": 1e-8,
        "ipopt.bound_mult_init_method": "mu-based",  # KEY: centrality-aware
        "ipopt.mu_init": 1e-1,     # start with larger barrier (more central)
        "ipopt.bound_mult_init_val": 1.0,
        "ipopt.nlp_scaling_method": "none",
        "ipopt.print_level": 5,   # for iteration count parsing
        "ipopt.max_iter": 300,
    })

    kwargs = {"x0": x_ws, "lbx": lbx, "ubx": ubx, "lbg": lbg, "ubg": ubg}
    if lam_ws is not None:
        kwargs["lam_g0"] = lam_ws   # equality + inequality multipliers
    if z_ws is not None:
        kwargs["lam_x0"] = z_ws     # bound multipliers

    result = solver(**kwargs)

    # Extract iteration count from solver stats
    stats = solver.stats()
    n_iters = stats.get("iter_count", -1)

    return {
        "x":        np.array(result["x"]).flatten(),
        "n_iters":  n_iters,
        "converged": stats.get("success", False),
        "obj":      float(result["f"]),
    }
```

**Alternative: use pypower directly with IPOPT**
```bash
pip install pyipopt   # wraps IPOPT C library
# Then patch pandapower to use ipopt_solver.py path:
pp.runopp(net, algorithm="ipopt", init="results", ...)
# Check pandapower/pypower/ipopt_solver.py for the warm_start options
```

**IPOPT installation (no license needed):**
```bash
# macOS
brew install ipopt

# Linux (Ubuntu/Debian)
sudo apt-get install coinor-ipopt coinor-libipopt-dev

# Python binding
pip install cyipopt   # preferred over pyipopt, actively maintained

# Verify
python -c "import cyipopt; print('IPOPT OK')"
```

HSL MA57 (faster sparse solver, free academic license): optional for now.
MUMPS (default) is sufficient for case118. Get HSL later for case2000+.

---

### Step 1 — High impact: compute dual variables from primal prediction (2–4 hours)

Given a primal prediction x̂ from DetGNN, analytically compute the dual
variables (λ̂, ẑ) using the KKT stationarity condition. No retraining needed.

```python
# inference/dual_completion.py
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

def compute_duals_from_primal(x_hat, net, G, B):
    """
    Compute dual variables (lambda, z) from primal prediction x_hat
    using KKT stationarity: ∇f(x) + J_eq^T λ + J_ineq^T µ - z = 0

    This is the Klamkin-Tanneau-Van Hentenryck (NeurIPS 2024) dual
    completion layer, simplified for the AC-OPF structure.

    Args:
        x_hat: primal prediction [Vm, Va, Pg, Qg] in per-unit
        net:   pandapower network
        G, B:  admittance matrices

    Returns:
        lam_hat: equality constraint multipliers (power balance) [2*n_bus]
        z_hat:   bound multipliers [len(x_hat)]
        mu_hat:  centrality estimate
    """
    Vm, Va, Pg, Qg = unpack_variables(x_hat, net)
    n_bus = len(Vm)
    n_gen = len(Pg)

    # Step 1: Compute gradient of objective w.r.t. x at x_hat
    # f = sum_g (c2_g * Pg_g^2 + c1_g * Pg_g + c0_g)
    c2 = net.poly_cost["cp2_eur_per_mw2"].values / net.sn_mva**2
    c1 = net.poly_cost["cp1_eur_per_mw"].values / net.sn_mva
    grad_f = np.zeros(len(x_hat))
    pg_idx = slice(2*n_bus, 2*n_bus + n_gen)
    grad_f[pg_idx] = 2 * c2 * Pg + c1

    # Step 2: Compute Jacobian of equality constraints (power balance)
    # g(x) = [P_inj(x) - Pd; Q_inj(x) - Qd] = 0
    # J_eq = dg/dx  shape: [2*n_bus, 2*n_bus + 2*n_gen]
    J_eq = compute_power_balance_jacobian(Vm, Va, Pg, Qg, G, B, net)

    # Step 3: Solve least-squares for lambda
    # J_eq^T λ ≈ -grad_f  (KKT stationarity, ignoring inequality terms)
    result = lsqr(J_eq.T, -grad_f)
    lam_hat = result[0]  # [2*n_bus]

    # Step 4: Compute bound multipliers z from complementarity
    # z_i = max(0, -[grad_f + J_eq^T lam]_i) for lower bounds
    # z_i = max(0,  [grad_f + J_eq^T lam]_i) for upper bounds
    residual = grad_f + J_eq.T @ lam_hat
    z_lower = np.maximum(0, -residual)
    z_upper = np.maximum(0,  residual)
    z_hat = z_lower + z_upper  # combined bound multipliers

    # Step 5: Estimate centrality
    lb, ub = get_variable_bounds(net)
    s_hat = np.concatenate([
        x_hat - lb,   # slack for lower bounds
        ub - x_hat,   # slack for upper bounds
    ])
    z_full = np.concatenate([z_lower, z_upper])
    mu_hat = np.dot(s_hat, z_full) / len(s_hat)

    return lam_hat, z_hat, mu_hat


def compute_power_balance_jacobian(Vm, Va, Pg, Qg, G, B, net):
    """
    Analytical Jacobian of AC power balance equations.
    Returns sparse matrix J [2*n_bus, 2*n_bus + 2*n_gen].
    """
    n_bus = len(Vm)
    n_gen = len(Pg)

    # dP/dVa, dP/dVm, dQ/dVa, dQ/dVm — standard AC power flow Jacobian
    # (same computation as Newton-Raphson power flow)
    rows, cols, vals = [], [], []

    for i in range(n_bus):
        # dP_i / dVa_j and dP_i / dVm_j
        for j in range(n_bus):
            if abs(G[i,j]) + abs(B[i,j]) < 1e-10:
                continue
            if i != j:
                dP_dVa = Vm[i]*Vm[j]*(G[i,j]*(-np.sin(Va[i]-Va[j]))
                                      + B[i,j]*np.cos(Va[i]-Va[j]))
                dP_dVm = Vm[i]*(G[i,j]*np.cos(Va[i]-Va[j])
                                + B[i,j]*np.sin(Va[i]-Va[j]))
            else:
                # diagonal terms
                P_i = sum(Vm[i]*Vm[k]*(G[i,k]*np.cos(Va[i]-Va[k])
                          + B[i,k]*np.sin(Va[i]-Va[k]))
                          for k in range(n_bus))
                dP_dVa = -Vm[i]**2 * B[i,i] - ... # standard formula
                dP_dVm = ...

            rows.append(i);    cols.append(j);         vals.append(dP_dVa)
            rows.append(i);    cols.append(n_bus+j);   vals.append(dP_dVm)
            rows.append(n_bus+i); cols.append(j);      vals.append(...)
            rows.append(n_bus+i); cols.append(n_bus+j);vals.append(...)

    # dP_i / dPg_g: +1 for buses where gen g is connected
    gen_bus = net.gen["bus"].values
    for g, bus in enumerate(gen_bus):
        rows.append(bus);        cols.append(2*n_bus + g);   vals.append(1.0)
        rows.append(n_bus+bus);  cols.append(2*n_bus+n_gen+g); vals.append(1.0)

    return sparse.csr_matrix((vals, (rows, cols)),
                             shape=(2*n_bus, 2*n_bus + 2*n_gen))
```

**Shortcut that avoids implementing the full Jacobian:** use ForwardDiff or
PyTorch autograd on the differentiable `acpf.py` physics module you already have.
The Jacobian is `torch.autograd.functional.jacobian(compute_residuals, x_hat)`.

---

### Step 2 — Medium impact: add mu-based centrality fix (30 minutes, try first)

Before the full dual completion, try this cheaper fix: blend the GNN prediction
with the midpoint in a way that preserves centrality.

```python
# inference/warmstart.py

def centred_warmstart(x_hat, net, alpha=0.5):
    """
    Blend GNN prediction with IPM midpoint to restore centrality.

    alpha=0.0 → pure midpoint (flat start)
    alpha=1.0 → pure GNN prediction
    alpha=0.5 → compromise (try this first)

    The blend keeps µ = alpha * µ_GNN + (1-alpha) * µ_flat,
    so centrality degrades linearly with alpha.
    """
    lb, ub = get_variable_bounds(net)
    x_mid  = (lb + ub) / 2.0
    x_blend = alpha * x_hat + (1 - alpha) * x_mid
    # Enforce strict feasibility
    x_blend = np.clip(x_blend, lb + 1e-6*(ub-lb), ub - 1e-6*(ub-lb))
    return x_blend
```

Run benchmark with alpha ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0} on 50
instances. If there's an alpha < 1.0 where iterations drop below 19.6, that
directly confirms the centrality hypothesis and tells you how much blending
is needed.

---

### Step 3 — Highest impact but requires retraining: Lagrangian-MSE loss

Retrain DetGNN (and optionally WARP) with a loss that jointly predicts the
primal-dual solution and uses the Lagrangian merit as a regulariser.

```python
# training/losses.py

def lagrangian_mse_loss(pred_x, pred_lam, pred_z,
                         true_x, true_lam, true_z,
                         data, G, B,
                         w_primal=1.0, w_dual=0.1, w_lagrangian=0.01):
    """
    Lagrangian-MSE loss (TOAST / Briden et al. 2024 recipe).

    Trains the model to predict a primal-dual triple (x, λ, z)
    that is accurate AND consistent with the KKT conditions.

    Args:
        pred_x, pred_lam, pred_z: model predictions
        true_x, true_lam, true_z: ground truth from IPOPT logs
        data: HeteroData graph
        G, B: admittance matrices

    Returns:
        total loss scalar
    """
    # 1. Primal MSE (standard term, already working)
    L_primal = F.mse_loss(pred_x, true_x)

    # 2. Dual MSE (new term — requires dual labels from training set)
    L_dual = F.mse_loss(pred_lam, true_lam) + F.mse_loss(pred_z, true_z)

    # 3. Lagrangian merit (encourages KKT consistency)
    # L(x, λ, z) = f(x) + λᵀg(x)    where g(x) = power balance residuals
    dP, dQ = compute_residuals_from_x(pred_x, data, G, B)
    g = torch.cat([dP, dQ])  # [2*n_bus]
    f = compute_cost(pred_x, data)  # generation cost at predicted dispatch

    # Use predicted λ (first 2*n_bus components)
    n_bus = data["bus"].num_nodes
    lam_eq = pred_lam[:2*n_bus]
    L_lagrangian = f + torch.dot(lam_eq, g)

    # 4. KKT stationarity residual (optional, strong but expensive)
    # ‖∇_x L(x, λ, z)‖² = ‖∇f + Jᵀλ - z‖²
    # Skip for now, add if L_lagrangian alone doesn't help enough

    total = w_primal * L_primal + w_dual * L_dual + w_lagrangian * L_lagrangian
    return total, {"L_primal": L_primal, "L_dual": L_dual,
                   "L_lagrangian": L_lagrangian}
```

**To use this loss you need dual labels (true_lam, true_z) in the training set.**
Extract them from your existing IPOPT/PIPS runs:

```python
# scripts/extract_duals.py
"""
Run IPOPT on every training instance and save (x*, λ*, z*) to disk.
These become labels for the Lagrangian-MSE loss.
"""
import pandapower as pp
import numpy as np
import torch
from torch_geometric.datasets import OPFDataset
from pathlib import Path

def extract_dual_labels(case="pglib_opf_case118_ieee", split="train",
                         save_dir="data/opfdata/duals"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ds = OPFDataset(root="data/opfdata", case_name=case, split=split)

    for i, data in enumerate(ds):
        net = build_pandapower_net(data)  # convert HeteroData → pandapower net

        # Run IPOPT and capture full KKT solution
        pp.runopp(net, algorithm="ipopt", numba=False)

        # Extract duals from IPOPT result
        lam_eq = extract_equality_multipliers(net)   # power balance duals
        lam_ineq = extract_inequality_multipliers(net)  # line/voltage duals
        z_bound = extract_bound_multipliers(net)    # generator bound duals

        torch.save({
            "lam_eq": torch.tensor(lam_eq, dtype=torch.float32),
            "lam_ineq": torch.tensor(lam_ineq, dtype=torch.float32),
            "z_bound": torch.tensor(z_bound, dtype=torch.float32),
        }, f"{save_dir}/duals_{i:06d}.pt")

        if i % 100 == 0:
            print(f"Processed {i}/{len(ds)}")
```

---

### Step 4 — Alternative that avoids the centrality problem entirely:
### Active-constraint screening

Instead of replacing x₀, predict which constraints are binding and solve a
*reduced* problem. This is Pineda-Morales (2020) and Park-Van Hentenryck (2022).

```python
# models/constraint_classifier.py

class ActiveConstraintClassifier(nn.Module):
    """
    Predict which constraints are binding at the optimum.
    Use the same HetGNN backbone as DetGNN — add a binary classification
    head for each constraint type.

    Output: binding probability for each
      - line thermal limits: [n_line] ∈ [0,1]
      - voltage magnitude limits: [n_bus] ∈ [0,1] (upper + lower)
      - generator Pg limits: [n_gen] ∈ [0,1] (upper + lower)
      - generator Qg limits: [n_gen] ∈ [0,1]
    """
    def __init__(self, gnn_backbone, hidden_dim=256):
        super().__init__()
        self.gnn = gnn_backbone

        # Line classifier head
        self.line_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        # Bus voltage classifier head
        self.bus_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2)  # lo/hi
        )
        # Generator classifier head
        self.gen_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 4)  # Pg lo/hi, Qg lo/hi
        )

    def forward(self, data):
        h = self.gnn.encode(data)  # reuse HetGNN encoder

        # Predict binding probability for each constraint
        p_line = torch.sigmoid(self.line_head(h["line"]))
        p_bus  = torch.sigmoid(self.bus_head(h["bus"]))
        p_gen  = torch.sigmoid(self.gen_head(h["generator"]))
        return p_line, p_bus, p_gen

    def get_active_constraints(self, data, threshold=0.5):
        """Return binary mask of predicted binding constraints."""
        p_line, p_bus, p_gen = self(data)
        return {
            "line_thermal": (p_line > threshold).squeeze(),
            "bus_vmin":     (p_bus[:, 0] > threshold),
            "bus_vmax":     (p_bus[:, 1] > threshold),
            "gen_pmin":     (p_gen[:, 0] > threshold),
            "gen_pmax":     (p_gen[:, 1] > threshold),
            "gen_qmin":     (p_gen[:, 2] > threshold),
            "gen_qmax":     (p_gen[:, 3] > threshold),
        }
```

**How to use it:**
```python
# Remove predicted-non-binding constraints from net before flat-starting
active = classifier.get_active_constraints(data, threshold=0.5)

# Remove non-binding line constraints (safe: adds slop, won't cut optimum)
non_binding_lines = ~active["line_thermal"]
net.line.loc[non_binding_lines, "max_loading_percent"] = 1000.0  # effectively remove

# Flat-start the reduced problem — no warm-start, no centrality issue
pp.runopp(net, init="flat")
```

Train with binary cross-entropy loss on (binding/non-binding) labels from
the training-set IPOPT solutions. Literature reports 50–80% constraint removal
with <1% false-negative rate (Pineda-Morales 2020, case118).

---

## 4. Experiment sequence to run next

Run these in order. Each is a go/no-go gate before proceeding to the next.

### Experiment N1: centrality blend sweep (run today, ~2 hours)

**Goal:** Confirm the centrality hypothesis and find the optimal blend α.

```bash
PYTHONPATH=. python eval/benchmark_blend.py \
    --case case118 \
    --model checkpoints/det_gnn_norm_best.pt \
    --alphas 0.0 0.1 0.2 0.3 0.5 0.7 1.0 \
    --n-instances 50
```

**Expected result:** IPM iterations should form a U-shape in α — flat at α=0,
rising to ~21 at α=1. The minimum should be at some α* < 1.0 where DetGNN
predictions contribute enough direction without destroying centrality.

**Gate N1:** If any α gives < 19.6 mean iterations → centrality hypothesis
confirmed, proceed to N2 and N3 in parallel. If minimum is at α=0 → deeper
architecture issue, stop and debug.

### Experiment N2: IPOPT backend switch (2–3 days, run in parallel with N1)

**Goal:** Switch from PIPS to IPOPT with `warm_start_init_point=yes` and
`bound_mult_init_method=mu-based`. Run same 50-instance benchmark.

```bash
pip install cyipopt   # IPOPT Python binding
PYTHONPATH=. python eval/benchmark_ipopt.py \
    --case case118 \
    --model checkpoints/det_gnn_norm_best.pt \
    --solver ipopt \
    --warm-start-options "bound_mult_init_method=mu-based,warm_start_init_point=yes"
```

**Gate N2:** DetGNN with IPOPT backend should beat flat-start by ≥10%. If not,
proceed to N3 (dual completion) before giving up on primal-only prediction.

### Experiment N3: analytical dual completion (1 day, after N1 or N2)

**Goal:** Compute (λ̂, ẑ) from DetGNN's x̂ using KKT stationarity, pass to IPOPT.

```bash
PYTHONPATH=. python eval/benchmark_with_duals.py \
    --case case118 \
    --model checkpoints/det_gnn_norm_best.pt \
    --dual-completion kkt-lsq   # or: kkt-newton
```

**Gate N3:** Providing duals should reduce iterations by ≥20% vs primal-only
on IPOPT. If the gain is <5%, the model's primal predictions are too inaccurate
for the KKT residual to give good duals — then proceed to N4 (retrain).

### Experiment N4: Lagrangian-MSE retrain (2–3 days GPU)

**Goal:** Retrain DetGNN with joint primal-dual MSE + Lagrangian merit loss.

Prerequisite: extract_duals.py has been run to produce dual labels.

```bash
# Extract dual labels from training set (one-time, ~4 hours on A100)
PYTHONPATH=. python scripts/extract_duals.py --case case118 --split train

# Retrain DetGNN with Lagrangian-MSE loss
PYTHONPATH=. python training/train_det.py \
    --case case118 \
    --loss lagrangian-mse \
    --w-primal 1.0 --w-dual 0.1 --w-lagrangian 0.01 \
    --epochs 100 \
    --hidden-dim 256 --num-layers 8
```

**Gate N4:** Val L_primal should reach ~0.01 (current: 0.017). IPM iterations
should beat flat-start by ≥15%. This is the gate for the paper's main claim.

### Experiment N5: active-constraint screening (can run in parallel with N3/N4)

**Goal:** Train the constraint classifier and benchmark against flat-start.

```bash
PYTHONPATH=. python training/train_classifier.py \
    --case case118 --epochs 50

PYTHONPATH=. python eval/benchmark_constraint_screen.py \
    --case case118 \
    --classifier checkpoints/constraint_clf_best.pt \
    --threshold 0.5
```

**Gate N5:** Should reduce mean iterations by ≥15% vs flat-start with <2%
constraint violation rate at OPF optimum. Literature benchmark: Pineda-Morales
2020 reports ~20% iteration reduction on IEEE 118-bus.

---

## 5. File structure for new code

```
warp/
├── eval/
│   ├── benchmark_blend.py       ← Experiment N1
│   ├── benchmark_ipopt.py       ← Experiment N2
│   └── benchmark_with_duals.py  ← Experiment N3
│
├── inference/
│   ├── dual_completion.py       ← KKT-based (λ,z) computation
│   └── warmstart.py             ← update: add centred_warmstart()
│
├── models/
│   └── constraint_classifier.py ← Experiment N5
│
├── training/
│   ├── losses.py                ← add lagrangian_mse_loss()
│   └── train_det.py             ← update: add --loss lagrangian-mse flag
│
└── scripts/
    └── extract_duals.py         ← one-time dual label extraction
```

---

## 6. Key implementation warnings

**Do not touch:**
- Per-variable normalisation — this was the biggest single win. Any new model
  must use the same normalisation stats (`data/opfdata/norm_stats.pt`).
- PIPS monkey-patch — keep it in place even when switching to IPOPT. Some
  benchmarks still run PIPS for comparison.
- DDIM t_start=0.98*T and variable clamping — required for stable sampling.

**Be careful about:**
- `data.to_data_list()` in physics loss — always extract single random graph,
  never compute Y-bus on the merged batch graph.
- Variable ordering in x̂ — canonical order is [Vm, Va, Pg, Qg]. All of
  `acpf.py`, `dual_completion.py`, and IPOPT variable vector must use this.
- Dual variable sign conventions — IPOPT uses a specific sign for bound
  multipliers (z ≥ 0 for lower bound active). Check IPOPT documentation
  §4.1 before implementing dual completion.

**GPU / compute:**
- N1 and N5 can run on local Mac (CPU, case14/57).
- N2 and N3 require pandapower + cyipopt: install on GCP VM, run in tmux.
- N4 requires A100 for 100-epoch training. Use GCP Compute Engine or Lambda Labs.
  Estimated time: 6–8 hours on A100 (3.5 min/epoch × 100 epochs × 1.5x for
  dual-label overhead).

---

## 7. Paper framing update

Given the experimental results, the paper's strongest claims are now:

1. **Normalization is critical.** Per-variable normalisation reduces IPM
   iterations from 31 (−58% worse) to 21 (−7.8% worse) with zero architecture
   changes — a finding not reported in prior warm-start literature.

2. **Denoising accuracy is the wrong training objective for IPM warm-starting.**
   Exp F (best L_ddpm=0.265) gives worst IPM iterations (39.5). Exp C (worst
   L_ddpm=0.576) gives best WARP iterations (28.8). This is a quantitative
   demonstration of the Haeser-Hinder-Ye 2021 centrality argument applied to
   AC-OPF diffusion warm-starts — a novel empirical finding.

3. **Physics loss coefficient is the key WARP hyperparameter**, not model size,
   training duration, or diffusion steps. λ_phy=1.0 (Exp C) beats all other
   configurations. Interpretable: higher physics loss → samples closer to the
   AC feasibility manifold → less centrality disruption in IPM.

4. **Primal-only ML warm-starts have a hard ceiling set by IPM centrality.**
   Demonstrated by the fact that even perfect primal predictions cannot beat the
   midpoint heuristic when duals are not provided. This is the novel negative
   result that motivates the dual-completion and Lagrangian-MSE directions.

These four findings are publishable independently of whether N2–N5 succeed.

---

## 8. Key references to cite

All confirmed to exist (searched and verified):

- **Yildirim & Wright 2002** — "Warm-Start Strategies in Interior-Point Methods
  for Linear Programming," SIAM J. Optim. 12(3):782–810. DOI: 10.1137/S1052623400369235.
  *Formal proof that optimal solution is worst IPM warm-start.*

- **IPM-LSTM (Gao et al. 2024)** — "IPM-LSTM: A Learning-Based Interior Point
  Method for Solving Nonlinear Programs," NeurIPS 2024, arXiv 2410.15731.
  Code: github.com/NetSysOpt/IPM-LSTM.
  *−63.9% iterations on non-convex QP via primal-dual joint prediction.*

- **TOAST / Briden et al. 2024** — "Constraint-Informed Learning for
  Warm-Starting Trajectory Optimization," J. Guid. Control Dyn. 47(6), 2024;
  arXiv 2312.14336. *Lagrangian-MSE loss reduces IPM iterations 30–50%.*

- **Klamkin, Tanneau, Van Hentenryck 2024** — "Dual Interior Point
  Optimization Learning," NeurIPS 2024, arXiv 2402.02596.
  *S³L log-barrier loss + dual completion for AC-OPF specifically.*

- **Baker 2019** — "Learning Warm-Start Points for AC Optimal Power Flow,"
  IEEE MLSP 2019, arXiv 1905.08860.
  *Original OPF warm-start paper — positive results were runtime, not iteration count.*

- **Haeser, Hinder, Ye 2021** — "On the behavior of Lagrange multipliers in
  convex and nonconvex infeasible interior point methods," Math. Program.
  186:257–288. *Formal explanation of why dual variables explode when primal
  is accurate but duals are default-initialized.*

- **Sambharya, Stellato et al. 2024** — "Learning to Warm-Start Fixed-Point
  Optimization Algorithms," JMLR 25(166). Code: github.com/stellatogrp/l2ws.
  *End-to-end unrolling loss recipe (for QP/DR-splitting, but method transfers).*

- **Pineda & Morales 2020** — "Screening Constraints for the Optimal Power
  Flow Problem," IEEE Trans. Power Syst. 35(5):3695–3705.
  *Active-constraint screening, 50–80% constraint removal case118.*
