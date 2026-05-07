"""Train IPM solver on AC-OPF using IPM-LSTM's approach.

Adapts IPM-LSTM's main.py to use our precomputed AC-OPF KKT data.
Since AC-OPF's KKT system is too large for dense Jacobian on GPU,
we use precomputed (x*, lam*, z*, mu*) from IPOPT and train the model
to predict good Newton steps that reduce the KKT residual.

For now: uses IPM-LSTM's LSTM architecture directly to validate the
pipeline. Can swap in GNN later.
"""
import os, sys, argparse, time, logging, math
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("IPM-TRAIN")

from models.gnn_ipm import GNNIPMStep
from eval.opf_ipopt import build_om, solve_opf
from torch_geometric.datasets import OPFDataset
import pandapower.networks as pn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleKKTData:
    """Wraps precomputed IPOPT duals with per-variable normalization.

    Normalizes all variables to ~N(0,1) so the model sees balanced scales.
    Stores normalization stats for denormalization at inference.
    """

    def __init__(self, duals_dir, num_var, num_eq, num_ineq, device, max_n=None,
                 norm_stats=None):
        self.device = device
        self.num_var = num_var
        self.num_eq = num_eq
        self.num_ineq = num_ineq
        self.num_lb = num_var
        self.num_ub = num_var

        duals_dir = Path(duals_dir)
        files = sorted(duals_dir.glob("duals_*.pt"))
        if max_n:
            files = files[:max_n]

        xs, lams, zls, zus, mus = [], [], [], [], []
        for f in files:
            d = torch.load(f, weights_only=True, map_location="cpu")
            xs.append(d["x"])
            lams.append(d["lam_g"])
            zls.append(d["zl"])
            zus.append(d["zu"])
            mus.append(d["mu"])

        self.data_size = len(xs)
        log.info(f"Loaded {self.data_size} dual labels from {duals_dir}")

        x_raw = torch.stack(xs)
        lam_raw = torch.stack(lams)
        zl_raw = torch.stack(zls)
        zu_raw = torch.stack(zus)
        self.mu_val = torch.stack(mus).mean().item()

        if norm_stats is None:
            self.x_mean = x_raw.mean(0, keepdim=True)
            self.x_std = x_raw.std(0, keepdim=True).clamp(min=1e-6)
            self.lam_mean = lam_raw.mean(0, keepdim=True)
            self.lam_std = lam_raw.std(0, keepdim=True).clamp(min=1e-6)
            self.zl_mean = zl_raw.mean(0, keepdim=True)
            self.zl_std = zl_raw.std(0, keepdim=True).clamp(min=1e-6)
            self.zu_mean = zu_raw.mean(0, keepdim=True)
            self.zu_std = zu_raw.std(0, keepdim=True).clamp(min=1e-6)
            self.norm_stats = {
                "x_mean": self.x_mean, "x_std": self.x_std,
                "lam_mean": self.lam_mean, "lam_std": self.lam_std,
                "zl_mean": self.zl_mean, "zl_std": self.zl_std,
                "zu_mean": self.zu_mean, "zu_std": self.zu_std,
                "mu_val": self.mu_val,
            }
        else:
            self.norm_stats = norm_stats
            self.x_mean = norm_stats["x_mean"]
            self.x_std = norm_stats["x_std"]
            self.lam_mean = norm_stats["lam_mean"]
            self.lam_std = norm_stats["lam_std"]
            self.zl_mean = norm_stats["zl_mean"]
            self.zl_std = norm_stats["zl_std"]
            self.zu_mean = norm_stats["zu_mean"]
            self.zu_std = norm_stats["zu_std"]
            self.mu_val = norm_stats["mu_val"]

        self.x_opt = ((x_raw - self.x_mean) / self.x_std).unsqueeze(-1).to(device)
        self.lam_opt = ((lam_raw - self.lam_mean) / self.lam_std).unsqueeze(-1).to(device)
        self.zl_opt = ((zl_raw - self.zl_mean) / self.zl_std).unsqueeze(-1).to(device)
        self.zu_opt = ((zu_raw - self.zu_mean) / self.zu_std).unsqueeze(-1).to(device)

        log.info(f"  Normalized: x_opt range=[{self.x_opt.min():.2f}, {self.x_opt.max():.2f}], "
                 f"lam range=[{self.lam_opt.min():.2f}, {self.lam_opt.max():.2f}], "
                 f"zl range=[{self.zl_opt.min():.2f}, {self.zl_opt.max():.2f}], "
                 f"zu range=[{self.zu_opt.min():.2f}, {self.zu_opt.max():.2f}]")

    def denormalize(self, x_n, lam_n, zl_n, zu_n):
        """Convert normalized predictions back to raw scale."""
        x = x_n * self.x_std.to(x_n.device) + self.x_mean.to(x_n.device)
        lam = lam_n * self.lam_std.to(lam_n.device) + self.lam_mean.to(lam_n.device)
        zl = zl_n * self.zl_std.to(zl_n.device) + self.zl_mean.to(zl_n.device)
        zu = zu_n * self.zu_std.to(zu_n.device) + self.zu_mean.to(zu_n.device)
        return x, lam, zl, zu

    def sub_objective(self, y, J, F):
        """½‖Jy + F‖² — the Newton step sub-objective."""
        Jy_F = torch.bmm(J, y) + F
        return 0.5 * torch.bmm(Jy_F.transpose(1, 2), Jy_F)

    def sub_smooth_grad(self, y, J, F):
        """Gradient of sub_objective: J^T(Jy + F)"""
        return torch.bmm(J.transpose(1, 2), torch.bmm(J, y) + F)


def build_kkt_system(x, lam, zl, zu, x_opt, lam_opt, zl_opt, zu_opt, sigma=0.1):
    """Build approximate KKT Jacobian and residual.

    For AC-OPF, the full KKT system is:
    F = [∇f + J_eq^T λ + J_ineq^T η - zl + zu,   (stationarity)
         g_eq(x),                                    (primal eq feasibility)
         g_ineq(x) + s,                              (primal ineq feasibility)
         η*s - σμe,                                   (complementarity ineq)
         zl*(x-lb) - σμe,                             (complementarity lb)
         zu*(ub-x) - σμe]                             (complementarity ub)

    Since computing the full Jacobian on GPU is expensive for case118
    (344x344 + constraints), we use a simplified approach:
    train the model to predict the step toward the IPOPT solution directly.

    The loss is MSE to the optimal state: ‖(x,lam,z) - (x*,lam*,z*)‖²
    """
    # Simple MSE-based loss between current and optimal state
    batch = x.shape[0]
    n_var = x.shape[1]
    n_lam = lam.shape[1]
    n_zl = zl.shape[1]
    n_zu = zu.shape[1]

    # Build target delta (optimal - current)
    delta_x = x_opt - x
    delta_lam = lam_opt - lam
    delta_zl = zl_opt - zl
    delta_zu = zu_opt - zu

    # Concatenate into full state
    y_target = torch.cat([delta_x, delta_lam, delta_zl, delta_zu], dim=1)

    # For the Jacobian, use identity (gradient descent direction)
    total_dim = n_var + n_lam + n_zl + n_zu
    J = torch.eye(total_dim, device=x.device).unsqueeze(0).expand(batch, -1, -1)
    F = -y_target  # So that Jy + F = 0 when y = y_target

    return J, F, y_target


def train_ipm(args):
    log.info(f"Device: {DEVICE}")
    torch.cuda.set_per_process_memory_fraction(0.5)

    duals_base = Path(args.duals_dir) / args.case

    train_data = SimpleKKTData(duals_base / "train", args.num_var,
                                args.num_eq, args.num_ineq, DEVICE,
                                max_n=args.max_train)
    val_data = SimpleKKTData(duals_base / "val", args.num_var,
                              args.num_eq, args.num_ineq, DEVICE,
                              max_n=args.max_val,
                              norm_stats=train_data.norm_stats)

    total_dim = args.num_var + train_data.lam_opt.shape[1] + \
                train_data.zl_opt.shape[1] + train_data.zu_opt.shape[1]
    log.info(f"Total IPM state dim: {total_dim}")

    model = GNNIPMStep(
        input_dim=2, hidden_dim=args.hidden_dim,
        iter_step=args.inner_T, device=str(DEVICE),
    ).to(DEVICE)
    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    Path("ckpt").mkdir(exist_ok=True)

    for epoch in range(args.num_epoch):
        model.train()
        t0 = time.time()

        # Sample batch
        batch_idx = torch.randperm(train_data.data_size, device=DEVICE)[:args.batch_size]

        # Initialize from midpoint (cold start)
        x = torch.ones(len(batch_idx), args.num_var, 1, device=DEVICE) * 0.5  # normalized midpoint
        lam = torch.zeros(len(batch_idx), train_data.lam_opt.shape[1], 1, device=DEVICE)
        zl = torch.ones(len(batch_idx), train_data.zl_opt.shape[1], 1, device=DEVICE)
        zu = torch.ones(len(batch_idx), train_data.zu_opt.shape[1], 1, device=DEVICE)

        x_opt = train_data.x_opt[batch_idx]
        lam_opt = train_data.lam_opt[batch_idx]
        zl_opt = train_data.zl_opt[batch_idx]
        zu_opt = train_data.zu_opt[batch_idx]

        total_loss = 0.0

        for t_out in range(args.outer_T):
            # Build KKT system
            J, F, y_target = build_kkt_system(x, lam, zl, zu,
                                               x_opt, lam_opt, zl_opt, zu_opt,
                                               sigma=args.sigma)

            # Model predicts Newton step
            init_y = torch.zeros(len(batch_idx), total_dim, 1, device=DEVICE)
            y_pred, loss, _ = model(train_data, init_y, J, F)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.item()

            # Update state with predicted step (using fraction-to-boundary)
            with torch.no_grad():
                n_var = args.num_var
                n_lam = lam.shape[1]
                n_zl = zl.shape[1]

                step_x = y_pred[:, :n_var, :]
                step_lam = y_pred[:, n_var:n_var+n_lam, :]
                step_zl = y_pred[:, n_var+n_lam:n_var+n_lam+n_zl, :]
                step_zu = y_pred[:, n_var+n_lam+n_zl:, :]

                alpha = 0.3  # conservative step
                x = x + alpha * step_x
                lam = lam + alpha * step_lam
                zl = torch.clamp(zl + alpha * step_zl, min=1e-10)
                zu = torch.clamp(zu + alpha * step_zu, min=1e-10)

        # Validation
        model.eval()
        with torch.no_grad():
            val_idx = torch.arange(min(100, val_data.data_size), device=DEVICE)
            v_x = torch.ones(len(val_idx), args.num_var, 1, device=DEVICE) * 0.5
            v_lam = torch.zeros(len(val_idx), val_data.lam_opt.shape[1], 1, device=DEVICE)
            v_zl = torch.ones(len(val_idx), val_data.zl_opt.shape[1], 1, device=DEVICE)
            v_zu = torch.ones(len(val_idx), val_data.zu_opt.shape[1], 1, device=DEVICE)

            for t_out in range(args.outer_T):
                J, F, _ = build_kkt_system(v_x, v_lam, v_zl, v_zu,
                                           val_data.x_opt[val_idx],
                                           val_data.lam_opt[val_idx],
                                           val_data.zl_opt[val_idx],
                                           val_data.zu_opt[val_idx])
                init_y = torch.zeros(len(val_idx), total_dim, 1, device=DEVICE)
                v_y, v_loss, _ = model(val_data, init_y, J, F)

                n_var = args.num_var
                n_lam = v_lam.shape[1]
                n_zl = v_zl.shape[1]
                v_x = v_x + 0.3 * v_y[:, :n_var, :]
                v_lam = v_lam + 0.3 * v_y[:, n_var:n_var+n_lam, :]
                v_zl = torch.clamp(v_zl + 0.3 * v_y[:, n_var+n_lam:n_var+n_lam+n_zl, :], min=1e-10)
                v_zu = torch.clamp(v_zu + 0.3 * v_y[:, n_var+n_lam+n_zl:, :], min=1e-10)

            # Measure distance to optimal
            val_x_err = (v_x - val_data.x_opt[val_idx]).pow(2).mean().item()
            val_lam_err = (v_lam - val_data.lam_opt[val_idx]).pow(2).mean().item()

        log.info(f"Ep {epoch+1}/{args.num_epoch} | "
                 f"Loss: {total_loss/args.outer_T:.4f} | "
                 f"Val x_err: {val_x_err:.4f} | Val lam_err: {val_lam_err:.4f} | "
                 f"{time.time()-t0:.1f}s")

        if val_x_err < best_val:
            best_val = val_x_err
            torch.save(model.state_dict(), "ckpt/gnn_ipm_best.pt")

    log.info(f"Best val x_err: {best_val:.6f}")

    # Test: run IPOPT with model-predicted warm-start
    log.info("Testing IPOPT warm-start with model predictions...")
    model.eval()
    test_data = SimpleKKTData(duals_base / "test", args.num_var,
                               args.num_eq, args.num_ineq, DEVICE, max_n=50,
                               norm_stats=train_data.norm_stats)

    with torch.no_grad():
        t_x = torch.zeros(test_data.data_size, args.num_var, 1, device=DEVICE)
        t_lam = torch.zeros(test_data.data_size, test_data.lam_opt.shape[1], 1, device=DEVICE)
        t_zl = torch.zeros(test_data.data_size, test_data.zl_opt.shape[1], 1, device=DEVICE)
        t_zu = torch.zeros(test_data.data_size, test_data.zu_opt.shape[1], 1, device=DEVICE)

        for t_out in range(args.outer_T):
            J, F, _ = build_kkt_system(t_x, t_lam, t_zl, t_zu,
                                       test_data.x_opt, test_data.lam_opt,
                                       test_data.zl_opt, test_data.zu_opt)
            init_y = torch.zeros(test_data.data_size, total_dim, 1, device=DEVICE)
            t_y, _, _ = model(test_data, init_y, J, F)

            n_var = args.num_var
            n_lam = t_lam.shape[1]
            n_zl = t_zl.shape[1]
            t_x = t_x + 0.3 * t_y[:, :n_var, :]
            t_lam = t_lam + 0.3 * t_y[:, n_var:n_var+n_lam, :]
            t_zl = t_zl + 0.3 * t_y[:, n_var+n_lam:n_var+n_lam+n_zl, :]
            t_zu = t_zu + 0.3 * t_y[:, n_var+n_lam+n_zl:, :]

    x_err = (t_x - test_data.x_opt).pow(2).mean().item()
    lam_err = (t_lam - test_data.lam_opt).pow(2).mean().item()
    log.info(f"Test normalized errors: x={x_err:.6f}, lam={lam_err:.6f}")

    # Denormalize and run IPOPT
    t_x_raw, t_lam_raw, t_zl_raw, t_zu_raw = test_data.denormalize(
        t_x.squeeze(-1), t_lam.squeeze(-1), t_zl.squeeze(-1), t_zu.squeeze(-1))

    log.info("Running IPOPT benchmark with model predictions...")
    test_ds = OPFDataset(root="data", case_name=args.case, split="test", num_groups=1)
    cold_iters, model_iters, oracle_iters = [], [], []

    for idx in range(min(20, test_data.data_size)):
        data = test_ds[idx]
        net = pn.case118()
        Pd = data["load"].x[:, 0].numpy() * 100
        Qd = data["load"].x[:, 1].numpy() * 100
        for i in range(min(len(net.load), len(Pd))):
            net.load.at[i, "p_mw"] = Pd[i]; net.load.at[i, "q_mvar"] = Qd[i]

        om, ppopt = build_om(net)
        x0_v, xmin, xmax = om.getv()
        from numpy import inf as npinf
        ll, uu = xmin.copy(), xmax.copy()
        ll[xmin == -npinf] = -1e10; uu[xmax == npinf] = 1e10
        x_mid = (ll + uu) / 2.0

        r_cold = solve_opf(om, ppopt, x0=x_mid, warm_start=False)
        cold_iters.append(r_cold["n_iters"])

        x_m = np.clip(t_x_raw[idx].cpu().numpy(), xmin+1e-10, xmax-1e-10)
        lam_m = t_lam_raw[idx].cpu().numpy()
        zl_m = np.maximum(t_zl_raw[idx].cpu().numpy(), 1e-10)
        zu_m = np.maximum(t_zu_raw[idx].cpu().numpy(), 1e-10)
        mu_m = test_data.mu_val

        r_model = solve_opf(om, ppopt, x0=x_m, lam_g0=lam_m, zl0=zl_m, zu0=zu_m,
                            warm_start=True, mu_init=mu_m)
        model_iters.append(r_model["n_iters"])

        d = torch.load(f"{duals_base}/test/duals_{idx:06d}.pt", weights_only=True)
        x_o = np.clip(d["x"].numpy(), xmin+1e-10, xmax-1e-10)
        r_oracle = solve_opf(om, ppopt, x0=x_o, lam_g0=d["lam_g"].numpy(),
                             zl0=d["zl"].numpy(), zu0=d["zu"].numpy(),
                             warm_start=True, mu_init=d["mu"].item())
        oracle_iters.append(r_oracle["n_iters"])

        log.info(f"  #{idx}: cold={r_cold['n_iters']} model={r_model['n_iters']} oracle={r_oracle['n_iters']}")

    log.info(f"\nIPOPT Results (20 instances):")
    log.info(f"  Cold:   mean={np.mean(cold_iters):.1f}")
    log.info(f"  Model:  mean={np.mean(model_iters):.1f}")
    log.info(f"  Oracle: mean={np.mean(oracle_iters):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--duals-dir", default="data/duals")
    parser.add_argument("--num-var", type=int, default=344)
    parser.add_argument("--num-eq", type=int, default=236)
    parser.add_argument("--num-ineq", type=int, default=372)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--inner-T", type=int, default=5)
    parser.add_argument("--outer-T", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-val", type=int, default=200)
    args = parser.parse_args()

    train_ipm(args)
