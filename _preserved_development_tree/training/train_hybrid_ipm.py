"""Train HetGNN-IPM: topology-aware Newton step prediction for AC-OPF.

Instead of a flat LSTM, uses heterogeneous message passing on the power
grid graph. Each bus/gen node carries its primal-dual state; the GNN
predicts per-node Newton step directions.

Training: minimize MSE to IPOPT-optimal (x*, lam*, zl*, zu*) with
per-variable normalization.
"""
import os, sys, argparse, time, logging
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("HETGNN-IPM")

from models.hetgnn_lstm_ipm import HetGNNLSTMIPM as HetGNNIPM
from eval.opf_ipopt import build_om, solve_opf
from torch_geometric.datasets import OPFDataset
import pandapower.networks as pn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DualNormalizer:
    """Per-dimension normalization for primal-dual variables."""

    def __init__(self):
        self.stats = {}

    def fit(self, duals_dir, max_n=2000):
        duals_dir = Path(duals_dir)
        files = sorted(duals_dir.glob("duals_*.pt"))[:max_n]

        xs, lams, zls, zus = [], [], [], []
        for f in files:
            d = torch.load(f, weights_only=True, map_location="cpu")
            xs.append(d["x"]); lams.append(d["lam_g"])
            zls.append(d["zl"]); zus.append(d["zu"])

        x = torch.stack(xs); lam = torch.stack(lams)
        zl = torch.stack(zls); zu = torch.stack(zus)

        self.stats = {
            "x_mean": x.mean(0), "x_std": x.std(0).clamp(min=1e-6),
            "lam_mean": lam.mean(0), "lam_std": lam.std(0).clamp(min=1e-6),
            "zl_mean": zl.mean(0), "zl_std": zl.std(0).clamp(min=1e-6),
            "zu_mean": zu.mean(0), "zu_std": zu.std(0).clamp(min=1e-6),
            "mu_val": 3.25e-08,
        }
        return self

    def normalize(self, key, val):
        return (val - self.stats[f"{key}_mean"].to(val.device)) / self.stats[f"{key}_std"].to(val.device)

    def denormalize(self, key, val):
        return val * self.stats[f"{key}_std"].to(val.device) + self.stats[f"{key}_mean"].to(val.device)


def pack_bus_state(x_norm, lam_norm, zl_norm, zu_norm, n_bus):
    """Pack normalized primal-dual into per-bus state [n_bus, 8]."""
    Va = x_norm[:n_bus]
    Vm = x_norm[n_bus:2*n_bus]
    lam_P = lam_norm[:n_bus]
    lam_Q = lam_norm[n_bus:2*n_bus]
    zl_Va = zl_norm[:n_bus]
    zl_Vm = zl_norm[n_bus:2*n_bus]
    zu_Va = zu_norm[:n_bus]
    zu_Vm = zu_norm[n_bus:2*n_bus]
    return torch.stack([Va, Vm, lam_P, lam_Q, zl_Va, zl_Vm, zu_Va, zu_Vm], dim=-1)


def pack_gen_state(x_norm, zl_norm, zu_norm, n_bus, n_gen):
    """Pack normalized primal-dual into per-gen state [n_gen, 6]."""
    Pg = x_norm[2*n_bus:2*n_bus+n_gen]
    Qg = x_norm[2*n_bus+n_gen:2*n_bus+2*n_gen]
    zl_Pg = zl_norm[2*n_bus:2*n_bus+n_gen]
    zl_Qg = zl_norm[2*n_bus+n_gen:2*n_bus+2*n_gen]
    zu_Pg = zu_norm[2*n_bus:2*n_bus+n_gen]
    zu_Qg = zu_norm[2*n_bus+n_gen:2*n_bus+2*n_gen]
    return torch.stack([Pg, Qg, zl_Pg, zl_Qg, zu_Pg, zu_Qg], dim=-1)


def unpack_bus_delta(delta_bus, n_bus):
    """Unpack per-bus delta [n_bus, 8] into flat vectors."""
    d_Va = delta_bus[:, 0]; d_Vm = delta_bus[:, 1]
    d_lam_P = delta_bus[:, 2]; d_lam_Q = delta_bus[:, 3]
    d_zl_Va = delta_bus[:, 4]; d_zl_Vm = delta_bus[:, 5]
    d_zu_Va = delta_bus[:, 6]; d_zu_Vm = delta_bus[:, 7]
    dx = torch.cat([d_Va, d_Vm])
    dlam = torch.cat([d_lam_P, d_lam_Q])
    dzl = torch.cat([d_zl_Va, d_zl_Vm])
    dzu = torch.cat([d_zu_Va, d_zu_Vm])
    return dx, dlam, dzl, dzu


def unpack_gen_delta(delta_gen, n_bus, n_gen):
    """Unpack per-gen delta [n_gen, 6] into flat vectors."""
    d_Pg = delta_gen[:, 0]; d_Qg = delta_gen[:, 1]
    d_zl_Pg = delta_gen[:, 2]; d_zl_Qg = delta_gen[:, 3]
    d_zu_Pg = delta_gen[:, 4]; d_zu_Qg = delta_gen[:, 5]
    dx = torch.cat([d_Pg, d_Qg])
    dzl = torch.cat([d_zl_Pg, d_zl_Qg])
    dzu = torch.cat([d_zu_Pg, d_zu_Qg])
    return dx, dzl, dzu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--duals-dir", default="data/duals")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--outer-T", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-val", type=int, default=200)
    parser.add_argument("--n-test-ipopt", type=int, default=20)
    args = parser.parse_args()

    log.info(f"Device: {DEVICE}")
    torch.cuda.set_per_process_memory_fraction(0.5)

    duals_base = Path(args.duals_dir) / args.case

    # Fit normalizer
    norm = DualNormalizer().fit(duals_base / "train", max_n=args.max_train)
    log.info(f"Normalizer fit on {args.max_train} instances")

    # Load datasets
    train_ds = OPFDataset(root="data", case_name=args.case, split="train", num_groups=1)
    val_ds = OPFDataset(root="data", case_name=args.case, split="val", num_groups=1)

    train_files = sorted((duals_base / "train").glob("duals_*.pt"))[:args.max_train]
    val_files = sorted((duals_base / "val").glob("duals_*.pt"))[:args.max_val]
    log.info(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Reference net for dimensions
    net_ref = pn.case118()
    om_ref, _ = build_om(net_ref)
    vv = om_ref.get_idx()[0]
    n_bus = vv['N']['Va']
    n_gen = vv['N']['Pg']
    log.info(f"n_bus={n_bus}, n_gen={n_gen}")

    model = HetGNNIPM(gnn_hidden=args.hidden_dim, gnn_layers=args.num_layers,
                      lstm_hidden=args.hidden_dim, inner_T=3).to(DEVICE)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val = float("inf")
    Path("ckpt").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0; n_samples = 0
        t0 = time.time()

        indices = np.random.permutation(len(train_files))
        for ii in indices:
            duals = torch.load(train_files[ii], weights_only=True, map_location="cpu")
            data = train_ds[int(train_files[ii].stem.split("_")[1])].to(DEVICE)

            x_opt_n = norm.normalize("x", duals["x"]).to(DEVICE)
            lam_opt_n = norm.normalize("lam", duals["lam_g"]).to(DEVICE)
            zl_opt_n = norm.normalize("zl", duals["zl"]).to(DEVICE)
            zu_opt_n = norm.normalize("zu", duals["zu"]).to(DEVICE)

            target_bus = pack_bus_state(x_opt_n, lam_opt_n, zl_opt_n, zu_opt_n, n_bus)
            target_gen = pack_gen_state(x_opt_n, zl_opt_n, zu_opt_n, n_bus, n_gen)

            # Initialize from zero (normalized midpoint)
            cur_bus = torch.zeros_like(target_bus)
            cur_gen = torch.zeros_like(target_gen)

            loss = torch.tensor(0.0, device=DEVICE)

            for t in range(args.outer_T):
                delta_bus, delta_gen = model(data, cur_bus, cur_gen)

                # Target: step toward optimal
                err_bus = target_bus - cur_bus
                err_gen = target_gen - cur_gen

                # Loss: predict the right direction
                loss += F.mse_loss(delta_bus, err_bus) + F.mse_loss(delta_gen, err_gen)

                # Update state
                with torch.no_grad():
                    cur_bus = cur_bus + args.alpha * delta_bus.detach()
                    cur_gen = cur_gen + args.alpha * delta_gen.detach()

            loss = loss / args.outer_T

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_samples += 1

        # Validation
        model.eval()
        val_loss = 0; val_n = 0
        with torch.no_grad():
            for ii in range(min(args.max_val, len(val_files))):
                duals = torch.load(val_files[ii], weights_only=True, map_location="cpu")
                idx = int(val_files[ii].stem.split("_")[1])
                if idx >= len(val_ds):
                    continue
                data = val_ds[idx].to(DEVICE)

                x_opt_n = norm.normalize("x", duals["x"]).to(DEVICE)
                lam_opt_n = norm.normalize("lam", duals["lam_g"]).to(DEVICE)
                zl_opt_n = norm.normalize("zl", duals["zl"]).to(DEVICE)
                zu_opt_n = norm.normalize("zu", duals["zu"]).to(DEVICE)

                target_bus = pack_bus_state(x_opt_n, lam_opt_n, zl_opt_n, zu_opt_n, n_bus)
                target_gen = pack_gen_state(x_opt_n, zl_opt_n, zu_opt_n, n_bus, n_gen)

                cur_bus = torch.zeros_like(target_bus)
                cur_gen = torch.zeros_like(target_gen)

                for t in range(args.outer_T):
                    delta_bus, delta_gen = model(data, cur_bus, cur_gen)
                    cur_bus = cur_bus + args.alpha * delta_bus
                    cur_gen = cur_gen + args.alpha * delta_gen

                val_loss += (F.mse_loss(cur_bus, target_bus) + F.mse_loss(cur_gen, target_gen)).item()
                val_n += 1

        vl = val_loss / max(val_n, 1)
        log.info(f"Ep {epoch+1}/{args.epochs} | Train: {total_loss/n_samples:.6f} | "
                 f"Val: {vl:.6f} | {time.time()-t0:.0f}s")

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), "ckpt/hybrid_ipm_best.pt")

    log.info(f"Best val loss: {best_val:.6f}")

    # IPOPT benchmark
    log.info(f"=== IPOPT BENCHMARK ({args.n_test_ipopt} instances) ===")
    model.load_state_dict(torch.load("ckpt/hybrid_ipm_best.pt", map_location=DEVICE, weights_only=True))
    model.eval()

    test_ds = OPFDataset(root="data", case_name=args.case, split="test", num_groups=1)
    test_files = sorted((duals_base / "test").glob("duals_*.pt"))[:args.n_test_ipopt]

    cold_i, model_i, oracle_i = [], [], []

    with torch.no_grad():
        for ii in range(min(args.n_test_ipopt, len(test_files))):
            duals = torch.load(test_files[ii], weights_only=True, map_location="cpu")
            idx = int(test_files[ii].stem.split("_")[1])
            data = test_ds[idx].to(DEVICE)

            # Predict from zero
            cur_bus = torch.zeros(n_bus, 8, device=DEVICE)
            cur_gen = torch.zeros(n_gen, 6, device=DEVICE)
            for t in range(args.outer_T):
                delta_bus, delta_gen = model(data, cur_bus, cur_gen)
                cur_bus = cur_bus + args.alpha * delta_bus
                cur_gen = cur_gen + args.alpha * delta_gen

            # Unpack and denormalize
            dx_bus, dlam_bus, dzl_bus, dzu_bus = unpack_bus_delta(cur_bus, n_bus)
            dx_gen, dzl_gen, dzu_gen = unpack_gen_delta(cur_gen, n_bus, n_gen)

            x_pred_n = torch.cat([dx_bus, dx_gen])
            lam_pred_n = dlam_bus
            zl_pred_n = torch.cat([dzl_bus, dzl_gen])
            zu_pred_n = torch.cat([dzu_bus, dzu_gen])

            x_raw = norm.denormalize("x", x_pred_n.cpu())
            lam_raw = norm.denormalize("lam", lam_pred_n.cpu())
            zl_raw = norm.denormalize("zl", zl_pred_n.cpu())
            zu_raw = norm.denormalize("zu", zu_pred_n.cpu())

            # Build IPOPT model
            data_cpu = test_ds[idx]
            net = pn.case118()
            Pd = data_cpu["load"].x[:, 0].numpy() * 100
            Qd = data_cpu["load"].x[:, 1].numpy() * 100
            for i in range(min(len(net.load), len(Pd))):
                net.load.at[i, "p_mw"] = Pd[i]; net.load.at[i, "q_mvar"] = Qd[i]

            om, ppopt = build_om(net)
            x0_v, xmin, xmax = om.getv()
            from numpy import inf as npinf
            ll, uu = xmin.copy(), xmax.copy()
            ll[xmin == -npinf] = -1e10; uu[xmax == npinf] = 1e10
            x_mid = (ll + uu) / 2.0

            # Cold
            r_cold = solve_opf(om, ppopt, x0=x_mid, warm_start=False)
            cold_i.append(r_cold["n_iters"])

            # Model
            x_m = np.clip(x_raw.numpy(), xmin + 1e-10, xmax - 1e-10)
            lam_m = lam_raw.numpy()
            # Pad lam to full constraint size (eq + ineq)
            lam_full = np.zeros(236 + 372)
            lam_full[:min(len(lam_m), 236)] = lam_m[:236]
            zl_m = np.maximum(zl_raw.numpy(), 1e-10)
            zu_m = np.maximum(zu_raw.numpy(), 1e-10)

            r_model = solve_opf(om, ppopt, x0=x_m, lam_g0=lam_full,
                                zl0=zl_m, zu0=zu_m,
                                warm_start=True, mu_init=norm.stats["mu_val"])
            model_i.append(r_model["n_iters"])

            # Oracle
            x_o = np.clip(duals["x"].numpy(), xmin + 1e-10, xmax - 1e-10)
            r_oracle = solve_opf(om, ppopt, x0=x_o,
                                 lam_g0=duals["lam_g"].numpy(),
                                 zl0=duals["zl"].numpy(), zu0=duals["zu"].numpy(),
                                 warm_start=True, mu_init=duals["mu"].item())
            oracle_i.append(r_oracle["n_iters"])

            log.info(f"  #{idx}: cold={cold_i[-1]} model={model_i[-1]} oracle={oracle_i[-1]}")

    log.info(f"\n{'='*60}")
    log.info(f"  HETGNN-IPM RESULTS ({len(cold_i)} instances)")
    log.info(f"{'='*60}")
    log.info(f"  Cold:    mean={np.mean(cold_i):.1f}  median={np.median(cold_i):.0f}")
    log.info(f"  HetGNN:  mean={np.mean(model_i):.1f}  median={np.median(model_i):.0f}  "
             f"vs cold: {(1-np.mean(model_i)/np.mean(cold_i))*100:+.1f}%")
    log.info(f"  Oracle:  mean={np.mean(oracle_i):.1f}  median={np.median(oracle_i):.0f}  "
             f"vs cold: {(1-np.mean(oracle_i)/np.mean(cold_i))*100:+.1f}%")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
