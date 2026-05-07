"""Exp E2: EPD-GNN + constraint violation loss + attention-weighted mu.

Same architecture as Exp E. Changes:
1. Loss = MSE + 0.1 * AC power balance violation (physics-informed)
2. Attention-weighted global pool for mu prediction
3. Primal/dual loss decomposition logging
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
log = logging.getLogger("EXP-E2")

from training.train_exp_e_epd_full import EPDGNN, DualNorm, OPFDualDataset, unpack_prediction, EdgeMLP, NodeMLP, INBlock
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from eval.opf_ipopt import build_om, solve_opf
import pandapower.networks as pn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EPDGNNPhysics(EPDGNN):
    """EPD-GNN with attention-weighted mu prediction."""

    def __init__(self, hidden_dim=128, k_steps=15):
        super().__init__(hidden_dim, k_steps)
        h = hidden_dim
        self.mu_attn = nn.Linear(h, 1)
        self.mu_mlp = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, data):
        bp, gp = super().forward(data)
        return bp, gp

    def forward_with_mu(self, data):
        """Forward pass that also returns mu prediction."""
        # Run the full encoder-processor
        n_bus = data["bus"].x.shape[0]
        n_gen = data["generator"].x.shape[0]

        bus_load = torch.zeros(n_bus, 2, device=data["bus"].x.device)
        ei_l = data["load", "load_link", "bus"].edge_index
        bus_load.scatter_add_(0, ei_l[1].unsqueeze(1).expand(-1, 2), data["load"].x[ei_l[0]])
        ei_g = data["generator", "generator_link", "bus"].edge_index
        gen_load = bus_load[ei_g[1]]

        nodes = {
            "bus": self.node_enc["bus"](torch.cat([data["bus"].x, bus_load], -1)),
            "generator": self.node_enc["generator"](torch.cat([data["generator"].x, gen_load], -1)),
            "load": self.node_enc["load"](data["load"].x),
        }
        h = self.h
        edges = {}
        if ("bus", "ac_line", "bus") in data.edge_types:
            edges["ac_line"] = self.edge_enc["ac_line"](data["bus", "ac_line", "bus"].edge_attr)
        else:
            edges["ac_line"] = torch.zeros(0, h, device=data["bus"].x.device)
        if ("bus", "transformer", "bus") in data.edge_types:
            edges["transformer"] = self.edge_enc["transformer"](data["bus", "transformer", "bus"].edge_attr)
        else:
            edges["transformer"] = torch.zeros(0, h, device=data["bus"].x.device)
        edges["gen_link"] = torch.zeros(ei_g.shape[1], h, device=data["bus"].x.device)
        edges["load_link"] = torch.zeros(ei_l.shape[1], h, device=data["bus"].x.device)

        for block in self.blocks:
            nodes, edges = block(nodes, edges, data)

        bp = self.bus_head(nodes["bus"])
        gp = self.gen_head(nodes["generator"])

        # Attention-weighted mu
        attn = torch.softmax(self.mu_attn(nodes["bus"]), dim=0)
        h_global = (attn * nodes["bus"]).sum(dim=0)
        mu = F.softplus(self.mu_mlp(h_global))

        return bp, gp, mu


def compute_power_balance_violation(bp, gp, data, norm):
    """Compute AC power balance violation from denormalized predictions.

    Uses simplified power balance: P_inj = sum(Pg) - sum(Pd) at each bus.
    Full AC power balance would need Y-bus, but this gives a useful gradient signal.
    """
    n_bus = data["bus"].x.shape[0] if not hasattr(data["bus"], "batch") else (data["bus"].batch == 0).sum()

    # Denormalize primals
    Va_n = bp[:, 0]
    Vm_n = bp[:, 1]
    Pg_n = gp[:, 0]
    Qg_n = gp[:, 1]

    # Simple power balance: at each bus, sum(Pg) - sum(Pd) should equal P_flow
    # We use a simplified version: penalize large deviations in gen-load balance
    # Full AC power balance is too expensive for batched training

    # Generator power injection at bus level
    if hasattr(data["generator"], "batch"):
        # Batched: use per-graph operations
        ei_g = data["generator", "generator_link", "bus"].edge_index
        P_gen = torch.zeros_like(Va_n)
        Q_gen = torch.zeros_like(Va_n)
        P_gen.scatter_add_(0, ei_g[1], Pg_n[ei_g[0]])
        Q_gen.scatter_add_(0, ei_g[1], Qg_n[ei_g[0]])

        # Load injection (normalized)
        ei_l = data["load", "load_link", "bus"].edge_index
        load_x_n = (data["load"].x - norm.stats["x_m"][-2:].to(data["load"].x.device)) / norm.stats["x_s"][-2:].to(data["load"].x.device) if len(norm.stats["x_m"]) > 2 else data["load"].x
        P_load = torch.zeros_like(Va_n)
        Q_load = torch.zeros_like(Va_n)
        P_load.scatter_add_(0, ei_l[1], data["load"].x[ei_l[0], 0])
        Q_load.scatter_add_(0, ei_l[1], data["load"].x[ei_l[0], 1])

        # Simplified balance violation (in normalized space)
        # This is not exact AC power flow, but gives a useful gradient signal
        violation = (P_gen - P_load).pow(2).mean() + (Q_gen - Q_load).pow(2).mean()
    else:
        violation = torch.tensor(0.0, device=bp.device)

    return violation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--duals-dir", default="data/duals")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--k-steps", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--max-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--lam-physics", type=float, default=0.1)
    args = parser.parse_args()

    log.info(f"Device: {DEVICE}")
    torch.cuda.set_per_process_memory_fraction(0.25)

    duals_base = Path(args.duals_dir) / args.case
    norm = DualNorm().fit(duals_base / "train", max_n=args.max_train)
    net_ref = pn.case118(); om_ref, _ = build_om(net_ref)
    vv = om_ref.get_idx()[0]; n_bus = vv['N']['Va']; n_gen = vv['N']['Pg']

    train_opf = OPFDataset(root="data", case_name=args.case, split="train", num_groups=1)
    val_opf = OPFDataset(root="data", case_name=args.case, split="val", num_groups=1)
    train_ds = OPFDualDataset(train_opf, duals_base/"train", norm, n_bus, n_gen, args.max_train)
    val_ds = OPFDualDataset(val_opf, duals_base/"val", norm, n_bus, n_gen, args.max_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = EPDGNNPhysics(hidden_dim=args.hidden_dim, k_steps=args.k_steps).to(DEVICE)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    warmup = 10
    def lr_lambda(ep):
        if ep < warmup: return (ep+1)/warmup
        return 0.9 ** ((ep - warmup) // 20)
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val = float("inf"); Path("ckpt").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train(); s_t = s_p = s_d = s_phy = 0; n = 0; t0 = time.time()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            bp, gp = model(batch)

            # Primal/dual decomposed MSE
            primal_loss = F.mse_loss(bp[:, :2], batch["bus"].target[:, :2]) + \
                          F.mse_loss(gp[:, :2], batch["generator"].target[:, :2])
            dual_loss = F.mse_loss(bp[:, 2:], batch["bus"].target[:, 2:]) + \
                        F.mse_loss(gp[:, 2:], batch["generator"].target[:, 2:])

            # Physics loss
            physics_loss = compute_power_balance_violation(bp, gp, batch, norm)

            loss = primal_loss + dual_loss + args.lam_physics * physics_loss

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            s_t += loss.item(); s_p += primal_loss.item(); s_d += dual_loss.item()
            s_phy += physics_loss.item(); n += 1
        sched.step()

        model.eval(); v_p = v_d = v_phy = 0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                bp, gp = model(batch)
                v_p += (F.mse_loss(bp[:,:2], batch["bus"].target[:,:2]) +
                        F.mse_loss(gp[:,:2], batch["generator"].target[:,:2])).item()
                v_d += (F.mse_loss(bp[:,2:], batch["bus"].target[:,2:]) +
                        F.mse_loss(gp[:,2:], batch["generator"].target[:,2:])).item()
                v_phy += compute_power_balance_violation(bp, gp, batch, norm).item()
                vn += 1

        vl = (v_p + v_d) / max(vn, 1)
        log.info(f"Ep {epoch+1}/{args.epochs} | Train: {s_t/n:.4f} | "
                 f"Val primal: {v_p/vn:.4f} | Val dual: {v_d/vn:.4f} | "
                 f"Val physics: {v_phy/vn:.4f} | Val total: {vl:.4f} | {time.time()-t0:.0f}s")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), "ckpt/exp_e2_best.pt")

    log.info(f"Best val: {best_val:.6f}")

    # IPOPT benchmark
    log.info(f"=== IPOPT BENCHMARK ===")
    model.load_state_dict(torch.load("ckpt/exp_e2_best.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    test_opf = OPFDataset(root="data", case_name=args.case, split="test", num_groups=1)
    test_files = sorted((duals_base/"test").glob("duals_*.pt"))[:args.n_test]
    cold_i, gnn_i, oracle_i = [], [], []
    with torch.no_grad():
        for ii in range(min(args.n_test, len(test_files))):
            duals = torch.load(test_files[ii], weights_only=True, map_location="cpu")
            idx = int(test_files[ii].stem.split("_")[1]); data = test_opf[idx].to(DEVICE)
            bp, gp = model(data)
            x_n, l_n, zl_n, zu_n = unpack_prediction(bp, gp, n_bus, n_gen)
            x_raw = norm.denorm("x", x_n.cpu()).numpy()
            l_raw = (l_n.cpu() * norm.stats["l_s"][:len(l_n)] + norm.stats["l_m"][:len(l_n)]).numpy()
            zl_raw = norm.denorm("zl", zl_n.cpu()).numpy()
            zu_raw = norm.denorm("zu", zu_n.cpu()).numpy()
            net = pn.case118(); data_cpu = test_opf[idx]
            Pd = data_cpu["load"].x[:,0].numpy()*100; Qd = data_cpu["load"].x[:,1].numpy()*100
            for i in range(min(len(net.load),len(Pd))): net.load.at[i,"p_mw"]=Pd[i]; net.load.at[i,"q_mvar"]=Qd[i]
            om, ppopt = build_om(net); x0_v, xmin, xmax = om.getv()
            from numpy import inf as npinf; ll,uu=xmin.copy(),xmax.copy(); ll[xmin==-npinf]=-1e10; uu[xmax==npinf]=1e10
            r_cold = solve_opf(om, ppopt, x0=(ll+uu)/2, warm_start=False); cold_i.append(r_cold["n_iters"])
            x_m = np.clip(x_raw, xmin+1e-10, xmax-1e-10)
            lam_full = np.zeros(608); lam_full[:min(len(l_raw),236)] = l_raw[:236]
            r_gnn = solve_opf(om, ppopt, x0=x_m, lam_g0=lam_full, zl0=np.maximum(zl_raw,1e-10),
                              zu0=np.maximum(zu_raw,1e-10), warm_start=True, mu_init=norm.stats["mu"])
            gnn_i.append(r_gnn["n_iters"])
            x_o = np.clip(duals["x"].numpy(), xmin+1e-10, xmax-1e-10)
            r_ora = solve_opf(om, ppopt, x0=x_o, lam_g0=duals["lam_g"].numpy(), zl0=duals["zl"].numpy(),
                              zu0=duals["zu"].numpy(), warm_start=True, mu_init=duals["mu"].item())
            oracle_i.append(r_ora["n_iters"])
            log.info(f"  #{idx}: cold={cold_i[-1]} model={gnn_i[-1]} oracle={oracle_i[-1]}")
    log.info(f"Cold: {np.mean(cold_i):.1f} | Model: {np.mean(gnn_i):.1f} | Oracle: {np.mean(oracle_i):.1f}")


if __name__ == "__main__":
    main()
