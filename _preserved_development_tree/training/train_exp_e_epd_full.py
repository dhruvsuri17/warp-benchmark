"""Experiment E: Full EPD with edge updates + unshared weights + dual prediction.

This is our own architecture — NOT CANOS. The contribution: extending the
encode-process-decode interaction network to predict the full (x, λ, z, μ)
IPM state for warm-starting IPOPT. CANOS only predicts primals.

Architecture:
  ENCODER: per-type Linear for all node and edge types
  PROCESSOR: 15 unshared IN blocks, each updating edges then nodes
  DECODER: per-type MLP for bus (8 dims) and gen (6 dims) + global μ
"""
import os, sys, argparse, time, logging, math
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("EXP-E")

from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from eval.opf_ipopt import build_om, solve_opf
import pandapower.networks as pn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EdgeMLP(nn.Module):
    """Edge update: e' = MLP(concat(e, h_sender, h_receiver))"""
    def __init__(self, h):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, h))
        self.ln = nn.LayerNorm(h)

    def forward(self, e, h_s, h_r):
        return self.ln(self.mlp(torch.cat([e, h_s, h_r], dim=-1)))


class NodeMLP(nn.Module):
    """Node update: h' = MLP(concat(h, agg))"""
    def __init__(self, h):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, h))
        self.ln = nn.LayerNorm(h)

    def forward(self, h, agg):
        return self.ln(self.mlp(torch.cat([h, agg], dim=-1)))


class INBlock(nn.Module):
    """One Interaction Network block with edge + node updates (unshared)."""
    def __init__(self, h):
        super().__init__()
        self.edge_mlps = nn.ModuleDict({
            "ac_line": EdgeMLP(h),
            "transformer": EdgeMLP(h),
            "gen_link": EdgeMLP(h),
            "load_link": EdgeMLP(h),
        })
        self.node_mlps = nn.ModuleDict({
            "bus": NodeMLP(h),
            "generator": NodeMLP(h),
            "load": NodeMLP(h),
        })

    def forward(self, nodes, edges, data):
        h = nodes
        e = edges
        n_bus = h["bus"].shape[0]

        # Edge updates + message aggregation
        agg_bus = torch.zeros_like(h["bus"])
        agg_gen = torch.zeros_like(h["generator"])
        agg_load = torch.zeros_like(h["load"])

        # AC lines: bus → bus
        if ("bus", "ac_line", "bus") in data.edge_types:
            ei = data["bus", "ac_line", "bus"].edge_index
            s, d = ei[0], ei[1]
            e_upd = self.edge_mlps["ac_line"](e["ac_line"], h["bus"][s], h["bus"][d])
            e["ac_line"] = e["ac_line"] + e_upd
            agg_bus.scatter_add_(0, d.unsqueeze(1).expand_as(e_upd), e_upd)
            agg_bus.scatter_add_(0, s.unsqueeze(1).expand_as(e_upd), e_upd)

        # Transformers: bus → bus
        if ("bus", "transformer", "bus") in data.edge_types:
            ei = data["bus", "transformer", "bus"].edge_index
            s, d = ei[0], ei[1]
            e_upd = self.edge_mlps["transformer"](e["transformer"], h["bus"][s], h["bus"][d])
            e["transformer"] = e["transformer"] + e_upd
            agg_bus.scatter_add_(0, d.unsqueeze(1).expand_as(e_upd), e_upd)
            agg_bus.scatter_add_(0, s.unsqueeze(1).expand_as(e_upd), e_upd)

        # Gen → bus
        ei = data["generator", "generator_link", "bus"].edge_index
        e_upd = self.edge_mlps["gen_link"](e["gen_link"], h["generator"][ei[0]], h["bus"][ei[1]])
        e["gen_link"] = e["gen_link"] + e_upd
        agg_bus.scatter_add_(0, ei[1].unsqueeze(1).expand_as(e_upd), e_upd)
        agg_gen = torch.zeros_like(h["generator"])
        # Reverse: bus → gen
        ei_rev = data["bus", "generator_link", "generator"].edge_index
        agg_gen.scatter_add_(0, ei_rev[1].unsqueeze(1).expand_as(e_upd), e_upd)

        # Load → bus
        ei = data["load", "load_link", "bus"].edge_index
        e_upd = self.edge_mlps["load_link"](e["load_link"], h["load"][ei[0]], h["bus"][ei[1]])
        e["load_link"] = e["load_link"] + e_upd
        agg_bus.scatter_add_(0, ei[1].unsqueeze(1).expand_as(e_upd), e_upd)

        # Node updates with residual
        h_new = {}
        h_new["bus"] = h["bus"] + self.node_mlps["bus"](h["bus"], agg_bus)
        h_new["generator"] = h["generator"] + self.node_mlps["generator"](h["generator"], agg_gen)
        h_new["load"] = h["load"] + self.node_mlps["load"](h["load"], agg_load)

        return h_new, e


class EPDGNN(nn.Module):
    """Encode-Process-Decode GNN with full primal-dual prediction.

    Our contribution: extends EPD interaction networks to predict
    (x, λ, z, μ) for IPOPT warm-starting. CANOS only predicts primals.
    """
    def __init__(self, hidden_dim=128, k_steps=15):
        super().__init__()
        h = hidden_dim

        # ENCODER: per-type linear projections
        self.node_enc = nn.ModuleDict({
            "bus": nn.Linear(4 + 2, h),     # bus features + injected load
            "generator": nn.Linear(11 + 2, h),  # gen features + connected bus load
            "load": nn.Linear(2, h),
        })
        self.edge_enc = nn.ModuleDict({
            "ac_line": nn.Linear(9, h),
            "transformer": nn.Linear(11, h),
        })
        self.h = h

        # PROCESSOR: k_steps unshared IN blocks
        self.blocks = nn.ModuleList([INBlock(h) for _ in range(k_steps)])
        self.k_steps = k_steps

        # DECODER: per-type output heads
        self.bus_head = nn.Sequential(
            nn.Linear(h, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 8))
        self.gen_head = nn.Sequential(
            nn.Linear(h, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 6))

    def forward(self, data):
        n_bus = data["bus"].x.shape[0]
        n_gen = data["generator"].x.shape[0]
        h = self.node_enc["bus"].weight.shape[1]

        # Inject loads onto bus and gen nodes
        bus_load = torch.zeros(n_bus, 2, device=data["bus"].x.device)
        ei_l = data["load", "load_link", "bus"].edge_index
        bus_load.scatter_add_(0, ei_l[1].unsqueeze(1).expand(-1, 2), data["load"].x[ei_l[0]])
        ei_g = data["generator", "generator_link", "bus"].edge_index
        gen_load = bus_load[ei_g[1]]

        # Encode nodes
        nodes = {
            "bus": self.node_enc["bus"](torch.cat([data["bus"].x, bus_load], -1)),
            "generator": self.node_enc["generator"](torch.cat([data["generator"].x, gen_load], -1)),
            "load": self.node_enc["load"](data["load"].x),
        }

        # Encode edges
        edges = {}
        if ("bus", "ac_line", "bus") in data.edge_types:
            edges["ac_line"] = self.edge_enc["ac_line"](data["bus", "ac_line", "bus"].edge_attr)
        else:
            edges["ac_line"] = torch.zeros(0, h, device=data["bus"].x.device)
        if ("bus", "transformer", "bus") in data.edge_types:
            edges["transformer"] = self.edge_enc["transformer"](data["bus", "transformer", "bus"].edge_attr)
        else:
            edges["transformer"] = torch.zeros(0, h, device=data["bus"].x.device)
        h = self.h
        n_gen_edges = data["generator", "generator_link", "bus"].edge_index.shape[1]
        n_load_edges = data["load", "load_link", "bus"].edge_index.shape[1]
        edges["gen_link"] = torch.zeros(n_gen_edges, h, device=data["bus"].x.device)
        edges["load_link"] = torch.zeros(n_load_edges, h, device=data["bus"].x.device)

        # Process
        for block in self.blocks:
            nodes, edges = block(nodes, edges, data)

        # Decode
        return self.bus_head(nodes["bus"]), self.gen_head(nodes["generator"])


# ============ Data and Training (reuse from train_gnn_kkt_batched) ============

class DualNorm:
    def __init__(self):
        self.stats = {}
    def fit(self, duals_dir, max_n=5000):
        files = sorted(Path(duals_dir).glob("duals_*.pt"))[:max_n]
        xs, lams, zls, zus = [], [], [], []
        for f in files:
            d = torch.load(f, weights_only=True, map_location="cpu")
            xs.append(d["x"]); lams.append(d["lam_g"]); zls.append(d["zl"]); zus.append(d["zu"])
        x=torch.stack(xs); lam=torch.stack(lams); zl=torch.stack(zls); zu=torch.stack(zus)
        self.stats = {
            "x_m": x.mean(0), "x_s": x.std(0).clamp(min=1e-6),
            "l_m": lam.mean(0), "l_s": lam.std(0).clamp(min=1e-6),
            "zl_m": zl.mean(0), "zl_s": zl.std(0).clamp(min=1e-6),
            "zu_m": zu.mean(0), "zu_s": zu.std(0).clamp(min=1e-6),
            "mu": 3.25e-08,
        }
        return self
    def norm(self, k, v): return (v - self.stats[f"{k}_m"].to(v.device)) / self.stats[f"{k}_s"].to(v.device)
    def denorm(self, k, v): return v * self.stats[f"{k}_s"].to(v.device) + self.stats[f"{k}_m"].to(v.device)


class OPFDualDataset(torch.utils.data.Dataset):
    def __init__(self, opf_ds, duals_dir, norm, n_bus, n_gen, max_n=None):
        self.opf = opf_ds; self.norm = norm; self.n_bus = n_bus; self.n_gen = n_gen
        self.items = []
        files = sorted(Path(duals_dir).glob("duals_*.pt"))
        if max_n: files = files[:max_n]
        for f in files:
            idx = int(f.stem.split("_")[1])
            if idx < len(opf_ds): self.items.append((idx, f))
        log.info(f"OPFDualDataset: {len(self.items)} instances from {duals_dir}")
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        idx, f = self.items[i]
        data = self.opf[idx].clone()
        duals = torch.load(f, weights_only=True, map_location="cpu")
        x_n = self.norm.norm("x", duals["x"]); l_n = self.norm.norm("l", duals["lam_g"])
        zl_n = self.norm.norm("zl", duals["zl"]); zu_n = self.norm.norm("zu", duals["zu"])
        nb = self.n_bus; ng = self.n_gen
        data["bus"].target = torch.stack([x_n[:nb], x_n[nb:2*nb], l_n[:nb], l_n[nb:2*nb],
                                          zl_n[:nb], zl_n[nb:2*nb], zu_n[:nb], zu_n[nb:2*nb]], dim=-1)
        data["generator"].target = torch.stack([x_n[2*nb:2*nb+ng], x_n[2*nb+ng:2*nb+2*ng],
                                                zl_n[2*nb:2*nb+ng], zl_n[2*nb+ng:2*nb+2*ng],
                                                zu_n[2*nb:2*nb+ng], zu_n[2*nb+ng:2*nb+2*ng]], dim=-1)
        return data


def unpack_prediction(bp, gp, n_bus, n_gen):
    x = torch.cat([bp[:, 0], bp[:, 1], gp[:, 0], gp[:, 1]])
    lam = torch.cat([bp[:, 2], bp[:, 3]])
    zl = torch.cat([bp[:, 4], bp[:, 5], gp[:, 2], gp[:, 3]])
    zu = torch.cat([bp[:, 6], bp[:, 7], gp[:, 4], gp[:, 5]])
    return x, lam, zl, zu


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
    args = parser.parse_args()

    log.info(f"Device: {DEVICE}")
    torch.cuda.set_per_process_memory_fraction(0.3)

    duals_base = Path(args.duals_dir) / args.case
    norm = DualNorm().fit(duals_base / "train", max_n=args.max_train)

    net_ref = pn.case118(); om_ref, _ = build_om(net_ref)
    vv = om_ref.get_idx()[0]
    n_bus = vv['N']['Va']; n_gen = vv['N']['Pg']

    train_opf = OPFDataset(root="data", case_name=args.case, split="train", num_groups=1)
    val_opf = OPFDataset(root="data", case_name=args.case, split="val", num_groups=1)
    train_ds = OPFDualDataset(train_opf, duals_base/"train", norm, n_bus, n_gen, args.max_train)
    val_ds = OPFDualDataset(val_opf, duals_base/"val", norm, n_bus, n_gen, args.max_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = EPDGNN(hidden_dim=args.hidden_dim, k_steps=args.k_steps).to(DEVICE)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # LR warmup 10 epochs, then multiply by 0.9 every 20 epochs
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.9 ** ((epoch - warmup_epochs) // 20)
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val = float("inf")
    Path("ckpt").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train(); s_loss = 0; n = 0; t0 = time.time()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            bp, gp = model(batch)
            loss = F.mse_loss(bp, batch["bus"].target) + F.mse_loss(gp, batch["generator"].target)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            s_loss += loss.item(); n += 1
        sched.step()

        model.eval(); v_loss = 0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                bp, gp = model(batch)
                v_loss += (F.mse_loss(bp, batch["bus"].target) + F.mse_loss(gp, batch["generator"].target)).item()
                vn += 1

        vl = v_loss / max(vn, 1)
        log.info(f"Ep {epoch+1}/{args.epochs} | Train: {s_loss/n:.6f} | Val: {vl:.6f} | "
                 f"LR: {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), "ckpt/epd_gnn_best.pt")

    log.info(f"Best val: {best_val:.6f}")

    # IPOPT benchmark
    log.info(f"=== IPOPT BENCHMARK ({args.n_test} instances) ===")
    model.load_state_dict(torch.load("ckpt/epd_gnn_best.pt", map_location=DEVICE, weights_only=True))
    model.eval()

    test_opf = OPFDataset(root="data", case_name=args.case, split="test", num_groups=1)
    test_files = sorted((duals_base/"test").glob("duals_*.pt"))[:args.n_test]
    cold_i, gnn_i, oracle_i = [], [], []

    with torch.no_grad():
        for ii in range(min(args.n_test, len(test_files))):
            duals = torch.load(test_files[ii], weights_only=True, map_location="cpu")
            idx = int(test_files[ii].stem.split("_")[1])
            data = test_opf[idx].to(DEVICE)
            bp, gp = model(data)
            x_n, l_n, zl_n, zu_n = unpack_prediction(bp, gp, n_bus, n_gen)
            x_raw = norm.denorm("x", x_n.cpu()).numpy()
            l_raw_n = l_n.cpu()
            l_raw = (l_raw_n * norm.stats["l_s"][:len(l_raw_n)] + norm.stats["l_m"][:len(l_raw_n)]).numpy()
            zl_raw = norm.denorm("zl", zl_n.cpu()).numpy()
            zu_raw = norm.denorm("zu", zu_n.cpu()).numpy()

            data_cpu = test_opf[idx]
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

            r_cold = solve_opf(om, ppopt, x0=(ll+uu)/2, warm_start=False)
            cold_i.append(r_cold["n_iters"])
            x_m = np.clip(x_raw, xmin+1e-10, xmax-1e-10)
            lam_full = np.zeros(236+372); lam_full[:min(len(l_raw),236)] = l_raw[:236]
            r_gnn = solve_opf(om, ppopt, x0=x_m, lam_g0=lam_full,
                              zl0=np.maximum(zl_raw,1e-10), zu0=np.maximum(zu_raw,1e-10),
                              warm_start=True, mu_init=norm.stats["mu"])
            gnn_i.append(r_gnn["n_iters"])
            x_o = np.clip(duals["x"].numpy(), xmin+1e-10, xmax-1e-10)
            r_ora = solve_opf(om, ppopt, x0=x_o, lam_g0=duals["lam_g"].numpy(),
                              zl0=duals["zl"].numpy(), zu0=duals["zu"].numpy(),
                              warm_start=True, mu_init=duals["mu"].item())
            oracle_i.append(r_ora["n_iters"])
            log.info(f"  #{idx}: cold={cold_i[-1]} epd={gnn_i[-1]} oracle={oracle_i[-1]}")

    log.info(f"\n{'='*60}")
    log.info(f"  EPD-GNN RESULTS ({len(cold_i)} instances)")
    log.info(f"{'='*60}")
    log.info(f"  Cold:    mean={np.mean(cold_i):.1f}")
    log.info(f"  EPD-GNN: mean={np.mean(gnn_i):.1f}  vs cold: {(1-np.mean(gnn_i)/np.mean(cold_i))*100:+.1f}%")
    log.info(f"  Oracle:  mean={np.mean(oracle_i):.1f}  vs cold: {(1-np.mean(oracle_i)/np.mean(cold_i))*100:+.1f}%")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
