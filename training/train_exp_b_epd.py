"""Experiment B: Encode-Process-Decode with UNSHARED weights.

Change vs baseline HetGNNKKT:
- Encoder: per-type Linear(raw→H) for all node and edge types (one-time)
- Processor: L=15 steps, each with its OWN edge MLP and node MLP (not shared).
  At each step: update edges (e'=MLP_k(e,h_s,h_r)+e), aggregate, update nodes (h'=MLP_k(h,agg)+h),
  LayerNorm after residual
- Decoder: MLP(H→256→256→output_dim) per node type
- H=128, L=15
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
log = logging.getLogger("ExpB-EPD")

from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
from eval.opf_ipopt import build_om, solve_opf
import pandapower.networks as pn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mlp(in_dim, h, out_dim, n=2):
    layers = []
    for i in range(n):
        di = in_dim if i == 0 else h
        do = out_dim if i == n-1 else h
        layers.append(nn.Linear(di, do))
        if i < n-1:
            layers += [nn.LayerNorm(do), nn.SiLU()]
    return nn.Sequential(*layers)


class EPDProcessorStep(nn.Module):
    """One processor step with its OWN (unshared) edge and node MLPs."""

    def __init__(self, h):
        super().__init__()
        self.edge_mlp_ac = _mlp(2*h + h, h, h)
        self.edge_mlp_xfmr = _mlp(2*h + h, h, h)
        self.edge_mlp_g2b = _mlp(2*h + h, h, h)
        self.edge_mlp_b2g = _mlp(2*h + h, h, h)
        self.edge_mlp_l2b = _mlp(2*h + h, h, h)

        self.edge_ln_ac = nn.LayerNorm(h)
        self.edge_ln_xfmr = nn.LayerNorm(h)
        self.edge_ln_g2b = nn.LayerNorm(h)
        self.edge_ln_b2g = nn.LayerNorm(h)
        self.edge_ln_l2b = nn.LayerNorm(h)

        self.node_mlp_bus = _mlp(2*h, h, h)
        self.node_mlp_gen = _mlp(2*h, h, h)
        self.node_ln_bus = nn.LayerNorm(h)
        self.node_ln_gen = nn.LayerNorm(h)

    def forward(self, hb, hg, hl, e_ac, e_xfmr, e_g2b, e_b2g, e_l2b, data):
        nb, h = hb.shape

        # Edge updates
        if ("bus", "ac_line", "bus") in data.edge_types:
            ei = data["bus", "ac_line", "bus"].edge_index
            s, d = ei[0], ei[1]
            e_ac = e_ac + self.edge_ln_ac(self.edge_mlp_ac(torch.cat([hb[s], hb[d], e_ac], -1)))

        if ("bus", "transformer", "bus") in data.edge_types:
            ei = data["bus", "transformer", "bus"].edge_index
            s, d = ei[0], ei[1]
            e_xfmr = e_xfmr + self.edge_ln_xfmr(self.edge_mlp_xfmr(torch.cat([hb[s], hb[d], e_xfmr], -1)))

        ei_g2b = data["generator", "generator_link", "bus"].edge_index
        e_g2b = e_g2b + self.edge_ln_g2b(self.edge_mlp_g2b(torch.cat([hg[ei_g2b[0]], hb[ei_g2b[1]], e_g2b], -1)))

        ei_bg = data["bus", "generator_link", "generator"].edge_index
        e_b2g = e_b2g + self.edge_ln_b2g(self.edge_mlp_b2g(torch.cat([hb[ei_bg[0]], hg[ei_bg[1]], e_b2g], -1)))

        ei_l = data["load", "load_link", "bus"].edge_index
        e_l2b = e_l2b + self.edge_ln_l2b(self.edge_mlp_l2b(torch.cat([hl[ei_l[0]], hb[ei_l[1]], e_l2b], -1)))

        # Aggregate updated edges → nodes
        agg_bus = torch.zeros(nb, h, device=hb.device)
        if ("bus", "ac_line", "bus") in data.edge_types:
            ei = data["bus", "ac_line", "bus"].edge_index
            agg_bus.scatter_add_(0, ei[1].unsqueeze(1).expand_as(e_ac), e_ac)
            agg_bus.scatter_add_(0, ei[0].unsqueeze(1).expand_as(e_ac), e_ac)
        if ("bus", "transformer", "bus") in data.edge_types:
            ei = data["bus", "transformer", "bus"].edge_index
            agg_bus.scatter_add_(0, ei[1].unsqueeze(1).expand_as(e_xfmr), e_xfmr)
            agg_bus.scatter_add_(0, ei[0].unsqueeze(1).expand_as(e_xfmr), e_xfmr)
        agg_bus.scatter_add_(0, ei_g2b[1].unsqueeze(1).expand_as(e_g2b), e_g2b)
        agg_bus.scatter_add_(0, ei_l[1].unsqueeze(1).expand_as(e_l2b), e_l2b)

        agg_gen = torch.zeros(hg.shape[0], h, device=hg.device)
        agg_gen.scatter_add_(0, ei_bg[1].unsqueeze(1).expand_as(e_b2g), e_b2g)

        # Node updates with residual + LN
        hb = hb + self.node_ln_bus(self.node_mlp_bus(torch.cat([hb, agg_bus], -1)))
        hg = hg + self.node_ln_gen(self.node_mlp_gen(torch.cat([hg, agg_gen], -1)))

        return hb, hg, hl, e_ac, e_xfmr, e_g2b, e_b2g, e_l2b


class HetGNNKKT_EPD(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=15):
        super().__init__()
        h = hidden_dim
        # Encoder: per-type
        self.bus_enc = nn.Linear(4 + 2, h)
        self.gen_enc = nn.Linear(11 + 2, h)
        self.load_enc = nn.Linear(2, h)
        self.edge_enc_ac = nn.Linear(9, h)
        self.edge_enc_xfmr = nn.Linear(11, h)

        # Processor: L unshared steps
        self.steps = nn.ModuleList([EPDProcessorStep(h) for _ in range(num_layers)])

        # Decoder: deeper MLP per node type
        self.bus_head = nn.Sequential(
            nn.Linear(h, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 8))
        self.load_skip = nn.Sequential(
            nn.Linear(2, h // 2), nn.SiLU(), nn.Linear(h // 2, h // 2))
        self.gen_head = nn.Sequential(
            nn.Linear(h + h // 2, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 6))
        self.h = h

    def forward(self, data):
        n_bus = data["bus"].x.shape[0]
        n_gen = data["generator"].x.shape[0]

        bus_load = torch.zeros(n_bus, 2, device=data["bus"].x.device)
        ei_load = data["load", "load_link", "bus"].edge_index
        load_vals = data["load"].x
        bus_load.scatter_add_(0, ei_load[1].unsqueeze(1).expand(-1, 2), load_vals[ei_load[0]])
        ei_gen = data["generator", "generator_link", "bus"].edge_index
        gen_load = bus_load[ei_gen[1]]

        # Encode
        hb = self.bus_enc(torch.cat([data["bus"].x, bus_load], dim=-1))
        hg = self.gen_enc(torch.cat([data["generator"].x, gen_load], dim=-1))
        hl = self.load_enc(data["load"].x)

        if ("bus", "ac_line", "bus") in data.edge_types:
            e_ac = self.edge_enc_ac(data["bus", "ac_line", "bus"].edge_attr)
        else:
            e_ac = torch.zeros(0, self.h, device=hb.device)
        if ("bus", "transformer", "bus") in data.edge_types:
            e_xfmr = self.edge_enc_xfmr(data["bus", "transformer", "bus"].edge_attr)
        else:
            e_xfmr = torch.zeros(0, self.h, device=hb.device)

        n_g2b = data["generator", "generator_link", "bus"].edge_index.shape[1]
        e_g2b = torch.zeros(n_g2b, self.h, device=hb.device)
        n_b2g = data["bus", "generator_link", "generator"].edge_index.shape[1]
        e_b2g = torch.zeros(n_b2g, self.h, device=hb.device)
        n_l2b = data["load", "load_link", "bus"].edge_index.shape[1]
        e_l2b = torch.zeros(n_l2b, self.h, device=hb.device)

        # Process
        for step in self.steps:
            hb, hg, hl, e_ac, e_xfmr, e_g2b, e_b2g, e_l2b = step(
                hb, hg, hl, e_ac, e_xfmr, e_g2b, e_b2g, e_l2b, data)

        # Decode
        bus_out = self.bus_head(hb)

        if hasattr(data["load"], "batch"):
            load_batch = data["load"].batch
            n_graphs = load_batch.max().item() + 1
            load_sum = torch.zeros(n_graphs, 2, device=hg.device)
            load_sum.scatter_add_(0, load_batch.unsqueeze(1).expand(-1, 2), data["load"].x)
            gen_batch = data["generator"].batch
            load_per_gen = load_sum[gen_batch]
        else:
            load_sum = data["load"].x.sum(0, keepdim=True)
            load_per_gen = load_sum.expand(n_gen, -1)

        skip = self.load_skip(load_per_gen)
        gen_out = self.gen_head(torch.cat([hg, skip], dim=-1))
        return bus_out, gen_out


# ── Dataset & utilities (same as baseline) ──────────────────────────────────

class DualNorm:
    def __init__(self):
        self.stats = {}

    def fit(self, duals_dir, max_n=5000):
        files = sorted(Path(duals_dir).glob("duals_*.pt"))[:max_n]
        xs, lams, zls, zus = [], [], [], []
        for f in files:
            d = torch.load(f, weights_only=True, map_location="cpu")
            xs.append(d["x"]); lams.append(d["lam_g"])
            zls.append(d["zl"]); zus.append(d["zu"])
        x = torch.stack(xs); lam = torch.stack(lams)
        zl = torch.stack(zls); zu = torch.stack(zus)
        self.stats = {
            "x_m": x.mean(0), "x_s": x.std(0).clamp(min=1e-6),
            "l_m": lam.mean(0), "l_s": lam.std(0).clamp(min=1e-6),
            "zl_m": zl.mean(0), "zl_s": zl.std(0).clamp(min=1e-6),
            "zu_m": zu.mean(0), "zu_s": zu.std(0).clamp(min=1e-6),
            "mu": 3.25e-08,
        }
        return self

    def norm(self, k, v):
        return (v - self.stats[f"{k}_m"].to(v.device)) / self.stats[f"{k}_s"].to(v.device)

    def denorm(self, k, v):
        return v * self.stats[f"{k}_s"].to(v.device) + self.stats[f"{k}_m"].to(v.device)


class OPFDualDataset(torch.utils.data.Dataset):
    def __init__(self, opf_ds, duals_dir, norm, n_bus, n_gen, max_n=None):
        self.opf = opf_ds
        self.norm = norm
        self.n_bus = n_bus
        self.n_gen = n_gen
        duals_dir = Path(duals_dir)
        self.items = []
        files = sorted(duals_dir.glob("duals_*.pt"))
        if max_n:
            files = files[:max_n]
        for f in files:
            idx = int(f.stem.split("_")[1])
            if idx < len(opf_ds):
                self.items.append((idx, f))
        log.info(f"OPFDualDataset: {len(self.items)} instances from {duals_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        idx, f = self.items[i]
        data = self.opf[idx].clone()
        duals = torch.load(f, weights_only=True, map_location="cpu")
        x_n = self.norm.norm("x", duals["x"])
        l_n = self.norm.norm("l", duals["lam_g"])
        zl_n = self.norm.norm("zl", duals["zl"])
        zu_n = self.norm.norm("zu", duals["zu"])
        nb = self.n_bus; ng = self.n_gen
        data["bus"].target = torch.stack([
            x_n[:nb], x_n[nb:2*nb],
            l_n[:nb], l_n[nb:2*nb],
            zl_n[:nb], zl_n[nb:2*nb],
            zu_n[:nb], zu_n[nb:2*nb],
        ], dim=-1)
        data["generator"].target = torch.stack([
            x_n[2*nb:2*nb+ng], x_n[2*nb+ng:2*nb+2*ng],
            zl_n[2*nb:2*nb+ng], zl_n[2*nb+ng:2*nb+2*ng],
            zu_n[2*nb:2*nb+ng], zu_n[2*nb+ng:2*nb+2*ng],
        ], dim=-1)
        data["_duals_x"] = duals["x"]
        data["_duals_lam"] = duals["lam_g"]
        data["_duals_zl"] = duals["zl"]
        data["_duals_zu"] = duals["zu"]
        data["_duals_mu"] = duals["mu"]
        return data


def unpack_prediction(bus_pred, gen_pred, n_bus, n_gen):
    x = torch.cat([bus_pred[:, 0], bus_pred[:, 1], gen_pred[:, 0], gen_pred[:, 1]])
    lam = torch.cat([bus_pred[:, 2], bus_pred[:, 3]])
    zl = torch.cat([bus_pred[:, 4], bus_pred[:, 5], gen_pred[:, 2], gen_pred[:, 3]])
    zu = torch.cat([bus_pred[:, 6], bus_pred[:, 7], gen_pred[:, 4], gen_pred[:, 5]])
    return x, lam, zl, zu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--duals-dir", default="data/duals")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--max-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    log.info(f"Device: {DEVICE}")
    torch.cuda.set_per_process_memory_fraction(0.2)

    duals_base = Path(args.duals_dir) / args.case
    norm = DualNorm().fit(duals_base / "train", max_n=args.max_train)

    net_ref = pn.case118(); om_ref, _ = build_om(net_ref)
    vv = om_ref.get_idx()[0]
    n_bus = vv['N']['Va']; n_gen = vv['N']['Pg']

    train_opf = OPFDataset(root="data", case_name=args.case, split="train", num_groups=1)
    val_opf = OPFDataset(root="data", case_name=args.case, split="val", num_groups=1)

    train_ds = OPFDualDataset(train_opf, duals_base/"train", norm, n_bus, n_gen, args.max_train)
    val_ds = OPFDualDataset(val_opf, duals_base/"val", norm, n_bus, n_gen, args.max_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = HetGNNKKT_EPD(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(DEVICE)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = args.epochs * len(train_loader)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)

    best_val = float("inf")
    Path("ckpt").mkdir(exist_ok=True)
    ckpt_path = "ckpt/exp_b_epd_best.pt"

    for epoch in range(args.epochs):
        model.train(); s_loss = 0; n = 0; t0 = time.time()

        for batch in train_loader:
            batch = batch.to(DEVICE)
            bp, gp = model(batch)
            bus_loss = F.mse_loss(bp, batch["bus"].target)
            gen_err = (gp - batch["generator"].target) ** 2
            gen_w = torch.ones(6, device=DEVICE)
            gen_w[1] = 5.0; gen_w[3] = 5.0; gen_w[5] = 5.0
            gen_loss = (gen_err * gen_w).mean()
            loss = bus_loss + gen_loss
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            s_loss += loss.item(); n += 1

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
            torch.save(model.state_dict(), ckpt_path)

    log.info(f"Best val: {best_val:.6f}")

    # ── IPOPT benchmark ──────────────────────────────────────────────────
    log.info(f"=== IPOPT BENCHMARK ({args.n_test} instances) ===")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
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

            log.info(f"  #{idx}: cold={cold_i[-1]} gnn={gnn_i[-1]} oracle={oracle_i[-1]}")

    log.info(f"\n{'='*60}")
    log.info(f"  ExpB EPD RESULTS ({len(cold_i)} instances)")
    log.info(f"{'='*60}")
    log.info(f"  Cold:    mean={np.mean(cold_i):.1f}  median={np.median(cold_i):.0f}")
    log.info(f"  GNN:     mean={np.mean(gnn_i):.1f}  median={np.median(gnn_i):.0f}  "
             f"vs cold: {(1-np.mean(gnn_i)/np.mean(cold_i))*100:+.1f}%")
    log.info(f"  Oracle:  mean={np.mean(oracle_i):.1f}  median={np.median(oracle_i):.0f}  "
             f"vs cold: {(1-np.mean(oracle_i)/np.mean(cold_i))*100:+.1f}%")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
