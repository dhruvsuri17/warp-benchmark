"""Train HetGNN with KKT sub-objective (same loss as LSTM baseline).

The GNN encodes the graph topology, then a readout head produces the full
flattened primal-dual state. Training uses ½‖y - y*‖² in normalized space —
the same objective that gave the LSTM 81% iteration reduction.

The GNN advantage: it sees the grid structure, so it should generalize
better to unseen load patterns and potentially to different grid sizes.
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
log = logging.getLogger("GNN-KKT")

from torch_geometric.datasets import OPFDataset
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


class HetGNNLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.msg_ac = _mlp(2*h+9, h, h)
        self.msg_xfmr = _mlp(2*h+11, h, h)
        self.msg_g2b = _mlp(2*h, h, h)
        self.msg_b2g = _mlp(2*h, h, h)
        self.msg_l2b = _mlp(2*h, h, h)
        self.upd_bus = _mlp(h, h, h)
        self.upd_gen = _mlp(h, h, h)
        self.norm_b = nn.LayerNorm(h)
        self.norm_g = nn.LayerNorm(h)

    def forward(self, hb, hg, hl, data):
        nb, h = hb.shape
        mb = torch.zeros(nb, h, device=hb.device)
        if ("bus","ac_line","bus") in data.edge_types:
            ei = data["bus","ac_line","bus"].edge_index
            ea = data["bus","ac_line","bus"].edge_attr
            s, d = ei[0], ei[1]
            m = self.msg_ac(torch.cat([hb[d], hb[s], ea], -1))
            mb.scatter_add_(0, d.unsqueeze(1).expand_as(m), m)
            m2 = self.msg_ac(torch.cat([hb[s], hb[d], ea], -1))
            mb.scatter_add_(0, s.unsqueeze(1).expand_as(m2), m2)
        if ("bus","transformer","bus") in data.edge_types:
            ei = data["bus","transformer","bus"].edge_index
            ea = data["bus","transformer","bus"].edge_attr
            s, d = ei[0], ei[1]
            m = self.msg_xfmr(torch.cat([hb[d], hb[s], ea], -1))
            mb.scatter_add_(0, d.unsqueeze(1).expand_as(m), m)
            m2 = self.msg_xfmr(torch.cat([hb[s], hb[d], ea], -1))
            mb.scatter_add_(0, s.unsqueeze(1).expand_as(m2), m2)
        ei = data["generator","generator_link","bus"].edge_index
        m = self.msg_g2b(torch.cat([hb[ei[1]], hg[ei[0]]], -1))
        mb.scatter_add_(0, ei[1].unsqueeze(1).expand_as(m), m)
        ei = data["load","load_link","bus"].edge_index
        m = self.msg_l2b(torch.cat([hb[ei[1]], hl[ei[0]]], -1))
        mb.scatter_add_(0, ei[1].unsqueeze(1).expand_as(m), m)

        mg = torch.zeros(hg.shape[0], h, device=hg.device)
        ei = data["bus","generator_link","generator"].edge_index
        m = self.msg_b2g(torch.cat([hg[ei[1]], hb[ei[0]]], -1))
        mg.scatter_add_(0, ei[1].unsqueeze(1).expand_as(m), m)

        hb = hb + self.upd_bus(self.norm_b(hb + mb))
        hg = hg + self.upd_gen(self.norm_g(hg + mg))
        return hb, hg, hl


class HetGNNKKT(nn.Module):
    """HetGNN that predicts the full normalized primal-dual state.

    Output: flattened [x_norm, lam_norm, zl_norm, zu_norm] vector.
    Per-bus output: Va, Vm (primals) + lam_P, lam_Q (eq duals) +
                    zl_Va, zl_Vm, zu_Va, zu_Vm (bound duals) = 8 dims
    Per-gen output: Pg, Qg (primals) + zl_Pg, zl_Qg, zu_Pg, zu_Qg = 6 dims
    """

    def __init__(self, hidden_dim=128, num_layers=6):
        super().__init__()
        h = hidden_dim
        self.bus_enc = nn.Linear(4, h)   # bus features
        self.gen_enc = nn.Linear(11, h)  # gen features
        self.load_enc = nn.Linear(2, h)  # load features
        self.layers = nn.ModuleList([HetGNNLayer(h) for _ in range(num_layers)])
        self.bus_head = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 8))
        self.gen_head = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 6))

    def forward(self, data):
        hb = self.bus_enc(data["bus"].x)
        hg = self.gen_enc(data["generator"].x)
        hl = self.load_enc(data["load"].x)
        for layer in self.layers:
            hb, hg, hl = layer(hb, hg, hl, data)
        return self.bus_head(hb), self.gen_head(hg)


class DualNorm:
    def __init__(self):
        self.stats = {}

    def fit(self, duals_dir, max_n=2000):
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


def pack_target(x_n, lam_n, zl_n, zu_n, n_bus, n_gen):
    """Pack normalized targets into per-node format matching model output."""
    bus_target = torch.stack([
        x_n[:n_bus], x_n[n_bus:2*n_bus],       # Va, Vm
        lam_n[:n_bus], lam_n[n_bus:2*n_bus],    # lam_P, lam_Q
        zl_n[:n_bus], zl_n[n_bus:2*n_bus],      # zl_Va, zl_Vm
        zu_n[:n_bus], zu_n[n_bus:2*n_bus],       # zu_Va, zu_Vm
    ], dim=-1)
    gen_target = torch.stack([
        x_n[2*n_bus:2*n_bus+n_gen], x_n[2*n_bus+n_gen:2*n_bus+2*n_gen],
        zl_n[2*n_bus:2*n_bus+n_gen], zl_n[2*n_bus+n_gen:2*n_bus+2*n_gen],
        zu_n[2*n_bus:2*n_bus+n_gen], zu_n[2*n_bus+n_gen:2*n_bus+2*n_gen],
    ], dim=-1)
    return bus_target, gen_target


def unpack_prediction(bus_pred, gen_pred, n_bus, n_gen):
    """Unpack per-node predictions into flat vectors for IPOPT."""
    x = torch.cat([bus_pred[:, 0], bus_pred[:, 1],   # Va, Vm
                   gen_pred[:, 0], gen_pred[:, 1]])    # Pg, Qg
    lam = torch.cat([bus_pred[:, 2], bus_pred[:, 3]])  # lam_P, lam_Q
    zl = torch.cat([bus_pred[:, 4], bus_pred[:, 5],
                    gen_pred[:, 2], gen_pred[:, 3]])
    zu = torch.cat([bus_pred[:, 6], bus_pred[:, 7],
                    gen_pred[:, 4], gen_pred[:, 5]])
    return x, lam, zl, zu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--duals-dir", default="data/duals")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=20)
    args = parser.parse_args()

    log.info(f"Device: {DEVICE}")
    torch.cuda.set_per_process_memory_fraction(0.5)

    duals_base = Path(args.duals_dir) / args.case
    norm = DualNorm().fit(duals_base / "train", max_n=args.max_train)

    train_ds = OPFDataset(root="data", case_name=args.case, split="train", num_groups=1)
    val_ds = OPFDataset(root="data", case_name=args.case, split="val", num_groups=1)
    train_files = sorted((duals_base / "train").glob("duals_*.pt"))[:args.max_train]
    val_files = sorted((duals_base / "val").glob("duals_*.pt"))[:args.max_val]

    net_ref = pn.case118(); om_ref, _ = build_om(net_ref)
    vv = om_ref.get_idx()[0]
    n_bus = vv['N']['Va']; n_gen = vv['N']['Pg']

    model = HetGNNKKT(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(DEVICE)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = args.epochs * len(train_files)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)

    best_val = float("inf")
    Path("ckpt").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train(); s_loss = 0; n = 0; t0 = time.time()

        for f in np.random.permutation(len(train_files)):
            duals = torch.load(train_files[f], weights_only=True, map_location="cpu")
            idx = int(train_files[f].stem.split("_")[1])
            data = train_ds[idx].to(DEVICE)

            x_n = norm.norm("x", duals["x"]).to(DEVICE)
            l_n = norm.norm("l", duals["lam_g"]).to(DEVICE)
            zl_n = norm.norm("zl", duals["zl"]).to(DEVICE)
            zu_n = norm.norm("zu", duals["zu"]).to(DEVICE)
            bt, gt = pack_target(x_n, l_n, zl_n, zu_n, n_bus, n_gen)

            bp, gp = model(data)
            loss = F.mse_loss(bp, bt) + F.mse_loss(gp, gt)

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            s_loss += loss.item(); n += 1

        model.eval(); v_loss = 0; vn = 0
        with torch.no_grad():
            for f in val_files[:args.max_val]:
                duals = torch.load(f, weights_only=True, map_location="cpu")
                idx = int(f.stem.split("_")[1])
                if idx >= len(val_ds): continue
                data = val_ds[idx].to(DEVICE)
                x_n = norm.norm("x", duals["x"]).to(DEVICE)
                l_n = norm.norm("l", duals["lam_g"]).to(DEVICE)
                zl_n = norm.norm("zl", duals["zl"]).to(DEVICE)
                zu_n = norm.norm("zu", duals["zu"]).to(DEVICE)
                bt, gt = pack_target(x_n, l_n, zl_n, zu_n, n_bus, n_gen)
                bp, gp = model(data)
                v_loss += (F.mse_loss(bp, bt) + F.mse_loss(gp, gt)).item()
                vn += 1

        vl = v_loss / max(vn, 1)
        log.info(f"Ep {epoch+1}/{args.epochs} | Train: {s_loss/n:.6f} | Val: {vl:.6f} | {time.time()-t0:.0f}s")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), "ckpt/gnn_kkt_best.pt")

    log.info(f"Best val: {best_val:.6f}")

    # IPOPT benchmark
    log.info(f"=== IPOPT BENCHMARK ({args.n_test} instances) ===")
    model.load_state_dict(torch.load("ckpt/gnn_kkt_best.pt", map_location=DEVICE, weights_only=True))
    model.eval()

    test_ds = OPFDataset(root="data", case_name=args.case, split="test", num_groups=1)
    test_files = sorted((duals_base / "test").glob("duals_*.pt"))[:args.n_test]
    cold_i, gnn_i, oracle_i = [], [], []

    with torch.no_grad():
        for ii in range(min(args.n_test, len(test_files))):
            duals = torch.load(test_files[ii], weights_only=True, map_location="cpu")
            idx = int(test_files[ii].stem.split("_")[1])
            data = test_ds[idx].to(DEVICE)

            bp, gp = model(data)
            x_n, l_n, zl_n, zu_n = unpack_prediction(bp, gp, n_bus, n_gen)
            x_raw = norm.denorm("x", x_n.cpu()).numpy()
            l_raw = norm.denorm("l", l_n.cpu()).numpy()
            zl_raw = norm.denorm("zl", zl_n.cpu()).numpy()
            zu_raw = norm.denorm("zu", zu_n.cpu()).numpy()

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

            r_cold = solve_opf(om, ppopt, x0=(ll+uu)/2, warm_start=False)
            cold_i.append(r_cold["n_iters"])

            x_m = np.clip(x_raw, xmin+1e-10, xmax-1e-10)
            lam_full = np.zeros(236+372); lam_full[:min(len(l_raw), 236)] = l_raw[:236]
            r_gnn = solve_opf(om, ppopt, x0=x_m, lam_g0=lam_full,
                              zl0=np.maximum(zl_raw, 1e-10), zu0=np.maximum(zu_raw, 1e-10),
                              warm_start=True, mu_init=norm.stats["mu"])
            gnn_i.append(r_gnn["n_iters"])

            x_o = np.clip(duals["x"].numpy(), xmin+1e-10, xmax-1e-10)
            r_ora = solve_opf(om, ppopt, x0=x_o, lam_g0=duals["lam_g"].numpy(),
                              zl0=duals["zl"].numpy(), zu0=duals["zu"].numpy(),
                              warm_start=True, mu_init=duals["mu"].item())
            oracle_i.append(r_ora["n_iters"])

            log.info(f"  #{idx}: cold={cold_i[-1]} gnn={gnn_i[-1]} oracle={oracle_i[-1]}")

    log.info(f"\n{'='*60}")
    log.info(f"  HetGNN-KKT RESULTS ({len(cold_i)} instances)")
    log.info(f"{'='*60}")
    log.info(f"  Cold:    mean={np.mean(cold_i):.1f}")
    log.info(f"  HetGNN:  mean={np.mean(gnn_i):.1f}  vs cold: {(1-np.mean(gnn_i)/np.mean(cold_i))*100:+.1f}%")
    log.info(f"  Oracle:  mean={np.mean(oracle_i):.1f}  vs cold: {(1-np.mean(oracle_i)/np.mean(cold_i))*100:+.1f}%")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
