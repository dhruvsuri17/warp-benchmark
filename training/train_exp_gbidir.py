"""Exp G-bidir: EPD-GNN with binding-mask loss + bidirectional messages.

Same as Exp G but adds reverse message aggregation for gen_link and load_link.
gen_link edge updates are scattered to both bus AND generator nodes.
load_link edge updates are scattered to both bus AND load nodes.
(ac_line and transformer already scatter to both s and d.)
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
log = logging.getLogger("EXP-G-BIDIR")

from training.train_exp_e_epd_full import EdgeMLP, NodeMLP, DualNorm, OPFDualDataset, unpack_prediction
from training.train_exp_g_binding_mask import binding_mask_loss
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from eval.opf_ipopt import build_om, solve_opf
import pandapower.networks as pn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class INBlockBidir(nn.Module):
    """IN block with bidirectional message passing on all edge types."""
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

        agg_bus = torch.zeros_like(h["bus"])
        agg_gen = torch.zeros_like(h["generator"])
        agg_load = torch.zeros_like(h["load"])

        # AC lines: bus → bus (already bidirectional)
        if ("bus", "ac_line", "bus") in data.edge_types:
            ei = data["bus", "ac_line", "bus"].edge_index
            s, d = ei[0], ei[1]
            e_upd = self.edge_mlps["ac_line"](e["ac_line"], h["bus"][s], h["bus"][d])
            e["ac_line"] = e["ac_line"] + e_upd
            agg_bus.scatter_add_(0, d.unsqueeze(1).expand_as(e_upd), e_upd)
            agg_bus.scatter_add_(0, s.unsqueeze(1).expand_as(e_upd), e_upd)

        # Transformers: bus → bus (already bidirectional)
        if ("bus", "transformer", "bus") in data.edge_types:
            ei = data["bus", "transformer", "bus"].edge_index
            s, d = ei[0], ei[1]
            e_upd = self.edge_mlps["transformer"](e["transformer"], h["bus"][s], h["bus"][d])
            e["transformer"] = e["transformer"] + e_upd
            agg_bus.scatter_add_(0, d.unsqueeze(1).expand_as(e_upd), e_upd)
            agg_bus.scatter_add_(0, s.unsqueeze(1).expand_as(e_upd), e_upd)

        # Gen → bus: scatter to bus AND generator (bidirectional)
        ei = data["generator", "generator_link", "bus"].edge_index
        e_upd = self.edge_mlps["gen_link"](e["gen_link"], h["generator"][ei[0]], h["bus"][ei[1]])
        e["gen_link"] = e["gen_link"] + e_upd
        agg_bus.scatter_add_(0, ei[1].unsqueeze(1).expand_as(e_upd), e_upd)
        agg_gen.scatter_add_(0, ei[0].unsqueeze(1).expand_as(e_upd), e_upd)
        # Also use the reverse edge index
        ei_rev = data["bus", "generator_link", "generator"].edge_index
        agg_gen.scatter_add_(0, ei_rev[1].unsqueeze(1).expand_as(e_upd), e_upd)

        # Load → bus: scatter to bus AND load (bidirectional)
        ei = data["load", "load_link", "bus"].edge_index
        e_upd = self.edge_mlps["load_link"](e["load_link"], h["load"][ei[0]], h["bus"][ei[1]])
        e["load_link"] = e["load_link"] + e_upd
        agg_bus.scatter_add_(0, ei[1].unsqueeze(1).expand_as(e_upd), e_upd)
        agg_load.scatter_add_(0, ei[0].unsqueeze(1).expand_as(e_upd), e_upd)

        # Node updates with residual
        h_new = {}
        h_new["bus"] = h["bus"] + self.node_mlps["bus"](h["bus"], agg_bus)
        h_new["generator"] = h["generator"] + self.node_mlps["generator"](h["generator"], agg_gen)
        h_new["load"] = h["load"] + self.node_mlps["load"](h["load"], agg_load)

        return h_new, e


class EPDGNNBidir(nn.Module):
    """EPDGNN with bidirectional message passing."""
    def __init__(self, hidden_dim=128, k_steps=15):
        super().__init__()
        h = hidden_dim
        self.node_enc = nn.ModuleDict({
            "bus": nn.Linear(4 + 2, h),
            "generator": nn.Linear(11 + 2, h),
            "load": nn.Linear(2, h),
        })
        self.edge_enc = nn.ModuleDict({
            "ac_line": nn.Linear(9, h),
            "transformer": nn.Linear(11, h),
        })
        self.h = h
        self.blocks = nn.ModuleList([INBlockBidir(h) for _ in range(k_steps)])
        self.k_steps = k_steps
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
        n_gen_edges = data["generator", "generator_link", "bus"].edge_index.shape[1]
        n_load_edges = data["load", "load_link", "bus"].edge_index.shape[1]
        edges["gen_link"] = torch.zeros(n_gen_edges, h, device=data["bus"].x.device)
        edges["load_link"] = torch.zeros(n_load_edges, h, device=data["bus"].x.device)

        for block in self.blocks:
            nodes, edges = block(nodes, edges, data)

        return self.bus_head(nodes["bus"]), self.gen_head(nodes["generator"])


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
    torch.cuda.set_per_process_memory_fraction(0.2)

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

    model = EPDGNNBidir(hidden_dim=args.hidden_dim, k_steps=args.k_steps).to(DEVICE)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    def lr_lambda(ep):
        if ep < 10: return (ep+1)/10
        return 0.9 ** ((ep - 10) // 20)
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val = float("inf"); Path("ckpt").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train(); s_t = s_p = s_d = 0; n = 0; t0 = time.time()
        for batch in train_loader:
            batch = batch.to(DEVICE); bp, gp = model(batch)
            loss, pl, dl, _ = binding_mask_loss(bp, gp, batch["bus"].target, batch["generator"].target)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            s_t += loss.item(); s_p += pl; s_d += dl; n += 1
        sched.step()

        model.eval(); v_t = v_p = v_d = v_bf = 0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE); bp, gp = model(batch)
                _, pl, dl, bf = binding_mask_loss(bp, gp, batch["bus"].target, batch["generator"].target)
                v_p += pl; v_d += dl; v_t += pl + dl; v_bf += bf; vn += 1

        vl = v_t / max(vn, 1)
        log.info(f"Ep {epoch+1}/{args.epochs} | Train: {s_t/n:.4f} | "
                 f"Val primal: {v_p/vn:.4f} | Val dual: {v_d/vn:.4f} | "
                 f"binding%: {v_bf/vn:.1%} | {time.time()-t0:.0f}s")
        if vl < best_val:
            best_val = vl; torch.save(model.state_dict(), "ckpt/exp_gbidir_best.pt")

    log.info(f"Best val: {best_val:.6f}")

    log.info(f"=== IPOPT BENCHMARK ===")
    model.load_state_dict(torch.load("ckpt/exp_gbidir_best.pt", map_location=DEVICE, weights_only=True)); model.eval()
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
            zl_raw = norm.denorm("zl", zl_n.cpu()).numpy(); zu_raw = norm.denorm("zu", zu_n.cpu()).numpy()
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
