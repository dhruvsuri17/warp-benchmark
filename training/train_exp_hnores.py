"""Exp H-nores: Two-stage decode + no node residuals.

Combines: predict primals → condition duals on primals (Exp H)
with: no node residuals (gnores breakthrough).
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
log = logging.getLogger("H-NORES")

from training.train_exp_gnores_bias import INBlockNoNodeRes
from training.train_exp_e_epd_full import EPDGNN, DualNorm, OPFDualDataset, unpack_prediction
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from eval.opf_ipopt import build_om, solve_opf
import pandapower.networks as pn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EPDGNNTwoStageNores(EPDGNN):
    """EPD-GNN: no node residuals + two-stage decode (primals → duals)."""
    def __init__(self, hidden_dim=128, k_steps=15):
        super().__init__(hidden_dim, k_steps)
        self.blocks = nn.ModuleList([INBlockNoNodeRes(hidden_dim) for _ in range(k_steps)])
        h = hidden_dim
        # Replace single heads with two-stage
        self.bus_primal_head = nn.Sequential(nn.Linear(h, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 2))
        self.bus_dual_head = nn.Sequential(nn.Linear(h+2, 256), nn.LayerNorm(256), nn.ReLU(),
                                           nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 6))
        self.gen_primal_head = nn.Sequential(nn.Linear(h, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 2))
        self.gen_dual_head = nn.Sequential(nn.Linear(h+2, 256), nn.LayerNorm(256), nn.ReLU(),
                                           nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 4))
        del self.bus_head; del self.gen_head

    def forward(self, data):
        n_bus = data["bus"].x.shape[0]; n_gen = data["generator"].x.shape[0]
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

        # Two-stage: primals first
        bus_p = self.bus_primal_head(nodes["bus"])
        gen_p = self.gen_primal_head(nodes["generator"])
        # Then duals conditioned on primals (detached)
        bus_d = self.bus_dual_head(torch.cat([nodes["bus"], bus_p.detach()], -1))
        gen_d = self.gen_dual_head(torch.cat([nodes["generator"], gen_p.detach()], -1))
        return torch.cat([bus_p, bus_d], -1), torch.cat([gen_p, gen_d], -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--k-steps", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-test", type=int, default=20)
    args = parser.parse_args()

    log.info(f"Device: {DEVICE}")
    torch.cuda.set_per_process_memory_fraction(0.3)
    duals_base = Path("data/duals/pglib_opf_case118_ieee")
    norm = DualNorm().fit(duals_base / "train", max_n=5000)
    net_ref = pn.case118(); om_ref, _ = build_om(net_ref)
    vv = om_ref.get_idx()[0]; n_bus = vv['N']['Va']; n_gen = vv['N']['Pg']

    train_opf = OPFDataset(root="data", case_name="pglib_opf_case118_ieee", split="train", num_groups=1)
    val_opf = OPFDataset(root="data", case_name="pglib_opf_case118_ieee", split="val", num_groups=1)
    train_ds = OPFDualDataset(train_opf, duals_base/"train", norm, n_bus, n_gen, 5000)
    val_ds = OPFDualDataset(val_opf, duals_base/"val", norm, n_bus, n_gen, 500)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = EPDGNNTwoStageNores(hidden_dim=args.hidden_dim, k_steps=args.k_steps).to(DEVICE)
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
            primal_loss = F.mse_loss(bp[:,:2], batch["bus"].target[:,:2]) + F.mse_loss(gp[:,:2], batch["generator"].target[:,:2])
            dual_loss = F.mse_loss(bp[:,2:], batch["bus"].target[:,2:]) + F.mse_loss(gp[:,2:], batch["generator"].target[:,2:])
            loss = primal_loss + dual_loss
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            s_t += loss.item(); s_p += primal_loss.item(); s_d += dual_loss.item(); n += 1
        sched.step()
        model.eval(); v_p = v_d = 0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE); bp, gp = model(batch)
                v_p += (F.mse_loss(bp[:,:2], batch["bus"].target[:,:2]) + F.mse_loss(gp[:,:2], batch["generator"].target[:,:2])).item()
                v_d += (F.mse_loss(bp[:,2:], batch["bus"].target[:,2:]) + F.mse_loss(gp[:,2:], batch["generator"].target[:,2:])).item()
                vn += 1
        vl = (v_p + v_d) / max(vn, 1)
        log.info(f"Ep {epoch+1}/{args.epochs} | Train: {s_t/n:.4f} | Val primal: {v_p/vn:.4f} | Val dual: {v_d/vn:.4f} | {time.time()-t0:.0f}s")
        if vl < best_val: best_val = vl; torch.save(model.state_dict(), "ckpt/hnores_best.pt")

    log.info(f"Best val: {best_val:.6f}")

    # IPOPT benchmark
    log.info("=== IPOPT BENCHMARK ===")
    model.load_state_dict(torch.load("ckpt/hnores_best.pt", map_location=DEVICE, weights_only=True)); model.eval()
    test_opf = OPFDataset(root="data", case_name="pglib_opf_case118_ieee", split="test", num_groups=1)
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
