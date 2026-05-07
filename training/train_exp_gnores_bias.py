"""Exp gnores-bias: EPD-GNN with no node residuals + per-node learned bias.

Adds a tiny per-node correction (1,268 params) to the shared GNN predictions.
The GNN provides topology-aware embeddings; the bias corrects per-node idiosyncrasies.
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
log = logging.getLogger("GNORES-BIAS")

from training.train_exp_e_epd_full import EPDGNN, DualNorm, OPFDualDataset, unpack_prediction, EdgeMLP, NodeMLP, INBlock
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from eval.opf_ipopt import build_om, solve_opf
import pandapower.networks as pn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_BUS = 118
N_GEN = 54


class INBlockNoNodeRes(INBlock):
    """IN block with edge residuals but NO node residuals."""
    def forward(self, nodes, edges, data):
        h = nodes; e = edges
        agg_bus = torch.zeros_like(h["bus"])
        agg_gen = torch.zeros_like(h["generator"])
        agg_load = torch.zeros_like(h["load"])

        if ("bus", "ac_line", "bus") in data.edge_types:
            ei = data["bus", "ac_line", "bus"].edge_index; s, d = ei[0], ei[1]
            e_upd = self.edge_mlps["ac_line"](e["ac_line"], h["bus"][s], h["bus"][d])
            e["ac_line"] = e["ac_line"] + e_upd
            agg_bus.scatter_add_(0, d.unsqueeze(1).expand_as(e_upd), e_upd)
            agg_bus.scatter_add_(0, s.unsqueeze(1).expand_as(e_upd), e_upd)
        if ("bus", "transformer", "bus") in data.edge_types:
            ei = data["bus", "transformer", "bus"].edge_index; s, d = ei[0], ei[1]
            e_upd = self.edge_mlps["transformer"](e["transformer"], h["bus"][s], h["bus"][d])
            e["transformer"] = e["transformer"] + e_upd
            agg_bus.scatter_add_(0, d.unsqueeze(1).expand_as(e_upd), e_upd)
            agg_bus.scatter_add_(0, s.unsqueeze(1).expand_as(e_upd), e_upd)
        ei = data["generator", "generator_link", "bus"].edge_index
        e_upd = self.edge_mlps["gen_link"](e["gen_link"], h["generator"][ei[0]], h["bus"][ei[1]])
        e["gen_link"] = e["gen_link"] + e_upd
        agg_bus.scatter_add_(0, ei[1].unsqueeze(1).expand_as(e_upd), e_upd)
        agg_gen = torch.zeros_like(h["generator"])
        ei_rev = data["bus", "generator_link", "generator"].edge_index
        agg_gen.scatter_add_(0, ei_rev[1].unsqueeze(1).expand_as(e_upd), e_upd)
        ei = data["load", "load_link", "bus"].edge_index
        e_upd = self.edge_mlps["load_link"](e["load_link"], h["load"][ei[0]], h["bus"][ei[1]])
        e["load_link"] = e["load_link"] + e_upd
        agg_bus.scatter_add_(0, ei[1].unsqueeze(1).expand_as(e_upd), e_upd)

        h_new = {}
        h_new["bus"] = self.node_mlps["bus"](h["bus"], agg_bus)
        h_new["generator"] = self.node_mlps["generator"](h["generator"], agg_gen)
        h_new["load"] = self.node_mlps["load"](h["load"], agg_load)
        return h_new, e


class EPDGNNNoresBias(EPDGNN):
    """EPD-GNN with no node residuals + per-node learned bias."""
    def __init__(self, hidden_dim=128, k_steps=15, n_bus=N_BUS, n_gen=N_GEN):
        super().__init__(hidden_dim, k_steps)
        self.blocks = nn.ModuleList([INBlockNoNodeRes(hidden_dim) for _ in range(k_steps)])
        self.bus_bias = nn.Parameter(torch.zeros(n_bus, 8))
        self.gen_bias = nn.Parameter(torch.zeros(n_gen, 6))

    def forward(self, data):
        bp, gp = super().forward(data)
        n_bus = bp.shape[0]
        n_gen = gp.shape[0]
        if n_bus == self.bus_bias.shape[0]:
            bp = bp + self.bus_bias
        if n_gen == self.gen_bias.shape[0]:
            gp = gp + self.gen_bias
        return bp, gp


def binding_mask_loss(bp, gp, bt, gt):
    bus_primal = F.mse_loss(bp[:, :2], bt[:, :2])
    gen_primal = F.mse_loss(gp[:, :2], gt[:, :2])
    lam_loss = F.mse_loss(bp[:, 2:4], bt[:, 2:4])
    z_pred = torch.cat([bp[:, 4:8].reshape(-1), gp[:, 2:6].reshape(-1)])
    z_true = torch.cat([bt[:, 4:8].reshape(-1), gt[:, 2:6].reshape(-1)])
    is_binding = z_true.abs() > 0.1
    z_binding = F.mse_loss(z_pred[is_binding], z_true[is_binding]) if is_binding.any() else torch.tensor(0.0, device=bp.device)
    z_nonbinding = z_pred[~is_binding].pow(2).mean() if (~is_binding).any() else torch.tensor(0.0, device=bp.device)
    primal = bus_primal + gen_primal
    dual = lam_loss + z_binding + 0.1 * z_nonbinding
    return primal + dual, primal.item(), dual.item()


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

    model = EPDGNNNoresBias(hidden_dim=args.hidden_dim, k_steps=args.k_steps,
                             n_bus=n_bus, n_gen=n_gen).to(DEVICE)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,} "
             f"(bias: {model.bus_bias.numel() + model.gen_bias.numel()})")

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
            loss, pl, dl = binding_mask_loss(bp, gp, batch["bus"].target, batch["generator"].target)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            s_t += loss.item(); s_p += pl; s_d += dl; n += 1
        sched.step()
        model.eval(); v_p = v_d = 0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE); bp, gp = model(batch)
                _, pl, dl = binding_mask_loss(bp, gp, batch["bus"].target, batch["generator"].target)
                v_p += pl; v_d += dl; vn += 1
        vl = (v_p + v_d) / max(vn, 1)
        log.info(f"Ep {epoch+1}/{args.epochs} | Train: {s_t/n:.4f} | Val primal: {v_p/vn:.4f} | Val dual: {v_d/vn:.4f} | {time.time()-t0:.0f}s")
        if vl < best_val: best_val = vl; torch.save(model.state_dict(), "ckpt/gnores_bias_best.pt")

    log.info(f"Best val: {best_val:.6f}")

    # IPOPT benchmark
    log.info("=== IPOPT BENCHMARK ===")
    model.load_state_dict(torch.load("ckpt/gnores_bias_best.pt", map_location=DEVICE, weights_only=True)); model.eval()
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
