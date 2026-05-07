"""Exp T: Transformer baseline — no graph structure, full self-attention.

All bus+gen nodes treated as tokens. Laplacian PE for positional encoding.
Tests whether graph inductive bias matters or if a transformer can match
the GNN by learning structure from data.
"""
import os, sys, argparse, time, logging
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("EXP-T")

from training.train_exp_e_epd_full import DualNorm, OPFDualDataset, unpack_prediction
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from eval.opf_ipopt import build_om, solve_opf
import pandapower.networks as pn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PE_DIM = 16  # Laplacian PE dimensions


def compute_laplacian_pe(data, k=PE_DIM):
    """Compute top-k Laplacian eigenvectors for positional encoding."""
    n_bus = data["bus"].x.shape[0]
    # Build adjacency from ac_line edges
    adj = torch.zeros(n_bus, n_bus)
    if ("bus", "ac_line", "bus") in data.edge_types:
        ei = data["bus", "ac_line", "bus"].edge_index
        adj[ei[0], ei[1]] = 1; adj[ei[1], ei[0]] = 1
    if ("bus", "transformer", "bus") in data.edge_types:
        ei = data["bus", "transformer", "bus"].edge_index
        adj[ei[0], ei[1]] = 1; adj[ei[1], ei[0]] = 1

    L = laplacian(csr_matrix(adj.numpy()), normed=True)
    try:
        eigenvalues, eigenvectors = eigsh(L, k=min(k, n_bus-2), which='SM')
        pe = torch.tensor(eigenvectors, dtype=torch.float32)
    except Exception:
        pe = torch.zeros(n_bus, k)

    if pe.shape[1] < k:
        pe = F.pad(pe, (0, k - pe.shape[1]))
    return pe


class TransformerOPF(nn.Module):
    """Pure transformer over bus+gen node tokens with Laplacian PE."""

    def __init__(self, n_bus_feat=4, n_gen_feat=11, n_load_feat=2,
                 pe_dim=PE_DIM, hidden=256, n_layers=6, n_heads=8):
        super().__init__()
        self.hidden = hidden
        self.pe_dim = pe_dim

        # Token embedders
        self.bus_embed = nn.Linear(n_bus_feat + 2 + pe_dim, hidden)  # +2 for injected load
        self.gen_embed = nn.Linear(n_gen_feat + 2 + pe_dim, hidden)  # +2 for bus load

        # Type embedding (bus vs gen)
        self.type_embed = nn.Embedding(2, hidden)

        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden*4,
            dropout=0.1, batch_first=True, norm_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Output heads
        self.bus_head = nn.Sequential(
            nn.Linear(hidden, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 8))
        self.gen_head = nn.Sequential(
            nn.Linear(hidden, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 6))

    def forward(self, data):
        n_bus = data["bus"].x.shape[0]
        n_gen = data["generator"].x.shape[0]
        device = data["bus"].x.device

        # Inject loads onto buses
        bus_load = torch.zeros(n_bus, 2, device=device)
        ei_l = data["load", "load_link", "bus"].edge_index
        bus_load.scatter_add_(0, ei_l[1].unsqueeze(1).expand(-1, 2), data["load"].x[ei_l[0]])

        # Inject loads onto generators
        ei_g = data["generator", "generator_link", "bus"].edge_index
        gen_load = bus_load[ei_g[1]]

        # Compute Laplacian PE (cached on data if available)
        if not hasattr(data, '_lap_pe') or data._lap_pe is None:
            data._lap_pe = compute_laplacian_pe(data, self.pe_dim).to(device)
        bus_pe = data._lap_pe[:n_bus]

        # Gen PE: use PE of connected bus
        gen_pe = bus_pe[ei_g[1]] if ei_g.shape[1] == n_gen else torch.zeros(n_gen, self.pe_dim, device=device)

        # Build tokens
        bus_tokens = self.bus_embed(torch.cat([data["bus"].x, bus_load, bus_pe], dim=-1))
        gen_tokens = self.gen_embed(torch.cat([data["generator"].x, gen_load, gen_pe], dim=-1))

        # Add type embeddings
        bus_tokens = bus_tokens + self.type_embed(torch.zeros(n_bus, dtype=torch.long, device=device))
        gen_tokens = gen_tokens + self.type_embed(torch.ones(n_gen, dtype=torch.long, device=device))

        # Concatenate all tokens
        tokens = torch.cat([bus_tokens, gen_tokens], dim=0).unsqueeze(0)  # [1, n_bus+n_gen, hidden]

        # Self-attention
        h = self.encoder(tokens).squeeze(0)  # [n_bus+n_gen, hidden]

        bus_out = self.bus_head(h[:n_bus])
        gen_out = self.gen_head(h[n_bus:])
        return bus_out, gen_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--duals-dir", default="data/duals")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--max-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=20)
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

    # Single-instance processing for transformer (batching across graphs is complex)
    model = TransformerOPF(hidden=args.hidden_dim, n_layers=args.n_layers,
                           n_heads=args.n_heads).to(DEVICE)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = 10
    def lr_lambda(ep):
        if ep < warmup: return (ep+1)/warmup
        return 0.9 ** ((ep - warmup) // 20)
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val = float("inf"); Path("ckpt").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train(); s_t = s_p = s_d = 0; n = 0; t0 = time.time()
        indices = np.random.permutation(len(train_ds))
        for ii in indices:
            data = train_ds[ii].to(DEVICE)
            bp, gp = model(data)
            primal_loss = F.mse_loss(bp[:,:2], data["bus"].target[:,:2]) + \
                          F.mse_loss(gp[:,:2], data["generator"].target[:,:2])
            dual_loss = F.mse_loss(bp[:,2:], data["bus"].target[:,2:]) + \
                        F.mse_loss(gp[:,2:], data["generator"].target[:,2:])
            loss = primal_loss + dual_loss
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            s_t += loss.item(); s_p += primal_loss.item(); s_d += dual_loss.item(); n += 1
        sched.step()

        model.eval(); v_p = v_d = 0; vn = 0
        with torch.no_grad():
            for ii in range(min(200, len(val_ds))):
                data = val_ds[ii].to(DEVICE)
                bp, gp = model(data)
                v_p += (F.mse_loss(bp[:,:2], data["bus"].target[:,:2]) +
                        F.mse_loss(gp[:,:2], data["generator"].target[:,:2])).item()
                v_d += (F.mse_loss(bp[:,2:], data["bus"].target[:,2:]) +
                        F.mse_loss(gp[:,2:], data["generator"].target[:,2:])).item()
                vn += 1

        vl = (v_p + v_d) / max(vn, 1)
        log.info(f"Ep {epoch+1}/{args.epochs} | Train: {s_t/n:.4f} | "
                 f"Val primal: {v_p/vn:.4f} | Val dual: {v_d/vn:.4f} | "
                 f"Val total: {vl:.4f} | {time.time()-t0:.0f}s")
        if vl < best_val:
            best_val = vl; torch.save(model.state_dict(), "ckpt/exp_t_best.pt")

    log.info(f"Best val: {best_val:.6f}")

    # IPOPT benchmark
    log.info(f"=== IPOPT BENCHMARK ===")
    model.load_state_dict(torch.load("ckpt/exp_t_best.pt", map_location=DEVICE, weights_only=True))
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
            log.info(f"  #{idx}: cold={cold_i[-1]} transformer={gnn_i[-1]} oracle={oracle_i[-1]}")
    log.info(f"Cold: {np.mean(cold_i):.1f} | Transformer: {np.mean(gnn_i):.1f} | Oracle: {np.mean(oracle_i):.1f}")


if __name__ == "__main__":
    main()
