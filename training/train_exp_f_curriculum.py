"""Exp F: EPD-GNN with curriculum loss (primal first, then duals).

Same architecture as Exp E. Changes:
1. Primal/dual loss decomposition logging
2. Curriculum: dual_weight ramps from 0 to 1 over 50 epochs
3. Lets the model learn good embeddings from easy primals first
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
log = logging.getLogger("EXP-F")

from training.train_exp_e_epd_full import EPDGNN, DualNorm, OPFDualDataset, unpack_prediction
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from eval.opf_ipopt import build_om, solve_opf
import pandapower.networks as pn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def primal_dual_loss(bp, gp, bt, gt, dual_weight=1.0):
    """Decomposed loss: primals (Va,Vm,Pg,Qg) vs duals (lam,zl,zu)."""
    # Bus: [0:2] = primals (Va,Vm), [2:8] = duals (lam_P,lam_Q,zl,zu)
    # Gen: [0:2] = primals (Pg,Qg), [2:6] = duals (zl,zu)
    bus_primal = F.mse_loss(bp[:, :2], bt[:, :2])
    bus_dual = F.mse_loss(bp[:, 2:], bt[:, 2:])
    gen_primal = F.mse_loss(gp[:, :2], gt[:, :2])
    gen_dual = F.mse_loss(gp[:, 2:], gt[:, 2:])
    primal = bus_primal + gen_primal
    dual = bus_dual + gen_dual
    total = primal + dual_weight * dual
    return total, primal.item(), dual.item()


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
    parser.add_argument("--curriculum-epochs", type=int, default=50)
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

    model = EPDGNN(hidden_dim=args.hidden_dim, k_steps=args.k_steps).to(DEVICE)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    warmup = 10
    def lr_lambda(ep):
        if ep < warmup: return (ep+1)/warmup
        return 0.9 ** ((ep - warmup) // 20)
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val = float("inf")
    Path("ckpt").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        dual_weight = min(1.0, epoch / args.curriculum_epochs)
        model.train(); s_total = s_primal = s_dual = 0; n = 0; t0 = time.time()

        for batch in train_loader:
            batch = batch.to(DEVICE)
            bp, gp = model(batch)
            loss, p_l, d_l = primal_dual_loss(bp, gp, batch["bus"].target, batch["generator"].target, dual_weight)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            s_total += loss.item(); s_primal += p_l; s_dual += d_l; n += 1
        sched.step()

        model.eval(); v_total = v_primal = v_dual = 0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                bp, gp = model(batch)
                _, p_l, d_l = primal_dual_loss(bp, gp, batch["bus"].target, batch["generator"].target, 1.0)
                v_primal += p_l; v_dual += d_l; v_total += p_l + d_l; vn += 1

        vl = v_total / max(vn, 1)
        log.info(f"Ep {epoch+1}/{args.epochs} | Total: {s_total/n:.4f} | "
                 f"Val primal: {v_primal/vn:.4f} | Val dual: {v_dual/vn:.4f} | "
                 f"Val total: {vl:.4f} | dw={dual_weight:.2f} | {time.time()-t0:.0f}s")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), "ckpt/exp_f_best.pt")

    log.info(f"Best val: {best_val:.6f}")

    # IPOPT benchmark (same as Exp E)
    log.info(f"=== IPOPT BENCHMARK ===")
    model.load_state_dict(torch.load("ckpt/exp_f_best.pt", map_location=DEVICE, weights_only=True))
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
            l_raw = (l_n.cpu() * norm.stats["l_s"][:len(l_n)] + norm.stats["l_m"][:len(l_n)]).numpy()
            zl_raw = norm.denorm("zl", zl_n.cpu()).numpy()
            zu_raw = norm.denorm("zu", zu_n.cpu()).numpy()
            net = pn.case118(); data_cpu = test_opf[idx]
            Pd = data_cpu["load"].x[:, 0].numpy() * 100; Qd = data_cpu["load"].x[:, 1].numpy() * 100
            for i in range(min(len(net.load), len(Pd))):
                net.load.at[i, "p_mw"] = Pd[i]; net.load.at[i, "q_mvar"] = Qd[i]
            om, ppopt = build_om(net); x0_v, xmin, xmax = om.getv()
            from numpy import inf as npinf
            ll, uu = xmin.copy(), xmax.copy(); ll[xmin==-npinf]=-1e10; uu[xmax==npinf]=1e10
            r_cold = solve_opf(om, ppopt, x0=(ll+uu)/2, warm_start=False); cold_i.append(r_cold["n_iters"])
            x_m = np.clip(x_raw, xmin+1e-10, xmax-1e-10)
            lam_full = np.zeros(236+372); lam_full[:min(len(l_raw),236)] = l_raw[:236]
            r_gnn = solve_opf(om, ppopt, x0=x_m, lam_g0=lam_full,
                              zl0=np.maximum(zl_raw,1e-10), zu0=np.maximum(zu_raw,1e-10),
                              warm_start=True, mu_init=norm.stats["mu"]); gnn_i.append(r_gnn["n_iters"])
            x_o = np.clip(duals["x"].numpy(), xmin+1e-10, xmax-1e-10)
            r_ora = solve_opf(om, ppopt, x0=x_o, lam_g0=duals["lam_g"].numpy(),
                              zl0=duals["zl"].numpy(), zu0=duals["zu"].numpy(),
                              warm_start=True, mu_init=duals["mu"].item()); oracle_i.append(r_ora["n_iters"])
            log.info(f"  #{idx}: cold={cold_i[-1]} model={gnn_i[-1]} oracle={oracle_i[-1]}")
    log.info(f"Cold: {np.mean(cold_i):.1f} | Model: {np.mean(gnn_i):.1f} | Oracle: {np.mean(oracle_i):.1f}")

if __name__ == "__main__":
    main()
