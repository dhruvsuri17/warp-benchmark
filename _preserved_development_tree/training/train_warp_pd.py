"""Train WARP-PD: diffusion over the full IPM state.

Uses the same data pipeline as the deterministic model.
Trains a DDPM noise prediction model on normalized (x, λ, z, μ) targets.
At inference, samples K=5 candidates and selects by KKT residual.
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
log = logging.getLogger("WARP-PD")

from models.warp_pd import WARPPD
from training.train_exp_e_epd_full import DualNorm, OPFDualDataset, unpack_prediction
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from eval.opf_ipopt import build_om, solve_opf
import pandapower.networks as pn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--k-steps", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--ddim-steps", type=int, default=50)
    args = parser.parse_args()

    log.info(f"Device: {DEVICE}")
    torch.cuda.set_per_process_memory_fraction(0.5)
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

    model = WARPPD(hidden_dim=args.hidden_dim, k_steps=args.k_steps, T=args.T).to(DEVICE)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    def lr_lambda(ep):
        if ep < 10: return (ep+1)/10
        return 0.9 ** ((ep - 10) // 20)
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val = float("inf"); Path("ckpt").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train(); s_loss = 0; n = 0; t0 = time.time()
        for data in train_loader:
            data = data.to(DEVICE)
            clean_bus = data["bus"].target
            clean_gen = data["generator"].target
            loss = model(data, clean_bus, clean_gen)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            s_loss += loss.item(); n += 1
        sched.step()

        model.eval(); v_loss = 0; vn = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                loss = model(data, data["bus"].target, data["generator"].target)
                v_loss += loss.item(); vn += 1
                if vn >= 200: break

        vl = v_loss / max(vn, 1)
        log.info(f"Ep {epoch+1}/{args.epochs} | Train: {s_loss/n:.4f} | Val: {vl:.4f} | {time.time()-t0:.0f}s")
        if vl < best_val:
            best_val = vl; torch.save(model.state_dict(), "ckpt/warp_pd_best.pt")

    log.info(f"Best val: {best_val:.6f}")

    # IPOPT benchmark: deterministic (K=1) vs K=3 vs K=5
    log.info(f"=== IPOPT BENCHMARK ===")
    model.load_state_dict(torch.load("ckpt/warp_pd_best.pt", map_location=DEVICE, weights_only=True))
    model.eval()

    test_opf = OPFDataset(root="data", case_name="pglib_opf_case118_ieee", split="test", num_groups=1)
    test_files = sorted((duals_base/"test").glob("duals_*.pt"))[:args.n_test]

    for K in [1, 3, 5]:
        cold_i, warp_i, oracle_i = [], [], []
        with torch.no_grad():
            for ii in range(min(args.n_test, len(test_files))):
                duals = torch.load(test_files[ii], weights_only=True, map_location="cpu")
                idx = int(test_files[ii].stem.split("_")[1])
                data = test_opf[idx].to(DEVICE)

                # Sample K candidates
                best_bus, best_gen, _ = model.sample_and_score(
                    data, norm, K=K, steps=args.ddim_steps)

                x_n, l_n, zl_n, zu_n = unpack_prediction(best_bus, best_gen, n_bus, n_gen)
                x_raw = norm.denorm("x", x_n.cpu()).numpy()
                l_raw = (l_n.cpu() * norm.stats["l_s"][:len(l_n)] + norm.stats["l_m"][:len(l_n)]).numpy()
                zl_raw = norm.denorm("zl", zl_n.cpu()).numpy()
                zu_raw = norm.denorm("zu", zu_n.cpu()).numpy()

                data_cpu = test_opf[idx]
                net = pn.case118()
                Pd = data_cpu["load"].x[:,0].numpy()*100; Qd = data_cpu["load"].x[:,1].numpy()*100
                for i in range(min(len(net.load),len(Pd))):
                    net.load.at[i,"p_mw"]=Pd[i]; net.load.at[i,"q_mvar"]=Qd[i]
                om, ppopt = build_om(net); x0_v, xmin, xmax = om.getv()
                from numpy import inf as npinf
                ll,uu=xmin.copy(),xmax.copy(); ll[xmin==-npinf]=-1e10; uu[xmax==npinf]=1e10

                r_cold = solve_opf(om, ppopt, x0=(ll+uu)/2, warm_start=False)
                cold_i.append(r_cold["n_iters"])

                x_m = np.clip(x_raw, xmin+1e-10, xmax-1e-10)
                lam_full = np.zeros(608); lam_full[:min(len(l_raw),236)] = l_raw[:236]
                r_warp = solve_opf(om, ppopt, x0=x_m, lam_g0=lam_full,
                                   zl0=np.maximum(zl_raw,1e-10),
                                   zu0=np.maximum(zu_raw,1e-10),
                                   warm_start=True, mu_init=norm.stats["mu"])
                warp_i.append(r_warp["n_iters"])

                x_o = np.clip(duals["x"].numpy(), xmin+1e-10, xmax-1e-10)
                r_ora = solve_opf(om, ppopt, x0=x_o, lam_g0=duals["lam_g"].numpy(),
                                  zl0=duals["zl"].numpy(), zu0=duals["zu"].numpy(),
                                  warm_start=True, mu_init=duals["mu"].item())
                oracle_i.append(r_ora["n_iters"])

        log.info(f"K={K}: Cold={np.mean(cold_i):.1f} | WARP-PD={np.mean(warp_i):.1f} | Oracle={np.mean(oracle_i):.1f}")

    log.info("Done!")


if __name__ == "__main__":
    main()
