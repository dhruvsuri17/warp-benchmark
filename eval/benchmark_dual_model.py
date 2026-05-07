"""Benchmark: DetGNN-dual model predictions → IPOPT.

Compares:
1. Cold start (midpoint)
2. DetGNN primal only → IPOPT
3. DetGNN primal + model-predicted duals → IPOPT
4. DetGNN primal + oracle duals → IPOPT (ceiling)
"""
import os, sys, argparse, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pathlib import Path
from numpy import inf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("DUAL-MODEL")

from models.det_gnn_dual import DetGNNDual
from normalizer import VariableNormalizer
from eval.opf_ipopt import build_om, solve_opf
from torch_geometric.datasets import OPFDataset
import pandapower.networks as pn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_loads(net, data):
    Pd = data["load"].x[:, 0].cpu().numpy() * 100
    Qd = data["load"].x[:, 1].cpu().numpy() * 100
    for i in range(min(len(net.load), len(Pd))):
        net.load.at[i, "p_mw"] = Pd[i]
        net.load.at[i, "q_mvar"] = Qd[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--ckpt-dir", default="ckpt")
    parser.add_argument("--duals-dir", default="data/duals")
    parser.add_argument("--n-test", type=int, default=50)
    args = parser.parse_args()

    norm = VariableNormalizer().load(f"{args.ckpt_dir}/normalizer_stats.json")

    model = DetGNNDual(hidden_dim=128, num_layers=6).to(DEVICE)
    ckpt = torch.load(f"{args.ckpt_dir}/det_dual_best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info(f"Loaded model, val_primal={ckpt.get('val_primal', '?')}")

    test_ds = OPFDataset(root="data", case_name=args.case, split="test", num_groups=1)
    duals_dir = Path(args.duals_dir) / args.case / "test"

    cold, primal, model_dual, oracle_dual = [], [], [], []

    for idx in range(min(args.n_test, len(test_ds))):
        data = test_ds[idx]
        net = pn.case118(); set_loads(net, data)
        om, ppopt = build_om(net)
        x0_v, xmin, xmax = om.getv()
        ll, uu = xmin.copy(), xmax.copy()
        ll[xmin == -inf] = -1e10; uu[xmax == inf] = 1e10
        x_mid = (ll + uu) / 2.0
        vv = om.get_idx()[0]

        # Model prediction
        with torch.no_grad():
            out = model(data.to(DEVICE))

        bp = norm.denormalize_bus(out["bus_pred"])
        gp = norm.denormalize_gen(out["gen_pred"])

        n_bus = vv['N']['Va']; n_gen = vv['N']['Pg']
        x_pred = x0_v.copy()
        x_pred[vv['i1']['Va']:vv['iN']['Va']] = np.degrees(bp[:, 0].cpu().numpy())[:n_bus]
        x_pred[vv['i1']['Vm']:vv['iN']['Vm']] = bp[:, 1].cpu().numpy()[:n_bus]
        x_pred[vv['i1']['Pg']:vv['i1']['Pg']+min(n_gen, gp.shape[0])] = gp[:, 0].cpu().numpy()[:n_gen]
        x_pred[vv['i1']['Qg']:vv['i1']['Qg']+min(n_gen, gp.shape[0])] = gp[:, 1].cpu().numpy()[:n_gen]
        x_pred = np.clip(x_pred, xmin + 1e-10, xmax - 1e-10)

        # Model-predicted duals
        lam_eq = out["lam_eq"].detach().cpu().numpy().flatten()
        n_eq = 236; n_ineq = 372
        lam_g_model = np.zeros(n_eq + n_ineq)
        lam_g_model[:min(n_eq, len(lam_eq))] = lam_eq[:n_eq]

        zl_model = np.zeros(len(x_pred))
        zu_model = np.zeros(len(x_pred))
        zl_b = out["zl_bus"].detach().cpu().numpy().flatten()
        zu_b = out["zu_bus"].detach().cpu().numpy().flatten()
        zl_g = out["zl_gen"].detach().cpu().numpy().flatten()
        zu_g = out["zu_gen"].detach().cpu().numpy().flatten()
        zl_model[:min(2*n_bus, len(zl_b))] = zl_b[:2*n_bus]
        zu_model[:min(2*n_bus, len(zu_b))] = zu_b[:2*n_bus]
        zl_model[2*n_bus:2*n_bus+min(2*n_gen, len(zl_g))] = zl_g[:2*n_gen]
        zu_model[2*n_bus:2*n_bus+min(2*n_gen, len(zu_g))] = zu_g[:2*n_gen]

        mu_model = out["mu"].detach().cpu().item()

        # 1. Cold start
        r1 = solve_opf(om, ppopt, x0=x_mid, warm_start=False)
        cold.append(r1["n_iters"])

        # 2. Primal only
        r2 = solve_opf(om, ppopt, x0=x_pred, warm_start=True, mu_init=1e-1)
        primal.append(r2["n_iters"])

        # 3. Model duals
        r3 = solve_opf(om, ppopt, x0=x_pred,
                       lam_g0=lam_g_model, zl0=zl_model, zu0=zu_model,
                       warm_start=True, mu_init=max(mu_model, 1e-8))
        model_dual.append(r3["n_iters"])

        # 4. Oracle duals
        dp = duals_dir / f"duals_{idx:06d}.pt"
        if dp.exists():
            d = torch.load(dp, weights_only=True)
            r4 = solve_opf(om, ppopt, x0=d["x"].numpy(),
                           lam_g0=d["lam_g"].numpy(),
                           zl0=d["zl"].numpy(), zu0=d["zu"].numpy(),
                           warm_start=True, mu_init=d["mu"].item())
            oracle_dual.append(r4["n_iters"])
        else:
            oracle_dual.append(None)

        log.info(f"  #{idx:3d}  cold={cold[-1]}  primal={primal[-1]}  "
                 f"model_dual={model_dual[-1]}  oracle={oracle_dual[-1]}")

    def _s(v):
        v2 = [x for x in v if x is not None]
        return (np.mean(v2), np.median(v2)) if v2 else (float('nan'), float('nan'))

    fm = _s(cold)[0]
    print(f"\n{'='*70}", flush=True)
    print(f"  DETGNN-DUAL MODEL BENCHMARK ({len(cold)} instances)", flush=True)
    print(f"{'='*70}", flush=True)
    for label, vals in [("Cold (midpoint)", cold), ("Primal only", primal),
                        ("Model duals", model_dual), ("Oracle duals", oracle_dual)]:
        m, md = _s(vals)
        delta = (1 - m/fm)*100 if fm > 0 else 0
        print(f"  {label:20s}  mean={m:5.1f}  median={md:4.0f}  vs cold: {delta:+.1f}%", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
