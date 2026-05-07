import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings('ignore')
import torch, numpy as np
from numpy import inf
from training.train_gnn_kkt import HetGNNKKT, DualNorm, unpack_prediction
from eval.opf_ipopt import build_om, solve_opf
from torch_geometric.datasets import OPFDataset
import pandapower.networks as pn
from pathlib import Path

DEVICE = "cuda"
norm = DualNorm().fit("data/duals/pglib_opf_case118_ieee/train", max_n=2000)
model = HetGNNKKT(hidden_dim=128, num_layers=6).to(DEVICE)
model.load_state_dict(torch.load("ckpt/gnn_kkt_best.pt", map_location=DEVICE, weights_only=True))
model.eval()

test_ds = OPFDataset(root="data", case_name="pglib_opf_case118_ieee", split="test", num_groups=1)
test_files = sorted(Path("data/duals/pglib_opf_case118_ieee/test").glob("duals_*.pt"))[:20]

net_ref = pn.case118(); om_ref, _ = build_om(net_ref)
vv = om_ref.get_idx()[0]
n_bus = vv['N']['Va']; n_gen = vv['N']['Pg']

cold_i, gnn_i, oracle_i = [], [], []

with torch.no_grad():
    for ii in range(20):
        duals = torch.load(test_files[ii], weights_only=True, map_location="cpu")
        idx = int(test_files[ii].stem.split("_")[1])
        data = test_ds[idx].to(DEVICE)
        bp, gp = model(data)
        x_n, l_n, zl_n, zu_n = unpack_prediction(bp, gp, n_bus, n_gen)
        x_raw = norm.denorm("x", x_n.cpu()).numpy()
        # GNN only predicts eq duals (2*n_bus=236), normalizer has full 608
        l_raw_n = l_n.cpu()
        l_raw = (l_raw_n * norm.stats["l_s"][:len(l_raw_n)] + norm.stats["l_m"][:len(l_raw_n)]).numpy()
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
        ll, uu = xmin.copy(), xmax.copy()
        ll[xmin == -inf] = -1e10; uu[xmax == inf] = 1e10

        r_cold = solve_opf(om, ppopt, x0=(ll+uu)/2, warm_start=False)
        cold_i.append(r_cold["n_iters"])

        x_m = np.clip(x_raw, xmin+1e-10, xmax-1e-10)
        lam_full = np.zeros(236+372)
        lam_full[:min(len(l_raw), 236)] = l_raw[:236]
        r_gnn = solve_opf(om, ppopt, x0=x_m, lam_g0=lam_full,
                          zl0=np.maximum(zl_raw, 1e-10),
                          zu0=np.maximum(zu_raw, 1e-10),
                          warm_start=True, mu_init=norm.stats["mu"])
        gnn_i.append(r_gnn["n_iters"])

        x_o = np.clip(duals["x"].numpy(), xmin+1e-10, xmax-1e-10)
        r_ora = solve_opf(om, ppopt, x0=x_o,
                          lam_g0=duals["lam_g"].numpy(),
                          zl0=duals["zl"].numpy(),
                          zu0=duals["zu"].numpy(),
                          warm_start=True, mu_init=duals["mu"].item())
        oracle_i.append(r_ora["n_iters"])

        print(f"#{idx}: cold={cold_i[-1]} gnn={gnn_i[-1]} oracle={oracle_i[-1]}", flush=True)

print(f"\nCold:   mean={np.mean(cold_i):.1f}", flush=True)
print(f"HetGNN: mean={np.mean(gnn_i):.1f}  vs cold: {(1-np.mean(gnn_i)/np.mean(cold_i))*100:+.1f}%", flush=True)
print(f"Oracle: mean={np.mean(oracle_i):.1f}  vs cold: {(1-np.mean(oracle_i)/np.mean(cold_i))*100:+.1f}%", flush=True)
