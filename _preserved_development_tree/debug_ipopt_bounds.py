import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings('ignore')
import numpy as np; from numpy import inf
import torch
import pandapower.networks as pn
from eval.ipopt_opf_v2 import build_om, solve_opf
from colab_warp import DetGNN, DEVICE
from normalizer import VariableNormalizer

norm = VariableNormalizer().load("ckpt/normalizer_stats.json")
det_model = DetGNN(hidden_dim=128, num_layers=6).to(DEVICE)
det_ckpt = torch.load("ckpt/det_norm_best.pt", map_location=DEVICE, weights_only=True)
det_model.load_state_dict(det_ckpt["model"]); det_model.eval()

from torch_geometric.datasets import OPFDataset
test_ds = OPFDataset(root="data", case_name="pglib_opf_case118_ieee", split="test", num_groups=1)
data = test_ds[0]

with torch.no_grad():
    bp, gp = det_model(data.to(DEVICE))
bp = norm.denormalize_bus(bp); gp = norm.denormalize_gen(gp)

net = pn.case118()
Pd = data["load"].x[:, 0].cpu().numpy() * 100
Qd = data["load"].x[:, 1].cpu().numpy() * 100
for i in range(min(len(net.load), len(Pd))):
    net.load.at[i, "p_mw"] = Pd[i]; net.load.at[i, "q_mvar"] = Qd[i]

om, ppopt = build_om(net)
vv = om.get_idx()[0]
x0, xmin, xmax = om.getv()
ll, uu = xmin.copy(), xmax.copy()
ll[xmin == -inf] = -1e10; uu[xmax == inf] = 1e10
x_mid = (ll + uu) / 2.0

n_va = vv['N']['Va']; n_vm = vv['N']['Vm']
n_pg = vv['N']['Pg']; n_qg = vv['N']['Qg']

Vm = bp[:, 1].cpu().numpy(); Va = bp[:, 0].cpu().numpy()
Pg = gp[:, 0].cpu().numpy() * 100; Qg = gp[:, 1].cpu().numpy() * 100

va_s = slice(vv['i1']['Va'], vv['iN']['Va'])
vm_s = slice(vv['i1']['Vm'], vv['iN']['Vm'])
pg_s = slice(vv['i1']['Pg'], vv['iN']['Pg'])
qg_s = slice(vv['i1']['Qg'], vv['iN']['Qg'])

print(f"Prediction ranges:", flush=True)
print(f"  Va(deg): [{np.degrees(Va.min()):.1f}, {np.degrees(Va.max()):.1f}]", flush=True)
print(f"  Vm: [{Vm.min():.3f}, {Vm.max():.3f}]", flush=True)
print(f"  Pg(MW): [{Pg.min():.1f}, {Pg.max():.1f}]", flush=True)
print(f"  Qg(Mvar): [{Qg.min():.1f}, {Qg.max():.1f}]", flush=True)

print(f"IPOPT bounds:", flush=True)
print(f"  Va: [{xmin[va_s].min():.1f}, {xmax[va_s].max():.1f}]", flush=True)
print(f"  Vm: [{xmin[vm_s].min():.3f}, {xmax[vm_s].max():.3f}]", flush=True)
print(f"  Pg: [{xmin[pg_s].min():.1f}, {xmax[pg_s].max():.1f}]", flush=True)
print(f"  Qg: [{xmin[qg_s].min():.1f}, {xmax[qg_s].max():.1f}]", flush=True)

x_pred = x_mid.copy()
x_pred[va_s] = np.degrees(Va[:n_va])
x_pred[vm_s] = Vm[:n_vm]
x_pred[pg_s] = Pg[:min(n_pg, len(Pg))]
x_pred[qg_s] = Qg[:min(n_qg, len(Qg))]
x_pred_clipped = np.clip(x_pred, xmin + 1e-8, xmax - 1e-8)

violations = ((x_pred < xmin) | (x_pred > xmax))
print(f"Bound violations: {violations.sum()} / {len(x_pred)}", flush=True)

print(f"\nTest: IPOPT cold GNN (no warm-start mode)...", flush=True)
r = solve_opf(om, ppopt, x0=x_pred_clipped, warm_start=False, print_level=0, max_iter=50)
print(f"  Iters={r['n_iters']}, Conv={r['converged']}, Status={r['status']}", flush=True)

print(f"\nTest: IPOPT flat...", flush=True)
r2 = solve_opf(om, ppopt, x0=x_mid, warm_start=False, print_level=0, max_iter=50)
print(f"  Iters={r2['n_iters']}, Conv={r2['converged']}", flush=True)
