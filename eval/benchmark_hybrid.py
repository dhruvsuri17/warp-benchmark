"""Direction 3: Hybrid — constraint screening + voltage warm-start.

Combines the two approaches:
1. Use DetGNN predictions to screen out non-binding constraints
2. Warm-start the reduced problem with voltage predictions (NOT flat start)

Hypothesis: constraint screening alone helps by reducing problem size.
Adding voltage warm-start to the reduced problem might help additionally
because the reduced problem has wider effective Vm bounds (after removing
tight voltage constraints), reducing the centrality penalty.
"""
import os, sys, argparse, time, io, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("D3-HYBRID")

from colab_warp import DetGNN, DEVICE
from normalizer import VariableNormalizer
from inference.warmstart import get_variable_bounds, inject_warmstart
from eval.benchmark_constraint_screen import screen_constraints, set_loads, run_opf
from torch_geometric.datasets import OPFDataset

import pandapower as pp
import pandapower.networks as pn
import pandapower.pypower.opf_execute as _opf_exec

_orig_solver = _opf_exec.pipsopf_solver
def _patched_solver(om, ppopt, out_opt=None):
    if ppopt.get('INIT') == 'results':
        ppopt = dict(ppopt); ppopt['INIT'] = 'pf'
    return _orig_solver(om, ppopt, out_opt)
_opf_exec.pipsopf_solver = _patched_solver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--ckpt-dir", default="ckpt")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--use-norm", action="store_true", default=True)
    parser.add_argument("--vm-margin", type=float, default=0.02)
    args = parser.parse_args()

    norm = None
    if args.use_norm:
        norm = VariableNormalizer().load(f"{args.ckpt_dir}/normalizer_stats.json")

    det_model = DetGNN(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(DEVICE)
    ckpt_name = "det_norm_best.pt" if args.use_norm else "det_best.pt"
    det_ckpt = torch.load(f"{args.ckpt_dir}/{ckpt_name}", map_location=DEVICE, weights_only=True)
    det_model.load_state_dict(det_ckpt["model"])
    det_model.eval()

    make_net = pn.case118
    test_ds = OPFDataset(root="data", case_name=args.case, split="test", num_groups=1)

    flat_i, screen_flat_i, screen_ws_i, ws_only_i = [], [], [], []

    log.info(f"=== DIRECTION 3: HYBRID (screen + warm-start) ===")

    for idx in range(min(args.n_test, len(test_ds))):
        data = test_ds[idx]

        with torch.no_grad():
            bp, gp = det_model(data.to(DEVICE))
        if norm:
            bp = norm.denormalize_bus(bp)
            gp = norm.denormalize_gen(gp)

        Vm = bp[:, 1].cpu().numpy()
        Va = bp[:, 0].cpu().numpy()
        Va_deg = np.degrees(Va)
        Pg = gp[:, 0].cpu().numpy() * 100
        Qg = gp[:, 1].cpu().numpy() * 100

        n_bus = len(make_net().bus)
        n_gen = len(make_net().gen)

        # 1. Baseline: flat start, original problem
        net = make_net(); set_loads(net, data)
        _, nf, _ = run_opf(net, "flat")
        flat_i.append(nf)

        # 2. Screen + flat start (Direction 1)
        net = make_net(); set_loads(net, data)
        net, stats = screen_constraints(net, Vm, Va, Pg, Qg,
                                         vm_margin=args.vm_margin)
        _, ns_flat, _ = run_opf(net, "flat")
        screen_flat_i.append(ns_flat)

        # 3. Screen + warm-start voltages (Direction 3 — hybrid)
        net = make_net(); set_loads(net, data)
        net, _ = screen_constraints(net, Vm, Va, Pg, Qg,
                                     vm_margin=args.vm_margin)
        lb, ub = get_variable_bounds(net)
        x_mid = (lb + ub) / 2.0
        x_ws = x_mid.copy()
        x_ws[:n_bus] = Vm[:n_bus]
        x_ws[n_bus:2*n_bus] = Va_deg[:n_bus]
        x_ws = np.clip(x_ws, lb + 1e-6*(ub-lb), ub - 1e-6*(ub-lb))
        inject_warmstart(net, x_ws)
        _, ns_ws, _ = run_opf(net, "results")
        screen_ws_i.append(ns_ws)

        # 4. Warm-start only, no screening (for comparison)
        net = make_net(); set_loads(net, data)
        x_ws2 = x_mid.copy()
        x_ws2[:n_bus] = Vm[:n_bus]
        x_ws2[n_bus:2*n_bus] = Va_deg[:n_bus]
        x_ws2 = np.clip(x_ws2, lb + 1e-6*(ub-lb), ub - 1e-6*(ub-lb))
        inject_warmstart(net, x_ws2)
        _, nw, _ = run_opf(net, "results")
        ws_only_i.append(nw)

        log.info(f"  #{idx:3d}  flat={nf}  screen+flat={ns_flat}  "
                 f"screen+ws={ns_ws}  ws_only={nw}  removed={stats['removal_pct']:.0f}%")

    def _s(v):
        v2 = [x for x in v if x is not None]
        return np.mean(v2), np.median(v2)

    fm = _s(flat_i)[0]
    print("\n" + "=" * 80)
    print("  DIRECTION 3: HYBRID RESULTS")
    print("=" * 80)
    for label, vals in [("Flat (baseline)", flat_i),
                        ("Screen + flat", screen_flat_i),
                        ("Screen + WS", screen_ws_i),
                        ("WS only (no screen)", ws_only_i)]:
        m, md = _s(vals)
        print(f"  {label:22s}  mean={m:5.1f}  median={md:4.0f}  vs flat: {(1-m/fm)*100:+.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
