"""Power Flow (PF) warm-start benchmark.

Unlike OPF which uses interior-point methods, power flow uses Newton-Raphson
which genuinely benefits from good starting points. DetGNN predictions should
directly reduce NR iteration count if they're close to the PF solution.

This tests whether WARP's predictions help in the PF setting where the
IPM centrality problem doesn't exist.
"""
import os, sys, argparse, time, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("PF-BENCH")

from colab_warp import DetGNN, DEVICE
from normalizer import VariableNormalizer
from torch_geometric.datasets import OPFDataset

import pandapower as pp
import pandapower.networks as pn


def run_pf(net, init="flat", max_iter=100):
    """Run power flow and return iteration count."""
    try:
        pp.runpp(net, init=init, numba=False, max_iteration=max_iter,
                 calculate_voltage_angles=True)
        conv = net.converged
    except Exception:
        conv = False

    n_iter = net._ppc["iterations"] if conv and "_ppc" in net else None
    return conv, n_iter


def set_loads(net, data):
    Pd = data["load"].x[:, 0].cpu().numpy() * 100
    Qd = data["load"].x[:, 1].cpu().numpy() * 100
    for i in range(min(len(net.load), len(Pd))):
        net.load.at[i, "p_mw"] = Pd[i]
        net.load.at[i, "q_mvar"] = Qd[i]


def set_gen_dispatch(net, data):
    """Set generator active power from dataset (needed for PF, not OPF)."""
    Pg = data["generator"].y[:, 0].cpu().numpy() * 100
    for i in range(min(len(net.gen), len(Pg))):
        net.gen.at[i, "p_mw"] = Pg[i]


def set_pf_warmstart(net, Vm, Va):
    """Set voltage warm-start for power flow."""
    n_bus = len(net.bus)
    net.res_bus["vm_pu"] = Vm[:n_bus]
    net.res_bus["va_degree"] = np.degrees(Va[:n_bus])
    net.res_bus["p_mw"] = 0.0
    net.res_bus["q_mvar"] = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--ckpt-dir", default="ckpt")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--use-norm", action="store_true", default=True)
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

    flat_iters, dc_iters, ws_iters = [], [], []
    gt_iters = []

    log.info(f"=== POWER FLOW WARM-START BENCHMARK ===")
    log.info(f"Testing {min(args.n_test, len(test_ds))} instances")

    for idx in range(min(args.n_test, len(test_ds))):
        data = test_ds[idx]

        with torch.no_grad():
            bp, gp = det_model(data.to(DEVICE))
        if norm:
            bp = norm.denormalize_bus(bp)
            gp = norm.denormalize_gen(gp)

        Vm_pred = bp[:, 1].cpu().numpy()
        Va_pred = bp[:, 0].cpu().numpy()

        Vm_gt = data["bus"].y[:, 1].cpu().numpy()
        Va_gt = data["bus"].y[:, 0].cpu().numpy()

        # 1. Flat start
        net = make_net(); set_loads(net, data); set_gen_dispatch(net, data)
        conv_f, ni_f = run_pf(net, "flat")
        flat_iters.append(ni_f)

        # 2. DC init (pandapower's built-in DC power flow init)
        net = make_net(); set_loads(net, data); set_gen_dispatch(net, data)
        conv_dc, ni_dc = run_pf(net, "dc")
        dc_iters.append(ni_dc)

        # 3. DetGNN warm-start
        net = make_net(); set_loads(net, data); set_gen_dispatch(net, data)
        set_pf_warmstart(net, Vm_pred, Va_pred)
        conv_ws, ni_ws = run_pf(net, "results")
        ws_iters.append(ni_ws)

        # 4. Ground-truth warm-start (oracle — upper bound on what's possible)
        net = make_net(); set_loads(net, data); set_gen_dispatch(net, data)
        set_pf_warmstart(net, Vm_gt, Va_gt)
        conv_gt, ni_gt = run_pf(net, "results")
        gt_iters.append(ni_gt)

        if idx % 10 == 0 or idx < 5:
            bus_rmse = np.sqrt(np.mean((Vm_pred[:len(Vm_gt)] - Vm_gt)**2
                                       + (Va_pred[:len(Va_gt)] - Va_gt)**2))
            log.info(f"  #{idx:3d}  flat={ni_f}  dc={ni_dc}  DetGNN={ni_ws}  "
                     f"oracle={ni_gt}  bus_rmse={bus_rmse:.4f}")

    def _s(v):
        v2 = [x for x in v if x is not None]
        return (np.mean(v2), np.median(v2), len(v2)) if v2 else (float('nan'), float('nan'), 0)

    fm, fmd, fn = _s(flat_iters)
    print("\n" + "=" * 80)
    print("  POWER FLOW WARM-START BENCHMARK")
    print("=" * 80)
    for label, vals in [("Flat start", flat_iters), ("DC init", dc_iters),
                        ("DetGNN WS", ws_iters), ("Oracle (GT)", gt_iters)]:
        m, md, n = _s(vals)
        delta = (1 - m / fm) * 100 if fm > 0 else 0
        print(f"  {label:15s}  mean={m:5.2f}  median={md:4.0f}  "
              f"vs flat: {delta:+6.1f}%  (n={n})")

    print("=" * 80)

    if _s(ws_iters)[0] < fm:
        print(f"\n  ✓ DetGNN warm-start BEATS flat start for power flow!")
        print(f"    {(1-_s(ws_iters)[0]/fm)*100:.1f}% reduction in NR iterations")
    else:
        print(f"\n  ✗ DetGNN warm-start does not beat flat start for power flow.")
    print("=" * 80)


if __name__ == "__main__":
    main()
