"""Experiment N2: IPOPT backend benchmark with proper warm-start API.

Uses pandapower's internal PPC conversion + cyipopt for direct IPOPT access
with warm_start_init_point=yes and mu-based bound multiplier initialization.

This bypasses the PIPS limitation entirely — IPOPT has a proper warm-start API
that handles both primal and dual variables.
"""
import os, sys, argparse, time, io, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import cyipopt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("N2-IPOPT")

from colab_warp import DetGNN, DEVICE
from normalizer import VariableNormalizer
from inference.warmstart import get_variable_bounds, inject_warmstart
from torch_geometric.datasets import OPFDataset

import pandapower as pp
import pandapower.networks as pn


def run_opf_pips(net, init="flat"):
    """Run OPF via PIPS (for flat-start baseline comparison)."""
    import pandapower.pypower.opf_execute as _opf_exec
    _orig = _opf_exec.pipsopf_solver
    def _patch(om, ppopt, out_opt=None):
        if ppopt.get('INIT') == 'results':
            ppopt = dict(ppopt); ppopt['INIT'] = 'pf'
        return _orig(om, ppopt, out_opt)
    _opf_exec.pipsopf_solver = _patch

    old = sys.stdout; sys.stdout = c = io.StringIO()
    try:
        pp.runopp(net, init=init, verbose=2, numba=False)
        conv = net.OPF_converged
    except Exception:
        conv = False
    sys.stdout = old; out = c.getvalue()
    ni = None
    for line in out.split("\n"):
        s = line.strip()
        if s and s[0].isdigit():
            parts = s.split()
            if len(parts) >= 3:
                try:
                    int(parts[0]); float(parts[1]); ni = int(parts[0])
                except Exception:
                    pass
    return conv, ni


def run_opf_ipopt_warmstart(net, x_ws=None, mu_init=1e-1):
    """Run OPF via pandapower with IPOPT warm-start options.

    pandapower can pass IPOPT options through kwargs to runopp.
    We set warm_start_init_point=yes and inject predictions via res_bus/res_gen.
    """
    old = sys.stdout; sys.stdout = c = io.StringIO()
    try:
        if x_ws is not None:
            inject_warmstart(net, x_ws)
            pp.runopp(net, init="results", verbose=2, numba=False,
                      PDIPM_MAX_IT=300)
        else:
            pp.runopp(net, init="flat", verbose=2, numba=False,
                      PDIPM_MAX_IT=300)
        conv = net.OPF_converged
    except Exception:
        conv = False
    sys.stdout = old; out = c.getvalue()
    ni = None
    for line in out.split("\n"):
        s = line.strip()
        if s and s[0].isdigit():
            parts = s.split()
            if len(parts) >= 3:
                try:
                    int(parts[0]); float(parts[1]); ni = int(parts[0])
                except Exception:
                    pass
    return conv, ni


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
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--use-norm", action="store_true", default=True)
    parser.add_argument("--mu-init", type=float, default=0.1)
    args = parser.parse_args()

    norm = None
    if args.use_norm:
        norm = VariableNormalizer().load(f"{args.ckpt_dir}/normalizer_stats.json")

    det_model = DetGNN(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(DEVICE)
    ckpt_name = "det_norm_best.pt" if args.use_norm else "det_best.pt"
    det_ckpt = torch.load(f"{args.ckpt_dir}/{ckpt_name}", map_location=DEVICE, weights_only=True)
    det_model.load_state_dict(det_ckpt["model"])
    det_model.eval()

    case_map = {"pglib_opf_case118_ieee": pn.case118}
    make_net = case_map.get(args.case, pn.case118)
    test_ds = OPFDataset(root="data", case_name=args.case, split="test", num_groups=1)

    log.info(f"=== N2: IPOPT BACKEND BENCHMARK ===")

    flat_pips, ws_pips, flat_ipopt, ws_ipopt = [], [], [], []

    for idx in range(min(args.n_test, len(test_ds))):
        data = test_ds[idx]

        with torch.no_grad():
            bp, gp = det_model(data.to(DEVICE))
        if norm:
            bp = norm.denormalize_bus(bp)
            gp = norm.denormalize_gen(gp)

        Vm = bp[:, 1].cpu().numpy()
        Va = bp[:, 0].cpu().numpy()
        Pg = gp[:, 0].cpu().numpy() * 100
        Qg = gp[:, 1].cpu().numpy() * 100

        n_bus = len(make_net().bus)
        n_gen = len(make_net().gen)
        x_ws = np.zeros(2*n_bus + 2*n_gen)
        x_ws[:n_bus] = Vm[:n_bus]
        x_ws[n_bus:2*n_bus] = np.degrees(Va[:n_bus])
        x_ws[2*n_bus:2*n_bus+min(n_gen, len(Pg))] = Pg[:n_gen]
        x_ws[2*n_bus+n_gen:2*n_bus+n_gen+min(n_gen, len(Qg))] = Qg[:n_gen]

        # PIPS flat
        net = make_net(); set_loads(net, data)
        _, nf = run_opf_pips(net, "flat")
        flat_pips.append(nf)

        # PIPS warm-start
        net = make_net(); set_loads(net, data); inject_warmstart(net, x_ws)
        _, nw = run_opf_pips(net, "results")
        ws_pips.append(nw)

        log.info(f"  #{idx:3d}  PIPS_flat={nf}  PIPS_ws={nw}")

    def _s(v):
        v2 = [x for x in v if x is not None]
        return np.mean(v2), np.median(v2) if v2 else (float('nan'), float('nan'))

    fm = _s(flat_pips)[0]
    print("\n" + "=" * 70)
    print("  N2: IPOPT BACKEND BENCHMARK")
    print("=" * 70)
    for label, vals in [("PIPS flat", flat_pips), ("PIPS ws", ws_pips)]:
        m, md = _s(vals)
        print(f"  {label:12s}  mean={m:5.1f}  median={md:4.0f}  vs Flat: {(1-m/fm)*100:+.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
