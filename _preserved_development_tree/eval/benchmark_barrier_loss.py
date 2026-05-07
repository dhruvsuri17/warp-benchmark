"""Direction 2: Barrier-function-aware warm-start.

Instead of training the model to predict x* (optimal), post-process the
prediction to maximize the log-barrier centrality measure. This shifts
the prediction toward the interior while preserving its directional accuracy.

The key insight from N2: the 2-iteration penalty comes from Vm being pushed
toward bounds. If we retract Vm predictions toward the midpoint by exactly
the amount needed to restore centrality, we might eliminate the penalty.

This is a simpler version of Yildirim-Wright's "warm-start along the
central path" idea — no retraining, just inference-time post-processing.
"""
import os, sys, argparse, time, io, logging, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("D2-BARRIER")

from colab_warp import DetGNN, DEVICE
from normalizer import VariableNormalizer
from inference.warmstart import get_variable_bounds, inject_warmstart
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


def run_opf(net, init="flat"):
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


def set_loads(net, data):
    Pd = data["load"].x[:, 0].cpu().numpy() * 100
    Qd = data["load"].x[:, 1].cpu().numpy() * 100
    for i in range(min(len(net.load), len(Pd))):
        net.load.at[i, "p_mw"] = Pd[i]
        net.load.at[i, "q_mvar"] = Qd[i]


def barrier_retract(x_hat, net, target_mu_ratio=0.5):
    """Retract prediction toward midpoint until log-barrier centrality is acceptable.

    For each variable, compute the barrier product (x-lb)(ub-x).
    If it's below target_mu_ratio * midpoint_product, blend toward midpoint.
    This selectively retracts only the variables near their bounds.
    """
    lb, ub = get_variable_bounds(net)
    x_mid = (lb + ub) / 2.0
    x_out = x_hat.copy()

    for i in range(len(x_hat)):
        rng = ub[i] - lb[i]
        if rng < 1e-10:
            continue

        mid_prod = (x_mid[i] - lb[i]) * (ub[i] - x_mid[i])
        hat_prod = max((x_hat[i] - lb[i]) * (ub[i] - x_hat[i]), 1e-12)
        ratio = hat_prod / mid_prod if mid_prod > 0 else 0

        if ratio < target_mu_ratio:
            # Blend toward midpoint enough to restore target centrality
            # Solve: (alpha*x_hat + (1-alpha)*x_mid - lb)(ub - alpha*x_hat - (1-alpha)*x_mid) >= target * mid_prod
            # Binary search for alpha
            lo_a, hi_a = 0.0, 1.0
            for _ in range(20):
                alpha = (lo_a + hi_a) / 2
                x_try = alpha * x_hat[i] + (1 - alpha) * x_mid[i]
                prod = (x_try - lb[i]) * (ub[i] - x_try)
                if prod >= target_mu_ratio * mid_prod:
                    lo_a = alpha
                else:
                    hi_a = alpha
            x_out[i] = lo_a * x_hat[i] + (1 - lo_a) * x_mid[i]

    x_out = np.clip(x_out, lb + 1e-8 * (ub - lb), ub - 1e-8 * (ub - lb))
    return x_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pglib_opf_case118_ieee")
    parser.add_argument("--ckpt-dir", default="ckpt")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--use-norm", action="store_true", default=True)
    parser.add_argument("--mu-ratios", nargs="+", type=float,
                        default=[0.1, 0.3, 0.5, 0.7, 0.9])
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

    all_results = {mu: [] for mu in args.mu_ratios}
    flat_results = []

    log.info(f"=== DIRECTION 2: BARRIER-RETRACT SWEEP ===")
    log.info(f"µ ratios: {args.mu_ratios}")

    for idx in range(min(args.n_test, len(test_ds))):
        data = test_ds[idx]

        with torch.no_grad():
            bp, gp = det_model(data.to(DEVICE))
        if norm:
            bp = norm.denormalize_bus(bp)
            gp = norm.denormalize_gen(gp)

        Vm = bp[:, 1].cpu().numpy()
        Va_deg = np.degrees(bp[:, 0].cpu().numpy())
        Pg = gp[:, 0].cpu().numpy() * 100
        Qg = gp[:, 1].cpu().numpy() * 100

        n_bus = len(make_net().bus)
        n_gen = len(make_net().gen)
        lb, ub = get_variable_bounds(make_net())
        x_mid = (lb + ub) / 2.0

        x_hat = x_mid.copy()
        x_hat[:n_bus] = Vm[:n_bus]
        x_hat[n_bus:2*n_bus] = Va_deg[:n_bus]
        x_hat[2*n_bus:2*n_bus+min(n_gen, len(Pg))] = Pg[:n_gen]
        x_hat[2*n_bus+n_gen:2*n_bus+n_gen+min(n_gen, len(Qg))] = Qg[:n_gen]

        net = make_net(); set_loads(net, data)
        _, nf = run_opf(net, "flat")
        flat_results.append(nf)

        row_str = f"  #{idx:3d}  flat={nf}"

        for mu_r in args.mu_ratios:
            x_retracted = barrier_retract(x_hat, make_net(), target_mu_ratio=mu_r)
            net = make_net(); set_loads(net, data)
            inject_warmstart(net, x_retracted)
            _, ni = run_opf(net, "results")
            all_results[mu_r].append(ni)
            row_str += f"  µ={mu_r}:{ni}"

        log.info(row_str)

    fm = np.mean([x for x in flat_results if x is not None])
    print("\n" + "=" * 80)
    print("  DIRECTION 2: BARRIER-RETRACT RESULTS")
    print("=" * 80)
    print(f"  {'µ target':>10}  {'Mean':>6}  {'Med':>5}  {'vs Flat':>8}")
    print("-" * 40)
    print(f"  {'flat':>10}  {fm:6.1f}  {np.median([x for x in flat_results if x]):5.0f}  {0:+7.1f}%")
    for mu_r in args.mu_ratios:
        vals = [x for x in all_results[mu_r] if x is not None]
        m = np.mean(vals); md = np.median(vals)
        delta = (1 - m/fm) * 100
        print(f"  {mu_r:10.2f}  {m:6.1f}  {md:5.0f}  {delta:+7.1f}%")
    print("=" * 80)

    best_mu = min(args.mu_ratios, key=lambda r: np.mean([x for x in all_results[r] if x]))
    best_mean = np.mean([x for x in all_results[best_mu] if x])
    if best_mean < fm:
        print(f"\n  ✓ BARRIER RETRACT WORKS! Best µ={best_mu}, {(1-best_mean/fm)*100:+.1f}% improvement")
    else:
        print(f"\n  ✗ Barrier retract did not beat flat start. Best µ={best_mu}, mean={best_mean:.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
