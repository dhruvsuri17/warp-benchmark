"""Test the IPM-LSTM-style OPF IPOPT solver."""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings('ignore')
import numpy as np
from numpy import inf
import pandapower.networks as pn
from eval.opf_ipopt import build_om, solve_opf

net = pn.case118()
om, ppopt = build_om(net)
x0, xmin, xmax = om.getv()
ll, uu = xmin.copy(), xmax.copy()
ll[xmin == -inf] = -1e10; uu[xmax == inf] = 1e10
x_mid = (ll + uu) / 2.0

print("Test 1: IPOPT cold start (midpoint)", flush=True)
r1 = solve_opf(om, ppopt, x0=x_mid, warm_start=False)
print(f"  iters={r1['n_iters']}, conv={r1['converged']}, "
      f"obj={r1['obj']:.1f}, mu_final={r1['mu_final']:.2e}", flush=True)

if r1["converged"]:
    print("\nTest 2: warm-start primal only (from optimal, no duals)", flush=True)
    r2 = solve_opf(om, ppopt, x0=r1["x"], warm_start=True)
    print(f"  iters={r2['n_iters']}, conv={r2['converged']}", flush=True)

    print("\nTest 3: warm-start primal + duals (no mu)", flush=True)
    r3 = solve_opf(om, ppopt, x0=r1["x"],
                   lam_g0=r1["lam_g"], zl0=r1["zl"], zu0=r1["zu"],
                   warm_start=True)
    print(f"  iters={r3['n_iters']}, conv={r3['converged']}", flush=True)

    print("\nTest 4: warm-start primal + duals + mu (full IPM-LSTM style)", flush=True)
    r4 = solve_opf(om, ppopt, x0=r1["x"],
                   lam_g0=r1["lam_g"], zl0=r1["zl"], zu0=r1["zu"],
                   warm_start=True, mu_init=r1["mu_final"])
    print(f"  iters={r4['n_iters']}, conv={r4['converged']}", flush=True)

    print("\nTest 5: warm-start with slightly higher mu (1e-4)", flush=True)
    r5 = solve_opf(om, ppopt, x0=r1["x"],
                   lam_g0=r1["lam_g"], zl0=r1["zl"], zu0=r1["zu"],
                   warm_start=True, mu_init=1e-4)
    print(f"  iters={r5['n_iters']}, conv={r5['converged']}", flush=True)

    print(f"\nSummary:", flush=True)
    print(f"  Cold (midpoint):       {r1['n_iters']} iters", flush=True)
    print(f"  Primal only:           {r2['n_iters']} iters", flush=True)
    print(f"  Primal + duals:        {r3['n_iters']} iters", flush=True)
    print(f"  Primal + duals + mu*:  {r4['n_iters']} iters", flush=True)
    print(f"  Primal + duals + 1e-4: {r5['n_iters']} iters", flush=True)
