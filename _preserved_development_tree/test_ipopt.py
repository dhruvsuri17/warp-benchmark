import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings('ignore')
import numpy as np
from numpy import inf
import pandapower.networks as pn
from eval.ipopt_opf_v2 import build_om, solve_opf

net = pn.case118()
om, ppopt = build_om(net)
x0, xmin, xmax = om.getv()
ll, uu = xmin.copy(), xmax.copy()
ll[xmin == -inf] = -1e10; uu[xmax == inf] = 1e10
x0_mid = (ll + uu) / 2.0

print("Test 1: IPOPT flat", flush=True)
r1 = solve_opf(om, ppopt, x0=x0_mid, warm_start=False, print_level=0, max_iter=200)
print(f"  Status={r1['status']}, Iters={r1['n_iters']}, Conv={r1['converged']}, Obj={r1['obj']:.1f}", flush=True)

if r1['converged']:
    print("Test 2: primal-only warm-start from optimal", flush=True)
    r2 = solve_opf(om, ppopt, x0=r1['x'], warm_start=True, mu_init=1e-4, print_level=0)
    print(f"  Status={r2['status']}, Iters={r2['n_iters']}, Conv={r2['converged']}", flush=True)

    print("Test 3: primal+dual warm-start from optimal", flush=True)
    r3 = solve_opf(om, ppopt, x0=r1['x'], lam_g0=r1['lam_g'], zl0=r1['zl'], zu0=r1['zu'],
                   warm_start=True, mu_init=1e-6, print_level=0)
    print(f"  Status={r3['status']}, Iters={r3['n_iters']}, Conv={r3['converged']}", flush=True)
