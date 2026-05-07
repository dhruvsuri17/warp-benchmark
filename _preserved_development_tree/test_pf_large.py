"""Test PF on large cases. Since pandapower doesn't have case500/case2000
built-in, we use pandapower's converter to build nets from pypower case files,
or we test with pandapower's built-in large cases (case1354pegase, case2869pegase)."""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings('ignore')
import numpy as np
import pandapower as pp
import pandapower.networks as pn

# Test with pandapower's built-in large cases
cases = [
    ("case118", pn.case118),
    ("case1354pegase", pn.case1354pegase),
    ("case2869pegase", pn.case2869pegase),
]

for name, fn in cases:
    print(f"\n=== {name} ===", flush=True)
    net = fn()
    print(f"  {len(net.bus)} buses, {len(net.gen)} gen, {len(net.line)} lines", flush=True)

    # Flat start
    pp.runpp(net, init="flat", numba=False, max_iteration=100)
    n_flat = net._ppc["iterations"]
    Vm_sol = net.res_bus["vm_pu"].values.copy()
    Va_sol = net.res_bus["va_degree"].values.copy()
    print(f"  Flat start: {n_flat} NR iterations", flush=True)

    # DC init
    net2 = fn()
    pp.runpp(net2, init="dc", numba=False, max_iteration=100)
    n_dc = net2._ppc["iterations"]
    print(f"  DC init: {n_dc} NR iterations", flush=True)

    # Oracle warm-start (from converged solution)
    net3 = fn()
    net3.res_bus["vm_pu"] = Vm_sol
    net3.res_bus["va_degree"] = Va_sol
    net3.res_bus["p_mw"] = 0.0
    net3.res_bus["q_mvar"] = 0.0
    pp.runpp(net3, init="results", numba=False, max_iteration=100)
    n_oracle = net3._ppc["iterations"]
    print(f"  Oracle (GT): {n_oracle} NR iterations", flush=True)

    # Perturbed warm-start (add noise to simulate ML prediction)
    for noise_std in [0.001, 0.005, 0.01, 0.02]:
        net4 = fn()
        net4.res_bus["vm_pu"] = Vm_sol + np.random.randn(len(Vm_sol)) * noise_std
        net4.res_bus["va_degree"] = Va_sol + np.random.randn(len(Va_sol)) * noise_std * 10
        net4.res_bus["p_mw"] = 0.0
        net4.res_bus["q_mvar"] = 0.0
        try:
            pp.runpp(net4, init="results", numba=False, max_iteration=100)
            n_pert = net4._ppc["iterations"]
            print(f"  Perturbed (σ={noise_std}): {n_pert} NR iterations", flush=True)
        except Exception:
            print(f"  Perturbed (σ={noise_std}): DIVERGED", flush=True)

    print(f"  Vm solution range: [{Vm_sol.min():.4f}, {Vm_sol.max():.4f}]", flush=True)
    print(f"  Va solution range: [{Va_sol.min():.1f}°, {Va_sol.max():.1f}°]", flush=True)
