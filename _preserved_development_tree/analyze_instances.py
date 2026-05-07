"""Instance-level analysis: which load scenarios does the model help vs hurt?

Reads the CSV produced by benchmark_v2.py and prints diagnostics.
"""
import os, sys, csv
import numpy as np

def analyze(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        for k in r:
            if k != "idx":
                try:
                    r[k] = float(r[k]) if r[k] else None
                except ValueError:
                    r[k] = None

    print(f"\n{'='*80}")
    print(f"  Instance Analysis — {len(rows)} test instances")
    print(f"{'='*80}\n")

    # DetGNN projected: helped vs hurt
    helped = [r for r in rows if r["det_proj"] is not None and r["flat"] is not None
              and r["det_proj"] < r["flat"]]
    hurt = [r for r in rows if r["det_proj"] is not None and r["flat"] is not None
            and r["det_proj"] > r["flat"]]
    same = [r for r in rows if r["det_proj"] is not None and r["flat"] is not None
            and r["det_proj"] == r["flat"]]

    print(f"DetGNN (projected):")
    print(f"  Helped: {len(helped)}/{len(rows)} instances")
    print(f"  Hurt:   {len(hurt)}/{len(rows)} instances")
    print(f"  Same:   {len(same)}/{len(rows)} instances")

    if helped:
        deltas_h = [r["flat"] - r["det_proj"] for r in helped]
        rmses_h = [r["det_bus_rmse"] for r in helped if r["det_bus_rmse"] is not None]
        print(f"  Helped — avg iter reduction: {np.mean(deltas_h):.1f}, "
              f"avg bus RMSE: {np.mean(rmses_h):.4f}")
    if hurt:
        deltas_u = [r["det_proj"] - r["flat"] for r in hurt]
        rmses_u = [r["det_bus_rmse"] for r in hurt if r["det_bus_rmse"] is not None]
        print(f"  Hurt   — avg iter increase: {np.mean(deltas_u):.1f}, "
              f"avg bus RMSE: {np.mean(rmses_u):.4f}")

    # WARP projected: helped vs hurt
    helped_w = [r for r in rows if r["warp_proj"] is not None and r["flat"] is not None
                and r["warp_proj"] < r["flat"]]
    hurt_w = [r for r in rows if r["warp_proj"] is not None and r["flat"] is not None
              and r["warp_proj"] > r["flat"]]

    print(f"\nWARP-K3 (projected):")
    print(f"  Helped: {len(helped_w)}/{len(rows)} instances")
    print(f"  Hurt:   {len(hurt_w)}/{len(rows)} instances")

    if helped_w:
        deltas_hw = [r["flat"] - r["warp_proj"] for r in helped_w]
        print(f"  Helped — avg iter reduction: {np.mean(deltas_hw):.1f}")
    if hurt_w:
        deltas_uw = [r["warp_proj"] - r["flat"] for r in hurt_w]
        print(f"  Hurt   — avg iter increase: {np.mean(deltas_uw):.1f}")

    # Projection impact
    print(f"\nProjection impact (raw → projected):")
    det_raw = [r["det_raw"] for r in rows if r["det_raw"] is not None]
    det_proj = [r["det_proj"] for r in rows if r["det_proj"] is not None]
    warp_raw = [r["warp_raw"] for r in rows if r["warp_raw"] is not None]
    warp_proj = [r["warp_proj"] for r in rows if r["warp_proj"] is not None]

    if det_raw and det_proj:
        print(f"  DetGNN:  raw mean={np.mean(det_raw):.1f} → proj mean={np.mean(det_proj):.1f} "
              f"(Δ={np.mean(det_proj)-np.mean(det_raw):+.1f})")
    if warp_raw and warp_proj:
        print(f"  WARP:    raw mean={np.mean(warp_raw):.1f} → proj mean={np.mean(warp_proj):.1f} "
              f"(Δ={np.mean(warp_proj)-np.mean(warp_raw):+.1f})")

    # RMSE threshold analysis
    print(f"\nRMSE threshold analysis (DetGNN projected):")
    for thresh in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        below = [r for r in rows if r["det_bus_rmse"] is not None
                 and r["det_bus_rmse"] < thresh and r["det_proj"] is not None]
        if below:
            avg_delta = np.mean([r["det_proj"] - r["flat"] for r in below])
            n_helped = sum(1 for r in below if r["det_proj"] < r["flat"])
            print(f"  RMSE < {thresh:.2f}: {len(below)} instances, "
                  f"mean Δiters={avg_delta:+.1f}, helped={n_helped}/{len(below)}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="logs/benchmark_v2_pglib_opf_case118_ieee.csv")
    args = parser.parse_args()
    analyze(args.csv)
