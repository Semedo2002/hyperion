"""Benchmark against open experimental data (aegis/experimental_suite.py).

Runs copper plate impacts with the solver (tabulated EOS from tools/generate_sesame.py) at a set of particle
velocities and compares the shock velocities with the 145 Marsh (1980) LASL Hugoniot shots bundled with the code;
writes results/shot_benchmark/shot_benchmark_comparison.png and summary.json (RMS, reduced chi^2 for an assumed
measurement error, the data's own linear fit).

Your own shot traces (VISAR, B-dot, radiography; CSV columns t, value, sigma) can be compared with an FCG or MagLIF
run through aegis/validation.py: see examples/experimental_shot_fitting.py.

    python examples/shot_benchmark.py                      # 10 impacts, Up 0.2-3 km/s
    python examples/shot_benchmark.py --n-up 6 --cells 400
"""
import os
import sys
import json
import argparse
import contextlib
import io

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    from aegis import experimental_suite as ex


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="copper_hugoniot", choices=list(ex.DATASETS))
    ap.add_argument("--n-up", dest="n_up", type=int, default=10)
    ap.add_argument("--up-max", dest="up_max", type=float, default=3.0)
    ap.add_argument("--cells", type=int, default=480)
    ap.add_argument("--sigma-rel", dest="sigma_rel", type=float, default=0.02,
                    help="assumed relative error of the measured Us (the data carry none)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "shot_benchmark"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = ex.hugoniot_benchmark(a.dataset, ups=np.linspace(0.2, a.up_max, a.n_up), sigma_rel=a.sigma_rel,
                              up_max=a.up_max, n=a.cells, out_png=os.path.join(a.out, "shot_benchmark_comparison.png"))
    json.dump(r, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(f"{a.dataset}: {r['n_points']} shots, RMS {r['rms_km_s']:.3f} km/s ({100 * r['rms_rel']:.1f}%), chi2_nu "
          f"{r['chi2_nu']:.2f} (sigma {100 * a.sigma_rel:.0f}% assumed); simulated vs table Hugoniot "
          f"{100 * r['sim_vs_table_max_rel']:.2f}%; data fit Us = {r['data_linear_fit']['c0']:.3f} + "
          f"{r['data_linear_fit']['s']:.3f} Up")
    print(f"results in {a.out}")


if __name__ == "__main__":
    main()
