"""Grid convergence of the 2D lock boundary near 10% field contrast.

    python experiments/convergence_lock.py --res 1 2      # 400x100 and 800x200 (~1.5 h)
    python experiments/convergence_lock.py --res 4        # 1600x400, boundary cases only (~7 h)
    python experiments/convergence_lock.py --summary      # table from saved results
    python experiments/convergence_lock.py --res 4 --cases "uniform Bx=3" --merge   # add a case to saved results

Cases straddle the boundary found in sweeps C/D at the base grid: uniform B = 3 (locked reference),
B0 = 3 and 3.5 with eps = 0.9 (locked, 10%), B0 = 4 with eps = 0.9 (just failed, 10%), B0 = 3.5 with
eps = 0.8 (failed, 20%). Each resolution has its own hydrodynamic reference. Lock criterion as in the
paper: late mixing-layer enstrophy < 0.2 x hydro and seeded-mode growth a4(t_end)/a4(0.05) < 0.5 x hydro;
the all-mode roughness ratio is reported as well. Results: experiments/data/exp_conv{res}.npz.
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runner import run_sweep, tail, DATA_DIR

CASES = {
    "hydro": {"B_transverse": 0.0, "B_field_type": "uniform"},
    "uniform Bx=3": {"B_transverse": 3.0, "B_gap_fraction": 0.99},
    "B0=3 eps=0.9": {"B_transverse": 3.0, "B_gap_fraction": 0.9},
    "B0=3.5 eps=0.9": {"B_transverse": 3.5, "B_gap_fraction": 0.9},
    "B0=4 eps=0.9": {"B_transverse": 4.0, "B_gap_fraction": 0.9},
    "B0=3.5 eps=0.8": {"B_transverse": 3.5, "B_gap_fraction": 0.8},
}
# the 4x grid costs ~1.7 h per run: keep the reference and the three cases nearest the boundary
CASES_4X = ("hydro", "B0=3.5 eps=0.9", "B0=4 eps=0.9", "B0=3.5 eps=0.8")


def contrast(name):
    if name.startswith("uniform"):
        b, e = float(name.split("=")[1]), 0.99
    else:
        b, e = (float(v.split("=")[1]) for v in name.split())
    return 0.5 * b**2 * (1 - e**2)


def load(res):
    path = os.path.join(DATA_DIR, f"exp_conv{res}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    names = sorted({k.split("__")[0] for k in d.files})
    return {n: {k.split("__")[1]: d[k] for k in d.files if k.startswith(n + "__")} for n in names}


def ratios(S):
    h = S["hydro"]
    g = lambda d: d["a"][-1] / d["a"][np.argmin(np.abs(d["t"] - 0.05))]
    out = {}
    for n, d in S.items():
        if n == "hydro":
            continue
        e = tail(d["emix"], d["t"]) / tail(h["emix"], h["t"])
        gr = g(d) / g(h)
        rr = d["rms"][-1] / h["rms"][-1]
        out[n] = (e, gr, rr, e < 0.2 and gr < 0.5)
    return out


def export_csv(path):
    """All cases and resolutions: absolute and hydro-relative enstrophy, roughness, a4, a8."""
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["resolution", "nx", "ny", "case", "dP_B", "emix", "emix_over_hydro", "roughness",
                    "roughness_over_hydro", "a4", "a8", "a4_growth_over_hydro"])
        for r in (1, 2, 4):
            S = load(r)
            if S is None:
                continue
            h = S["hydro"]
            eh, rh = tail(h["emix"], h["t"]), h["rms"][-1]
            gh = h["a"][-1] / h["a"][np.argmin(np.abs(h["t"] - 0.05))]
            for n, d in S.items():
                e = tail(d["emix"], d["t"])
                g = d["a"][-1] / d["a"][np.argmin(np.abs(d["t"] - 0.05))]
                w.writerow([f"{r}x", 400 * r, 100 * r, n, "" if n == "hydro" else f"{contrast(n):.4f}",
                            f"{e:.6g}", f"{e / eh:.6g}", f"{d['rms'][-1]:.6g}", f"{d['rms'][-1] / rh:.6g}",
                            f"{d['m4'][-1]:.6g}", f"{d['m8'][-1]:.6g}", f"{g / gh:.6g}"])


def summary():
    data = {r: load(r) for r in (1, 2, 4)}
    data = {r: ratios(S) for r, S in data.items() if S is not None}
    if not data:
        print("no results yet"); return
    print(f"{'case':<16}{'dP_B':>7}" + "".join(f"{f'emix/h x{r}':>12}{f'growth x{r}':>11}{f'lock x{r}':>8}"
                                            for r in data))
    for n in CASES:
        if n == "hydro":
            continue
        row = f"{n:<16}{contrast(n):7.2f}"
        for r, R in data.items():
            if n in R:
                e, gr, rr, lk = R[n]
                row += f"{e:12.3f}{gr:11.3f}{('yes' if lk else 'no'):>8}"
            else:
                row += f"{'-':>12}{'-':>11}{'-':>8}"
        print(row)
    # boundary estimate: dP_B where emix/h crosses 0.2 (log-linear interpolation between neighbours)
    for r, R in data.items():
        pts = sorted((contrast(n), R[n][0]) for n in R if not n.startswith("uniform"))
        cross = None
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if y0 < 0.2 <= y1:
                cross = x0 + (np.log(0.2) - np.log(y0)) / (np.log(y1) - np.log(y0)) * (x1 - x0)
        print(f"x{r}: enstrophy threshold crossed at dP_B ~ {cross:.2f}" if cross else
              f"x{r}: threshold not bracketed by these cases")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs="*", default=[])
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--cases", nargs="*", help="run only these cases")
    ap.add_argument("--merge", action="store_true",
                    help="add the results to the existing exp_conv<res>.npz instead of replacing it")
    args = ap.parse_args()
    for r in args.res:
        names = args.cases or (CASES_4X if r >= 4 else tuple(CASES))
        grid = {"nx": 400 * r, "ny": 100 * r}
        tag = f"conv{r}"
        old = os.path.join(DATA_DIR, f"exp_{tag}.npz")
        keep = dict(np.load(old)) if args.merge and os.path.exists(old) else {}
        run_sweep(f"{tag}_new" if keep else tag, {"B_field_type": "striped", **grid}, {n: CASES[n] for n in names})
        if keep:
            new_path = os.path.join(DATA_DIR, f"exp_{tag}_new.npz")
            new = dict(np.load(new_path))
            np.savez(old, **{**keep, **new})
            os.remove(new_path)
            print(f"merged {sorted({k.split('__')[0] for k in new})} into {old}")
    summary()
    out_csv = os.path.join(os.path.dirname(DATA_DIR), "..", "results", "convergence", "convergence_summary.csv")
    export_csv(os.path.normpath(out_csv))
    print(f"wrote {os.path.normpath(out_csv)}")
