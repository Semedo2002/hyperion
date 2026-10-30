"""Shot calibration: fit a pulsed-power liner simulation (examples/fcg_flux_compression.py with the RLC drive) to
diagnostic traces, VISAR free-surface velocity, a B-dot probe and radiography radii, by scanning one circuit
parameter and minimising chi^2 (aegis/validation.py).

Without data files the script makes a synthetic shot first, so that the whole chain can be run and checked:
the "experiment" is a simulation at a hidden parameter value (16.2 nH series inductance), sampled like the
instruments, with Gaussian noise and timing-fiducial offsets, and saved as CSV under synthetic_shot/ (clearly
labelled synthetic). With --visar / --bdot / --radiography the traces come from your files instead (CSV:
time in us, value, sigma).

    python examples/experimental_shot_fitting.py                             # synthetic shot, 5-point scan
    python examples/experimental_shot_fitting.py --visar v.csv --bdot b.csv --radiography r.csv \\
        --param inductance_nh --range 10 25 --n 7 --voltage-kv 80 --capacitance-uf 200
    python examples/experimental_shot_fitting.py --plot-only

Outputs (results/shot_fitting): shot_calibration_fit.png (data, best fit, ensemble band, residuals),
chi2_scan.png, calibration.json (metrics per diagnostic, calibrated value and its 1-sigma range).
"""
import os
import sys
import csv
import json
import argparse
import contextlib
import io

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))
with contextlib.redirect_stdout(io.StringIO()):
    import fcg_flux_compression as fcg
    from aegis import validation as val

PARAMS = {"inductance_nh": "series inductance (nH)", "voltage_kv": "charge voltage (kV)",
          "resistance_mohm": "series resistance (mOhm)", "capacitance_uf": "capacitance (uF)"}


def simulate(a, value, out):
    """one FCG run with the scanned parameter at value; returns the observables"""
    ns = argparse.Namespace(liner_material="al", b0_tesla=a.b0_tesla, v0_kms=0.0, enable_strength=False,
                            eta_model="ideal", sigma_gas=None, sigma_min=5e6, r0=a.r0, dr=a.dr, r_max=a.r_max,
                            lz=a.lz, mode=1, amp=0.0, rho_gas=0.1, p_gas_bar=0.1, nr=a.nr, nz=a.nz, t_end=a.t_end,
                            frame_dt=100.0, diag_every=2, capacitance_uf=a.capacitance_uf, voltage_kv=a.voltage_kv,
                            inductance_nh=a.inductance_nh, resistance_mohm=a.resistance_mohm, r_return_mm=a.r_max)
    setattr(ns, a.param, value)
    with contextlib.redirect_stdout(io.StringIO()):
        summ = fcg.run(ns, out)
    rows = list(csv.DictReader(open(os.path.join(out, "diagnostics.csv"))))
    d = {k: np.array([float(r[k]) for r in rows]) for k in ("t_us", "I_MA", "r_out_mm", "R_cavity_mm", "v_surface_kms")}
    obs = val.observables_from_fcg(d, a.probe_r_mm, a.probe_area_mm2, a.fwhm_ns * 1e-3)
    return obs, summ


def synthetic_shot(a, out):
    """a hidden-parameter run sampled like the instruments, with noise and timing offsets"""
    obs, summ = simulate(a, a.truth, os.path.join(out, "runs", "truth"))
    rng = np.random.default_rng(a.seed)
    t_end = summ["t_turnaround_us"] - 0.1
    dt_v, dt_b = rng.normal(0, a.timing_ns * 1e-3, 2)                 # fiducial errors of the two timed channels
    spec = {"visar": (np.arange(1.0, t_end, 0.02), lambda y: 0.05 + 0.02 * np.abs(y), dt_v),
            "bdot": (np.arange(0.2, t_end, 0.01), lambda y: 0.03 * np.max(np.abs(y)) + 0 * y, dt_b),
            "radiography": (np.linspace(1.6, t_end, 6), lambda y: 0.08 + 0 * y, 0.0)}
    traces = []
    os.makedirs(os.path.join(out, "synthetic_shot"), exist_ok=True)
    for kind, (t, sig, offset) in spec.items():
        ts, ys = obs[kind]
        y = np.interp(t - offset, ts, ys)            # the instrument's clock is off by `offset`
        s = sig(y)
        tr = val.Trace(kind, t, y + rng.normal(0, s), s, val.UNITS[kind], kind,
                       meta={"synthetic": True, "hidden": {a.param: a.truth}, "timing_offset_us": float(offset)})
        val.save_trace(os.path.join(out, "synthetic_shot", f"{kind}.csv"), tr)
        traces.append(tr)
    return traces


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--visar"); ap.add_argument("--bdot"); ap.add_argument("--radiography")
    ap.add_argument("--sigma", default="3%", help="uncertainty for files without a sigma column")
    ap.add_argument("--param", choices=list(PARAMS), default="inductance_nh")
    ap.add_argument("--range", type=float, nargs=2, default=[12.0, 22.0])
    ap.add_argument("--n", type=int, default=5, help="coarse scan points")
    ap.add_argument("--refine", type=int, default=12, help="golden-section runs after the coarse scan")
    ap.add_argument("--tol", type=float, default=0.01, help="stop when the bracket is narrower than this")
    ap.add_argument("--truth", type=float, default=16.2, help="hidden value for the synthetic shot")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-shift", dest="no_shift", action="store_true", help="do not fit timing offsets")
    ap.add_argument("--timing-ns", dest="timing_ns", type=float, default=5.0,
                    help="timing-fiducial uncertainty of VISAR and B-dot (prior on their fitted shifts), ns")
    ap.add_argument("--capacitance-uf", dest="capacitance_uf", type=float, default=200.0)
    ap.add_argument("--voltage-kv", dest="voltage_kv", type=float, default=80.0)
    ap.add_argument("--inductance-nh", dest="inductance_nh", type=float, default=15.0)
    ap.add_argument("--resistance-mohm", dest="resistance_mohm", type=float, default=1.0)
    ap.add_argument("--probe-r-mm", dest="probe_r_mm", type=float, default=12.5, help="B-dot probe radius (return gap)")
    ap.add_argument("--probe-area-mm2", dest="probe_area_mm2", type=float, default=1.0)
    ap.add_argument("--fwhm-ns", dest="fwhm_ns", type=float, default=10.0, help="instrument response FWHM, ns")
    ap.add_argument("--b0-tesla", dest="b0_tesla", type=float, default=5.0)
    ap.add_argument("--r0", type=float, default=10.0); ap.add_argument("--dr", type=float, default=1.0)
    ap.add_argument("--r-max", dest="r_max", type=float, default=14.0); ap.add_argument("--lz", type=float, default=10.0)
    ap.add_argument("--nr", type=int, default=64); ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--t-end", dest="t_end", type=float, default=6.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "shot_fitting"))
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)
    if a.plot_only:
        return plot_saved(out)

    files = {"visar": a.visar, "bdot": a.bdot, "radiography": a.radiography}
    if any(files.values()):
        traces = [val.load_trace(p, k, k, val.UNITS[k], a.sigma) for k, p in files.items() if p]
        synthetic = False
    else:
        print(f"synthetic shot: hidden {PARAMS[a.param]} = {a.truth:g}", flush=True)
        traces = synthetic_shot(a, out)
        synthetic = True

    values = list(np.linspace(*a.range, a.n))
    runs = {}

    def run(v):
        if v not in runs:
            obs, summ = simulate(a, v, os.path.join(out, "runs", f"{a.param}_{v:.6g}"))
            runs[v] = (obs, summ["t_turnaround_us"])
            print(f"  {a.param} = {v:.5g}: turnaround {summ['t_turnaround_us']:.3f} us", flush=True)
        return runs[v]
    for v in values:
        run(v)
    # fit window: the data up to just before the earliest simulated turnaround, so every run covers it
    t_cut = min(r[1] for r in runs.values()) - 0.05
    traces = [val.Trace(tr.name, tr.t[tr.t <= t_cut], tr.y[tr.t <= t_cut], tr.sigma[tr.t <= t_cut], tr.units, tr.kind,
                        tr.meta) for tr in traces]
    shifted = lambda tr: (not a.no_shift) and tr.kind in ("visar", "bdot")    # traces with a timing fiducial

    def chi2_of(v):
        obs = run(v)[0]
        return sum(val.compare(tr, *obs[tr.kind], fit_shift=shifted(tr), max_shift=0.03,
                               shift_sigma=a.timing_ns * 1e-3)["chi2_total"] for tr in traces)
    # coarse scan, then golden-section search inside the bracket of the best scan point (chi^2 is sharply
    # peaked and not quadratic far from its minimum), then a least-squares parabola over the whole valley
    chi = {v: chi2_of(v) for v in values}
    k = int(np.argmin([chi[v] for v in values]))
    lo, hi = values[max(k - 1, 0)], values[min(k + 1, len(values) - 1)]
    g = (np.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = chi2_of(x1), chi2_of(x2)
    chi[x1], chi[x2] = f1, f2
    for _ in range(a.refine):
        if hi - lo < a.tol:
            break
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo); f1 = chi[x1] = chi2_of(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo); f2 = chi[x2] = chi2_of(x2)
    pts = sorted(chi)
    values, chi2 = pts, [chi[p] for p in pts]
    n_data = sum(len(tr.t) for tr in traces)
    best, sig, cmin, scale = val.calibrate_valley(values, chi2, n_data - 1)
    print(f"calibrated {a.param} = {best:.3f} +- {sig:.3f} (chi^2 min {cmin:.1f})", flush=True)
    ens = {tr.kind: [] for tr in traces}
    best_obs = None
    for v in (best - sig, best, best + sig):
        obs = run(v)[0]
        for k in ens:
            ens[k].append(obs[k])
        if v == best:
            best_obs = obs
    metrics = {tr.kind: val.compare(tr, *best_obs[tr.kind], n_params=1, fit_shift=shifted(tr), max_shift=0.03,
                                    shift_sigma=a.timing_ns * 1e-3) for tr in traces}
    total = sum(m["chi2"] for m in metrics.values())
    dof = sum(m["n"] for m in metrics.values()) - 1 - sum(shifted(tr) for tr in traces)
    res = {"synthetic": synthetic, "param": a.param, "scan": {"values": list(map(float, values)), "chi2": chi2},
           "calibrated": best, "sigma": sig, "sigma_scale_factor": scale, "chi2_min_parabola": cmin,
           "timing_fiducial_sigma_ns": a.timing_ns, "chi2_total": total, "dof": dof,
           "chi2_nu_total": total / max(dof, 1), "fit_window_us": [0.0, t_cut], "metrics": metrics}
    if synthetic:
        res["hidden"] = a.truth
        res["hidden_within_sigma"] = abs(best - a.truth) / sig
        res["timing_offsets_us"] = {tr.kind: tr.meta.get("timing_offset_us") for tr in traces}
    json.dump(res, open(os.path.join(out, "calibration.json"), "w"), indent=2)
    np.savez(os.path.join(out, "fit_curves.npz"), **{f"best_{k}_{i}": np.asarray(v[i]) for k, v in best_obs.items() for i in (0, 1)},
             **{f"ens_{k}_{j}_{i}": np.asarray(c[i]) for k, cs in ens.items() for j, c in enumerate(cs) for i in (0, 1)})
    for tr in traces:
        val.save_trace(os.path.join(out, f"fitted_data_{tr.kind}.csv"), tr)
    for k, m in metrics.items():
        print(f"  {k:12s} RMS {m['rms']:.4g} {val.UNITS[k]}  chi2_nu {m['chi2_nu']:.2f}  peak offset "
              f"{1e3 * m['peak_offset']:+.0f} ns  timing shift {1e3 * m['shift']:+.1f} ns", flush=True)
    if synthetic:
        print(f"  hidden value {a.truth:g}: recovered within {res['hidden_within_sigma']:.2f} sigma", flush=True)
    plot_saved(out)


def plot_saved(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.load(open(os.path.join(out, "calibration.json")))
    f = np.load(os.path.join(out, "fit_curves.npz"))
    kinds = list(res["metrics"])
    traces = [val.load_trace(os.path.join(out, f"fitted_data_{k}.csv"), k, k, val.UNITS[k]) for k in kinds]
    best = {k: (f[f"best_{k}_0"], f[f"best_{k}_1"]) for k in kinds}
    ens = {k: [(f[f"ens_{k}_{j}_0"], f[f"ens_{k}_{j}_1"]) for j in range(3)] for k in kinds}
    lab = PARAMS[res["param"]]
    title = (f"{'SYNTHETIC shot (hidden ' + lab + ' = ' + format(res['hidden'], 'g') + ')' if res['synthetic'] else 'Shot'}"
             f": calibrated {lab} = {res['calibrated']:.2f} $\\pm$ {res['sigma']:.2f}, total $\\chi^2_\\nu$ = "
             f"{res['chi2_nu_total']:.2f}")
    val.report(os.path.join(out, "shot_calibration_fit.png"), traces, best, ens, res["metrics"], title)
    v, c = np.array(res["scan"]["values"]), np.array(res["scan"]["chi2"])
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))
    b, s, m = res["calibrated"], res["sigma"], res["chi2_min_parabola"]
    for k, x in enumerate(ax):
        x.semilogy(v, c, "o", color="#1f5f99") if k == 0 else x.plot(v, c, "o", color="#1f5f99")
        x.axvspan(b - s, b + s, color="#7fb3d5", alpha=0.4, lw=0)
    near = np.abs(v - b) < 8 * s
    vv = np.linspace(b - 6 * s, b + 6 * s, 200)
    sc = res.get("sigma_scale_factor", 1.0)
    ax[1].plot(vv, m + ((vv - b) * sc / s)**2, color="#888888", lw=1)
    ax[1].set_xlim(b - 6 * s, b + 6 * s)
    if near.any():
        ax[1].set_ylim(m - 2, max(m + 40, float(c[near].max()) * 1.05))
    ax[0].set_title("scan", loc="left"); ax[1].set_title("refined minimum", loc="left")
    ax = ax[1]
    if res["synthetic"]:
        ax.axvline(res["hidden"], color="#c0392b", ls="--", lw=1, label="hidden value")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel(lab); ax.set_ylabel(r"$\chi^2$ (all diagnostics)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "chi2_scan.png"), dpi=170); plt.close(fig)
    print(f"plots in {out}")


if __name__ == "__main__":
    main()
