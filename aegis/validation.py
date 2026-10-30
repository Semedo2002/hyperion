# validation.py - comparing simulated observables with shot data: traces, goodness of fit, calibration.
#
# A Trace is a measured signal y(t) with 1-sigma uncertainties (VISAR free-surface velocity, a B-dot probe
# voltage, radiography radii, ...). compare() samples a simulated signal at the data times and returns
#     RMS error, RMS / peak |y|, chi^2 = sum ((y - f) / sigma)^2, chi^2_nu = chi^2 / (N - n_params),
#     peak offset = delay of the simulation against the data that maximises their cross-correlation,
# optionally after fitting a timing shift (fiducial uncertainty) that minimises chi^2. calibrate() fits a
# parabola to chi^2 over a scan of one model parameter: the minimum is the calibrated value and its 1-sigma
# range is where chi^2 rises by 1 above it (the usual Delta chi^2 = 1 rule, valid when the model is close to
# linear in the parameter there and the sigmas are right). Observables for a pulsed-power liner (FCG with the
# RLC drive) are in observables_from_fcg().
import csv
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trace:
    name: str
    t: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    units: str = ""
    kind: str = ""                                   # visar, bdot, radiography, ...
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t, self.y = np.asarray(self.t, float), np.asarray(self.y, float)
        s = np.asarray(self.sigma, float)
        self.sigma = np.broadcast_to(s, self.y.shape).copy()


def load_trace(path, name=None, kind="", units="", sigma=None, t_col=0, y_col=1, s_col=2, t_scale=1.0, y_scale=1.0):
    """CSV with a header row: time, value[, sigma]. Without a sigma column: `sigma` (absolute, or a string like
    '3%' of the peak |value|)."""
    rows = [r for r in csv.reader(open(path)) if r and not r[0].lstrip().startswith("#")]
    head, data = rows[0], np.array([[float(x) for x in r] for r in rows[1:]])
    t, y = data[:, t_col] * t_scale, data[:, y_col] * y_scale
    if data.shape[1] > s_col:
        s = data[:, s_col] * y_scale
    elif isinstance(sigma, str) and sigma.endswith("%"):
        s = float(sigma[:-1]) / 100 * np.max(np.abs(y))
    elif sigma is not None:
        s = float(sigma)
    else:
        raise ValueError(f"{path}: no sigma column and no sigma given")
    return Trace(name or head[y_col].strip(), t, y, s, units, kind)


def save_trace(path, tr):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        fh.write(f"# {tr.name} ({tr.kind}), {tr.units}" + (f"; {tr.meta}" if tr.meta else "") + "\n")
        w.writerow(["t", tr.name, "sigma"])
        w.writerows(zip(tr.t, tr.y, tr.sigma))


def sample(t_sim, y_sim, t, shift=0.0):
    """simulated signal at the data times (linear interpolation; NaN outside the simulated span)"""
    return np.interp(t - shift, t_sim, y_sim, left=np.nan, right=np.nan)


def _peak_time(t, y):
    """time of the largest |y|, refined by a parabola through the sample and its neighbours"""
    k = int(np.nanargmax(np.abs(y)))
    if 0 < k < len(t) - 1:
        a, b, c = np.abs(y[k - 1:k + 2])
        den = a - 2 * b + c
        if den != 0:
            return t[k] + 0.5 * (a - c) / den * 0.5 * (t[k + 1] - t[k - 1])
    return t[k]


def compare(tr, t_sim, y_sim, n_params=0, fit_shift=False, max_shift=None, shift_sigma=None):
    """goodness of fit of a simulated signal to a trace (see module notes); with fit_shift the timing shift
    that minimises chi^2 (grid then golden section within +-max_shift) is applied first and counts as one
    parameter; shift_sigma (the timing-fiducial uncertainty) adds the prior (shift / shift_sigma)^2, reported
    as chi2_prior and included in chi2_total"""
    def chi2(shift, prior=True):
        f = sample(t_sim, y_sim, tr.t, shift)
        ok = np.isfinite(f)
        c = float(np.sum(((tr.y[ok] - f[ok]) / tr.sigma[ok])**2))
        if prior and shift_sigma:
            c += (shift / shift_sigma)**2
        return c, ok
    shift = 0.0
    if fit_shift:
        L = max_shift if max_shift is not None else 0.1 * (tr.t[-1] - tr.t[0])
        grid = np.linspace(-L, L, 121)               # dense search first: chi^2(shift) can be multimodal
        k = int(np.argmin([chi2(x)[0] for x in grid]))
        a, b = grid[max(k - 1, 0)], grid[min(k + 1, 120)]
        g = (np.sqrt(5) - 1) / 2
        x1, x2 = b - g * (b - a), a + g * (b - a)
        f1, f2 = chi2(x1)[0], chi2(x2)[0]
        for _ in range(60):
            if f1 < f2:
                b, x2, f2 = x2, x1, f1
                x1 = b - g * (b - a); f1 = chi2(x1)[0]
            else:
                a, x1, f1 = x1, x2, f2
                x2 = a + g * (b - a); f2 = chi2(x2)[0]
        shift = 0.5 * (a + b)
        n_params += 1
    c2, ok = chi2(shift, prior=False)
    cp = (shift / shift_sigma)**2 if (fit_shift and shift_sigma) else 0.0
    f = sample(t_sim, y_sim, tr.t, shift)
    r = tr.y[ok] - f[ok]
    n = int(ok.sum())
    dof = max(n - n_params, 1)
    tp_sim = _peak_time(t_sim + shift, y_sim)
    return {"name": tr.name, "n": n, "rms": float(np.sqrt(np.mean(r**2))),
            "nrms": float(np.sqrt(np.mean(r**2)) / max(np.max(np.abs(tr.y)), 1e-300)),
            "chi2": c2, "chi2_prior": cp, "chi2_total": c2 + cp, "dof": dof, "chi2_nu": c2 / dof, "shift": shift,
            "peak_offset": xcorr_lag(tr, t_sim, y_sim, max_shift if max_shift is not None else 0.1 * np.ptp(tr.t)),
            "peak_time_offset": float(tp_sim - _peak_time(tr.t, tr.y)), "coverage": n / len(tr.t)}


def xcorr_lag(tr, t_sim, y_sim, max_lag, n=401):
    """peak alignment offset: the delay of the simulation relative to the data that maximises their
    cross-correlation over the data window (parabolic refinement); robust to noise and to oscillating
    signals, unlike the time of the largest sample"""
    lags = np.linspace(-max_lag, max_lag, n)
    C = []
    for L in lags:
        f = sample(t_sim, y_sim, tr.t + L)
        ok = np.isfinite(f)
        C.append(np.sum(tr.y[ok] * f[ok]) / max(np.sqrt(np.sum(f[ok]**2) * np.sum(tr.y[ok]**2)), 1e-300))
    C = np.array(C)
    k = int(np.argmax(C))
    if 0 < k < n - 1:
        a, b, c = C[k - 1:k + 2]
        den = a - 2 * b + c
        if den != 0:
            return float(lags[k] + 0.5 * (a - c) / den * (lags[1] - lags[0]))
    return float(lags[k])


def calibrate(values, chi2s):
    """parabola through chi^2(parameter) around its minimum: (best, sigma, chi2_min); sigma from
    chi^2(best +- sigma) = chi2_min + 1. Uses the lowest point and its neighbours (all points if <= 3)."""
    v, c = np.asarray(values, float), np.asarray(chi2s, float)
    order = np.argsort(v)
    v, c = v[order], c[order]
    k = int(np.argmin(c))
    lo, hi = max(k - 1, 0), min(k + 2, len(v))
    if hi - lo < 3:
        lo, hi = max(0, hi - 3), max(3, hi)
    A, B, C = np.polyfit(v[lo:hi], c[lo:hi], 2)
    if A <= 0:                                       # no curvature: the minimum is at the scan edge
        return float(v[k]), float("inf"), float(c[k])
    best = -B / (2 * A)
    return float(best), float(1 / np.sqrt(A)), float(C - B * B / (4 * A))


def calibrate_valley(values, chi2s, dof, window=None):
    """least-squares parabola through every point of the chi^2 valley (chi^2 <= chi2_min + window, default
    max(100, chi2_min)): robust where chi^2 is jagged on small parameter steps (numerical noise of the model).
    Returns (best, sigma, chi2_min_fit, scale): sigma from Delta chi^2 = 1 on the parabola, multiplied by
    scale = sqrt(chi2_min_fit / dof) when that exceeds 1 (the fit is worse than the stated errors)."""
    v, c = np.asarray(values, float), np.asarray(chi2s, float)
    cmin = c.min()
    w = window if window is not None else max(100.0, cmin)
    sel = c <= cmin + w
    if sel.sum() < 3:
        sel = np.argsort(c)[:3]
    A, B, C = np.polyfit(v[sel], c[sel], 2)
    if A <= 0:
        k = int(np.argmin(c))
        return float(v[k]), float("inf"), float(cmin), 1.0
    best = -B / (2 * A)
    cfit = C - B * B / (4 * A)
    scale = max(1.0, np.sqrt(max(cfit, 0.0) / max(dof, 1)))
    return float(best), float(scale / np.sqrt(A)), float(cfit), float(scale)


def band(curves, t):
    """ensemble band on a common time grid: (min, max, mean) over curves = [(t_i, y_i), ...]"""
    Y = np.array([np.interp(t, ti, yi, left=np.nan, right=np.nan) for ti, yi in curves])
    return np.nanmin(Y, axis=0), np.nanmax(Y, axis=0), np.nanmean(Y, axis=0)


# ------------------------------------------------------------------ pulsed-power liner observables

def instrument(t, y, fwhm):
    """signal seen through a Gaussian instrument response of the given FWHM (same time units): resampled onto
    a uniform grid, convolved, returned on that grid"""
    if fwhm <= 0:
        return t, y
    dt = min(np.median(np.diff(t)), fwhm / 10)
    tu = np.arange(t[0], t[-1], dt)
    yu = np.interp(tu, t, y)
    s = fwhm / 2.3548 / dt
    k = np.arange(-int(4 * s) - 1, int(4 * s) + 2)
    w = np.exp(-0.5 * (k / s)**2)
    w /= w.sum()
    pad = len(k) // 2
    yp = np.concatenate([np.full(pad, yu[0]), yu, np.full(pad, yu[-1])])
    return tu, np.convolve(yp, w, mode="valid")


def observables_from_fcg(d, probe_r_mm=20.0, probe_area_mm2=1.0, fwhm_us=0.01):
    """simulated diagnostics from an FCG run with the RLC drive (columns of its diagnostics.csv):
    visar: free-surface velocity of the liner (inward positive), v_r at its outer surface when the run
    logged v_surface_kms, else -d r_out/dt (km/s); bdot: a one-turn probe of area A at radius
    r_p in the return gap sees B_theta = mu0 I / (2 pi r_p), so V = A mu0 / (2 pi r_p) dI/dt (V);
    radiography: the cavity radius (mm). Times in us. Each goes through a Gaussian instrument response of
    FWHM fwhm_us (10 ns), the instruments' bandwidth, which also damps the step-to-step noise of the logged
    signals (differences between runs at that level would otherwise swamp chi^2)."""
    mu0 = 4e-7 * np.pi
    t = np.asarray(d["t_us"], float)
    k = np.unique(t, return_index=True)[1]
    t = t[k]
    I = np.asarray(d["I_MA"], float)[k] * 1e6
    r_out = np.asarray(d["r_out_mm"], float)[k]
    dIdt = np.gradient(I, t * 1e-6)
    vis = -np.asarray(d["v_surface_kms"], float)[k] if "v_surface_kms" in d else -np.gradient(r_out, t)
    raw = {"visar": vis, "bdot": probe_area_mm2 * 1e-6 * mu0 / (2 * np.pi * probe_r_mm * 1e-3) * dIdt,
           "radiography": np.asarray(d["R_cavity_mm"], float)[k]}
    return {key: instrument(t, y, fwhm_us) for key, y in raw.items()}


UNITS = {"visar": "km/s", "bdot": "V", "radiography": "mm"}
LABELS = {"visar": "VISAR free-surface velocity", "bdot": "B-dot probe voltage", "radiography": "radiography core radius"}


def report(path, traces, best, ens=None, metrics=None, title=""):
    """shot_calibration_fit.png: data with error bars, the best-fit simulation, the ensemble band and residuals"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    n = len(traces)
    fig, ax = plt.subplots(2, n, figsize=(4.2 * n, 5.6), sharex="col", gridspec_kw={"height_ratios": [3, 1]},
                           squeeze=False)
    for j, tr in enumerate(traces):
        a, r = ax[0, j], ax[1, j]
        tb, yb = best[tr.kind]
        m = (metrics or {}).get(tr.kind, {})
        sh = m.get("shift", 0.0)
        if ens:
            tg = np.linspace(tr.t.min(), tr.t.max(), 400)
            lo, hi, _ = band([(ti + sh, yi) for ti, yi in ens[tr.kind]], tg)
            a.fill_between(tg, lo, hi, color="#7fb3d5", alpha=0.35, lw=0, label="simulation ensemble (1 sigma)")
        a.plot(tb + sh, yb, color="#1f5f99", lw=1.4, label="best-fit simulation")
        a.errorbar(tr.t, tr.y, yerr=tr.sigma, fmt="o", ms=2.5, color="#c0392b", ecolor="#e6a39a", elinewidth=0.8,
                   label="shot data")
        a.set_title(LABELS.get(tr.kind, tr.name), loc="left")
        a.set_ylabel(tr.units)
        if m:
            a.text(0.02, 0.97, f"RMS {m['rms']:.3g} {tr.units}\n$\\chi^2_\\nu$ = {m['chi2_nu']:.2f}\n"
                   f"peak offset {m['peak_offset'] * 1e3:+.0f} ns", transform=a.transAxes, va="top", fontsize=8)
        f = sample(tb, yb, tr.t, sh)
        r.axhline(0, color="#999999", lw=0.7)
        r.errorbar(tr.t, (tr.y - f) / tr.sigma, yerr=1.0, fmt="o", ms=2, color="#555555", elinewidth=0.6)
        r.set_ylabel("residual / $\\sigma$")
        r.set_xlabel("t ($\\mu$s)")
        x0, x1 = tr.t.min() - 0.05 * np.ptp(tr.t), tr.t.max() + 0.05 * np.ptp(tr.t)
        a.set_xlim(x0, x1)
        inwin = (tb + sh >= x0) & (tb + sh <= x1)
        lo_ = min(float(np.min(tr.y - tr.sigma)), float(np.min(yb[inwin])))
        hi_ = max(float(np.max(tr.y + tr.sigma)), float(np.max(yb[inwin])))
        a.set_ylim(lo_ - 0.08 * (hi_ - lo_), hi_ + 0.08 * (hi_ - lo_))
    ax[0, -1].legend(*ax[0, 0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="lower left")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
