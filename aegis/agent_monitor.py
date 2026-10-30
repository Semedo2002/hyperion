# agent_monitor.py - automatic run diagnostics: instability modes, magnetic insulation, numerical health.
#
# A rule-based monitor (statistics and physics thresholds, not a learned model) that watches a run and logs
# alerts with suggested design changes. Call observe(solver) every few steps; it keeps time series and checks:
#   - liner interface modes: the inner liner radius R(theta) on chosen slices (3D, or a 2D x-y cross-section) is
#     Fourier analysed; for every azimuthal mode m the growth rate is a least-squares fit of log|a_m| over a sliding
#     window. A mode growing faster than `growth_alert` (e-folds per unit time) or to more than `amp_alert` of the
#     mean radius is reported; modes that were not seeded (`seeded_modes`) are flagged as unexpected (e.g. m = 3
#     wall modes from a drive asymmetry);
#   - magnetic insulation: the axial flux through the fuel (cells with tracer < tracer_max) and the fraction of
#     fuel cells with Hall parameter x = cond_x0 |B| theta^1.5 / rho below 1. Flux loss beyond `flux_loss_alert` or
#     an unmagnetized fraction beyond `unmag_alert` is reported;
#   - numerical health: relative total-energy drift (closed domains) and max |div B| relative to |B| / dx.
# Alerts go to a JSON-lines log (one record per alert: time, kind, mode, measured values, recommendation) and to
# monitor.alerts; each kind / mode is reported once until it clears.
import json
import time

import numpy as np

RECOMMEND = {
    "unexpected_mode": "unseeded m = {m} mode growing at {g:.2f} / time unit: check the azimuthal symmetry of the "
                       "drive (current return-can posts, feed slots) and the liner wall-thickness tolerance; a thicker "
                       "or pre-magnetized liner (axial B) raises the stabilising tension",
    "fast_mode": "m = {m} grows at {g:.2f} / time unit (amplitude {a:.1%} of the radius): reduce the seed (surface "
                 "finish, corrugation depth), increase the liner aspect ratio (thicker wall) or the axial field",
    "large_mode": "m = {m} amplitude reached {a:.1%} of the mean radius: the liner is close to break-up there; "
                  "consider a thicker liner or an earlier stagnation (higher drive)",
    "flux_loss": "{f:.1%} of the axial flux has left the fuel: resistive / Nernst losses dominate; raise the seed "
                 "field, use a liner with higher conductivity, or shorten the implosion",
    "insulation_loss": "{f:.1%} of the fuel has Hall parameter below 1 (unmagnetized): thermal insulation is lost "
                       "there; raise the seed field or the fuel temperature at magnetization",
    "energy_drift": "total energy drifted by {e:.1e} (relative) in a closed domain: check the time step and the "
                    "boundary conditions",
    "divb": "max |div B| dx / |B| = {d:.1e}: use constrained transport (div_b='ct') or more GLM cleaning",
}


def interface_modes(r_theta, mmax=8):
    """complex Fourier amplitudes a_m (m = 0..mmax) of R(theta) sampled uniformly; |a_m| is the radius
    perturbation amplitude of mode m (a_0 the mean radius)"""
    n = len(r_theta)
    f = np.fft.rfft(r_theta) / n
    a = f[:mmax + 1].copy()
    a[1:] *= 2
    return a


def inner_radius_2d(x, y, C, n_theta=128, center=(0.0, 0.0), threshold=0.5, r_max=None):
    """inner radius of a tracer ring (C > threshold) against theta on a 2D x-y slice (bilinear sampling)"""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    r_max = r_max or 0.98 * min(abs(x[-1] - center[0]), abs(x[0] - center[0]), abs(y[-1] - center[1]),
                                abs(y[0] - center[1]))
    r = np.linspace(0.5 * (x[1] - x[0]), r_max, 600)
    X = center[0] + r[None, :] * np.cos(th)[:, None]
    Y = center[1] + r[None, :] * np.sin(th)[:, None]
    fi = np.clip((X - x[0]) / (x[1] - x[0]), 0, len(x) - 1.001)
    fj = np.clip((Y - y[0]) / (y[1] - y[0]), 0, len(y) - 1.001)
    i0, j0 = fi.astype(int), fj.astype(int)
    a, b = fi - i0, fj - j0
    v = (1 - a) * (1 - b) * C[i0, j0] + a * (1 - b) * C[i0 + 1, j0] + (1 - a) * b * C[i0, j0 + 1] + a * b * C[i0 + 1, j0 + 1]
    hit = v > threshold
    k = np.argmax(hit, axis=1)
    rr = r[k]
    # sub-sample position of the crossing
    k0 = np.maximum(k - 1, 0)
    v0, v1 = v[np.arange(n_theta), k0], v[np.arange(n_theta), k]
    frac = np.where(v1 != v0, (threshold - v0) / np.where(v1 != v0, v1 - v0, 1.0), 0.0)
    rr = r[k0] + np.clip(frac, 0, 1) * (r[k] - r[k0])
    return th, np.where(hit.any(axis=1), rr, np.nan)


class AgentMonitor:
    def __init__(self, log_path=None, seeded_modes=(), mmax=8, window=8, growth_alert=3.0, amp_alert=0.1,
                 min_amp=5e-3, flux_loss_alert=0.2, unmag_alert=0.3, energy_alert=1e-8, divb_alert=1e-8,
                 tracer_max=0.5, center=(0.0, 0.0), slices=None, closed_domain=None):
        self.log_path = log_path
        self.seeded = set(seeded_modes)
        self.mmax, self.window = mmax, window
        self.growth_alert, self.amp_alert, self.min_amp = growth_alert, amp_alert, min_amp
        self.flux_loss_alert, self.unmag_alert = flux_loss_alert, unmag_alert
        self.energy_alert, self.divb_alert = energy_alert, divb_alert
        self.tracer_max, self.center, self.slices = tracer_max, center, slices
        self.closed = closed_domain
        self.t, self.modes, self.flux, self.unmag, self.energy, self.divb = [], [], [], [], [], []
        self.alerts, self._active = [], set()

    # ---------------- measurements ----------------
    def measure(self, solver):
        """(mode amplitudes |a_m| / a_0 averaged over slices, fuel flux, unmagnetized fuel fraction, energy, div B)"""
        from .physics import cons_to_prim
        from .config import iRHO, iPR, iCLR, iBX, iBY, iBZ
        s, cfg, ng = solver, solver.cfg, solver.ng
        three_d = hasattr(s, "nz_tot")
        W = cons_to_prim(s.U, s.eos)
        get = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
        x, y = get(s.x)[ng:-ng], get(s.y)[ng:-ng]
        C = get(W[iCLR])
        if three_d:
            I = (slice(ng, -ng),) * 3
            ks = self.slices or [cfg.nz // 2]
            planes = [C[I][:, :, k] for k in ks]
        else:
            I = (slice(ng, -ng),) * 2
            planes = [C[I]]
        amps = []
        for P in planes:
            th, rr = inner_radius_2d(x, y, P, center=self.center)
            if np.isnan(rr).any():
                continue
            a = np.abs(interface_modes(rr, self.mmax))
            amps.append(a / max(a[0], 1e-300))
        amp = np.mean(amps, axis=0) if amps else np.full(self.mmax + 1, np.nan)
        rho, th_ = get(W[iRHO])[I], (get(W[iPR]) / get(W[iRHO]))[I]
        Bz = get(W[iBZ])[I]
        Bm = np.sqrt(get(W[iBX])[I]**2 + get(W[iBY])[I]**2 + Bz**2)
        fuel = C[I] < self.tracer_max
        dA = cfg.dx * cfg.dy
        flux = float(np.sum(np.where(fuel, Bz, 0.0)) * dA) / (Bz.shape[2] if three_d else 1)
        if cfg.cond_x0 > 0:
            xh = cfg.cond_x0 * Bm * np.maximum(th_, 0) ** 1.5 / rho
            unmag = float(np.mean(xh[fuel] < 1.0)) if fuel.any() else 0.0
        else:
            unmag = np.nan
        energy = float(s.total_energy()) if hasattr(s, "total_energy") else float(get(s.U)[7][I].sum())
        bmax = max(float(Bm.max()), 1e-300)
        divb = float(s.div_b_max()) * min(cfg.dx, cfg.dy) / bmax
        return amp, flux, unmag, energy, divb

    def observe(self, solver, t=None):
        t = solver.t if t is None else t
        amp, flux, unmag, energy, divb = self.measure(solver)
        self.t.append(t)
        self.modes.append(amp)
        self.flux.append(flux)
        self.unmag.append(unmag)
        self.energy.append(energy)
        self.divb.append(divb)
        return self.check()

    def growth_rates(self, with_quality=False):
        """least-squares growth rate of log|a_m| over the last `window` observations, per mode, using only samples
        above the measurable amplitude min_amp (below it the extracted interface is noise); NaN with fewer than 3
        such samples or a poor fit (R^2 < 0.7)"""
        n = min(len(self.t), self.window)
        g = np.full(self.mmax + 1, np.nan)
        r2 = np.full(self.mmax + 1, np.nan)
        if n < 3:
            return (g, r2) if with_quality else g
        t = np.asarray(self.t[-n:])
        A = np.asarray(self.modes[-n:])
        for m in range(self.mmax + 1):
            ok = np.isfinite(A[:, m]) & (A[:, m] >= self.min_amp)
            if ok.sum() < 3:
                continue
            tt, la = t[ok], np.log(A[ok, m])
            tc, lc = tt - tt.mean(), la - la.mean()
            if (tc**2).sum() <= 0:
                continue
            g[m] = (tc * lc).sum() / (tc**2).sum()
            ss = (lc**2).sum()
            r2[m] = 1.0 - ((lc - g[m] * tc)**2).sum() / ss if ss > 0 else 1.0
            if r2[m] < 0.7:
                g[m] = np.nan
        return (g, r2) if with_quality else g

    # ---------------- rules ----------------
    def _alert(self, kind, key, values):
        tag = (kind, key)
        if tag in self._active:
            return None
        self._active.add(tag)
        rec = dict(t=float(self.t[-1]), kind=kind, mode=key, values={k: float(v) for k, v in values.items()},
                   recommendation=RECOMMEND[kind].format(**values), wall_time=time.time())
        self.alerts.append(rec)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    def _clear(self, kind, key):
        self._active.discard((kind, key))

    def check(self):
        new = []
        g = self.growth_rates()
        amp = self.modes[-1]
        for m in range(1, self.mmax + 1):
            if not np.isfinite(g[m]) or not np.isfinite(amp[m]) or amp[m] < self.min_amp:
                continue
            vals = dict(m=m, g=g[m], a=amp[m])
            if g[m] > self.growth_alert:
                unexpected = bool(self.seeded) and m not in self.seeded
                new.append(self._alert("unexpected_mode" if unexpected else "fast_mode", m, vals))
            elif g[m] < 0.5 * self.growth_alert:
                self._clear("fast_mode", m)
                self._clear("unexpected_mode", m)
            if amp[m] > self.amp_alert:
                new.append(self._alert("large_mode", m, vals))
        if len(self.flux) > 1 and abs(self.flux[0]) > 0:
            f = 1.0 - self.flux[-1] / self.flux[0]
            if f > self.flux_loss_alert:
                new.append(self._alert("flux_loss", 0, dict(f=f)))
        if np.isfinite(self.unmag[-1]) and self.unmag[-1] > self.unmag_alert:
            new.append(self._alert("insulation_loss", 0, dict(f=self.unmag[-1])))
        if self.closed and len(self.energy) > 1:
            e = abs(self.energy[-1] - self.energy[0]) / max(abs(self.energy[0]), 1e-300)
            if e > self.energy_alert:
                new.append(self._alert("energy_drift", 0, dict(e=e)))
        if self.divb[-1] > self.divb_alert:
            new.append(self._alert("divb", 0, dict(d=self.divb[-1])))
        return [r for r in new if r]

    def summary(self):
        g = self.growth_rates()
        return dict(t=self.t[-1] if self.t else None,
                    modes={int(m): dict(amp=float(self.modes[-1][m]), growth=float(g[m])) for m in range(1, self.mmax + 1)}
                    if self.modes else {},
                    flux_fraction=float(self.flux[-1] / self.flux[0]) if self.flux and self.flux[0] else None,
                    unmagnetized=self.unmag[-1] if self.unmag else None,
                    divb=self.divb[-1] if self.divb else None, alerts=len(self.alerts))
