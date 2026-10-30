"""Inverse design of an imploding liner with adjoint gradients (aegis/autodiff.py).

An axisymmetric (r, z) liner of fixed mass and fixed kinetic energy implodes onto a uniform axial seed field
B0 in a light gas fill (ideal MHD, constrained transport). Its inner radius is R(z) = R0 + sum_k a_k cos(2 pi k
z / L). The objective is the peak axial field in a small window on the axis at the mid-plane z = L/2, as a
smooth maximum over time: B_peak = (sum_n b_n^q dt / T)^(1/q), b_n = mean B_z in the window after step n.

Each design iteration is one forward run and one adjoint pass, which give dB_peak/da_k for every mode at
once; the design is updated by projected gradient ascent (Adam, |a_k| <= a_max). The script checks one
adjoint derivative against a central finite difference before it starts.

With --drive the liner also gets a magnetic piston (B_phi on the outer boundary, current amplitude I as a
parameter) and the script reports dB_peak/dI alongside the shape derivatives.

Dimensionless units (liner density 1, implosion speed 1, R0 = 1). Axisymmetric: the "shape" is R(z); a
fully 3D liner is outside the differentiable twin's scope (2D only).

    python examples/optimize_liner_shape.py                     # 3 modes, 12 iterations
    python examples/optimize_liner_shape.py --quick             # small grid smoke run
    python examples/optimize_liner_shape.py --drive 0.3 --iters 0   # gradients only, with a drive
"""
import os
import sys
import json
import time
import argparse
import contextlib
import io

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config, BY
    from aegis.solver import MHDSolver
from aegis import autodiff as ad


def build(a):
    cfg = Config(nx=a.nr, ny=a.nz, x_min=0.0, x_max=a.r_max, y_min=0.0, y_max=a.lz, cfl=0.3, gamma=5 / 3,
                 geometry="cylindrical", bc_x_type="extrapolation", bc_y_type="periodic",
                 diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    drive = None
    if a.drive:                                       # B_phi(t) = I r_max-normalised, ramped over t_ramp
        drive = lambda t, p: p[-1] * (1.0 - np.exp(-t / 0.1))
    sim = ad.DiffMHD(s, drive=drive, xp=np)
    r = s.rc.get()[:, None] * np.ones((1, s.ny_tot))
    z = np.ones((s.nx_tot, 1)) * s.y.get()[None, :]
    dV = r * cfg.dx * cfg.dy
    K = a.modes

    def liner(p):
        Rz = a.r0 + sum(p[k] * ad.const(np.cos(2 * np.pi * (k + 1) * z / a.lz)) for k in range(K)) if K else a.r0
        rr = ad.const(r)
        prof = 0.5 * (ad.tanh((rr - Rz) / a.w) - ad.tanh((rr - Rz - a.dr) / a.w))
        rho_l = prof * 1.0
        v_l = prof * ad.const(-(a.r0 + 0.5 * a.dr) / np.maximum(r, 1e-3))     # v ~ 1/r, speed 1 at mid-wall
        return prof, rho_l, v_l

    zero = np.zeros(K + (1 if a.drive else 0))
    _, rho0, v0 = liner(ad.const(zero))
    sl = (slice(2, -2), slice(2, -2))
    M0 = float(np.sum((rho0.v * dV)[sl]))
    KE0 = float(np.sum((0.5 * rho0.v * v0.v**2 * dV)[sl]))

    def W_fn(p):
        prof, rho_l, v_l = liner(p)
        M = ad.tsum(ad.getitem(rho_l * ad.const(dV), sl))
        rho_l = rho_l * (M0 / M)                      # fixed liner mass
        KE = ad.tsum(ad.getitem(0.5 * rho_l * v_l * v_l * ad.const(dV), sl))
        v = v_l * ad.sqrt(KE0 / KE)                   # fixed kinetic energy
        return [a.rho_gas + rho_l, v, 0.0, 0.0, 0.0, a.b0, 0.0, a.p0, 0.0, prof, 0.0]

    init = lambda p: ad.cells_to_state(sim, W_fn, p)
    win = ((r < a.core_r) & (np.abs(z - 0.5 * a.lz) < a.core_dz))[sl].astype(float)
    win /= win.sum()

    def b_core(st):
        return ad.tsum(ad.interior(sim, st[BY]) * ad.const(win))

    q = a.q
    step_obj = lambda st, t, p: b_core(st) ** q
    return s, sim, init, step_obj, liner, dict(r=r, z=z, win=win, M0=M0, KE0=KE0)


def peak(J, T, q):
    return (J / T) ** (1.0 / q)


def evaluate(sim, init, step_obj, p, a, dts=None):
    J, g, info = sim.value_and_grad(init, p, t_end=None if dts else a.t_end, dts=dts, step_objective=step_obj)
    T = info["t"]
    B = peak(J, T, a.q)
    return B, g * B / (a.q * J), info                # dB_peak/dp from dJ/dp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nr", type=int, default=56)
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--r-max", type=float, default=1.6)
    ap.add_argument("--lz", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0, help="mean inner liner radius")
    ap.add_argument("--dr", type=float, default=0.2, help="liner thickness")
    ap.add_argument("--w", type=float, default=0.03, help="liner edge width")
    ap.add_argument("--rho-gas", type=float, default=0.05)
    ap.add_argument("--p0", type=float, default=0.01)
    ap.add_argument("--b0", type=float, default=0.1)
    ap.add_argument("--t-end", type=float, default=1.1)
    ap.add_argument("--modes", type=int, default=3)
    ap.add_argument("--a-max", type=float, default=0.15)
    ap.add_argument("--a-start", type=float, default=0.02, help="initial amplitude of every mode")
    ap.add_argument("--core-r", type=float, default=0.12)
    ap.add_argument("--core-dz", type=float, default=0.1)
    ap.add_argument("--q", type=float, default=8.0, help="smooth-max exponent over time")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--drive", type=float, default=0.0, help="magnetic piston amplitude (B_phi at r_max)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "liner_optimization"))
    a = ap.parse_args()
    if a.quick:
        a.nr, a.nz, a.iters, a.t_end = 28, 16, 2, 0.6
    os.makedirs(a.out, exist_ok=True)
    s, sim, init, step_obj, liner, geo = build(a)
    K = a.modes
    # start off the z-symmetric design: there the z slopes vanish and the van Leer limiter (and the corner EMF
    # upwinding) is not differentiable, so one-sided derivatives differ
    p = np.full(K + (1 if a.drive else 0), a.a_start)
    if a.drive:
        p[-1] = a.drive

    t0 = time.time()
    B, g, info = evaluate(sim, init, step_obj, p, a)
    dts = info["dts"]
    print(f"start: B_peak = {B:.6g} (B0 = {a.b0}), {len(dts)} steps, forward + adjoint {time.time() - t0:.1f}s")
    names = [f"a{k + 1}" for k in range(K)] + (["I_drive"] if a.drive else [])
    for n, v in zip(names, g):
        print(f"  dB_peak/d{n} = {v:+.6e}")
    # finite-difference check of the first derivative, same step sequence. The scheme is piecewise smooth with
    # many limiter kinks (cells where a slope changes sign): the step must stay inside one smooth piece, so it is
    # small (1e-6 already crosses kinks on this grid)
    h = 1e-8
    k = 0
    pp, pm = p.copy(), p.copy()
    pp[k] += h
    pm[k] -= h
    Jp, _ = sim.run(init, pp, dts=dts, step_objective=step_obj)
    Jm, _ = sim.run(init, pm, dts=dts, step_objective=step_obj)
    fd = (peak(Jp, info["t"], a.q) - peak(Jm, info["t"], a.q)) / (2 * h)
    print(f"  check d/d{names[k]}: adjoint {g[k]:+.8e}, finite difference {fd:+.8e} (rel {abs(fd - g[k]) / abs(g[k]):.1e})")

    hist = [dict(iter=0, B_peak=B, params=p.tolist(), grad=g.tolist())]
    m, v = np.zeros_like(p), np.zeros_like(p)
    best = (B, p.copy())
    for it in range(1, a.iters + 1):
        gs = g.copy()
        if a.drive:
            gs[-1] = 0.0                              # the drive is a fixed input, not a design variable
        m = 0.9 * m + 0.1 * gs
        v = 0.999 * v + 0.001 * gs**2
        lr = a.lr * 0.5 * (1 + np.cos(np.pi * (it - 1) / a.iters))      # cosine decay
        step = lr * (m / (1 - 0.9**it)) / (np.sqrt(v / (1 - 0.999**it)) + 1e-12)
        p = p + step
        p[:K] = np.clip(p[:K], -a.a_max, a.a_max)
        t1 = time.time()
        B, g, info = evaluate(sim, init, step_obj, p, a)
        hist.append(dict(iter=it, B_peak=B, params=p.tolist(), grad=g.tolist()))
        if B > best[0]:
            best = (B, p.copy())
        print(f"iter {it:2d}: B_peak = {B:.6g}  a = {np.array2string(p[:K], precision=4)}  ({time.time() - t1:.0f}s)")

    summary = dict(B0=a.b0, B_peak_start=hist[0]["B_peak"], B_peak_best=best[0], params_best=best[1].tolist(),
                   gain=best[0] / hist[0]["B_peak"], fd_check=dict(param=names[0], adjoint=hist[0]["grad"][0], fd=fd),
                   history=hist, args=vars(a))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"best B_peak {best[0]:.6g} vs start {hist[0]['B_peak']:.6g} (x{summary['gain']:.3f})")
    plot(a, sim, init, liner, geo, hist, best[1])


def plot(a, sim, init, liner, geo, hist, p_best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p0 = np.full_like(p_best, a.a_start)
    p0[a.modes:] = p_best[a.modes:]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    z = np.linspace(0, a.lz, 200)
    for p, lab, c in ((p0, "start", "0.5"), (p_best, "optimised", "C3")):
        Rz = a.r0 + sum(p[k] * np.cos(2 * np.pi * (k + 1) * z / a.lz) for k in range(a.modes))
        ax[0].plot(Rz, z, color=c, label=lab)
        ax[0].plot(Rz + a.dr, z, color=c, ls="--")
    ax[0].set_xlabel("r"); ax[0].set_ylabel("z"); ax[0].set_title("liner inner / outer radius")
    ax[0].axhspan(0.5 * a.lz - a.core_dz, 0.5 * a.lz + a.core_dz, xmax=a.core_r / ax[0].get_xlim()[1], color="C0",
                  alpha=0.3, label="target window")
    ax[0].legend(fontsize=8)
    ax[1].plot([h["iter"] for h in hist], [h["B_peak"] / a.b0 for h in hist], "o-")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("B_peak / B0"); ax[1].set_title("objective")
    # B_z history in the window for both designs
    for p, lab, c in ((p0, "start", "0.5"), (p_best, "optimised", "C3")):
        st = init(ad.const(p))
        t, ts, bs = 0.0, [], []
        while t < a.t_end - 1e-12:
            dt = min(sim.compute_dt(st), a.t_end - t)
            st = sim.step(st, dt, t, ad.const(p))
            t += dt
            ts.append(t)
            bs.append(float(np.sum(ad.interior(sim, st[BY]).v * geo["win"])))
        ax[2].plot(ts, np.array(bs) / a.b0, color=c, label=lab)
    ax[2].set_xlabel("t"); ax[2].set_ylabel("B_z / B0 in the window"); ax[2].set_title("flux compression")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "liner_optimization.png"), dpi=130)
    print(f"wrote {os.path.join(a.out, 'liner_optimization.png')}")


if __name__ == "__main__":
    main()
