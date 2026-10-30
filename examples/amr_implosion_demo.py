"""Adaptive mesh refinement on a magnetized liner implosion (2D cross-section, aegis/amr.py).

A dense liner (radii 0.5-0.7, density 10, a 1% mode-6 ripple on its inner surface) implodes at Mach 4 onto a
core holding an out-of-plane seed field, like examples/mtif_flux_compression.py. The same problem runs three
ways: a uniform 128^2 grid, a uniform 512^2 grid (the reference), and 128^2 with ratio-4 patches on the
liner, the converging shock and the hot core (density-gradient and pressure-jump flags, regridding every 4 steps;
poorly filled boxes are split, so the ring is covered by many touching patches). --region fuel refines only
inside the gas the liner encloses.
At 1 mm per code unit (core radius 0.5 mm) the patches resolve 3.9 um against 15.6 um outside. Reported: peak
core density and temperature, core radius, density error against the 512^2 run, fine-cell fraction, wall time.

    python examples/amr_implosion_demo.py               # 3 runs, plots in results/amr_implosion
    python examples/amr_implosion_demo.py --quick       # 64 / 256
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
    from aegis.config import Config, NVAR, iRHO, iVX, iVY, iBZ, iPR, iCLR
    from aegis.physics import xp, prim_to_cons, cons_to_prim
    from aegis.solver import MHDSolver, to_numpy
    from aegis.amr import AMR, AMRParams
    from aegis.diagnostics import enclosed_by_tracer

G, R0, R1 = 5.0 / 3.0, 0.5, 0.7


def cfg_for(n, t_end):
    return Config(nx=n, ny=n, x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, t_end=t_end, cfl=0.3, mach=1.0, gamma=G,
                  B_transverse=1.0, bc_x_type="extrapolation", bc_y_type="extrapolation", powell_source=False,
                  material_tracer="volume_fraction", diag_interval=10**9, snapshot_times=[])


def initial(s, B0=1.0, mach=4.0):
    X, Y = s.X, s.Y
    r = xp.sqrt(X**2 + Y**2) + 1e-12
    th = xp.arctan2(Y, X)
    w = 1.5 * s.cfg.dx
    Rin = R0 * (1 + 0.01 * xp.cos(6 * th))
    liner = 0.5 * (1 + xp.tanh((r - Rin) / w)) * 0.5 * (1 - xp.tanh((r - R1) / w))
    core = 0.5 * (1 - xp.tanh((r - Rin) / w))
    V = mach * np.sqrt(G)
    W = xp.zeros((NVAR,) + s.U.shape[1:])
    W[iRHO] = 1.0 + 9.0 * liner
    W[iBZ] = B0 * core
    W[iPR] = 1.0 + 0.5 * B0**2 * (1 - core)              # total pressure balance across the core edge
    W[iVX], W[iVY] = -V * liner * X / r, -V * liner * Y / r
    W[iCLR] = liner
    return prim_to_cons(W, s.eos)


def metrics(rho, p, C, dx):
    core = C < 0.5
    rr = np.hypot(*np.meshgrid((np.arange(rho.shape[0]) + 0.5) * dx - 1, (np.arange(rho.shape[1]) + 0.5) * dx - 1,
                               indexing="ij"))
    inside = core & (rr < 0.45)
    return {"rho_core_max": float(rho[inside].max()), "T_core_max": float((p / rho)[inside].max()),
            "R_core": float(np.sqrt(inside.sum() * dx * dx / np.pi))}


def run_uniform(n, t_end):
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg_for(n, t_end))
    s.U = initial(s); s.t = 0.0; s.step = 0
    t0 = time.time()
    while s.t < t_end - 1e-12:
        s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), t_end - s.t)
        s.step_ssprk3(); s.t += s.dt; s.step += 1
    W = to_numpy(cons_to_prim(s.U, s.eos)[:, s.ng:-s.ng, s.ng:-s.ng])
    return W, s.step, time.time() - t0


def run_amr(n, t_end, ratio=4, region="all"):
    c = cfg_for(n, t_end)
    def fuel(s):                                     # the gas the liner encloses: hot spot and converging shock
        ng = s.ng
        C = xp.clip(cons_to_prim(s.U, s.eos)[iCLR, ng:-ng, ng:-ng], 0, 1)
        X, Y = s.X[ng:-ng, ng:-ng], s.Y[ng:-ng, ng:-ng]
        return to_numpy(xp.asarray(enclosed_by_tracer(C, X, Y)) & (C < 0.5))
    a = AMR(c, AMRParams(ratio=ratio, regrid_every=4, tol_rho=0.08, tol_p=0.15, margin=3, buffer=2,
                         region=fuel if region == "fuel" else None, efficiency=0.7))
    with contextlib.redirect_stdout(io.StringIO()):
        U = initial(MHDSolver(c))
    a.set_state(U)
    a.regrid()
    t0, frac, boxes = time.time(), [], []
    E0 = a.totals()
    while a.t < t_end - 1e-12:
        a.advance(min(a.compute_dt(), t_end - a.t))
        frac.append(a.fine_fraction())
        boxes.append(list(a.boxes))
    E1 = a.totals()
    Wc = {k: a.composite(v) for k, v in (("rho", iRHO), ("p", iPR), ("C", iCLR))}
    return a, Wc, frac, time.time() - t0, E0, E1, boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--t-end", dest="t_end", type=float, default=0.2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--region", choices=["all", "fuel"], default="all",
                    help="refine everywhere the flags fire (liner, shocks, core) or only in the enclosed fuel")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "amr_implosion"))
    a = ap.parse_args()
    if a.quick:
        a.n = 64
    os.makedirs(a.out, exist_ok=True)
    r = 4
    print(f"liner implosion to t = {a.t_end}: uniform {a.n}^2, uniform {r * a.n}^2, AMR {a.n}^2 + ratio-{r} patches",
          flush=True)
    Wc, nc, tc = run_uniform(a.n, a.t_end)
    print(f"  uniform {a.n}^2: {nc} steps, {tc:.0f} s", flush=True)
    Wf, nf, tf = run_uniform(r * a.n, a.t_end)
    print(f"  uniform {r * a.n}^2: {nf} steps, {tf:.0f} s", flush=True)
    amr, Wa, frac, ta, E0, E1, boxes = run_amr(a.n, a.t_end, r, a.region)
    print(f"  AMR: {amr.step} coarse steps, {ta:.0f} s, fine cells {100 * np.mean(frac):.0f}% (max {100 * max(frac):.0f}%) "
          f"of the uniform fine grid", flush=True)
    dxf = 2.0 / (r * a.n)
    up = lambda q: np.repeat(np.repeat(q, r, 0), r, 1)
    ref = metrics(Wf[iRHO], Wf[iPR], Wf[iCLR], dxf)
    m_c = metrics(up(Wc[iRHO]), up(Wc[iPR]), up(Wc[iCLR]), dxf)
    m_a = metrics(Wa["rho"], Wa["p"], Wa["C"], dxf)
    l1 = lambda q: float(np.mean(np.abs(q - Wf[iRHO])) / np.mean(Wf[iRHO]))
    E_err = abs(E1["E"] - E0["E"] - E1["boundary"]) / abs(E0["E"])
    summ = {"t_end": a.t_end, "n_coarse": a.n, "ratio": r, "reference": ref, "uniform_coarse": m_c, "amr": m_a,
            "L1_rel_density_vs_fine": {"uniform_coarse": l1(up(Wc[iRHO])), "amr": l1(Wa["rho"])},
            "fine_fraction_mean": float(np.mean(frac)), "fine_fraction_max": float(max(frac)),
            "wall_s": {"uniform_coarse": tc, "uniform_fine": tf, "amr": ta}, "amr_energy_error": E_err,
            "amr_div_B": amr.div_b(), "patches_final": len(amr.boxes), "region": a.region, "cell_um": {"coarse": 2000.0 / a.n, "fine": 2000.0 / (r * a.n)}}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    for name, m in (("uniform coarse", m_c), ("AMR", m_a), ("uniform fine", ref)):
        print(f"  {name:15s} peak core density {m['rho_core_max']:.3f}, peak core T {m['T_core_max']:.3f}, core radius "
              f"{m['R_core']:.4f}", flush=True)
    print(f"  density L1 vs fine: coarse {summ['L1_rel_density_vs_fine']['uniform_coarse']:.4f}, AMR "
          f"{summ['L1_rel_density_vs_fine']['amr']:.4f} | AMR energy {E_err:.1e}, div B {summ['amr_div_B']:.1e}",
          flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    lims = (-0.75, 0.75)
    data = [(up(Wc[iRHO]), f"uniform {a.n}$^2$ ({2000 / a.n:.1f} $\\mu$m cells)"),
            (Wa["rho"], f"AMR {a.n}$^2$ + ratio-{r} patches"), (Wf[iRHO], f"uniform {r * a.n}$^2$ ({2000 / (r * a.n):.1f} $\\mu$m)")]
    vmin, vmax = np.log10(Wf[iRHO].min()), np.log10(Wf[iRHO].max())
    for k, (q, t) in enumerate(data):
        ax[k].imshow(np.log10(q).T, origin="lower", extent=(-1, 1, -1, 1), cmap="inferno", vmin=vmin, vmax=vmax)
        ax[k].set_xlim(lims); ax[k].set_ylim(lims); ax[k].set_title(t, fontsize=10)
        ax[k].set_xlabel("x (mm)")
    dxc = 2.0 / a.n
    for (i0, i1, j0, j1) in amr.boxes:
        ax[1].add_patch(Rectangle((-1 + i0 * dxc, -1 + j0 * dxc), (i1 - i0) * dxc, (j1 - j0) * dxc, fill=False,
                                  ec="#5fd3ff", lw=1.2))
    ax[0].set_ylabel("y (mm)")
    fig.suptitle(f"log10 density at t = {a.t_end}; AMR used {100 * np.mean(frac):.0f}% of the fine cells on average",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "amr_implosion.png"), dpi=170)
    print(f"plots in {a.out}")


if __name__ == "__main__":
    main()
