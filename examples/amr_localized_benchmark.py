"""AMR against a uniform grid at the fine resolution on localized shock / interface problems (aegis/amr.py).

    bubble  a Mach-3 shock (B_x = 0.3) runs into a dense cylinder (radius 0.12, density 10) in a 2 x 1 box:
            256 x 128 coarse cells + ratio-4 patches on the shock, the bubble and its wake, against 1024 x 512
    fuel    the magnetized liner implosion of examples/amr_implosion_demo.py, refining only inside the fuel the
            liner encloses (converging shock, hot spot): 128^2 + ratio 4 against 512^2

Reported: wall time of each run, fine cells used (fraction of the uniform fine grid), density L1 error against the
uniform fine run for AMR and for the uniform coarse run. The AMR level is replayed from CUDA graphs.

    python examples/amr_localized_benchmark.py                  # both, results/amr_benchmark
    python examples/amr_localized_benchmark.py --case bubble
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
sys.path.insert(0, os.path.join(ROOT, "examples"))
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config, NVAR, iRHO, iVX, iPR, iCLR, iBX
    from aegis.physics import xp, cons_to_prim, prim_to_cons
    from aegis.solver import MHDSolver, to_numpy
    from aegis.amr import AMR, AMRParams
    from aegis.diagnostics import enclosed_by_tracer
    import amr_implosion_demo as imp

G = 5.0 / 3.0


def bubble_cfg(n, t):
    return Config(nx=2 * n, ny=n, x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0, t_end=t, cfl=0.3, mach=1.0, gamma=G,
                  B_transverse=0.3, bc_x_type="extrapolation", bc_y_type="periodic", powell_source=False,
                  material_tracer="volume_fraction", diag_interval=10**9, snapshot_times=[])


def bubble_init(s, M=3.0):
    X, Y = s.X, s.Y
    W = xp.zeros((NVAR,) + s.U.shape[1:])
    p2 = 1 + 2 * G / (G + 1) * (M**2 - 1)                     # shock jump into rho = 1, p = 1 at rest
    r2 = (G + 1) * M**2 / ((G - 1) * M**2 + 2)
    u2 = M * np.sqrt(G) * (1 - 1 / r2)
    post = X < 0.3
    bub = 0.5 * (1 - xp.tanh((xp.sqrt((X - 0.6)**2 + (Y - 0.5)**2) - 0.12) / (1.5 * s.cfg.dx)))
    W[iRHO] = xp.where(post, r2, 1.0 + 9.0 * bub)
    W[iPR] = xp.where(post, p2, 1.0)
    W[iVX] = xp.where(post, u2, 0.0)
    W[iBX] = 0.3
    W[iCLR] = bub * (~post)
    return prim_to_cons(W, s.eos)


def fuel_region(s):
    ng = s.ng
    C = xp.clip(cons_to_prim(s.U, s.eos)[iCLR, ng:-ng, ng:-ng], 0, 1)
    return to_numpy(xp.asarray(enclosed_by_tracer(C, s.X[ng:-ng, ng:-ng], s.Y[ng:-ng, ng:-ng])) & (C < 0.5))


CASES = {
    "bubble": dict(cfg=bubble_cfg, init=bubble_init, n=128, t_end=0.15, region=None, tol_rho=0.1, tol_p=0.2),
    "fuel": dict(cfg=imp.cfg_for, init=imp.initial, n=128, t_end=0.1, region=fuel_region, tol_rho=0.08, tol_p=0.15),
}


def uniform(c, m):
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(c["cfg"](m, c["t_end"]))
    s.U = c["init"](s); s.t = 0.0; s.step = 0
    xp.cuda.Device().synchronize(); t0 = time.time()
    while s.t < c["t_end"] - 1e-12:
        s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), c["t_end"] - s.t)
        s.step_ssprk3(); s.t += s.dt; s.step += 1
    xp.cuda.Device().synchronize()
    return to_numpy(cons_to_prim(s.U, s.eos)[iRHO, s.ng:-s.ng, s.ng:-s.ng]), s.step, time.time() - t0


def amr(c, r=4):
    cfg = c["cfg"](c["n"], c["t_end"])
    a = AMR(cfg, AMRParams(ratio=r, regrid_every=4, margin=3, buffer=2, efficiency=0.7, region=c["region"],
                           tol_rho=c["tol_rho"], tol_p=c["tol_p"]))
    with contextlib.redirect_stdout(io.StringIO()):
        U = c["init"](MHDSolver(cfg))
    a.set_state(U); a.regrid()
    E0, frac = a.totals(), []
    xp.cuda.Device().synchronize(); t0 = time.time()
    while a.t < c["t_end"] - 1e-12:
        a.advance(min(a.compute_dt(), c["t_end"] - a.t))
        frac.append(a.fine_fraction())
    xp.cuda.Device().synchronize()
    E1 = a.totals()
    return a, a.composite(iRHO), frac, time.time() - t0, abs(E1["E"] - E0["E"] - E1["boundary"] + E0["boundary"]) / abs(E0["E"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=["bubble", "fuel", "all"], default="all")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "amr_benchmark"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    names = list(CASES) if args.case == "all" else [args.case]
    for name in names:
        c, r = CASES[name], 4
        a, Wa, frac, ta, e_err = amr(c, r)
        print(f"{name}: AMR {ta:.0f} s, {a.step} coarse steps, fine cells mean {100 * np.mean(frac):.0f}% (max "
              f"{100 * max(frac):.0f}%), energy {e_err:.1e}, div B {a.div_b():.1e}", flush=True)
        Wf, nf, tf = uniform(c, r * c["n"])
        print(f"{name}: uniform fine {tf:.0f} s, {nf} steps", flush=True)
        Wc, nc, tc = uniform(c, c["n"])
        up = lambda q: np.repeat(np.repeat(q, r, 0), r, 1)
        l1 = lambda q: float(np.mean(np.abs(q - Wf)) / np.mean(Wf))
        summary[name] = {"wall_s": {"amr": ta, "uniform_fine": tf, "uniform_coarse": tc},
                         "amr_over_uniform_fine": ta / tf, "fine_fraction_mean": float(np.mean(frac)),
                         "fine_fraction_max": float(max(frac)), "L1_density_vs_fine": {"amr": l1(Wa), "coarse": l1(up(Wc))},
                         "amr_energy_error": e_err, "amr_div_b": a.div_b(), "coarse_steps": a.step, "fine_steps": nf,
                         "graph_launches": dict(a.graph_launches)}
        print(f"{name}: AMR / uniform fine wall time {ta / tf:.2f}; density L1 vs fine: AMR {l1(Wa):.4f}, coarse "
              f"{l1(up(Wc)):.4f}", flush=True)
        if name == "bubble":
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            fig, ax = plt.subplots(3, 1, figsize=(8, 10))
            vmin, vmax = np.log10(Wf.min()), np.log10(Wf.max())
            for k, (q, t) in enumerate(((up(Wc), f"uniform {2 * c['n']} x {c['n']}"),
                                        (Wa, f"AMR {2 * c['n']} x {c['n']} + ratio-{r} patches ({ta:.0f} s)"),
                                        (Wf, f"uniform {2 * r * c['n']} x {r * c['n']} ({tf:.0f} s)"))):
                ax[k].imshow(np.log10(q).T, origin="lower", extent=(0, 2, 0, 1), cmap="inferno", vmin=vmin, vmax=vmax)
                ax[k].set_title(t, fontsize=10)
            dxc = 1.0 / c["n"]
            for (i0, i1, j0, j1) in a.boxes:
                ax[1].add_patch(Rectangle((i0 * dxc, j0 * dxc), (i1 - i0) * dxc, (j1 - j0) * dxc, fill=False,
                                          ec="#5fd3ff", lw=0.8))
            fig.suptitle(f"Mach-3 shock on a dense bubble, log10 density at t = {c['t_end']}", x=0.01, ha="left")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, "shock_bubble.png"), dpi=150)
            plt.close(fig)
    old = {}
    path = os.path.join(args.out, "summary.json")
    if os.path.exists(path):
        old = json.load(open(path))
    old.update(summary)
    json.dump(old, open(path, "w"), indent=2)
    print(f"results in {args.out}")


if __name__ == "__main__":
    main()
