"""Step time of the tabulated liner EOS against the analytic Mie-Gruneisen liner it tabulates.

A 2D aluminium plate impact (2 km/s, g/cc - km/s - GPa units, liner fraction 1 everywhere except a gas gap)
runs N steps with each EOS after a warm-up; reported: ms per step and the ratio.

    python scripts/bench_tabular.py [--n 256] [--steps 40]
"""
import os
import sys
import time
import argparse
import tempfile
import contextlib
import io

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config, NVAR, iRHO, iVX, iPR, iCLR
    from aegis.physics import xp, prim_to_cons, cons_to_prim
    from aegis.solver import MHDSolver
    from aegis import tabular


def make(n, table):
    mg = dict(eos_liner="mie_gruneisen", mg_rho0=2.7, mg_c0=5.328, mg_s=1.338, mg_gamma0=2.0)
    cfg = Config(nx=n, ny=n, x_max=1.0, y_max=1.0, t_end=1.0, cfl=0.3, mach=1.0, gamma=1.4, B_transverse=0.0,
                 bc_x_type="extrapolation", bc_y_type="periodic", material_tracer="volume_fraction", eos_table=table,
                 tab_rho_unit=1000.0, tab_v_unit=1000.0, diag_interval=10**9, snapshot_times=[], **mg)
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    W = xp.zeros((NVAR,) + s.U.shape[1:])
    gap = (s.X > 0.45) & (s.X < 0.5)                 # a thin gas gap: mixed cells and pure cells both present
    W[iRHO] = xp.where(gap, 0.01, 2.7)
    W[iPR] = 1e-4
    W[iCLR] = xp.where(gap, 0.0, 1.0)
    W[iVX] = xp.where(s.X < 0.45, 2.0 * (1 + 0.05 * xp.sin(6.283 * s.Y)), 0.0)
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    return s


def run(s, steps):
    for k in range(steps):
        s.dt = s.compute_dt(cons_to_prim(s.U, s.eos))
        s.step_ssprk3(); s.t += s.dt; s.step += 1
    xp.cuda.Stream.null.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=40)
    a = ap.parse_args()
    path = os.path.join(tempfile.mkdtemp(prefix="bench_tab_"), "al.npz")
    np.savez(path, **tabular.mie_gruneisen_table(2700.0, 5328.0, 1.338, 2.0, 900.0, (2000.0, 4500.0), (10.0, 4e4),
                                                 80, 80))
    res = {}
    for name, table in (("analytic", ""), ("tabular", path)):
        s = make(a.n, table)
        run(s, 5)
        t0 = time.perf_counter()
        run(s, a.steps)
        res[name] = (time.perf_counter() - t0) / a.steps * 1e3
        res[name + "_p"] = float(cons_to_prim(s.U, s.eos)[iPR].max())
        print(f"  {name:9s} {res[name]:8.2f} ms/step  (max p {res[name + '_p']:.4f} GPa after {s.step} steps)",
              flush=True)
    print(f"  {a.n}^2: tabular / analytic = {res['tabular'] / res['analytic']:.2f}")


if __name__ == "__main__":
    main()
