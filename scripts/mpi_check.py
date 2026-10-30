"""Domain-decomposition check: run under mpiexec, compare with a single-domain run on rank 0.

    mpiexec -n 2 python scripts/mpi_check.py
    mpiexec -n 3 python scripts/mpi_check.py --quick

Cases: 2D Orszag-Tang with resistivity (periodic x), 2D stiffened Al with strength and spall (zero-gradient
x ends), 3D Orszag-Tang-like field in a periodic box. Each must match the single-domain solver to round-off,
with div B at round-off and the same total energy. Prints one line per case and MPI_CHECK PASS/FAIL.
"""
import os
import sys
import argparse
import contextlib
import io

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mpi4py import MPI  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config, NVAR, EN, iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iCLR
    from aegis.physics import xp, prim_to_cons, cons_to_prim
    from aegis.solver import MHDSolver, to_numpy
    from aegis.solver3d import MHDSolver3D
    from aegis.parallel import distributed
    from aegis.eos import stiffened_from_hugoniot

comm = MPI.COMM_WORLD
TAU = 2 * np.pi


def ot2d(s):
    X, Y = s.X, s.Y
    W = xp.zeros((NVAR,) + s.U.shape[1:])
    W[iRHO], W[iPR] = 25 / (36 * np.pi), 5 / (12 * np.pi)
    W[iVX], W[iVY] = -xp.sin(TAU * Y), xp.sin(TAU * X)
    B0 = 1 / np.sqrt(4 * np.pi)
    W[iBX], W[iBY] = -B0 * xp.sin(TAU * Y), B0 * xp.sin(2 * TAU * X)
    return W


def metal2d(s):
    X, Y = s.X, s.Y
    W = xp.zeros((NVAR,) + s.U.shape[1:])
    r2 = (X - 0.5)**2 + (Y - 0.45)**2
    W[iRHO], W[iPR], W[iCLR] = 2.7, 1e-4, 1.0
    W[iVX] = 0.5 * (X - 0.5) / 0.1 * xp.exp(-r2 / 0.01)
    W[iVY] = 0.5 * (Y - 0.45) / 0.1 * xp.exp(-r2 / 0.01)
    return W


def ot3d(s):
    X = s.x[:, None, None]; Y = s.y[None, :, None]; Z = s.z[None, None, :]
    W = xp.zeros((NVAR,) + s.U.shape[1:])
    o = xp.ones(s.U.shape[1:])
    W[iRHO], W[iPR] = 25 / (36 * np.pi) * o, 5 / (12 * np.pi) * o
    W[iVX] = -xp.sin(TAU * Y) * o + 0.2 * xp.sin(TAU * Z)
    W[iVY] = xp.sin(TAU * X) * o
    W[iVZ] = 0.2 * xp.sin(TAU * X) * xp.cos(TAU * Y) * o
    B0 = 1 / np.sqrt(4 * np.pi)
    W[iBX] = -B0 * xp.sin(TAU * Y) * xp.cos(TAU * Z) * o      # each component independent of its own axis
    W[iBY] = B0 * xp.sin(2 * TAU * X) * o + 0.1 * B0 * xp.sin(TAU * Z)
    W[iBZ] = 0.2 * B0 * xp.cos(TAU * X) * xp.sin(TAU * Y) * o
    return W


def run(cls, cfg, init, t_end, dist):
    with contextlib.redirect_stdout(io.StringIO()):
        s = distributed(cls, cfg) if dist else cls(cfg)
    s.U = prim_to_cons(init(s), s.eos)
    s.t, s.step = 0.0, 0
    while s.t < t_end - 1e-14:
        s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), t_end - s.t)
        s.step_ssprk3(); s.t += s.dt; s.step += 1
    return s


def interior(s):
    ng = s.ng
    sl = (slice(None),) + (slice(ng, -ng),) * (s.U.ndim - 1)
    return cons_to_prim(s.U, s.eos)[sl], s.U[EN][sl[1:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n = 48 if a.quick else 96
    gA, piA = stiffened_from_hugoniot(2.7, 5.328, 1.338, 1.0)
    base = dict(t_end=1.0, cfl=0.3, mach=1.0, B_transverse=0.0, powell_source=False, bc_y_type="periodic",
                diag_interval=10**9, snapshot_times=[])
    cases = [
        ("2D Orszag-Tang, eta 1e-3, periodic x", MHDSolver, ot2d, 0.1,
         dict(nx=n, ny=n // 2, x_max=1.0, y_max=1.0, gamma=5 / 3, B_transverse=1.0, bc_x_type="periodic",
              eta_resistivity=1e-3)),
        ("2D Al, strength + spall, open x", MHDSolver, metal2d, 0.02,
         dict(nx=n, ny=n // 2, x_max=1.0, y_max=1.0, bc_x_type="extrapolation", eos_liner="stiffened",
              gamma_liner=gA, p_inf_liner=piA, material_tracer="volume_fraction", strength=True, shear_modulus=26.0,
              yield_strength=0.29, spall_strength=1.2)),
        ("3D Orszag-Tang-like, periodic", MHDSolver3D, ot3d, 0.05,
         dict(nx=n // 2, ny=n // 4, nz=n // 4, x_max=1.0, y_max=1.0, z_max=1.0, gamma=5 / 3, B_transverse=1.0,
              bc_x_type="periodic")),
    ]
    ok = True
    for name, cls, init, t_end, kw in cases:
        cfg = Config(**{**base, **kw})
        sd = run(cls, cfg, init, t_end, True)
        Wd, Ed = interior(sd)
        Wg = sd.gather_x(Wd, ax=1)
        Etot = sd.global_sum(xp.sum(Ed))
        divd = sd.div_b_max()
        steps = sd.step
        if comm.rank == 0:
            ss = run(cls, cfg, init, t_end, False)
            Ws, Es = interior(ss)
            Ws = to_numpy(Ws)
            keys = [iRHO, iPR, iVX, iVY, iBX, iBY, iBZ] + ([iCLR] if "Al" in name else [])
            d = max(float(np.max(np.abs(Wg[k] - Ws[k]))) / max(float(np.max(np.abs(Ws[k]))), 1e-300) for k in keys
                    if np.max(np.abs(Ws[k])) > 0)
            dE = abs(Etot - float(xp.sum(Es))) / abs(float(xp.sum(Es)))
            good = d < 1e-11 and dE < 1e-12 and divd < 1e-12 and steps == ss.step
            ok = ok and good
            print(f"  {name}: {comm.size} ranks vs 1, {steps}/{ss.step} steps, max rel diff {d:.1e}, total energy "
                  f"{dE:.1e}, div B {divd:.1e} (halo {sd.halo.backend})  [{'PASS' if good else 'FAIL'}]", flush=True)
    ok = comm.bcast(ok, root=0)
    if comm.rank == 0:
        print("MPI_CHECK", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
