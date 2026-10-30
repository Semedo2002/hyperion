"""3D MHD implosion of a CAD liner, straight from the file onto the grid (aegis/cad_importer.py).

The target is read from a CAD file (.stl here; .step / .stp / .iges / .igs through gmsh), and its solid volume
fraction on every cell of a 3D Cartesian grid becomes the liner: density, material tracer and inward velocity.
No mesh generation, no manual masks. By default the file is examples/data/corrugated_liner.stl, a cylindrical
shell with an m = 4 azimuthal by k = 1 axial corrugation of its inner surface (written by --make-stl). An
optional central rod (--rod-radius) is a second CAD body treated as a rigid immersed wall (aperture-weighted
HLLD fluxes).

The liner coasts inward (v ~ 1/r at speed v0 mid-wall) and compresses the axial seed field B0 in a light fill;
the corrugation seeds an m = 4 perturbation of the inner surface whose growth is tracked. The domain is periodic
in z (the liner's axial period), open in x and y.

    python examples/cad_liner_implosion.py                       # 64 x 64 x 32
    python examples/cad_liner_implosion.py --rod-radius 0.15     # with a rigid central rod
    python examples/cad_liner_implosion.py --cad my_target.step  # any closed CAD solid, axis along z
    python examples/cad_liner_implosion.py --quick
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
    from aegis.config import Config, NVAR, iRHO, iVX, iVY, iPR, iBZ, iCLR
    from aegis.physics import xp, prim_to_cons, cons_to_prim
    from aegis.solver3d import MHDSolver3D
from aegis import cad_importer as ci

DEFAULT_STL = os.path.join(ROOT, "examples", "data", "corrugated_liner.stl")


def make_stl(path):
    tris = ci.corrugated_liner_mesh(r_in=1.0, thickness=0.2, length=2.0, m=4, kz=1, eps=0.06, n_theta=128, n_z=48,
                                    overhang=0.1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ci.write_stl(path, tris, name="corrugated_liner")
    return tris


def inner_radius(s, C, z_index):
    """inner liner radius against theta on one z slice: the first radius where the tracer crosses 0.5"""
    ng = s.ng
    x, y = s.x[ng:-ng].get(), s.y[ng:-ng].get()
    c = C[ng:-ng, ng:-ng, z_index]
    th = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    r = np.linspace(0.02, 1.5, 600)
    X = r[None, :] * np.cos(th)[:, None]
    Y = r[None, :] * np.sin(th)[:, None]
    i = np.clip((X - x[0]) / (x[1] - x[0]), 0, len(x) - 1.001)
    j = np.clip((Y - y[0]) / (y[1] - y[0]), 0, len(y) - 1.001)
    i0, j0 = i.astype(int), j.astype(int)
    a, b = i - i0, j - j0
    v = (1 - a) * (1 - b) * c[i0, j0] + a * (1 - b) * c[i0 + 1, j0] + (1 - a) * b * c[i0, j0 + 1] + a * b * c[i0 + 1, j0 + 1]
    k = np.argmax(v > 0.5, axis=1)
    return th, r[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cad", default=DEFAULT_STL)
    ap.add_argument("--make-stl", action="store_true", help="(re)write the default corrugated liner STL")
    ap.add_argument("--n", type=int, default=64, help="cells across x and y")
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--half-width", type=float, default=1.6)
    ap.add_argument("--length", type=float, default=2.0, help="axial period (domain length in z)")
    ap.add_argument("--rho-liner", type=float, default=1.0)
    ap.add_argument("--rho-gas", type=float, default=0.05)
    ap.add_argument("--p0", type=float, default=0.01)
    ap.add_argument("--b0", type=float, default=0.1)
    ap.add_argument("--v0", type=float, default=1.0)
    ap.add_argument("--rod-radius", type=float, default=0.0)
    ap.add_argument("--t-end", type=float, default=0.75)
    ap.add_argument("--sub", type=int, default=4, help="ray samples per cell and direction")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cad_liner"))
    a = ap.parse_args()
    if a.quick:
        a.n, a.nz, a.t_end, a.sub = 32, 16, 0.3, 2
    os.makedirs(a.out, exist_ok=True)
    if a.make_stl or (a.cad == DEFAULT_STL and not os.path.exists(DEFAULT_STL)):
        make_stl(DEFAULT_STL)
    t0 = time.time()
    tris = ci.read_cad(a.cad)
    L, H = a.length, a.half_width
    cfg = Config(nx=a.n, ny=a.n, nz=a.nz, x_min=-H, x_max=H, y_min=-H, y_max=H, z_min=0.0, z_max=L, cfl=0.3,
                 gamma=5 / 3, bc_x_type="extrapolation", bc_y_type="extrapolation", powell_source=False,
                 t_end=a.t_end, diag_interval=10, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver3D(cfg)
    edges = ci.solver_edges(s)
    alpha = ci.Geometry(tris, edges, sub=a.sub).volume_fraction()            # liner fraction per cell
    t_geo = time.time() - t0
    print(f"{os.path.basename(a.cad)}: {len(tris)} triangles, enclosed volume {ci.mesh_volume(tris):.4f}; on the "
          f"grid {alpha.sum() * cfg.dx * cfg.dy * cfg.dz:.4f} (inside the domain) in {t_geo:.1f}s")
    ng = s.ng
    A = np.zeros((s.nx_tot, s.ny_tot, s.nz_tot))
    A[ng:-ng, ng:-ng, ng:-ng] = alpha
    A = xp.asarray(A)
    X = s.x[:, None, None] * xp.ones((1, s.ny_tot, s.nz_tot))
    Y = s.y[None, :, None] * xp.ones((s.nx_tot, 1, s.nz_tot))
    R = xp.sqrt(X**2 + Y**2) + 1e-12
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot, s.nz_tot))
    W[iRHO] = a.rho_gas + (a.rho_liner - a.rho_gas) * A
    vr = -a.v0 * 1.1 / R * A                                               # v ~ 1/r, v0 at the mid-wall radius
    W[iVX], W[iVY] = vr * X / R, vr * Y / R
    W[iPR] = a.p0
    W[iBZ] = a.b0
    W[iCLR] = A
    walls = None
    if a.rod_radius > 0:                                                   # rigid rod: an immersed wall
        rod = ci.cylinder_mesh(r=a.rod_radius, length=L, overhang=0.1)
        walls = ci.ImmersedWalls.from_geometry(s, ci.Geometry(rod, edges, sub=a.sub))
        s.walls = walls
    s.U = prim_to_cons(W, cfg.gamma)
    zmid = a.nz // 4                                                       # z = L/8: the corrugation's crest
    th, r0 = inner_radius(s, cons_to_prim(s.U, s.eos)[iCLR].get(), ng + zmid)
    t1 = time.time()
    hist = []
    snaps = []
    targets = list(np.linspace(0, a.t_end, 6)[1:])
    while s.t < a.t_end - 1e-12:
        s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), a.t_end - s.t)
        s.step_ssprk3()
        s.t += s.dt
        s.step += 1
        if s.step % 5 == 0 or s.t >= a.t_end - 1e-12:
            C = cons_to_prim(s.U, s.eos)[iCLR].get()
            th, rr = inner_radius(s, C, ng + zmid)
            m4 = 2 * abs(np.mean(rr * np.exp(-4j * th)))
            hist.append(dict(t=s.t, r_mean=float(rr.mean()), m4=float(m4), E=s.total_energy(), divB=float(s.div_b_max())))
        if targets and s.t >= targets[0] - 1e-12:
            targets.pop(0)
            snaps.append((s.t, cons_to_prim(s.U, s.eos)[iRHO].get()[ng:-ng, ng:-ng, ng:-ng].copy()))
    wall = time.time() - t1
    m4_0 = 2 * abs(np.mean(r0 * np.exp(-4j * th)))
    print(f"{s.step} steps in {wall:.0f}s; inner radius {r0.mean():.3f} -> {hist[-1]['r_mean']:.3f}, m = 4 amplitude "
          f"{m4_0:.4f} -> {hist[-1]['m4']:.4f}; max |div B| {max(h['divB'] for h in hist):.1e}")
    summary = dict(cad=os.path.relpath(a.cad, ROOT), triangles=int(len(tris)), mesh_volume=ci.mesh_volume(tris),
                   geometry_seconds=t_geo, steps=s.step, run_seconds=wall, r_inner_initial=float(r0.mean()),
                   m4_initial=float(m4_0), history=hist, args=vars(a))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    plot(a, s, snaps, hist, m4_0)


def plot(a, s, snaps, hist, m4_0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H = a.half_width
    fig, ax = plt.subplots(2, 4, figsize=(16, 7.5))
    for k, (t, rho) in enumerate(snaps[:3] if len(snaps) >= 3 else snaps):
        kz = rho.shape[2] // 4
        ax[0, k].imshow(rho[:, :, kz].T, origin="lower", extent=[-H, H, -H, H], cmap="magma")
        ax[0, k].set_title(f"density, z = L/8, t = {t:.2f}")
        ax[1, k].imshow(rho[:, rho.shape[1] // 2, :].T, origin="lower", extent=[-H, H, 0, a.length], cmap="magma",
                        aspect="auto")
        ax[1, k].set_title(f"density, y = 0, t = {t:.2f}")
    t = [h["t"] for h in hist]
    ax[0, 3].plot(t, [h["r_mean"] for h in hist])
    ax[0, 3].set_xlabel("t"); ax[0, 3].set_title("mean inner radius (z = L/8)")
    ax[1, 3].plot([0] + t, np.array([m4_0] + [h["m4"] for h in hist]) / max(m4_0, 1e-12))
    ax[1, 3].set_xlabel("t"); ax[1, 3].set_title("m = 4 amplitude / initial")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "cad_liner_implosion.png"), dpi=120)
    print(f"wrote {os.path.join(a.out, 'cad_liner_implosion.png')}")


if __name__ == "__main__":
    main()
