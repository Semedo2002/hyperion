"""Low-resolution 3D proof of concept: do the stripe shear layers break down in 3D?

    python experiments/3d_tearing_sweep.py            # 192 x 48 x 24 (h = 0.042), t = 0.6, ~20 min
    python experiments/3d_tearing_sweep.py --quick    # 128 x 32 x 16, smoke test

Questions and the diagnostic that answers each, always against seeded 3D hydro:
  3D breakdown     growth of E_kz = int rho v_z^2/2 and of the z-roughness of the interface
  dynamo           growth of E_perp = int (B_y^2 + B_z^2)/2 (field generated across B_x)
  current sheets   max|J|, volume fraction with |J| > 5 J_rms, vs the z-invariant control
  mixing           mixing-layer enstrophy and interface roughness
Results: experiments/data/exp_3d.npz and exp_3d.png.
"""
import os
import sys
import io
import time
import argparse
import contextlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config
    from aegis.solver3d import MHDSolver3D

DATA = os.path.join(ROOT, "experiments", "data")

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()

nx, ny, nz = (128, 32, 16) if args.quick else (192, 48, 24)
Lx, Ly, Lz, t_end = 8.0, 2.0, 1.0, (0.3 if args.quick else 0.6)
h = Lx / nx
SEED = dict(z_pert_vel=0.01, z_pert_amp=0.005, z_pert_mode=4, z_pert_seed=1)
BASE = dict(div_b="glm", nx=nx, ny=ny, nz=nz, x_max=Lx, y_max=Ly, z_max=Lz, t_end=t_end, cfl=0.3, mach=10.0,
            interface_x=1.5, perturbation_amp=0.02, perturbation_mode=4, density_ratio=3.0,
            bc_x_type="inflow_outflow", B_modulation_mode=4, diag_interval=10)
DEEP = dict(B_field_type="striped", B_transverse=6.0)               # field-free gaps: B_x reverses
CASES = {
    "hydro 3D": dict(B_transverse=0.0, **SEED),
    "deep, z-invariant": dict(**DEEP, nz=4, z_max=4 * h),           # quasi-2D control, no seed
    "deep (eps=0) 3D": dict(**DEEP, **SEED),
    "deep (eps=0.3) 3D": dict(**DEEP, B_gap_fraction=0.3, **SEED),  # no field reversal
    "uniform B=3 3D": dict(B_field_type="striped", B_transverse=3.0, B_gap_fraction=0.99, **SEED),
}

out, snaps = {}, {}
for name, c in CASES.items():
    s = MHDSolver3D(Config(**{**BASE, **c}))
    s.initialize()
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        d = s.run(snapshot_times=(t_end,), verbose=False)
    d["energy_error"] = s.energy_error()
    out[name] = d
    snaps[name] = s.snapshots.get(f"t={t_end:.3f}", {})
    print(f"[3d] {name:<20} {s.step} steps, {time.time() - t0:.0f}s, energy error {d['energy_error']:.1e}", flush=True)
    del s

os.makedirs(DATA, exist_ok=True)
np.savez(os.path.join(DATA, "exp_3d.npz"),
         **{f"{n}__{k}": np.asarray(v) for n, d in out.items() for k, v in d.items()},
         **{f"{n}__snap_{k}": np.asarray(v) for n, sn in snaps.items() for k, v in sn.items()})


def growth(d, key, t0=0.1):
    """growth factor of key between t0 and the end, and the late exponential rate"""
    t, v = d["t"], np.maximum(d[key], 1e-300)
    i0 = np.argmin(np.abs(t - t0))
    late = t > t[-1] * 0.6
    rate = np.polyfit(t[late], np.log(v[late]), 1)[0] if late.sum() > 2 else np.nan
    return v[-1] / v[i0], rate


hyd = out["hydro 3D"]
print(f"\n[3d] {'case':<20}{'Ekz x':>9}{'Ekz rate':>9}{'Eperp':>10}{'Eperp x':>9}{'J_max':>8}"
      f"{'sheets':>8}{'rough/h':>8}{'rz/rough':>9}{'emix/h':>8}{'divB':>7}")
for name, d in out.items():
    gk, rk = growth(d, "Ekz")
    gp, _ = growth(d, "Eperp")
    print(f"[3d] {name:<20}{gk:9.2g}{rk:9.2f}{d['Eperp'][-1]:10.2e}{gp:9.2g}{d['J_max'][-1]:8.1f}"
          f"{d['sheet_frac'][-1]:8.4f}{d['rough'][-1] / hyd['rough'][-1]:8.2f}"
          f"{d['rough_z'][-1] / max(d['rough'][-1], 1e-12):9.3f}{d['emix'][-1] / hyd['emix'][-1]:8.2f}"
          f"{d['divB_rel'][-1]:7.3f}")

# figure: time series + mid-plane |J|
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
cols = {"hydro 3D": "#222222", "deep, z-invariant": "#8a8a8a", "deep (eps=0) 3D": "#c0392b",
        "deep (eps=0.3) 3D": "#8e6cbf", "uniform B=3 3D": "#2e8b57"}
fig, ax = plt.subplots(2, 3, figsize=(10, 5.6))
for name, d in out.items():
    kw = dict(color=cols[name], lw=1.3, label=name)
    ax[0, 0].semilogy(d["t"], np.maximum(d["Ekz"], 1e-12), **kw)
    ax[0, 1].semilogy(d["t"], np.maximum(d["Eperp"], 1e-12), **kw)
    ax[0, 2].plot(d["t"], d["J_max"], **kw)
    ax[1, 0].plot(d["t"], d["rough"], **kw)
    ax[1, 1].semilogy(d["t"], np.maximum(d["emix"], 1e-6), **kw)
for a, ttl in zip(ax.flat, (r"(a) $E_{kz}=\int\rho v_z^2/2$", r"(b) $E_\perp=\int(B_y^2+B_z^2)/2$",
                            r"(c) max $|\mathbf{J}|$", "(d) interface roughness", "(e) mixing-layer enstrophy")):
    a.set_title(ttl, loc="left", fontsize=9); a.set_xlabel("$t$")
ax[0, 0].legend(frameon=False, fontsize=7)
sn = snaps.get("deep (eps=0) 3D", {})
if "J_yz" in sn:
    im = ax[1, 2].imshow(sn["J_yz"].T, origin="lower", extent=[0, Ly, 0, Lz], cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax[1, 2], label=r"$|\mathbf{J}|$")
    ax[1, 2].set_title(r"(f) $|\mathbf{J}|$ in the mixing layer ($y$-$z$ cut)", loc="left", fontsize=9)
    ax[1, 2].set_xlabel("$y$"); ax[1, 2].set_ylabel("$z$")
fig.tight_layout()
fig.savefig(os.path.join(DATA, "exp_3d.png"), dpi=200)
print(f"\nsaved {os.path.join(DATA, 'exp_3d.npz')} and exp_3d.png")
