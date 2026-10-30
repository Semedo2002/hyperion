"""3D sweep 2: wide field-reversing stripes, explicit resistivity, seed scan, resolution check.

    python experiments/3d_tearing_sweep2.py          # ~50 min on one GPU
    python experiments/3d_tearing_sweep2.py --quick  # base grid halved, seed scan only

Set-up: x stripes with field-free gaps (eps = 0, so B_x reverses at the gaps: two current sheets
per domain), B_modulation_mode = 1 (|B| period 1.0), B0 = 6, eta = 0.02. With the post-shock bar
Alfven speed v_A ~ 3 and half-spacing L = 0.5 of the reversal layers, S = v_A L / eta ~ 75.
Base grid 192 x 48 x 24 (h = 0.042, 24 cells per stripe period); resolution check 256 x 64 x 32.

Classification, fixed before running:
  instability   E_kz grows after the seed transient at a seed-independent rate, or saturates at a
                seed-independent level
  driven        E_kz and the 3D enstrophy excess scale like seed^2 and do not grow
  tearing       current sheets acquire z-structure (J_zvar) and exceed the z-invariant control
Results: experiments/data/exp_3d2.npz, exp_3d2.png.
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

ETA, B0, L_SHEET = 0.02, 6.0, 0.5
Lx, Ly, Lz, T_END = 8.0, 2.0, 1.0, 0.6
SEEDS = (0.003, 0.01, 0.03)
BASE = dict(div_b="glm", x_max=Lx, y_max=Ly, z_max=Lz, t_end=T_END, cfl=0.3, mach=10.0, interface_x=1.5,
            perturbation_amp=0.02, perturbation_mode=4, density_ratio=3.0, bc_x_type="inflow_outflow",
            diag_interval=10, B_modulation_mode=1)
STRIPES = dict(B_field_type="striped", B_transverse=B0, eta_resistivity=ETA)
# both seeds scale together (interface z-ripple = 0.5 v_z), so the scan really varies the seed amplitude
seed = lambda v: dict(z_pert_vel=v, z_pert_amp=0.5 * v, z_pert_mode=4, z_pert_seed=1)


def grid(nx, ny, nz):
    return dict(nx=nx, ny=ny, nz=nz)


lo = grid(96, 24, 12) if args.quick else grid(192, 48, 24)
hi = grid(256, 64, 32)
h_lo = Lx / lo["nx"]
h_hi = Lx / hi["nx"]
CASES = {
    "hydro": {**lo, "B_transverse": 0.0, **seed(0.01)},
    "z-invariant": {**lo, **STRIPES, "nz": 4, "z_max": 4 * h_lo},
    **{f"seed {v:g}": {**lo, **STRIPES, **seed(v)} for v in SEEDS},
}
if not args.quick:
    CASES.update({
        "hydro hi": {**hi, "B_transverse": 0.0, **seed(0.01)},
        "z-invariant hi": {**hi, **STRIPES, "nz": 4, "z_max": 4 * h_hi},
        "seed 0.01 hi": {**hi, **STRIPES, **seed(0.01)},
    })

out, snaps = {}, {}
for name, c in CASES.items():
    s = MHDSolver3D(Config(**{**BASE, **c}))
    s.initialize()
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        d = s.run(snapshot_times=(T_END,), verbose=False)
    d["energy_error"] = s.energy_error()
    out[name] = d
    snaps[name] = s.snapshots.get(f"t={T_END:.3f}", {})
    if name == "z-invariant":
        eq = s.equilibrium
    print(f"[3d2] {name:<16} {s.step} steps {time.time() - t0:.0f}s  energy error {d['energy_error']:.1e}", flush=True)
    del s

os.makedirs(DATA, exist_ok=True)
np.savez(os.path.join(DATA, "exp_3d2.npz"),
         **{f"{n}__{k}": np.asarray(v) for n, d in out.items() for k, v in d.items()},
         **{f"{n}__snap_{k}": np.asarray(v) for n, sn in snaps.items() for k, v in sn.items()})


def late_rate(d, key, frac=0.5):
    """exponential rate of key over the last half of the run"""
    t, v = d["t"], np.maximum(d[key], 1e-300)
    m = t > t[-1] * (1 - frac)
    return np.polyfit(t[m], np.log(v[m]), 1)[0]


rA = 2 * B0 / np.sqrt(3.88)   # rough post-shock bar Alfven speed scale (B unchanged, rho x r)
print(f"\n[3d2] eta = {ETA}, Lundquist S ~ v_A L / eta = {B0 / np.sqrt(3.88) * L_SHEET / ETA:.0f}")
print(f"[3d2] {'case':<16}{'Ekz end':>10}{'Ekz/seed^2':>11}{'Ekz rate':>9}{'emix':>9}{'emix-ctrl':>10}"
      f"{'J_max':>8}{'J_zvar':>8}{'sheets':>8}{'Eperp':>10}{'rough':>8}{'E err':>9}")
for name, d in out.items():
    v = CASES[name].get("z_pert_vel", 0.0)
    ctrl = out["z-invariant hi" if name.endswith("hi") else "z-invariant"]
    print(f"[3d2] {name:<16}{d['Ekz'][-1]:10.2e}{(d['Ekz'][-1] / v**2 if v else np.nan):11.3g}"
          f"{late_rate(d, 'Ekz') if v else np.nan:9.2f}{d['emix'][-1]:9.3g}{d['emix'][-1] - ctrl['emix'][-1]:10.3g}"
          f"{d['J_max'][-1]:8.1f}{d['J_zvar'][-1]:8.3f}{d['sheet_frac'][-1]:8.4f}{d['Eperp'][-1]:10.2e}"
          f"{d['rough'][-1]:8.4f}{d['energy_error']:9.1e}")

# seed scaling of the 3D excess: slope of log(quantity) against log(seed); 2 means driven (quadratic)
vs = np.array(SEEDS)
ekz = np.array([out[f"seed {v:g}"]["Ekz"][-1] for v in SEEDS])
exc = np.array([out[f"seed {v:g}"]["emix"][-1] - out["z-invariant"]["emix"][-1] for v in SEEDS])
p_ekz = np.polyfit(np.log(vs), np.log(ekz), 1)[0]
p_exc = np.polyfit(np.log(vs), np.log(np.maximum(np.abs(exc), 1e-30)), 1)[0]
print(f"[3d2] seed scaling exponents: E_kz ~ seed^{p_ekz:.2f}, emix excess ~ seed^{p_exc:.2f} "
      f"(2 = linear driven response; ~0 = saturated instability)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
col = {"hydro": "#222222", "z-invariant": "#8a8a8a", "seed 0.003": "#f0a04b", "seed 0.01": "#c0392b",
       "seed 0.03": "#7b1e1e", "hydro hi": "#222222", "z-invariant hi": "#8a8a8a", "seed 0.01 hi": "#1f5f99"}
fig, ax = plt.subplots(2, 3, figsize=(10.5, 5.8))
for name, d in out.items():
    v = CASES[name].get("z_pert_vel", 0.0)
    kw = dict(color=col[name], lw=1.3, ls="--" if name.endswith("hi") else "-", label=name)
    if v:
        ax[0, 0].semilogy(d["t"], np.maximum(d["Ekz"] / v**2, 1e-12), **kw)
    ax[0, 1].semilogy(d["t"], np.maximum(d["emix"], 1e-6), **kw)
    ax[0, 2].plot(d["t"], d["J_max"], **kw)
    ax[1, 0].plot(d["t"], d["J_zvar"], **kw)
    ax[1, 1].semilogy(d["t"], np.maximum(d["Eperp"], 1e-12), **kw)
titles = (r"(a) $E_{kz}$ / seed$^2$", "(b) mixing-layer enstrophy", r"(c) max $|\mathbf{J}|$",
          r"(d) $z$-variation of $|\mathbf{J}|$ / $J_{rms}$", r"(e) $E_\perp=\int(B_y^2+B_z^2)/2$")
for a, ttl in zip(ax.flat, titles):
    a.set_title(ttl, loc="left", fontsize=9); a.set_xlabel("$t$")
ax[0, 1].legend(frameon=False, fontsize=6.5)
sn = snaps.get("seed 0.01 hi" if "seed 0.01 hi" in snaps else "seed 0.01", {})
if "J_yz" in sn:
    im = ax[1, 2].imshow(sn["J_yz"].T, origin="lower", extent=[0, Ly, 0, Lz], cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax[1, 2], label=r"$|\mathbf{J}|$")
    ax[1, 2].set_title(r"(f) $|\mathbf{J}|$, $y$-$z$ cut at the interface", loc="left", fontsize=9)
    ax[1, 2].set_xlabel("$y$"); ax[1, 2].set_ylabel("$z$")
fig.tight_layout()
fig.savefig(os.path.join(DATA, "exp_3d2.png"), dpi=200)
print(f"[3d2] saved exp_3d2.npz and exp_3d2.png")
