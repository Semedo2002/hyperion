"""3D sweep 3: interface frame, long residence time for the current sheets.

    python experiments/3d_tearing_sweep3.py          # 240 x 48 x 24 to t = 5, ~1.5 h
    python experiments/3d_tearing_sweep3.py --quick  # 120 x 24 x 12 to t = 1.5, smoke test

The frame moves with the interface (U measured from a lab-frame run of the same set-up: 6.49 with
stripes, 7.16 hydro), so the shocked heavy gas behind the interface is nearly at rest and the
field-reversal current sheets in it stay in the domain. Interface at x = 4 on a 10-long domain keeps
the subsonic left inflow away from it. eta = 0.02 (S ~ 75), B_modulation_mode = 1, B0 = 6, eps = 0.

Expected tearing rate in the heavy gas: half-thickness a ~ 0.3, v_A = B0/sqrt(rho_heavy) ~ 1.75,
S_a = a v_A / eta ~ 26, gamma ~ S_a^(-1/2) v_A / a ~ 1.1 per unit time, i.e. ~5 e-foldings by t = 5
(resistive spreading of the sheet, a^2 / eta ~ 4.5, competes with it).

Boundaries: x_min_bc = x_max_bc = "plateau". Each end supplies the inflow state until the outgoing
shock (reflected at x_min, transmitted at x_max) has left, then the shocked plateau behind it, i.e. the
true exterior state. Fixed inflow states reflect those shocks (interface reversal at t ~ 3 in the
first attempt); zero-gradient drains the subsonic shocked gas. A gate stops the sweep if the control's
interface drifts by more than 0.25 over the run.

Tearing signature: |J| along the sheets breaks up along x (J_xmod grows) in the z-invariant control
as well (tearing is a 2D mode, k along B); a 3D effect is anything the seeded run adds beyond it.
Results: experiments/data/exp_3d3.npz, exp_3d3.png.
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

ETA, B0, U_MHD, U_HYD = 0.02, 6.0, 6.49, 7.16
Lx, Ly, Lz = 10.0, 2.0, 1.0
nx, ny, nz = (120, 24, 12) if args.quick else (240, 48, 24)
T_END = 1.5 if args.quick else 5.0
h = Lx / nx
BASE = dict(div_b="glm", nx=nx, ny=ny, nz=nz, x_max=Lx, y_max=Ly, z_max=Lz, t_end=T_END, cfl=0.3, mach=10.0,
            interface_x=4.0, perturbation_amp=0.02, perturbation_mode=4, density_ratio=3.0,
            bc_x_type="inflow_outflow", diag_interval=20, B_modulation_mode=1,
            x_min_bc="plateau", x_max_bc="plateau")
STRIPES = dict(B_field_type="striped", B_transverse=B0, eta_resistivity=ETA, frame_velocity=U_MHD)
SEED = dict(z_pert_vel=0.01, z_pert_amp=0.005, z_pert_mode=4, z_pert_seed=1)
CASES = {                                       # control first: it gates the expensive runs
    "z-invariant": dict(**STRIPES, nz=4, z_max=4 * h),
    "stripes 3D": dict(**STRIPES, **SEED),
    "hydro 3D": dict(B_transverse=0.0, frame_velocity=U_HYD, **SEED),
}
SNAP_T = tuple(np.round(np.linspace(T_END / 5, T_END, 5), 3))

out, snaps = {}, {}
for name, c in CASES.items():
    s = MHDSolver3D(Config(**{**BASE, **c}))
    s.initialize()
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        d = s.run(snapshot_times=SNAP_T, verbose=False)
    d["energy_error"] = s.energy_error()
    out[name], snaps[name] = d, s.snapshots
    print(f"[3d3] {name:<12} {s.step} steps {time.time() - t0:.0f}s  energy error {d['energy_error']:.1e}  "
          f"interface x {d['x_int'][0]:.2f} -> {d['x_int'][-1]:.2f}", flush=True)
    del s
    if name == "z-invariant":
        t, xi = d["t"], d["x_int"]
        drift = xi[-1] - xi[np.argmin(np.abs(t - 0.5))]
        if abs(drift) > 0.25:
            sys.exit(f"[3d3] gate failed: interface drifted {drift:+.2f} after t = 0.5; boundaries not clean")
        print(f"[3d3] gate passed: interface drift {drift:+.3f} after t = 0.5", flush=True)

os.makedirs(DATA, exist_ok=True)
np.savez(os.path.join(DATA, "exp_3d3.npz"),
         **{f"{n}__{k}": np.asarray(v) for n, d in out.items() for k, v in d.items()},
         **{f"{n}__{lab}__{k}": np.asarray(v) for n, sn in snaps.items() for lab, f in sn.items()
            for k, v in f.items()})


def rate(d, key, t0, t1):
    m = (d["t"] >= t0) & (d["t"] <= t1)
    return np.polyfit(d["t"][m], np.log(np.maximum(d[key][m], 1e-300)), 1)[0] if m.sum() > 3 else np.nan


T1, T2 = 0.3 * T_END, T_END
print(f"\n[3d3] window for rates: t = {T1:.2f}-{T2:.2f}")
print(f"[3d3] {'case':<12}{'Ekz end':>10}{'Ekz rate':>9}{'emix end':>10}{'emix rate':>10}{'J_max':>8}"
      f"{'J_xmod 0.3T':>12}{'J_xmod end':>11}{'J_zvar':>8}{'Eperp':>10}{'rough':>8}")
for name, d in out.items():
    i3 = np.argmin(np.abs(d["t"] - T1))
    print(f"[3d3] {name:<12}{d['Ekz'][-1]:10.2e}{rate(d, 'Ekz', T1, T2):9.2f}{d['emix'][-1]:10.3g}"
          f"{rate(d, 'emix', T1, T2):10.2f}{d['J_max'][-1]:8.1f}{d['J_xmod'][i3]:12.4f}{d['J_xmod'][-1]:11.4f}"
          f"{d['J_zvar'][-1]:8.4f}{d['Eperp'][-1]:10.3e}{d['rough'][-1]:8.4f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
col = {"hydro 3D": "#222222", "z-invariant": "#8a8a8a", "stripes 3D": "#c0392b"}
fig = plt.figure(figsize=(10.5, 7.2))
gs = fig.add_gridspec(3, 3)
axs = [fig.add_subplot(gs[0, i]) for i in range(3)] + [fig.add_subplot(gs[1, i]) for i in range(3)]
for name, d in out.items():
    kw = dict(color=col[name], lw=1.3, label=name)
    axs[0].semilogy(d["t"], np.maximum(d["Ekz"], 1e-14), **kw)
    axs[1].semilogy(d["t"], np.maximum(d["emix"], 1e-6), **kw)
    axs[2].plot(d["t"], d["J_max"], **kw)
    axs[3].plot(d["t"], d["J_xmod"], **kw)
    axs[4].semilogy(d["t"], np.maximum(d["Eperp"], 1e-14), **kw)
    axs[5].plot(d["t"], d["x_int"], **kw)
for a, ttl in zip(axs, (r"(a) $E_{kz}$", "(b) mixing-layer enstrophy", r"(c) max $|\mathbf{J}|$",
                        r"(d) sheet break-up along $x$ (J$_{xmod}$)", r"(e) $E_\perp$", "(f) interface position")):
    a.set_title(ttl, loc="left", fontsize=9); a.set_xlabel("$t$")
axs[0].legend(frameon=False, fontsize=7)
last = f"t={SNAP_T[-1]:.3f}"
for i, name in enumerate(("z-invariant", "stripes 3D")):
    sn = snaps[name].get(last, {})
    a = fig.add_subplot(gs[2, i])
    if "J_mid" in sn:
        im = a.imshow(sn["J_mid"].T, origin="lower", extent=[0, Lx, 0, Ly], cmap="magma", aspect="auto")
        fig.colorbar(im, ax=a, label=r"$|\mathbf{J}|$")
    a.set_title(f"({'gh'[i]}) {name}: $|\\mathbf{{J}}|$, mid-plane, t = {SNAP_T[-1]:g}", loc="left", fontsize=9)
    a.set_xlabel("$x$"); a.set_ylabel("$y$")
a = fig.add_subplot(gs[2, 2])
sn = snaps["stripes 3D"].get(last, {})
if "J_yz" in sn:
    im = a.imshow(sn["J_yz"].T, origin="lower", extent=[0, Ly, 0, Lz], cmap="magma", aspect="auto")
    fig.colorbar(im, ax=a, label=r"$|\mathbf{J}|$")
a.set_title(r"(i) stripes 3D: $|\mathbf{J}|$, $y$-$z$ cut", loc="left", fontsize=9)
a.set_xlabel("$y$"); a.set_ylabel("$z$")
fig.tight_layout()
fig.savefig(os.path.join(DATA, "exp_3d3.png"), dpi=200)
print("[3d3] saved exp_3d3.npz and exp_3d3.png")
