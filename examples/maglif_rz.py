"""MagLIF-style liner implosion in axisymmetric (r, z) geometry.

A dense liner (radii 0.5-0.7, density 10) implodes at Mach 4 onto a core (density 1, pressure 1) holding an
axial seed field B_z = B0, in total-pressure balance; z is periodic with period Lz. The liner surfaces carry an
axial ripple R (1 + amp cos(2 pi m z / Lz)) that seeds the magneto-Rayleigh-Taylor instability when the liner
decelerates on the core. Unlike the Cartesian cross-section (examples/mtif_flux_compression.py) there is no
grid imprint: the geometry is exactly axisymmetric, the axis is a reflecting boundary, and constrained
transport keeps div B = 0 to round-off.

Diagnostics: core radius per z row (first liner cell outward from the axis), its mean and its axial
perturbation amplitude, mean core field against the ideal (R0/R)^2, core flux per z row (frozen in: spread
between rows and drift measure numerical reconnection), energy budget.

    python examples/maglif_rz.py                   # 256 x 256, run + plots
    python examples/maglif_rz.py --quick --check   # smoke test
    python examples/maglif_rz.py --plot-only
    python examples/maglif_rz.py --capacitance-uf 2 --voltage-kv 40 --inductance-nh 20   # liner at rest, RLC drive

Needs the AEGIS-MHD solver: pip install git+https://github.com/Semedo2002/MagSieve-3D
"""
import os
import sys
import csv
import json
import time
import argparse
import contextlib
import io

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # run from a repo checkout without installing
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config, NVAR, EN, iRHO, iVX, iBZ, iBY, iPR, iCLR
    from aegis.physics import xp, prim_to_cons, cons_to_prim, pressure_margin
    from aegis.solver import MHDSolver, to_numpy
    from aegis.diagnostics import cavitation_split
    from aegis.units import (add_resistivity_args, resistivity_from_args, add_eos_args, eos_from_args,
                             add_radiation_args, radiation_from_args, add_conduction_args, conduction_from_args)
    from aegis.circuit import LinerDrive, add_circuit_args, circuit_from_args

GAMMA = 5.0 / 3.0
R0, R1 = 0.5, 0.7


def setup(a):
    L0, v0 = a.r_ref_mm * 1e-3 / R0, a.v_ref_kms * 1e3
    phys = {}
    phys.update(resistivity_from_args(a, L0, v0))
    eos, rho_metal = eos_from_args(a, a.rho_ref, v0)
    phys.update(eos)
    phys.update(radiation_from_args(a, a.rho_ref, v0, L0))
    phys.update(conduction_from_args(a, a.rho_ref, v0, L0, a.B0, 1.0, 1.0))
    a.phys = phys
    if a.rho_liner is None:
        a.rho_liner = rho_metal or 10.0
    cfg = Config(**phys, nx=a.nr, ny=a.nz, x_min=0.0, x_max=1.0, y_min=0.0, y_max=a.lz, t_end=a.t_end, cfl=0.3,
                 gamma=GAMMA, B_transverse=a.B0, B_field_type="uniform", geometry="cylindrical",
                 bc_x_type="extrapolation", bc_y_type="periodic", material_tracer="volume_fraction",
                 diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    r = s.x[:, None] * xp.ones((1, s.ny_tot))
    z = s.y[None, :] * xp.ones((s.nx_tot, 1))
    pert = 1 + a.amp * xp.cos(2 * np.pi * a.mode * z / a.lz)
    w = 1.5 * cfg.dx
    step = lambda R: 0.5 * (1 + xp.tanh((r - R * pert) / w))
    core, liner = 1 - step(R0), step(R0) * (1 - step(R1))
    V = a.mach * np.sqrt(GAMMA)
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
    W[iRHO] = 1.0 + (a.rho_liner - 1.0) * liner
    # axial field in the unperturbed core: it must depend on r only (B_r = 0), or div B != 0 from the start
    core0 = 1 - 0.5 * (1 + xp.tanh((r - R0) / w))
    W[iBY] = a.B0 * core0
    W[iPR] = 1.0 + 0.5 * a.B0**2 - 0.5 * W[iBY]**2           # total pressure balance
    W[iVX] = -V * liner
    W[iCLR] = liner
    s.U = prim_to_cons(W, s.eos)
    s.t, s.step = 0.0, 0
    return s, V


def diag(s, state):
    ng = s.ng
    U = s.U[:, ng:-ng, ng:-ng]
    W = cons_to_prim(s.U, s.eos)[:, ng:-ng, ng:-ng]
    C = xp.clip(W[iCLR], 0, 1)
    r = to_numpy(s.x[ng:-ng])
    c = to_numpy(C)
    hit = c >= 0.5
    first = np.where(hit.any(axis=0), hit.argmax(axis=0), len(r) - 1)   # per z row
    Rz = r[first] - 0.5 * s.cfg.dx
    core = xp.asarray(np.arange(len(r))[:, None] < first[None, :]) & (C < 0.5)
    Bz = W[iBY]
    rr = s.x[ng:-ng][:, None]
    flux_z = to_numpy(xp.sum(xp.where(core, Bz * rr, 0.0), axis=0)) * s.cfg.dx * 2 * np.pi
    area = float(xp.sum(xp.where(core, rr, 0.0))) * s.cfg.dx * s.cfg.dy * 2 * np.pi / s.cfg.y_max
    p_pure, n_mixed = cavitation_split(pressure_margin(U, s.eos), C)
    E = float(xp.sum(U[EN] * s.dV))
    budget = s._cumulative_boundary_energy + s._cumulative_powell_energy - s.cumulative_radiated_energy
    drive = state.get("drive")
    if drive:
        budget += drive.work_grid
    state.setdefault("E0", E)
    state.setdefault("flux0", float(np.mean(flux_z)))
    row = {"t": s.t, "step": s.step, "R_mean": float(np.mean(Rz)), "R_amp": float(0.5 * (Rz.max() - Rz.min())),
            "R_area": float(np.sqrt(area / np.pi)),
            "B_core_mean": float(xp.sum(xp.where(core, Bz * rr, 0.0)) / max(float(xp.sum(xp.where(core, rr, 0.0))), 1e-300)),
            "B_max": float(xp.max(xp.abs(Bz))), "core_flux": float(np.mean(flux_z)),
            "core_flux_spread": float(np.std(flux_z) / max(abs(np.mean(flux_z)), 1e-300)),
            "div_B": s.div_b_max(), "p_min": p_pure, "mixed_cells_at_floor": n_mixed,
            "energy_error": (E - state["E0"] - budget) / state["E0"]}
    if drive:
        row.update({"I_MA": drive.c.I / 1e6, "P_drive_GPa": drive.P_B, "L_load_nH": drive.L_energy * 1e9,
                    "L_load_geom_nH": drive.L_geom * 1e9})
    return row


def run(a, out):
    os.makedirs(out, exist_ok=True)
    s, V = setup(a)
    state = {}
    bank = circuit_from_args(a) if hasattr(a, "capacitance_uf") else None
    if bank:
        L0 = a.r_ref_mm * 1e-3 / R0
        r_ret = (a.r_return_mm * 1e-3) if a.r_return_mm else s.cfg.x_max * L0
        state["drive"] = LinerDrive(s, bank, r_ret, L0, a.rho_ref, a.v_ref_kms * 1e3)
    print(f"r-z grid {a.nr} x {a.nz}, liner Mach {a.mach:g} (v = {V:.3f}), rho_liner {a.rho_liner:g}, "
          f"B0 {a.B0:g}, axial mode {a.mode} amp {a.amp:g}"
          + (f", RLC drive {bank.E0 / 1e3:.3g} kJ" if bank else ""), flush=True)
    rows = [diag(s, state)]
    frames, times = [], list(np.linspace(0, a.t_end, a.frames))
    best = (np.inf, None)
    t0 = time.time()
    while s.t < a.t_end - 1e-12:
        if times and s.t >= times[0] - 1e-12:
            times.pop(0)
            frames.append((s.t, to_numpy(cons_to_prim(s.U, s.eos)[iRHO, s.ng:-s.ng, s.ng:-s.ng]).astype(np.float32)))
        s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), a.t_end - s.t)
        s.step_ssprk3()
        if bank:
            state["drive"].step(s.dt)
        s.t += s.dt
        s.step += 1
        if s.step % 5 == 0:
            row = diag(s, state)
            rows.append(row)
            if row["p_min"] <= 0 or not np.isfinite(row["energy_error"]):
                raise RuntimeError(f"non-physical state at t={s.t:.4f}")
            if row["R_mean"] < best[0]:
                best = (row["R_mean"], to_numpy(cons_to_prim(s.U, s.eos)[iRHO, s.ng:-s.ng, s.ng:-s.ng]).astype(np.float32))
    rows.append(diag(s, state))
    with open(os.path.join(out, "diagnostics.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    d = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    i = int(np.argmin(d["R_mean"]))
    summary = {
        "note": "axisymmetric r-z; illustrative dimensionless parameters (see --r-ref-mm, --v-ref-kms, --rho-ref)",
        "params": {k: v for k, v in vars(a).items() if k != "phys"}, "physics_code": a.phys,
        "steps": s.step, "wall_s": time.time() - t0,
        "t_turnaround": float(d["t"][i]), "convergence_ratio": float(R0 / d["R_mean"][i]),
        "B_core_over_B0_at_turnaround": float(d["B_core_mean"][i] / a.B0),
        "ideal_(R0/R)^2_at_turnaround": float((d["R_area"][0] / d["R_area"][i])**2),
        "B_max_over_B0": float(d["B_max"].max() / a.B0),
        "R_perturbation_initial": float(d["R_amp"][0]), "R_perturbation_at_turnaround": float(d["R_amp"][i]),
        "core_flux_retained_at_turnaround": float(d["core_flux"][i] / d["core_flux"][0]),
        "core_flux_spread_max": float(d["core_flux_spread"].max()),
        "div_B_max": float(d["div_B"].max()), "energy_error_max": float(np.max(np.abs(d["energy_error"]))),
        "p_min": float(d["p_min"].min()),
    }
    if bank:
        summary["circuit"] = {"bank_energy_J": bank.E0, "I_peak_MA": float(d["I_MA"].max()),
                              "P_drive_peak_GPa": float(d["P_drive_GPa"].max()),
                              "work_on_liner_J": state["drive"].work_grid * state["drive"].Eu,
                              "work_mismatch_over_bank": state["drive"].work_mismatch(),
                              "circuit_budget_error": bank.budget_error()}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(os.path.join(out, "frames.npz"), r=to_numpy(s.x[s.ng:-s.ng]), z=to_numpy(s.y[s.ng:-s.ng]),
                        t=np.array([f[0] for f in frames]), rho=np.array([f[1] for f in frames]),
                        rho_turnaround=best[1])
    print(f"done: {s.step} steps, {summary['wall_s']:.0f}s | CR {summary['convergence_ratio']:.2f}, core B "
          f"{summary['B_core_over_B0_at_turnaround']:.1f} B0 vs ideal {summary['ideal_(R0/R)^2_at_turnaround']:.1f}, "
          f"ripple {summary['R_perturbation_initial']:.4f} -> {summary['R_perturbation_at_turnaround']:.4f} | div B "
          f"{summary['div_B_max']:.1e}, energy error {summary['energy_error_max']:.1e}", flush=True)
    return summary


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    rows = list(csv.DictReader(open(os.path.join(out, "diagnostics.csv"))))
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    summ = json.load(open(os.path.join(out, "summary.json")))
    fr = np.load(os.path.join(out, "frames.npz"))
    B0 = summ["params"]["B0"]
    fig = plt.figure(figsize=(14, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15])
    ax = [fig.add_subplot(gs[0, k]) for k in range(4)]
    ax[0].semilogy(d["t"], d["B_core_mean"] / B0, color="#1f5f99", lw=1.8, label="mean core $B_z/B_0$")
    ax[0].semilogy(d["t"], (d["R_area"][0] / d["R_area"])**2, color="#888888", ls="--", lw=1.2, label=r"ideal $(R_0/R)^2$")
    ax[0].semilogy(d["t"], d["B_max"] / B0, color="#c0392b", lw=1.2, label="$B_{max}/B_0$")
    ax[0].set_title("(a) flux compression", loc="left"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(d["t"], d["R_mean"] / R0, color="#222222", lw=1.8)
    ax[1].set_title(r"(b) mean core radius $R/R_0$", loc="left")
    ax[2].semilogy(d["t"], np.maximum(d["R_amp"], 1e-6), color="#e08a1e", lw=1.6, label="inner-surface ripple")
    ax[2].semilogy(d["t"], np.maximum(np.abs(d["core_flux"] / d["core_flux"][0] - 1), 1e-16), color="#2e8b57", lw=1.2,
                   label="core flux drift")
    ax[2].set_title("(c) ripple amplitude, flux drift", loc="left"); ax[2].legend(frameon=False, fontsize=8)
    for x in ax[:3]:
        x.set_xlabel("$t$")
    rho = fr["rho_turnaround"]
    r, z = fr["r"], fr["z"]
    full = np.concatenate([rho[::-1], rho], axis=0)            # mirror through the axis
    rr = np.concatenate([-r[::-1], r])
    im = ax[3].pcolormesh(rr, z, np.log10(full).T, cmap="inferno", shading="auto", rasterized=True)
    ax[3].set_xlim(-0.6, 0.6); ax[3].set_aspect("equal")
    ax[3].set_xlabel("r"); ax[3].set_ylabel("z")
    ax[3].set_title(r"(d) $\log_{10}\rho$ at turnaround (mirrored)", loc="left")
    fig.colorbar(im, ax=ax[3], shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "maglif_rz.png"), dpi=200)
    plt.close(fig)
    print(f"plots in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nr", type=int, default=256)
    ap.add_argument("--nz", type=int, default=256)
    ap.add_argument("--lz", type=float, default=1.0, help="axial period")
    ap.add_argument("--mode", type=int, default=4, help="axial liner ripple mode number")
    ap.add_argument("--amp", type=float, default=0.01, help="relative ripple amplitude")
    ap.add_argument("--mach", type=float, default=None, help="liner Mach number (default 4; 0 with a drive circuit)")
    ap.add_argument("--B0", type=float, default=1.0)
    ap.add_argument("--rho-liner", dest="rho_liner", type=float, default=None)
    ap.add_argument("--t-end", dest="t_end", type=float, default=0.25)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--r-ref-mm", dest="r_ref_mm", type=float, default=8.0)
    ap.add_argument("--v-ref-kms", dest="v_ref_kms", type=float, default=1.0)
    ap.add_argument("--rho-ref", dest="rho_ref", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "maglif_rz"))
    ap.add_argument("--quick", action="store_true", help="96 x 96")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    add_resistivity_args(ap)
    add_eos_args(ap)
    add_radiation_args(ap)
    add_conduction_args(ap)
    add_circuit_args(ap)
    a = ap.parse_args()
    if a.mach is None:
        a.mach = 0.0 if circuit_from_args(a) else 4.0
    if a.quick:
        a.nr = a.nz = 96
        a.frames = 10
        if a.out == ap.get_default("out"):
            a.out += "_quick"
    if not a.plot_only:
        summ = run(a, a.out)
        if a.check:
            ok = (summ["p_min"] > 0 and summ["energy_error_max"] < 1e-10 and summ["div_B_max"] < 1e-12
                  and summ["B_core_over_B0_at_turnaround"] > 2)
            print("smoke check:", "PASS" if ok else "FAIL")
            if not ok:
                sys.exit(1)
    if not a.no_plot:
        plot(a.out)


if __name__ == "__main__":
    main()
