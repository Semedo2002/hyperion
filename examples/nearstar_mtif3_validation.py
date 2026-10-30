"""Synthetic magnetic probe signals for a liner-driven magnetized-target implosion (MTIF-3-style set-up).

A dense liner moving inward at >= 10 km/s compresses a pre-magnetized fuel column (axial seed field)
in a 2D cross-section. Virtual probes at fixed radii record B(t) and dB/dt like b-dot channels.

IMPORTANT: the default machine parameters below are illustrative placeholders, not NearStar's MTIF-3
values, and the default solver is ideal-gas MHD (no radiation; a solid aluminium liner only with
--eos-liner stiffened or mie_gruneisen, magnetic diffusion into the liner only with --sigma-liner). Use the traces for signal shape and scaling; replace the parameters with
the real shot values, and add the missing physics, before comparing amplitudes with measured data.

    python examples/nearstar_mtif3_validation.py                  # 384 x 384, run + plots
    python examples/nearstar_mtif3_validation.py --quick --check  # smoke test
    python examples/nearstar_mtif3_validation.py --plot-only
    python examples/nearstar_mtif3_validation.py --sigma-liner 3.5e7            # resistive Al liner
    python examples/nearstar_mtif3_validation.py --eta-model spitzer --sigma-liner 3.5e7
    python examples/nearstar_mtif3_validation.py --eos-liner stiffened --sigma-liner 3.5e7    # solid Al liner
    python examples/nearstar_mtif3_validation.py --enable-rad --t-rad-min-ev 10              # bremsstrahlung
    python examples/nearstar_mtif3_validation.py --enable-conduction --braginskii             # conduction

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
    from aegis.config import Config, NVAR, EN, iRHO, iVX, iVY, iBZ, iPR, iCLR
    from aegis.physics import xp, prim_to_cons, cons_to_prim, pressure_margin
    from aegis.solver import MHDSolver, to_numpy
    from aegis.diagnostics import enclosed_by_tracer, cavitation_split, hotspot_conduction
    from aegis.units import (add_resistivity_args, resistivity_from_args, add_eos_args, eos_from_args,
                             add_radiation_args, radiation_from_args, add_conduction_args, conduction_from_args)
    from aegis.physics import internal_energy

GAMMA = 5.0 / 3.0
MU0 = 4e-7 * np.pi
R_CODE = 0.5      # fuel radius in code units (domain is [-1, 1]^2)


def units(a):
    """Code units: length L0 = R_fuel / R_CODE, density = fuel density, pressure = fuel pressure."""
    L0 = a.r_fuel_mm * 1e-3 / R_CODE
    rho0 = a.rho_fuel
    v0 = np.sqrt(a.p_fuel_mpa * 1e6 / rho0)
    p0 = rho0 * v0**2
    return {"L0": L0, "rho0": rho0, "v0": v0, "t0": L0 / v0, "p0": p0, "B0": np.sqrt(MU0 * p0)}


def setup(a, u):
    a.eta = resistivity_from_args(a, u["L0"], u["v0"])
    a.eos, rho_metal = eos_from_args(a, u["rho0"], u["v0"])
    a.rad = radiation_from_args(a, u["rho0"], u["v0"], u["L0"])
    # fuel starts at code density 1, code temperature p/rho = 1, field b_seed
    a.cond = conduction_from_args(a, u["rho0"], u["v0"], u["L0"], a.b_seed_t / u["B0"], 1.0, 1.0)
    if a.liner_ratio is None:
        a.liner_ratio = rho_metal or 10.0                       # solid density for a metal EOS
    # tracer = liner volume fraction in every run (also with an ideal-gas liner)
    cfg = Config(**a.eta, **a.eos, **a.rad, **a.cond, material_tracer="volume_fraction", nx=a.n, ny=a.n, x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, t_end=a.t_end_us * 1e-6 / u["t0"],
                 cfl=0.3, gamma=GAMMA, B_transverse=1.0, B_field_type="uniform", powell_source=True,
                 bc_x_type="extrapolation", bc_y_type="outflow", diag_interval=10**9, snapshot_times=[])
    s = MHDSolver(cfg)
    X, Y = s.X, s.Y
    r = xp.sqrt(X**2 + Y**2) + 1e-12
    th = xp.arctan2(Y, X)
    w = 1.5 * cfg.dx
    R1 = R_CODE + a.liner_mm * 1e-3 / u["L0"]
    pert = 1 + a.amp * xp.cos(a.mode * th)
    step = lambda R: 0.5 * (1 + xp.tanh((r - R * pert) / w))
    fuel, liner, outside = 1 - step(R_CODE), step(R_CODE) * (1 - step(R1)), step(R1)

    B0 = a.b_seed_t / u["B0"]                     # seed z-pinch field in code units
    V = a.v_liner_kms * 1e3 / u["v0"]             # liner speed in code units
    Bz = B0 * fuel
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
    W[iRHO] = fuel + a.liner_ratio * liner + outside
    W[iPR] = 1.0 + 0.5 * B0**2 - 0.5 * Bz**2      # total pressure balance at t = 0
    W[iVX] = -V * liner * X / r
    W[iVY] = -V * liner * Y / r
    W[iBZ] = Bz
    W[iCLR] = liner                                # material tracer: 1 in the liner only
    s.U = prim_to_cons(W, s.eos)
    s.t, s.step = 0.0, 0
    return s, B0, V


def probe_index(s, radii_mm, u):
    """bilinear weights for probes on the 45 deg diagonal (off the grid axes)"""
    ng = s.ng
    x = to_numpy(s.x[ng:-ng])
    dx = x[1] - x[0]
    out = []
    for r_mm in radii_mm:
        rc = r_mm * 1e-3 / u["L0"]
        px = py = rc / np.sqrt(2)
        fx, fy = (px - x[0]) / dx, (py - x[0]) / dx
        i, j = int(np.floor(fx)), int(np.floor(fy))
        out.append((i + ng, j + ng, fx - i, fy - j))
    return out


def sample(s, probes):
    Bz = s.U[6]                                   # conserved B_z = primitive B_z
    vals = []
    for i, j, fx, fy in probes:
        c = Bz[i:i + 2, j:j + 2]
        vals.append((1 - fx) * (1 - fy) * c[0, 0] + fx * (1 - fy) * c[1, 0] + (1 - fx) * fy * c[0, 1] + fx * fy * c[1, 1])
    return [float(v) for v in xp.stack(vals)]


def core_radius(s):
    ng = s.ng
    C = xp.clip(cons_to_prim(s.U, s.eos)[iCLR, ng:-ng, ng:-ng], 0, 1)
    # fuel = gas enclosed by the liner (the tracer marks the liner only)
    inside = xp.asarray(enclosed_by_tracer(C, s.X[ng:-ng, ng:-ng], s.Y[ng:-ng, ng:-ng]))
    return float(np.sqrt(float(xp.sum((1 - C) * inside)) * s.cfg.dx * s.cfg.dy / np.pi))


def track_conduction(s):
    """accumulate heat conducted out of the hot spot since the last call; record the suppression (field off /
    field on) of the instantaneous conductive power"""
    ng = s.ng
    C = xp.clip(cons_to_prim(s.U, s.eos)[iCLR, ng:-ng, ng:-ng], 0, 1)
    core = xp.asarray(enclosed_by_tracer(C, s.X[ng:-ng, ng:-ng], s.Y[ng:-ng, ng:-ng])) & (C < 0.5)
    P, supp, E_hot = hotspot_conduction(s.U[:, ng:-ng, ng:-ng], s.eos, s.cfg, (s.cfg.dx, s.cfg.dy), (False, False), core)
    dt = s.t - getattr(s, "_cond_t", s.t)
    s._cond_loss = getattr(s, "_cond_loss", 0.0) + P * dt
    s._E_hot_max = max(getattr(s, "_E_hot_max", 0.0), E_hot)
    s._cond_t = s.t
    s._cond_samples = getattr(s, "_cond_samples", []) + [(s.t, supp)]


def check_state(s, E0):
    ng = s.ng
    U = s.U[:, ng:-ng, ng:-ng]
    # p above cavitation (p for an ideal gas); mixed cells at a free surface facing vacuum may dip below
    # and are held by the solver's floor, so only pure cells count toward p_min
    C = xp.clip(cons_to_prim(s.U, s.eos)[iCLR, ng:-ng, ng:-ng], 0, 1)
    p_pure, n_mixed = cavitation_split(pressure_margin(U, s.eos), C)
    s._mixed_at_floor = max(getattr(s, "_mixed_at_floor", 0), n_mixed)
    E = float(xp.sum(U[EN])) * s.cfg.dx * s.cfg.dy
    E0 = E if E0 is None else E0
    err = (E - E0 - s._cumulative_boundary_energy - s._cumulative_powell_energy + s.cumulative_radiated_energy) / E0
    return p_pure, float(xp.min(U[0])), err, E0


def run(a, out):
    os.makedirs(out, exist_ok=True)
    u = units(a)
    s, B0, V = setup(a, u)
    # reference energies for the radiated fraction: initial kinetic energy, initial fuel thermal energy
    ng = s.ng
    U0 = s.U[:, ng:-ng, ng:-ng]
    C0 = xp.clip(cons_to_prim(s.U, s.eos)[iCLR, ng:-ng, ng:-ng], 0, 1)
    dA = s.cfg.dx * s.cfg.dy
    Ekin0 = float(xp.sum(0.5 * (U0[1]**2 + U0[2]**2) / U0[0])) * dA
    Eth_fuel0 = float(xp.sum(internal_energy(U0) * (C0 < 0.1))) * dA
    mach = V / np.sqrt(GAMMA)
    print(f"liner {a.v_liner_kms:g} km/s = Mach {mach:.1f} in the fuel; seed {a.b_seed_t:g} T "
          f"(beta = {2 / B0**2:.3g}); time unit {u['t0'] * 1e6:.3g} us", flush=True)
    if a.v_liner_kms < 10 or mach < 5:
        print("warning: liner below 10 km/s or Mach 5", flush=True)
    if a.rad:
        T_fuel = a.p_fuel_mpa * 1e6 / a.rho_fuel * a.ion_mass_amu * 1.66053906660e-27 / (2 * 1.602176634e-19)
        print(f"radiation: bremsstrahlung, Z_eff {a.z_eff:g}, off below {a.t_rad_min_ev:g} eV (initial fuel "
              f"{T_fuel:.3g} eV)", flush=True)
    if a.eos_liner != "ideal":
        print(f"liner EOS: {s.eos.describe()}, liner density {a.liner_ratio * u['rho0']:.0f} kg/m^3", flush=True)
    if a.eta:
        Rm = V * R_CODE / a.eta["eta_liner"] if a.eta.get("eta_liner", 0) > 0 else float("inf")
        print(f"resistivity: {a.eta_model}, liner eta {a.eta.get('eta_liner', 0) * u['L0'] * u['v0']:.3g} m^2/s "
              f"(Rm = {Rm:.3g}), cap {a.eta['eta_max'] * u['L0'] * u['v0']:.3g} m^2/s", flush=True)
    probes = probe_index(s, a.probes_mm, u)
    rows = []
    R_min, rho_peak, t_peak = 1e9, None, 0.0
    p_min, rho_min, E0 = 1e9, 1e9, None
    err_max = 0.0
    t0 = time.time()
    while s.t < s.cfg.t_end:
        b = sample(s, probes)
        R = core_radius(s)
        rows.append([s.t * u["t0"] * 1e6, R * u["L0"] * 1e3] + [v * u["B0"] for v in b])
        if R < R_min:                             # keep the density map at peak compression
            R_min, t_peak = R, s.t
            if s.step % 5 == 0 or rho_peak is None:
                rho_peak = to_numpy(s.U[0, s.ng:-s.ng, s.ng:-s.ng])
        if s.step % 10 == 0:
            if a.cond:
                track_conduction(s)
            pm, rm, err, E0 = check_state(s, E0)
            p_min, rho_min, err_max = min(p_min, pm), min(rho_min, rm), max(err_max, abs(err))
            if pm <= 0 or rm <= 0:
                raise RuntimeError(f"non-physical state at step {s.step}: p_min={pm:.3g}")
        s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), s.cfg.t_end - s.t)
        s.step_ssprk3()
        s.t += s.dt
        s.step += 1
    pm, rm, err, E0 = check_state(s, E0)
    p_min, rho_min, err_max = min(p_min, pm), min(rho_min, rm), max(err_max, abs(err))

    # probe table + b-dot
    T = np.array(rows)
    t_us = T[:, 0]
    B = T[:, 2:]
    dB = np.gradient(B, t_us, axis=0)            # T/us
    # Ohmic heat, J/m of column: code energy per unit length * p0 * L0^2
    ohmic_J_per_m = s.cumulative_ohmic_heating * u["p0"] * u["L0"]**2
    E_rad = s.cumulative_radiated_energy
    head = ["t_us", "R_core_mm"] + [f"B_{r:g}mm_T" for r in a.probes_mm] + [f"dBdt_{r:g}mm_T_per_us" for r in a.probes_mm]
    with open(os.path.join(out, "synthetic_probes.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(head)
        for k in range(len(t_us)):
            wr.writerow([f"{t_us[k]:.6g}", f"{T[k, 1]:.6g}"] + [f"{v:.6g}" for v in B[k]] + [f"{v:.6g}" for v in dB[k]])

    # observables at the innermost probe
    b1 = B[:, 0]
    k_pk = int(np.argmax(b1))
    rise = b1[0] + np.array([0.1, 0.9]) * (b1[k_pk] - b1[0])
    t10 = float(np.interp(rise[0], b1[:k_pk + 1], t_us[:k_pk + 1]))
    t90 = float(np.interp(rise[1], b1[:k_pk + 1], t_us[:k_pk + 1]))
    Rmm = T[:, 1]
    near = np.where(Rmm <= 1.1 * Rmm.min())[0]
    summary = {
        "note": "illustrative parameters, " + ("ideal-gas" if a.eos_liner == "ideal" else f"{a.eos_liner} Al liner")
                + " MHD" + ("" if not a.eta else f" with {a.eta_model} resistivity")
                + "; not a prediction of MTIF-3 shot data",
        "eta_model": a.eta_model, "eta_code": a.eta, "eos_liner": a.eos_liner, "eos_code": a.eos, "ohmic_heat_J_per_m": ohmic_J_per_m,
        "radiation": bool(a.rad), "rad_code": a.rad, "radiated_J_per_m": E_rad * u["p0"] * u["L0"]**2,
        "radiated_pct_of_Ekin0": 100 * E_rad / Ekin0, "radiated_pct_of_fuel_Eth0": 100 * E_rad / Eth_fuel0,
        "params": vars(a), "units": u, "liner_mach": mach, "beta_seed": 2 / B0**2,
        "B_peak_T": float(B.max()), "B_peak_innermost_T": float(b1[k_pk]), "B_peak_over_seed": float(b1[k_pk] / a.b_seed_t),
        "t_peak_us": float(t_us[k_pk]), "rise_time_10_90_us": t90 - t10,
        "dBdt_peak_T_per_us": float(np.abs(dB[:, 0]).max()),
        "R_min_mm": float(Rmm.min()), "convergence_ratio": float(Rmm[0] / Rmm.min()),
        "dwell_time_us": float(t_us[near[-1]] - t_us[near[0]]),
        "dwell_definition": "time the core radius stays within 10% of its minimum",
        "rise_definition": "10-90% of the field increase at the innermost probe",
        "steps": s.step, "wall_s": time.time() - t0,
        "p_min_code": p_min, "rho_min_code": rho_min, "energy_error_max": err_max,
        "mixed_cells_at_floor_max": getattr(s, "_mixed_at_floor", 0),
        "conduction": bool(a.cond), "cond_code": a.cond,
    }
    if a.cond:
        track_conduction(s)
        Eh = max(s._E_hot_max, 1e-300)
        ts = np.array([x[0] for x in s._cond_samples])
        summary.update({
            "cond_loss_pct_of_peak_hotspot_Eth": 100 * s._cond_loss / Eh,
            "cond_suppression_at_peak_compression": float(s._cond_samples[int(np.argmin(np.abs(ts - t_peak)))][1]),
            "cond_note": "heat conducted out of the hot spot (fuel above half the peak core temperature), integrated over the run, as a percentage of the hot spot's peak thermal energy; suppression = sum of grad T . K . grad T over the hot spot with the field off / with it on, for the state at peak compression (sampled every 10 steps) (instantaneous). In 2D the field is axial, so all in-plane conduction is across B; the fuel conducts into the liner as an isothermal wall (cond_wall_T), with its anisotropic conductivity. For an unmagnetized run to compare with, use --kappa-ratio 1.",
        })
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    if a.cond:
        print(f"conduction: hot spot lost {summary['cond_loss_pct_of_peak_hotspot_Eth']:.3g}% of its peak thermal "
              f"energy across B; at peak compression the field suppresses conduction in the hot spot "
              f"{summary['cond_suppression_at_peak_compression']:.3g}x", flush=True)
    x_mm = to_numpy(s.x[s.ng:-s.ng]) * u["L0"] * 1e3
    np.savez_compressed(os.path.join(out, "peak_compression.npz"), x_mm=x_mm, rho=rho_peak * u["rho0"],
                        t_us=t_peak * u["t0"] * 1e6)
    print(f"done: {s.step} steps, {summary['wall_s']:.0f}s | B_peak {summary['B_peak_T']:.1f} T, "
          f"rise {summary['rise_time_10_90_us']:.3g} us, dwell {summary['dwell_time_us']:.3g} us, "
          f"CR {summary['convergence_ratio']:.2f} | energy error {err_max:.1e}", flush=True)
    return summary


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    with open(os.path.join(out, "synthetic_probes.csv")) as fh:
        rd = csv.reader(fh)
        head = next(rd)
        D = np.array([[float(v) for v in row] for row in rd])
    summ = json.load(open(os.path.join(out, "summary.json")))
    pk = np.load(os.path.join(out, "peak_compression.npz"))
    radii = summ["params"]["probes_mm"]
    nb = len(radii)
    t = D[:, 0]
    cols = ["#c0392b", "#1f5f99", "#2e8b57", "#8e6cbf", "#f0a04b"]

    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 1.2])
    axB, axD = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[1, :2])
    for k, r in enumerate(radii):
        axB.plot(t, D[:, 2 + k], color=cols[k % 5], lw=1.4, label=f"r = {r:g} mm")
        axD.plot(t, D[:, 2 + nb + k], color=cols[k % 5], lw=1.1)
    axB.axvline(summ["t_peak_us"], color="#888888", ls=":", lw=1)
    axB.set_ylabel("$B$ (T)"); axB.legend(frameon=False, fontsize=8.5, ncol=nb)
    axB.set_title(f"synthetic probe traces: $B_{{peak}}$ = {summ['B_peak_innermost_T']:.1f} T, "
                  f"rise (10-90%) = {summ['rise_time_10_90_us'] * 1e3:.0f} ns, "
                  f"dwell = {summ['dwell_time_us'] * 1e3:.0f} ns", loc="left", fontsize=10)
    axD.set_ylabel(r"$dB/dt$ (T/$\mu$s)"); axD.set_xlabel(r"$t$ ($\mu$s)")
    axR = fig.add_subplot(gs[:, 2])
    x = pk["x_mm"]
    im = axR.pcolormesh(x, x, np.log10(pk["rho"]).T, cmap="inferno", shading="auto", rasterized=True)
    for k, r in enumerate(radii):
        axR.plot(r / np.sqrt(2), r / np.sqrt(2), "o", ms=5, mfc="none", mec=cols[k % 5], mew=1.5)
    lim = 1.6 * summ["params"]["r_fuel_mm"]
    axR.set_xlim(-lim, lim); axR.set_ylim(-lim, lim); axR.set_aspect("equal")
    axR.set_xlabel("x (mm)"); axR.set_ylabel("y (mm)")
    axR.set_title(f"peak compression, $t$ = {float(pk['t_us']):.3f} $\\mu$s", loc="left", fontsize=10)
    fig.colorbar(im, ax=axR, shrink=0.85, label=r"$\log_{10}\rho$ (kg/m$^3$)")
    eta_note = "" if summ.get("eta_model", "ideal") == "ideal" else f" with {summ['eta_model']} resistivity"
    eos = summ.get("eos_liner", "ideal")
    liner = "stiffened-gas" if eos == "stiffened" else "Mie-Gruneisen"
    phys = "ideal-gas MHD" if eos == "ideal" else f"MHD with a {liner} Al liner"
    if summ.get("radiation"):
        eta_note += (f"; bremsstrahlung radiated {summ['radiated_pct_of_Ekin0']:.2g}% of the initial kinetic and "
                     f"{summ['radiated_pct_of_fuel_Eth0']:.2g}% of the initial fuel thermal energy")
    fig.text(0.01, 0.005, f"Illustrative parameters, {phys}{eta_note}: signal shape and scaling only, not a prediction "
             "of MTIF-3 shot data.", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(out, "nearstar_mtif3_probe_signals.png"), dpi=220)
    plt.close(fig)
    print(f"plot in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # machine parameters (illustrative defaults: replace with the shot values)
    ap.add_argument("--r-fuel-mm", dest="r_fuel_mm", type=float, default=8.0, help="fuel radius")
    ap.add_argument("--liner-mm", dest="liner_mm", type=float, default=3.0, help="liner thickness")
    ap.add_argument("--rho-fuel", dest="rho_fuel", type=float, default=1.0, help="fuel density, kg/m^3")
    ap.add_argument("--p-fuel-mpa", dest="p_fuel_mpa", type=float, default=1.0, help="fuel pressure, MPa")
    ap.add_argument("--liner-ratio", dest="liner_ratio", type=float, default=None,
                    help="liner / fuel density (default 10, or solid Al for a metal EOS)")
    ap.add_argument("--v-liner-kms", dest="v_liner_kms", type=float, default=10.0, help="liner speed, km/s")
    ap.add_argument("--b-seed-t", dest="b_seed_t", type=float, default=10.0, help="seed axial field, T")
    ap.add_argument("--probes-mm", dest="probes_mm", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    ap.add_argument("--t-end-us", dest="t_end_us", type=float, default=2.5)
    # numerics
    ap.add_argument("--n", type=int, default=384, help="cells per side")
    ap.add_argument("--mode", type=int, default=6)
    ap.add_argument("--amp", type=float, default=0.005)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "nearstar_mtif3"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    add_resistivity_args(ap)
    add_eos_args(ap)
    add_radiation_args(ap)
    add_conduction_args(ap)
    a = ap.parse_args()
    if a.quick:
        a.n = 128
        if a.out == ap.get_default("out"):
            a.out += "_quick"
    if not a.plot_only:
        summ = run(a, a.out)
        if a.check:
            ok = (summ["p_min_code"] > 0 and summ["rho_min_code"] > 0 and summ["energy_error_max"] < 1e-10
                  and os.path.exists(os.path.join(a.out, "synthetic_probes.csv")) and summ["B_peak_over_seed"] > 1)
            print("smoke check:", "PASS" if ok else "FAIL")
            if not ok:
                sys.exit(1)
    if not a.no_plot:
        plot(a.out)
        if a.check and not os.path.exists(os.path.join(a.out, "nearstar_mtif3_probe_signals.png")):
            print("smoke check: FAIL (plot missing)"); sys.exit(1)


if __name__ == "__main__":
    main()
