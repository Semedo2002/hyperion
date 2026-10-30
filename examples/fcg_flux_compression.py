"""Magnetic flux compression generator (FCG): a solid Al or Cu liner, launched inward at a few km/s (as by an
explosive or pulsed-power drive), crushes an axial seed field in axisymmetric (r, z).

The liner (inner radius R0, thickness dR) coasts inward with no further drive, v0 at mid-wall and v ~ 1/r
across it; the cavity holds a low-pressure gas fill (0.1 kg/m^3: its Alfven speed is several times v0, so the
cavity field stays near uniform, as in an evacuated generator) and the seed field B0; the field is zero in and
outside the liner. Physics: constrained transport (div B = 0 to round-off), Mie-Gruneisen liner, optional von Mises strength and spall (--enable-strength), resistivity with
the liner at its metal conductivity and Spitzer (capped) in the cavity gas, Ohmic heating. A small axial ripple
on the liner seeds feedthrough of the deceleration instability near turnaround.

With a capacitor bank (--capacitance-uf, --voltage-kv) the liner starts at rest and is driven instead by an RLC
circuit (aegis/circuit.py): the bank current runs up the liner and back through a return can, and its B_theta
pushes on the liner's outer surface with P_B = mu0 I^2 / (8 pi^2 r^2) while the load inductance grows as the
liner implodes.

Code units: mm, km/s, us, 1000 kg/m^3, GPa; B_unit = sqrt(mu0 * 1e9 Pa) = 35.45 T.

    python examples/fcg_flux_compression.py --enable-strength          # Al liner, 5 T, 3 km/s
    python examples/fcg_flux_compression.py --liner-material cu --enable-strength
    python examples/fcg_flux_compression.py --quick --check            # smoke test
    python examples/fcg_flux_compression.py --capacitance-uf 200 --voltage-kv 80 --inductance-nh 15 \
        --r0 10 --dr 1 --r-max 14 --lz 10 --nr 192 --nz 32 --out results/fcg_rlc   # pulsed-power drive
    python examples/fcg_flux_compression.py --plot-only
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
sys.path.insert(0, ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config, NVAR, EN, iRHO, iVX, iVY, iBY, iPR, iCLR
    from aegis.physics import xp, prim_to_cons, cons_to_prim
    from aegis.solver import MHDSolver, to_numpy
    from aegis.units import MU0, resistivity_kwargs, strength_kwargs, fld_kwargs, temperature_eV_per_code
    from aegis import strength as st
    from aegis.circuit import LinerDrive, add_circuit_args, circuit_from_args

RHO_U, V_U, L_U = 1000.0, 1000.0, 1e-3            # kg/m^3, m/s, m
B_U = np.sqrt(MU0 * RHO_U * V_U**2)                # tesla per code unit (35.45 T)
E_J = RHO_U * V_U**2 * L_U**3 * 2 * np.pi          # code energy (per radian) -> joules over the full circle

# Us-Up Hugoniots (Marsh 1980), Gruneisen gamma, cold conductivity, shear modulus / yield / spall (GPa)
METALS = {
    "al": dict(rho0=2700.0, c0=5328.0, s=1.338, g0=2.0, sigma=3.5e7, G=26.0, Y=0.29, spall=1.2),   # 6061-T6
    "cu": dict(rho0=8960.0, c0=3940.0, s=1.489, g0=2.0, sigma=5.8e7, G=47.7, Y=0.12, spall=1.3),   # OFHC
}


def setup(a):
    m = METALS[a.liner_material]
    phys = dict(eos_liner="mie_gruneisen", mg_rho0=m["rho0"] / RHO_U, mg_c0=m["c0"] / V_U, mg_s=m["s"],
                mg_gamma0=m["g0"])
    if a.eta_model != "ideal":
        phys.update(resistivity_kwargs(a.eta_model, L_U, V_U, sigma_liner=m["sigma"], sigma_fuel=a.sigma_gas,
                                       sigma_min=a.sigma_min, T_min_eV=1.0, ion_mass_amu=14.0))
    if a.enable_strength:
        phys.update(strength_kwargs(RHO_U, V_U, m["G"], m["Y"], m["spall"]))
    if getattr(a, "eos_table", None):                # tabulated liner, Mie-Gruneisen beyond the table
        phys.update(eos_table=a.eos_table, eos_table_material=a.eos_table_material, tab_rho_unit=RHO_U, tab_v_unit=V_U)
    if getattr(a, "enable_fld", False) or getattr(a, "opacity_table", None):    # FLD radiation in the cavity gas
        phys.update(fld_kwargs(RHO_U, V_U, L_U, 14.0, 1.0, preset=None if a.opacity_table else "free-free"))
        if a.opacity_table:
            phys.update(fld_opacity_table=a.opacity_table, tab_rho_unit=RHO_U, tab_L_unit=L_U,
                        tab_T_unit=temperature_eV_per_code(V_U, 14.0, 1.0) * 11604.518)
    a.phys = phys
    R1 = a.r0 + a.dr
    cfg = Config(**phys, nx=a.nr, ny=a.nz, x_min=0.0, x_max=a.r_max, y_min=0.0, y_max=a.lz, t_end=a.t_end,
                 cfl=0.3, gamma=1.4, B_transverse=a.b0_tesla / B_U, B_field_type="uniform", geometry="cylindrical",
                 bc_x_type="extrapolation", bc_y_type="periodic", material_tracer="volume_fraction",
                 diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    r = s.x[:, None] * xp.ones((1, s.ny_tot))
    z = s.y[None, :] * xp.ones((s.nx_tot, 1))
    w = 1.0 * cfg.dx
    step = lambda R: 0.5 * (1 + xp.tanh((r - R) / w))
    ripple = 1 + a.amp * xp.cos(2 * np.pi * a.mode * z / a.lz)   # axial seed on the inner surface
    liner = step(a.r0 * ripple) * (1 - step(R1))
    B0 = a.b0_tesla / B_U
    cavity = 1 - step(a.r0)                        # seed field B_z(r) only, so div B = 0 exactly
    p_gas = a.p_gas_bar * 1e-4                     # bar -> GPa
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
    W[iRHO] = a.rho_gas / RHO_U + (m["rho0"] - a.rho_gas) / RHO_U * liner
    W[iBY] = B0 * cavity
    # liner holds the seed field's pressure; the cold gas outside sits at p_gas (a 10 MPa step, far below yield)
    W[iPR] = p_gas + 0.5 * B0**2 * (1 - cavity) * (1 - step(R1))
    # coasting liner, explosive drive already done. Every cell holding any metal gets the full v0 (a velocity
    # ramp across the smeared edges makes the liner hit its own slow skin), so the velocity edge sits a few
    # cells inside the density edge and the gas outside moves with the liner.
    # v ~ 1/r (v0 at mid-wall) is the incompressible coasting profile; a uniform v0 makes a thick shell ring
    # radially at ~rho c v0 dr/R, a few GPa, enough to spall it before the field does anything
    W[iVX] = -a.v0_kms * (a.r0 + 0.5 * a.dr) / xp.maximum(r, w) * step(a.r0 * ripple - 4 * w)
    W[iCLR] = liner
    s.U = prim_to_cons(W, s.eos)
    s.t, s.step = 0.0, 0
    s.metal = a.liner_material
    return s


def stress_state(s, C, P):
    """von Mises stress / yield in the intact liner, spall damage"""
    if not s.strength:
        return 0.0, 0.0, 0, 0.0
    ng = s.ng
    S = s.S[:, ng:-ng, ng:-ng]
    Szz = -(S[st.SXX] + S[st.SYY])                 # hoop
    seq = xp.sqrt(1.5 * (S[st.SXX]**2 + S[st.SYY]**2 + Szz**2 + 2 * S[st.SXY]**2))
    lin = C > 0.5
    dmg = S[st.DMG] > 0.5
    intact = lin & ~dmg
    Y = s.cfg.yield_strength
    n = max(int(xp.sum(intact)), 1)
    mean_ratio = float(xp.sum(xp.where(intact, seq, 0.0))) / n / Y
    at_yield = float(xp.sum(intact & (seq > 0.99 * Y * C))) / n
    rr = s.x[ng:-ng][:, None]
    m_dmg = float(xp.sum(xp.where(dmg & lin, P[iRHO] * rr, 0.0)) / max(float(xp.sum(xp.where(lin, P[iRHO] * rr, 0.0))), 1e-300))
    return mean_ratio, at_yield, int(xp.sum(dmg & lin)), m_dmg


def diag(s, state):
    ng = s.ng
    U = s.U[:, ng:-ng, ng:-ng]
    P = cons_to_prim(s.U, s.eos)[:, ng:-ng, ng:-ng]
    C = xp.clip(P[iCLR], 0, 1)
    r = to_numpy(s.x[ng:-ng])
    c = to_numpy(C)
    mid = c.argmax(axis=0)                         # liner centre per z row
    inner = np.arange(len(r))[:, None] < mid[None, :]
    cav = xp.asarray(inner)
    # cavity radius per z row from the gas area inside the liner (consistent with the smeared interface)
    Rz = np.sqrt(np.sum(np.where(inner, (1 - c) * 2 * r[:, None], 0.0), axis=0) * s.cfg.dx)
    width = float(np.mean(np.sum(inner & (c > 0.05) & (c < 0.95), axis=0)))   # inner interface, cells
    Bz = P[iBY]
    rr = s.x[ng:-ng][:, None]
    # flux out to mid-liner (what is trapped, field soaked into the inner skin included) and flux left in the
    # cavity gas (what a load would see); the difference is the field the liner has absorbed
    da = s.cfg.dx * 2 * np.pi * s.cfg.dy / s.cfg.y_max
    flux = float(xp.sum(xp.where(cav, Bz * rr, 0.0))) * da
    flux_gas = float(xp.sum(xp.where(cav & (C < 0.5), Bz * rr, 0.0))) * da
    B_axis = float(xp.mean(Bz[0]))                 # on-axis probe
    B_max = float(xp.max(xp.abs(Bz)))
    lin = C > 0.5
    p_lin_min = float(xp.min(xp.where(lin, P[iPR], xp.inf)))
    seq_ratio, at_yield, n_dmg, m_dmg = stress_state(s, C, P)
    E = float(xp.sum(U[EN] * s.dV))
    v2 = P[iVX]**2 + P[iVY]**2
    # free-surface velocity (what VISAR sees): v_r of the outermost cell holding over half the row's peak metal
    # fraction (a coarse grid smears a thin liner below any fixed threshold), mean over z rows
    metal = C > 0.5 * xp.max(C, axis=0)[None, :]
    io = (C.shape[0] - 1) - xp.argmax(metal[::-1], axis=0)
    v_surf = float(xp.mean(xp.take_along_axis(P[iVX], io[None, :], axis=0)))
    KE_lin = float(xp.sum(xp.where(lin, 0.5 * P[iRHO] * v2 * s.dV, 0.0))) * E_J
    drive = state.get("drive")
    W_in = s._cumulative_boundary_energy + (drive.work_grid if drive else 0.0)
    state.setdefault("E0", E)
    state.setdefault("flux0", flux)
    state.setdefault("flux_gas0", flux_gas)
    # track dB/dt voltage pulse on the axis (finite difference between diagnostic samples)
    dBdt = 0.0 if "Bprev" not in state else (B_axis - state["Bprev"][1]) / max(s.t - state["Bprev"][0], 1e-300)
    state["Bprev"] = (s.t, B_axis)
    Y = METALS[s.metal]["Y"] * 1e9 / (RHO_U * V_U**2)  # P_B/Y reference, also with strength off
    row = {"t_us": s.t, "step": s.step, "R_cavity_mm": float(np.mean(Rz)),
            "R_field_mm": float(np.sqrt(max(flux, 0.0) / (np.pi * max(abs(B_axis), 1e-300)))),
            "interface_cells": width,
            "R_ripple_mm": float(0.5 * (Rz.max() - Rz.min())),
            "B_axis_T": B_axis * B_U, "B_max_T": B_max * B_U, "dBdt_T_per_us": dBdt * B_U,
            "probe_V_per_mm2": dBdt * B_U,       # 1-turn 1 mm^2 loop: V = A dB/dt, T/us * 1e-6 m^2 = V
            "flux_retention": flux_gas / state["flux_gas0"], "flux_trapped": flux / state["flux0"],
            "PB_over_Y": 0.5 * B_max**2 / Y,
            "seq_over_Y_mean": seq_ratio, "frac_at_yield": at_yield,
            "spall_cells": n_dmg, "spall_mass_frac": m_dmg, "p_liner_min_GPa": p_lin_min,
            "KE_liner_J": KE_lin, "v_surface_kms": v_surf, "div_B": s.div_b_max(),
            "energy_error": (E - state["E0"] - W_in) / state["E0"]}
    if drive:
        c = drive.c
        row.update({"I_MA": c.I / 1e6, "V_cap_kV": c.V / 1e3, "L_load_nH": drive.L_energy * 1e9,
                    "L_load_geom_nH": drive.L_geom * 1e9,
                    "P_drive_GPa": drive.P_B, "r_out_mm": drive.r_out,
                    "W_grid_J": drive.work_grid * E_J, "W_circuit_J": c.work_mech, "circuit_ohmic_J": c.ohmic,
                    "circuit_budget_error": c.budget_error()})
    return row


def frame(s):
    ng = s.ng
    P = cons_to_prim(s.U, s.eos)[:, ng:-ng, ng:-ng]
    rr = s.x[ng:-ng][:, None]
    psi = xp.cumsum(P[iBY] * rr, axis=0) * s.cfg.dx   # flux function: field lines are psi = const
    out = {"rho": to_numpy(P[iRHO]).astype(np.float32), "psi": to_numpy(psi).astype(np.float32)}
    if s.strength:
        S = s.S[:, ng:-ng, ng:-ng]
        Szz = -(S[st.SXX] + S[st.SYY])
        seq = xp.sqrt(1.5 * (S[st.SXX]**2 + S[st.SYY]**2 + Szz**2 + 2 * S[st.SXY]**2)) / s.cfg.yield_strength
        out["seq"] = to_numpy(seq).astype(np.float32)
    else:
        out["seq"] = np.zeros_like(out["rho"])
    return out


def run(a, out):
    os.makedirs(out, exist_ok=True)
    s = setup(a)
    bank = circuit_from_args(a) if hasattr(a, "capacitance_uf") else None
    state = {}
    if bank:
        r_ret = (a.r_return_mm or a.r_max) * 1e-3
        state["drive"] = LinerDrive(s, bank, r_ret, L_U, RHO_U, V_U)
        drv = (f"RLC drive {a.capacitance_uf:g} uF at {a.voltage_kv:g} kV ({bank.E0 / 1e3:.0f} kJ), "
               f"{a.inductance_nh:g} nH, {a.resistance_mohm:g} mOhm, return can at {r_ret * 1e3:g} mm")
    else:
        drv = f"coasting at v0 {a.v0_kms} km/s"
    print(f"FCG r-z {a.nr} x {a.nz}: {a.liner_material} liner R0 {a.r0} mm, {a.dr} mm thick, {drv}, "
          f"B0 {a.b0_tesla} T, strength {'on' if s.strength else 'off'}, eta {a.eta_model}", flush=True)
    rows = [diag(s, state)]
    frames, ftimes = [], []
    t0 = time.time()
    Rmin, t_turn = np.inf, None
    next_frame, dt0, stop = 0.0, None, ""
    while s.t < a.t_end - 1e-12:
        if s.t >= next_frame - 1e-12:
            frames.append(frame(s)); ftimes.append(s.t)
            next_frame += a.frame_dt
        s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), a.t_end - s.t)
        dt0 = dt0 or s.dt
        if s.dt < 1e-4 * dt0:                      # tenuous cavity gas after turnaround: Alfven speed blows up
            stop = f"dt fell to {s.dt:.1e} us"
            break
        s.step_ssprk3()
        if bank:
            state["drive"].step(s.dt)
        s.t += s.dt
        s.step += 1
        if s.step % a.diag_every == 0:
            row = diag(s, state)
            rows.append(row)
            if not np.isfinite(row["energy_error"]) or not np.isfinite(row["B_max_T"]):
                raise RuntimeError(f"non-physical state at t={s.t:.4f} us")
            if row["R_cavity_mm"] < Rmin:
                Rmin, t_turn = row["R_cavity_mm"], s.t
            # stop after turnaround: cavity re-expanded 10% (the axis field rings with cavity waves, so not that)
            if Rmin < 0.7 * a.r0 and row["R_cavity_mm"] > 1.1 * Rmin:
                stop = f"past turnaround at {s.t:.2f} us"
                break
    if stop:
        print("  stopped:", stop, flush=True)
    rows.append(diag(s, state))
    frames.append(frame(s)); ftimes.append(s.t)
    with open(os.path.join(out, "diagnostics.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    d = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    i = int(np.argmin(d["R_cavity_mm"]))
    summary = {
        "note": ("liner driven by an RLC circuit (magnetic piston on its outer surface; the B_theta outside it is "
                 "not on the grid)" if bank else "coasting liner (explosive/pulsed-power drive not modelled)")
                + "; liner at its cold metal conductivity, Spitzer (capped at sigma_min) in the cavity gas; "
                  "illustrative, not a specific device",
        "params": {k: v for k, v in vars(a).items() if k != "phys"}, "physics_code": a.phys, "B_unit_T": B_U,
        "steps": s.step, "wall_s": time.time() - t0, "stopped": stop,
        "t_turnaround_us": float(d["t_us"][i]),
        "B_axis_peak_T": float(d["B_axis_T"].max()), "B_max_peak_T": float(d["B_max_T"].max()),
        # field radius: where the retained flux sits at the on-axis field; tracer radius: gas area inside the
        # liner. They differ by the smeared interface, so check interface_cells against R_field / dr
        "R_field_at_turnaround_mm": float(d["R_field_mm"][i]), "R_tracer_at_turnaround_mm": float(d["R_cavity_mm"][i]),
        "convergence_ratio_field": float(a.r0 / d["R_field_mm"][i]),
        "interface_cells_at_turnaround": float(d["interface_cells"][i]), "cell_mm": a.r_max / a.nr,
        "flux_retention_at_turnaround": float(d["flux_retention"][i]),
        "flux_trapped_at_turnaround": float(d["flux_trapped"][i]),
        "dBdt_peak_T_per_us": float(d["dBdt_T_per_us"].max()),
        "PB_over_Y_peak": float(d["PB_over_Y"].max()),
        "frac_at_yield_max": float(d["frac_at_yield"].max()),
        "spall_cells_final": int(d["spall_cells"][-1]), "spall_mass_frac_final": float(d["spall_mass_frac"][-1]),
        "KE_liner_peak_J": float(d["KE_liner_J"].max()),
        "div_B_max": float(d["div_B"].max()), "energy_error_max": float(np.max(np.abs(d["energy_error"]))),
    }
    if bank:
        drive = state["drive"]
        summary["circuit"] = {
            "bank_energy_J": bank.E0, "I_peak_MA": float(d["I_MA"].max()),
            "t_I_peak_us": float(d["t_us"][int(np.argmax(d["I_MA"]))]),
            "P_drive_peak_GPa": float(d["P_drive_GPa"].max()), "L_load_final_nH": float(d["L_load_nH"][-1]),
            # load inductance from the work done vs from the liner's outer radius: smeared-surface error
            "L_load_geom_final_nH": float(d["L_load_geom_nH"][-1]),
            "L_energy_vs_geom_max_rel": float(np.max(np.abs(d["L_load_nH"] - d["L_load_geom_nH"])
                                                     / np.maximum(d["L_load_geom_nH"], 1e-30))),
            "work_on_liner_J": drive.work_grid * E_J, "work_from_circuit_J": bank.work_mech,
            "KE_liner_peak_over_bank": float(d["KE_liner_J"].max()) / bank.E0,
            "circuit_ohmic_J": bank.ohmic,
            # grid kick vs circuit energy: the coupling's discretisation error, as a fraction of the bank
            "work_mismatch_over_bank": drive.work_mismatch(),
            "circuit_budget_error": bank.budget_error(),
        }
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(os.path.join(out, "frames.npz"), r=to_numpy(s.x[s.ng:-s.ng]), z=to_numpy(s.y[s.ng:-s.ng]),
                        t=np.array(ftimes), rho=np.array([f["rho"] for f in frames]),
                        psi=np.array([f["psi"] for f in frames]), seq=np.array([f["seq"] for f in frames]))
    print(f"done: {s.step} steps, {summary['wall_s']:.0f}s | turnaround t {summary['t_turnaround_us']:.2f} us, "
          f"field radius {summary['R_field_at_turnaround_mm']:.2f} mm (CR {summary['convergence_ratio_field']:.1f}, "
          f"interface {summary['interface_cells_at_turnaround']:.1f} cells) | B axis peak "
          f"{summary['B_axis_peak_T']:.0f} T, flux kept "
          f"{summary['flux_retention_at_turnaround']:.3f} in the cavity ({summary['flux_trapped_at_turnaround']:.3f} to mid-liner) | P_B/Y peak {summary['PB_over_Y_peak']:.0f}, spall cells "
          f"{summary['spall_cells_final']} | div B {summary['div_B_max']:.1e}, energy error "
          f"{summary['energy_error_max']:.1e}", flush=True)
    if bank:
        c = summary["circuit"]
        print(f"circuit: I peak {c['I_peak_MA']:.2f} MA at {c['t_I_peak_us']:.2f} us, drive pressure peak "
              f"{c['P_drive_peak_GPa']:.1f} GPa, liner KE peak {100 * c['KE_liner_peak_over_bank']:.1f}% of the bank | "
              f"work on liner {c['work_on_liner_J']:.4g} J vs from circuit {c['work_from_circuit_J']:.4g} J "
              f"(mismatch {c['work_mismatch_over_bank']:.1e} of the bank), circuit budget {c['circuit_budget_error']:.1e} | "
              f"load L from work vs from radius: max {100 * c['L_energy_vs_geom_max_rel']:.2g}% apart",
              flush=True)
    return summary


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    rows = list(csv.DictReader(open(os.path.join(out, "diagnostics.csv"))))
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    summ = json.load(open(os.path.join(out, "summary.json")))
    par = summ["params"]
    t = d["t_us"]

    # oscilloscope-style traces
    driven = "I_MA" in d
    fig, ax = plt.subplots(5 if driven else 4, 1, figsize=(7.5, 10.5 if driven else 8.5), sharex=True)
    scope = dict(lw=1.4)
    ax[0].semilogy(t, d["B_axis_T"], color="#1f5f99", label="on-axis $B_z$", **scope)
    ax[0].semilogy(t, d["B_max_T"], color="#888888", ls="--", lw=1.0, label="$B_{max}$")
    ax[0].set_ylabel("B (T)"); ax[0].legend(frameon=False, fontsize=8)
    how = (f"RLC {par['capacitance_uf']:g} $\\mu$F, {par['voltage_kv']:g} kV" if driven
           else f"$v_0$ = {par['v0_kms']} km/s")
    ax[0].set_title(f"FCG, {par['liner_material'].upper()} liner, {how}, $B_0$ = {par['b0_tesla']} T", loc="left")
    ax[1].plot(t, d["probe_V_per_mm2"], color="#c0392b", **scope)
    ax[1].set_ylabel("b-dot V per mm$^2$\n(1 turn)")
    ax[2].plot(t, d["flux_retention"], color="#2e8b57", label="in the cavity gas", **scope)
    if "flux_trapped" in d:
        ax[2].plot(t, d["flux_trapped"], color="#2e8b57", ls="--", lw=1.0, label="out to mid-liner")
    ax[2].set_ylabel(r"$\Phi/\Phi_0$"); ax[2].legend(frameon=False, fontsize=8, loc="lower left")
    ax[2].ticklabel_format(axis="y", useOffset=False)
    axr = ax[2].twinx()
    axr.plot(t, d["R_field_mm"], color="#222222", ls=":", lw=1.1)
    axr.set_ylabel("field radius (mm)")
    ax[3].semilogy(t, np.maximum(d["PB_over_Y"], 1e-3), color="#7d3c98", label="$P_B/Y$", **scope)
    if summ["physics_code"].get("strength"):
        ax[3].semilogy(t, np.maximum(d["seq_over_Y_mean"], 1e-3), color="#e08a1e", label=r"mean $\sigma_{eq}/Y$ (liner)", **scope)
        ax[3].semilogy(t, np.maximum(d["spall_mass_frac"], 1e-3), color="#555555", label="spalled mass fraction", **scope)
    ax[3].set_ylabel("liner stress state"); ax[3].legend(frameon=False, fontsize=8)
    if driven:
        ax[4].plot(t, d["I_MA"], color="#1f5f99", **scope)
        ax[4].set_ylabel("I (MA)")
        axp = ax[4].twinx()
        axp.plot(t, d["P_drive_GPa"], color="#c0392b", lw=1.0, ls="--")
        axp.set_ylabel("drive pressure (GPa)", color="#c0392b")
    ax[-1].set_xlabel(r"$t$ ($\mu$s)")
    for x in ax:
        x.axvline(summ["t_turnaround_us"], color="#bbbbbb", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fcg_signals.png"), dpi=180)
    plt.close(fig)

    # implosion animation: log density (left) and sigma_eq / Y (right), field lines on both
    fr = np.load(os.path.join(out, "frames.npz"))
    r, z = fr["r"], fr["z"]
    rr = np.concatenate([-r[::-1], r])
    rmax = 1.2 * (par["r0"] + par["dr"])
    fig, axa = plt.subplots(figsize=(8, 8 * par["lz"] / (2 * rmax) + 1.2))
    lr = np.log10(fr["rho"])
    vmin, vmax = float(lr.min()), float(lr.max())

    def draw(k):
        axa.clear()
        rho, seq, psi = lr[k], fr["seq"][k], fr["psi"][k]
        left = rho[::-1]
        axa.pcolormesh(-r[::-1], z, left.T, cmap="inferno", vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
        axa.pcolormesh(r, z, np.where(fr["rho"][k] > 1.0, seq, np.nan).T, cmap="viridis", vmin=0, vmax=1.05,
                       shading="auto", rasterized=True)
        full = np.concatenate([psi[::-1], psi], axis=0)
        lv = np.linspace(0, float(fr["psi"][0].max()), 9)[1:-1]
        axa.contour(rr, z, full.T, levels=lv, colors="white", linewidths=0.7)
        axa.set_xlim(-rmax, rmax); axa.set_aspect("equal")
        axa.set_xlabel("r (mm)"); axa.set_ylabel("z (mm)")
        axa.set_title(rf"t = {fr['t'][k]:.2f} $\mu$s   left: $\log_{{10}}\rho$, right: $\sigma_{{eq}}/Y$, lines: B",
                      loc="left", fontsize=9)

    ani = FuncAnimation(fig, draw, frames=len(fr["t"]))
    ani.save(os.path.join(out, "fcg_implosion.gif"), writer=PillowWriter(fps=8), dpi=90)
    plt.close(fig)
    print(f"plots in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--liner-material", dest="liner_material", choices=["al", "cu"], default="al")
    ap.add_argument("--b0-tesla", dest="b0_tesla", type=float, default=5.0)
    ap.add_argument("--v0-kms", dest="v0_kms", type=float, default=None,
                    help="initial liner speed at mid-wall, km/s (default 3; 0 with a drive circuit)")
    ap.add_argument("--enable-strength", dest="enable_strength", action="store_true")
    ap.add_argument("--eta-model", dest="eta_model", choices=["spitzer", "material", "ideal"], default="spitzer")
    ap.add_argument("--sigma-gas", dest="sigma_gas", type=float, default=None, help="cavity gas S/m (material model)")
    ap.add_argument("--sigma-min", dest="sigma_min", type=float, default=5e6, help="conductivity floor, S/m")
    ap.add_argument("--r0", type=float, default=30.0, help="liner inner radius, mm")
    ap.add_argument("--dr", type=float, default=3.0, help="liner thickness, mm")
    ap.add_argument("--r-max", dest="r_max", type=float, default=40.0)
    ap.add_argument("--lz", type=float, default=20.0, help="axial period, mm")
    ap.add_argument("--mode", type=int, default=2, help="axial ripple mode number")
    ap.add_argument("--amp", type=float, default=0.005, help="relative ripple amplitude")
    # low-pressure fill: Alfven speed well above v0 keeps the cavity field near uniform, as in a vacuum cavity
    ap.add_argument("--rho-gas", dest="rho_gas", type=float, default=0.1, help="cavity gas, kg/m^3")
    ap.add_argument("--p-gas-bar", dest="p_gas_bar", type=float, default=0.1)
    ap.add_argument("--nr", type=int, default=256)
    ap.add_argument("--nz", type=int, default=128)
    ap.add_argument("--t-end", dest="t_end", type=float, default=15.0, help="us (the run stops after turnaround)")
    ap.add_argument("--frame-dt", dest="frame_dt", type=float, default=0.25, help="us between GIF frames")
    ap.add_argument("--diag-every", dest="diag_every", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "fcg_compression"))
    ap.add_argument("--quick", action="store_true", help="coarse grid")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    add_circuit_args(ap)
    ap.add_argument("--eos-table", dest="eos_table", default=None,
                    help="tabulated liner EOS (SESAME 301 or .npz, SI); the metal's Mie-Gruneisen fit beyond it")
    ap.add_argument("--eos-table-material", dest="eos_table_material", type=int, default=0)
    ap.add_argument("--enable-fld", dest="enable_fld", action="store_true", help="grey FLD radiation in the cavity gas")
    ap.add_argument("--opacity-table", dest="opacity_table", default=None,
                    help="Rosseland / Planck opacities for FLD (SESAME 502/505 or .npz); implies --enable-fld")
    a = ap.parse_args()
    if a.v0_kms is None:
        a.v0_kms = 0.0 if circuit_from_args(a) else 3.0
    if a.quick:
        a.nr, a.nz, a.frame_dt = 128, 16, 1.0
        if a.out == ap.get_default("out"):
            a.out += "_quick"
    if not a.plot_only:
        summ = run(a, a.out)
        if a.check:
            ok = smoke_ok(summ)
            print("smoke check:", "PASS" if ok else "FAIL")
            if not ok:
                sys.exit(1)
    if not a.no_plot:
        plot(a.out)


def smoke_ok(summ):
    return (summ["energy_error_max"] < 1e-12 and summ["div_B_max"] < 1e-12 and summ["B_axis_peak_T"] > 20 * 5.0
            and 0.3 < summ["flux_retention_at_turnaround"] <= 1.0 + 1e-9)


if __name__ == "__main__":
    main()
