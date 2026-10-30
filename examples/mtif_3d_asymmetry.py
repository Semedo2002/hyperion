"""3D target asymmetries in liner-driven flux compression: axis tilt and azimuthal liner modes.

A dense liner implodes onto a magnetized core in 3D (x-y cross-section, z along the axis, periodic in z).
The seed field B_z is uniform everywhere (pressure balanced), so div B = 0 holds exactly however the
target is shaped. Asymmetries:
  - tilt: the target axis is displaced as dx(z) = A sin(2 pi z / Lz), with A set so the largest axis slope
    equals the tilt angle (a uniform tilt cannot be periodic in z); the target is then misaligned with the
    field lines by up to that angle;
  - azimuthal modes: liner radii R (1 + amp cos(m phi)), m = 2 or 4.
Each 3D case is compared with a 2D axisymmetric baseline and with an unperturbed 3D reference (which
separates 3D discretization effects from the asymmetry). Metrics: peak mean core field, spread of the
peak field between z slices, field-line tilt in the core, heat conducted out of the hot spot, and how
much the field suppresses conduction there at turnaround (diagnostics.hotspot_conduction).

Default scales (r_ref 0.5 mm, v_ref 100 km/s, rho_ref 1 kg/m^3) put the core near 100 eV so magnetized
conduction matters; the liner then moves at Mach 4 relative to the core sound speed, ~520 km/s, which is
an illustrative, not a design, value.

    python examples/mtif_3d_asymmetry.py                      # 2D baseline, 3D reference, tilt 1 deg, m = 4
    python examples/mtif_3d_asymmetry.py --tilts 0.5 1 2 --modes 2 4
    python examples/mtif_3d_asymmetry.py --quick --check      # smoke test
    python examples/mtif_3d_asymmetry.py --tilts 2 --modes --vtk-case "tilt 2 deg"   # + 3D frames and renders
    python examples/mtif_3d_asymmetry.py --plot-only

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
    from aegis.config import Config, NVAR, EN, iRHO, iVX, iVY, iBX, iBY, iBZ, iPR, iCLR
    from aegis.physics import xp, prim_to_cons, cons_to_prim, pressure_margin
    from aegis.solver import MHDSolver, to_numpy
    from aegis.solver3d import MHDSolver3D
    from aegis.diagnostics import enclosed_by_tracer, cavitation_split, hotspot_conduction
    from aegis.units import (add_resistivity_args, resistivity_from_args, add_eos_args, eos_from_args,
                             add_radiation_args, radiation_from_args, add_conduction_args, conduction_from_args)

GAMMA = 5.0 / 3.0
R0, R1, LZ = 0.5, 0.7, 1.0            # core radius, liner outer radius, axial period (code units)


def physics(a):
    """Config keywords for resistivity, EOS, radiation and conduction from the SI scales."""
    L0, v0 = a.r_ref_mm * 1e-3 / R0, a.v_ref_kms * 1e3
    kw = {}
    kw.update(resistivity_from_args(a, L0, v0))
    eos, rho_metal = eos_from_args(a, a.rho_ref, v0)
    kw.update(eos)
    kw.update(radiation_from_args(a, a.rho_ref, v0, L0))
    kw.update(conduction_from_args(a, a.rho_ref, v0, L0, a.B0, 1.0, 1.0))
    if a.rho_liner is None:
        a.rho_liner = rho_metal or 10.0
    return kw


def build(a, dims, tilt_deg=0.0, m=0):
    """2D (dims = 2) or 3D solver with the target set up."""
    n = a.n
    cfg_kw = dict(nx=n, ny=n, x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, t_end=a.t_end, cfl=0.3, gamma=GAMMA,
                  B_transverse=a.B0, B_field_type="uniform", powell_source=True, bc_x_type="extrapolation",
                  bc_y_type="periodic", material_tracer="volume_fraction", diag_interval=10**9, snapshot_times=[],
                  **a.phys)
    if dims == 3:
        cfg_kw.update(nz=a.nz, z_min=0.0, z_max=LZ)
    with contextlib.redirect_stdout(io.StringIO()):
        s = (MHDSolver3D if dims == 3 else MHDSolver)(Config(**cfg_kw))
    sh = s.U.shape[1:]
    X = s.x.reshape((-1,) + (1,) * (len(sh) - 1))
    Y = s.y.reshape((1, -1) + (1,) * (len(sh) - 2))
    dx = 0.0
    if dims == 3:
        Z = s.z.reshape(1, 1, -1)
        A = LZ * np.tan(np.radians(tilt_deg)) / (2 * np.pi)     # max axis slope = tan(tilt)
        dx = A * xp.sin(2 * np.pi * Z / LZ)
    Xc = X - dx
    r = xp.sqrt(Xc**2 + Y**2) + 1e-12
    pert = 1 + (a.amp * xp.cos(m * xp.arctan2(Y, Xc)) if m else 0.0)
    w = 1.5 * s.cfg.dx
    step = lambda R: 0.5 * (1 + xp.tanh((r - R * pert) / w))
    liner = step(R0) * (1 - step(R1))
    cs = np.sqrt(GAMMA)                                        # core sound speed (rho = p = 1)
    V = a.mach * cs
    W = xp.zeros((NVAR,) + sh)
    W[iRHO] = 1.0 + (a.rho_liner - 1.0) * liner
    W[iPR] = 1.0                                               # uniform B: uniform p is in balance
    W[iVX] = -V * liner * Xc / r
    W[iVY] = -V * liner * Y / r
    W[iBZ] = a.B0
    W[iCLR] = liner
    s.U = prim_to_cons(W, s.eos)
    s.t, s.step = 0.0, 0
    s.apply_bc(s.U)
    return s, V


def core_mask(s, C):
    """gas the liner encloses, per z slice in 3D"""
    ng = s.ng
    X = s.x[ng:-ng][:, None] * xp.ones((1, s.cfg.ny))
    Y = s.y[ng:-ng][None, :] * xp.ones((s.cfg.nx, 1))
    if C.ndim == 2:
        return xp.asarray(enclosed_by_tracer(C, X, Y)) & (C < 0.5)
    return xp.stack([xp.asarray(enclosed_by_tracer(C[:, :, k], X, Y)) for k in range(C.shape[2])], axis=2) & (C < 0.5)


def diag(s, a, state):
    """one row of diagnostics; state carries running integrals between calls"""
    ng = s.ng
    I = (slice(None),) + (slice(ng, -ng),) * (s.U.ndim - 1)
    U = s.U[I]
    W = cons_to_prim(s.U, s.eos)[I]
    C = xp.clip(W[iCLR], 0, 1)
    core = core_mask(s, C)
    is3d = U.ndim == 4
    h = (s.cfg.dx, s.cfg.dy) + ((s.cfg.dz,) if is3d else ())
    dV = float(np.prod(h))
    depth = LZ if is3d else 1.0
    Bz = W[iBZ]
    area = float(xp.sum(core)) * dV / depth                  # mean cross-section area of the core
    R = np.sqrt(area / np.pi)
    B_core = float(xp.sum(xp.where(core, Bz, 0.0))) / max(float(xp.sum(core)), 1.0)
    if is3d:
        # spread of the peak field between z slices (a maximum over cells changes continuously; slice sums
        # over a core mask jump by one cell's flux, ~3% of a 3-cell-radius core, whenever a cell crosses)
        bmax_z = to_numpy(xp.max(xp.abs(Bz), axis=(0, 1)))
        bz_spread = float(np.std(bmax_z) / max(np.mean(bmax_z), 1e-300))
    else:
        bz_spread = 0.0
    tilt2 = (W[iBX]**2 + W[iBY]**2) / xp.maximum(Bz**2, 1e-30)
    field_tilt = float(xp.sqrt(xp.sum(xp.where(core, tilt2, 0.0)) / max(float(xp.sum(core)), 1.0)))
    p_pure, n_mixed = cavitation_split(pressure_margin(U, s.eos), C)
    E = float(xp.sum(U[EN])) * dV
    if is3d:
        budget = float(s._acc_boundary) + float(s._acc_powell) - float(s._acc_rad)
    else:
        budget = s._cumulative_boundary_energy + s._cumulative_powell_energy - s.cumulative_radiated_energy
    state.setdefault("E0", E)
    row = {"t": s.t, "step": s.step, "R": R, "B_core_mean": B_core, "B_max": float(xp.max(xp.abs(Bz))),
           "bz_peak_spread": bz_spread, "field_tilt_rms": field_tilt, "p_min": p_pure, "mixed_cells_at_floor": n_mixed,
           "energy_error": (E - state["E0"] - budget) / state["E0"]}
    if s.cfg.cond_kappa0 > 0:
        per = (False, True, True)[:len(h)]
        P, supp, E_hot = hotspot_conduction(U, s.eos, s.cfg, h, per, core)
        dt = s.t - state.get("t", s.t)
        state["loss"] = state.get("loss", 0.0) + P * dt / depth
        state["t"] = s.t
        row.update({"cond_loss": state["loss"], "cond_suppression": supp, "E_th_hotspot": E_hot / depth})
    return row


def slices(s):
    """density in the x-y plane at the z of largest axis displacement, and in the x-z plane through y = 0"""
    ng = s.ng
    rho = to_numpy(cons_to_prim(s.U, s.eos)[iRHO])[ng:-ng, ng:-ng]
    if rho.ndim == 2:
        return rho.astype(np.float32), None
    kz = rho.shape[2] // 4                                   # z = Lz/4: sin = 1
    return rho[:, :, kz].astype(np.float32), rho[:, rho.shape[1] // 2, :].astype(np.float32)


def run_case(a, name, dims, tilt=0.0, m=0):
    s, V = build(a, dims, tilt, m)
    state, rows = {}, []
    rows.append(diag(s, a, state))
    rec = None
    if dims == 3 and getattr(a, "vtk_case", None) == name:      # 3D snapshots for scripts/render_3d.py
        from aegis.io3d import Recorder
        rec = Recorder(os.path.join(a.out, "vtk_" + name.replace(" ", "_").replace("=", "")), fmt=("vti",))
        rec(s)
        next_rec = a.t_end / a.vtk_frames
    best = (np.inf, None, None)
    t0 = time.time()
    while s.t < a.t_end - 1e-12:
        s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), a.t_end - s.t)
        s.step_ssprk3()
        s.t += s.dt
        s.step += 1
        if s.step % a.diag_every == 0:
            row = diag(s, a, state)
            rows.append(row)
            if row["p_min"] <= 0 or not np.isfinite(row["energy_error"]):
                raise RuntimeError(f"{name}: non-physical state at t={s.t:.4f} (p_min {row['p_min']:.3g})")
            if row["R"] < best[0]:
                best = (row["R"], *slices(s))
        if rec is not None and s.t >= next_rec - 1e-12:
            rec(s)
            next_rec += a.t_end / a.vtk_frames
    rows.append(diag(s, a, state))
    if rec is not None and not getattr(a, "no_render", False):
        try:                                         # render the series now (PNGs + MP4), if pyvista is there
            sys.path.insert(0, os.path.join(ROOT, "scripts"))
            import render_3d
            pngs, mp4 = render_3d.render(render_3d.frames_from([os.path.join(rec.out, "series.pvd")]),
                                         os.path.join(rec.out, "renders"))
            print(f"  rendered {len(pngs)} frames of {name} -> {os.path.relpath(rec.out, ROOT)}/renders", flush=True)
        except ImportError:
            print("  (pyvista not installed: render later with scripts/render_3d.py)", flush=True)
    d = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    i = int(np.argmin(d["R"]))
    res = {"case": name, "dims": dims, "tilt_deg": tilt, "mode": m, "steps": s.step, "wall_s": time.time() - t0,
           "t_turnaround": float(d["t"][i]), "convergence_ratio": float(R0 / d["R"][i]),
           "B_core_peak": float(d["B_core_mean"].max()), "B_max_peak": float(d["B_max"].max()),
           "bz_peak_spread_at_turnaround": float(d["bz_peak_spread"][i]), "bz_peak_spread_max": float(d["bz_peak_spread"].max()),
           "field_tilt_rms_at_turnaround": float(d["field_tilt_rms"][i]),
           "energy_error_max": float(np.max(np.abs(d["energy_error"]))), "p_min": float(d["p_min"].min()),
           "mixed_cells_at_floor_max": int(d["mixed_cells_at_floor"].max())}
    if "cond_loss" in d:
        Eh = float(d["E_th_hotspot"].max())
        res.update({"cond_loss_pct": 100 * float(d["cond_loss"][-1]) / Eh,
                    "cond_suppression_at_turnaround": float(d["cond_suppression"][i])})
    print(f"  {name:14s} {s.step:5d} steps {res['wall_s']:6.0f}s | CR {res['convergence_ratio']:.2f}, peak core B "
          f"{res['B_core_peak']:.2f} B0, peak-B spread in z {res['bz_peak_spread_at_turnaround']:.2e}, field tilt "
          f"{res['field_tilt_rms_at_turnaround']:.2e}" + (f", hot-spot conduction loss {res['cond_loss_pct']:.3g}% "
          f"(field suppression {res['cond_suppression_at_turnaround']:.3g}x)" if "cond_loss_pct" in res else "") + f" | energy error {res['energy_error_max']:.1e}", flush=True)
    return res, rows, best


def run(a, out):
    os.makedirs(out, exist_ok=True)
    a.phys = physics(a)
    cases = [("2D baseline", 2, 0.0, 0), ("3D reference", 3, 0.0, 0)]
    cases += [(f"tilt {t:g} deg", 3, t, 0) for t in a.tilts]
    cases += [(f"m = {m}", 3, 0.0, m) for m in a.modes]
    print(f"grid {a.n}^2" + f" (x {a.nz} in z)" + f", liner Mach {a.mach:g}, rho_liner {a.rho_liner:g}, B0 {a.B0:g}"
          + (f", conduction k_par/k_perp {1 + (a.phys['cond_x0'] * a.B0)**2:.3g} in the initial core" if a.phys.get("cond_kappa0") else "")
          + f"; {len(cases)} runs", flush=True)
    results, frames = [], {}
    with open(os.path.join(out, "diagnostics.csv"), "w", newline="") as fh:
        wr = None
        for name, dims, tilt, m in cases:
            res, rows, best = run_case(a, name, dims, tilt, m)
            results.append(res)
            frames[name] = best
            for row in rows:
                row = {"case": name, **row}
                if wr is None:
                    wr = csv.DictWriter(fh, fieldnames=list(row), extrasaction="ignore")
                    wr.writeheader()
                wr.writerow(row)
    base = results[0]
    ref = results[1]
    for r in results[1:]:
        r["B_core_peak_vs_2D"] = r["B_core_peak"] / base["B_core_peak"]
        r["B_core_peak_vs_3D_reference"] = r["B_core_peak"] / ref["B_core_peak"]
        if "cond_loss_pct" in r and "cond_loss_pct" in ref:
            r["cond_loss_vs_3D_reference"] = r["cond_loss_pct"] / max(ref["cond_loss_pct"], 1e-300)
    L0, v0 = a.r_ref_mm * 1e-3 / R0, a.v_ref_kms * 1e3
    summary = {
        "note": "illustrative 3D study; tilt is a sinusoidal axis displacement with maximum slope equal to the "
                "tilt angle (periodic z); the fuel conducts into the liner as an isothermal wall; compare each 3D case with "
                "the 3D reference to separate the asymmetry from 3D discretization effects",
        "params": {k: v for k, v in vars(a).items() if k != "phys"}, "physics_code": a.phys,
        "si_scales": {"length_unit_m": L0, "velocity_unit_m_s": v0, "liner_speed_km_s": a.mach * np.sqrt(GAMMA) * a.v_ref_kms,
                      "core_T0_eV": 2 * 1.66053906660e-27 * v0**2 / (2 * 1.602176634e-19)},
        "cases": results,
    }
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(os.path.join(out, "turnaround_slices.npz"),
                        **{f"{k}_xy": v[1] for k, v in frames.items() if v[1] is not None},
                        **{f"{k}_xz": v[2] for k, v in frames.items() if v[2] is not None})
    return summary


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    summ = json.load(open(os.path.join(out, "summary.json")))
    rows = list(csv.DictReader(open(os.path.join(out, "diagnostics.csv"))))
    names = [c["case"] for c in summ["cases"]]
    get = lambda n, k: np.array([float(r[k]) for r in rows if r["case"] == n and r.get(k, "") != ""])
    cols = ["#222222", "#888888", "#c0392b", "#e08a1e", "#1f5f99", "#2e8b57", "#8e6cbf", "#b5651d"]
    has_cond = "cond_loss" in rows[0]
    fig, ax = plt.subplots(1, 4 if has_cond else 3, figsize=(16 if has_cond else 12.5, 3.8))
    for i, n in enumerate(names):
        st = dict(color=cols[i % len(cols)], lw=1.8 if i < 2 else 1.4, ls="--" if i == 0 else "-")
        ax[0].plot(get(n, "t"), get(n, "B_core_mean"), label=n, **st)
        if i > 0:
            ax[1].semilogy(get(n, "t"), np.maximum(get(n, "bz_peak_spread"), 1e-16), **st)
            ax[2].semilogy(get(n, "t"), np.maximum(get(n, "field_tilt_rms"), 1e-16), **st)
        if has_cond:
            Eh = get(n, "E_th_hotspot").max()
            ax[3].plot(get(n, "t"), 100 * get(n, "cond_loss") / Eh, **st)
    ax[0].set_title(r"(a) mean core $B/B_0$", loc="left"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].set_title("(b) spread of the peak field between z slices", loc="left")
    ax[2].set_title(r"(c) field-line tilt in the core, rms $B_\perp/B_z$", loc="left")
    if has_cond:
        ax[3].set_title("(d) heat conducted out of the hot spot, %", loc="left")
    for x in ax:
        x.set_xlabel("$t$")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "asymmetry_curves.png"), dpi=200)
    plt.close(fig)

    sl = np.load(os.path.join(out, "turnaround_slices.npz"))
    pert = [n for n in names[2:] if f"{n}_xy" in sl]
    if pert:
        fig, ax = plt.subplots(2, len(pert), figsize=(3.4 * len(pert), 6.4), squeeze=False)
        for j, n in enumerate(pert):
            xy, xz = sl[f"{n}_xy"], sl[f"{n}_xz"]
            ax[0, j].imshow(np.log10(xy).T, origin="lower", extent=(-1, 1, -1, 1), cmap="inferno")
            ax[0, j].set_title(f"{n}: x-y at z = Lz/4", fontsize=9, loc="left")
            ax[1, j].imshow(np.log10(xz).T, origin="lower", extent=(-1, 1, 0, LZ), cmap="inferno", aspect="auto")
            ax[1, j].set_title("x-z at y = 0", fontsize=9, loc="left")
            for k in (0, 1):
                ax[k, j].set_xlim(-0.4, 0.4)
            ax[0, j].set_ylim(-0.4, 0.4)
        fig.suptitle(r"$\log_{10}\rho$ at turnaround", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "turnaround_slices.png"), dpi=200)
        plt.close(fig)
    print(f"plots in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=64, help="cells across x and y")
    ap.add_argument("--nz", type=int, default=32, help="cells along z (period Lz = 1)")
    ap.add_argument("--tilts", type=float, nargs="*", default=[1.0], help="axis tilt angles, degrees")
    ap.add_argument("--modes", type=int, nargs="*", default=[4], help="azimuthal liner modes")
    ap.add_argument("--amp", type=float, default=0.02, help="relative amplitude of the azimuthal modes")
    ap.add_argument("--mach", type=float, default=4.0)
    ap.add_argument("--B0", type=float, default=1.0, help="uniform seed field, code units (core beta = 2/B0^2)")
    ap.add_argument("--rho-liner", dest="rho_liner", type=float, default=None)
    ap.add_argument("--t-end", dest="t_end", type=float, default=0.25)
    ap.add_argument("--r-ref-mm", dest="r_ref_mm", type=float, default=0.5, help="core radius in mm (SI scale)")
    ap.add_argument("--v-ref-kms", dest="v_ref_kms", type=float, default=100.0, help="core sqrt(p/rho) in km/s")
    ap.add_argument("--rho-ref", dest="rho_ref", type=float, default=1.0, help="core density in kg/m^3")
    ap.add_argument("--no-conduction", dest="no_conduction", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "mtif_3d_asymmetry"))
    ap.add_argument("--quick", action="store_true", help="40^2 x 12, smoke test size")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--vtk-case", dest="vtk_case", default=None,
                    help='write 3D .vti snapshots of one case (e.g. "tilt 2 deg") and render them at the end')
    ap.add_argument("--vtk-frames", dest="vtk_frames", type=int, default=24)
    ap.add_argument("--no-render", dest="no_render", action="store_true", help="snapshots only, no PyVista render")
    add_resistivity_args(ap)
    add_eos_args(ap)
    add_radiation_args(ap)
    add_conduction_args(ap)
    a = ap.parse_args()
    a.enable_conduction = a.enable_conduction or not a.no_conduction
    a.diag_every = 5
    if a.quick:
        a.n, a.nz = 40, 12
        if a.out == ap.get_default("out"):
            a.out += "_quick"
    if not a.plot_only:
        summ = run(a, a.out)
        if a.check:
            c = summ["cases"]
            ok = (all(r["energy_error_max"] < 1e-10 and r["p_min"] > 0 for r in c)
                  and abs(c[1]["B_core_peak_vs_2D"] - 1) < 0.15                 # 3D reference ~ 2D baseline
                  and c[1]["bz_peak_spread_max"] < 1e-8                         # unperturbed 3D stays z-invariant
                  and all(r["bz_peak_spread_max"] > c[1]["bz_peak_spread_max"] for r in c[2:] if r["tilt_deg"] > 0))
            print("smoke check:", "PASS" if ok else "FAIL")
            if not ok:
                sys.exit(1)
    if not a.no_plot:
        plot(a.out)


if __name__ == "__main__":
    main()
