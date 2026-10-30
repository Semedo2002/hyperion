"""High-Mach Richtmyer-Meshkov instability with magnetic suppression at high density ratio.

A planar shock (Mach >= 5) in a light gas hits a single-mode interface x(y) = x0 + eta0 cos(k y) with a
heavy gas (rho2/rho1 = 10, A ~ 0.82). The field is off, uniform along the shock normal ("parallel", B_x)
or uniform across it ("perpendicular", B_y). Repo convention: the shock travels in +x.

Jumps: hydro and parallel fields use the gas-dynamic Rankine-Hugoniot conditions (a normal field is
continuous across the shock); perpendicular fields use the MHD conditions, with the Mach number referred
to the fast magnetosonic speed. For parallel fields the incident, reflected and transmitted shocks must be
fast MHD shocks (downstream flow super-Alfvenic in the shock frame); inadmissible B0 is rejected.

Diagnostics (every diag_every steps): bubble/spike positions from the first C >= 0.5 crossing along x in
each row, a(t) = (x_bubble - x_spike)/2, max vorticity, the exact energy budget, div B norms and the
minimum raw pressure and density. Outputs in out_dir: diagnostics.csv, summary.json, frames.npz,
vtk/frame_NNNN.vti + rmi.pvd (ParaView time series).
"""
import os
import csv
import json
import time
import struct
from dataclasses import dataclass, asdict

import numpy as np

from ..config import Config, NVAR, EN, iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iCLR
from ..physics import xp, prim_to_cons, cons_to_prim, mhd_rankine_hugoniot
from ..solver import MHDSolver, to_numpy

FIELDS = ("none", "parallel", "perpendicular")


@dataclass
class RMIParams:
    name: str = "case"
    mach: float = 5.0
    rho1: float = 1.0
    p1: float = 1.0
    density_ratio: float = 10.0
    gamma: float = 5.0 / 3.0
    eta0: float = 0.05            # interface amplitude (k eta0 = 0.31 for mode 1 on Ly = 1)
    mode: int = 1
    field: str = "none"
    B0: float = 0.0
    nx: int = 1024
    ny: int = 128
    x_max: float = 8.0
    y_max: float = 1.0
    x_interface: float = 1.0
    shock_gap: float = 0.2        # shock starts this far ahead of the mean interface
    interface_cells: float = 1.5  # tanh half-width of the interface, in cells
    t_end: float = 1.5
    cfl: float = 0.3
    diag_every: int = 10
    n_frames: int = 31
    div_b: str = "glm"            # the committed results/rmi_mhd runs used GLM; "ct" for constrained transport

    def validate(self):
        if self.field not in FIELDS:
            raise ValueError(f"field must be one of {FIELDS}, got {self.field!r}")
        if (self.field == "none") != (self.B0 == 0.0):
            raise ValueError("B0 must be 0 for field='none' and > 0 otherwise")
        if self.shock_gap <= self.eta0 + 3 * self.interface_cells * self.x_max / self.nx:
            raise ValueError("shock_gap too small: the shock would start inside the interface")


# ---------------------------------------------------------------- jump conditions

def incident_jump(p):
    """Post-shock state behind the incident shock (lab frame, upstream at rest)."""
    g = p.gamma
    if p.field == "perpendicular":
        rho2, p2, u2, By2, vs = mhd_rankine_hugoniot(g, p.mach, p.rho1, p.p1, p.B0)
        return {"rho": rho2, "p": p2, "u": u2, "B_t": By2, "vs": vs}
    M2 = p.mach**2
    r = (g + 1) * M2 / ((g - 1) * M2 + 2)
    vs = p.mach * np.sqrt(g * p.p1 / p.rho1)
    return {"rho": p.rho1 * r, "p": p.p1 * (2 * g * M2 - (g - 1)) / (g + 1), "u": vs * (1 - 1 / r),
            "B_t": 0.0, "vs": vs}


def riemann_star(rhoL, uL, pL, rhoR, uR, pR, g):
    """Exact ideal-gas Riemann solver (Toro ch. 4): star pressure, velocity and densities."""
    cL, cR = np.sqrt(g * pL / rhoL), np.sqrt(g * pR / rhoR)

    def f(p, rho, pk, ck):
        if p > pk:                                                   # shock
            A, B = 2 / ((g + 1) * rho), (g - 1) / (g + 1) * pk
            q = np.sqrt(A / (p + B))
            return (p - pk) * q, q * (1 - 0.5 * (p - pk) / (p + B))
        r = (p / pk)**((g - 1) / (2 * g))                            # rarefaction
        return 2 * ck / (g - 1) * (r - 1), r / (rho * ck)

    p = max(0.5 * (pL + pR), 1e-12)
    for _ in range(100):
        fL, dL = f(p, rhoL, pL, cL)
        fR, dR = f(p, rhoR, pR, cR)
        p_new = max(p - (fL + fR + uR - uL) / (dL + dR), 1e-12)
        if abs(p_new - p) < 1e-12 * p:
            p = p_new
            break
        p = p_new
    fL, _ = f(p, rhoL, pL, cL)
    fR, _ = f(p, rhoR, pR, cR)
    u = 0.5 * (uL + uR) + 0.5 * (fR - fL)

    def rho_star(rho, pk):
        if p > pk:
            k = (g - 1) / (g + 1)
            return rho * (p / pk + k) / (k * p / pk + 1)
        return rho * (p / pk)**(1 / g)

    return {"p": p, "u": u, "rhoL": rho_star(rhoL, pL), "rhoR": rho_star(rhoR, pR),
            "left_shock": p > pL, "right_shock": p > pR}


def interaction(p, jump):
    """1D shock-interface interaction: contact velocity, post-interaction densities, wave speeds."""
    g = p.gamma
    rho_h = p.rho1 * p.density_ratio
    st = riemann_star(jump["rho"], jump["u"], jump["p"], rho_h, 0.0, p.p1, g)
    out = dict(st)
    if st["right_shock"]:          # transmitted shock into the heavy gas (at rest)
        out["S_transmitted"] = st["rhoR"] * st["u"] / (st["rhoR"] - rho_h)
    if st["left_shock"]:           # reflected shock into the post-incident light gas
        out["S_reflected"] = (st["rhoL"] * st["u"] - jump["rho"] * jump["u"]) / (st["rhoL"] - jump["rho"])
    return out


def check_admissible(p, jump, inter):
    """Parallel field: every shock must leave a super-Alfvenic downstream flow (fast shock).
    Returns the margins (downstream Alfven Mach numbers); raises if any is <= 1."""
    if p.field != "parallel":
        return {}
    B = p.B0
    m = {"incident": (jump["vs"] - jump["u"]) * np.sqrt(jump["rho"]) / B}
    if "S_transmitted" in inter:
        m["transmitted"] = (inter["S_transmitted"] - inter["u"]) * np.sqrt(inter["rhoR"]) / B
    if "S_reflected" in inter:
        m["reflected"] = abs(inter["u"] - inter["S_reflected"]) * np.sqrt(inter["rhoL"]) / B
    bad = {k: v for k, v in m.items() if v <= 1.0}
    if bad:
        raise ValueError(f"B0={B:g} is not admissible: downstream Alfven Mach <= 1 for {bad} "
                         f"(switch-on regime; lower B0 or raise the Mach number)")
    return m


def linear_theory(p, jump, inter):
    """Richtmyer impulsive model and the Meyer-Blewett variant for a(t) after shock impact."""
    rho_h = p.rho1 * p.density_ratio
    du = inter["u"]                                         # interface velocity jump
    A_pre = (rho_h - p.rho1) / (rho_h + p.rho1)
    A_post = (inter["rhoR"] - inter["rhoL"]) / (inter["rhoR"] + inter["rhoL"])
    k = 2 * np.pi * p.mode / p.y_max
    a0_post = p.eta0 * (1 - du / jump["vs"])                # amplitude compressed by the shock
    t_impact = p.shock_gap / jump["vs"]
    return {"k": k, "du": du, "A_pre": A_pre, "A_post": A_post, "a0": p.eta0, "a0_post": a0_post,
            "t_impact": t_impact,
            "v_impulsive": k * A_post * du * a0_post,
            "v_meyer_blewett": k * du * 0.5 * (A_post * a0_post + A_pre * p.eta0)}


# ---------------------------------------------------------------- set-up

def build_solver(p):
    p.validate()
    jump = incident_jump(p)
    inter = interaction(p, jump)
    margins = check_admissible(p, jump, inter)
    cfg = Config(nx=p.nx, ny=p.ny, x_min=0.0, x_max=p.x_max, y_min=0.0, y_max=p.y_max, t_end=p.t_end,
                 cfl=p.cfl, gamma=p.gamma, mach=p.mach, B_transverse=p.B0, B_field_type="uniform",
                 powell_source=True, bc_x_type="inflow_outflow", bc_y_type="periodic", div_b=p.div_b,
                 diag_interval=10**9, snapshot_times=[])
    s = MHDSolver(cfg)
    X, Y = s.X, s.Y
    ky = 2 * np.pi * p.mode / p.y_max
    x_if = p.x_interface + p.eta0 * xp.cos(ky * Y)
    phi = 0.5 * (1 + xp.tanh((X - x_if) / (p.interface_cells * cfg.dx)))
    post = X < p.x_interface - p.shock_gap
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
    W[iRHO] = xp.where(post, jump["rho"], p.rho1 * (1 - phi) + p.rho1 * p.density_ratio * phi)
    W[iVX] = xp.where(post, jump["u"], 0.0)
    W[iPR] = xp.where(post, jump["p"], p.p1)
    W[iCLR] = xp.where(post, 0.0, phi)
    if p.field == "parallel":
        W[iBX] = p.B0
    elif p.field == "perpendicular":
        W[iBY] = xp.where(post, jump["B_t"], p.B0)
    s.U = prim_to_cons(W, p.gamma)
    s._inflow_U = s.U[:, :s.ng].copy()
    s.t, s.step = 0.0, 0
    return s, {"incident": jump, "interaction": inter, "alfven_margins": margins,
               "linear": linear_theory(p, jump, inter)}


# ---------------------------------------------------------------- diagnostics

def diagnostics(s, p):
    ng = s.ng
    g = p.gamma
    U = s.U[:, ng:-ng, ng:-ng]
    W = cons_to_prim(s.U, g)[:, ng:-ng, ng:-ng]
    dx, dy = s.cfg.dx, s.cfg.dy
    x = s.x[ng:-ng]

    # first heavy-fluid crossing along x in each row (sub-cell linear interpolation)
    C = W[iCLR]
    heavy = C >= 0.5
    i1 = xp.argmax(heavy, axis=0)
    i0 = xp.maximum(i1 - 1, 0)
    cols = xp.arange(C.shape[1])
    c0, c1 = C[i0, cols], C[i1, cols]
    frac = xp.clip((0.5 - c0) / xp.maximum(c1 - c0, 1e-12), 0.0, 1.0)
    front = x[i0] + frac * (x[i1] - x[i0])
    front = xp.where(xp.any(heavy, axis=0), front, xp.nan)
    x_spike, x_bubble = float(xp.nanmin(front)), float(xp.nanmax(front))

    omega = xp.gradient(W[iVY], dx, axis=0) - xp.gradient(W[iVX], dy, axis=1)
    divB = xp.gradient(W[iBX], dx, axis=0) + xp.gradient(W[iBY], dy, axis=1)
    bmax = float(xp.max(xp.sqrt(W[iBX]**2 + W[iBY]**2 + W[iBZ]**2)))
    rho_raw = U[0]
    ke = 0.5 * (U[1]**2 + U[2]**2 + U[3]**2) / xp.maximum(rho_raw, 1e-300)
    p_raw = (g - 1) * (U[EN] - ke - 0.5 * (U[4]**2 + U[5]**2 + U[6]**2))
    E = float(xp.sum(U[EN])) * dx * dy
    inflow, powell = s._cumulative_boundary_energy, s._cumulative_powell_energy
    return {
        "t": s.t, "step": s.step,
        "a": 0.5 * (x_bubble - x_spike), "x_spike": x_spike, "x_bubble": x_bubble,
        "x_mean": float(xp.nanmean(front)),
        "omega_max": float(xp.max(xp.abs(omega))),
        "E": E, "E_inflow": inflow, "E_powell": powell,
        "divB_L1": float(xp.mean(xp.abs(divB))), "divB_L2": float(xp.sqrt(xp.mean(divB**2))),
        "divB_Linf": float(xp.max(xp.abs(divB))),
        "divB_rel": float(xp.max(xp.abs(divB))) * min(dx, dy) / bmax if bmax > 0 else 0.0,
        "p_min": float(xp.min(p_raw)), "rho_min": float(xp.min(rho_raw)),
        "finite": bool(xp.all(xp.isfinite(U))),
    }


# ---------------------------------------------------------------- VTK output

def write_vti(path, cell_arrays, origin, spacing):
    """VTK XML ImageData with cell data, raw appended binary (no VTK dependency).
    cell_arrays: name -> (nx, ny) scalar or (nx, ny, 3) vector, float."""
    first = next(iter(cell_arrays.values()))
    nx, ny = first.shape[:2]
    blobs, header, offset = [], [], 0
    for name, a in cell_arrays.items():
        a = np.asarray(a, dtype="<f4")
        ncomp = 3 if a.ndim == 3 else 1
        data = (a.transpose(1, 0, 2) if ncomp == 3 else a.T).tobytes(order="C")   # x fastest
        header.append(f'        <DataArray type="Float32" Name="{name}" NumberOfComponents="{ncomp}" '
                      f'format="appended" offset="{offset}"/>')
        blobs.append(struct.pack("<I", len(data)) + data)
        offset += 4 + len(data)
    xml = (f'<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
           f'header_type="UInt32">\n  <ImageData WholeExtent="0 {nx} 0 {ny} 0 0" '
           f'Origin="{origin[0]} {origin[1]} 0" Spacing="{spacing[0]} {spacing[1]} 1">\n'
           f'    <Piece Extent="0 {nx} 0 {ny} 0 0">\n      <CellData>\n' + "\n".join(header) +
           '\n      </CellData>\n    </Piece>\n  </ImageData>\n  <AppendedData encoding="raw">\n   _')
    with open(path, "wb") as fh:
        fh.write(xml.encode())
        for b in blobs:
            fh.write(b)
        fh.write(b"\n  </AppendedData>\n</VTKFile>\n")


def write_pvd(path, entries):
    rows = "\n".join(f'    <DataSet timestep="{t:.6g}" file="{f}"/>' for t, f in entries)
    with open(path, "w") as fh:
        fh.write(f'<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n  <Collection>\n'
                 f'{rows}\n  </Collection>\n</VTKFile>\n')


def _frame(s, p):
    ng = s.ng
    W = cons_to_prim(s.U, p.gamma)[:, ng:-ng, ng:-ng]
    return {k: to_numpy(W[i]).astype(np.float32) for k, i in
            (("rho", iRHO), ("p", iPR), ("C", iCLR), ("vx", iVX), ("vy", iVY), ("Bx", iBX), ("By", iBY))}


# ---------------------------------------------------------------- run

def run(p, out_dir, verbose=True):
    """Run one case and write all outputs to out_dir. Returns the summary dict."""
    os.makedirs(os.path.join(out_dir, "vtk"), exist_ok=True)
    s, info = build_solver(p)
    cfg = s.cfg
    frame_times = list(np.linspace(0.0, p.t_end, p.n_frames))
    rows, frames, pvd = [], {}, []
    t0 = time.time()
    E0 = None

    def record():
        nonlocal E0
        d = diagnostics(s, p)
        if E0 is None:
            E0 = d["E"]
        d["energy_error"] = (d["E"] - E0 - d["E_inflow"] - d["E_powell"]) / E0
        rows.append(d)
        return d

    def snapshot():
        k = len(pvd)
        f = _frame(s, p)
        frames[f"t{k:03d}"] = np.array(s.t)
        for name, a in f.items():
            frames[f"{name}{k:03d}"] = a
        vel = np.stack([f["vx"], f["vy"], np.zeros_like(f["vx"])], axis=-1)
        mag = np.stack([f["Bx"], f["By"], np.zeros_like(f["Bx"])], axis=-1)
        fname = f"vtk/frame_{k:04d}.vti"
        write_vti(os.path.join(out_dir, fname),
                  {"density": f["rho"], "pressure": f["p"], "heavy_fraction": f["C"], "velocity": vel,
                   "magnetic_field": mag},
                  (cfg.x_min, cfg.y_min), (cfg.dx, cfg.dy))
        pvd.append((s.t, fname))

    record()
    snapshot()
    frame_times.pop(0)
    while s.t < p.t_end:
        s.dt = min(s.compute_dt(cons_to_prim(s.U, p.gamma)), p.t_end - s.t)
        if s.dt <= 1e-14:
            break
        s.step_ssprk3()
        s.t += s.dt
        s.step += 1
        if s.step % p.diag_every == 0:
            d = record()
            if not d["finite"] or d["p_min"] <= 0 or d["rho_min"] <= 0:
                raise RuntimeError(f"{p.name}: non-physical state at t={s.t:.4f} "
                                   f"(p_min={d['p_min']:.3g}, rho_min={d['rho_min']:.3g})")
        while frame_times and s.t >= frame_times[0] - 0.5 * s.dt:
            frame_times.pop(0)
            snapshot()
        if verbose and s.step % 500 == 0:
            d = rows[-1]
            print(f"  [{p.name}] step {s.step} t={s.t:.3f} a={d['a']:.4f} omega_max={d['omega_max']:.1f}",
                  flush=True)
    if rows[-1]["step"] != s.step:
        record()
    wall = time.time() - t0

    keys = [k for k in rows[0] if k != "finite"]
    with open(os.path.join(out_dir, "diagnostics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(out_dir, "frames.npz"), x=to_numpy(s.x[s.ng:-s.ng]),
                        y=to_numpy(s.y[s.ng:-s.ng]), **frames)
    write_pvd(os.path.join(out_dir, "rmi.pvd"), pvd)

    def clean(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        return v

    summary = {
        "params": asdict(p), **clean(info),
        "result": {
            "steps": s.step, "wall_s": wall, "t_end": s.t,
            "a_final": rows[-1]["a"], "a_max": max(r["a"] for r in rows),
            "omega_max_peak": max(r["omega_max"] for r in rows),
            "energy_error_max": max(abs(r["energy_error"]) for r in rows),
            "divB_rel_max": max(r["divB_rel"] for r in rows),
            "p_min": min(r["p_min"] for r in rows), "rho_min": min(r["rho_min"] for r in rows),
            "n_frames": len(pvd),
        },
        "files": {"diagnostics": "diagnostics.csv", "frames": "frames.npz", "vtk_series": "rmi.pvd"},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    if verbose:
        r = summary["result"]
        print(f"  [{p.name}] done: {s.step} steps, {wall:.0f}s, a_final={r['a_final']:.4f}, "
              f"energy error max {r['energy_error_max']:.1e}, divB_rel max {r['divB_rel_max']:.3f}", flush=True)
    return summary
