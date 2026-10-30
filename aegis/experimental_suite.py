# experimental_suite.py - open experimental data bundled with the code, and benchmarks that run the solver against
# them (RMS, reduced chi^2, a comparison figure).
#
# Bundled data (aegis/data/experimental/), each with its provenance and a SHA-256 of its (LF-normalised) content,
# checked on load:
#   copper_hugoniot, nickel_hugoniot   shock velocity Us against particle velocity Up (and P, rho) from plate-impact
#       experiments, Marsh (ed.), "LASL Shock Hugoniot Data", University of California Press (1980), as transcribed
#       in LLNL's BALSCD repository (github.com/llnl/BALSCD, MIT licence, notice files alongside the data)
# No open B-dot, VISAR-trace or x-ray-radius data from pulsed-power shots are bundled: none were found in an open,
# citable machine-readable form. Shot traces of your own load with load_shot() (CSV: t, value, sigma) and are
# compared with simulations through aegis/validation.py (see examples/experimental_shot_fitting.py).
#
# hugoniot_benchmark() runs symmetric plate impacts with the solver (2D grid, one cell row of interest) at a set of
# particle velocities, measures the shock speed from the shock front's motion, and compares Us(Up) with the data.
# The data carry no error bars; sigma = sigma_rel Us (default 2%, typical of these plate-impact measurements) is an
# assumption that sets the chi^2 scale.
import hashlib
import io
import os
import contextlib

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "experimental")
_MARSH = ("S. P. Marsh (ed.), LASL Shock Hugoniot Data, University of California Press, Berkeley (1980); "
          "transcribed in github.com/llnl/BALSCD (MIT licence)")
DATASETS = {
    "copper_hugoniot": dict(file="copper_marsh.csv", kind="hugoniot", material="cu", source=_MARSH,
                            url="https://raw.githubusercontent.com/llnl/BALSCD/main/data/copper_marsh.csv",
                            sha256="9f38186ed4b1de37729fc7122e2520389a2e04c106d48c8451792c170882889c"),
    "nickel_hugoniot": dict(file="nickel_marsh.csv", kind="hugoniot", material="ni", source=_MARSH,
                            url="https://raw.githubusercontent.com/llnl/BALSCD/main/data/nickel_marsh.csv",
                            sha256="45985b78f584bc7264fe8036d6dce85a3423e6f87ef18893797679ec4fd03ea3"),
}


def _sha(path):
    raw = open(path, "rb").read().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def load_dataset(name):
    """dict of column arrays of a bundled dataset (checksum verified) plus its metadata under 'meta'"""
    meta = DATASETS[name]
    path = os.path.join(DATA_DIR, meta["file"])
    h = _sha(path)
    if h != meta["sha256"]:
        raise ValueError(f"{meta['file']}: checksum {h} does not match the bundled {meta['sha256']}")
    lines = open(path).read().strip().splitlines()
    head = lines[0].split(",")
    rows = [ln.split(",") for ln in lines[1:]]
    out = {"meta": meta}
    for k, col in enumerate(head):
        vals = [r[k] for r in rows]
        try:
            out[col] = np.array([float(v) for v in vals])
        except ValueError:
            out[col] = np.array(vals)
    return out


def load_shot(path, kind, units="", sigma=None, **kw):
    """a measured shot trace (CSV: t, value[, sigma]) as a validation.Trace"""
    from . import validation
    return validation.load_trace(path, kind=kind, units=units, sigma=sigma, **kw)


# ------------------------------------------------------------------ plate-impact simulations

def impact_shock_speed(up, eos_kw, rho0, n=480, L=1.0, p0=1e-4):
    """shock speed (km/s) in a stationary target hit by a flyer of the same material at 2 up (symmetric impact):
    the 2D solver with the given EOS keywords (g/cc, km/s, GPa units), the front tracked at half the shocked
    pressure between two times; returns (Us, shocked pressure)"""
    from .config import Config, NVAR, iRHO, iVX, iPR, iCLR
    from .physics import xp, prim_to_cons, cons_to_prim
    from .solver import MHDSolver, to_numpy
    cfg = Config(nx=n, ny=4, x_max=L, y_max=4 * L / n, t_end=1.0, cfl=0.3, mach=1.0, gamma=1.4, B_transverse=0.0,
                 bc_x_type="extrapolation", bc_y_type="periodic", material_tracer="volume_fraction",
                 diag_interval=10**9, snapshot_times=[], **eos_kw)
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    W = xp.zeros((NVAR,) + s.U.shape[1:])
    x0 = 0.3 * L
    W[iRHO], W[iPR], W[iCLR] = rho0, p0, 1.0
    W[iVX] = xp.where(s.X < x0, 2.0 * up, 0.0)
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    ng = s.ng
    x = to_numpy(s.x[ng:-ng])
    Us_guess = 4.0 + 1.5 * up
    times = [0.25 * L / Us_guess, 0.55 * L / Us_guess]
    fronts, ps = [], []
    for T in times:
        while s.t < T - 1e-12:
            s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), T - s.t)
            s.step_ssprk3(); s.t += s.dt; s.step += 1
        p = to_numpy(cons_to_prim(s.U, s.eos)[iPR, ng:-ng, ng])
        right = x > x0
        p_sh = float(np.median(p[(x > x0) & (x < x0 + 0.5 * up * T + 0.02 * L) | ((x > x0) & (p > 0.5 * p.max()))]))
        half = 0.5 * (p_sh + p0)
        k = np.nonzero(right & (p > half))[0][-1]           # last cell above half: the front, interpolated
        xf = x[k] + (p[k] - half) / (p[k] - p[k + 1]) * (x[k + 1] - x[k])
        fronts.append(xf); ps.append(p_sh)
    return (fronts[1] - fronts[0]) / (times[1] - times[0]), ps[-1]


def hugoniot_benchmark(dataset="copper_hugoniot", ups=None, table=None, sigma_rel=0.02, up_max=3.0, n=480,
                       out_png=None):
    """simulate Us at particle velocities `ups` (km/s) with the material table from tools/generate_sesame.py (made
    on the fly if `table` is None) and compare with the data (Up <= up_max, ambient density within 1% of the
    table's): returns a dict with the metrics; writes the comparison figure if out_png is given"""
    import sys
    from . import validation
    d = load_dataset(dataset)
    mat = d["meta"]["material"]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "tools"))
    import generate_sesame as gs
    if mat not in gs.MATERIALS:
        raise ValueError(f"no material model for {mat} in tools/generate_sesame.py")
    m = gs.MATERIALS[mat]
    if table is None:
        import tempfile
        with np.errstate(all="ignore"):
            tab = gs.table(mat, 300, 160, opacity=False)
        tab.pop("adjust_max", None)
        table = os.path.join(tempfile.mkdtemp(prefix="hug_"), f"{mat}.npz")
        np.savez(table, **tab)
    rho0 = m["rho0"] / 1000.0
    c0, s = m["hug"]
    eos_kw = dict(eos_liner="mie_gruneisen", mg_rho0=rho0, mg_c0=c0, mg_s=s, mg_gamma0=m["G0"], eos_table=table,
                  tab_rho_unit=1000.0, tab_v_unit=1000.0)
    ups = np.asarray(ups if ups is not None else np.linspace(0.25, up_max, 8))
    sim = np.array([impact_shock_speed(u, eos_kw, rho0, n=n) for u in ups])
    Us_sim, P_sim = sim[:, 0], sim[:, 1]
    sel = (d["Up_km_s"] > 0) & (d["Up_km_s"] <= up_max) & (np.abs(d["rho0_g_cc"] / rho0 - 1) < 0.01)
    up_d, Us_d = d["Up_km_s"][sel], d["Us_km_s"][sel]
    Us_m = np.interp(up_d, ups, Us_sim)                      # simulated Hugoniot at the data's Up
    sig = sigma_rel * Us_d
    res = Us_d - Us_m
    chi2 = float(np.sum((res / sig)**2))
    dof = max(len(res), 1)
    H = gs.hugoniot(mat, ups)                                # the table's own Hugoniot (no flow solver)
    out = {"dataset": dataset, "source": d["meta"]["source"], "n_points": int(len(res)),
           "ups": ups.tolist(), "Us_sim": Us_sim.tolist(), "P_sim_GPa": P_sim.tolist(), "Us_table": H[:, 0].tolist(),
           "rms_km_s": float(np.sqrt(np.mean(res**2))), "rms_rel": float(np.sqrt(np.mean((res / Us_d)**2))),
           "chi2_nu": chi2 / dof, "sigma_rel_assumed": sigma_rel,
           "sim_vs_table_max_rel": float(np.max(np.abs(Us_sim - H[:, 0]) / H[:, 0]))}
    lin = np.polyfit(up_d, Us_d, 1)
    out["data_linear_fit"] = {"c0": float(lin[1]), "s": float(lin[0])}
    if out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(7, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
        uu = np.linspace(0, up_max, 200)
        ax[0].fill_between(uu, (1 - sigma_rel) * np.polyval(lin, uu), (1 + sigma_rel) * np.polyval(lin, uu),
                           color="#9bb7d4", alpha=0.35, lw=0, label=f"data linear fit +/- {100 * sigma_rel:.0f}%")
        ax[0].plot(up_d, Us_d, "o", ms=3.5, color="#1f4e79", label=f"Marsh (1980), {len(up_d)} shots")
        ax[0].plot(ups, H[:, 0], "-", color="#7f7f7f", lw=1.2, label="model Hugoniot (table, no flow solver)")
        ax[0].plot(ups, Us_sim, "s", ms=6, mfc="none", color="#c0392b", mew=1.5, label="simulated plate impacts")
        ax[0].set_ylabel("shock velocity Us (km/s)")
        ax[0].legend(frameon=False, fontsize=8)
        ax[0].set_title(f"{mat.capitalize()} Hugoniot: RMS {out['rms_km_s']:.3f} km/s ({100 * out['rms_rel']:.1f}%), "
                        f"chi2_nu {out['chi2_nu']:.2f} (sigma = {100 * sigma_rel:.0f}% Us)", fontsize=10)
        ax[1].axhspan(-sigma_rel * 100, sigma_rel * 100, color="#9bb7d4", alpha=0.35, lw=0)
        ax[1].plot(up_d, 100 * res / Us_d, "o", ms=3.5, color="#1f4e79")
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_ylabel("data - sim (%)")
        ax[1].set_xlabel("particle velocity Up (km/s)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
    return out
