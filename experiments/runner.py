"""Run a set of cases against a hydro reference and report interface-lock metrics.

Lock criterion (pre-registered): late-time mixing-layer enstrophy < 0.2 x hydro AND
final interface roughness (rms over all y-modes) < 0.5 x hydro.
"""
import os
import sys
import io
import time
import contextlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config, RHO, MX
    from aegis.solver import MHDSolver
    from aegis.physics import xp

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 12 x 2 domain, small ripple (k a0 = 0.25), Mach 10, x stripes in pressure balance. div_b="glm": the
# published study used GLM cleaning; the solver default is now constrained transport
BASE = dict(div_b="glm", nx=400, ny=100, x_min=0.0, x_max=12.0, y_min=0.0, y_max=2.0, t_end=0.8, cfl=0.30,
            mach=10.0, interface_x=1.5, perturbation_amp=0.02, perturbation_mode=4, density_ratio=3.0,
            interface_width=2.0, powell_source=True, bc_x_type="inflow_outflow", bc_y_type="periodic",
            B_modulation_mode=4, localized_rh=True, diag_interval=20, snapshot_times=[])


def inflow_gap_bar(s):
    """gap/bar momentum flux of the imposed inflow, used to normalise the measured ratio."""
    U = s._inflow_U[:, -1, s.ng:-s.ng]
    mf = xp.asnumpy(U[MX]**2 / U[RHO])
    return mf[s._gap_mask_y].mean() / mf[s._bar_mask_y].mean()


def tail(x, t, frac=1/3, t0=0.05):
    """mean over the last third of t > t0"""
    m = t > t0
    x = x[m]
    return float(np.mean(x[-max(int(len(x) * frac), 1):]))


def run_case(params):
    s = MHDSolver(Config(**params))
    with contextlib.redirect_stdout(io.StringIO()):
        s.initialize()
        if s._bar_mask_y is None:
            s._build_bar_gap_masks()  # hydro gets the same masks as the striped runs
        s.run()
    t = np.array(s.diag_times)
    A = np.array(s.diag_mode_amps)
    return dict(t=t, a=np.array(s.diag_perturbation_amp),
                rms=np.sqrt(0.5 * np.sum(A[:, 1:]**2, axis=1)), m4=A[:, 4], m8=A[:, 8],
                en=np.array(s.diag_enstrophy_local), emix=np.array(s.diag_enstrophy_mix),
                circ=np.array(s.diag_interface_circulation), mw=np.array(s.diag_mixing_width_integral),
                gap=np.array(s.diag_jet_momentum_flux_gap), bar=np.array(s.diag_jet_momentum_flux_bar),
                base=inflow_gap_bar(s), div=s.diag_divB_rel_max[-1],
                E=np.array(s.diag_energy_total), inflow=np.array(s.diag_boundary_flux_cumulative),
                powell=np.array(s.diag_powell_cumulative),
                cons=abs(s.energy_budget()["conservation_error_rel"]), steps=s.step)


def run_sweep(tag, common, cases):
    """cases: {name: overrides}; one case should be named 'hydro'."""
    os.makedirs(DATA_DIR, exist_ok=True)
    out = {}
    for name, c in cases.items():
        t0 = time.time()
        out[name] = run_case({**BASE, **common, **c})
        print(f"[{tag}] {name} done {time.time() - t0:.0f}s {out[name]['steps']} steps", flush=True)
    np.savez(os.path.join(DATA_DIR, f"exp_{tag}.npz"),
             **{f"{n}__{k}": v for n, d in out.items() for k, v in d.items() if isinstance(v, np.ndarray)})
    report(tag, out)
    return out


def report(tag, out):
    ref = out.get("hydro")
    en_h = tail(ref["emix"], ref["t"]) if ref else None
    circ_h = tail(ref["circ"], ref["t"]) if ref else None
    rms_h = ref["rms"][-1] if ref else None
    print(f"[{tag}] {'case':<22}{'m4 end':>8}{'m8 end':>8}{'rms end':>8}{'rms/h':>7}{'emix/h':>8}{'circ/h':>7}"
          f"{'g/b norm':>9}{'divB':>7}{'cons':>8}  verdict")
    for name, d in out.items():
        t = d["t"]
        en_r = tail(d["emix"], t) / en_h if en_h else np.nan
        circ_r = tail(d["circ"], t) / circ_h if circ_h else np.nan
        rms_r = d["rms"][-1] / rms_h if rms_h else np.nan
        gb = tail(d["gap"], t) / max(tail(d["bar"], t), 1e-12) / d["base"]
        locked = (en_r < 0.2) and (rms_r < 0.5) and name != "hydro"
        print(f"[{tag}] {name:<22}{d['m4'][-1]:8.4f}{d['m8'][-1]:8.4f}{d['rms'][-1]:8.4f}{rms_r:7.3f}{en_r:8.3f}"
              f"{circ_r:7.3f}{gb:9.3f}{d['div']:7.3f}{d['cons']:8.1e}  {'LOCKED' if locked else '-'}")
