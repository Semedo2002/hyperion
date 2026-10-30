"""Where does the enstrophy live? Mixing layer (0.05 < C < 0.95) vs rest of the +-0.5 window,
plus interface circulation |Gamma| per half wavelength (the RMI driver)."""
import sys, os, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config
    from aegis.solver import MHDSolver

base = dict(div_b="glm", nx=400, ny=100, x_min=0.0, x_max=12.0, y_min=0.0, y_max=2.0, t_end=0.8, mach=10.0,
            perturbation_amp=0.02, perturbation_mode=4, density_ratio=3.0, bc_x_type="inflow_outflow",
            B_modulation_mode=4, localized_rh=True, diag_interval=10**9, snapshot_times=[0.2, 0.4, 0.8])
cases = {"hydro": dict(B_transverse=0.0),
         "uniform Bx=3": dict(B_transverse=3.0, B_field_type="striped", B_gap_fraction=0.99),
         "cos B0=6 (UD)": dict(B_transverse=6.0, B_field_type="striped")}
for name, c in cases.items():
    s = MHDSolver(Config(**base, **c))
    with contextlib.redirect_stdout(io.StringIO()):
        s.initialize(); s.run()
    dx, dy = s.cfg.dx, s.cfg.dy
    row = []
    for key in sorted(s.snapshots, key=lambda k: s.snapshots[k]['t']):
        sn = s.snapshots[key]
        if sn['t'] < 0.15: continue
        om = np.gradient(sn['vy'], dx, axis=0) - np.gradient(sn['vx'], dy, axis=1)
        ens = sn['rho'] * om**2 * dx * dy
        C = sn['C']; x = sn['x']
        mix = (C > 0.05) & (C < 0.95)
        xc = np.sum(x[:, None] * mix) / max(mix.sum(), 1)
        win = np.abs(x - xc)[:, None] < 0.5 + 0 * C
        # circulation in the mixing layer per half wavelength of the interface mode
        lam2 = 2.0 / 4 / 2
        nb = (s.cfg.y_max - s.cfg.y_min) / lam2
        om_mix = np.where(mix, om, 0.0).sum(axis=0) * dx          # integral over x, per y
        bands = np.array([np.sum(np.abs(om_mix)) * dy / nb])
        row.append(f"t={sn['t']:.2f}: mix={ens[mix].sum():8.1f} win-rest={ens[win & ~mix].sum():8.1f} "
                   f"|Gamma|={np.mean(np.abs(bands)):.3f}")
    print(f"> {name:<14} " + " | ".join(row), flush=True)
