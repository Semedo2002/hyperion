"""Interface spectrum: RMI mode (4), stripe mode (8) and total roughness over time."""
import sys, os, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config
    from aegis.solver import MHDSolver

base = dict(div_b="glm", nx=400, ny=100, x_min=0.0, x_max=12.0, y_min=0.0, y_max=2.0, t_end=0.8, mach=10.0,
            perturbation_amp=0.02, perturbation_mode=4, density_ratio=3.0, bc_x_type="inflow_outflow",
            B_modulation_mode=4, localized_rh=True, diag_interval=20, snapshot_times=[],
            B_field_type="striped")
cases = {"hydro": dict(B_transverse=0.0, B_field_type="uniform"),
         "uniform Bx=3": dict(B_transverse=3.0, B_gap_fraction=0.99),
         "B0=3.5 eps=0.9": dict(B_transverse=3.5, B_gap_fraction=0.9),
         "B0=6 eps=0.3": dict(B_transverse=6.0, B_gap_fraction=0.3),
         "cos B0=6": dict(B_transverse=6.0)}
for name, c in cases.items():
    s = MHDSolver(Config(**{**base, **c}))
    with contextlib.redirect_stdout(io.StringIO()):
        s.initialize(); s.run()
    t = np.array(s.diag_times); A = np.array(s.diag_mode_amps)
    row = []
    for tt in (0.1, 0.4, 0.8):
        i = np.argmin(np.abs(t - tt)); a = A[i]
        rms = np.sqrt(0.5 * np.sum(a[1:]**2))       # rms interface displacement from all modes
        row.append(f"t={tt}: m4={a[4]:.4f} m8={a[8]:.4f} rms={rms:.4f}")
    print(f"> {name:<15} " + " | ".join(row), flush=True)
