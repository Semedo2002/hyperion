"""Long-run side-by-side showcase: hydro vs locked fine sieve vs deep stripes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from aegis.config import Config
from aegis.solver import MHDSolver, PostProcessor

t_end = 0.8
base = dict(div_b="glm", nx=400, ny=100, x_min=0.0, x_max=12.0, y_min=0.0, y_max=2.0, t_end=t_end, mach=10.0,
            perturbation_amp=0.02, perturbation_mode=4, density_ratio=3.0, bc_x_type="inflow_outflow",
            B_modulation_mode=4, localized_rh=True, diag_interval=20,
            snapshot_times=list(np.linspace(0.0, t_end, 61)))
cases = {
    "Hydro": dict(B_transverse=0.0),
    "Fine sieve 3.5->3.15 (locked)": dict(B_transverse=3.5, B_field_type="striped", B_gap_fraction=0.9),
    "Deep stripes 6->1.8": dict(B_transverse=6.0, B_field_type="striped", B_gap_fraction=0.3),
}
solvers = {}
for label, c in cases.items():
    s = MHDSolver(Config(**{**base, **c})); s.initialize(); s.run()
    solvers[label] = s
PostProcessor(solvers).plot_all()
