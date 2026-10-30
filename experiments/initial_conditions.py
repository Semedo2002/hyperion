"""Initial fields for the paper: corrected x stripes vs the legacy By(y) stripes.

Saves experiments/data/initial_conditions.npz (density, field, pressure, div B at t = 0).
"""
import os
import sys
import io
import contextlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config, iRHO, iBX, iBY, iPR
    from aegis.solver import MHDSolver
    from aegis.physics import cons_to_prim

from runner import BASE, DATA_DIR


def init(**kw):
    s = MHDSolver(Config(**{**BASE, **kw}))
    with contextlib.redirect_stdout(io.StringIO()):
        s.initialize()
    W = cons_to_prim(s.U, s.cfg.gamma).get()
    ng = s.ng
    f = {k: W[i, ng:-ng, ng:-ng] for k, i in (("rho", iRHO), ("Bx", iBX), ("By", iBY), ("p", iPR))}
    dx, dy = s.cfg.dx, s.cfg.dy
    f["divB"] = np.gradient(f["Bx"], dx, axis=0) + np.gradient(f["By"], dy, axis=1)
    f["x"] = s.x[ng:-ng].get()
    f["y"] = s.y[ng:-ng].get()
    return f


# corrected: x stripes, pressure balance (deep, so the profiles are visible)
new = init(B_field_type="striped", B_transverse=6.0, B_gap_fraction=0.3)
# legacy: By = B0 cos(k y), same field strength, uniform upstream pressure
old = init(B_field_type="striped", B_transverse=6.0, B_stripe_component="y")
np.savez(os.path.join(DATA_DIR, "initial_conditions.npz"),
         **{f"new_{k}": v for k, v in new.items()}, **{f"old_{k}": v for k, v in old.items()})
print(f"> max|div B| t=0: corrected {np.abs(new['divB']).max():.3g}, legacy {np.abs(old['divB']).max():.3g}")
