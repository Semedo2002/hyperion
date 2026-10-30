import sys, os, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from aegis.config import Config, iVX, iVY, iPR
from aegis.solver import MHDSolver
from aegis.physics import xp, cons_to_prim, prim_to_cons

def far_v(nx, balanced=True, B0=5.0, t_end=0.05):
    cfg = Config(div_b="glm", nx=nx, ny=nx // 2, t_end=t_end, mach=10.0, B_transverse=B0, B_field_type="striped",
                 B_stripe_component="x", bc_x_type="inflow_outflow", diag_interval=10**9, snapshot_times=[])
    s = MHDSolver(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        s.initialize()
        if not balanced:  # undo equilibrium: uniform upstream pressure P_total ahead of the shock
            W = cons_to_prim(s.U, cfg.gamma)
            ahead = s.X > cfg.interface_x - 0.3
            W[iPR] = xp.where(ahead, s._equilibrium["P_total"], W[iPR])
            s.U = prim_to_cons(W, cfg.gamma)
            s._inflow_U = s.U[:, :s.ng, :].copy()
        s.run()
    W = cons_to_prim(s.U, cfg.gamma); ng = s.ng
    far = s.x[ng:-ng] > 3.5
    v = xp.sqrt(W[iVX, ng:-ng, ng:-ng]**2 + W[iVY, ng:-ng, ng:-ng]**2)[far]
    return float(xp.max(v)) / np.sqrt(cfg.gamma), float(xp.sqrt(xp.mean(v**2))) / np.sqrt(cfg.gamma)

prev = None
res = []
for nx in (100, 200, 400):
    mx, rms = far_v(nx)
    res.append((nx, mx, rms))
    order = f"  order {np.log2(prev/mx):.2f}" if prev else ""
    print(f"> balanced   nx={nx}: max|v|/cs={mx:.2e} rms={rms:.2e}{order}", flush=True)
    prev = mx
mx_u, rms_u = far_v(200, balanced=False)
print(f"> unbalanced nx=200: max|v|/cs={mx_u:.2e} rms={rms_u:.2e}", flush=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "equilibrium_convergence.npz")
np.savez(out, nx=np.array([r[0] for r in res]), vmax=np.array([r[1] for r in res]),
         vrms=np.array([r[2] for r in res]), unbalanced_vmax=mx_u, unbalanced_vrms=rms_u)
