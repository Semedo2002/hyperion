import sys, os, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.physics import xp, parallel_stripe_state

# Gap/bar momentum flux rho*vx^2 of the post-shock inflow state alone,
# using the same y-masks as the diagnostics (bar: |cos| > 0.5).
y = (np.arange(100) + 0.5) * 0.02
prof = np.cos(2 * np.pi * 4 / 2.0 * y)
bar, gap = np.abs(prof) > 0.5, np.abs(prof) <= 0.5
for B0 in (3.0, 6.0):
    st = parallel_stripe_state(5 / 3, 10.0, 1.0, "uniform_density", xp.asarray(B0 * prof))
    mf = xp.asnumpy(st["rho2"] * st["vx2"]**2)
    M = xp.asnumpy(st["M_local"])
    print(f"> B0={B0:g}: inflow gap/bar momentum flux = {mf[gap].mean() / mf[bar].mean():.3f}  "
          f"(gap mean Mach {M[gap].mean():.2f}, bar mean Mach {M[bar].mean():.2f})")
