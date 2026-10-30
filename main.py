# main.py - entry point: verification tests, single cases, and the (legacy) campaign.

import time
import os
import sys
import csv
import io
import contextlib
import numpy as np
from typing import Dict, List, Tuple

from aegis.config import Config, NVAR, EN, RHOSE, iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iCLR
from aegis.physics import xp, cp, cons_to_prim, prim_to_cons, mhd_rankine_hugoniot, sound_speed2
from aegis.solver import MHDSolver, PostProcessor, save_figure, to_numpy, OUTPUT_DIR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Verification Tests (unchanged from working code)
# ============================================================
def brio_wu_test(nx=800, t_end=0.1, plot=True):
    print("\n--- Brio-Wu Shock Tube ---"); sys.stdout.flush()
    cfg = Config(nx=nx, ny=4, x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.05,
        t_end=t_end, cfl=0.30, gamma=2.0, mach=1.0, B_transverse=0.0,
        interface_x=0.5, perturbation_amp=0.0, density_ratio=1.0,
        diag_interval=10000, snapshot_times=[t_end],
        powell_source=False, use_char_bc=False,
        bc_x_type="extrapolation", bc_y_type="periodic")
    solver = MHDSolver(cfg); ng = solver.ng
    W = xp.zeros((NVAR, solver.nx_tot, solver.ny_tot))
    x = solver.x; left = x < 0.5
    W[iRHO] = xp.where(left[:, None], 1.0, 0.125)
    W[iPR] = xp.where(left[:, None], 1.0, 0.1)
    W[iBX] = 0.75
    W[iBY] = xp.where(left[:, None], 1.0, -1.0)
    W[iCLR] = xp.where(left[:, None], 1.0, 0.0)
    solver.U = prim_to_cons(W, cfg.gamma); solver.t = 0.0; solver.step = 0
    while solver.t < cfg.t_end and solver.step < cfg.max_steps:
        W_c = cons_to_prim(solver.U, cfg.gamma)
        solver.dt = min(solver.compute_dt(W_c), cfg.t_end - solver.t)
        if solver.dt <= 1e-16: break
        solver.step_ssprk3(); solver.t += solver.dt; solver.step += 1
    W_f = cons_to_prim(solver.U, cfg.gamma)
    rho_1d = to_numpy(W_f[iRHO, ng:-ng, ng]); By_1d = to_numpy(W_f[iBY, ng:-ng, ng])
    vx_1d = to_numpy(W_f[iVX, ng:-ng, ng])
    passed = (0.1 < np.min(rho_1d) < 0.2) and (0.9 < np.max(rho_1d) < 1.05) and bool(np.any(By_1d[:-1]*By_1d[1:]<0))
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if plot:
        p_1d = to_numpy(W_f[iPR, ng:-ng, ng]); x_1d = to_numpy(x[ng:-ng])
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, data, yl, tl, c in [(axes[0,0],rho_1d,r'$\rho$','Density','b'),
            (axes[0,1],p_1d,'$p$','Pressure','r'), (axes[1,0],vx_1d,'$v_x$','Velocity','g'),
            (axes[1,1],By_1d,'$B_y$','Transverse B','m')]:
            ax.plot(x_1d, data, f'{c}-', lw=1); ax.set_ylabel(yl); ax.set_title(tl); ax.grid(True, alpha=0.3)
        fig.suptitle(f'Brio-Wu (GPU), t={t_end}, nx={nx}', fontweight='bold')
        plt.tight_layout(); save_figure(fig, 'brio_wu_test.png')
    return passed

def alfven_convergence_test(plot=True):
    print("\n--- Alfven Wave Convergence ---"); sys.stdout.flush()
    resolutions = [32, 64, 128, 256]; errors = []
    rho0=1.0; p0=0.1; Bx0=1.0; amp=1e-6; gamma=5.0/3.0; Lx=1.0
    vA = Bx0/np.sqrt(rho0); period = Lx/vA
    for nx in resolutions:
        cfg = Config(nx=nx, ny=4, x_min=0.0, x_max=Lx, y_min=0.0, y_max=0.05,
            t_end=period, cfl=0.25, gamma=gamma, mach=1.0, B_transverse=0.0,
            interface_x=0.5, perturbation_amp=0.0, density_ratio=1.0,
            diag_interval=100000, snapshot_times=[period],
            powell_source=False, bc_x_type="periodic", bc_y_type="periodic")
        s = MHDSolver(cfg); ng = s.ng; x = s.x; kx = 2.0*np.pi/Lx
        W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
        W[iRHO]=rho0; W[iPR]=p0; W[iBX]=Bx0
        W[iBY]=amp*xp.sin(kx*x)[:,None]*xp.ones(s.ny_tot)[None,:]
        W[iVY]=-amp*xp.sin(kx*x)[:,None]*xp.ones(s.ny_tot)[None,:]/np.sqrt(rho0)
        W_init = W.copy()
        s.U = prim_to_cons(W, cfg.gamma); s.t=0.0; s.step=0
        while s.t < cfg.t_end and s.step < 100000:
            W_c = cons_to_prim(s.U, cfg.gamma)
            s.dt = min(s.compute_dt(W_c), cfg.t_end - s.t)
            if s.dt <= 1e-16: break
            s.step_ssprk3(); s.t += s.dt; s.step += 1
        W_f = cons_to_prim(s.U, cfg.gamma)
        errors.append(float(np.mean(np.abs(to_numpy(W_f[iBY,ng:-ng,ng])-to_numpy(W_init[iBY,ng:-ng,ng])))))
    orders = [np.log(errors[i-1]/errors[i])/np.log(resolutions[i]/resolutions[i-1]) for i in range(1,len(errors)) if errors[i]>0 and errors[i-1]>0]
    mean_order = np.mean(orders) if orders else 0
    passed = mean_order >= 1.5
    print(f"  Mean order: {mean_order:.2f} ({'PASS' if passed else 'FAIL'})")
    if plot:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(resolutions, errors, 'bo-', lw=2, ms=8, label=f'Measured ({mean_order:.2f})')
        ref = np.array([resolutions[0], resolutions[-1]], dtype=float)
        ax.loglog(ref, errors[0]*resolutions[0]**2/ref**2, 'k--', lw=1, alpha=0.5, label='2nd order')
        ax.set_xlabel('$N_x$'); ax.set_ylabel('$L_1$ error'); ax.set_title('Alfven Wave Convergence', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout(); save_figure(fig, 'convergence_alfven.png')
    return passed


def _advance(s, t_end, max_steps=100000):
    """Advance a solver whose state was set by hand (no diagnostics)."""
    while s.t < t_end and s.step < max_steps:
        W_c = cons_to_prim(s.U, s.cfg.gamma)
        s.dt = min(s.compute_dt(W_c), t_end - s.t)
        if s.dt <= 1e-16: break
        s.step_ssprk3(); s.t += s.dt; s.step += 1


def boundary_reflection_test(nx=200, eps=1e-4, tol=1e-2):
    """Send small right-going fast and Alfven pulses through the x_max outflow
    boundary and measure the fraction of the pulse left in the domain."""
    print("\n--- Outflow Boundary Reflection ---"); sys.stdout.flush()
    rho0, p0, B0, gamma, sigma, x0 = 1.0, 1.0, 1.0, 5.0/3.0, 0.05, 0.5
    results = {}
    for wave in ("fast", "alfven"):
        cfg = Config(nx=nx, ny=4, x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.02,
            cfl=0.3, gamma=gamma, mach=1.0, powell_source=False,
            bc_x_type="inflow_outflow", bc_y_type="periodic")
        s = MHDSolver(cfg); x = s.x[:, None] * xp.ones(s.ny_tot)[None, :]
        f = eps * xp.exp(-0.5 * ((x - x0) / sigma)**2)
        W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
        W[iRHO] = rho0; W[iPR] = p0
        if wave == "fast":    # B perpendicular to x: right-going fast eigenvector
            cs2 = gamma * p0 / rho0; c = np.sqrt(cs2 + B0**2 / rho0)
            W[iBY] = B0 * (1 + f); W[iRHO] = rho0 * (1 + f)
            W[iVX] = c * f; W[iPR] = p0 + rho0 * cs2 * f
            probe = lambda Wf: xp.abs(Wf[iRHO] - rho0) / rho0
        else:                 # B along x: right-going Alfven eigenvector
            c = B0 / np.sqrt(rho0)
            W[iBX] = B0; W[iBY] = B0 * f; W[iVY] = -B0 * f / np.sqrt(rho0)
            probe = lambda Wf: xp.abs(Wf[iBY]) / B0
        s.U = prim_to_cons(W, gamma); s.t = 0.0; s.step = 0
        background = W.copy(); background[iVX] = 0; background[iVY] = 0
        background[iRHO] = rho0; background[iPR] = p0
        background[iBY] = B0 if wave == "fast" else 0.0
        s._inflow_U = prim_to_cons(background, gamma)[:, :s.ng, :].copy()
        _advance(s, (cfg.x_max - x0 + 8 * sigma) / c)
        ng = s.ng
        left = float(xp.max(probe(cons_to_prim(s.U, gamma))[ng:-ng, ng:-ng])) / eps
        results[wave] = left
        print(f"  {wave:>6s} pulse: residual amplitude after exit = {left:.2e} of incident")
    passed = all(v < tol for v in results.values())
    print(f"  Result: {'PASS' if passed else 'FAIL'} (tolerance {tol:.0e})")
    return passed


def conservation_test(nx=100, ny=50, t_end=0.05, tol=1e-10, div_tol=0.1):
    """Short sieve runs: energy must balance boundary fluxes to round-off. With constrained transport the
    face divergence must stay at round-off for every field; the legacy GLM path (cell-centred B) only
    keeps h max|div B|/max|B| below div_tol (legacy By(y) stripes start at h*k_mod ~ 0.4 and fail this)."""
    print("\n--- Conservation & Solenoidality ---"); sys.stdout.flush()
    base = dict(nx=nx, ny=ny, t_end=t_end, mach=10.0, localized_rh=True,
                bc_x_type="inflow_outflow", diag_interval=10**9, snapshot_times=[])
    cases = {
        "uniform By=1, CT": dict(B_transverse=1.0, B_field_type="uniform"),
        "striped Bz, eta=1e-2, CT": dict(B_transverse=2.0, B_field_type="striped",
                                          B_stripe_component="z", eta_resistivity=1e-2),
        "striped Bx (tension), CT": dict(B_transverse=2.0, B_field_type="striped",
                                         B_stripe_component="x"),
        "uniform By=1, GLM+Powell": dict(B_transverse=1.0, B_field_type="uniform", div_b="glm"),
    }
    passed = True
    for label, c in cases.items():
        s = MHDSolver(Config(**base, **c))
        with contextlib.redirect_stdout(io.StringIO()):
            s.initialize(); s.run()
        b = s.energy_budget()
        err = abs(b["conservation_error_rel"]); div_rel = s.div_b_max()
        exact = s.ct or c.get("B_stripe_component") == "z"
        ok = err < tol and div_rel < (1e-12 if exact else div_tol)
        passed &= ok
        print(f"  {label:<26s} E change {100*b['raw_change_rel']:+7.1f}% (inflow)  "
              f"conservation err {err:.1e}  h*max|divB|/max|B| {div_rel:.1e}  [{'PASS' if ok else 'FAIL'}]")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def equilibrium_test(B0=5.0, t_end=0.05, tol=5e-2):
    """The gas ahead of the shock must stay at rest. The balance p + Bx^2/2 = P_total
    holds exactly in the initial data but only to O(h^2) at MUSCL-reconstructed
    faces, so the residual flow is truncation error and must fall at second order:
    with CT measured 1.1e-1, 2.6e-2, 3.8e-3 c_s at nx = 100, 200, 400 (GLM, whose
    signal speeds are clamped to the global maximum and damp it more: 1.6e-2, 4.0e-3,
    1.2e-3), against 0.30 c_s with no pressure balance. Checked: order >= 1.5 between
    nx = 100 and 200 and the nx = 200 residual below tol. A well-balanced scheme would
    be needed for round-off."""
    print("\n--- Upstream Pressure Equilibrium ---"); sys.stdout.flush()
    passed = True
    for closure in ("uniform_density", "uniform_temperature"):
        res = []
        for nx in (100, 200):
            cfg = Config(nx=nx, ny=nx // 2, t_end=t_end, mach=10.0, B_transverse=B0,
                         B_field_type="striped", B_stripe_component="x",
                         equilibrium_closure=closure, bc_x_type="inflow_outflow",
                         diag_interval=10**9, snapshot_times=[])
            s = MHDSolver(cfg)
            with contextlib.redirect_stdout(io.StringIO()):
                s.initialize(); s.run()
            W = cons_to_prim(s.U, s.eos); ng = s.ng
            far = s.x[ng:-ng] > 3.5   # shock is at x ~ 1.9 at t_end
            v = xp.sqrt(W[iVX, ng:-ng, ng:-ng]**2 + W[iVY, ng:-ng, ng:-ng]**2)[far]
            res.append(float(xp.max(v)) / np.sqrt(cfg.gamma * cfg.p_bar))
        order = np.log2(res[0] / res[1])
        ok = res[1] < tol and order >= 1.5
        passed &= ok
        eq = s._equilibrium
        print(f"  {closure:<20s} B0={B0:g}: P_total={eq['P_total']:.4g}, gap Mach={eq['M_gap']:.3g}, "
              f"max|v|/c_s upstream = {res[0]:.1e}, {res[1]:.1e} at nx = 100, 200 (order {order:.2f})  "
              f"[{'PASS' if ok else 'FAIL'}]")
    try:
        Config(nx=10, eta_resistivty=1e-2)
        print("  misspelled config key accepted  [FAIL]"); passed = False
    except ValueError as e:
        print(f"  misspelled config key rejected: {e}  [PASS]")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def z_invariance_test(tol=1e-8):
    """3D runs with a z-invariant initial state must reproduce the 2D solver (ideal and resistive)."""
    print("\n--- 3D: z-invariance against the 2D solver ---"); sys.stdout.flush()
    # constant eta, and eta varying 8x across the interface through the material model
    cases = ({"eta_resistivity": 0.0}, {"eta_resistivity": 0.02},
             {"eta_model": "material", "eta_fuel": 0.005, "eta_liner": 0.04})
    return all([_z_invariance_case(kw, tol) for kw in cases])


def _z_invariance_case(eta_kw, tol):
    from aegis.solver3d import MHDSolver3D
    nx, ny = 128, 32
    dx, dy = 6.0 / nx, 2.0 / ny
    common = dict(nx=nx, ny=ny, x_max=6.0, t_end=0.1, mach=10.0, B_transverse=3.0, B_field_type="striped",
                  B_gap_fraction=0.9, perturbation_amp=0.02, bc_x_type="inflow_outflow",
                  diag_interval=10**9, snapshot_times=[], **eta_kw)
    s2 = MHDSolver(Config(**common))
    with contextlib.redirect_stdout(io.StringIO()):
        s2.initialize(); s2.run()
    s3 = MHDSolver3D(Config(**common, nz=4, z_max=4 * max(dx, dy)))
    s3.initialize(); s3.run(verbose=False)
    ng = s2.ng
    r2 = cons_to_prim(s2.U, 5.0 / 3.0)[iRHO, ng:-ng, ng:-ng]
    r3 = cons_to_prim(s3.U, 5.0 / 3.0)[iRHO, ng:-ng, ng:-ng, ng:-ng]
    diff = float(xp.max(xp.abs(r3 - r2[:, :, None]) / r2[:, :, None]))
    zvar = float(xp.max(xp.abs(r3 - r3[:, :, :1])))
    err = s3.energy_error()
    ok = diff < tol and zvar == 0.0 and err < 1e-10 and s2.step == s3.step
    label = ", ".join(f"{k}={v:g}" if not isinstance(v, str) else v for k, v in eta_kw.items())
    print(f"  {label}: steps 2D/3D {s2.step}/{s3.step}, max rel diff in rho {diff:.1e}, z-variation {zvar:.1e}, "
          f"energy error {err:.1e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def alfven_z_test():
    """Alfven wave travelling along z: tests the z-face fluxes, expects ~2nd order."""
    print("\n--- 3D: Alfven wave along z ---"); sys.stdout.flush()
    from aegis.solver3d import MHDSolver3D
    errors, res = [], [16, 32, 64]
    for nz in res:
        cfg = Config(nx=4, ny=4, nz=nz, x_max=4.0 / nz, y_max=4.0 / nz, z_max=1.0, t_end=1.0, cfl=0.25,
                     mach=1.0, powell_source=False, bc_x_type="periodic", diag_interval=10**9)
        s = MHDSolver3D(cfg)
        z = s.z[None, None, :] * xp.ones((s.nx_tot, s.ny_tot, 1))
        W = xp.zeros((NVAR, s.nx_tot, s.ny_tot, s.nz_tot))
        amp, k = 1e-6, 2 * np.pi
        W[iRHO] = 1.0; W[iPR] = 0.1; W[iBZ] = 1.0
        W[iBX] = amp * xp.sin(k * z); W[iVX] = -amp * xp.sin(k * z)
        W0 = W.copy()
        s.U = prim_to_cons(W, cfg.gamma)
        s._inflow_U = s.U[:, :s.ng].copy()
        s.run(verbose=False)
        ng = s.ng
        Wf = cons_to_prim(s.U, cfg.gamma)
        errors.append(float(xp.mean(xp.abs(Wf[iBX, ng:-ng, ng:-ng, ng:-ng] - W0[iBX, ng:-ng, ng:-ng, ng:-ng]))))
    orders = [np.log2(errors[i - 1] / errors[i]) for i in range(1, len(errors))]
    ok = np.mean(orders) >= 1.5
    print(f"  L1 errors {', '.join(f'{e:.2e}' for e in errors)}; orders {', '.join(f'{o:.2f}' for o in orders)}"
          f"  [{'PASS' if ok else 'FAIL'}]")
    return ok


def resistive_z_test(eta=0.01, tol=0.01):
    """Weak B_x = b sin(k z) in a static plasma decays as exp(-eta k^2 t); tests the z-face
    resistive fluxes and that Ohmic heating conserves total energy."""
    print("\n--- 3D: resistive diffusion along z ---"); sys.stdout.flush()
    from aegis.solver3d import MHDSolver3D
    nz, b, k, t_end = 64, 1e-3, 2 * np.pi, 1.0
    cfg = Config(nx=4, ny=4, nz=nz, x_max=4.0 / nz, y_max=4.0 / nz, z_max=1.0, t_end=t_end, cfl=0.3,
                 mach=1.0, powell_source=False, bc_x_type="periodic", eta_resistivity=eta, diag_interval=10**9)
    s = MHDSolver3D(cfg)
    z = s.z[None, None, :] * xp.ones((s.nx_tot, s.ny_tot, 1))
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot, s.nz_tot))
    W[iRHO] = 1.0; W[iPR] = 1.0; W[iBX] = b * xp.sin(k * z)
    s.U = prim_to_cons(W, cfg.gamma)
    s._inflow_U = s.U[:, :s.ng].copy()
    s.run(verbose=False)
    ng = s.ng
    Bx = cons_to_prim(s.U, cfg.gamma)[iBX, ng, ng, ng:-ng]
    zc = s.z[ng:-ng]
    amp = float(2 * xp.mean(Bx * xp.sin(k * zc)))
    expected = b * np.exp(-eta * k**2 * t_end)
    rel = abs(amp - expected) / expected
    err = s.energy_error()
    ok = rel < tol and err < 1e-12
    print(f"  amplitude {amp:.4e} vs exp(-eta k^2 t) {expected:.4e} (rel. error {rel:.1e}); "
          f"energy error {err:.1e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def _march(s, t_end, max_steps=None):
    """Plain time loop for hand-built 2D states (energy budget accumulates inside step_ssprk3)."""
    n0 = s.step
    while s.t < t_end - 1e-14 and (max_steps is None or s.step - n0 < max_steps):
        s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), t_end - s.t)
        s.step_ssprk3(); s.t += s.dt; s.step += 1


def _energy(s):
    ng = s.ng
    return float(xp.sum(s.U[EN, ng:-ng, ng:-ng] * getattr(s, "dV", s.cfg.dx * s.cfg.dy)))


def _two_layer_run(face_mean, nx, eta1, eta2, b, t_end):
    # static plasma, Bz(x) held at b on x = 0 and 0 on x = 1, eta1 left of x = 0.5, eta2 right
    class Dirichlet(MHDSolver):
        def apply_bc(self, U):
            super().apply_bc(U)
            ng = self.ng
            for sl, mirror, val in ((slice(0, ng), slice(2 * ng - 1, ng - 1, -1), b),
                                    (slice(-ng, None), slice(-ng - 1, -2 * ng - 1, -1), 0.0)):
                bz = 2 * val - U[iBZ, mirror]                  # face value = val
                U[EN, sl] += 0.5 * (bz**2 - U[iBZ, sl]**2)     # keep the ghost pressure unchanged
                U[iBZ, sl] = bz
            return U
    cfg = Config(nx=nx, ny=4, x_max=1.0, y_max=4.0 / nx, t_end=t_end, cfl=0.3, mach=1.0, B_transverse=0.0,
                 powell_source=False, bc_x_type="extrapolation", bc_y_type="periodic", eta_model="material",
                 eta_fuel=eta1, eta_liner=eta2, eta_face_mean=face_mean, diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = Dirichlet(cfg)
    x = s.x[:, None] * xp.ones((1, s.ny_tot))
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
    W[iRHO] = 1.0
    W[iBZ] = b * (1 - x)
    W[iPR] = 1.0 - 0.5 * W[iBZ]**2
    W[iCLR] = xp.where(x > 0.5, 1.0, 0.0)
    s.U = prim_to_cons(W, cfg.gamma); s.t = 0.0; s.step = 0
    E0 = _energy(s)
    _march(s, t_end)
    ng = s.ng
    xc = to_numpy(s.x[ng:-ng])
    Bz = to_numpy(cons_to_prim(s.U, cfg.gamma)[iBZ, ng:-ng, ng])
    s1 = -2 * b / (1 + eta1 / eta2); s2 = eta1 * s1 / eta2
    exact = np.where(xc < 0.5, b + s1 * xc, b + 0.5 * s1 + s2 * (xc - 0.5))
    err = np.max(np.abs(Bz - exact)) / b
    budget = abs(_energy(s) - E0 - s._cumulative_boundary_energy) / abs(E0)
    return err, budget, s.step


def resistive_layers_test(tol=1e-3):
    """Two-layer steady magnetic diffusion: eta jumps 4x at x = 0.5 (on a cell face). The steady Bz is
    piecewise linear with eta dBz/dx continuous (continuity of E_y). With harmonic face averaging the
    discrete steady state is exact, so the error is only the decay of the transient; the arithmetic mean
    misplaces the kink and is reported for comparison."""
    print("\n--- Resistive: two-layer steady diffusion ---"); sys.stdout.flush()
    eta1, eta2, b, nx, t_end = 0.04, 0.01, 1e-3, 16, 40.0
    err_h, bud_h, steps = _two_layer_run("harmonic", nx, eta1, eta2, b, t_end)
    err_a, bud_a, _ = _two_layer_run("arithmetic", nx, eta1, eta2, b, t_end)
    ok = err_h < tol and err_a > 10 * err_h and max(bud_h, bud_a) < 1e-12
    print(f"  eta {eta1:g} | {eta2:g}, nx={nx}, {steps} steps: max |Bz - exact|/b harmonic {err_h:.1e}, "
          f"arithmetic {err_a:.1e}; energy budget residual {max(bud_h, bud_a):.1e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def ohmic_heating_test(tol=0.01):
    """Periodic box, all three B components, eta varying smoothly by 10x through the material model.
    Total energy must close to round-off, and the thermal energy gained must match the accumulated
    Ohmic heating integral of eta |J|^2 (to the time-stepping error). The field is not force-free, so
    the magnetic loss is Ohmic heat plus work done by the Lorentz force (reported, not asserted)."""
    print("\n--- Resistive: Ohmic heating energy closure ---"); sys.stdout.flush()
    n, b = 64, 0.05
    cfg = Config(nx=n, ny=n, x_max=1.0, y_max=1.0, t_end=1.0, cfl=0.3, mach=1.0, B_transverse=0.0,
                 powell_source=False, bc_x_type="periodic", bc_y_type="periodic", eta_model="material",
                 eta_fuel=0.002, eta_liner=0.02, diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    X, Y = s.X, s.Y
    k = 2 * np.pi
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
    W[iRHO] = 1.0
    W[iBX] = b * xp.sin(k * Y)
    W[iBY] = b * xp.sin(k * X)
    W[iBZ] = b * xp.sin(k * X) * xp.sin(k * Y)
    W[iPR] = 1.0 - 0.5 * (W[iBX]**2 + W[iBY]**2 + W[iBZ]**2)
    W[iCLR] = 0.5 + 0.5 * xp.cos(k * (X + Y))
    s.U = prim_to_cons(W, cfg.gamma); s.t = 0.0; s.step = 0
    s.apply_bc(s.U)
    ng = s.ng; dA = cfg.dx * cfg.dy

    def parts():
        Wi = cons_to_prim(s.U, cfg.gamma)[:, ng:-ng, ng:-ng]
        mag = 0.5 * float(xp.sum(Wi[iBX]**2 + Wi[iBY]**2 + Wi[iBZ]**2)) * dA
        th = float(xp.sum(Wi[iPR])) * dA / (cfg.gamma - 1)
        kin = 0.5 * float(xp.sum(Wi[iRHO] * (Wi[iVX]**2 + Wi[iVY]**2 + Wi[iVZ]**2))) * dA
        return mag, th, kin

    E0 = _energy(s); m0, th0, k0 = parts()
    _march(s, cfg.t_end)
    m1, th1, k1 = parts()
    Q = s.cumulative_ohmic_heating
    err = abs(_energy(s) - E0) / E0
    rel_th = abs((th1 - th0) - Q) / Q
    ok = err < 1e-12 and rel_th < tol
    print(f"  {s.step} steps: Ohmic heat {Q:.5e}, thermal gain {th1 - th0:.5e} ({100 * rel_th:.2f}% off); "
          f"magnetic loss {m0 - m1:.5e} = Ohmic + kinetic gain {k1 - k0:.2e} + rest {m0 - m1 - Q - (k1 - k0):.1e}; "
          f"total energy error {err:.1e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


# Multi-material EOS tests. Units: g/cm^3, km/s, mm, us, GPa (1 g/cm^3 (km/s)^2 = 1 GPa).
AL = dict(rho0=2.70, c0=5.328, s=1.338, gamma0=2.0)
CU = dict(rho0=8.93, c0=3.940, s=1.489, gamma0=2.0)


def _eos_tube(nx, t_end, x0, left, right, bc="extrapolation", max_steps=None, **cfg_kw):
    """1D Riemann problem on an nx x 4 strip; left/right = (rho, u, p, liner fraction)."""
    cfg = Config(nx=nx, ny=4, x_max=1.0, y_max=4.0 / nx, t_end=t_end, cfl=0.3, mach=1.0, B_transverse=0.0,
                 powell_source=False, bc_x_type=bc, bc_y_type="periodic", diag_interval=10**9,
                 snapshot_times=[], **cfg_kw)
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    x = s.x[:, None] * xp.ones((1, s.ny_tot))
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
    inside = x0(x) if callable(x0) else (x < x0)
    for k, i in ((0, iRHO), (1, iVX), (2, iPR), (3, iCLR)):
        W[i] = xp.where(inside, left[k], right[k])
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    s.apply_bc(s.U)
    E0 = _energy(s)
    _march(s, t_end, max_steps)
    ng = s.ng
    Wf = cons_to_prim(s.U, s.eos)
    prof = {k: to_numpy(Wf[i, ng:-ng, ng]) for k, i in (("rho", iRHO), ("u", iVX), ("p", iPR), ("a", iCLR))}
    prof["x"] = to_numpy(s.x[ng:-ng])
    prof["c2"] = to_numpy(sound_speed2(Wf, s.eos)[ng:-ng, ng])
    prof["budget"] = abs(_energy(s) - E0 - s._cumulative_boundary_energy) / abs(E0)
    return s, prof


def _shock_from(prof, p_lo, p_hi, side):
    """x where p first exceeds the mid value, scanning in from the given side"""
    mid = 0.5 * (p_lo + p_hi)
    i = np.where(prof["p"] > mid)[0]
    return prof["x"][i[-1]] if side == "right" else prof["x"][i[0]]


def eos_impact_test(tol=0.02):
    """Al flyer at 10 km/s on targets at rest, compared with the exact jump conditions.
    (1) stiffened-gas Al on stiffened-gas Cu (two materials, volume-fraction interface): impedance match of
        the two stiffened-gas Hugoniots gives interface velocity, pressure and both shocks.
    (2) Mie-Gruneisen Al on Mie-Gruneisen Al: symmetric, u_p = V/2 and the state must lie on the reference
        Hugoniot Us = c0 + s u_p, p = rho0 Us u_p.
    Shock speeds are measured relative to the unshocked material ahead of each shock."""
    from aegis.eos import stiffened_from_hugoniot, stiffened_hugoniot
    print("\n--- EOS: 10 km/s aluminium impact ---"); sys.stdout.flush()
    V, p0, nx, t_end = 10.0, 1e-4, 400, 0.035
    gA, piA = stiffened_from_hugoniot(AL["rho0"], AL["c0"], AL["s"], 5.0)
    gC, piC = stiffened_from_hugoniot(CU["rho0"], CU["c0"], CU["s"], 3.0)
    # impedance match: p_Cu(u) = p_Al(V - u)
    lo, hi = 0.0, V
    for _ in range(200):
        u = 0.5 * (lo + hi)
        if stiffened_hugoniot(CU["rho0"], p0, gC, piC, u)[0] > stiffened_hugoniot(AL["rho0"], p0, gA, piA, V - u)[0]:
            hi = u
        else:
            lo = u
    pC, rC, usC = stiffened_hugoniot(CU["rho0"], p0, gC, piC, u)
    pA, rA, usA = stiffened_hugoniot(AL["rho0"], p0, gA, piA, V - u)
    s, P = _eos_tube(nx, t_end, 0.5, (AL["rho0"], V, p0, 1.0), (CU["rho0"], 0.0, p0, 0.0),
                     eos_fuel="stiffened", gamma=gC, p_inf=piC, eos_liner="stiffened", gamma_liner=gA,
                     p_inf_liner=piA)
    t = s.t
    xs_C, xs_A = _shock_from(P, p0, pC, "right"), _shock_from(P, p0, pA, "left")
    xc = 0.5 + u * t
    h = 1.0 / nx
    cu = (P["x"] > xc + 8 * h) & (P["x"] < xs_C - 8 * h)
    al = (P["x"] > xs_A + 8 * h) & (P["x"] < xc - 8 * h)
    errs = {"Us_Cu": abs((xs_C - 0.5) / t - usC) / usC,
            "Us_Al": abs((0.5 - xs_A) / t + V - usA) / usA,             # relative to the unshocked flyer
            "p": abs(np.mean(P["p"][cu | al]) - pC) / pC,
            "rho_Cu": abs(np.mean(P["rho"][cu]) - rC) / rC, "rho_Al": abs(np.mean(P["rho"][al]) - rA) / rA,
            "u": abs(np.mean(P["u"][cu | al]) - u) / u}
    ok1 = max(errs.values()) < tol and P["budget"] < 1e-12
    print(f"  stiffened Al (gamma {gA:.3f}, p_inf {piA:.2f} GPa) -> Cu (gamma {gC:.3f}, p_inf {piC:.2f} GPa): "
          f"exact p {pC:.1f} GPa, u {u:.3f}, Us Cu {usC:.3f}, rho Cu {rC:.3f}, rho Al {rA:.3f}")
    print("    rel. errors " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items())
          + f"; energy budget {P['budget']:.1e}  [{'PASS' if ok1 else 'FAIL'}]")
    mg = dict(eos_liner="mie_gruneisen", mg_rho0=AL["rho0"], mg_c0=AL["c0"], mg_s=AL["s"], mg_gamma0=AL["gamma0"])
    s, P = _eos_tube(nx, t_end, 0.5, (AL["rho0"], V, 0.0, 1.0), (AL["rho0"], 0.0, 0.0, 1.0), **mg)
    up = V / 2
    us = AL["c0"] + AL["s"] * up
    pH, rH = AL["rho0"] * us * up, AL["rho0"] * us / (us - up)
    xr, xl = _shock_from(P, 0.0, pH, "right"), _shock_from(P, 0.0, pH, "left")
    mid = (P["x"] > xl + 8 * h) & (P["x"] < xr - 8 * h) & (np.abs(P["x"] - 0.5 - up * s.t) > 8 * h)
    errs = {"Us": abs((xr - 0.5) / s.t - us) / us, "Us_left": abs((0.5 - xl) / s.t + V - us) / us,
            "p": abs(np.mean(P["p"][mid]) - pH) / pH,
            "rho": abs(np.mean(P["rho"][mid]) - rH) / rH, "u": abs(np.mean(P["u"][mid]) - up) / up}
    ok2 = max(errs.values()) < tol and P["budget"] < 1e-12
    print(f"  Mie-Gruneisen Al -> Al: Hugoniot u_p {up:g}, Us {us:.3f}, p {pH:.1f} GPa, rho {rH:.3f}")
    print("    rel. errors " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items())
          + f"; energy budget {P['budget']:.1e}  [{'PASS' if ok2 else 'FAIL'}]")
    return ok1 and ok2


def eos_interface_test():
    """Aluminium slab (stiffened, p_inf ~ 27 GPa) carried through gas at uniform p and u, periodic box.
    The exact solution keeps p and u uniform. Volume-fraction tracer: must hold to round-off. Mass-fraction
    blending (the conserved tracer) is run for 20 steps for comparison and reported: its spurious pressures
    reach the cavitation floor within a few steps and the fake sound speeds collapse the time step."""
    from aegis.eos import stiffened_from_hugoniot
    print("\n--- EOS: material interface in pressure equilibrium ---"); sys.stdout.flush()
    gA, piA = stiffened_from_hugoniot(AL["rho0"], AL["c0"], AL["s"], 5.0)
    p0, u0 = 1e-4, 1.0
    slab = lambda x: (x > 0.3) & (x < 0.6)
    res = {}
    for mode, t_end, nmax in (("volume_fraction", 0.2, None), ("mass_fraction", 0.002, 20)):
        try:
            s, P = _eos_tube(100, t_end, slab, (AL["rho0"], u0, p0, 1.0), (1e-3, u0, p0, 0.0), bc="periodic", max_steps=nmax,
                             gamma=1.4, eos_liner="stiffened", gamma_liner=gA, p_inf_liner=piA,
                             material_tracer=mode)
            res[mode] = (np.max(np.abs(P["p"] - p0)) / p0, np.max(np.abs(P["u"] - u0)) / u0, P["budget"])
        except Exception as e:                      # mass-fraction blending can crash outright
            res[mode] = (float("inf"), float("inf"), float("inf"))
            print(f"  {mode}: {type(e).__name__}: {e}")
    vf, mf = res["volume_fraction"], res["mass_fraction"]
    ok = vf[0] < 1e-8 and vf[1] < 1e-8 and vf[2] < 1e-12
    print(f"  max |p - p0|/p0: volume fraction {vf[0]:.1e} (t = 0.2), mass fraction {mf[0]:.1e} (20 steps); max |u - u0|/u0: "
          f"{vf[1]:.1e}, {mf[1]:.1e}; energy budget (vf) {vf[2]:.1e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def eos_positivity_test():
    """Aluminium (stiffened) pulled apart symmetrically. At v = 5 km/s the centre approaches cavitation;
    the exact centre state has P = p + p_inf = P0 (1 - (g-1) v/(2 c0))^(2g/(g-1)). At v = 8 km/s (beyond
    2 c0/(g-1)) the exact solution has a cavity. Checks p + p_inf > 0, c^2 > 0, finite state; also a gas /
    aluminium interface pulled apart."""
    from aegis.eos import stiffened_from_hugoniot
    print("\n--- EOS: positivity under extreme expansion ---"); sys.stdout.flush()
    gA, piA = stiffened_from_hugoniot(AL["rho0"], AL["c0"], AL["s"], 5.0)
    st = dict(eos_liner="stiffened", gamma_liner=gA, p_inf_liner=piA)
    p0 = 1e-4
    c0 = np.sqrt(gA * (p0 + piA) / AL["rho0"])
    ok, lines = True, []
    for name, v, left, right in (("Al, v = 5", 5.0, None, None), ("Al, v = 8 (cavitates)", 8.0, None, None),
                                 ("gas | Al, v = 8", 8.0, (1e-3, -8.0, p0, 0.0), (AL["rho0"], 8.0, p0, 1.0))):
        L = left or (AL["rho0"], -v, p0, 1.0)
        R = right or (AL["rho0"], v, p0, 1.0)
        s, P = _eos_tube(200, 0.04, 0.5, L, R, gamma=1.4, **st)
        pinf_loc = P["a"] * piA                      # p_inf of the local mixture (exact for a = 0, 1)
        fine = all(np.all(np.isfinite(P[k])) for k in ("rho", "u", "p"))
        good = fine and np.min(P["c2"]) >= 0 and np.min(P["p"] + pinf_loc) > 0 and np.min(P["rho"]) > 0
        msg = (f"  {name}: min(p + p_inf) {np.min(P['p'] + pinf_loc):.2e} GPa, min c^2 {np.min(P['c2']):.2e}, "
               f"min rho {np.min(P['rho']):.2e}")
        if v == 5.0 and left is None:
            Pc = (p0 + piA) * (1 - (gA - 1) * v / (2 * c0))**(2 * gA / (gA - 1))
            num = np.min(P["p"]) + piA
            msg += f"; centre P {num:.3f} vs exact {Pc:.3f} GPa"
        lines.append(msg + f"  [{'PASS' if good else 'FAIL'}]")
        ok = ok and good
    print("\n".join(lines))
    return ok


def radiation_decay_test():
    """Uniform hydrogen plasma, T0 = 500 eV, n = 1e20 cm^-3, periodic box (the interior of a hot sphere
    before any rarefaction arrives). Optically thin bremsstrahlung gives 3 n k dT/dt = -1.69e-38 n^2 sqrt(T)
    (SI, T in eV), so T(t) = (sqrt(T0) - K t/2)^2 with K = 1.69e-38 n/(3 e), until T reaches T_min = 10 eV,
    where cooling stops. K is computed here from the SI constants, independently of the code-unit
    conversion in units.py. Also: no cooling in liner material (tracer 1) or below T_min."""
    from aegis.units import bremsstrahlung_kwargs, AMU, E_CHARGE
    print("\n--- Radiation: bremsstrahlung decay of a uniform hot plasma ---"); sys.stdout.flush()
    mp, n0, T0, Tmin = 1.67262192e-27, 1e26, 500.0, 10.0
    v0 = np.sqrt(2 * E_CHARGE * T0 / mp)              # code temperature p/rho = 1 is 500 eV
    t0 = 1.0 / v0                                     # L0 = 1 m
    rad = bremsstrahlung_kwargs(n0 * mp, v0, 1.0, z_eff=1.0, ion_mass_amu=mp / AMU, t_min_ev=Tmin)
    K = 1.69e-38 * n0 / (3 * E_CHARGE)

    def box(theta0, C, **kw):
        cfg = Config(nx=8, ny=8, x_max=1.0, y_max=1.0, t_end=1.0, cfl=0.3, mach=1.0, gamma=5.0 / 3.0,
                     B_transverse=0.0, powell_source=False, bc_x_type="periodic", bc_y_type="periodic",
                     diag_interval=10**9, snapshot_times=[], **rad, **kw)
        with contextlib.redirect_stdout(io.StringIO()):
            s = MHDSolver(cfg)
        W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
        W[iRHO], W[iPR], W[iCLR] = 1.0, theta0, C
        s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
        return s

    def theta(s):
        Wi = cons_to_prim(s.U, s.eos)[:, s.ng:-s.ng, s.ng:-s.ng]
        th = Wi[iPR] / Wi[iRHO]
        return float(xp.mean(th)), float(xp.max(th) - xp.min(th))

    s = box(1.0, 0.0)
    E0 = _energy(s)
    err_T, err_E, spread = 0.0, 0.0, 0.0
    for t_us in (2.0, 5.0, 8.0, 10.5, 12.0):
        _march(s, t_us * 1e-6 / t0)
        th, sp = theta(s)
        T_exact = max((np.sqrt(T0) - 0.5 * K * t_us * 1e-6)**2, Tmin)
        err_T = max(err_T, abs(th * T0 - T_exact) / T_exact)
        err_E = max(err_E, abs(_energy(s) + s.cumulative_radiated_energy - E0) / E0)
        spread = max(spread, sp)
    t_floor = 2 * (np.sqrt(T0) - np.sqrt(Tmin)) / K
    steps = s.step
    # cutoffs: liner material (tracer 1, kept exact by the volume-fraction tracer) and T < T_min
    liner = box(1.0, 1.0, material_tracer="volume_fraction")
    cold = box(0.5 * Tmin / T0, 0.0)
    for b in (liner, cold):
        _march(b, 2e-6 / t0)
    still = (liner.cumulative_radiated_energy == 0.0 and cold.cumulative_radiated_energy == 0.0
             and abs(theta(liner)[0] - 1.0) < 1e-14 and abs(theta(cold)[0] - 0.5 * Tmin / T0) < 1e-14)
    ok = err_T < 1e-10 and err_E < 1e-13 and spread < 1e-12 and still
    print(f"  K = {K:.4e} eV^1/2/s, T reaches {Tmin:g} eV at {t_floor * 1e6:.2f} us; {steps} steps to 12 us: "
          f"max |T - T_exact|/T {err_T:.1e}, |E + E_rad - E0|/E0 {err_E:.1e}, radiated "
          f"{100 * s.cumulative_radiated_energy / E0:.1f}% of E0; no cooling in liner / below T_min: "
          f"{'yes' if still else 'NO'}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def _hot_spot(cls, n, angle, kappa0, x0, amp=1e-3, sigma=0.1, nz=None, dz_factor=1.0):
    """Static uniform plasma (rho = 1, T = p/rho = 1) with a small Gaussian hot spot and a uniform field of
    unit strength in the x-y plane at `angle`; periodic box [-1, 1]^2 (x [-1, 1]^2 x thin z in 3D)."""
    kw = dict(nx=n, ny=n, x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, t_end=1.0, cfl=0.3, mach=1.0,
              gamma=5.0 / 3.0, B_transverse=0.0, powell_source=False, bc_x_type="periodic", bc_y_type="periodic",
              diag_interval=10**9, snapshot_times=[], cond_kappa0=kappa0, cond_x0=x0)
    if nz:
        kw.update(nz=nz, z_max=nz * dz_factor * 2.0 / n)
    with contextlib.redirect_stdout(io.StringIO()):
        s = cls(Config(**kw))
    shape = (NVAR,) + s.U.shape[1:]
    W = xp.zeros(shape)
    X = s.x.reshape((-1,) + (1,) * (len(shape) - 2))
    Y = s.y.reshape((1, -1) + (1,) * (len(shape) - 3))
    W[iRHO] = 1.0
    W[iPR] = 1.0 + amp * xp.exp(-(X**2 + Y**2) / (2 * sigma**2))
    W[iBX], W[iBY] = np.cos(angle), np.sin(angle)
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    s.apply_bc(s.U)
    return s


def conduction_test():
    """Magnetized conduction: small Gaussian hot spot (dT/T = 1e-3) in a uniform field, k_par/k_perp = 1e4,
    D_par = 0.01, integrated to t = 2 with four RKL2 super-steps. For linear diffusion the second moment of
    the excess temperature grows by exactly 2 D t along each principal direction, so the spread along and
    across B is compared with that. Field along x (grid-aligned: exact up to the 0.25% variation of k with
    T) and at 45 deg, where the limited transverse gradients add numerical perpendicular diffusion that must
    fall with resolution (64^2, 128^2). Also: no new extrema (to round-off), energy conserved."""
    from aegis.physics import conduct
    print("\n--- Conduction: anisotropic hot spot ---"); sys.stdout.flush()
    from aegis.transport import x_for_kappa_ratio
    g, D_par, ratio, t_end = 5.0 / 3.0, 0.01, 1e4, 2.0
    kappa0, x0 = D_par / (g - 1), x_for_kappa_ratio(ratio)       # Braginskii k_perp fit
    ok, out, leak = True, [], {}
    for name, ang, n in (("B along x, 64^2", 0.0, 64), ("B at 45 deg, 64^2", np.pi / 4, 64),
                         ("B at 45 deg, 128^2", np.pi / 4, 128)):
        s = _hot_spot(MHDSolver, n, ang, kappa0, x0)
        ng, cfg = s.ng, s.cfg
        Ui = s.U[:, ng:-ng, ng:-ng]
        X, Y = (to_numpy(a)[ng:-ng, ng:-ng] for a in (s.X, s.Y))
        u, v = X * np.cos(ang) + Y * np.sin(ang), -X * np.sin(ang) + Y * np.cos(ang)

        def moments():
            w = to_numpy(cons_to_prim(s.U, s.eos)[iPR, ng:-ng, ng:-ng]) - 1.0
            return np.sum(w * u**2) / np.sum(w), np.sum(w * v**2) / np.sum(w), w
        su0, sv0, w0 = moments()
        E0 = float(xp.sum(Ui[EN]))
        stages = 0
        for _ in range(4):
            stages += conduct(Ui, s.eos, cfg.cond_kappa0, cfg.cond_x0, cfg.cond_tracer_max, (cfg.dx, cfg.dy),
                              (True, True), t_end / 4, perp=cfg.cond_perp)
        su, sv, w = moments()
        par, perp = (su - su0) / (2 * D_par * t_end), (sv - sv0) / (2 * D_par * t_end)
        bounded = w.max() <= w0.max() and w.min() >= -1e-10 * w0.max()
        dE = abs(float(xp.sum(Ui[EN])) - E0) / E0
        if ang == 0:
            good = abs(par - 1) < 0.01 and abs(perp * ratio - 1) < 0.05
        else:
            leak[n] = perp
            good = abs(par - 1) < 0.05 and perp < 0.05 and (n == 64 or perp < leak[64])
        good = good and bounded and dE < 1e-13
        ok = ok and good
        out.append(f"  {name}: spread along B {par:.4f} x 2 D_par t, across B {perp:.2e} x 2 D_par t "
                   f"(exact {1 / ratio:.0e}); max {w.max() / w0.max():.3f} of initial, min {w.min() / w0.max():+.1e}; "
                   f"{stages} RKL2 stages vs {t_end / (0.5 * cfg.dx**2 / (4 * D_par)):.0f} explicit steps; energy "
                   f"{dE:.1e}  [{'PASS' if good else 'FAIL'}]")
    print("\n".join(out))
    print(f"  45 deg numerical cross-field leakage falls with resolution at order {np.log2(leak[64] / leak[128]):.2f}")
    return ok


def wall_conduction_test(tol=0.01):
    """Fuel-liner heat exchange: a fuel slab on [0, 0.5] (insulated at x = 0) at T0 + d against a cold liner
    wall at x = 0.5 held at T0 (cond_wall="isothermal"). The mean excess temperature follows the exact series
    sum 8/((2n+1)^2 pi^2) exp(-D k_n^2 t), k_n = (2n+1) pi/(2L). With the field along the wall normal the wall
    sees k_par; with the field parallel to the wall it sees k_perp = k_par/10, so the same decay takes 10x
    longer: checks the wall flux and its magnetic insulation. Energy moved into the liner cells is conserved."""
    from aegis.physics import conduct
    print("\n--- Conduction: fuel-liner wall exchange ---"); sys.stdout.flush()
    from aegis.transport import x_for_kappa_ratio
    g, D, ratio, d, nx = 5.0 / 3.0, 0.01, 10.0, 1e-3, 64
    kappa0, x0, Lf = D / (g - 1), x_for_kappa_ratio(ratio), 0.5
    ok, out = True, []
    for name, Bdir, Deff in (("field along the wall normal", (1.0, 0.0), D), ("field along the wall", (0.0, 1.0), D / ratio)):
        cfg = Config(nx=nx, ny=4, x_max=1.0, y_max=4.0 / nx, t_end=1.0, mach=1.0, gamma=g, B_transverse=0.0,
                     bc_x_type="extrapolation", bc_y_type="periodic", material_tracer="volume_fraction",
                     cond_kappa0=kappa0, cond_x0=x0, cond_wall="isothermal", cond_wall_T=1.0,
                     diag_interval=10**9, snapshot_times=[])
        with contextlib.redirect_stdout(io.StringIO()):
            s = MHDSolver(cfg)
        x = s.x[:, None] * xp.ones((1, s.ny_tot))
        fuel = x < Lf
        W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
        W[iRHO] = 1.0
        W[iPR] = xp.where(fuel, 1.0 + d, 1.0)
        W[iCLR] = xp.where(fuel, 0.0, 1.0)
        W[iBX], W[iBY] = Bdir
        s.U = prim_to_cons(W, s.eos)
        ng = s.ng
        Ui = s.U[:, ng:-ng, ng:-ng]
        E0 = float(xp.sum(Ui[EN]))
        t_end = 1.0 / (Deff * (np.pi / (2 * Lf))**2)         # one e-fold of the slowest mode
        for _ in range(8):
            conduct(Ui, s.eos, kappa0, x0, 0.1, (cfg.dx, cfg.dy), (False, True), t_end / 8, wall_T=1.0, perp=cfg.cond_perp)
        Wf = cons_to_prim(s.U, s.eos)[:, ng:-ng, ng:-ng]
        f = to_numpy(fuel[ng:-ng, ng:-ng])
        excess = float(np.mean(to_numpy(Wf[iPR] / Wf[iRHO])[f] - 1.0)) / d
        exact = sum(8 / ((2 * n + 1)**2 * np.pi**2) * np.exp(-(2 * n + 1)**2) for n in range(200))
        dE = abs(float(xp.sum(Ui[EN])) - E0) / E0
        good = abs(excess / exact - 1) < tol and dE < 1e-13
        ok = ok and good
        out.append(f"  {name}: t = {t_end:.1f}, mean excess {excess:.4f} of the initial (exact {exact:.4f}); "
                   f"energy {dE:.1e}  [{'PASS' if good else 'FAIL'}]")
    # second law at the wall: field 80 deg from the wall normal, k_par/k_perp = 1e4, fuel temperature varying strongly
    # along the wall; the cold (T = 0) liner may only gain heat
    from aegis.physics import conduction_div
    n = 32
    X, Yg = xp.meshgrid(xp.arange(n) / n, xp.arange(n) / n, indexing="ij")
    a = xp.where(X >= 0.5, 1.0, 0.0)
    theta = xp.where(a > 0, 0.0, 1.0 + 0.9 * xp.sin(2 * np.pi * 3 * Yg))
    ang = np.radians(80.0)
    B = (xp.full_like(X, np.cos(ang)), xp.full_like(X, np.sin(ang)), xp.zeros_like(X))
    div = conduction_div(theta, B, xp.ones_like(X), a, 1.0, x_for_kappa_ratio(1e4), 0.1, (1 / n, 1 / n),
                         (False, True), wall_T=0.0)
    gain_min = float(xp.min(div[n // 2]))                       # liner column at the wall
    good = gain_min >= 0.0
    ok = ok and good
    out.append(f"  oblique field at a cold wall, k_par/k_perp = 1e4: min heat gain of a liner cell {gain_min:.2e} "
               f"(must be >= 0)  [{'PASS' if good else 'FAIL'}]")
    print("\n".join(out))
    return ok


def transport_test():
    """Magnetized transport (aegis/transport.py). (1) Braginskii Z = 1 fits: k_perp/k_par = 1 at x = 0 and
    4.664/(3.1616 x^2) at large x; beta_wedge = 0.81 x and 1.5/x. (2) Nernst advection: an isobaric slab
    (p = 1, rho = 1/T) with T = 1 + 0.02 (x - 0.5) and weak field bumps in B_y (constrained-transport faces)
    and in the out-of-plane B_w (cell centred); both must travel down the temperature gradient along the exact
    path dX/dt = v_N(X), v_N = -c_N (beta_wedge/|B|) dT/dx, with div B at round-off. (3) Ettingshausen: a
    strong sheared field in a periodic box conserves total energy to round-off while its heat flux
    redistributes thermal energy."""
    from aegis import transport as tr
    print("\n--- Transport: Braginskii fits, Nernst, Ettingshausen ---"); sys.stdout.flush()
    fits = (abs(tr.kperp_ratio(0.0) - 1) < 1e-3 and abs(tr.kperp_ratio(1e3) * 3.1616 * 1e6 / 4.664 - 1) < 1e-3
            and abs(tr.beta_wedge_over_x(0.0) - 3.053 / 3.7703) < 1e-12 and abs(1e3 * 1e3 * tr.beta_wedge_over_x(1e3) / 1.5 - 1) < 1e-3)
    # (2) Nernst advection; small x so beta_wedge/|B| = 0.81 x0 T^1.5/rho
    nx, cN, x0, t_end, X0, w = 200, 31.0, 1.0, 0.6, 0.7, 0.05
    cfg = Config(nx=nx, ny=4, x_max=1.0, y_max=4.0 / nx, t_end=t_end, cfl=0.3, mach=1.0, gamma=5 / 3,
                 B_transverse=1e-6, bc_x_type="extrapolation", bc_y_type="periodic", cond_x0=x0, nernst_coeff=cN,
                 diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    T = lambda x: 1.0 + 0.02 * (x - 0.5)
    x = s.x[:, None] * xp.ones((1, s.ny_tot))
    bump = 1e-4 * xp.exp(-((x - X0) / w)**2)
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
    W[iPR], W[iRHO] = 1.0, 1.0 / T(x)
    W[iBY], W[iBZ] = bump, bump
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    E0 = _energy(s)
    _march(s, t_end)
    ng = s.ng
    xc = to_numpy(s.x[ng:-ng])
    Wf = cons_to_prim(s.U, s.eos)
    cen = lambda b: float(np.sum(xc * b) / np.sum(b))
    xy, xw = cen(to_numpy(Wf[iBY, ng:-ng, ng])), cen(to_numpy(Wf[iBZ, ng:-ng, ng]))
    Xe, n = X0, 4000                                         # exact path: dX/dt = v_N(X)
    for _ in range(n):
        th = T(Xe)
        Xe += -cN * x0 * th**1.5 * th * (3.053 / 3.7703) * 0.02 * (t_end / n)   # rho = 1/T
    budget = abs(_energy(s) - E0 - s._cumulative_boundary_energy) / abs(E0)
    div = s.div_b_max()
    nern_ok = abs(xy - Xe) < 1.5 / nx and abs(xw - Xe) < 1.5 / nx and div < 1e-12 and budget < 1e-12
    # (3) Ettingshausen in a periodic box: B_w(x) shear, uniform T
    cfg = Config(nx=64, ny=4, x_max=1.0, y_max=4.0 / 64, t_end=0.2, cfl=0.3, mach=1.0, gamma=5 / 3,
                 B_transverse=1.0, bc_x_type="periodic", bc_y_type="periodic", cond_x0=2.0, nernst_coeff=1.0,
                 diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
    W[iRHO], W[iPR] = 1.0, 1.0
    W[iBZ] = xp.sin(2 * np.pi * s.x)[:, None]
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    p0 = to_numpy(cons_to_prim(s.U, s.eos)[iPR, s.ng:-s.ng, s.ng]).copy()
    E0 = _energy(s)
    _march(s, 0.2)
    dE = abs(_energy(s) - E0) / E0
    dp = float(np.max(np.abs(to_numpy(cons_to_prim(s.U, s.eos)[iPR, s.ng:-s.ng, s.ng]) - p0)))
    ett_ok = dE < 1e-13 and dp > 1e-4
    ok = fits and nern_ok and ett_ok
    print(f"  Braginskii fits (limits of k_perp/k_par and beta_wedge): {'ok' if fits else 'WRONG'}")
    print(f"  Nernst: field bumps moved {X0:.2f} -> {xy:.4f} (B_y, faces) and {xw:.4f} (B_w, cells), exact {Xe:.4f}, "
          f"toward the cold side; div B {div:.1e}, energy budget {budget:.1e}")
    print(f"  Ettingshausen: periodic box, total energy {dE:.1e}, pressure redistributed by up to {dp:.2e}  "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def twotemp_test():
    """Two-temperature plasma (Z = 1). (1) Uniform plasma at theta_e = 1.2, theta_i = 0.8: theta_e - theta_i
    relaxes as d(dT)/dt = -2 nu dT, nu = c_eq rho/theta_e^1.5 (theta_e = 1 + dT/2), checked against an RK4
    solution of that ODE; total energy unchanged. (2) Sod shock tube, gamma = 5/3, T_e = T_i initially: behind
    the shock the electron entropy equals the upstream one (electrons only compressed adiabatically) and the
    ions take the shock heating (T_i > T_e). (3) Bremsstrahlung in two-temperature mode cools the electrons
    only: theta_i unchanged to round-off."""
    from aegis import twotemp as tt
    print("\n--- Two-temperature plasma ---"); sys.stdout.flush()
    g = 5.0 / 3.0

    def box(nx, ny, **kw):
        cfg = Config(nx=nx, ny=ny, x_max=1.0, y_max=ny / nx, t_end=1.0, cfl=0.3, mach=1.0, gamma=g, B_transverse=0.0,
                     bc_y_type="periodic", two_temp=True, diag_interval=10**9, snapshot_times=[], **kw)
        with contextlib.redirect_stdout(io.StringIO()):
            return MHDSolver(cfg)
    # (1) equilibration
    c_eq, T_end = 0.5, 2.0
    s = box(8, 8, bc_x_type="periodic", te_eq_coeff=c_eq)
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
    W[iRHO], W[iPR] = 1.0, 1.0
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    tt.set_pe(s.U, xp.full_like(s.U[0], 0.6), g)             # theta_e = 1.2, theta_i = 0.8
    s._te_owner = s.U
    E0 = _energy(s)
    _march(s, T_end)
    te, ti = (float(xp.mean(a[s.ng:-s.ng, s.ng:-s.ng])) for a in tt.temperatures(s.U, g, 1.0))
    d, n = 0.4, 20000
    f = lambda d: -2 * c_eq / (1 + d / 2)**1.5 * d
    hstep = T_end / n
    for _ in range(n):
        k1 = f(d); k2 = f(d + 0.5 * hstep * k1); k3 = f(d + 0.5 * hstep * k2); k4 = f(d + hstep * k3)
        d += hstep * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    eq_diff = te - ti
    eq_err = abs(eq_diff / d - 1)
    eq_E = abs(_energy(s) - E0) / E0
    # (2) Sod with two temperatures
    s = box(400, 4, bc_x_type="extrapolation")
    x = s.x[:, None] * xp.ones((1, s.ny_tot))
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
    W[iRHO] = xp.where(x < 0.5, 1.0, 0.125)
    W[iPR] = xp.where(x < 0.5, 1.0, 0.1)
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    _march(s, 0.2)
    ng = s.ng
    rho = to_numpy(s.U[0, ng:-ng, ng])
    se = to_numpy(s.U[10, ng:-ng, ng]) / rho
    te, ti = (to_numpy(a[ng:-ng, ng]) for a in tt.temperatures(s.U, g, 1.0))
    xc = to_numpy(s.x[ng:-ng])
    seR = 0.05 / 0.125**g                                    # p_e = p/2 upstream (right)
    shocked = (xc > 0.73) & (xc < 0.83)                     # between the contact (~0.69) and the shock (~0.85)
    se_err = float(np.max(np.abs(se[shocked] / seR - 1)))
    hot_ions = float(np.min(ti[shocked] / te[shocked]))
    # (3) electron-only radiation
    s = box(8, 8, bc_x_type="periodic", rad_coeff=0.05, rad_T_min=0.1)
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
    W[iRHO], W[iPR] = 1.0, 1.0
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    ti0 = float(xp.mean(s.temperatures()[1]))
    _march(s, 1.0)
    te1, ti1 = (float(xp.mean(a[s.ng:-s.ng, s.ng:-s.ng])) for a in s.temperatures())
    ok = eq_err < 0.01 and eq_E < 1e-13 and se_err < 0.01 and hot_ions > 1.0 and abs(ti1 - ti0) < 1e-12 and te1 < ti1
    print(f"  equilibration: theta_e - theta_i = {eq_diff:.5f} vs ODE {d:.5f} "
          f"({100 * eq_err:.2f}%), energy {eq_E:.1e}")
    print(f"  Sod: electron entropy behind the shock = upstream to {100 * se_err:.2f}%, T_i/T_e >= {hot_ions:.2f} there")
    print(f"  radiation cools electrons only: theta_i {ti0:.6f} -> {ti1:.6f}, theta_e -> {te1:.4f}  "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def viscosity_test(tol=0.05):
    """Braginskii parallel viscosity: a sound wave along x (k = 2 pi, rho = p = 1, gamma = 5/3) damps at the extra
    rate mu_eff k^2/(2 rho) over the inviscid scheme's own damping (a run with eta = 0 subtracted), with
    mu_eff = 4/3 eta for B along k, 1/3 eta for B across k (Pi = -3 eta (bb - I/3)(bb:grad v - div v/3)),
    and 4/3 eta for B = 0 (isotropic Navier-Stokes). Total energy conserved in the periodic box."""
    print("\n--- Braginskii viscosity: sound-wave damping ---"); sys.stdout.flush()
    nx, k, A, eta, T = 128, 2 * np.pi, 1e-4, 0.002, 2.0
    c = np.sqrt(5.0 / 3.0)

    def run(B, visc):
        cfg = Config(nx=nx, ny=4, x_max=1.0, y_max=4.0 / nx, t_end=T, cfl=0.3, mach=1.0, gamma=5 / 3,
                     B_transverse=1e-8, bc_x_type="periodic", bc_y_type="periodic", visc_coeff=visc,
                     diag_interval=10**9, snapshot_times=[])
        with contextlib.redirect_stdout(io.StringIO()):
            s = MHDSolver(cfg)
        x = s.x[:, None] * xp.ones((1, s.ny_tot))
        W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
        W[iRHO] = 1.0 + A * xp.sin(k * x)
        W[iVX] = c * A * xp.sin(k * x)
        W[iPR] = 1.0 + c**2 * A * xp.sin(k * x)
        W[iBX], W[iBY] = B
        s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
        E0 = _energy(s)
        _march(s, T)
        xc = to_numpy(s.x[s.ng:-s.ng])
        dr = to_numpy(s.U[0, s.ng:-s.ng, s.ng]) - 1.0
        amp = 2 * np.hypot(np.mean(dr * np.sin(k * xc)), np.mean(dr * np.cos(k * xc)))
        return -np.log(amp / A) / T, abs(_energy(s) - E0) / E0
    g0, _ = run((1e-6, 0.0), 0.0)
    out, ok = [], True
    for name, B, mu in (("B along k", (1e-6, 0.0), 4 / 3), ("B across k", (0.0, 1e-6), 1 / 3), ("B = 0", (0.0, 0.0), 4 / 3)):
        gv, dE = run(B, eta)
        exact = mu * eta * k**2 / 2
        err = abs((gv - g0) / exact - 1)
        good = err < (tol if mu > 1 else 2 * tol) and dE < 1e-12
        ok = ok and good
        out.append(f"  {name}: extra damping {gv - g0:.5f} vs {exact:.5f} ({100 * err:.1f}%), energy {dE:.1e}  "
                   f"[{'PASS' if good else 'FAIL'}]")
    print(f"  scheme damping without viscosity: {g0:.5f}")
    print("\n".join(out))
    return ok


def strength_test():
    """Liner strength on stiffened-gas aluminium (g/cm^3, km/s, mm, us, GPa; G = 26, Y = 0.29, spall 1.2).
    (1) A small velocity pulse travels at the longitudinal speed c_L = sqrt(c0^2 + 4G/(3 rho)) = 6.42 km/s,
    not the bulk c0 = 5.33. (2) A 0.5 km/s impact makes an elastic precursor at the Hugoniot elastic limit
    sigma_HEL = (K + 4G/3) Y/(2G) moving at c_L; the plastic wave reaches the yield surface, sqrt(3 J2) = Y
    (S_xx = -2Y/3 in uniaxial strain) and the target stays there behind it. The flyer is long enough that the
    weak unloading wave its shock reflects off the zero-gradient left boundary stays out of the target. (3) A bar pulled apart at +-0.2 km/s (tension ~3 GPa): spall damage forms at the centre
    only and the intact metal never carries more tension than the spall strength."""
    from aegis.eos import stiffened_from_hugoniot
    from aegis import strength as st
    print("\n--- Liner strength: elastic waves, Hugoniot elastic limit, spall ---"); sys.stdout.flush()
    gA, piA = stiffened_from_hugoniot(AL["rho0"], AL["c0"], AL["s"], 1.0)
    G, Y, sp = 26.0, 0.29, 1.2
    rho0, c0 = AL["rho0"], AL["c0"]
    cL = np.sqrt(c0**2 + 4 * G / (3 * rho0))

    def bar(nx, L, u_of, t_end, spall=0.0, yield_=Y):
        cfg = Config(nx=nx, ny=4, x_max=L, y_max=4 * L / nx, t_end=t_end, cfl=0.3, mach=1.0, B_transverse=0.0,
                     bc_x_type="extrapolation", bc_y_type="periodic", eos_liner="stiffened", gamma_liner=gA,
                     p_inf_liner=piA, material_tracer="volume_fraction", strength=True, shear_modulus=G,
                     yield_strength=yield_, spall_strength=spall, diag_interval=10**9, snapshot_times=[])
        with contextlib.redirect_stdout(io.StringIO()):
            s = MHDSolver(cfg)
        x = s.x[:, None] * xp.ones((1, s.ny_tot))
        W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
        W[iRHO], W[iPR], W[iCLR] = rho0, 1e-4, 1.0
        W[iVX] = u_of(x)
        s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
        E0 = _energy(s)
        _march(s, t_end)
        ng = s.ng
        P = cons_to_prim(s.U, s.eos)
        prof = dict(x=to_numpy(s.x[ng:-ng]), u=to_numpy(P[iVX, ng:-ng, ng]), p=to_numpy(P[iPR, ng:-ng, ng]),
                    Sxx=to_numpy(s.S[st.SXX, ng:-ng, ng]), dmg=to_numpy(s.S[st.DMG, ng:-ng, ng]))
        prof["budget"] = abs(_energy(s) - E0 - s._cumulative_boundary_energy) / abs(E0)
        return s, prof
    # (1) elastic pulse
    t1 = 0.04
    s, P = bar(800, 1.0, lambda x: 1e-3 * xp.exp(-((x - 0.5) / 0.02)**2), t1, yield_=1e3)
    right = P["x"] > 0.5
    i = int(np.argmax(np.where(right, P["u"], -1)))
    a_, b_, c_ = P["u"][i - 1], P["u"][i], P["u"][i + 1]
    xpk = P["x"][i] + 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_) * (P["x"][1] - P["x"][0])
    speed = (xpk - 0.5) / t1
    e1 = abs(speed / cL - 1)
    # (2) Hugoniot elastic limit
    K = rho0 * c0**2
    hel = (K + 4 * G / 3) * Y / (2 * G)
    t2, x0 = 0.12, 0.4
    s, P = bar(1120, 1.4, lambda x: xp.where(x < x0, 0.5, 0.0), t2)
    sig = P["p"] - P["Sxx"]                                  # compressive normal stress
    xs = P["x"]
    tgt = xs > x0 + 0.02
    front = xs[np.where(tgt & (sig > 0.5 * hel))[0][-1]]
    plastic = xs[np.where(tgt & (sig > 2.0 * hel))[0][-1]]
    h = xs[1] - xs[0]
    pre = (xs > plastic + 6 * h) & (xs < front - 6 * h)
    hel_num = float(np.mean(sig[pre]))
    behind = (xs > x0 + 0.1) & (xs < plastic - 6 * h)
    sxx_pl = float(np.mean(P["Sxx"][behind]))
    near = (xs > plastic - 6 * h) & (xs < plastic + 2 * h)
    seq_peak = float(np.max(1.5 * np.abs(P["Sxx"][near])))      # sqrt(3 J2) = 1.5 |S_xx| in uniaxial strain
    e2y = abs(seq_peak / Y - 1)
    e2 = abs(hel_num / hel - 1)
    e2s = abs(sxx_pl / (-2 * Y / 3) - 1)
    e2f = abs((front - x0) / t2 / cL - 1)
    # (3) spall
    s, P = bar(400, 1.0, lambda x: xp.where(x < 0.5, -0.2, 0.2), 0.04, spall=sp)
    dm = P["dmg"] > 0.5
    intact_pmin = float(np.min(P["p"][~dm]))
    central = bool(dm.any()) and bool(np.all(np.abs(P["x"][dm] - 0.5) < 0.05))
    ok = (e1 < 0.01 and e2 < 0.05 and e2y < 0.01 and e2s < 0.01 and e2f < 0.03 and central and intact_pmin > -1.1 * sp
          and P["budget"] < 1e-12)
    print(f"  elastic pulse: {speed:.3f} km/s vs c_L {cL:.3f} ({100 * e1:.2f}%; bulk c0 {c0:.3f})")
    print(f"  impact at 0.5 km/s: precursor {hel_num:.3f} GPa vs HEL {hel:.3f} ({100 * e2:.1f}%), front at "
          f"{(front - x0) / t2:.2f} km/s; plastic front reaches sqrt(3 J2) = {seq_peak:.4f} (Y = {Y}, {100 * e2y:.2f}%), "
          f"plateau S_xx {sxx_pl:.4f} vs -2Y/3 {-2 * Y / 3:.4f} ({100 * e2s:.2f}%)")
    print(f"  bar pulled apart: {int(dm.sum())} spalled cells, all within 0.05 mm of the centre: {central}; "
          f"intact metal min p {intact_pmin:.3f} GPa (spall {-sp}); energy budget {P['budget']:.1e}  "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def parity3d_test(tol=1e-12):
    """3D slabs one cell pattern deep in z must reproduce the 2D solver for the Phase 2-4 physics:
    (1) stiffened Al on stiffened Cu at 10 km/s (two materials, volume-fraction tracer);
    (2) bremsstrahlung decay of a uniform hot plasma; (3) magnetized conduction from a hot spot in an
    oblique field, with the hydrodynamics running. In (3) the slab has dz = 10 dx: the RKL2 stage count
    comes from the explicit limit, which counts 1/dz^2, and equal stage counts are needed for round-off
    agreement (with dz = dx the 3D run takes more stages and differs by the RKL2 time error, 1e-4 of the
    hot-spot amplitude). The z faces still run and must contribute exactly zero."""
    from aegis.solver3d import MHDSolver3D
    from aegis.eos import stiffened_from_hugoniot
    from aegis.units import bremsstrahlung_kwargs, AMU, E_CHARGE
    print("\n--- 3D parity: EOS, radiation, conduction ---"); sys.stdout.flush()
    res = []

    def build(cls, nx, ny, x0, left, right, bc, nz=None, extent=(1.0, None), **kw):
        Lx, Ly = extent
        Ly = Ly or ny * Lx / nx
        c = dict(nx=nx, ny=ny, x_max=Lx, y_max=Ly, t_end=1.0, cfl=0.3, mach=1.0, B_transverse=0.0,
                 powell_source=False, bc_x_type=bc, bc_y_type="periodic", diag_interval=10**9, snapshot_times=[], **kw)
        if nz:
            c.update(nz=nz, z_max=nz * Lx / nx)
        with contextlib.redirect_stdout(io.StringIO()):
            s = cls(Config(**c))
        W = xp.zeros((NVAR,) + s.U.shape[1:])
        x = s.x.reshape((-1,) + (1,) * (W.ndim - 2))
        inside = x < x0
        for k, i in ((0, iRHO), (1, iVX), (2, iPR), (3, iCLR)):
            W[i] = xp.where(inside, left[k], right[k])
        s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
        s.apply_bc(s.U)
        return s

    def compare(s2, s3, keys=(iRHO, iPR)):
        ng = s2.ng
        W2 = cons_to_prim(s2.U, s2.eos)[:, ng:-ng, ng:-ng]
        W3 = cons_to_prim(s3.U, s3.eos)[:, ng:-ng, ng:-ng, ng:-ng]
        d = max(float(xp.max(xp.abs(W3[i] - W2[i][:, :, None]))) / float(xp.max(xp.abs(W2[i]))) for i in keys)
        zv = max(float(xp.max(xp.abs(W3[i] - W3[i][:, :, :1]))) for i in keys)
        return d, zv

    # (1) two-material impact
    gA, piA = stiffened_from_hugoniot(AL["rho0"], AL["c0"], AL["s"], 5.0)
    gC, piC = stiffened_from_hugoniot(CU["rho0"], CU["c0"], CU["s"], 3.0)
    mm = dict(eos_fuel="stiffened", gamma=gC, p_inf=piC, eos_liner="stiffened", gamma_liner=gA, p_inf_liner=piA)
    L, R = (AL["rho0"], 10.0, 1e-4, 1.0), (CU["rho0"], 0.0, 1e-4, 0.0)
    s2 = build(MHDSolver, 200, 4, 0.5, L, R, "extrapolation", **mm)
    s3 = build(MHDSolver3D, 200, 4, 0.5, L, R, "extrapolation", nz=4, **mm)
    for s in (s2, s3):
        _march(s, 0.03)
    res.append(("stiffened Al -> Cu impact", *compare(s2, s3, (iRHO, iPR, iCLR)), s2.step, s3.step))

    # (2) radiation decay (same set-up as the radiation test)
    mp, n0, T0 = 1.67262192e-27, 1e26, 500.0
    v0 = np.sqrt(2 * E_CHARGE * T0 / mp)
    rad = bremsstrahlung_kwargs(n0 * mp, v0, 1.0, 1.0, mp / AMU, 10.0)
    hot = (1.0, 0.0, 1.0, 0.0)
    s2 = build(MHDSolver, 8, 8, 2.0, hot, hot, "periodic", gamma=5.0 / 3.0, **rad)
    s3 = build(MHDSolver3D, 8, 8, 2.0, hot, hot, "periodic", nz=4, gamma=5.0 / 3.0, **rad)
    for s in (s2, s3):
        _march(s, 5e-6 * v0)
    d, zv = compare(s2, s3)
    # the 3D integral is over the slab depth; the 2D one is per unit depth
    erad = abs(s3.cumulative_radiated_energy / (s3.cfg.dz * s3.cfg.nz) - s2.cumulative_radiated_energy) \
        / s2.cumulative_radiated_energy
    res.append(("bremsstrahlung decay", max(d, erad), zv, s2.step, s3.step))

    # (3) conduction with hydrodynamics, oblique field
    s2 = _hot_spot(MHDSolver, 32, np.pi / 6, 0.15, 30.0)
    s3 = _hot_spot(MHDSolver3D, 32, np.pi / 6, 0.15, 30.0, nz=4, dz_factor=10.0)
    for s in (s2, s3):
        _march(s, 0.2)
    res.append(("conduction, field at 30 deg", *compare(s2, s3), s2.step, s3.step))

    ok = True
    for name, d, zv, n2, n3 in res:
        good = d < tol and zv == 0.0 and n2 == n3
        ok = ok and good
        print(f"  {name}: steps 2D/3D {n2}/{n3}, max rel diff {d:.1e}, z-variation {zv:.1e}  "
              f"[{'PASS' if good else 'FAIL'}]")
    return ok


def cylindrical_test():
    """Axisymmetric (r, z) geometry. (1) A static uniform magnetized state (axial field) must stay static to
    round-off: the r-weighted flux divergence and the p_T/r source balance exactly. (2) Cylindrical Noh
    problem, gamma = 5/3: gas at v_r = -1 stagnates on the axis behind an outgoing shock at r = t/3 with
    density ((g+1)/(g-1))^2 = 16, ahead of it rho = 1 + t/r. Checked at t = 0.6 away from the axis
    (wall-heating layer) and from the outer boundary; plus the energy budget and div B with a weak axial
    field."""
    print("\n--- Cylindrical (r, z) geometry ---"); sys.stdout.flush()
    base = dict(x_min=0.0, x_max=1.0, y_min=0.0, cfl=0.3, gamma=5.0 / 3.0, mach=1.0, B_transverse=1e-3,
                geometry="cylindrical", bc_x_type="extrapolation", bc_y_type="periodic", diag_interval=10**9,
                snapshot_times=[])
    # (1) static uniform state
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(Config(nx=64, ny=8, y_max=8 / 64, t_end=0.5, **base))
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
    W[iRHO], W[iPR], W[iBY] = 1.0, 1.0, 0.7
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    _march(s, 0.5)
    Wf = cons_to_prim(s.U, s.eos)[:, s.ng:-s.ng, s.ng:-s.ng]
    vmax = float(xp.max(xp.abs(Wf[iVX]))) + float(xp.max(xp.abs(Wf[iVY])))
    dp = float(xp.max(xp.abs(Wf[iPR] - 1.0)))
    # (2) Noh
    nx, t_end = 200, 0.6
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(Config(nx=nx, ny=4, y_max=4 / nx, t_end=t_end, **base))
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
    W[iRHO], W[iPR], W[iVX], W[iBY] = 1.0, 1e-6, -1.0, 1e-3
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    E0 = _energy(s)
    _march(s, t_end)
    ng = s.ng
    P = cons_to_prim(s.U, s.eos)
    r, rho = to_numpy(s.x[ng:-ng]), to_numpy(P[iRHO, ng:-ng, ng])
    plateau = float(np.mean(rho[(r > 0.06) & (r < 0.17)]))
    rs = r[np.where(rho > 0.5 * (16 + 1 + t_end / 0.2))[0][-1]]            # last cell above mid-jump
    pre = float(np.mean(rho[(r > 0.28) & (r < 0.32)]))
    pre_exact = float(np.mean(1 + t_end / r[(r > 0.28) & (r < 0.32)]))
    budget = abs(_energy(s) - E0 - s._cumulative_boundary_energy) / abs(E0)
    div = s.div_b_max()
    ok = (vmax < 1e-12 and dp < 1e-12 and abs(plateau / 16 - 1) < 0.05 and abs(rs - t_end / 3) < 2.5 / nx
          and abs(pre / pre_exact - 1) < 0.03 and budget < 1e-12 and div < 1e-12)
    print(f"  static magnetized state, 64x8 to t = 0.5: max |v| {vmax:.1e}, max |dp| {dp:.1e}")
    print(f"  Noh, {nx} cells, t = {t_end}: post-shock rho {plateau:.3f} (exact 16), shock at r = {rs:.4f} "
          f"({t_end / 3:.4f}), pre-shock rho {pre:.3f} ({pre_exact:.3f}); energy budget {budget:.1e}, div B {div:.1e}  "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def ct_test():
    """Constrained transport. (1) Field loop advection (Gardiner & Stone 2005): a weak loop A_z = A0 (R - r)
    carried diagonally through a periodic box for one period; div B on the faces must stay at round-off,
    and the loop keeps a reported fraction of its magnetic energy (GLM shown for comparison). (2) A fully 3D
    field from a vector potential in a uniform 3D flow, periodic: div B at round-off after 40 steps."""
    from aegis.solver3d import MHDSolver3D
    print("\n--- Constrained transport: div B ---"); sys.stdout.flush()
    A0, R = 1e-3, 0.3
    loop = lambda X, Y: A0 * np.maximum(R - np.sqrt(X**2 + Y**2), 0.0)
    res = {}
    for mode in ("ct", "glm"):
        cfg = Config(nx=128, ny=64, x_min=-1, x_max=1, y_min=-0.5, y_max=0.5, t_end=2.0, cfl=0.4, gamma=5 / 3,
                     mach=1.0, B_transverse=1e-3, bc_x_type="periodic", bc_y_type="periodic", div_b=mode,
                     diag_interval=10**9, snapshot_times=[])
        with contextlib.redirect_stdout(io.StringIO()):
            s = MHDSolver(cfg)
        W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
        W[iRHO], W[iPR], W[iVX], W[iVY] = 1.0, 1.0, 2.0, 1.0
        s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
        s.set_vector_potential(loop)
        Em0 = float(xp.sum(s.U[iBX]**2 + s.U[iBY]**2))
        div0 = s.div_b_max()
        _march(s, cfg.t_end)
        res[mode] = (float(xp.sum(s.U[iBX]**2 + s.U[iBY]**2)) / Em0, div0, s.div_b_max())
    # 3D: ABC-like potential, uniform oblique flow
    k = 2 * np.pi
    cfg = Config(nx=16, ny=16, nz=16, x_max=1.0, y_max=1.0, z_max=1.0, t_end=1.0, cfl=0.3, mach=1.0,
                 B_transverse=0.1, bc_x_type="periodic", diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s3 = MHDSolver3D(cfg)
    W = xp.zeros((NVAR,) + s3.U.shape[1:])
    W[iRHO], W[iPR], W[iVX], W[iVY], W[iVZ] = 1.0, 1.0, 1.0, 0.5, 0.25
    s3.U = prim_to_cons(W, s3.eos); s3.t = 0.0; s3.step = 0
    s3.set_vector_potential(lambda x, y, z: 0.1 * np.sin(k * z) / k, lambda x, y, z: 0.1 * np.sin(k * x) / k,
                            lambda x, y, z: 0.1 * np.sin(k * y) / k)
    d3_0 = s3.div_b_max()
    for _ in range(40):
        s3.dt = s3.compute_dt(cons_to_prim(s3.U, s3.eos)); s3.step_ssprk3(); s3.t += s3.dt
    d3 = s3.div_b_max()
    ct_ok = res["ct"][1] < 1e-12 and res["ct"][2] < 1e-12 and res["ct"][0] > 0.7 and d3_0 < 1e-12 and d3 < 1e-12
    print(f"  field loop, 128x64, one period: CT div B {res['ct'][1]:.1e} -> {res['ct'][2]:.1e}, magnetic energy "
          f"kept {res['ct'][0]:.3f}; GLM div B {res['glm'][2]:.1e}, energy kept {res['glm'][0]:.3f}")
    print(f"  3D field from a vector potential in a 3D flow, 16^3: div B {d3_0:.1e} -> {d3:.1e} after 40 steps  "
          f"[{'PASS' if ct_ok else 'FAIL'}]")
    return ct_ok


def galilean_test(U=6.49):
    """Interface frame: the interface must stay put, energy must close, and the interface growth must match
    the lab frame (within 10%). Mixing-layer enstrophy is reported, not asserted: the scheme is not exactly
    Galilean invariant, and advecting the interface across the grid in the lab frame adds numerical
    diffusion (measured 15-36% at h = 0.042-0.083)."""
    print("\n--- 3D: Galilean frame ---"); sys.stdout.flush()
    from aegis.solver3d import MHDSolver3D
    h = 10.0 / 120
    base = dict(nx=120, ny=24, nz=4, x_max=10.0, z_max=4 * h, t_end=0.4, mach=10.0, interface_x=4.0,
                B_field_type="striped", B_transverse=6.0, eta_resistivity=0.02, B_modulation_mode=1,
                bc_x_type="inflow_outflow", diag_interval=5)
    res = {}
    for name, fv in (("lab", 0.0), ("co-moving", U)):
        s = MHDSolver3D(Config(**base, frame_velocity=fv))
        s.initialize(); res[name] = s.run(verbose=False); res[name]["err"] = s.energy_error()
    lab, cm = res["lab"], res["co-moving"]
    d_rough = abs(cm["rough"][-1] - lab["rough"][-1]) / lab["rough"][-1]
    d_emix = abs(cm["emix"][-1] - lab["emix"][-1]) / lab["emix"][-1]
    drift = cm["x_int"][-1] - cm["x_int"][len(cm["t"]) // 4]
    err = max(lab["err"], cm["err"])
    ok = d_rough < 0.10 and abs(drift) < 0.1 and err < 1e-10
    print(f"  roughness lab/co-moving {lab['rough'][-1]:.4f}/{cm['rough'][-1]:.4f} ({100*d_rough:.1f}%); "
          f"interface drift {drift:+.3f}; energy error {err:.1e}; enstrophy differs by {100*d_emix:.0f}% "
          f"(numerical diffusion, reported only)  [{'PASS' if ok else 'FAIL'}]")
    return ok


def rmi_mhd_smoke_test():
    """High-Mach RMI benchmark at low resolution: no crash, positive pressure/density, energy closes,
    all output files written, and the shock-parallel field reduces the amplitude."""
    print("\n--- RMI-MHD benchmark smoke test ---"); sys.stdout.flush()
    import json
    import tempfile
    from aegis.problems.rmi_mhd import RMIParams, run
    out = tempfile.mkdtemp(prefix="rmi_smoke_")
    res = {}
    for name, field, B0 in (("hydro", "none", 0.0), ("mhd_parallel", "parallel", 2.5)):
        p = RMIParams(name=name, field=field, B0=B0, nx=256, ny=32, t_end=0.5, n_frames=3)
        with contextlib.redirect_stdout(io.StringIO()):
            res[name] = run(p, os.path.join(out, name))["result"]
    files = all(os.path.exists(os.path.join(out, n, f)) for n in res
                for f in ("diagnostics.csv", "summary.json", "frames.npz", "rmi.pvd", "vtk/frame_0002.vti"))
    ok = (files and all(r["p_min"] > 0 and r["rho_min"] > 0 and r["energy_error_max"] < 1e-10 for r in res.values())
          and res["mhd_parallel"]["a_final"] < res["hydro"]["a_final"])
    print(f"  a(t_end): hydro {res['hydro']['a_final']:.4f}, B_x=2.5 {res['mhd_parallel']['a_final']:.4f}; "
          f"min p {min(r['p_min'] for r in res.values()):.3g}; energy error "
          f"{max(r['energy_error_max'] for r in res.values()):.1e}; files {'ok' if files else 'MISSING'}  "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def fcg_smoke_test():
    """Flux compression generator (examples/fcg_flux_compression.py) on a small r-z grid: Al liner (R0 = 10 mm,
    1.5 mm thick) at 3 km/s on a 5 T seed with Mie-Gruneisen, strength/spall, CT and resistivity together.
    Energy closes, div B stays at round-off, the axis field passes 20 B0 with >80% of the flux left in the
    cavity gas, the liner does not spall, and the output files are written."""
    print("\n--- FCG flux compression smoke test ---"); sys.stdout.flush()
    import argparse
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples"))
    import fcg_flux_compression as fcg
    out = tempfile.mkdtemp(prefix="fcg_smoke_")
    a = argparse.Namespace(liner_material="al", b0_tesla=5.0, v0_kms=3.0, enable_strength=True, eta_model="spitzer",
                           sigma_gas=None, sigma_min=5e6, r0=10.0, dr=1.5, r_max=14.0, lz=1.0, mode=1, amp=0.0,
                           rho_gas=0.1, p_gas_bar=0.1, nr=64, nz=4, t_end=8.0, frame_dt=10.0, diag_every=5)
    with contextlib.redirect_stdout(io.StringIO()):
        r = fcg.run(a, out)
    files = all(os.path.exists(os.path.join(out, f)) for f in ("diagnostics.csv", "summary.json", "frames.npz"))
    ok = (files and r["energy_error_max"] < 1e-12 and r["div_B_max"] < 1e-12 and r["B_axis_peak_T"] > 20 * a.b0_tesla
          and r["flux_retention_at_turnaround"] > 0.8 and r["spall_cells_final"] == 0)
    print(f"  {r['steps']} steps: axis B peak {r['B_axis_peak_T']:.0f} T from {a.b0_tesla:g} T, flux kept "
          f"{r['flux_retention_at_turnaround']:.3f}, P_B/Y peak {r['PB_over_Y_peak']:.0f}, spalled cells "
          f"{r['spall_cells_final']}; div B {r['div_B_max']:.1e}, energy error {r['energy_error_max']:.1e}; files "
          f"{'ok' if files else 'MISSING'}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def solid_nernst_3d_test(tol=1e-12):
    """3D liner strength and Nernst transport (aegis/solver3d.py). Parity: z-invariant 3D slabs reproduce the
    2D solver to round-off for (1) stiffened Al with strength, a velocity blob that drives the metal plastic, and
    (2) Nernst + Ettingshausen with a temperature varying in x and y and field on every face (1e-10: the 2D
    solver moves the out-of-plane field as a cell-centred upwind divergence, the 3D one through CT edges, the
    same sums in a different order). With spall on, the same blob spalls its core: a cell at the threshold
    may flip a step apart on round-off, so the damaged regions are compared, not the fields. Genuinely 3D: (3) an elastic shear wave along z (S_xz, absent in 2D) oscillates at
    w = k sqrt(G/rho) (Jaumann/Hooke in the third direction); (4) a field bump in a temperature gradient along z
    travels down it on the exact Nernst path, div B at round-off."""
    from aegis.solver3d import MHDSolver3D
    from aegis.eos import stiffened_from_hugoniot
    from aegis import strength as st
    print("\n--- 3D strength and Nernst ---"); sys.stdout.flush()
    gA, piA = stiffened_from_hugoniot(AL["rho0"], AL["c0"], AL["s"], 1.0)
    G, rho0 = 26.0, AL["rho0"]
    metal = dict(eos_liner="stiffened", gamma_liner=gA, p_inf_liner=piA, material_tracer="volume_fraction",
                 strength=True, shear_modulus=G)

    def make(cls, n, L, nz=None, Lz=None, **kw):
        c = dict(nx=n[0], ny=n[1], x_max=L[0], y_max=L[1], t_end=1.0, cfl=0.3, mach=1.0, B_transverse=0.0,
                 powell_source=False, bc_x_type="periodic", bc_y_type="periodic", diag_interval=10**9,
                 snapshot_times=[])
        c.update(kw)
        if nz:
            c.update(nz=nz, z_max=Lz)
        with contextlib.redirect_stdout(io.StringIO()):
            return cls(Config(**c))

    def grids(s):
        sh = s.U.shape[1:]
        X = s.x.reshape((-1,) + (1,) * (len(sh) - 1))
        Y = s.y.reshape((1, -1) + (1,) * (len(sh) - 2))
        return X, Y, sh

    def compare(s2, s3, keys, extra=()):
        ng = s2.ng
        W2 = cons_to_prim(s2.U, s2.eos)[:, ng:-ng, ng:-ng]
        W3 = cons_to_prim(s3.U, s3.eos)[:, ng:-ng, ng:-ng, ng:-ng]
        pairs = [(W2[i], W3[i]) for i in keys] + [(a[ng:-ng, ng:-ng], b[ng:-ng, ng:-ng, ng:-ng]) for a, b in extra]
        d = max(float(xp.max(xp.abs(b - a[:, :, None]))) / max(float(xp.max(xp.abs(a))), 1e-300) for a, b in pairs)
        zv = max(float(xp.max(xp.abs(b - b[:, :, :1]))) for _, b in pairs)
        return d, zv

    res = []
    # (1) strength: outward velocity blob, plastic flow (and with spall on, a spalled core)
    def blob(spall):
        out = []
        for cls, nz in ((MHDSolver, None), (MHDSolver3D, 4)):
            s = make(cls, (48, 48), (1.0, 1.0), nz, 4 / 48, yield_strength=0.29, spall_strength=spall, **metal)
            X, Y, sh = grids(s)
            r2 = (X - 0.5)**2 + (Y - 0.45)**2
            W = xp.zeros((NVAR,) + sh)
            W[iRHO], W[iPR], W[iCLR] = rho0, 1e-4, 1.0
            W[iVX] = 0.5 * (X - 0.5) / 0.1 * xp.exp(-r2 / 0.1**2) + 0.0 * Y
            W[iVY] = 0.5 * (Y - 0.45) / 0.1 * xp.exp(-r2 / 0.1**2) + 0.0 * X
            s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
            _march(s, 0.02)
            out.append(s)
        return out
    s2, s3 = blob(0.0)
    d, zv = compare(s2, s3, (iRHO, iPR, iVX, iVY), [(s2.S[c], s3.S[c]) for c in (st.SXX, st.SYY, st.SXY)])
    plastic = float(xp.max(st.von_mises(s3.S))) > 0.99 * 0.29
    off = max(float(xp.max(xp.abs(s3.S[c]))) for c in (st.SXZ, st.SYZ))
    res.append(("strength (plastic flow)", d, zv, s2.step, s3.step, plastic and off == 0.0, tol))
    s2, s3 = blob(1.2)
    n2, n3 = int(xp.sum(s2.S[st.DMG] > 0.5)), int(xp.sum(s3.S[st.DMG, :, :, s3.ng:-s3.ng] > 0.5))
    D3 = s3.S[st.DMG, s3.ng:-s3.ng, s3.ng:-s3.ng, s3.ng:-s3.ng]
    dz = float(xp.max(xp.abs(D3 - D3[:, :, :1])))
    spall_ok = n2 > 0 and abs(n3 / 4 - n2) <= 0.02 * n2 and dz == 0.0
    print(f"  spall: damaged cells 2D {n2}, 3D {n3 / 4:g} per z layer, z-invariant {dz == 0.0}  "
          f"[{'PASS' if spall_ok else 'FAIL'}]")

    # (2) Nernst + Ettingshausen
    for_both = []
    for cls, nz in ((MHDSolver, None), (MHDSolver3D, 4)):
        s = make(cls, (64, 16), (1.0, 0.25), nz, 4 / 64, gamma=5 / 3, B_transverse=1e-6, bc_x_type="extrapolation",
                 cond_x0=1.0, nernst_coeff=31.0)
        X, Y, sh = grids(s)
        T = 1.0 + 0.02 * (X - 0.5) + 0.01 * xp.cos(2 * np.pi * Y / 0.25)
        W = xp.zeros((NVAR,) + sh)
        W[iPR], W[iRHO] = 1.0, 1.0 / T
        bump = 1e-2 * xp.exp(-((X - 0.6) / 0.08)**2)
        W[iBX] = 1e-2 * xp.cos(2 * np.pi * Y / 0.25) + 0.0 * X
        W[iBY] = bump + 0.0 * Y
        W[iBZ] = bump * (1 + 0.5 * xp.cos(2 * np.pi * Y / 0.25))
        s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
        _march(s, 0.3)
        for_both.append(s)
    s2, s3 = for_both
    d, zv = compare(s2, s3, (iRHO, iPR, iVX, iVY, iBX, iBY, iBZ))
    res.append(("Nernst + Ettingshausen", d, zv, s2.step, s3.step, s3.div_b_max() < 1e-12, 1e-10))

    ok = spall_ok
    for name, d, zv, n2, n3, extra, tl in res:
        good = d < tl and zv == 0.0 and n2 == n3 and extra
        ok = ok and good
        print(f"  parity, {name}: steps 2D/3D {n2}/{n3}, max rel diff {d:.1e}, z-variation {zv:.1e}  "
              f"[{'PASS' if good else 'FAIL'}]")

    # (3) shear wave along z: v_x = A sin(k z) cos(w t), w = k sqrt(G/rho)
    Lz, A = 1.0, 1e-4
    s = make(MHDSolver3D, (4, 4), (4 / 64, 4 / 64), 64, Lz, yield_strength=1e3, **metal)
    sh = s.U.shape[1:]
    Z = s.z.reshape(1, 1, -1)
    W = xp.zeros((NVAR,) + sh)
    W[iRHO], W[iPR], W[iCLR] = rho0, 1e-4, 1.0
    W[iVX] = A * xp.sin(2 * np.pi * Z / Lz) + 0.0 * W[iRHO]
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    k, ng = 2 * np.pi / Lz, s.ng
    w = k * np.sqrt(G / rho0)
    proj = lambda: float(xp.mean(cons_to_prim(s.U, s.eos)[iVX, ng:-ng, ng:-ng, ng:-ng] * xp.sin(k * Z[..., ng:-ng]))) * 2 / A
    a0, errs = proj(), []
    for frac in (0.125, 0.25, 0.375):
        _march(s, frac * 2 * np.pi / w)
        errs.append(abs(proj() / a0 - np.cos(w * s.t)))
    shear_ok = max(errs) < 0.01
    print(f"  shear wave along z (S_xz): amplitude vs cos(k sqrt(G/rho) t) at 1/8, 1/4, 3/8 period, max error "
          f"{max(errs):.1e} of the amplitude  [{'PASS' if shear_ok else 'FAIL'}]")

    # (4) Nernst along z: isobaric periodic T(z) = 1 + 0.01 sin(2 pi z), field bumps in B_x and B_y travel down
    # the gradient (from z = 0.5, where it is steepest, toward the cold side)
    nz, cN, x0, t_end, Z0, wb = 200, 31.0, 1.0, 0.1, 0.5, 0.05
    s = make(MHDSolver3D, (4, 4), (4 / nz, 4 / nz), nz, 1.0, gamma=5 / 3, B_transverse=1e-6, cond_x0=x0,
             nernst_coeff=cN)
    sh = s.U.shape[1:]
    Z = s.z.reshape(1, 1, -1)
    T = 1.0 + 0.01 * xp.sin(2 * np.pi * Z)
    W = xp.zeros((NVAR,) + sh)
    W[iPR], W[iRHO] = 1.0, 1.0 / T + 0.0 * W[iPR]
    bump = 1e-4 * xp.exp(-((Z - Z0) / wb)**2) + 0.0 * W[iPR]
    W[iBX], W[iBY] = bump, bump
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    _march(s, t_end)
    zc = to_numpy(s.z[ng:-ng])
    Wf = cons_to_prim(s.U, s.eos)
    cen = lambda b: float(np.sum(zc * b) / np.sum(b))
    zx, zy = cen(to_numpy(Wf[iBX, ng, ng, ng:-ng])), cen(to_numpy(Wf[iBY, ng, ng, ng:-ng]))
    Ze, m = Z0, 4000
    for _ in range(m):                               # dZ/dt = -c_N x0 theta^2.5 (beta_wedge/x)(0) dtheta/dz
        th = 1.0 + 0.01 * np.sin(2 * np.pi * Ze)
        Ze += -cN * x0 * th**2.5 * (3.053 / 3.7703) * 0.02 * np.pi * np.cos(2 * np.pi * Ze) * (t_end / m)
    nz_ok = abs(zx - Ze) < 1.5 / nz and abs(zy - Ze) < 1.5 / nz and s.div_b_max() < 1e-12
    print(f"  Nernst along z: bumps moved {Z0:.2f} -> {zx:.4f} (B_x), {zy:.4f} (B_y), exact {Ze:.4f}; div B "
          f"{s.div_b_max():.1e}  [{'PASS' if nz_ok else 'FAIL'}]")
    return ok and shear_ok and nz_ok


def mpi_test():
    """Slab domain decomposition (aegis/parallel.py): scripts/mpi_check.py under mpiexec with 2 and 3 ranks
    (uneven slabs) must match the single-domain solver to round-off for 2D Orszag-Tang with resistivity, 2D
    aluminium with strength and spall, and a 3D Orszag-Tang-like box: same steps, same total energy, div B at
    round-off. Skipped (reported, not failed) when mpi4py or mpiexec is missing."""
    import shutil
    import subprocess
    print("\n--- MPI domain decomposition ---"); sys.stdout.flush()
    here = os.path.dirname(os.path.abspath(__file__))
    exe = shutil.which("mpiexec") or next((p for p in (os.path.join(sys.prefix, "Library", "bin", "mpiexec.exe"),
                                                       os.path.join(sys.prefix, "bin", "mpiexec")) if os.path.exists(p)), None)
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        exe = None
    if exe is None:
        print("  SKIPPED: mpi4py or mpiexec not found (pip install mpi4py, plus an MPI runtime)")
        return True
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(exe) + os.pathsep + env.get("PATH", "")
    ok = True
    for n in (2, 3):
        r = subprocess.run([exe, "-n", str(n), sys.executable, os.path.join(here, "scripts", "mpi_check.py"), "--quick"],
                           capture_output=True, text=True, env=env, timeout=3600)
        lines = [l for l in r.stdout.splitlines() if l.startswith("  ") or l.startswith("MPI_CHECK")]
        print("\n".join(lines) if lines else (r.stdout + r.stderr)[-2000:])
        ok = ok and r.returncode == 0 and any(l == "MPI_CHECK PASS" for l in lines)
    return ok


def fld_test():
    """Grey flux-limited diffusion radiation (aegis/fld.py). (1) Uniform gas (theta = 1) and no radiation, Planck
    coupling: theta and E_r follow the exact ODE pair to 1% (backward Euler, rate x dt = 0.01) and settle on the
    equilibrium a theta^4 = E_r with gas + radiation energy conserved to round-off. (2) Optically thick (chi
    R << 1, so D = c/(3 chi)): a Gaussian E_r pulse spreads as <x^2> + 2 D t per direction, in 2D, in r-z
    (<r^2> + 4 D t) and in 3D (<r^2> + 6 D t), each to 1%. (3) Optically thin, a hot wall: the Levermore-
    Pomraning limiter keeps the radiation front at or behind c t (without it diffusion would fill the slab at
    once). (4) Radiation carried by a uniform flow with the hydrodynamics running: the pulse moves with the gas
    and gas + radiation energy closes to round-off."""
    from aegis import fld
    from aegis.solver3d import MHDSolver3D
    print("\n--- Grey FLD radiation ---"); sys.stdout.flush()

    def make(cls, n, L, **kw):
        c = dict(nx=n[0], ny=n[1], x_max=L[0], y_max=L[1], t_end=1.0, cfl=0.3, mach=1.0, gamma=5 / 3,
                 B_transverse=0.0, bc_x_type="periodic", bc_y_type="periodic", diag_interval=10**9, snapshot_times=[],
                 fld=True, fld_c=1.0, fld_kr0=1.0)
        if len(n) == 3:
            c.update(nz=n[2], z_max=L[2])
        c.update(kw)
        with contextlib.redirect_stdout(io.StringIO()):
            s = cls(Config(**c))
        W = xp.zeros((NVAR,) + s.U.shape[1:])
        W[iRHO], W[iPR] = 1.0, 1.0
        s.U = prim_to_cons(W, s.eos)
        s.t, s.step = 0.0, 0
        return s

    def interior(s):
        return s.U[(slice(None),) + (slice(s.ng, -s.ng),) * (s.U.ndim - 1)]

    def per(s):
        return s._periodic()

    ok = True
    # (1) equilibration: rho cv theta' = -c k (a theta^4 - E), E' = c k (a theta^4 - E), cv = 3/2
    a, c, k, dt = 1.0, 10.0, 1.0, 1e-3
    s = make(MHDSolver, (8, 8), (1.0, 1.0), fld_a=a, fld_c=c, fld_kp0=k, fld_advect=False)
    Ui = interior(s)
    E0 = float(xp.sum(Ui[EN])) * s.dV + s.radiation_energy()
    th, Er, ref = 1.0, 0.0, []
    for _ in range(50000):                                    # RK4 reference to t = 0.05
        f = lambda y: np.array([-c * k * (a * y[0]**4 - y[1]) / 1.5, c * k * (a * y[0]**4 - y[1])])
        y = np.array([th, Er]); h = 0.05 / 50000
        k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
        th, Er = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    for i in range(int(0.05 / dt + 0.5)):
        fld.step(s.Er, Ui, s.eos, s.cfg, (s.cfg.dx, s.cfg.dy), per(s), dt, advect=False)
    th_s = float(xp.mean(cons_to_prim(Ui, s.eos)[iPR]))
    e_tr = abs(th_s - th) / abs(1.0 - th)
    for i in range(2000):
        fld.step(s.Er, Ui, s.eos, s.cfg, (s.cfg.dx, s.cfg.dy), per(s), dt, advect=False)
    th_f = float(xp.mean(cons_to_prim(Ui, s.eos)[iPR]))
    eq = abs(a * th_f**4 - float(xp.mean(s.Er))) / float(xp.mean(s.Er))
    dE = abs(float(xp.sum(Ui[EN])) * s.dV + s.radiation_energy() - E0) / E0
    good = e_tr < 0.01 and eq < 1e-6 and dE < 1e-13
    ok = ok and good
    print(f"  equilibration: gas temperature at t = 0.05 within {100 * e_tr:.2f}% of the exact ODE (of its change); "
          f"final a T^4 = E_r to {eq:.1e}; gas + radiation energy {dE:.1e}  [{'PASS' if good else 'FAIL'}]")

    # (2) optically thick diffusion of a Gaussian: chi = 1e4, D = c / (3 chi) = 0.1
    D, t2, w0 = 0.1, 0.05, 0.05
    cases = []
    for name, cls, n, L, kw, dims in (("2D", MHDSolver, (128, 128), (1.0, 1.0), {}, 2),
                                      ("r-z", MHDSolver, (128, 8), (1.0, 8 / 128),
                                       dict(geometry="cylindrical", bc_x_type="extrapolation", div_b="ct"), 2),
                                      ("3D", MHDSolver3D, (48, 48, 48), (1.0, 1.0, 1.0), {}, 3)):
        s = make(cls, n, L, fld_c=3e3, fld_kr0=1e4, fld_advect=False, **kw)
        ng = s.ng
        cyl = kw.get("geometry") == "cylindrical"
        X = s.x[ng:-ng].reshape((-1,) + (1,) * (len(n) - 1))
        Y = s.y[ng:-ng].reshape((1, -1) + (1,) * (len(n) - 2))
        r2 = X**2 if cyl else (X - 0.5)**2 + (Y - 0.5)**2
        if dims == 3:
            r2 = r2 + (s.z[ng:-ng].reshape(1, 1, -1) - 0.5)**2
        s.Er[...] = 1.0 + xp.exp(-r2 / w0**2) + 0.0 * s.Er          # background 1 keeps R << 1
        Vw = getattr(s, "dV", s.cfg.dx * s.cfg.dy * s.cfg.dz)
        m2 = lambda: float(xp.sum((s.Er - 1.0) * r2 * Vw)) / float(xp.sum((s.Er - 1.0) * Vw))
        a0 = m2()
        Ui = interior(s)
        h = (s.cfg.dx, s.cfg.dy) + ((s.cfg.dz,) if dims == 3 else ())
        radial = (s.rf[ng:ng + s.cfg.nx + 1], s.rc[ng:ng + s.cfg.nx]) if cyl else None
        for _ in range(10):
            fld.step(s.Er, Ui, s.eos, s.cfg, h, per(s), t2 / 10, radial=radial, advect=False)
        ndir = 2 if cyl else dims
        err = abs((m2() - a0) / (2 * ndir * D * t2) - 1)
        cases.append((name, err, ndir))
    good = all(e < 0.01 for _, e, _ in cases)
    ok = ok and good
    print("  optically thick Gaussian: second moment growth vs 2 D t per direction: "
          + ", ".join(f"{nm} {100 * e:.2f}% off" for nm, e, _ in cases) + f"  [{'PASS' if good else 'FAIL'}]")

    # (3) optically thin slab lit from x = 0 (E = 1): front at most c t
    s = make(MHDSolver, (400, 4), (1.0, 0.01), bc_x_type="extrapolation", fld_kr0=1e-6, fld_E_xmin=1.0,
             fld_advect=False)
    s.Er[...] = 1e-10
    Ui = interior(s)
    t, dt3 = 0.0, 0.0025
    while t < 0.5 - 1e-12:
        fld.step(s.Er, Ui, s.eos, s.cfg, (s.cfg.dx, s.cfg.dy), per(s), dt3, advect=False)
        t += dt3
    E = to_numpy(s.Er[:, 0])
    x = (np.arange(400) + 0.5) / 400
    front = float(x[np.argmax(E < 0.5)])
    good = 0.9 * t <= front <= t + 2 / 400
    ok = ok and good
    print(f"  optically thin, hot wall: front (E_r = 1/2) at x = {front:.4f} at c t = {t:.2f}  "
          f"[{'PASS' if good else 'FAIL'}]")

    # (4) advection by a uniform flow with the hydrodynamics running
    s = make(MHDSolver, (128, 4), (1.0, 4 / 128), fld_c=1.0, fld_kr0=1e8, fld_a=0.0)
    W = cons_to_prim(s.U, s.eos)
    W[iVX] = 0.5
    s.U = prim_to_cons(W, s.eos)
    xc = to_numpy(s.x[s.ng:-s.ng])
    s.Er[...] = xp.asarray(np.exp(-((xc - 0.3) / 0.05)**2))[:, None] + 0.0 * s.Er
    cen0 = float(np.sum(to_numpy(s.Er[:, 0]) * xc) / np.sum(to_numpy(s.Er[:, 0])))
    E0 = _energy(s) + s.radiation_energy()
    _march(s, 0.4)
    cen = float(np.sum(to_numpy(s.Er[:, 0]) * xc) / np.sum(to_numpy(s.Er[:, 0])))
    dE = abs(_energy(s) + s.radiation_energy() - E0 - s._cumulative_boundary_energy) / E0
    good = abs(cen - cen0 - 0.5 * 0.4) < 1e-3 and dE < 1e-13
    ok = ok and good
    print(f"  advection with the flow (v = 0.5, t = 0.4): pulse centre moved {cen - cen0:.4f} (exact 0.2000); "
          f"gas + radiation energy {dE:.1e}  [{'PASS' if good else 'FAIL'}]")
    return ok


def render3d_test():
    """3D output and rendering (aegis/io3d.py, scripts/render_3d.py): a 32^3 state with a liner shell, a hot core
    and a twisted field is written as .vti (+ .pvd) and .h5, read back to the same arrays, and rendered off
    screen (volume, liner isosurface, hot-spot contours, field lines) to non-blank PNGs and an MP4. Skipped
    (reported, not failed) without pyvista."""
    import tempfile
    print("\n--- 3D output and rendering ---"); sys.stdout.flush()
    try:
        import pyvista  # noqa: F401
    except ImportError:
        print("  SKIPPED: pyvista not installed (pip install pyvista imageio imageio-ffmpeg h5py)")
        return True
    from aegis.solver3d import MHDSolver3D
    from aegis import io3d
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
    import render_3d
    cfg = Config(nx=32, ny=32, nz=32, x_min=-1, x_max=1, y_min=-1, y_max=1, z_max=2.0, t_end=1.0, cfl=0.3,
                 mach=1.0, B_transverse=1.0, bc_x_type="periodic", bc_y_type="periodic", powell_source=False,
                 diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver3D(cfg)
    X, Y, Z = s.x[:, None, None], s.y[None, :, None], s.z[None, None, :]
    r = xp.sqrt(X**2 + Y**2) + 0.0 * Z
    W = xp.zeros((NVAR,) + s.U.shape[1:])
    shell = 0.5 * (1 + xp.tanh((r - 0.5) / 0.05)) * 0.5 * (1 - xp.tanh((r - 0.7) / 0.05))
    W[iRHO] = 1.0 + 9.0 * shell
    W[iPR] = 1.0 + 4.0 * xp.exp(-(r / 0.25)**2 - ((Z - 1.0) / 0.4)**2)
    W[iCLR] = shell
    W[iBZ] = 1.0 + 0.0 * r
    W[iBX], W[iBY] = -0.3 * Y * xp.exp(-r**2), 0.3 * X * xp.exp(-r**2)     # twist about the axis
    s.U = prim_to_cons(W, s.eos)
    out = tempfile.mkdtemp(prefix="render3d_")
    rec = io3d.Recorder(out, fmt=("vti", "h5"))
    rec(s)
    s.t = 0.1
    s.U[iBZ] *= 1.2                                          # conserved and primitive B share the index
    rec(s)
    a = io3d.fields(s)
    g1, g2 = render_3d.load(os.path.join(out, "vtk", "frame_0001.vti")), render_3d.load(os.path.join(out, "h5", "frame_0001.h5"))
    back = max(float(np.max(np.abs(np.asarray(g.cell_data["rho"]) - a["rho"].ravel(order="F")))) for g in (g1, g2))
    with contextlib.redirect_stdout(io.StringIO()):
        pngs, mp4 = render_3d.render(render_3d.frames_from([os.path.join(out, "series.pvd")]),
                                     os.path.join(out, "renders"), size=(640, 480), fps=4)
    import imageio.v2 as imageio
    spread = min(float(np.std(imageio.imread(p)[..., :3])) for p in pngs)
    good = back < 1e-6 and len(pngs) == 2 and spread > 5.0 and mp4 is not None and os.path.getsize(mp4) > 1000
    print(f"  wrote 2 frames (.vti/.pvd and .h5), read back to {back:.1e}; rendered {len(pngs)} PNGs (pixel spread "
          f"{spread:.0f}) and {os.path.basename(mp4) if mp4 else 'no movie'}  [{'PASS' if good else 'FAIL'}]")
    return good


def tabular_test():
    """Tabulated EOS and opacities (aegis/tabular.py). (1) A Mie-Gruneisen aluminium table with power-law
    opacities written as SESAME ASCII (tables 301, 502, 505) reads back to the same arrays. (2) An ideal-gas
    table as the liner over the whole of a Sod shock tube reproduces the analytic ideal-gas run (log-log
    interpolation is exact for power laws). (3) The aluminium table reproduces the analytic Mie-Gruneisen
    liner in a 2 km/s impact, and p(rho, e) stays continuous where the states leave the table for the
    analytic fallback. (4) FLD with the opacities from a table matches FLD with the same power laws."""
    import tempfile
    from aegis import tabular, fld
    print("\n--- Tabular EOS and opacities ---"); sys.stdout.flush()
    tmp = tempfile.mkdtemp(prefix="tab_")
    ok = True
    # (1) SESAME round trip
    tab = tabular.mie_gruneisen_table(2700.0, 5328.0, 1.338, 2.0, 900.0, (1500.0, 6000.0), (10.0, 3e5), 60, 70,
                                      kappa=(1e-2, 4e-2, 1.0, -1.5))
    nr, nt = len(tab["rho"]), len(tab["T"])
    w301 = np.concatenate([[nr, nt], tab["rho"] / 1e3, tab["T"], tab["p"].T.ravel() / 1e9, tab["e"].T.ravel() / 1e6,
                           np.zeros(nr * nt)])
    lk = lambda k: np.concatenate([[nr, nt], np.log10(tab["rho"] / 1e3), np.log10(tab["T"] / 11604.518),
                                   np.log10(k.T.ravel() / 0.1)])
    path = os.path.join(tmp, "al_test.ses")
    tabular.write_sesame(path, 3719, {301: w301, 502: lk(tab["kappa_R"]), 505: lk(tab["kappa_P"])})
    back = tabular.load(path)
    rt = max(float(np.max(np.abs(back[k] - tab[k]) / np.max(np.abs(tab[k])))) for k in ("rho", "T", "p", "e", "kappa_R", "kappa_P"))
    good = rt < 1e-12
    ok = ok and good
    print(f"  SESAME ASCII (301, 502, 505) round trip: {rt:.1e}  [{'PASS' if good else 'FAIL'}]")

    # (2) ideal-gas table vs ideal gas, Sod (code units = SI here: R = 1, theta = T)
    ig = os.path.join(tmp, "ideal.npz")
    np.savez(ig, **tabular.ideal_gas_table(1.4, 1.0, (0.01, 10.0), (0.01, 100.0), 40, 50))
    runs = []
    for table in ("", ig):
        cfg = Config(nx=400, ny=4, x_max=1.0, y_max=0.01, t_end=0.2, cfl=0.3, mach=1.0, gamma=1.4, B_transverse=0.0,
                     bc_x_type="extrapolation", bc_y_type="periodic", material_tracer="volume_fraction",
                     eos_liner="ideal", gamma_liner=1.4, eos_table=table, tab_rho_unit=1.0, tab_v_unit=1.0,
                     diag_interval=10**9, snapshot_times=[])
        with contextlib.redirect_stdout(io.StringIO()):
            s = MHDSolver(cfg)
        W = xp.zeros((NVAR,) + s.U.shape[1:])
        left = s.X < 0.5
        W[iRHO], W[iPR], W[iCLR] = xp.where(left, 1.0, 0.125), xp.where(left, 1.0, 0.1), 1.0
        s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
        _march(s, 0.2)
        runs.append((to_numpy(cons_to_prim(s.U, s.eos)[:, s.ng:-s.ng, s.ng]), s.step))
    (Wa, na), (Wt, nt_) = runs
    d = max(float(np.max(np.abs(Wt[k] - Wa[k]))) / float(np.max(np.abs(Wa[k]))) for k in (iRHO, iPR, iVX))
    good = d < 1e-8 and na == nt_
    ok = ok and good
    print(f"  ideal-gas table vs ideal gas, Sod: {na}/{nt_} steps, max rel diff {d:.1e}  [{'PASS' if good else 'FAIL'}]")

    # (3) Mie-Gruneisen table vs analytic MG: Al on Al at 2 km/s (g/cc, km/s, GPa)
    alp = os.path.join(tmp, "al.npz")
    np.savez(alp, **tabular.mie_gruneisen_table(2700.0, 5328.0, 1.338, 2.0, 900.0, (2000.0, 4500.0), (10.0, 4e4), 80, 80))
    mg = dict(eos_liner="mie_gruneisen", mg_rho0=2.7, mg_c0=5.328, mg_s=1.338, mg_gamma0=2.0)
    runs = []
    for table in ("", alp):
        cfg = Config(nx=400, ny=4, x_max=1.0, y_max=0.01, t_end=0.06, cfl=0.3, mach=1.0, gamma=1.4, B_transverse=0.0,
                     bc_x_type="extrapolation", bc_y_type="periodic", material_tracer="volume_fraction", eos_table=table,
                     tab_rho_unit=1000.0, tab_v_unit=1000.0, diag_interval=10**9, snapshot_times=[], **mg)
        with contextlib.redirect_stdout(io.StringIO()):
            s = MHDSolver(cfg)
        W = xp.zeros((NVAR,) + s.U.shape[1:])
        W[iRHO], W[iPR], W[iCLR] = 2.7, 1e-4, 1.0
        W[iVX] = xp.where(s.X < 0.5, 2.0, 0.0)
        s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
        _march(s, 0.06)
        runs.append(to_numpy(cons_to_prim(s.U, s.eos)[:, s.ng:-s.ng, s.ng]))
    Wa, Wt = runs
    pa, pt = float(Wa[iPR][Wa[iVX].size // 2]), float(Wt[iPR][Wt[iVX].size // 2])
    d = max(float(np.max(np.abs(Wt[k] - Wa[k]))) / float(np.max(np.abs(Wa[k]))) for k in (iRHO, iPR, iVX))
    # continuity into the fallback: sweep e across the top of the table at fixed density
    tm = s.eos.tab
    rho = xp.full(4000, 3.0)
    e = xp.linspace(0.0, 2.0 * float(tm.g.host["e"].max()) if not tm.g.logs["e"] else 2 * float(10**tm.g.host["e"].max()), 4000)
    p = to_numpy(tm.pressure(rho, e))
    jump = float(np.max(np.abs(np.diff(p)))) / float(np.max(np.abs(p)))
    good = d < 5e-3 and jump < 2e-3
    ok = ok and good
    print(f"  Mie-Gruneisen table vs analytic, 2 km/s Al impact: shocked p {pt:.3f} vs {pa:.3f} GPa, max rel diff "
          f"{d:.1e}; p(e) across the table edge: largest step {jump:.1e} of the range (4000 samples)  "
          f"[{'PASS' if good else 'FAIL'}]")

    # (3b) device EOS (eos_fused.cu, used inline by the conversions and the HLLD kernel) vs the array EOS:
    # random states over pure liner, the 0.98-1 blend band, mixed cells, off-table states; then the impact again
    rng = np.random.default_rng(1)
    m4 = 40000
    a_ = xp.asarray(np.concatenate([np.ones(m4), rng.uniform(0.97, 1.0, m4), rng.uniform(0, 1, m4), np.zeros(m4)]))
    r_ = xp.asarray(rng.uniform(1.0, 6.0, 4 * m4)) * xp.maximum(a_, 0.01)
    p_ = xp.asarray(10**rng.uniform(-4, 2.5, 4 * m4))
    eos = s.eos
    dv = [eos.eps(r_, p_, a_), eos.c2(r_, p_, a_), eos.p_floor(r_, a_)]
    dv.append(eos.pressure(r_, dv[0], a_))
    eos._device_eos = None
    rf = [eos.eps(r_, p_, a_), eos.c2(r_, p_, a_), eos.p_floor(r_, a_)]
    rf.append(eos.pressure(r_, rf[0], a_))
    dpar = max(float(xp.max(xp.abs(u - v) / (xp.abs(v) + 1e-12 * float(xp.max(xp.abs(v)))))) for u, v in zip(dv, rf))
    with contextlib.redirect_stdout(io.StringIO()):
        s2 = MHDSolver(cfg)
    s2.eos._device_eos = None                        # the same impact with the array EOS
    W = xp.zeros((NVAR,) + s2.U.shape[1:])
    W[iRHO], W[iPR], W[iCLR] = 2.7, 1e-4, 1.0
    W[iVX] = xp.where(s2.X < 0.5, 2.0, 0.0)
    s2.U = prim_to_cons(W, s2.eos); s2.t = 0.0; s2.step = 0
    _march(s2, 0.06)
    Wr = to_numpy(cons_to_prim(s2.U, s2.eos)[:, s2.ng:-s2.ng, s2.ng])
    drun = max(float(np.max(np.abs(Wt[k] - Wr[k]))) / float(np.max(np.abs(Wr[k]))) for k in (iRHO, iPR, iVX))
    good = dpar < 1e-9 and drun < 1e-9
    ok = ok and good
    print(f"  device EOS (fused kernels) vs array EOS: 160k random states {dpar:.1e}, impact run {drun:.1e}  "
          f"[{'PASS' if good else 'FAIL'}]")

    # (4) opacity table vs the same power laws in FLD (code = SI; kappa = k0 rho theta^-1.5)
    op = os.path.join(tmp, "kap.npz")
    rho_k, T_k = np.geomspace(0.1, 10, 30), np.geomspace(0.1, 10, 30)
    R_, T_ = np.meshgrid(rho_k, T_k, indexing="ij")
    np.savez(op, rho=rho_k, T=T_k, kappa_R=2.0 * R_ * T_**-1.5, kappa_P=5.0 * R_ * T_**-1.5)
    res = []
    for table in ("", op):
        kw = dict(fld=True, fld_c=10.0, fld_a=1.0, fld_advect=False)
        kw.update(dict(fld_opacity_table=op, tab_rho_unit=1.0, tab_L_unit=1.0, tab_T_unit=1.0) if table else
                  dict(fld_kr0=2.0, fld_kr_rho=1.0, fld_kr_T=-1.5, fld_kp0=5.0, fld_kp_rho=1.0, fld_kp_T=-1.5))
        cfg = Config(nx=64, ny=64, x_max=1.0, y_max=1.0, t_end=1.0, cfl=0.3, mach=1.0, gamma=5 / 3, B_transverse=0.0,
                     bc_x_type="periodic", bc_y_type="periodic", diag_interval=10**9, snapshot_times=[], **kw)
        with contextlib.redirect_stdout(io.StringIO()):
            s = MHDSolver(cfg)
        W = xp.zeros((NVAR,) + s.U.shape[1:])
        r2 = (s.X - 0.5)**2 + (s.Y - 0.5)**2
        W[iRHO], W[iPR] = 1.0 + 0.5 * xp.exp(-r2 / 0.02), 1.0 + 2.0 * xp.exp(-r2 / 0.01)
        s.U = prim_to_cons(W, s.eos)
        s.Er[...] = 0.5
        Ui = s.U[:, s.ng:-s.ng, s.ng:-s.ng]
        for _ in range(20):
            fld.step(s.Er, Ui, s.eos, s.cfg, (s.cfg.dx, s.cfg.dy), (True, True), 2e-3, advect=False)
        res.append((to_numpy(s.Er), to_numpy(cons_to_prim(Ui, s.eos)[iPR])))
    d = max(float(np.max(np.abs(res[1][k] - res[0][k]))) / float(np.max(np.abs(res[0][k]))) for k in (0, 1))
    good = d < 1e-9
    ok = ok and good
    print(f"  FLD with tabulated opacities vs the same power laws: max rel diff {d:.1e}  [{'PASS' if good else 'FAIL'}]")
    return ok


def validation_test():
    """Goodness of fit and calibration (aegis/validation.py) on analytic signals with known noise:
    (1) data = model + Gaussian noise (sigma known): chi^2_nu of the true model within its statistical spread
    (|chi^2_nu - 1| < 4 sqrt(2/dof)), RMS within 10% of sigma; (2) a model delayed by 37 ns: the peak offset and
    the fitted timing shift recover the delay to 2 ns; (3) a 1-parameter scan: the parabola recovers the true
    amplitude within 3 sigma, and sigma matches the Fisher value sigma / sqrt(sum f^2) to 5%; (4) CSV round
    trip of a trace."""
    import tempfile
    from aegis import validation as val
    print("\n--- Validation metrics and calibration ---"); sys.stdout.flush()
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 4.0, 801)
    model = lambda t, A=1.0, d=0.0: A * np.exp(-((t - 2.0 - d) / 0.4)**2) * np.sin(6 * (t - d))
    sig = 0.02
    tr = val.Trace("sig", t, model(t) + rng.normal(0, sig, t.size), sig, "V", "bdot")
    ts = np.linspace(-0.5, 4.5, 5001)
    m = val.compare(tr, ts, model(ts))
    ok1 = abs(m["chi2_nu"] - 1) < 4 * np.sqrt(2 / m["dof"]) and abs(m["rms"] / sig - 1) < 0.1
    m2 = val.compare(tr, ts, model(ts, d=0.037), fit_shift=True, max_shift=0.1)
    ok2 = abs(m2["peak_offset"] - 0.037) < 0.002 and abs(m2["shift"] + 0.037) < 0.002
    vals = np.linspace(0.9, 1.1, 5)
    chi = [val.compare(tr, ts, model(ts, A=v))["chi2"] for v in vals]
    best, s, _ = val.calibrate(vals, chi)
    fisher = sig / np.sqrt(np.sum(model(t)**2 / 1.0**2))
    ok3 = abs(best - 1.0) < 3 * fisher and abs(s / fisher - 1) < 0.05
    p = os.path.join(tempfile.mkdtemp(prefix="val_"), "t.csv")
    val.save_trace(p, tr)
    back = val.load_trace(p, kind="bdot")
    ok4 = float(np.max(np.abs(back.y - tr.y))) < 1e-12 and float(np.max(np.abs(back.sigma - tr.sigma))) < 1e-15
    print(f"  true model: chi2_nu {m['chi2_nu']:.3f} (dof {m['dof']}), RMS {m['rms'] / sig:.3f} sigma  "
          f"[{'PASS' if ok1 else 'FAIL'}]")
    print(f"  37 ns delay: peak offset {1e3 * m2['peak_offset']:.1f} ns, fitted shift {1e3 * m2['shift']:+.1f} ns  "
          f"[{'PASS' if ok2 else 'FAIL'}]")
    print(f"  amplitude scan: {best:.4f} +- {s:.4f} (true 1, Fisher sigma {fisher:.4f})  [{'PASS' if ok3 else 'FAIL'}]")
    print(f"  CSV round trip  [{'PASS' if ok4 else 'FAIL'}]")
    return ok1 and ok2 and ok3 and ok4


def amr_test():
    """Adaptive mesh refinement (aegis/amr.py), refinement ratio 4. (1) Periodic magnetized box (Orszag-Tang
    vortex plus a dense blob) with regridding every 4 steps: mass, momentum and energy conserved to round-off,
    div B at round-off on both levels, patches created and moved. (2) A gas / stiffened-gas interface carried at
    uniform velocity and pressure across fixed coarse-fine boundaries: p and u stay uniform (the volume-fraction
    equilibrium survives refluxing, prolongation and restriction). (3) A magnetized blast wave: the AMR density
    error against a uniform run at the fine resolution is under half the coarse run's, with fewer fine cells."""
    from aegis.amr import AMR, AMRParams
    print("\n--- Adaptive mesh refinement ---"); sys.stdout.flush()
    ok = True

    def box(n, L=1.0, **kw):
        c = dict(nx=n, ny=n, x_max=L, y_max=L, t_end=1.0, cfl=0.3, mach=1.0, gamma=5 / 3, B_transverse=1.0,
                 bc_x_type="periodic", bc_y_type="periodic", powell_source=False, diag_interval=10**9,
                 snapshot_times=[])
        c.update(kw)
        return Config(**c)

    def state(s, fn):
        W = xp.zeros((NVAR,) + s.U.shape[1:])
        fn(W, s.X, s.Y)
        return prim_to_cons(W, s.eos)

    # (1) conservation and div B with regridding
    def ot(W, X, Y):
        W[iRHO] = 25 / (36 * np.pi) * (1 + 3 * xp.exp(-((X - 0.3)**2 + (Y - 0.6)**2) / 0.003))
        W[iPR] = 5 / (12 * np.pi)
        W[iVX], W[iVY] = -xp.sin(2 * np.pi * Y), xp.sin(2 * np.pi * X)
        B0 = 1 / np.sqrt(4 * np.pi)
        W[iBX], W[iBY] = -B0 * xp.sin(2 * np.pi * Y), B0 * xp.sin(4 * np.pi * X)
    cfg = box(64)
    a = AMR(cfg, AMRParams(ratio=4, regrid_every=4, tol_rho=0.15, tol_p=0.3, margin=3))
    with contextlib.redirect_stdout(io.StringIO()):
        U = state(MHDSolver(cfg), ot)
    a.set_state(U)
    a.regrid()
    T0 = a.totals()
    boxes0, moved, div = list(a.boxes), False, 0.0
    for _ in range(24):
        a.advance(a.compute_dt())
        moved = moved or a.boxes != boxes0
        div = max(div, a.div_b())
    T1 = a.totals()
    cons = max(abs(T1[k] - T0[k]) / max(abs(T0[k]), abs(T0["E"])) for k in ("mass", "mx", "my", "E"))
    good = cons < 1e-12 and div < 1e-12 and len(a.patches) > 0 and moved
    ok = ok and good
    print(f"  periodic MHD box, 24 coarse steps, {len(a.patches)} patches (moved: {moved}, fine cells "
          f"{100 * a.fine_fraction():.0f}% of a uniform fine grid): mass/momentum/energy {cons:.1e}, div B {div:.1e}  "
          f"[{'PASS' if good else 'FAIL'}]")

    # (1b) touching patches: two share an edge, a third touches one of them at a corner
    cfg = box(64)
    a = AMR(cfg, AMRParams(ratio=4, regrid_every=0))
    with contextlib.redirect_stdout(io.StringIO()):
        U = state(MHDSolver(cfg), ot)
    a.set_state(U)
    a.regrid([(12, 28, 20, 36), (28, 44, 16, 40), (44, 52, 40, 48)])
    T0 = a.totals()
    div = 0.0
    for _ in range(24):
        a.advance(a.compute_dt())
        div = max(div, a.div_b())
    T1 = a.totals()
    cons = max(abs(T1[k] - T0[k]) / max(abs(T0[k]), abs(T0["E"])) for k in ("mass", "mx", "my", "E"))
    ng = a.coarse.ng
    fA, fB = a.patches[0], a.patches[1]                  # shared edge x = 28, y in [20, 36]
    ea = fA.Bf[0][ng + fA.cfg.nx, ng:ng + fA.cfg.ny]
    eb = fB.Bf[0][ng, ng + 4 * (20 - 16):ng + 4 * (36 - 16)]
    seam = float(xp.max(xp.abs(ea - eb))) / float(xp.max(xp.abs(fA.Bf[0])))
    good = cons < 1e-12 and div < 1e-12 and seam < 1e-12
    ok = ok and good
    print(f"  three touching patches (edge and corner), 24 steps: mass/momentum/energy {cons:.1e}, div B {div:.1e}, "
          f"shared-edge faces agree to {seam:.1e}  [{'PASS' if good else 'FAIL'}]")

    # (1b') a T-junction: the top corners of two side-by-side patches lie on the bottom edge of a third, so that
    # corner is shared by three patches (the coarse divergence once grew there, step after step)
    a = AMR(box(64), AMRParams(ratio=4, regrid_every=0))
    with contextlib.redirect_stdout(io.StringIO()):
        U = state(MHDSolver(box(64)), ot)
    a.set_state(U)
    a.regrid([(12, 24, 12, 32), (24, 36, 18, 32), (16, 44, 32, 44)])
    T0, div = a.totals(), 0.0
    for _ in range(60):
        a.advance(a.compute_dt())
        div = max(div, a.div_b())
    T1 = a.totals()
    cons = max(abs(T1[k] - T0[k]) / max(abs(T0[k]), abs(T0["E"])) for k in ("mass", "mx", "my", "E"))
    good = cons < 1e-12 and div < 1e-12
    ok = ok and good
    print(f"  T-junction of three patches, 60 steps: mass/momentum/energy {cons:.1e}, div B (both levels) {div:.1e}  "
          f"[{'PASS' if good else 'FAIL'}]")

    # (1c) the recorded CUDA graphs (coarse step, fine sub-step, synchronisation) replay exactly what the eager
    # code computes, through regrids
    runs = []
    for graphs in (False, True):
        a = AMR(box(64), AMRParams(ratio=4, regrid_every=4, tol_rho=0.15, tol_p=0.3, margin=3, graphs=graphs))
        with contextlib.redirect_stdout(io.StringIO()):
            U = state(MHDSolver(box(64)), ot)
        a.set_state(U)
        a.regrid()
        for _ in range(12):
            a.advance(a.compute_dt())
        runs.append((a.composite(iRHO), a.composite(iBX), a.graph_launches))
    dg = max(float(np.max(np.abs(runs[1][k] - runs[0][k]))) for k in (0, 1))
    good = dg == 0.0 and len(runs[1][2]) == 3
    ok = ok and good
    print(f"  CUDA-graph replay vs eager, 12 steps with regrids: max difference {dg:.1e} (replays: "
          f"{', '.join(f'{k} {v}' for k, v in sorted(runs[1][2].items()))})  [{'PASS' if good else 'FAIL'}]")

    # (2) material interface across coarse-fine boundaries at uniform p, u
    mm = dict(B_transverse=0.0, material_tracer="volume_fraction", gamma=1.4, eos_liner="stiffened", gamma_liner=4.4,
              p_inf_liner=6.0)
    cfg = box(64, **mm)

    def slab(W, X, Y):
        a_ = 0.5 * (1 + xp.tanh((0.15 - xp.abs(X - 0.3)) / 0.01))
        W[iRHO], W[iPR], W[iVX], W[iVY], W[iCLR] = 1.0 + 9.0 * a_, 1.0, 1.0, 0.5, a_
    a = AMR(cfg, AMRParams(ratio=4, regrid_every=0))
    with contextlib.redirect_stdout(io.StringIO()):
        U = state(MHDSolver(cfg), slab)
    a.set_state(U)
    a.regrid([(20, 44, 16, 40)])                     # fixed patch: the slab sweeps through its edges
    for _ in range(60):
        a.advance(a.compute_dt())
    W = [cons_to_prim(a.coarse.U, a.coarse.eos)] + [cons_to_prim(f.U, f.eos) for f in a.patches]
    ng = a.coarse.ng
    dp = max(float(xp.max(xp.abs(w[iPR, ng:-ng, ng:-ng] - 1.0))) for w in W)
    du = max(float(xp.max(xp.abs(w[iVX, ng:-ng, ng:-ng] - 1.0))) + float(xp.max(xp.abs(w[iVY, ng:-ng, ng:-ng] - 0.5)))
             for w in W)
    good = dp < 1e-8 and du < 1e-8
    ok = ok and good
    print(f"  gas / stiffened-gas slab through a fixed patch (60 steps): max |p - p0| {dp:.1e}, |u - u0| {du:.1e}  "
          f"[{'PASS' if good else 'FAIL'}]")

    # (3) magnetized blast wave: AMR vs uniform coarse and uniform fine
    def blast(W, X, Y):
        r2 = (X - 0.5)**2 + (Y - 0.5)**2
        W[iRHO] = 1.0
        W[iPR] = 0.1 + 9.9 * (r2 < 0.1**2)
        W[iBX] = W[iBY] = 1.0 / np.sqrt(2.0)
    t_end, res = 0.06, {}
    for name, n in (("fine", 256), ("coarse", 64)):
        cfg = box(n, gamma=1.4)
        with contextlib.redirect_stdout(io.StringIO()):
            s = MHDSolver(cfg)
        s.U = state(s, blast); s.t = 0.0; s.step = 0
        _march(s, t_end)
        res[name] = to_numpy(cons_to_prim(s.U, s.eos)[iRHO, s.ng:-s.ng, s.ng:-s.ng])
    cfg = box(64, gamma=1.4)
    a = AMR(cfg, AMRParams(ratio=4, regrid_every=2, tol_rho=0.03, tol_p=0.1, margin=3, buffer=2))
    with contextlib.redirect_stdout(io.StringIO()):
        U = state(MHDSolver(cfg), blast)
    a.set_state(U)
    a.regrid()
    frac = []
    while a.t < t_end - 1e-12:
        a.advance(min(a.compute_dt(), t_end - a.t))
        frac.append(a.fine_fraction())
    ref = res["fine"]
    l1 = lambda f: float(np.mean(np.abs(f - ref)))
    e_amr, e_c = l1(a.composite(iRHO)), l1(np.repeat(np.repeat(res["coarse"], 4, 0), 4, 1))
    good = e_amr < 0.5 * e_c and max(frac) < 0.5
    ok = ok and good
    print(f"  magnetized blast wave vs a uniform 256^2 run: L1 density error coarse 64^2 {e_c:.4f}, AMR 64^2 + ratio-4 "
          f"patches {e_amr:.4f} ({e_amr / e_c:.2f}x) with at most {100 * max(frac):.0f}% (mean {100 * np.mean(frac):.0f}%) "
          f"of the fine cells  [{'PASS' if good else 'FAIL'}]")
    return ok


def circuit_test():
    """RLC drive circuit (aegis/circuit.py). (1) Fixed load: the underdamped discharge I = V0/(w L) e^(-a t)
    sin(w t) to 1e-6 of the peak at 4000 steps, bank energy budget to round-off. (2) An Al liner (R 10-11 mm,
    10 mm long) at rest, driven by a 200 uF, 30 kV bank on a 64 x 4 r-z grid: bank + grid conserve energy
    to round-off, the hydro energy budget closes with the drive work, and at 5 us (drive below 1.5 GPa, so
    the metal barely compresses) the outer radius and current match an incompressible thick shell on the
    same circuit."""
    print("\n--- RLC drive circuit ---"); sys.stdout.flush()
    import argparse
    import csv as _csv
    import tempfile
    from aegis.circuit import RLCCircuit, thick_shell
    C, V, L, R, n, T = 100e-6, 40e3, 30e-9, 2e-3, 4000, 8e-6
    c = RLCCircuit(C, V, L, R)
    a_, w = R / (2 * L), np.sqrt(1 / (L * C) - (R / (2 * L))**2)
    err = 0.0
    for i in range(n):
        c.advance(T / n, L)
        t = (i + 1) * T / n
        err = max(err, abs(c.I - V / (w * L) * np.exp(-a_ * t) * np.sin(w * t)))
    rel = err / (V / (w * L) * np.exp(-a_ * np.arctan(w / a_) / w) * np.sin(np.arctan(w / a_)))
    ok1 = rel < 1e-6 and abs(c.budget_error()) < 1e-13
    print(f"  fixed load: current within {rel:.1e} of the exact discharge, bank budget {c.budget_error():.1e}  "
          f"[{'PASS' if ok1 else 'FAIL'}]")

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples"))
    import fcg_flux_compression as fcg
    out = tempfile.mkdtemp(prefix="circuit_")
    t_end = 5.0
    a = argparse.Namespace(liner_material="al", b0_tesla=1.0, v0_kms=0.0, enable_strength=False, eta_model="ideal",
                           sigma_gas=None, sigma_min=5e6, r0=10.0, dr=1.0, r_max=14.0, lz=10.0, mode=1, amp=0.0,
                           rho_gas=0.1, p_gas_bar=0.1, nr=64, nz=4, t_end=t_end, frame_dt=10.0, diag_every=5,
                           capacitance_uf=200.0, voltage_kv=30.0, inductance_nh=15.0, resistance_mohm=1.0,
                           r_return_mm=14.0)
    with contextlib.redirect_stdout(io.StringIO()):
        r = fcg.run(a, out)
    rows = list(_csv.DictReader(open(os.path.join(out, "diagnostics.csv"))))
    last = {k: float(v) for k, v in rows[-1].items()}
    ts, rs, Is = thick_shell(200e-6, 30e3, 15e-9, 1e-3, 11e-3, 10e-3, 2700.0, 10e-3, 14e-3, last["t_us"] * 1e-6,
                            n=20000)
    dr_ = abs(last["r_out_mm"] - rs[-1] * 1e3) / (11.0 - rs[-1] * 1e3)    # error relative to the distance moved
    dI = abs(last["I_MA"] - Is[-1] / 1e6) / (np.abs(Is).max() / 1e6)           # relative to the peak
    cc = r["circuit"]
    ok2 = (r["energy_error_max"] < 1e-12 and cc["work_mismatch_over_bank"] < 1e-12
           and abs(cc["circuit_budget_error"]) < 1e-12 and dr_ < 0.03 and dI < 0.01)
    print(f"  driven liner, {r['steps']} steps to {last['t_us']:.2f} us: I {last['I_MA']:.3f} MA (thick shell "
          f"{Is[-1] / 1e6:.3f}), outer radius {last['r_out_mm']:.3f} mm (thick shell {rs[-1] * 1e3:.3f}; "
          f"{100 * dr_:.1f}% of the distance moved); bank + grid energy {cc['work_mismatch_over_bank']:.1e}, hydro "
          f"budget {r['energy_error_max']:.1e}; load L from work vs radius {100 * cc['L_energy_vs_geom_max_rel']:.2g}% "
          f"apart  [{'PASS' if ok2 else 'FAIL'}]")
    return ok1 and ok2


def amr_physics_test():
    """Non-ideal terms across AMR levels (aegis/amr.py), 32^2 coarse, ratio 2, for each of resistive MHD,
    conduction, grey FLD and liner strength: (1) three touching patches (an edge and a corner shared) that move
    every 4 coarse steps: mass, momentum and total energy (gas + radiation) conserved to round-off and div B at
    round-off; with the new flux registers switched off the energy (or momentum) error is 1e-7 or more, so the
    refluxing of the resistive, conductive, radiative and stress fluxes is what closes the budget; (2) a feature
    inside one patch (initialised at the fine resolution): the AMR solution in the patch matches a uniform fine run
    to under 2% of the coarse run's error."""
    from aegis.amr import AMR, AMRParams, _Recorder
    from aegis.eos import stiffened_from_hugoniot
    from aegis import ct as _ct
    print("\n--- AMR with resistive, conduction, FLD and strength terms ---"); sys.stdout.flush()
    gA, piA = stiffened_from_hugoniot(AL["rho0"], AL["c0"], AL["s"], 1.0)
    g2 = lambda X, Y, w: xp.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / w)

    def init_resist(W, X, Y):
        W[iRHO], W[iPR], W[iBZ], W[iBX] = 1.0, 20.0, g2(X, Y, 0.01), 0.2

    def init_cond(W, X, Y):
        W[iRHO], W[iPR] = 1.0, 1.0 + 2.0 * g2(X, Y, 0.01)

    def init_fld(W, X, Y):
        W[iRHO], W[iPR], W[iVX] = 1.0, 1.0 + g2(X, Y, 0.01), 0.3

    def init_strength(W, X, Y):
        W[iRHO], W[iPR], W[iCLR], W[iVY] = AL["rho0"], 1e-3, 1.0, 0.05 * g2(X, Y, 0.004)
    cases = [("resistive", dict(eta_resistivity=2e-3, B_transverse=1.0), init_resist, 0.05, iBZ),
             ("conduction", dict(cond_kappa0=2e-3, cond_x0=0.0), init_cond, 0.03, iPR),
             ("FLD", dict(fld=True, fld_c=10.0, fld_a=1.0, fld_kr0=30.0, fld_kp0=1.0, fld_advect=True), init_fld, 0.05,
              iPR),
             ("strength", dict(eos_liner="stiffened", gamma_liner=gA, p_inf_liner=piA, material_tracer="volume_fraction",
                               strength=True, shear_modulus=26.0, yield_strength=100.0), init_strength, 0.03, iVY)]

    def cfg_of(n, kw):
        c = dict(nx=n, ny=n, x_max=1.0, y_max=1.0, t_end=1.0, cfl=0.3, mach=1.0, gamma=5 / 3, B_transverse=0.0,
                 bc_x_type="periodic", bc_y_type="periodic", powell_source=False, diag_interval=10**9, snapshot_times=[])
        c.update(kw)
        return Config(**c)

    def fill(s, fn):
        W = xp.zeros((NVAR,) + s.U.shape[1:])
        fn(W, s.X, s.Y)
        if s.Er is not None:
            s.Er[...] = 1.0 + g2(s.X[s.ng:-s.ng, s.ng:-s.ng], s.Y[s.ng:-s.ng, s.ng:-s.ng], 0.005)
        return prim_to_cons(W, s.eos)

    def amr_run(kw, fn, steps=None, t_end=None, boxes=None, move=False, exact=False):
        n, r = 32, 2
        cfg = cfg_of(n, kw)
        a = AMR(cfg, AMRParams(ratio=r, regrid_every=0))
        with contextlib.redirect_stdout(io.StringIO()):
            s0 = MHDSolver(cfg)
        a.set_state(fill(s0, fn))
        if s0.Er is not None:
            a.coarse.Er[...] = s0.Er
        a.regrid(boxes)
        if exact:                                    # patch data at the fine resolution, not prolonged
            for f in a.patches:
                ng = f.ng
                f.U[:, ng:-ng, ng:-ng] = fill(f, fn)[:, ng:-ng, ng:-ng]
                _ct.cells_from_faces(f.U, f.Bf, (slice(ng, -ng), slice(ng, -ng)))
        T0, div, k = a.totals(), 0.0, 0
        shift = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        while (steps is not None and k < steps) or (t_end is not None and a.t < t_end - 1e-12):
            a.advance(a.compute_dt() if t_end is None else min(a.compute_dt(), t_end - a.t))
            k += 1
            div = max(div, a.div_b())
            if move and k % 4 == 0:
                dx_, dy_ = shift[(k // 4 - 1) % 4]
                a.regrid([(b[0] + dx_, b[1] + dx_, b[2] + dy_, b[3] + dy_) for b in a.boxes])
        T1 = a.totals()
        cons = max(abs(T1[q] - T0[q] - (T1["boundary"] - T0["boundary"] if q == "E" else 0.0)) / abs(T0["E"])
                   for q in ("mass", "mx", "my", "E"))
        return a, cons, div

    def uniform(n, kw, fn, t_end, var):
        with contextlib.redirect_stdout(io.StringIO()):
            s = MHDSolver(cfg_of(n, kw))
        s.U = fill(s, fn); s.t = 0.0; s.step = 0
        _march(s, t_end)
        return to_numpy(cons_to_prim(s.U, s.eos)[var, s.ng:-s.ng, s.ng:-s.ng])

    touching = [(9, 17, 11, 19), (17, 23, 9, 21), (23, 27, 21, 25)]
    ok = True
    for name, kw, fn, t_end, var in cases:
        _, cons, div = amr_run(kw, fn, steps=12, boxes=touching, move=True)
        saved = (_Recorder._add_flux, _Recorder._add_split_flux)
        _Recorder._add_flux = _Recorder._add_split_flux = lambda *a_: None
        try:
            _, cons_off, _ = amr_run(kw, fn, steps=8, boxes=touching, move=True)
        finally:
            _Recorder._add_flux, _Recorder._add_split_flux = saved
        a, _, _ = amr_run(kw, fn, t_end=t_end, boxes=[(6, 26, 6, 26)], exact=True)
        fine, coarse = uniform(64, kw, fn, t_end, var), uniform(32, kw, fn, t_end, var)
        up = lambda q: np.repeat(np.repeat(q, 2, 0), 2, 1)
        sl = (slice(12, 52), slice(12, 52))
        e_amr = float(np.mean(np.abs(a.composite(var)[sl] - fine[sl])))
        e_c = float(np.mean(np.abs(up(coarse)[sl] - fine[sl])))
        good = cons < 1e-13 and div < 1e-12 and cons_off > 1e-8 and e_amr < 0.02 * e_c
        ok = ok and good
        print(f"  {name:10s}: moving touching patches, 12 steps: conservation {cons:.1e} (registers off: "
              f"{cons_off:.1e}), div B {div:.1e}; feature in a patch: L1 error vs uniform 64^2 {e_amr:.1e} "
              f"(coarse {e_c:.1e})  [{'PASS' if good else 'FAIL'}]")
    return ok


def amr_transport_test():
    """The remaining terms across AMR levels (aegis/amr.py), 32^2 coarse, ratio 2: Hall + anisotropic resistivity,
    Nernst + Ettingshausen (two-temperature), two-temperature electron and ion conduction with equilibration,
    gyroviscosity, Braginskii viscosity and optically thin radiation. For each: (1) three touching patches moving
    every 4 steps conserve mass, momentum and total energy (radiated energy counted) to round-off with div B at
    round-off, and switching off the new registers (for radiation: the coarse level radiating under patches too)
    breaks the budget; (2) a feature inside one patch, initialised at the fine resolution, matches a uniform fine
    run to under 5% of the coarse run's L1 error. (3) The electron-energy register: under electron conduction only,
    the refluxed heat must stay with the electrons; without the register the coarse cells next to the patch heat
    their ions: the change of T_i over the run departs from the uniform fine run's (the change, so that the coarse
    sampling of the initial state cancels)."""
    from aegis.amr import AMR, AMRParams, _Recorder
    from aegis import ct as _ct, twotemp
    print("\n--- AMR with Hall, Nernst, two-temperature, gyroviscosity, viscosity, radiation ---"); sys.stdout.flush()
    g2 = lambda X, Y, w: xp.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / w)

    def st_hall(W, X, Y):
        W[iRHO], W[iPR], W[iBX], W[iBZ] = 1.0, 20.0, 0.3, 0.5 + g2(X, Y, 0.01)

    def st_hot(W, X, Y):
        W[iRHO], W[iPR], W[iBZ], W[iBX] = 1.0, 1.0 + 2.0 * g2(X, Y, 0.01), 1.0, 0.2

    def st_shear(W, X, Y):
        W[iRHO], W[iPR], W[iBZ], W[iBX] = 1.0, 1.0 + g2(X, Y, 0.01), 1.0, 0.3
        W[iVX], W[iVY] = 0.3 * g2(X, Y, 0.01), 0.2 * g2(X, Y, 0.02)
    Te = lambda U: twotemp.temperatures(U, 5 / 3, 1.0)[0]
    tt = dict(two_temp=True, cond_kappa0=2e-3, cond_x0=0.5, cond_kappa_i0=1e-3, cond_x0_i=0.5, te_eq_coeff=0.5)
    cases = [("Hall", dict(hall_di=0.05, eta_resistivity=2e-3, eta_perp_ratio=1.96), st_hall, 0.01, iBZ),
             ("Nernst", dict(nernst_coeff=0.05, cond_x0=2.0, two_temp=True), st_hot, 0.03, iBZ),
             ("two-temp", tt, st_hot, 0.03, Te),
             ("gyroviscous", dict(gyro_di=0.3), st_shear, 0.03, iVY),
             ("viscosity", dict(visc_coeff=2e-3), st_shear, 0.03, iVX),
             ("radiation", dict(rad_coeff=0.5, rad_T_min=0.1), st_hot, 0.05, iPR)]

    def cfg_of(n, kw):
        c = dict(nx=n, ny=n, x_max=1.0, y_max=1.0, t_end=1.0, cfl=0.3, mach=1.0, gamma=5 / 3, B_transverse=1.0,
                 bc_x_type="periodic", bc_y_type="periodic", powell_source=False, diag_interval=10**9, snapshot_times=[])
        c.update(kw)
        return Config(**c)

    def fill(s, fn):
        W = xp.zeros((NVAR,) + s.U.shape[1:])
        fn(W, s.X, s.Y)
        U = prim_to_cons(W, s.eos)
        if fn is st_electron:                        # p = 1 uniform, p_e = 0.5 (1 + bump), p_i takes the rest
            twotemp.set_pe(U, 0.5 * (1.0 + 0.8 * g2(s.X, s.Y, 0.01)), 5 / 3)
        return U

    def st_electron(W, X, Y):
        W[iRHO], W[iPR], W[iBZ] = 1.0, 1.0, 1.0

    def amr_run(kw, fn, steps=None, t_end=None, boxes=None, move=False, exact=False, rad_off=False, at0=None):
        cfg = cfg_of(32, kw)
        a = AMR(cfg, AMRParams(ratio=2, regrid_every=0))
        with contextlib.redirect_stdout(io.StringIO()):
            s0 = MHDSolver(cfg)
        a.set_state(fill(s0, fn))
        a.regrid(boxes)
        if exact:                                    # patch data at the fine resolution, not prolonged
            for f in a.patches:
                ng = f.ng
                f.U[:, ng:-ng, ng:-ng] = fill(f, fn)[:, ng:-ng, ng:-ng]
                _ct.cells_from_faces(f.U, f.Bf, (slice(ng, -ng), slice(ng, -ng)))
                if cfg.two_temp and fn is not st_electron:
                    twotemp.init_equal(f.U, cfg.gamma, cfg.z_ion)
        if at0 is not None:
            at0.append(a.composite(at0[0]))
        T0, div, k = a.totals(), 0.0, 0
        shift = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        while (steps is not None and k < steps) or (t_end is not None and a.t < t_end - 1e-12):
            if rad_off:
                a.coarse._uncovered = None
            a.advance(a.compute_dt() if t_end is None else min(a.compute_dt(), t_end - a.t))
            k += 1
            div = max(div, a.div_b())
            if move and k % 4 == 0:
                dx_, dy_ = shift[(k // 4 - 1) % 4]
                a.regrid([(b[0] + dx_, b[1] + dx_, b[2] + dy_, b[3] + dy_) for b in a.boxes])
        T1 = a.totals()
        dE = T1["E"] - T0["E"] - (T1["boundary"] - T0["boundary"]) + (T1["radiated"] - T0["radiated"])
        cons = max([abs(T1[q] - T0[q]) / abs(T0["E"]) for q in ("mass", "mx", "my")] + [abs(dE) / abs(T0["E"])])
        return a, cons, div

    def uniform(n, kw, fn, t_end, var, change=False):
        with contextlib.redirect_stdout(io.StringIO()):
            s = MHDSolver(cfg_of(n, kw))
        s.U = fill(s, fn); s.t = 0.0; s.step = 0
        q0 = var(s.U).copy() if change else 0.0
        _march(s, t_end)
        if change:
            return to_numpy((var(s.U) - q0)[s.ng:-s.ng, s.ng:-s.ng])
        q = var(s.U) if callable(var) else cons_to_prim(s.U, s.eos)[var]
        return to_numpy(q[s.ng:-s.ng, s.ng:-s.ng])

    touching = [(9, 17, 11, 19), (17, 23, 9, 21), (23, 27, 21, 25)]
    up = lambda q: np.repeat(np.repeat(q, 2, 0), 2, 1)
    sl = (slice(12, 52), slice(12, 52))
    ok = True
    for name, kw, fn, t_end, var in cases:
        _, cons, div = amr_run(kw, fn, steps=12, boxes=touching, move=True)
        saved = (_Recorder._add_flux, _Recorder._add_split_flux)
        if name != "radiation":
            _Recorder._add_flux = _Recorder._add_split_flux = lambda *a_: None
        try:
            _, cons_off, _ = amr_run(kw, fn, steps=8, boxes=touching, move=True, rad_off=(name == "radiation"))
        finally:
            _Recorder._add_flux, _Recorder._add_split_flux = saved
        a, _, _ = amr_run(kw, fn, t_end=t_end, boxes=[(6, 26, 6, 26)], exact=True)
        fine, coarse = uniform(64, kw, fn, t_end, var), uniform(32, kw, fn, t_end, var)
        e_amr = float(np.mean(np.abs(a.composite(var)[sl] - fine[sl])))
        e_c = float(np.mean(np.abs(up(coarse)[sl] - fine[sl])))
        good = cons < 1e-13 and div < 1e-12 and cons_off > 1e-8 and e_amr < 0.05 * e_c
        ok = ok and good
        print(f"  {name:11s}: moving touching patches, 12 steps: conservation {cons:.1e} (control {cons_off:.1e}), "
              f"div B {div:.1e}; feature in a patch: L1 error vs uniform 64^2 {e_amr:.1e} (coarse {e_c:.1e})  "
              f"[{'PASS' if good else 'FAIL'}]")

    # electron register: T_i (which electron conduction must not touch), patch edge inside the hot region
    Ti = lambda U: twotemp.temperatures(U, 5 / 3, 1.0)[1]
    kw = dict(two_temp=True, cond_kappa0=4e-3, cond_x0=0.0)
    fine = uniform(64, kw, st_electron, 0.03, Ti, change=True)
    errs = []
    for on in (True, False):
        saved = _Recorder._add_split_flux
        if not on:
            _Recorder._add_split_flux = lambda self, key, *a_: None if key == "Fe" else saved(self, key, *a_)
        try:
            at0 = [Ti]
            a, cons, _ = amr_run(kw, st_electron, t_end=0.03, boxes=[(12, 20, 10, 22)], exact=True, at0=at0)
        finally:
            _Recorder._add_split_flux = saved
        errs.append(float(np.max(np.abs(a.composite(Ti) - at0[1] - fine))))
    good = errs[0] < 0.2 * errs[1]
    ok = ok and good
    print(f"  electron-energy register (electron conduction across a patch edge): max |dT_i - dT_i(uniform 64^2)| "
          f"{errs[0]:.1e} with it, {errs[1]:.1e} without  [{'PASS' if good else 'FAIL'}]")
    return ok


def cylindrical3d_test():
    """3D cylindrical (r, theta, z) geometry in solver3d. (1) Without theta variation the 3D solver reproduces the 2D
    r-z solver (swirl, azimuthal and axial field, z-dependent flow, the axis) to round-off over 30 steps of the same
    dt. (2) Non-axisymmetric (m = 1): a density blob carried straight across the axis by a uniform Cartesian flow
    along a uniform Cartesian field (v_r = U cos theta, v_theta = -U sin theta, faces from A_z = B0 y), here onto the
    axis: the errors in the blob's centroid (exact: moves U t) and in the field (exact: uniform) fall with
    resolution, energy is conserved to round-off and the cylindrical face divergence stays at round-off."""
    from aegis.solver3d import MHDSolver3D
    print("\n--- 3D cylindrical geometry (r, theta, z) ---"); sys.stdout.flush()
    ok = True
    base = dict(cfl=0.3, gamma=5 / 3, mach=1.0, B_transverse=1.0, geometry="cylindrical", div_b="ct",
                bc_x_type="extrapolation", bc_y_type="periodic", powell_source=False, diag_interval=10**9,
                snapshot_times=[], t_end=1.0)
    Nr, Nz, Nth = 32, 24, 4
    with contextlib.redirect_stdout(io.StringIO()):
        s2 = MHDSolver(Config(nx=Nr, ny=Nz, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, **base))
        s3 = MHDSolver3D(Config(nx=Nr, ny=Nth, nz=Nz, x_min=0.0, x_max=1.0, y_min=0.0, y_max=2 * np.pi, z_min=0.0,
                                z_max=1.0, **base))
    R, Z = s2.X, s2.Y
    W2 = xp.zeros((NVAR,) + s2.U.shape[1:])
    W2[iRHO] = 1.0 + 0.3 * xp.exp(-((R - 0.4)**2 + (Z - 0.5)**2) / 0.02)
    W2[iPR] = 1.0 + 0.5 * xp.exp(-(R**2) / 0.05)
    W2[iVX] = 0.2 * xp.sin(2 * np.pi * Z) * R * xp.exp(-R**2 / 0.2)
    W2[iVY] = 0.1 * xp.cos(2 * np.pi * Z)
    W2[iVZ] = -0.3 * R * xp.exp(-R**2 / 0.1)          # w = -v_phi: swirl
    W2[iBY] = 0.5 + 0.2 * xp.exp(-R**2 / 0.1)
    W2[iBZ] = -0.4 * R * xp.exp(-R**2 / 0.1)          # B_w = -B_phi
    s2.U = prim_to_cons(W2, s2.eos); s2.t = 0.0
    m3 = lambda q: xp.repeat(q[:, None, :], s3.U.shape[2], axis=1)
    W3 = xp.zeros((NVAR,) + s3.U.shape[1:])
    W3[iRHO], W3[iPR] = m3(W2[iRHO]), m3(W2[iPR])
    W3[iVX], W3[iVY], W3[iVZ] = m3(W2[iVX]), -m3(W2[iVZ]), m3(W2[iVY])
    W3[iBX], W3[iBY], W3[iBZ] = m3(W2[iBX]), -m3(W2[iBZ]), m3(W2[iBY])
    s3.U = prim_to_cons(W3, s3.eos); s3.t = 0.0
    for _ in range(30):
        dt = s2.compute_dt(cons_to_prim(s2.U, s2.eos))
        s2.dt = s3.dt = dt
        s2.step_ssprk3(); s2.t += dt
        s3.step_ssprk3(); s3.t += dt
    ng = s2.ng
    A = to_numpy(cons_to_prim(s2.U, s2.eos)[:, ng:-ng, ng:-ng])
    Bq = to_numpy(cons_to_prim(s3.U, s3.eos)[:, ng:-ng, ng:-ng, ng:-ng])
    worst = 0.0
    for v2, v3, sg in ((iRHO, iRHO, 1), (iPR, iPR, 1), (iVX, iVX, 1), (iVY, iVZ, 1), (iVZ, iVY, -1), (iBX, iBX, 1),
                       (iBY, iBZ, 1), (iBZ, iBY, -1)):
        sc = max(float(np.max(np.abs(A[v2]))), 1e-30)
        worst = max(worst, max(float(np.max(np.abs(sg * Bq[v3][:, j, :] - A[v2]))) for j in range(Nth)) / sc)
    good = worst < 1e-12 and s3.div_b_max() < 1e-12
    ok = ok and good
    print(f"  3D cylindrical with d/dtheta = 0 vs the 2D r-z solver, 30 steps: largest difference {worst:.1e}, div B "
          f"{s3.div_b_max():.1e}  [{'PASS' if good else 'FAIL'}]")
    res = [_cyl3d_blob(Nr_, Nt_, T=0.3) for Nr_, Nt_ in ((16, 24), (32, 48))]
    e_c = [abs(r_["xc"] - r_["xc_exact"]) for r_ in res]
    good = (res[1]["dB"] < 0.7 * res[0]["dB"] and e_c[1] < 0.7 * e_c[0] and max(r_["dE"] for r_ in res) < 1e-13
            and max(r_["div"] for r_ in res) < 1e-12)
    ok = ok and good
    print(f"  m = 1 blob carried onto the axis along a uniform field (t = 0.3): centroid error {e_c[0]:.1e} / {e_c[1]:.1e}, "
          f"field deviation {res[0]['dB']:.1e} / {res[1]['dB']:.1e} at 16 x 24 / 32 x 48 cells (r x theta); energy "
          f"{max(r_['dE'] for r_ in res):.1e}, div B {max(r_['div'] for r_ in res):.1e}  [{'PASS' if good else 'FAIL'}]")
    return ok


def _cyl3d_blob(Nr, Nth, T=0.5, U=1.0, B0=0.5, cyl_filter=False):
    """the m = 1 blob of cylindrical3d_test: returns centroid, field deviation, energy error, max div B"""
    from aegis.solver3d import MHDSolver3D
    from aegis import ct as _ct
    c = Config(nx=Nr, ny=Nth, nz=4, x_min=0.0, x_max=1.0, y_min=0.0, y_max=2 * np.pi, z_min=0.0, z_max=4.0 / Nr,
               cfl=0.3, gamma=5 / 3, mach=1.0, B_transverse=1.0, geometry="cylindrical", div_b="ct",
               bc_x_type="extrapolation", bc_y_type="periodic", powell_source=False, diag_interval=10**9,
               snapshot_times=[], t_end=T, cyl_filter=cyl_filter)
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver3D(c)
    ng = s.ng
    r = to_numpy(s.x)[:, None, None]
    th = to_numpy(s.y)[None, :, None]
    X, Y = r * np.cos(th), r * np.sin(th)
    rf = to_numpy(s.rf)
    thf = c.y_min + (np.arange(s.ny_tot + 1) - ng) * c.dy
    Az = B0 * rf[:, None] * np.sin(thf)[None, :]                       # A_z = B0 y on the z-edges
    Br = (Az[:, 1:] - Az[:, :-1]) / (np.where(rf > 0, rf, np.inf)[:, None] * c.dy)
    Bt = -(Az[1:, :] - Az[:-1, :]) / c.dx
    nz = s.nz_tot
    Bf = [xp.asarray(np.repeat(Br[:, :, None], nz, 2)), xp.asarray(np.repeat(Bt[:, :, None], nz, 2)),
          xp.zeros((s.nx_tot, s.ny_tot, nz + 1))]
    W = xp.zeros((NVAR,) + s.U.shape[1:])
    W[iRHO] = xp.asarray(1.0 + np.exp(-((X + 0.3)**2 + Y**2) / 0.01) + 0 * r * th)
    W[iPR] = 1.0
    W[iVX] = xp.asarray(U * np.cos(th) + 0 * r)
    W[iVY] = xp.asarray(-U * np.sin(th) + 0 * r)
    s.U = prim_to_cons(W, s.eos)
    s.Bf = Bf; s._faces_owner = s.U
    s._Bf1 = [xp.empty_like(f) for f in Bf]; s._dBf = [xp.empty_like(f) for f in Bf]
    _ct.cells_from_faces(s.U, Bf, (slice(None),) * 3)
    s.U[EN] += 0.5 * (s.U[iBX]**2 + s.U[iBY]**2 + s.U[iBZ]**2)
    s.t, s.step = 0.0, 0
    E0, div = s.total_energy(), 0.0
    while s.t < T - 1e-12:
        s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), T - s.t)
        s.step_ssprk3(); s.t += s.dt; s.step += 1
        div = max(div, s.div_b_max())
    Wn = to_numpy(cons_to_prim(s.U, s.eos))
    I = (slice(ng, -ng),) * 3
    dV = to_numpy(s.dV)
    rho = Wn[iRHO][I] - 1.0
    thi = th[:, ng:-ng]
    xc = float(np.sum(rho * X[ng:-ng, ng:-ng] * dV) / np.sum(rho * dV))
    Bx = Wn[iBX][I] * np.cos(thi) - Wn[iBY][I] * np.sin(thi)
    By = Wn[iBX][I] * np.sin(thi) + Wn[iBY][I] * np.cos(thi)
    return {"xc": xc, "xc_exact": -0.3 + U * T, "dB": max(float(np.max(np.abs(Bx - B0))), float(np.max(np.abs(By)))) / B0,
            "dE": abs(s.total_energy() - E0 - float(s._acc_boundary)) / E0, "div": div, "steps": s.step}


def multigroup_test():
    """Multigroup FLD (aegis/radiation_multigroup.py). (1) One group spanning all frequencies is grey FLD: equal to
    fld.step to the solver tolerance. (2) Eight groups with frequency-independent opacity, optically thick: the sum
    over groups follows grey FLD (to the per-group limiter, 2e-3), gas + radiation energy conserved to round-off.
    (3) Kramers (nu^-3) opacity with a photo-ionization edge in a uniform box: the groups relax to the Planck
    spectrum a T^4 b_g(T) at the common final temperature, energy to round-off. (4) Decoupled groups (no absorption)
    in the diffusion limit: each Gaussian spreads with its own D_g = c / (3 rho kR_g), the group Rosseland means of
    the Kramers shape spanning three decades. (5) In the 2D solver with a group-opacity table from
    tools/generate_sesame.py (Al, bound-free edges): a hot spot radiates, total energy conserved to round-off."""
    import tempfile
    from aegis import fld, radiation_multigroup as mg
    print("\n--- Multigroup flux-limited diffusion ---"); sys.stdout.flush()
    ok = True

    def cfg(**kw):
        c = dict(nx=48, ny=48, x_max=1.0, y_max=1.0, t_end=1.0, cfl=0.3, mach=1.0, gamma=5 / 3, B_transverse=0.0,
                 bc_x_type="periodic", bc_y_type="periodic", diag_interval=10**9, snapshot_times=[], fld=True,
                 fld_c=10.0, fld_a=1.0, fld_advect=False, fld_tol=1e-12)
        c.update(kw)
        return Config(**c)

    def gas(n, p):
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        W = xp.zeros((NVAR, n, n))
        W[iRHO] = 1.0
        W[iPR] = xp.asarray(p(X, Y))
        return prim_to_cons(W, 5 / 3), X, Y
    bump = lambda X, Y, w: np.exp(-((X - .5)**2 + (Y - .5)**2) / w)
    h, per = (1 / 48, 1 / 48), (True, True)

    # (1) one group = grey
    U1, X, Y = gas(48, lambda X, Y: 1.0 + 2.0 * bump(X, Y, 0.01))
    U2 = U1.copy()
    E1 = xp.asarray(0.5 + 0.5 * bump(X, Y, 0.005))[None].copy()
    E2 = E1[0].copy()
    kw = dict(fld_kr0=5.0, fld_kp0=1.0)
    for _ in range(10):
        mg.step(E1, U1, 5 / 3, cfg(fld_groups=1, **kw), h, per, 2e-3, advect=False)
        fld.step(E2, U2, 5 / 3, cfg(**kw), h, per, 2e-3, advect=False)
    d1 = max(float(xp.max(xp.abs(E1[0] - E2)) / xp.max(E2)), float(xp.max(xp.abs(U1[EN] - U2[EN])) / xp.max(U2[EN])))
    good = d1 < 1e-8
    ok = ok and good
    print(f"  one group vs grey FLD, 10 steps: {d1:.1e}  [{'PASS' if good else 'FAIL'}]")

    # (2) eight grey groups, thick: sum over groups vs grey
    U1, X, Y = gas(48, lambda X, Y: 1.0 + 2.0 * bump(X, Y, 0.01))
    U2 = U1.copy()
    c8 = cfg(fld_groups=8, fld_kr0=50.0, fld_kp0=1.0)
    W = cons_to_prim(U1, 5 / 3)
    B, _ = mg.planck_groups(mg.edges_of(c8), W[iPR] / W[iRHO], 1.0)
    Er0 = xp.asarray(0.5 + 0.5 * bump(X, Y, 0.005))
    Eg = B / xp.sum(B, axis=0) * Er0[None]
    Egrey = Er0.copy()
    tot0 = float(xp.sum(U1[EN]) + xp.sum(Eg))
    for _ in range(10):
        mg.step(Eg, U1, 5 / 3, c8, h, per, 2e-3, advect=False)
        fld.step(Egrey, U2, 5 / 3, cfg(fld_kr0=50.0, fld_kp0=1.0), h, per, 2e-3, advect=False)
    d2 = float(xp.max(xp.abs(xp.sum(Eg, axis=0) - Egrey)) / xp.max(Egrey))
    c2_ = abs(float(xp.sum(U1[EN]) + xp.sum(Eg)) - tot0) / tot0
    good = d2 < 2e-3 and c2_ < 1e-14
    ok = ok and good
    print(f"  8 grey groups vs grey FLD (thick): sum of groups within {d2:.1e}, energy {c2_:.1e}  [{'PASS' if good else 'FAIL'}]")

    # (3) Kramers + edge, uniform box: relaxation to the Planck spectrum
    ck = cfg(nx=8, ny=8, fld_groups=12, fld_mg_opacity="kramers", fld_kp0=50.0, fld_kr0=5.0, fld_edge=3.0,
             fld_edge_jump=10.0, fld_group_emin=0.05, fld_group_emax=30.0)
    U, _, _ = gas(8, lambda X, Y: 2.0 + 0 * X)
    E = xp.zeros((12, 8, 8)) + 0.01
    tot0 = float(xp.sum(U[EN]) + xp.sum(E))
    for _ in range(300):
        mg.step(E, U, 5 / 3, ck, (1 / 8, 1 / 8), per, 0.05, advect=False)
    W = cons_to_prim(U, 5 / 3)
    Bg, _ = mg.planck_groups(mg.edges_of(ck), W[iPR] / W[iRHO], 1.0)
    d3 = float(xp.max(xp.abs(E - Bg)) / xp.max(Bg))
    c3_ = abs(float(xp.sum(U[EN]) + xp.sum(E)) - tot0) / tot0
    good = d3 < 1e-4 and c3_ < 1e-14
    ok = ok and good
    print(f"  Kramers + edge, 12 groups, uniform box: relaxed spectrum vs a T^4 b_g(T) {d3:.1e}, energy {c3_:.1e}  "
          f"[{'PASS' if good else 'FAIL'}]")

    # (4) decoupled groups in the diffusion limit vs analytic Gaussian spreading
    n4 = 96
    U, X, Y = gas(n4, lambda X, Y: 1.0 + 0 * X)
    s0 = 0.004
    E = xp.asarray(np.repeat((1e-8 + bump(X, Y, s0))[None], 4, 0))
    th = xp.ones((n4, n4))
    # the Kramers group opacities scale with the grey kP coefficient (1e4: every group optically thick);
    # fld_mg_absorption = 0 removes the exchange with the gas, so the groups only diffuse
    c4 = cfg(nx=n4, ny=n4, fld_groups=4, fld_mg_opacity="kramers", fld_kp0=1e4, fld_kr0=1.0,
             fld_group_emin=0.5, fld_group_emax=4.0, fld_picard=1, fld_a=0.0, fld_mg_absorption=0.0)
    _, kR = mg._kramers_means(mg.edges_of(c4), th, xp.full((n4, n4), 1e4), 0.0, 0.0)
    T4, nst = 0.02, 20
    for _ in range(nst):
        mg.step(E, U, 5 / 3, c4, (1 / n4, 1 / n4), per, T4 / nst, advect=False)
    errs, Ds = [], []
    for g in range(4):
        D = 10.0 / (3.0 * float(kR[g][0, 0]))
        amp = float(xp.max(E[g]))
        errs.append(abs(amp / (s0 / (s0 + 4 * D * T4)) - 1))
        Ds.append(D)
    good = max(errs) < 0.02 and Ds[-1] / Ds[0] > 100
    ok = ok and good
    print(f"  decoupled groups, diffusion limit: D_g = c/(3 rho kR_g) from {Ds[0]:.1e} to {Ds[-1]:.1e}; peak vs exact "
          f"Gaussian within {max(errs):.1e}  [{'PASS' if good else 'FAIL'}]")

    # (5) solver run with a generated group table
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
    import generate_sesame as gs
    with np.errstate(all="ignore"):
        gt = gs.group_table("al", [0.0, 10.0, 100.0, 1000.0, np.inf], nr=24, nt=24, pts=48)
    path = os.path.join(tempfile.mkdtemp(prefix="mg_"), "al_groups.npz")
    np.savez(path, **gt)
    # code units: rho in g/cc, theta in units of 1 eV, lengths in mm; opacity conversion via tab_* units
    c5 = Config(nx=32, ny=32, x_max=1.0, y_max=1.0, t_end=1.0, cfl=0.3, mach=1.0, gamma=5 / 3, B_transverse=0.0,
                bc_x_type="periodic", bc_y_type="periodic", diag_interval=10**9, snapshot_times=[], fld=True,
                fld_c=50.0, fld_a=1e-3, fld_kr0=1.0, fld_advect=True, fld_groups=4, fld_mg_table=path, tab_rho_unit=1000.0,
                tab_L_unit=1e-3, tab_T_unit=11604.518, fld_tol=1e-10)
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(c5)
    W = xp.zeros((NVAR,) + s.U.shape[1:])
    W[iRHO] = 0.01
    W[iPR] = 0.01 * (5.0 + 200.0 * xp.exp(-((s.X - 0.5)**2 + (s.Y - 0.5)**2) / 0.01))
    W[iVX] = 0.5
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    E0 = _energy(s) + s.radiation_energy()
    for _ in range(15):
        s.dt = s.compute_dt(cons_to_prim(s.U, s.eos)); s.step_ssprk3(); s.t += s.dt; s.step += 1
    c5_ = abs(_energy(s) + s.radiation_energy() - E0) / E0
    erad = s.radiation_energy()
    good = c5_ < 1e-13 and erad > 0 and bool(xp.all(xp.isfinite(s.Er)))
    ok = ok and good
    print(f"  2D solver, Al group table (4 groups, bound-free edges), hot spot, 15 steps: gas + radiation energy "
          f"{c5_:.1e}, radiation now {erad / E0:.1e} of the total  [{'PASS' if good else 'FAIL'}]")
    return ok


def experimental_test():
    """Bundled open data and the benchmark that runs the solver against it (aegis/experimental_suite.py).
    (1) The copper and nickel Hugoniot datasets (Marsh 1980, via LLNL BALSCD, MIT) load with their checksums; a
    tampered copy is refused. (2) A shot trace written as CSV loads back through load_shot. (3) Copper plate
    impacts at five particle velocities with the generated Cu table: the simulated shock speeds follow the table's
    own Hugoniot (the flow solver adds no bias, within 1.5%) and the measured Us(Up) within 4% RMS, with chi^2_nu
    reported for an assumed 2% measurement error."""
    import tempfile
    import shutil
    from aegis import experimental_suite as ex, validation as val
    print("\n--- Experimental data suite ---"); sys.stdout.flush()
    ok = True
    cu, ni = ex.load_dataset("copper_hugoniot"), ex.load_dataset("nickel_hugoniot")
    tmp = tempfile.mkdtemp(prefix="exp_")
    bad = os.path.join(tmp, "copper_marsh.csv")
    shutil.copy(os.path.join(ex.DATA_DIR, "copper_marsh.csv"), bad)
    with open(bad, "a") as fh:
        fh.write("8.930,5.000,1.000,44.650,0.0900,11.1,0.800,xx\n")
    saved = ex.DATA_DIR
    ex.DATA_DIR = tmp
    try:
        ex.load_dataset("copper_hugoniot")
        refused = False
    except ValueError:
        refused = True
    finally:
        ex.DATA_DIR = saved
    good = len(cu["Us_km_s"]) == 145 and len(ni["Us_km_s"]) == 19 and refused
    ok = ok and good
    print(f"  datasets: copper {len(cu['Us_km_s'])} shots, nickel {len(ni['Us_km_s'])} shots, checksums verified; a "
          f"tampered file refused: {refused}  [{'PASS' if good else 'FAIL'}]")
    t = np.linspace(0, 1, 50)
    tr = val.Trace("vis", t, np.sin(3 * t), 0.01, "km/s", "visar")
    path = os.path.join(tmp, "shot.csv")
    val.save_trace(path, tr)
    back = ex.load_shot(path, "visar", "km/s")
    d = float(np.max(np.abs(back.y - tr.y)))
    good = d < 1e-12 and back.kind == "visar"
    ok = ok and good
    print(f"  shot trace CSV round trip: {d:.1e}  [{'PASS' if good else 'FAIL'}]")
    r = ex.hugoniot_benchmark("copper_hugoniot", ups=[0.3, 0.8, 1.4, 2.0, 2.8], n=400,
                              out_png=os.path.join(tmp, "shot_benchmark_comparison.png"))
    good = r["sim_vs_table_max_rel"] < 0.015 and r["rms_rel"] < 0.04 and os.path.exists(os.path.join(tmp, "shot_benchmark_comparison.png"))
    ok = ok and good
    print(f"  copper plate impacts (5 Up values) vs {r['n_points']} Marsh shots: RMS {r['rms_km_s']:.3f} km/s "
          f"({100 * r['rms_rel']:.1f}%), chi2_nu {r['chi2_nu']:.2f} (sigma 2% assumed); simulated vs the table's own "
          f"Hugoniot within {100 * r['sim_vs_table_max_rel']:.2f}%  [{'PASS' if good else 'FAIL'}]")
    return ok


def autodiff_test():
    """Reverse-mode differentiable twin of the ideal-MHD step (aegis/autodiff.py).
    (1) Forward parity: 20 steps of an Orszag-Tang state (periodic, Cartesian) and of an imploding r-z shell
    with B_phi and B_z match MHDSolver (CUDA HLLD, fused MUSCL / EMF kernels) to round-off. (2) Conservation in
    the twin: periodic total energy to round-off, face div B at round-off, and the adjoint of the conserved
    energy carries its gradient back unchanged (dE_final/dp = dE_initial/dp). (3) Adjoint gradients against
    central finite differences with the same step sequence, for a shape parameter of the initial state, an
    amplitude and a boundary-drive current (r-z, 30 steps, objective: core B_z^2 plus a running B_phi^2)."""
    from aegis import autodiff as ad
    from aegis.config import iVZ, iBZ
    print("\n--- Differentiable MHD (adjoint) ---"); sys.stdout.flush()
    ok = True

    def solver(cyl):
        if cyl:
            cfg = Config(nx=24, ny=16, x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0, cfl=0.3, gamma=5 / 3,
                         geometry="cylindrical", bc_x_type="extrapolation", bc_y_type="periodic",
                         diag_interval=10**9, snapshot_times=[])
        else:
            cfg = Config(nx=24, ny=20, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, cfl=0.3, gamma=5 / 3,
                         bc_x_type="periodic", bc_y_type="periodic", diag_interval=10**9, snapshot_times=[])
        with contextlib.redirect_stdout(io.StringIO()):
            return MHDSolver(cfg)

    # (1) forward parity
    for cyl in (False, True):
        s = solver(cyl)
        X, Y = s.X, s.Y
        W = xp.zeros_like(s.U)
        B0 = 1 / np.sqrt(4 * np.pi)
        if cyl:
            shell = xp.exp(-((X - 1.2)**2) / 0.02)
            W[iRHO] = 1.0 + 4.0 * shell * (1 + 0.2 * xp.cos(2 * np.pi * Y))
            W[iVX] = -shell
            W[iPR] = 0.2
            W[iBZ] = 0.1 * X
            W[iBY] = 0.5
            W[iCLR] = (X > 1.1).astype(float)
        else:
            W[iRHO] = 25 / (36 * np.pi); W[iPR] = 5 / (12 * np.pi)
            W[iVX] = -xp.sin(2 * np.pi * Y); W[iVY] = xp.sin(2 * np.pi * X); W[iVZ] = 0.1 * xp.cos(2 * np.pi * Y)
            W[iCLR] = 0.5 + 0.5 * xp.sin(2 * np.pi * X)
        s.U[...] = prim_to_cons(W, s.cfg.gamma)
        if not cyl:
            s.set_vector_potential(lambda x, y: B0 * (np.cos(4 * np.pi * x) / (4 * np.pi) + np.cos(2 * np.pi * y) / (2 * np.pi)))
        s._ct_sync(); s.apply_bc(s.U); s._ghost_faces(s.Bf, s.U)
        sim = ad.DiffMHD(s, xp=np)
        st = sim.state_from_solver()
        E0 = float(ad.total_energy(sim, st).v)
        t = 0.0
        for n in range(20):
            s.dt = s.compute_dt(cons_to_prim(s.U, s.eos))
            s.step_ssprk3()
            st = sim.step(st, s.dt, t)
            t += s.dt
        ng, nx, ny = s.ng, s.cfg.nx, s.cfg.ny
        Uc = to_numpy(s.U)[:, ng:ng + nx, ng:ng + ny]
        Ut = np.array([a.v for a in st[:NVAR]])[:, ng:ng + nx, ng:ng + ny]
        scale = np.abs(Uc).max(axis=(1, 2), keepdims=True) + 1e-300
        cell = float(np.max(np.abs(Uc - Ut) / scale))
        face = max(float(np.max(np.abs(to_numpy(f) - b.v))) for f, b in zip(s.Bf, st[NVAR:]))
        good = cell < 1e-11 and face < 1e-12
        if cyl:
            print(f"  r-z shell, 20 steps: twin vs MHDSolver max relative cell difference {cell:.1e}, faces {face:.1e}"
                  f"  [{'PASS' if good else 'FAIL'}]")
        else:
            Bf = [b.v for b in st[NVAR:]]
            divb = float(np.max(np.abs((Bf[0][ng + 1:ng + nx + 1, ng:ng + ny] - Bf[0][ng:ng + nx, ng:ng + ny]) / s.cfg.dx
                                       + (Bf[1][ng:ng + nx, ng + 1:ng + ny + 1] - Bf[1][ng:ng + nx, ng:ng + ny]) / s.cfg.dy)))
            dE = abs(float(ad.total_energy(sim, st).v) - E0) / E0
            good = good and dE < 1e-13 and divb < 1e-11
            print(f"  Orszag-Tang, 20 steps: twin vs MHDSolver max relative cell difference {cell:.1e}, faces {face:.1e};"
                  f" twin energy change {dE:.1e}, max |div B| {divb:.1e}  [{'PASS' if good else 'FAIL'}]")
            ok = ok and good
            # (2) the adjoint of a conserved quantity: dE(T)/dp = dE(0)/dp
            def init_ot(p):
                def Wf(p):
                    Xn, Yn = to_numpy(s.X), to_numpy(s.Y)
                    return [25 / (36 * np.pi) * (1 + p[0] * ad.const(np.cos(2 * np.pi * Xn))),
                            -p[1] * ad.const(np.sin(2 * np.pi * Yn)), ad.const(np.sin(2 * np.pi * Xn)), 0.0,
                            0.0, 0.0, 0.2, 5 / (12 * np.pi), 0.0, 0.5, 0.0]
                return ad.cells_to_state(sim, Wf, p)
            Efin = lambda st_, t_, p_: ad.total_energy(sim, st_)
            pq = np.array([0.1, 1.0])
            _, gT, _ = sim.value_and_grad(init_ot, pq, n_steps=10, final_objective=Efin)
            with ad.record():
                pl = ad.leaf(pq)
                ad.backward(ad.total_energy(sim, init_ot(pl)))
            dg = float(np.max(np.abs(gT - pl.g)) / np.max(np.abs(pl.g)))
            good = dg < 1e-12
            print(f"  adjoint of the conserved energy over 10 steps: dE(T)/dp vs dE(0)/dp {dg:.1e}  [{'PASS' if good else 'FAIL'}]")
        ok = ok and good

    # (3) gradients against finite differences
    s = solver(True)
    sim = ad.DiffMHD(s, drive=lambda t, p: p[2] * (1.0 - np.exp(-t / 0.05)), xp=np)
    R = to_numpy(s.rc)[:, None] * np.ones((1, s.ny_tot))
    Z = np.ones((s.nx_tot, 1)) * to_numpy(s.y)[None, :]

    def Wf(p):
        prof = ad.exp(-((ad.const(R) - (1.2 + p[0] * ad.const(np.cos(2 * np.pi * Z))))**2) / 0.02)
        return [0.2 + 4.0 * prof, -p[1] * prof, 0.0, 0.0, 0.0, 0.5, 0.0, 0.2, 0.0, prof, 0.0]
    init = lambda p: ad.cells_to_state(sim, Wf, p)
    core = np.exp(-(R / 0.5)**2)[2:-2, 2:-2]
    final = lambda st_, t_, p_: ad.tsum(ad.interior(sim, st_[iBY])**2 * core)
    running = lambda st_, t_, p_: ad.tsum(ad.interior(sim, st_[iBZ])**2)
    p0 = np.array([0.05, 1.0, 0.3])
    J, g, info = sim.value_and_grad(init, p0, n_steps=30, final_objective=final, step_objective=running)
    errs = []
    for k in range(3):
        fd = ad.finite_difference(sim, init, p0, k, 1e-6, dts=info["dts"], final_objective=final, step_objective=running)
        errs.append(abs(fd - g[k]) / abs(g[k]))
    good = max(errs) < 1e-6
    ok = ok and good
    print(f"  r-z adjoint vs finite differences (30 steps): shape {errs[0]:.1e}, velocity {errs[1]:.1e}, drive current "
          f"{errs[2]:.1e}  [{'PASS' if good else 'FAIL'}]")
    return ok


def hybrid_pic_test():
    """Adaptive kinetic-fluid hybrid (aegis/hybrid_pic.py). (1) Free streaming: a density mode in a collisionless
    gas (every cell kinetic by its Knudsen number) decays as exp(-k^2 theta t^2 / 2), where the fluid would ring as
    a sound wave; the domain totals stay at round-off. (2) Boris push: one ion in a uniform field keeps |v| to
    round-off and gyrates at r_L = v / Omega. (3) Mixed run: Orszag-Tang with a low-density channel; cells switch
    to particles by Kn and r_L / dx and back (hysteresis), mass, momentum and energy of the domain conserved to
    round-off through every exchange, div B at round-off."""
    from aegis.hybrid_pic import HybridPIC
    from aegis.config import RHO, MX, MY, MZ
    print("\n--- Kinetic-fluid hybrid (PIC) ---"); sys.stdout.flush()
    ok = True

    def run(cfg, init, T, times=None):
        with contextlib.redirect_stdout(io.StringIO()):
            s = MHDSolver(cfg)
        init(s)
        t, out = 0.0, []
        for Tk in (times or [T]):
            while t < Tk - 1e-12:
                s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), Tk - t)
                s.step_ssprk3()
                t += s.dt
            out.append(to_numpy(s.U))
        return s, out

    def totals(s, U):
        ng = s.ng
        return np.array([U[v, ng:-ng, ng:-ng].sum() for v in (RHO, MX, MY, MZ, EN)])

    # (1) free streaming
    th, A, k = 1.0, 0.1, 2 * np.pi
    cfg = Config(nx=64, ny=4, x_min=0, x_max=1, y_min=0, y_max=1 / 16, cfl=0.3, gamma=5 / 3, bc_x_type="periodic",
                 bc_y_type="periodic", pic=True, pic_mfp0=1e3, pic_ppc=1024, diag_interval=10**9, snapshot_times=[])

    def init1(s):
        W = xp.zeros_like(s.U)
        W[iRHO] = 1 + A * xp.cos(k * s.X)
        W[iPR] = th * W[iRHO]
        s.U[...] = prim_to_cons(W, s.cfg.gamma)
        s._U0 = to_numpy(s.U)
    times = [0.1, 0.2, 0.3]
    s, Us = run(cfg, init1, None, times)
    ng = s.ng
    x = to_numpy(s.x[ng:-ng])
    amps = [2 * np.mean(U[RHO, ng:-ng, ng:-ng].mean(axis=1) * np.cos(k * x)) / A for U in Us]
    exact = [np.exp(-k * k * th * T * T / 2) for T in times]
    fluid = [np.cos(k * np.sqrt(5 / 3 * th) * T) for T in times]
    t0, t1 = totals(s, s._U0), totals(s, Us[-1])
    cons = float(np.max(np.abs(t1 - t0) / np.maximum(np.abs(t0), 1.0)))
    err = max(abs(a - e) for a, e in zip(amps, exact))
    good = err < 0.03 and cons < 1e-13 and int(s.pic.report()["kinetic_cells"]) == 64 * 4
    ok = ok and good
    print("  free streaming, mode amplitude / A at t = " + ", ".join(f"{T}: {a:+.3f} (exact {e:+.3f}, fluid {f:+.3f})"
          for T, a, e, f in zip(times, amps, exact, fluid)) + f"; totals {cons:.1e}  [{'PASS' if good else 'FAIL'}]")

    # (2) Boris gyration
    cfg2 = Config(nx=8, ny=8, x_min=0, x_max=10, y_min=0, y_max=10, gamma=5 / 3, bc_x_type="periodic",
                  bc_y_type="periodic", pic=True, pic_di=1.0, diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s2 = MHDSolver(cfg2)
    W2 = xp.zeros_like(s2.U)
    W2[iRHO], W2[iPR], W2[iBZ] = 1.0, 1.0, 2.0
    s2.U[...] = prim_to_cons(W2, 5 / 3)
    s2.apply_bc(s2.U)
    pc = HybridPIC(s2)
    pc.P = dict(x=xp.array([5.0]), y=xp.array([5.0]), vx=xp.array([1.0]), vy=xp.array([0.0]), vz=xp.array([0.3]),
                w=xp.array([1.0]))
    Om, dt = 2.0, 0.05
    xs = []
    for _ in range(int(round(2 * np.pi / Om / dt)) * 10):
        pc.push(dt, s2.U)
        xs.append((float(pc.P["x"][0]), float(pc.P["y"][0])))
    xs = np.array(xs)
    r = np.hypot(xs[:, 0] - xs[:, 0].mean(), xs[:, 1] - xs[:, 1].mean())
    dv = abs(float(pc.P["vx"][0]**2 + pc.P["vy"][0]**2) - 1.0)
    good = dv < 1e-13 and abs(r.mean() * Om - 1.0) < 3e-3
    ok = ok and good
    print(f"  Boris gyration, 10 orbits at Omega dt = 0.1: |v| change {dv:.1e}, r_L Omega / v = {r.mean() * Om:.4f}"
          f"  [{'PASS' if good else 'FAIL'}]")

    # (3) mixed fluid / kinetic run with switching
    cfg3 = Config(nx=48, ny=48, x_min=0, x_max=1, y_min=0, y_max=1, cfl=0.3, gamma=5 / 3, bc_x_type="periodic",
                  bc_y_type="periodic", pic=True, pic_mfp0=2e-4, pic_di=0.002, pic_ppc=64, diag_interval=10**9,
                  snapshot_times=[])
    B0 = 1 / np.sqrt(4 * np.pi)

    def init3(s):
        W = xp.zeros_like(s.U)
        W[iRHO] = 25 / (36 * np.pi) * (1 - 0.97 * xp.exp(-((s.X - 0.5)**2) / 0.004))
        W[iPR] = 5 / (12 * np.pi)
        W[iVX] = -xp.sin(2 * np.pi * s.Y)
        W[iVY] = xp.sin(2 * np.pi * s.X)
        s.U[...] = prim_to_cons(W, s.cfg.gamma)
        s.set_vector_potential(lambda x, y: B0 * (np.cos(4 * np.pi * x) / (4 * np.pi) + np.cos(2 * np.pi * y) / (2 * np.pi)))
        s._U0 = to_numpy(s.U)
    s, Us = run(cfg3, init3, 0.15)
    t0, t1 = totals(s, s._U0), totals(s, Us[-1])
    dm, de = abs(t1[0] - t0[0]) / t0[0], abs(t1[4] - t0[4]) / t0[4]
    dp = float(np.max(np.abs(t1[1:4] - t0[1:4])))
    divb = float(s.div_b_max())
    rep = s.pic.report()
    frac = rep["kinetic_cells"] / (48 * 48)
    good = (dm < 1e-13 and de < 1e-13 and dp < 1e-12 and divb < 1e-12 and rep["switched_on"] > 0
            and rep["switched_off"] > 0 and 0.02 < frac < 0.5)
    ok = ok and good
    print(f"  Orszag-Tang with a kinetic channel to t = 0.15: {100 * frac:.0f}% of cells kinetic ({rep['particles']} "
          f"particles), {rep['switched_on']} switches to particles, {rep['switched_off']} back; mass {dm:.1e}, "
          f"momentum {dp:.1e} (absolute), energy {de:.1e}, max |div B| {divb:.1e}  [{'PASS' if good else 'FAIL'}]")
    return ok


def fno_test():
    """Fourier neural operator predictor (aegis/fno_subgrid.py) and its guard in the multigroup step.
    (1) The spectral convolution with identity weights is the FFT low-pass of the kept modes. (2) Gradients of a
    loss through the whole network (tape of aegis/autodiff.py) agree with central differences. (3) Guard, with an
    untrained (random) network: forcing acceptance (huge tolerance) still conserves gas + radiation energy to
    round-off, because the step ends with the conservative update built from whatever solution it has; forcing
    the fallback (tiny tolerance) gives the exact solver's answer to its tolerance, and a prediction no better
    than E^n costs no extra Krylov iterations (the solve then starts from E^n)."""
    import tempfile
    from aegis import fno_subgrid as fs, autodiff as ad, radiation_multigroup as mg
    print("\n--- Fourier neural operator predictor ---"); sys.stdout.flush()
    ok = True
    rng = np.random.default_rng(0)
    m = fs.FNO(3, 2, width=4, modes=5, layers=2, xp=np)
    nx, ny = 16, 12
    x = rng.normal(size=(2, 4, nx, ny))
    wr = np.zeros((4, 4, 9, 5))
    for c in range(4):
        wr[c, c] = 1.0
    y = m._spectral(ad.const(x), ad.const(wr), ad.const(np.zeros_like(wr))).v
    kx, ky = np.fft.fftfreq(nx, 1 / nx), np.fft.fftfreq(ny, 1 / ny)
    yref = np.real(np.fft.ifft2(np.fft.fft2(x) * ((np.abs(kx)[:, None] <= 4) & (np.abs(ky)[None, :] <= 4))))
    d1 = float(np.abs(y - yref).max())
    xin, tgt = rng.normal(size=(2, 3, nx, ny)), rng.normal(size=(2, 2, nx, ny))

    def loss(P):
        d = m.forward(ad.const(xin), P) - ad.const(tgt)
        return ad.tsum(d * d)
    with ad.record():
        P = {k: ad.leaf(v) for k, v in m.params.items()}
        ad.backward(loss(P))
    gerr = 0.0
    for k in ("sr0", "si1", "pw0", "lift_w", "proj1_w", "proj2_b"):
        idx = tuple(int(rng.integers(0, n)) for n in m.params[k].shape)
        p0, h = m.params[k][idx], 1e-6
        m.params[k][idx] = p0 + h
        Lp = float(loss({q: ad.const(v) for q, v in m.params.items()}).v)
        m.params[k][idx] = p0 - h
        Lm = float(loss({q: ad.const(v) for q, v in m.params.items()}).v)
        m.params[k][idx] = p0
        fd = (Lp - Lm) / (2 * h)
        gerr = max(gerr, abs(fd - P[k].g[idx]) / max(abs(fd), 1e-3))
    good = d1 < 1e-13 and gerr < 1e-5
    ok = ok and good
    print(f"  spectral layer vs FFT low-pass {d1:.0e}; network gradients vs central differences {gerr:.0e}"
          f"  [{'PASS' if good else 'FAIL'}]")

    # (3) the guard in the multigroup step
    G, n = 4, 24
    path = os.path.join(tempfile.mkdtemp(prefix="fno_"), "random.npz")
    fs.FNO(2 * G + 3, G, width=8, modes=6, layers=2, xp=xp).save(path)

    def cfg(**kw):
        c = dict(nx=n, ny=n, x_max=1.0, y_max=1.0, gamma=5 / 3, bc_x_type="periodic", bc_y_type="periodic",
                 diag_interval=10**9, snapshot_times=[], fld=True, fld_c=10.0, fld_a=1.0, fld_advect=False, fld_tol=1e-10,
                 fld_groups=G, fld_mg_opacity="kramers", fld_kp0=20.0, fld_kr0=2.0, fld_edge=3.0, fld_edge_jump=10.0,
                 fld_group_emin=0.1, fld_group_emax=20.0)
        c.update(kw)
        return Config(**c)
    xg = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    W = xp.zeros((NVAR, n, n))
    W[iRHO] = xp.asarray(1.0 + 0.3 * np.cos(2 * np.pi * X))
    W[iPR] = W[iRHO] * xp.asarray(1.0 + 0.8 * np.exp(-((X - .5)**2 + (Y - .5)**2) / 0.02))
    U0 = prim_to_cons(W, 5 / 3)
    E0 = xp.full((G, n, n), 0.2)
    h, per, dt = (1 / n, 1 / n), (True, True), 0.01
    out = {}
    for name, c in (("exact", cfg()), ("accept", cfg(fld_fno_model=path, fld_fno_tol=1e300)),
                    ("fallback", cfg(fld_fno_model=path, fld_fno_tol=1e-300))):
        fs.Stats.reset()
        U, E = U0.copy(), E0.copy()
        tot0 = float(xp.sum(U[EN]) + xp.sum(E))
        _, _, its = mg.step(E, U, 5 / 3, c, h, per, dt, advect=False)
        out[name] = (E, abs(float(xp.sum(U[EN]) + xp.sum(E)) - tot0) / tot0, its, fs.Stats.accepted, fs.Stats.fallback)
    dfb = float(xp.max(xp.abs(out["fallback"][0] - out["exact"][0])) / xp.max(out["exact"][0]))
    good = (out["accept"][1] < 1e-14 and out["accept"][3] == 1 and out["fallback"][4] == 1
            and dfb < 1e-7 and out["fallback"][2] <= out["exact"][2] and out["exact"][1] < 1e-14)
    ok = ok and good
    print(f"  guard with a random network: forced acceptance (first Picard pass, no Krylov solve) keeps energy to "
          f"{out['accept'][1]:.0e}; forced fallback -> "
          f"{out['fallback'][2]} iterations (exact solve {out['exact'][2]}), differs from it by {dfb:.0e}, energy "
          f"{out['fallback'][1]:.0e}  [{'PASS' if good else 'FAIL'}]")
    return ok


def cad_test():
    """CAD ingestion and immersed walls (aegis/cad_importer.py). (1) STL: binary and ASCII round trip; volume
    fractions of a triangulated sphere converge to its enclosed volume with the ray sampling, along each ray axis,
    and the wall area vectors from the apertures close (sum to zero). (2) STEP through gmsh / OpenCASCADE (if
    installed): a cylinder written as STEP reads back with its volume. (3) 2D walls: a magnetized gas at rest
    around a ring-shaped wall stays at rest exactly; a blast sealed inside the ring keeps the cavity mass, the
    total mass and energy to round-off, div B 0; a planar shock reflecting off a wall at 45 and 30 degrees to the
    grid reaches the exact reflected pressure (first order near the wall). (4) 3D walls: a sphere in
    MHDSolver3D (B normal to some wall faces), rest preserved to round-off, a blast beside it conserves mass and energy, div B at round-off."""
    import tempfile
    from aegis import cad_importer as ci
    from aegis.solver3d import MHDSolver3D
    from aegis.config import RHO
    print("\n--- CAD ingestion and immersed walls ---"); sys.stdout.flush()
    ok = True
    tmp = tempfile.mkdtemp(prefix="cad_")

    # (1) STL and volume fractions
    tr = ci.sphere_mesh(0.37, (0.5, 0.5, 0.5), n=96)
    V = ci.mesh_volume(tr)
    ci.write_stl(os.path.join(tmp, "b.stl"), tr)
    ci.write_stl(os.path.join(tmp, "a.stl"), tr, binary=False)
    rb = float(np.abs(ci.read_stl(os.path.join(tmp, "b.stl")) - tr).max())
    ra = float(np.abs(ci.read_stl(os.path.join(tmp, "a.stl")) - tr).max())
    e = [ci.grid_edges(0, 1, 32)] * 3
    errs = []
    for sub in (1, 2, 4, 8):
        g = ci.Geometry(tr, e, sub)
        errs.append(max(abs(g.volume_fraction(ax).sum() / 32**3 - V) / V for ax in range(3)))
    closure = float(np.abs(ci.Geometry(tr, e, 4).wall_area_vectors().sum(axis=(1, 2, 3))).max())
    good = rb < 1e-6 and ra < 1e-12 and errs[-1] < 5e-5 and errs[0] > errs[1] > errs[2] and closure < 1e-12
    ok = ok and good
    print(f"  STL round trip binary {rb:.0e} (float32), ASCII {ra:.0e}; sphere volume on a 32^3 grid with 1, 2, 4, 8 "
          f"rays per cell side: " + ", ".join(f"{x:.1e}" for x in errs) + f"; wall area vectors close to {closure:.0e}"
          f"  [{'PASS' if good else 'FAIL'}]")

    # (2) STEP through gmsh
    try:
        import gmsh
        gmsh.initialize(interruptible=False)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("c")
        gmsh.model.occ.addCylinder(0.5, 0.5, 0.2, 0, 0, 0.6, 0.3)
        gmsh.model.occ.synchronize()
        step = os.path.join(tmp, "cyl.step")
        gmsh.write(step)
        gmsh.finalize()
        tc = ci.read_cad(step, mesh_size=0.03)
        Vc = ci.mesh_volume(tc)
        ac = ci.Geometry(tc, e, 4).volume_fraction().sum() / 32**3
        exact = np.pi * 0.3**2 * 0.6
        good = abs(Vc - exact) / exact < 1e-2 and abs(ac - Vc) / Vc < 2e-3
        print(f"  STEP cylinder via gmsh/OpenCASCADE: {len(tc)} triangles, volume {Vc:.5f} (exact {exact:.5f}), on the "
              f"grid {ac:.5f}  [{'PASS' if good else 'FAIL'}]")
    except ImportError:
        good = True
        print("  STEP / IGES: gmsh not installed, skipped (pip install gmsh)  [SKIP]")
    ok = ok and good

    def solver2d(n, bc, L):
        cfg = Config(nx=n, ny=n, x_min=0, x_max=L, y_min=0, y_max=L, cfl=0.3, gamma=5 / 3, bc_x_type=bc, bc_y_type=bc,
                     diag_interval=10**9, snapshot_times=[])
        with contextlib.redirect_stdout(io.StringIO()):
            return MHDSolver(cfg)

    def march(s, T):
        t = 0.0
        while t < T - 1e-12:
            s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), T - t)
            s.step_ssprk3()
            t += s.dt

    # (3) 2D walls: rest and a sealed blast in a ring
    ring = ci.corrugated_liner_mesh(r_in=0.25, thickness=0.12, length=1.0, eps=0.0, center=(0.5, 0.5), z0=-0.5,
                                    overhang=0.1)
    s = solver2d(64, "periodic", 1.0)
    g = ci.Geometry(ring, ci.solver_edges(s), sub=4)
    s.walls = ci.ImmersedWalls.from_geometry(s, g)
    W = xp.zeros_like(s.U)
    W[iRHO], W[iPR], W[iBZ] = 1.0, 1.0, 0.7
    s.U[...] = prim_to_cons(W, 5 / 3)
    march(s, 0.3)
    vrest = float(xp.max(xp.abs(s.U[1:3])))
    s = solver2d(64, "periodic", 1.0)
    s.walls = ci.ImmersedWalls.from_geometry(s, g)
    r = xp.sqrt((s.X - 0.5)**2 + (s.Y - 0.5)**2)
    W = xp.zeros_like(s.U)
    W[iRHO], W[iBZ] = 1.0, 0.3
    W[iPR] = xp.where(r < 0.1, 10.0, 0.1)
    s.U[...] = prim_to_cons(W, 5 / 3)
    ng = s.ng
    I = (slice(ng, -ng), slice(ng, -ng))
    cav = (r < 0.31)[I]
    q0 = [float(xp.sum(s.U[RHO][I][cav])), float(xp.sum(s.U[RHO][I])), float(xp.sum(s.U[EN][I]))]
    march(s, 0.3)
    q1 = [float(xp.sum(s.U[RHO][I][cav])), float(xp.sum(s.U[RHO][I])), float(xp.sum(s.U[EN][I]))]
    dq = [abs(a - b) / b for a, b in zip(q1, q0)]
    pout = float(xp.max(cons_to_prim(s.U, 5 / 3)[iPR][I][(r > 0.4)[I]]))
    good = vrest == 0.0 and max(dq) < 1e-13 and float(s.div_b_max()) < 1e-12 and abs(pout - 0.1) < 1e-12
    ok = ok and good
    print(f"  ring wall: gas at rest with B_z keeps max |rho v| = {vrest:.0e}; sealed blast: cavity mass {dq[0]:.1e}, total "
          f"mass {dq[1]:.1e}, energy {dq[2]:.1e}, outside untouched, div B {float(s.div_b_max()):.0e}  [{'PASS' if good else 'FAIL'}]")

    # oblique shock reflection
    g_, M = 5 / 3, 2.0
    p2 = 1 + 2 * g_ / (g_ + 1) * (M * M - 1)
    rho2 = (g_ + 1) * M * M / ((g_ - 1) * M * M + 2)
    us = M * np.sqrt(g_)
    u2 = us * (1 - 1 / rho2)
    ur = lambda p3: (p3 - p2) * np.sqrt(2 / ((g_ + 1) * rho2) / (p3 + (g_ - 1) / (g_ + 1) * p2))
    lo, hi = p2, 100 * p2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ur(mid) < u2 else (lo, mid)
    p3 = 0.5 * (lo + hi)
    res = {}
    for th_deg in (45.0, 30.0):
        n = 256
        s = solver2d(n, "extrapolation", 4.0)
        th = np.radians(th_deg)
        c = 2.4 + 0.37 / n
        Lb = 6.0
        box = np.array([[c, -Lb, -1], [c + Lb, -Lb, -1], [c + Lb, Lb, -1], [c, Lb, -1], [c, -Lb, 1], [c + Lb, -Lb, 1],
                        [c + Lb, Lb, 1], [c, Lb, 1]])
        F = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7],
             [2, 7, 3], [3, 7, 4], [3, 4, 0]]
        Rm = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        o = np.array([2.0, 2.0, 0.0])
        s.walls = ci.ImmersedWalls.from_geometry(s, ci.Geometry(((box - o) @ Rm.T + o)[np.array(F)], ci.solver_edges(s), 4))
        xi = (s.X - 2) * np.cos(th) + (s.Y - 2) * np.sin(th)
        eta = -(s.X - 2) * np.sin(th) + (s.Y - 2) * np.cos(th)
        W = xp.zeros_like(s.U)
        sh = xi < 0.0
        W[iRHO] = xp.where(sh, rho2, 1.0)
        W[iPR] = xp.where(sh, p2, 1.0)
        W[iVX] = xp.where(sh, u2 * np.cos(th), 0.0)
        W[iVY] = xp.where(sh, u2 * np.sin(th), 0.0)
        s.U[...] = prim_to_cons(W, 5 / 3)
        wall = c - 2.0
        march(s, wall / us + 0.12)
        band = (xi > wall - 0.12) & (xi < wall - 0.05) & (xp.abs(eta) < 0.3)
        res[th_deg] = abs(float(xp.mean(cons_to_prim(s.U, 5 / 3)[iPR][band])) - p3) / p3
    good = res[45.0] < 0.03 and res[30.0] < 0.1
    ok = ok and good
    print(f"  shock (M = 2) reflecting off an inclined wall, 256^2: pressure behind the reflected shock within "
          f"{100 * res[45.0]:.1f}% (45 deg) and {100 * res[30.0]:.1f}% (30 deg) of the exact {p3:.3f}  [{'PASS' if good else 'FAIL'}]")

    # (4) 3D: sphere wall
    cfg = Config(nx=24, ny=24, nz=24, x_max=1.0, y_max=1.0, z_max=1.0, cfl=0.3, gamma=5 / 3, bc_x_type="periodic",
                 bc_y_type="periodic", powell_source=False, diag_interval=10**9, snapshot_times=[])
    q = []
    for blast in (False, True):
        with contextlib.redirect_stdout(io.StringIO()):
            s = MHDSolver3D(cfg)
        s.walls = ci.ImmersedWalls.from_geometry(s, ci.Geometry(ci.sphere_mesh(0.2, (0.5, 0.5, 0.5), 48), ci.solver_edges(s), 4))
        W = xp.zeros((NVAR, s.nx_tot, s.ny_tot, s.nz_tot))
        W[iRHO], W[iPR], W[iBZ] = 1.0, 1.0, 0.5
        if blast:
            X = s.x[:, None, None]
            W[iPR] = 1.0 + 9.0 * xp.exp(-((X - 0.15)**2) / 0.003) * xp.ones_like(W[iPR])
        s.U = prim_to_cons(W, 5 / 3)
        ng = s.ng
        I3 = (slice(ng, -ng),) * 3
        m0, E0 = float(xp.sum(s.U[RHO][I3])), float(xp.sum(s.U[EN][I3]))
        t = 0.0
        while t < 0.15 - 1e-12:
            s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), 0.15 - t)
            s.step_ssprk3()
            t += s.dt
        q.append((float(xp.max(xp.abs(s.U[1:4]))), abs(float(xp.sum(s.U[RHO][I3])) - m0) / m0,
                  abs(float(xp.sum(s.U[EN][I3])) - E0) / E0, float(s.div_b_max())))
    good = q[0][0] < 1e-13 and q[1][1] < 1e-13 and q[1][2] < 1e-13 and q[1][3] < 1e-12
    ok = ok and good
    print(f"  3D sphere wall: rest kept (max |rho v| = {q[0][0]:.0e}); blast beside it: mass {q[1][1]:.1e}, energy "
          f"{q[1][2]:.1e}, div B {q[1][3]:.1e}  [{'PASS' if good else 'FAIL'}]")
    return ok


def monitor_test():
    """Run monitor (aegis/agent_monitor.py) and live viewer server (scripts/server.py). (1) Mode analysis: a
    synthetic interface with an m = 3 mode growing at a known rate is decomposed and the rate recovered; a flux
    history losing 30% raises flux_loss. (2) A 2D liner cross-section implosion with a seeded m = 4 corrugation and
    an undeclared 3% m = 3 drive asymmetry: the monitor reports the unexpected m = 3 growth; without the asymmetry
    it reports no unexpected mode. (3) The server on a free local port: the viewer page (WebGPU path), a frame with
    the density / temperature slices, contours and monitor data, the Server-Sent-Events stream, /status."""
    import importlib.util
    import json
    import urllib.request
    from aegis.agent_monitor import AgentMonitor, interface_modes
    print("\n--- Run monitor and live server ---"); sys.stdout.flush()
    ok = True
    # (1) synthetic
    th = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    mon = AgentMonitor(seeded_modes=(4,), growth_alert=1.5)
    g_true = 2.0
    for t in np.linspace(0, 1, 11):
        r = 1.0 + 0.01 * np.cos(4 * th) + 0.006 * np.exp(g_true * t) * np.cos(3 * th + 0.4)
        a = np.abs(interface_modes(r))
        mon.t.append(t); mon.modes.append(a / a[0]); mon.flux.append(1.0 - 0.3 * t); mon.unmag.append(np.nan)
        mon.energy.append(1.0); mon.divb.append(0.0)
        mon.check()
    g3 = mon.growth_rates()[3]
    kinds = {(a["kind"], a["mode"]) for a in mon.alerts}
    good = abs(g3 - g_true) / g_true < 1e-6 and ("unexpected_mode", 3) in kinds and ("flux_loss", 0) in kinds \
        and not any(k == "unexpected_mode" and m != 3 for k, m in kinds)
    ok = ok and good
    print(f"  synthetic interface: m = 3 growth rate {g3:.6f} (exact {g_true}); alerts {sorted(kinds)}"
          f"  [{'PASS' if good else 'FAIL'}]")
    # (2) the 2D preset, with and without the m = 3 drive asymmetry
    spec = importlib.util.spec_from_file_location("aegis_server", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                               "scripts", "server.py"))
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    found = {}
    for m3 in (0.03, 0.0):
        S = srv.Session("liner2d", every=10, n=96, drive_m3=m3)
        S.run()
        found[m3] = [(a["kind"], a["mode"], a["values"]["g"]) for a in S.mon.alerts if a["kind"] == "unexpected_mode"]
    hit = [x for x in found[0.03] if x[1] == 3]
    good = bool(hit) and not found[0.0] and S.error is None
    ok = ok and good
    print(f"  liner cross-section with a 3% m = 3 drive asymmetry: unexpected m = 3 reported (growth "
          f"{hit[0][2]:.1f} per unit time) " if hit else "  m = 3 not reported ", end="")
    print(f"; without it: {len(found[0.0])} unexpected-mode alerts  [{'PASS' if good else 'FAIL'}]")
    # (3) server
    S = srv.Session("liner2d", every=5, n=48)
    httpd = srv.serve(S, "127.0.0.1", 0)
    port = httpd.server_address[1]
    import threading
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    S.start(max_steps=20).join(timeout=300)
    base = f"http://127.0.0.1:{port}"
    page = urllib.request.urlopen(base + "/", timeout=30).read().decode()
    fr = json.loads(urllib.request.urlopen(base + "/frame", timeout=30).read())
    st = json.loads(urllib.request.urlopen(base + "/status", timeout=30).read())
    ev = urllib.request.urlopen(base + "/events", timeout=30)
    first = b""
    while not first.startswith(b"data: "):
        first = ev.readline()
    evf = json.loads(first[6:])
    ev.close()
    httpd.shutdown()
    import base64
    npx = len(base64.b64decode(fr["rho"]["data"]))
    good = ("navigator.gpu" in page and "@fragment" in page and npx == 48 * 48 and len(fr["contours"]) > 0
            and "modes" in fr["monitor"] and st["error"] is None and st["step"] == 20 and evf["step"] == fr["step"])
    ok = ok and good
    print(f"  server on port {port}: page with the WebGPU renderer, frame at step {fr['step']} ({npx} px slice, "
          f"{len(fr['contours'])} contour lines), SSE frame at step {evf['step']}, status {st}  [{'PASS' if good else 'FAIL'}]")
    return ok


def gmg_test():
    """Geometric multigrid preconditioner for the FLD solves (aegis/radiation_gmg.py). Stiff grey FLD (D dt / dx^2
    ~ 20-80) on 64^2 and 128^2 and 6-group Kramers multigroup on 64^2, three steps each: the multigrid
    V-cycle needs far fewer Krylov iterations than Jacobi, about the same number at both grid sizes (O(N) work),
    the solutions agree to the solver tolerance, and gas + radiation energy stays at round-off."""
    from aegis import fld, radiation_multigroup as mgr
    print("\n--- Multigrid preconditioner for FLD ---"); sys.stdout.flush()
    ok = True
    rows = {}
    for groups, n in ((0, 64), (0, 128), (6, 64)):
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        res = {}
        for pc in ("jacobi", "mg"):
            kw = dict(nx=n, ny=n, x_max=1.0, y_max=1.0, gamma=5 / 3, bc_x_type="periodic", bc_y_type="periodic",
                      diag_interval=10**9, snapshot_times=[], fld=True, fld_c=10.0, fld_a=1.0, fld_advect=False,
                      fld_tol=1e-10, fld_kr0=2.0, fld_kp0=1.0, fld_precond=pc, fld_maxit=20000)
            if groups:
                kw.update(fld_groups=groups, fld_mg_opacity="kramers", fld_kp0=20.0, fld_edge=3.0, fld_edge_jump=10.0,
                          fld_group_emin=0.1, fld_group_emax=20.0)
            cfg = Config(**kw)
            W = xp.zeros((NVAR, n, n))
            W[iRHO] = xp.asarray(1 + 0.5 * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y))
            W[iPR] = W[iRHO] * xp.asarray(1 + 2 * np.exp(-((X - .5)**2 + (Y - .5)**2) / 0.01))
            U = prim_to_cons(W, 5 / 3)
            E = xp.asarray(0.5 + 0.5 * np.exp(-((X - .3)**2 + (Y - .6)**2) / 0.005))
            if groups:
                E = xp.stack([E / groups] * groups)
            tot0 = float(xp.sum(U[EN]) + xp.sum(E))
            its = 0
            for _ in range(3):
                f = mgr.step if groups else fld.step
                its += f(E, U, 5 / 3, cfg, (1 / n, 1 / n), (True, True), 0.01, advect=False)[2]
            res[pc] = (its, E, abs(float(xp.sum(U[EN]) + xp.sum(E)) - tot0) / tot0)
        d = float(xp.max(xp.abs(res["mg"][1] - res["jacobi"][1])) / xp.max(res["jacobi"][1]))
        rows[(groups, n)] = (res["jacobi"][0], res["mg"][0], d, max(res["jacobi"][2], res["mg"][2]))
    good = all(j >= 5 * m and d < 1e-6 and e < 1e-14 for j, m, d, e in rows.values()) \
        and rows[(0, 128)][1] <= 1.3 * rows[(0, 64)][1]
    ok = ok and good
    print("  Krylov iterations over 3 steps, Jacobi -> multigrid: " + "; ".join(
        f"{'grey' if g == 0 else f'{g} groups'} {n}^2: {j} -> {m} (x{j / m:.0f}), solutions within {d:.0e}, energy {e:.0e}"
        for (g, n), (j, m, d, e) in rows.items()) + f"  [{'PASS' if good else 'FAIL'}]")
    return ok


def cyl_filter_test():
    """Near-axis azimuthal Fourier filter in 3D cylindrical geometry (solver3d, cyl_filter). (1) The filter keeps
    modes m <= m_crit(r) exactly (m <= 2 everywhere) and removes the rest on the filtered rings, leaving outer rings
    alone. (2) The blob carried across the axis by a uniform flow (cylindrical3d_test): with the filter the time
    step is set by dr as in r-z (many times fewer steps), the centroid and field error agree with the unfiltered
    run, energy and div B stay at round-off."""
    from aegis.solver3d import MHDSolver3D
    print("\n--- 3D cylindrical: near-axis azimuthal filter ---"); sys.stdout.flush()
    ok = True
    c = Config(nx=32, ny=64, nz=4, x_min=0.0, x_max=1.0, y_min=0.0, y_max=2 * np.pi, z_min=0.0, z_max=0.125, cfl=0.3,
               gamma=5 / 3, geometry="cylindrical", div_b="ct", bc_x_type="extrapolation", bc_y_type="periodic",
               powell_source=False, diag_interval=10**9, snapshot_times=[], cyl_filter=True)
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver3D(c)
    az = s._az_setup()
    th = (np.arange(64) + 0.5) * 2 * np.pi / 64
    A = np.zeros((s.nx_tot, 64 + 2 * s.ng, 3))
    low = 1.0 + 0.3 * np.cos(th) + 0.2 * np.sin(2 * th)
    A[:, s.ng:-s.ng, :] = (low + 0.1 * np.cos(10 * th))[None, :, None]
    Ad = xp.asarray(A)
    rows = [i for i in az["rows_c"] if s.ng <= i < s.ng + 32]
    s._az_apply(Ad, rows, az["mask_c"], slice(s.ng, s.ng + 64))
    An = to_numpy(Ad)[:, s.ng:-s.ng, 0]
    inner = [i for i in rows if az["mc"][i] < 10]
    outer = [i for i in range(s.ng, s.ng + 32) if i not in rows]
    e_in = max(float(np.max(np.abs(An[i] - low))) for i in inner)
    e_out = max(float(np.max(np.abs(An[i] - low - 0.1 * np.cos(10 * th)))) for i in outer)
    good = e_in < 1e-13 and e_out < 1e-13 and len(inner) > 0 and len(outer) > 0
    ok = ok and good
    print(f"  filter: {len(rows)} of 32 rings filtered (r dtheta < dr); m = 10 removed on {len(inner)} inner rings, "
          f"m <= 2 kept to {e_in:.0e}; outer rings untouched ({e_out:.0e})  [{'PASS' if good else 'FAIL'}]")
    ra, rb = (_cyl3d_blob(32, 64, T=0.25, cyl_filter=f) for f in (False, True))
    na, nb = ra["steps"], rb["steps"]
    good = (na >= 5 * nb and abs(ra["xc"] - rb["xc"]) < 2e-3 and abs(ra["dB"] - rb["dB"]) < 2e-3
            and rb["dE"] < 1e-13 and rb["div"] < 1e-12)
    ok = ok and good
    print(f"  blob across the axis (32 x 64, t = 0.25): {na} steps without the filter, {nb} with it (x{na / nb:.0f}); "
          f"centroid {ra['xc']:.4f} vs {rb['xc']:.4f}, field error {ra['dB']:.4f} vs {rb['dB']:.4f}; with the filter "
          f"energy {rb['dE']:.0e}, div B {rb['div']:.0e}  [{'PASS' if good else 'FAIL'}]")
    return ok


def _mode_run(cfg, init, T, var_fn, k):
    """march a 1D-like periodic run to T, recording the complex amplitude of the k Fourier mode of var_fn(W) along
    x; returns (measured omega from the phase slope, final / initial amplitude, steps, dt limiter)"""
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    W = xp.zeros((NVAR,) + s.U.shape[1:])
    init(W, s.X)
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    ng = s.ng
    x = to_numpy(s.x[ng:-ng])

    def amp():
        q = to_numpy(var_fn(cons_to_prim(s.U, s.eos))[ng:-ng, ng])
        return np.sum(q * np.exp(-1j * k * x)) / x.size
    ts, a = [0.0], [amp()]
    while s.t < T - 1e-12:
        s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), T - s.t)
        s.step_ssprk3(); s.t += s.dt; s.step += 1
        ts.append(s.t); a.append(amp())
    a = np.array(a)
    return -np.polyfit(ts, np.unwrap(np.angle(a)), 1)[0], abs(a[-1]) / abs(a[0]), s.step, s._dt_limiter, s


def hall_test():
    """Generalized Ohm's law (aegis/hall.py). (1) Circularly polarized waves along B_0 with the Hall term
    (d_i k = 1): the whistler and ion-cyclotron branches, omega = k (sqrt(1 + (k d_i/2)^2) -+ k d_i/2), measured from
    the phase of B_y + i B_z, converge at second order. (2) Anisotropic resistivity: a B_z ripple decays at
    eta_par k^2 when its current is along B_0 and at 1.96 eta_par k^2 when across it. (3) Hall + anisotropic
    resistivity in a periodic sheared, magnetized box: total energy conserved to round-off, div B at round-off."""
    print("\n--- Hall term and anisotropic resistivity ---"); sys.stdout.flush()
    ok = True
    k = 2 * np.pi
    di = 1 / k
    base = dict(ny=4, x_max=1.0, t_end=1.0, cfl=0.3, mach=1.0, gamma=5 / 3, B_transverse=1.0, bc_x_type="periodic",
                bc_y_type="periodic", powell_source=False, diag_interval=10**9, snapshot_times=[])
    for branch in ("whistler", "ion cyclotron"):
        a = k * di / 2
        w = -k * (np.sqrt(1 + a * a) + a) if branch == "whistler" else k * (np.sqrt(1 + a * a) - a)
        errs = []
        for N in (32, 64):
            def init(W, X, w=w):
                W[iRHO], W[iPR], W[iBX] = 1.0, 10.0, 1.0
                W[iBY], W[iBZ] = 1e-4 * xp.cos(k * X), 1e-4 * xp.sin(k * X)
                W[iVY], W[iVZ] = -(k / w) * W[iBY], -(k / w) * W[iBZ]
            cfg = Config(nx=N, y_max=4.0 / N, hall_di=di, **base)
            wm, amp, n, lim, _ = _mode_run(cfg, init, 2 * np.pi / abs(w), lambda W: W[iBY] + 1j * W[iBZ], k)
            errs.append(abs(wm - w) / abs(w))
        good = errs[0] < 1e-2 and errs[1] < 2.5e-3 and errs[0] / errs[1] > 3.0
        ok = ok and good
        print(f"  {branch:13s} (d_i k = 1): omega exact {w:+.4f}; relative error N=32 {errs[0]:.1e}, N=64 {errs[1]:.1e} "
              f"(order {np.log2(errs[0] / errs[1]):.1f})  [{'PASS' if good else 'FAIL'}]")

    eta, ratio = 0.01, 1.96
    rates = {}
    # the ripple varies along x, across B_0 in both cases (k . B_0 = 0: no Alfven wave, which resistivity would damp
    # at half the rate): B_0 along y with a B_z ripple (J along y, parallel), B_0 along z with a B_z ripple held in
    # total pressure balance (J along y, perpendicular)
    for name, B0 in (("parallel", iBY), ("perpendicular", iBZ)):
        def init(W, X, B0=B0):
            W[iRHO], W[iPR], W[B0] = 1.0, 100.0, 1.0
            W[iBZ] = W[iBZ] + 1e-3 * xp.sin(k * X)
            if B0 == iBZ:                            # p + B^2/2 uniform
                W[iPR] = 100.0 - 0.5 * (W[iBZ]**2 - 1.0)
        cfg = Config(nx=64, y_max=4.0 / 64, eta_resistivity=eta, eta_perp_ratio=ratio, **base)
        T = 1.0
        _, amp, _, _, _ = _mode_run(cfg, init, T, lambda W: W[iBZ], k)
        rates[name] = -np.log(amp) / T
    exp_par, exp_perp = eta * k * k, ratio * eta * k * k
    e1, e2 = abs(rates["parallel"] / exp_par - 1), abs(rates["perpendicular"] / exp_perp - 1)
    good = e1 < 0.01 and e2 < 0.01
    ok = ok and good
    print(f"  anisotropic resistivity (eta_perp = 1.96 eta_par): B_z ripple decay rate with J along B "
          f"{rates['parallel']:.4f} (eta_par k^2 = {exp_par:.4f}), across B {rates['perpendicular']:.4f} "
          f"(eta_perp k^2 = {exp_perp:.4f})  [{'PASS' if good else 'FAIL'}]")

    cfg = Config(nx=48, ny=48, x_max=1.0, y_max=1.0, t_end=1.0, cfl=0.3, mach=1.0, gamma=5 / 3, B_transverse=1.0,
                 bc_x_type="periodic", bc_y_type="periodic", powell_source=False, hall_di=0.05, eta_resistivity=2e-3,
                 eta_perp_ratio=1.96, diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    W = xp.zeros((NVAR,) + s.U.shape[1:])
    X, Y = s.X, s.Y
    W[iRHO], W[iPR] = 1.0 + 0.2 * xp.sin(2 * np.pi * X), 2.0
    W[iVX], W[iVY] = 0.3 * xp.sin(2 * np.pi * Y), 0.2 * xp.cos(2 * np.pi * X)
    W[iBX], W[iBY], W[iBZ] = 0.5 + 0.3 * xp.sin(2 * np.pi * Y), 0.4 * xp.cos(4 * np.pi * X), 0.6 + 0.2 * xp.sin(2 * np.pi * (X + Y))
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    E0 = _energy(s)
    div = 0.0
    for _ in range(60):
        s.dt = s.compute_dt(cons_to_prim(s.U, s.eos)); s.step_ssprk3(); s.t += s.dt; s.step += 1
        div = max(div, s.div_b_max())
    dE = abs(_energy(s) - E0) / abs(E0)
    good = dE < 1e-13 and div < 1e-12
    ok = ok and good
    print(f"  Hall + anisotropic resistivity, periodic box, 60 steps ({s._dt_limiter}-limited): energy {dE:.1e}, "
          f"div B {div:.1e}  [{'PASS' if good else 'FAIL'}]")
    return ok


def ion_transport_test():
    """Ion Braginskii transport (aegis/ions.py). (1) Gyroviscosity: a fast magnetosonic wave across B = B_z has the
    finite-Larmor-radius dispersion omega^2 = k^2 c_f^2 + (eta_3 k^2 / rho)^2, eta_3 = p_i d_i / (2 B) (here
    the FLR term doubles omega^2), converging at second order; without it omega = k c_f. (2) Ion conduction
    (two-temperature): a T_i ripple decays at D k^2, D = k_i (1 + Z)(gamma - 1) / rho, and at D k^2 / (1 + x_i^2) across
    a field, with the electron entropy untouched and the energy conserved; (3) both on in a full run: energy to
    round-off."""
    from aegis.physics import conduct
    from aegis import twotemp
    print("\n--- Ion gyroviscosity and ion conduction ---"); sys.stdout.flush()
    ok = True
    k, g = 2 * np.pi, 5 / 3
    base = dict(ny=4, x_max=1.0, t_end=1.0, cfl=0.3, mach=1.0, gamma=g, B_transverse=1.0, bc_x_type="periodic",
                bc_y_type="periodic", powell_source=False, diag_interval=10**9, snapshot_times=[])
    rho, p, B0 = 1.0, 1.0, 1.0
    cs2, cf2 = g * p / rho, g * p / rho + B0**2 / rho
    for di in (0.0, 1.04):
        eta3 = (p / 2) * di / (2 * B0)                       # p_i = p / (1 + Z) at T_e = T_i
        w = k * np.sqrt(cf2 + (eta3 * k / rho)**2)
        errs = []
        for N in (32, 64):
            def init(W, X, w=w, eta3=eta3):
                e = 1e-4
                W[iVX] = e * xp.cos(k * X)
                W[iVY] = -(eta3 * k * k / (w * rho)) * e * xp.sin(k * X)
                drho = rho * (k / w) * e * xp.cos(k * X)
                W[iRHO], W[iPR], W[iBZ] = rho + drho, p + cs2 * drho, B0 * (1 + drho / rho)
            cfg = Config(nx=N, y_max=4.0 / N, gyro_di=di, **base)
            wm, amp, n, lim, _ = _mode_run(cfg, init, 2 * np.pi / w, lambda W: W[iVX], k)
            errs.append(abs(wm - w) / w)
        good = errs[1] < 5e-3 and (di == 0 or errs[0] / errs[1] > 3.0)
        ok = ok and good
        print(f"  fast wave across B, gyro d_i = {di:.2f}: omega exact {w:.4f} (no FLR {k * np.sqrt(cf2):.4f}); relative "
              f"error N=32 {errs[0]:.1e}, N=64 {errs[1]:.1e}  [{'PASS' if good else 'FAIL'}]")

    kap, Z = 0.01, 1.0
    rates, ent, econs = {}, 0.0, 0.0
    for name, x0, Bkey in (("unmagnetized", 0.0, None), ("across B (x_i = 3)", 3.0, iBY)):
        cfg = Config(nx=64, y_max=4.0 / 64, two_temp=True, z_ion=Z, cond_kappa_i0=kap, cond_x0_i=x0, **base)
        with contextlib.redirect_stdout(io.StringIO()):
            s = MHDSolver(cfg)
        W = xp.zeros((NVAR,) + s.U.shape[1:])
        W[iRHO], W[iPR] = 1.0, 0.5 * (1.0 + 1.0)            # p_e = 0.5 (theta_e = 1) set below, p_i = 0.5
        if Bkey is not None:
            W[Bkey] = 1.0
        U = prim_to_cons(W, s.eos)
        twotemp.set_pe(U, xp.full(U[0].shape, 0.5), g)
        W[iPR] = 1.0 + 0.5 * 1e-3 * xp.cos(k * s.X)          # theta_i = 1 + 1e-3 cos kx
        U2 = prim_to_cons(W, s.eos)
        U2[RHOSE] = U[RHOSE]
        ng = s.ng
        Ui = U2[:, ng:-ng, ng:-ng].copy()
        x = to_numpy(s.x[ng:-ng])
        se0, E0 = Ui[RHOSE].copy(), float(xp.sum(Ui[EN]))
        ti = lambda: to_numpy(twotemp.temperatures(Ui, g, Z)[1][:, 0])
        a0 = abs(np.sum(ti() * np.exp(-1j * k * x)))
        T, nsteps = 1.0, 50
        for _ in range(nsteps):
            conduct(Ui, g, kap, x0, cfg.cond_tracer_max, (cfg.dx, cfg.dy), (True, True), T / nsteps, perp="simple",
                    ion_Z=Z)
        rates[name] = -np.log(abs(np.sum(ti() * np.exp(-1j * k * x))) / a0) / T
        ent = max(ent, float(xp.max(xp.abs(Ui[RHOSE] - se0))))
        econs = max(econs, abs(float(xp.sum(Ui[EN])) - E0) / abs(E0))
    D = kap * (1 + Z) * (g - 1) / 1.0
    e1 = abs(rates["unmagnetized"] / (D * k * k) - 1)
    e2 = abs(rates["across B (x_i = 3)"] / (D * k * k / 10.0) - 1)
    good = e1 < 0.01 and e2 < 0.01 and ent == 0.0 and econs < 1e-14
    ok = ok and good
    print(f"  ion conduction: T_i ripple decay {rates['unmagnetized']:.4f} (D k^2 = {D * k * k:.4f}), across B with "
          f"x_i = 3: {rates['across B (x_i = 3)']:.4f} (D k^2 / 10 = {D * k * k / 10:.4f}); electron entropy change "
          f"{ent:.1e}, energy {econs:.1e}  [{'PASS' if good else 'FAIL'}]")

    cfg = Config(nx=48, ny=48, x_max=1.0, y_max=1.0, t_end=1.0, cfl=0.3, mach=1.0, gamma=g, B_transverse=1.0,
                 bc_x_type="periodic", bc_y_type="periodic", powell_source=False, two_temp=True, cond_kappa_i0=2e-3,
                 cond_x0_i=1.0, gyro_di=0.3, diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    W = xp.zeros((NVAR,) + s.U.shape[1:])
    X, Y = s.X, s.Y
    W[iRHO], W[iPR] = 1.0 + 0.2 * xp.sin(2 * np.pi * X), 1.0 + 0.3 * xp.cos(2 * np.pi * Y)
    W[iVX], W[iVY] = 0.2 * xp.sin(2 * np.pi * Y), 0.2 * xp.cos(2 * np.pi * X)
    W[iBX], W[iBZ] = 0.3 * xp.sin(2 * np.pi * Y), 1.0
    s.U = prim_to_cons(W, s.eos); s.t = 0.0; s.step = 0
    E0 = _energy(s)
    for _ in range(40):
        s.dt = s.compute_dt(cons_to_prim(s.U, s.eos)); s.step_ssprk3(); s.t += s.dt; s.step += 1
    dE = abs(_energy(s) - E0) / abs(E0)
    good = dE < 1e-13 and s.div_b_max() < 1e-12
    ok = ok and good
    print(f"  gyroviscosity + ion conduction, two-temperature periodic box, 40 steps: energy {dE:.1e}, div B "
          f"{s.div_b_max():.1e}  [{'PASS' if good else 'FAIL'}]")
    return ok


def sesame_test():
    """Material tables from tools/generate_sesame.py (Al, Cu, D-T, Au, Pb; free-energy model, no LANL data):
    (1) every table loads into the solver's tabular EOS (p and e increasing in T) and Al survives a SESAME ASCII
    (301 / 502 / 505) round trip; (2) metals: p(rho0, 300 K) = 0 and the ambient sound speed at the measured bulk
    value (both are fitted: the cold reference density and K0); (3) independently, the principal Hugoniot within 5%
    of the measured Us-up fits at up = 1, 2, 3 km/s;
    (4) fully ionized hot plasma (D-T, Al) gives p = (1 + Z) rho k T / m within 1%, and kappa_R its Thomson floor;
    (5) thermodynamic consistency de/drho|_T = (p - T dp/dT|_rho) / rho^2 at random states; (6) the tabular EOS
    p(rho, e) against the model at random in-table states."""
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
    import generate_sesame as gs
    from aegis import tabular
    from aegis.eos import Material
    print("\n--- Generated material tables (tools/generate_sesame.py) ---"); sys.stdout.flush()
    tmp = tempfile.mkdtemp(prefix="ses_")
    ok = True
    rng = np.random.default_rng(5)
    with np.errstate(all="ignore"):
        loaded, adj = [], 0.0
        for name in gs.MATERIALS:
            tab = gs.table(name, 50, 56)
            adj = max(adj, tab.pop("adjust_max"))
            gs.write(name, tab, tmp, sesame=(name == "al"))
            d = tabular.load(os.path.join(tmp, f"{name}.npz"))
            fb = Material("stiffened", 3.0, 1.0)
            tabular.TabularMaterial(d, 1000.0, 1000.0, fb)
            loaded.append(name)
        back = tabular.load(os.path.join(tmp, "al.ses"))
        ref = np.load(os.path.join(tmp, "al.npz"))
        rt = max(float(np.max(np.abs(back[k] - ref[k]) / np.max(np.abs(ref[k])))) for k in ("p", "e", "kappa_R"))
        good = len(loaded) == 5 and rt < 1e-12 and adj < 1e-4
        ok = ok and good
        print(f"  tables {', '.join(loaded)} load into the tabular EOS (largest monotonicity fix {adj:.1e} of a row); "
              f"Al SESAME ASCII round trip {rt:.1e}  [{'PASS' if good else 'FAIL'}]")

        rows, worst_p, worst_c, worst_h = [], 0.0, 0.0, 0.0
        for name in ("al", "cu", "au", "pb"):
            m = gs.MATERIALS[name]
            r0, T0 = np.array([m["rho0"]]), np.array([300.0])
            p0, _, c2 = gs.thermo(m, r0, T0)
            c0, sl = m["hug"]
            H = gs.hugoniot(name, [1.0, 2.0, 3.0])
            dh = float(np.max(np.abs(H[:, 0] - (c0 + sl * np.array([1.0, 2.0, 3.0]))) / (c0 + sl * np.array([1.0, 2.0, 3.0]))))
            worst_p = max(worst_p, abs(p0[0]) / m["K0"])
            worst_c = max(worst_c, abs(np.sqrt(c2[0]) / 1e3 / c0 - 1))
            worst_h = max(worst_h, dh)
            rows.append(f"{name} c_s {np.sqrt(c2[0]) / 1e3:.2f} ({c0}) Us(2) {H[1, 0]:.2f} ({c0 + 2 * sl:.2f})")
        good = worst_p < 1e-9 and worst_c < 1e-3 and worst_h < 0.05
        ok = ok and good
        print(f"  ambient and Hugoniot (km/s, measured in brackets): {'; '.join(rows)}; |p(rho0, 300 K)| / K0 "
              f"{worst_p:.0e}, sound speed (fitted) within {100 * worst_c:.1f}%, Hugoniot (not fitted) within {100 * worst_h:.1f}%  "
              f"[{'PASS' if good else 'FAIL'}]")

        lim = []
        for name, rho, T in (("dt", 1.0, 1e7), ("al", 1.0, 1e8)):
            m = gs.MATERIALS[name]
            p, _ = gs.eos(m, np.array([rho]), np.array([T]))
            ideal = (1 + m["Z"]) * rho * gs.KB * T / (m["A"] * gs.MU)
            kR, _ = gs.opacities(m, np.array([1e-3]), np.array([1e9]))
            th = 6.6524587e-29 * m["Z"] / (m["A"] * gs.MU)
            lim.append((name, p[0] / ideal, kR[0] / th))
        good = all(abs(a - 1) < 0.01 and abs(b - 1) < 0.02 for _, a, b in lim)
        ok = ok and good
        print(f"  fully ionized limit: p / ((1 + Z) rho k T / m) " + ", ".join(f"{n} {a:.4f}" for n, a, _ in lim)
              + "; kappa_R / Thomson " + ", ".join(f"{n} {b:.4f}" for n, _, b in lim) + f"  [{'PASS' if good else 'FAIL'}]")

        res = 0.0
        for name in gs.MATERIALS:
            m = gs.MATERIALS[name]
            r = m["rho0"] * 10**rng.uniform(-3, 1.3, 40)
            T = 10**rng.uniform(np.log10(m["T"][0]) + 0.5, 7.5, 40)
            h = 1e-3
            p, e = gs.eos(m, r, T)
            er = (gs.eos(m, r * (1 + h), T)[1] - gs.eos(m, r * (1 - h), T)[1]) / (2 * h * r)
            pT = (gs.eos(m, r, T * (1 + h))[0] - gs.eos(m, r, T * (1 - h))[0]) / (2 * h * T)
            scale = (np.abs(p) + T * np.abs(pT)) / r**2
            res = max(res, float(np.max(np.abs(er - (p - T * pT) / r**2) / scale)))
        good = res < 1e-4
        ok = ok and good
        print(f"  thermodynamic consistency, 200 random states: largest residual {res:.1e}  [{'PASS' if good else 'FAIL'}]")

        m = gs.MATERIALS["al"]
        tab = gs.table("al", 300, 160, opacity=False)
        tab.pop("adjust_max")
        tm = tabular.TabularMaterial(tab, 1000.0, 1000.0, Material("stiffened", 3.0, 1.0))
        errs = {}
        for region, (lr, lT) in (("compressed (rho0-10 rho0, 1e3-1e7 K)", ((0.0, 1.0), (3.0, 7.0))),
                                 ("hot (0.01-10 rho0, 2e4-1e7 K)", ((-2.0, 1.0), (4.3, 7.0))),
                                 ("expanded liquid-vapour (0.01-1 rho0, 1e3-2e4 K)", ((-2.0, 0.0), (3.0, 4.3)))):
            r = m["rho0"] * 10**rng.uniform(*lr, 300)
            T = 10**rng.uniform(*lT, 300)
            p, e = gs.eos(m, r, T)
            pt = to_numpy(tm.pressure(xp.asarray(r / 1000.0), xp.asarray(e / 1e6))) * 1e9
            errs[region] = np.abs(pt - p) / (np.abs(p) + 1e-3 * m["K0"])
        single = np.concatenate([v for k, v in errs.items() if not k.startswith("expanded")])
        good = float(np.median(single)) < 2e-3 and float(np.max(single)) < 3e-2
        ok = ok and good
        print(f"  tabular p(rho, e) vs the model, Al 300 x 160 table, 300 random states per region (error relative to "
              f"|p| + 1e-3 K0): " + "; ".join(f"{k} median {np.median(v):.1e}, largest {np.max(v):.1e}" for k, v in errs.items())
              + f" (the last, where tension and thermal pressure cancel, is reported, not tested)  [{'PASS' if good else 'FAIL'}]")
    return ok


VERIFICATION_TESTS = {
    "brio_wu": brio_wu_test,
    "alfven": alfven_convergence_test,
    "reflection": boundary_reflection_test,
    "conservation": conservation_test,
    "ct": ct_test,
    "cylindrical": cylindrical_test,
    "equilibrium": equilibrium_test,
    "z_invariance": z_invariance_test,
    "alfven_z": alfven_z_test,
    "resistive_z": resistive_z_test,
    "resistive_layers": resistive_layers_test,
    "ohmic": ohmic_heating_test,
    "eos_impact": eos_impact_test,
    "eos_interface": eos_interface_test,
    "eos_positivity": eos_positivity_test,
    "radiation": radiation_decay_test,
    "conduction": conduction_test,
    "wall_conduction": wall_conduction_test,
    "transport": transport_test,
    "twotemp": twotemp_test,
    "viscosity": viscosity_test,
    "strength": strength_test,
    "parity3d": parity3d_test,
    "galilean": galilean_test,
    "rmi_mhd": rmi_mhd_smoke_test,
    "fcg": fcg_smoke_test,
    "circuit": circuit_test,
    "solid_nernst_3d": solid_nernst_3d_test,
    "mpi": mpi_test,
    "fld": fld_test,
    "render3d": render3d_test,
    "tabular": tabular_test,
    "validation": validation_test,
    "amr": amr_test,
    "amr_physics": amr_physics_test,
    "amr_transport": amr_transport_test,
    "cylindrical3d": cylindrical3d_test,
    "multigroup": multigroup_test,
    "experimental": experimental_test,
    "autodiff": autodiff_test,
    "hybrid_pic": hybrid_pic_test,
    "fno": fno_test,
    "cad": cad_test,
    "monitor": monitor_test,
    "gmg": gmg_test,
    "cyl_filter": cyl_filter_test,
    "hall": hall_test,
    "ion_transport": ion_transport_test,
    "sesame": sesame_test,
}


# ============================================================
# Theoretical V_hydro (from Richtmyer linear theory)
# ============================================================
def compute_V_hydro(gamma, mach, rho1, density_ratio, a0, pert_mode, Ly):
    M2 = mach**2
    r = min((gamma+1)*M2/((gamma-1)*M2+2), (gamma+1)/(gamma-1)-0.01)
    cs1 = np.sqrt(gamma/rho1)
    vs = mach*cs1
    delta_U = vs*(1.0-1.0/r)
    A_post = ((density_ratio*r-r)/((density_ratio*r+r)))
    a0_post = a0/r
    k_int = 2.0*np.pi*pert_mode/Ly
    V_hydro = k_int*delta_U*A_post*a0_post
    rho_avg = 0.5*(rho1*r + rho1*density_ratio*r)
    return V_hydro, {'k_int': k_int, 'delta_U': delta_U, 'A_post': A_post,
                      'a0_post': a0_post, 'r': r, 'rho_avg': rho_avg, 'vs': vs}


# ============================================================
# Enstrophy-based locking metric
# ============================================================
def extract_enstrophy_locking(solver, hydro_enstrophy, t_skip=0.05):
    """Returns True if late-time enstrophy is suppressed by > (1-fraction) of hydro."""
    t = np.array(solver.diag_times); mask = t > t_skip
    if not np.any(mask) or hydro_enstrophy < 1e-10:
        return False, 1.0
    en = np.array(solver.diag_enstrophy_local)[mask]
    tail = max(len(en)//3, 1)
    mean_en = float(np.mean(en[-tail:]))
    ratio = mean_en / hydro_enstrophy
    locked = ratio < solver.cfg.enstrophy_locking_fraction
    return locked, ratio

def extract_momentum_locking(solver, t_skip=0.05):
    """Original gap/bar momentum ratio criterion."""
    t = np.array(solver.diag_times); mask = t > t_skip
    if not np.any(mask): return 1.0
    gap = np.array(solver.diag_jet_momentum_flux_gap)[mask]
    bar = np.array(solver.diag_jet_momentum_flux_bar)[mask]
    tail = max(len(gap)//3, 1)
    mg = float(np.mean(gap[-tail:])); mb = float(np.mean(bar[-tail:]))
    return mg/max(mb, 1e-14)


# ============================================================
# Bisection search for B_crit
# ============================================================
def find_B_critical(base_params, B_low, B_high, hydro_enstrophy=None,
                     threshold_momentum=0.80, tol=0.05, max_iter=6):
    history = []
    for it in range(max_iter):
        B_mid = 0.5*(B_low + B_high)
        params = {**base_params, 'B_transverse': B_mid,
                  'B_field_type': 'striped', 'B_modulation_mode': base_params.get('B_modulation_mode', 4),
                  'snapshot_times': [0.0, base_params.get('t_end', 0.25)], 'diag_interval': 40}
        cfg = Config(**params)
        s = MHDSolver(cfg); s.initialize(); s.run()
        mom_ratio = extract_momentum_locking(s)
        # Also check enstrophy criterion if hydro baseline available
        if hydro_enstrophy is not None:
            locked_en, en_ratio = extract_enstrophy_locking(s, hydro_enstrophy)
        else:
            locked_en, en_ratio = False, 1.0
        history.append({'B0': B_mid, 'mom_ratio': mom_ratio, 'en_ratio': en_ratio})
        locked = mom_ratio < threshold_momentum
        print(f"      iter {it}: B0={B_mid:.3f}  gap/bar={mom_ratio:.3f}  "
              f"en_frac={en_ratio:.3f}  [{'LOCKED' if locked else 'UNLOCKED'}]")
        if locked: B_high = B_mid
        else: B_low = B_mid
        if (B_high - B_low) < tol: break
        del s; cp.get_default_memory_pool().free_all_blocks()
    return 0.5*(B_low+B_high), history


# ============================================================
# MAIN
# ============================================================
def run_campaign():
    """Full research campaign (verification, showcase, resistivity, Psi map, GCI, sweeps)."""
    print("\n" + "="*72)
    print("  MAGNETIC SIEVE - campaign")
    print("  Localized R-H | Resistive MHD | Psi Mapping | GCI | Enstrophy Locking")
    print("="*72); sys.stdout.flush()
    total_t0 = time.time()

    # ========== Phase 1: Verification ==========
    print("\n" + "="*64 + "\n  PHASE 1: VERIFICATION\n" + "="*64)
    bw = brio_wu_test(); af = alfven_convergence_test()
    boundary_reflection_test(); conservation_test()
    print(f"  Overall: {'ALL PASSED' if (bw and af) else 'SOME FAILED'}")

    # ========== Phase 2: Showcase runs with localized R-H ==========
    print("\n" + "="*64 + "\n  PHASE 2: SHOWCASE (localized R-H, 400x200)\n" + "="*64)

    base = dict(nx=400, ny=200, x_min=0.0, x_max=6.0, y_min=0.0, y_max=2.0,
        t_end=0.25, cfl=0.30, mach=10.0, interface_x=1.5, perturbation_amp=0.15,
        perturbation_mode=4, density_ratio=3.0, interface_width=2.0,
        powell_source=True, use_char_bc=True, bc_x_type="characteristic",
        bc_y_type="periodic", enable_smoothing=False, diag_interval=20,
        localized_rh=True)

    cases = {
        "Hydro (B=0)": dict(B_transverse=0.0, B_field_type="uniform", localized_rh=False),
        "Uniform (By=1.0)": dict(B_transverse=1.0, B_field_type="uniform"),
        "Striped (B0=1.5)": dict(B_transverse=1.5, B_field_type="striped", B_modulation_mode=4),
        "Striped (B0=2.0)": dict(B_transverse=2.0, B_field_type="striped", B_modulation_mode=4),
    }

    solvers: Dict[str, MHDSolver] = {}
    for label, cp_dict in cases.items():
        print(f"\n{'='*64}\n  CASE: {label}\n{'='*64}")
        params = {**base, **cp_dict}
        cfg = Config(**params)
        s = MHDSolver(cfg); s.initialize(); s.run()
        solvers[label] = s

    post = PostProcessor(solvers)
    post.plot_all()

    # Get hydro enstrophy baseline for enstrophy-based locking
    hydro_key = list(solvers.keys())[0]
    hydro_solver = solvers[hydro_key]
    t_arr = np.array(hydro_solver.diag_times)
    mask = t_arr > 0.05
    hydro_enstrophy = float(np.mean(np.array(hydro_solver.diag_enstrophy_local)[mask])) if np.any(mask) else 1.0

    # ========== Enstrophy-based locking report ==========
    print("\n  --- Enstrophy-Based Locking ---")
    print(f"  Hydro baseline enstrophy: {hydro_enstrophy:.1f}")
    for label, s in solvers.items():
        locked, ratio = extract_enstrophy_locking(s, hydro_enstrophy)
        status = "LOCKED" if locked else "unlocked"
        print(f"  {label:<25s}: en/en_hydro = {ratio:.3f}  [{status}]")

    # ========== Phase 3: Resistivity comparison ==========
    print("\n" + "="*64 + "\n  PHASE 3: RESISTIVE MHD COMPARISON\n" + "="*64)

    eta_values = [0.0, 1e-3, 1e-2]
    resist_solvers = {}
    for eta in eta_values:
        label = f"Striped B0=2.0, eta={eta:.0e}" if eta > 0 else "Striped B0=2.0, ideal"
        print(f"\n  --- {label} ---")
        params = {**base, 'nx': 200, 'ny': 100,
                  'B_transverse': 2.0, 'B_field_type': 'striped', 'B_modulation_mode': 4,
                  'eta_resistivity': eta, 'localized_rh': True,
                  'snapshot_times': [0.0, 0.10, 0.25], 'diag_interval': 40}
        cfg = Config(**params)
        s = MHDSolver(cfg); s.initialize(); s.run()
        resist_solvers[label] = s

    # Plot resistivity comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['C0', 'C1', 'C2']
    for i, (label, s) in enumerate(resist_solvers.items()):
        t = np.array(s.diag_times)
        axes[0].plot(t, s.diag_enstrophy_local, color=colors[i], lw=2, label=label)
        axes[1].plot(t, s.diag_jet_max_vx, color=colors[i], lw=2, label=label)
        axes[2].plot(t, s.diag_jet_collimation, color=colors[i], lw=2, label=label)
    for ax, yl, tl in [(axes[0], r'$\langle\rho\omega^2\rangle$', 'Enstrophy'),
                        (axes[1], '$v_x^{max}$', 'Peak Jet Velocity'),
                        (axes[2], 'Collimation', 'Gap Momentum Fraction')]:
        ax.set_xlabel('Time $t$'); ax.set_ylabel(yl); ax.set_title(tl, fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle('Resistive MHD: Does Reconnection Break the Sieve?', fontsize=14, fontweight='bold')
    plt.tight_layout(); save_figure(fig, 'resistivity_comparison.png')

    # ========== Phase 4: Psi(k_mod/k_int) functional mapping ==========
    print("\n" + "="*64 + "\n  PHASE 4: Psi MAPPING (k_mod/k_int sweep)\n" + "="*64)

    k_ratios = [0.5, 1.0, 2.0, 4.0]
    pert_mode = 4; Ly = 2.0
    gamma = 5.0/3.0; M = 10.0; eta_r = 3.0
    V_hydro, theory = compute_V_hydro(gamma, M, 1.0, eta_r, 0.15, pert_mode, Ly)
    sqrt_rho = np.sqrt(theory['rho_avg'])

    psi_results = []
    for kr in k_ratios:
        mod_mode = int(round(pert_mode * kr))
        if mod_mode < 1: mod_mode = 1
        actual_kr = mod_mode / pert_mode
        print(f"\n  --- k_mod/k_int = {actual_kr:.2f} (mod_mode={mod_mode}) ---")
        sweep_base = dict(nx=200, ny=100, x_min=0.0, x_max=6.0, y_min=0.0, y_max=2.0,
            t_end=0.25, cfl=0.30, mach=M, interface_x=1.5, perturbation_amp=0.15,
            perturbation_mode=pert_mode, density_ratio=eta_r, interface_width=2.0,
            powell_source=True, use_char_bc=True, bc_x_type="characteristic",
            bc_y_type="periodic", B_modulation_mode=mod_mode, localized_rh=True)

        B_est = V_hydro * sqrt_rho * 3.0  # rough estimate using Psi~3
        B_crit, hist = find_B_critical(sweep_base, max(0.5, B_est*0.3), B_est*2.0,
                                        hydro_enstrophy=hydro_enstrophy)
        Psi_measured = B_crit / (sqrt_rho * V_hydro) if V_hydro > 1e-10 else 0
        psi_results.append({'k_ratio': actual_kr, 'mod_mode': mod_mode,
                            'B_crit': B_crit, 'Psi': Psi_measured})
        print(f"    B_crit = {B_crit:.3f}, Psi = {Psi_measured:.3f}")
        cp.get_default_memory_pool().free_all_blocks()

    # Plot Psi mapping
    fig, ax = plt.subplots(figsize=(8, 5))
    krs = [r['k_ratio'] for r in psi_results]
    psis = [r['Psi'] for r in psi_results]
    ax.plot(krs, psis, 'ro-', ms=10, lw=2, label='Measured $\\Psi$')
    ax.axhline(psis[krs.index(1.0)] if 1.0 in krs else np.mean(psis),
               color='gray', ls='--', lw=1, alpha=0.5, label='$k_{mod}/k_{int}=1$ value')
    ax.set_xlabel('$k_{mod}/k_{int}$', fontsize=13); ax.set_ylabel('$\\Psi$', fontsize=13)
    ax.set_title('Geometric Nozzle Factor: The Universal Function', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    save_figure(fig, 'psi_mapping.png')

    # ========== Phase 5: Grid Convergence Index (Richardson) ==========
    print("\n" + "="*64 + "\n  PHASE 5: GRID CONVERGENCE INDEX\n" + "="*64)

    gci_resolutions = [(100, 50), (200, 100), (400, 200)]
    gci_B_crits = []

    for (nx, ny) in gci_resolutions:
        print(f"\n  --- Resolution {nx}x{ny} ---")
        gci_base = dict(nx=nx, ny=ny, x_min=0.0, x_max=6.0, y_min=0.0, y_max=2.0,
            t_end=0.25, cfl=0.30, mach=10.0, interface_x=1.5, perturbation_amp=0.15,
            perturbation_mode=4, density_ratio=3.0, interface_width=2.0,
            powell_source=True, use_char_bc=True, bc_x_type="characteristic",
            bc_y_type="periodic", B_modulation_mode=4, localized_rh=True)
        B_crit, _ = find_B_critical(gci_base, 5.0, 40.0,
                                     hydro_enstrophy=hydro_enstrophy, max_iter=5)
        gci_B_crits.append(B_crit)
        print(f"    B_crit({nx}x{ny}) = {B_crit:.3f}")
        cp.get_default_memory_pool().free_all_blocks()

    # Richardson extrapolation
    if len(gci_B_crits) == 3:
        h = [6.0/nx for nx, _ in gci_resolutions]  # grid spacings
        f1, f2, f3 = gci_B_crits  # coarse, medium, fine
        r21 = h[0]/h[1]; r32 = h[1]/h[2]
        # Observed order
        if abs(f3-f2) > 1e-10 and abs(f2-f1) > 1e-10:
            p_obs = np.log(abs((f1-f2)/(f2-f3))) / np.log(r21)
            f_exact = f3 + (f3-f2)/(r32**p_obs - 1)
            gci_fine = 1.25 * abs(f3-f2)/(r32**p_obs - 1) / abs(f3) * 100
            print(f"\n  Richardson Extrapolation:")
            print(f"    B_crit(h->0) = {f_exact:.3f}")
            print(f"    Observed order p = {p_obs:.2f}")
            print(f"    GCI_fine = {gci_fine:.2f}%")
        else:
            print("  Grid solutions too close for Richardson extrapolation")
            f_exact = f3; p_obs = 2.0; gci_fine = 0.0

    # Plot GCI
    fig, ax = plt.subplots(figsize=(7, 5))
    h_arr = np.array([6.0/nx for nx, _ in gci_resolutions])
    ax.plot(h_arr, gci_B_crits, 'bs-', ms=10, lw=2, label='$B_{crit}(h)$')
    if len(gci_B_crits) == 3:
        ax.axhline(f_exact, color='red', ls='--', lw=1.5, label=f'$B_{{crit}}(h\\to 0) = {f_exact:.2f}$')
    ax.set_xlabel('Grid spacing $h = L_x/N_x$', fontsize=12)
    ax.set_ylabel('$B_{crit}$', fontsize=12)
    ax.set_title('Grid Convergence Index (Richardson Extrapolation)', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3); ax.invert_xaxis()
    plt.tight_layout(); save_figure(fig, 'gci_convergence.png')

    # ========== Phase 6: Mach + Density sweeps (scaling law proof) ==========
    print("\n" + "="*64 + "\n  PHASE 6: SCALING LAW SWEEPS\n" + "="*64)

    sweep_base_common = dict(nx=200, ny=100, x_min=0.0, x_max=6.0, y_min=0.0, y_max=2.0,
        t_end=0.25, cfl=0.30, interface_x=1.5, perturbation_amp=0.15,
        perturbation_mode=4, interface_width=2.0,
        powell_source=True, use_char_bc=True, bc_x_type="characteristic",
        bc_y_type="periodic", B_modulation_mode=4, localized_rh=True)

    all_sweep = []
    # Mach sweep
    for M_val in [5.0, 10.0, 15.0]:
        V, th = compute_V_hydro(5.0/3.0, M_val, 1.0, 3.0, 0.15, 4, 2.0)
        B_est = V*np.sqrt(th['rho_avg'])*3.0
        sb = {**sweep_base_common, 'mach': M_val, 'density_ratio': 3.0}
        Bc, _ = find_B_critical(sb, max(0.5, B_est*0.3), B_est*2.0, hydro_enstrophy=hydro_enstrophy)
        all_sweep.append({'sweep':'mach', 'param': M_val, 'V_hydro': V,
                          'sqrt_rho': np.sqrt(th['rho_avg']), 'B_crit': Bc})
        cp.get_default_memory_pool().free_all_blocks()

    # Density sweep
    for eta_val in [1.5, 3.0, 5.0]:
        V, th = compute_V_hydro(5.0/3.0, 10.0, 1.0, eta_val, 0.15, 4, 2.0)
        B_est = V*np.sqrt(th['rho_avg'])*3.0
        sb = {**sweep_base_common, 'mach': 10.0, 'density_ratio': eta_val}
        Bc, _ = find_B_critical(sb, max(0.5, B_est*0.3), B_est*2.0, hydro_enstrophy=hydro_enstrophy)
        all_sweep.append({'sweep':'density', 'param': eta_val, 'V_hydro': V,
                          'sqrt_rho': np.sqrt(th['rho_avg']), 'B_crit': Bc})
        cp.get_default_memory_pool().free_all_blocks()

    # Scaling law plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    mach_r = [r for r in all_sweep if r['sweep']=='mach']
    dens_r = [r for r in all_sweep if r['sweep']=='density']
    all_V = np.array([r['V_hydro'] for r in all_sweep])
    all_Bn = np.array([r['B_crit']/r['sqrt_rho'] for r in all_sweep])
    Psi_fit = float(np.sum(all_V*all_Bn)/np.sum(all_V**2)) if len(all_V)>1 else 0

    ax = axes[0]
    if mach_r:
        ax.plot([r['V_hydro'] for r in mach_r], [r['B_crit']/r['sqrt_rho'] for r in mach_r],
                'rs-', ms=10, lw=2, label='Mach sweep')
    if dens_r:
        ax.plot([r['V_hydro'] for r in dens_r], [r['B_crit']/r['sqrt_rho'] for r in dens_r],
                'b^-', ms=10, lw=2, label='Density sweep')
    Vf = np.linspace(0, max(all_V)*1.2, 100)
    ax.plot(Vf, Psi_fit*Vf, 'k--', lw=1.5, label=f'$\\Psi = {Psi_fit:.2f}$')
    pred = Psi_fit*all_V; ss_res = np.sum((all_Bn-pred)**2)
    ss_tot = np.sum((all_Bn-np.mean(all_Bn))**2)
    R2 = 1-ss_res/max(ss_tot,1e-14)
    ax.text(0.05, 0.92, f'$R^2 = {R2:.4f}$', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$V_{hydro}$'); ax.set_ylabel(r'$B_{crit}/\sqrt{\bar\rho}$')
    ax.set_title('THE SCALING LAW', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    ax = axes[1]
    if mach_r:
        ax.plot([r['param'] for r in mach_r], [r['B_crit'] for r in mach_r], 'rs-', ms=10, lw=2, label='Measured')
        ax.plot([r['param'] for r in mach_r], [Psi_fit*r['V_hydro']*r['sqrt_rho'] for r in mach_r],
                'k^--', ms=8, lw=1.5, label=f'Theory ($\\Psi={Psi_fit:.2f}$)')
    ax.set_xlabel('Mach $M$'); ax.set_ylabel('$B_{crit}$')
    ax.set_title('$B_{crit}$ vs Mach', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    if dens_r:
        ax.plot([r['param'] for r in dens_r], [r['B_crit'] for r in dens_r], 'b^-', ms=10, lw=2, label='Measured')
        ax.plot([r['param'] for r in dens_r], [Psi_fit*r['V_hydro']*r['sqrt_rho'] for r in dens_r],
                'k^--', ms=8, lw=1.5, label=f'Theory ($\\Psi={Psi_fit:.2f}$)')
    ax.set_xlabel(r'$\rho_H/\rho_L$'); ax.set_ylabel('$B_{crit}$')
    ax.set_title('$B_{crit}$ vs Density Ratio', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle('Magnetic Sieve Scaling Law (Localized R-H)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); save_figure(fig, 'scaling_law_proof.png')

    # ========== Final Summary ==========
    total_elapsed = time.time() - total_t0
    print("\n" + "="*72)
    print("  CAMPAIGN COMPLETE")
    print("="*72)
    print(f"  Verification: {'ALL PASSED' if (bw and af) else 'SOME FAILED'}")
    print(f"  Psi(k_mod/k_int=1) = {Psi_fit:.4f}, R^2 = {R2:.4f}")
    print(f"  Psi mapping: {len(psi_results)} k-ratio points")
    for r in psi_results:
        print(f"    k_mod/k_int = {r['k_ratio']:.2f} -> Psi = {r['Psi']:.3f}")
    print(f"  GCI: B_crit(h->0) = {f_exact:.3f}, GCI_fine = {gci_fine:.2f}%")
    print(f"  Resistivity: {len(resist_solvers)} eta values tested")
    print(f"  Total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
    print("="*72); sys.stdout.flush()


# ============================================================
# Single showcase case
# ============================================================
SHOWCASE_CASES = {
    "hydro": dict(B_transverse=0.0, B_field_type="uniform", localized_rh=False),
    "uniform": dict(B_transverse=1.0, B_field_type="uniform"),
    "striped": dict(B_transverse=2.0, B_field_type="striped", B_modulation_mode=4),
}


def run_case(name, nx=400, ny=100, t_end=0.8, B0=None, stripe="x", eta=0.0, gap_fraction=0.0,
             a0=0.02, n_frames=61, compare_hydro=False, plot=True):
    """One showcase run on a 12 x 2 domain (long enough to keep the interface in view
    to t ~ 1). Snapshots are evenly spaced so the GIFs play smoothly for ~12 s."""
    params = dict(nx=nx, ny=ny, x_min=0.0, x_max=12.0, y_min=0.0, y_max=2.0,
        t_end=t_end, cfl=0.30, mach=10.0, interface_x=1.5, perturbation_amp=a0,
        perturbation_mode=4, density_ratio=3.0, interface_width=2.0,
        powell_source=True, bc_x_type="inflow_outflow", bc_y_type="periodic",
        diag_interval=20, localized_rh=True, B_stripe_component=stripe,
        eta_resistivity=eta, snapshot_times=list(np.linspace(0.0, t_end, n_frames)))
    runs = {}
    if compare_hydro and name != "hydro":
        runs["hydro"] = {**params, **SHOWCASE_CASES["hydro"]}
    case = {**params, **SHOWCASE_CASES[name]}
    if B0 is not None:
        case["B_transverse"] = B0
    if case.get("B_field_type") == "striped":
        case["B_gap_fraction"] = gap_fraction
    runs[name if B0 is None else f"{name} B0={B0:g}"] = case
    solvers = {}
    for label, p in runs.items():
        s = MHDSolver(Config(**p)); s.initialize(); s.run()
        solvers[label] = s
    if plot:
        PostProcessor(solvers).plot_all()
    return solvers


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Magnetic Sieve: GPU HLLD-MHD solver for the magnetised Richtmyer-Meshkov problem.",
        epilog="examples:\n"
               "  python main.py --test brio_wu          # ~10 s verification\n"
               "  python main.py --test all              # full verification suite\n"
               "  python main.py --case striped --nx 200 # one sieve run + plots\n"
               "  python main.py --campaign              # full scaling-law campaign (hours)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test", choices=[*VERIFICATION_TESTS, "all"],
                      help="run a verification test (exit code 1 on failure)")
    mode.add_argument("--case", choices=list(SHOWCASE_CASES), help="run a single showcase case")
    mode.add_argument("--campaign", action="store_true", help="run the full research campaign")
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--ny", type=int, default=None, help="default: nx/4")
    parser.add_argument("--t-end", type=float, default=0.8)
    parser.add_argument("--B0", type=float, default=None, help="override field strength")
    parser.add_argument("--stripe", choices=["x", "z", "y"], default="x",
                        help="striped component: x (tension, default), z (pressure bars), y (legacy, div B != 0)")
    parser.add_argument("--eta", type=float, default=0.0, help="resistivity")
    parser.add_argument("--gap-fraction", type=float, default=0.0,
                        help="field left in the stripe gaps as a fraction of B0 (x stripes)")
    parser.add_argument("--a0", type=float, default=0.02, help="initial interface amplitude")
    parser.add_argument("--frames", type=int, default=61, help="snapshots / GIF frames")
    parser.add_argument("--compare-hydro", action="store_true",
                        help="also run the hydro case and plot side by side")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    if args.test:
        names = list(VERIFICATION_TESTS) if args.test == "all" else [args.test]
        results = {}
        for n in names:
            fn = VERIFICATION_TESTS[n]
            results[n] = fn(plot=not args.no_plots) if n in ("brio_wu", "alfven") else fn()
        print("\n" + "\n".join(f"  {n:<14s} {'PASS' if ok else 'FAIL'}" for n, ok in results.items()))
        return 0 if all(results.values()) else 1
    if args.case:
        run_case(args.case, nx=args.nx, ny=args.ny or args.nx // 4, t_end=args.t_end,
                 B0=args.B0, stripe=args.stripe, eta=args.eta, gap_fraction=args.gap_fraction,
                 a0=args.a0, n_frames=args.frames, compare_hydro=args.compare_hydro,
                 plot=not args.no_plots)
        return 0
    if args.campaign:
        run_campaign()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
