# physics.py - CUDA HLLD kernel wrapper, variable conversions, MUSCL, jump conditions,
# resistive terms and the shock-parallel stripe equilibrium.

import sys
import os
import sysconfig
from pathlib import Path
import numpy as np


def _configure_cuda_dll_search_path():
    """Expose the bundled CUDA wheel DLLs to CuPy on Windows."""
    if os.name != 'nt':
        return

    candidates = []
    for base in [
        sysconfig.get_paths().get('purelib'),
        sysconfig.get_paths().get('platlib'),
        str(Path(sys.executable).resolve().parent.parent / 'Lib' / 'site-packages'),
    ]:
        if base:
            candidates.append(Path(base))

    for base in candidates:
        for wheel_root in ('nvidia/cuda_runtime', 'nvidia/cuda_nvrtc', 'nvidia/cublas', 'nvidia/cufft'):
            wheel_dir = base / wheel_root
            bin_dir = wheel_dir / 'bin'
            if not bin_dir.exists():
                continue
            os.environ['PATH'] = str(bin_dir) + os.pathsep + os.environ.get('PATH', '')
            if not os.environ.get('CUDA_PATH') or not os.path.exists(os.environ['CUDA_PATH']):
                os.environ['CUDA_PATH'] = str(wheel_dir)
            try:
                os.add_dll_directory(str(bin_dir))
            except (OSError, AttributeError):
                pass


_configure_cuda_dll_search_path()

# ============================================================
# CuPy / CUDA setup
# ============================================================
try:
    import cupy as cp
    HAS_CUPY = True
    dev = cp.cuda.Device(0)
    dev.use()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    dev_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    print(f"CuPy CUDA acceleration: ENABLED  (GPU: {dev_name})")
    print(f"  Compute capability: {props['major']}.{props['minor']}")
    print(f"  Total GPU memory: {props['totalGlobalMem'] / 1e9:.1f} GB")
except ImportError:
    print("ERROR: CuPy is required for this GPU version.")
    print("Install with: pip install cupy-cuda12x  (or appropriate CUDA version)")
    sys.exit(1)

xp = cp  # alias: all array ops go through this

print(f"NumPy {np.__version__}, CuPy {cp.__version__}, Python {sys.version.split()[0]}")

from .config import (
    NVAR, FLOOR_RHO, FLOOR_PR,
    RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI, RHOC, RHOSE,
    iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iPSI, iCLR, iSE,
)


# ============================================================
# CUDA Raw Kernels
# ============================================================

# CUDA source is compiled by CuPy at module import time: the HLLD kernel together with the device EOS
# (eos_fused.cu), which it calls inline for the multi-material and tabulated equations of state.
from . import eos_fused as _ef
_hlld_kernel = _ef.hlld_kernel


# ============================================================
# GPU Vectorized Operations
# ============================================================
# fused single-launch kernels for the hot paths (aegis/fused.py); off on the NumPy fallback
try:
    from . import fused as _fz
    _FUSED = xp.__name__ == "cupy"
except Exception:                            # pragma: no cover - CPU-only installs
    _FUSED = False


def _ideal(eos):
    return isinstance(eos, (int, float))


def _tracer_is_vf(eos):
    return not _ideal(eos) and eos.volume_fraction


def cons_to_prim(U, gamma):
    """Conservative to primitive -- fully on GPU. gamma is a float (one ideal gas) or an eos.EOS;
    with a volume-fraction tracer U[RHOC] holds the fraction itself, not rho times it."""
    if _FUSED and _ideal(gamma):
        return _fz.cons_to_prim_ideal(U, gamma)
    if not _ideal(gamma):
        W = _ef.cons_to_prim(U, gamma)
        if W is not None:
            return W
    W = xp.empty_like(U)
    rho = xp.maximum(U[RHO], FLOOR_RHO)
    ri = 1.0 / rho
    W[iRHO] = rho
    W[iVX] = U[MX] * ri
    W[iVY] = U[MY] * ri
    W[iVZ] = U[MZ] * ri
    W[iBX] = U[BX]
    W[iBY] = U[BY]
    W[iBZ] = U[BZ]
    W[iPSI] = U[PSI]
    W[iCLR] = U[RHOC] if _tracer_is_vf(gamma) else U[RHOC] * ri
    W[iSE] = U[RHOSE] * ri
    KE = 0.5 * rho * (W[iVX]**2 + W[iVY]**2 + W[iVZ]**2)
    ME = 0.5 * (U[BX]**2 + U[BY]**2 + U[BZ]**2)
    if _ideal(gamma):
        W[iPR] = xp.maximum((gamma - 1) * (U[EN] - KE - ME), FLOOR_PR)
    else:
        a = W[iCLR]
        W[iPR] = xp.maximum(gamma.pressure(rho, U[EN] - KE - ME, a), gamma.p_floor(rho, a))
    return W


def prim_to_cons(W, gamma):
    """Primitive to conservative -- fully on GPU."""
    if _FUSED and _ideal(gamma):
        return _fz.prim_to_cons_ideal(W, gamma)
    if not _ideal(gamma):
        U = _ef.prim_to_cons(W, gamma)
        if U is not None:
            return U
    U = xp.empty_like(W)
    rho = xp.maximum(W[iRHO], FLOOR_RHO)
    U[RHO] = rho
    U[MX] = rho * W[iVX]
    U[MY] = rho * W[iVY]
    U[MZ] = rho * W[iVZ]
    U[BX] = W[iBX]
    U[BY] = W[iBY]
    U[BZ] = W[iBZ]
    U[PSI] = W[iPSI]
    U[RHOC] = W[iCLR] if _tracer_is_vf(gamma) else rho * W[iCLR]
    U[RHOSE] = rho * W[iSE]
    KE = 0.5 * rho * (W[iVX]**2 + W[iVY]**2 + W[iVZ]**2)
    ME = 0.5 * (W[iBX]**2 + W[iBY]**2 + W[iBZ]**2)
    if _ideal(gamma):
        U[EN] = W[iPR] / (gamma - 1) + KE + ME
    else:
        U[EN] = gamma.eps(rho, W[iPR], W[iCLR]) + KE + ME
    return U


def internal_energy(U):
    """eps = E - kinetic - magnetic, per volume (no floors)"""
    rho = xp.maximum(U[RHO], FLOOR_RHO)
    return U[EN] - 0.5 * (U[MX]**2 + U[MY]**2 + U[MZ]**2) / rho - 0.5 * (U[BX]**2 + U[BY]**2 + U[BZ]**2)


def pressure_margin(U, gamma):
    """raw pressure (before any floor) minus the cavitation pressure: p for an ideal gas, p + p_inf for a
    stiffened gas. Must stay positive."""
    eps = internal_energy(U)
    if _ideal(gamma):
        return (gamma - 1) * eps
    rho = xp.maximum(U[RHO], FLOOR_RHO)
    a = U[RHOC] if gamma.volume_fraction else U[RHOC] / rho
    return gamma.pressure(rho, eps, a) - gamma.p_min(rho, a)


def radiate(U, gamma, coeff, T_min, tracer_max, dt, electron_Z=None, mask=None):
    """Optically thin bremsstrahlung over dt, in place on U[EN]; returns the energy removed per cell.

    Q = coeff rho^2 sqrt(theta), theta = p/rho. At fixed density the internal energy changes as
    d eps = phi1 dp (eps = phi1 p + phi2), so d theta/dt = -K sqrt(theta) with K = coeff rho/phi1, which
    has the exact solution sqrt(theta_new) = sqrt(theta) - K dt/2. Cooling stops at theta = T_min and acts
    only where the liner fraction is below tracer_max (and mask, if given: AMR leaves out coarse cells under a
    patch, which the patch radiates). Exact, so no stability limit on dt."""
    W = cons_to_prim(U, gamma)
    rho, p, a = W[iRHO], W[iPR], xp.clip(W[iCLR], 0.0, 1.0)
    if electron_Z is not None:               # two-temperature: the electrons radiate, at T_e
        from . import twotemp
        Z = electron_Z
        _, _, pe = twotemp.pressures(U, gamma)
        theta = (1 + Z) / Z * pe / rho
        phi1 = Z / ((1 + Z) * (gamma - 1))   # d eps_e / d(rho theta_e)
    else:
        theta = p / rho
        phi1 = 1.0 / (gamma - 1) if _ideal(gamma) else gamma.coeffs(rho, a)[0]
    on = (theta > T_min) & (a < tracer_max)
    if mask is not None:
        on = on & mask
    K = coeff * rho / phi1
    s = xp.maximum(xp.sqrt(theta) - 0.5 * K * dt, np.sqrt(T_min))
    dE = xp.where(on, phi1 * rho * (theta - s * s), 0.0)
    U[EN] -= dE
    if electron_Z is not None:
        from . import twotemp
        twotemp.set_pe(U, pe - (gamma - 1) * dE, gamma)
    return dE


def _along(nd, axis, s, rest=slice(None)):
    """index tuple for a spatial array: s on axis, rest elsewhere"""
    idx = [rest] * nd
    idx[axis] = s
    return tuple(idx)


def vf_advection_rate(F, WL, WR, a_c, axis, h, wf=None, wc=None):
    """Volume-fraction tracer, da/dt + u.grad(a) = 0, along one spatial axis: -(div(u a) - a div u) with the
    face a the HLLD kernel used for the energy flux (F_a = mass flux * a_upwind) and u_f = mass flux /
    upwind density. F, WL, WR hold the faces along `axis` (variables first); a_c is the tracer on the
    cells, interior along `axis`. Shared by the 2D and 3D solvers."""
    F0, Fa = F[RHO], F[RHOC]
    rho_up = xp.where(F0 >= 0, WL[iRHO], WR[iRHO])
    u_f = F0 / rho_up
    a_f = xp.where(xp.abs(F0) > 1e-300, Fa / xp.where(F0 == 0, 1.0, F0), 0.5 * (WL[iCLR] + WR[iCLR]))
    ua = u_f * a_f
    nd, n = a_c.ndim, a_c.shape[axis]
    hi, lo = _along(nd, axis, slice(1, 1 + n)), _along(nd, axis, slice(0, n))
    if wf is not None:                       # cylindrical radius: face weights r_f, cell weights r_c
        ua, u_f = ua * wf, u_f * wf
        return -((ua[hi] - ua[lo]) - a_c * (u_f[hi] - u_f[lo])) / (h * wc)
    return -((ua[hi] - ua[lo]) - a_c * (u_f[hi] - u_f[lo])) / h


def braginskii_kperp_ratio(x):
    """Braginskii (1965) electron k_perp / k_par, Z = 1: (4.664 x^2 + 11.92) / (3.1616 (x^4 + 14.79 x^2 + 3.7703))"""
    return (4.664 * x * x + 11.92) / (3.1616 * (x**4 + 14.79 * x * x + 3.7703))


def _kperp(kpar, x, perp):
    return kpar * braginskii_kperp_ratio(x) if perp == "braginskii" else kpar / (1.0 + x * x)


def _vl(a, b):
    """van Leer (harmonic) limiter"""
    return xp.where(a * b > 0, 2.0 * a * b / xp.where(a + b == 0, 1.0, a + b), 0.0)


def _pad1(f, periodic):
    for ax, per in enumerate(periodic):
        width = [(0, 0)] * f.ndim
        width[ax] = (1, 1)
        f = xp.pad(f, width, mode="wrap" if per else "edge")
    return f


def conduction_div(theta, B, rho, a, kappa0, x0, tracer_max, h, periodic, T_exp=2.5, radial=None, wall_T=None,
                   perp="braginskii", ghost=None, return_flux=False):
    """-div q for magnetized (Braginskii-type) conduction on interior cells, 2D or 3D:
        q = -[k_perp I + (k_par - k_perp) b b] . grad(theta),  k_par = kappa0 theta^T_exp,
        k_perp = k_par / (1 + x^2),  x = omega_ce tau_e = x0 |B| theta^1.5 / rho.
    Face fluxes, so the energy update telescopes. Normal gradient: compact two-point difference; transverse
    gradients: van Leer limited average of the four one-sided differences around the face (Sharma &
    Hammett 2007), which keeps strongly anisotropic conduction from moving heat from cold to hot. Faces
    touching the liner (tracer >= tracer_max) and non-periodic domain faces carry no heat.
    theta, rho, a: interior arrays; B: three interior arrays (Bz enters |B| and b in 2D).
    radial = (r_faces, r_cells) along axis 0 for cylindrical geometry: div q = (1/r) d(r q_r)/dr + ...
    wall_T: if given, a face between fuel and liner conducts from the fuel cell to a wall at temperature wall_T
    on the face (half a cell away, i.e. a Dirichlet condition with the fuel's anisotropic k); the heat goes into
    the liner cell, so energy stays conserved. Otherwise such faces are insulating.
    ghost = (theta, rho, a, (Bx, By, Bz)) arrays padded by one cell: the outer ring is used as neighbour data and
    the domain faces conduct (an AMR patch, ghost data from the coarse level). return_flux: also return the face
    fluxes q (toward +ax, per axis; times r on radial faces)."""
    nd = theta.ndim
    if ghost is None:
        tp, rp, ap = (_pad1(f, periodic) for f in (theta, rho, a))
        Bp = [_pad1(f, periodic) for f in B]
    else:
        tp = ghost[0].copy()
        tp[(slice(1, -1),) * nd] = theta
        rp, ap, Bp = ghost[1], ghost[2], ghost[3]
    qs = []
    div = xp.zeros_like(theta)
    core = slice(1, -1)
    for ax in range(nd):
        L, R = _along(nd, ax, slice(0, -1), core), _along(nd, ax, slice(1, None), core)
        gn = (tp[R] - tp[L]) / h[ax]
        Bf = [0.5 * (Bc[L] + Bc[R]) for Bc in Bp]
        Bm = xp.sqrt(Bf[0]**2 + Bf[1]**2 + Bf[2]**2)
        inv = xp.where(Bm > 0, 1.0 / xp.where(Bm > 0, Bm, 1.0), 0.0)
        b = [c * inv for c in Bf]
        tf = xp.maximum(0.5 * (tp[L] + tp[R]), 0.0)
        kpar = kappa0 * tf**T_exp
        x = x0 * Bm * tf**1.5 / (0.5 * (rp[L] + rp[R]))
        kperp = _kperp(kpar, x, perp)
        bg = b[ax] * gn
        fuelL, fuelR = ap[L] < tracer_max, ap[R] < tracer_max
        wall = wall_T is not None
        if wall:                                 # fuel on one side, liner on the other: conduct to the wall
            wL = fuelL & ~fuelR
            tf = xp.where(wL, tp[L], tp[R])
            gw = xp.where(wL, wall_T - tp[L], tp[R] - wall_T) / (0.5 * h[ax])
            kw = kappa0 * tf**T_exp
            xw = x0 * Bm * tf**1.5 / xp.where(wL, rp[L], rp[R])
            kpw = _kperp(kw, xw, perp)
            bgw = b[ax] * gw
        for t in range(nd):
            if t == ax:
                continue
            side = []
            for S in (L, R):
                c, up, dn = list(S), list(S), list(S)
                up[t], dn[t] = slice(2, None), slice(0, -2)
                c, up, dn = tuple(c), tuple(up), tuple(dn)
                side.append(_vl((tp[c] - tp[dn]) / h[t], (tp[up] - tp[c]) / h[t]))
            bg = bg + b[t] * _vl(side[0], side[1])
            if wall:                             # transverse gradient from the fuel side only
                bgw = bgw + b[t] * xp.where(wL, side[0], side[1])
        q = -(kperp * gn + (kpar - kperp) * b[ax] * bg)
        if wall:
            qw = -(kpw * gw + (kw - kpw) * b[ax] * bgw)
            # second law at the wall: an oblique field's transverse term must not pump heat from the wall up
            # the fuel-wall temperature difference
            qw = xp.where(qw * gw > 0, 0.0, qw)
            q = xp.where(fuelL & fuelR, q, xp.where(fuelL ^ fuelR, qw, 0.0))
        else:
            q = xp.where(fuelL & fuelR, q, 0.0)
        if not periodic[ax] and ghost is None:
            n = q.shape[ax]
            q[_along(nd, ax, 0)] = 0.0
            q[_along(nd, ax, n - 1)] = 0.0
        if ax == 0 and radial is not None:
            q = q * radial[0].reshape((-1,) + (1,) * (nd - 1))
            div -= ((q[_along(nd, ax, slice(1, None))] - q[_along(nd, ax, slice(0, -1))])
                    / (h[ax] * radial[1].reshape((-1,) + (1,) * (nd - 1))))
            qs.append(q)
            continue
        div -= (q[_along(nd, ax, slice(1, None))] - q[_along(nd, ax, slice(0, -1))]) / h[ax]
        qs.append(q)
    return (div, qs) if return_flux else div


def rkl2(y0, L, dt, dt_expl, s_max=60):
    """Second-order Runge-Kutta-Legendre super-time-step (Meyer, Balsara & Aslam 2014) of dy/dt = L(y) over
    dt, given the forward-Euler stable step dt_expl. Stable for dt <= dt_expl (s^2 + s - 2)/4 with s stages;
    longer steps are split into sub-steps of at most s_max stages. Returns (y, total stages).
    If L returns (L(y), fluxes) with L(y) = -div(fluxes), the fluxes go through the same linear recursion, so
    y_end = y0 - div(Q) exactly with Q the time-integrated face flux; then returns (y, stages, Q)."""
    ratio = dt / dt_expl
    stages = lambda r: max(2, int(np.ceil(0.5 * (np.sqrt(9.0 + 16.0 * r) - 1.0))))
    s, nsub = stages(ratio), 1
    if s > s_max:                                   # each sub-step covers dt/nsub: ratio/nsub
        nsub = int(np.ceil(4.0 * ratio / (s_max * s_max + s_max - 2.0)))
        s = stages(ratio / nsub)
    h = dt / nsub
    b = [1.0 / 3.0] * 3 + [(j * j + j - 2.0) / (2.0 * j * (j + 1.0)) for j in range(3, s + 1)]
    w1 = 4.0 / (s * s + s - 2.0)
    y = y0
    Q = None
    for _ in range(nsub):
        L0 = L(y)
        track = isinstance(L0, tuple)
        if track:
            L0, q0 = L0
            Qm2, Qm1 = [0.0 * q for q in q0], [b[1] * w1 * h * q for q in q0]
        ym2, ym1 = y, y + b[1] * w1 * h * L0
        for j in range(2, s + 1):
            mu = (2.0 * j - 1.0) / j * b[j] / b[j - 1]
            nu = -(j - 1.0) / j * b[j] / b[j - 2]
            mut = mu * w1
            gt = -(1.0 - b[j - 1]) * mut
            Lj = L(ym1)
            if track:
                Lj, qj = Lj
                Qj = [mu * a1 + nu * a2 + mut * h * c1 + gt * h * c0 for a1, a2, c1, c0 in zip(Qm1, Qm2, qj, q0)]
                Qm2, Qm1 = Qm1, Qj
            yj = mu * ym1 + nu * ym2 + (1.0 - mu - nu) * y + mut * h * Lj + gt * h * L0
            ym2, ym1 = ym1, yj
        y = ym1
        if track:
            Q = Qm1 if Q is None else [a + c for a, c in zip(Q, Qm1)]
    return (y, s * nsub) if Q is None else (y, s * nsub, Q)


def conduct(U, gamma, kappa0, x0, tracer_max, h, periodic, dt, cfl=0.5, radial=None, wall_T=None,
            perp="braginskii", electron_Z=None, ghost_U=None, flux_out=None, ion_Z=None):
    """Split step: anisotropic conduction over dt on the interior conservative array U (in place on U[EN]),
    with density, momentum and B frozen, by RKL2. Returns the number of stages used (0 if nothing to do).
    ghost_U: the ghost-inclusive array U is the interior of; its first ghost ring is the neighbour data of the
    edge cells (held fixed over the step) and the domain faces conduct. flux_out: a list that receives the
    time-integrated face fluxes (one array per axis, energy per area). ion_Z: two-temperature ion conduction
    instead, at theta_i = (1 + Z) p_i / rho with the electron pressure held (the electron entropy is untouched)."""
    rho = xp.maximum(U[RHO], FLOOR_RHO)
    vf = not _ideal(gamma) and gamma.volume_fraction
    a = xp.clip(U[RHOC] if vf else U[RHOC] / rho, 0.0, 1.0)
    fixed = 0.5 * (U[MX]**2 + U[MY]**2 + U[MZ]**2) / rho + 0.5 * (U[BX]**2 + U[BY]**2 + U[BZ]**2)
    B = (U[BX], U[BY], U[BZ])

    def p_of(E):
        eps = E - fixed
        return (gamma - 1) * eps if _ideal(gamma) else gamma.pressure(rho, eps, a)

    if electron_Z is not None:               # two-temperature: electron conduction at T_e, ions held fixed
        from . import twotemp
        Z = electron_Z
        _, p_start, pe_start = twotemp.pressures(U, gamma)
        p_ion = p_start - pe_start

        def theta_of(E):
            return (1 + Z) / Z * xp.maximum(p_of(E) - p_ion, 0.0) / rho
    elif ion_Z is not None:
        from . import twotemp
        _, _, pe_fixed = twotemp.pressures(U, gamma)

        def theta_of(E):
            return (1 + ion_Z) * xp.maximum(p_of(E) - pe_fixed, 0.0) / rho
    else:
        def theta_of(E):
            return xp.maximum(p_of(E), 0.0) / rho

    ghost = None
    if ghost_U is not None:                          # one ring of neighbour cells, held fixed over the step
        ng = (ghost_U.shape[1] - U.shape[1]) // 2
        ring = ghost_U[(slice(None),) + tuple(slice(ng - 1, ng + n + 1) for n in U.shape[1:])]
        Wg = cons_to_prim(ring, gamma)
        tg = xp.maximum(Wg[iPR], 0.0) / Wg[iRHO]
        if electron_Z is not None or ion_Z is not None:
            from . import twotemp
            te_g, ti_g = twotemp.temperatures(ring, gamma, electron_Z if electron_Z is not None else ion_Z)
            tg = te_g if electron_Z is not None else ti_g
        ghost = (tg, Wg[iRHO], xp.clip(Wg[iCLR], 0.0, 1.0), (Wg[iBX], Wg[iBY], Wg[iBZ]))
    track = flux_out is not None

    def rate(E):
        return conduction_div(theta_of(E), B, rho, a, kappa0, x0, tracer_max, h, periodic, radial=radial,
                              wall_T=wall_T, perp=perp, ghost=ghost, return_flux=track)

    # forward-Euler limit from the largest diffusivity k_par / (rho phi1), phi1 = d eps / d p
    phi1 = 1.0 / (gamma - 1) if _ideal(gamma) else gamma.coeffs(rho, a)[0]
    if electron_Z is not None:
        phi1 = electron_Z / ((1 + electron_Z) * (gamma - 1))
    elif ion_Z is not None:                      # d eps / d theta_i = rho / ((1 + Z)(gamma - 1))
        phi1 = 1.0 / ((1 + ion_Z) * (gamma - 1))
    D = xp.where(a < tracer_max, kappa0 * theta_of(U[EN])**2.5 / (rho * phi1), 0.0)
    D_max = float(xp.max(D))
    if D_max <= 0:
        return 0
    dt_expl = cfl / (2.0 * D_max * sum(1.0 / hh**2 for hh in h))
    if wall_T is not None:
        dt_expl *= 0.5                           # a wall face sits half a cell away: twice the stiffness
    out = rkl2(U[EN], rate, dt, dt_expl)
    U[EN], stages = out[0], out[1]
    if track:
        flux_out.extend(out[2])
    if electron_Z is not None:
        twotemp.set_pe(U, xp.maximum(p_of(U[EN]) - p_ion, 0.0), gamma)
    return stages


def conduction_rate(U, gamma, kappa0, x0, tracer_max, h, periodic, radial=None, wall_T=None, perp="braginskii"):
    """Instantaneous -div q (energy per volume per time) of the conduction model for a conservative
    interior array U: a diagnostic (e.g. heat leaving a hot spot, or the same with x0 = 0 to see how much
    the field insulates). The time integration is done by conduct()."""
    W = cons_to_prim(U, gamma)
    rho = xp.maximum(W[iRHO], FLOOR_RHO)
    theta = xp.maximum(W[iPR], 0.0) / rho
    return conduction_div(theta, (W[iBX], W[iBY], W[iBZ]), rho, xp.clip(W[iCLR], 0.0, 1.0), kappa0, x0,
                          tracer_max, h, periodic, radial=radial, wall_T=wall_T, perp=perp)


def sound_speed2(W, gamma):
    """squared sound speed of a primitive state"""
    rho = xp.maximum(W[iRHO], FLOOR_RHO)
    if _ideal(gamma):
        return gamma * xp.maximum(W[iPR], FLOOR_PR) / rho
    return xp.maximum(gamma.c2(rho, W[iPR], W[iCLR]), 0.0)


def _floor_faces(WL, WR, eos):
    if not _ideal(eos) and _ef.floor_faces(WL, eos) and _ef.floor_faces(WR, eos):
        return
    for Wf in (WL, WR):
        Wf[iRHO] = xp.maximum(Wf[iRHO], FLOOR_RHO)
        if _ideal(eos):
            Wf[iPR] = xp.maximum(Wf[iPR], FLOOR_PR)
        else:
            Wf[iCLR] = xp.clip(Wf[iCLR], 0.0, 1.0)
            Wf[iPR] = xp.maximum(Wf[iPR], eos.p_floor(Wf[iRHO], Wf[iCLR]))


def muscl_x(W, eos=None):
    """MUSCL reconstruction in x -- GPU vectorized with van Leer limiter."""
    if _FUSED:
        e = 1.4 if eos is None else eos
        WL, WR = _fz.muscl(W, 0, floor_p=_ideal(e))
        if not _ideal(e):
            _floor_faces(WL, WR, e)
        return WL, WR
    dm = W[:, 1:-1, :] - W[:, :-2, :]
    dp = W[:, 2:, :] - W[:, 1:-1, :]
    ab = dm * dp
    d = xp.where(ab > 0, 2.0 * ab / (dm + dp + 1e-30), 0.0)
    WL = W[:, 1:-2, :] + 0.5 * d[:, :-1, :]
    WR = W[:, 2:-1, :] - 0.5 * d[:, 1:, :]
    _floor_faces(WL, WR, 1.4 if eos is None else eos)
    return WL, WR


def muscl_y(W, eos=None):
    """MUSCL reconstruction in y -- GPU vectorized with van Leer limiter."""
    if _FUSED:
        e = 1.4 if eos is None else eos
        WL, WR = _fz.muscl(W, 1, floor_p=_ideal(e))
        if not _ideal(e):
            _floor_faces(WL, WR, e)
        return WL, WR
    dm = W[:, :, 1:-1] - W[:, :, :-2]
    dp = W[:, :, 2:] - W[:, :, 1:-1]
    ab = dm * dp
    d = xp.where(ab > 0, 2.0 * ab / (dm + dp + 1e-30), 0.0)
    WL = W[:, :, 1:-2] + 0.5 * d[:, :, :-1]
    WR = W[:, :, 2:-1] - 0.5 * d[:, :, 1:]
    _floor_faces(WL, WR, 1.4 if eos is None else eos)
    return WL, WR


_NO_EOS = None


def hlld_flux_x(WL, WR, gamma, ch, axn=0):
    """HLLD flux using CUDA kernel -- one thread per interface. With an eos.EOS the kernel gets each face
    state's internal energy and squared sound speed instead of one gamma."""
    global _NO_EOS
    n_faces = WL.shape[1]
    F = xp.empty_like(WL)
    smax_arr = xp.empty(n_faces, dtype=xp.float64)

    block = 256
    grid = (n_faces + block - 1) // block
    if _NO_EOS is None:
        _NO_EOS = xp.zeros(1, dtype=xp.float64)
    eP = eT = _NO_EOS
    dev = None if _ideal(gamma) else _ef.device(gamma)
    if _ideal(gamma):
        eL = eR = c2L = c2R = _NO_EOS
        use_eos, g0 = 0, gamma
    elif dev is not None:                    # the kernel evaluates the EOS at each face state itself
        eL = eR = c2L = c2R = _NO_EOS
        (eP, eT), use_eos, g0 = dev, 2, 0.0
    else:
        eL, eR = (xp.ascontiguousarray(gamma.eps(Wf[iRHO], Wf[iPR], Wf[iCLR])) for Wf in (WL, WR))
        c2L, c2R = (xp.ascontiguousarray(xp.maximum(gamma.c2(Wf[iRHO], Wf[iPR], Wf[iCLR]), 0.0)) for Wf in (WL, WR))
        use_eos, g0 = 1, 0.0

    _hlld_kernel(
        (grid,), (block,),
        (WL, WR, F, smax_arr,
         np.int32(n_faces), np.float64(g0), np.float64(ch),
         np.float64(FLOOR_RHO), np.float64(FLOOR_PR),
         eL, eR, c2L, c2R, np.int32(use_eos), np.int32(axn), eP, eT)
    )

    return F, smax_arr                       # per-face max signal speed, left on the device (no sync)


def hlld_flux_y(WL, WR, gamma, ch):
    """HLLD flux in y (the kernel rotates the components itself)"""
    return hlld_flux_x(WL, WR, gamma, ch, axn=1)


# ============================================================
# MHD Rankine-Hugoniot
# ============================================================
def mhd_rankine_hugoniot(gamma, M, rho1, p1, By1):
    """Solve MHD Rankine-Hugoniot (CPU -- called once at init).
    For striped fields, use the RMS By for shock jump estimates."""
    M2 = M * M
    if abs(By1) < 1e-14:
        r = (gamma + 1) * M2 / ((gamma - 1) * M2 + 2)
        cs1 = np.sqrt(gamma * p1 / rho1)
        vs = M * cs1
        p2 = p1 * (2 * gamma * M2 - (gamma - 1)) / (gamma + 1)
        vx2 = vs * (1 - 1.0 / r)
        return rho1 * r, p2, vx2, 0.0, vs

    cs1_sq = gamma * p1 / rho1
    va1_sq = By1**2 / rho1
    cf1 = np.sqrt(cs1_sq + va1_sq)
    vs = M * cf1

    def get_p2(r):
        return p1 + rho1 * vs**2 * (1 - 1.0/r) + 0.5 * By1**2 * (1 - r**2)

    def energy_residual(r):
        p2_ = get_p2(r)
        lhs = 0.5*vs**2 + gamma/(gamma-1)*p1/rho1 + By1**2/rho1
        rhs = 0.5*vs**2/r**2 + gamma/(gamma-1)*p2_/(rho1*r) + r*By1**2/rho1
        return lhs - rhs

    r = (gamma + 1) * M2 / ((gamma - 1) * M2 + 2)
    r = min(r, (gamma + 1) / (gamma - 1) - 0.01)

    for _ in range(100):
        f = energy_residual(r)
        dr = 1e-6 * r
        fp = (energy_residual(r + dr) - energy_residual(r - dr)) / (2 * dr)
        if abs(fp) < 1e-30: break
        r_new = r - f / fp
        r_new = max(r_new, 1.001)
        r_new = min(r_new, (gamma + 1) / (gamma - 1) - 0.001)
        if abs(r_new - r) < 1e-12:
            r = r_new; break
        r = r_new

    rho2 = rho1 * r
    p2 = get_p2(r)
    By2 = r * By1
    vx2 = vs * (1 - 1.0 / r)
    return rho2, p2, vx2, By2, vs


# ============================================================
# Shock-parallel striped field: upstream pressure equilibrium
# ============================================================
def parallel_stripe_limits(gamma, M, p_bar, closure, rho_bar=1.0, gap_fraction=0.0):
    """Largest admissible B0 for Bx = B0 cos(k y) stripes in pressure balance.

    Both conditions are evaluated for a planar shock of speed vs = M c_s(bar):
      * fast shock: the hydrodynamic jump is an evolutionary MHD shock only if
        the downstream flow stays super-Alfvenic, u2 = vs/r > vA2, which binds
        at the bar centres: B0 < vs sqrt(rho_bar / r);
      * gap Mach > 1 (uniform density only): the gaps carry
        p_bar + (1 - eps^2) B0^2/2 (eps = gap_fraction), so
        M_gap = M sqrt(p_bar / (p_bar + (1 - eps^2) B0^2/2)) must exceed 1.
    """
    cs_bar = np.sqrt(gamma * p_bar / rho_bar)
    vs = M * cs_bar
    r = (gamma + 1) * M**2 / ((gamma - 1) * M**2 + 2)
    b_fast = vs * np.sqrt(rho_bar / r)
    b_gap = (np.sqrt(2 * p_bar * (M**2 - 1) / (1 - gap_fraction**2))
             if closure == "uniform_density" else np.inf)
    return {"vs": vs, "B0_max_fast_shock": b_fast, "B0_max_gap_mach": b_gap,
            "B0_max": min(b_fast, b_gap)}


def parallel_stripe_state(gamma, M, p_bar, closure, Bx, rho_bar=1.0):
    """Upstream equilibrium and per-row hydrodynamic jump for Bx(y) stripes.

    Returns upstream (rho1, p1), post-shock (rho2, p2, vx2), the shock speed,
    the local upstream Mach number and the downstream Alfven Mach number.
    """
    B0 = float(xp.max(xp.abs(Bx)))
    P_total = p_bar + 0.5 * B0**2
    p1 = P_total - 0.5 * Bx**2
    if closure == "uniform_density":
        rho1 = xp.full_like(Bx, rho_bar)
    elif closure == "uniform_temperature":
        rho1 = rho_bar * p1 / p_bar
    else:
        raise ValueError(f"equilibrium_closure must be 'uniform_density' or "
                         f"'uniform_temperature', got {closure!r}")
    vs = M * np.sqrt(gamma * p_bar / rho_bar)
    M_loc = vs / xp.sqrt(gamma * p1 / rho1)
    M2 = M_loc**2
    r = (gamma + 1) * M2 / ((gamma - 1) * M2 + 2)
    p2 = p1 * (2 * gamma * M2 - (gamma - 1)) / (gamma + 1)
    rho2 = r * rho1
    vx2 = vs * (1.0 - 1.0 / r)
    # Downstream Alfven Mach number in the shock frame (> 1 for a fast shock).
    absB = xp.abs(Bx)
    MA2 = xp.where(absB > 0, (vs / r) * xp.sqrt(rho2) / xp.maximum(absB, 1e-300), xp.inf)
    return {"P_total": P_total, "rho1": rho1, "p1": p1, "rho2": rho2, "p2": p2,
            "vx2": vx2, "vs": vs, "M_local": M_loc, "MA_downstream": MA2}


# ============================================================
# Smoothing Helper
# ============================================================
def smooth(y, n=5, enabled=True):
    """Optional smoothing filter for diagnostics."""
    if not enabled:
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 2 * n:
        return y
    kernel = np.ones(n) / n
    yp = np.concatenate([np.full(n, y[0]), y, np.full(n, y[-1])])
    return np.convolve(yp, kernel, mode='same')[n:-n]


# ============================================================
# Linear Theory
# ============================================================
def richtmyer_linear_theory(t_arr, gamma, mach, rho1, rho_heavy, a0, k, By=0.0):
    """Compute linear RMI growth (CPU)."""
    M2 = mach * mach
    if abs(By) < 1e-14:
        cs1 = np.sqrt(gamma * 1.0 / rho1)
        vs = mach * cs1
    else:
        cs1_sq = gamma * 1.0 / rho1
        va1_sq = By**2 / rho1
        cf1 = np.sqrt(cs1_sq + va1_sq)
        vs = mach * cf1

    r = (gamma + 1) * M2 / ((gamma - 1) * M2 + 2)
    r = min(r, (gamma + 1) / (gamma - 1) - 0.01)
    a0_post = a0 / r
    rho2_light = rho1 * r
    rho2_heavy = rho_heavy * r
    A_post = (rho2_heavy - rho2_light) / (rho2_heavy + rho2_light)
    delta_v = vs * (1.0 - 1.0 / r)
    x_shock_init = 1.2
    x_interface = 1.5
    t_shock = (x_interface - x_shock_init) / vs
    da_dt = A_post * k * a0_post * delta_v

    a_linear = np.zeros_like(t_arr)
    for i, t in enumerate(t_arr):
        if t < t_shock:
            a_linear[i] = a0
        else:
            dt_val = t - t_shock
            if abs(By) < 1e-14:
                a_linear[i] = a0_post + da_dt * dt_val
            else:
                rho_avg = 0.5 * (rho2_light + rho2_heavy)
                vA = By / np.sqrt(rho_avg)
                omega_A = k * vA
                if omega_A > 1e-14:
                    a_linear[i] = a0_post * np.cos(omega_A * dt_val) + \
                                  (da_dt / omega_A) * np.sin(omega_A * dt_val)
                else:
                    a_linear[i] = a0_post + da_dt * dt_val

    info = {
        'A_post': A_post, 'da_dt': da_dt, 'delta_v': delta_v,
        'vs': vs, 't_shock': t_shock, 'a0_post': a0_post, 'r': r,
    }
    return np.abs(a_linear), t_shock, info


# ============================================================
# Localized Rankine-Hugoniot (GPU-vectorised)
# ============================================================
def mhd_rankine_hugoniot_local_gpu(gamma, M, rho1, p1, By_field):
    """Solve MHD R-H jump conditions at every grid cell on the GPU.

    Instead of using a single RMS By_eff, this computes the post-shock
    state (rho2, p2, vx2, By2, vs) at each (x,y) using the local By(y).

    Parameters
    ----------
    gamma : float
    M : float -- shock Mach number
    rho1 : float -- pre-shock density (scalar, uniform)
    p1 : float -- pre-shock pressure (scalar, uniform)
    By_field : cupy array, shape (nx_tot, ny_tot) -- local By values

    Returns
    -------
    rho2, p2, vx2, By2, vs : cupy arrays, same shape as By_field
    """
    # For cells where |By| < threshold, use pure hydro jump
    M2 = M * M
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0

    # Hydro jump (By=0 limit)
    r_hydro = gp1 * M2 / (gm1 * M2 + 2.0)
    r_hydro = min(r_hydro, gp1 / gm1 - 0.01)
    cs1 = np.sqrt(gamma * p1 / rho1)
    vs_hydro = M * cs1
    p2_hydro = p1 * (2.0 * gamma * M2 - gm1) / gp1
    vx2_hydro = vs_hydro * (1.0 - 1.0 / r_hydro)

    By_abs = xp.abs(By_field)
    is_hydro = By_abs < 1e-10

    # For MHD cells, compute per-cell jump
    # Fast magnetosonic speed: cf = sqrt(cs^2 + va^2)
    cs1_sq = gamma * p1 / rho1
    va1_sq = By_field**2 / rho1
    cf1 = xp.sqrt(cs1_sq + va1_sq)
    vs_local = M * cf1

    # Initial guess for compression ratio (hydro value)
    r_init = gp1 * M2 / (gm1 * M2 + 2.0)
    r_init = min(r_init, gp1 / gm1 - 0.01)
    r = xp.full_like(By_field, r_init)

    # Vectorized Newton iteration with a per-cell residual mask. Cells whose
    # residual is already below 1e-6 are left unchanged and do not consume
    # further GPU work in the next iteration.
    tolerance = 1e-6
    max_iterations = 50
    for _ in range(max_iterations):
        p2_iter = p1 + rho1 * vs_local**2 * (1.0 - 1.0 / r) + 0.5 * By_field**2 * (1.0 - r**2)
        lhs = 0.5 * vs_local**2 + gamma / gm1 * p1 / rho1 + By_field**2 / rho1
        rhs = 0.5 * vs_local**2 / r**2 + gamma / gm1 * p2_iter / (rho1 * r) + r * By_field**2 / rho1
        f_val = lhs - rhs
        converged = xp.abs(f_val) < tolerance
        if bool(xp.all(converged)):
            break

        dr = 1e-6 * r
        p2_p = p1 + rho1 * vs_local**2 * (1.0 - 1.0 / (r + dr)) + 0.5 * By_field**2 * (1.0 - (r + dr)**2)
        rhs_p = 0.5 * vs_local**2 / (r + dr)**2 + gamma / gm1 * p2_p / (rho1 * (r + dr)) + (r + dr) * By_field**2 / rho1
        p2_m = p1 + rho1 * vs_local**2 * (1.0 - 1.0 / (r - dr)) + 0.5 * By_field**2 * (1.0 - (r - dr)**2)
        rhs_m = 0.5 * vs_local**2 / (r - dr)**2 + gamma / gm1 * p2_m / (rho1 * (r - dr)) + (r - dr) * By_field**2 / rho1
        fp_val = ((lhs - rhs_p) - (lhs - rhs_m)) / (2.0 * dr)

        fp_safe = xp.where(xp.abs(fp_val) > 1e-30, fp_val, xp.ones_like(fp_val))
        r_new = r - f_val / fp_safe
        r_new = xp.clip(r_new, 1.001, gp1 / gm1 - 0.001)
        r = xp.where(converged, r, r_new)

    # Compute post-shock state
    rho2 = rho1 * r
    p2 = p1 + rho1 * vs_local**2 * (1.0 - 1.0 / r) + 0.5 * By_field**2 * (1.0 - r**2)
    By2 = r * By_field
    vx2 = vs_local * (1.0 - 1.0 / r)

    # Fall back to hydro jump where |By| ~ 0
    rho2 = xp.where(is_hydro, rho1 * r_hydro, rho2)
    p2 = xp.where(is_hydro, p2_hydro, p2)
    vx2 = xp.where(is_hydro, vx2_hydro, vx2)
    By2 = xp.where(is_hydro, 0.0, By2)
    vs_out = xp.where(is_hydro, vs_hydro, vs_local)

    # Floor pressure
    p2 = xp.maximum(p2, FLOOR_PR)

    return rho2, p2, vx2, By2, vs_out


# ============================================================
# Resistive MHD (GPU-vectorised, conservative)
# ============================================================
def resistivity_field(W, cfg):
    """Cell-centred eta for the configured model. Returns a float for "constant" (so the
    uniform case runs exactly as before) and an array shaped like W[0] otherwise."""
    model = cfg.eta_model
    if model == "constant":
        eta = cfg.eta_resistivity
        return min(eta, cfg.eta_max) if cfg.eta_max > 0 else eta
    C = xp.clip(W[iCLR], 0.0, 1.0)
    if model == "material":
        eta = _blend(cfg.eta_fuel, cfg.eta_liner, C)
    else:
        # Spitzer: eta ~ T^-3/2, with T floored so cold cells don't blow up
        T = xp.maximum(W[iPR] / xp.maximum(W[iRHO], FLOOR_RHO), cfg.eta_T_min)
        eta = xp.minimum(cfg.eta_spitzer * T**-1.5, cfg.eta_max)
        if cfg.eta_liner > 0:
            eta = _blend(eta, cfg.eta_liner, C)
    if cfg.eta_max > 0:
        eta = xp.minimum(eta, cfg.eta_max)
    return eta


def _blend(a, b, C):
    # log blend keeps a mixed cell between two very different conductivities from being
    # dominated by the larger eta; falls back to linear if either side is zero
    if isinstance(a, float) and isinstance(b, float) and (a <= 0 or b <= 0):
        return (1 - C) * a + C * b
    a = xp.maximum(a, 1e-300); b = xp.maximum(b, 1e-300)
    return xp.exp((1 - C) * xp.log(a) + C * xp.log(b))


def eta_max_of(eta):
    return float(eta) if isinstance(eta, (int, float)) else float(xp.max(eta))


def face_eta(eta, axis, mean="harmonic"):
    """eta on the faces between neighbouring cells along `axis`. Harmonic mean by default:
    it gives the right series resistance across a conductivity jump (a face between a good
    and a bad conductor carries the current the good one allows, not the average)."""
    if isinstance(eta, (int, float)):
        return eta
    n = eta.shape[axis]
    a = xp.take(eta, xp.arange(0, n - 1), axis=axis)
    b = xp.take(eta, xp.arange(1, n), axis=axis)
    if mean == "arithmetic":
        return 0.5 * (a + b)
    return 2.0 * a * b / xp.maximum(a + b, 1e-300)


def compute_resistive_terms(U, eta, dx, dy, face_mean="harmonic", rc=None, rf=None, bz_flux=False):
    """Resistive induction and energy terms in conservative form.

    Induction: dB/dt = -curl(eta J), J = curl(B), with all three current
    components of the 2.5D problem (dz = 0).
    Energy:    dE/dt = -div(eta J x B)   (resistive Poynting flux).

    eta is a float or a cell-centred array; on faces it is averaged with face_eta.
    Both terms are differences of face quantities, so the energy update telescopes
    exactly and Ohmic heating eta|J|^2 shows up in the pressure through the drop in
    magnetic energy -- it is not added to E separately.

    Returns (R, Gx, Gy, q): R has shape U.shape with nonzero BX, BY, BZ, EN
    entries on cells [1:-1, 1:-1]; Gx (nx_tot-1, ny_tot) and Gy (nx_tot, ny_tot-1)
    are the energy fluxes on x- and y-faces; q is the Ohmic heating density on cells
    [1:-1, 1:-1] (diagnostic only), built from each current component on the faces
    where its derivative is compact. In 1D it equals the discrete magnetic energy
    loss exactly (summation by parts).
    rc, rf: cell and face radii along x for cylindrical geometry (x = r, y = z, z = -phi): then the axial
    current on r-faces is -(1/r) d(r B_w)/dr and the energy flux divergence is r-weighted.
    bz_flux: also return the B_z fluxes (E_y on x-faces, -E_x on y-faces), shaped like Gx, Gy.
    """
    R = xp.zeros_like(U)
    Gx = xp.zeros((U.shape[1] - 1, U.shape[2]), dtype=U.dtype)
    Gy = xp.zeros((U.shape[1], U.shape[2] - 1), dtype=U.dtype)
    q = xp.zeros(U.shape[1:], dtype=U.dtype)
    if isinstance(eta, (int, float)) and eta <= 0:
        return (R, Gx, Gy, q, (xp.zeros_like(Gx), xp.zeros_like(Gy))) if bz_flux else (R, Gx, Gy, q)
    ex = face_eta(eta, 0, face_mean)
    ey = face_eta(eta, 1, face_mean)

    Bx, By, Bz = U[BX], U[BY], U[BZ]

    # Cell-centred transverse derivatives (zero on the outermost ring).
    dBx_dy = xp.zeros_like(Bx); dBz_dy = xp.zeros_like(Bz)
    dBy_dx = xp.zeros_like(By); dBz_dx = xp.zeros_like(Bz)
    dBx_dy[:, 1:-1] = (Bx[:, 2:] - Bx[:, :-2]) / (2.0 * dy)
    dBz_dy[:, 1:-1] = (Bz[:, 2:] - Bz[:, :-2]) / (2.0 * dy)
    dBy_dx[1:-1, :] = (By[2:, :] - By[:-2, :]) / (2.0 * dx)
    dBz_dx[1:-1, :] = (Bz[2:, :] - Bz[:-2, :]) / (2.0 * dx)

    # Currents on x-faces (i+1/2, j): compact normal derivative, averaged transverse.
    Jx_xf = 0.5 * (dBz_dy[1:, :] + dBz_dy[:-1, :])
    if rc is None:
        Jy_xf = -(Bz[1:, :] - Bz[:-1, :]) / dx
    else:
        rcc, rff = rc[:, None], rf[1:-1, None]
        Jy_xf = -(rcc[1:] * Bz[1:, :] - rcc[:-1] * Bz[:-1, :]) / (dx * xp.where(rff == 0, 1.0, rff))
    Jz_xf = (By[1:, :] - By[:-1, :]) / dx - 0.5 * (dBx_dy[1:, :] + dBx_dy[:-1, :])

    # Currents on y-faces (i, j+1/2).
    Jx_yf = (Bz[:, 1:] - Bz[:, :-1]) / dy
    Jz_yf = 0.5 * (dBy_dx[:, 1:] + dBy_dx[:, :-1]) - (Bx[:, 1:] - Bx[:, :-1]) / dy

    # electric field E = eta J on the faces
    Ex_yf, Ez_yf = ey * Jx_yf, ey * Jz_yf
    Ey_xf, Ez_xf = ex * Jy_xf, ex * Jz_xf

    # Induction, curl form: dBx/dt = -d(Ez)/dy, dBy/dt = d(Ez)/dx,
    # dBz/dt = -d(Ey)/dx + d(Ex)/dy.
    R[BX, 1:-1, 1:-1] = -(Ez_yf[1:-1, 1:] - Ez_yf[1:-1, :-1]) / dy
    R[BY, 1:-1, 1:-1] = (Ez_xf[1:, 1:-1] - Ez_xf[:-1, 1:-1]) / dx
    R[BZ, 1:-1, 1:-1] = (-(Ey_xf[1:, 1:-1] - Ey_xf[:-1, 1:-1]) / dx
                         + (Ex_yf[1:-1, 1:] - Ex_yf[1:-1, :-1]) / dy)

    # Resistive Poynting flux E x B on faces.
    By_xf = 0.5 * (By[1:, :] + By[:-1, :]); Bz_xf = 0.5 * (Bz[1:, :] + Bz[:-1, :])
    Bx_yf = 0.5 * (Bx[:, 1:] + Bx[:, :-1]); Bz_yf = 0.5 * (Bz[:, 1:] + Bz[:, :-1])
    Gx[...] = Ey_xf * Bz_xf - Ez_xf * By_xf
    Gy[...] = Ez_yf * Bx_yf - Ex_yf * Bz_yf
    if rc is None:
        R[EN, 1:-1, 1:-1] = (-(Gx[1:, 1:-1] - Gx[:-1, 1:-1]) / dx
                             - (Gy[1:-1, 1:] - Gy[1:-1, :-1]) / dy)
    else:
        rG = Gx * rf[1:-1, None]
        R[EN, 1:-1, 1:-1] = (-(rG[1:, 1:-1] - rG[:-1, 1:-1]) / (dx * rc[1:-1, None])
                             - (Gy[1:-1, 1:] - Gy[1:-1, :-1]) / dy)

    # Ohmic heating: Jy is compact on x-faces, Jx on y-faces, Jz on both (half weight each)
    qx = Ey_xf * Jy_xf + 0.5 * Ez_xf * Jz_xf
    qy = Ex_yf * Jx_yf + 0.5 * Ez_yf * Jz_yf
    q[1:-1, 1:-1] = (0.5 * (qx[1:, 1:-1] + qx[:-1, 1:-1])
                     + 0.5 * (qy[1:-1, 1:] + qy[1:-1, :-1]))
    if bz_flux:
        return R, Gx, Gy, q, (Ey_xf, -Ex_yf)
    return R, Gx, Gy, q
