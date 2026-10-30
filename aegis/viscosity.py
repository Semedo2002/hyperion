# viscosity.py - Braginskii parallel ion viscosity (strongly magnetized limit), Cartesian 2D and 3D.
#
#     Pi = -3 eta_0 (b b - I/3) (b b : grad v - div v / 3),   eta_0 = visc_coeff theta_i^5/2
# (Braginskii 1965; the form used for magnetized plasmas, e.g. Dong & Stone 2009). Where |B| = 0 the stress is
# the isotropic Navier-Stokes one with the same eta. Momentum and energy (Pi . v) are face fluxes, so total
# energy is conserved exactly and the viscous heating appears as thermal (ion) energy. Velocity gradients on a
# face: compact difference along the normal, averaged centred differences across it. Integrated as a split
# step with RKL2, liner faces and non-periodic domain faces carry no stress.
from .config import RHO, MX, MY, MZ, EN, iBX, iBY, iBZ, FLOOR_RHO
from .physics import xp, _along, _pad1, rkl2


def stress_rates(rho, m, E, B, eta, h, periodic, fuel, ghost=None, fluxes=None):
    """(d m / dt for the three components, dE/dt) from the viscous stress, interior arrays. ghost: one-cell
    padded (v, B, eta, fuel) whose outer ring is used as neighbour data, the domain faces then carrying stress
    (an AMR patch); fluxes: a list receiving, per axis, the stacked face fluxes (momentum x, y, z, energy)"""
    nd = rho.ndim
    v = [mk / rho for mk in m]
    if ghost is None:
        vp = [_pad1(f, periodic) for f in v]
        Bp = [_pad1(f, periodic) for f in B]
        ep, fp = _pad1(eta, periodic), _pad1(fuel.astype(rho.dtype), periodic)
    else:
        core_ = (slice(1, -1),) * nd
        vp = []
        for f, g in zip(v, ghost[0]):
            q = g.copy()
            q[core_] = f
            vp.append(q)
        Bp, ep, fp = ghost[1], ghost[2], ghost[3]
    dm = [xp.zeros_like(rho) for _ in range(3)]
    dE = xp.zeros_like(rho)
    core = slice(1, -1)
    for ax in range(nd):
        L, R = _along(nd, ax, slice(0, -1), core), _along(nd, ax, slice(1, None), core)
        G = [[None] * 3 for _ in range(3)]                    # G[k][m] = d v_m / d x_k on the face
        for mm in range(3):
            for k in range(3):
                if k == ax:
                    G[k][mm] = (vp[mm][R] - vp[mm][L]) / h[ax]
                elif k < nd:
                    up, dn = list(L), list(L)
                    up[k], dn[k] = slice(2, None), slice(0, -2)
                    upR, dnR = list(R), list(R)
                    upR[k], dnR[k] = slice(2, None), slice(0, -2)
                    G[k][mm] = 0.25 * ((vp[mm][tuple(up)] - vp[mm][tuple(dn)])
                                       + (vp[mm][tuple(upR)] - vp[mm][tuple(dnR)])) / h[k]
                else:
                    G[k][mm] = 0.0
        Bf = [0.5 * (c[L] + c[R]) for c in Bp]
        Bm = xp.sqrt(Bf[0]**2 + Bf[1]**2 + Bf[2]**2)
        inv = xp.where(Bm > 0, 1.0 / xp.where(Bm > 0, Bm, 1.0), 0.0)
        b = [c * inv for c in Bf]
        divv = G[0][0] + G[1][1] + G[2][2]
        psi = sum(b[k] * b[mm] * G[k][mm] for k in range(3) for mm in range(3)) - divv / 3.0
        ef = 0.5 * (ep[L] + ep[R]) * fp[L] * fp[R]
        mag = Bm > 0
        vf = [0.5 * (c[L] + c[R]) for c in vp]
        flux_E = 0.0
        face = []
        for mm in range(3):
            d = 1.0 if mm == ax else 0.0
            Pi_par = -3.0 * ef * (b[ax] * b[mm] - d / 3.0) * psi
            Pi_iso = -ef * (G[ax][mm] + G[mm][ax] - (2.0 / 3.0) * d * divv)
            Pi = xp.where(mag, Pi_par, Pi_iso)
            if not periodic[ax] and ghost is None:
                n = Pi.shape[ax]
                Pi[_along(nd, ax, 0)] = 0.0
                Pi[_along(nd, ax, n - 1)] = 0.0
            dm[mm] -= (Pi[_along(nd, ax, slice(1, None))] - Pi[_along(nd, ax, slice(0, -1))]) / h[ax]
            flux_E = flux_E + Pi * vf[mm]
            face.append(Pi)
        dE -= (flux_E[_along(nd, ax, slice(1, None))] - flux_E[_along(nd, ax, slice(0, -1))]) / h[ax]
        if fluxes is not None:
            fluxes.append(xp.stack(face + [flux_E]))
    return dm, dE


def viscous_step(U, eta, h, periodic, fuel, dt, cfl=0.5, ghost=None, flux_out=None):
    """split step over dt on the interior conservative array U (in place: momentum and energy), eta frozen.
    Returns the RKL2 stage count. ghost: (v, B, eta, fuel) padded by one cell (see stress_rates); flux_out: a list
    receiving the time-integrated face fluxes per axis, stacked (momentum x, y, z, energy)."""
    rho = xp.maximum(U[RHO], FLOOR_RHO)
    B = (U[iBX], U[iBY], U[iBZ])
    track = flux_out is not None
    nu_max = float(xp.max(xp.where(fuel, 3.0 * eta / rho, 0.0)))
    if nu_max <= 0:
        return 0
    dt_expl = cfl / (2.0 * nu_max * sum(1.0 / hh**2 for hh in h))
    y0 = xp.stack([U[MX], U[MY], U[MZ], U[EN]])

    def rate(y):
        fl = [] if track else None
        dm, dE = stress_rates(rho, (y[0], y[1], y[2]), y[3], B, eta, h, periodic, fuel, ghost=ghost, fluxes=fl)
        out = xp.stack([dm[0], dm[1], dm[2], dE])
        return (out, fl) if track else out

    res = rkl2(y0, rate, dt, dt_expl)
    y, stages = res[0], res[1]
    if track:
        flux_out.extend(res[2])
    U[MX], U[MY], U[MZ], U[EN] = y[0], y[1], y[2], y[3]
    return stages
