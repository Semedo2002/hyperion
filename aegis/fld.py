# fld.py - grey flux-limited diffusion (FLD) radiation transport, 2D (Cartesian or r-z) and 3D.
#
#   dE_r/dt + div(v E_r) = div(D grad E_r) + c rho kappa_P (a T^4 - E_r),   gas: de/dt = -c rho kappa_P (a T^4 - E_r)
#   D = c lambda(R) / (rho kappa_R),  R = |grad E_r| / (rho kappa_R E_r),  lambda = (2 + R) / (6 + 3 R + R^2)
#
# Levermore & Pomraning (1981): lambda -> 1/3 when optically thick (D = c / (3 rho kappa_R)), lambda -> 1/R when
# thin, so the flux D |grad E_r| never exceeds c E_r. T is the code temperature theta = p/rho, a and c are in code
# units (units.fld_kwargs), opacities are power laws kappa = k0 rho^m theta^n per unit mass, or a table
# (fld_opacity_table: SESAME 502/505 or .npz, aegis/tabular.py).
#
# Each step (split, after the hydro): upwind advection of E_r with the gas, then the stiff part by backward Euler
# with the emission linearised, a (theta + dtheta)^4 = a theta^4 + 4 a theta^3 dtheta, and the limiter taken
# from the latest iterate over a few Picard passes (fld_picard). Eliminating dtheta
# leaves one symmetric positive-definite system for E_r^{n+1} (volume weighted, so it stays symmetric in r-z),
# solved by matrix-free conjugate gradients on the GPU, preconditioned by Jacobi or (fld_precond = "mg") a
# geometric multigrid V-cycle (aegis/radiation_gmg.py). The update is then written
# from the solution in face-flux form plus the cell exchange that the gas receives with the opposite sign, so
# gas + radiation energy is conserved to round-off whatever the CG residual. D is frozen at the old E_r.
# Radiation pressure and its work are left out (fine while E_r / 3 << p). Radiation lives in the fuel only
# (liner fraction below fld_tracer_max): faces touching the liner carry no radiation flux.
#
# An AMR patch (aegis/amr.py) passes `ghost`: E_r and the normal velocity in the cells just outside each of its
# edges, taken from the coarse level. Those faces then conduct to the ghost cell (a full cell away) and advect
# across it; `faces` receives the time-integrated face fluxes for refluxing.
import numpy as np

from .config import EN, iRHO, iPR, iVX, iCLR
from .physics import xp, cons_to_prim


def kappa(k0, m, n, rho, theta):
    return k0 * rho**m * xp.maximum(theta, 1e-30)**n


def _limiter(R):
    return (2.0 + R) / (6.0 + 3.0 * R + R * R)


def _sl(nd, ax, s):
    idx = [slice(None)] * nd
    idx[ax] = s
    return tuple(idx)


class Geometry:
    """cell volumes and face areas of the interior grid (per radian in r-z, radius along axis 0)"""

    def __init__(self, shape, h, periodic, radial=None):
        self.shape, self.h, self.per, self.nd = tuple(shape), tuple(h), tuple(periodic), len(shape)
        col = lambda a: a.reshape((-1,) + (1,) * (self.nd - 1))
        other = lambda ax: float(np.prod([h[k] for k in range(self.nd) if k != ax]))
        if radial is None:
            self.V = float(np.prod(h))
            self.A = [other(ax) for ax in range(self.nd)]
        else:
            rf, rc = radial
            self.V = col(rc) * float(np.prod(h))
            self.A = [col(rf) * other(0)] + [col(rc) * other(ax) for ax in range(1, self.nd)]

    def pairs(self, f, ax):
        """(left, right) cell values on the n + 1 faces normal to ax; end faces wrap (periodic) or see the
        edge cell on both sides"""
        n = f.shape[ax]
        if self.per[ax]:
            lo = xp.concatenate([f[_sl(self.nd, ax, slice(n - 1, n))], f], axis=ax)
            hi = xp.concatenate([f, f[_sl(self.nd, ax, slice(0, 1))]], axis=ax)
        else:
            lo = xp.concatenate([f[_sl(self.nd, ax, slice(0, 1))], f], axis=ax)
            hi = xp.concatenate([f, f[_sl(self.nd, ax, slice(n - 1, n))]], axis=ax)
        return lo, hi

    def divergence(self, G):
        """G[ax]: energy per time through each face toward +ax -> net energy per time into each cell, and the
        net inflow through the non-periodic domain faces"""
        net, inflow = 0.0, 0.0
        for ax, g in enumerate(G):
            n = self.shape[ax]
            net = net + g[_sl(self.nd, ax, slice(0, n))] - g[_sl(self.nd, ax, slice(1, n + 1))]
            if not self.per[ax]:
                inflow = inflow + xp.sum(g[_sl(self.nd, ax, slice(0, 1))]) - xp.sum(g[_sl(self.nd, ax, slice(n, n + 1))])
        return net, inflow


def _ends(geo, ax, lo, hi, vals, end_key, value=None):
    """put the ghost (or Dirichlet) values of axis ax into the end entries of the (left, right) face pairs"""
    n = geo.shape[ax]
    for end, idx, arr in (("lo", 0, lo), ("hi", n, hi)):
        if (ax, end) in vals:
            v = vals[(ax, end)] if value is None else value
            arr[_sl(geo.nd, ax, slice(idx, idx + 1))] = v if end_key is None else v[end_key]


def face_conductance(geo, E, chiR, c, open_, dirichlet, ghost=None):
    """D A / h on every face with the limiter evaluated on the face: R = |grad E| / (chi E) from the normal
    difference across the face (to the boundary value, half a cell out, on a Dirichlet face) plus the
    transverse cell gradients averaged onto it, E and chi the face means. Zero where either side is closed
    (liner) and on non-periodic end faces without a Dirichlet value (zero flux). ghost: E in the cells beyond
    the ends ({(ax, "lo"/"hi"): {"E": ...}}), a full cell away."""
    nd = geo.nd
    ghost = ghost or {}
    gc = []                                          # cell-centred gradients, for the transverse parts
    for ax in range(nd):
        lo, hi = geo.pairs(E, ax)
        d = (hi - lo) / geo.h[ax]
        gc.append(0.5 * (d[_sl(nd, ax, slice(0, -1))] + d[_sl(nd, ax, slice(1, None))]))
    K = []
    for ax in range(nd):
        n = geo.shape[ax]
        elo, ehi = geo.pairs(E, ax)
        hn = xp.full(elo.shape, geo.h[ax])
        if not geo.per[ax]:
            for end, idx, arr in (("lo", 0, elo), ("hi", n, ehi)):
                if (ax, end) in dirichlet:
                    arr[_sl(nd, ax, slice(idx, idx + 1))] = dirichlet[(ax, end)]
                    hn[_sl(nd, ax, slice(idx, idx + 1))] = 0.5 * geo.h[ax]
            _ends(geo, ax, elo, ehi, ghost, "E")
        g2 = ((ehi - elo) / hn)**2
        for t in range(nd):
            if t != ax:
                a, b = geo.pairs(gc[t], ax)
                g2 = g2 + (0.5 * (a + b))**2
        clo, chi = geo.pairs(chiR, ax)
        olo, ohi = geo.pairs(open_, ax)
        chif = 0.5 * (clo + chi)
        R = xp.sqrt(g2) / xp.maximum(chif * 0.5 * (elo + ehi), 1e-300)
        Df = xp.where(olo & ohi, c * _limiter(R) / xp.maximum(chif, 1e-300), 0.0)
        if not geo.per[ax]:
            for end, idx in (("lo", 0), ("hi", n)):
                if (ax, end) not in dirichlet and (ax, end) not in ghost:
                    Df[_sl(nd, ax, slice(idx, idx + 1))] = 0.0
        K.append(Df * geo.A[ax] / hn)
    return K


def diffusive(geo, E, K, dirichlet, boundary=True, ghost=None):
    """energy per time through each face, K (E_left - E_right) (toward +ax); Dirichlet and ghost faces use the
    boundary value (or 0 with boundary=False: the homogeneous operator)"""
    nd, G = geo.nd, []
    ghost = ghost or {}
    for ax in range(nd):
        lo, hi = geo.pairs(E, ax)
        n = geo.shape[ax]
        if not geo.per[ax]:
            if (ax, "lo") in dirichlet:
                lo[_sl(nd, ax, slice(0, 1))] = dirichlet[(ax, "lo")] if boundary else 0.0
            if (ax, "hi") in dirichlet:
                hi[_sl(nd, ax, slice(n, n + 1))] = dirichlet[(ax, "hi")] if boundary else 0.0
            _ends(geo, ax, lo, hi, ghost, "E", None if boundary else {"E": 0.0})
        G.append(K[ax] * (lo - hi))
    return G


def advective(geo, Er, W, h, ghost=None):
    """upwind energy per time v E_r through each face (cell velocities averaged onto the face; end faces of a
    non-periodic axis carry the edge cell outward, nothing inward, unless a ghost cell is given)"""
    G = []
    ghost = ghost or {}
    for ax in range(geo.nd):
        vlo, vhi = geo.pairs(W[iVX + ax], ax)
        elo, ehi = geo.pairs(Er, ax)
        _ends(geo, ax, vlo, vhi, ghost, "v")
        _ends(geo, ax, elo, ehi, ghost, "E")
        vf = 0.5 * (vlo + vhi)
        F = vf * xp.where(vf > 0, elo, ehi)
        if not geo.per[ax]:
            n = geo.shape[ax]
            if (ax, "lo") not in ghost:
                F[_sl(geo.nd, ax, slice(0, 1))] = xp.minimum(F[_sl(geo.nd, ax, slice(0, 1))], 0.0)
            if (ax, "hi") not in ghost:
                F[_sl(geo.nd, ax, slice(n, n + 1))] = xp.maximum(F[_sl(geo.nd, ax, slice(n, n + 1))], 0.0)
        G.append(F * geo.A[ax])
    return G


_TABLES = {}


def _opacity_table(cfg):
    key = (cfg.fld_opacity_table, cfg.tab_rho_unit, cfg.tab_L_unit, cfg.tab_T_unit)
    if key not in _TABLES:
        from . import tabular
        _TABLES[key] = tabular.OpacityTable(tabular.load(cfg.fld_opacity_table), cfg.tab_rho_unit, cfg.tab_L_unit,
                                            cfg.tab_T_unit)
    return _TABLES[key]


def _cg(op, b, x0, Minv, tol, maxit):
    x = x0.copy()
    r = b - op(x)
    M = Minv if callable(Minv) else (lambda v: Minv * v)
    z = M(r)
    p = z.copy()
    rz = xp.sum(r * z)
    bn = float(xp.sqrt(xp.sum(b * b))) or 1.0
    for it in range(maxit):
        if float(xp.sqrt(xp.sum(r * r))) <= tol * bn:
            return x, it
        Ap = op(p)
        alpha = rz / xp.sum(p * Ap)
        x += alpha * p
        r -= alpha * Ap
        z = M(r)
        rz_new = xp.sum(r * z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, maxit


def step(Er, U, eos, cfg, h, periodic, dt, radial=None, advect=True, ghost=None, faces=None):
    """one FLD step on interior arrays, in place on Er and U[EN]. Returns (energy the gas gained, energy that
    entered through the domain faces, CG iterations): cell integrals (per radian in r-z).
    ghost: {(axis, "lo"/"hi"): {"E": E_r, "v": normal velocity}} in the cells beyond those ends (AMR patches);
    faces: a list that receives the time-integrated E_r face fluxes per axis (energy per area, Cartesian)."""
    geo = Geometry(Er.shape, h, periodic, radial)
    W = cons_to_prim(U, eos)
    rho = W[iRHO]
    theta = xp.maximum(W[iPR], 0.0) / rho
    open_ = xp.clip(W[iCLR], 0.0, 1.0) < cfg.fld_tracer_max
    V = geo.V
    inflow = 0.0
    ghost = ghost or {}
    Gadv = None
    if advect:
        Gadv = advective(geo, Er, W, h, ghost)
        net, inflow = geo.divergence(Gadv)
        Er += dt * net / V
        inflow = dt * inflow
    dirichlet = {}
    if not periodic[0]:
        if cfg.fld_E_xmin >= 0:
            dirichlet[(0, "lo")] = cfg.fld_E_xmin
        if cfg.fld_E_xmax >= 0:
            dirichlet[(0, "hi")] = cfg.fld_E_xmax
    for key in ghost:
        dirichlet.pop(key, None)
    if cfg.fld_opacity_table:                        # tabulated kappa_R, kappa_P (aegis/tabular.py)
        kR, kP = _opacity_table(cfg)(rho, theta)
        kP = xp.where(open_, kP, 0.0)
        chiR = rho * kR
    else:
        kP = xp.where(open_, kappa(cfg.fld_kp0, cfg.fld_kp_rho, cfg.fld_kp_T, rho, theta), 0.0)
        chiR = rho * kappa(cfg.fld_kr0, cfg.fld_kr_rho, cfg.fld_kr_T, rho, theta)
    a, c = cfg.fld_a, cfg.fld_c
    cv = 1.0 / (cfg.gamma - 1.0)                     # fuel: ideal gas, e = rho cv theta
    B = a * theta**4
    s = c * rho * kP * cv / (cv + 4 * a * theta**3 * dt * c * kP)   # absorption rate with the gas eliminated
    diag_ = V * (1.0 + dt * s)
    nd = geo.nd
    zero = xp.zeros_like(Er)
    E1, its = Er, 0
    # Picard passes: the limiter from the latest iterate (frozen at the old E_r, the flat region ahead of a
    # front would see the unlimited D = c / (3 chi) and flood within one step)
    for _ in range(max(cfg.fld_picard, 1)):
        K = [dt * k for k in face_conductance(geo, xp.maximum(E1, 1e-300), chiR, c, open_, dirichlet, ghost)]

        def op(x):                                   # V (1 + dt s) x + sum over faces of K (x_i - x_j)
            net, _ = geo.divergence(diffusive(geo, x, K, dirichlet, boundary=False, ghost=ghost))
            return diag_ * x - net
        bnet, _ = geo.divergence(diffusive(geo, zero, K, dirichlet, ghost=ghost))  # boundary values -> rhs
        rhs = V * (Er + dt * s * B) + bnet
        kd = sum(k[_sl(nd, ax, slice(0, geo.shape[ax]))] + k[_sl(nd, ax, slice(1, geo.shape[ax] + 1))]
                 for ax, k in enumerate(K))
        if cfg.fld_precond == "mg":                  # geometric multigrid V-cycle (aegis/radiation_gmg.py)
            from .radiation_gmg import Multigrid
            mgv = Multigrid(xp.broadcast_to(diag_, Er.shape)[None], [k[None] for k in K], geo.per)
            Minv = lambda v: mgv(v[None])[0]
        else:
            Minv = 1.0 / (diag_ + kd)
        E1, it = _cg(op, rhs, E1, Minv, cfg.fld_tol, cfg.fld_maxit)
        its += it
    # conservative update from the solution: face fluxes telescope, the gas gets the exchange back
    Gd = diffusive(geo, E1, K, dirichlet, ghost=ghost)
    net, din = geo.divergence(Gd)
    if faces is not None:
        for ax in range(nd):
            G = Gd[ax] if Gadv is None else Gd[ax] + dt * Gadv[ax]
            faces.append(G / geo.A[ax])
    exch = dt * s * (B - E1) * V                     # energy the radiation takes from the gas, per cell
    Er += (net + exch) / V
    U[EN] -= exch / V
    return -float(xp.sum(exch)), float(inflow) + float(din), its
