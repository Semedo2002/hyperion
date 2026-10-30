# radiation_multigroup.py - multigroup flux-limited diffusion: the grey FLD of fld.py split over photon groups.
#
#   dE_g/dt + div(v E_g) = div(D_g grad E_g) + c rho kP_g (a T^4 b_g(T) - E_g),   gas: de/dt = -sum_g (same)
#   D_g = c lambda(R_g) / (rho kR_g),   b_g(T) = fraction of the Planck spectrum between the group edges
#
# Groups are bands of photon energy eps in code temperature units (eps / theta = h nu / k T); the first starts
# at 0 and the last runs to infinity, so sum_g b_g = 1 and a uniform medium relaxes to a Planck spectrum.
# Opacities per group (Planck mean kP_g, Rosseland mean kR_g):
#   "grey"     the grey power laws of fld.py in every group (the sum over groups then follows grey FLD)
#   "kramers"  kappa_nu = kP_grey(rho, theta) (pi^4/15) x^-3 (1 - e^-x), x = eps/theta (free-free shape, scaled so
#              its Planck mean over all frequencies is the grey kP), times (1 + fld_edge_jump) above a
#              photo-ionization edge at fld_edge; group means by Gauss-Legendre quadrature in each cell
#   table      kR_g, kP_g (rho, T) from a group-opacity .npz (tools/generate_sesame.py --groups)
#
# Each step (split, after the hydro): upwind advection of every group, then backward Euler with the emission
# linearised, a T^4 b_g(T + dT) = B_g + beta_g dT, beta_g = d(a T^4 b_g)/dT (sum_g beta_g = 4 a T^3), and dT
# eliminated through the gas energy equation. Per cell this couples the groups through one rank-one term, so the
# system is solved by BiCGStab (matrix-free, all groups at once, on the GPU) preconditioned with the exact inverse
# of each cell's group block (diagonal + rank one, Sherman-Morrison). The update is then written from the solution
# in face-flux form plus the exchange the gas receives with the opposite sign, so gas + radiation energy is
# conserved to round-off whatever the Krylov residual. The limiter is taken per group and per face as in fld.py.
import numpy as np

from .config import EN, iRHO, iPR, iCLR
from .physics import xp, cons_to_prim
from . import fld

_GL = np.polynomial.legendre.leggauss(8)


def planck_cdf(x):
    """fraction of the Planck energy spectrum below x = h nu / k T (series for small x, exponential sums above)"""
    x = xp.asarray(x, dtype=float)
    c = 15.0 / np.pi**4
    small = x < 2.0
    xs = xp.minimum(x, 2.0)
    s = xs**3 / 3 - xs**4 / 8 + xs**5 / 60 - xs**7 / 5040 + xs**9 / 272160 - xs**11 / 13305600
    xl = xp.maximum(x, 2.0)
    big = 0.0
    for n in range(1, 16):
        big = big + xp.exp(-n * xl) * (xl**3 / n + 3 * xl**2 / n**2 + 6 * xl / n**3 + 6 / n**4)
    return xp.where(small, c * s, 1.0 - c * big)


def planck_density(x):
    """d(planck_cdf)/dx = (15/pi^4) x^3 / (e^x - 1)"""
    x = xp.maximum(xp.asarray(x, dtype=float), 1e-300)
    return 15.0 / np.pi**4 * x**3 / xp.expm1(xp.minimum(x, 700.0))


def edges_of(cfg):
    """group edges in code temperature units, first 0 and last inf (from the table when one is given: eV ->
    K -> code temperature through tab_T_unit)"""
    if cfg.fld_mg_table:
        e = np.load(cfg.fld_mg_table)["group_edges_eV"] * 11604.518 / cfg.tab_T_unit
        e[0], e[-1] = 0.0, np.inf
        return e
    if cfg.fld_group_edges:
        inner = list(cfg.fld_group_edges)
    else:
        inner = list(np.geomspace(cfg.fld_group_emin, cfg.fld_group_emax, cfg.fld_groups - 1))
    return np.array([0.0] + inner + [np.inf])


def planck_groups(edges, theta, a):
    """(B_g = a theta^4 b_g, beta_g = dB_g/dtheta) for every group: arrays (G, ...)"""
    t = xp.maximum(theta, 1e-30)
    cdf, dcdT = [], []
    for e in edges:
        if e == 0.0:
            cdf.append(xp.zeros_like(t)); dcdT.append(xp.zeros_like(t))
        elif np.isinf(e):
            cdf.append(xp.ones_like(t)); dcdT.append(xp.zeros_like(t))
        else:
            x = e / t
            cdf.append(planck_cdf(x)); dcdT.append(-planck_density(x) * x / t)
    b = xp.stack([cdf[g + 1] - cdf[g] for g in range(len(edges) - 1)])
    db = xp.stack([dcdT[g + 1] - dcdT[g] for g in range(len(edges) - 1)])
    aT4 = a * t**4
    return aT4 * b, 4 * a * t**3 * b + aT4 * db


def _kramers_means(edges, theta, kP_grey, edge, jump):
    """group Planck and Rosseland means of kappa_nu = kP_grey (pi^4/15) x^-3 (1 - e^-x) (1 + jump [eps > edge])"""
    t = xp.maximum(theta, 1e-30)
    nodes, wts = _GL
    kP, kR = [], []
    for g in range(len(edges) - 1):
        lo = edges[g] / t
        hi = edges[g + 1] / t if np.isfinite(edges[g + 1]) else lo + 60.0
        lo = xp.maximum(lo, 1e-6) if edges[g] == 0.0 else lo
        num_p = den_p = num_r = den_r = 0.0
        # log-spaced sub-intervals so the steep ends of a wide group are resolved
        nsub = 4
        L = xp.log(xp.maximum(lo, 1e-6)); H = xp.log(xp.maximum(hi, 1e-6 * 1.0001))
        for k in range(nsub):
            a_, b_ = L + (H - L) * k / nsub, L + (H - L) * (k + 1) / nsub
            for xn, wn in zip(nodes, wts):
                lx = 0.5 * (a_ + b_) + 0.5 * (b_ - a_) * xn
                x = xp.exp(lx)
                w = wn * 0.5 * (b_ - a_) * x                  # dx = x dlnx
                shape = (np.pi**4 / 15) * (-xp.expm1(-x)) / x**3
                if edge > 0:
                    shape = shape * (1.0 + jump * (x * t > edge))
                kap = kP_grey * shape
                Bw = planck_density(x)                       # Planck weight
                Rw = Bw * x / -xp.expm1(-x)                  # dB/dT weight: x^4 e^x / (e^x - 1)^2
                num_p = num_p + w * kap * Bw
                den_p = den_p + w * Bw
                num_r = num_r + w * Rw / xp.maximum(kap, 1e-300)
                den_r = den_r + w * Rw
        kP.append(num_p / xp.maximum(den_p, 1e-300))
        kR.append(den_r / xp.maximum(num_r, 1e-300))
    return xp.stack(kP), xp.stack(kR)


_TABLES = {}


def group_opacities(cfg, edges, rho, theta, open_):
    """(kP_g, kR_g) per unit mass, code units, arrays (G, ...); kP zero in the liner"""
    G = len(edges) - 1
    model = cfg.fld_mg_opacity
    if cfg.fld_mg_table:
        key = (cfg.fld_mg_table, cfg.tab_rho_unit, cfg.tab_L_unit, cfg.tab_T_unit)
        if key not in _TABLES:
            from . import tabular
            d = np.load(cfg.fld_mg_table)
            tabs = [tabular.OpacityTable({"rho": d["rho"], "T": d["T"], "kappa_R": d["kappa_R_g"][g],
                                          "kappa_P": d["kappa_P_g"][g]}, cfg.tab_rho_unit, cfg.tab_L_unit,
                                         cfg.tab_T_unit) for g in range(d["kappa_R_g"].shape[0])]
            _TABLES[key] = tabs
        kk = [tb(rho, theta) for tb in _TABLES[key]]
        kR = xp.stack([k[0] for k in kk])
        kP = xp.stack([k[1] for k in kk])
    else:
        kPg = fld.kappa(cfg.fld_kp0, cfg.fld_kp_rho, cfg.fld_kp_T, rho, theta)
        if model == "grey":
            kRg = fld.kappa(cfg.fld_kr0, cfg.fld_kr_rho, cfg.fld_kr_T, rho, theta)
            kP = xp.stack([kPg] * G)
            kR = xp.stack([kRg] * G)
        else:
            kP, kR = _kramers_means(edges, theta, kPg, cfg.fld_edge, cfg.fld_edge_jump)
    return xp.where(open_[None], kP * cfg.fld_mg_absorption, 0.0), kR


def _bicgstab(op, b, x0, prec, tol, maxit):
    x = x0.copy()
    r = b - op(x)
    rh = r.copy()
    rho_o = alpha = w = 1.0
    v = p = xp.zeros_like(b)
    bn = float(xp.sqrt(xp.sum(b * b))) or 1.0
    for it in range(maxit):
        if float(xp.sqrt(xp.sum(r * r))) <= tol * bn:
            return x, it
        rho_n = float(xp.sum(rh * r))
        if rho_n == 0.0:
            return x, it
        beta = (rho_n / rho_o) * (alpha / w)
        p = r + beta * (p - w * v)
        ph = prec(p)
        v = op(ph)
        alpha = rho_n / float(xp.sum(rh * v))
        s = r - alpha * v
        sh = prec(s)
        tv = op(sh)
        tt = float(xp.sum(tv * tv))
        w = float(xp.sum(tv * s)) / tt if tt > 0 else 0.0
        x = x + alpha * ph + w * sh
        r = s - w * tv
        rho_o = rho_n
        if w == 0.0:
            return x, it + 1
    return x, maxit


RECORD = None                                        # a list here collects (E^n, theta, rho, B_g, dt, E^{n+1})


def step(Er, U, eos, cfg, h, periodic, dt, radial=None, advect=True):
    """one multigroup step on interior arrays, in place on Er (G, ...) and U[EN]. Returns (energy the gas gained,
    energy that entered through the domain faces, Krylov iterations)."""
    G = Er.shape[0]
    geo = fld.Geometry(Er.shape[1:], h, periodic, radial)
    W = cons_to_prim(U, eos)
    rho = W[iRHO]
    theta = xp.maximum(W[iPR], 0.0) / rho
    open_ = xp.clip(W[iCLR], 0.0, 1.0) < cfg.fld_tracer_max
    V = geo.V
    inflow = 0.0
    if advect:
        for g in range(G):
            net, din = geo.divergence(fld.advective(geo, Er[g], W, h))
            Er[g] += dt * net / V
            inflow = inflow + dt * din
    dirichlet = {}
    if not periodic[0]:                              # an imposed radiation energy splits by the Planck fractions
        for end, val in (("lo", cfg.fld_E_xmin), ("hi", cfg.fld_E_xmax)):
            if val >= 0:
                dirichlet[(0, end)] = val
    edges = edges_of(cfg)
    a, c = cfg.fld_a, cfg.fld_c
    cv = 1.0 / (cfg.gamma - 1.0)
    kP, kR = group_opacities(cfg, edges, rho, theta, open_)
    chiR = rho[None] * kR
    B, beta = planck_groups(edges, theta, a)
    w = c * rho[None] * kP                           # absorption rate per group
    Gam = cv * rho / dt + xp.sum(w * beta, axis=0)   # gas energy equation, linearised
    cpl = dt * V / Gam                               # rank-one coupling weight
    u = w * beta
    E0 = Er.copy()
    if dirichlet:                                    # boundary values per group: the Planck share at the edge
        bfrac = {}
        for key, val in dirichlet.items():
            ax, end = key
            idx = [slice(None)] * geo.nd
            idx[ax] = slice(0, 1) if end == "lo" else slice(-1, None)
            tb = theta[tuple(idx)]
            Bb, _ = planck_groups(edges, tb, 1.0)
            bfrac[key] = Bb / xp.maximum(xp.sum(Bb, axis=0), 1e-300) * val
    E1, its, first = Er.copy(), 0, True
    for _ in range(max(cfg.fld_picard, 1)):
        K, dg = [], []
        for g in range(G):
            dg_ = {k: v[g] for k, v in bfrac.items()} if dirichlet else {}
            Kg = [dt * k for k in fld.face_conductance(geo, xp.maximum(E1[g], 1e-300), chiR[g], c, open_, dg_)]
            K.append(Kg); dg.append(dg_)
        kd = xp.stack([sum(k[_sl(geo.nd, ax, slice(0, geo.shape[ax]))] + k[_sl(geo.nd, ax, slice(1, geo.shape[ax] + 1))]
                           for ax, k in enumerate(K[g])) for g in range(G)])
        diag = V * (1.0 + dt * w)

        def op(x):                                   # V (1 + dt w_g) x_g - V dt u_g (sum_h w_h x_h) / Gam - div
            s = xp.sum(w * x, axis=0)
            out = diag * x - cpl * u * s
            for g in range(G):
                net, _ = geo.divergence(fld.diffusive(geo, x[g], K[g], dg[g], boundary=False))
                out[g] -= net
            return out
        rhs = V * E0 + dt * V * w * B - cpl * u * xp.sum(w * B, axis=0)
        for g in range(G):
            bnet, _ = geo.divergence(fld.diffusive(geo, xp.zeros_like(E0[g]), K[g], dg[g]))
            rhs[g] += bnet
        D = diag + kd                                # per-cell block: diag(D) - cpl u w^T, Sherman-Morrison

        def prec(r):
            Di = r / D
            den = 1.0 - cpl * xp.sum(w * u / D, axis=0)
            return Di + (u / D) * (cpl * xp.sum(w * Di, axis=0) / den)
        if cfg.fld_precond == "mg":                  # geometric multigrid V-cycle (aegis/radiation_gmg.py)
            from .radiation_gmg import Multigrid
            prec = Multigrid(xp.broadcast_to(diag, E0.shape), [xp.stack([K[g][ax] for g in range(G)])
                                                                 for ax in range(geo.nd)], geo.per,
                             cu=cpl * u, w=w)
        if cfg.fld_fno_model and first and geo.nd == 2:
            # learned predictor (aegis/fno_subgrid.py), guarded by the residual of this linear system
            from . import fno_subgrid as fs
            x0 = fs.radiation_guess(cfg, E0, theta, rho, B, dt)
            bn = max(float(xp.sqrt(xp.sum(rhs * rhs))), 1e-300)
            res = lambda x: float(xp.sqrt(xp.sum((rhs - op(x))**2))) / bn
            rel, rel0 = res(x0), res(E1)
            fs.Stats.residuals.append(rel)
            fs.Stats.default_residuals.append(rel0)
            if rel <= cfg.fld_fno_tol:
                E1, it = x0, 0
                fs.Stats.accepted += 1
            else:                                    # exact solve, from the prediction only if clearly better
                E1, it = _bicgstab(op, rhs, x0 if rel < 0.5 * rel0 else E1, prec, cfg.fld_tol, cfg.fld_maxit)
                fs.Stats.fallback += 1
            fs.Stats.iterations += it
        else:
            E1, it = _bicgstab(op, rhs, E1, prec, cfg.fld_tol, cfg.fld_maxit)
        its += it
        first = False
        if RECORD is not None and not cfg.fld_fno_model:   # training data for the predictor: exact solutions
            RECORD.append(tuple(a.copy() for a in (E0, theta, rho, B)) + (dt, E1.copy()))
    # conservative update from the solution: face fluxes telescope per group, the gas gets the exchange back
    dT = xp.sum(w * (E1 - B), axis=0) / Gam
    exch_tot = 0.0
    din_tot = 0.0
    for g in range(G):
        net, din = geo.divergence(fld.diffusive(geo, E1[g], K[g], dg[g]))
        exch = dt * w[g] * (B[g] + beta[g] * dT - E1[g]) * V
        Er[g] = E0[g] + (net + exch) / V
        exch_tot = exch_tot + exch
        din_tot = din_tot + din
    U[EN] -= exch_tot / V
    return -float(xp.sum(exch_tot)), float(inflow) + float(din_tot), its


def _sl(nd, ax, s):
    idx = [slice(None)] * nd
    idx[ax] = s
    return tuple(idx)
