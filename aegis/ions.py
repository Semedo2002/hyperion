# ions.py - ion Braginskii transport beyond the parallel viscosity (viscosity.py): gyroviscosity, and the ion
# temperature used by ion heat conduction (physics.conduct with ion_Z).
#
# Gyroviscosity is the non-dissipative finite-Larmor-radius part of the ion stress (Braginskii eta_3, eta_4 terms),
# in the compact form (e.g. Hogan 1984; Ramos 2005)
#     Pi = p_i / (4 Omega_i) [ (b x W) . (I + 3 b b) + transpose ],   (b x W)_ij = eps_ikl b_k W_lj,
#     W_ij = d_i v_j + d_j v_i - (2/3) delta_ij div v,               Omega_i = |B| / d_i,
# which for b = z gives Pi_xx = -Pi_yy = -eta_3 W_xy, Pi_xy = eta_3 (W_xx - W_yy) / 2 with eta_3 = p_i / (2 Omega_i).
# Pi : W = 0, so it moves momentum without heating. It enters the momentum and energy equations as face fluxes
# (Pi . n, and (Pi . v) . n), so momentum and total energy stay conserved. Where B = 0 it vanishes.
# Normal derivatives on a face are compact differences, transverse ones the average of the two cells' central
# differences (2.5D: d/dz = 0, all three velocity components).
from .config import iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR
from .physics import xp
from . import twotemp


def ion_pressure(W, U, cfg):
    """p_i: p - p_e in two-temperature mode, else the ion share of p at T_e = T_i, p / (1 + Z)"""
    if cfg.two_temp:
        _, p, pe = twotemp.pressures(U, cfg.gamma)
        return xp.maximum(p - pe, 0.0)
    return xp.maximum(W[iPR], 0.0) / (1.0 + cfg.z_ion)


def _stress(grad, v_div, B, coef):
    """Pi (3 x 3 nested lists) from the velocity gradient grad[i][j] = d_i v_j (d_z = 0), B and p_i / (4 Omega_i)"""
    Wt = [[grad[i][j] + grad[j][i] - (2.0 / 3.0) * v_div * (i == j) for j in range(3)] for i in range(3)]
    Bm = xp.sqrt(B[0]**2 + B[1]**2 + B[2]**2)
    inv = xp.where(Bm > 0, 1.0 / xp.where(Bm > 0, Bm, 1.0), 0.0)
    b = [c * inv for c in B]
    eps = {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}
    bW = [[sum(s * b[k] * Wt[l][j] for (i_, k, l), s in eps.items() if i_ == i) for j in range(3)] for i in range(3)]
    bWb = [bW[i][0] * b[0] + bW[i][1] * b[1] + bW[i][2] * b[2] for i in range(3)]
    A = [[bW[i][j] + 3.0 * bWb[i] * b[j] for j in range(3)] for i in range(3)]
    return [[coef * (A[i][j] + A[j][i]) for j in range(3)] for i in range(3)]


def gyroviscous_fluxes(W, U, cfg, dx, dy, nx, ny, ng):
    """face fluxes of the gyroviscous stress on a ghost-inclusive 2D state: returns
    (fx, fy), fx = [momentum x, y, z, energy] on the x-faces k = 0..nx (all rows), fy likewise on the y-faces
    (all columns), laid out like the hydro fluxes, to be subtracted as divergences"""
    v = [W[iVX], W[iVY], W[iVZ]]
    B = [W[iBX], W[iBY], W[iBZ]]
    rho = W[iRHO]
    pi = ion_pressure(W, U, cfg)
    Bm = xp.sqrt(B[0]**2 + B[1]**2 + B[2]**2)
    coef_c = pi * cfg.gyro_di / (4.0 * xp.where(Bm > 0, Bm, 1.0))      # p_i / (4 Omega_i) on cells
    Nx, Ny = rho.shape
    # cell-centred central differences (ghost ring: one-sided zero)
    def ddx(f):
        g = xp.zeros_like(f)
        g[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dx)
        return g

    def ddy(f):
        g = xp.zeros_like(f)
        g[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dy)
        return g
    dvx = [ddx(c) for c in v]
    dvy = [ddy(c) for c in v]
    L, R = slice(ng - 1, ng + nx), slice(ng, ng + nx + 1)           # x-faces k = 0..nx: cells left / right
    D, Up = slice(ng - 1, ng + ny), slice(ng, ng + ny + 1)          # y-faces
    zero = lambda shape: xp.zeros(shape)
    fx = [zero((nx + 1, Ny)) for _ in range(4)]
    fy = [zero((Nx, ny + 1)) for _ in range(4)]
    rows, cols = slice(1, Ny - 1), slice(1, Nx - 1)
    # x-faces (rows with neighbours)
    m = lambda f: 0.5 * (f[L, rows] + f[R, rows])
    gx = [[(c[R, rows] - c[L, rows]) / dx for c in v],               # d_x v_j: compact
          [m(c) for c in dvy],                                       # d_y v_j: averaged
          [0.0 * m(v[0])] * 3]
    P = _stress(gx, gx[0][0] + gx[1][1], [m(c) for c in B], m(coef_c))
    vf = [m(c) for c in v]
    for j in range(3):
        fx[j][:, rows] = P[0][j]
    fx[3][:, rows] = P[0][0] * vf[0] + P[0][1] * vf[1] + P[0][2] * vf[2]
    # y-faces
    m = lambda f: 0.5 * (f[cols, D] + f[cols, Up])
    gy = [[m(c) for c in dvx],
          [(c[cols, Up] - c[cols, D]) / dy for c in v],
          [0.0 * m(v[0])] * 3]
    P = _stress(gy, gy[0][0] + gy[1][1], [m(c) for c in B], m(coef_c))
    vf = [m(c) for c in v]
    for j in range(3):
        fy[j][cols, :] = P[1][j]
    fy[3][cols, :] = P[1][0] * vf[0] + P[1][1] * vf[1] + P[1][2] * vf[2]
    return fx, fy


def dt_limit(W, U, cfg, dx, dy):
    """explicit limit of the dispersive FLR waves, omega = eta_3 k^2 / rho at the grid scale (as hall.dt_limit)"""
    Bm = xp.sqrt(W[iBX]**2 + W[iBY]**2 + W[iBZ]**2)
    eta3 = ion_pressure(W, U, cfg) * cfg.gyro_di / (2.0 * xp.where(Bm > 0, Bm, xp.inf))
    w = float(xp.max(eta3 / W[iRHO])) * (4.0 / dx**2 + 4.0 / dy**2)
    return 2.5 * cfg.cfl / w if w > 0 else float("inf")
