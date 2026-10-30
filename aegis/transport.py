# transport.py - magnetized electron transport: Braginskii (1965) Z = 1 coefficients, Nernst field advection
# and the Ettingshausen heat flux (2D, Cartesian or cylindrical, with constrained transport).
#
# With x = omega_ce tau_e and Delta = x^4 + 14.79 x^2 + 3.7703:
#   k_perp / k_par = (4.664 x^2 + 11.92) / (3.1616 Delta)        (1 at x = 0, 4.664/(3.1616 x^2) for x >> 1)
#   beta_wedge     = x (1.5 x^2 + 3.053) / Delta                 (0.81 x for x << 1, 1.5/x for x >> 1)
# Nernst: Ohm's law gains E_N = -v_N x B with v_N = -(beta_wedge/(e|B|)) grad_perp(kT_e): the field is carried
# down the temperature gradient, toward cold walls. Ettingshausen (its Onsager partner): heat flux
# q_E = (beta_wedge kT_e/e) b x J. In code units both take one coefficient c_N = T1/(B_unit L0 v0)
# (units.braginskii_kwargs): v_N = -c_N (beta_wedge/|B|) grad(theta), q_E = c_N theta (beta_wedge/|B|) B x J,
# with theta = p/rho and beta_wedge/|B| finite as B -> 0 (x is proportional to |B|).
import numpy as np

from .config import iRHO, iPR, iVX, iVY, iBX, iBY, iBZ, iCLR
from .physics import xp, braginskii_kperp_ratio


def _delta(x):
    return x**4 + 14.79 * x**2 + 3.7703


kperp_ratio = braginskii_kperp_ratio                         # k_perp / k_par for electrons, Z = 1


def beta_wedge_over_x(x):
    """Braginskii beta_wedge / x, Z = 1"""
    return (1.5 * x * x + 3.053) / _delta(x)


def x_for_kappa_ratio(R, model="braginskii"):
    """magnetization x giving k_par/k_perp = R"""
    if R <= 1:
        return 0.0
    if model == "simple":
        return float(np.sqrt(R - 1))
    lo, hi = 0.0, 1.0
    while 1.0 / kperp_ratio(hi) < R:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if 1.0 / kperp_ratio(mid) < R else (lo, mid)
    return 0.5 * (lo + hi)


def _bw_over_B(W, x0):
    """beta_wedge / |B| on cells (x0: magnetization coefficient, x = x0 |B| theta^1.5 / rho)"""
    rho = W[iRHO]
    theta = xp.maximum(W[iPR], 0.0) / rho
    Bm = xp.sqrt(W[iBX]**2 + W[iBY]**2 + W[iBZ]**2)
    xb = x0 * theta**1.5 / rho                               # x / |B|
    return xb * beta_wedge_over_x(xb * Bm), theta


def nernst_velocity(W, cN, x0, tracer_max, dx, dy, vf):
    """v_N on cells (zero outside the fuel and on the outer ring)"""
    fac, theta = _bw_over_B(W, x0)
    a = W[iCLR]
    fuel = a < tracer_max
    gx, gy = xp.zeros_like(theta), xp.zeros_like(theta)
    gx[1:-1] = (theta[2:] - theta[:-2]) / (2 * dx)
    gy[:, 1:-1] = (theta[:, 2:] - theta[:, :-2]) / (2 * dy)
    vx = xp.where(fuel, -cN * fac * gx, 0.0)
    vy = xp.where(fuel, -cN * fac * gy, 0.0)
    return vx, vy, fac, theta, fuel


def nernst_2d(W, Bf, cN, x0, tracer_max, dx, dy, nx, ny):
    """Nernst and Ettingshausen terms for the 2D solver (2 ghost layers, CT faces Bf = [Bx, By]).
    Returns (E corner EMF on x-faces 2..nx+2 x y-faces 2..ny+2, dBw/dt on interior cells for the out-of-plane
    field, Gx on x-faces 2..nx+2 x all rows, Gy on all columns x y-faces 2..ny+2, max |v_N|, div of the
    Ettingshausen flux on interior cells, and the face fluxes behind dBw and that divergence for AMR refluxing:
    (B_w flux x, y), (Ettingshausen flux x, y) in the Gx / Gy layouts)."""
    vx, vy, fac, theta, fuel = nernst_velocity(W, cN, x0, tracer_max, dx, dy, None)
    Bxf, Byf = Bf
    # corner EMF, upwind on the face fields: E = -(v_x B_y - v_y B_x)
    c = lambda f: 0.25 * (f[1:nx + 2, 1:ny + 2] + f[2:nx + 3, 1:ny + 2] + f[1:nx + 2, 2:ny + 3] + f[2:nx + 3, 2:ny + 3])
    vxc, vyc = c(vx), c(vy)
    By_up = xp.where(vxc > 0, Byf[1:nx + 2, 2:ny + 3], Byf[2:nx + 3, 2:ny + 3])
    Bx_up = xp.where(vyc > 0, Bxf[2:nx + 3, 1:ny + 2], Bxf[2:nx + 3, 2:ny + 3])
    E = -(vxc * By_up - vyc * Bx_up)
    # out-of-plane field: d Bw/dt = -div(v_N Bw), upwind fluxes
    Bw = W[iBZ]
    vxf = 0.5 * (vx[1:nx + 2] + vx[2:nx + 3])                # x-faces 2..nx+2, all rows
    Fx = vxf * xp.where(vxf > 0, Bw[1:nx + 2], Bw[2:nx + 3])
    vyf = 0.5 * (vy[:, 1:ny + 2] + vy[:, 2:ny + 3])          # all columns, y-faces 2..ny+2
    Fy = vyf * xp.where(vyf > 0, Bw[:, 1:ny + 2], Bw[:, 2:ny + 3])
    dBw = -(Fx[1:, 2:ny + 2] - Fx[:-1, 2:ny + 2]) / dx - (Fy[2:nx + 2, 1:] - Fy[2:nx + 2, :-1]) / dy
    # energy flux: Poynting flux of E_N plus the Ettingshausen heat flux, cell values averaged to faces
    Bx, By = W[iBX], W[iBY]
    Ex, Ey, Ew = -vy * Bw, vx * Bw, -(vx * By - vy * Bx)
    Jx, Jy, Jw = xp.zeros_like(Bw), xp.zeros_like(Bw), xp.zeros_like(Bw)
    Jx[:, 1:-1] = (Bw[:, 2:] - Bw[:, :-2]) / (2 * dy)
    Jy[1:-1] = -(Bw[2:] - Bw[:-2]) / (2 * dx)
    Jw[1:-1, 1:-1] = ((By[2:, 1:-1] - By[:-2, 1:-1]) / (2 * dx) - (Bx[1:-1, 2:] - Bx[1:-1, :-2]) / (2 * dy))
    qe = xp.where(fuel, cN * theta * fac, 0.0)               # q_E = qe (B x J)
    Gcx = (Ey * Bw - Ew * By) + qe * (By * Jw - Bw * Jy)
    Gcy = (Ew * Bx - Ex * Bw) + qe * (Bw * Jx - Bx * Jw)
    Gx = 0.5 * (Gcx[1:nx + 2] + Gcx[2:nx + 3])
    Gy = 0.5 * (Gcy[:, 1:ny + 2] + Gcy[:, 2:ny + 3])
    # Ettingshausen part alone (the electron heat flux), for the two-temperature electron energy
    qx = qe * (By * Jw - Bw * Jy)
    qy = qe * (Bw * Jx - Bx * Jw)
    qEx, qEy = 0.5 * (qx[1:nx + 2] + qx[2:nx + 3]), 0.5 * (qy[:, 1:ny + 2] + qy[:, 2:ny + 3])
    divqE = (qEx[1:, 2:ny + 2] - qEx[:-1, 2:ny + 2]) / dx + (qEy[2:nx + 2, 1:] - qEy[2:nx + 2, :-1]) / dy
    vmax = xp.max(xp.sqrt(vx**2 + vy**2))                    # device scalar (no host sync)
    return E, dBw, Gx, Gy, vmax, divqE, (Fx, Fy), (qEx, qEy)


# ------------------------------------------------------------------ 3D (CT on all three face fields)

def nernst_velocity_3d(W, cN, x0, tracer_max, h):
    """v_N on 3D cells (zero outside the fuel and on the outer ring of each axis)"""
    fac, theta = _bw_over_B(W, x0)
    fuel = W[iCLR] < tracer_max
    v = []
    for ax in range(3):
        g = xp.zeros_like(theta)
        c, up, dn = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
        c[ax], up[ax], dn[ax] = slice(1, -1), slice(2, None), slice(None, -2)
        g[tuple(c)] = (theta[tuple(up)] - theta[tuple(dn)]) / (2 * h[ax])
        v.append(xp.where(fuel, -cN * fac * g, 0.0))
    return v, fac, theta, fuel


def nernst_edge(va, vb, Ba, Bb, na, nb):
    """Nernst EMF E_c = -(v_a B_b - v_b B_a) on the edges along c, arrays ordered (a, b, c): the 2D corner
    formula, v averaged from the four cells around the edge and each face field upwinded by the velocity
    across it. Same edges as ct.edge_emf: a-faces 2..na+2 x b-faces 2..nb+2 x all c."""
    c = lambda f: 0.25 * (f[1:na + 2, 1:nb + 2] + f[2:na + 3, 1:nb + 2] + f[1:na + 2, 2:nb + 3] + f[2:na + 3, 2:nb + 3])
    vac, vbc = c(va), c(vb)
    Bb_up = xp.where(vac > 0, Bb[1:na + 2, 2:nb + 3], Bb[2:na + 3, 2:nb + 3])
    Ba_up = xp.where(vbc > 0, Ba[2:na + 3, 1:nb + 2], Ba[2:na + 3, 2:nb + 3])
    return -(vac * Bb_up - vbc * Ba_up)


def nernst_energy_3d(W, v, fac, theta, fuel, cN, h, n):
    """Poynting flux of E_N = -v_N x B plus the Ettingshausen heat flux qe (B x J), cell values averaged to the
    faces: returns per axis the face fluxes (faces 2..n+2 along it, all cells across), and div q_E on interior
    cells (the electron part, for two-temperature runs)"""
    B = [W[iBX], W[iBY], W[iBZ]]
    E = [-(v[1] * B[2] - v[2] * B[1]), -(v[2] * B[0] - v[0] * B[2]), -(v[0] * B[1] - v[1] * B[0])]

    def d(f, ax):
        g = xp.zeros_like(f)
        c, up, dn = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
        c[ax], up[ax], dn[ax] = slice(1, -1), slice(2, None), slice(None, -2)
        g[tuple(c)] = (f[tuple(up)] - f[tuple(dn)]) / (2 * h[ax])
        return g
    J = [d(B[2], 1) - d(B[1], 2), d(B[0], 2) - d(B[2], 0), d(B[1], 0) - d(B[0], 1)]
    qe = xp.where(fuel, cN * theta * fac, 0.0)
    G, divq = [], 0.0
    inner = [slice(2, 2 + m) for m in n]
    for ax in range(3):
        p, q = (ax + 1) % 3, (ax + 2) % 3
        qa = qe * (B[p] * J[q] - B[q] * J[p])
        ga = (E[p] * B[q] - E[q] * B[p]) + qa
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[ax], hi[ax] = slice(1, n[ax] + 2), slice(2, n[ax] + 3)
        G.append(0.5 * (ga[tuple(lo)] + ga[tuple(hi)]))
        qf = 0.5 * (qa[tuple(lo)] + qa[tuple(hi)])
        a, b = list(inner), list(inner)
        a[ax], b[ax] = slice(1, n[ax] + 1), slice(0, n[ax])
        divq = divq + (qf[tuple(a)] - qf[tuple(b)]) / h[ax]
    return G, divq
