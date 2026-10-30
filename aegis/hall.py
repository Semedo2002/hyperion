# hall.py - generalized Ohm's law beyond eta J: the Hall term and anisotropic (Spitzer) resistivity, for the 2D
# Cartesian solver with constrained transport.
#
#     E' = d_i (J x B) / rho + (eta_perp - eta_par) J_perp        (added to E = -v x B + eta_par J)
#
# d_i is the ion inertial length at unit density in code units (Hall term J x B / (e n_e) with n_e = rho / m_i
# for Z = 1; ion cyclotron frequency Omega_i = B / d_i). J_perp = J - (b . J) b; eta_perp = 1.96 eta_par for a
# magnetized Spitzer plasma (Z = 1), cfg.eta_perp_ratio. Where B = 0 there is no anisotropy.
#
# Discretisation (2.5D, d/dz = 0): J and B are formed on the cell corners (for E'_z, which joins the CT corner
# EMF and updates the in-plane face fields) and on the faces (E'_y on x-faces, E'_x on y-faces, which update
# the cell-centred B_z in flux form). Normal derivatives are compact two-point differences, transverse ones
# averages of cell-centred differences. The energy equation gets the Poynting flux E' x B on the faces, so total
# energy telescopes exactly; the Hall field does no work in the continuum (J . E_Hall = 0) and the anisotropic
# part heats at (eta_perp - eta_par) |J_perp|^2 through the drop in magnetic energy.
from .config import iRHO, iBX, iBY, iBZ
from .physics import xp

PERP_RATIO_SPITZER = 1.96


def active(cfg):
    return cfg.hall_di > 0 or (cfg.is_resistive() and cfg.eta_perp_ratio != 1.0)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _efield(J, B, rho, di, deta):
    """E' at one set of points: Hall + (eta_perp - eta_par) J_perp; deta may be a float or an array"""
    E = [0.0 * J[0], 0.0 * J[0], 0.0 * J[0]]
    if di > 0:
        H = _cross(J, B)
        E = [di * h / rho for h in H]
    if deta is not None:
        Bm2 = B[0]**2 + B[1]**2 + B[2]**2
        inv = xp.where(Bm2 > 0, 1.0 / xp.where(Bm2 > 0, Bm2, 1.0), 0.0)
        bJ = (B[0] * J[0] + B[1] * J[1] + B[2] * J[2]) * inv        # (b . J) / |B|
        mag = Bm2 > 0
        E = [e + xp.where(mag, deta * (j - bJ * b), 0.0) for e, j, b in zip(E, J, B)]
    return E


def ohm_extra(W, Bf, eta, cfg, dx, dy, nx, ny, ng):
    """E' of the generalized Ohm's law on a ghost-inclusive 2D state (W: primitive cells with B from the faces,
    Bf: CT faces, eta: cell resistivity, a float or None). Returns
        Ek   E'_z on the interior corners (nx + 1, ny + 1), to add to the CT EMF
        Fbx  B_z flux on the interior x-faces (nx + 1, ny):  E'_y
        Fby  B_z flux on the interior y-faces (nx, ny + 1): -E'_x
        Sx, Sy  energy fluxes (E' x B) on the same faces
        q    (eta_perp - eta_par) |J_perp|^2 on the interior cells (heating diagnostic / two-temperature)"""
    di = cfg.hall_di
    ratio = cfg.eta_perp_ratio if eta is not None else 1.0
    rho, Bxc, Byc, Bz = W[iRHO], W[iBX], W[iBY], W[iBZ]
    A, Bc = slice(ng - 1, ng + nx), slice(ng, ng + nx + 1)          # cells left / right of a corner or x-face
    C, D = slice(ng - 1, ng + ny), slice(ng, ng + ny + 1)            # cells below / above
    I, Jr = slice(ng, ng + nx), slice(ng, ng + ny)                   # interior cells

    def deta_at(avg):
        if eta is None or ratio == 1.0:
            return None
        return (ratio - 1.0) * (eta if isinstance(eta, (int, float)) else avg(eta))
    corner = lambda q: 0.25 * (q[A, C] + q[Bc, C] + q[A, D] + q[Bc, D])
    xface = lambda q: 0.5 * (q[A, Jr] + q[Bc, Jr])
    yface = lambda q: 0.5 * (q[I, C] + q[I, D])

    # corners
    Bk = (0.5 * (Bf[0][Bc, C] + Bf[0][Bc, D]), 0.5 * (Bf[1][A, D] + Bf[1][Bc, D]), corner(Bz))
    Jk = ((0.5 * (Bz[A, D] + Bz[Bc, D]) - 0.5 * (Bz[A, C] + Bz[Bc, C])) / dy,
          -(0.5 * (Bz[Bc, C] + Bz[Bc, D]) - 0.5 * (Bz[A, C] + Bz[A, D])) / dx,
          (Bf[1][Bc, D] - Bf[1][A, D]) / dx - (Bf[0][Bc, D] - Bf[0][Bc, C]) / dy)
    Ekv = _efield(Jk, Bk, corner(rho), di, deta_at(corner))

    # x-faces
    dBz_dy = (Bz[:, ng + 1:ng + ny + 1] - Bz[:, ng - 1:ng + ny - 1]) / (2 * dy)   # cells x all, rows interior
    Bx_ = (Bf[0][Bc, Jr], xface(Byc), xface(Bz))
    Jx_ = (0.5 * (dBz_dy[A] + dBz_dy[Bc]), -(Bz[Bc, Jr] - Bz[A, Jr]) / dx, 0.5 * (Jk[2][:, :-1] + Jk[2][:, 1:]))
    Ex_ = _efield(Jx_, Bx_, xface(rho), di, deta_at(xface))
    Ezx = 0.5 * (Ekv[2][:, :-1] + Ekv[2][:, 1:])

    # y-faces
    dBz_dx = (Bz[ng + 1:ng + nx + 1, :] - Bz[ng - 1:ng + nx - 1, :]) / (2 * dx)   # cells x interior, rows all
    By_ = (yface(Bxc), Bf[1][I, D], yface(Bz))
    Jy_ = ((Bz[I, D] - Bz[I, C]) / dy, -0.5 * (dBz_dx[:, C] + dBz_dx[:, D]), 0.5 * (Jk[2][:-1, :] + Jk[2][1:, :]))
    Ey_ = _efield(Jy_, By_, yface(rho), di, deta_at(yface))
    Ezy = 0.5 * (Ekv[2][:-1, :] + Ekv[2][1:, :])

    Sx = Ex_[1] * Bx_[2] - Ezx * Bx_[1]                              # (E x B)_x on x-faces
    Sy = Ezy * By_[0] - Ey_[0] * By_[2]                              # (E x B)_y on y-faces
    q = None
    if deta_at(corner) is not None:                                  # anisotropic Ohmic heating, corners -> cells
        qk = sum(e * j for e, j in zip(_efield(Jk, Bk, corner(rho), 0.0, deta_at(corner)), Jk))
        q = 0.25 * (qk[:-1, :-1] + qk[1:, :-1] + qk[:-1, 1:] + qk[1:, 1:])
    return Ekv[2], Ex_[1], -Ey_[0], Sx, Sy, q


def dt_limit(W, cfg, dx, dy, eta_max=None):
    """explicit limits: whistler (omega = d_i |B| k^2 / rho at the grid scale, kept well inside the SSP-RK3
    imaginary-axis stability bound) and the larger (perpendicular) resistive diffusivity"""
    dt = float("inf")
    lap = 4.0 / dx**2 + 4.0 / dy**2
    if cfg.hall_di > 0:
        B = xp.sqrt(W[iBX]**2 + W[iBY]**2 + W[iBZ]**2)
        w = float(xp.max(cfg.hall_di * B / W[iRHO])) * lap
        if w > 0:                                    # |omega dt| <= 2.5 cfl (0.75 at cfl 0.3), inside sqrt(3)
            dt = min(dt, 2.5 * cfg.cfl / w)
    return dt
