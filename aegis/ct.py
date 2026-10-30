# ct.py - constrained transport: face-centred B, edge EMFs (Gardiner & Stone 2005, "CT contact").
#
# The in-plane field lives on faces: B_a on the faces normal to axis a (in 2D only x and y; B_z stays a
# cell-centred conserved variable there). Faces are updated by dB/dt = -curl E with E on cell edges, so the
# face divergence (sum over a of the B_a difference across the cell) does not change: it stays at round-off.
# Cell-centred B is the average of the two faces. E on an edge along c (axes a, b, c cyclic) is the average
# of the four face values of E_c from the HLLD fluxes (E_c = -F_a[B_b] = +F_b[B_a]) plus upwinded gradient
# corrections against the cell-centred E_c = -(v_a B_b - v_b B_a), which makes it reduce to the 1D upwind
# result for grid-aligned flow. Face index convention: face i lies on the low side of cell i, so an axis
# with N cells (ghosts included) has N + 1 faces; the HLLD faces from muscl are faces 2 .. N-2.
from .config import RHO, BX, iVX, iBX
import numpy as np

from .physics import xp


def _up(m, lo, hi, tol):
    """value on the upwind side of a face with mass flux m; the average where |m| <= tol, so that
    round-off noise in a stagnant direction cannot flip the choice"""
    return xp.where(m > tol, lo, xp.where(m < -tol, hi, 0.5 * (lo + hi)))


def edge_emf(Fa, Fb, W, a, b, ha, hb, na, nb):
    """E_c on the edges where a-faces meet b-faces, arrays ordered (var, a, b, c).
    Fa: fluxes on a-faces 2..Na-2, all b and c; Fb: fluxes on b-faces 2..Nb-2; W: cell primitives.
    Returns edges on a-faces 2..na+2 x b-faces 2..nb+2 x all c (na, nb: interior cell counts, 2 ghosts)."""
    Ea = -Fa[BX + b]                                         # E_c on a-faces
    Eb = Fb[BX + a]                                          # E_c on b-faces
    Ec = -(W[iVX + a] * W[iBX + b] - W[iVX + b] * W[iBX + a])
    ma, mb = Fa[RHO], Fb[RHO]
    tol = xp.maximum(1e-10 * xp.maximum(xp.max(xp.abs(ma)), xp.max(xp.abs(mb))), 1e-300)   # on the device
    Ea_hi, Ea_lo = Ea[:, 2:nb + 3], Ea[:, 1:nb + 2]
    Eb_r, Eb_l = Eb[2:na + 3], Eb[1:na + 2]
    Ec_c, Ec_l = Ec[2:na + 3, 2:nb + 3], Ec[1:na + 2, 2:nb + 3]
    Ec_d, Ec_ld = Ec[2:na + 3, 1:nb + 2], Ec[1:na + 2, 1:nb + 2]
    db_hi = _up(ma[:, 2:nb + 3], 2 * (Ec_l - Eb_l) / hb, 2 * (Ec_c - Eb_r) / hb, tol)
    db_lo = _up(ma[:, 1:nb + 2], 2 * (Eb_l - Ec_ld) / hb, 2 * (Eb_r - Ec_d) / hb, tol)
    da_r = _up(mb[2:na + 3], 2 * (Ec_d - Ea_lo) / ha, 2 * (Ec_c - Ea_hi) / ha, tol)
    da_l = _up(mb[1:na + 2], 2 * (Ea_lo - Ec_ld) / ha, 2 * (Ea_hi - Ec_l) / ha, tol)
    return (0.25 * (Ea_hi + Ea_lo + Eb_r + Eb_l) + 0.125 * hb * (db_lo - db_hi)
            + 0.125 * ha * (da_l - da_r))


def edge_current(Bf, a, b, ha, hb, na, nb):
    """J_c = dB_b/da - dB_a/db on the same edges as edge_emf, from the face fields (ordered a, b, c)"""
    Ba, Bb = Bf[a], Bf[b]                                    # Ba: (Na+1, Nb, Nc), Bb: (Na, Nb+1, Nc)
    return ((Bb[2:na + 3, 2:nb + 3] - Bb[1:na + 2, 2:nb + 3]) / ha
            - (Ba[2:na + 3, 2:nb + 3] - Ba[2:na + 3, 1:nb + 2]) / hb)


def edge_eta(eta, na, nb):
    """harmonic mean of the four cells around each edge (eta ordered a, b, c)"""
    if isinstance(eta, (int, float)):
        return eta
    e = [eta[1:na + 2, 1:nb + 2], eta[2:na + 3, 1:nb + 2], eta[1:na + 2, 2:nb + 3], eta[2:na + 3, 2:nb + 3]]
    return 4.0 / sum(1.0 / xp.maximum(x, 1e-300) for x in e)


def faces_from_cells(U, ndim):
    """face fields by averaging neighbouring cells (exact for fields whose normal component does not vary
    along its own axis; a general field should start from a vector potential)"""
    Bf = []
    for a in range(ndim):
        c = U[BX + a]
        n = c.shape[a]
        shape = list(c.shape)
        shape[a] = n + 1
        f = xp.empty(shape)
        lo = [slice(None)] * c.ndim
        hi = [slice(None)] * c.ndim
        mid = [slice(None)] * c.ndim
        lo[a], hi[a], mid[a] = slice(0, n - 1), slice(1, n), slice(1, n)
        f[tuple(mid)] = 0.5 * (c[tuple(lo)] + c[tuple(hi)])
        e0, e1, s0, s1 = [slice(None)] * c.ndim, [slice(None)] * c.ndim, [slice(None)] * c.ndim, [slice(None)] * c.ndim
        e0[a], s0[a], e1[a], s1[a] = 0, 0, n, n - 1
        f[tuple(e0)] = c[tuple(s0)]
        f[tuple(e1)] = c[tuple(s1)]
        Bf.append(f)
    return Bf


def cells_from_faces(U, Bf, inner):
    """cell-centred B = average of the two faces, on the cells selected by `inner` (spatial slices)"""
    for a, f in enumerate(Bf):
        n = U.shape[1 + a]
        lo = [slice(None)] * (U.ndim - 1)
        hi = [slice(None)] * (U.ndim - 1)
        lo[a], hi[a] = slice(0, n), slice(1, n + 1)
        U[(BX + a,) + inner] = (0.5 * (f[tuple(lo)] + f[tuple(hi)]))[inner]


def ghost_faces(Bf, U, ng, n, periodic):
    """faces outside the interior: along periodic axes, wrap-around copies of the interior faces (cell
    averages there would be a smoothed copy and break periodicity); elsewhere the averages of the ghost
    cells the cell boundary conditions produced. Interior faces, domain-boundary faces included, are
    owned by CT and left alone."""
    nd = len(Bf)
    avg = faces_from_cells(U, nd)
    for a, f in enumerate(Bf):
        inner = [slice(ng, ng + m) for m in n]
        inner[a] = slice(ng, ng + n[a] + 1)
        inner = tuple(inner)
        keep = f[inner].copy()
        f[...] = avg[a]
        f[inner] = keep
        for t in range(nd):
            if not periodic[t]:
                continue
            N = f.shape[t]
            lo, hi = [slice(None)] * nd, [slice(None)] * nd
            if t == a:                               # faces 0..ng-1 and ng+n+1..N-1 (face index = cell index)
                lo[t], hi[t] = slice(0, ng), slice(n[t], n[t] + ng)
                f[tuple(lo)] = f[tuple(hi)]
                lo[t], hi[t] = slice(ng + n[t] + 1, N), slice(ng + 1, N - n[t])
                f[tuple(lo)] = f[tuple(hi)]
            else:                                    # ghost cell layers along t
                lo[t], hi[t] = slice(0, ng), slice(n[t], n[t] + ng)
                f[tuple(lo)] = f[tuple(hi)]
                lo[t], hi[t] = slice(ng + n[t], N), slice(ng, N - n[t])
                f[tuple(lo)] = f[tuple(hi)]


def div_faces(Bf, h, ng, n):
    """face divergence on interior cells"""
    nd = len(Bf)
    d = 0.0
    for a, f in enumerate(Bf):
        idx_hi = [slice(ng, ng + m) for m in n]
        idx_lo = [slice(ng, ng + m) for m in n]
        idx_hi[a] = slice(ng + 1, ng + n[a] + 1)
        d = d + (f[tuple(idx_hi)] - f[tuple(idx_lo)]) / h[a]
    return d


def faces_from_potential(A, coords_c, coords_f):
    """Face fields as the discrete curl of a vector potential, so the face divergence is zero to round-off.
    2D: A = Az(x, y) evaluated at cell corners. 3D: A = (Ax, Ay, Az) callables, each on its edges.
    coords_c / coords_f: per axis, cell-centre and face coordinates (numpy, ghosts included)."""
    h = [cf[1] - cf[0] for cf in coords_f]
    if len(coords_c) == 2:
        X, Y = np.meshgrid(coords_f[0], coords_f[1], indexing="ij")
        Az = A(X, Y)
        return [xp.asarray((Az[:, 1:] - Az[:, :-1]) / h[1]), xp.asarray(-(Az[1:, :] - Az[:-1, :]) / h[0])]
    (xc, yc, zc), (xf, yf, zf) = coords_c, coords_f
    g = lambda a, b, c: np.meshgrid(a, b, c, indexing="ij")
    Ax = A[0](*g(xc, yf, zf))                                # x-edges
    Ay = A[1](*g(xf, yc, zf))                                # y-edges
    Az = A[2](*g(xf, yf, zc))                                # z-edges
    Bx = (Az[:, 1:, :] - Az[:, :-1, :]) / h[1] - (Ay[:, :, 1:] - Ay[:, :, :-1]) / h[2]
    By = (Ax[:, :, 1:] - Ax[:, :, :-1]) / h[2] - (Az[1:, :, :] - Az[:-1, :, :]) / h[0]
    Bz = (Ay[1:, :, :] - Ay[:-1, :, :]) / h[0] - (Ax[:, 1:, :] - Ax[:, :-1, :]) / h[1]
    return [xp.asarray(Bx), xp.asarray(By), xp.asarray(Bz)]
