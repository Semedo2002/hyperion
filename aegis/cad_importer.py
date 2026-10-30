# cad_importer.py - CAD geometry straight onto the solver grid: volume fractions, face apertures, immersed walls.
#
# Input: .stl (ASCII or binary, read here), .step / .stp / .iges / .igs (B-rep, triangulated through gmsh's
# OpenCASCADE kernel when the optional `gmsh` package is installed). A geometry is a closed triangle surface.
#
# Volume fractions and apertures by ray columns: for a ray direction (one grid axis) the surface is crossed at
# exactly computed points along each ray; sorted crossings pair into inside intervals (parity, so the triangle
# orientation does not matter). The inside length of each interval within each cell along the ray is exact, and
# the rays are a sub x sub sample per cell across it (midpoint rule). So alpha_cell = solid volume / cell volume is
# exact along the ray axis and second order across it (error ~ (h / sub)^2 for a smooth surface), and the
# aperture of a face normal to the ray axis (its solid area fraction) comes from the same crossings. Running the
# columns along each axis gives the apertures of every face orientation; the wall area vector of a cell follows
# from them by the discrete divergence theorem, A_wall = -sum_faces (1 - aperture) A_face n_face, so walls and
# faces close exactly.
#
# Two uses:
#   - material: alpha becomes the liner volume fraction / density of an initial state (no meshing: the CAD solid
#     is the target);
#   - immersed rigid walls (ImmersedWalls): each HLLD face flux becomes a_f F + (1 - a_f) F_wall, a_f the face's
#     fluid aperture and F_wall the flux of a stationary wall (no mass, energy, tracer or induction flux; the
#     pressure + Maxwell stress row p + B^2/2 - B_n^2, -B_n B_t on the momentum), from the side with more fluid.
#     Cell volumes stay whole (a porosity-type immersed boundary): there is no small-cut-cell time-step problem,
#     uniform states at rest stay at rest exactly, mass and energy are conserved exactly, and CT keeps div B at
#     round-off; cells under half fluid count as solid (all faces closed, no update); the wall's normal and blockage are sub-cell, its position along a grid axis is resolved to a
#     cell (no volume-normalised cut-cell update).
import os
import struct

import numpy as np


# ============================================================
# reading and writing surfaces
# ============================================================
def read_stl(path):
    """triangles (T, 3, 3) from an ASCII or binary STL"""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):                 # binary
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                                count=n, offset=84)
            return rec["v"].astype(np.float64)
    text = data.decode("ascii", errors="replace").split()
    if not text or text[0].lower() != "solid":
        raise ValueError(f"{path}: not an STL file")
    v = np.asarray(_ascii_vertices(text), dtype=np.float64)
    if v.size % 9:
        raise ValueError(f"{path}: truncated ASCII STL")
    return v.reshape(-1, 3, 3)


def _ascii_vertices(tok):
    out = []
    for i, t in enumerate(tok):
        if t == "vertex":
            out.extend((float(tok[i + 1]), float(tok[i + 2]), float(tok[i + 3])))
    return out


def write_stl(path, tris, binary=True, name="aegis"):
    tris = np.asarray(tris, dtype=np.float64)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
    if binary:
        rec = np.zeros(len(tris), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        rec["n"], rec["v"] = nrm, tris
        with open(path, "wb") as f:
            f.write(name.encode()[:80].ljust(80, b" "))
            f.write(struct.pack("<I", len(tris)))
            f.write(rec.tobytes())
    else:
        with open(path, "w") as f:
            f.write(f"solid {name}\n")
            for t, n in zip(tris, nrm):
                f.write(f" facet normal {n[0]:.9e} {n[1]:.9e} {n[2]:.9e}\n  outer loop\n")
                for p in t:
                    f.write(f"   vertex {p[0]:.12e} {p[1]:.12e} {p[2]:.12e}\n")
                f.write("  endloop\n endfacet\n")
            f.write(f"endsolid {name}\n")


def read_cad(path, mesh_size=None):
    """triangles (T, 3, 3) from .stl, or from .step/.stp/.iges/.igs through gmsh (OpenCASCADE)"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl":
        return read_stl(path)
    if ext not in (".step", ".stp", ".iges", ".igs"):
        raise ValueError(f"{path}: unsupported CAD format {ext!r} (.stl, .step, .stp, .iges, .igs)")
    try:
        import gmsh
    except ImportError as e:
        raise ImportError("STEP / IGES import needs the gmsh package (pip install gmsh), which provides the "
                          "OpenCASCADE kernel; STL files load without it") from e
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cad")
        gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()
        if mesh_size:
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.25 * mesh_size)
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        pos = {t: i for i, t in enumerate(tags)}
        xyz = coords.reshape(-1, 3)
        tris = []
        for dim, tag in gmsh.model.getEntities(2):
            types, _, nodes = gmsh.model.mesh.getElements(dim, tag)
            for ty, nd in zip(types, nodes):
                if ty == 2:                          # 3-node triangles
                    idx = np.array([pos[t] for t in nd]).reshape(-1, 3)
                    tris.append(xyz[idx])
        return np.concatenate(tris).astype(np.float64)
    finally:
        gmsh.finalize()


def mesh_volume(tris):
    """enclosed volume of a closed, consistently oriented surface (divergence theorem, sign-free)"""
    return abs(float(np.sum(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])))) / 6.0)


# ============================================================
# ray columns
# ============================================================
def _columns(tris, ax, edges, sub):
    """crossings of the surface with rays along axis `ax` through a sub x sub sample of every cell of the
    two other axes. Returns (column ids of each interval, interval start, end) sorted, plus the column grid
    shape (n1 * sub, n2 * sub), axes (a1, a2)."""
    a1, a2 = [d for d in range(3) if d != ax]
    e1, e2 = np.asarray(edges[a1], float), np.asarray(edges[a2], float)
    n1, n2 = (len(e1) - 1) * sub, (len(e2) - 1) * sub
    # sample points: sub per cell (cells may be non-uniform); a tiny irrational shift keeps rays off edges / vertices
    def pts(e):
        f = (np.arange(sub) + 0.5) / sub
        return (e[:-1, None] + np.diff(e)[:, None] * f[None, :]).ravel()
    u, v = pts(e1), pts(e2)
    du, dv = np.min(np.diff(e1)) / sub, np.min(np.diff(e2)) / sub
    u = u + 1.234567e-7 * du * np.sqrt(2.0)
    v = v + 1.234567e-7 * dv * np.sqrt(3.0)
    P, Q, Z = tris[:, :, a1], tris[:, :, a2], tris[:, :, ax]
    lo_i = np.searchsorted(u, P.min(axis=1), side="left")
    hi_i = np.searchsorted(u, P.max(axis=1), side="right")
    lo_j = np.searchsorted(v, Q.min(axis=1), side="left")
    hi_j = np.searchsorted(v, Q.max(axis=1), side="right")
    ni, nj = np.maximum(hi_i - lo_i, 0), np.maximum(hi_j - lo_j, 0)
    cnt = ni * nj
    keep = cnt > 0
    t = np.repeat(np.nonzero(keep)[0], cnt[keep])
    if t.size == 0:
        return np.zeros(0, int), np.zeros(0), np.zeros(0), (n1, n2)
    start = np.repeat(np.cumsum(cnt[keep]) - cnt[keep], cnt[keep])
    k = np.arange(t.size) - start
    i = lo_i[t] + k // nj[t]
    j = lo_j[t] + k % nj[t]
    pu, pv = u[i], v[j]
    x0, y0 = P[t, 0], Q[t, 0]
    x1, y1 = P[t, 1] - x0, Q[t, 1] - y0
    x2, y2 = P[t, 2] - x0, Q[t, 2] - y0
    det = x1 * y2 - x2 * y1
    ok = np.abs(det) > 1e-300
    det = np.where(ok, det, 1.0)
    bx, by = pu - x0, pv - y0
    l1 = (bx * y2 - x2 * by) / det
    l2 = (x1 * by - bx * y1) / det
    hit = ok & (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
    t, i, j, l1, l2 = t[hit], i[hit], j[hit], l1[hit], l2[hit]
    z = Z[t, 0] + l1 * (Z[t, 1] - Z[t, 0]) + l2 * (Z[t, 2] - Z[t, 0])
    col = i * n2 + j
    o = np.lexsort((z, col))
    col, z = col[o], z[o]
    # pair consecutive crossings per column (parity); an odd count (open surface) drops the last crossing
    first = np.r_[True, col[1:] != col[:-1]]
    rank = np.arange(col.size) - np.maximum.accumulate(np.where(first, np.arange(col.size), 0))
    cnt_col = np.bincount(col, minlength=n1 * n2)
    usable = rank < (cnt_col[col] // 2) * 2
    col, z, rank = col[usable], z[usable], rank[usable]
    a = (rank % 2) == 0
    return col[a], z[a], z[~a], (n1, n2)


def _layer_lengths(cols, za, zb, ncol, levels):
    """inside length below each level (ncol, L) for intervals [za, zb] in columns `cols`"""
    out = np.zeros((ncol, len(levels)))
    if cols.size:
        Lv = np.clip(levels[None, :] - za[:, None], 0.0, (zb - za)[:, None])
        np.add.at(out, cols, Lv)
    return out


class Geometry:
    """a closed surface (triangles) placed on a rectilinear grid; edges: cell edges along x, y, z"""

    def __init__(self, tris, edges, sub=4):
        self.tris = np.asarray(tris, dtype=np.float64)
        self.edges = [np.asarray(e, float) for e in edges]
        self.sub = sub
        self.shape = tuple(len(e) - 1 for e in self.edges)
        self._cols = {}

    def _col(self, ax):
        if ax not in self._cols:
            self._cols[ax] = _columns(self.tris, ax, self.edges, self.sub)
        return self._cols[ax]

    def _per_cell(self, ax, f):
        """average an (n1 * sub, n2 * sub, ...) column field over the sub-columns of each cell"""
        a1, a2 = [d for d in range(3) if d != ax]
        n1, n2, s = self.shape[a1], self.shape[a2], self.sub
        g = f.reshape((n1, s, n2, s) + f.shape[2:]).mean(axis=(1, 3))
        return g                                     # (n1, n2, ...)

    def volume_fraction(self, ax=2):
        """solid fraction of every cell (nx, ny, nz), exact along axis ax"""
        cols, za, zb, (n1, n2) = self._col(ax)
        e = self.edges[ax]
        L = _layer_lengths(cols, za, zb, n1 * n2, e)                  # (ncol, n_ax + 1)
        frac = np.diff(L, axis=1) / np.diff(e)[None, :]
        g = self._per_cell(ax, frac.reshape(n1, n2, -1))              # (n1, n2, n_ax)
        return np.moveaxis(g, 2, ax)

    def aperture(self, ax):
        """solid area fraction of the faces normal to axis ax, shape with n_ax + 1 along ax"""
        cols, za, zb, (n1, n2) = self._col(ax)
        e = self.edges[ax]
        inside = np.zeros((n1 * n2, len(e)))
        if cols.size:
            np.add.at(inside, cols, ((e[None, :] > za[:, None]) & (e[None, :] < zb[:, None])).astype(float))
        g = self._per_cell(ax, np.minimum(inside, 1.0).reshape(n1, n2, -1))
        return np.moveaxis(g, 2, ax)

    def wall_area_vectors(self):
        """(3, nx, ny, nz): area vector of the wall in each cell, pointing out of the fluid (into the solid),
        from the fluid apertures by the discrete divergence theorem"""
        h = [np.diff(e) for e in self.edges]
        out = np.zeros((3,) + self.shape)
        for ax in range(3):
            fl = 1.0 - self.aperture(ax)
            o = [d for d in range(3) if d != ax]
            area = np.multiply.outer(h[o[0]], h[o[1]])
            lo = np.take(fl, range(0, self.shape[ax]), axis=ax)
            hi = np.take(fl, range(1, self.shape[ax] + 1), axis=ax)
            out[ax] = (lo - hi) * np.expand_dims(area, ax)            # minus the outward fluid face vectors
        return out


# ============================================================
# test and demo shapes
# ============================================================
def _grid_surface(X, Y, Z, flip=False):
    """triangles of a structured (n1, n2) surface patch"""
    P = np.stack([X, Y, Z], axis=-1)
    a, b, c, d = P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]
    t1, t2 = np.stack([a, b, c], axis=-2), np.stack([a, c, d], axis=-2)
    t = np.concatenate([t1.reshape(-1, 3, 3), t2.reshape(-1, 3, 3)])
    return t[:, ::-1] if flip else t


def sphere_mesh(r=1.0, center=(0.0, 0.0, 0.0), n=48):
    """closed UV sphere (2 n^2 triangles; the poles hold degenerate ones of zero area)"""
    th = np.linspace(0, np.pi, n + 1)
    ph = np.linspace(0, 2 * np.pi, 2 * n + 1)
    T, Ph = np.meshgrid(th, ph, indexing="ij")
    c = np.asarray(center, float)
    return _grid_surface(c[0] + r * np.sin(T) * np.cos(Ph), c[1] + r * np.sin(T) * np.sin(Ph), c[2] + r * np.cos(T))


def corrugated_liner_mesh(r_in=1.0, thickness=0.2, length=2.0, m=4, kz=1, eps=0.05, n_theta=128, n_z=64,
                          center=(0.0, 0.0), z0=0.0, overhang=0.05):
    """closed cylindrical shell along z with an inner radius r_in (1 + eps cos(m theta) cos(2 pi kz z / L)) and
    constant thickness, capped at z0 - overhang and z0 + length + overhang (caps outside a z-periodic domain
    [z0, z0 + length], so rays along z see no cap inside it)"""
    th = np.linspace(0, 2 * np.pi, n_theta + 1)
    z = np.linspace(z0 - overhang, z0 + length + overhang, n_z + 1)
    Th, Zz = np.meshgrid(th, z, indexing="ij")
    Ri = r_in * (1 + eps * np.cos(m * Th) * np.cos(2 * np.pi * kz * (Zz - z0) / length))
    Ro = Ri + thickness
    cx, cy = center
    inner = _grid_surface(cx + Ri * np.cos(Th), cy + Ri * np.sin(Th), Zz, flip=True)
    outer = _grid_surface(cx + Ro * np.cos(Th), cy + Ro * np.sin(Th), Zz)
    caps = []
    for k, flip in ((0, False), (-1, True)):
        rr = np.stack([Ri[:, k], Ro[:, k]], axis=1)
        tt = np.stack([th, th], axis=1)
        caps.append(_grid_surface(cx + rr * np.cos(tt), cy + rr * np.sin(tt), np.full_like(rr, z[k]), flip=flip))
    return np.concatenate([inner, outer] + caps)


def cylinder_mesh(r=0.2, length=2.0, center=(0.0, 0.0), z0=0.0, overhang=0.05, n_theta=96, n_z=8):
    """closed solid cylinder along z (side + fan caps), capped outside [z0, z0 + length] like the liner"""
    th = np.linspace(0, 2 * np.pi, n_theta + 1)
    z = np.linspace(z0 - overhang, z0 + length + overhang, n_z + 1)
    Th, Zz = np.meshgrid(th, z, indexing="ij")
    cx, cy = center
    side = _grid_surface(cx + r * np.cos(Th), cy + r * np.sin(Th), Zz)
    caps = []
    for zc, flip in ((z[0], True), (z[-1], False)):
        c = np.array([cx, cy, zc])
        p = np.stack([cx + r * np.cos(th), cy + r * np.sin(th), np.full_like(th, zc)], axis=1)
        t = np.stack([np.repeat(c[None], n_theta, 0), p[:-1], p[1:]], axis=1)
        caps.append(t[:, ::-1] if flip else t)
    return np.concatenate([side] + caps)


def grid_edges(lo, hi, n):
    return np.linspace(lo, hi, n + 1)


def solver_edges(solver):
    """cell edges of a solver's interior (2D: a unit slab in z)"""
    cfg = solver.cfg
    ex = grid_edges(cfg.x_min, cfg.x_max, cfg.nx)
    ey = grid_edges(cfg.y_min, cfg.y_max, cfg.ny)
    ez = grid_edges(cfg.z_min, cfg.z_max, cfg.nz) if hasattr(solver, "nz_tot") else np.array([-0.5, 0.5])
    return [ex, ey, ez]


# ============================================================
# immersed rigid walls in the solvers
# ============================================================
class ImmersedWalls:
    """aperture-weighted HLLD fluxes with a conducting-wall flux on the blocked part of every face.

    fluid_face[a]: fluid aperture of the faces normal to axis a over the solver's interior (n_a + 1 along a);
    fluid_cell: fluid fraction of the interior cells. Built by from_geometry(); the solver calls apply()."""

    def __init__(self, solver, fluid_face, fluid_cell, min_fluid=0.5):
        from .physics import xp
        self.s = solver
        ng = solver.ng
        nd = fluid_cell.ndim
        # cells with less than min_fluid of fluid are solid: all their faces closed. Otherwise a sliver of fluid in
        # a full-volume cell, and the solid cells behind it, would take part in the flow through partly open faces
        # and soak up gas (the wall would act as a sponge)
        fluid_cell = np.where(fluid_cell >= min_fluid, fluid_cell, 0.0)
        open_ = fluid_cell > 0
        fluid_face = [f.copy() for f in fluid_face]
        for a in range(nd):
            n = fluid_cell.shape[a]
            lo = np.concatenate([np.take(open_, [0], axis=a), open_], axis=a)       # cell on the low side
            hi = np.concatenate([open_, np.take(open_, [n - 1], axis=a)], axis=a)   # cell on the high side
            fluid_face[a] = np.where(lo & hi, fluid_face[a], 0.0)
        pad = [(ng, ng)] * nd
        fc = np.pad(fluid_cell, pad, mode="edge")
        self.fluid_cell = xp.asarray(fc)
        self._open = (self.fluid_cell > 0).astype(self.fluid_cell.dtype)
        self.face = []
        for a in range(nd):
            p = list(pad)
            p[a] = (0, 0)                            # flux layout: faces ng..ng+n along a, all cells across
            self.face.append(xp.asarray(np.pad(fluid_face[a], p, mode="edge")))

    @classmethod
    def from_geometry(cls, solver, geom, min_fluid=0.5):
        if getattr(solver, "cyl", False):
            raise NotImplementedError("immersed walls: Cartesian grids (2D x-y or 3D x-y-z)")
        nd = 3 if hasattr(solver, "nz_tot") else 2
        alpha = geom.volume_fraction()
        faces = [1.0 - geom.aperture(a) for a in range(nd)]
        if nd == 2:
            alpha = alpha[:, :, 0]
            faces = [f[:, :, 0] for f in faces]
        return cls(solver, faces, 1.0 - alpha, min_fluid)

    def mask_rates(self, R):
        """no update in fully solid cells: their faces are closed (no mass or energy flux), but the wall pressure on
        one face and their own on the opposite one would push the frozen gas inside them to runaway speeds"""
        R *= self._open[None]

    def apply(self, F, WL, WR, a):
        """F, WL, WR: (NVAR, ...) fluxes and face states along axis a (flux layout); modified in place"""
        from .physics import xp
        from .config import RHO, MX, BX, EN, PSI, RHOC, RHOSE, iPR, iBX
        ap = self.face[a]
        sl = [slice(None)] * ap.ndim
        lo = list(sl)
        hi = list(sl)
        n = ap.shape[a]
        lo[a] = slice(self.s.ng - 1, self.s.ng - 1 + n)
        hi[a] = slice(self.s.ng, self.s.ng + n)
        left_more = self.fluid_cell[tuple(lo)] >= self.fluid_cell[tuple(hi)]
        # the wall carries the momentum-flux row of the pressure + Maxwell stress of the fluid-side state:
        # p + B^2/2 - B_n^2 normal, -B_n B_t tangential (the flux of that state at rest, so any uniform state at rest
        # stays at rest); no mass, energy, tracer or induction flux
        p, B = [xp.where(left_more, WL[k], WR[k]) for k in (iPR,)][0],             [xp.where(left_more, WL[iBX + d], WR[iBX + d]) for d in range(3)]
        Bn = B[a]
        blocked = 1.0 - ap
        F *= ap[None]
        for d in range(3):
            stress = p + 0.5 * (B[0]**2 + B[1]**2 + B[2]**2) - Bn * Bn if d == a else -Bn * B[d]
            F[MX + d] += blocked * stress
        return F
