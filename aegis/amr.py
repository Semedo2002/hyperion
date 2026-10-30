# amr.py - block-structured adaptive mesh refinement for the 2D Cartesian solver (Berger-Colella style).
#
# One coarse MHDSolver covers the domain; fine patches (each an MHDSolver on a sub-rectangle, cell size dx/r)
# cover the flagged regions and take r sub-steps per coarse step. Per coarse step:
#   1. the coarse level steps, recording (time-weighted over its SSP-RK3 stages) the fluxes, the
#      volume-fraction face terms and the corner EMFs on every patch boundary;
#   2. each patch sub-steps, its ghost cells filled from the coarse level at the stage time (limited linear in
#      space, linear in time between the coarse states before and after the step), recording the same
#      boundary quantities;
#   3. synchronisation: the coarse cells next to a patch are refluxed with the fine fluxes (mass, momentum and
#      energy then conserved exactly across the coarse-fine boundary), the coarse faces next to a patch get the
#      fine edge EMFs instead of the coarse ones (Balsara 2001), and the coarse cells and faces under a patch
#      become the fine averages. The coarse face divergence stays at round-off, and so does the fine one.
# The volume fraction (non-conservative, Shyue) is refluxed with both of its face terms, u a and u, and the
# prolongation uses one slope limiter for all conserved variables of a cell, so a material interface at
# uniform pressure and velocity stays in equilibrium across the coarse-fine boundary.
#
# Patches may touch: on a shared edge the lower-index patch owns the faces and corners, and the other is
# refluxed and EMF-corrected with its values, so conservation and div B hold across fine-fine seams too
# (ghost cells there still come from the coarse level).
#
# Every `regrid_every` coarse steps the level is re-flagged: |grad rho| dx / rho > tol_rho, relative pressure
# jumps between neighbours > tol_p, interface curvature |div n| dx > tol_curv (n = grad a / |grad a|, in mixed
# cells) and field-line curvature |b . grad b| dx > tol_bcurv (where |B| is above 5% of its maximum). Flags are
# buffered, grouped into rectangles (connected components, merged when close), and kept away from the domain
# edges, and split where they are poorly filled; an optional region mask limits refinement to where it
# matters (e.g. the fuel core). New patches take the old fine data where they overlap an old patch and are prolonged elsewhere; their
# face fields come from a vector potential A_z (old fine A_z in the overlap, coarse A_z interpolated
# elsewhere), so they are divergence-free by construction and consistent with the coarse faces.
#
# Non-ideal terms couple across the levels the same way:
#   - resistive MHD: the eta J part of the corner EMF is in the EMF correction; the resistive Poynting flux and
#     the B_z diffusion flux are added to the recorded face fluxes and refluxed with them;
#   - strength: the deviatoric stress S is prolonged, restricted and time-interpolated into patch ghosts like
#     U, and the stress fluxes of momentum and energy are refluxed;
#   - conduction (RKL2) and grey FLD (backward Euler) are split steps: on a patch they conduct to ghost cells
#     taken from the coarse level at the end of the sub-step (and from touching patches), and each returns its
#     time-integrated face fluxes, which are refluxed into the coarse energy (conduction) or the coarse E_r
#     (FLD; E_r is prolonged and restricted too). Gas + radiation energy stays conserved to round-off.
#   - Hall term, anisotropic resistivity, Nernst and Ettingshausen: their corner EMFs are part of the corrected
#     EMF, their B_z and energy face fluxes are refluxed; gyroviscous stresses likewise;
#   - viscosity and ion conduction: split steps like conduction (ghost ring, time-integrated face fluxes);
#   - two-temperature: the electron entropy is advected with the hydro fluxes (refluxed with them), and an
#     electron-energy register (electron conduction, Ettingshausen) moves the refluxed electron share into the
#     coarse electron entropy; equilibration is local;
#   - optically thin radiation is local: the coarse level does not radiate under a patch (the patch does) and
#     totals() counts what all levels radiated.
# Scope: 2D Cartesian with constrained transport.
#
# Launch overhead: a fine sub-step of every patch (all three RK stages, the boundary registers, and the
# post-step when it has no split diffusion) is recorded once as a CUDA graph and replayed, so a level with
# tens of patches costs one graph launch per sub-step instead of tens of thousands of kernel launches from
# Python. dt and the ghost-interpolation weights are device scalars, ghost frames live in persistent
# buffers and the registers accumulate in place, so the same graph serves every sub-step until the next regrid.
import dataclasses

import numpy as np

from .config import Config, NVAR, RHO, MX, MY, MZ, BX, BY, BZ, EN, RHOC, RHOSE, iRHO, iVX, iVY, iBX, iBY, iBZ, iPR, iCLR
from .physics import xp, cons_to_prim
from .solver import MHDSolver, to_numpy
from . import ct


def _check(cfg):
    if cfg.geometry != "cartesian" or cfg.div_b != "ct":
        raise NotImplementedError("AMR: 2D Cartesian with constrained transport")
    if cfg.fld_groups:
        raise NotImplementedError("AMR: multigroup FLD is not coupled across levels (grey FLD is)")
    if cfg.pic:
        raise NotImplementedError("AMR: the kinetic-fluid hybrid (pic) runs on a single level")


# ------------------------------------------------------------------ kernels

_KSRC = r"""
extern "C" {
// limited-linear prolongation (see prolong()): one thread per fine cell, all variables; Uc (nv, NX, NY),
// out (nv, nfx, nfy); fine cell (fi, fj) lies in coarse array cell (i0 + ng + floor(xf), ...)
__global__ void prolong2d(const double* __restrict__ Uc, double* __restrict__ out, const int nv, const int NX,
                          const int NY, const int nfx, const int nfy, const int i0, const int j0, const int r,
                          const int ng) {
    long t = (long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= (long)nfx * nfy) return;
    int fi = t / nfy, fj = t % nfy;
    double xf = (fi - ng + 0.5) / r, yf = (fj - ng + 0.5) / r;
    double fx = floor(xf), fy = floor(yf);
    int ic = (int)fx + i0 + ng, jc = (int)fy + j0 + ng;
    double ox = xf - fx - 0.5, oy = yf - fy - 0.5;
    long nc = (long)NX * NY, k = (long)ic * NY + jc;
    double rx = 1e300, ry = 1e300;
    for (int v = 0; v < nv; v++) {
        const double* C = Uc + v * nc;
        double c0 = C[k];
        double dm = c0 - C[k - NY], dp = C[k + NY] - c0, c = 0.5 * (dm + dp);
        double mm = dm * dp > 0 ? (dm > 0 ? 1.0 : -1.0) * fmin(fabs(dm), fabs(dp)) : 0.0;
        rx = fmin(rx, fabs(c) > 1e-300 ? mm / (c == 0 ? 1.0 : c) : 1.0);
        dm = c0 - C[k - 1]; dp = C[k + 1] - c0; c = 0.5 * (dm + dp);
        mm = dm * dp > 0 ? (dm > 0 ? 1.0 : -1.0) * fmin(fabs(dm), fabs(dp)) : 0.0;
        ry = fmin(ry, fabs(c) > 1e-300 ? mm / (c == 0 ? 1.0 : c) : 1.0);
    }
    double px = fmin(fmax(rx, 0.0), 1.0), py = fmin(fmax(ry, 0.0), 1.0);
    for (int v = 0; v < nv; v++) {
        const double* C = Uc + v * nc;
        double c0 = C[k];
        double cx = 0.5 * (C[k + NY] - C[k - NY]), cy = 0.5 * (C[k + 1] - C[k - 1]);
        out[v * (long)nfx * nfy + t] = c0 + (px * cx) * ox + (py * cy) * oy;
    }
}

// ghost ring of A (nv, N0, N1) from the time-interpolated frames: (1 - th) Po + th Pn
__global__ void ghost_ring(double* __restrict__ A, const double* __restrict__ Po, const double* __restrict__ Pn,
                           const double* __restrict__ th_p, const int nv, const int N0, const int N1, const int ng) {
    long t = (long)blockDim.x * blockIdx.x + threadIdx.x;
    long n = (long)N0 * N1;
    if (t >= n * nv) return;
    long c = t % n;
    int i = c / N1, j = c % N1;
    if (i >= ng && i < N0 - ng && j >= ng && j < N1 - ng) return;
    double th = th_p[0];
    A[t] = (1.0 - th) * Po[t] + th * Pn[t];
}
}
"""
_KMOD = None


def _kern(name):
    global _KMOD
    if _KMOD is None:
        import cupy as cp
        _KMOD = cp.RawModule(code=_KSRC)
    return _KMOD.get_function(name)


def _blocks(n):
    return ((int(n) + 255) // 256,)


def _ghost_ring(A, Po, Pn, th, ng):
    """A's ghost ring from the frames at weight th (float or device scalar), in place, one launch"""
    if xp.__name__ != "cupy" or not (A.flags.c_contiguous and Po.flags.c_contiguous and Pn.flags.c_contiguous):
        G = (1.0 - th) * Po + th * Pn
        A[:, :ng, :] = G[:, :ng, :]; A[:, -ng:, :] = G[:, -ng:, :]
        A[:, :, :ng] = G[:, :, :ng]; A[:, :, -ng:] = G[:, :, -ng:]
        return
    thd = th if isinstance(th, xp.ndarray) else xp.full((), float(th))
    nv, N0, N1 = A.shape
    _kern("ghost_ring")(_blocks(A.size), (256,), (A, Po, Pn, thd, np.int32(nv), np.int32(N0), np.int32(N1),
                                                np.int32(ng)))


# ------------------------------------------------------------------ recording levels

class _Recorder:
    """mixin: accumulate, weighted by the RK stage weights, the fluxes / vf terms / EMFs on the boundaries of
    the boxes in self._rec_boxes (coarse cell indices (i0, i1, j0, j1), interior numbering, faces k = 0..n)"""

    def _rec_alloc(self, er=False):
        """zeroed registers for the current boxes, so a graph can be recorded on the first step after a regrid"""
        self._rec = []
        for (i0, i1, j0, j1) in self._rec_boxes:
            ri0, ri1, rj0, rj1 = (i0, i1, j0, j1) if self._rec_r == 0 else (0, self.cfg.nx, 0, self.cfg.ny)
            L = [rj1 - rj0] * 2 + [ri1 - ri0] * 2
            acc = {"F": [xp.zeros((NVAR, n)) for n in L], "u": [xp.zeros(n) for n in L],
                   "ua": [xp.zeros(n) for n in L], "E": [xp.zeros(n + 1) for n in L]}
            if er:
                acc["Er"] = [xp.zeros(n) for n in L]
            if self.cfg.two_temp:
                acc["Fe"] = [xp.zeros(n) for n in L]
            self._rec.append(acc)
        self._stage = None

    def _rec_reset(self, boxes, ratio):
        """new boxes: fresh registers; the same boxes again: zero them in place (graph replays write into them)"""
        old = getattr(self, "_rec", None)
        if old and all(a is not None for a in old) and list(boxes) == list(self._rec_boxes) \
                and ratio == self._rec_r:
            for acc in old:
                for arrs in acc.values():
                    for a in arrs:
                        if a is not None:
                            a.fill(0.0)
            self._stage = None
            return
        self._rec_boxes, self._rec_r = boxes, ratio
        self._rec = [None] * len(boxes)
        self._stage = None

    def _flux_hook(self, Fx, Fy, WLx, WRx, WLy, WRy):
        if not getattr(self, "_rec_boxes", None):
            return
        ng = self.ng

        def vf(F, WL, WR):                           # u_f and u_f a_f, as vf_advection_rate builds them
            F0 = F[RHO]
            u = F0 / xp.where(F0 >= 0, WL[iRHO], WR[iRHO])
            a = xp.where(xp.abs(F0) > 1e-300, F[RHOC] / xp.where(F0 == 0, 1.0, F0), 0.5 * (WL[iCLR] + WR[iCLR]))
            return u, u * a
        ux, uax = vf(Fx, WLx, WRx)
        uy, uay = vf(Fy, WLy, WRy)
        self._stage = []
        self._stage_own = False
        for (i0, i1, j0, j1) in self._rec_boxes:
            r = self._rec_r
            # this level's own box boundary in its index space (fine: the whole patch, coarse: the box)
            ri0, ri1, rj0, rj1 = (i0, i1, j0, j1) if r == 0 else (0, self.cfg.nx, 0, self.cfg.ny)
            rows = slice(ng + rj0, ng + rj1)
            cols = slice(ng + ri0, ng + ri1)
            rec = {"F": [Fx[:, ri0, rows], Fx[:, ri1, rows], Fy[:, cols, rj0], Fy[:, cols, rj1]],
                   "u": [ux[ri0, rows], ux[ri1, rows], uy[cols, rj0], uy[cols, rj1]],
                   "ua": [uax[ri0, rows], uax[ri1, rows], uay[cols, rj0], uay[cols, rj1]]}
            if self.cfg.two_temp:                    # electron energy fluxes (Ettingshausen) added below
                rec["Fe"] = [xp.zeros_like(a) for a in rec["u"]]
            self._stage.append(rec)

    @property
    def _records_fluxes(self):
        return bool(getattr(self, "_rec_boxes", None))

    def _sides(self, k, Gx, Gy, interior):
        """the four boundary pieces (left, right, bottom, top) of box k from x- and y-face arrays: Fx / Fy
        layout (rows / columns with ghosts) or interior faces only"""
        i0, i1, j0, j1 = self._rec_boxes[k]
        ri0, ri1, rj0, rj1 = (i0, i1, j0, j1) if self._rec_r == 0 else (0, self.cfg.nx, 0, self.cfg.ny)
        o = 0 if interior else self.ng
        rows, cols = slice(o + rj0, o + rj1), slice(o + ri0, o + ri1)
        return [Gx[..., ri0, rows], Gx[..., ri1, rows], Gy[..., cols, rj0], Gy[..., cols, rj1]]

    def _add_flux(self, var, Gx, Gy):
        if self._stage is None:
            return
        if var == "Fe":                              # electron energy flux (own register)
            for k, st in enumerate(self._stage):
                for m, g in enumerate(self._sides(k, Gx, Gy, False)):
                    st["Fe"][m] += g
            return
        if not self._stage_own:                      # the recorded pieces are views of Fx, Fy: copy before adding
            for st in self._stage:
                st["F"] = [a.copy() for a in st["F"]]
            self._stage_own = True
        for k, st in enumerate(self._stage):
            for m, g in enumerate(self._sides(k, Gx, Gy, False)):
                st["F"][m][var] += g

    def _add_split_flux(self, key, var, Gx, Gy):
        for k, acc in enumerate(self._rec):
            if acc is None:
                continue
            for m, g in enumerate(self._sides(k, Gx, Gy, True)):
                if key == "F":
                    acc["F"][m][var] += g
                else:
                    acc.setdefault(key, [None] * 4)
                    if acc[key][m] is None:
                        acc[key][m] = g.copy()
                    else:
                        acc[key][m] += g

    def _emf_hook(self, E):
        if getattr(self, "_rec_boxes", None) and self._stage is not None:
            for k, (i0, i1, j0, j1) in enumerate(self._rec_boxes):
                ri0, ri1, rj0, rj1 = (i0, i1, j0, j1) if self._rec_r == 0 else (0, self.cfg.nx, 0, self.cfg.ny)
                self._stage[k]["E"] = [E[ri0, rj0:rj1 + 1], E[ri1, rj0:rj1 + 1], E[ri0:ri1 + 1, rj0], E[ri0:ri1 + 1, rj1]]
        return E

    def _accumulate_energy_budget(self, weight):
        super()._accumulate_energy_budget(weight)
        if not getattr(self, "_rec_boxes", None) or self._stage is None:
            return
        for k, st in enumerate(self._stage):
            acc = self._rec[k]
            if acc is None:
                self._rec[k] = {key: [weight * a for a in st[key]] for key in st}
            else:
                for key in st:
                    for a_acc, a in zip(acc[key], st[key]):
                        _axpy(weight, a, a_acc)      # in place: a recorded graph replays into the same buffers
        self._stage = None


_AXPY = None


def _axpy(w, a, out):
    """out += w a (w a float or a device scalar) in one launch"""
    global _AXPY
    if xp.__name__ != "cupy":
        out += w * a
        return
    if _AXPY is None:
        import cupy as cp
        _AXPY = cp.ElementwiseKernel("raw float64 w, float64 a", "float64 y", "y += w[0] * a", "amr_axpy")
    _AXPY(w if isinstance(w, xp.ndarray) else xp.full((), float(w)), a, out)


class CoarseLevel(_Recorder, MHDSolver):
    _uncovered = None                                # interior cells not under a patch (set at regrid)

    def _rad_mask(self):
        return self._uncovered


class FinePatch(_Recorder, MHDSolver):
    """an MHDSolver on a sub-rectangle. Ghost cells and faces come from neighbouring patches where they overlap
    one (same stage, the patches advance in lockstep) and from the coarse level elsewhere (space-time
    interpolated). gx0, gy0: global fine index of the first interior cell."""

    neighbours = ()
    _th_d = None                                     # device ghost weights: RK stages 0, 1, 2 and the sub-step end

    def _set_ghost_source(self, P_old, P_new, t0, dtc, S=None, E=None):
        """coarse data prolonged onto this patch (with ghosts) before and after the coarse step: U, and the
        stress S and radiation energy E_r frames when those are on. Copied into persistent buffers (a recorded
        graph reads those addresses)."""
        def put(name, val):
            cur = getattr(self, name, None)
            if val is None or cur is None or cur.shape != val.shape:
                setattr(self, name, val)
            else:
                cur[...] = val
        put("_P_old", P_old); put("_P_new", P_new)
        for nm, fr in (("_S", S), ("_E", E)):
            put(nm + "_old", None if fr is None else fr[0]); put(nm + "_new", None if fr is None else fr[1])
        self._S_frames = None if S is None else (self._S_old, self._S_new)
        self._E_frames = None if E is None else (self._E_old, self._E_new)
        self._t0, self._dtc = t0, dtc
        self._k = 0


    def _ghost_fill(self, A, frames, cur, interior_only=False):
        """ghost-inclusive array of a field from the coarse frames at the current theta, with touching patches'
        current values (attribute `cur`) where they overlap; A (if given) keeps its interior"""
        th = getattr(self, "_theta", 1.0)
        G = (1.0 - th) * frames[0] + th * frames[1]
        ng = self.ng
        for o in self.neighbours:
            ov = self._overlaps(o, 0, 0)
            if ov is not None:
                src = getattr(o, cur)
                if interior_only:                    # E_r has no ghost cells
                    G[(slice(None),) + ov[0]] = src[None][(slice(None),) + tuple(slice(t.start - ng, t.stop - ng) for t in ov[1])]
                else:
                    G[(slice(None),) + ov[0]] = src[(slice(None),) + ov[1]]
        if A is not None:
            G[:, ng:-ng, ng:-ng] = A[:, ng:-ng, ng:-ng]
        return G

    def _halo(self, A):
        """ghost cells of the stress array (compute_rhs calls this after the domain boundary fill)"""
        if getattr(self, "_S_frames", None) is not None and A is not None and A.shape == self.S.shape:
            _ghost_ring(A, self._S_frames[0], self._S_frames[1], getattr(self, "_theta", 1.0), self.ng)
            for o in self.neighbours:
                ov = self._ov(o, 0, 0)
                if ov is not None:
                    A[(slice(None),) + ov[0]] = o._cur_S[(slice(None),) + ov[1]]

    def _post_step(self, dt):
        if self._th_d is not None:                   # end of this sub-step: the split steps' ghosts
            self._theta = self._th_d[3]
        else:
            self._theta = (self.t + dt - self._t0) / self._dtc
        self.apply_bc(self.U)
        super()._post_step(dt)

    def _split_ghosts(self):
        return self.U

    def _fld_ghosts(self):
        if getattr(self, "_E_frames", None) is None:
            return None
        ng, nx, ny = self.ng, self.cfg.nx, self.cfg.ny
        G = self._ghost_fill(None, self._E_frames, "Er", interior_only=True)[0]
        rho = xp.maximum(self.U[RHO], 1e-300)
        vx, vy = self.U[MX] / rho, self.U[MY] / rho
        J, I = slice(ng, ng + ny), slice(ng, ng + nx)
        return {(0, "lo"): {"E": G[ng - 1, J][None], "v": vx[ng - 1, J][None]},
                (0, "hi"): {"E": G[ng + nx, J][None], "v": vx[ng + nx, J][None]},
                (1, "lo"): {"E": G[I, ng - 1][:, None], "v": vy[I, ng - 1][:, None]},
                (1, "hi"): {"E": G[I, ng + ny][:, None], "v": vy[I, ng + ny][:, None]}}

    def compute_rhs(self, U, out=None, Bf=None, dBf=None, S=None, dS=None):
        if self._th_d is not None:
            self._theta = self._th_d[self._k % 3]
        else:
            ts = (0.0, self.dt, 0.5 * self.dt)[self._k % 3]
            self._theta = (self.t + ts - self._t0) / self._dtc
        self._k += 1
        return super().compute_rhs(U, out, Bf, dBf, S, dS)

    def _overlaps(self, o, sx, sy):
        """(my index slices, o's index slices) where o's interior (shifted by sx, sy in its index space, 0 for
        cells, 1 to include the upper boundary face) covers my ghost-inclusive array"""
        ng = self.ng
        N = (self.U.shape[1] + sx, self.U.shape[2] + sy)
        lo = (max(0, o.gx0 - self.gx0 + ng), max(0, o.gy0 - self.gy0 + ng))
        hi = (min(N[0], o.gx0 + o.cfg.nx + sx - self.gx0 + ng), min(N[1], o.gy0 + o.cfg.ny + sy - self.gy0 + ng))
        if lo[0] >= hi[0] or lo[1] >= hi[1]:
            return None
        mine = (slice(lo[0], hi[0]), slice(lo[1], hi[1]))
        theirs = tuple(slice(l + s - o_ + ng, h + s - o_ + ng) for l, h, s, o_ in
                       ((lo[0], hi[0], self.gx0 - ng, o.gx0), (lo[1], hi[1], self.gy0 - ng, o.gy0)))
        return mine, theirs

    def _ov(self, o, sx, sy):
        key = (id(o), sx, sy)
        c = self.__dict__.setdefault("_ov_cache", {})
        if key not in c:
            c[key] = self._overlaps(o, sx, sy)
        return c[key]

    def apply_bc(self, U):
        _ghost_ring(U, self._P_old, self._P_new, getattr(self, "_theta", 1.0), self.ng)
        for o in self.neighbours:                    # fine data from touching patches, same stage (it lands in
            ov = self._ov(o, 0, 0)                   # the ghost ring only: patch interiors never overlap)
            if ov is not None:
                U[(slice(None),) + ov[0]] = o._cur_U[(slice(None),) + ov[1]]
        return U

    def _ghost_faces(self, Bf, U):
        super()._ghost_faces(Bf, U)
        ng, nx, ny = self.ng, self.cfg.nx, self.cfg.ny
        keep = [Bf[0][ng:ng + nx + 1, ng:ng + ny].copy(), Bf[1][ng:ng + nx, ng:ng + ny + 1].copy()]
        for o in self.neighbours:
            for a, (sx, sy) in enumerate(((1, 0), (0, 1))):
                ov = self._ov(o, sx, sy)
                if ov is not None:
                    Bf[a][ov[0]] = o._cur_F[a][ov[1]]
        Bf[0][ng:ng + nx + 1, ng:ng + ny] = keep[0]
        Bf[1][ng:ng + nx, ng:ng + ny + 1] = keep[1]

    # SSP-RK3 split into stages so that all patches can take each stage together (_rk_core, stage by stage)
    def rk_begin(self, dt):
        self.dt = dt
        self._glm_ch_frozen = self.cfg.glm_ch
        self._ct_sync()
        self._cur_U, self._cur_F = self.U, self.Bf
        self._cur_S = self.S if self.strength else None
        self._U0 = self.U

    def stage_rhs(self, k):
        w = (1 / 6, 1 / 6, 2 / 3)[k] * self.dt
        U = self.U if k == 0 else self.U_temp
        F = self.Bf if k == 0 else self._Bf1
        S, dS = ((self.S if k == 0 else self._S1), self._dS) if self.strength else (None, None)
        self.compute_rhs(U, out=self.dU, Bf=F, dBf=self._dBf, S=S, dS=dS)
        self._accumulate_energy_budget(w)

    def stage_update(self, k):
        dt, U0, F0, F1, dF = self.dt, self.U, self.Bf, self._Bf1, self._dBf
        if self.strength:
            S0, S1, dS = self.S, self._S1, self._dS
            if k == 0:
                xp.add(S0, dt * dS, out=S1)
            elif k == 1:
                S1[...] = 0.75 * S0 + 0.25 * (S1 + dt * dS)
            else:
                S0[...] = S0 / 3.0 + 2.0 / 3.0 * (S1 + dt * dS)
            self._cur_S = S0 if k == 2 else S1
        if k == 0:
            for f0, f1, d in zip(F0, F1, dF):
                xp.add(f0, dt * d, out=f1)
            xp.add(U0, dt * self.dU, out=self.U_temp)
        elif k == 1:
            for f0, f1, d in zip(F0, F1, dF):
                f1[...] = 0.75 * f0 + 0.25 * (f1 + dt * d)
            self.U_temp[...] = 0.75 * U0 + 0.25 * (self.U_temp + dt * self.dU)
        else:
            for f0, f1, d in zip(F0, F1, dF):
                f0[...] = f0 / 3.0 + 2.0 / 3.0 * (f1 + dt * d)
            self.U[...] = U0 / 3.0 + 2.0 / 3.0 * (self.U_temp + dt * self.dU)
        target = self.U if k == 2 else self.U_temp
        self.enforce_scalar_bounds(target)
        if k == 2:
            ct.cells_from_faces(self.U, self.Bf, (slice(self.ng, self.ng + self.cfg.nx),
                                                  slice(self.ng, self.ng + self.cfg.ny)))
        self._cur_U, self._cur_F = target, (self.Bf if k == 2 else F1)


# ------------------------------------------------------------------ prolongation, vector potential

_PROLONG_IDX = {}


def _prolong_index(i0, j0, nxf, nyf, r, ng):
    """device index / offset arrays of prolong(), cached (they were re-uploaded on every call)"""
    key = (i0, j0, nxf, nyf, r, ng)
    if key not in _PROLONG_IDX:
        if len(_PROLONG_IDX) > 4096:
            _PROLONG_IDX.clear()
        xf = (np.arange(nxf + 2 * ng) - ng + 0.5) / r       # fine centres in coarse-cell units from the box origin
        yf = (np.arange(nyf + 2 * ng) - ng + 0.5) / r
        ic = np.floor(xf).astype(int) + i0 + ng             # coarse array index (ghost-inclusive)
        jc = np.floor(yf).astype(int) + j0 + ng
        lo_i, hi_i, lo_j, hi_j = ic.min() - 1, ic.max() + 2, jc.min() - 1, jc.max() + 2
        _PROLONG_IDX[key] = (xp.asarray(xf - np.floor(xf) - 0.5)[:, None], xp.asarray(yf - np.floor(yf) - 0.5)[None, :],
                             (lo_i, hi_i, lo_j, hi_j), xp.asarray(ic - lo_i)[:, None], xp.asarray(jc - lo_j)[None, :])
    return _PROLONG_IDX[key]


def prolong(Uc, i0, j0, nxf, nyf, r, ng):
    """conservative limited-linear prolongation of the coarse conserved array (with ghosts) onto a fine array of
    nxf x nyf interior cells (+ ng ghosts) starting at coarse interior cell (i0, j0); one limiter factor per
    coarse cell for all variables, so linear relations between them survive. One kernel launch on the GPU."""
    if xp.__name__ == "cupy":
        Uc = xp.ascontiguousarray(Uc)
        nv, NX, NY = Uc.shape
        nfx, nfy = nxf + 2 * ng, nyf + 2 * ng
        out = xp.empty((nv, nfx, nfy))
        _kern("prolong2d")(_blocks(nfx * nfy), (256,), (Uc, out, np.int32(nv), np.int32(NX), np.int32(NY),
                                                       np.int32(nfx), np.int32(nfy), np.int32(i0), np.int32(j0),
                                                       np.int32(r), np.int32(ng)))
        return out
    ox, oy, (lo_i, hi_i, lo_j, hi_j), I, J = _prolong_index(i0, j0, nxf, nyf, r, ng)
    C = Uc[:, lo_i:hi_i, lo_j:hi_j]

    def slopes(C, ax):
        f = C.shape[1 + ax]
        lo, mid, hi = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
        lo[1 + ax], mid[1 + ax], hi[1 + ax] = slice(0, f - 2), slice(1, f - 1), slice(2, f)
        dm, dp = C[tuple(mid)] - C[tuple(lo)], C[tuple(hi)] - C[tuple(mid)]
        c = 0.5 * (dm + dp)
        mm = xp.where(dm * dp > 0, xp.sign(dm) * xp.minimum(xp.abs(dm), xp.abs(dp)), 0.0)
        ratio = xp.where(xp.abs(c) > 1e-300, mm / xp.where(c == 0, 1.0, c), 1.0)
        phi = xp.clip(xp.min(ratio, axis=0), 0.0, 1.0)      # one factor for all variables
        out = xp.zeros_like(C)
        out[tuple(mid)] = phi[None] * c
        return out
    Sx, Sy = slopes(C, 0), slopes(C, 1)
    return C[:, I, J] + Sx[:, I, J] * ox + Sy[:, I, J] * oy


def potential(Bxf, Byf, k0, k1, l0, l1, h):
    """A_z on the corners (k0..k1) x (l0..l1) (face indices) from face fields: dA/dy = B_x, dA/dx = -B_y,
    A = 0 at (k0, l0); exact for a divergence-free field (path independent)"""
    dx, dy = h
    col = xp.concatenate([xp.zeros(1), xp.cumsum(Bxf[k0, l0:l1]) * dy])           # up the left edge
    row = -xp.cumsum(Byf[k0:k1, l0:l1 + 1], axis=0) * dx                            # then along x
    A = xp.concatenate([col[None, :], col[None, :] + row], axis=0)
    return A


# ------------------------------------------------------------------ driver

@dataclasses.dataclass
class AMRParams:
    ratio: int = 4
    regrid_every: int = 8
    tol_rho: float = 0.05
    tol_p: float = 0.1
    tol_curv: float = 0.0                            # 0: off
    tol_bcurv: float = 0.0
    buffer: int = 3
    margin: int = 3
    min_size: int = 4
    max_patches: int = 64
    efficiency: float = 0.7                          # split boxes filled less than this
    region: object = None                            # optional callable(coarse solver) -> host bool mask (interior)
    graphs: bool = True                              # record the fine sub-step as a CUDA graph


class AMR:
    """two-level AMR driver over a 2D Cartesian Config (the coarse grid)"""

    def __init__(self, cfg, params=None):
        _check(cfg)
        self.p = params or AMRParams()
        self.cfg = cfg
        phys = dataclasses.replace(cfg.physics, use_cuda_graphs=False)
        c = Config.__new__(Config)
        c.grid, c.physics, c.io = cfg.grid, phys, cfg.io
        self.coarse = CoarseLevel(c)
        self.coarse._rec_reset([], 0)
        self.patches, self.boxes = [], []
        self.t, self.step = 0.0, 0
        self.fine_cell_steps = 0
        self._rad_retired = 0.0
        self._dt_d, self._th_d = xp.zeros(()), xp.zeros(4)    # fine dt and ghost weights, read by the patches
        self._cdt_d = xp.zeros(())
        self._graphs, self._warm = {}, set()
        self.graph_launches = {}                     # replays per graph kind (diagnostic)

    # -- setup
    def set_state(self, U):
        """coarse conserved state (with ghosts); call regrid() afterwards"""
        self.coarse.U = U
        self.coarse.t, self.coarse.step = 0.0, 0
        self.coarse.apply_bc(self.coarse.U)
        self.coarse._ct_sync()
        self.coarse._te_sync()                       # two-temperature: T_e = T_i where s_e was never set

    def _fine_cfg(self, box):
        i0, i1, j0, j1 = box
        c, r = self.cfg, self.p.ratio
        grid = dataclasses.replace(c.grid, nx=r * (i1 - i0), ny=r * (j1 - j0), x_min=c.x_min + i0 * c.dx,
                                   x_max=c.x_min + i1 * c.dx, y_min=c.y_min + j0 * c.dy, y_max=c.y_min + j1 * c.dy,
                                   x_spacing=c.dx / r)
        phys = dataclasses.replace(c.physics, use_cuda_graphs=False, bc_x_type="extrapolation",
                                   bc_y_type="extrapolation")
        out = Config.__new__(Config)
        out.grid, out.physics, out.io = grid, phys, c.io
        return out

    # -- flagging and clustering
    def flags(self):
        s, p = self.coarse, self.p
        ng = s.ng
        W = cons_to_prim(s.U, s.eos)
        rho, pr = W[iRHO], W[iPR]
        I = (slice(ng, -ng), slice(ng, -ng))
        gx = 0.5 * (rho[2:, 1:-1] - rho[:-2, 1:-1])
        gy = 0.5 * (rho[1:-1, 2:] - rho[1:-1, :-2])
        f = xp.sqrt(gx**2 + gy**2)[1:-1, 1:-1] / rho[I] > p.tol_rho
        jump = lambda a, b: xp.abs(a - b) / xp.minimum(xp.abs(a), xp.abs(b))
        jp = xp.maximum(xp.maximum(jump(pr[I], pr[ng + 1:-ng + 1, ng:-ng]), jump(pr[I], pr[ng - 1:-ng - 1, ng:-ng])),
                        xp.maximum(jump(pr[I], pr[ng:-ng, ng + 1:-ng + 1]), jump(pr[I], pr[ng:-ng, ng - 1:-ng - 1])))
        f = f | (jp > p.tol_p)
        if p.tol_curv > 0:                           # interface curvature, mixed cells
            a = xp.clip(W[iCLR], 0, 1)
            ax, ay = xp.gradient(a, axis=0), xp.gradient(a, axis=1)
            m = xp.sqrt(ax**2 + ay**2) + 1e-12
            k = xp.gradient(ax / m, axis=0) + xp.gradient(ay / m, axis=1)
            f = f | ((xp.abs(k[I]) > p.tol_curv) & (a[I] > 0.05) & (a[I] < 0.95))
        if p.tol_bcurv > 0:                          # field-line curvature where the field matters
            bx, by = W[iBX], W[iBY]
            bm = xp.sqrt(bx**2 + by**2 + W[iBZ]**2) + 1e-30
            ux, uy = bx / bm, by / bm
            kx = ux * xp.gradient(ux, axis=0) + uy * xp.gradient(ux, axis=1)
            ky = ux * xp.gradient(uy, axis=0) + uy * xp.gradient(uy, axis=1)
            strong = bm[I] > 0.05 * float(xp.max(bm[I]))
            f = f | ((xp.sqrt(kx**2 + ky**2)[I] > p.tol_bcurv) & strong)
        f = to_numpy(f)
        if p.region is not None:                     # refine only where the caller says (e.g. the fuel core)
            f &= np.asarray(p.region(s), bool)
        return f

    def cluster(self, f):
        """flagged cells -> rectangles: buffered flags, connected components, overlapping boxes merged, then
        boxes filled less than `efficiency` split (at a hole in the flag signature, else at the middle of the
        longer side) and shrunk to their flags, recursively. Boxes may touch, never overlap."""
        p = self.p
        nx, ny = f.shape
        b, m = p.buffer, p.margin
        g = np.zeros_like(f)
        for i, j in np.argwhere(f):
            g[max(i - b, 0):i + b + 1, max(j - b, 0):j + b + 1] = True
        g[:m, :] = g[nx - m:, :] = False
        g[:, :m] = g[:, ny - m:] = False
        lab = np.zeros(f.shape, int)
        boxes, n = [], 0
        for i, j in np.argwhere(g):                  # connected components (host flood fill, small grid)
            if lab[i, j]:
                continue
            n += 1
            stack, lab[i, j] = [(i, j)], n
            lo_i = hi_i = i; lo_j = hi_j = j
            while stack:
                a, c = stack.pop()
                lo_i, hi_i, lo_j, hi_j = min(lo_i, a), max(hi_i, a), min(lo_j, c), max(hi_j, c)
                for da, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, c + dc
                    if 0 <= x < nx and 0 <= y < ny and g[x, y] and not lab[x, y]:
                        lab[x, y] = n
                        stack.append((x, y))
            boxes.append([lo_i, hi_i + 1, lo_j, hi_j + 1])
        merged = True                                # bounding boxes of different components may overlap
        while merged:
            merged = False
            for a_ in range(len(boxes)):
                for c_ in range(a_ + 1, len(boxes)):
                    A, B = boxes[a_], boxes[c_]
                    if A[0] < B[1] and B[0] < A[1] and A[2] < B[3] and B[2] < A[3]:
                        boxes[a_] = [min(A[0], B[0]), max(A[1], B[1]), min(A[2], B[2]), max(A[3], B[3])]
                        boxes.pop(c_)
                        merged = True
                        break
                if merged:
                    break

        def shrink(bx):
            i0, i1, j0, j1 = bx
            sub = g[i0:i1, j0:j1]
            if not sub.any():
                return None
            ii, jj = np.nonzero(sub.any(axis=1))[0], np.nonzero(sub.any(axis=0))[0]
            return [i0 + ii[0], i0 + ii[-1] + 1, j0 + jj[0], j0 + jj[-1] + 1]

        def split(bx):
            bx = shrink(bx)
            if bx is None:
                return []
            i0, i1, j0, j1 = bx
            sub = g[i0:i1, j0:j1]
            ax = 0 if i1 - i0 >= j1 - j0 else 1
            L = sub.shape[ax]
            if sub.mean() >= p.efficiency or L < 2 * p.min_size:
                return [bx]
            sig = sub.sum(axis=1 - ax)
            holes = [k for k in range(p.min_size, L - p.min_size + 1) if sig[k - 1] == 0 or (k < L and sig[k] == 0)]
            cut = min(holes, key=lambda k: abs(k - L / 2)) if holes else L // 2
            if ax == 0:
                return split([i0, i0 + cut, j0, j1]) + split([i0 + cut, i1, j0, j1])
            return split([i0, i1, j0, j0 + cut]) + split([i0, i1, j0 + cut, j1])
        out = []
        for bx in boxes:
            out += split(bx)
        return [tuple(int(v) for v in bx) for bx in out if bx[1] - bx[0] >= 2 and bx[3] - bx[2] >= 2][:p.max_patches]

    # -- (re)gridding
    def regrid(self, boxes=None):
        """new patches on boxes (default: from the flags). Data: the old fine data where an old patch covers the
        cells, prolonged coarse data elsewhere. Faces: one composite fine-resolution A_z over the domain (coarse
        A_z interpolated, old fine A_z offset-matched into the old patches), so every new patch, and two that
        touch, gets divergence-free faces that agree on shared edges and sum to the coarse face fluxes."""
        boxes = self.cluster(self.flags()) if boxes is None else boxes
        if [tuple(b_) for b_ in boxes] == self.boxes and self.patches:
            return                                   # same boxes: keep the patches (and their recorded graphs)
        old = list(zip(self.boxes, self.patches))
        s, r, ng = self.coarse, self.p.ratio, self.coarse.ng
        nx, ny = s.cfg.nx, s.cfg.ny
        self._rad_retired += sum(float(op._acc_rad) for _, op in old)      # what replaced patches radiated
        cov = np.zeros((nx, ny), bool)
        for (i0, i1, j0, j1) in boxes:
            cov[i0:i1, j0:j1] = True
        s._uncovered = xp.asarray(~cov)
        s._ct_sync()
        Ac = potential(s.Bf[0], s.Bf[1], ng, ng + nx, ng, ng + ny, (s.cfg.dx, s.cfg.dy))
        xs, ys = np.arange(r * nx + 1) / r, np.arange(r * ny + 1) / r
        ii = np.minimum(np.floor(xs).astype(int), nx - 1); wx = xp.asarray(xs - ii)[:, None]
        jj = np.minimum(np.floor(ys).astype(int), ny - 1); wy = xp.asarray(ys - jj)[None, :]
        I, J = xp.asarray(ii)[:, None], xp.asarray(jj)[None, :]
        Af = ((1 - wx) * (1 - wy) * Ac[I, J] + wx * (1 - wy) * Ac[I + 1, J] + (1 - wx) * wy * Ac[I, J + 1]
              + wx * wy * Ac[I + 1, J + 1])
        Erc = self._padded_Er()
        for (ob, op) in old:
            Ao = potential(op.Bf[0], op.Bf[1], ng, ng + op.cfg.nx, ng, ng + op.cfg.ny, (op.cfg.dx, op.cfg.dy))
            ci, cj = r * ob[0], r * ob[2]
            Af[ci:ci + Ao.shape[0], cj:cj + Ao.shape[1]] = Ao + (Af[ci, cj] - Ao[0, 0])
        new = []
        for box in boxes:
            i0, i1, j0, j1 = box
            fc = self._fine_cfg(box)
            f = FinePatch(fc)
            f._rec_reset([box], r)
            nxf, nyf = fc.nx, fc.ny
            U = prolong(s.U, i0, j0, nxf, nyf, r, ng)
            S = prolong(s.S, i0, j0, nxf, nyf, r, ng) if s.strength else None
            Er = prolong(Erc, i0, j0, nxf, nyf, r, ng)[0, ng:-ng, ng:-ng] if Erc is not None else None
            for (ob, op) in old:                     # overlap with an old patch: take its fine data
                a0, a1, b0, b1 = max(i0, ob[0]), min(i1, ob[1]), max(j0, ob[2]), min(j1, ob[3])
                if a0 >= a1 or b0 >= b1:
                    continue
                fi = slice(ng + r * (a0 - i0), ng + r * (a1 - i0)); fj = slice(ng + r * (b0 - j0), ng + r * (b1 - j0))
                oi = slice(ng + r * (a0 - ob[0]), ng + r * (a1 - ob[0])); oj = slice(ng + r * (b0 - ob[2]), ng + r * (b1 - ob[2]))
                U[:, fi, fj] = op.U[:, oi, oj]
                if S is not None:
                    S[:, fi, fj] = op.S[:, oi, oj]
                if Er is not None:
                    Er[fi.start - ng:fi.stop - ng, fj.start - ng:fj.stop - ng] = \
                        op.Er[oi.start - ng:oi.stop - ng, oj.start - ng:oj.stop - ng]
            f.U = U
            if S is not None:
                f.S = xp.ascontiguousarray(S)
            if Er is not None:
                f.Er = xp.ascontiguousarray(Er)
            A = Af[r * i0:r * i1 + 1, r * j0:r * j1 + 1]
            Bx = xp.zeros((nxf + 2 * ng + 1, nyf + 2 * ng)); By = xp.zeros((nxf + 2 * ng, nyf + 2 * ng + 1))
            Bx[ng:ng + nxf + 1, ng:ng + nyf] = (A[:, 1:] - A[:, :-1]) / fc.dy
            By[ng:ng + nxf, ng:ng + nyf + 1] = -(A[1:, :] - A[:-1, :]) / fc.dx
            f.Bf = [Bx, By]
            f._faces_owner = f.U
            f._Bf1 = [xp.empty_like(a) for a in f.Bf]
            f._dBf = [xp.empty_like(a) for a in f.Bf]
            inner = (slice(ng, ng + nxf), slice(ng, ng + nyf))
            ct.cells_from_faces(f.U, f.Bf, inner)          # total energy kept as prolonged (conservative)
            f.t, f.step = self.t, 0
            f.gx0, f.gy0 = r * i0, r * j0
            f._cur_U, f._cur_F = f.U, f.Bf
            new.append(f)
        for f, bf in zip(new, boxes):                # touching or close: within a ghost layer of each other
            f.neighbours = [o for o, bo in zip(new, boxes) if o is not f and bo[0] <= bf[1] and bf[0] <= bo[1]
                            and bo[2] <= bf[3] and bf[2] <= bo[3]]
        for k, f in enumerate(new):
            f._th_d = self._th_d
            f._rec_alloc(self.cfg.fld)
        self.boxes, self.patches = [tuple(b_) for b_ in boxes], new
        s._rec_reset(self.boxes, 0)
        s._rec_alloc(self.cfg.fld)
        # corners on several patch boundaries: the lower-index patch owns them (EMF correction in _sync)
        setc = np.zeros((nx + 1, ny + 1), bool)
        self._emf_free = []
        for (i0, i1, j0, j1) in self.boxes:
            masks = []
            for ksl, lsl in ((i0, slice(j0, j1 + 1)), (i1, slice(j0, j1 + 1)), (slice(i0, i1 + 1), j0),
                             (slice(i0, i1 + 1), j1)):
                masks.append(xp.asarray(~setc[ksl, lsl]))
                setc[ksl, lsl] = True
            self._emf_free.append(masks)
        self._graphs = {}                            # new patches: record new graphs

    # -- time stepping
    def compute_dt(self):
        s = self.coarse
        dt = s.compute_dt(cons_to_prim(s.U, s.eos))
        for f in self.patches:                       # ghost cells hold the last sub-step's values: fine for dt
            dt = min(dt, self.p.ratio * f.compute_dt(cons_to_prim(f.U, f.eos)))
        return dt

    def _padded_Er(self, Er=None):
        """coarse E_r with ng edge-copied ghost cells, shaped (1, ...) for prolong(); None without FLD"""
        s = self.coarse
        Er = s.Er if Er is None else Er
        return None if Er is None else xp.pad(Er, s.ng, mode="edge")[None]

    def _ghost_frames(self, f, old=None):
        """coarse data prolonged onto patch f before (old = (U, S, E_r) saved before the coarse step) and after
        the coarse step: (U frames, S frames or None, E_r frames or None)"""
        s, ng, r = self.coarse, self.coarse.ng, self.p.ratio
        i0, i1, j0, j1 = self.boxes[self.patches.index(f)]
        P = lambda A: prolong(A, i0, j0, f.cfg.nx, f.cfg.ny, r, ng)
        new = (s.U, s.S if s.strength else None, self._padded_Er())
        old = new if old is None else old
        return tuple(None if a is None else (P(a), P(b)) for a, b in zip(old, new))

    def advance(self, dt):
        s, r, ng = self.coarse, self.p.ratio, self.coarse.ng
        old = (s.U.copy(), s.S.copy() if s.strength else None, self._padded_Er())
        s._rec_reset(self.boxes, 0)
        s.dt = dt
        self._coarse_step(dt)
        s.t += dt
        s.step += 1
        P = self.patches
        for k, f in enumerate(P):
            (Po, Pn), Sf, Ef = self._ghost_frames(f, old)
            f._set_ghost_source(Po, Pn, self.t, dt, Sf, Ef)
            f._rec_reset([self.boxes[k]], r)
        h = dt / r
        c = self.cfg                                 # split diffusion decides on the host (stage counts)
        post_in_graph = not (c.cond_kappa0 > 0 or c.fld or c.cond_kappa_i0 > 0 or c.visc_coeff > 0)
        self._dt_d.fill(h)
        for m in range(r):                           # all patches take each RK stage together
            t1 = self.t + m * h
            self._th_d.set(np.array([t1 - self.t, t1 + h - self.t, t1 + 0.5 * h - self.t, t1 + h - self.t]) / dt)
            self._fine_substep(P, post_in_graph)
            for f in P:
                if not post_in_graph:
                    f._post_step(h)
                f.t += h
                f.step += 1
        for f in P:
            f._theta = 1.0
            f.apply_bc(f.U)
            self.fine_cell_steps += r * f.cfg.nx * f.cfg.ny
        self._sync()
        self.t += dt
        self.step += 1
        if self.p.regrid_every and self.step % self.p.regrid_every == 0:
            self.regrid()

    def _coarse_step(self, dt):
        """the coarse SSP-RK3 core with its recording hooks, as a CUDA graph like the fine sub-step (eager
        right after a regrid, then recorded and replayed); the split post-step runs eagerly"""
        s = self.coarse
        s._glm_ch_frozen = s.cfg.glm_ch
        s._ct_sync()
        s._te_sync()
        self._cdt_d.fill(dt)
        self._graphed("coarse", lambda: s._rk_core(self._cdt_d))
        s._post_step(dt)

    def _graphed(self, name, fn):
        """run fn as a CUDA graph: recorded on first use after a regrid (the very first call of each kind runs
        eagerly, so kernels are compiled and buffers exist), replayed afterwards; eager without CuPy or patches"""
        if not (self.p.graphs and xp.__name__ == "cupy" and self.patches):
            return fn()
        if name not in self._warm:
            self._warm.add(name)
            return fn()
        import cupy as cp
        if name not in self._graphs:
            pool, stream = cp.cuda.MemoryPool(), cp.cuda.Stream(non_blocking=True)
            cp.cuda.Device().synchronize()
            with cp.cuda.using_allocator(pool.malloc), stream:
                stream.begin_capture()
                fn()
                self._graphs[name] = (stream.end_capture(), stream, pool)
        g, stream, _ = self._graphs[name]
        self.graph_launches[name] = self.graph_launches.get(name, 0) + 1
        cp.cuda.Device().synchronize()
        g.launch(stream=stream)
        stream.synchronize()

    def _fine_core(self, P, post):
        for f in P:
            f.rk_begin(self._dt_d)
        for k in range(3):
            for f in P:
                f.stage_rhs(k)
            for f in P:
                f.stage_update(k)
        if post:
            for f in P:
                f._post_step(self._dt_d)

    def _fine_substep(self, P, post):
        """one fine sub-step of every patch, as one recorded graph (the registers accumulate in place)"""
        self._graphed("fine" + ("+post" if post else ""), lambda: self._fine_core(P, post))

    def _sync(self):
        """synchronisation after the fine sub-steps (_sync_core), as one recorded graph"""
        self._graphed("sync", self._sync_core)

    def _sync_core(self):
        s, r, ng = self.coarse, self.p.ratio, self.coarse.ng
        dx, dy = s.cfg.dx, s.cfg.dy
        nx, ny = s.cfg.nx, s.cfg.ny
        dU = xp.zeros_like(s.U)
        dEr = None if s.Er is None else xp.zeros_like(s.Er)
        dEc = xp.zeros((nx + 1, ny + 1))
        a_c = cons_to_prim(s.U, s.eos)[iCLR]
        for k, (box, f) in enumerate(zip(self.boxes, self.patches)):
            i0, i1, j0, j1 = box
            C, Fr = s._rec[k], f._rec[0]
            agg = lambda a: a.reshape(a.shape[:-1] + (-1, r)).mean(axis=-1)          # r fine faces -> one coarse
            dF = [agg(Fr["F"][m]) - C["F"][m] for m in range(4)]
            for d in dF:                             # the tracer is refluxed with its own two terms below
                d[RHOC] = 0.0
            du = [agg(Fr["u"][m]) - C["u"][m] for m in range(4)]
            dua = [agg(Fr["ua"][m]) - C["ua"][m] for m in range(4)]
            rows, cols = slice(ng + j0, ng + j1), slice(ng + i0, ng + i1)
            # reflux: left (outside cell's high face), right (low face), bottom, top
            for m, (ci, cj, sign, h) in enumerate(((ng + i0 - 1, rows, -1, dx), (ng + i1, rows, +1, dx),
                                                   (cols, ng + j0 - 1, -1, dy), (cols, ng + j1, +1, dy))):
                dU[:, ci, cj] += sign * dF[m] / h
                dU[RHOC, ci, cj] += sign * (dua[m] - a_c[ci, cj] * du[m]) / h
                if "Fe" in C:                        # electron energy: p_e += (gamma - 1) e_e, rho s_e = p_e rho^(1-g)
                    g_ = s.cfg.gamma
                    dU[RHOSE, ci, cj] += sign * (g_ - 1) * (agg(Fr["Fe"][m]) - C["Fe"][m]) / h \
                        * xp.maximum(s.U[RHO, ci, cj], 1e-300)**(1 - g_)
                if dEr is not None and "Er" in C:    # radiation energy fluxes (FLD), interior indices
                    ei = ci - ng if isinstance(ci, int) else slice(ci.start - ng, ci.stop - ng)
                    ej = cj - ng if isinstance(cj, int) else slice(cj.start - ng, cj.stop - ng)
                    dEr[ei, ej] += sign * (agg(Fr["Er"][m]) - C["Er"][m]) / h
            # EMF correction on the boundary corners (fine corners every r-th, time-weighted)
            for m, (ksl, lsl) in enumerate(((i0, slice(j0, j1 + 1)), (i1, slice(j0, j1 + 1)),
                                            (slice(i0, i1 + 1), j0), (slice(i0, i1 + 1), j1))):
                dEc[ksl, lsl] = xp.where(self._emf_free[k][m], Fr["E"][m][::r] - C["E"][m], dEc[ksl, lsl])
        dU[BX:BZ] = 0.0                              # in-plane B lives on the faces (B_z is a cell variable: refluxed)
        s.U += dU
        if dEr is not None:
            s.Er += dEr
        # faces next to the patches: dBx = -(dE(k, l+1) - dE(k, l)) / dy, dBy = (dE(k+1, l) - dE(k, l)) / dx
        s.Bf[0][ng:ng + nx + 1, ng:ng + ny] += -(dEc[:, 1:] - dEc[:, :-1]) / dy
        s.Bf[1][ng:ng + nx, ng:ng + ny + 1] += (dEc[1:, :] - dEc[:-1, :]) / dx
        self._fine_fine()
        # restriction: cells and faces under each patch become the fine averages
        for box, f in zip(self.boxes, self.patches):
            i0, i1, j0, j1 = box
            nxf, nyf = f.cfg.nx, f.cfg.ny
            Uf = f.U[:, ng:ng + nxf, ng:ng + nyf]
            Uc = Uf.reshape(NVAR, i1 - i0, r, j1 - j0, r).mean(axis=(2, 4))
            s.U[:, ng + i0:ng + i1, ng + j0:ng + j1] = Uc
            fx = f.Bf[0][ng:ng + nxf + 1:r, ng:ng + nyf]
            s.Bf[0][ng + i0:ng + i1 + 1, ng + j0:ng + j1] = fx.reshape(i1 - i0 + 1, j1 - j0, r).mean(axis=2)
            fy = f.Bf[1][ng:ng + nxf, ng:ng + nyf + 1:r]
            s.Bf[1][ng + i0:ng + i1, ng + j0:ng + j1 + 1] = fy.reshape(i1 - i0, r, j1 - j0 + 1).mean(axis=1)
            if s.strength:
                Sf = f.S[:, ng:ng + nxf, ng:ng + nyf]
                s.S[:, ng + i0:ng + i1, ng + j0:ng + j1] = Sf.reshape(Sf.shape[0], i1 - i0, r, j1 - j0, r).mean(axis=(2, 4))
            if s.Er is not None:
                s.Er[i0:i1, j0:j1] = f.Er.reshape(i1 - i0, r, j1 - j0, r).mean(axis=(1, 3))
        # cell B from the faces; the total energy keeps the conserved (averaged) value
        ct.cells_from_faces(s.U, s.Bf, (slice(ng, ng + nx), slice(ng, ng + ny)))
        s.apply_bc(s.U)
        s._ghost_faces(s.Bf, s.U)

    def _fine_fine(self):
        """patches that touch: the lower-index one owns the shared faces and corners. The other one's cells next
        to the shared faces are refluxed with the owner's fluxes (and both tracer terms), and its faces get the
        owner's corner EMFs, so the two agree on the shared edge and both stay conservative and divergence-free.
        Patches touching at a single corner share only that corner's EMF. A corner on the boundary of several
        patches (a T-junction, or four patches meeting) takes the EMF of the lowest-index one, once: corrected
        twice, a patch's faces there would no longer match its neighbours' and the coarse divergence would grow."""
        r, ng = self.p.ratio, self.coarse.ng
        bx = self.boxes
        dEs, owned = {}, {}
        for a in range(len(bx)):
            for b in range(a + 1, len(bx)):
                A, B = bx[a], bx[b]
                fA, fB = self.patches[a], self.patches[b]
                RA, RB = fA._rec[0], fB._rec[0]
                # (owner side, other side, other's cell index along the normal, sign, h, axis of the edge)
                for sa, sb, ax in ((1, 0, 0), (0, 1, 0), (3, 2, 1), (2, 3, 1)):
                    line_a = (A[1] if sa == 1 else A[0]) if ax == 0 else (A[3] if sa == 3 else A[2])
                    line_b = (B[0] if sb == 0 else B[1]) if ax == 0 else (B[2] if sb == 2 else B[3])
                    if line_a != line_b:
                        continue
                    lo_a, lo_b = (A[2], B[2]) if ax == 0 else (A[0], B[0])
                    o0 = max(A[2], B[2]) if ax == 0 else max(A[0], B[0])
                    o1 = min(A[3], B[3]) if ax == 0 else min(A[1], B[1])
                    if o0 > o1:
                        continue
                    fa0, fb0, nf = r * (o0 - lo_a), r * (o0 - lo_b), r * (o1 - o0)
                    nxB, nyB = fB.cfg.nx, fB.cfg.ny
                    if nf > 0:                       # shared faces: reflux B's adjacent cells
                        dF = RA["F"][sa][:, fa0:fa0 + nf] - RB["F"][sb][:, fb0:fb0 + nf]
                        du = RA["u"][sa][fa0:fa0 + nf] - RB["u"][sb][fb0:fb0 + nf]
                        dua = RA["ua"][sa][fa0:fa0 + nf] - RB["ua"][sb][fb0:fb0 + nf]
                        dF[RHOC] = 0.0
                        dF[BX:BZ] = 0.0
                        h = fB.cfg.dx if ax == 0 else fB.cfg.dy
                        sign = +1 if sb in (0, 2) else -1        # B's low side: the face is its cells' low face
                        if ax == 0:
                            ci = ng if sb == 0 else ng + nxB - 1
                            cj = slice(ng + fb0, ng + fb0 + nf)
                        else:
                            ci = slice(ng + fb0, ng + fb0 + nf)
                            cj = ng if sb == 2 else ng + nyB - 1
                        Ue = fB.U[:, ci, cj]
                        a_c = Ue[RHOC] if fB._vf else Ue[RHOC] / xp.maximum(Ue[RHO], 1e-300)
                        fB.U[:, ci, cj] += sign * dF / h
                        fB.U[RHOC, ci, cj] += sign * (dua - a_c * du) / h
                        if "Fe" in RA and "Fe" in RB:
                            g_ = fB.cfg.gamma
                            fB.U[RHOSE, ci, cj] += sign * (g_ - 1) * (RA["Fe"][sa][fa0:fa0 + nf] - RB["Fe"][sb][fb0:fb0 + nf]) / h \
                                * xp.maximum(fB.U[RHO, ci, cj], 1e-300)**(1 - g_)
                        if "Er" in RA and "Er" in RB:
                            ei = ci - ng if isinstance(ci, int) else slice(ci.start - ng, ci.stop - ng)
                            ej = cj - ng if isinstance(cj, int) else slice(cj.start - ng, cj.stop - ng)
                            fB.Er[ei, ej] += sign * (RA["Er"][sa][fa0:fa0 + nf] - RB["Er"][sb][fb0:fb0 + nf]) / h
                    # shared corners: B's faces get the owner's EMFs (corners not already owned by a lower patch;
                    # the ownership is host bookkeeping, fixed by the boxes, so this also records into a graph)
                    if b not in dEs:
                        dEs[b], owned[b] = xp.zeros((nxB + 1, nyB + 1)), np.zeros((nxB + 1, nyB + 1), bool)
                    line = (0 if sb == 0 else nxB) if ax == 0 else (0 if sb == 2 else nyB)
                    idx = (line, slice(fb0, fb0 + nf + 1)) if ax == 0 else (slice(fb0, fb0 + nf + 1), line)
                    free = ~owned[b][idx]
                    k = 0
                    while k <= nf:                   # contiguous runs of free corners
                        if not free[k]:
                            k += 1
                            continue
                        k1 = k
                        while k1 <= nf and free[k1]:
                            k1 += 1
                        run = slice(fb0 + k, fb0 + k1)
                        d = RA["E"][sa][fa0 + k:fa0 + k1] - RB["E"][sb][fb0 + k:fb0 + k1]
                        if ax == 0:
                            dEs[b][line, run] = d
                        else:
                            dEs[b][run, line] = d
                        k = k1
                    owned[b][idx] = True
        for b, dE in dEs.items():
            fB = self.patches[b]
            nxB, nyB = fB.cfg.nx, fB.cfg.ny
            fB.Bf[0][ng:ng + nxB + 1, ng:ng + nyB] += -(dE[:, 1:] - dE[:, :-1]) / fB.cfg.dy
            fB.Bf[1][ng:ng + nxB, ng:ng + nyB + 1] += (dE[1:, :] - dE[:-1, :]) / fB.cfg.dx
            ct.cells_from_faces(fB.U, fB.Bf, (slice(ng, ng + nxB), slice(ng, ng + nyB)))

    # -- diagnostics
    def totals(self):
        """mass, momentum, total energy (gas + radiation) on the (synchronised) coarse level, and the boundary
        energy inflow"""
        s, ng = self.coarse, self.coarse.ng
        I = (slice(ng, -ng), slice(ng, -ng))
        dV = s.cfg.dx * s.cfg.dy
        out = {k: float(xp.sum(s.U[(v,) + I])) * dV for k, v in (("mass", RHO), ("mx", MX), ("my", MY), ("E", EN))}
        out["E_rad"] = s.radiation_energy()
        out["E"] += out["E_rad"]
        out["boundary"] = s._cumulative_boundary_energy + s.fld_inflow
        out["radiated"] = float(s._acc_rad) + sum(float(f._acc_rad) for f in self.patches) + self._rad_retired
        return out

    def div_b(self):
        return max([self.coarse.div_b_max()] + [f.div_b_max() for f in self.patches])

    def fine_fraction(self):
        """fine cells / cells of a uniform grid at the fine resolution"""
        r = self.p.ratio
        return sum(f.cfg.nx * f.cfg.ny for f in self.patches) / (r * r * self.cfg.nx * self.cfg.ny)

    def composite(self, var=iRHO):
        """primitive variable (an index, or a function of the conserved array) on a uniform grid at the fine
        resolution: patches where they exist, the coarse level repeated elsewhere (host array)"""
        s, r, ng = self.coarse, self.p.ratio, self.coarse.ng
        q = (lambda lev: var(lev.U)) if callable(var) else (lambda lev: cons_to_prim(lev.U, lev.eos)[var])
        Wc = to_numpy(q(s)[ng:-ng, ng:-ng])
        out = np.repeat(np.repeat(Wc, r, axis=0), r, axis=1)
        for (i0, i1, j0, j1), f in zip(self.boxes, self.patches):
            out[r * i0:r * i1, r * j0:r * j1] = to_numpy(q(f)[ng:-ng, ng:-ng])
        return out
