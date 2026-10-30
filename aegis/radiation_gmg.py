# radiation_gmg.py - geometric multigrid preconditioner for the implicit flux-limited-diffusion solves.
#
# The backward-Euler FLD systems (fld.py grey, radiation_multigroup.py per group) have the form
#
#   A x = D x - div(K grad x) - c u (w . x)
#
# on the interior cells: D the volume-weighted diagonal (V (1 + dt kappa)), K the face conductances dt D A / h of
# each group (end faces of a non-periodic axis act on a zero boundary value, the homogeneous operator; zero-flux
# faces have K = 0), and, for multigroup only, a rank-one coupling between the groups of each cell (the gas
# temperature eliminated; c u = cpl u_g, w = w_h). Stiff diffusion (D dt / dx^2 >> 1) makes Jacobi-type
# preconditioners lose a constant factor per grid doubling; a multigrid V-cycle removes the smooth error on coarse
# grids and keeps the iteration count roughly independent of the grid.
#
# Cell-centred multigrid with 2:1 coarsening along every axis that is even and at least 4 cells long:
#   - coarse operator by rediscretisation: D and the coupling vectors c u summed over the children, w averaged,
#     face conductances summed over the children faces across the face and halved along a coarsened axis (the
#     face area adds up, the spacing doubles);
#   - restriction: sum over the children (the residual is volume integrated); prolongation: the coarse correction
#     copied to the children (the transpose);
#   - smoother: red-black Gauss-Seidel on the cell blocks (cells of one colour are not neighbours, so updating
#     them together is exact Gauss-Seidel); the block of a cell (its groups) is inverted exactly by
#     Sherman-Morrison, as in radiation_multigroup's block preconditioner;
#   - pre-smoothing red then black, post-smoothing black then red, so the V-cycle is a symmetric operator and
#     is a valid preconditioner for conjugate gradients (grey) as well as BiCGStab (multigroup).
# The coarsest grid gets `coarse_sweeps` smoothing sweeps. Only the Krylov convergence changes: both solvers write
# their conservative face-flux update from the solution, so energy conservation is unaffected.
from .physics import xp


def _sl(nd, ax, s):
    idx = [slice(None)] * (nd + 1)                   # leading group axis
    idx[ax + 1] = s
    return tuple(idx)


class Level:
    def __init__(self, D, K, per, cu=None, w=None):
        self.D, self.K, self.per = D, K, tuple(per)  # D: (G, *shape); K[ax]: (G, ..., n_ax + 1, ...)
        self.cu, self.w = cu, w                      # (G, *shape) or None
        self.G, self.shape = D.shape[0], D.shape[1:]
        self.nd = len(self.shape)
        kd = 0.0
        for ax, k in enumerate(K):
            n = self.shape[ax]
            kd = kd + k[_sl(self.nd, ax, slice(0, n))] + k[_sl(self.nd, ax, slice(1, n + 1))]
        self.diag = D + kd
        if cu is not None:
            self.den = 1.0 - xp.sum(w * cu / self.diag, axis=0)
        par = 0
        for ax, n in enumerate(self.shape):
            par = par + xp.arange(n).reshape((-1,) + (1,) * (self.nd - 1 - ax))
        self.red = (par % 2 == 0)[None]
        self.coarse = None

    def apply(self, x):
        out = self.D * x
        if self.cu is not None:
            out = out - self.cu * xp.sum(self.w * x, axis=0)[None]
        nd = self.nd
        for ax, k in enumerate(self.K):
            n = self.shape[ax]
            if self.per[ax]:
                lo = xp.concatenate([x[_sl(nd, ax, slice(n - 1, n))], x], axis=ax + 1)
                hi = xp.concatenate([x, x[_sl(nd, ax, slice(0, 1))]], axis=ax + 1)
            else:
                z = xp.zeros_like(x[_sl(nd, ax, slice(0, 1))])
                lo = xp.concatenate([z, x], axis=ax + 1)
                hi = xp.concatenate([x, z], axis=ax + 1)
            g = k * (lo - hi)
            out = out - (g[_sl(nd, ax, slice(0, n))] - g[_sl(nd, ax, slice(1, n + 1))])
        return out

    def block_inv(self, r):
        Di = r / self.diag
        if self.cu is None:
            return Di
        u = self.cu / self.diag
        return Di + u * (xp.sum(self.w * Di, axis=0) / self.den)[None]

    def smooth(self, x, b, order):
        for colour in order:
            m = self.red if colour == 0 else ~self.red
            x = x + xp.where(m, self.block_inv(b - self.apply(x)), 0.0)
        return x

    def build_coarse(self):
        f = [2 if (n % 2 == 0 and n >= 4) else 1 for n in self.shape]
        if all(v == 1 for v in f):
            return None
        G, nd = self.G, self.nd

        def csum(a):                                 # sum over the children of each coarse cell
            shp = [G]
            for n, v in zip(self.shape, f):
                shp += [n // v, v]
            return a.reshape(shp).sum(axis=tuple(2 + 2 * i for i in range(nd)))
        Kc = []
        for ax, k in enumerate(self.K):
            if f[ax] == 2:                           # every other face along ax: faces 0, 2, ..., n
                k = k[_sl(nd, ax, slice(0, None, 2))]
            shp = [G]
            for d, (n, v) in enumerate(zip(self.shape, f)):
                if d == ax:
                    shp += [k.shape[d + 1], 1]
                else:
                    shp += [n // v, v]
            kc = k.reshape(shp).sum(axis=tuple(2 + 2 * i for i in range(nd)))
            Kc.append(kc * (0.5 if f[ax] == 2 else 1.0))
        nchild = 1
        for v in f:
            nchild *= v
        cu = csum(self.cu) if self.cu is not None else None
        w = csum(self.w) / nchild if self.w is not None else None
        self.factors = f
        self.coarse = Level(csum(self.D), Kc, self.per, cu, w)
        return self.coarse

    def restrict(self, r):
        G, nd, f = self.G, self.nd, self.factors
        shp = [G]
        for n, v in zip(self.shape, f):
            shp += [n // v, v]
        return r.reshape(shp).sum(axis=tuple(2 + 2 * i for i in range(nd)))

    def prolong(self, e):
        for ax, v in enumerate(self.factors):
            if v == 2:
                e = xp.repeat(e, 2, axis=ax + 1)
        return e


class Multigrid:
    """V-cycle preconditioner: M(r) approximates A^{-1} r. Arrays carry a leading group axis (G = 1 for grey)."""

    def __init__(self, D, K, per, cu=None, w=None, nu=2, coarse_sweeps=4, max_levels=12):
        self.levels = [Level(D, K, per, cu, w)]
        while len(self.levels) < max_levels:
            c = self.levels[-1].build_coarse()
            if c is None:
                break
            self.levels.append(c)
        self.nu, self.coarse_sweeps = nu, coarse_sweeps

    def _vcycle(self, k, b):
        L = self.levels[k]
        x = xp.zeros_like(b)
        if k == len(self.levels) - 1:
            for _ in range(self.coarse_sweeps):
                x = L.smooth(x, b, (0, 1))
            for _ in range(self.coarse_sweeps):
                x = L.smooth(x, b, (1, 0))
            return x
        for _ in range(self.nu):
            x = L.smooth(x, b, (0, 1))
        e = self._vcycle(k + 1, L.restrict(b - L.apply(x)))
        x = x + L.prolong(e)
        for _ in range(self.nu):
            x = L.smooth(x, b, (1, 0))
        return x

    def __call__(self, r):
        return self._vcycle(0, r)
