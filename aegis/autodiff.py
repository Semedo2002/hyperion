# autodiff.py - reverse-mode differentiation of the 2D ideal-MHD solver (adjoint gradients, inverse design).
#
# Two parts:
#   1. A small tape for arrays (Tensor): every operation on a Tensor records its vector-Jacobian product, and
#      backward() runs them in reverse. It works on NumPy or CuPy arrays (the backend of the leaves).
#   2. DiffMHD, a twin of MHDSolver's ideal-MHD step written with those operations: MUSCL (van Leer), the HLLD
#      flux, constrained transport with the Gardiner-Stone corner EMF, SSP-RK3, the same boundary conditions,
#      Cartesian or axisymmetric (r, z). The arithmetic follows hlld_kernel.cu / fused.py line by line, so a
#      forward run reproduces MHDSolver to round-off (checked by --test autodiff), and its gradients are the
#      exact derivatives of the discrete scheme (checked against finite differences).
#
# Gradients come from a discrete adjoint with checkpointing: the forward pass stores the state after every step
# (no tape); the backward pass re-runs one step at a time with the tape on, from the last step to the first,
# carrying the adjoint state lambda_n = dJ/dU_n back through each step. Memory is one state per step plus one
# step's tape, and one backward pass gives dJ/dp for every parameter p at about 3x the cost of the forward run.
#
# What is differentiated: the initial state (any function of the parameters built from Tensor operations),
# an optional boundary drive (B_phi in the outer radial ghost cells, a function of time and the parameters),
# a per-step objective (summed with weight dt) and a final objective. Time steps are fixed by the forward run's
# CFL condition and held constant (the gradient does not include d(dt)/dp, the usual choice for adjoints of
# explicit schemes; finite differences with the same dt sequence agree with it). Floors, limiters and the
# HLLD wave-branch choices are piecewise smooth: the derivative is that of the branch taken.
#
# Scope: ideal gas, ideal MHD with div_b='ct', no extra physics (resistivity, conduction, radiation, EOS
# tables, strength ...). DiffMHD refuses configurations outside it.
import numpy as np

from .config import (NVAR, FLOOR_RHO, FLOOR_PR, RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI, RHOC, RHOSE)

# ============================================================
# tape
# ============================================================
_REC = [False]
_TAPE = []


def _xp(*a):
    try:
        import cupy
        return cupy.get_array_module(*a)
    except ImportError:
        return np


class Tensor:
    """an array with a gradient slot; operations between Tensors (and with plain numbers / arrays) record
    their backward rules while the tape is on"""
    __slots__ = ("v", "parents", "g", "rg")
    __array_priority__ = 1000

    def __init__(self, v, requires_grad=True):
        self.v, self.parents, self.g, self.rg = v, (), None, requires_grad

    shape = property(lambda s: s.v.shape)
    ndim = property(lambda s: s.v.ndim)

    def __add__(s, o): return add(s, o)
    __radd__ = __add__
    def __sub__(s, o): return sub(s, o)
    def __rsub__(s, o): return sub(o, s)
    def __mul__(s, o): return mul(s, o)
    __rmul__ = __mul__
    def __truediv__(s, o): return div(s, o)
    def __rtruediv__(s, o): return div(o, s)
    def __neg__(s): return mul(s, -1.0)
    def __pow__(s, k): return power(s, k)
    def __getitem__(s, idx): return getitem(s, idx)
    def __abs__(s): return absolute(s)
    # comparisons give plain boolean arrays (no gradient)
    def __gt__(s, o): return s.v > val(o)
    def __ge__(s, o): return s.v >= val(o)
    def __lt__(s, o): return s.v < val(o)
    def __le__(s, o): return s.v <= val(o)

    def sum(s, axis=None): return tsum(s, axis)


def val(x):
    return x.v if isinstance(x, Tensor) else x


def _node(v, parents):
    """result of an operation; parents: (input, vjp) pairs, kept only for inputs that need a gradient"""
    t = Tensor.__new__(Tensor)
    t.v, t.g = v, None
    if _REC[0]:
        ps = tuple(p for p in parents if isinstance(p[0], Tensor) and p[0].rg)
        if ps:
            t.parents, t.rg = ps, True
            _TAPE.append(t)
            return t
    t.parents, t.rg = (), False
    return t


def _unbroadcast(g, shape):
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, n in enumerate(shape):
        if n == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def add(a, b):
    return _node(val(a) + val(b), ((a, lambda g: g), (b, lambda g: g)))


def sub(a, b):
    return _node(val(a) - val(b), ((a, lambda g: g), (b, lambda g: -g)))


def mul(a, b):
    av, bv = val(a), val(b)
    return _node(av * bv, ((a, lambda g: g * bv), (b, lambda g: g * av)))


def div(a, b):
    av, bv = val(a), val(b)
    v = av / bv
    return _node(v, ((a, lambda g: g / bv), (b, lambda g: -g * v / bv)))


def power(a, k):
    av = val(a)
    return _node(av ** k, ((a, lambda g: g * k * av ** (k - 1)),))


def sqrt(a):
    av = val(a)
    xp = _xp(av)
    v = xp.sqrt(av)
    # zero derivative at 0 (the kernel only takes sqrt of quantities floored at 0 there)
    return _node(v, ((a, lambda g: xp.where(v > 0, 0.5 * g / xp.where(v > 0, v, 1.0), 0.0)),))


def absolute(a):
    av = val(a)
    xp = _xp(av)
    return _node(xp.abs(av), ((a, lambda g: g * xp.sign(av)),))


def maximum(a, b):
    av, bv = val(a), val(b)
    xp = _xp(av, bv)
    m = av >= bv                                     # ties go to the first argument (fmax(x, floor))
    return _node(xp.maximum(av, bv), ((a, lambda g: xp.where(m, g, 0.0)), (b, lambda g: xp.where(m, 0.0, g))))


def minimum(a, b):
    av, bv = val(a), val(b)
    xp = _xp(av, bv)
    m = av <= bv
    return _node(xp.minimum(av, bv), ((a, lambda g: xp.where(m, g, 0.0)), (b, lambda g: xp.where(m, 0.0, g))))


def where(c, a, b):
    """c: a plain boolean array (no gradient flows through the choice)"""
    xp = _xp(c)
    return _node(xp.where(c, val(a), val(b)), ((a, lambda g: xp.where(c, g, 0.0)), (b, lambda g: xp.where(c, 0.0, g))))


def _unary(f, df):
    def op(a):
        av = val(a)
        v = f(_xp(av), av)
        return _node(v, ((a, lambda g: g * df(_xp(av), av, v)),))
    return op


exp = _unary(lambda xp, a: xp.exp(a), lambda xp, a, v: v)
log = _unary(lambda xp, a: xp.log(a), lambda xp, a, v: 1.0 / a)
tanh = _unary(lambda xp, a: xp.tanh(a), lambda xp, a, v: 1.0 - v * v)
cos = _unary(lambda xp, a: xp.cos(a), lambda xp, a, v: -xp.sin(a))
sin = _unary(lambda xp, a: xp.sin(a), lambda xp, a, v: xp.cos(a))


def getitem(a, idx):
    av = val(a)
    xp = _xp(av)

    def back(g):
        z = xp.zeros_like(av)
        z[idx] = g
        return z
    return _node(av[idx], ((a, back),))


def assign(a, idx, b):
    """a copy of a with a[idx] = b (basic indexing)"""
    av, bv = val(a), val(b)
    xp = _xp(av)
    v = av.copy()
    v[idx] = bv
    shape_b = np.shape(bv)

    def back_a(g):
        g = g.copy()
        g[idx] = 0.0
        return g
    return _node(v, ((a, back_a), (b, lambda g: _unbroadcast(g[idx], shape_b))))


def tsum(a, axis=None):
    av = val(a)
    xp = _xp(av)
    v = av.sum(axis=axis)

    def back(g):
        if axis is not None:
            g = xp.expand_dims(g, axis)
        return xp.broadcast_to(g, av.shape)
    return _node(v, ((a, back),))


def einsum(spec, a, b):
    """two-operand einsum; every index of an operand must appear in the other operand or the output"""
    av, bv = val(a), val(b)
    xp = _xp(av, bv)
    ins, out = spec.split("->")
    sa, sb = ins.split(",")
    return _node(xp.einsum(spec, av, bv), ((a, lambda g: xp.einsum(f"{out},{sb}->{sa}", g, bv)),
                                          (b, lambda g: xp.einsum(f"{out},{sa}->{sb}", g, av))))


def gelu(a):
    """GELU, tanh form"""
    av = val(a)
    xp = _xp(av)
    c = 0.7978845608028654
    u = c * (av + 0.044715 * av**3)
    t = xp.tanh(u)
    v = 0.5 * av * (1.0 + t)
    return _node(v, ((a, lambda g: g * (0.5 * (1.0 + t) + 0.5 * av * (1.0 - t * t) * c * (1.0 + 3 * 0.044715 * av**2))),))


def reshape(a, shape):
    av = val(a)
    return _node(av.reshape(shape), ((a, lambda g: g.reshape(av.shape)),))


def stack(ts, axis=0):
    vs = [val(t) for t in ts]
    xp = _xp(vs[0])
    return _node(xp.stack(vs, axis=axis), tuple((t, (lambda k: lambda g: xp.take(g, k, axis=axis))(k))
                                               for k, t in enumerate(ts)))


def backward(outputs, seeds=None):
    """accumulate gradients into the leaves; outputs: Tensors (a scalar objective, seed 1) or with seeds"""
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
        seeds = None if seeds is None else [seeds]
    for k, t in enumerate(outputs):
        s = (1.0 if t.v.ndim == 0 else None) if seeds is None else seeds[k]
        if s is None or not t.rg:
            continue
        xp = _xp(t.v)
        s = xp.asarray(s, dtype=t.v.dtype) if not hasattr(s, "shape") else s
        t.g = s if t.g is None else t.g + s
    for node in reversed(_TAPE):
        g = node.g
        if g is None:
            continue
        for p, fn in node.parents:
            gp = _unbroadcast(fn(g), p.v.shape)
            p.g = gp if p.g is None else p.g + gp
        node.g = None
    _TAPE.clear()


class record:
    """with record(): ... -- operations inside are taped"""
    def __enter__(self):
        _TAPE.clear()
        _REC[0] = True
        return self

    def __exit__(self, *exc):
        _REC[0] = False
        if exc[0] is not None:
            _TAPE.clear()


def leaf(v):
    return Tensor(v, requires_grad=True)


def const(v):
    return Tensor(v, requires_grad=False)


# ============================================================
# differentiable ideal-MHD step
# ============================================================
_SUPPORTED_OFF = ("rad_coeff", "cond_kappa0", "cond_kappa_i0", "visc_coeff", "nernst_coeff", "hall_di", "gyro_di")


class DiffMHD:
    """differentiable twin of MHDSolver for ideal MHD with constrained transport.

    Built from an MHDSolver (its config, grid and boundary conditions). A state is a list of NVAR cell
    arrays (ghost cells included, as MHDSolver.U) followed by the two face arrays (Bf[0], Bf[1]), each a
    Tensor. drive(t, params) -> B_phi at the outer radial boundary (r = x_max) is optional (cylindrical
    only): the outer ghost cells then carry B_phi = B_d r_max / r with the pressure extrapolated, a magnetic
    piston for a current I = 2 pi r_max B_d / mu0 flowing on the outer boundary."""

    def __init__(self, solver, drive=None, xp=None):
        cfg = solver.cfg
        bad = [k for k in _SUPPORTED_OFF if getattr(cfg, k, 0)]
        if not solver.ct or not isinstance(solver.eos, (int, float)) or cfg.is_resistive() or cfg.fld \
                or cfg.two_temp or solver.strength or bad:
            raise ValueError("DiffMHD covers ideal-gas ideal MHD with div_b='ct' only (found: "
                             + ", ".join(bad or ["resistivity / EOS / FLD / two-temperature / strength / GLM"]) + ")")
        if drive is not None and not solver.cyl:
            raise ValueError("a boundary drive needs cylindrical geometry")
        self.s, self.cfg, self.drive = solver, cfg, drive
        self.xp = xp or _xp(solver.U)
        x = self.xp
        self.g = float(cfg.gamma)
        self.ng, self.nx, self.ny = solver.ng, cfg.nx, cfg.ny
        self.Nx, self.Ny = solver.nx_tot, solver.ny_tot
        self.dx, self.dy = cfg.dx, cfg.dy
        self.cyl = solver.cyl
        self.per = (cfg.get_bc_x() == "periodic", cfg.bc_y_type == "periodic")
        if cfg.get_bc_x() not in ("periodic", "extrapolation"):
            raise ValueError("DiffMHD supports periodic or extrapolation x boundaries")
        asx = lambda a: x.asarray(a.get() if hasattr(a, "get") else a)
        self.rc = asx(solver.rc)
        self.rf = asx(solver.rf)
        self.dV = asx(solver.dV) if not np.isscalar(solver.dV) else solver.dV

    # ---------------- state helpers ----------------
    def state_from_solver(self, requires_grad=False):
        """the solver's current U and face fields as a state (faces built from U first if needed)"""
        s, x = self.s, self.xp
        s._ct_sync()
        h = lambda a: x.asarray(a.get() if hasattr(a, "get") else a)
        mk = leaf if requires_grad else const
        return [mk(h(s.U[k]).copy()) for k in range(NVAR)] + [mk(h(f).copy()) for f in s.Bf]

    def raw(self, state):
        return [val(a) for a in state]

    def to_solver(self, state):
        """write a state back into the solver (U and faces)"""
        s = self.s
        X = _xp(s.U)
        for k in range(NVAR):
            s.U[k] = X.asarray(val(state[k]))
        for f, a in zip(s.Bf, state[NVAR:]):
            f[...] = X.asarray(val(a))

    def compute_dt(self, state):
        """CFL step of MHDSolver.compute_dt (ideal gas), from the raw state"""
        x, g, cfg = self.xp, self.g, self.cfg
        U = self.raw(state)
        rho = x.maximum(U[RHO], FLOOR_RHO)
        vx, vy, vz = U[MX] / rho, U[MY] / rho, U[MZ] / rho
        B2 = U[BX]**2 + U[BY]**2 + U[BZ]**2
        p = x.maximum((g - 1) * (U[EN] - 0.5 * rho * (vx**2 + vy**2 + vz**2) - 0.5 * B2), FLOOR_PR)
        cf = x.sqrt(g * p / rho + B2 / rho)
        sx, sy = x.abs(vx) + cf, x.abs(vy) + cf
        ch = max(float(x.max(x.maximum(sx, sy))), 1e-12)
        inv = max(float(x.max(x.maximum(sx / self.dx, sy / self.dy))), ch / min(self.dx, self.dy))
        return cfg.cfl / inv if inv > 1e-14 else cfg.cfl * min(self.dx, self.dy)

    # ---------------- boundary conditions ----------------
    def _bc(self, U, t, params):
        """MHDSolver.apply_bc on a list of cell Tensors (functional)"""
        ng, nx, ny = self.ng, self.nx, self.ny
        out = []
        for k, a in enumerate(U):
            if self.per[0]:
                a = assign(a, (slice(0, ng), slice(None)), a[-2 * ng:-ng, :])
                a = assign(a, (slice(-ng, None), slice(None)), a[ng:2 * ng, :])
            else:
                for i in range(ng):
                    a = assign(a, (i, slice(None)), a[ng, :])
                    a = assign(a, (-(i + 1), slice(None)), a[-(ng + 1), :])
                if self.cyl:                         # axis: mirror, odd in v_r, v_phi, B_r, B_phi
                    for q in range(ng):
                        src = a[ng + q, :]
                        a = assign(a, (ng - 1 - q, slice(None)), -src if k in (MX, MZ, BX, BZ) else src)
            out.append(a)
        if self.drive is not None:                   # magnetic piston on the outer radial boundary
            Bd = self.drive(t, params)
            rmax = self.cfg.x_max
            for i in range(ng):
                row = self.Nx - ng + i
                bnew = Bd * (rmax / float(self.rc[row]))
                bold = out[BZ][row, :]
                out[EN] = assign(out[EN], (row, slice(None)), out[EN][row, :] - 0.5 * bold * bold + 0.5 * bnew * bnew)
                out[BZ] = assign(out[BZ], (row, slice(None)), bnew + 0.0 * bold)
        res = []
        for a in out:
            if self.per[1]:
                a = assign(a, (slice(None), slice(0, ng)), a[:, -2 * ng:-ng])
                a = assign(a, (slice(None), slice(-ng, None)), a[:, ng:2 * ng])
            else:
                for j in range(ng):
                    a = assign(a, (slice(None), j), a[:, ng])
                    a = assign(a, (slice(None), -(j + 1)), a[:, -(ng + 1)])
            res.append(a)
        return res

    def _faces_from_cells(self, c, axis):
        """ct.faces_from_cells for one component"""
        n = c.shape[axis]
        if axis == 0:
            mid = 0.5 * (c[0:n - 1, :] + c[1:n, :])
            f = Tensor(self.xp.empty((n + 1, c.shape[1])), False)
            f = assign(f, (slice(1, n), slice(None)), mid)
            f = assign(f, (0, slice(None)), c[0, :])
            return assign(f, (n, slice(None)), c[n - 1, :])
        mid = 0.5 * (c[:, 0:n - 1] + c[:, 1:n])
        f = Tensor(self.xp.empty((c.shape[0], n + 1)), False)
        f = assign(f, (slice(None), slice(1, n)), mid)
        f = assign(f, (slice(None), 0), c[:, 0])
        return assign(f, (slice(None), n), c[:, n - 1])

    def _ghost_faces(self, Bf, U):
        """ct.ghost_faces (+ the axis faces in r-z): every face outside the interior from the ghost cells"""
        ng, n = self.ng, (self.nx, self.ny)
        out = []
        for a in range(2):
            f = Bf[a]
            avg = self._faces_from_cells(U[BX + a], a)
            inner = [slice(ng, ng + m) for m in n]
            inner[a] = slice(ng, ng + n[a] + 1)
            inner = tuple(inner)
            f = assign(avg, inner, f[inner])
            for t in range(2):
                if not self.per[t]:
                    continue
                N = f.shape[t]
                lo, hi = [slice(None)] * 2, [slice(None)] * 2
                if t == a:
                    lo[t], hi[t] = slice(0, ng), slice(n[t], n[t] + ng)
                    f = assign(f, tuple(lo), f[tuple(hi)])
                    lo[t], hi[t] = slice(ng + n[t] + 1, N), slice(ng + 1, N - n[t])
                    f = assign(f, tuple(lo), f[tuple(hi)])
                else:
                    lo[t], hi[t] = slice(0, ng), slice(n[t], n[t] + ng)
                    f = assign(f, tuple(lo), f[tuple(hi)])
                    lo[t], hi[t] = slice(ng + n[t], N), slice(ng, N - n[t])
                    f = assign(f, tuple(lo), f[tuple(hi)])
            out.append(f)
        if self.cyl:                                 # behind the axis: B_r odd, B_z even
            f0, f1 = out
            f0 = assign(f0, (ng, slice(None)), 0.0 * f0[ng, :])
            for k in range(1, ng + 1):
                f0 = assign(f0, (ng - k, slice(None)), -f0[ng + k, :])
            for k in range(ng):
                f1 = assign(f1, (ng - 1 - k, slice(None)), f1[ng + k, :])
            out = [f0, f1]
        return out

    # ---------------- fluxes ----------------
    def _prim(self, U):
        g = self.g
        rho = maximum(U[RHO], FLOOR_RHO)
        ri = 1.0 / rho
        vx, vy, vz = U[MX] * ri, U[MY] * ri, U[MZ] * ri
        bx, by, bz = U[BX], U[BY], U[BZ]
        ke = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
        me = 0.5 * (bx * bx + by * by + bz * bz)
        p = maximum((g - 1.0) * (U[EN] - ke - me), FLOOR_PR)
        return [rho, vx, vy, vz, bx, by, bz, p, U[PSI], U[RHOC] * ri, U[RHOSE] * ri]

    @staticmethod
    def _muscl(W, axis):
        """fused.muscl: van Leer, face k between cells k+1 and k+2, floors on rho and p"""
        WL, WR = [], []
        for v, w in enumerate(W):
            n = w.shape[axis]
            sl = lambda a, b: (slice(a, b), slice(None)) if axis == 0 else (slice(None), slice(a, b))
            w0, w1, w2, w3 = w[sl(0, n - 3)], w[sl(1, n - 2)], w[sl(2, n - 1)], w[sl(3, n)]
            dm1, dp1 = w1 - w0, w2 - w1
            ab1 = dm1 * dp1
            d1 = where(val(ab1) > 0, 2.0 * ab1 / (dm1 + dp1 + 1e-30), 0.0)
            dm2, dp2 = w2 - w1, w3 - w2
            ab2 = dm2 * dp2
            d2 = where(val(ab2) > 0, 2.0 * ab2 / (dm2 + dp2 + 1e-30), 0.0)
            l, r = w1 + 0.5 * d1, w2 - 0.5 * d2
            if v == 0:
                l, r = maximum(l, FLOOR_RHO), maximum(r, FLOOR_RHO)
            elif v == 7:
                l, r = maximum(l, FLOOR_PR), maximum(r, FLOOR_PR)
            WL.append(l)
            WR.append(r)
        return WL, WR

    def _hlld(self, WL, WR, axn):
        """hlld_kernel.cu with ch = 0 (CT), ideal gas; returns the 11 flux components in global order"""
        g = self.g
        c1, c2, c3 = (1, 2, 3) if axn == 0 else (2, 1, 3)
        T, F = 1e-14, FLOOR_RHO

        def side(Wf):
            rho = maximum(Wf[0], F)
            return (rho, Wf[c1], Wf[c2], Wf[c3], Wf[c1 + 3], Wf[c2 + 3], Wf[c3 + 3], maximum(Wf[7], FLOOR_PR),
                    Wf[8], Wf[9], Wf[10])
        rhoL, vxL, vyL, vzL, BxL, ByL, BzL, pL, psiL, CL, SEL = side(WL)
        rhoR, vxR, vyR, vzR, BxR, ByR, BzR, pR, psiR, CR, SER = side(WR)
        Bx = 0.5 * (BxL + BxR)
        Bx2 = Bx * Bx
        BmL2 = Bx2 + ByL * ByL + BzL * BzL
        BmR2 = Bx2 + ByR * ByR + BzR * BzR
        ptL, ptR = pL + 0.5 * BmL2, pR + 0.5 * BmR2
        aL2, aR2 = g * pL / rhoL, g * pR / rhoR
        eiL, eiR = pL / (g - 1.0), pR / (g - 1.0)

        def fast(a2, Bm2, rho):
            tmp = a2 + Bm2 / rho
            disc = maximum(tmp * tmp - 4.0 * a2 * Bx2 / rho, 0.0)
            return sqrt(maximum(0.5 * (tmp + sqrt(disc)), 0.0))
        cfL, cfR = fast(aL2, BmL2, rhoL), fast(aR2, BmR2, rhoR)
        SL = minimum(vxL - cfL, vxR - cfR)
        SR = maximum(vxL + cfL, vxR + cfR)
        dSL, dSR = SL - vxL, SR - vxR
        den = dSR * rhoR - dSL * rhoL
        xp = _xp(val(rhoL))
        den = where(xp.abs(val(den)) < T, xp.where(val(den) >= 0.0, T, -T), den)
        SM = (dSR * rhoR * vxR - dSL * rhoL * vxL - ptR + ptL) / den
        SM = where(val(SM) < val(SL) + T, SL + T, SM)
        SM = where(val(SM) > val(SR) - T, SR - T, SM)
        ptS = ptL + rhoL * dSL * (SM - vxL)
        dSL_SM, dSR_SM = SL - SM, SR - SM
        dSL_SM = where(abs(val(dSL_SM)) < T, T, dSL_SM)
        dSR_SM = where(abs(val(dSR_SM)) < T, T, dSR_SM)
        rhoLS = maximum(rhoL * dSL / dSL_SM, F)
        rhoRS = maximum(rhoR * dSR / dSR_SM, F)
        sqLS, sqRS = sqrt(rhoLS), sqrt(rhoRS)
        tol = 1e-8 * xp.maximum(xp.abs(val(ptS)), 1e-300)

        def star(rho, dS, dS_SM, vx, vy, vz, By, Bz):
            dn = rho * dS * dS_SM - Bx2
            ok = xp.abs(val(dn)) > tol
            dns = where(ok, dn, 1.0)
            vyS = where(ok, vy - Bx * By * (SM - vx) / dns, vy)
            vzS = where(ok, vz - Bx * Bz * (SM - vx) / dns, vz)
            ByS = where(ok, By * (rho * dS * dS - Bx2) / dns, By)
            BzS = where(ok, Bz * (rho * dS * dS - Bx2) / dns, Bz)
            return vyS, vzS, ByS, BzS
        vyLS, vzLS, ByLS, BzLS = star(rhoL, dSL, dSL_SM, vxL, vyL, vzL, ByL, BzL)
        vyRS, vzRS, ByRS, BzRS = star(rhoR, dSR, dSR_SM, vxR, vyR, vzR, ByR, BzR)
        vBL = vxL * Bx + vyL * ByL + vzL * BzL
        vBLS = SM * Bx + vyLS * ByLS + vzLS * BzLS
        EL = eiL + 0.5 * rhoL * (vxL * vxL + vyL * vyL + vzL * vzL) + 0.5 * BmL2
        ELS = ((SL - vxL) * EL - ptL * vxL + ptS * SM + Bx * (vBL - vBLS)) / dSL_SM
        vBR = vxR * Bx + vyR * ByR + vzR * BzR
        vBRS = SM * Bx + vyRS * ByRS + vzRS * BzRS
        ER = eiR + 0.5 * rhoR * (vxR * vxR + vyR * vyR + vzR * vzR) + 0.5 * BmR2
        ERS = ((SR - vxR) * ER - ptR * vxR + ptS * SM + Bx * (vBR - vBRS)) / dSR_SM
        sBx = xp.where(val(Bx) >= 0.0, 1.0, -1.0)
        SAL = SM - abs(Bx) / maximum(sqLS, T)
        SAR = SM + abs(Bx) / maximum(sqRS, T)
        SAL = where(val(SAL) < val(SL), SL, SAL)
        SAL = where(val(SAL) > val(SM), SM, SAL)
        SAR = where(val(SAR) < val(SM), SM, SAR)
        SAR = where(val(SAR) > val(SR), SR, SAR)
        dds = maximum(sqLS + sqRS, T)
        vyDS = (sqLS * vyLS + sqRS * vyRS + (ByRS - ByLS) * sBx) / dds
        vzDS = (sqLS * vzLS + sqRS * vzRS + (BzRS - BzLS) * sBx) / dds
        ByDS = (sqLS * ByRS + sqRS * ByLS + sqLS * sqRS * (vyRS - vyLS) * sBx) / dds
        BzDS = (sqLS * BzRS + sqRS * BzLS + sqLS * sqRS * (vzRS - vzLS) * sBx) / dds
        vBDS = SM * Bx + vyDS * ByDS + vzDS * BzDS
        ELDS = ELS - sqLS * (vBLS - vBDS) * sBx
        ERDS = ERS + sqRS * (vBRS - vBDS) * sBx

        def flux(rho, vx, vy, vz, By, Bz, pt, E, vB, C, SE):
            return [rho * vx, rho * vx * vx + pt - Bx2, rho * vy * vx - Bx * By, rho * vz * vx - Bx * Bz, None,
                    By * vx - Bx * vy, Bz * vx - Bx * vz, (E + pt) * vx - Bx * vB, None, rho * C * vx, rho * SE * vx]
        FL = flux(rhoL, vxL, vyL, vzL, ByL, BzL, ptL, EL, vBL, CL, SEL)
        FR = flux(rhoR, vxR, vyR, vzR, ByR, BzR, ptR, ER, vBR, CR, SER)
        UL = [rhoL, rhoL * vxL, rhoL * vyL, rhoL * vzL, None, ByL, BzL, EL, psiL, rhoL * CL, rhoL * SEL]
        UR = [rhoR, rhoR * vxR, rhoR * vyR, rhoR * vzR, None, ByR, BzR, ER, psiR, rhoR * CR, rhoR * SER]
        ULS = [rhoLS, rhoLS * SM, rhoLS * vyLS, rhoLS * vzLS, None, ByLS, BzLS, ELS, 0.0, rhoLS * CL, rhoLS * SEL]
        URS = [rhoRS, rhoRS * SM, rhoRS * vyRS, rhoRS * vzRS, None, ByRS, BzRS, ERS, 0.0, rhoRS * CR, rhoRS * SER]
        ULDS = [rhoLS, rhoLS * SM, rhoLS * vyDS, rhoLS * vzDS, None, ByDS, BzDS, ELDS, 0.0, rhoLS * CL, rhoLS * SEL]
        URDS = [rhoRS, rhoRS * SM, rhoRS * vyDS, rhoRS * vzDS, None, ByDS, BzDS, ERDS, 0.0, rhoRS * CR, rhoRS * SER]
        cL, cR, cAL, cM, cAR = val(SL) >= 0.0, val(SR) <= 0.0, val(SAL) >= 0.0, val(SM) >= 0.0, val(SAR) >= 0.0
        zero = xp.zeros_like(val(SL))
        f = [None] * NVAR
        for v in range(NVAR):
            if v == 4:                               # normal field: zero in every branch with ch = 0
                f[v] = const(zero)
                continue
            fl = FL[v] if FL[v] is not None else 0.0
            fr = FR[v] if FR[v] is not None else 0.0
            FLS = fl + SL * (ULS[v] - UL[v])
            FRS = fr + SR * (URS[v] - UR[v])
            FLDS = FLS + SAL * (ULDS[v] - ULS[v])
            FRDS = FRS + SAR * (URDS[v] - URS[v])
            star_f = where(cAL, FLS, where(cM, FLDS, where(cAR, FRDS, FRS)))
            f[v] = where(cL, fl, where(cR, fr, star_f))
            if not isinstance(f[v], Tensor):
                f[v] = const(f[v])
        gmap = {1: c1, 2: c2, 3: c3, 4: c1 + 3, 5: c2 + 3, 6: c3 + 3}
        out = [None] * NVAR
        for v in range(NVAR):
            out[gmap.get(v, v)] = f[v]
        return out

    def _emf(self, Fx, Fy, W):
        """fused.emf2d / ct.edge_emf: corner E_z (Gardiner & Stone 2005) on (nx+1, ny+1) corners"""
        nx, ny, hx, hy = self.nx, self.ny, self.dx, self.dy
        xp = self.xp
        Ea, Eb = -Fx[BY], Fy[BX]
        Ec = -(W[1] * W[5] - W[2] * W[4])
        ma, mb = val(Fx[RHO]), val(Fy[RHO])
        tol = max(1e-10 * max(float(xp.max(xp.abs(ma))), float(xp.max(xp.abs(mb)))), 1e-300)
        Ea_hi, Ea_lo = Ea[:, 2:ny + 3], Ea[:, 1:ny + 2]
        Eb_r, Eb_l = Eb[2:nx + 3, :], Eb[1:nx + 2, :]
        Ec_c, Ec_l = Ec[2:nx + 3, 2:ny + 3], Ec[1:nx + 2, 2:ny + 3]
        Ec_d, Ec_ld = Ec[2:nx + 3, 1:ny + 2], Ec[1:nx + 2, 1:ny + 2]

        def up(m, lo, hi):
            return where(m > tol, lo, where(m < -tol, hi, 0.5 * (lo + hi)))
        db_hi = up(ma[:, 2:ny + 3], 2.0 * (Ec_l - Eb_l) / hy, 2.0 * (Ec_c - Eb_r) / hy)
        db_lo = up(ma[:, 1:ny + 2], 2.0 * (Eb_l - Ec_ld) / hy, 2.0 * (Eb_r - Ec_d) / hy)
        da_r = up(mb[2:nx + 3, :], 2.0 * (Ec_d - Ea_lo) / hx, 2.0 * (Ec_c - Ea_hi) / hx)
        da_l = up(mb[1:nx + 2, :], 2.0 * (Ea_lo - Ec_ld) / hx, 2.0 * (Ea_hi - Ec_l) / hx)
        return (0.25 * (Ea_hi + Ea_lo + Eb_r + Eb_l) + 0.125 * hy * (db_lo - db_hi)
                + 0.125 * hx * (da_l - da_r))

    def rhs(self, U, Bf, t, params):
        """MHDSolver.compute_rhs (ideal MHD, CT): returns (cell rates on the interior, face rates, U with the
        boundary conditions applied, faces with ghosts)"""
        ng, nx, ny, xp = self.ng, self.nx, self.ny, self.xp
        inner = (slice(ng, ng + nx), slice(ng, ng + ny))
        U = list(U)
        for a in range(2):                           # cell B = face average on the interior
            f = Bf[a]
            n = U[0].shape[a]
            avg = 0.5 * (f[0:n, :] + f[1:n + 1, :]) if a == 0 else 0.5 * (f[:, 0:n] + f[:, 1:n + 1])
            U[BX + a] = assign(U[BX + a], inner, avg[inner])
        U = self._bc(U, t, params)
        Bf = self._ghost_faces(Bf, U)
        W = self._prim(U)
        WLx, WRx = self._muscl(W, 0)
        WLx[BX] = WRx[BX] = Bf[0][2:self.Nx - 1, :]
        Fx = self._hlld(WLx, WRx, 0)
        WLy, WRy = self._muscl(W, 1)
        WLy[BY] = WRy[BY] = Bf[1][:, 2:self.Ny - 1]
        Fy = self._hlld(WLy, WRy, 1)
        R = [None] * NVAR
        for v in range(NVAR):
            if v in (BX, BY):
                continue
            if self.cyl and v != BZ:
                rfx = self.rf[2:self.Nx - 1][:, None]
                rF = Fx[v] * rfx
                dFx = (rF[1:1 + nx, :] - rF[0:nx, :]) / self.rc[ng:ng + nx][:, None]
            else:
                dFx = Fx[v][1:1 + nx, :] - Fx[v][0:nx, :]
            dFy = Fy[v][:, 1:1 + ny] - Fy[v][:, 0:ny]
            R[v] = (0.0 - dFx[:, ng:ng + ny] / self.dx) - dFy[ng:ng + nx, :] / self.dy
        if self.cyl:
            Wi = [w[inner] for w in W]
            ri = self.rc[ng:ng + nx][:, None]
            pT = Wi[7] + 0.5 * (Wi[4] * Wi[4] + Wi[5] * Wi[5] + Wi[6] * Wi[6])
            R[MX] = R[MX] + (Wi[0] * Wi[3] * Wi[3] - Wi[6] * Wi[6] + pT) / ri
            R[MZ] = R[MZ] - (Wi[0] * Wi[1] * Wi[3] - Wi[4] * Wi[6]) / ri
        E = self._emf(Fx, Fy, W)
        if self.cyl:
            E = assign(E, (0, slice(None)), 0.0 * E[0, :])
            dB0 = -(E[:, 1:] - E[:, :-1]) / self.dy
            rE = E * self.rf[2:nx + 3][:, None]
            dB1 = (rE[1:, :] - rE[:-1, :]) / (self.dx * self.rc[2:nx + 2][:, None])
        else:
            dB0 = -(E[:, 1:] - E[:, :-1]) / self.dy
            dB1 = (E[1:, :] - E[:-1, :]) / self.dx
        return R, (dB0, dB1), U, Bf

    def _update(self, A, R, dt, inner):
        """A + dt R on the interior (R covers interior cells)"""
        return assign(A, inner, A[inner] + R * dt) if R is not None else A

    def step(self, state, dt, t, params=None):
        """one SSP-RK3 step (MHDSolver._rk_core + the boundary refresh of _post_step); state -> state"""
        ng, nx, ny = self.ng, self.nx, self.ny
        inner = (slice(ng, ng + nx), slice(ng, ng + ny))
        fx_in = (slice(2, nx + 3), slice(2, ny + 2))
        fy_in = (slice(2, nx + 2), slice(2, ny + 3))
        U0, F0 = list(state[:NVAR]), list(state[NVAR:])
        stage = lambda U, Bf, tt: self.rhs(U, Bf, tt, params)

        def bounds(U):
            rho = maximum(U[RHO], FLOOR_RHO)
            C = minimum(maximum(U[RHOC] / rho, 0.0), 1.0)
            U = list(U)
            U[RHOC] = rho * C
            return U

        # stage 1
        R, dB, U0b, F0g = stage(U0, F0, t)
        U1 = [self._update(U0b[k], R[k], dt, inner) for k in range(NVAR)]
        F1 = [assign(f, sl, f[sl] + d * dt) for f, d, sl in zip(F0g, dB, (fx_in, fy_in))]
        U1 = self._bc(bounds(U1), t + dt, params)
        # stage 2: U2 = 3/4 U0 + 1/4 (U1 + dt L(U1))
        R, dB, U1b, F1g = stage(U1, F1, t + dt)
        U2 = []
        for k in range(NVAR):
            a = self._update(U1b[k], R[k], dt, inner)
            U2.append(U0b[k] * 0.75 + a * 0.25)
        F2 = []
        for f0, f1, d, sl in zip(F0g, F1g, dB, (fx_in, fy_in)):
            F2.append(assign(f1, sl, 0.75 * f0[sl] + 0.25 * (f1[sl] + d * dt)))
        U2 = self._bc(bounds(U2), t + 0.5 * dt, params)
        # stage 3: U = 1/3 U0 + 2/3 (U2 + dt L(U2))
        R, dB, U2b, F2g = stage(U2, F2, t + 0.5 * dt)
        U3 = []
        for k in range(NVAR):
            a = self._update(U2b[k], R[k], dt, inner)
            U3.append(U0b[k] * (1.0 / 3.0) + a * (2.0 / 3.0))
        F3 = []
        for f0, f2, d, sl in zip(F0g, F2g, dB, (fx_in, fy_in)):
            F3.append(assign(f0, sl, f0[sl] / 3.0 + 2.0 / 3.0 * (f2[sl] + d * dt)))
        U3 = bounds(U3)
        for a in range(2):                           # cell B from the faces, then boundaries (as _post_step)
            f = F3[a]
            n = U3[0].shape[a]
            avg = 0.5 * (f[0:n, :] + f[1:n + 1, :]) if a == 0 else 0.5 * (f[:, 0:n] + f[:, 1:n + 1])
            U3[BX + a] = assign(U3[BX + a], inner, avg[inner])
        U3 = self._bc(U3, t + dt, params)
        F3 = self._ghost_faces(F3, U3)
        return U3 + F3

    # ---------------- forward run and adjoint ----------------
    def run(self, init, params, t_end=None, n_steps=None, dts=None, step_objective=None, final_objective=None,
            keep=False):
        """forward run (no tape). init(params) -> state; the step count is fixed by dts, n_steps or t_end.
        Returns (J, info) with info["dts"] the step sequence (reuse it for consistent finite differences) and,
        if keep, info["states"] the raw state after every step."""
        p = const(params) if not isinstance(params, Tensor) else const(params.v)
        st = init(p)
        states = [self.raw(st)] if keep else None
        J = 0.0
        t, k, used = 0.0, 0, []
        while True:
            if dts is not None:
                if k >= len(dts):
                    break
                dt = dts[k]
            else:
                if (n_steps is not None and k >= n_steps) or (t_end is not None and t >= t_end - 1e-14):
                    break
                dt = self.compute_dt(st)
                if t_end is not None:
                    dt = min(dt, t_end - t)
            if step_objective is not None:
                J += float(val(step_objective(st, t, p))) * dt
            st = self.step(st, dt, t, p)
            t += dt
            k += 1
            used.append(dt)
            if keep:
                states.append(self.raw(st))
        if final_objective is not None:
            J += float(val(final_objective(st, t, p)))
        return J, dict(dts=used, t=t, state=st, states=states)

    def value_and_grad(self, init, params, t_end=None, n_steps=None, dts=None, step_objective=None,
                       final_objective=None):
        """J and dJ/dparams by the checkpointed discrete adjoint (one forward + one backward pass)"""
        xp = self.xp
        params = xp.asarray(params, dtype=float)
        J, info = self.run(init, params, t_end, n_steps, dts, step_objective, final_objective, keep=True)
        dts, states = info["dts"], info["states"]
        times = np.concatenate([[0.0], np.cumsum(dts)])
        gp = xp.zeros_like(params)
        # adjoint of the final objective
        lam = [xp.zeros_like(a) for a in states[-1]]
        if final_objective is not None:
            with record():
                p = leaf(params)
                s = [leaf(a) for a in states[-1]]
                backward(final_objective(s, times[-1], p))
            lam = [a.g if a.g is not None else xp.zeros_like(a.v) for a in s]
            gp += p.g if p.g is not None else 0.0
        # one step at a time, backwards
        for n in range(len(dts) - 1, -1, -1):
            with record():
                p = leaf(params)
                s = [leaf(a) for a in states[n]]
                out = self.step(s, dts[n], times[n], p)
                outs, seeds = list(out), list(lam)
                if step_objective is not None:
                    outs.append(step_objective(s, times[n], p) * dts[n])
                    seeds.append(1.0)
                backward(outs, seeds)
            lam = [a.g if a.g is not None else xp.zeros_like(a.v) for a in s]
            if p.g is not None:
                gp += p.g
        # through the initial state
        with record():
            p = leaf(params)
            backward(list(init(p)), lam)
        if p.g is not None:
            gp += p.g
        return J, gp, info


# ============================================================
# helpers for building states and objectives
# ============================================================
def cells_to_state(sim, W_fn, params, Bf=None):
    """a state from primitive cell fields: W_fn(params) -> list of NVAR primitive Tensors / arrays over the
    full grid (rho, v, B, p, psi, C, s_e). Faces: given (raw arrays, parameter independent) or averaged from
    the cell B (exact when the normal component does not vary along its own axis, e.g. a uniform B_z in r-z)."""
    W = [w if isinstance(w, Tensor) else const(sim.xp.asarray(w) + 0.0 * sim.xp.zeros((sim.Nx, sim.Ny)))
         for w in W_fn(params)]
    g = sim.g
    rho = maximum(W[0], FLOOR_RHO)
    U = [rho, rho * W[1], rho * W[2], rho * W[3], W[4], W[5], W[6],
         W[7] / (g - 1.0) + 0.5 * rho * (W[1] * W[1] + W[2] * W[2] + W[3] * W[3])
         + 0.5 * (W[4] * W[4] + W[5] * W[5] + W[6] * W[6]), W[8], rho * W[9], rho * W[10]]
    if Bf is None:
        Bf = [sim._faces_from_cells(U[BX], 0), sim._faces_from_cells(U[BY], 1)]
    else:
        Bf = [const(sim.xp.asarray(f)) for f in Bf]
    return U + list(Bf)


def interior(sim, a):
    ng = sim.ng
    return a[ng:ng + sim.nx, ng:ng + sim.ny]


def total_energy(sim, state):
    """sum of E dV over the interior (per radian in r-z)"""
    return tsum(interior(sim, state[EN]) * sim.dV)


def finite_difference(sim, init, params, k, h, **kw):
    """central difference of J along parameter k (same dt sequence as the adjoint run: pass dts=...)"""
    p = np.array(params, dtype=float)
    pp, pm = p.copy(), p.copy()
    pp[k] += h
    pm[k] -= h
    Jp, _ = sim.run(init, sim.xp.asarray(pp), **kw)
    Jm, _ = sim.run(init, sim.xp.asarray(pm), **kw)
    return (Jp - Jm) / (2 * h)
