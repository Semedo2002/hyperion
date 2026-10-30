# tabular.py - tabulated equation of state and opacities: SESAME ASCII (EOS table 301, opacity tables 502 and
# 505) or an SI .npz table, interpolated on the GPU.
#
# On load a table is resampled onto a uniform grid in log10(rho) and log10(T) (linear interpolation in those
# coordinates). p, e and the opacities are interpolated in log space when they are positive everywhere (power
# laws, e.g. an ideal gas, are then exact) and linearly otherwise. The hydrodynamics asks for p(rho, e) and
# e(rho, p): both invert the table along T at fixed rho (binary search, then bilinear interpolation), in one
# CUDA kernel per call. c^2 comes from the table when it has one, otherwise from its derivatives,
#     c^2 = p_rho - p_T e_rho / e_T + p p_T / (rho^2 e_T)       (subscripts: partial derivatives).
# States beyond the table (in rho, or in e / p along T) are extrapolated linearly from the edge interval and
# blended into an analytic fallback (Mie-Gruneisen or stiffened gas) over `margin` grid cells, so the EOS stays
# continuous. Code units are set by (rho_unit, v_unit): rho / rho_unit, p / (rho_unit v_unit^2), e / v_unit^2.
#
# In the two-material solver a tabulated liner is used in pure liner cells; mixed cells (liner fraction below
# 0.98) keep the tangent stiffened gas of the volume-fraction closure, blended over 0.98 ... 1 as for the
# Mie-Gruneisen liner (TabularEOS).
import re

import numpy as np

from .physics import xp

_FLOAT = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?")


# ------------------------------------------------------------------ reading

def read_sesame(path, table, material=None):
    """words of one SESAME ASCII table (e.g. 301, 502): a header line '<flag> <material> <table> <nwords> ...'
    followed by the numbers, five per line. Returns (material, np.ndarray of words)."""
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines):
        tok = lines[i].split()
        if len(tok) >= 4 and all(t.lstrip("-").isdigit() for t in tok[:4]) and int(tok[0]) in (0, 1, 2):
            mat, tab, nw = int(tok[1]), int(tok[2]), int(tok[3])
            words, j = [], i + 1
            while len(words) < nw and j < len(lines):
                words += [float(x.replace("D", "E").replace("d", "e")) for x in _FLOAT.findall(lines[j])]
                j += 1
            if tab == table and (material is None or mat == material):
                return mat, np.array(words[:nw])
            i = j
        else:
            i += 1
    raise ValueError(f"table {table}" + (f" for material {material}" if material else "") + f" not in {path}")


def write_sesame(path, material, tables):
    """write SESAME ASCII: tables = {table id: words}; five E22.15 numbers per line"""
    with open(path, "w") as fh:
        for tab, words in tables.items():
            words = np.asarray(words, dtype=float)
            fh.write(f" 0 {material:6d} {tab:6d} {len(words):8d}   010126   1\n")
            for k in range(0, len(words), 5):
                fh.write("".join(f"{w:22.15E}" for w in words[k:k + 5]) + "\n")


def sesame_301(words):
    """total EOS: NR, NT, rho (g/cc), T (K), P (GPa), E (MJ/kg) with rho fastest -> SI arrays (nr, nt)"""
    nr, nt = int(words[0]), int(words[1])
    o = 2
    rho = words[o:o + nr] * 1e3; o += nr
    T = words[o:o + nt]; o += nt
    P = words[o:o + nr * nt].reshape(nt, nr).T * 1e9; o += nr * nt
    E = words[o:o + nr * nt].reshape(nt, nr).T * 1e6
    return rho, T, P, E


def sesame_opacity(words):
    """500-series opacity table: NR, NT, log10 rho (g/cc), log10 T (eV), log10 kappa (cm^2/g), rho fastest
    -> SI (rho kg/m^3, T K, kappa m^2/kg). Check that your library uses this convention."""
    nr, nt = int(words[0]), int(words[1])
    o = 2
    rho = 10**words[o:o + nr] * 1e3; o += nr
    T = 10**words[o:o + nt] * 11604.518; o += nt
    k = 10**words[o:o + nr * nt].reshape(nt, nr).T * 0.1
    return rho, T, k


def load(path, material=None):
    """dict of SI arrays from a SESAME file (tables 301 and, if present, 502 / 505) or an .npz with rho, T,
    p, e and optionally cs, kappa_R, kappa_P ((nr, nt) arrays, rho [kg/m^3], T [K], SI values)"""
    if path.endswith(".npz"):
        d = np.load(path)
        return {k: np.asarray(d[k], dtype=float) for k in d.files}
    out = {}
    try:
        _, w = read_sesame(path, 301, material)
        out["rho"], out["T"], out["p"], out["e"] = sesame_301(w)
    except ValueError:
        pass
    for tab, name in ((502, "kappa_R"), (505, "kappa_P")):
        try:
            _, w = read_sesame(path, tab, material)
            r, T, k = sesame_opacity(w)
            out.setdefault("rho_k", r); out.setdefault("T_k", T)
            out[name] = k
        except ValueError:
            pass
    if not out:
        raise ValueError(f"no 301 / 502 / 505 table in {path}")
    return out


# ------------------------------------------------------------------ uniform log grid

def _resample(lr, lt, f, LR, LT):
    """bilinear resampling of f(lr, lt) (nr, nt) onto the grid LR x LT (all log10)"""
    fr = np.interp(LR, lr, np.arange(len(lr)))
    ft = np.interp(LT, lt, np.arange(len(lt)))
    i = np.clip(np.floor(fr).astype(int), 0, len(lr) - 2); wr = (fr - i)[:, None]
    j = np.clip(np.floor(ft).astype(int), 0, len(lt) - 2); wt = (ft - j)[None, :]
    i, j = i[:, None], j[None, :]
    return ((1 - wr) * (1 - wt) * f[i, j] + wr * (1 - wt) * f[i + 1, j] + (1 - wr) * wt * f[i, j + 1]
            + wr * wt * f[i + 1, j + 1])


class Grid:
    """a quantity set on a uniform log10 rho x log10 T grid (code units), stored flat on the device"""

    def __init__(self, rho, T, fields, refine=2):
        lr, lt = np.log10(rho), np.log10(T)
        self.nr, self.nt = refine * (len(lr) - 1) + 1, refine * (len(lt) - 1) + 1
        LR, LT = np.linspace(lr[0], lr[-1], self.nr), np.linspace(lt[0], lt[-1], self.nt)
        self.lr0, self.dlr, self.lt0, self.dlt = LR[0], LR[1] - LR[0], LT[0], LT[1] - LT[0]
        self.logs, self.host, self.dev = {}, {}, {}
        for name, f in fields.items():
            lg = bool(np.all(f > 0))
            g = _resample(lr, lt, np.log10(f) if lg else f, LR, LT)
            self.logs[name] = lg
            self.host[name] = g
            self.dev[name] = xp.asarray(np.ascontiguousarray(g).ravel())
        self.LR, self.LT = LR, LT


_KERNEL = None


def _kernel():
    """CUDA elementwise kernel: invert table Q along T at fixed rho (mode 0) or evaluate at given log T
    (mode 1), returning the other tables bilinearly; plus the out-of-table distance in grid cells"""
    global _KERNEL
    if _KERNEL is None:
        import cupy as cp
        _KERNEL = cp.ElementwiseKernel(
            "float64 lrho, float64 q, raw float64 Q, raw float64 A, raw float64 B, raw float64 C, "
            "int32 nr, int32 nt, float64 lr0, float64 dlr, int32 mode",
            "float64 a, float64 b, float64 c, float64 lt, float64 out",
            r"""
            double fr = (lrho - lr0) / dlr;
            int ir = (int)floor(fr);
            ir = ir < 0 ? 0 : (ir > nr - 2 ? nr - 2 : ir);
            double wr = fr - ir;                                  // outside [0, 1]: extrapolated in rho
            double dr = wr < 0 ? -wr : (wr > 1 ? wr - 1 : 0.0);
            int jt; double wt;
            if (mode == 0) {
                #define QV(jj) ((1 - wr) * Q[ir * nt + (jj)] + wr * Q[(ir + 1) * nt + (jj)])
                int lo = 0, hi = nt - 1;
                double qlo = QV(0), qhi = QV(nt - 1);
                if (q <= qlo) { jt = 0; }
                else if (q >= qhi) { jt = nt - 2; }
                else {
                    while (hi - lo > 1) { int m = (lo + hi) >> 1; if (QV(m) <= q) lo = m; else hi = m; }
                    jt = lo;
                }
                double q0 = QV(jt), q1 = QV(jt + 1);
                wt = (q - q0) / (q1 - q0 != 0 ? q1 - q0 : 1e-300);
                #undef QV
            } else {
                double ft = q;                                   // q is the fractional T index
                jt = (int)floor(ft); jt = jt < 0 ? 0 : (jt > nt - 2 ? nt - 2 : jt);
                wt = ft - jt;
            }
            double dt = wt < 0 ? -wt : (wt > 1 ? wt - 1 : 0.0);
            int k00 = ir * nt + jt, k10 = (ir + 1) * nt + jt;
            double w00 = (1 - wr) * (1 - wt), w10 = wr * (1 - wt), w01 = (1 - wr) * wt, w11 = wr * wt;
            a = w00 * A[k00] + w10 * A[k10] + w01 * A[k00 + 1] + w11 * A[k10 + 1];
            b = w00 * B[k00] + w10 * B[k10] + w01 * B[k00 + 1] + w11 * B[k10 + 1];
            c = w00 * C[k00] + w10 * C[k10] + w01 * C[k00 + 1] + w11 * C[k10 + 1];
            lt = jt + wt;
            out = dr > dt ? dr : dt;
            """, "aegis_tab_lookup")
    return _KERNEL


def _lookup(g, lrho, q, Q, A, B, C, mode):
    if xp.__name__ == "cupy":
        return _kernel()(lrho, q, g.dev[Q], g.dev[A], g.dev[B], g.dev[C], np.int32(g.nr), np.int32(g.nt),
                         float(g.lr0), float(g.dlr), np.int32(mode))
    # NumPy path: the same algorithm, vectorised (bisection over the T index)
    H = {k: g.host[k] for k in (Q, A, B, C)}
    fr = (lrho - g.lr0) / g.dlr
    i = np.clip(np.floor(fr).astype(int), 0, g.nr - 2)
    wr = fr - i
    dr = np.maximum(np.maximum(-wr, wr - 1), 0)
    if mode == 0:
        QV = lambda jj: (1 - wr) * H[Q][i, jj] + wr * H[Q][i + 1, jj]
        lo, hi = np.zeros_like(i), np.full_like(i, g.nt - 1)
        for _ in range(int(np.ceil(np.log2(g.nt))) + 1):
            m = (lo + hi) // 2
            up = QV(m) <= q
            lo, hi = np.where(up & (hi - lo > 1), m, lo), np.where(~up & (hi - lo > 1), m, hi)
        j = np.where(q <= QV(np.zeros_like(i)), 0, np.where(q >= QV(np.full_like(i, g.nt - 1)), g.nt - 2, lo))
        q0, q1 = QV(j), QV(j + 1)
        wt = (q - q0) / np.where(q1 - q0 != 0, q1 - q0, 1e-300)
    else:
        j = np.clip(np.floor(q).astype(int), 0, g.nt - 2)
        wt = q - j
    dt = np.maximum(np.maximum(-wt, wt - 1), 0)
    bl = lambda f: ((1 - wr) * (1 - wt) * f[i, j] + wr * (1 - wt) * f[i + 1, j] + (1 - wr) * wt * f[i, j + 1]
                    + wr * wt * f[i + 1, j + 1])
    return bl(H[A]), bl(H[B]), bl(H[C]), j + wt, np.maximum(dr, dt)


# ------------------------------------------------------------------ EOS

class TabularMaterial:
    """p(rho, e), e(rho, p), c^2 from a table, in code units; fallback: an eos.Material for the states beyond it"""

    def __init__(self, data, rho_unit, v_unit, fallback, margin=2.0, refine=2):
        pu = rho_unit * v_unit**2
        rho = data["rho"] / rho_unit
        p, e = data["p"] / pu, data["e"] / v_unit**2
        if "cs" in data:
            c2 = (data["cs"] / v_unit)**2
        else:                                        # thermodynamic sound speed from the table's derivatives
            lr, lt = np.log10(data["rho"]), np.log10(data["T"])
            R, T = np.meshgrid(rho, data["T"], indexing="ij")

            def d(f, ax):                            # df/drho or df/dT; through log f where f > 0 (power laws exact)
                l, X = (lr, R) if ax == 0 else (lt, T)
                if np.all(f > 0):
                    return f / X * np.gradient(np.log10(f), l, axis=ax, edge_order=2)
                return np.gradient(f, l, axis=ax, edge_order=2) / (X * np.log(10))
            p_r, p_T, e_r, e_T = d(p, 0), d(p, 1), d(e, 0), d(e, 1)
            e_T = np.where(np.abs(e_T) > 0, e_T, 1e-300)
            c2 = np.maximum(p_r - p_T * e_r / e_T + p * p_T / (R**2 * e_T), 1e-30)
        if not (np.all(np.diff(e, axis=1) > 0) and np.all(np.diff(p, axis=1) > 0)):
            raise ValueError("table: e and p must increase with T at every density")
        self.g = Grid(rho, data["T"], {"p": p, "e": e, "c2": c2}, refine)
        self.fallback, self.margin = fallback, margin
        self.kind = "tabular"
        self.rho_range = (float(rho.min()), float(rho.max()))

    def _fb(self, rho):
        return self.fallback.phis(rho, xp)

    def _blend(self, out, tab, fb_fn):
        """table values, blended into the fallback (evaluated only if some state is off the table)"""
        if not bool(xp.any(out > 0)):
            return tab
        w = xp.clip(1.0 - out / self.margin, 0.0, 1.0)
        return w * tab + (1 - w) * fb_fn()

    def _decode(self, name, v):
        return 10**v if self.g.logs[name] else v

    def _encode(self, name, v):
        return xp.log10(xp.maximum(v, 1e-300)) if self.g.logs[name] else v

    def pressure(self, rho, e):
        """p from density and specific internal energy"""
        lr = xp.log10(xp.maximum(rho, 1e-300))
        p, _, c2, lt, out = _lookup(self.g, lr, self._encode("e", e), "e", "p", "e", "c2", 0)

        def fb():
            f1, f2, _, _ = self._fb(rho)
            return (rho * e - f2) / f1
        return self._blend(out, self._decode("p", p), fb)

    def energy(self, rho, p):
        """specific internal energy from density and pressure"""
        lr = xp.log10(xp.maximum(rho, 1e-300))
        _, e, _, lt, out = _lookup(self.g, lr, self._encode("p", p), "p", "p", "e", "c2", 0)

        def fb():
            f1, f2, _, _ = self._fb(rho)
            return (f1 * p + f2) / rho
        return self._blend(out, self._decode("e", e), fb)

    def c2(self, rho, p):
        lr = xp.log10(xp.maximum(rho, 1e-300))
        _, _, c2, _, out = _lookup(self.g, lr, self._encode("p", p), "p", "p", "e", "c2", 0)

        def fb():
            f1, f2, d1, d2 = self._fb(rho)
            return ((f1 * p + f2 + p) / rho - d2 - p * d1) / f1
        return self._blend(out, self._decode("c2", c2), fb)

    def temperature(self, rho, e):
        """T (K) from density and specific internal energy (inside the table)"""
        lr = xp.log10(xp.maximum(rho, 1e-300))
        _, _, _, lt, _ = _lookup(self.g, lr, self._encode("e", e), "e", "p", "e", "c2", 0)
        return 10**(self.g.lt0 + lt * self.g.dlt)

    def describe(self):
        r = self.rho_range
        return f"tabular ({self.g.nr} x {self.g.nt} log grid, rho {r[0]:.3g}-{r[1]:.3g} code) -> {self.fallback.describe()}"


class TabularEOS:
    """two-material EOS (same interface as eos.EOS) whose liner is a TabularMaterial in pure liner cells; mixed
    cells use the mixture of the fuel and the fallback's tangent stiffened gas, blended in over a = 0.98 ... 1"""

    @property
    def no_graph(self):                              # the array eps() branches on the host: no CUDA-graph
        from .eos_fused import device                # capture unless the device EOS (eos_fused.cu) covers it
        return device(self) is None

    def __init__(self, base, liner_table):
        self.base, self.tab = base, liner_table
        self.fuel, self.liner = base.fuel, base.liner
        self.volume_fraction = base.volume_fraction
        tangent = liner_table.fallback.tangent_stiffened() if liner_table.fallback.kind == "mie_gruneisen" \
            else liner_table.fallback
        from .eos import EOS
        self._mix = EOS(base.fuel, tangent, base.volume_fraction)

    def _w(self, a):
        w = xp.clip((xp.clip(a, 0, 1) - 0.98) / 0.02, 0.0, 1.0)
        return w * w * (3 - 2 * w)

    def coeffs(self, rho, a):                        # linear-form coefficients of the mixed-cell closure
        return self._mix.coeffs(rho, a)

    def pressure(self, rho, eps, a):
        from .eos import _dev
        r = _dev(self, "pressure", rho, eps, a)
        if r is not None:
            return r
        w = self._w(a)
        p_mix = self._mix.pressure(rho, eps, a)
        p_tab = self.tab.pressure(rho, eps / xp.maximum(rho, 1e-300))
        return (1 - w) * p_mix + w * p_tab

    def eps(self, rho, p, a):
        from .eos import _dev
        r = _dev(self, "eps", rho, p, a)
        if r is not None:
            return r
        w = self._w(a)
        e_mix = self._mix.eps(rho, p, a)
        e_tab = rho * self.tab.energy(rho, p)
        e = (1 - w) * e_mix + w * e_tab
        idx = xp.nonzero(((w > 0) & (w < 1)).ravel())[0]
        if idx.size:                                 # exact inverse of pressure() in the blend band: Newton,
            r_, p_, a_ = (q.ravel()[idx] for q in (rho, p, a))   # on those cells only
            e0 = e.ravel()[idx]                      # secant from the blended guess and a 0.1% step
            e1 = e0 * 1.001 + 1e-12
            f0 = self.pressure(r_, e0, a_) - p_
            for _ in range(4):
                f1 = self.pressure(r_, e1, a_) - p_
                d = f1 - f0
                e2 = e1 - f1 * (e1 - e0) / xp.where(d != 0, d, 1e-300)
                e0, f0, e1 = e1, f1, xp.where(d != 0, e2, e1)
            e_ = e1
            e = e.copy()
            e.ravel()[idx] = e_
        return e

    def c2(self, rho, p, a):
        from .eos import _dev
        r = _dev(self, "c2", rho, p, a)
        if r is not None:
            return r
        w = self._w(a)
        return (1 - w) * self._mix.c2(rho, p, a) + w * self.tab.c2(rho, p)

    def p_min(self, rho, a):
        return self._mix.p_min(rho, a)

    def p_floor(self, rho, a):
        return self._mix.p_floor(rho, a)

    def describe(self):
        return f"fuel: {self.fuel.describe()}; liner: {self.tab.describe()}; mixed cells: tangent stiffened"


# ------------------------------------------------------------------ opacities

class OpacityTable:
    """kappa_R(rho, T), kappa_P(rho, T) in code units (kappa rho_unit L0) at code temperature theta = T / T1"""

    def __init__(self, data, rho_unit, L0, T1, refine=1):
        rho = data.get("rho_k", data["rho"]) / rho_unit
        T = data.get("T_k", data["T"])
        f = {}
        for k in ("kappa_R", "kappa_P"):
            if k in data:
                f[k] = np.maximum(data[k], 1e-300) * rho_unit * L0
        if "kappa_R" not in f:
            raise ValueError("opacity table needs kappa_R")
        f.setdefault("kappa_P", f["kappa_R"])
        self.g = Grid(rho, T, {**f, "zero": np.ones_like(f["kappa_R"])}, refine)
        self.T1 = T1

    def __call__(self, rho, theta):
        """(kappa_R, kappa_P) per unit mass, code units; clamped to the table edges"""
        lr = xp.clip(xp.log10(xp.maximum(rho, 1e-300)), self.g.LR[0], self.g.LR[-1])
        lt = xp.clip(xp.log10(xp.maximum(theta * self.T1, 1e-300)), self.g.LT[0], self.g.LT[-1])
        ft = (lt - self.g.lt0) / self.g.dlt
        kr, kp, _, _, _ = _lookup(self.g, lr, ft, "kappa_R", "kappa_R", "kappa_P", "zero", 1)
        return self._d("kappa_R", kr), self._d("kappa_P", kp)

    def _d(self, name, v):
        return 10**v if self.g.logs[name] else v


# ------------------------------------------------------------------ synthetic tables (tests, examples)

def mie_gruneisen_table(rho0, c0, s, gamma0, cv, rho_range, T_range, nr=80, nt=80, kappa=None):
    """SI table of a Mie-Gruneisen solid with a constant heat capacity cv (J/kg/K): e = e_c(rho) + cv (T - T0),
    p = p_c(rho) + rho0 gamma0 cv (T - T0), cold curve from the linear Us-Up Hugoniot reference as in eos.py
    (T0 = 300 K). kappa: optional (kR0, kP0, m, n) power laws in SI. For tests and as a format example."""
    from .eos import Material
    m = Material("mie_gruneisen", rho0=rho0, c0=c0, s=s, gamma0=gamma0)
    rho = np.geomspace(*rho_range, nr)
    T = np.geomspace(*T_range, nt)
    phi1, phi2, _, _ = m.phis(rho, np)               # eps = phi1 p + phi2 on the reference (T0) state
    R, TT = np.meshgrid(rho, T, indexing="ij")
    T0 = 300.0
    p_ref = np.zeros_like(rho)                       # reference: the MG curve at p such that eps(p) = phi2 + ...
    e_ref = (phi1 * p_ref + phi2) / rho
    e = e_ref[:, None] + cv * (TT - T0)
    p = (R * e - phi2[:, None]) / phi1[:, None]      # MG: p = (eps - phi2) / phi1
    out = {"rho": rho, "T": T, "p": p, "e": e}
    if kappa is not None:
        kR0, kP0, mm, nn = kappa
        out["kappa_R"] = kR0 * R**mm * TT**nn
        out["kappa_P"] = kP0 * R**mm * TT**nn
    return out


def ideal_gas_table(gamma, R_gas, rho_range, T_range, nr=60, nt=60):
    """SI table of an ideal gas: p = rho R T, e = R T / (gamma - 1)"""
    rho = np.geomspace(*rho_range, nr)
    T = np.geomspace(*T_range, nt)
    Rr, TT = np.meshgrid(rho, T, indexing="ij")
    return {"rho": rho, "T": T, "p": Rr * R_gas * TT, "e": R_gas * TT / (gamma - 1) + 0 * Rr}
