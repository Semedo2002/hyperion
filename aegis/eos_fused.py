# eos_fused.py - host side of the device EOS (eos_fused.cu): compiles it together with the HLLD kernel, packs an
# eos.EOS / tabular.TabularEOS into a parameter vector and a table array, and launches the fused kernels.
#
# Covered: a fuel that is an ideal or stiffened gas, with a liner that is ideal, stiffened, Mie-Gruneisen or a
# table (with a stiffened or Mie-Gruneisen fallback). Anything else (a Mie-Gruneisen fuel) returns None from
# device() and keeps the array implementation in eos.py / tabular.py, as does AEGIS_ARRAY_EOS=1 in the
# environment (for comparisons and debugging).
import os
from pathlib import Path

import numpy as np
import cupy as cp

from .config import FLOOR_RHO

_HERE = Path(__file__).parent
_SRC = (_HERE / "eos_fused.cu").read_text(encoding="utf-8-sig") + "\n" + \
       (_HERE / "hlld_kernel.cu").read_text(encoding="utf-8-sig")
_mod = cp.RawModule(code=_SRC)
hlld_kernel = _mod.get_function("hlld_flux_x_kernel")
_k = {n: _mod.get_function(n) for n in ("eos_eval", "c2p_eos", "p2c_eos", "floor_faces_eos")}
_BLOCK = 256
MODES = {"pressure": 0, "eps": 1, "c2": 2, "p_floor": 3, "p_min": 4}


def _grid(n):
    return ((int(n) + _BLOCK - 1) // _BLOCK,)


def _const_phis(m):
    p1, p2, _, _ = m.phis(1.0, np)
    return float(p1), float(p2)


def _pack(eos):
    from .eos import EOS
    from .tabular import TabularEOS
    tab = isinstance(eos, TabularEOS)
    base = eos.base if tab else eos
    if not isinstance(base, EOS) or base.fuel.kind not in ("ideal", "stiffened"):
        return None
    P = np.zeros(28)
    P[0], P[1] = _const_phis(base.fuel)
    if tab:
        P[4] = 2
        P[2], P[3] = _const_phis(eos._mix.liner)
        fb = eos.tab.fallback
        mg = fb
        P[20] = 1 if fb.kind == "mie_gruneisen" else 0
        if fb.kind != "mie_gruneisen":
            P[21], P[22] = _const_phis(fb)
        g = eos.tab.g
        P[10:16] = g.nr, g.nt, g.lr0, g.dlr, g.lt0, g.dlt
        P[16:19] = [float(g.logs[k]) for k in ("p", "e", "c2")]
        P[19] = eos.tab.margin
        T = cp.ascontiguousarray(cp.concatenate([g.dev["p"], g.dev["e"], g.dev["c2"]]))
    else:
        liner = base.liner
        mg = liner
        if liner.kind == "mie_gruneisen":
            P[4] = 1
            P[2], P[3] = _const_phis(liner.tangent_stiffened())
        else:
            P[2], P[3] = _const_phis(liner)
        T = cp.zeros(1)
    if mg.kind == "mie_gruneisen":
        P[5:9] = mg.rho0, mg.c0, mg.s, mg.gamma0
    return cp.asarray(P), T


def device(eos):
    """(P, T) device arrays of eos, cached on the object; None if the device EOS does not cover it"""
    d = getattr(eos, "_device_eos", False)
    if d is False:
        d = None if os.environ.get("AEGIS_ARRAY_EOS") == "1" else _pack(eos)
        eos._device_eos = d
    return d


def evaluate(eos, mode, rho, y, a):
    """eos.pressure / eps / c2 / p_floor / p_min in one launch, or None (not covered, or no device array among
    the inputs: scalar and NumPy callers keep the array implementation)"""
    ins = (rho, y, a)
    if not any(isinstance(v, cp.ndarray) for v in ins) or any(isinstance(v, np.ndarray) for v in ins):
        return None
    d = device(eos)
    if d is None:
        return None
    shape = cp.broadcast(*[v for v in ins if v is not None]).shape
    X, Y, A = (cp.zeros(shape) if v is None else
               cp.ascontiguousarray(cp.broadcast_to(cp.asarray(v, dtype=cp.float64), shape)) for v in ins)
    out = cp.empty(shape)
    n = out.size
    if n:
        _k["eos_eval"](_grid(n), (_BLOCK,), (X, Y, A, out, np.int64(n), np.int32(MODES[mode]), d[0], d[1]))
    return out


def cons_to_prim(U, eos):
    d = device(eos)
    if d is None:
        return None
    U = cp.ascontiguousarray(U)
    W = cp.empty_like(U)
    n = U[0].size
    _k["c2p_eos"](_grid(n), (_BLOCK,), (U, W, np.int64(n), np.int32(eos.volume_fraction), np.float64(FLOOR_RHO),
                                        d[0], d[1]))
    return W


def prim_to_cons(W, eos):
    d = device(eos)
    if d is None:
        return None
    W = cp.ascontiguousarray(W)
    U = cp.empty_like(W)
    n = W[0].size
    _k["p2c_eos"](_grid(n), (_BLOCK,), (W, U, np.int64(n), np.int32(eos.volume_fraction), np.float64(FLOOR_RHO),
                                        d[0], d[1]))
    return U


def floor_faces(Wf, eos):
    """rho floor, tracer clip and cavitation floor on face states, in place; False if not covered"""
    d = device(eos)
    if d is None or not Wf.flags.c_contiguous:
        return False
    n = Wf[0].size
    _k["floor_faces_eos"](_grid(n), (_BLOCK,), (Wf, np.int64(n), np.float64(FLOOR_RHO), d[0]))
    return True
