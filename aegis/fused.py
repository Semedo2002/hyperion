# fused.py - single-launch CUDA kernels for the hot elementwise chains (conversions, MUSCL, CT corner EMF).
#
# On small and medium grids the step time was set by kernel launches (one CuPy launch per array operation,
# slow under Windows WDDM), not by arithmetic; these kernels do in one launch what took 10-40. They reproduce
# the arithmetic of the array versions in physics.py and ct.py (up to round-off) and are used for the ideal-gas
# path; multi-material runs keep the array versions where the EOS is needed.
import numpy as np
import cupy as cp

from .config import NVAR, FLOOR_RHO, FLOOR_PR

_SRC = r"""
#define NV __NVAR__
extern "C" {

__global__ void c2p_ideal(const double* __restrict__ U, double* __restrict__ W, const long n, const double g,
                          const int vf, const double frho, const double fpr) {
    long i = (long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double rho = fmax(U[0*n + i], frho), ri = 1.0 / rho;
    double vx = U[1*n + i] * ri, vy = U[2*n + i] * ri, vz = U[3*n + i] * ri;
    double bx = U[4*n + i], by = U[5*n + i], bz = U[6*n + i];
    W[0*n + i] = rho; W[1*n + i] = vx; W[2*n + i] = vy; W[3*n + i] = vz;
    W[4*n + i] = bx; W[5*n + i] = by; W[6*n + i] = bz; W[8*n + i] = U[8*n + i];
    W[9*n + i] = vf ? U[9*n + i] : U[9*n + i] * ri;
    W[10*n + i] = U[10*n + i] * ri;
    double ke = 0.5 * rho * (vx*vx + vy*vy + vz*vz), me = 0.5 * (bx*bx + by*by + bz*bz);
    W[7*n + i] = fmax((g - 1.0) * (U[7*n + i] - ke - me), fpr);
}

__global__ void p2c_ideal(const double* __restrict__ W, double* __restrict__ U, const long n, const double g,
                          const int vf, const double frho) {
    long i = (long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double rho = fmax(W[0*n + i], frho);
    double vx = W[1*n + i], vy = W[2*n + i], vz = W[3*n + i];
    double bx = W[4*n + i], by = W[5*n + i], bz = W[6*n + i];
    U[0*n + i] = rho; U[1*n + i] = rho*vx; U[2*n + i] = rho*vy; U[3*n + i] = rho*vz;
    U[4*n + i] = bx; U[5*n + i] = by; U[6*n + i] = bz; U[8*n + i] = W[8*n + i];
    U[9*n + i] = vf ? W[9*n + i] : rho * W[9*n + i];
    U[10*n + i] = rho * W[10*n + i];
    U[7*n + i] = W[7*n + i] / (g - 1.0) + 0.5 * rho * (vx*vx + vy*vy + vz*vz) + 0.5 * (bx*bx + by*by + bz*bz);
}

// van Leer MUSCL along axis `ax` of W[NV][A][B][C]; face k (0..N-4) lies between cells k+1 and k+2.
// Output WL, WR [NV][..faces..]; floors on rho and (if fp) on p.
__global__ void muscl(const double* __restrict__ W, double* __restrict__ WL, double* __restrict__ WR,
                      const long A, const long B, const long C, const int ax, const double frho,
                      const double fpr, const int fp) {
    long nA = A - 3*(ax == 0), nB = B - 3*(ax == 1), nC = C - 3*(ax == 2);
    long nf = nA * nB * nC;
    long t = (long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= nf * NV) return;
    long v = t / nf, r = t % nf;
    long c = r % nC, b = (r / nC) % nB, a = r / (nC * nB);
    long sA = B * C, sB = C, sC = 1, s = (ax == 0) ? sA : ((ax == 1) ? sB : sC);
    long base = v * A * B * C + a * sA + b * sB + c * sC;      // cell k (the face's index along ax)
    double w0 = W[base], w1 = W[base + s], w2 = W[base + 2*s], w3 = W[base + 3*s];
    double dm1 = w1 - w0, dp1 = w2 - w1, ab1 = dm1 * dp1;
    double d1 = ab1 > 0 ? 2.0 * ab1 / (dm1 + dp1 + 1e-30) : 0.0;
    double dm2 = w2 - w1, dp2 = w3 - w2, ab2 = dm2 * dp2;
    double d2 = ab2 > 0 ? 2.0 * ab2 / (dm2 + dp2 + 1e-30) : 0.0;
    double l = w1 + 0.5 * d1, rr = w2 - 0.5 * d2;
    if (v == 0) { l = fmax(l, frho); rr = fmax(rr, frho); }
    if (v == 7 && fp) { l = fmax(l, fpr); rr = fmax(rr, fpr); }
    WL[t] = l; WR[t] = rr;
}

// CT corner EMF (2D, Gardiner & Stone 2005 "CT contact"): see ct.edge_emf.
// Fx: [NV][nx+1][Ny] (x-faces 2..nx+2), Fy: [NV][Nx][ny+1] (y-faces 2..ny+2), W: [NV][Nx][Ny], out E: [nx+1][ny+1]
__global__ void emf2d(const double* __restrict__ Fx, const double* __restrict__ Fy, const double* __restrict__ W,
                      double* __restrict__ E, const long Nx, const long Ny, const long nx, const long ny,
                      const double hx, const double hy, const double* __restrict__ tolp) {
    const double tol = tolp[0];
    long t = (long)blockDim.x * blockIdx.x + threadIdx.x;
    long ne = (nx + 1) * (ny + 1);
    if (t >= ne) return;
    long k = t / (ny + 1), q = t % (ny + 1);   // corner on x-face I = k + 2, y-face J = q + 2
    long I = k + 2, J = q + 2;
    long nfx = (nx + 1) * Ny, nfy = Nx * (ny + 1), nc = Nx * Ny;
    // E_z on x-faces (rows J, J-1): -Fx[BY]; on y-faces (cols I, I-1): +Fy[BX]
    double Ea_hi = -Fx[5*nfx + k*Ny + J], Ea_lo = -Fx[5*nfx + k*Ny + (J - 1)];
    double Eb_r = Fy[4*nfy + I*(ny + 1) + q], Eb_l = Fy[4*nfy + (I - 1)*(ny + 1) + q];
    double ma_hi = Fx[0*nfx + k*Ny + J], ma_lo = Fx[0*nfx + k*Ny + (J - 1)];
    double mb_r = Fy[0*nfy + I*(ny + 1) + q], mb_l = Fy[0*nfy + (I - 1)*(ny + 1) + q];
    #define EC(i, j) (-(W[1*nc + (i)*Ny + (j)] * W[5*nc + (i)*Ny + (j)] - W[2*nc + (i)*Ny + (j)] * W[4*nc + (i)*Ny + (j)]))
    double Ec_c = EC(I, J), Ec_l = EC(I - 1, J), Ec_d = EC(I, J - 1), Ec_ld = EC(I - 1, J - 1);
    #define UP(m, lo, hi) ((m) > tol ? (lo) : ((m) < -tol ? (hi) : 0.5 * ((lo) + (hi))))
    double db_hi = UP(ma_hi, 2.0 * (Ec_l - Eb_l) / hy, 2.0 * (Ec_c - Eb_r) / hy);
    double db_lo = UP(ma_lo, 2.0 * (Eb_l - Ec_ld) / hy, 2.0 * (Eb_r - Ec_d) / hy);
    double da_r = UP(mb_r, 2.0 * (Ec_d - Ea_lo) / hx, 2.0 * (Ec_c - Ea_hi) / hx);
    double da_l = UP(mb_l, 2.0 * (Ea_lo - Ec_ld) / hx, 2.0 * (Ea_hi - Ec_l) / hx);
    E[t] = 0.25 * (Ea_hi + Ea_lo + Eb_r + Eb_l) + 0.125 * hy * (db_lo - db_hi) + 0.125 * hx * (da_l - da_r);
}

}
""".replace("__NVAR__", str(NVAR))

_mod = cp.RawModule(code=_SRC)
_k = {name: _mod.get_function(name) for name in ("c2p_ideal", "p2c_ideal", "muscl", "emf2d")}
_BLOCK = 256


def _grid(n):
    return ((int(n) + _BLOCK - 1) // _BLOCK,)


def cons_to_prim_ideal(U, gamma, vf=False):
    U = cp.ascontiguousarray(U)
    W = cp.empty_like(U)
    n = U[0].size
    _k["c2p_ideal"](_grid(n), (_BLOCK,), (U, W, np.int64(n), np.float64(gamma), np.int32(vf),
                                          np.float64(FLOOR_RHO), np.float64(FLOOR_PR)))
    return W


def prim_to_cons_ideal(W, gamma, vf=False):
    W = cp.ascontiguousarray(W)
    U = cp.empty_like(W)
    n = W[0].size
    _k["p2c_ideal"](_grid(n), (_BLOCK,), (W, U, np.int64(n), np.float64(gamma), np.int32(vf), np.float64(FLOOR_RHO)))
    return U


def muscl(W, axis, floor_p=True):
    """van Leer MUSCL along spatial axis `axis` of W (var, ...spatial); returns WL, WR with the face axis
    shortened by 3, floors on rho (and p if floor_p)"""
    W = cp.ascontiguousarray(W)
    sp = W.shape[1:]
    A, B, C = (sp + (1, 1, 1))[:3]
    fshape = list(sp)
    fshape[axis] -= 3
    WL = cp.empty((NVAR,) + tuple(fshape))
    WR = cp.empty_like(WL)
    n = WL.size
    _k["muscl"](_grid(n), (_BLOCK,), (W, WL, WR, np.int64(A), np.int64(B), np.int64(C), np.int32(axis),
                                      np.float64(FLOOR_RHO), np.float64(FLOOR_PR), np.int32(floor_p)))
    return WL, WR


def emf2d(Fx, Fy, W, nx, ny, hx, hy):
    Fx, Fy, W = cp.ascontiguousarray(Fx), cp.ascontiguousarray(Fy), cp.ascontiguousarray(W)
    Nx, Ny = W.shape[1], W.shape[2]
    E = cp.empty((nx + 1, ny + 1))
    tol = cp.maximum(1e-10 * cp.maximum(cp.max(cp.abs(Fx[0])), cp.max(cp.abs(Fy[0]))), 1e-300).reshape(1)
    _k["emf2d"](_grid(E.size), (_BLOCK,), (Fx, Fy, W, E, np.int64(Nx), np.int64(Ny), np.int64(nx), np.int64(ny),
                                           np.float64(hx), np.float64(hy), tol))
    return E
