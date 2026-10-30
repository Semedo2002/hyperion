// eos_fused.cu - the two-material EOS of eos.EOS and tabular.TabularEOS as device functions, so the
// conversions, the face floors and the HLLD solver (hlld_kernel.cu, compiled after this file) evaluate
// p(rho, eps), eps(rho, p) and c^2(rho, p) inline instead of in a chain of array operations.
//
// Parameters (double P[], built by eos_fused.py):
//   P[0], P[1]   fuel phi1, phi2 (ideal or stiffened: constant)
//   P[2], P[3]   liner phi1, phi2 used in mixed cells: the liner itself if stiffened / ideal, else its
//                tangent stiffened gas
//   P[4]         liner kind: 0 constant phis, 1 Mie-Gruneisen, 2 table
//   P[5..8]      Mie-Gruneisen rho0, c0, s, Gamma0 (the liner, or the table's fallback)
//   P[10..15]    table nr, nt, lr0, dlr (log10 rho grid), lt0, dlt (log10 T grid)
//   P[16..18]    1 if the p, e, c^2 tables hold log10 values
//   P[19]        fallback blend width (grid cells)
//   P[20..22]    table fallback: kind (0 constant phis, 1 Mie-Gruneisen), its phi1, phi2 if constant
// Table T[]: p, e, c^2 on the (nr, nt) grid, one after the other, index ir * nt + jt.
#define EOS_FLOOR_PR 1e-12

__device__ __forceinline__ double ef_clip01(double a) { return fmin(fmax(a, 0.0), 1.0); }

__device__ __forceinline__ double ef_blendw(double a) {     // 0 below a = 0.98, 1 at a = 1 (smoothstep)
    double w = fmin(fmax((ef_clip01(a) - 0.98) / 0.02, 0.0), 1.0);
    return w * w * (3.0 - 2.0 * w);
}

// eos.Material.phis for Mie-Gruneisen (linear Us-Up reference, linear in expansion, energy offset c0^2/G0)
__device__ void ef_mg(const double* __restrict__ P, double rho, double* p1, double* p2, double* d1, double* d2) {
    const double r0 = P[5], c0 = P[6], s = P[7], G0 = P[8], K = G0 * r0;
    double eta = fmin(1.0 - r0 / rho, 0.9 / s);
    double ec = fmax(eta, 0.0), om = 1.0 - s * ec;
    double pH = r0 * c0 * c0 * ec / (om * om);
    double dpH = r0 * c0 * c0 * (1.0 + s * ec) / (om * om * om);
    double f, df;
    if (eta > 0.0) { f = pH * (1.0 - 0.5 * G0 * ec); df = dpH * (1.0 - 0.5 * G0 * ec) - 0.5 * G0 * pH; }
    else { f = r0 * c0 * c0 * eta; df = r0 * c0 * c0; }
    df *= r0 / (rho * rho);
    double e_off = c0 * c0 / G0;
    *p1 = rho / K; *p2 = -rho * f / K + rho * e_off; *d1 = 1.0 / K; *d2 = -(f + rho * df) / K + e_off;
}

// eos.EOS.coeffs: mixture phis at fixed composition; with a table (kind 2) these are the mixed-cell closure
__device__ void ef_coeffs(const double* __restrict__ P, double rho, double a,
                          double* p1, double* p2, double* d1, double* d2) {
    a = ef_clip01(a);
    if (P[4] != 1.0) {
        *p1 = (1.0 - a) * P[0] + a * P[2]; *p2 = (1.0 - a) * P[1] + a * P[3]; *d1 = 0.0; *d2 = 0.0;
        return;
    }
    double w = fmin(fmax((a - 0.98) / 0.02, 0.0), 1.0);
    w = w * w * (3.0 - 2.0 * w);
    const double r0 = P[5], s = P[7];
    double rl = fmin(fmax(rho / fmax(a, 1e-12), 0.2 * r0), r0 / (1.0 - 0.9 / s));
    double m1, m2, md1, md2;
    ef_mg(P, rl, &m1, &m2, &md1, &md2);
    double l1 = (1.0 - w) * P[2] + w * m1, l2 = (1.0 - w) * P[3] + w * m2;
    *p1 = (1.0 - a) * P[0] + a * l1; *p2 = (1.0 - a) * P[1] + a * l2; *d1 = w * md1; *d2 = w * md2;
}

// ---------------------------------------------------------------- table (tabular.TabularMaterial)

// fractional table position of the state with table value q of Q (Q increasing along T) at log10 rho = lr;
// out: distance beyond the table in grid cells (0 inside)
__device__ void ef_find(const double* __restrict__ Q, int nr, int nt, double lr0, double dlr, double lr, double q,
                        int* ir_, double* wr_, int* jt_, double* wt_, double* out_) {
    double fr = (lr - lr0) / dlr;
    int ir = (int)floor(fr);
    ir = ir < 0 ? 0 : (ir > nr - 2 ? nr - 2 : ir);
    double wr = fr - ir;
    double dr = wr < 0 ? -wr : (wr > 1 ? wr - 1 : 0.0);
    const double* q0p = Q + ir * nt;
    const double* q1p = Q + (ir + 1) * nt;
    #define EF_QV(jj) ((1 - wr) * q0p[(jj)] + wr * q1p[(jj)])
    int jt, lo = 0, hi = nt - 1;
    if (q <= EF_QV(0)) jt = 0;
    else if (q >= EF_QV(nt - 1)) jt = nt - 2;
    else {
        while (hi - lo > 1) { int m = (lo + hi) >> 1; if (EF_QV(m) <= q) lo = m; else hi = m; }
        jt = lo;
    }
    double qa = EF_QV(jt), qb = EF_QV(jt + 1);
    #undef EF_QV
    double wt = (q - qa) / (qb - qa != 0 ? qb - qa : 1e-300);
    double dt = wt < 0 ? -wt : (wt > 1 ? wt - 1 : 0.0);
    *ir_ = ir; *wr_ = wr; *jt_ = jt; *wt_ = wt; *out_ = dr > dt ? dr : dt;
}

__device__ __forceinline__ double ef_bil(const double* __restrict__ F, int nt, int ir, double wr, int jt, double wt) {
    int k00 = ir * nt + jt, k10 = k00 + nt;
    return (1 - wr) * (1 - wt) * F[k00] + wr * (1 - wt) * F[k10] + (1 - wr) * wt * F[k00 + 1] + wr * wt * F[k10 + 1];
}

__device__ __forceinline__ double ef_enc(double v, double lg) { return lg != 0.0 ? log10(fmax(v, 1e-300)) : v; }
__device__ __forceinline__ double ef_dec(double v, double lg) { return lg != 0.0 ? pow(10.0, v) : v; }

__device__ void ef_fb(const double* __restrict__ P, double rho, double* p1, double* p2, double* d1, double* d2) {
    if (P[20] == 1.0) ef_mg(P, rho, p1, p2, d1, d2);
    else { *p1 = P[21]; *p2 = P[22]; *d1 = 0.0; *d2 = 0.0; }
}

// which = 0: p(rho, e); 1: e(rho, p); 2: c^2(rho, p)   (e specific internal energy)
__device__ double ef_table(const double* __restrict__ P, const double* __restrict__ T, int which, double rho, double x) {
    const int nr = (int)P[10], nt = (int)P[11], n = nr * nt;
    const double* Tp = T; const double* Te = T + n; const double* Tc = T + 2 * n;
    double lr = log10(fmax(rho, 1e-300));
    int ir, jt; double wr, wt, out, v;
    if (which == 0) {
        ef_find(Te, nr, nt, P[12], P[13], lr, ef_enc(x, P[17]), &ir, &wr, &jt, &wt, &out);
        v = ef_dec(ef_bil(Tp, nt, ir, wr, jt, wt), P[16]);
    } else {
        ef_find(Tp, nr, nt, P[12], P[13], lr, ef_enc(x, P[16]), &ir, &wr, &jt, &wt, &out);
        v = which == 1 ? ef_dec(ef_bil(Te, nt, ir, wr, jt, wt), P[17]) : ef_dec(ef_bil(Tc, nt, ir, wr, jt, wt), P[18]);
    }
    if (out > 0.0) {                                       // beyond the table: blend into the analytic fallback
        double w = fmin(fmax(1.0 - out / P[19], 0.0), 1.0);
        double f1, f2, g1, g2, fb;
        ef_fb(P, rho, &f1, &f2, &g1, &g2);
        if (which == 0) fb = (rho * x - f2) / f1;
        else if (which == 1) fb = (f1 * x + f2) / rho;
        else fb = ((f1 * x + f2 + x) / rho - g2 - x * g1) / f1;
        v = w * v + (1 - w) * fb;
    }
    return v;
}

// ---------------------------------------------------------------- two-material EOS

__device__ double ef_pressure(const double* __restrict__ P, const double* __restrict__ T, double rho, double eps, double a) {
    double p1, p2, d1, d2;
    ef_coeffs(P, rho, a, &p1, &p2, &d1, &d2);
    double pm = (eps - p2) / p1;
    if (P[4] != 2.0) return pm;
    double w = ef_blendw(a);
    if (w <= 0.0) return pm;
    double pt = ef_table(P, T, 0, rho, eps / fmax(rho, 1e-300));
    return w >= 1.0 ? pt : (1 - w) * pm + w * pt;
}

__device__ double ef_eps(const double* __restrict__ P, const double* __restrict__ T, double rho, double p, double a) {
    double p1, p2, d1, d2;
    ef_coeffs(P, rho, a, &p1, &p2, &d1, &d2);
    double em = p1 * p + p2;
    if (P[4] != 2.0) return em;
    double w = ef_blendw(a);
    if (w <= 0.0) return em;
    double et = rho * ef_table(P, T, 1, rho, p);
    if (w >= 1.0) return et;
    // blend band: the blend of the two inverses is not the inverse of the blend; secant on ef_pressure
    double e0 = (1 - w) * em + w * et, e1 = e0 * 1.001 + 1e-12;
    double f0 = ef_pressure(P, T, rho, e0, a) - p;
    for (int k = 0; k < 4; k++) {
        double f1 = ef_pressure(P, T, rho, e1, a) - p, d = f1 - f0;
        double e2 = e1 - f1 * (e1 - e0) / (d != 0 ? d : 1e-300);
        e0 = e1; f0 = f1; e1 = d != 0 ? e2 : e1;
    }
    return e1;
}

__device__ double ef_c2(const double* __restrict__ P, const double* __restrict__ T, double rho, double p, double a) {
    double p1, p2, d1, d2;
    ef_coeffs(P, rho, a, &p1, &p2, &d1, &d2);
    double cm = ((p1 * p + p2 + p) / rho - d2 - p * d1) / p1;
    if (P[4] != 2.0) return cm;
    double w = ef_blendw(a);
    if (w <= 0.0) return cm;
    double ct = ef_table(P, T, 2, rho, p);
    return w >= 1.0 ? ct : (1 - w) * cm + w * ct;
}

__device__ double ef_pmin(const double* __restrict__ P, double rho, double a) {   // cavitation pressure
    double p1, p2, d1, d2;
    ef_coeffs(P, rho, a, &p1, &p2, &d1, &d2);
    return -(p2 - rho * d2) / (1 + p1 - rho * d1);
}

__device__ __forceinline__ double ef_pfloor(const double* __restrict__ P, double rho, double a) {
    double pm = ef_pmin(P, rho, a);
    return pm + EOS_FLOOR_PR + 1e-12 * fabs(pm);
}

// ---------------------------------------------------------------- kernels

extern "C" {

// mode 0: p(rho = X, eps = Y, a); 1: eps(rho, p = Y, a); 2: c^2(rho, p, a); 3: p_floor(rho, a); 4: p_min(rho, a)
__global__ void eos_eval(const double* __restrict__ X, const double* __restrict__ Y, const double* __restrict__ A,
                         double* __restrict__ out, const long n, const int mode,
                         const double* __restrict__ P, const double* __restrict__ T) {
    long i = (long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double r = X[i], a = A[i];
    double v;
    if (mode == 0) v = ef_pressure(P, T, r, Y[i], a);
    else if (mode == 1) v = ef_eps(P, T, r, Y[i], a);
    else if (mode == 2) v = ef_c2(P, T, r, Y[i], a);
    else if (mode == 3) v = ef_pfloor(P, r, a);
    else v = ef_pmin(P, r, a);
    out[i] = v;
}

__global__ void c2p_eos(const double* __restrict__ U, double* __restrict__ W, const long n, const int vf,
                        const double frho, const double* __restrict__ P, const double* __restrict__ T) {
    long i = (long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double rho = fmax(U[0*n + i], frho), ri = 1.0 / rho;
    double vx = U[1*n + i] * ri, vy = U[2*n + i] * ri, vz = U[3*n + i] * ri;
    double bx = U[4*n + i], by = U[5*n + i], bz = U[6*n + i];
    double a = vf ? U[9*n + i] : U[9*n + i] * ri;
    W[0*n + i] = rho; W[1*n + i] = vx; W[2*n + i] = vy; W[3*n + i] = vz;
    W[4*n + i] = bx; W[5*n + i] = by; W[6*n + i] = bz; W[8*n + i] = U[8*n + i];
    W[9*n + i] = a; W[10*n + i] = U[10*n + i] * ri;
    double ke = 0.5 * rho * (vx*vx + vy*vy + vz*vz), me = 0.5 * (bx*bx + by*by + bz*bz);
    W[7*n + i] = fmax(ef_pressure(P, T, rho, U[7*n + i] - ke - me, a), ef_pfloor(P, rho, a));
}

__global__ void p2c_eos(const double* __restrict__ W, double* __restrict__ U, const long n, const int vf,
                        const double frho, const double* __restrict__ P, const double* __restrict__ T) {
    long i = (long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double rho = fmax(W[0*n + i], frho);
    double vx = W[1*n + i], vy = W[2*n + i], vz = W[3*n + i];
    double bx = W[4*n + i], by = W[5*n + i], bz = W[6*n + i];
    U[0*n + i] = rho; U[1*n + i] = rho*vx; U[2*n + i] = rho*vy; U[3*n + i] = rho*vz;
    U[4*n + i] = bx; U[5*n + i] = by; U[6*n + i] = bz; U[8*n + i] = W[8*n + i];
    U[9*n + i] = vf ? W[9*n + i] : rho * W[9*n + i];
    U[10*n + i] = rho * W[10*n + i];
    U[7*n + i] = ef_eps(P, T, rho, W[7*n + i], W[9*n + i]) + 0.5 * rho * (vx*vx + vy*vy + vz*vz)
                 + 0.5 * (bx*bx + by*by + bz*bz);
}

// face states in place: rho floor, tracer clipped to [0, 1], p above cavitation
__global__ void floor_faces_eos(double* __restrict__ Wf, const long n, const double frho,
                                const double* __restrict__ P) {
    long i = (long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double rho = fmax(Wf[0*n + i], frho), a = ef_clip01(Wf[9*n + i]);
    Wf[0*n + i] = rho; Wf[9*n + i] = a;
    Wf[7*n + i] = fmax(Wf[7*n + i], ef_pfloor(P, rho, a));
}

}
