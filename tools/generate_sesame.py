"""SESAME-compatible EOS and opacity tables from analytic physics models (no LANL data).

For Al, Cu, D-T, Au and Pb this writes 2D tables p, e, c_s, kappa_R, kappa_P on a log rho x log T grid, as an SI
.npz (what aegis/tabular.py reads) and optionally SESAME ASCII (tables 301, 502, 505). Everything follows from one
Helmholtz free energy per unit mass, so the EOS is thermodynamically consistent by construction:

    F(rho, T) = E_cold(rho) + F_ion(rho, T) + F_e(rho, T) - F_e(rho, 0)

  cold     Vinet curve from (rho0, K0, K0'): Murnaghan-like near rho0, but its energy saturates at the cohesive
           energy 9 K0 / (rho0 xi^2) in expansion (p -> 0) instead of a constant tension
  ions     F_ion = -(k T / m) [1.5 ln(T / theta_g) + 1.5 ln(T / (T + T_m))], theta_g ~ rho^(2/3), Lindemann melting
           T_m = T_m0 (rho0/rho)^(2/3) exp(2 Gamma0 (1 - rho0/rho)) with rho Gamma = rho0 Gamma0 (as in Cowan's
           model): e_ion = (k T / m)(1.5 + 1.5 T_m / (T + T_m)); below melting p_ion = 3 Gamma rho k T / m exactly
           (Mie-Gruneisen), far above it the ideal gas
  electrons ideal Fermi-Dirac gas of n_e = Z* n_i free electrons (exact Fermi integrals: degenerate to classical)
           with the Dirac exchange term, minus its T = 0 value (the cold curve already holds the T = 0 physics);
           Z*(rho, T) from the Thomas-Fermi ionization fit of More (1985, used in QEOS), and for hydrogen (D-T) the
           larger of that and Saha (13.6 eV, LTE), so thermal and pressure ionization are both there
p = rho^2 dF/drho and e = F - T dF/dT are taken analytically for the cold and ion parts and by central differences
(relative step 1e-5) for the electron part; c_s^2 = p_rho + T p_T^2 / (rho^2 e_T).

Opacities (LTE, average ion of charge Z*): free-free (Kramers with stimulated emission, Rybicki & Lightman 5.18b,
Gaunt factor 1), bound-free from the outermost occupied shell of a screened-hydrogenic average atom
(sigma_n = 7.91e-18 cm^2 n / Z_eff^2 (nu_n / nu)^3 above its edge), and Thomson scattering in the Rosseland mean.
The Planck and Rosseland means are integrated over frequency, so their ratio is the model's own. No line
(bound-bound) opacity: for partially ionized mid- and high-Z matter the means are lower bounds.

These are physics-based, smooth and consistent tables for development and for the solver's tabular path; they are
not a substitute for evaluated SESAME / TOPS data where accuracy matters. Reference checks (python main.py --test
sesame): ambient density at zero pressure, sound speed, the shock Hugoniot against measured Us-up fits, the fully
ionized ideal-gas limit, and thermodynamic consistency of the tables.

    python tools/generate_sesame.py                         # all materials, tables/ (npz)
    python tools/generate_sesame.py --materials al cu --sesame --nr 400 --nt 200
"""
import os
import sys
import argparse

import numpy as np

KB, MU, ME, HBAR, QE, EPS0 = 1.380649e-23, 1.66053907e-27, 9.1093837e-31, 1.054571817e-34, 1.602176634e-19, 8.8541878128e-12
EV = QE / KB                                                     # K per eV

# Z, A, rho0 (kg/m^3), K0 (Pa), K0', Gamma0, T_melt0 (K), theta_D (K), SESAME id, (rho range / rho0), (T range K),
# measured Hugoniot Us = c0 + s up (km/s) for the reference check
MATERIALS = {
    "al": dict(Z=13, A=26.982, rho0=2700.0, K0=76e9, K1=4.7, G0=2.0, Tm=933.5, thD=428.0, id=93719,
               rho=(1e-4, 30.0), T=(100.0, 1e8), hug=(5.33, 1.34)),
    "cu": dict(Z=29, A=63.546, rho0=8960.0, K0=137e9, K1=5.3, G0=2.0, Tm=1357.8, thD=343.0, id=93337,
               rho=(1e-4, 30.0), T=(100.0, 1e8), hug=(3.94, 1.49)),
    "dt": dict(Z=1, A=2.515, rho0=250.0, K0=0.3e9, K1=6.0, G0=2.0, Tm=19.8, thD=100.0, id=95271,
               rho=(1e-4, 4000.0), T=(5.0, 1e8), hug=None, Tref=10.0),
    "au": dict(Z=79, A=196.97, rho0=19300.0, K0=167e9, K1=5.9, G0=3.0, Tm=1337.3, thD=165.0, id=92700,
               rho=(1e-4, 30.0), T=(100.0, 1e8), hug=(3.06, 1.57)),
    "pb": dict(Z=82, A=207.2, rho0=11340.0, K0=46e9, K1=5.5, G0=2.6, Tm=600.6, thD=105.0, id=93200,
               rho=(1e-4, 30.0), T=(50.0, 1e8), hug=(2.05, 1.46)),
}


# ------------------------------------------------------------------ ionization

def z_tf(Z, A, rho, T):
    """mean ionization of the Thomas-Fermi atom, More's fit (rho kg/m^3, T K)"""
    T0 = (T / EV) / Z**(4.0 / 3.0)
    R = (rho * 1e-3) / (Z * A)
    Tf = T0 / (1 + T0)
    a = 0.003323 * T0**0.9718 + 9.26148e-5 * T0**3.10165
    b = -np.exp(-1.7630 + 1.43175 * Tf + 0.31546 * Tf**7)
    c = -0.366667 * Tf + 0.983333
    Q1 = a * R**b
    Q = (R**c + Q1**c)**(1.0 / c)
    x = 14.3139 * Q**0.6624
    return Z * x / (1 + x + np.sqrt(1 + 2 * x))


def z_saha_h(A, rho, T):
    """ionized fraction of hydrogen in LTE (Saha, 13.6 eV, statistical weights 1/2 x 2)"""
    n = rho / (A * MU)
    S = (ME * KB * T / (2 * np.pi * HBAR**2))**1.5 * np.exp(-13.6 * EV / T) / n
    return 2.0 / (1.0 + np.sqrt(1.0 + 4.0 / np.maximum(S, 1e-300)))     # x^2 / (1 - x) = S


def zbar(m, rho, T):
    z = z_tf(m["Z"], m["A"], rho, T)
    if m["Z"] == 1:                                  # pressure (TF) or thermal (Saha) ionization, smoothly
        zs = z_saha_h(m["A"], rho, T)
        z = z + zs - z * zs
    return np.clip(z, 1e-6, m["Z"])


# ------------------------------------------------------------------ Fermi-Dirac electrons

_ETA = np.linspace(-40.0, 60.0, 8001)


def _fd_table():
    """F_-1/2, F_1/2, F_3/2 on _ETA by quadrature in t = sqrt(x) (no singularity at x = 0)"""
    t = np.linspace(0.0, np.sqrt(125.0), 6001)[None, :]
    out = []
    for k in range(0, len(_ETA), 1000):
        occ = 1.0 / (np.exp(np.clip(t * t - _ETA[k:k + 1000, None], -700, 700)) + 1.0)
        out.append([np.trapz(2 * t**q * occ, t, axis=1) for q in (0, 2, 4)])   # dx = 2 t dt: t^(2j+1)
    return [np.concatenate([o[j] for o in out]) for j in range(3)]


_FM12, _F12, _F32 = _fd_table()


def _sommerfeld(j, eta):
    """F_j(eta) for large eta, to eta^-4: eta^(j+1)/(j+1) [1 + (j+1) j pi^2/6 eta^-2 + (j+1) j (j-1)(j-2) 7 pi^4/360 eta^-4]"""
    return eta**(j + 1) / (j + 1) * (1 + (j + 1) * j * np.pi**2 / 6 / eta**2
                                     + (j + 1) * j * (j - 1) * (j - 2) * 7 * np.pi**4 / 360 / eta**4)


def _hermite(eta, F, dF):
    """cubic Hermite interpolation on _ETA with the exact derivatives dF (C1, error ~ h^4)"""
    h = _ETA[1] - _ETA[0]
    k = np.clip(((eta - _ETA[0]) / h).astype(int), 0, len(_ETA) - 2)
    s = (eta - _ETA[k]) / h
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * F[k] + h10 * h * dF[k] + h01 * F[k + 1] + h11 * h * dF[k + 1]


def fermi(eta):
    """(F_1/2, F_3/2), F_j(eta) = int x^j / (exp(x - eta) + 1) dx: Hermite-interpolated quadrature (dF_j/deta =
    j F_j-1) inside -40 < eta < 60, the classical limit below and the Sommerfeld series above"""
    eta = np.asarray(eta, float)
    e_in = np.clip(eta, _ETA[0], _ETA[-1])
    f12 = _hermite(e_in, _F12, 0.5 * _FM12)
    f32 = _hermite(e_in, _F32, 1.5 * _F12)
    lo, hi = eta < _ETA[0], eta > _ETA[-1]
    e = np.exp(np.minimum(eta, 0.0))
    f12 = np.where(lo, 0.5 * np.sqrt(np.pi) * e, f12)
    f32 = np.where(lo, 0.75 * np.sqrt(np.pi) * e, f32)
    et = np.maximum(eta, 1.0)
    f12 = np.where(hi, _sommerfeld(0.5, et), f12)
    f32 = np.where(hi, _sommerfeld(1.5, et), f32)
    return f12, f32


def eta_of(y):
    """inverse of F_1/2 (bisection in a transformed variable, vectorised)"""
    lo, hi = np.full(np.shape(y), -700.0), np.full(np.shape(y), 1e9)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        up = fermi(mid)[0] < y
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def f_electron(m, rho, T):
    """free energy per unit mass of the free electrons (thermal ideal Fermi gas + Dirac exchange)"""
    z = zbar(m, rho, T)
    ne = z * rho / (m["A"] * MU)
    lam = np.sqrt(2.0) * (ME * KB * T)**1.5 / (np.pi**2 * HBAR**3)  # n_e = lam F_1/2(eta)
    eta = eta_of(ne / lam)
    f12, f32 = fermi(eta)
    per_e = KB * T * (eta - (2.0 / 3.0) * f32 / f12)
    EF = HBAR**2 * (3 * np.pi**2 * ne)**(2.0 / 3.0) / (2 * ME)
    ex = -0.75 * (3 / np.pi)**(1 / 3) * QE**2 / (4 * np.pi * EPS0) * ne**(1 / 3) / (1 + (KB * T / EF)**2)
    return z / (m["A"] * MU) * (per_e + ex) + _f_hydrogen(m, rho, T, z)


def _f_hydrogen(m, rho, T, x):
    """hydrogen (Z = 1): binding energy of the neutral fraction (13.6 eV, reduced as pressure ionization sets in)
    and the ion / neutral mixing entropy with the neutral spin weight, the terms that make Saha the minimizer"""
    if m["Z"] != 1:
        return 0.0
    x = np.clip(x, 1e-12, 1 - 1e-12)
    I = 13.6 * QE * (1 - np.clip(z_tf(1, m["A"], rho, 1e-6), 0, 1))
    return (KB * T * (x * np.log(x) + (1 - x) * np.log(1 - x) - (1 - x) * np.log(2.0)) - (1 - x) * I) / (m["A"] * MU)


def f_electron0(m, rho):
    """the T -> 0 limit of f_electron: (3/5) E_F + exchange per electron, ionization at T = 0"""
    z = np.clip(z_tf(m["Z"], m["A"], rho, 1e-6), 1e-6, m["Z"])
    ne = z * rho / (m["A"] * MU)
    EF = HBAR**2 * (3 * np.pi**2 * ne)**(2.0 / 3.0) / (2 * ME)
    ex = -0.75 * (3 / np.pi)**(1 / 3) * QE**2 / (4 * np.pi * EPS0) * ne**(1 / 3)
    return z / (m["A"] * MU) * (0.6 * EF + ex) + _f_hydrogen(m, rho, 0.0, z)


# ------------------------------------------------------------------ EOS

def _rho_cold(m):
    """zero-temperature reference density of the cold curve: the one that puts p(rho0, T_ref) at zero, rho0 being
    the measured density at T_ref (300 K; 10 K for solid D-T): the thermal pressure of a solid there is ~ a GPa"""
    if "rho_c" not in m:
        r0 = np.array([m["rho0"]]); T = np.array([m.get("Tref", 300.0)])
        lo, hi = m["rho0"], 1.5 * m["rho0"]
        m["rho_c"] = hi
        if eos(m, r0, T)[0][0] > 0:
            # no root: the classical ion heat capacity (no quantum freeze-out below theta_D) gives a light solid
            # (D-T at 10 K) more thermal pressure than its cold curve can hold in tension; keep rho0
            m["rho_c"] = m["rho0"]
            return m["rho_c"]
        for _ in range(60):
            m["rho_c"] = 0.5 * (lo + hi)
            p, _ = eos(m, r0, T)
            lo, hi = (m["rho_c"], hi) if p[0] > 0 else (lo, m["rho_c"])
        m["rho_c"] = 0.5 * (lo + hi)
    return m["rho_c"]


def calibrate(m):
    """materials with a measured bulk sound speed c0 (the Hugoniot intercept): scale K0 so that the adiabatic
    sound speed at (rho0, T_ref) equals c0, re-solving the cold reference density each time"""
    if m.get("hug") is None or m.get("_fit"):
        return
    m["_fit"] = True
    T = np.array([m.get("Tref", 300.0)])
    for _ in range(6):
        m.pop("rho_c", None)
        _rho_cold(m)
        _, _, c2 = thermo(m, np.array([m["rho0"]]), T)
        m["K0"] *= (m["hug"][0] * 1e3)**2 / c2[0]
    m.pop("rho_c", None)
    _rho_cold(m)


def cold(m, rho):
    """Vinet cold curve: (E_c per mass, p_c); energy zero at the cold reference density"""
    calibrate(m)
    rc = m.get("rho_c") or _rho_cold(m)
    x = (rc / rho)**(1.0 / 3.0)
    xi = 1.5 * (m["K1"] - 1.0)
    ex = np.exp(xi * (1 - x))
    E = 9 * m["K0"] / (rc * xi**2) * (1 - (1 - xi * (1 - x)) * ex)
    p = 3 * m["K0"] * (1 - x) / x**2 * ex
    return E, p


def ions(m, rho, T):
    """(e_ion, p_ion) of the solid-to-gas ion model"""
    mi = m["A"] * MU
    r = m["rho0"] / rho
    Tm = m["Tm"] * r**(2.0 / 3.0) * np.exp(np.clip(2 * m["G0"] * (1 - r), -700, 700))
    Gm = -2.0 / 3.0 + 2 * m["G0"] * r                             # d ln T_m / d ln rho
    e = KB * T / mi * (1.5 + 1.5 * Tm / (T + Tm))
    p = rho * KB * T / mi * (1.0 + 1.5 * Gm * Tm / (T + Tm))
    return e, p


def eos(m, rho, T, h=1e-5):
    """p (Pa), e (J/kg, zero at rho0 and T = 0), on broadcast arrays rho, T"""
    Ec, pc = cold(m, rho)
    ei, pi = ions(m, rho, T)
    fe = lambda r, t: f_electron(m, r, t) - f_electron0(m, r)
    # p_e = rho^2 dF_e/drho, e_e = F_e - T dF_e/dT (central differences in log rho and log T)
    dFr = (fe(rho * (1 + h), T) - fe(rho * (1 - h), T)) / (2 * h)            # rho dF/drho
    FT1, FT0 = fe(rho, T * (1 + h)), fe(rho, T * (1 - h))
    Fe = fe(rho, T)
    pe = rho * dFr
    ee = Fe - (FT1 - FT0) / (2 * h)
    return pc + pi + pe, Ec + ei + ee


def thermo(m, rho, T, h=1e-4):
    """p, e, c_s^2 and the derivatives used for c_s"""
    p, e = eos(m, rho, T)
    pr = (eos(m, rho * (1 + h), T)[0] - eos(m, rho * (1 - h), T)[0]) / (2 * h * rho)
    pT1, eT1 = eos(m, rho, T * (1 + h))
    pT0, eT0 = eos(m, rho, T * (1 - h))
    pT, eT = (pT1 - pT0) / (2 * h * T), (eT1 - eT0) / (2 * h * T)
    c2 = pr + T * pT**2 / (rho**2 * np.maximum(eT, 1e-300))
    return p, e, c2


# ------------------------------------------------------------------ opacities

def opacities(m, rho, T, nu_pts=300, band=None):
    """(kappa_R, kappa_P) in m^2/kg: frequency-integrated Rosseland and Planck means; band = (E_lo, E_hi) photon
    energies in eV restricts both means to that band (multigroup tables; E_hi may be inf)"""
    z = zbar(m, rho, T)
    ni = rho / (m["A"] * MU)
    ne = z * ni
    shp = np.shape(rho * T)
    if band is None:
        u = np.geomspace(1e-3, 60.0, nu_pts)                        # h nu / k T
        U = u.reshape((1,) * len(shp) + (-1,))
    else:                                            # per-cell u range of the band (log-spaced)
        kT = np.broadcast_to(T, shp)[..., None] / EV                # eV
        lo = np.maximum(band[0] / kT, 1e-4)
        hi = (band[1] / kT) if np.isfinite(band[1]) else lo + 60.0
        hi = np.maximum(hi, lo * (1 + 1e-9))
        s_ = np.linspace(0.0, 1.0, nu_pts)
        U = np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * s_)
        u = None
    Tc, rc, zc, nic, nec = (np.broadcast_to(a, shp)[..., None] for a in (T, rho, z, ni, ne))
    nu = U * KB * Tc / (2 * np.pi * HBAR)
    stim = -np.expm1(-U)
    # free-free, cgs: alpha = 3.692e8 (1 - e^-u) Z^2 n_e n_i T^-1/2 nu^-3 [cm^-1]
    a_ff = 3.692e8 * stim * zc**2 * (nec * 1e-6) * (nic * 1e-6) / np.sqrt(Tc) / nu**3 * 1e2     # m^-1
    # bound-free: the outermost occupied shell of a screened-hydrogenic average ion
    Nb = m["Z"] - zc
    shells = np.array([2, 8, 18, 32, 50, 72, 98])
    cum = np.cumsum(shells)
    n_out = 1 + np.sum(Nb > cum, axis=-1, keepdims=True)            # full shells below the outermost one
    filled = np.where(n_out > 1, cum[np.clip(n_out - 2, 0, len(cum) - 1)], 0)
    N_out = np.clip(Nb - filled, 0, None)
    Zeff = zc + 1.0
    I = 13.6 * QE * Zeff**2 / n_out**2                              # J
    nu_n = I / (2 * np.pi * HBAR)
    sig = 7.91e-22 * n_out / Zeff**2 * (nu_n / nu)**3 * (nu >= nu_n)  # m^2
    a_bf = nic * N_out * sig * stim
    a_abs = (a_ff + a_bf) / rc                                      # absorption per mass
    k_es = 6.6524587e-29 * nec / rc
    w_P = U**3 / np.expm1(U)                                        # Planck weight
    w_R = U**4 * np.exp(U) / np.expm1(U)**2                         # dB/dT weight
    X = u if u is not None else U
    integ = (lambda f: np.trapz(f, u, axis=-1)) if u is not None else (lambda f: np.trapz(f, X, axis=-1))
    kP = integ(a_abs * w_P) / np.maximum(integ(w_P), 1e-300)
    kR = integ(w_R) / np.maximum(integ(w_R / (a_abs + k_es)), 1e-300)
    return kR, kP


def group_table(name, edges_eV, nr=120, nt=110, pts=96):
    """multigroup opacity table: kappa_R_g, kappa_P_g (G, nr, nt) in m^2/kg for the photon-energy groups between
    edges_eV (first 0, last inf allowed), on the material's rho x T grid; with group_edges_eV for the solver"""
    m = MATERIALS[name]
    rho = m["rho0"] * np.geomspace(*m["rho"], nr)
    T = np.geomspace(*m["T"], nt)
    edges = np.asarray(edges_eV, float)
    G = len(edges) - 1
    kR, kP = np.zeros((G, nr, nt)), np.zeros((G, nr, nt))
    for g in range(G):
        for i in range(nr):
            kR[g, i], kP[g, i] = opacities(m, np.full(nt, rho[i]), T, nu_pts=pts, band=(edges[g], edges[g + 1]))
    return {"rho": rho, "T": T, "kappa_R_g": np.maximum(kR, 1e-300), "kappa_P_g": np.maximum(kP, 1e-300),
            "group_edges_eV": edges}


# ------------------------------------------------------------------ tables

def monotone(q, tol=1e-3):
    """q (nr, nt) made strictly increasing along T with the smallest upward changes (the tabular reader inverts
    e(T) and p(T)); returns (q, largest change relative to its row). Changes come from finite-difference noise
    in degenerate states (1e-9) and from the solid-vapour corner (cold curve in tension at low T and density);
    a change above tol means the model itself is wrong there and raises."""
    q = q.copy()
    scale = np.max(np.abs(q), axis=1)
    d = 1e-12 * scale
    worst = 0.0
    for j in range(1, q.shape[1]):
        low = q[:, j - 1] + d
        bump = np.maximum(low - q[:, j], 0.0)
        worst = max(worst, float(np.max(bump / np.maximum(scale, 1e-300))))
        q[:, j] = np.maximum(q[:, j], low)
    if worst > tol:
        raise ValueError(f"table not monotone in T beyond {tol:g} of a row ({worst:.2e})")
    return q, worst


def table(name, nr=300, nt=160, opacity=True):
    """SI table dict (rho, T, p, e, cs, kappa_R, kappa_P) of a material"""
    m = MATERIALS[name]
    rho = m["rho0"] * np.geomspace(*m["rho"], nr)
    T = np.geomspace(*m["T"], nt)
    R, TT = np.meshgrid(rho, T, indexing="ij")
    p, e, c2 = thermo(m, R, TT)
    adj = {}
    for k_, q in (("p", p), ("e", e)):                               # strictly increasing along T (see monotone)
        q[...], adj[k_] = monotone(q)
    out = {"rho": rho, "T": T, "p": p, "e": e, "cs": np.sqrt(np.maximum(c2, 1e-6)), "adjust_max": max(adj.values())}
    if opacity:
        kR, kP = np.zeros_like(R), np.zeros_like(R)
        for i in range(nr):                                          # row by row: memory
            kR[i], kP[i] = opacities(m, R[i], TT[i])
        out["kappa_R"], out["kappa_P"] = kR, kP
    return out


def hugoniot(name, up_kms, T0=None, n=240):
    """(Us km/s, p GPa) on the principal Hugoniot from (rho0, T_ref) at particle velocities up (km/s): the
    energy jump e - e0 = (p + p0)(1/rho0 - 1/rho)/2 is solved along T at each density of a grid (one vectorised EOS
    evaluation), then Us and up follow from mass and momentum and are interpolated at the requested up"""
    m = MATERIALS[name]
    T0 = m.get("Tref", 300.0) if T0 is None else T0
    rho0 = m["rho0"]
    p0, e0 = eos(m, np.array([rho0]), np.array([T0]))
    rho = rho0 * np.linspace(1.002, 3.2, n)
    T = np.geomspace(T0, 3e6, n)
    R, TT = np.meshgrid(rho, T, indexing="ij")
    p, e = eos(m, R, TT)
    H = e - e0[0] - 0.5 * (p + p0[0]) * (1 / rho0 - 1 / R)
    Us, up = [], []
    for i in range(n):                                   # H increases with T at fixed rho: first sign change
        k = np.nonzero(H[i] > 0)[0]
        if len(k) == 0 or k[0] == 0:
            continue
        j = k[0]
        w = -H[i, j - 1] / (H[i, j] - H[i, j - 1])
        pH = (1 - w) * p[i, j - 1] + w * p[i, j]
        mu = 1 - rho0 / rho[i]
        u_s = np.sqrt(max(pH - p0[0], 0.0) / (rho0 * mu))
        Us.append(u_s); up.append(u_s * mu)
    Us, up = np.array(Us), np.array(up)
    q = np.atleast_1d(up_kms) * 1e3
    us_q = np.interp(q, up, Us)
    return np.stack([us_q / 1e3, rho0 * us_q * q / 1e9], axis=1)


def write(name, tab, out_dir, sesame=False):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez(path, **tab)
    if sesame:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from aegis import tabular
        nr, nt = len(tab["rho"]), len(tab["T"])
        w301 = np.concatenate([[nr, nt], tab["rho"] / 1e3, tab["T"], tab["p"].T.ravel() / 1e9,
                               tab["e"].T.ravel() / 1e6, np.zeros(nr * nt)])
        lk = lambda k: np.concatenate([[nr, nt], np.log10(tab["rho"] / 1e3), np.log10(tab["T"] / EV),
                                       np.log10(np.maximum(k, 1e-300).T.ravel() / 0.1)])
        tables = {301: w301}
        if "kappa_R" in tab:
            tables.update({502: lk(tab["kappa_R"]), 505: lk(tab["kappa_P"])})
        tabular.write_sesame(os.path.join(out_dir, f"{name}.ses"), MATERIALS[name]["id"], tables)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--materials", nargs="+", default=list(MATERIALS), choices=list(MATERIALS))
    ap.add_argument("--nr", type=int, default=300, help="densities (the solid branch near rho0 needs ~300 over the "
                    "default 5.5 decades: the reader resamples onto a uniform log grid)")
    ap.add_argument("--nt", type=int, default=160)
    ap.add_argument("--no-opacity", action="store_true")
    ap.add_argument("--sesame", action="store_true", help="also write SESAME ASCII (301, 502, 505)")
    ap.add_argument("--groups", type=float, nargs="+", default=None,
                    help="also write <material>_groups.npz with multigroup opacities: inner group edges in eV "
                         "(the first group starts at 0, the last runs to infinity)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables"))
    a = ap.parse_args()
    for name in a.materials:
        tab = table(name, a.nr, a.nt, not a.no_opacity)
        path = write(name, tab, a.out, a.sesame)
        m = MATERIALS[name]
        p0, e0 = eos(m, np.array([m["rho0"]]), np.array([300.0]))
        if a.groups:
            gt = group_table(name, [0.0] + sorted(a.groups) + [np.inf])
            np.savez(os.path.join(a.out, f"{name}_groups.npz"), **gt)
        print(f"{name}: {path} ({a.nr} x {a.nt}); p(rho0, 300 K) = {p0[0] / 1e9:+.3f} GPa, "
              f"Z*(rho0, 1 eV) = {zbar(m, m['rho0'], EV):.2f}")


if __name__ == "__main__":
    main()
