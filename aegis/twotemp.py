# twotemp.py - two-temperature plasma (T_e != T_i) for an ideal-gas fuel.
#
# The electron entropy s_e = p_e / rho^gamma is carried as a passive scalar (conserved rho s_e, advected by the
# HLLD mass flux like the tracer): electrons follow adiabatic compression exactly and shock dissipation goes to
# the ions, the usual single-fluid three-temperature closure (as in FLASH). The ions hold p_i = p - p_e.
# Code temperatures share the scale of theta = p/rho at T_e = T_i (Z = ion charge, n_e = Z n_i):
#     theta_e = (1 + Z)/Z * p_e/rho,   theta_i = (1 + Z) * p_i/rho
# Electron processes (conduction, bremsstrahlung, Ohmic heating, Ettingshausen) change p_e through set_pe;
# electron-ion equilibration relaxes theta_e - theta_i at the rate nu (1 + Z), nu = c_eq rho / theta_e^1.5
# (nu = 2 m_e / (m_i tau_e)), conserving the total thermal energy.
from .config import RHO, MX, MY, MZ, BX, BY, BZ, EN, RHOSE, FLOOR_RHO
from .physics import xp


def _thermal(U):
    rho = xp.maximum(U[RHO], FLOOR_RHO)
    eps = U[EN] - 0.5 * (U[MX]**2 + U[MY]**2 + U[MZ]**2) / rho - 0.5 * (U[BX]**2 + U[BY]**2 + U[BZ]**2)
    return rho, eps


def pressures(U, gamma):
    """(rho, p, p_e) with p_e clipped to [0, p]"""
    rho, eps = _thermal(U)
    p = xp.maximum((gamma - 1) * eps, 0.0)
    pe = xp.clip(U[RHOSE] / rho * rho**gamma, 0.0, p)
    return rho, p, pe


def temperatures(U, gamma, Z):
    """(theta_e, theta_i)"""
    rho, p, pe = pressures(U, gamma)
    return (1 + Z) / Z * pe / rho, (1 + Z) * (p - pe) / rho


def set_pe(U, pe, gamma):
    """write the electron pressure back as rho s_e"""
    rho = xp.maximum(U[RHO], FLOOR_RHO)
    U[RHOSE] = xp.maximum(pe, 0.0) * rho**(1 - gamma)


def init_equal(U, gamma, Z):
    """electron entropy for T_e = T_i: p_e = Z/(1 + Z) p"""
    rho, p, _ = pressures(U, gamma)
    set_pe(U, Z / (1 + Z) * p, gamma)


def equilibrate(U, gamma, Z, c_eq, fuel, dt):
    """exact relaxation of theta_e - theta_i over dt with nu frozen at the start; total thermal energy kept"""
    rho, p, pe = pressures(U, gamma)
    te, ti = (1 + Z) / Z * pe / rho, (1 + Z) * (p - pe) / rho
    nu = c_eq * rho / xp.maximum(te, 1e-30)**1.5
    d = (te - ti) * xp.exp(-nu * (1 + Z) * dt)
    # theta = (Z theta_e + theta_i)/(1 + Z) is fixed by the total p; new theta_e = theta + d/(1 + Z)
    theta = p / rho
    te_new = theta + d / (1 + Z)
    set_pe(U, xp.where(fuel, Z / (1 + Z) * rho * te_new, pe), gamma)
