# eos.py - two-material equation of state: ideal gas, stiffened gas, Mie-Gruneisen (linear Us-Up).
#
# Every material is written in "Mie-Gruneisen form": internal energy per volume
#     eps = phi1(rho) * p + phi2(rho)
#   ideal       phi1 = 1/(g-1)          phi2 = 0
#   stiffened   phi1 = 1/(g-1)          phi2 = g p_inf/(g-1)       ->  p = (g-1) eps - g p_inf
#   MG          phi1 = rho/(G0 rho0)    phi2 = -rho f(rho)/(G0 rho0),  f = p_H (1 - G0 eta/2)
# (MG with rho Gamma = rho0 Gamma0 and Hugoniot p_H = rho0 c0^2 eta/(1 - s eta)^2, eta = 1 - rho0/rho, for
# compression; in expansion (eta < 0) the reference curve is linear, f = rho0 c0^2 eta, which joins with a
# continuous slope and keeps the cavitation pressure near -rho0 c0^2/G0 instead of letting it vanish.)
# A mixed cell uses phi = (1 - a) phi_fuel + a phi_liner with a the tracer. When a is a volume fraction this
# is the isobaric mixture rule and, for constant phis (ideal, stiffened), keeps uniform p, u uniform
# (Abgrall 1996, Shyue 1998). A Mie-Gruneisen liner is therefore replaced in mixed cells by its tangent
# stiffened gas at rho0 and used in full only where a > 0.98 (see EOS.coeffs).
# The sound speed follows from the phis:
#     c^2 = ((eps + p)/rho - phi2' - p phi1') / phi1
# which gives g (p + p_inf)/rho for a stiffened gas and f' + G0 rho0 p/rho^2 for MG.
import numpy as np

from .config import FLOOR_PR


class Material:
    def __init__(self, kind, gamma=1.4, p_inf=0.0, rho0=0.0, c0=0.0, s=0.0, gamma0=0.0):
        self.kind, self.gamma, self.p_inf = kind, gamma, p_inf
        self.rho0, self.c0, self.s, self.gamma0 = rho0, c0, s, gamma0

    def phis(self, rho, xp):
        """(phi1, phi2, dphi1/drho, dphi2/drho); floats for the constant-coefficient laws."""
        if self.kind in ("ideal", "stiffened"):
            g = self.gamma
            return 1.0 / (g - 1), g * self.p_inf / (g - 1), 0.0, 0.0
        r0, c0, s, G0 = self.rho0, self.c0, self.s, self.gamma0
        K = G0 * r0
        # keep away from the Hugoniot singularity at eta = 1/s (rho = s rho0/(s-1))
        eta = xp.minimum(1.0 - r0 / rho, 0.9 / s)
        ec = xp.maximum(eta, 0.0)                     # compression branch only
        pH = r0 * c0**2 * ec / (1 - s * ec)**2
        dpH = r0 * c0**2 * (1 + s * ec) / (1 - s * ec)**3
        comp = eta > 0
        f = xp.where(comp, pH * (1 - 0.5 * G0 * ec), r0 * c0**2 * eta)
        df = xp.where(comp, dpH * (1 - 0.5 * G0 * ec) - 0.5 * G0 * pH, r0 * c0**2) * r0 / rho**2
        # specific-energy zero shifted by c0^2/G0 so that at rho0 eps(p) equals that of the tangent
        # stiffened gas (tangent_stiffened); only energy differences enter the dynamics
        e_off = c0**2 / G0
        return rho / K, -rho * f / K + rho * e_off, 1.0 / K, -(f + rho * df) / K + e_off

    def tangent_stiffened(self):
        """stiffened gas matching MG at rho0: gamma = 1 + G0 (same Gruneisen), p_inf = rho0 c0^2/gamma (same c0)"""
        g = 1.0 + self.gamma0
        return Material("stiffened", g, self.rho0 * self.c0**2 / g)

    def describe(self):
        if self.kind == "ideal":
            return f"ideal gamma={self.gamma:g}"
        if self.kind == "stiffened":
            return f"stiffened gamma={self.gamma:g} p_inf={self.p_inf:.4g}"
        return f"Mie-Gruneisen rho0={self.rho0:g} c0={self.c0:g} s={self.s:g} Gamma0={self.gamma0:g}"


def _dev(eos, mode, rho, y, a):
    """the same quantity from the device EOS (eos_fused.cu) in one kernel launch, or None"""
    from .eos_fused import evaluate
    return evaluate(eos, mode, rho, y, a)


class EOS:
    """Two-material EOS. Arguments a are the tracer (liner fraction) on the same grid as rho. On device
    arrays the methods below run as one fused kernel each (eos_fused.cu) with the same arithmetic."""

    def __init__(self, fuel, liner, volume_fraction=True):
        self.fuel, self.liner = fuel, liner
        self.volume_fraction = volume_fraction

    def coeffs(self, rho, a):
        """mixture (phi1, phi2, dphi1/drho, dphi2/drho) at fixed composition. The liner term is
        a phi_l(rho/a), whose rho-derivative is phi_l'(rho/a): no factor a on the derivatives."""
        from .physics import xp
        a = xp.clip(a, 0.0, 1.0)
        f1, f2, fd1, fd2 = self.fuel.phis(rho, xp)
        if self.liner.kind != "mie_gruneisen":
            l1, l2, ld1, ld2 = self.liner.phis(rho, xp)   # constant coefficients: density not needed
            return ((1 - a) * f1 + a * l1, (1 - a) * f2 + a * l2, (1 - a) * fd1 + a * ld1, (1 - a) * fd2 + a * ld2)
        # Mixed cells: the MG phis depend on density, which breaks the pressure-equilibrium property of the
        # volume-fraction mixture (the cold energy drifts as a and rho evolve differently). So the liner in
        # a mixed cell is MG's tangent stiffened gas (constant phis, same eps(p) at rho0), switching to full
        # MG over a = 0.98 ... 1, where rho/a is the liner density. d(a phi_l(rho/a))/drho = phi_l'(rho/a).
        s1, s2, _, _ = self.liner.tangent_stiffened().phis(rho, xp)
        w = xp.clip((a - 0.98) / 0.02, 0.0, 1.0)
        w = w * w * (3 - 2 * w)
        r0 = self.liner.rho0
        rho_l = xp.clip(rho / xp.maximum(a, 1e-12), 0.2 * r0, r0 / (1 - 0.9 / self.liner.s))
        m1, m2, md1, md2 = self.liner.phis(rho_l, xp)
        l1, l2 = (1 - w) * s1 + w * m1, (1 - w) * s2 + w * m2
        return ((1 - a) * f1 + a * l1, (1 - a) * f2 + a * l2, (1 - a) * fd1 + w * md1, (1 - a) * fd2 + w * md2)

    def eps(self, rho, p, a):
        r = _dev(self, "eps", rho, p, a)
        if r is not None:
            return r
        p1, p2, _, _ = self.coeffs(rho, a)
        return p1 * p + p2

    def pressure(self, rho, eps, a):
        r = _dev(self, "pressure", rho, eps, a)
        if r is not None:
            return r
        p1, p2, _, _ = self.coeffs(rho, a)
        return (eps - p2) / p1

    def c2(self, rho, p, a):
        r = _dev(self, "c2", rho, p, a)
        if r is not None:
            return r
        p1, p2, d1, d2 = self.coeffs(rho, a)
        return ((p1 * p + p2 + p) / rho - d2 - p * d1) / p1

    def p_min(self, rho, a):
        """pressure at which c^2 = 0 (cavitation): -p_inf for a stiffened gas, 0 for an ideal gas"""
        r = _dev(self, "p_min", rho, None, a)
        if r is not None:
            return r
        p1, p2, d1, d2 = self.coeffs(rho, a)
        return -(p2 - rho * d2) / (1 + p1 - rho * d1)

    def p_floor(self, rho, a):
        r = _dev(self, "p_floor", rho, None, a)
        if r is not None:
            return r
        pm = self.p_min(rho, a)
        from .physics import xp
        return pm + FLOOR_PR + 1e-12 * xp.abs(pm)

    def describe(self):
        mode = "volume fraction" if self.volume_fraction else "mass fraction"
        return f"fuel: {self.fuel.describe()}; liner: {self.liner.describe()}; tracer as {mode}"


def make_eos(cfg):
    """cfg.gamma (a float, the old single-material path) unless more than one material is configured."""
    if not cfg.is_multimaterial() and cfg.material_tracer != "volume_fraction":
        return cfg.gamma
    fuel = Material(cfg.eos_fuel, cfg.gamma, cfg.p_inf)
    gl = cfg.gamma if cfg.gamma_liner is None else cfg.gamma_liner
    liner = Material(cfg.eos_liner, gl, cfg.p_inf_liner, cfg.mg_rho0, cfg.mg_c0, cfg.mg_s, cfg.mg_gamma0)
    eos = EOS(fuel, liner, cfg.material_tracer != "mass_fraction")
    if cfg.eos_table:                                # tabulated liner, the analytic liner as its fallback
        from . import tabular
        data = tabular.load(cfg.eos_table, cfg.eos_table_material or None)
        eos = tabular.TabularEOS(eos, tabular.TabularMaterial(data, cfg.tab_rho_unit, cfg.tab_v_unit, liner))
    from .eos_fused import device
    device(eos)                                      # pack the device EOS now, not inside a CUDA-graph capture
    return eos


def stiffened_from_hugoniot(rho0, c0, s, up_ref):
    """Stiffened-gas (gamma, p_inf) matching c0 at rest and the linear Hugoniot Us = c0 + s up at up_ref.
    The stiffened-gas Hugoniot is Us = k up + sqrt((k up)^2 + c0^2), k = (gamma + 1)/4."""
    us = c0 + s * up_ref
    k = (us**2 - c0**2) / (2 * us * up_ref)
    g = 4 * k - 1
    return g, rho0 * c0**2 / g


def stiffened_hugoniot(rho0, p0, gamma, p_inf, up):
    """Shock state for particle-velocity jump up into a stiffened gas at rest (rho0, p0): (p, rho, Us).
    Ideal-gas jump relations in P = p + p_inf."""
    c0 = np.sqrt(gamma * (p0 + p_inf) / rho0)
    k = (gamma + 1) / 4
    us = k * up + np.sqrt((k * up)**2 + c0**2)
    return p0 + rho0 * us * up, rho0 * us / (us - up), us
