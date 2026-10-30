# circuit.py - RLC drive circuit coupled to a cylindrical (r, z) liner: the bank's current runs axially up the
# liner and back through a return can at r_return, and its B_theta pushes on the liner's outer surface.
#
#   d(L I)/dt = Q/C - R I,  dQ/dt = -I,  L = L0 + L_load(t),  L_load = mu0 l / (2 pi) <ln(r_return / r_out(z))>
#
# which is dI/dt = (V - R I - I dL/dt) / L. The force on the liner is the J x B of its surface current,
# f = P_B(r) dC/dr on the outer side of the liner (C the liner fraction), P_B = mu0 I^2 / (8 pi^2 r^2): it
# integrates to P_B per unit area across the smeared surface, and the vacuum B_theta outside is force-free.
import numpy as np

from .config import MX, EN, iRHO, iCLR
from .physics import xp, cons_to_prim
from .units import MU0


class RLCCircuit:
    """capacitor bank C charged to V0, series inductance L0 and resistance R; SI throughout.

    advance() takes one step of the flux-form trapezoid rule, L1 I1 - L0 I0 = dt (V_mid - R I_mid),
    Q1 = Q0 - dt I_mid, closed form since it is linear in I1. For a fixed load it conserves
    Q^2/2C + L I^2/2 + int R I^2 dt exactly; with a moving load, the energy that leaves the circuit
    beyond the resistive loss and the change in L I^2/2 is the mechanical work on the liner (work_mech)."""

    def __init__(self, C, V0, L0, R):
        self.C, self.L0, self.R = float(C), float(L0), float(R)
        self.Q = self.C * float(V0)
        self.I = 0.0
        self.L = None                                # total inductance at the last step
        self.E0 = 0.5 * self.C * V0**2
        self.ohmic = 0.0
        self.work_mech = 0.0

    @property
    def V(self):
        return self.Q / self.C

    def energy_stored(self):
        return 0.5 * self.Q**2 / self.C + 0.5 * self.L * self.I**2

    def advance(self, dt, L_new=None, work=None):
        """one step to L_new; or, given work (J) done on the load this step, to the L for which the step's
        mechanical work, which for this scheme is exactly (L1 - L0) I0 I1 / 2, equals it"""
        if self.L is None:
            self.L = L_new
        L0, I0, Q0, C, R = self.L, self.I, self.Q, self.C, self.R
        a = dt * dt / (4 * C) + 0.5 * dt * R
        N = L0 * I0 + dt * Q0 / C - a * I0
        if work is not None and I0 != 0.0:
            L_new = L0 + 2 * work * (L0 + a) / (I0 * N - 2 * work)
        I1 = N / (L_new + a)
        Im = 0.5 * (I0 + I1)
        Q1 = Q0 - dt * Im
        E_before = 0.5 * Q0**2 / C + 0.5 * L0 * I0**2
        self.ohmic += R * Im * Im * dt
        self.I, self.Q, self.L = I1, Q1, L_new
        # whatever the bank gave up that is neither in the field, nor in R, went into the liner
        self.work_mech += E_before - (0.5 * Q1**2 / C + 0.5 * L_new * I1**2) - R * Im * Im * dt
        return I1

    def budget_error(self):
        """(E_bank0 - stored - ohmic - mechanical) / E_bank0: zero to round-off by construction"""
        return (self.E0 - self.energy_stored() - self.ohmic - self.work_mech) / self.E0


class LinerDrive:
    """couples an RLCCircuit to a cylindrical MHDSolver whose liner is marked by the volume-fraction tracer.

    Code units are set by L_unit (m), rho_unit (kg/m^3), v_unit (m/s); the solver's domain length in z is the
    load length. Call step(dt) right after each solver.step_ssprk3(): it kicks the liner's outer surface with
    the current of the step start (exact kinetic-energy update, so work_grid is the energy the grid received),
    then advances the circuit so that its mechanical work equals that energy: bank + grid conserve energy to
    round-off. The load inductance this implies (L_energy) is checked against the coaxial inductance of the
    liner's actual outer radius (L_geom); they differ by the smeared surface's discretisation error."""

    def __init__(self, solver, circuit, r_return_m, L_unit, rho_unit, v_unit):
        self.s, self.c = solver, circuit
        cfg = solver.cfg
        self.Lu, self.pu = L_unit, rho_unit * v_unit**2
        self.tu = L_unit / v_unit
        self.Eu = self.pu * L_unit**3 * 2 * np.pi        # code energy (per radian) -> joules for the full circle
        self.length = (cfg.y_max - cfg.y_min) * L_unit
        self.r_return = float(r_return_m)
        ng = solver.ng
        self.r = solver.rc[ng:ng + cfg.nx]
        self.rn = xp.arange(cfg.nx)[:, None]
        self.work_grid = 0.0                         # code units, per radian (as the solver's energy budget)
        self.P_B = 0.0
        self.r_out = self.outer_radius()
        if self.r_out >= self.r_return / L_unit:
            raise ValueError("the return conductor must sit outside the liner")
        self.L_geom = self.inductance()
        circuit.L = circuit.L0 + self.L_geom

    def _liner(self):
        ng, cfg = self.s.ng, self.s.cfg
        return xp.clip(cons_to_prim(self.s.U, self.s.eos)[iCLR, ng:ng + cfg.nx, ng:ng + cfg.ny], 0.0, 1.0)

    def outer_radius(self, C=None):
        """outer liner radius per z row (code units): gas area inside the liner plus the liner area"""
        C = self._liner() if C is None else C
        dr = self.s.cfg.dx
        rr = self.r[:, None]
        inner = xp.arange(C.shape[0])[:, None] < xp.argmax(C, axis=0)[None, :]
        r_in2 = xp.sum(xp.where(inner, (1 - C) * 2 * rr, 0.0), axis=0) * dr
        self.r_out_z = xp.sqrt(r_in2 + xp.sum(C * 2 * rr, axis=0) * dr)
        return float(xp.mean(self.r_out_z))

    def inductance(self):
        """load inductance (H): coaxial gap from the liner's outer surface to the return can, averaged in z"""
        return MU0 * self.length / (2 * np.pi) * float(xp.mean(xp.log(self.r_return / (self.r_out_z * self.Lu))))

    def pressure_gpa(self, r_m=None):
        """drive pressure at radius r (m), default the mean outer liner radius"""
        r_m = self.r_out * self.Lu if r_m is None else r_m
        return MU0 * self.c.I**2 / (8 * np.pi**2 * r_m**2) / 1e9

    def step(self, dt):
        s, cfg, ng = self.s, self.s.cfg, self.s.ng
        I = self.c.I
        if I != 0.0:
            U = s.U[:, ng:ng + cfg.nx, ng:ng + cfg.ny]
            C = self._liner()
            m = xp.argmax(C, axis=0)[None, :]
            Cm = xp.take_along_axis(C, m, axis=0)
            # -dC/dr on the outer side, split so it sums to exactly the peak fraction: (C[i-1] - C[i+1]) / 2
            # beyond the peak, (C[m] - C[m+1]) / 2 at it; normalised by the peak, so the force per row is
            # P_B x area even when the smeared liner has lost its flat top
            Cp = xp.concatenate([C[:1], C, xp.zeros_like(C[:1])], axis=0)
            g = xp.where(self.rn > m, 0.5 * (Cp[:-2] - Cp[2:]), 0.0)
            g = g + xp.where(self.rn == m, 0.5 * (Cp[1:-1] - Cp[2:]), 0.0)
            # f = -P_B(r) dC/dr: the surface current's J x B, inward
            P = MU0 * I * I / (8 * np.pi**2 * (self.r[:, None] * self.Lu)**2) / self.pu
            f = -P * g / (xp.maximum(Cm, 1e-12) * cfg.dx)
            rho = U[iRHO]
            dm = f * dt
            dE = dm * (U[MX] + 0.5 * dm) / rho          # exact change of kinetic energy, internal energy kept
            U[MX] += dm
            U[EN] += dE
            dW = float(xp.sum(dE * s.dV))
            self.work_grid += dW
            s.apply_bc(s.U)
        else:
            dW = 0.0
        self.r_out = self.outer_radius()
        self.L_geom = self.inductance()
        if I != 0.0:
            self.c.advance(dt * self.tu, work=dW * self.Eu)
        else:
            self.c.advance(dt * self.tu, self.c.L0 + self.L_geom)
        self.P_B = self.pressure_gpa()

    @property
    def L_energy(self):
        """load inductance implied by the work done on the liner (H)"""
        return self.c.L - self.c.L0

    def work_mismatch(self):
        """|work the grid received - work the circuit delivered| / bank energy: round-off by construction"""
        return abs(self.work_grid * self.Eu - self.c.work_mech) / self.c.E0


def thick_shell(C, V0, L0, R, r_out0, r_in0, rho, length, r_return, t_end, n=20000):
    """reference: an incompressible thick liner on the same circuit, pushed on its outer radius x. With
    v ~ 1/r across the wall its kinetic energy is M(x) x'^2 / 2, M = 2 pi rho l x^2 ln(x / r_in), r_in^2 =
    x^2 - (r_out0^2 - r_in0^2), so M x'' + M'(x) x'^2 / 2 = -mu0 I^2 l / (4 pi x). RK4 in SI; returns arrays
    t, r_out, I. Valid while the liner stays incompressible (drive pressure well below its bulk modulus)."""
    A = r_out0**2 - r_in0**2
    k = MU0 * length / (2 * np.pi)

    def rhs(y):
        x, u, Q, I = y
        ri2 = x * x - A
        lg = 0.5 * np.log(x * x / ri2)
        M = 2 * np.pi * rho * length * x * x * lg
        dM = 2 * np.pi * rho * length * (2 * x * lg + x - x**3 / ri2)
        L = L0 + k * np.log(r_return / x)
        return np.array([u, (-MU0 * I * I * length / (4 * np.pi * x) - 0.5 * dM * u * u) / M, -I,
                         (Q / C - R * I + I * k * u / x) / L])

    y = np.array([r_out0, 0.0, C * V0, 0.0])
    dt = t_end / n
    out = np.empty((n + 1, 4)); out[0] = y
    for i in range(n):
        k1 = rhs(y); k2 = rhs(y + 0.5 * dt * k1); k3 = rhs(y + 0.5 * dt * k2); k4 = rhs(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = y
    return np.linspace(0, t_end, n + 1), out[:, 0], out[:, 3]


def add_circuit_args(ap):
    g = ap.add_argument_group("RLC drive circuit (off unless --capacitance-uf and --voltage-kv are set)")
    g.add_argument("--capacitance-uf", dest="capacitance_uf", type=float, default=0.0, help="bank capacitance, uF")
    g.add_argument("--voltage-kv", dest="voltage_kv", type=float, default=0.0, help="charge voltage, kV")
    g.add_argument("--inductance-nh", dest="inductance_nh", type=float, default=20.0,
                   help="series inductance of bank and feed, nH")
    g.add_argument("--resistance-mohm", dest="resistance_mohm", type=float, default=1.0, help="series resistance, mOhm")
    g.add_argument("--r-return-mm", dest="r_return_mm", type=float, default=None,
                   help="return-current can radius, mm (default: the outer domain edge)")


def circuit_from_args(a):
    """RLCCircuit from the command line, or None when no bank is given"""
    if a.capacitance_uf <= 0 or a.voltage_kv <= 0:
        return None
    return RLCCircuit(a.capacitance_uf * 1e-6, a.voltage_kv * 1e3, a.inductance_nh * 1e-9, a.resistance_mohm * 1e-3)
