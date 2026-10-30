# solver3d.py - 3D extension of MHDSolver: unsplit MUSCL-HLLD in x, y, z with CT or GLM + Powell,
# conservative resistive terms, Nernst/Ettingshausen transport (CT) and liner strength,
# the same driven-shock boundaries (z periodic) and the pressure-balanced stripe set-up.
import io
import time
import contextlib

import numpy as np

from .config import (
    NVAR, FLOOR_RHO, FLOOR_PR,
    RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI, RHOC, RHOSE,
    iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iPSI, iCLR,
)
from .physics import (xp, cons_to_prim, prim_to_cons, hlld_flux_x, parallel_stripe_state,
                      resistivity_field, face_eta, eta_max_of, sound_speed2, radiate, vf_advection_rate,
                      conduct, _floor_faces)
from .eos import make_eos
from . import ct
from . import twotemp
from . import transport
from . import strength as stren
from . import fld
from . import radiation_multigroup
from .viscosity import viscous_step
from .solver import MHDSolver, to_numpy

# variable permutations that rotate the y or z face normal onto x for the HLLD kernel
_PERM = {1: list(range(NVAR)), 2: list(range(NVAR)), 3: list(range(NVAR))}
_PERM[2][iVX], _PERM[2][iVY] = iVY, iVX
_PERM[2][iBX], _PERM[2][iBY] = iBY, iBX
_PERM[3][iVX], _PERM[3][iVZ] = iVZ, iVX
_PERM[3][iBX], _PERM[3][iBZ] = iBZ, iBX


def _sl(ax, s):
    idx = [slice(None)] * 4
    idx[ax] = s
    return tuple(idx)


def muscl(W, ax, eos=None):
    """van Leer MUSCL along array axis ax (1, 2, 3); face k lies between cells k+1 and k+2."""
    from .physics import _FUSED, _ideal
    if _FUSED:
        from . import fused
        e = 1.4 if eos is None else eos
        WL, WR = fused.muscl(W, ax - 1, floor_p=_ideal(e))
        if not _ideal(e):
            _floor_faces(WL, WR, e)
        return WL, WR
    dm = W[_sl(ax, slice(1, -1))] - W[_sl(ax, slice(None, -2))]
    dp = W[_sl(ax, slice(2, None))] - W[_sl(ax, slice(1, -1))]
    ab = dm * dp
    d = xp.where(ab > 0, 2.0 * ab / (dm + dp + 1e-30), 0.0)
    WL = W[_sl(ax, slice(1, -2))] + 0.5 * d[_sl(ax, slice(None, -1))]
    WR = W[_sl(ax, slice(2, -1))] - 0.5 * d[_sl(ax, slice(1, None))]
    _floor_faces(WL, WR, 1.4 if eos is None else eos)
    return WL, WR


def hlld_flux(WL, WR, ax, gamma, ch):
    """HLLD flux through faces normal to axis ax, via variable rotation onto x."""
    shape = WL.shape
    F, _ = hlld_flux_x(xp.ascontiguousarray(WL).reshape(NVAR, -1), xp.ascontiguousarray(WR).reshape(NVAR, -1),
                       gamma, ch, axn=ax - 1)
    return F.reshape(shape)


def _levi(i, j, k):
    return 1.0 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0


def resistive_terms_3d(U, eta, h, ng, n, face_mean="harmonic"):
    """Resistive induction and energy terms as differences of face fluxes.

    Induction -curl(eta J) is the divergence of F_j(B_i) = eps_ijk eta J_k; energy uses the
    resistive Poynting flux eta (J x B). J on a face: compact difference along the face normal,
    averaged centred differences across it (reduces to the 2D form when d/dz = 0). eta is a float
    or a cell-centred array, averaged onto faces with face_eta (harmonic by default).
    Returns (R over all cells, net resistive energy inflow through the domain faces).
    """
    B = [U[BX], U[BY], U[BZ]]
    D = [[None] * 3 for _ in range(3)]          # cell-centred dB_c/dx_d, zero on the outer ring
    for c in range(3):
        for d in range(3):
            a = xp.zeros_like(B[c])
            mid = [slice(None)] * 3; mid[d] = slice(1, -1)
            up = [slice(None)] * 3; up[d] = slice(2, None)
            dn = [slice(None)] * 3; dn[d] = slice(None, -2)
            a[tuple(mid)] = (B[c][tuple(up)] - B[c][tuple(dn)]) / (2 * h[d])
            D[c][d] = a
    R = xp.zeros_like(U)
    rate = xp.zeros(())
    inner = tuple(slice(ng, ng + m) for m in n)
    for ax in range(3):
        lo = [slice(None)] * 3; lo[ax] = slice(None, -1); lo = tuple(lo)
        hi = [slice(None)] * 3; hi[ax] = slice(1, None); hi = tuple(hi)

        def dface(c, d):
            if d == ax:
                return (B[c][hi] - B[c][lo]) / h[ax]
            return 0.5 * (D[c][d][lo] + D[c][d][hi])

        J = (dface(2, 1) - dface(1, 2), dface(0, 2) - dface(2, 0), dface(1, 0) - dface(0, 1))
        ef = face_eta(eta, ax, face_mean)
        Bf = [0.5 * (B[c][lo] + B[c][hi]) for c in range(3)]
        # face index k lies between cells k and k+1: interior cells use F[c] - F[c-1]
        a_hi = list(inner); a_hi[ax] = slice(ng, ng + n[ax]); a_hi = tuple(a_hi)
        a_lo = list(inner); a_lo[ax] = slice(ng - 1, ng + n[ax] - 1); a_lo = tuple(a_lo)
        for i in range(3):
            if i == ax:
                continue
            k = 3 - i - ax
            F = _levi(i, ax, k) * ef * J[k]
            R[(BX + i,) + inner] -= (F[a_hi] - F[a_lo]) / h[ax]
        G = ef * (J[(ax + 1) % 3] * Bf[(ax + 2) % 3] - J[(ax + 2) % 3] * Bf[(ax + 1) % 3])
        R[(EN,) + inner] -= (G[a_hi] - G[a_lo]) / h[ax]
        f_lo = list(inner); f_lo[ax] = ng - 1
        f_hi = list(inner); f_hi[ax] = ng + n[ax] - 1
        area = np.prod([h[j] for j in range(3) if j != ax])
        rate = rate + xp.sum(G[tuple(f_lo)] - G[tuple(f_hi)]) * area
    return R, rate


def _curl(ax_, ay_, az_, dx, dy, dz):
    """curl of a cell-centred vector on interior arrays (central differences)."""
    g = lambda f, h, a: xp.gradient(f, h, axis=a)
    cx = g(az_, dy, 1) - g(ay_, dz, 2)
    cy = g(ax_, dz, 2) - g(az_, dx, 0)
    cz = g(ay_, dx, 0) - g(ax_, dy, 1)
    return cx, cy, cz


class MHDSolver3D:
    """3D MHD with HLLD fluxes on all three face orientations (unsplit, SSP-RK3), optional constant eta.

    geometry="cylindrical": (x, y, z) = (r, theta, z), scale factors (1, r, 1). Cells are r dr dtheta dz; the radial
    flux divergence is (1/r) d(r F)/dr, the azimuthal one dF/(r dtheta), with the momentum sources
    (rho v_theta^2 - B_theta^2 + p_T)/r (radial) and -(rho v_r v_theta - B_r B_theta)/r (azimuthal). The CT faces
    (B_r on r-faces, B_theta on theta-faces, B_z on z-faces) follow Stokes on the curved cell faces:
        dB_r/dt = -dE_z/(r dtheta) + dE_theta/dz,  dB_theta/dt = -dE_r/dz + dE_z/dr,
        dB_z/dt = -(1/r) d(r E_theta)/dr + dE_r/(r dtheta),
    so the cylindrical face divergence stays at round-off. With x_min = 0 the domain holds the axis: theta must
    span 2 pi with an even cell count, and the ghost cells behind the axis are the cells at theta + pi (radial and
    azimuthal components reversed). Without theta variation this reproduces the 2D r-z solver to round-off.
    Ideal MHD with the EOS options and the volume-fraction tracer; the other 3D physics modules are Cartesian."""

    def __init__(self, config):
        cfg = config
        if cfg.nz < 4:
            raise ValueError("solver3d needs nz >= 4")
        self.walls = None                            # immersed rigid walls (aegis/cad_importer.ImmersedWalls)
        self.cyl = cfg.geometry == "cylindrical"
        if self.cyl:
            if cfg.div_b != "ct" or cfg.x_min < 0:
                raise ValueError("3D cylindrical geometry needs div_b='ct' and x_min >= 0")
            if (cfg.is_resistive() or cfg.nernst_coeff > 0 or cfg.strength or cfg.cond_kappa0 > 0 or cfg.fld
                    or cfg.visc_coeff > 0 or cfg.rad_coeff > 0 or cfg.two_temp):
                raise NotImplementedError("3D cylindrical geometry: ideal MHD (with the EOS options) only")
            self.axis = cfg.x_min == 0
            if self.axis and (abs(cfg.y_max - cfg.y_min - 2 * np.pi) > 1e-9 or cfg.ny % 2):
                raise ValueError("3D cylindrical with the axis (x_min = 0) needs theta over 2 pi, ny even")
        if cfg.hall_di > 0 or cfg.eta_perp_ratio != 1.0 or cfg.gyro_di > 0 or cfg.cond_kappa_i0 > 0:
            raise NotImplementedError("the Hall term, anisotropic resistivity and ion transport are in the 2D solver")
        self.cfg = cfg
        self.strength = cfg.strength
        # float gamma for one ideal gas, eos.EOS for two materials (volume-fraction tracer)
        self.eos = make_eos(cfg)
        self._vf = not isinstance(self.eos, (int, float)) and self.eos.volume_fraction
        # constrained transport: Bf = [Bx on x-faces, By on y-faces, Bz on z-faces], built from U on first use
        self.ct = cfg.div_b == "ct"
        self.Bf = None
        self._faces_owner = None
        self.ng = ng = 2
        self.nx_tot, self.ny_tot, self.nz_tot = cfg.nx + 2 * ng, cfg.ny + 2 * ng, cfg.nz + 2 * ng
        self.x = xp.asarray(cfg.x_min + (np.arange(self.nx_tot) - ng + 0.5) * cfg.dx)
        self.y = xp.asarray(cfg.y_min + (np.arange(self.ny_tot) - ng + 0.5) * cfg.dy)
        self.z = xp.asarray(cfg.z_min + (np.arange(self.nz_tot) - ng + 0.5) * cfg.dz)
        if self.cyl:                                 # face and cell radii, interior cell volumes r dr dtheta dz
            self.rf = xp.asarray(cfg.x_min + (np.arange(self.nx_tot + 1) - ng) * cfg.dx)
            self.rc = self.x
            self.dV = (self.rc[ng:ng + cfg.nx] * cfg.dx * cfg.dy * cfg.dz)[:, None, None]
        shape = (NVAR, self.nx_tot, self.ny_tot, self.nz_tot)
        self.U = xp.zeros(shape)
        self.U_temp = xp.empty(shape)
        self.U_stage = xp.empty(shape)
        self.dU = xp.empty(shape)
        if cfg.strength:                             # S_xx, S_yy, S_xy, damage, S_xz, S_yz
            self.S = xp.zeros((6,) + shape[1:])
            self._S1 = xp.zeros_like(self.S)
            self._dS = xp.zeros_like(self.S)
        self.t = 0.0
        self.step = 0
        self.dt = 0.0
        self._ch = 0.0
        self._inflow_U = None
        self._inflow_R = None
        self._sigma = None          # sponge off until initialize() sets it up (hand-built states skip that)
        self._acc_boundary = xp.zeros(())
        self._acc_powell = xp.zeros(())
        self._acc_rad = xp.zeros(())
        self.Er = None if not cfg.fld else xp.zeros(((cfg.fld_groups,) if cfg.fld_groups else ()) + (cfg.nx, cfg.ny, cfg.nz))
        self.fld_inflow = 0.0
        self._rate_boundary = xp.zeros(())
        self._rate_powell = xp.zeros(())
        self.diag = {}
        self.snapshots = {}

    # ------------------------------------------------------------ initial condition

    def initialize(self):
        """2D pressure-balanced set-up extruded in z, plus optional 3D seeding."""
        cfg = self.cfg
        if cfg.is_striped() and cfg.B_stripe_component != "x":
            raise ValueError("solver3d supports hydro or x stripes (pressure-balanced)")
        if cfg.is_mhd() and not cfg.is_striped():
            raise ValueError("solver3d supports hydro or x stripes (pressure-balanced)")
        s2 = MHDSolver(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            s2.initialize()
        W2 = cons_to_prim(s2.U, self.eos)
        W = xp.repeat(W2[:, :, :, None], self.nz_tot, axis=3)
        self.equilibrium = getattr(s2, "_equilibrium", None)
        self._bar_mask_y, self._gap_mask_y = s2._bar_mask_y, s2._gap_mask_y

        rng = np.random.default_rng(cfg.z_pert_seed)
        X = self.x[:, None, None]
        Y = self.y[None, :, None]
        Z = self.z[None, None, :]
        Ly, Lz = cfg.y_max - cfg.y_min, cfg.z_max - cfg.z_min

        if cfg.z_pert_amp > 0:
            # interface ripple in z: rebuild density and tracer ahead of the shock
            ky = 2 * np.pi * cfg.perturbation_mode / Ly
            kz = 2 * np.pi * cfg.z_pert_mode / Lz
            x_if = cfg.interface_x + cfg.perturbation_amp * xp.sin(ky * Y) + cfg.z_pert_amp * xp.sin(kz * Z)
            phi = 0.5 * (1.0 + xp.tanh((X - x_if) / max(cfg.interface_width * cfg.dx, 1e-10)))
            ahead = (X >= cfg.interface_x - 0.3) * xp.ones_like(phi)
            if cfg.is_striped():
                st = parallel_stripe_state(cfg.gamma, cfg.mach, cfg.p_bar, cfg.equilibrium_closure,
                                           W2[iBX, 0, :])
                rho_l = st["rho1"][None, :, None]
            else:
                rho_l = xp.ones((1, self.ny_tot, 1))
            rho_up = rho_l * (1 - phi) + cfg.density_ratio * rho_l * phi
            W[iRHO] = xp.where(ahead > 0, rho_up, W[iRHO])
            W[iCLR] = xp.where(ahead > 0, phi, W[iCLR])

        self._vz_pattern = None
        if cfg.z_pert_vel > 0:
            cs_bar = np.sqrt(cfg.gamma * cfg.p_bar)
            amp = cfg.z_pert_vel * cs_bar
            W[iVZ] = W[iVZ] + amp * xp.asarray(self._smooth_noise(rng, (self.nx_tot, self.ny_tot, self.nz_tot)))
            self._vz_pattern = amp * xp.asarray(self._smooth_noise(rng, (self.ng, self.ny_tot, self.nz_tot)))

        if cfg.frame_velocity != 0.0:
            W[iVX] = W[iVX] - cfg.frame_velocity      # Galilean shift to the interface frame
        self.U = prim_to_cons(W, self.eos)
        self._inflow_U = self.U[:, :self.ng, :, :].copy()
        # in the shifted frame the upstream gas enters at x_max (supersonic for U > c_s upstream)
        right_inflow = cfg.x_max_bc in ("inflow", "plateau") or (cfg.x_max_bc == "auto" and cfg.frame_velocity > 0)
        if cfg.x_max_bc not in ("auto", "inflow", "transmissive", "plateau"):
            raise ValueError(f"x_max_bc must be auto, inflow, transmissive or plateau, got {cfg.x_max_bc!r}")
        self._plateau_captured = False
        self._p_upstream = float(xp.mean(W[iPR, -self.ng - 1]))
        if cfg.x_min_bc not in ("inflow", "plateau"):
            raise ValueError(f"x_min_bc must be inflow or plateau, got {cfg.x_min_bc!r}")
        self._left_captured = False
        self._p_inflow = float(xp.mean(W[iPR, self.ng]))
        self._inflow_R = self.U[:, -self.ng:, :, :].copy() if right_inflow else None
        self._setup_sponge()
        if self._vz_pattern is not None:
            Win = cons_to_prim(self._inflow_U, self.eos)
            Win[iVZ] = self._vz_pattern
            self._inflow_U = prim_to_cons(Win, self.eos)
        self.t, self.step = 0.0, 0
        self._acc_boundary = xp.zeros(())
        self._acc_powell = xp.zeros(())
        self._acc_rad = xp.zeros(())

    def _update_plateau(self):
        """x_max_bc = "plateau": upstream inflow until the transmitted shock reaches x_max, then hold the
        shocked state found just inside the domain (the exterior behind the departing shock).
        x_min_bc = "plateau": the same at x_min for the reflected shock."""
        ng = self.ng
        if self.cfg.x_min_bc == "plateau" and not self._left_captured:
            p_edge = float(xp.mean(cons_to_prim(self.U[:, ng:ng + 1], self.eos)[iPR]))
            if p_edge > 1.3 * self._p_inflow:
                self._inflow_U = xp.repeat(self.U[:, ng + 9:ng + 10].copy(), ng, axis=1)
                self._left_captured = True
                self.left_plateau_time = self.t
        if self.cfg.x_max_bc != "plateau" or self._plateau_captured:
            return
        p_edge = float(xp.mean(cons_to_prim(self.U[:, -ng - 1:-ng], self.eos)[iPR]))
        if p_edge > 3.0 * self._p_upstream:
            plateau = self.U[:, -ng - 10:-ng - 9].copy()       # 10 cells behind the boundary
            self._inflow_R = xp.repeat(plateau, ng, axis=1)
            self._plateau_captured = True
            self.plateau_time = self.t

    def _setup_sponge(self):
        """sigma(x): zero in the interior, rising quadratically to sponge_rate at each x end."""
        cfg = self.cfg
        self._sigma = None
        if cfg.sponge_width <= 0 or cfg.sponge_rate <= 0:
            return
        x = self.x
        d_left = (cfg.x_min + cfg.sponge_width - x) / cfg.sponge_width
        d_right = (x - (cfg.x_max - cfg.sponge_width)) / cfg.sponge_width
        depth = xp.clip(xp.maximum(d_left, d_right), 0.0, 1.0)
        self._sigma = (cfg.sponge_rate * depth**2)[None, :, None, None]
        self._U_mean = self.U.copy()
        self._acc_sponge = xp.zeros(())

    def _apply_sponge(self, dt):
        """relax toward a running mean (time constant sponge_tau); energy removed is booked in the budget"""
        if self._sigma is None:
            return
        a = min(dt / self.cfg.sponge_tau, 1.0)
        self._U_mean += a * (self.U - self._U_mean)
        self.U[...] = self._U_mean + (self.U - self._U_mean) * xp.exp(-self._sigma * dt)
        ng = self.ng
        I = (slice(ng, -ng),) * 3
        dV = self.cfg.dx * self.cfg.dy * self.cfg.dz
        # only interior cells enter the energy diagnostic
        self._acc_sponge += (xp.sum(self.U[EN][I]) - xp.sum(self._U_before_sponge[EN][I])) * dV

    @staticmethod
    def _smooth_noise(rng, shape, kmax=6):
        """unit-rms random field keeping the lowest kmax modes in every direction"""
        a = rng.standard_normal(shape)
        f = np.fft.rfftn(a)
        for ax, n in enumerate(shape):
            k = np.abs(np.fft.fftfreq(n) * n) if ax < len(shape) - 1 else np.fft.rfftfreq(n) * n
            mask = (k <= kmax).reshape([-1 if i == ax else 1 for i in range(len(shape))])
            f = f * mask
        a = np.fft.irfftn(f, s=shape)
        return a / max(a.std(), 1e-30)

    # ------------------------------------------------------------ boundaries and time step

    def _behind_axis(self, A, sign_rows=(), shift_faces=False):
        """ghost rows i < ng of a (var, x, theta, z) array from the rows across the axis, theta + pi: row ng-1-k
        takes row ng+k (cells; faces: row ng-k takes ng+k, the axis face itself zero), components in sign_rows
        reversed. Only interior theta entries; the periodic theta fill follows."""
        ng, nth = self.ng, self.cfg.ny
        half = nth // 2
        th = slice(ng, ng + nth)
        pairs = [(ng - k, ng + k) for k in range(1, ng + 1)] if shift_faces else \
            [(ng - 1 - k, ng + k) for k in range(ng)]
        for dst, src in pairs:
            A[:, dst, th] = xp.roll(A[:, src, th], -half, axis=1)
            for v in sign_rows:
                A[v, dst, th] = -A[v, dst, th]

    def apply_bc(self, U):
        ng = self.ng
        if self.cyl:
            for i in range(ng):                          # outer radius: zero gradient
                U[:, -(i + 1)] = U[:, -(ng + 1)]
            if self.axis:
                self._behind_axis(U, (MX, MY, BX, BY))
            else:
                for i in range(ng):
                    U[:, i] = U[:, ng]
        elif self.cfg.get_bc_x() == "periodic":
            U[:, :ng] = U[:, -2 * ng:-ng]
            U[:, -ng:] = U[:, ng:2 * ng]
        elif self.cfg.get_bc_x() == "extrapolation":
            for i in range(ng):                          # zero gradient at both x ends
                U[:, i] = U[:, ng]
                U[:, -(i + 1)] = U[:, -(ng + 1)]
        else:
            U[:, :ng] = self._inflow_U                   # driven inflow at x_min
            if self._inflow_R is not None:
                U[:, -ng:] = self._inflow_R              # upstream inflow at x_max (co-moving frame)
            else:
                for i in range(ng):
                    U[:, -(i + 1)] = U[:, -(ng + 1)]     # transmissive outflow
        U[:, :, :ng] = U[:, :, -2 * ng:-ng]              # periodic y
        U[:, :, -ng:] = U[:, :, ng:2 * ng]
        U[:, :, :, :ng] = U[:, :, :, -2 * ng:-ng]        # periodic z
        U[:, :, :, -ng:] = U[:, :, :, ng:2 * ng]
        return U

    def compute_dt(self, W):
        cfg = self.cfg
        rho = xp.maximum(W[iRHO], FLOOR_RHO)
        c2 = sound_speed2(W, self.eos)
        if self.strength:                            # longitudinal elastic waves: c^2 + 4 G / (3 rho)
            c2 = c2 + 4.0 * self._shear(W) / (3.0 * rho)
        cf = xp.sqrt(c2 + (W[iBX]**2 + W[iBY]**2 + W[iBZ]**2) / rho)
        sx, sy, sz = xp.abs(W[iVX]) + cf, xp.abs(W[iVY]) + cf, xp.abs(W[iVZ]) + cf
        self._ch = max(float(xp.max(xp.maximum(xp.maximum(sx, sy), sz))), 1e-12)
        h3 = (cfg.dx, cfg.dy, cfg.dz)
        h = min(h3)
        hy = cfg.dy * xp.maximum(self.rc, 0.5 * cfg.dx)[:, None, None] if self.cyl else cfg.dy
        if self.cyl and cfg.cyl_filter:              # filtered rings resolve only m <= m_crit: spacing pi r / m_crit
            hy = xp.maximum(hy, xp.asarray(self._az_setup()["h_cell"])[:, None, None])
        inv = max(float(xp.max(xp.maximum(xp.maximum(sx / cfg.dx, sy / hy), sz / cfg.dz))), self._ch / h)
        dt = cfg.cfl / inv
        if cfg.is_resistive():
            # explicit diffusion: dt <= 1 / (2 eta_max (1/dx^2 + 1/dy^2 + 1/dz^2))
            eta_max = max(eta_max_of(resistivity_field(W, cfg)), 1e-300)
            dt = min(dt, 0.5 / (eta_max * (cfg.dx**-2 + cfg.dy**-2 + cfg.dz**-2)))
        if cfg.nernst_coeff > 0:                     # Nernst advection of B
            v, _, _, _ = transport.nernst_velocity_3d(W, cfg.nernst_coeff, cfg.cond_x0, cfg.cond_tracer_max, h3)
            vN = float(xp.max(xp.maximum(xp.maximum(xp.abs(v[0]) / cfg.dx, xp.abs(v[1]) / cfg.dy),
                                         xp.abs(v[2]) / cfg.dz)))
            if vN > 0:
                dt = min(dt, cfg.cfl / vN)
        return dt

    # ---------------- near-axis azimuthal Fourier filter (3D cylindrical) ----------------
    # On a ring of radius r the azimuthal spacing r dtheta shrinks toward the axis and sets the explicit time step.
    # With cyl_filter, rings with r dtheta below cyl_filter_frac dr keep only the azimuthal modes
    # m <= m_crit(r) = max(cyl_filter_mkeep, floor(pi r / (cyl_filter_frac dr))) in their tendencies: the RK rates
    # of the cell variables and the edge EMFs. Their shortest resolved half-wavelength is then pi r / m_crit, which
    # the time step uses in place of r dtheta (about dr, as in r-z). Mode m = 0 is untouched, so ring sums (mass,
    # momentum, energy on equal-volume cells) are unchanged, and B still follows from -curl E, so div B stays at
    # round-off; the low modes m <= cyl_filter_mkeep are never filtered.
    def _az_setup(self):
        if getattr(self, "_az", None) is not None:
            return self._az
        cfg = self.cfg
        nth, dr = cfg.ny, cfg.dx
        get = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)

        def mcrit(r):
            m = np.floor(np.pi * np.maximum(r, 0.0) / (cfg.cyl_filter_frac * dr)).astype(int)
            return np.clip(np.maximum(m, cfg.cyl_filter_mkeep), 0, nth // 2)
        rc, rf = get(self.rc), get(self.rf)
        mc, mf = mcrit(rc), mcrit(rf)
        k = np.arange(nth // 2 + 1)
        az = dict(mc=mc, mf=mf, rows_c=np.nonzero(mc < nth // 2)[0], rows_f=np.nonzero(mf < nth // 2)[0],
                  mask_c=xp.asarray((k[None, :] <= mc[:, None]).astype(float)),
                  mask_f=xp.asarray((k[None, :] <= mf[:, None]).astype(float)),
                  h_cell=np.where(mc < nth // 2, np.pi * np.maximum(rc, 0.5 * dr) / np.maximum(mc, 1), 0.0))
        self._az = az
        return az

    def _az_apply(self, A, rows, mask, th):
        """keep modes <= m_crit along axis 1 of A[rows] over the nth entries th (a slice), in place"""
        if len(rows) == 0:
            return
        nth = self.cfg.ny
        r = xp.asarray(rows)
        sub = A[r][:, th]
        f = xp.fft.rfft(sub, axis=1) * mask[r][:, :, None]
        A[r, th] = xp.fft.irfft(f, n=nth, axis=1)

    def _az_filter_rates(self, out):
        az, ng, cfg = self._az_setup(), self.ng, self.cfg
        rows = [i for i in az["rows_c"] if ng <= i < ng + cfg.nx]
        th = slice(ng, ng + cfg.ny)
        for v in range(out.shape[0]):
            if BX <= v <= BZ and self.ct:
                continue
            self._az_apply(out[v], rows, az["mask_c"], th)

    def _az_filter_emf(self, E):
        """E[0] (E_r): radial cells x theta faces; E[1] (E_theta): radial faces x theta cells; E[2] (E_z): radial
        faces x theta faces (theta face nth repeats face 0)"""
        az, ng, nth = self._az_setup(), self.ng, self.cfg.ny
        self._az_apply(E[0], list(az["rows_c"]), az["mask_c"], slice(0, nth))
        E[0][:, nth] = E[0][:, 0]
        rows_f = [i - ng for i in az["rows_f"] if ng <= i <= ng + self.cfg.nx]   # face index k = rf[ng + k]
        mask_f = az["mask_f"][ng:]
        self._az_apply(E[2], rows_f, mask_f, slice(0, nth))
        E[2][:, nth] = E[2][:, 0]
        self._az_apply(E[1], rows_f, mask_f, slice(ng, ng + nth))

    def _shear(self, W, S=None):
        S = self.S if S is None else S
        return self.cfg.shear_modulus * xp.clip(W[iCLR], 0, 1) * (1 - S[stren.DMG])

    def _periodic(self):
        return (self.cfg.get_bc_x() == "periodic", True, True)

    # hooks a domain-decomposed subclass (aegis/parallel.py) overrides: ghost faces, edge EMFs, extra arrays
    def _ghost_faces(self, Bf, U):
        ct.ghost_faces(Bf, U, self.ng, (self.cfg.nx, self.cfg.ny, self.cfg.nz), self._periodic())
        if self.cyl and self.axis:                   # B_r odd (zero on the axis face), B_theta odd, B_z even
            ng, nth = self.ng, self.cfg.ny
            Bf[0][ng] = 0.0
            self._behind_axis(Bf[0][None], (0,), shift_faces=True)
            Bt = Bf[1]
            self._behind_axis(Bt[None], (0,))       # theta-faces of the ghost rows (interior faces ng..ng+nth-1)
            self._behind_axis(Bf[2][None], ())
            for A, nf in ((Bf[0], 0), (Bt, 1), (Bf[2], 0)):  # periodic theta for the rows just filled
                rows = slice(0, ng + 1) if A is Bf[0] else slice(0, ng)
                if nf:
                    A[rows, ng + nth] = A[rows, ng]
                    A[rows, :ng] = A[rows, nth:nth + ng]
                    A[rows, ng + nth + 1:] = A[rows, ng + 1:2 * ng + 1]
                else:
                    A[rows, :ng] = A[rows, nth:nth + ng]
                    A[rows, ng + nth:] = A[rows, ng:2 * ng]

    def _emf_hook(self, E):
        return E

    def _halo(self, A):
        pass

    # ------------------------------------------------------------ right-hand side

    def _ct_sync(self):
        """face fields from self.U if it was (re)assigned since they were made (total energy unchanged)"""
        if not self.ct or self._faces_owner is self.U:
            return
        U = self.U
        if self.cyl:
            self.apply_bc(U)
        self.Bf = ct.faces_from_cells(U, 3)
        if self.cyl and self.axis:
            self.Bf[0][self.ng] = 0.0                    # B_r = 0 on the axis
        self._cells_from_faces(U, self.Bf, (slice(None),) * 3)      # E kept, so the budget is unaffected
        self._faces_owner = U
        self._Bf1 = [xp.empty_like(f) for f in self.Bf]
        self._dBf = [xp.empty_like(f) for f in self.Bf]

    def _cells_from_faces(self, U, Bf, region):
        """cell B from the faces (averages); in a cylindrical axis cell the inner r-face has no area and carries
        B_r = 0, which is right for the axisymmetric part (B_r ~ r) but not for m >= 1 (B_r finite on the axis):
        there the cell takes half the theta-mean of the outer-face B_r plus all of its theta variation"""
        ct.cells_from_faces(U, Bf, region)
        if self.cyl and self.axis:
            ng, nth = self.ng, self.cfg.ny
            th = slice(ng, ng + nth)
            Bo = Bf[0][ng + 1, th]
            U[BX, ng, th] = Bo - 0.5 * xp.mean(Bo, axis=0, keepdims=True)

    def set_vector_potential(self, Ax, Ay, Az):
        """Set B from A = (Ax, Ay, Az) (numpy callables, each evaluated on its cell edges): faces are its
        discrete curl, so div B = 0 to round-off; cell B and the magnetic energy in E follow, pressure kept.
        Call after setting U."""
        cfg, ng = self.cfg, self.ng
        fc = lambda lo, h, N: lo + (np.arange(N + 1) - ng) * h
        cf = (fc(cfg.x_min, cfg.dx, self.nx_tot), fc(cfg.y_min, cfg.dy, self.ny_tot), fc(cfg.z_min, cfg.dz, self.nz_tot))
        Bf = ct.faces_from_potential((Ax, Ay, Az), (to_numpy(self.x), to_numpy(self.y), to_numpy(self.z)), cf)
        U = self.U
        B2 = U[BX]**2 + U[BY]**2 + U[BZ]**2
        if self.ct:
            self.Bf = Bf
            self._faces_owner = U
            self._Bf1 = [xp.empty_like(f) for f in Bf]
            self._dBf = [xp.empty_like(f) for f in Bf]
        self._cells_from_faces(U, Bf, (slice(None),) * 3)
        U[EN] += 0.5 * (U[BX]**2 + U[BY]**2 + U[BZ]**2 - B2)

    def div_b_max(self):
        """max |div B| h / max |B| on interior cells (face divergence with CT)"""
        cfg, ng = self.cfg, self.ng
        n, h = (cfg.nx, cfg.ny, cfg.nz), (cfg.dx, cfg.dy, cfg.dz)
        if self.ct and self.cyl:
            self._ct_sync()
            Br, Bt, Bz = self.Bf
            I = [slice(ng, ng + m) for m in n]
            rf = self.rf[ng:ng + n[0] + 1][:, None, None]
            rc = self.rc[ng:ng + n[0]][:, None, None]
            rB = Br[ng:ng + n[0] + 1, I[1], I[2]] * rf
            d = ((rB[1:] - rB[:-1]) / (rc * h[0]) + (Bt[I[0], ng + 1:ng + n[1] + 1, I[2]] - Bt[I[0], ng:ng + n[1], I[2]])
                 / (rc * h[1]) + (Bz[I[0], I[1], ng + 1:ng + n[2] + 1] - Bz[I[0], I[1], ng:ng + n[2]]) / h[2])
            h = (h[0], float(self.rc[ng]) * h[1], h[2])      # smallest physical spacing for the normalisation
        elif self.ct:
            self._ct_sync()
            d = ct.div_faces(self.Bf, h, ng, n)
        else:
            W = cons_to_prim(self.U, self.eos)
            d = ((W[iBX][2:, 1:-1, 1:-1] - W[iBX][:-2, 1:-1, 1:-1]) / (2 * cfg.dx)
                 + (W[iBY][1:-1, 2:, 1:-1] - W[iBY][1:-1, :-2, 1:-1]) / (2 * cfg.dy)
                 + (W[iBZ][1:-1, 1:-1, 2:] - W[iBZ][1:-1, 1:-1, :-2]) / (2 * cfg.dz))[(slice(ng - 1, -ng + 1),) * 3]
        Bm = float(xp.max(xp.sqrt(self.U[BX]**2 + self.U[BY]**2 + self.U[BZ]**2)))
        return float(xp.max(xp.abs(d))) * min(h) / max(Bm, 1e-300)

    def _ct_face_rates(self, Fs, W, Bf, dBf, eta, vN=None):
        """dB/dt = -curl E on the faces, E on the edges from edge_emf (plus eta J, plus the Nernst EMF)"""
        cfg, ng = self.cfg, self.ng
        n, h = (cfg.nx, cfg.ny, cfg.nz), (cfg.dx, cfg.dy, cfg.dz)
        E = []
        for c in range(3):
            a, b = (c + 1) % 3, (c + 2) % 3
            order = (a, b, c)
            perm = (0,) + tuple(1 + k for k in order)
            Ec = ct.edge_emf(Fs[a].transpose(perm), Fs[b].transpose(perm), W.transpose(perm), a, b,
                             h[a], h[b], n[a], n[b])
            if eta is not None:
                Bp = {a: Bf[a].transpose(order), b: Bf[b].transpose(order)}
                J = ct.edge_current(Bp, a, b, h[a], h[b], n[a], n[b])
                e = eta if isinstance(eta, (int, float)) else eta.transpose(order)
                Ec = Ec + ct.edge_eta(e, n[a], n[b]) * J
            if vN is not None:
                Ec = Ec + transport.nernst_edge(vN[a].transpose(order), vN[b].transpose(order),
                                                Bf[a].transpose(order), Bf[b].transpose(order), n[a], n[b])
            E.append(Ec.transpose(tuple(int(k) for k in np.argsort(order))))   # back to (x, y, z)
        E = self._emf_hook(E)
        if self.cyl and cfg.cyl_filter:              # near-axis azimuthal filter of the edge EMFs (div B kept)
            self._az_filter_emf(E)
        if self.cyl and self.axis:
            E[1][0] = 0.0                            # E_theta = 0 on the axis
            # E_z on the axis is one value: the theta average of the edge EMFs there (otherwise the axis cells'
            # divergence changes at the rate d E_z(axis) / dtheta)
            nth = cfg.ny
            E[2][0] = xp.mean(E[2][0, :nth], axis=0, keepdims=True)
        for f in range(3):
            p, q = (f + 1) % 3, (f + 2) % 3
            # dB_f = -(d_p E_q - d_q E_p); E_q: faces on f and p, cells on q; E_p: faces on q and f, cells on p
            Eq, Ep = E[q], E[p]
            sq = [slice(None)] * 3
            sq[q] = slice(ng, ng + n[q])
            hi, lo = list(sq), list(sq)
            hi[p], lo[p] = slice(1, None), slice(0, -1)
            hi_p, lo_p = hi, lo
            dEq = (Eq[tuple(hi)] - Eq[tuple(lo)]) / h[p]
            sp = [slice(None)] * 3
            sp[p] = slice(ng, ng + n[p])
            hi, lo = list(sp), list(sp)
            hi[q], lo[q] = slice(1, None), slice(0, -1)
            dEp = (Ep[tuple(hi)] - Ep[tuple(lo)]) / h[q]
            if self.cyl:                             # scale factors: r on theta lengths, r-weighted E_theta
                rf = self.rf[ng:ng + n[0] + 1][:, None, None]
                rc = self.rc[ng:ng + n[0]][:, None, None]
                if f == 0:                           # r-faces: dE_z / (r dtheta) (zero area on the axis face)
                    dEq = dEq / xp.where(rf > 0, rf, xp.inf)
                elif f == 2:                         # z-faces: (1/r) d(r E_theta)/dr - dE_r/(r dtheta)
                    rEq = Eq * self.rf[ng:ng + n[0] + 1][:, None, None]
                    dEq = (rEq[tuple(hi_p)] - rEq[tuple(lo_p)]) / (h[p] * rc)
                    dEp = dEp / rc
            dBf[f][...] = 0.0
            tgt = [slice(ng, ng + m) for m in n]
            tgt[f] = slice(ng, ng + n[f] + 1)
            dBf[f][tuple(tgt)] = -(dEq - dEp)

    def compute_rhs(self, U, out, Bf=None, dBf=None, S=None, dS=None):
        cfg = self.cfg
        ng, g, ch = self.ng, self.eos, self._ch
        n = (cfg.nx, cfg.ny, cfg.nz)
        h = (cfg.dx, cfg.dy, cfg.dz)
        if self.ct:
            ch = 0.0
            self._cells_from_faces(U, Bf, tuple(slice(ng, ng + m) for m in n))
        U = self.apply_bc(U)
        if self.ct:
            self._ghost_faces(Bf, U)
        W = cons_to_prim(U, g)
        if self.strength:
            stren.bc3d(S, ng, n, self._periodic())
            self._halo(S)
            spalled = (S[stren.DMG] > 0.5) & (W[iCLR] > 0.5)       # spalled liner carries no tension
            W[iPR] = xp.where(spalled, xp.maximum(W[iPR], 0.0), W[iPR])
        Fs = [None] * 3
        WLs, WRs = [None] * 3, [None] * 3
        out[...] = 0.0
        inner = (slice(None),) + tuple(slice(ng, ng + m) for m in n)
        rate = xp.zeros(())
        if self._vf:
            vf_rate = xp.zeros(n)
        for ax in (1, 2, 3):
            WL, WR = muscl(W, ax, g)
            if self.ct:                              # normal field on a face is the face value
                N = W.shape[ax]
                sel = [slice(None)] * 3
                sel[ax - 1] = slice(2, N - 1)
                WL[iBX + ax - 1] = WR[iBX + ax - 1] = Bf[ax - 1][tuple(sel)]
            F = hlld_flux(WL, WR, ax, g, ch)
            if self.walls is not None:               # immersed rigid walls (aegis/cad_importer.ImmersedWalls)
                self.walls.apply(F, WL, WR, ax - 1)
            Fs[ax - 1] = F
            WLs[ax - 1], WRs[ax - 1] = WL, WR
            m = n[ax - 1]
            if self._vf:
                # volume fraction: da/dt + u.grad(a) = 0 from the same face values as the energy flux
                a_c = W[iCLR][tuple(slice(ng, ng + m) if d == ax - 1 else slice(None) for d in range(3))]
                if self.cyl and ax == 1:
                    rv = vf_advection_rate(F, WL, WR, a_c, 0, h[0], self.rf[ng:ng + m + 1][:, None, None],
                                           self.rc[ng:ng + m][:, None, None])
                elif self.cyl and ax == 2:
                    rv = vf_advection_rate(F, WL, WR, a_c, 1, h[1] * self.rc[:, None, None])
                else:
                    rv = vf_advection_rate(F, WL, WR, a_c, ax - 1, h[ax - 1])
                vf_rate += rv[tuple(slice(None) if d == ax - 1 else slice(ng, ng + n[d]) for d in range(3))]
            sel = list(inner)
            sel[ax] = slice(None)
            if self.cyl and ax == 1:                 # (1/r) d(r F)/dr
                rF = F * self.rf[ng:ng + m + 1][None, :, None, None]
                dF = (rF[:, 1:1 + m] - rF[:, 0:m]) / self.rc[ng:ng + m][None, :, None, None]
                out[inner] -= dF[tuple(sel)] / h[0]
            elif self.cyl and ax == 2:               # dF / (r dtheta)
                dF = F[_sl(ax, slice(1, 1 + m))] - F[_sl(ax, slice(0, m))]
                out[inner] -= dF[tuple(sel)] / (h[1] * self.rc[ng:ng + n[0]][None, :, None, None])
            else:
                dF = F[_sl(ax, slice(1, 1 + m))] - F[_sl(ax, slice(0, m))]
                out[inner] -= dF[tuple(sel)] / h[ax - 1]
            # net energy inflow through the two domain faces normal to ax
            lo = list(sel); lo[0] = EN; lo[ax] = 0
            hi = list(sel); hi[0] = EN; hi[ax] = m
            if self.cyl:                             # face areas r_f dtheta dz, dr dz, r_c dr dtheta
                if ax == 1:
                    rate = rate + (xp.sum(F[tuple(lo)]) * self.rf[ng] - xp.sum(F[tuple(hi)]) * self.rf[ng + m]) * h[1] * h[2]
                elif ax == 2:
                    rate = rate + xp.sum(F[tuple(lo)] - F[tuple(hi)]) * h[0] * h[2]
                else:
                    rc_ = self.rc[ng:ng + n[0]][:, None]
                    rate = rate + xp.sum((F[tuple(lo)] - F[tuple(hi)]) * rc_) * h[0] * h[1]
            else:
                face_area = np.prod([h[i] for i in range(3) if i != ax - 1])
                rate = rate + xp.sum(F[tuple(lo)] - F[tuple(hi)]) * face_area
        if self._vf:
            out[(RHOC,) + inner[1:]] = vf_rate
        if self.cyl:                                 # geometric sources (x = r, y = theta)
            Wi = W[inner]
            ri = self.rc[ng:ng + n[0]][:, None, None]
            pT = Wi[iPR] + 0.5 * (Wi[iBX]**2 + Wi[iBY]**2 + Wi[iBZ]**2)
            out[(MX,) + inner[1:]] += (Wi[iRHO] * Wi[iVY]**2 - Wi[iBY]**2 + pT) / ri
            out[(MY,) + inner[1:]] -= (Wi[iRHO] * Wi[iVX] * Wi[iVY] - Wi[iBX] * Wi[iBY]) / ri
        self._rate_boundary = rate
        self._rate_powell = xp.zeros(())
        if cfg.powell_source and cfg.is_mhd() and not self.ct:
            divB = xp.zeros_like(W[iBX])
            c = (slice(1, -1),) * 3
            divB[c] = ((W[iBX][2:, 1:-1, 1:-1] - W[iBX][:-2, 1:-1, 1:-1]) / (2 * cfg.dx)
                       + (W[iBY][1:-1, 2:, 1:-1] - W[iBY][1:-1, :-2, 1:-1]) / (2 * cfg.dy)
                       + (W[iBZ][1:-1, 1:-1, 2:] - W[iBZ][1:-1, 1:-1, :-2]) / (2 * cfg.dz))
            vB = W[iVX] * W[iBX] + W[iVY] * W[iBY] + W[iVZ] * W[iBZ]
            S = xp.zeros_like(U)
            S[MX], S[MY], S[MZ] = -divB * W[iBX], -divB * W[iBY], -divB * W[iBZ]
            S[BX], S[BY], S[BZ] = -divB * W[iVX], -divB * W[iVY], -divB * W[iVZ]
            S[EN] = -divB * vB
            out[inner] += S[inner]
            self._rate_powell = xp.sum(S[EN][inner[1:]]) * cfg.dx * cfg.dy * cfg.dz
        eta = None
        if cfg.is_resistive():
            eta = resistivity_field(W, cfg)
            Rr, rate_r = resistive_terms_3d(U, eta, h, ng, n, cfg.eta_face_mean)
            out[inner] += Rr[inner]
            self._rate_boundary = self._rate_boundary + rate_r
        dV = cfg.dx * cfg.dy * cfg.dz

        def face_flux(var, G):                       # subtract div G (face arrays per axis), book the domain faces
            for ax in range(3):
                hi, lo = list(inner[1:]), list(inner[1:])
                hi[ax], lo[ax] = slice(1, n[ax] + 1), slice(0, n[ax])
                out[(var,) + inner[1:]] -= (G[ax][tuple(hi)] - G[ax][tuple(lo)]) / h[ax]
                if var == EN:
                    f0, f1 = list(inner[1:]), list(inner[1:])
                    f0[ax], f1[ax] = 0, n[ax]
                    self._rate_boundary = self._rate_boundary + xp.sum(G[ax][tuple(f0)] - G[ax][tuple(f1)]) * dV / h[ax]
        vN = None
        if cfg.nernst_coeff > 0:
            if not self.ct:
                raise NotImplementedError("Nernst transport in 3D needs div_b='ct'")
            vN, fac, theta, fuel = transport.nernst_velocity_3d(W, cfg.nernst_coeff, cfg.cond_x0,
                                                                cfg.cond_tracer_max, h)
            G, divqE = transport.nernst_energy_3d(W, vN, fac, theta, fuel, cfg.nernst_coeff, h, n)
            face_flux(EN, G)
            if cfg.two_temp:                         # the Ettingshausen heat flux is carried by the electrons
                rho_i = W[(iRHO,) + inner[1:]]
                out[(RHOSE,) + inner[1:]] -= (cfg.gamma - 1) * divqE * rho_i**(1 - cfg.gamma)
        if self.strength:
            dSi, sf = stren.rates3d(S, W, Fs, WLs, WRs, self._shear(W, S), h, ng, n)
            dS[...] = 0.0
            dS[(slice(None),) + inner[1:]] = dSi
            for k, var in enumerate((MX, MY, MZ, EN)):
                face_flux(var, [sf[ax][k] for ax in range(3)])
        if self.ct:
            out[BX:BZ + 1] = 0.0                     # B lives on the faces
            self._ct_face_rates(Fs, W, Bf, dBf, eta, vN)
        if self.cyl and cfg.cyl_filter:
            self._az_filter_rates(out)
        if self.walls is not None:
            self.walls.mask_rates(out)
        return out

    def total_energy(self):
        """total energy on the interior cells (r dr dtheta dz volumes in cylindrical geometry)"""
        ng = self.ng
        I = (slice(ng, -ng),) * 3
        dV = self.dV if self.cyl else self.cfg.dx * self.cfg.dy * self.cfg.dz
        return float(xp.sum(self.U[EN][I] * dV))

    def radiation_energy(self):
        """FLD radiation energy on the grid"""
        return float(xp.sum(self.Er)) * self.cfg.dx * self.cfg.dy * self.cfg.dz if self.Er is not None else 0.0

    @property
    def cumulative_radiated_energy(self):
        """Energy radiated so far, integrated over the 3D domain."""
        return float(self._acc_rad)

    def _acc(self, w):
        self._acc_boundary += w * self._rate_boundary
        self._acc_powell += w * self._rate_powell

    def step_ssprk3(self):
        dt = self.dt
        U0 = self.U
        self._ct_sync()
        if self.cfg.two_temp and getattr(self, "_te_owner", None) is not self.U:
            if float(xp.max(xp.abs(self.U[RHOSE]))) == 0.0:
                twotemp.init_equal(self.U, self.cfg.gamma, self.cfg.z_ion)
            self._te_owner = self.U
        F0, F1, dF = (self.Bf, self._Bf1, self._dBf) if self.ct else (None, None, None)
        S0, S1, dS = (self.S, self._S1, self._dS) if self.strength else (None, None, None)
        self.compute_rhs(U0, self.dU, F0, dF, S0, dS); self._acc(dt / 6)
        if self.ct:
            for f0, f1, d in zip(F0, F1, dF):
                xp.add(f0, dt * d, out=f1)
        if self.strength:
            xp.add(S0, dt * dS, out=S1)
        xp.add(U0, dt * self.dU, out=self.U_temp)
        self.compute_rhs(self.U_temp, self.dU, F1, dF, S1, dS); self._acc(dt / 6)
        if self.ct:
            for f0, f1, d in zip(F0, F1, dF):
                f1[...] = 0.75 * f0 + 0.25 * (f1 + dt * d)
        if self.strength:
            S1[...] = 0.75 * S0 + 0.25 * (S1 + dt * dS)
        self.U_stage[...] = 0.75 * U0 + 0.25 * (self.U_temp + dt * self.dU)
        self.compute_rhs(self.U_stage, self.dU, F1, dF, S1, dS); self._acc(2 * dt / 3)
        if self.ct:
            for f0, f1, d in zip(F0, F1, dF):
                f0[...] = f0 / 3.0 + 2.0 / 3.0 * (f1 + dt * d)
        if self.strength:
            S0[...] = S0 / 3.0 + 2.0 / 3.0 * (S1 + dt * dS)
        self.U[...] = U0 / 3 + 2.0 / 3.0 * (self.U_stage + dt * self.dU)
        if self.ct:
            n3 = (self.cfg.nx, self.cfg.ny, self.cfg.nz)
            self._cells_from_faces(self.U, self.Bf, tuple(slice(self.ng, self.ng + m) for m in n3))
        if self._vf:
            self.U[RHOC] = xp.clip(self.U[RHOC], 0.0, 1.0)
        else:
            rho = xp.maximum(self.U[RHO], FLOOR_RHO)
            self.U[RHOC] = rho * xp.clip(self.U[RHOC] / rho, 0.0, 1.0)
        cfg, ng = self.cfg, self.ng
        inner = (slice(None),) + tuple(slice(ng, ng + m) for m in (cfg.nx, cfg.ny, cfg.nz))
        if self.strength:                            # radial return to the yield surface, spall damage
            Wn = cons_to_prim(self.U, self.eos)
            stren.plastic(self.S, xp.clip(Wn[iCLR], 0, 1), Wn[iPR], cfg.yield_strength, cfg.spall_strength, ng)
        if cfg.rad_coeff > 0:
            dE = radiate(self.U[inner], self.eos, cfg.rad_coeff, cfg.rad_T_min, cfg.rad_tracer_max, dt,
                         electron_Z=cfg.z_ion if cfg.two_temp else None)
            self._acc_rad += xp.sum(dE) * cfg.dx * cfg.dy * cfg.dz
        if cfg.fld:
            step = radiation_multigroup.step if cfg.fld_groups else fld.step
            _, din, self.fld_iters = step(self.Er, self.U[inner], self.eos, cfg, (cfg.dx, cfg.dy, cfg.dz),
                                          self._periodic(), dt, advect=cfg.fld_advect)
            self.fld_inflow += din
        if cfg.cond_kappa0 > 0:
            per = (cfg.get_bc_x() == "periodic", True, True)
            wall = cfg.cond_wall_T if cfg.cond_wall == "isothermal" else None
            self.cond_stages = conduct(self.U[inner], self.eos, cfg.cond_kappa0, cfg.cond_x0, cfg.cond_tracer_max,
                                       (cfg.dx, cfg.dy, cfg.dz), per, dt, wall_T=wall, perp=cfg.cond_perp,
                                       electron_Z=cfg.z_ion if cfg.two_temp else None)
        if cfg.two_temp and cfg.te_eq_coeff > 0:
            Ui = self.U[inner]
            fuel = xp.clip(cons_to_prim(Ui, self.eos)[iCLR], 0, 1) < cfg.cond_tracer_max
            twotemp.equilibrate(Ui, cfg.gamma, cfg.z_ion, cfg.te_eq_coeff, fuel, dt)
        if cfg.visc_coeff > 0:
            Ui = self.U[inner]
            Wi = cons_to_prim(Ui, self.eos)
            fuel = xp.clip(Wi[iCLR], 0, 1) < cfg.cond_tracer_max
            theta_i = twotemp.temperatures(Ui, cfg.gamma, cfg.z_ion)[1] if cfg.two_temp else Wi[iPR] / Wi[iRHO]
            eta = cfg.visc_coeff * xp.maximum(theta_i, 0.0)**2.5
            self.visc_stages = viscous_step(Ui, eta, (cfg.dx, cfg.dy, cfg.dz), (cfg.get_bc_x() == "periodic", True, True),
                                            fuel, dt)
        if self._sigma is not None:
            self._U_before_sponge = self.U.copy()
            self._apply_sponge(dt)
        self._update_plateau()
        self.apply_bc(self.U)
        if self.ct:
            self._ghost_faces(self.Bf, self.U)
        if self._ch > 0 and not self.ct:
            h = min(self.cfg.dx, self.cfg.dy, self.cfg.dz)
            self.U[PSI] *= np.exp(-self.cfg.glm_alpha * self._ch * dt / h)

    # ------------------------------------------------------------ diagnostics

    def diagnostics(self):
        cfg = self.cfg
        ng = self.ng
        I = (slice(ng, -ng),) * 3
        W = cons_to_prim(self.U, self.eos)
        rho, C = W[iRHO][I], xp.clip(W[iCLR][I], 0, 1)
        vx, vy, vz = W[iVX][I], W[iVY][I], W[iVZ][I]
        bx, by, bz = W[iBX][I], W[iBY][I], W[iBZ][I]
        dx, dy, dz = cfg.dx, cfg.dy, cfg.dz
        dV = dx * dy * dz
        dA = dV / (cfg.z_max - cfg.z_min)    # per unit depth: comparable across z-extents and with 2D
        mix = (C > 0.05) & (C < 0.95)

        # interface displacement eta(y, z) = x_min + int (1 - C) dx
        eta = cfg.x_min + xp.sum(1.0 - C, axis=0) * dx
        eta_f = eta - xp.mean(eta)
        eta_zmean = xp.mean(eta, axis=1)
        prof = to_numpy(eta_zmean)
        a = np.abs(np.fft.rfft(prof - prof.mean())) * 2 / cfg.ny   # host FFT: cuFFT not required
        k4 = cfg.perturbation_mode

        wx, wy, wz = _curl(vx, vy, vz, dx, dy, dz)
        jx, jy, jz = _curl(bx, by, bz, dx, dy, dz)
        J = xp.sqrt(jx**2 + jy**2 + jz**2)
        J_rms = float(xp.sqrt(xp.mean(J**2)))
        divB = (xp.gradient(bx, dx, axis=0) + xp.gradient(by, dy, axis=1) + xp.gradient(bz, dz, axis=2))
        bmax = float(xp.max(xp.sqrt(bx**2 + by**2 + bz**2)))
        E = float(xp.sum(self.U[EN][I]) * dV)
        return {
            "t": self.t,
            "a4": float(a[k4]) if k4 < a.size else 0.0,
            "x_int": float(xp.mean(eta)),
            "rough": float(xp.sqrt(xp.mean(eta_f**2))),
            "rough_z": float(xp.sqrt(xp.mean((eta - eta_zmean[:, None])**2))),   # z-variation of the interface
            "emix": float(xp.sum(xp.where(mix, rho * (wx**2 + wy**2 + wz**2), 0.0)) * dA),
            "Ekz": float(xp.sum(0.5 * rho * vz**2) * dA),
            "Eperp": float(xp.sum(0.5 * (by**2 + bz**2)) * dA),           # field generated across B_x
            "Emag": float(xp.sum(0.5 * (bx**2 + by**2 + bz**2)) * dA),
            "J_max": float(xp.max(J)),
            "J_rms": J_rms,
            "J_mix_rms": float(xp.sqrt(xp.sum(xp.where(mix, J**2, 0.0)) / max(int(xp.sum(mix)), 1))),
            "sheet_frac": float(xp.mean(J > 5 * J_rms)) if J_rms > 0 else 0.0,
            "J_xmod": self._sheet_xmod(J, C),
            "J_zvar": float(xp.sqrt(xp.mean((J - xp.mean(J, axis=2, keepdims=True))**2))) / J_rms if J_rms > 0 else 0.0,
            "divB_rel": float(xp.max(xp.abs(divB))) * min(dx, dy, dz) / bmax if bmax > 0 else 0.0,
            "E": E,
            "inflow": float(self._acc_boundary),
            "powell": float(self._acc_powell),
            "sponge": float(self._acc_sponge) if getattr(self, "_sigma", None) is not None else 0.0,
            "radiated": float(self._acc_rad),
        }

    def _sheet_xmod(self, J, C):
        """rms x-variation / mean of z-averaged |J| along the current-sheet rows, in the shocked heavy gas
        more than 0.5 behind the interface (tearing breaks a sheet into islands along x = along B)"""
        Jm = xp.mean(J, axis=2)                       # (nx, ny)
        heavy = xp.mean(C, axis=(1, 2)) > 0.95
        if int(xp.sum(heavy)) < 8:
            return 0.0
        prof = xp.mean(Jm[heavy], axis=0)
        rows = prof > 0.5 * xp.max(prof)
        seg = Jm[heavy][:, rows]
        return float(xp.mean(xp.std(seg, axis=0) / xp.maximum(xp.mean(seg, axis=0), 1e-30)))

    def save_snapshot(self, label):
        cfg = self.cfg
        ng = self.ng
        I = (slice(ng, -ng),) * 3
        W = cons_to_prim(self.U, self.eos)
        bx, by, bz = W[iBX][I], W[iBY][I], W[iBZ][I]
        vx, vy, vz = W[iVX][I], W[iVY][I], W[iVZ][I]
        jx, jy, jz = _curl(bx, by, bz, cfg.dx, cfg.dy, cfg.dz)
        wx, wy, wz = _curl(vx, vy, vz, cfg.dx, cfg.dy, cfg.dz)
        k = cfg.nz // 2
        self.snapshots[label] = {
            "t": self.t,
            "rho_mid": to_numpy(W[iRHO][I][:, :, k]),
            "C_mid": to_numpy(W[iCLR][I][:, :, k]),
            "J_mid": to_numpy(xp.sqrt(jx**2 + jy**2 + jz**2)[:, :, k]),
            "w_mid": to_numpy(xp.sqrt(wx**2 + wy**2 + wz**2)[:, :, k]),
            "vz_mid": to_numpy(vz[:, :, k]),
            # y-z cut through the mixing layer at the mean interface position
            "J_yz": to_numpy(xp.sqrt(jx**2 + jy**2 + jz**2)[int(xp.argmax(xp.mean(
                xp.abs(xp.gradient(W[iCLR][I], axis=0)), axis=(1, 2))))]),
        }

    # ------------------------------------------------------------ run

    def run(self, snapshot_times=(), verbose=True):
        cfg = self.cfg
        t0 = time.time()
        self.diag = {}

        def record():
            for k, v in self.diagnostics().items():
                self.diag.setdefault(k, []).append(v)

        record()
        pending = sorted(snapshot_times)
        while self.t < cfg.t_end and self.step < cfg.max_steps:
            self.dt = min(self.compute_dt(cons_to_prim(self.U, self.eos)), cfg.t_end - self.t)
            if self.dt <= 1e-16:
                break
            self.step_ssprk3()
            self.t += self.dt
            self.step += 1
            if self.step % cfg.diag_interval == 0:
                record()
            while pending and self.t >= pending[0] - 0.5 * self.dt:
                self.save_snapshot(f"t={pending.pop(0):.3f}")
            if verbose and self.step % 200 == 0:
                d = self.diag
                print(f"  step {self.step:5d} t={self.t:.4f} dt={self.dt:.2e} rough={d['rough'][-1]:.4f} "
                      f"Ekz={d['Ekz'][-1]:.3e} Eperp={d['Eperp'][-1]:.3e}", flush=True)
        record()
        self.wall = time.time() - t0
        self.diag = {k: np.array(v) for k, v in self.diag.items()}
        return self.diag

    def energy_error(self):
        """|E(t) - E(0) - boundary inflow - Powell - sponge + radiated| / E(0)"""
        d = self.diag
        return abs(d["E"][-1] - d["E"][0] - d["inflow"][-1] - d["powell"][-1] - d["sponge"][-1]
                   + d["radiated"][-1]) / abs(d["E"][0])
