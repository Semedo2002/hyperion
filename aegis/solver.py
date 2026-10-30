# solver.py - MHDSolver (HLLD, MUSCL, SSP-RK3, GLM) and PostProcessor (figures, GIFs).

import time
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib import cm

matplotlib.rcParams.update({
    "font.size": 11, "font.family": "serif", "mathtext.fontset": "cm",
    "figure.dpi": 140, "savefig.dpi": 180,
    "axes.labelsize": 12, "axes.titlesize": 13, "legend.fontsize": 9,
    "figure.facecolor": "white",
})

from .config import (
    Config, NVAR, FLOOR_RHO, FLOOR_PR,
    RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI, RHOC, RHOSE,
    iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iPSI, iCLR,
)
from .diagnostics import diagnostics_for_solver
from .eos import make_eos
from . import fld
from . import radiation_multigroup
from . import ct
from . import fused
from . import hall
from . import ions
from . import transport
from . import twotemp
from .viscosity import viscous_step
from . import strength as stren
from .physics import (
    xp, cp,
    cons_to_prim, prim_to_cons,
    muscl_x, muscl_y,
    hlld_flux_x, hlld_flux_y,
    mhd_rankine_hugoniot,
    mhd_rankine_hugoniot_local_gpu,
    parallel_stripe_limits,
    parallel_stripe_state,
    compute_resistive_terms,
    sound_speed2,
    radiate,
    vf_advection_rate,
    conduct,
    resistivity_field,
    eta_max_of,
    richtmyer_linear_theory,
    smooth,
)

OUTPUT_DIR = "rmi_output_gpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


def save_figure(fig, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=180)
    plt.close(fig)
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        return True, filepath, size_kb
    return False, filepath, 0


def to_numpy(a):
    if isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def _record_diagnostics(solver):
    diagnostics = diagnostics_for_solver(solver)
    solver.diag_times.append(solver.t)
    target_by_key = {
        "mixing_width_integral": solver.diag_mixing_width_integral,
        "mixing_width_thresh": solver.diag_mixing_width_thresh,
        "mixedness": solver.diag_mixedness,
        "enstrophy": solver.diag_enstrophy,
        "enstrophy_local": solver.diag_enstrophy_local,
        "enstrophy_mix": solver.diag_enstrophy_mix,
        "interface_circulation": solver.diag_interface_circulation,
        "perturbation_amp": solver.diag_perturbation_amp,
        "mode_amps": solver.diag_mode_amps,
        "stag_pressure": solver.diag_stag_pressure,
        "divB_max": solver.diag_divB_max,
        "divB_L2": solver.diag_divB_L2,
        "divB_rel_max": solver.diag_divB_rel_max,
        "energy_total": solver.diag_energy_total,
        "boundary_flux_cumulative": solver.diag_boundary_flux_cumulative,
        "powell_cumulative": solver.diag_powell_cumulative,
        "jet_max_vx": solver.diag_jet_max_vx,
        "jet_collimation": solver.diag_jet_collimation,
        "jet_momentum_flux_gap": solver.diag_jet_momentum_flux_gap,
        "jet_momentum_flux_bar": solver.diag_jet_momentum_flux_bar,
    }
    for key, value in diagnostics.items():
        if key in target_by_key:
            target_by_key[key].append(np.asarray(value) if key == "mode_amps" else value)


# ============================================================
# MHD Solver Class
# ============================================================
class MHDSolver:
    """2D Ideal MHD solver with HLLD, MUSCL, SSP-RK3, GLM+Powell.
    Supports uniform and spatially modulated (striped) magnetic fields."""

    def __init__(self, config):
        self.cfg = config
        # float gamma for one ideal gas, eos.EOS for two materials
        self.eos = make_eos(config)
        self._vf = not isinstance(self.eos, (int, float)) and self.eos.volume_fraction
        # constrained transport: face fields Bf = [Bx on x-faces, By on y-faces], built from U on first use
        self.ct = config.div_b == "ct"
        self.Bf = None
        self._faces_owner = None
        self.cyl = config.geometry == "cylindrical"
        self.strength = config.strength
        self.pic = None                              # adaptive kinetic-fluid hybrid (aegis/hybrid_pic.py)
        self.walls = None                            # immersed rigid walls (aegis/cad_importer.ImmersedWalls)
        if self.cyl and abs(config.x_min) > 0:
            raise ValueError("cylindrical geometry needs x_min = 0 (the axis)")
        self.ng = 2
        self.nx_tot = config.nx + 2 * self.ng
        self.ny_tot = config.ny + 2 * self.ng

        self.x = xp.asarray(config.x_min + (np.arange(self.nx_tot) - self.ng + 0.5) * config.dx)
        # radii (cylindrical) and the interior cell volume element, dx dy or r dr dz (per radian)
        self.rf = xp.asarray(config.x_min + (np.arange(self.nx_tot + 1) - self.ng) * config.dx)
        self.rc = self.x
        self._r0, self._r1 = config.x_min, config.x_max          # radii of the x boundaries (no device reads)
        if config.geometry == "cylindrical":
            self.dV = (self.rc[self.ng:self.ng + config.nx] * config.dx * config.dy)[:, None]
        else:
            self.dV = config.dx * config.dy
        self.y = xp.asarray(config.y_min + (np.arange(self.ny_tot) - self.ng + 0.5) * config.dy)
        self.X, self.Y = xp.meshgrid(self.x, self.y, indexing='ij')

        self.U = xp.zeros((NVAR, self.nx_tot, self.ny_tot))
        self.U_temp = xp.empty_like(self.U)
        self.U_stage = xp.empty_like(self.U)
        self.dU = xp.empty_like(self.U)
        if config.strength:                          # deviatoric stress (S_xx, S_yy, S_xy) and damage
            self.S = xp.zeros((4, self.nx_tot, self.ny_tot))
            self._S1 = xp.zeros_like(self.S)
            self._dS = xp.zeros_like(self.S)
        self.t = 0.0
        self.step = 0
        self.dt = 0.0
        self._glm_ch_frozen = 0.0

        self.diag_times = []
        self.diag_mixing_width_integral = []
        self.diag_mixing_width_thresh = []
        self.diag_mixedness = []
        self.diag_enstrophy = []
        self.diag_enstrophy_local = []
        self.diag_enstrophy_mix = []
        self.diag_interface_circulation = []
        self.diag_perturbation_amp = []
        self.diag_mode_amps = []
        self.diag_stag_pressure = []
        self.diag_divB_max = []
        self.diag_divB_L2 = []
        self.diag_divB_rel_max = []
        self.diag_energy_total = []
        self.diag_boundary_flux_cumulative = []
        self.diag_powell_cumulative = []
        # Nozzle / jet diagnostics
        self.diag_jet_max_vx = []
        self.diag_jet_collimation = []
        self.diag_jet_momentum_flux_gap = []
        self.diag_jet_momentum_flux_bar = []
        # Exact energy budget, accumulated on the GPU from the face fluxes that
        # actually update U[EN] (SSP-RK3 stage weighted):
        #   E(t) - E(0) = boundary inflow + Powell source   (to round-off)
        self._acc_boundary_energy = xp.zeros(())
        self._acc_powell_energy = xp.zeros(())
        self._stage_boundary_rate = xp.zeros(())
        self._stage_powell_rate = xp.zeros(())
        # Ohmic heating integral (diagnostic; already inside E through the Poynting flux)
        self._acc_ohmic = xp.zeros(())
        self._stage_ohmic_rate = xp.zeros(())
        # radiated energy (split step after each RK step; removed from E, so it is a budget term)
        self._acc_rad = xp.zeros(())
        # grey FLD radiation energy density on interior cells, and the radiation energy that entered the domain
        self.Er = None if not config.fld else xp.zeros(((config.fld_groups,) if config.fld_groups else ())
                                                       + (config.nx, config.ny))    # (groups, x, y) if multigroup
        self.fld_inflow = 0.0
        self._inflow_U = None
        self._inflow_superfast = None
        self._dt_limiter = "hyperbolic"
        self.snapshots = {}
        self._rho_light = 1.0
        self._rho_heavy = 1.0
        self._rh_results = None
        self._bar_mask_y = None
        self._gap_mask_y = None

    def _build_bar_gap_masks(self):
        """Build y-index masks for bar (high |B|) and gap (low |B|) regions."""
        cfg = self.cfg
        ng = self.ng
        y_arr = to_numpy(self.y[ng:-ng])
        Ly = cfg.y_max - cfg.y_min
        k_mod = 2 * np.pi * cfg.B_modulation_mode / Ly
        By_profile = np.cos(k_mod * y_arr + cfg.B_stripe_phase)
        self._bar_mask_y = np.abs(By_profile) > 0.5
        self._gap_mask_y = np.abs(By_profile) <= 0.5

    def initialize(self):
        """Set up initial conditions with uniform or striped magnetic field.
        Supports both global-average and localized per-cell R-H jump conditions."""
        cfg = self.cfg
        g = cfg.gamma
        M = cfg.mach
        rho1 = 1.0; p1 = 1.0
        Ly = cfg.y_max - cfg.y_min

        rho_h = rho1 * cfg.density_ratio
        x_shock = cfg.interface_x - 0.3
        ky = 2 * np.pi * cfg.perturbation_mode / Ly

        x_if = cfg.interface_x + cfg.perturbation_amp * xp.sin(ky * self.Y)
        interface_delta = cfg.interface_width * cfg.dx
        phi = 0.5 * (1.0 + xp.tanh((self.X - x_if) / max(interface_delta, 1e-10)))

        W = xp.zeros((NVAR, self.nx_tot, self.ny_tot))
        post = self.X < x_shock
        rho_pre = rho1 * (1 - phi) + rho_h * phi

        # --- Build the pre-shock magnetic field ---
        # By_pre is the shock-transverse field that enters the R-H jump; it is
        # stored in component bt_idx (By, or Bz for solenoidal stripes).
        # Bx_pre is a shock-normal field (continuous across the shock).
        bt_idx = iBY
        Bx_pre = None
        stripe = cfg.B_stripe_component
        if cfg.is_striped():
            if stripe not in ("x", "y", "z"):
                raise ValueError(f"B_stripe_component must be 'x', 'y' or 'z', got {stripe!r}")
            k_mod = 2 * np.pi * cfg.B_modulation_mode / Ly
            eps = cfg.B_gap_fraction
            if not 0.0 <= eps < 1.0:
                raise ValueError(f"B_gap_fraction must be in [0, 1), got {eps}")
            if eps > 0.0:
                if stripe != "x":
                    raise ValueError("B_gap_fraction is only implemented for x stripes")
                profile = cfg.B_transverse * xp.sqrt(eps**2 + (1 - eps**2) * xp.cos(k_mod * self.Y + cfg.B_stripe_phase)**2)
            else:
                profile = cfg.B_transverse * xp.cos(k_mod * self.Y + cfg.B_stripe_phase)
            self._build_bar_gap_masks()
            if stripe == "x":
                Bx_pre = profile
                By_pre = xp.zeros_like(self.Y)
            else:
                By_pre = profile
                bt_idx = iBZ if stripe == "z" else iBY
        elif cfg.is_mhd():
            By_pre = xp.full_like(self.Y, cfg.B_transverse)
        else:
            By_pre = xp.zeros_like(self.Y)
        has_transverse = bool(xp.any(By_pre != 0))

        # --- Rankine-Hugoniot: localized or global ---
        if Bx_pre is not None:
            # Shock-parallel stripes: upstream in pressure balance, per-row
            # hydrodynamic jump (B_n is continuous), planar shock.
            self._init_parallel_stripes(W, Bx_pre, post, phi)
        elif cfg.localized_rh and has_transverse:
            # per-cell R-H using local By(y)
            rho2_field, p2_field, vx2_field, By2_field, vs_field = \
                mhd_rankine_hugoniot_local_gpu(g, M, rho1, p1, By_pre)
            W[iRHO] = xp.where(post, rho2_field, rho_pre)
            W[iVX] = xp.where(post, vx2_field, 0.0)
            W[iPR] = xp.where(post, p2_field, p1)
            W[bt_idx] = xp.where(post, By2_field, By_pre)
            vs_avg = float(xp.mean(vs_field))
            self._rh_results = {
                'rho2': float(xp.mean(rho2_field[post])),
                'p2': float(xp.mean(p2_field[post])),
                'vx2': float(xp.mean(vx2_field[post])),
                'By2': float(xp.mean(By2_field[post])),
                'vs': vs_avg
            }
            print(f"  LOCALIZED R-H: per-cell jump conditions applied")
            print(f"  Mean post-shock: rho2={self._rh_results['rho2']:.3f}, "
                  f"p2={self._rh_results['p2']:.3f}, vx2={self._rh_results['vx2']:.3f}")
        else:
            # Original: global-average R-H
            By_pre_eff = cfg.B_transverse if has_transverse else 0.0
            if cfg.is_striped() and has_transverse:
                By_pre_eff = cfg.B_transverse / np.sqrt(2.0)
            rho2, p2, vx2, By2, vs = mhd_rankine_hugoniot(g, M, rho1, p1, By_pre_eff)
            self._rh_results = {'rho2': rho2, 'p2': p2, 'vx2': vx2, 'By2': By2, 'vs': vs}
            W[iRHO] = xp.where(post, rho2, rho_pre)
            W[iVX] = xp.where(post, vx2, 0.0)
            W[iPR] = xp.where(post, p2, p1)
            if cfg.is_striped() and has_transverse:
                By2_spatial = By_pre * (rho2 / rho1)
                W[bt_idx] = xp.where(post, By2_spatial, By_pre)
            elif has_transverse:
                W[iBY] = xp.where(post, By2, cfg.B_transverse)

        W[iCLR] = xp.where(post, 0.0, phi)

        self.U = prim_to_cons(W, self.eos)
        self.t = 0.0; self.step = 0
        self._acc_boundary_energy = xp.zeros(())
        self._acc_powell_energy = xp.zeros(())
        self._acc_ohmic = xp.zeros(())
        self._acc_rad = xp.zeros(())
        self._rho_light = rho1; self._rho_heavy = rho_h
        # Post-shock state held fixed in the x_min ghost cells (driven shock).
        self._inflow_U = self.U[:, :self.ng, :].copy()
        self._check_inflow_characteristics(W[:, :self.ng, :])

        vs = self._rh_results['vs']
        rho2 = self._rh_results['rho2']
        p2 = self._rh_results['p2']
        vx2 = self._rh_results['vx2']
        print(f"  Mach={M:.1f}, Vs={vs:.3f}, rho2={rho2:.3f}, p2={p2:.3f}, vx2={vx2:.3f}")
        if cfg.is_striped():
            comp = {"x": "Bx", "y": "By", "z": "Bz"}[stripe]
            print(f"  STRIPED B-FIELD: {comp} = {cfg.B_transverse:.2f} * cos({cfg.B_modulation_mode}*k*y)")
            if stripe == "x":
                eq = self._equilibrium
                print(f"  EQUILIBRIUM ({cfg.equilibrium_closure}): P_total={eq['P_total']:.4g}, "
                      f"p_bar={cfg.p_bar:.4g}, gap Mach={eq['M_gap']:.3g}, "
                      f"min downstream Alfven Mach={eq['MA_min']:.3g}, admissible B0 < {eq['B0_max']:.4g}")
            else:
                print(f"  WARNING: {comp} stripes are not initialised in pressure balance "
                      f"(p + B^2/2 varies in y); only 'x' stripes are.")
            if stripe == "y":
                div0 = cfg.B_transverse * 2 * np.pi * cfg.B_modulation_mode / Ly
                print(f"  WARNING: legacy By(y) stripes are not solenoidal: max|div B| = {div0:.3g} "
                      f"ahead of the shock and ~{self._rh_results['rho2']:.2f}x that behind it. "
                      f"Use B_stripe_component='z' or 'x'.")
        elif cfg.is_mhd():
            print(f"  UNIFORM B-FIELD: By = {cfg.B_transverse:.3f}")
        else:
            print(f"  B_y=0 (pure hydro)")
        if cfg.is_resistive():
            if cfg.eta_model == "constant":
                print(f"  RESISTIVE MHD: eta = {cfg.eta_resistivity:.4e}")
            else:
                print(f"  RESISTIVE MHD: eta model {cfg.eta_model}, fuel {cfg.eta_fuel:.3e}, liner "
                      f"{cfg.eta_liner:.3e}, cap {cfg.eta_max:.3e}, {cfg.eta_face_mean} face mean")
        print(f"  Grid: {cfg.nx}x{cfg.ny}, dx={cfg.dx:.4f}, R-H: {'localized' if cfg.localized_rh else 'global'}")
        if cfg.powell_source:
            print(f"  Powell source terms: ENABLED")
        print(f"  BC x: {cfg.get_bc_x()}, BC y: {cfg.bc_y_type}")
        sys.stdout.flush()

    def _init_parallel_stripes(self, W, Bx_pre, post, phi):
        cfg = self.cfg
        lim = parallel_stripe_limits(cfg.gamma, cfg.mach, cfg.p_bar, cfg.equilibrium_closure,
                                     gap_fraction=cfg.B_gap_fraction)
        if cfg.B_transverse >= lim["B0_max"]:
            why = ("the gaps would be subsonic" if lim["B0_max_gap_mach"] < lim["B0_max_fast_shock"]
                   else "the hydrodynamic jump is not a fast MHD shock (switch-on regime)")
            raise ValueError(
                f"B0={cfg.B_transverse:g} is not admissible for x stripes at Mach {cfg.mach:g}, "
                f"p_bar={cfg.p_bar:g}: B0 must be < {lim['B0_max']:.4g} ({why}). "
                f"Only B0/sqrt(p_bar) matters; raising p_bar just rescales units.")
        st = parallel_stripe_state(cfg.gamma, cfg.mach, cfg.p_bar, cfg.equilibrium_closure, Bx_pre)
        rho_up = st["rho1"] * (1 - phi) + cfg.density_ratio * st["rho1"] * phi
        W[iRHO] = xp.where(post, st["rho2"], rho_up)
        W[iVX] = xp.where(post, st["vx2"], 0.0)
        W[iPR] = xp.where(post, st["p2"], st["p1"])
        W[iBX] = Bx_pre
        self._rh_results = {
            'rho2': float(xp.mean(st["rho2"][post])), 'p2': float(xp.mean(st["p2"][post])),
            'vx2': float(xp.mean(st["vx2"][post])), 'By2': 0.0, 'vs': st["vs"]}
        self._equilibrium = {
            "P_total": st["P_total"], "M_gap": float(xp.min(st["M_local"])),
            "MA_min": float(xp.min(st["MA_downstream"])), **lim}

    def apply_bc(self, U):
        ng = self.ng; cfg = self.cfg; bc_x = cfg.get_bc_x()
        if bc_x == "periodic":
            U[:, :ng, :] = U[:, -2*ng:-ng, :]
            U[:, -ng:, :] = U[:, ng:2*ng, :]
        elif bc_x == "inflow_outflow":
            self._apply_inflow_outflow_bc_x(U)
        elif bc_x == "extrapolation":
            for i in range(ng):
                U[:, i, :] = U[:, ng, :]
                U[:, -(i+1), :] = U[:, -(ng+1), :]
            if self.cyl:                             # axis: mirror, odd in v_r, v_phi, B_r, B_phi
                for k in range(ng):
                    U[:, ng - 1 - k, :] = U[:, ng + k, :]
                for v in (MX, MZ, BX, BZ):
                    U[v, :ng, :] = -U[v, :ng, :]
        else:
            raise ValueError(f"Unknown bc_x_type {bc_x!r}")
        if cfg.bc_y_type == "periodic":
            U[:, :, :ng] = U[:, :, -2*ng:-ng]
            U[:, :, -ng:] = U[:, :, ng:2*ng]
        else:
            for j in range(ng):
                U[:, :, j] = U[:, :, ng]
                U[:, :, -(j+1)] = U[:, :, -(ng+1)]
        return U

    def _apply_inflow_outflow_bc_x(self, U):
        """Driven-shock boundaries.

        x_min: the post-shock state is imposed in the ghost cells. For a
        super-fast inflow all seven MHD characteristics enter the domain, so
        prescribing the full state is the characteristic boundary condition
        (checked at initialisation by _check_inflow_characteristics).
        x_max: transmissive (zero-gradient) outflow; the upwind HLLD flux then
        carries outgoing waves out of the domain. psi is extrapolated too, so
        GLM divergence waves leave instead of reflecting off psi = 0.
        """
        ng = self.ng
        if self._inflow_U is not None:
            U[:, :ng, :] = self._inflow_U
        else:
            for i in range(ng):
                U[:, i, :] = U[:, ng, :]
        for i in range(ng):
            U[:, -(i+1), :] = U[:, -(ng+1), :]

    def _check_inflow_characteristics(self, W_in):
        """Report whether the Dirichlet inflow is super-fast (well posed)."""
        g = self.cfg.gamma
        rho = xp.maximum(W_in[iRHO], FLOOR_RHO)
        a2 = g * xp.maximum(W_in[iPR], FLOOR_PR) / rho
        b2 = (W_in[iBX]**2 + W_in[iBY]**2 + W_in[iBZ]**2) / rho
        bx2 = W_in[iBX]**2 / rho
        cf = xp.sqrt(0.5 * (a2 + b2 + xp.sqrt(xp.maximum((a2 + b2)**2 - 4 * a2 * bx2, 0.0))))
        margin = float(xp.min(W_in[iVX] - cf))
        self._inflow_superfast = margin > 0
        if self.cfg.get_bc_x() == "inflow_outflow" and not self._inflow_superfast:
            print(f"  NOTE: inflow is sub-fast (min(vx - cf) = {margin:.3g}); the Dirichlet inflow "
                  f"over-specifies the outgoing fast characteristic.")

    def enforce_scalar_bounds(self, U):
        if self._vf:
            U[RHOC] = xp.clip(U[RHOC], 0.0, 1.0)
            return U
        rho = xp.maximum(U[RHO], FLOOR_RHO)
        C = xp.clip(U[RHOC] / rho, 0.0, 1.0)
        U[RHOC] = rho * C
        return U

    def compute_dt(self, W):
        cfg = self.cfg
        rho = xp.maximum(W[iRHO], FLOOR_RHO)
        B2 = W[iBX]**2 + W[iBY]**2 + W[iBZ]**2
        c2 = sound_speed2(W, self.eos)
        if self.strength:                            # longitudinal elastic waves: c^2 + 4 G / (3 rho)
            G = cfg.shear_modulus * xp.clip(W[iCLR], 0, 1) * (1 - self.S[stren.DMG])
            c2 = c2 + 4.0 * G / (3.0 * rho)
        cf = xp.sqrt(c2 + B2 / rho)
        sx = xp.abs(W[iVX]) + cf
        sy = xp.abs(W[iVY]) + cf
        # GLM speed = largest MHD signal speed (Dedner et al. 2002). psi waves
        # travel at ch in the lab frame (they are not advected), so this choice
        # cleans as fast as possible without tightening the CFL limit.
        smax = float(xp.max(xp.maximum(sx, sy)))
        cfg.glm_ch = max(smax, 1e-12)
        inv_dt = max(float(xp.max(xp.maximum(sx / cfg.dx, sy / cfg.dy))),
                     cfg.glm_ch / min(cfg.dx, cfg.dy))
        dt = cfg.cfl / inv_dt if inv_dt > 1e-14 else cfg.cfl * min(cfg.dx, cfg.dy)
        self._dt_limiter = "hyperbolic"
        if cfg.nernst_coeff > 0:                     # Nernst advection of B
            vx, vy, _, _, _ = transport.nernst_velocity(W, cfg.nernst_coeff, cfg.cond_x0, cfg.cond_tracer_max,
                                                        cfg.dx, cfg.dy, None)
            vN = float(xp.max(xp.maximum(xp.abs(vx) / cfg.dx, xp.abs(vy) / cfg.dy)))
            if vN > 0 and cfg.cfl / vN < dt:
                dt = cfg.cfl / vN
                self._dt_limiter = "nernst"
        if cfg.is_resistive():
            # Explicit diffusion of B: FTCS 2D limit dt <= 1 / (2 eta (1/dx^2 + 1/dy^2))
            # (= dx^2 / 4 eta for dx = dy); SSP-RK3 is stable to ~2.5x this.
            eta_max = eta_max_of(resistivity_field(W, cfg)) * max(1.0, cfg.eta_perp_ratio)
            dt_par = 0.5 / (max(eta_max, 1e-300) * (1.0 / cfg.dx**2 + 1.0 / cfg.dy**2))
            if dt_par < dt:
                dt = dt_par
                self._dt_limiter = "parabolic"
        if cfg.gyro_di > 0:
            dt_g = ions.dt_limit(W, self.U, cfg, cfg.dx, cfg.dy)
            if dt_g < dt:
                dt = dt_g
                self._dt_limiter = "gyroviscous"
        if self.pic is not None:
            dt_p = self.pic.dt_limit()
            if dt_p < dt:
                dt = dt_p
                self._dt_limiter = "gyration"
        if cfg.hall_di > 0:
            dt_h = hall.dt_limit(W, cfg, cfg.dx, cfg.dy)
            if dt_h < dt:
                dt = dt_h
                self._dt_limiter = "whistler"
        return dt

    @property
    def _cumulative_boundary_energy(self):
        return float(self._acc_boundary_energy)

    @property
    def _cumulative_powell_energy(self):
        return float(self._acc_powell_energy)

    @property
    def cumulative_radiated_energy(self):
        """Energy radiated away so far (optically thin bremsstrahlung)."""
        return float(self._acc_rad)

    def radiation_energy(self):
        """FLD radiation energy on the grid (integral of E_r; per radian in r-z)"""
        return float(xp.sum(self.Er * self.dV)) if self.Er is not None else 0.0

    @property
    def cumulative_ohmic_heating(self):
        """Integral of eta |J|^2 over the domain and time so far."""
        return float(self._acc_ohmic)

    def _compute_powell_source(self, U, W):
        cfg = self.cfg
        S = xp.zeros_like(U)
        Bx = W[iBX]; By_f = W[iBY]
        divB = xp.zeros_like(Bx)
        divB[1:-1, 1:-1] = (
            (0.5*(Bx[1:-1, 1:-1] + Bx[2:, 1:-1]) -
             0.5*(Bx[:-2, 1:-1] + Bx[1:-1, 1:-1])) / cfg.dx +
            (0.5*(By_f[1:-1, 1:-1] + By_f[1:-1, 2:]) -
             0.5*(By_f[1:-1, :-2] + By_f[1:-1, 1:-1])) / cfg.dy
        )
        vB = W[iVX]*W[iBX] + W[iVY]*W[iBY] + W[iVZ]*W[iBZ]
        S[MX] = -divB * W[iBX]; S[MY] = -divB * W[iBY]; S[MZ] = -divB * W[iBZ]
        S[BX] = -divB * W[iVX]; S[BY] = -divB * W[iVY]; S[BZ] = -divB * W[iVZ]
        S[EN] = -divB * vB
        return S

    def _ct_sync(self):
        """build the face fields from self.U if it was (re)assigned since they were made; cell B becomes the
        face average (total energy unchanged)"""
        if not self.ct or self._faces_owner is self.U:
            return
        U = self.U
        if self.cyl:
            self.apply_bc(U)
        self.Bf = ct.faces_from_cells(U, 2)
        if self.cyl:
            self.Bf[0][self.ng] = 0.0                    # B_r = 0 on the axis
        ct.cells_from_faces(U, self.Bf, (slice(None), slice(None)))   # E kept, so the budget is unaffected
        self._faces_owner = U
        self._Bf1 = [xp.empty_like(f) for f in self.Bf]
        self._dBf = [xp.empty_like(f) for f in self.Bf]

    def set_vector_potential(self, Az):
        """Set the in-plane field from Az(x, y) (numpy callable, evaluated on cell corners): faces are its
        discrete curl, so div B = 0 to round-off; cell B and the magnetic energy in E follow, pressure kept.
        Call after setting U."""
        cfg, ng = self.cfg, self.ng
        xf = cfg.x_min + (np.arange(self.nx_tot + 1) - ng) * cfg.dx
        yf = cfg.y_min + (np.arange(self.ny_tot + 1) - ng) * cfg.dy
        Bf = ct.faces_from_potential(Az, (to_numpy(self.x), to_numpy(self.y)), (xf, yf))
        U = self.U
        B2 = U[BX]**2 + U[BY]**2
        if self.ct:
            self.Bf = Bf
            self._faces_owner = U
            self._Bf1 = [xp.empty_like(f) for f in Bf]
            self._dBf = [xp.empty_like(f) for f in Bf]
        ct.cells_from_faces(U, Bf, (slice(None), slice(None)))
        U[EN] += 0.5 * (U[BX]**2 + U[BY]**2 - B2)

    def _te_sync(self):
        """two-temperature: start electrons at T_e = T_i where the electron entropy was never set"""
        if not self.cfg.two_temp or getattr(self, "_te_owner", None) is self.U:
            return
        if float(xp.max(xp.abs(self.U[RHOSE]))) == 0.0:
            twotemp.init_equal(self.U, self.cfg.gamma, self.cfg.z_ion)
        self._te_owner = self.U

    def temperatures(self):
        """(theta_e, theta_i) on all cells; equal to p/rho without two_temp"""
        if self.cfg.two_temp:
            self._te_sync()
            return twotemp.temperatures(self.U, self.cfg.gamma, self.cfg.z_ion)
        W = cons_to_prim(self.U, self.eos)
        return W[iPR] / W[iRHO], W[iPR] / W[iRHO]

    def _axis_faces(self, Bf):
        """faces behind the axis: B_r odd, B_z even"""
        ng = self.ng
        Bf[0][ng] = 0.0
        for k in range(1, ng + 1):
            Bf[0][ng - k] = -Bf[0][ng + k]
        for k in range(ng):
            Bf[1][ng - 1 - k] = Bf[1][ng + k]

    def _periodic(self):
        return (self.cfg.get_bc_x() == "periodic", self.cfg.bc_y_type == "periodic")

    # hooks a domain-decomposed subclass (aegis/parallel.py) overrides: ghost faces, edge EMFs, extra arrays
    def _ghost_faces(self, Bf, U):
        ct.ghost_faces(Bf, U, self.ng, (self.cfg.nx, self.cfg.ny), self._periodic())
        if self.cyl:
            self._axis_faces(Bf)

    def _emf_hook(self, E):
        return E

    def _flux_hook(self, Fx, Fy, WLx, WRx, WLy, WRy):    # AMR records the fluxes on patch boundaries
        pass

    def _halo(self, A):
        pass

    # AMR (aegis/amr.py): non-ideal face fluxes on patch boundaries, ghost data for the split diffusion steps
    _records_fluxes = False

    def _add_flux(self, var, Gx, Gy):
        """more face fluxes of variable var in this RK stage, laid out like the hydro Fx, Fy"""

    def _add_split_flux(self, key, var, Gx, Gy):
        """time-integrated face fluxes of a split step on interior faces ((nx+1, ny), (nx, ny+1)); key "F" (U
        variable var) or "Er" (radiation energy)"""

    def _pad_faces(self, a, axis):
        """face array on interior rows (x-faces, axis 0) or columns (y-faces, axis 1) -> the Fx / Fy layout"""
        ng, nx, ny = self.ng, self.cfg.nx, self.cfg.ny
        if axis == 0:
            out = xp.zeros((nx + 1, self.ny_tot))
            out[:, ng:ng + ny] = a
        else:
            out = xp.zeros((self.nx_tot, ny + 1))
            out[ng:ng + nx, :] = a
        return out

    def _rad_mask(self):
        """interior cells the thin radiation acts on besides its own criteria (None: all; AMR coarse level: not
        under a patch)"""
        return None

    def _split_ghosts(self):
        """ghost-inclusive U whose first ghost ring the split conduction step conducts to (None: domain walls)"""
        return None

    def _fld_ghosts(self):
        """FLD ghost cells beyond the edges (see fld.step), None: domain boundaries"""
        return None

    def div_b_max(self):
        """max |div B| h / max |B| on interior cells: face divergence with CT, central differences with GLM"""
        cfg, ng = self.cfg, self.ng
        if self.ct and self.cyl:
            self._ct_sync()
            Bx, By = self.Bf
            rf, rc = self.rf[ng:ng + cfg.nx + 1, None], self.rc[ng:ng + cfg.nx, None]
            rB = Bx[ng:ng + cfg.nx + 1, ng:ng + cfg.ny] * rf
            d = (rB[1:] - rB[:-1]) / (rc * cfg.dx) + (By[ng:ng + cfg.nx, ng + 1:ng + cfg.ny + 1]
                                                      - By[ng:ng + cfg.nx, ng:ng + cfg.ny]) / cfg.dy
        elif self.ct:
            self._ct_sync()
            d = ct.div_faces(self.Bf, (cfg.dx, cfg.dy), ng, (cfg.nx, cfg.ny))
        else:
            W = cons_to_prim(self.U, self.eos)
            d = ((W[iBX][ng+1:-ng+1 or None, ng:-ng] - W[iBX][ng-1:-ng-1, ng:-ng]) / (2 * cfg.dx)
                 + (W[iBY][ng:-ng, ng+1:-ng+1 or None] - W[iBY][ng:-ng, ng-1:-ng-1]) / (2 * cfg.dy))
        Bm = float(xp.max(xp.sqrt(self.U[BX]**2 + self.U[BY]**2 + self.U[BZ]**2)))
        return float(xp.max(xp.abs(d))) * min(cfg.dx, cfg.dy) / max(Bm, 1e-300)

    def compute_rhs(self, U, out=None, Bf=None, dBf=None, S=None, dS=None):
        cfg = self.cfg; ng = self.ng; nx = cfg.nx; ny = cfg.ny
        g = cfg.gamma; ch = self._glm_ch_frozen
        inner = (slice(ng, ng + nx), slice(ng, ng + ny))
        if self.ct:
            ch = 0.0
            ct.cells_from_faces(U, Bf, inner)
        U = self.apply_bc(U)
        if self.ct:
            self._ghost_faces(Bf, U)
        eos = self.eos
        W = cons_to_prim(U, eos)
        if self.strength:
            stren.bc(S, ng, nx, ny, self._periodic(), self.cyl)
            self._halo(S)
            spalled = (S[stren.DMG] > 0.5) & (W[iCLR] > 0.5)       # spalled liner carries no tension
            W[iPR] = xp.where(spalled, xp.maximum(W[iPR], 0.0), W[iPR])
        WLx, WRx = muscl_x(W, eos); s = WLx.shape
        if self.ct:                                  # normal field on a face is the face value
            WLx[iBX] = WRx[iBX] = Bf[0][2:self.nx_tot - 1, :]
        Fx, _ = hlld_flux_x(xp.ascontiguousarray(WLx.reshape(NVAR, -1)),
                             xp.ascontiguousarray(WRx.reshape(NVAR, -1)), eos, ch)
        Fx = Fx.reshape(s)
        if self.walls is not None:
            self.walls.apply(Fx, WLx, WRx, 0)
        WLy, WRy = muscl_y(W, eos); s2 = WLy.shape
        if self.ct:
            WLy[iBY] = WRy[iBY] = Bf[1][:, 2:self.ny_tot - 1]
        Fy, _ = hlld_flux_y(xp.ascontiguousarray(WLy.reshape(NVAR, -1)),
                             xp.ascontiguousarray(WRy.reshape(NVAR, -1)), eos, ch)
        Fy = Fy.reshape(s2)
        if self.walls is not None:
            self.walls.apply(Fy, WLy, WRy, 1)
        self._flux_hook(Fx, Fy, WLx, WRx, WLy, WRy)
        if self.cyl:                                 # (1/r) d(r F)/dr with face radii
            rfx = self.rf[2:self.nx_tot - 1][None, :, None]
            rFx = Fx * rfx
            dFx = (rFx[:, 1:1+nx, :] - rFx[:, 0:nx, :]) / self.rc[ng:ng+nx][None, :, None]
            # B_phi: (curl E)_phi = d_z E_r - d_r E_z has no metric factor, so plain differences (conserves the
            # toroidal flux sum B_phi dr dz; the 3D cylindrical CT update on theta-faces is the same)
            dFx[BZ] = Fx[BZ, 1:1+nx, :] - Fx[BZ, 0:nx, :]
        else:
            dFx = Fx[:, 1:1+nx, :] - Fx[:, 0:nx, :]
        dFy = Fy[:, :, 1:1+ny] - Fy[:, :, 0:ny]
        R = xp.zeros_like(U) if out is None else out
        R[...] = 0.0
        R[:, ng:ng+nx, ng:ng+ny] -= dFx[:, :, ng:ng+ny] / cfg.dx
        R[:, ng:ng+nx, ng:ng+ny] -= dFy[:, ng:ng+nx, :] / cfg.dy
        if self._vf:
            # volume fraction: da/dt + u.grad(a) = 0, written as div(u a) - a div(u) with the face a
            # the kernel used for the energy flux (F_a = mass flux * a_upwind), so uniform p and u stay
            # uniform across the material interface
            R[RHOC, ng:ng+nx, ng:ng+ny] = (self._vf_rate(Fx, WLx, WRx, W, 0, cfg.dx, self.cyl)[:, ng:ng+ny]
                                          + self._vf_rate(Fy, WLy, WRy, W, 1, cfg.dy)[ng:ng+nx, :])
        # Net energy inflow through the domain faces, from the same fluxes that
        # update U[EN]. Fx[:, 0] / Fx[:, nx] are the x_min / x_max faces.
        if self.cyl:
            r0, r1 = self._r0, self._r1
            boundary_rate = (xp.sum(r0 * Fx[EN, 0, ng:ng+ny] - r1 * Fx[EN, nx, ng:ng+ny]) * cfg.dy
                             + xp.sum((Fy[EN, ng:ng+nx, 0] - Fy[EN, ng:ng+nx, ny]) * self.rc[ng:ng+nx]) * cfg.dx)
        else:
            boundary_rate = (xp.sum(Fx[EN, 0, ng:ng+ny] - Fx[EN, nx, ng:ng+ny]) * cfg.dy
                             + xp.sum(Fy[EN, ng:ng+nx, 0] - Fy[EN, ng:ng+nx, ny]) * cfg.dx)
        if self.cyl:
            # geometric sources (x = r, y = z, third = -phi): radial momentum (rho w^2 - B_w^2 + p_T)/r,
            # azimuthal momentum -(rho u w - B_x B_w)/r (B_phi needs none, see dFx above)
            Wi = W[:, ng:ng+nx, ng:ng+ny]
            ri = self.rc[ng:ng+nx][:, None]
            pT = Wi[iPR] + 0.5 * (Wi[iBX]**2 + Wi[iBY]**2 + Wi[iBZ]**2)
            R[MX, ng:ng+nx, ng:ng+ny] += (Wi[iRHO] * Wi[iVZ]**2 - Wi[iBZ]**2 + pT) / ri
            R[MZ, ng:ng+nx, ng:ng+ny] -= (Wi[iRHO] * Wi[iVX] * Wi[iVZ] - Wi[iBX] * Wi[iBZ]) / ri
        powell_rate = xp.zeros(())
        ohmic_rate = xp.zeros(())
        # Powell source for ANY MHD case (uniform or striped); not with CT
        if cfg.powell_source and cfg.is_mhd() and not self.ct:
            S = self._compute_powell_source(U, W)
            R[:, ng:ng+nx, ng:ng+ny] += S[:, ng:ng+nx, ng:ng+ny]
            powell_rate = xp.sum(S[EN, ng:ng+nx, ng:ng+ny]) * cfg.dx * cfg.dy
        # Resistive MHD: -curl(eta J) in induction, -div(eta J x B) in energy
        eta = None
        if cfg.is_resistive():
            eta = resistivity_field(W, cfg)
            rc_, rf_ = (self.rc, self.rf) if self.cyl else (None, None)
            R_resist, Gx, Gy, q, (Gbx, Gby) = compute_resistive_terms(U, eta, cfg.dx, cfg.dy, cfg.eta_face_mean,
                                                                      rc_, rf_, bz_flux=True)
            R[:, ng:ng+nx, ng:ng+ny] += R_resist[:, ng:ng+nx, ng:ng+ny]
            if self._records_fluxes:                 # face k of Fx is face ng - 1 + k of the full arrays
                self._add_flux(EN, Gx[ng-1:ng+nx], Gy[:, ng-1:ng+ny])
                self._add_flux(BZ, Gbx[ng-1:ng+nx], Gby[:, ng-1:ng+ny])
            ohmic_rate = xp.sum(q[ng:ng+nx, ng:ng+ny] * self.dV)
            if cfg.two_temp:                         # Ohmic heat goes to the electrons
                rho_i = W[iRHO, ng:ng+nx, ng:ng+ny]
                R[RHOSE, ng:ng+nx, ng:ng+ny] += (cfg.gamma - 1) * q[ng:ng+nx, ng:ng+ny] * rho_i**(1 - cfg.gamma)
            if self.cyl:
                boundary_rate = boundary_rate + (
                    xp.sum(self._r0 * Gx[ng-1, ng:ng+ny] - self._r1 * Gx[ng+nx-1, ng:ng+ny]) * cfg.dy
                    + xp.sum((Gy[ng:ng+nx, ng-1] - Gy[ng:ng+nx, ng+ny-1]) * self.rc[ng:ng+nx]) * cfg.dx)
            else:
                boundary_rate = boundary_rate + (
                    xp.sum(Gx[ng-1, ng:ng+ny] - Gx[ng+nx-1, ng:ng+ny]) * cfg.dy
                    + xp.sum(Gy[ng:ng+nx, ng-1] - Gy[ng:ng+nx, ng+ny-1]) * cfg.dx)
        if self.ct:
            # in-plane B lives on the faces: dB/dt = -curl E with E_z on the cell corners
            R[BX] = 0.0
            R[BY] = 0.0
            E = fused.emf2d(Fx, Fy, W, nx, ny, cfg.dx, cfg.dy)
            if cfg.nernst_coeff > 0:
                # Nernst EMF (and the out-of-plane field it carries), Poynting + Ettingshausen energy flux
                En, dBw, Gnx, Gny, _, divqE, FBw, FqE = transport.nernst_2d(W, Bf, cfg.nernst_coeff, cfg.cond_x0,
                                                                            cfg.cond_tracer_max, cfg.dx, cfg.dy,
                                                                            nx, ny)
                if self._records_fluxes:             # AMR: B_w flux, Poynting + Ettingshausen, electron share
                    self._add_flux(BZ, *FBw)
                    self._add_flux(EN, Gnx, Gny)
                    if cfg.two_temp:
                        self._add_flux("Fe", *FqE)
                if cfg.two_temp:                     # the Ettingshausen heat flux is carried by the electrons
                    rho_i = W[iRHO, ng:ng+nx, ng:ng+ny]
                    R[RHOSE, ng:ng+nx, ng:ng+ny] -= (cfg.gamma - 1) * divqE * rho_i**(1 - cfg.gamma)
                E = E + En
                R[BZ, ng:ng+nx, ng:ng+ny] += dBw
                if self.cyl:
                    rG = Gnx * self.rf[2:nx + 3][:, None]
                    R[EN, ng:ng+nx, ng:ng+ny] -= (rG[1:, ng:ng+ny] - rG[:-1, ng:ng+ny]) / (cfg.dx * self.rc[ng:ng+nx][:, None])
                    boundary_rate = boundary_rate + (
                        xp.sum(self._r0 * Gnx[0, ng:ng+ny] - self._r1 * Gnx[nx, ng:ng+ny]) * cfg.dy
                        + xp.sum((Gny[ng:ng+nx, 0] - Gny[ng:ng+nx, ny]) * self.rc[ng:ng+nx]) * cfg.dx)
                else:
                    R[EN, ng:ng+nx, ng:ng+ny] -= (Gnx[1:, ng:ng+ny] - Gnx[:-1, ng:ng+ny]) / cfg.dx
                    boundary_rate = boundary_rate + (
                        xp.sum(Gnx[0, ng:ng+ny] - Gnx[nx, ng:ng+ny]) * cfg.dy
                        + xp.sum(Gny[ng:ng+nx, 0] - Gny[ng:ng+nx, ny]) * cfg.dx)
                R[EN, ng:ng+nx, ng:ng+ny] -= (Gny[ng:ng+nx, 1:] - Gny[ng:ng+nx, :-1]) / cfg.dy
            if eta is not None:
                J = ct.edge_current([f[..., None] for f in Bf], 0, 1, cfg.dx, cfg.dy, nx, ny)[..., 0]
                E = E + ct.edge_eta(eta, nx, ny) * J
            if hall.active(cfg):                     # Hall term, anisotropic resistivity (aegis/hall.py)
                Ek, Fbx, Fby, Sx, Sy, qa = hall.ohm_extra(W, Bf, eta, cfg, cfg.dx, cfg.dy, nx, ny, ng)
                E = E + Ek
                R[BZ, ng:ng+nx, ng:ng+ny] -= (Fbx[1:] - Fbx[:-1]) / cfg.dx + (Fby[:, 1:] - Fby[:, :-1]) / cfg.dy
                R[EN, ng:ng+nx, ng:ng+ny] -= (Sx[1:] - Sx[:-1]) / cfg.dx + (Sy[:, 1:] - Sy[:, :-1]) / cfg.dy
                boundary_rate = boundary_rate + (xp.sum(Sx[0] - Sx[nx]) * cfg.dy + xp.sum(Sy[:, 0] - Sy[:, ny]) * cfg.dx)
                if cfg.two_temp and qa is not None:  # anisotropic Ohmic heat goes to the electrons
                    rho_i = W[iRHO, ng:ng+nx, ng:ng+ny]
                    R[RHOSE, ng:ng+nx, ng:ng+ny] += (cfg.gamma - 1) * qa * rho_i**(1 - cfg.gamma)
                if self._records_fluxes:             # interior-row face arrays -> the Fx / Fy layout
                    self._add_flux(EN, self._pad_faces(Sx, 0), self._pad_faces(Sy, 1))
                    self._add_flux(BZ, self._pad_faces(Fbx, 0), self._pad_faces(Fby, 1))
            E = self._emf_hook(E)
            dBf[0][...] = 0.0
            dBf[1][...] = 0.0
            if self.cyl:
                E[0, :] = 0.0                        # E_phi = 0 on the axis
                dBf[0][2:nx + 3, 2:ny + 2] = -(E[:, 1:] - E[:, :-1]) / cfg.dy
                rE = E * self.rf[2:nx + 3][:, None]
                dBf[1][2:nx + 2, 2:ny + 3] = (rE[1:, :] - rE[:-1, :]) / (cfg.dx * self.rc[2:nx + 2][:, None])
            else:
                dBf[0][2:nx + 3, 2:ny + 2] = -(E[:, 1:] - E[:, :-1]) / cfg.dy
                dBf[1][2:nx + 2, 2:ny + 3] = (E[1:, :] - E[:-1, :]) / cfg.dx
        if self.strength:
            G = cfg.shear_modulus * xp.clip(W[iCLR], 0, 1) * (1 - S[stren.DMG])
            rc_, rf_ = (self.rc, self.rf) if self.cyl else (None, None)
            dSi, fx, fy, hoop = stren.rates(S, W, Fx, Fy, WLx, WRx, WLy, WRy, G, (cfg.dx, cfg.dy), ng, (nx, ny),
                                            rc_, rf_)
            dS[...] = 0.0
            dS[:, ng:ng+nx, ng:ng+ny] = dSi
            for var, gx, gy in ((MX, fx[0], fy[0]), (MY, fx[1], fy[1]), (EN, fx[2], fy[2])):
                if self._records_fluxes:
                    self._add_flux(var, gx, gy)
                if self.cyl:
                    rg = gx * self.rf[2:nx + 3][:, None]
                    R[var, ng:ng+nx, ng:ng+ny] -= (rg[1:, ng:ng+ny] - rg[:-1, ng:ng+ny]) / (cfg.dx * self.rc[ng:ng+nx][:, None])
                else:
                    R[var, ng:ng+nx, ng:ng+ny] -= (gx[1:, ng:ng+ny] - gx[:-1, ng:ng+ny]) / cfg.dx
                R[var, ng:ng+nx, ng:ng+ny] -= (gy[ng:ng+nx, 1:] - gy[ng:ng+nx, :-1]) / cfg.dy
            if self.cyl:
                R[MX, ng:ng+nx, ng:ng+ny] += hoop
                r0, r1 = self._r0, self._r1
                boundary_rate = boundary_rate + (
                    xp.sum(r0 * fx[2][0, ng:ng+ny] - r1 * fx[2][nx, ng:ng+ny]) * cfg.dy
                    + xp.sum((fy[2][ng:ng+nx, 0] - fy[2][ng:ng+nx, ny]) * self.rc[ng:ng+nx]) * cfg.dx)
            else:
                boundary_rate = boundary_rate + (
                    xp.sum(fx[2][0, ng:ng+ny] - fx[2][nx, ng:ng+ny]) * cfg.dy
                    + xp.sum(fy[2][ng:ng+nx, 0] - fy[2][ng:ng+nx, ny]) * cfg.dx)
        if cfg.gyro_di > 0:                          # ion gyroviscosity (aegis/ions.py)
            gfx, gfy = ions.gyroviscous_fluxes(W, U, cfg, cfg.dx, cfg.dy, nx, ny, ng)
            for var, gx, gy in ((MX, gfx[0], gfy[0]), (MY, gfx[1], gfy[1]), (MZ, gfx[2], gfy[2]), (EN, gfx[3], gfy[3])):
                R[var, ng:ng+nx, ng:ng+ny] -= ((gx[1:, ng:ng+ny] - gx[:-1, ng:ng+ny]) / cfg.dx
                                               + (gy[ng:ng+nx, 1:] - gy[ng:ng+nx, :-1]) / cfg.dy)
                if self._records_fluxes:
                    self._add_flux(var, gx, gy)
            boundary_rate = boundary_rate + (xp.sum(gfx[3][0, ng:ng+ny] - gfx[3][nx, ng:ng+ny]) * cfg.dy
                                             + xp.sum(gfy[3][ng:ng+nx, 0] - gfy[3][ng:ng+nx, ny]) * cfg.dx)
        self._stage_boundary_rate = boundary_rate
        self._stage_powell_rate = powell_rate
        self._stage_ohmic_rate = ohmic_rate
        if self.walls is not None:
            self.walls.mask_rates(R)
        return R

    def _vf_rate(self, F, WL, WR, W, axis, h, radial=False):
        """-(div(u a) - a div u) along one axis, on interior cells along that axis"""
        ng, n = self.ng, (self.cfg.nx if axis == 0 else self.cfg.ny)
        a_c = W[iCLR, ng:ng+n, :] if axis == 0 else W[iCLR, :, ng:ng+n]
        if radial:
            return vf_advection_rate(F, WL, WR, a_c, axis, h, self.rf[2:self.nx_tot - 1][:, None],
                                     self.rc[ng:ng+n][:, None])
        return vf_advection_rate(F, WL, WR, a_c, axis, h)

    def _accumulate_energy_budget(self, weight):
        self._acc_boundary_energy += weight * self._stage_boundary_rate
        self._acc_powell_energy += weight * self._stage_powell_rate
        self._acc_ohmic += weight * self._stage_ohmic_rate

    def step_ssprk3(self):
        dt = self.dt; self._glm_ch_frozen = self.cfg.glm_ch
        self._ct_sync()
        self._te_sync()
        if self.ct and self.cfg.use_cuda_graphs and xp.__name__ == "cupy" and not getattr(self.eos, "no_graph", False):
            self._rk_graph(dt)
        else:
            self._rk_core(dt)
        self._post_step(dt)

    def _rk_graph(self, dt):
        """the RK core as a captured CUDA graph: captured once (dt read from a device scalar, temporaries from
        a private memory pool the rest of the code never touches), replayed every step; recaptured if U, the
        faces or the stress array are replaced"""
        import cupy as cp
        key = (id(self.U), id(self.Bf[0]), id(getattr(self, "S", None)))
        if getattr(self, "_graph_key", None) != key:
            if getattr(self, "_graph_warm", None) != key:  # first call: run normally so every buffer exists
                self._graph_warm = key
                self._rk_core(dt)
                return
            self._dtd = cp.zeros(())
            self._gpool = cp.cuda.MemoryPool()
            self._gstream = cp.cuda.Stream(non_blocking=True)
            cp.cuda.Device().synchronize()
            with cp.cuda.using_allocator(self._gpool.malloc), self._gstream:
                self._gstream.begin_capture()
                self._rk_core(self._dtd)
                self._graph = self._gstream.end_capture()
            self._graph_key = key
        self._dtd.fill(dt)
        cp.cuda.Device().synchronize()
        self._graph.launch(stream=self._gstream)
        self._gstream.synchronize()

    def _rk_core(self, dt):
        """SSP-RK3 stages (U, CT faces, stress); dt is a float or a device scalar"""
        U0 = self.U
        F0, F1, dF = (self.Bf, self._Bf1, self._dBf) if self.ct else (None, None, None)
        S0, S1, dS = (self.S, self._S1, self._dS) if self.strength else (None, None, None)

        # SSP-RK3: U^{n+1} = U^n + dt (L0 + L1 + 4 L2) / 6 (face fields and stress alongside)
        self.compute_rhs(U0, out=self.dU, Bf=F0, dBf=dF, S=S0, dS=dS)
        self._accumulate_energy_budget(dt / 6.0)
        if self.strength:
            xp.add(S0, dt * dS, out=S1)
        if self.ct:
            for f0, f1, d in zip(F0, F1, dF):
                xp.add(f0, dt * d, out=f1)
        xp.multiply(self.dU, dt, out=self.dU)
        xp.add(U0, self.dU, out=self.U_temp)
        self.enforce_scalar_bounds(self.U_temp)
        self.apply_bc(self.U_temp)

        self.compute_rhs(self.U_temp, out=self.dU, Bf=F1, dBf=dF, S=S1, dS=dS)
        self._accumulate_energy_budget(dt / 6.0)
        if self.strength:
            S1[...] = 0.75 * S0 + 0.25 * (S1 + dt * dS)
        if self.ct:
            for f0, f1, d in zip(F0, F1, dF):
                f1[...] = 0.75 * f0 + 0.25 * (f1 + dt * d)
        xp.multiply(self.dU, dt, out=self.dU)
        xp.add(self.U_temp, self.dU, out=self.U_stage)
        xp.multiply(self.U_stage, 0.25, out=self.U_stage)
        xp.multiply(U0, 0.75, out=self.U_temp)
        xp.add(self.U_temp, self.U_stage, out=self.U_temp)
        self.enforce_scalar_bounds(self.U_temp)
        self.apply_bc(self.U_temp)

        self.compute_rhs(self.U_temp, out=self.dU, Bf=F1, dBf=dF, S=S1, dS=dS)
        self._accumulate_energy_budget(2.0 * dt / 3.0)
        if self.strength:
            S0[...] = S0 / 3.0 + 2.0 / 3.0 * (S1 + dt * dS)
        if self.ct:
            for f0, f1, d in zip(F0, F1, dF):
                f0[...] = f0 / 3.0 + 2.0 / 3.0 * (f1 + dt * d)
        xp.multiply(self.dU, dt, out=self.dU)
        xp.add(self.U_temp, self.dU, out=self.U_stage)
        xp.multiply(self.U_stage, 2.0 / 3.0, out=self.U_stage)
        xp.multiply(U0, 1.0 / 3.0, out=self.dU)
        xp.add(self.dU, self.U_stage, out=self.dU)
        xp.copyto(self.U, self.dU)
        self.enforce_scalar_bounds(self.U)
        if self.ct:
            ct.cells_from_faces(self.U, self.Bf, (slice(self.ng, self.ng + self.cfg.nx),
                                                  slice(self.ng, self.ng + self.cfg.ny)))

    def _post_step(self, dt):
        """split steps after the RK stages: kinetic cells, plasticity, radiation, conduction, equilibration, viscosity"""
        if self.cfg.pic:                             # particles replace the fluid ions in the kinetic cells
            if self.pic is None:
                from .hybrid_pic import HybridPIC
                self.pic = HybridPIC(self)
            self.pic.step(dt)
        if self.strength:                            # radial return to the yield surface, spall damage
            Wn = cons_to_prim(self.U, self.eos)
            stren.plastic(self.S, xp.clip(Wn[iCLR], 0, 1), Wn[iPR], self.cfg.yield_strength,
                          self.cfg.spall_strength, self.ng)
        if self.cfg.rad_coeff > 0:
            ng, cfg = self.ng, self.cfg
            inner = (slice(None), slice(ng, ng + cfg.nx), slice(ng, ng + cfg.ny))
            Ui = self.U[inner]
            dE = radiate(Ui, self.eos, cfg.rad_coeff, cfg.rad_T_min, cfg.rad_tracer_max, dt,
                         electron_Z=cfg.z_ion if cfg.two_temp else None, mask=self._rad_mask())
            self.U[inner] = Ui
            self._acc_rad += xp.sum(dE * self.dV)
        if self.cfg.fld:
            ng, cfg = self.ng, self.cfg
            Ui = self.U[:, ng:ng + cfg.nx, ng:ng + cfg.ny]
            radial = (self.rf[ng:ng + cfg.nx + 1], self.rc[ng:ng + cfg.nx]) if self.cyl else None
            faces = [] if self._records_fluxes else None
            if cfg.fld_groups:                       # multigroup (aegis/radiation_multigroup.py)
                _, din, self.fld_iters = radiation_multigroup.step(self.Er, Ui, self.eos, cfg, (cfg.dx, cfg.dy),
                                                                   self._periodic(), dt, radial=radial,
                                                                   advect=cfg.fld_advect)
            else:
                _, din, self.fld_iters = fld.step(self.Er, Ui, self.eos, cfg, (cfg.dx, cfg.dy), self._periodic(), dt,
                                                 radial=radial, advect=cfg.fld_advect, ghost=self._fld_ghosts(),
                                                 faces=faces)
            self.fld_inflow += din
            if faces:
                self._add_split_flux("Er", None, *faces)
        if self.cfg.cond_kappa0 > 0:
            ng, cfg = self.ng, self.cfg
            Ui = self.U[:, ng:ng + cfg.nx, ng:ng + cfg.ny]
            per = (cfg.get_bc_x() == "periodic", cfg.bc_y_type == "periodic")
            radial = (self.rf[ng:ng + cfg.nx + 1], self.rc[ng:ng + cfg.nx]) if self.cyl else None
            wall = cfg.cond_wall_T if cfg.cond_wall == "isothermal" else None
            faces = [] if self._records_fluxes else None
            self.cond_stages = conduct(Ui, self.eos, cfg.cond_kappa0, cfg.cond_x0, cfg.cond_tracer_max,
                                       (cfg.dx, cfg.dy), per, dt, radial=radial, wall_T=wall, perp=cfg.cond_perp,
                                       electron_Z=cfg.z_ion if cfg.two_temp else None,
                                       ghost_U=self._split_ghosts(), flux_out=faces)
            if faces:
                self._add_split_flux("F", EN, *faces)
                if cfg.two_temp:                     # all of it is electron heat
                    self._add_split_flux("Fe", None, *faces)
        cfg, ng = self.cfg, self.ng
        if cfg.cond_kappa_i0 > 0:                    # ion conduction on T_i (two-temperature)
            Ui = self.U[:, ng:ng + cfg.nx, ng:ng + cfg.ny]
            per = (cfg.get_bc_x() == "periodic", cfg.bc_y_type == "periodic")
            faces = [] if self._records_fluxes else None
            self.cond_i_stages = conduct(Ui, self.eos, cfg.cond_kappa_i0, cfg.cond_x0_i, cfg.cond_tracer_max,
                                         (cfg.dx, cfg.dy), per, dt, perp="simple", ion_Z=cfg.z_ion,
                                         ghost_U=self._split_ghosts(), flux_out=faces)
            if faces:
                self._add_split_flux("F", EN, *faces)
        if cfg.two_temp and cfg.te_eq_coeff > 0:
            Ui = self.U[:, ng:ng + cfg.nx, ng:ng + cfg.ny]
            fuel = xp.clip(cons_to_prim(Ui, self.eos)[iCLR], 0, 1) < cfg.cond_tracer_max
            twotemp.equilibrate(Ui, cfg.gamma, cfg.z_ion, cfg.te_eq_coeff, fuel, dt)
        if cfg.visc_coeff > 0:
            Ui = self.U[:, ng:ng + cfg.nx, ng:ng + cfg.ny]
            Wi = cons_to_prim(Ui, self.eos)
            fuel = xp.clip(Wi[iCLR], 0, 1) < cfg.cond_tracer_max
            if cfg.two_temp:
                theta_i = twotemp.temperatures(Ui, cfg.gamma, cfg.z_ion)[1]
            else:
                theta_i = Wi[iPR] / Wi[iRHO]
            eta = cfg.visc_coeff * xp.maximum(theta_i, 0.0)**2.5
            ghost, faces = None, ([] if self._records_fluxes else None)
            Ug = self._split_ghosts()
            if Ug is not None:                       # AMR patch: neighbour ring for v, B, eta and the fuel mask
                ring = Ug[:, ng - 1:ng + cfg.nx + 1, ng - 1:ng + cfg.ny + 1]
                Wr = cons_to_prim(ring, self.eos)
                ti_r = twotemp.temperatures(ring, cfg.gamma, cfg.z_ion)[1] if cfg.two_temp else Wr[iPR] / Wr[iRHO]
                ghost = ([Wr[iVX], Wr[iVY], Wr[iVZ]], [Wr[iBX], Wr[iBY], Wr[iBZ]],
                         cfg.visc_coeff * xp.maximum(ti_r, 0.0)**2.5,
                         (xp.clip(Wr[iCLR], 0, 1) < cfg.cond_tracer_max).astype(Wr.dtype))
            self.visc_stages = viscous_step(Ui, eta, (cfg.dx, cfg.dy), self._periodic(), fuel, dt, ghost=ghost,
                                            flux_out=faces)
            if faces:
                for k, var in enumerate((MX, MY, MZ, EN)):
                    self._add_split_flux("F", var, faces[0][k], faces[1][k])
        self.apply_bc(self.U)
        if self.ct:
            self._ghost_faces(self.Bf, self.U)
        if self._glm_ch_frozen > 0 and not self.ct:
            decay = np.exp(-self.cfg.glm_alpha * self._glm_ch_frozen * dt / min(self.cfg.dx, self.cfg.dy))
            self.U[PSI] *= decay

    def energy_budget(self):
        """Energy balance over the run: E(t) - E(0) = inflow + Powell - radiated + residual."""
        e0 = self.diag_energy_total[0]; ef = self.diag_energy_total[-1]
        inflow = self._cumulative_boundary_energy
        powell = self._cumulative_powell_energy
        rad = self.cumulative_radiated_energy
        scale = max(abs(e0), 1e-14)
        return {
            "E0": e0, "E": ef, "dE": ef - e0,
            "boundary_inflow": inflow, "powell": powell, "radiated": rad,
            "raw_change_rel": (ef - e0) / scale,
            "powell_rel": powell / scale,
            "conservation_error_rel": (ef - e0 - inflow - powell + rad) / scale,
        }

    def energy_budget_report(self):
        b = self.energy_budget()
        return [
            f"  Energy budget: E0={b['E0']:.6e}  E={b['E']:.6e}  (change {100*b['raw_change_rel']:+.2f}% "
            f"of E0, all of it accounted for by boundary inflow {b['boundary_inflow']:+.6e} + Powell)",
            f"  Powell source (non-conservative by design): {100*b['powell_rel']:+.3e}% of E0",
            f"  Conservation error |dE - inflow - Powell|/E0: {abs(b['conservation_error_rel']):.2e}",
        ]

    def save_snapshot(self, label=None):
        ng = self.ng
        W = cons_to_prim(self.U, self.eos)
        key = label or f"t={self.t:.4f}"
        self.snapshots[key] = {
            'rho': to_numpy(W[iRHO, ng:-ng, ng:-ng]),
            'p': to_numpy(W[iPR, ng:-ng, ng:-ng]),
            'vx': to_numpy(W[iVX, ng:-ng, ng:-ng]),
            'vy': to_numpy(W[iVY, ng:-ng, ng:-ng]),
            'Bx': to_numpy(W[iBX, ng:-ng, ng:-ng]),
            'By': to_numpy(W[iBY, ng:-ng, ng:-ng]),
            'Bz': to_numpy(W[iBZ, ng:-ng, ng:-ng]),
            'C': to_numpy(W[iCLR, ng:-ng, ng:-ng]),
            't': self.t,
            'x': to_numpy(self.x[ng:-ng]),
            'y': to_numpy(self.y[ng:-ng]),
        }

    def run(self):
        cfg = self.cfg; g = cfg.gamma
        field_desc = "STRIPED" if cfg.is_striped() else ("UNIFORM" if cfg.is_mhd() else "HYDRO")
        print(f"\n--- Running (GPU) [{field_desc}]: B0 = {cfg.B_transverse} ---")
        sys.stdout.flush()
        t0 = time.time()
        _record_diagnostics(self)
        self.save_snapshot(label="initial")
        while self.t < cfg.t_end and self.step < cfg.max_steps:
            W = cons_to_prim(self.U, self.eos)
            dt_cfl = self.compute_dt(W)
            self.dt = min(dt_cfl, cfg.t_end - self.t)
            if self.dt <= 1e-16: break
            self.step_ssprk3()
            self.t += self.dt; self.step += 1
            if self.step % cfg.diag_interval == 0:
                _record_diagnostics(self)
            for st in cfg.snapshot_times:
                lbl = f"t~{st:.3f}"
                if abs(self.t - st) < 1.5*self.dt and lbl not in self.snapshots:
                    self.save_snapshot(label=lbl)
            if self.step % 200 == 0:
                mwi = self.diag_mixing_width_integral[-1] if self.diag_mixing_width_integral else 0
                theta = self.diag_mixedness[-1] if self.diag_mixedness else 0
                jvx = self.diag_jet_max_vx[-1] if self.diag_jet_max_vx else 0
                col = self.diag_jet_collimation[-1] if self.diag_jet_collimation else 0
                print(f"  Step {self.step:5d}  t={self.t:.4f}  dt={self.dt:.2e}  "
                      f"MW={mwi:.4f}  theta={theta:.3f}  jet_vx={jvx:.2f}  collim={col:.3f}")
                sys.stdout.flush()
        _record_diagnostics(self)
        self.save_snapshot(label="final")
        elapsed = time.time() - t0
        if len(self.diag_energy_total) >= 2:
            for line in self.energy_budget_report():
                print(line)
        if self.diag_divB_max:
            print(f"  Final max|divB|: {self.diag_divB_max[-1]:.3g}, L2: {self.diag_divB_L2[-1]:.3g}, "
                  f"h*max|divB|/max|B|: {self.diag_divB_rel_max[-1]:.3g}")
        print(f"  Done: {self.step} steps, {elapsed:.1f}s")
        sys.stdout.flush()


print("GPU Solver OK")


# ============================================================
# PostProcessor - figures and animations
# ============================================================
class PostProcessor:
    """Generate all analysis plots including Magnetic Nozzle / Sieve visualizations."""

    def __init__(self, solvers_dict):
        self.solvers = solvers_dict
        self._colors = ['C3', 'C2', 'C0', 'C4', 'C5', 'C6', 'C7']
        self._styles = ['-', '--', '-.', ':', '-', '--', '-.']
        self._saved_files = []

    @staticmethod
    def _final(s):
        return s.snapshots.get('final', list(s.snapshots.values())[-1])

    def _save(self, fig, filename):
        ok, path, sz = save_figure(fig, filename)
        self._saved_files.append((filename, ok, sz))
        return ok, path, sz

    def _get_smoothing(self):
        for s in self.solvers.values():
            return s.cfg.enable_smoothing
        return False

    # ---- ORIGINAL PLOTS (preserved) ----

    def plot_density_comparison(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, self.solvers.items()):
            snap = self._final(solver); rho = snap['rho']
            im = ax.pcolormesh(snap['x'], snap['y'], rho.T, cmap='inferno', shading='auto',
                               vmin=rho.min(), vmax=np.percentile(rho, 98))
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\rho$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Density Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'density_comparison.png')

    def plot_passive_scalar(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, self.solvers.items()):
            snap = self._final(solver); C = np.clip(snap['C'], 0, 1)
            im = ax.pcolormesh(snap['x'], snap['y'], C.T, cmap='coolwarm', shading='auto', vmin=0, vmax=1)
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label='$C$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Passive Scalar (Mixing Tracer)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'passive_scalar.png')

    def plot_schlieren(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, self.solvers.items()):
            snap = self._final(solver); rho = snap['rho']; dx, dy = solver.cfg.dx, solver.cfg.dy
            grad_rho = np.sqrt(np.gradient(rho, dx, axis=0)**2 + np.gradient(rho, dy, axis=1)**2)
            schlieren = np.log10(grad_rho / np.maximum(rho, 1e-10) + 1e-10)
            vmin_s = np.percentile(schlieren, 3); vmax_s = np.percentile(schlieren, 99)
            im = ax.pcolormesh(snap['x'], snap['y'], schlieren.T, cmap='gray_r', shading='auto', vmin=vmin_s, vmax=vmax_s)
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}: Schlieren", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82)
        axes[0].set_ylabel('$y$')
        fig.suptitle('Numerical Schlieren', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'schlieren.png')

    def plot_vorticity_comparison(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]; omegas, snaps = [], []
        for label, solver in self.solvers.items():
            snap = self._final(solver); snaps.append(snap)
            omegas.append(np.gradient(snap['vy'], solver.cfg.dx, axis=0) - np.gradient(snap['vx'], solver.cfg.dy, axis=1))
        vmax_om = max(np.percentile(np.abs(om), 99) for om in omegas)
        vmax_om = max(vmax_om, 1e-10)
        for ax, (label, _), snap, omega in zip(axes, self.solvers.items(), snaps, omegas):
            im = ax.pcolormesh(snap['x'], snap['y'], omega.T, cmap='RdBu_r', shading='auto', vmin=-vmax_om, vmax=vmax_om)
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\omega_z$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Vorticity Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'vorticity.png')

    def plot_magnetic_pressure_with_fieldlines(self):
        mhd = {k: v for k, v in self.solvers.items() if v.cfg.is_mhd()}
        if not mhd: return
        n = len(mhd)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, mhd.items()):
            snap = self._final(solver)
            Pmag = 0.5*(snap['Bx']**2 + snap['By']**2 + snap['Bz']**2)
            im = ax.pcolormesh(snap['x'], snap['y'], Pmag.T, cmap='plasma', shading='auto')
            try:
                ax.streamplot(snap['x'], snap['y'], snap['Bx'].T, snap['By'].T,
                              color='white', linewidth=0.5, density=1.2, arrowsize=0.5)
            except Exception: pass
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$P_B$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Magnetic Pressure & Field Lines', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'mag_pressure.png')

    def plot_diagnostics(self):
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        colors, styles = self._colors, self._styles
        sm_en = self._get_smoothing(); ns = 5; pairs = list(self.solvers.items())

        for ax_idx, (diag_key, ylabel, title) in enumerate([
            ('diag_mixing_width_integral', r'$W$', 'Integral Mixing Width'),
            ('diag_mixing_width_thresh', 'Threshold Width', 'Threshold MW'),
            ('diag_mixedness', r'$\theta$', 'Mixedness'),
            ('diag_perturbation_amp', r'$a_k$', 'Perturbation Amplitude'),
        ]):
            ax = axes[0, ax_idx]
            for i, (label, s) in enumerate(pairs):
                t = np.array(s.diag_times); val = np.array(getattr(s, diag_key))
                ax.plot(t, smooth(val, ns, sm_en), color=colors[i%7], ls=styles[i%7], lw=2, label=label)
            ax.set_xlabel('$t$'); ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)
            if 'mixedness' in diag_key: ax.set_ylim(0, 1)
            else: ax.set_ylim(bottom=0)

        for ax_idx, (diag_key, ylabel, title) in enumerate([
            ('diag_enstrophy_local', r'$\langle\rho\omega_z^2\rangle$', 'Local Enstrophy'),
            ('diag_jet_max_vx', r'$v_x^{max}$', 'Peak Jet Velocity'),
            ('diag_jet_collimation', 'Collimation', 'Gap Momentum Fraction'),
            ('diag_energy_total', '$E_{tot}$', 'Total Energy'),
        ]):
            ax = axes[1, ax_idx]
            for i, (label, s) in enumerate(pairs):
                t = np.array(s.diag_times); val = np.array(getattr(s, diag_key))
                ax.plot(t, val, color=colors[i%7], ls=styles[i%7], lw=2, label=label)
            ax.set_xlabel('$t$'); ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)

        plt.tight_layout()
        self._save(fig, 'diagnostics.png')

    def plot_evolution(self):
        n_cases = len(self.solvers); n_cols = 4
        fig, axes = plt.subplots(n_cases, n_cols, figsize=(4*n_cols, 3*n_cases), sharex=True, sharey=True, squeeze=False)
        for row, (label, solver) in enumerate(self.solvers.items()):
            keys = sorted(solver.snapshots.keys(), key=lambda k: solver.snapshots[k]['t'])
            if len(keys) > n_cols:
                idx = np.linspace(0, len(keys)-1, n_cols, dtype=int)
                keys = [keys[i] for i in idx]
            for col in range(min(n_cols, len(keys))):
                ax = axes[row, col]; snap = solver.snapshots[keys[col]]; rho = snap['rho']
                ax.pcolormesh(snap['x'], snap['y'], rho.T, cmap='inferno', shading='auto', vmin=0.5, vmax=rho.max()*0.95)
                ax.set_title(f"$t={snap['t']:.3f}$", fontsize=10); ax.set_aspect('equal')
                if col == 0: ax.set_ylabel(f"{label}\n$y$")
                if row == n_cases-1: ax.set_xlabel('$x$')
        fig.suptitle('Density Evolution', fontsize=13, fontweight='bold')
        plt.tight_layout()
        self._save(fig, 'evolution.png')

    def plot_interface_shape(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (label, solver) in enumerate(self.solvers.items()):
            snap = self._final(solver); C = np.clip(snap['C'], 0, 1)
            ax.contour(snap['x'], snap['y'], C.T, levels=[0.5],
                       colors=[self._colors[i%7]], linewidths=2)
            ax.plot([], [], color=self._colors[i%7], lw=2, label=label)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_title('Interface Shape ($C=0.5$ contour)', fontweight='bold')
        ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        plt.tight_layout()
        self._save(fig, 'interface_shape.png')

    # ---- NEW: MAGNETIC NOZZLE / SIEVE PLOTS ----

    def plot_nozzle_velocity_map(self):
        """Downstream x-velocity map showing jet formation at gaps."""
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, self.solvers.items()):
            snap = self._final(solver); vx = snap['vx']
            im = ax.pcolormesh(snap['x'], snap['y'], vx.T, cmap='hot', shading='auto',
                               vmin=0, vmax=np.percentile(vx, 99))
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$v_x$')
            # Overlay bar/gap regions for striped cases
            if solver.cfg.is_striped():
                Ly = solver.cfg.y_max - solver.cfg.y_min
                k_mod = 2*np.pi*solver.cfg.B_modulation_mode / Ly
                y_arr = snap['y']
                for yn in y_arr[::1]:
                    if abs(np.cos(k_mod * yn + solver.cfg.B_stripe_phase)) < 0.1:
                        ax.axhline(yn, color='cyan', lw=0.5, alpha=0.4)
        axes[0].set_ylabel('$y$')
        fig.suptitle('X-Velocity: Jet Formation Map', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'nozzle_velocity_map.png')

    def plot_momentum_flux_comparison(self):
        """Bar chart comparing momentum flux at bars vs gaps for all cases."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = self._colors; pairs = list(self.solvers.items())

        ax = axes[0]
        for i, (label, s) in enumerate(pairs):
            t = np.array(s.diag_times)
            ax.plot(t, s.diag_jet_momentum_flux_gap, color=colors[i%7], ls='-', lw=2, label=f'{label} (gap)')
            ax.plot(t, s.diag_jet_momentum_flux_bar, color=colors[i%7], ls=':', lw=1.5, alpha=0.6, label=f'{label} (bar)')
        ax.set_xlabel('Time $t$'); ax.set_ylabel(r'$\langle \rho v_x^2 \rangle$')
        ax.set_title('Momentum Flux: Gaps vs Bars', fontweight='bold')
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[1]
        for i, (label, s) in enumerate(pairs):
            t = np.array(s.diag_times)
            gap = np.array(s.diag_jet_momentum_flux_gap)
            bar = np.array(s.diag_jet_momentum_flux_bar)
            ratio = gap / np.maximum(bar, 1e-14)
            ax.plot(t, ratio, color=colors[i%7], ls=self._styles[i%7], lw=2, label=label)
        ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
        ax.set_xlabel('Time $t$'); ax.set_ylabel('Gap/Bar Momentum Ratio')
        ax.set_title('Nozzle Effectiveness (ratio > 1 = jet focusing)', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)

        plt.tight_layout()
        self._save(fig, 'nozzle_momentum_flux.png')

    def plot_collimation_timeseries(self):
        """Collimation ratio over time for all cases."""
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (label, s) in enumerate(self.solvers.items()):
            t = np.array(s.diag_times)
            ax.plot(t, s.diag_jet_collimation, color=self._colors[i%7],
                    ls=self._styles[i%7], lw=2.5, label=label)
        ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.5, label='Isotropic (0.5)')
        ax.set_xlabel('Time $t$', fontsize=13); ax.set_ylabel('Gap Momentum Fraction', fontsize=12)
        ax.set_title('Jet Collimation: Splash vs Pressure Washer', fontweight='bold', fontsize=14)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(0, 1)
        plt.tight_layout()
        self._save(fig, 'nozzle_collimation.png')

    def plot_jet_velocity_profiles(self):
        """Transverse (y) profiles of vx at a downstream slice for each case."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        pairs = list(self.solvers.items()); colors = self._colors

        # Profile at x = interface + 1.0
        for ax, x_offset, title in [
            (axes[0], 0.5, 'Near-field ($x = x_{if} + 0.5$)'),
            (axes[1], 1.5, 'Far-field ($x = x_{if} + 1.5$)'),
        ]:
            for i, (label, s) in enumerate(pairs):
                snap = self._final(s)
                x_target = s.cfg.interface_x + x_offset
                ix = np.argmin(np.abs(snap['x'] - x_target))
                vx_profile = snap['vx'][ix, :]
                ax.plot(snap['y'], vx_profile, color=colors[i%7], ls=self._styles[i%7], lw=2, label=label)
                # Mark bar/gap boundaries for striped cases
                if s.cfg.is_striped():
                    Ly = s.cfg.y_max - s.cfg.y_min
                    k_mod = 2*np.pi*s.cfg.B_modulation_mode / Ly
                    for yn in snap['y']:
                        if abs(np.cos(k_mod * yn + s.cfg.B_stripe_phase)) < 0.05:
                            ax.axvline(yn, color=colors[i%7], lw=0.3, alpha=0.3)
            ax.set_xlabel('$y$'); ax.set_ylabel('$v_x$')
            ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        fig.suptitle('Jet Velocity Profiles (transverse cuts)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'nozzle_jet_profiles.png')

    def plot_magnetic_tension_map(self):
        """Map of the in-plane Lorentz force |J x B| showing bars vs gaps.

        (B.grad)B alone is not a force when div B != 0 and misses the magnetic
        pressure of out-of-plane fields, so the full J x B is plotted."""
        mhd = {k: v for k, v in self.solvers.items() if v.cfg.is_mhd()}
        if not mhd: return
        n = len(mhd)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, mhd.items()):
            snap = self._final(solver)
            Bx, By_f, Bz = snap['Bx'], snap['By'], snap['Bz']
            dx, dy = solver.cfg.dx, solver.cfg.dy
            # J = curl B with d/dz = 0
            Jx = np.gradient(Bz, dy, axis=1)
            Jy = -np.gradient(Bz, dx, axis=0)
            Jz = np.gradient(By_f, dx, axis=0) - np.gradient(Bx, dy, axis=1)
            force_x = Jy*Bz - Jz*By_f
            force_y = Jz*Bx - Jx*Bz
            tension_mag = np.sqrt(force_x**2 + force_y**2)
            vmax_t = np.percentile(tension_mag, 98)
            im = ax.pcolormesh(snap['x'], snap['y'], tension_mag.T, cmap='magma',
                               shading='auto', vmin=0, vmax=max(vmax_t, 1e-10))
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$|\mathbf{J}\times\mathbf{B}|$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Lorentz Force: The Bars That Pin the Interface', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'nozzle_tension_map.png')

    def plot_nozzle_hero(self):
        """Hero comparison: hydro splash vs nozzle jets (density + velocity overlay)."""
        keys = list(self.solvers.keys())
        if len(keys) < 2: return
        fig, axes = plt.subplots(2, len(keys), figsize=(5.5*len(keys), 10), squeeze=False)

        for col, key in enumerate(keys):
            solver = self.solvers[key]; snap = self._final(solver)
            rho = snap['rho']; vx = snap['vx']

            # Top row: density
            ax = axes[0, col]
            im = ax.pcolormesh(snap['x'], snap['y'], rho.T, cmap='inferno', shading='auto',
                               vmin=rho.min(), vmax=np.percentile(rho, 98))
            ax.set_aspect('equal'); ax.set_title(f"{key}", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\rho$')
            if col == 0: ax.set_ylabel('$y$')
            ax.set_xlabel('$x$')

            # Bottom row: x-momentum (rho*vx) showing directed jets
            ax = axes[1, col]
            mom = rho * vx
            vmax_m = np.percentile(np.abs(mom), 98)
            im = ax.pcolormesh(snap['x'], snap['y'], mom.T, cmap='RdYlBu_r', shading='auto',
                               vmin=-vmax_m, vmax=vmax_m)
            ax.set_aspect('equal')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\rho v_x$')
            if col == 0: ax.set_ylabel('$y$')
            ax.set_xlabel('$x$')

            # Overlay bar/gap for striped
            if solver.cfg.is_striped():
                Ly = solver.cfg.y_max - solver.cfg.y_min
                k_mod = 2*np.pi*solver.cfg.B_modulation_mode / Ly
                for yn in snap['y']:
                    if abs(np.cos(k_mod * yn + solver.cfg.B_stripe_phase)) < 0.05:
                        for r in range(2):
                            axes[r, col].axhline(yn, color='lime', lw=0.5, alpha=0.4)

        fig.suptitle('Magnetic Nozzle: Splash vs Focused Jets\n(top: density, bottom: x-momentum)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'nozzle_hero.png')

    def plot_y_striped_momentum(self):
        """x-momentum as function of y, averaged over downstream region."""
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (label, s) in enumerate(self.solvers.items()):
            snap = self._final(s)
            x_arr = snap['x']; y_arr = snap['y']
            x_ds = s.cfg.interface_x + 0.3
            ix_ds = np.argmin(np.abs(x_arr - x_ds))
            rho_ds = snap['rho'][ix_ds:, :]; vx_ds = snap['vx'][ix_ds:, :]
            mom_y_profile = np.mean(rho_ds * vx_ds, axis=0)
            ax.plot(y_arr, mom_y_profile, color=self._colors[i%7], ls=self._styles[i%7], lw=2, label=label)

        # Mark gap locations for striped cases
        for label, s in self.solvers.items():
            if s.cfg.is_striped():
                snap = self._final(s)
                Ly = s.cfg.y_max - s.cfg.y_min
                k_mod = 2*np.pi*s.cfg.B_modulation_mode / Ly
                for yn in snap['y']:
                    if abs(np.cos(k_mod * yn + s.cfg.B_stripe_phase)) < 0.05:
                        ax.axvline(yn, color='green', lw=0.5, alpha=0.3)
                break  # only need one set of markers

        ax.set_xlabel('$y$', fontsize=13); ax.set_ylabel(r'$\langle \rho v_x \rangle_{downstream}$', fontsize=12)
        ax.set_title('Downstream Momentum Profile: Where Does the Plasma Go?', fontweight='bold', fontsize=14)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, 'nozzle_y_momentum.png')

    def plot_sieve_initial_condition(self):
        """Visualize the initial striped B-field setup: the sieve before the shock hits."""
        striped = {k: v for k, v in self.solvers.items() if v.cfg.is_striped()}
        if not striped: return
        label, solver = next(iter(striped.items()))
        snap = solver.snapshots.get('initial', list(solver.snapshots.values())[0])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        im = ax.pcolormesh(snap['x'], snap['y'], snap['rho'].T, cmap='inferno', shading='auto')
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_aspect('equal')
        ax.set_title('Initial Density', fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\rho$')

        ax = axes[1]
        comp = {"x": "Bx", "y": "By", "z": "Bz"}[solver.cfg.B_stripe_component]
        B_init = snap[comp]
        vmax_b = max(np.max(np.abs(B_init)), 1e-12)
        im = ax.pcolormesh(snap['x'], snap['y'], B_init.T, cmap='RdBu_r', shading='auto',
                           vmin=-vmax_b, vmax=vmax_b)
        ax.set_xlabel('$x$'); ax.set_aspect('equal')
        ax.set_title(f'Initial $B_{comp[1]}$ (Striped)', fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.82, label=f'$B_{comp[1]}$')

        ax = axes[2]
        Pmag = 0.5*(snap['Bx']**2 + snap['By']**2 + snap['Bz']**2)
        im = ax.pcolormesh(snap['x'], snap['y'], Pmag.T, cmap='plasma', shading='auto')
        ax.set_xlabel('$x$'); ax.set_aspect('equal')
        ax.set_title(r'Initial $P_{mag}$ (bars = high, gaps = zero)', fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.82, label=r'$B^2/2$')

        fig.suptitle(f'Magnetic Sieve Setup: {label}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'sieve_initial_condition.png')

    def plot_bar_vs_gap_density_profiles(self):
        """Compare density profiles at bar and gap y-locations."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        pairs = list(self.solvers.items()); colors = self._colors

        for ax, region, title in [
            (axes[0], 'bar', 'Bar Region (high |B|)'),
            (axes[1], 'gap', 'Gap Region (B ~ 0)'),
        ]:
            for i, (label, s) in enumerate(pairs):
                snap = self._final(s)
                if s._bar_mask_y is not None:
                    mask = s._bar_mask_y if region == 'bar' else s._gap_mask_y
                    rho_region = np.mean(snap['rho'][:, mask], axis=1)
                else:
                    rho_region = np.mean(snap['rho'], axis=1)
                ax.plot(snap['x'], rho_region, color=colors[i%7], ls=self._styles[i%7], lw=2, label=label)
            ax.set_xlabel('$x$'); ax.set_ylabel(r'$\langle \rho \rangle_y$')
            ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        fig.suptitle('Density Profiles: Bar vs Gap Regions', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'nozzle_bar_gap_density.png')

    def plot_summary_bars(self):
        t_skip = 0.03
        labels_list = []; mwi_peaks = []; enst_means = []; col_means = []; jet_peaks = []
        sm_en = self._get_smoothing()
        for label, s in self.solvers.items():
            t = np.array(s.diag_times); mask = t > t_skip
            labels_list.append(label)
            mwi = smooth(np.array(s.diag_mixing_width_integral)[mask], 7, sm_en)
            mwi_peaks.append(float(np.max(mwi)) if len(mwi) > 0 else 0)
            en = np.array(s.diag_enstrophy_local)[mask]
            enst_means.append(float(np.mean(en)) if len(en) > 0 else 0)
            col = np.array(s.diag_jet_collimation)[mask]
            col_means.append(float(np.mean(col)) if len(col) > 0 else 0)
            jvx = np.array(s.diag_jet_max_vx)[mask]
            jet_peaks.append(float(np.max(jvx)) if len(jvx) > 0 else 0)

        x_pos = np.arange(len(labels_list)); bar_colors = self._colors[:len(labels_list)]
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 4.5))
        for ax, vals, ylabel, title in [
            (ax1, mwi_peaks, 'Peak MW', 'Mixing Width'),
            (ax2, enst_means, 'Mean Enstrophy', 'Enstrophy'),
            (ax3, col_means, 'Mean Collimation', 'Jet Collimation'),
            (ax4, jet_peaks, r'Peak $v_x$', 'Peak Jet Velocity'),
        ]:
            bars = ax.bar(x_pos, vals, color=bar_colors, edgecolor='k')
            ax.set_xticks(x_pos); ax.set_xticklabels(labels_list, fontsize=7, rotation=15)
            ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold'); ax.grid(True, alpha=0.3, axis='y')
            fmt = '.4f' if max(vals) < 1 else '.1f'
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                        f"{val:{fmt}}", ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        self._save(fig, 'summary_bars.png')

    def plot_divB_comparison(self):
        mhd_cases = {k: v for k, v in self.solvers.items() if v.cfg.is_mhd()}
        if not mhd_cases: return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5)); colors = self._colors
        for ax, diag, ylabel, title in [
            (axes[0], 'diag_divB_max', r'max $|\nabla\cdot\mathbf{B}|$', 'Max norm'),
            (axes[1], 'diag_divB_L2', r'$L_2$ norm', '$L_2$ norm'),
        ]:
            for i, (label, s) in enumerate(mhd_cases.items()):
                ax.plot(s.diag_times, getattr(s, diag), color=colors[i%7], ls=self._styles[i%7], lw=2, label=label)
            ax.set_xlabel('Time $t$'); ax.set_ylabel(ylabel)
            ax.set_title(f'div(B) Control ({title})', fontweight='bold')
            ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        plt.tight_layout()
        self._save(fig, 'divB.png')

    def plot_all(self):
        print("\n=== Generating figures ==="); sys.stdout.flush()
        # Core comparison plots
        self.plot_density_comparison()
        self.plot_passive_scalar()
        self.plot_schlieren()
        self.plot_vorticity_comparison()
        self.plot_magnetic_pressure_with_fieldlines()
        self.plot_diagnostics()
        self.plot_evolution()
        self.plot_interface_shape()
        self.plot_summary_bars()
        self.plot_divB_comparison()
        # Magnetic Nozzle / Sieve specific plots
        self.plot_sieve_initial_condition()
        self.plot_nozzle_hero()
        self.plot_nozzle_velocity_map()
        self.plot_jet_velocity_profiles()
        self.plot_momentum_flux_comparison()
        self.plot_collimation_timeseries()
        self.plot_magnetic_tension_map()
        self.plot_y_striped_momentum()
        self.plot_bar_vs_gap_density_profiles()

        print(f"\n=== Figure manifest ({len(self._saved_files)} files) ===")
        total_kb = 0
        for fname, ok, sz in self._saved_files:
            print(f"  {'OK' if ok else 'FAIL'} {fname:45s}  {sz:7.0f} KB")
            total_kb += sz
        print(f"  {'Total':>47s}  {total_kb:7.0f} KB")
        print(f"  Directory: {os.path.abspath(OUTPUT_DIR)}")
        print("=== All figures done ===\n"); sys.stdout.flush()

        # Dynamic animations (run after static plots)
        self.animate_density_evolution()
        self.animate_vorticity_evolution()
        self.animate_jet_formation()

    # ---- DYNAMIC ANIMATIONS ----

    @staticmethod
    def _anim_axes(pairs):
        """Side-by-side panels, or stacked rows for wide domains (Lx/Ly > 2)."""
        n_cases = len(pairs)
        cfg = pairs[0][1].cfg
        aspect = (cfg.x_max - cfg.x_min) / (cfg.y_max - cfg.y_min)
        if aspect > 2:
            fig, axes = plt.subplots(n_cases, 1, figsize=(12, 1.2 + 12 / aspect * 1.25 * n_cases),
                                     sharex=True, squeeze=False)
            return fig, axes[:, 0]
        fig, axes = plt.subplots(1, n_cases, figsize=(5*n_cases, 4.5), sharey=True, squeeze=False)
        return fig, axes[0]

    def _get_sorted_snapshots(self, solver):
        return sorted(solver.snapshots.keys(),
                       key=lambda k: solver.snapshots[k]['t'])

    def animate_density_evolution(self):
        """Animated density: all cases side-by-side, cycling through snapshots."""
        from matplotlib.animation import FuncAnimation
        pairs = list(self.solvers.items())
        n_cases = len(pairs)
        all_keys = [self._get_sorted_snapshots(s) for _, s in pairs]
        n_frames = min(len(ks) for ks in all_keys)
        if n_frames < 2:
            print("  [SKIP] animate_density_evolution: need >= 2 snapshots"); return

        rho_max = max(np.percentile(s.snapshots[k]['rho'], 98)
                       for _, s in pairs for k in s.snapshots)

        fig, axes = self._anim_axes(pairs)
        ims = []
        for ax, (label, s), keys in zip(axes, pairs, all_keys):
            snap = s.snapshots[keys[0]]
            im = ax.pcolormesh(snap['x'], snap['y'], snap['rho'].T,
                               cmap='inferno', shading='auto', vmin=0.5, vmax=rho_max)
            ax.set_aspect('equal'); ax.set_xlabel('$x$')
            ax.set_title(f"{label}\n$t={snap['t']:.3f}$", fontweight='bold', fontsize=10)
            ims.append(im)
        axes[0].set_ylabel('$y$')
        fig.suptitle('Density Evolution', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        def update(frame):
            for i, ((label, s), keys) in enumerate(zip(pairs, all_keys)):
                snap = s.snapshots[keys[frame]]
                ims[i].set_array(snap['rho'].T.ravel())
                axes[i].set_title(f"{label}\n$t={snap['t']:.3f}$",
                                   fontweight='bold', fontsize=10)
            return ims

        anim = FuncAnimation(fig, update, frames=n_frames, interval=500, blit=False)
        filepath = os.path.join(OUTPUT_DIR, 'anim_density.gif')
        try:
            anim.save(filepath, writer='pillow', fps=max(2, round(n_frames / 12)), dpi=120)
            sz = os.path.getsize(filepath) / 1024
            self._saved_files.append(('anim_density.gif', True, sz))
            print(f"  Saved: {filepath} ({sz:.0f} KB)")
        except Exception as e:
            print(f"  [WARN] anim_density: {e}")
        plt.close(fig)

    def animate_vorticity_evolution(self):
        """Animated vorticity showing KH at bar-gap boundaries."""
        from matplotlib.animation import FuncAnimation
        pairs = list(self.solvers.items())
        n_cases = len(pairs)
        all_keys = [self._get_sorted_snapshots(s) for _, s in pairs]
        n_frames = min(len(ks) for ks in all_keys)
        if n_frames < 2:
            print("  [SKIP] animate_vorticity_evolution: need >= 2 snapshots"); return

        vmax = 0
        for _, s in pairs:
            for snap in s.snapshots.values():
                om = np.gradient(snap['vy'], s.cfg.dx, axis=0) - np.gradient(snap['vx'], s.cfg.dy, axis=1)
                vmax = max(vmax, np.percentile(np.abs(om), 99))
        vmax = max(vmax, 1e-10)

        fig, axes = self._anim_axes(pairs)
        ims = []
        for ax, (label, s), keys in zip(axes, pairs, all_keys):
            snap = s.snapshots[keys[0]]
            om = np.gradient(snap['vy'], s.cfg.dx, axis=0) - np.gradient(snap['vx'], s.cfg.dy, axis=1)
            im = ax.pcolormesh(snap['x'], snap['y'], om.T, cmap='RdBu_r', shading='auto',
                               vmin=-vmax, vmax=vmax)
            ax.set_aspect('equal'); ax.set_xlabel('$x$')
            ax.set_title(f"{label}\n$t={snap['t']:.3f}$", fontweight='bold', fontsize=10)
            ims.append(im)
        axes[0].set_ylabel('$y$')
        fig.suptitle('Vorticity Evolution $\\omega_z$', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        def update(frame):
            for i, ((label, s), keys) in enumerate(zip(pairs, all_keys)):
                snap = s.snapshots[keys[frame]]
                om = np.gradient(snap['vy'], s.cfg.dx, axis=0) - np.gradient(snap['vx'], s.cfg.dy, axis=1)
                ims[i].set_array(om.T.ravel())
                axes[i].set_title(f"{label}\n$t={snap['t']:.3f}$", fontweight='bold', fontsize=10)
            return ims

        anim = FuncAnimation(fig, update, frames=n_frames, interval=500, blit=False)
        filepath = os.path.join(OUTPUT_DIR, 'anim_vorticity.gif')
        try:
            anim.save(filepath, writer='pillow', fps=max(2, round(n_frames / 12)), dpi=120)
            sz = os.path.getsize(filepath) / 1024
            self._saved_files.append(('anim_vorticity.gif', True, sz))
            print(f"  Saved: {filepath} ({sz:.0f} KB)")
        except Exception as e:
            print(f"  [WARN] anim_vorticity: {e}")
        plt.close(fig)

    def animate_jet_formation(self):
        """Animated x-momentum showing jets emerging from gaps."""
        from matplotlib.animation import FuncAnimation
        pairs = list(self.solvers.items())
        n_cases = len(pairs)
        all_keys = [self._get_sorted_snapshots(s) for _, s in pairs]
        n_frames = min(len(ks) for ks in all_keys)
        if n_frames < 2:
            print("  [SKIP] animate_jet_formation: need >= 2 snapshots"); return

        mom_max = 0
        for _, s in pairs:
            for snap in s.snapshots.values():
                mom_max = max(mom_max, np.percentile(np.abs(snap['rho'] * snap['vx']), 98))
        mom_max = max(mom_max, 1e-10)

        fig, axes = self._anim_axes(pairs)
        ims = []
        for ax, (label, s), keys in zip(axes, pairs, all_keys):
            snap = s.snapshots[keys[0]]
            mom = snap['rho'] * snap['vx']
            im = ax.pcolormesh(snap['x'], snap['y'], mom.T, cmap='RdYlBu_r', shading='auto',
                               vmin=-mom_max, vmax=mom_max)
            ax.set_aspect('equal'); ax.set_xlabel('$x$')
            ax.set_title(f"{label}\n$t={snap['t']:.3f}$", fontweight='bold', fontsize=10)
            if s.cfg.is_striped():
                Ly = s.cfg.y_max - s.cfg.y_min
                k_mod = 2*np.pi*s.cfg.B_modulation_mode / Ly
                for yn in snap['y']:
                    if abs(np.cos(k_mod * yn + s.cfg.B_stripe_phase)) < 0.05:
                        ax.axhline(yn, color='lime', lw=0.5, alpha=0.4)
            ims.append(im)
        axes[0].set_ylabel('$y$')
        fig.suptitle('Jet Formation: $\\rho v_x$', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        def update(frame):
            for i, ((label, s), keys) in enumerate(zip(pairs, all_keys)):
                snap = s.snapshots[keys[frame]]
                mom = snap['rho'] * snap['vx']
                ims[i].set_array(mom.T.ravel())
                axes[i].set_title(f"{label}\n$t={snap['t']:.3f}$", fontweight='bold', fontsize=10)
            return ims

        anim = FuncAnimation(fig, update, frames=n_frames, interval=500, blit=False)
        filepath = os.path.join(OUTPUT_DIR, 'anim_jets.gif')
        try:
            anim.save(filepath, writer='pillow', fps=max(2, round(n_frames / 12)), dpi=120)
            sz = os.path.getsize(filepath) / 1024
            self._saved_files.append(('anim_jets.gif', True, sz))
            print(f"  Saved: {filepath} ({sz:.0f} KB)")
        except Exception as e:
            print(f"  [WARN] anim_jets: {e}")
        plt.close(fig)


print("PostProcessor OK")

