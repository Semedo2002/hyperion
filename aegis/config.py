# config.py - Structured solver configuration.
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional


class ConsVar(IntEnum):
    RHO = 0; MX = 1; MY = 2; MZ = 3
    BX = 4; BY = 5; BZ = 6; EN = 7
    PSI = 8; RHOC = 9; RHOSE = 10      # RHOSE: rho s_e, electron entropy (s_e = p_e/rho^gamma), passive


class PrimVar(IntEnum):
    RHO = 0; VX = 1; VY = 2; VZ = 3
    BX = 4; BY = 5; BZ = 6; PR = 7
    PSI = 8; CLR = 9; SE = 10


NVAR = len(ConsVar)
RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI, RHOC, RHOSE = tuple(ConsVar)
iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iPSI, iCLR, iSE = tuple(PrimVar)
FLOOR_RHO = 1e-12
FLOOR_PR = 1e-12


@dataclass(frozen=True)
class GridConfig:
    nx: int = 400
    ny: int = 200
    x_min: float = 0.0
    x_max: float = 6.0
    y_min: float = 0.0
    y_max: float = 2.0
    # z is used only by solver3d (periodic)
    nz: int = 1
    z_min: float = 0.0
    z_max: float = 1.0
    # exact cell width in x, if > 0 (a subdomain of a decomposed grid keeps the global dx to the last bit)
    x_spacing: float = 0.0

    @property
    def dx(self) -> float:
        return self.x_spacing if self.x_spacing > 0 else (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.ny

    @property
    def dz(self) -> float:
        return (self.z_max - self.z_min) / self.nz


@dataclass(frozen=True)
class PhysicsConfig:
    t_end: float = 0.25
    cfl: float = 0.30
    max_steps: int = 200000
    gamma: float = 5.0 / 3.0
    mach: float = 10.0
    interface_x: float = 1.5
    perturbation_amp: float = 0.15
    perturbation_mode: int = 4
    density_ratio: float = 3.0
    interface_width: float = 2.0
    B_transverse: float = 0.0
    B_field_type: str = "uniform"
    B_modulation_mode: int = 4
    # Field component carrying the striped modulation cos(k_mod*y):
    #   "z": out-of-plane magnetic-pressure bars (exactly solenoidal in 2D)
    #   "x": shock-parallel field lines with tension (exactly solenoidal)
    #   "y": legacy By(y); div(B) = -B0*k_mod*sin(k_mod*y) != 0, kept for reproduction only
    B_stripe_component: str = "x"
    # Upstream equilibrium for "x" stripes: p_gas(y) = P_total - B(y)^2/2 with
    # P_total = p_bar + B0^2/2, so the bar centres (|B| = B0) keep (rho=1, p=p_bar)
    # and the Mach number refers to that state. equilibrium_closure picks how
    # density follows the pressure: "uniform_density" or "uniform_temperature".
    p_bar: float = 1.0
    # Field left in the gaps of "x" stripes, as a fraction eps of B0:
    #   eps = 0:  Bx = B0 cos(k y)                       (field-free gaps)
    #   eps > 0:  Bx = B0 sqrt(eps^2 + (1-eps^2) cos^2)  (tension everywhere, no nulls)
    B_gap_fraction: float = 0.0
    # Phase phi of the stripes, Bx ~ cos(k_mod y + phi), relative to the interface
    # ripple sin(k_int y) (radians).
    B_stripe_phase: float = 0.0
    # 3D seeding (solver3d): random v_z in the domain and at the inflow, as a fraction of the
    # bar sound speed, and a z-ripple of the interface; seed fixes the random pattern
    z_pert_vel: float = 0.0
    z_pert_amp: float = 0.0
    z_pert_mode: int = 4
    z_pert_seed: int = 0
    # Galilean frame (solver3d): velocity subtracted from v_x everywhere; with frame_velocity > 0 the
    # x_max boundary becomes a fixed inflow of upstream gas moving at -frame_velocity
    frame_velocity: float = 0.0
    # x_max boundary in solver3d: "auto" (upstream inflow when frame_velocity > 0, else transmissive),
    # "inflow", "transmissive" or "plateau". In the co-moving frame use "plateau": upstream inflow until
    # the transmitted shock arrives, then the shocked state behind it. Holding the upstream state
    # reflects a compression; zero-gradient lets the subsonic shocked gas drain and sends a rarefaction.
    x_max_bc: str = "auto"
    # x_min boundary in solver3d: "inflow" (fixed post-shock inflow) or "plateau" (inflow until the
    # reflected shock leaves through x_min, then the post-reflected state behind it)
    x_min_bc: str = "inflow"
    # absorbing layer (solver3d): over sponge_width at each x end, relax U toward a running time-average
    # of the local state at rate sponge_rate * (depth into the layer)^2; damps waves, not the mean state
    sponge_width: float = 0.0
    sponge_rate: float = 0.0
    sponge_tau: float = 0.5
    equilibrium_closure: str = "uniform_density"
    # div B control: "ct" (constrained transport, face-centred B, div B = 0 to round-off; Powell and GLM
    # are off) or "glm" (cell-centred B with GLM cleaning and optional Powell terms, the scheme the
    # published sieve study used)
    div_b: str = "ct"
    # replay each SSP-RK3 step as one captured CUDA graph (2D, CT): removes the per-kernel launch overhead
    use_cuda_graphs: bool = True
    # "cartesian" or "cylindrical" (2D solver only): x is the radius r (x_min = 0 is the axis, reflecting),
    # y the axial coordinate z, and the out-of-plane components are -v_phi, -B_phi. Requires CT and an
    # extrapolation (outflow) boundary at r = x_max.
    geometry: str = "cartesian"
    # 3D cylindrical (r, theta, z): near-axis azimuthal Fourier filter of the tendencies (solver3d), rings with
    # r dtheta < cyl_filter_frac dr keep modes m <= max(cyl_filter_mkeep, floor(pi r / (cyl_filter_frac dr)))
    cyl_filter: bool = False
    cyl_filter_mkeep: int = 2
    cyl_filter_frac: float = 1.0
    glm_ch: float = 0.0
    # GLM damping, psi *= exp(-glm_alpha * ch * dt / h). Values in [0, 1] keep the
    # cleaning transport-dominated (Mignone & Tzeferacos 2010); 1.5 measured worse.
    glm_alpha: float = 0.4
    powell_source: bool = True
    use_char_bc: bool = True
    # x boundaries: "inflow_outflow" (Dirichlet post-shock inflow at x_min, transmissive x_max),
    # "extrapolation" (zero-gradient both ends), "periodic", or "auto" (from use_char_bc).
    # "characteristic" is accepted as an alias for "inflow_outflow".
    bc_x_type: str = "auto"
    bc_y_type: str = "periodic"
    enable_smoothing: bool = False
    eta_resistivity: float = 0.0
    # resistivity model: "constant" (eta_resistivity everywhere), "material" (eta_fuel where the
    # tracer C = 0, eta_liner where C = 1, log blend in between) or "spitzer" (eta_spitzer * T^-3/2
    # with T = p/rho floored at eta_T_min; blended toward eta_liner by C when eta_liner > 0).
    # eta_max > 0 caps eta in any model (required for spitzer). Face values use eta_face_mean.
    eta_model: str = "constant"
    eta_fuel: float = 0.0
    eta_liner: float = 0.0
    eta_spitzer: float = 0.0
    eta_T_min: float = 1e-2
    eta_max: float = 0.0
    eta_face_mean: str = "harmonic"
    # equation of state, two materials: "fuel" where the tracer is 0, "liner" where it is 1.
    # fuel: ideal or stiffened (gamma, p_inf); liner: ideal, stiffened (gamma_liner, p_inf_liner) or
    # mie_gruneisen (linear Us-Up Hugoniot reference: mg_rho0, mg_c0, mg_s, mg_gamma0, code units).
    # With more than one material the tracer becomes a volume fraction advected non-conservatively
    # (keeps pressure equilibrium across the interface); material_tracer = "mass_fraction" blends by the
    # conserved tracer instead and is kept for comparison. Setting "volume_fraction" explicitly also
    # makes a single ideal gas carry its tracer as a volume fraction (same pressures, a tracer that
    # measures volume); left as None, a single gas keeps the conserved mass-fraction tracer.
    eos_fuel: str = "ideal"
    p_inf: float = 0.0
    eos_liner: str = "ideal"
    gamma_liner: Optional[float] = None
    p_inf_liner: float = 0.0
    mg_rho0: float = 0.0
    mg_c0: float = 0.0
    mg_s: float = 0.0
    mg_gamma0: float = 0.0
    material_tracer: Optional[str] = None
    # optically thin bremsstrahlung, Q = rad_coeff rho^2 sqrt(p/rho) in code units (0 = off; see
    # units.bremsstrahlung_kwargs), only where the liner fraction < rad_tracer_max and p/rho > rad_T_min.
    # Applied as a split step with the exact solution at fixed density, so it never limits the time step.
    rad_coeff: float = 0.0
    rad_T_min: float = 0.0
    rad_tracer_max: float = 0.1
    # grey flux-limited diffusion radiation (aegis/fld.py; units.fld_kwargs): radiation energy E_r with
    # a theta^4 emission, speed of light fld_c, power-law opacities k0 rho^m theta^n (Planck kp, Rosseland kr)
    # per unit mass; fuel only (liner fraction < fld_tracer_max). fld_E_xmin / fld_E_xmax >= 0 hold E_r at
    # the non-periodic x faces (negative: zero flux there).
    # tabulated liner EOS (aegis/tabular.py: SESAME ASCII or .npz, SI) in pure liner cells, eos_liner as the
    # fallback beyond the table and the tangent closure in mixed cells; the table is converted with the code
    # units tab_rho_unit (kg/m^3) and tab_v_unit (m/s). fld_opacity_table: kappa_R / kappa_P for FLD instead of
    # the power laws (tab_L_unit m, tab_T_unit kelvin per code temperature).
    eos_table: str = ""
    eos_table_material: int = 0
    tab_rho_unit: float = 1000.0
    tab_v_unit: float = 1000.0
    tab_L_unit: float = 1e-3
    tab_T_unit: float = 1.0
    fld_opacity_table: str = ""
    fld: bool = False
    fld_a: float = 0.0
    fld_c: float = 0.0
    fld_kp0: float = 0.0
    fld_kp_rho: float = 0.0
    fld_kp_T: float = 0.0
    fld_kr0: float = 0.0
    fld_kr_rho: float = 0.0
    fld_kr_T: float = 0.0
    fld_tracer_max: float = 0.1
    fld_E_xmin: float = -1.0
    fld_E_xmax: float = -1.0
    fld_advect: bool = True
    fld_tol: float = 1e-11
    fld_maxit: int = 5000
    fld_picard: int = 4
    # multigroup FLD (aegis/radiation_multigroup.py): fld_groups > 0 photon groups, edges in code temperature units
    # (fld_group_edges, or log-spaced between fld_group_emin and fld_group_emax; the first group starts at 0, the
    # last runs to infinity); opacities "grey" (the power laws in every group), "kramers" (free-free nu^-3 shape
    # scaled to the grey Planck mean, times 1 + fld_edge_jump above a photo-ionization edge at fld_edge), or a
    # group table (fld_mg_table, from tools/generate_sesame.py --groups)
    fld_groups: int = 0
    fld_group_edges: tuple = ()
    fld_group_emin: float = 0.1
    fld_group_emax: float = 100.0
    fld_mg_opacity: str = "grey"
    fld_edge: float = 0.0
    fld_edge_jump: float = 0.0
    fld_mg_table: str = ""
    fld_mg_absorption: float = 1.0                   # scales every kP_g (0: groups only diffuse, no exchange)
    # learned predictor for the multigroup implicit solve (aegis/fno_subgrid.py, a .npz model; 2D): accepted when
    # the relative residual of the linear system is <= fld_fno_tol, otherwise BiCGStab starts from it
    fld_precond: str = "jacobi"                      # Krylov preconditioner: "jacobi" (block Jacobi) or "mg" (multigrid)
    fld_fno_model: str = ""
    fld_fno_tol: float = 1e-3
    # magnetized electron conduction (physics.conduction_div), 0 = off: k_par = cond_kappa0 T^5/2,
    # k_perp = k_par/(1 + x^2), x = cond_x0 |B| T^3/2 / rho, T = p/rho; fuel only (tracer < cond_tracer_max).
    # Split step integrated with RKL2 super-time-stepping, so it does not limit the time step.
    cond_kappa0: float = 0.0
    cond_x0: float = 0.0
    cond_tracer_max: float = 0.1
    # fuel-liner faces: "isothermal" (the liner is a heat sink at temperature cond_wall_T, code units p/rho of
    # the fuel; the fuel conducts to it across half a cell with its own anisotropic k) or "insulating"
    cond_wall: str = "isothermal"
    cond_wall_T: float = 0.0
    # cross-field conductivity: "braginskii" (full Z = 1 fit, transport.kperp_ratio) or "simple" (1/(1 + x^2))
    cond_perp: str = "braginskii"
    # Nernst field advection and Ettingshausen heat flux (aegis/transport.py), coefficient c_N in code units
    # (units.braginskii_kwargs); 0 = off. Uses cond_x0 for the magnetization; 2D solver with CT.
    nernst_coeff: float = 0.0
    # two-temperature fuel (aegis/twotemp.py): electron entropy carried as a passive scalar; electron conduction,
    # bremsstrahlung, Ohmic and Ettingshausen heating act on the electrons; equilibration rate te_eq_coeff
    # (units.twotemp_kwargs). Needs an ideal-gas fuel with gamma = 5/3. z_ion: fuel ion charge.
    two_temp: bool = False
    z_ion: float = 1.0
    te_eq_coeff: float = 0.0
    # Braginskii parallel ion viscosity, eta_0 = visc_coeff theta_i^2.5 (units.viscosity_kwargs); 0 = off.
    # Cartesian geometry; fuel only (tracer < cond_tracer_max).
    visc_coeff: float = 0.0
    # generalized Ohm's law (aegis/hall.py), 2D Cartesian with CT: Hall term d_i (J x B)/rho with hall_di the ion
    # inertial length at unit density in code units (0 = off); anisotropic resistivity eta_perp = eta_perp_ratio
    # eta_par with eta (any eta_model) the parallel value (1.96 for a magnetized Z = 1 Spitzer plasma; 1 = isotropic)
    hall_di: float = 0.0
    eta_perp_ratio: float = 1.0
    # ion Braginskii transport (aegis/ions.py): ion conduction on T_i in two-temperature mode, k_i,par =
    # cond_kappa_i0 theta_i^5/2, k_i,perp = k_i,par / (1 + x_i^2), x_i = cond_x0_i |B| theta_i^3/2 / rho (0 = off);
    # ion gyroviscosity (the non-dissipative finite-Larmor-radius stress) with gyro_di the ion inertial length at
    # unit density (Omega_i = |B| / gyro_di; 0 = off), 2D Cartesian
    cond_kappa_i0: float = 0.0
    cond_x0_i: float = 0.0
    gyro_di: float = 0.0
    # adaptive kinetic-fluid hybrid (aegis/hybrid_pic.py), 2D Cartesian, ideal gas gamma = 5/3: cells with a local
    # Knudsen number lambda / L > pic_kn_on (lambda = pic_mfp0 theta^2 / rho, L the density / pressure gradient
    # length) or an ion Larmor radius r_L > pic_rl_on dx (r_L = pic_di sqrt(theta) / |B|, pic_di the ion inertial
    # length at unit density, which also sets q/m = 1 / pic_di) carry their ions as particles; they return to the
    # fluid when both fall below pic_hysteresis times the threshold. pic_mfp0 = 0 or pic_di = 0 switches that
    # criterion off (pic_di = 0: neutral particles, no Lorentz force).
    pic: bool = False
    pic_kn_on: float = 0.1
    pic_rl_on: float = 1.0
    pic_hysteresis: float = 0.5
    pic_mfp0: float = 0.0
    pic_di: float = 0.0
    pic_ppc: int = 64
    pic_seed: int = 0
    # liner strength (aegis/strength.py): elastic-perfectly-plastic, von Mises yield, spall damage; code units
    # (units.strength_kwargs); scaled by the liner fraction. 2D solver.
    strength: bool = False
    shear_modulus: float = 0.0
    yield_strength: float = 0.0
    spall_strength: float = 0.0
    localized_rh: bool = False
    enstrophy_locking_fraction: float = 0.20

    def __post_init__(self):
        if self.eta_model not in ("constant", "material", "spitzer"):
            raise ValueError(f"eta_model must be constant, material or spitzer, not {self.eta_model!r}")
        if self.eta_face_mean not in ("harmonic", "arithmetic"):
            raise ValueError(f"eta_face_mean must be harmonic or arithmetic, not {self.eta_face_mean!r}")
        if self.eta_model == "spitzer" and not (self.eta_T_min > 0 and self.eta_max > 0 and self.eta_spitzer > 0):
            raise ValueError("eta_model='spitzer' needs eta_spitzer > 0, eta_T_min > 0 and eta_max > 0")
        if min(self.eta_resistivity, self.eta_fuel, self.eta_liner, self.eta_max) < 0:
            raise ValueError("resistivities must be non-negative")
        if self.geometry not in ("cartesian", "cylindrical"):
            raise ValueError(f"geometry must be cartesian or cylindrical, not {self.geometry!r}")
        if self.geometry == "cylindrical" and (self.div_b != "ct" or self.bc_x_type != "extrapolation"):
            raise ValueError("cylindrical geometry needs div_b='ct' and bc_x_type='extrapolation' (outer boundary)")
        if self.div_b not in ("ct", "glm"):
            raise ValueError(f"div_b must be ct or glm, not {self.div_b!r}")
        if self.eos_fuel not in ("ideal", "stiffened"):
            raise ValueError(f"eos_fuel must be ideal or stiffened, not {self.eos_fuel!r}")
        if self.eos_liner not in ("ideal", "stiffened", "mie_gruneisen"):
            raise ValueError(f"eos_liner must be ideal, stiffened or mie_gruneisen, not {self.eos_liner!r}")
        if self.material_tracer not in (None, "volume_fraction", "mass_fraction"):
            raise ValueError(f"material_tracer must be volume_fraction or mass_fraction")
        if (self.eos_fuel == "ideal" and self.p_inf != 0) or (self.eos_liner == "ideal" and self.p_inf_liner != 0):
            raise ValueError("p_inf is only used by the stiffened EOS; set eos_fuel/eos_liner='stiffened'")
        if min(self.p_inf, self.p_inf_liner) < 0:
            raise ValueError("p_inf must be non-negative")
        if self.two_temp and (abs(self.gamma - 5.0 / 3.0) > 1e-12 or self.eos_fuel != "ideal"):
            raise ValueError("two_temp needs an ideal-gas fuel with gamma = 5/3")
        if min(self.te_eq_coeff, self.visc_coeff) < 0 or self.z_ion <= 0:
            raise ValueError("te_eq_coeff, visc_coeff must be >= 0 and z_ion > 0")
        if self.strength and min(self.shear_modulus, self.yield_strength) <= 0:
            raise ValueError("strength needs shear_modulus > 0 and yield_strength > 0")
        if min(self.shear_modulus, self.yield_strength, self.spall_strength) < 0:
            raise ValueError("shear_modulus, yield_strength, spall_strength must be >= 0")
        if self.visc_coeff > 0 and self.geometry != "cartesian":
            raise ValueError("viscosity is implemented in Cartesian geometry")
        if self.cond_perp not in ("braginskii", "simple"):
            raise ValueError("cond_perp must be braginskii or simple")
        if self.nernst_coeff < 0 or (self.nernst_coeff > 0 and (self.cond_x0 <= 0 or self.div_b != "ct")):
            raise ValueError("nernst_coeff > 0 needs cond_x0 > 0 (magnetization) and div_b='ct'")
        if self.cond_wall not in ("isothermal", "insulating") or self.cond_wall_T < 0:
            raise ValueError("cond_wall must be isothermal or insulating, cond_wall_T >= 0")
        if min(self.cond_kappa0, self.cond_x0) < 0 or not 0 < self.cond_tracer_max <= 1:
            raise ValueError("cond_kappa0, cond_x0 must be >= 0 and 0 < cond_tracer_max <= 1")
        if min(self.rad_coeff, self.rad_T_min) < 0 or not 0 < self.rad_tracer_max <= 1:
            raise ValueError("rad_coeff, rad_T_min must be >= 0 and 0 < rad_tracer_max <= 1")
        if self.fld and (self.fld_c <= 0 or (self.fld_kr0 <= 0 and not self.fld_opacity_table) or self.fld_a < 0
                         or self.fld_kp0 < 0):
            raise ValueError("fld needs fld_c > 0, fld_kr0 > 0, fld_a >= 0, fld_kp0 >= 0")
        if self.fld_groups < 0 or (self.fld_groups > 0 and not self.fld) or self.fld_mg_opacity not in ("grey", "kramers"):
            raise ValueError("fld_groups > 0 needs fld=True; fld_mg_opacity is grey or kramers")
        if self.fld_group_edges and len(self.fld_group_edges) != self.fld_groups - 1:
            raise ValueError("fld_group_edges holds the fld_groups - 1 inner edges")
        if self.cyl_filter and (self.geometry != "cylindrical" or self.cyl_filter_mkeep < 1 or self.cyl_filter_frac <= 0):
            raise ValueError("cyl_filter needs cylindrical geometry (3D), cyl_filter_mkeep >= 1, cyl_filter_frac > 0")
        if self.fld_precond not in ("jacobi", "mg"):
            raise ValueError(f"fld_precond must be jacobi or mg, not {self.fld_precond!r}")
        if self.fld_fno_model and (self.fld_groups < 1 or self.fld_fno_tol <= 0):
            raise ValueError("fld_fno_model needs fld_groups >= 1 and fld_fno_tol > 0")
        if self.fld and self.two_temp:
            raise ValueError("fld couples to a single gas temperature: not with two_temp")
        if min(self.hall_di, self.cond_kappa_i0, self.cond_x0_i, self.gyro_di) < 0 or self.eta_perp_ratio <= 0:
            raise ValueError("hall_di, cond_kappa_i0, cond_x0_i, gyro_di must be >= 0 and eta_perp_ratio > 0")
        if (self.hall_di > 0 or self.eta_perp_ratio != 1.0 or self.gyro_di > 0) and                 (self.div_b != "ct" or self.geometry != "cartesian"):
            raise ValueError("the Hall term, anisotropic resistivity and gyroviscosity need div_b='ct' and Cartesian "
                             "geometry")
        if self.pic and (self.geometry != "cartesian" or abs(self.gamma - 5.0 / 3.0) > 1e-12 or self.two_temp
                         or self.eos_fuel != "ideal" or self.eos_liner != "ideal" or self.pic_ppc < 8
                         or not 0 < self.pic_hysteresis <= 1 or min(self.pic_mfp0, self.pic_di) < 0):
            raise ValueError("pic needs Cartesian geometry, an ideal gas with gamma = 5/3 (monatomic particles), no "
                             "two_temp, pic_ppc >= 8, 0 < pic_hysteresis <= 1, pic_mfp0 and pic_di >= 0")
        if self.cond_kappa_i0 > 0 and not self.two_temp:
            raise ValueError("ion conduction (cond_kappa_i0) acts on T_i: it needs two_temp")
        if self.eos_liner == "mie_gruneisen" and min(self.mg_rho0, self.mg_c0, self.mg_s, self.mg_gamma0) <= 0:
            raise ValueError("eos_liner='mie_gruneisen' needs mg_rho0, mg_c0, mg_s, mg_gamma0 > 0")

    def get_bc_x(self) -> str:
        if self.bc_x_type == "characteristic":
            return "inflow_outflow"
        if self.bc_x_type != "auto":
            return self.bc_x_type
        return "inflow_outflow" if self.use_char_bc else "extrapolation"

    def is_striped(self) -> bool:
        return self.B_field_type == "striped" and self.B_transverse > 0

    def is_multimaterial(self) -> bool:
        gl = self.gamma if self.gamma_liner is None else self.gamma_liner
        return self.eos_fuel != "ideal" or self.eos_liner != "ideal" or gl != self.gamma or bool(self.eos_table)

    def is_mhd(self) -> bool:
        return self.B_transverse > 0

    def is_resistive(self) -> bool:
        # independent of B_transverse: a field set up by hand (e.g. axial Bz) still diffuses
        if self.eta_model == "constant":
            return self.eta_resistivity > 0
        if self.eta_model == "material":
            return max(self.eta_fuel, self.eta_liner) > 0
        return True


@dataclass(frozen=True)
class IOConfig:
    diag_interval: int = 20
    snapshot_times: List[float] = field(default_factory=lambda: [
        0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15,
        0.18, 0.20, 0.22, 0.25,
    ])


@dataclass(init=False)
class Config:
    """Compatibility facade over independently typed configuration groups."""

    def __init__(self, **kwargs):
        known = (set(GridConfig.__dataclass_fields__) | set(PhysicsConfig.__dataclass_fields__)
                 | set(IOConfig.__dataclass_fields__))
        unknown = sorted(set(kwargs) - known)
        if unknown:
            raise ValueError(f"Unknown configuration key(s): {', '.join(unknown)}")
        self.grid = GridConfig(**{key: kwargs[key] for key in GridConfig.__dataclass_fields__ if key in kwargs})
        self.physics = PhysicsConfig(**{key: kwargs[key] for key in PhysicsConfig.__dataclass_fields__ if key in kwargs})
        self.io = IOConfig(**{key: kwargs[key] for key in IOConfig.__dataclass_fields__ if key in kwargs})

    def __getattr__(self, name):
        for group in (self.grid, self.physics, self.io):
            if name in group.__dataclass_fields__:
                return getattr(group, name)
        raise AttributeError(name)

    @property
    def dx(self) -> float:
        return self.grid.dx

    @property
    def dy(self) -> float:
        return self.grid.dy

    @property
    def dz(self) -> float:
        return self.grid.dz

    def get_bc_x(self) -> str:
        return self.physics.get_bc_x()

    def is_striped(self) -> bool:
        return self.physics.is_striped()

    def is_mhd(self) -> bool:
        return self.physics.is_mhd()

    def is_resistive(self) -> bool:
        return self.physics.is_resistive()

    def is_multimaterial(self) -> bool:
        return self.physics.is_multimaterial()
