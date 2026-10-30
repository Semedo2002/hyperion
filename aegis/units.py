# units.py - SI helpers for resistivity: conductivity -> code-unit magnetic diffusivity, Spitzer.
import numpy as np

MU0 = 4e-7 * np.pi
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27


def eta_code_from_sigma(sigma, L0, v0):
    """Magnetic diffusivity 1/(mu0 sigma) in code units (L0 * v0). sigma in S/m; None or inf -> 0 (ideal)."""
    if sigma is None or not np.isfinite(sigma):
        return 0.0
    return 1.0 / (MU0 * sigma * L0 * v0)


def temperature_eV_per_code(v0, ion_mass_amu=2.0, Z=1.0):
    """kT in eV for code temperature p/rho = 1 (units of v0^2), fully ionised, T_e = T_i."""
    return ion_mass_amu * AMU * v0**2 / ((1.0 + Z) * E_CHARGE)


def spitzer_coefficient(L0, v0, ion_mass_amu=2.0, Z=1.0, coulomb_log=10.0):
    """eta_spitzer for Config: code-unit diffusivity at code temperature 1, so that eta = coef * T^-3/2.
    Uses eta [Ohm m] ~ 5.2e-5 Z lnL / T_eV^(3/2) (NRL formulary, transverse ~2x parallel; order of magnitude)."""
    T1 = temperature_eV_per_code(v0, ion_mass_amu, Z)
    return 5.2e-5 * Z * coulomb_log / MU0 / (L0 * v0) / T1**1.5


def resistivity_kwargs(model, L0, v0, sigma_liner=None, sigma_fuel=None, sigma_min=1e6, T_min_eV=1.0,
                       ion_mass_amu=2.0, Z=1.0, coulomb_log=10.0):
    """Config keywords for the resistivity model chosen on an example's command line.

    model: "ideal", "material" (sigma_fuel where the tracer is 0, sigma_liner where it is 1) or
    "spitzer" (Spitzer in the fuel, sigma_liner in the liner if given). sigma_min sets the eta cap,
    which also bounds the explicit diffusion time step."""
    if model == "ideal":
        return {}
    eta_max = eta_code_from_sigma(sigma_min, L0, v0)
    kw = dict(eta_model=model, eta_liner=eta_code_from_sigma(sigma_liner, L0, v0), eta_max=eta_max)
    if model == "material":
        kw["eta_fuel"] = eta_code_from_sigma(sigma_fuel, L0, v0)
    elif model == "spitzer":
        kw["eta_spitzer"] = spitzer_coefficient(L0, v0, ion_mass_amu, Z, coulomb_log)
        kw["eta_T_min"] = T_min_eV / temperature_eV_per_code(v0, ion_mass_amu, Z)
    else:
        raise ValueError(f"unknown eta model {model!r}")
    return kw


def add_resistivity_args(ap):
    g = ap.add_argument_group("resistivity (SI)")
    g.add_argument("--eta-model", dest="eta_model", choices=["ideal", "material", "spitzer"], default=None,
                   help="default: material if a conductivity is given, else ideal")
    g.add_argument("--sigma-liner", dest="sigma_liner", type=float, default=None, help="liner conductivity, S/m (Al ~3.5e7)")
    g.add_argument("--sigma-fuel", dest="sigma_fuel", type=float, default=None, help="fuel conductivity, S/m (default: ideal)")
    g.add_argument("--sigma-min", dest="sigma_min", type=float, default=1e6,
                   help="conductivity floor = eta cap, S/m; sets the smallest diffusion time step")
    g.add_argument("--t-min-ev", dest="t_min_ev", type=float, default=1.0, help="Spitzer temperature floor, eV")
    g.add_argument("--coulomb-log", dest="coulomb_log", type=float, default=10.0)
    g.add_argument("--ion-mass-amu", dest="ion_mass_amu", type=float, default=2.0, help="fuel ion mass (D = 2)")


def resistivity_from_args(a, L0, v0):
    model = a.eta_model or ("material" if (a.sigma_liner or a.sigma_fuel) else "ideal")
    a.eta_model = model
    return resistivity_kwargs(model, L0, v0, a.sigma_liner, a.sigma_fuel, a.sigma_min, a.t_min_ev,
                              a.ion_mass_amu, 1.0, a.coulomb_log)


# aluminium: linear Us-Up Hugoniot (Marsh 1980) and Gruneisen parameter
AL_HUGONIOT = dict(rho0=2700.0, c0=5328.0, s=1.338, gamma0=2.0)


def add_eos_args(ap):
    g = ap.add_argument_group("liner equation of state")
    g.add_argument("--eos-liner", dest="eos_liner", choices=["ideal", "stiffened", "mie_gruneisen"], default="ideal")
    g.add_argument("--gamma-liner", dest="gamma_liner", type=float, default=None,
                   help="ideal/stiffened liner gamma (stiffened default: fit to the Al Hugoniot at u_p = 1 km/s)")
    g.add_argument("--p-inf-gpa", dest="p_inf_gpa", type=float, default=None,
                   help="stiffened p_inf, GPa (default rho0 c0^2 / gamma for Al)")
    g.add_argument("--eos-table", dest="eos_table", default=None,
                   help="tabulated liner EOS (SESAME ASCII with table 301, or .npz in SI); --eos-liner is the fallback "
                        "beyond the table")
    g.add_argument("--eos-table-material", dest="eos_table_material", type=int, default=0,
                   help="SESAME material number (default: the first 301 table in the file)")


def eos_from_args(a, rho_unit, v_unit):
    """Config keywords for the liner EOS (code units) and the liner's natural density in code units
    (None for an ideal-gas liner). Mie-Gruneisen uses the aluminium Hugoniot. With --eos-table the table is
    the liner in pure cells and this analytic law its fallback."""
    kw, rho_l = _eos_analytic(a, rho_unit, v_unit)
    if getattr(a, "eos_table", None):
        kw.update(eos_table=a.eos_table, eos_table_material=a.eos_table_material, tab_rho_unit=rho_unit,
                  tab_v_unit=v_unit)
    return kw, rho_l


def _eos_analytic(a, rho_unit, v_unit):
    from .eos import stiffened_from_hugoniot
    H = AL_HUGONIOT
    p_unit = rho_unit * v_unit**2
    if a.eos_liner == "ideal":
        return ({} if a.gamma_liner is None else {"gamma_liner": a.gamma_liner}), None
    if a.eos_liner == "stiffened":
        g_fit, _ = stiffened_from_hugoniot(H["rho0"], H["c0"], H["s"], 1000.0)
        g = a.gamma_liner or g_fit
        pinf = a.p_inf_gpa * 1e9 if a.p_inf_gpa is not None else H["rho0"] * H["c0"]**2 / g
        return dict(eos_liner="stiffened", gamma_liner=g, p_inf_liner=pinf / p_unit), H["rho0"] / rho_unit
    return (dict(eos_liner="mie_gruneisen", mg_rho0=H["rho0"] / rho_unit, mg_c0=H["c0"] / v_unit, mg_s=H["s"],
                 mg_gamma0=H["gamma0"]), H["rho0"] / rho_unit)


# NRL Plasma Formulary: P_br = 1.69e-32 Z n_e^2 sqrt(T_eV) W/cm^3 with n in cm^-3 (Z_eff = Z for one ion
# species, n_e = Z n_i). In SI with n in m^-3: 1.69e-38 W/m^3.
BREMS_SI = 1.69e-38


def bremsstrahlung_kwargs(rho_unit, v_unit, L0, z_eff=1.0, ion_mass_amu=2.0, t_min_ev=10.0, tracer_max=0.1):
    """Config keywords for optically thin bremsstrahlung in code units. Q = rad_coeff rho^2 sqrt(p/rho):
    n_e = Z rho/(A m_u), kT = A m_u p/((1 + Z) rho), energy-density rate unit rho_unit v_unit^3 / L0."""
    A, Z = ion_mass_amu, z_eff
    T1 = temperature_eV_per_code(v_unit, A, Z)
    coeff = BREMS_SI * Z**3 / (A * AMU)**2 * rho_unit**2 * np.sqrt(T1) / (rho_unit * v_unit**3 / L0)
    return dict(rad_coeff=coeff, rad_T_min=t_min_ev / T1, rad_tracer_max=tracer_max)


def add_radiation_args(ap):
    g = ap.add_argument_group("radiation")
    g.add_argument("--enable-rad", dest="enable_rad", action="store_true", help="optically thin bremsstrahlung in the fuel")
    g.add_argument("--z-eff", dest="z_eff", type=float, default=1.0)
    g.add_argument("--t-rad-min-ev", dest="t_rad_min_ev", type=float, default=10.0,
                   help="no cooling below this temperature")
    g.add_argument("--enable-fld", dest="enable_fld", action="store_true",
                   help="grey flux-limited diffusion radiation in the fuel (free-free opacities unless --opacity-table)")
    g.add_argument("--opacity-table", dest="opacity_table", default=None,
                   help="Rosseland / Planck opacities for FLD (SESAME 502/505 or .npz in SI); implies --enable-fld")


def radiation_from_args(a, rho_unit, v_unit, L0):
    ion = getattr(a, "ion_mass_amu", 2.0)
    kw = bremsstrahlung_kwargs(rho_unit, v_unit, L0, a.z_eff, ion, a.t_rad_min_ev) if a.enable_rad else {}
    table = getattr(a, "opacity_table", None)
    if getattr(a, "enable_fld", False) or table:
        kw.update(fld_kwargs(rho_unit, v_unit, L0, ion, a.z_eff, preset=None if table else "free-free"))
        if table:
            kw.update(fld_opacity_table=table, tab_rho_unit=rho_unit, tab_L_unit=L0,
                      tab_T_unit=temperature_eV_per_code(v_unit, ion, a.z_eff) * 11604.518)
    return kw


M_E = 9.1093837015e-31


def braginskii_kwargs(rho_unit, v_unit, L0, z=1.0, ion_mass_amu=2.0, coulomb_log=10.0, tracer_max=0.1, nernst=False):
    """Config keywords for magnetized electron conduction in code units.
    Spitzer-Braginskii: k_par = 3.16 n_e kT tau_e / m_e (heat flux per gradient of kT; Z = 1 coefficient),
    tau_e = 3.44e5 T_eV^1.5 / (n_e[cm^-3] lnL) s (NRL), so k_par = C T_eV^5/2 independent of density;
    magnetization x = omega_ce tau_e = (e B / m_e) tau_e. In code units, with T = p/rho:
    k_par = cond_kappa0 T^5/2 and x = cond_x0 |B| T^3/2 / rho."""
    A, Z = ion_mass_amu, z
    T1 = temperature_eV_per_code(v_unit, A, Z)             # eV per code temperature
    tau_coef = 3.44e11 / coulomb_log                        # tau_e = tau_coef T_eV^1.5 / n_e[m^-3]
    C = 3.16 * E_CHARGE * tau_coef / M_E                    # k_par [1/(m s)] = C T_eV^2.5, flux per grad(kT)
    kappa0 = C * T1**2.5 * E_CHARGE * T1 / (L0 * rho_unit * v_unit**3)
    B_unit = np.sqrt(MU0 * rho_unit * v_unit**2)
    x0 = (E_CHARGE / M_E) * tau_coef * T1**1.5 * B_unit * A * AMU / (Z * rho_unit)
    kw = dict(cond_kappa0=kappa0, cond_x0=x0, cond_tracer_max=tracer_max)
    if nernst:
        # v_N = -c_N (beta_wedge/|B|) grad(theta): kT/e = T1 theta volts, |B| = B_unit B, grad = grad/L0
        kw["nernst_coeff"] = T1 / (B_unit * L0 * v_unit)
    return kw


def add_conduction_args(ap):
    g = ap.add_argument_group("thermal conduction")
    g.add_argument("--enable-conduction", dest="enable_conduction", action="store_true",
                   help="magnetized electron conduction in the fuel (Spitzer k_par ~ T^5/2)")
    g.add_argument("--braginskii", action="store_true",
                   help="cross-field suppression from omega_ce tau_e (the default)")
    g.add_argument("--kappa-ratio", dest="kappa_ratio", type=float, default=None,
                   help="instead: set k_par/k_perp to this value in the initial fuel; it then follows the "
                        "Braginskii fit with x ~ |B| T^1.5 / rho")
    g.add_argument("--nernst", action="store_true",
                   help="Nernst field advection and Ettingshausen heat flux (needs --enable-conduction)")


def conduction_from_args(a, rho_unit, v_unit, L0, B_fuel, T_fuel, rho_fuel):
    """Config keywords for conduction; B_fuel, T_fuel (= p/rho), rho_fuel: initial fuel state, code units."""
    if not a.enable_conduction:
        return {}
    if a.braginskii and a.kappa_ratio is not None:
        raise SystemExit("--braginskii and --kappa-ratio are alternatives; give one")
    from .transport import x_for_kappa_ratio
    kw = braginskii_kwargs(rho_unit, v_unit, L0, getattr(a, "z_eff", 1.0), getattr(a, "ion_mass_amu", 2.0),
                           getattr(a, "coulomb_log", 10.0), nernst=getattr(a, "nernst", False))
    if a.kappa_ratio is not None:
        if a.kappa_ratio < 1:
            raise SystemExit("--kappa-ratio must be >= 1")
        kw["cond_x0"] = x_for_kappa_ratio(a.kappa_ratio) * rho_fuel / (max(B_fuel, 1e-300) * T_fuel**1.5)
    return kw


def twotemp_kwargs(rho_unit, v_unit, L0, z=1.0, ion_mass_amu=2.0, coulomb_log=10.0):
    """Config keywords for the two-temperature model: electron-ion energy equilibration
    nu = 2 m_e/(m_i tau_e), tau_e = 3.44e5 T_eV^1.5/(Z n_e[cm^-3] lnL) s, in code units nu = te_eq_coeff rho/theta_e^1.5."""
    A, Z = ion_mass_amu, z
    T1 = temperature_eV_per_code(v_unit, A, Z)
    tau_coef = 3.44e11 / (coulomb_log * Z)                  # tau_e = tau_coef T_eV^1.5 / n_e[m^-3]
    c_eq = 2 * M_E / (A * AMU) * Z * rho_unit / (A * AMU) / (tau_coef * T1**1.5) * (L0 / v_unit)
    return dict(two_temp=True, z_ion=Z, te_eq_coeff=c_eq)


def strength_kwargs(rho_unit, v_unit, G_gpa=26.0, Y_gpa=0.29, spall_gpa=1.2):
    """liner strength in code units (defaults: aluminium 6061, G 26 GPa, yield 0.29 GPa, spall 1.2 GPa)"""
    p_unit = rho_unit * v_unit**2
    return dict(strength=True, shear_modulus=G_gpa * 1e9 / p_unit, yield_strength=Y_gpa * 1e9 / p_unit,
                spall_strength=spall_gpa * 1e9 / p_unit)


def viscosity_kwargs(rho_unit, v_unit, L0, z=1.0, ion_mass_amu=2.0, coulomb_log=10.0):
    """Braginskii parallel ion viscosity eta_0 = 0.96 n_i kT_i tau_i, tau_i = 2.09e7 T_i^1.5 A^0.5/(Z^4 n_i[cm^-3] lnL)
    s (NRL), independent of density: eta_0 = visc_coeff theta_i^2.5 in code units (theta_i = T_i code)."""
    A, Z = ion_mass_amu, z
    T1 = temperature_eV_per_code(v_unit, A, Z)
    eta_SI_per_T = 0.96 * 2.09e13 * E_CHARGE * np.sqrt(A) / (Z**4 * coulomb_log)     # Pa s / eV^2.5
    return dict(visc_coeff=eta_SI_per_T * T1**2.5 / (rho_unit * v_unit * L0))


def fld_kwargs(rho_unit, v_unit, L0, ion_mass_amu=2.0, Z=1.0, kappa_R=(0.0, 0.0, 0.0), kappa_P=(0.0, 0.0, 0.0),
               preset=None, tracer_max=0.1):
    """Config keywords for grey FLD radiation (aegis/fld.py) in code units.

    kappa_R, kappa_P: (k0, m, n) for kappa = k0 rho^m T^n in m^2/kg with rho in kg/m^3 and T in kelvin.
    preset="free-free": hydrogen-like bremsstrahlung (Kramers) opacity, kappa_R = 7.4e18 Z^2/A rho T^-3.5 and
    kappa_P = 37 kappa_R (the free-free Planck/Rosseland ratio); order of magnitude only, use a table for
    real materials. The code temperature theta = p/rho maps to T = theta * T1 (fully ionised, T_e = T_i)."""
    c_light, sigma_sb = 2.99792458e8, 5.670374419e-8
    T1 = temperature_eV_per_code(v_unit, ion_mass_amu, Z) * 11604.518
    if preset == "free-free":
        k = 7.4e18 * Z * Z / ion_mass_amu
        kappa_R, kappa_P = (k, 1.0, -3.5), (37 * k, 1.0, -3.5)
    code = lambda k: (k[0] * rho_unit**(k[1] + 1) * T1**k[2] * L0, k[1], k[2])
    kr, kp = code(kappa_R), code(kappa_P)
    return dict(fld=True, fld_a=4 * sigma_sb / c_light * T1**4 / (rho_unit * v_unit**2), fld_c=c_light / v_unit,
                fld_kr0=kr[0], fld_kr_rho=kr[1], fld_kr_T=kr[2], fld_kp0=kp[0], fld_kp_rho=kp[1], fld_kp_T=kp[2],
                fld_tracer_max=tracer_max)
