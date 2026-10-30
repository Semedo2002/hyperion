# AEGIS-MHD: GPU MHD solver platform, and the magnetic sieve study of striped fields and the Richtmyer-Meshkov instability

The solver package is `aegis` (`pip install aegis-mhd`; the platform was called MagSieve-3D / `sieve` before, and the GitHub repository keeps that name).

A 2D and 3D compressible MHD solver (CuPy + a custom CUDA HLLD kernel) and the parameter study built on it. The study asks whether a spatially striped magnetic field can suppress ("lock") the Richtmyer-Meshkov instability of a shocked density interface while channelling the flow through the weak-field gaps.

**Result in one paragraph.** A uniform shock-parallel field locks the interface, in line with earlier work on magnetised RMI. Striped fields roughen the interface in proportion to their bar/gap contrast: the stripe-scale mode grows as a8 ~ 0.027 dP_B, converged over three resolutions. (An earlier "10% contrast" lock threshold did not survive the convergence study: it was measured against hydrodynamic enstrophy, which keeps growing with resolution.) A shock-parallel field is not compressed by the shock while the gas is, so a striped field that is in pressure balance upstream produces a steady field-aligned shear downstream. Magnetic tension does not resist this shear, and it deforms the interface at the stripe scale. Suppression and collimation therefore work against each other in this regime. Aligning the stripe phase with the interface ripple produces a predictable zero crossing of the RMI mode, but not a lock. In 3D, at the Lundquist number this grid resolves (S ~ 75), the field-aligned shear layers neither tear nor break down. See `paper/` for the write-up, including the grid-convergence study and its limitations.

## Layout

```
aegis/            solver package
  config.py       typed configuration (unknown keys raise ValueError)
  physics.py      CUDA HLLD wrapper, MUSCL, jump conditions, resistive terms, stripe equilibrium
  eos.py          two-material EOS: ideal gas, stiffened gas, Mie-Gruneisen (linear Us-Up)
  units.py        SI helpers: conductivity to code-unit diffusivity, Spitzer coefficient, liner EOS,
                  bremsstrahlung and Braginskii conduction coefficients
  hlld_kernel.cu  HLLD Riemann solver (one thread per face)
  solver.py       MHDSolver (SSP-RK3, constrained transport or GLM + Powell, Cartesian or cylindrical r-z,
                  BCs, energy budget) and PostProcessor
  ct.py           constrained transport: face-centred B, upwinded edge EMFs, vector-potential initialisation
  transport.py    Braginskii Z = 1 coefficients, Nernst field advection, Ettingshausen heat flux
  solver3d.py     MHDSolver3D: unsplit HLLD in x, y, z (same kernel, rotated), z-periodic, 3D seeding,
                  current-density and generated-field diagnostics
  diagnostics.py  mixing, enstrophy, interface spectrum, div B, energy
  circuit.py      RLC drive circuit coupled to an r-z liner (magnetic piston, energy-consistent inductance)
  fld.py          grey flux-limited diffusion radiation (Levermore-Pomraning, implicit CG on the GPU)
  radiation_gmg.py  geometric multigrid V-cycle preconditioner for the grey and multigroup FLD solves
  radiation_multigroup.py  multigroup FLD: photon groups, group opacities, coupled implicit BiCGStab solve
  cad_importer.py STL / STEP / IGES import, ray-column volume fractions and apertures, immersed walls
  agent_monitor.py  automatic run diagnostics: interface mode growth, flux / insulation loss, numerical health
  fno_subgrid.py  Fourier neural operator (inference + training on the autodiff tape), guarded radiation predictor
  parallel.py     MPI slab domain decomposition of the 2D and 3D solvers (halo and CT edge exchange)
  io3d.py         3D snapshots: VTK ImageData (.vti/.pvd) and HDF5
  tabular.py      tabulated EOS / opacities (SESAME ASCII, .npz) with GPU inversion and analytic fallback
  eos_fused.cu    device two-material / tabulated EOS, called inline by the HLLD kernel and the conversions
  eos_fused.py    its host side: packing, launches
  validation.py   shot data vs simulation: chi^2, RMS, peak offset, timing shifts, calibration, report plot
  experimental_suite.py  bundled open data (Marsh 1980 Cu / Ni Hugoniots) and the plate-impact benchmark
  amr.py          block-structured adaptive mesh refinement (2D, CT, refluxing, touching patches)
  autodiff.py     reverse-mode tape and DiffMHD, the differentiable ideal-MHD twin (adjoint gradients)
  hybrid_pic.py   adaptive kinetic-fluid hybrid: Kn / Larmor-radius classification, particle ions, exact exchange
  hall.py         generalized Ohm's law: Hall term, anisotropic (Spitzer) resistivity, on the CT staggering
  ions.py         ion gyroviscosity (finite-Larmor-radius stress) and the ion pressure for ion transport
main.py           CLI: verification tests, single cases, legacy campaign
tools/            generate_sesame.py: Al, Cu, D-T, Au, Pb EOS / opacity tables from a free-energy model
examples/         MTIF liner flux compression; synthetic magnetic probes (SI units); 3D target asymmetries;
                  MagLIF-style r-z implosion; metal-liner flux compression generator (FCG)
experiments/      sweeps A-H and analysis scripts; data/ holds the saved results
paper/            figure and .docx generators, figures/, magnetic_sieve.docx
scripts/          RMI benchmark runner/plots, render_3d.py (PyVista volume renders + MP4), mpi_check.py
rmi_output_gpu/   figures and GIFs written by the solver
```

## Installation

An NVIDIA GPU with CUDA 12 is required.

```
pip install -e .                  # package + dependencies (pyproject.toml)
pip install -e .[paper]          # also python-docx, for paper/make_paper.py
pip install -e .[viz,mpi]        # 3D rendering (PyVista, ffmpeg, h5py) and MPI (needs an MPI runtime too)
pip install -r requirements.txt  # or the exact tested versions
```

See `CONTRIBUTING.md` for the development workflow.

## Usage

```
python main.py --test all                     # verification suite, ~20 min on a Quadro P1000
python main.py --case striped --B0 3.5 --gap-fraction 0.9 --compare-hydro
python main.py --help
```

`--case` runs a 12 x 2 domain to t = 0.8 and writes figures and 61-frame GIFs to `rmi_output_gpu/`.

### Equation of state

Two materials, selected by the tracer: `eos_fuel` (where it is 0: `ideal` with `gamma`, or `stiffened` with
`p_inf`) and `eos_liner` (where it is 1: `ideal`, `stiffened` with `gamma_liner` and `p_inf_liner`, or
`mie_gruneisen` with `mg_rho0`, `mg_c0`, `mg_s`, `mg_gamma0`). Each law is written as eps = phi1(rho) p + phi2(rho);
mixed cells blend phi1 and phi2 linearly in the tracer, which is then a volume fraction advected
non-conservatively with the face values the energy flux uses (Shyue 1998), so a material interface at uniform
p and u stays in equilibrium. The HLLD kernel takes each face state's internal energy and sound speed from the
EOS; with one ideal gas the solver runs the original code path. Both the 2D and 3D solvers support it. The examples
take an aluminium liner in SI:

```
python examples/nearstar_mtif3_validation.py --eos-liner stiffened                # Al, gamma/p_inf fitted to its Hugoniot
python examples/nearstar_mtif3_validation.py --eos-liner mie_gruneisen            # Al: rho0 2700, c0 5328, s 1.338, Gamma0 2
python examples/mtif_flux_compression.py --eos-liner stiffened --p-inf-gpa 20 --gamma-liner 3.8
```

### Constrained transport and geometry

The magnetic field is face centred and evolved by constrained transport (Gardiner & Stone 2005): edge EMFs
from the HLLD fluxes with upwinded corrections, so the face divergence stays at round-off (about 1e-15) in 2D
and 3D. `set_vector_potential` initialises any field divergence-free; otherwise faces are built from the cell
values. `div_b="glm"` keeps the cell-centred GLM/Powell scheme the published sieve study used (the experiment
scripts and the RMI benchmark pin it, so their results reproduce). `geometry="cylindrical"` runs the 2D solver
in axisymmetric (r, z): x is the radius with a reflecting axis at x = 0, y the axial coordinate, the
out-of-plane components are -v_phi and -B_phi; fluxes are area weighted, the geometric sources
(rho v_phi^2 - B_phi^2 + p_T)/r and -(rho v_r v_phi - B_r B_phi)/r are added, B_phi follows
(curl E)_phi = d_z E_r - d_r E_z with plain flux differences (no metric factor; the toroidal flux is conserved),
and CT updates B_z with r-weighted edge EMFs (E_phi = 0 on the axis).

The 3D solver runs `geometry="cylindrical"` in (r, theta, z), scale factors (1, r, 1):
- cells are r dr dtheta dz, the radial flux divergence is (1/r) d(r F)/dr and the azimuthal one dF/(r dtheta),
  with the same momentum sources as 2D;
- the CT faces (B_r, B_theta, B_z) follow Stokes on the curved faces: dB_r/dt = -dE_z/(r dtheta) + dE_theta/dz,
  dB_theta/dt = -dE_r/dz + dE_z/dr, dB_z/dt = -(1/r) d(r E_theta)/dr + dE_r/(r dtheta), so the cylindrical
  divergence stays at round-off;
- with x_min = 0 the domain contains the axis (theta over 2 pi, even cell count). Ghost cells behind the axis are
  the cells at theta + pi with r and theta components reversed, E_z on the axis takes its theta average (E_z is
  single-valued there), and the axis cell's cell-centred B_r keeps all the m >= 1 part of the outer face value;
- without theta variation it reproduces the 2D r-z solver to 1.6e-14, and an m = 1 blob carried onto the axis
  along a uniform field converges with resolution (`--test cylindrical3d`);
- it covers ideal MHD with the EOS options and the tracer; the other 3D physics modules are Cartesian;
- the azimuthal spacing r dtheta shrinks toward the axis and would set the time step. `cyl_filter=True` filters
  the rings with r dtheta < `cyl_filter_frac` dr:
  - their tendencies (RK rates of the cell variables and the edge EMFs) keep only the azimuthal Fourier modes
    m <= m_crit(r) = max(`cyl_filter_mkeep`, floor(pi r / dr)), with `cyl_filter_mkeep` = 2 by default so
    m = 0, 1, 2 are never touched;
  - the time step then uses the resolved spacing pi r / m_crit, about dr as in r-z;
  - m = 0 is untouched, so ring sums (mass, energy) are unchanged, and B still follows from -curl E, so div B
    stays at round-off;
  - the blob of `--test cylindrical3d` crosses the axis in 16x fewer steps with the same centroid and field error
    (`--test cyl_filter`);
  - the cost is that structure finer than m_crit is removed near the axis; it uses CuPy's FFT (cuFFT; with the
    pip CUDA wheels, `pip install aegis-mhd[fft]`).

### Thermal conduction

Magnetized electron conduction, q = -[k_perp I + (k_par - k_perp) b b] . grad T with k_par = `cond_kappa0` T^5/2
and k_perp from Braginskii's Z = 1 fit, k_perp/k_par = (4.664 x^2 + 11.92)/(3.1616 (x^4 + 14.79 x^2 + 3.7703)),
x = omega_ce tau_e = `cond_x0` |B| T^3/2 / rho (T = p/rho), in the fuel (liner fraction below
`cond_tracer_max`). At a fuel-liner face the fuel conducts to the liner as an isothermal wall at `cond_wall_T`
(a Dirichlet condition half a cell away, with the fuel's anisotropic k, so a field parallel to the wall
insulates it); the heat goes into the liner cell, so total energy is conserved exactly. `cond_wall="insulating"`
and `cond_perp="simple"` (1/(1 + x^2)) restore the earlier choices; non-periodic domain faces are insulating. Fluxes live on cell faces; the gradient normal to a face is a two-point
difference and the transverse gradients are van Leer limited (Sharma & Hammett 2007), which keeps strongly
anisotropic conduction from creating new temperature extrema. It is a split step integrated with RKL2
super-time-stepping (Meyer, Balsara & Aslam 2014), about sqrt(dt/dt_explicit) stages, so it never limits the
hydrodynamic time step. `units.braginskii_kwargs` gives the Spitzer-Braginskii coefficients in code units (k_par
at 1 keV within 1.1% of the Spitzer value). Ion conduction is left out.

### Nernst and Ettingshausen effects

`nernst_coeff` (from `units.braginskii_kwargs(..., nernst=True)`, or `--nernst` in the examples) adds the
Nernst term to Ohm's law, E_N = -v_N x B with v_N = -(beta_wedge/(e|B|)) grad_perp(kT_e) (Braginskii Z = 1
beta_wedge): the field is carried down the temperature gradient, toward cold walls. It enters CT as an
upwinded edge EMF, so div B stays at round-off, and the time step respects |v_N|. Its Onsager partner, the
Ettingshausen heat flux (beta_wedge kT_e/e) b x J, and the Poynting flux of E_N enter the energy equation as
face fluxes, so total energy is conserved exactly. 2D solver (Cartesian and r-z) and 3D: there the Nernst EMF
-v_N x B sits on all three edge orientations with the same upwinding, and a z-invariant 3D run reduces to the
2D scheme (to 7e-12 over 1240 steps, the two forms summing in a different order).

### Hall term and anisotropic resistivity

`aegis/hall.py` adds to the induction and energy equations the part of the generalized Ohm's law beyond eta J,

    E' = d_i (J x B) / rho + (eta_perp - eta_par) J_perp,     J_perp = J - (b . J) b,

with `hall_di` the ion inertial length at unit density in code units (the Hall term J x B / (e n_e) for Z = 1)
and `eta_perp_ratio` = eta_perp / eta_par (1.96 for a magnetized Z = 1 Spitzer plasma; the configured resistivity,
any `eta_model`, is the parallel value). J and B are formed on the cell corners for E'_z, which joins the CT corner
EMF, and on the faces for E'_x, E'_y, which move B_z in flux form; the energy equation gets the Poynting flux
E' x B, so total energy is conserved to round-off and div B stays at round-off. Whistlers set an explicit time
step, omega = d_i |B| k^2 / rho at the grid scale (kept at 0.75 of the SSP-RK3 imaginary-axis bound at cfl 0.3).
2D Cartesian with CT; the electron pressure gradient term is not included.

### Two-temperature plasma

`two_temp=True` carries the electron entropy s_e = p_e/rho^gamma as an advected scalar (electrons are
compressed adiabatically; shock heating goes to the ions), with theta_e = (1 + Z)/Z p_e/rho and theta_i = (1 + Z)
p_i/rho (`z_ion`). Electron-ion equilibration relaxes theta_e - theta_i exactly at rate (1 + Z) nu, nu =
`te_eq_coeff` rho/theta_e^3/2, at fixed total energy. Conduction, bremsstrahlung, Ohmic and Ettingshausen heating
act on the electrons. `units.twotemp_kwargs` gives the Spitzer equilibration coefficient in code units.

### Viscosity

`visc_coeff` adds Braginskii parallel viscosity, Pi = -3 eta_0 (b b - I/3)(b b : grad v - div v/3), which
reduces to isotropic Navier-Stokes where B = 0. Stresses are face fluxes of momentum and energy (energy
conserved exactly), integrated as an RKL2 split step like conduction (`units.viscosity_kwargs`).

### Ion conduction and gyroviscosity

In two-temperature mode `cond_kappa_i0` adds ion heat conduction on theta_i = (1 + Z) p_i / rho, k_i,par =
`cond_kappa_i0` theta_i^5/2, k_i,perp = k_i,par / (1 + x_i^2) with x_i = `cond_x0_i` |B| theta_i^3/2 / rho: the same
RKL2 split step as the electrons, with the electron pressure held (the electron entropy is untouched) and the
energy conserved. `gyro_di` adds the ion gyroviscous stress (Braginskii eta_3 / eta_4, the non-dissipative
finite-Larmor-radius part) in its compact form

    Pi = p_i / (4 Omega_i) [ (b x W) . (I + 3 b b) + transpose ],   Omega_i = |B| / gyro_di,

(for b = z: Pi_xx = -Pi_yy = -eta_3 W_xy, Pi_xy = eta_3 (W_xx - W_yy) / 2, eta_3 = p_i / (2 Omega_i)), as face fluxes
of momentum and energy, so both are conserved; it does no work (Pi : W = 0). A fast wave across B then has the
FLR dispersion omega^2 = k^2 c_f^2 + (eta_3 k^2 / rho)^2. The dispersive waves set an explicit step like the Hall
term. p_i is p - p_e in two-temperature mode and p / (1 + Z) otherwise. 2D Cartesian.

### Liner strength and spall

`strength=True` gives the liner (volume fraction above 0.5) a deviatoric stress (S_xx, S_yy, S_xy) evolved by a
hypoelastic Jaumann rate with shear modulus `shear_modulus`, returned radially to the von Mises surface sqrt(3
J2) = `yield_strength` after each step. The stress enters the momentum and energy fluxes (and the hoop term in
r-z), and the time step uses the longitudinal speed c_L = sqrt(c^2 + 4G/(3 rho)). Where the tension exceeds
`spall_strength` the cell is marked damaged: no tension (p >= 0) and no strength from then on.
`units.strength_kwargs` converts SI moduli. In 3D the stress gains S_xz and S_yz and the Jaumann rotation the
full spin tensor; a z-invariant 3D run matches the 2D solver to 1e-13, and a shear wave along z oscillates at
k sqrt(G/rho) to 0.3%.

### Radiation

Optically thin bremsstrahlung, P = 1.69e-38 Z n_e^2 sqrt(T_eV) W/m^3 (NRL formulary, n in m^-3), acts on the
fuel only: where the liner fraction is below `rad_tracer_max` (0.1) and T is above `rad_T_min`. It is a
split step after each time step using the exact solution at fixed density, sqrt(T) falling linearly in time, so
it is stable for any time step. The energy removed accumulates in `cumulative_radiated_energy` and enters the
energy budget. `units.bremsstrahlung_kwargs` converts to code units; the examples take `--enable-rad`,
`--z-eff`, `--t-rad-min-ev`. At the examples' default parameters the fuel stays below 1 eV and nothing
radiates; a hot dense core does:

```
python examples/mtif_flux_compression.py --enable-rad --v-ref-kms 100 --rho-ref 100
```

### Resistivity

`eta_model` selects `constant` (`eta_resistivity` everywhere, the default), `material` (`eta_fuel` where the
tracer is 0, `eta_liner` where it is 1, log blend between) or `spitzer` (`eta_spitzer` T^-3/2 with T = p/rho
floored at `eta_T_min`, capped at `eta_max`). Face values use the harmonic mean, so a conductivity jump carries
the series resistance of the two cells; the resistive terms stay in face-flux form and conserve energy exactly,
and `cumulative_ohmic_heating` integrates eta J^2. The explicit time step uses the largest eta on the grid. The
examples take SI conductivities:

```
python examples/mtif_flux_compression.py --sigma-liner 3.5e7          # aluminium liner, S/m
python examples/nearstar_mtif3_validation.py --eta-model spitzer --sigma-liner 3.5e7
```

### RLC drive circuit

`aegis/circuit.py` replaces a prescribed liner velocity with a capacitor bank (C, V0, series L0 and R) whose
current runs up an r-z liner and back through a return can at r_return. The circuit is d(LI)/dt = V - RI with
L = L0 + (mu0 l / 2 pi) <ln(r_return / r_out)>, i.e. dI/dt = (V - RI - I dL/dt) / L, integrated by the
trapezoid rule in flux form (exact energy identity for a fixed load). The liner feels the J x B of its surface
current, f = P_B(r) dC/dr on its outer side with P_B = mu0 I^2 / (8 pi^2 r^2), applied as an exact kinetic
energy kick each step. The inductance increment of each step is the one whose mechanical work, (L1 - L0) I0 I1
/ 2 for this scheme, equals the work the grid received, so bank + liner conserve energy to round-off; the
geometric inductance of the liner's actual outer radius is reported alongside (they agree to 1% at 64-96 cells
across the liner region). The FCG and MagLIF examples take `--capacitance-uf`, `--voltage-kv`,
`--inductance-nh`, `--resistance-mohm` and `--r-return-mm`; the liner then starts at rest. Verified against the
analytic RLC discharge (3e-7) and an incompressible thick-shell liner on the same circuit (outer radius to 1.3%
of the distance moved, current to 0.1%).

### Grey flux-limited diffusion radiation

`fld=True` (`units.fld_kwargs`) adds a radiation energy density E_r: dE_r/dt + div(v E_r) = div(D grad E_r) +
c rho kappa_P (a T^4 - E_r), with the Levermore-Pomraning limiter D = c lambda(R) / (rho kappa_R) evaluated on
each face, and power-law Planck and Rosseland opacities (a free-free preset, or your own). Each step advects E_r
with the gas, then solves the stiff part by backward Euler with the emission linearised, the gas temperature
eliminated, and the limiter re-evaluated over a few Picard passes; the symmetric system (volume weighted, so
also in r-z) goes to a matrix-free conjugate-gradient solver on the GPU, preconditioned by Jacobi or, with
`fld_precond="mg"`, a geometric multigrid V-cycle (below). The final update
is written in face-flux form from the solution, so gas + radiation energy is conserved to round-off whatever
the solver residual. 2D, r-z and 3D. Radiation pressure is left out (fine while E_r/3 << p), and FLD couples to
a single gas temperature (not with `two_temp`).

### Multigrid preconditioner for the radiation solves

`fld_precond="mg"` replaces the Jacobi (grey) or block-Jacobi (multigroup) preconditioner by a geometric
multigrid V-cycle (`aegis/radiation_gmg.py`), on the GPU arrays:
- **Hierarchy:** cell-centred, 2:1 coarsening along every axis that is even and at least 4 cells long. Coarse
  operators are rediscretised: volumes and the multigroup coupling vectors summed, face conductances summed across
  the face and halved along a coarsened axis.
- **Transfers:** restriction sums over the children, and prolongation (its transpose) copies the correction back.
- **Smoothing:** red-black Gauss-Seidel on the cell blocks, each group block inverted exactly by Sherman-Morrison,
  2 pre- and 2 post-sweeps in mirrored order. The V-cycle is symmetric, so it also preconditions the grey CG.
- **Iterations (stiff case, D dt / dx^2 ~ 20-80, three steps):**
  - Jacobi: grey 1962 / 3960 / 8010 at 64^2 / 128^2 / 256^2;
  - multigrid: 70 / 71 / 71 (O(N) work);
  - 6-group Kramers: 2401 / 5096 / 10985 against 85 / 103 / 122;
  - solutions agree to the tolerance, and energy stays at round-off (the conservative update is unchanged).
- **Wall time (P1000, idle):** the V-cycle launches many small kernels, so it pays off with size:
  - grey: 0.6x / 0.8x / 1.4x the Jacobi speed at 64^2 / 128^2 / 256^2;
  - 6 groups: 1.6x / 2.5x / 3.6x.
  These are short of a 5-10x wall-clock gain at these sizes; the iteration gain (28-113x) would show fully only on
  larger grids or with fused kernels.
- **Default:** Jacobi stays the default.

### Multigroup radiation diffusion

`fld_groups > 0` splits the FLD radiation into photon groups (`aegis/radiation_multigroup.py`):

    dE_g/dt + div(v E_g) = div(D_g grad E_g) + c rho kP_g (a T^4 b_g(T) - E_g),   D_g = c lambda(R_g) / (rho kR_g)

- **Groups:** bands of photon energy in code temperature units (`fld_group_edges`, or log-spaced between
  `fld_group_emin` and `fld_group_emax`). The first starts at 0 and the last runs to infinity, so the Planck
  fractions b_g sum to 1.
- **Group opacities (Planck and Rosseland means):**
  - `grey`: the power laws in every group;
  - `kramers`: a free-free nu^-3 (1 - e^-x) shape scaled to the grey Planck mean, times 1 + `fld_edge_jump`
    above a photo-ionization edge at `fld_edge`, with group means by quadrature;
  - a table (`fld_mg_table`): `python tools/generate_sesame.py --groups 10 100 1000` writes kR_g, kP_g from the
    generator's free-free, screened-hydrogenic bound-free and Thomson model (bound-free edges included, no
    lines).
- **Implicit step:** backward Euler with a T^4 b_g linearised and the gas temperature eliminated. This couples
  the groups per cell through one rank-one term, so the system is solved by BiCGStab on the GPU, preconditioned
  with each cell's exact group-block inverse (Sherman-Morrison).
- **Conservation:** the update is written in face-flux form plus the exchange the gas receives back, so gas +
  radiation energy is conserved to round-off.
- **Checks (`--test multigroup`):**
  - one group equals grey FLD;
  - grey groups sum to grey FLD in the thick limit;
  - Kramers groups with an edge relax to the Planck spectrum;
  - decoupled groups diffuse with their own D_g over three decades;
  - a solver run with an Al group table conserves energy.
- **Scope:** 2D (Cartesian, r-z) and 3D; not under AMR (grey FLD is).

### Neural-operator predictor for the multigroup solve

`aegis/fno_subgrid.py` is a small Fourier neural operator (FNO) that runs inside the time step as a
predictor for the implicit multigroup radiation solve:
- **Model:** lift, then L Fourier layers (lowest modes mixed by learned complex weights, plus a 1x1 path, GELU),
  then a projection. The truncated transforms are cosine / sine matrix products, so the model runs on CuPy or
  NumPy at any resolution. It is trained with the reverse-mode tape of `aegis/autodiff.py`, with no ML framework.
  GPU matrix products need cuBLAS (`pip install aegis-mhd[fno]`).
- **Use:** with `fld_fno_model`, `radiation_multigroup.step` asks the model for E^{n+1} of every group before the
  Krylov solve. The guard is the relative residual of that linear system:
  - at or below `fld_fno_tol`, the prediction is accepted and the first Picard pass skips BiCGStab;
  - otherwise BiCGStab runs, from the prediction only if its residual is under half that of E^n.
  Either way the step ends with the conservative face-flux + exchange update built from the solution, so gas +
  radiation energy stays at round-off whatever the network predicts.
- **Training:** `tools/train_fno_radiation.py` generates data with the exact solver, trains, and benchmarks:
  - data: random smooth T, rho, E_g fields on 32^2, 6 Kramers groups with an edge;
  - training: 720 samples, 150 epochs, ~54 min on a P1000.
- **Measured result (`results/fno/radiation_fno_report.json`), negative:**
  - validation loss 0.16 (normalised MSE) against 0.018 on the training set;
  - the predictor's linear-system residual is worse than E^n's (median 3.1 against 0.26), because the system is
    stiff (D dt / dx^2 ~ 20) and a 1% field error is amplified by the operator;
  - so it is never accepted, the solve starts from E^n, and the iteration count (219 per step) is unchanged;
  - inference costs 64 ms against 6.6 s for an exact solve, so the speed-up would only appear for a predictor
    accurate to ~1e-5.
  The engine and the guard work (`--test fno`), but this predictor does not pay off. A better preconditioner
  (multigrid) is the more promising route for this solve.
- **Not done:** no learned sub-grid turbulence closure.

### Tabulated EOS and opacities

`aegis/tabular.py` reads SESAME ASCII files (EOS table 301; opacity tables 502 Rosseland and 505 Planck, taken
as log10 kappa in cm^2/g against log10 rho in g/cc and log10 T in eV, so check your library's convention) or an
SI `.npz` table (rho, T, p, e, optional cs, kappa_R, kappa_P). Tables are resampled onto a uniform log rho x log
T grid; positive quantities are interpolated in log space, so power laws are reproduced exactly. The lookup
inverts e(rho, T) or p(rho, T) along T at fixed rho (binary search) and interpolates bilinearly; c^2 comes from
the table or from its thermodynamic derivatives. Beyond the table the values are
extrapolated from the edge interval and blended into the analytic liner law (Mie-Gruneisen or stiffened gas)
over two grid cells, so p(e) stays continuous. In the two-material solver the table is the liner in pure liner
cells; mixed cells keep the tangent stiffened gas of the volume-fraction closure (blended over a liner fraction of
0.98 ... 1, with e(p) an exact Newton inverse of p(e) in the blend band). Every example takes `--eos-table` (and
`--eos-table-material` for a SESAME file with several materials) and `--opacity-table` (which turns on FLD with
tabulated kappa_R and kappa_P). No SESAME data ship with the code (LANL distributes them under licence):
`tabular.mie_gruneisen_table` and `tabular.ideal_gas_table` write synthetic tables, which is how this is tested.

The two-material EOS, analytic or tabulated, runs as device functions (`aegis/eos_fused.cu`): the conversions,
the face floors and the HLLD kernel evaluate p(rho, eps), eps(rho, p) and c^2 per cell or face inline (table
search, bilinear interpolation, fallback blend and the blend-band secant all in the thread), and every
`EOS` / `TabularEOS` method on device arrays is one launch. They reproduce the array implementation to
round-off (1e-14 over a run; `AEGIS_ARRAY_EOS=1` switches back to it). `scripts/bench_tabular.py`, a 2D Al
impact on a Quadro P1000:

| grid | analytic MG, array EOS | table, array EOS | analytic MG, device EOS | table, device EOS | table / analytic |
|---|---|---|---|---|---|
| 64^2 | 70.7 ms/step | 873 ms/step (12x) | 15.7 ms/step | 15.4 ms/step | 0.98 |
| 256^2 | 286 ms/step | 759 ms/step (2.7x) | 70.7 ms/step | 68.8 ms/step | 0.97 |

Interpolation is bilinear (bicubic is not implemented).

### Generated material tables

`tools/generate_sesame.py` writes EOS and opacity tables for Al, Cu, D-T, Au and Pb without LANL data (SI .npz,
and SESAME ASCII 301 / 502 / 505 with `--sesame`; default 300 x 160 points over 1e-4 ... 30 rho0 and up to 1e8 K,
about a minute per material; tables/ is not committed). Everything follows from one Helmholtz free energy, so p, e
and c_s are thermodynamically consistent:
- cold: a Vinet curve (Murnaghan-like near rho0, energy saturating at the cohesive energy in expansion), with
  the 0 K reference density and K0 fitted so that p = 0 and the bulk sound speed equals the measured value at
  ambient conditions;
- ions: a Gruneisen solid with Lindemann melting going over to an ideal gas, built so that below melting
  p_ion = 3 Gamma rho k T / m exactly;
- electrons: an ideal Fermi-Dirac gas of Z* n_i free electrons (Fermi integrals by quadrature with exact-derivative
  Hermite interpolation and the Sommerfeld series) with Dirac exchange, thermal part only. Z* is the
  Thomas-Fermi ionization (More's fit), and for D-T also Saha with the binding and mixing-entropy terms;
- opacities: Planck and Rosseland means integrated over frequency of free-free, screened-hydrogenic
  bound-free and (Rosseland) Thomson scattering.

Checks (`--test sesame`):
- the principal Hugoniot of Al, Cu, Au and Pb is within 2.7% of the measured Us-up fits up to up = 3 km/s, which
  were not fitted;
- the fully ionized limit p = (1 + Z) rho k T / m and the Thomson floor of kappa_R are recovered;
- thermodynamic consistency holds to 1e-5;
- the solver's tabular EOS reproduces the model to about 1e-3 in the compressed and hot branches.

These are smooth, consistent development tables, not evaluated data:
- there is no line opacity (the means are lower bounds for partially ionized mid- and high-Z matter);
- the classical ion heat capacity leaves solid D-T at 10 K with 0.036 GPa of thermal pressure at its measured
  density;
- in the expanded liquid-vapour region tension and thermal pressure nearly cancel and the table interpolation
  error reaches several %;
- a sliver of the solid-vapour corner was raised by up to 1e-5 of its row to keep p and e increasing in T.

```
python tools/generate_sesame.py                                   # tables/al.npz ... pb.npz
python examples/fcg_flux_compression.py --eos-table tables/al.npz --opacity-table tables/al.npz
```

### Shot calibration against diagnostics

`aegis/validation.py` compares simulated observables with measured traces (VISAR free-surface velocity, B-dot
probe voltage, radiography radii; CSV with time, value, sigma): RMS error, reduced chi^2, and the peak alignment
offset (the delay maximising the cross-correlation of simulation and data). Timing shifts can be fitted as
nuisance parameters with a Gaussian prior from the fiducial uncertainty, and simulated signals pass through the
instruments' response (10 ns FWHM by default). A scan of one model parameter gives the calibrated value: a coarse
scan, a golden-section search inside its best bracket (chi^2 is sharply peaked and not quadratic far from the
minimum), and a least-squares parabola over the whole chi^2 valley, with sigma from Delta chi^2 = 1 inflated by
sqrt(chi^2_min / dof) when the fit is worse than the errors. For a pulsed-power liner the observables come from an
FCG run with the RLC drive: VISAR is the velocity of the liner's outer surface, a B-dot probe in the return gap
sees B_theta = mu0 I / (2 pi r_p) and so measures dI/dt, radiography is the cavity radius.

`examples/experimental_shot_fitting.py` runs the whole chain. Without data files it first makes a synthetic shot
(clearly labelled as such in every output): an FCG run at a hidden series inductance of 16.2 nH sampled like the
instruments, with Gaussian noise and timing offsets drawn from a 5 ns fiducial uncertainty. Scanning 12-22 nH
(5 points, then golden section: 23 runs plus the hidden one) it recovers 16.24 +- 0.03 nH, within 1.1 sigma of the hidden value, with
chi^2_nu 0.81 (VISAR), 0.90 (B-dot) and 2.6 (six radiography frames), 0.89 overall. With real traces:
`--visar v.csv --bdot b.csv --radiography r.csv --param inductance_nh --range 10 25`. Two lessons from building it:
implosion timing makes chi^2 so steep in the circuit inductance (about 0.1 us per nH) that a free timing shift per
diagnostic is nearly degenerate with it, so the shifts need a fiducial prior; and an observable taken from the
tracer-based radius rings at the cell-crossing frequency differently in every run, so VISAR must be the material
velocity at the surface.

![Shot calibration](results/shot_fitting/shot_calibration_fit.png)

### Open experimental data and benchmarks

`aegis/experimental_suite.py` bundles open, citable experimental data and runs the solver against it:
- **Bundled data:** copper (145 shots) and nickel (19 shots) shock Hugoniots, shock velocity against particle
  velocity from plate impacts. They come from Marsh (ed.), *LASL Shock Hugoniot Data* (University of
  California Press, 1980), as transcribed in LLNL's [BALSCD](https://github.com/llnl/BALSCD) repository (MIT
  licence; its notice files sit next to the data in `aegis/data/experimental/`). Each file's SHA-256 is checked
  on load.
- **Benchmark:** `hugoniot_benchmark()` runs symmetric plate impacts with the solver and the generated Cu table
  at a set of particle velocities, measures the shock speed from the front's motion, and reports RMS and
  chi^2_nu against the data (the data carry no error bars, so a 2% sigma is assumed and stated).
  `examples/shot_benchmark.py` writes `results/shot_benchmark/shot_benchmark_comparison.png` and `summary.json`:
  10 impacts against 143 shots with Up <= 3 km/s, RMS 0.100 km/s (1.6%), chi^2_nu 0.64. The residuals show the model
  Hugoniot slightly too shallow, 1-2% high below Up = 1 km/s and low above 2.5 km/s (its slope is a prediction;
  only the ambient sound speed was fitted).

![Cu Hugoniot benchmark](results/shot_benchmark/shot_benchmark_comparison.png)
- **Not bundled:** no open pulsed-power shot data (B-dot, VISAR traces, x-ray core radii) in a machine-readable,
  citable form were found, so none ship with the code. Your own traces load with `load_shot()` (CSV: t, value,
  sigma) and compare with FCG / MagLIF runs through `aegis/validation.py` (`examples/experimental_shot_fitting.py`).

### Differentiable solver and inverse design

`aegis/autodiff.py` computes exact gradients of objectives of a simulation with respect to its inputs by a
discrete adjoint (reverse mode):
- **Tape:** a small reverse-mode tape for NumPy / CuPy arrays (`Tensor`, `record()`, `backward()`): each operation
  stores its vector-Jacobian product.
- **Twin:** `DiffMHD`, a twin of `MHDSolver`'s ideal-MHD step written with those operations, following
  `hlld_kernel.cu` and the fused MUSCL / CT-EMF kernels line by line: van Leer MUSCL, HLLD, Gardiner-Stone
  corner EMF, SSP-RK3, the same boundary conditions, Cartesian or r-z. A forward run reproduces the CUDA solver
  to round-off.
- **Adjoint:** checkpointed. The forward pass stores the state after every step; the backward pass re-runs one
  step at a time under the tape and carries the adjoint state back. One backward pass gives dJ/dp for every
  parameter at about 3-4x the forward cost.
- **What can depend on parameters:**
  - the initial state, built from Tensor operations (`cells_to_state`);
  - a boundary drive (B_phi on the outer radial boundary, a magnetic piston for a current I(t));
  - a running objective (summed over steps with weight dt) and a final objective.
  Time steps come from the forward run's CFL condition and are held fixed.
- **Differentiability:** the scheme is piecewise smooth (limiter, HLLD branches, floors); the gradient is that of
  the branch taken. Exactly symmetric states sit on a kink. With z-uniform data the van Leer limiter sees zero
  slopes, so the one-sided derivatives differ there; start designs slightly off symmetry. Finite differences
  agree with the adjoint only where their step stays inside one smooth piece.
- **Scope:** ideal gas, ideal MHD with `div_b='ct'`, 2D. It refuses resistivity, EOS tables, transport,
  radiation, strength and 3D.
- **Inverse design:** `examples/optimize_liner_shape.py` optimises the axial shape R(z) = R0 + sum a_k cos(2 pi k z / L)
  of an imploding liner (fixed mass and kinetic energy) to maximise the peak axial field on the axis at the
  mid-plane, by projected Adam on adjoint gradients; `--drive` adds a magnetic piston and reports dB_peak/dI.
  The design is axisymmetric (a fully 3D liner is outside the 2D twin).
  Default run (56 x 32, 3 modes, 12 iterations, ~30 min on one CPU core): the peak window field rises
  steadily from 7.4 B0 to 11.7 B0 (smooth time maximum; the instantaneous peak goes from 9.3 to 14.8 B0). The
  optimiser pulls the liner inward at the mid-plane, with the first mode at its bound |a_1| = 0.15. The adjoint
  dB_peak/da_1 agrees with a central difference (h = 1e-8) to 4e-9
  (`results/liner_optimization/`).

### Adaptive kinetic-fluid hybrid (particles where MHD breaks down)

`pic=True` lets cells switch from fluid MHD to particle ions where the continuum description fails
(`aegis/hybrid_pic.py`, 2D Cartesian, gamma = 5/3):
- **Classification (every step, on the GPU):** local Knudsen number Kn = lambda / L, with
  lambda = `pic_mfp0` theta^2 / rho and L the density / pressure gradient length. A cell turns kinetic when
  Kn > `pic_kn_on` or the ion Larmor radius r_L = `pic_di` sqrt(theta) / |B| exceeds `pic_rl_on` cells. It
  returns to the fluid when both fall below `pic_hysteresis` times their thresholds.
- **Conversion:** a new kinetic cell gets `pic_ppc` particles sampled from a drifting Maxwellian, with the
  cell's mass, momentum and thermal energy matched exactly. A cell returning to the fluid keeps the particle
  moments as its state.
- **Kinetic step (a split step after the MHD stages):**
  - particles move with the Boris pusher in B and E = -u x B (q/m = 1 / `pic_di`), bilinear in the cell-centred
    fields;
  - those that leave the kinetic region are dropped, and sparse cells are resampled;
  - their NGP moments replace rho, rho u and E - B^2/2 in the kinetic cells; B is untouched (CT, div B at
    round-off);
  - an extra time-step limit keeps Omega_i dt <= 0.5.
- **Conservation:**
  - cells on the edge of the kinetic region, which exchange HLLD fluxes with the fluid, are matched cell by cell
    to the conservative fluid step;
  - every connected interior region is matched as a whole (weight scale, velocity shift, peculiar-velocity
    scale), and the round-off of the particle sums is put back through ordered device sums;
  - so mass, momentum and total energy equal those of the conservative fluid step to round-off, while inside a
    region the particles decide where mass, momentum and heat go.
- **Model limits:**
  - the particles carry the whole thermal energy (cold-electron hybrid);
  - E has no Hall or electron-pressure term;
  - the fluid sees only the particles' scalar pressure;
  - a kinetic region needs to be a few cells wide to matter, because its edge cells follow the fluid.

### Adaptive mesh refinement

`aegis/amr.py` is a two-level, block-structured AMR for the 2D Cartesian solver (Berger-Colella style):
ratio-r patches (each an `MHDSolver` on a sub-rectangle) take r sub-steps per coarse step, all patches
advancing stage by stage together. Ghost cells and faces come from touching patches (same stage) or from the
coarse level (limited linear in space, linear in time). At each synchronisation the coarse cells next to a
patch are refluxed with the fine fluxes, coarse faces next to a patch get the fine edge EMFs, coarse cells and
faces under a patch become fine averages, and on an edge shared by two patches the lower-index one owns the
faces and corners. Mass, momentum and energy are then conserved exactly and the face divergence stays at
round-off on both levels. The volume fraction is refluxed with both of its terms and prolongation uses one
limiter per cell for all variables, so a material interface at uniform pressure and velocity crosses
coarse-fine boundaries undisturbed (to 1e-13). Patches are regridded from |grad rho| dx / rho, pressure jumps,
interface curvature and field-line curvature flags (optionally inside a region mask), clustered into
rectangles that are split where poorly filled; new patches take the old fine data where they overlap old
ones, and their face fields come from one composite vector potential, so they are divergence-free and agree
on shared edges by construction.

Non-ideal terms are coupled across the levels with the same flux registers:
- resistive MHD: the eta J part of the corner EMF is in the EMF correction, and the resistive Poynting flux
  and the B_z diffusion flux are refluxed;
- liner strength: the 2D deviatoric stress (S_xx, S_yy, S_xy and damage) is prolonged, restricted and
  time-interpolated into patch ghosts, and the stress fluxes of momentum and energy are refluxed;
- Braginskii conduction (RKL2) and grey FLD (backward Euler) are split steps: on a patch they conduct to ghost
  cells from the coarse level at the end of the sub-step, and return their time-integrated face fluxes. The
  RKL2 fluxes go through the same linear recursion as the energy, so the energy change is exactly their
  divergence. These are refluxed into the coarse energy, or into the coarse E_r, which is prolonged and
  restricted like U.

- the Hall term, anisotropic resistivity, Nernst and Ettingshausen terms: their corner EMFs are part of the
  corrected EMF, and their B_z and energy face fluxes are refluxed;
- gyroviscous stresses: refluxed like the strength stresses;
- viscosity and ion conduction: split steps like conduction (ghost ring, time-integrated face fluxes);
- two-temperature: the electron entropy rides with the hydro fluxes, and an electron-energy register
  (electron conduction, Ettingshausen) puts the electron share of refluxed energy into the coarse electron
  entropy; without it that heat would land in the coarse cells' ions;
- optically thin radiation: local, with the coarse level not radiating under patches and `totals()` counting
  what every level radiated.

With patches that touch and move, mass, momentum and gas + radiation energy stay conserved to round-off for
every term; with the corresponding registers switched off the error is 1e-7 to 1e-3. Every 2D Cartesian module
now runs under AMR.

`examples/amr_implosion_demo.py` implodes the 2D magnetized liner (1% mode-6 ripple, 1 mm per code unit) to
t = 0.2 three ways: uniform 128^2 (15.6 um cells), uniform 512^2 (3.9 um) and AMR 128^2 + ratio-4 patches.
The AMR run resolves the instability fingers and spikes that the coarse run washes out (density error against
the 512^2 run 0.40 vs 0.62 for the coarse run; core radius 0.326 between coarse 0.272 and fine 0.365), using
44% of the fine cells on average with up to about 20 patches, energy to 7e-14 and div B at round-off, in 232 s against
273 s for the uniform 512^2 run on a Quadro P1000.

Launch overhead: the coarse step, a fine sub-step of every patch together (three RK stages, the flux registers and
the post-step) and the synchronisation are each recorded once as a CUDA graph and replayed until the next regrid
(dt and the ghost time-interpolation weights are device scalars, ghost frames sit in persistent buffers, the
registers accumulate in place, and a regrid that returns the same boxes keeps the patches). Prolongation is one
kernel, and so are the ghost-ring fill and the register updates. The replay matches the eager code bit for bit.
Before this the patches stepped stage by stage from Python and the same AMR run took 3270 s.
`examples/amr_localized_benchmark.py` times localized problems against the uniform fine grid:

| problem | fine cells | AMR | uniform fine | density L1 vs fine (coarse / AMR) |
|---|---|---|---|---|
| Mach-3 shock on a dense bubble, 256 x 128 + ratio 4 vs 1024 x 512 | 8% | 217 s | 371 s | 0.032 / 0.020 |
| liner implosion, refining only the fuel, 128^2 + ratio 4 vs 512^2 | 8% | 69 s | 111 s | 0.53 / 0.41 |
| liner implosion, refining liner, shocks and core (above) | 44% | 232 s | 273 s | 0.62 / 0.40 |

The remaining cost per fine cell is still about twice a uniform grid's (each patch is its own set of kernels:
with tens of patches the graph holds tens of thousands of small nodes), so AMR pays off when the refined region
is a modest fraction of the domain.

![AMR implosion](results/amr_implosion/amr_implosion.png)

### Multi-GPU domain decomposition

`aegis.parallel.distributed(MHDSolver3D, cfg)` (or `MHDSolver`) run under `mpiexec` splits the grid into x
slabs, one per rank, each keeping the global dx to the last bit. Ghost layers of the cells, the CT face fields
and the liner stress are swapped with the neighbours, the edge EMFs on a shared x-face are taken from the left
rank on both sides (so both update the shared face identically and div B stays at round-off), and dt is the
global minimum. Halos go device to device through NCCL where CuPy has it (Linux), otherwise through MPI staged
in host memory. `scripts/mpi_check.py` compares 2 and 3 ranks with the single-domain solver: 2D Orszag-Tang
with resistivity, aluminium with strength and spall, and a 3D field box agree to 1e-15 to 2e-14 with identical
total energy. Not decomposed yet: r-z geometry, conduction, viscosity and FLD (they raise). Only the MPI path has
been run here (one GPU, Windows, Intel MPI from `pip install impi-rt`); the NCCL path is written but untested,
and the 512^3 scale is a memory estimate (about 1 GB per field copy), not a measured run.

### CAD geometry on the grid (volume fractions, immersed walls)

`aegis/cad_importer.py` puts CAD solids straight onto the solver grid, with no mesh generation:
- **Formats:** `.stl` (ASCII or binary) is read natively. `.step` / `.stp` / `.iges` / `.igs` are triangulated
  through gmsh's OpenCASCADE kernel (`pip install aegis-mhd[cad]`).
- **Volume fractions and apertures:** by ray columns. Each ray along a grid axis crosses the surface at exactly
  computed points, and sorted crossings pair into inside intervals (parity, so orientation does not matter):
  - this is exact along the ray and a sub x sub midpoint rule across it;
  - a 32^3 grid reproduces a sphere's volume to 6e-4 / 1e-4 / 7e-5 / 2.5e-5 with 1 / 2 / 4 / 8 rays per cell
    side;
  - face apertures come from the same crossings, and wall area vectors follow by the discrete divergence theorem
    (they close to round-off).
- **Materials:** the solid fraction becomes an initial state (liner density, tracer, velocity).
  `examples/cad_liner_implosion.py` loads `examples/data/corrugated_liner.stl` (an m = 4 x k = 1 corrugated shell)
  onto a 64^2 x 32 grid in 0.1 s and runs the 3D MHD implosion. An optional central rod is a rigid immersed wall
  (`results/cad_liner/`).
- **Immersed rigid walls (`ImmersedWalls`, 2D and 3D Cartesian):** every HLLD face flux becomes
  a_f F + (1 - a_f) F_wall:
  - F_wall carries the pressure + Maxwell stress row of the fluid-side state and no mass, energy or induction
    flux;
  - cells with less than half fluid are solid (faces closed, no update);
  - cell volumes stay whole (a porosity-type immersed boundary, no small-cell time-step limit). The wall's normal
    and blockage are sub-cell; its position along an axis is resolved to a cell.
  - Verified: uniform states at rest stay at rest, mass and energy are conserved to round-off, div B at round-off,
    and a Mach 2 shock reflecting off walls inclined at 45 and 30 degrees reaches the exact reflected pressure
    within 0.4% at 256^2.

### Live viewer and automatic run monitor

`aegis/agent_monitor.py` watches a run and logs alerts with suggested design changes. It is rule-based
(statistics and physics thresholds), not a learned or language model:
- **Interface modes:** the inner liner radius R(theta) on chosen slices is Fourier analysed. Each mode's growth
  rate is a least-squares fit of log|a_m| over a sliding window, using only amplitudes above 0.5% of the radius
  and fits with R^2 >= 0.7. A mode growing faster than `growth_alert` or past `amp_alert` is reported; modes not
  declared as seeded are flagged as unexpected (e.g. an m = 3 wall mode from a drive asymmetry).
- **Magnetic insulation:** axial flux through the fuel (loss beyond `flux_loss_alert`) and the fuel fraction with
  Hall parameter below 1.
- **Numerical health:** energy drift in closed domains, and max |div B| dx / |B|.
- **Log:** alerts go to a JSON-lines file, once per kind / mode until they clear.

`scripts/server.py` runs a simulation in a background thread and streams it to a browser (standard-library HTTP,
Server-Sent Events, no web framework):
- **Each frame:** density and temperature on a slice, temperature contours and in-plane field lines traced on the
  server, and the monitor's mode table and alerts.
- **Rendering:** the page draws the slice as a colour-mapped texture with WebGPU (WGSL shader), or a 2D canvas when
  the browser has no WebGPU.
- **Presets:** `liner2d`, a liner cross-section with a seeded m = 4 corrugation and a hidden 3% m = 3 drive
  asymmetry; `cad3d`, the CAD liner of `examples/cad_liner_implosion.py`.

    python scripts/server.py                  # then open http://127.0.0.1:8765
    python scripts/server.py --preset cad3d

### 3D rendering

`aegis.io3d.Recorder` writes 3D snapshots (density, pressure, temperature, liner fraction, v, B, E_r) as VTK
ImageData with a ParaView `.pvd` series and optionally HDF5. `scripts/render_3d.py` renders them off screen with
PyVista: a ray-cast density volume, the liner isosurface with the quadrant facing the camera cut away, hot-spot
temperature contours and field lines, one PNG per frame and an MP4. `examples/mtif_3d_asymmetry.py --vtk-case
"tilt 2 deg"` records one case and renders it at the end of the run.

```
mpiexec -n 4 python scripts/mpi_check.py                       # decomposition check
python examples/mtif_3d_asymmetry.py --tilts 2 --modes --vtk-case "tilt 2 deg"
python scripts/render_3d.py results/mtif_3d_asymmetry/vtk_tilt_2_deg
python examples/experimental_shot_fitting.py                   # synthetic shot -> calibration report
python examples/amr_implosion_demo.py                          # uniform coarse / fine vs AMR
```

## Performance

The hot paths are fused CUDA kernels (conversions, MUSCL, the CT corner EMF; HLLD reads any axis without
transposing), the right-hand side has no host synchronisation, and in 2D with CT each SSP-RK3 step is captured
once as a CUDA graph and replayed (`use_cuda_graphs`), bit-identical to the uncaptured step. On a Quadro P1000
under Windows a 128 x 128 ideal MHD step went from about 140 ms to 22 ms, a 16 x 4 step from 140 ms to 9 ms.

## Verification

`python main.py --test all` runs the full suite and exits non-zero on failure:

| test | checks |
|---|---|
| brio_wu | Brio-Wu shock tube structure |
| alfven | Alfvén wave convergence order (measured 1.96) |
| reflection | residual of fast and Alfvén pulses after leaving through the outflow boundary (3.5e-5, 7e-10) |
| conservation | energy budget closes to round-off (~1e-14); with CT div B stays at 1e-15 for every field (GLM: 1.3e-2) |
| ct | field-loop advection one period: CT div B 7.5e-15 (GLM 4e-3), 78% of the magnetic energy kept at 128x64; a 3D field from a vector potential in a 3D flow keeps div B at 2.5e-15 |
| cylindrical | r-z: a static magnetized state stays static to 2e-15; cylindrical Noh problem: post-shock density 15.8 (exact 16), shock at r = 0.1975 (0.2), pre-shock density exact; energy budget 5e-14 |
| equilibrium | upstream gas stays at rest for pressure-balanced x stripes to truncation error, second order (1.1e-1, 2.6e-2 c_s at nx = 100, 200); misspelled config keys are rejected |
| z_invariance | z-invariant 3D runs reproduce the 2D solver: ideal, constant eta, and eta varying 8x across the interface |
| alfven_z | Alfven wave along z converges (orders 1.3, 1.9 at nz = 16-64) |
| resistive_z | weak B_x(z) decays as exp(-eta k^2 t) (error 3e-4); Ohmic heating conserves energy |
| resistive_layers | two-layer steady diffusion, eta jumps 4x: piecewise-linear B_z recovered to 3e-4 with harmonic face averaging (arithmetic: 1.7e-2) |
| ohmic | periodic box, eta varying 10x: thermal energy gained matches the integral of eta J^2 to 0.14%, total energy to 2e-14 |
| eos_impact | 10 km/s Al impact: stiffened Al on stiffened Cu vs the exact impedance match, and Mie-Gruneisen Al on Al vs the Us-Up Hugoniot; shock speeds, pressure, densities and velocity within 0.5% |
| eos_interface | Al slab in gas at uniform p and u: volume-fraction tracer keeps p uniform to 1e-9 (round-off times p_inf/p); mass-fraction blending, for comparison, is off by 3e5 p within 0.002 us |
| wall_conduction | fuel slab cooling into a cold liner wall: mean temperature follows the exact series to 0.3%, 10x slower with the field parallel to the wall (k_par/k_perp = 10); with an oblique field (80 deg, k_par/k_perp = 1e4) no liner cell gives heat back to the fuel (without the wall's second-law limit one did) |
| transport | Braginskii fit limits; Nernst: field bumps in B_y (faces) and B_w (cells) travel down a temperature gradient along the exact path to 0.06 cells; Ettingshausen conserves energy to 2e-14 |
| twotemp | theta_e - theta_i relaxes as the exact ODE to 0.3% at fixed energy (1.5e-16); Sod: electron entropy behind the shock equals the upstream value (electrons compressed adiabatically), T_i/T_e >= 1.13 there; bremsstrahlung cools only the electrons |
| viscosity | sound-wave damping from Braginskii parallel viscosity: 4/3 eta k^2/(2 rho) for B along k, 1/3 for B across k, 4/3 (Navier-Stokes) for B = 0, all within 3%; energy 6e-14 |
| strength | stiffened Al, G = 26 GPa, Y = 0.29 GPa: elastic pulse at c_L to 0.2% (not bulk c0); 0.5 km/s impact: elastic precursor at the HEL to 0.9% moving at c_L, plastic wave on the yield surface sqrt(3 J2) = Y and plateau S_xx = -2Y/3 to 0.01%; a bar pulled apart spalls at the centre only; energy budget 9e-15 |
| conduction | hot spot in a field with k_par/k_perp = 1e4: grid-aligned, the spread along and across B equals 2 D t to 0.04% and 1%; at 45 deg the numerical cross-field leakage is 2.7% (64^2) and 0.7% (128^2) of the parallel spread, second order; no new extrema, energy conserved to round-off |
| parity3d | 3D slabs reproduce the 2D solver to round-off for a stiffened Al/Cu impact (5e-14), bremsstrahlung decay (2e-15) and conduction in an oblique field (7e-15) |
| radiation | uniform hydrogen plasma, T0 = 500 eV, n = 1e20 cm^-3: T(t) matches (sqrt(T0) - K t/2)^2 to 1e-14 down to the 10 eV cutoff; E + E_rad conserved to 1e-15; no cooling in liner material or below the cutoff |
| eos_positivity | stiffened Al pulled apart at 5 and 8 km/s (cavitation) and a gas/Al interface: p + p_inf > 0, c^2 > 0, finite state |
| fcg | flux compression generator on a 64 x 4 r-z grid (Al liner, Mie-Gruneisen, strength and spall, CT, resistivity): axis field passes 20 B0 with >80% of the flux left in the cavity, no spall, div B at round-off, energy budget below 1e-12, output files written |
| circuit | RLC discharge vs the exact damped sine (3e-7); an Al liner driven from rest by a 200 uF, 30 kV bank: bank + grid energy to round-off, outer radius and current vs an incompressible thick shell on the same circuit (1.3% of the distance moved, 0.1%) |
| solid_nernst_3d | 3D vs 2D: plastic flow to 1e-13, Nernst + Ettingshausen to 7e-12 (1240 steps), identical spalled regions; a shear wave along z (S_xz) at k sqrt(G/rho) to 0.3%; a field bump on the exact Nernst path along z |
| mpi | 2 and 3 MPI ranks vs one domain: 2D Orszag-Tang with resistivity, Al with strength and spall, 3D field box, to 1e-15..2e-14 with identical total energy and div B at round-off (skipped without mpi4py/mpiexec) |
| fld | equilibration vs the exact ODE (0.7%, energy 5e-15); optically thick Gaussian spreading as 2 D t per direction in 2D, r-z and 3D (0.02%); optically thin front at 0.99 c t; advection with the flow, gas + radiation energy to round-off |
| render3d | 3D snapshots to .vti/.pvd and .h5 and back exactly; off-screen PyVista renders and an MP4 (skipped without pyvista) |
| tabular | SESAME ASCII 301/502/505 write and read back (1e-15); an ideal-gas table reproduces the ideal-gas Sod tube to 1e-13; a Mie-Gruneisen Al table reproduces the analytic 2 km/s impact (shocked p 17.998 vs 17.998 GPa, fields to 2e-4) and p(e) stays continuous into the fallback; the device EOS matches the array EOS on 160k random states (blend band, mixed, off-table) to 3e-10 and over the impact run to 1e-14; FLD with tabulated opacities matches the power laws to 2e-11 |
| validation | analytic signals with known noise: chi^2_nu of the true model 1.001, RMS 1.000 sigma; a 37 ns delay recovered by the cross-correlation offset and the fitted shift to 0.5 ns; a 1-parameter scan recovers the amplitude with the Fisher sigma; CSV round trip |
| amr | ratio-4 patches: periodic MHD box with regridding and three touching patches (edge and corner) conserve mass, momentum and energy to 2e-15 with div B at 5e-15 and shared faces equal; a T-junction of three patches keeps div B at 6e-15 (each shared corner takes the lowest-index patch's EMF once; corrected twice, as before, div B grew to 7e-3); CUDA-graph replay equals the eager run exactly; a gas / stiffened-gas slab crosses fixed coarse-fine boundaries with p and u uniform to 1e-13; a magnetized blast wave has 0.43x the coarse-grid density error with 21% of the fine cells |
| cylindrical3d | 3D (r, theta, z) with d/dtheta = 0 vs the 2D r-z solver (swirl, B_phi, B_z, z-dependent flow, the axis): 1.6e-14 after 30 steps; an m = 1 blob carried onto the axis along a uniform field: centroid error 3.4e-2 -> 1.4e-2 and field error 2.7e-2 -> 1.2e-2 from 16 x 24 to 32 x 48 cells, energy 4e-14, div B 2e-14 |
| multigroup | one group = grey FLD (1e-8 at the solver tolerance); 8 grey groups sum to grey FLD in the thick limit (within the per-group limiter), energy to round-off; 12 Kramers groups with an edge relax to the Planck spectrum; decoupled groups diffuse with D_g = c / (3 rho kR_g) over three decades of D (peak within 2% of the exact Gaussian); 2D solver with an Al group table: gas + radiation energy to round-off |
| experimental | Cu and Ni Hugoniot data load with verified checksums, a tampered file is refused; shot-trace CSV round trip; Cu plate impacts at 5 particle velocities follow the table's own Hugoniot to 0.3% (the flow solver adds no bias) and 143 Marsh shots (Up <= 3 km/s) to 1.8% RMS, chi^2_nu 0.82 for an assumed 2% error |
| autodiff | the differentiable twin matches MHDSolver after 20 steps (Orszag-Tang 2e-15, r-z shell with B_phi and B_z 4e-13); twin energy 1e-15 and div B 1e-14; the adjoint carries the gradient of the conserved energy back unchanged (2e-16); adjoint vs central finite differences in r-z for an initial-shape parameter, an amplitude and a boundary-drive current (6e-10, 4e-10, 3e-8) |
| hybrid_pic | free streaming of a density mode in a collisionless gas follows exp(-k^2 theta t^2/2) (0.816 / 0.444 / 0.163 vs 0.821 / 0.454 / 0.169; fluid would ring as a sound wave), totals 6e-15; Boris: \|v\| to 1e-15, r_L Omega / v = 1.001; Orszag-Tang with a kinetic low-density channel: 13% of cells kinetic, cells switching both ways, mass 1e-14, momentum 6e-14 (absolute), energy 1e-14, div B 4e-15 |
| fno | spectral layer with identity weights = FFT low-pass (2e-15); network gradients vs central differences (2e-6); guard with an untrained network: forced acceptance keeps gas + radiation energy at round-off, forced fallback reproduces the exact solve with the same iteration count |
| cad | STL round trip; sphere volume fractions converge with the ray sampling (6e-4 to 2.5e-5), wall area vectors close; STEP cylinder through gmsh / OpenCASCADE reads back its volume; ring wall in 2D: rest exact with B_z, a sealed blast keeps cavity mass, total mass and energy to ~1e-14; Mach 2 shock reflection off walls inclined at 45 and 30 degrees within 0.4% of the exact pressure; 3D sphere wall: rest to 2e-16, mass and energy 1.6e-15, div B 2e-15 |
| monitor | synthetic interface: m = 3 growth rate recovered exactly, flux loss flagged; 2D liner implosion: an undeclared 3% m = 3 drive asymmetry is reported as an unexpected mode (growth 5.3 per unit time), none without it; the server serves the WebGPU page, frames (slice, contours, monitor data), the SSE stream and its status |
| gmg | multigrid vs Jacobi preconditioning, three stiff FLD steps: grey 64^2 1962 -> 70, 128^2 3960 -> 71 iterations (flat with N), 6 Kramers groups 2401 -> 85; solutions within 3e-9, energy at round-off |
| cyl_filter | the azimuthal filter keeps m <= m_crit exactly and leaves outer rings alone; the blob crossing the axis (32 x 64): 1254 -> 79 steps, centroid and field error unchanged to 1e-4, energy 3e-15, div B 2e-15 |
| amr_transport | Hall + anisotropic resistivity, Nernst + Ettingshausen, two-temperature (electron and ion conduction, equilibration), gyroviscosity, viscosity and thin radiation each with three touching patches moving every 4 steps: conservation 6e-16 to 8e-16 (3e-7 to 1e-3 with the registers off), div B 2e-15; a feature in a patch matches a uniform fine run to 1e-2 to 1e-5 of the coarse error; the electron register keeps refluxed electron heat out of the ions (T_i change within 8e-4 of the fine run, 7e-3 without it) |
| hall | circularly polarized waves along B_0 with d_i k = 1: whistler and ion-cyclotron frequencies k (sqrt(1 + (k d_i/2)^2) -+ k d_i/2) to 8e-5 and 1.5e-3 at 64 cells (second order); a B_z ripple decays at eta_par k^2 (0.3945 vs 0.3948) with its current along B and at 1.96 eta_par k^2 (0.7685 vs 0.7738) across it; Hall + anisotropic resistivity in a sheared periodic box: energy 4e-15, div B 2e-15 |
| ion_transport | a fast wave across B with gyroviscosity (eta_3 k^2 / rho = k c_f) follows omega^2 = k^2 c_f^2 + (eta_3 k^2 / rho)^2 to 2e-6 (and k c_f to 6.5e-4 without it); a T_i ripple decays at D k^2 to 0.1% and at D k^2 / (1 + x_i^2) across a field with x_i = 3, electron entropy untouched, energy 4e-16; gyroviscosity + ion conduction in a two-temperature box: energy 2e-15 |
| sesame | generated Al, Cu, D-T, Au, Pb tables load into the tabular EOS; Al SESAME ASCII round trip 9e-15; p(rho0, 300 K) = 0 and fitted ambient sound speeds; principal Hugoniot within 2.7% of measured Us-up fits (not fitted) at up = 1-3 km/s; fully ionized p / ((1 + Z) rho k T / m) = 0.9997 (D-T), 1.0016 (Al), kappa_R at the Thomson value; thermodynamic consistency 1.4e-5; tabular p(rho, e) vs the model: median 1e-3, largest 8e-3 (compressed and hot branches) |
| amr_physics | resistive, conduction, FLD and strength each with three touching patches moving every 4 steps: conservation 5e-16 to 8e-16 (1e-7 to 3e-4 with the new registers off), div B at round-off; a feature inside one patch matches a uniform fine run to 1e-3 to 1e-4 of the coarse run's L1 error |

## Reproducing the study

```
cd experiments
python sweeps.py C            # one sweep, results in data/exp_C.npz
python sweeps.py all          # sweeps A-H, about 2.7 h
python interface_spectrum.py  # RMI mode vs stripe mode
python enstrophy_location.py  # mixing-layer vs window enstrophy
python initial_conditions.py  # t = 0 fields (Fig. 1, 5)
python equilibrium_convergence.py  # upstream residual vs resolution (Fig. 2a)
python 3d_tearing_sweep.py   # 3D proof of concept, ~20 min (--quick: 2 min)
python 3d_tearing_sweep2.py  # resistive 3D, seed scan + resolution check, ~50 min
python ../paper/make_figures.py
python ../paper/make_paper.py
```

| sweep | question |
|---|---|
| A | small ripple, long domain: field-free-gap stripes vs hydro |
| B | uniform-temperature closure and a uniform-field control |
| C | gap-fraction sweep at B0 = 4 and 6 |
| D | weakest locking uniform field; weak-modulation stripes |
| E | stripe/ripple wavenumber ratio and phase at B0 = 6 |
| F | phase-aligned stripes at 20% and 30% modulation |
| G | resolution check (ny = 100 vs 200) |
| H | figure set: spectrum, enstrophy location, energy budget |

## Multi-physics MTIF capabilities

AEGIS-MHD includes physics modules for magnetized target inertial fusion (MTIF) and magnetically driven
shock compression, each verified against an analytic solution in `python main.py --test all`:

- **Shock capturing and exact div B:** HLLD Riemann solver, MUSCL reconstruction, constrained transport: the
  face divergence stays at round-off (1e-15) in 2D, 3D and r-z.
- **Axisymmetric (r, z) geometry:** for MagLIF-style liners without Cartesian grid imprint; verified on the
  cylindrical Noh problem.
- **Variable resistivity:** constant, material-blended (liner/fuel) or Spitzer (T^-3/2 with a temperature floor
  and cap). The resistive terms are face fluxes, so energy is conserved exactly; harmonic face averaging gives
  the correct series resistance across a conductivity jump (two-layer steady diffusion recovered to 3e-4).
- **Two-material EOS:** ideal gas, stiffened gas and Mie-Gruneisen (linear Us-Up) for metallic liners. The
  interface uses a volume fraction advected non-conservatively (Shyue 1998), which keeps a gas/aluminium
  interface at uniform pressure to 1e-9. A 10 km/s aluminium impact matches the exact jump conditions to
  within 0.5%.
- **Optically thin bremsstrahlung:** an operator-split sink using the exact solution at fixed density, so it
  is stable at any time step. In the decay test T(t) matches the analytic solution to 1e-14 and E + E_rad is
  conserved to 7e-16.
- **Two-temperature plasma, Braginskii viscosity, Nernst and Ettingshausen transport.**
- **Liner strength:** von Mises plasticity with a Hugoniot elastic limit, and spall.
- **Flux compression generators:** Al or Cu liners crushing a seed field in r-z, with b-dot probe signals,
  flux retention and liner stress diagnostics (`examples/fcg_flux_compression.py`, below).
- **Magnetized thermal conduction:** anisotropic, k_par ~ T^5/2 along B and Braginskii k_perp across it, with limited transverse gradients and RKL2 super-time-stepping. A hot spot with
  k_par/k_perp = 1e4 spreads along B as 2 D t and stays confined across it; for an oblique field the numerical
  cross-field leakage falls at second order with resolution.

Resistivity, EOS, radiation, conduction, two-temperature and viscosity run in 2D and 3D; z-invariant 3D slabs
reproduce the 2D solver to round-off. Nernst and strength now also run in 3D; r-z geometry is 2D. The example
parameters are
illustrative, not taken from any experiment.

### Examples

```bash
# liner-driven flux compression: stiffened-gas aluminium liner with finite conductivity
python examples/mtif_flux_compression.py --eos-liner stiffened --sigma-liner 3.5e7

# the same at hot, dense core scales (~100 eV), where bremsstrahlung matters
python examples/mtif_flux_compression.py --enable-rad --v-ref-kms 100 --rho-ref 100

# synthetic b-dot probe signals for an MTIF-3-style implosion, Mie-Gruneisen aluminium liner
python examples/nearstar_mtif3_validation.py --eos-liner mie_gruneisen --sigma-liner 3.5e7

# magnetized conduction on a 0.5 mm target at ~100 eV (--braginskii: physical omega_ce tau_e, the
# default; --kappa-ratio R: fix k_par/k_perp = R in the initial fuel instead; --kappa-ratio 1: unmagnetized)
python examples/mtif_flux_compression.py --enable-conduction --v-ref-kms 100 --r-ref-mm 0.5

# 3D target asymmetries: axis tilt and azimuthal liner modes against 2D and 3D references
python examples/mtif_3d_asymmetry.py --tilts 0.5 1 2 --modes 2 4

# MagLIF-style implosion in axisymmetric (r, z) with an axial liner ripple (magneto-Rayleigh-Taylor seed);
# --enable-conduction --nernst --v-ref-kms 100 --r-ref-mm 0.5 adds magnetized transport on a hot small target
python examples/maglif_rz.py

# flux compression generator: a metal liner coasting at v0 squeezes a seed field in r-z (see below)
python examples/fcg_flux_compression.py --liner-material al --b0-tesla 5 --v0-kms 3 --enable-strength
python examples/fcg_flux_compression.py --liner-material cu --enable-strength --out results/fcg_compression_cu
```

With conduction on, the summaries report the heat conducted out of the hot spot (core gas above half the
peak core temperature) as a percentage of its peak thermal energy, and the field's suppression of conduction
there: the sum of grad T . K . grad T over the hot spot with the field off divided by the sum with it on, for
the state at turnaround. In 2D the field is axial, so all in-plane conduction is across B; the fuel conducts
into the liner as an isothermal wall. On the 0.5 mm, 100 km/s target (512^2) the hot spot loses 7.9% of its
peak thermal energy with Braginskii magnetization, and at turnaround the core holds 39.8 units of thermal
energy against 21.9 unmagnetized (`--kappa-ratio 1`): the field keeps 1.8x more heat in the fuel. Without a
field the hot-spot loss is not a useful number: the core is nearly isothermal, so heat crosses the moving
half-maximum surface back and forth and the integral (130x the hot spot's peak energy, more than the whole
system's thermal energy) counts that gross flow, not a net loss; compare core thermal energies instead. At
the examples' default scales (fuel near 0.01 eV) conduction is negligible.

### 3D multi-physics parity and magnetized conduction

The two-material EOS with volume-fraction interfaces, bremsstrahlung and anisotropic conduction all run in
the 3D solver, sharing their kernels with 2D; a z-invariant 3D slab reproduces the 2D solver to round-off
(`--test parity3d`). Conduction follows k_par ~ T^5/2 along B and k_perp = k_par/(1 + (omega_ce tau_e)^2)
across it, with Sharma & Hammett (2007) limited transverse gradients so that strongly anisotropic heat flow
creates no new temperature extrema, and RKL2 super-time-stepping so that it never limits the hydrodynamic time step.

### 3D target asymmetries

`examples/mtif_3d_asymmetry.py` implodes the liner in 3D (64^2 x 32, z periodic) with a uniform axial seed
field, so div B = 0 for any target shape. A tilt is an axis displacement dx(z) = A sin(2 pi z/Lz) whose
largest slope equals the tilt angle (a uniform tilt cannot be periodic); the modes are liner radii
R (1 + 0.02 cos(m phi)). Conduction on, Braginskii magnetization, 0.5 mm / 100 km/s scales:

| case | convergence ratio | peak mean core B / B0 | peak-B spread in z | field tilt (rms B_perp/B_z) | hot-spot conduction loss | field suppression at turnaround |
|---|---|---|---|---|---|---|
| 2D baseline | 5.01 | 21.52 | - | - | 3.64% | 5.2e4 |
| 3D reference | 5.01 | 21.52 | 0 | 0 | 3.64% | 5.2e4 |
| tilt 0.5 deg | 5.01 | 21.51 | 5.5e-4 | 4.3e-3 | 3.91% | 4.2e4 |
| tilt 1 deg | 5.01 | 21.51 | 8.9e-4 | 8.0e-3 | 4.12% | 2.9e4 |
| tilt 2 deg | 5.01 | 21.50 | 1.0e-3 | 1.6e-2 | 4.86% | 1.4e4 |
| m = 2 | 4.73 | 20.96 | 0 | 0 | 4.64% | 1.1e5 |
| m = 4 | 4.48 | 20.39 | 0 | 0 | 4.70% | 5.5e4 |

The unperturbed 3D run matches the 2D baseline exactly. Up to 2 deg, tilt barely changes the compression
(peak field -0.1% at 2 deg) but bends the core field lines in proportion to the tilt, which opens a parallel
conduction path: the field's suppression of conduction in the hot spot falls 3.8x between 0 and 2 deg and the
hot spot's conductive loss rises from 3.6% to 4.9% of its peak thermal energy. The 2% azimuthal modes cost
more peak field (-2.6% for m = 2, -5.3% for m = 4) and raise the loss to 4.6-4.7% through the larger hot-spot
surface, while the field lines stay axial. The axis displacement is at most 0.18 cells at this resolution, so
it is invisible in density slices and shows only in these integrated metrics; energy closes to 3e-14 or
better in every run. The loss counts only cells the scheme conducts as fuel (liner fraction below
`cond_tracer_max`); cells above it are the isothermal wall's heat sink.

## Flux compression generator (FCG)

`examples/fcg_flux_compression.py` models a cylindrical magnetic flux compression generator in axisymmetric
(r, z): a solid aluminium or copper liner, already launched inward at v0 by an explosive or pulsed-power
drive, crushes an axial seed field B0 trapped in its cavity. Constrained transport keeps div B at round-off;
the liner has a Mie-Gruneisen EOS (linear Us-Up Hugoniot), optional von Mises strength and spall
(`--enable-strength`), and its cold metal conductivity (Al 3.5e7, Cu 5.8e7 S/m), with Spitzer resistivity
(capped at `--sigma-min`) in the cavity gas and Ohmic heating everywhere. Each run writes `diagnostics.csv`
(time, on-axis and peak B, the b-dot voltage of a 1 mm^2 loop, flux in the cavity, magnetic pressure over
yield P_B/Y, von Mises stress, spall), `summary.json`, `fcg_signals.png` and `fcg_implosion.gif` (density and
von Mises stress with field lines).

```
python examples/fcg_flux_compression.py --liner-material al --b0-tesla 5 --v0-kms 3 --enable-strength
python examples/fcg_flux_compression.py --liner-material cu --enable-strength --out results/fcg_compression_cu
python examples/fcg_flux_compression.py --quick --check      # 128 x 16 smoke run
```

Liner R0 = 30 mm, 3 mm thick, 20 mm axial period with a 0.5% mode-2 ripple, v0 = 3 km/s at mid-wall, B0 = 5 T,
256 x 128 cells (0.16 mm):

| | Al (6061) | Cu (OFHC) |
|---|---|---|
| turnaround | 8.42 us | 8.46 us |
| peak on-axis field | 506 T (101 B0) | 1079 T (216 B0) |
| field-radius convergence | 10.1 | 14.6 |
| flux left in the cavity at turnaround | 93.6% (ideal MHD: 99.1%) | 93.3% |
| peak b-dot | 1.33e3 T/us (1.3 kV per mm^2 turn) | 1.34e3 T/us |
| peak P_B/Y | 358 | 3926 |
| spalled liner mass | 0 | 0 |
| div B, energy error | 4e-14, 7.9e-13 | 4e-14, 9.2e-13 |

![FCG signals, aluminium liner](results/fcg_compression/fcg_signals.png)

The liner yields within the first 0.1 us and stays plastic (P_B/Y in the hundreds to thousands), so near
peak field it behaves as a fluid and strength matters only early on. The copper liner carries 3.3x the
aluminium's kinetic energy and reaches twice the field. About 5.5% of the cavity flux soaks into the
aluminium liner by resistive diffusion; the ideal-MHD run loses 0.9%, which is the numerical diffusion at
this resolution. The early spikes in the on-axis field are fast magnetosonic waves from the liner focusing
on the axis (the cavity rings); by peak compression the cavity field is uniform. Energy closes to 1e-12 over
13 000 to 15 000 steps: the error grows linearly at about 6e-17 per step, so it is accumulated round-off.

**Pulsed-power drive.** With `--capacitance-uf 200 --voltage-kv 80 --inductance-nh 15` (640 kJ) an aluminium liner
at rest (R 10-11 mm, 10 mm long, 5 T seed, strength on, 192 x 32 cells of 0.073 mm) is driven by the circuit
instead: the current peaks at 8.0 MA at 2.5 us with 109 GPa of drive pressure, the liner takes 7.2% of the bank
as kinetic energy (21% of the bank is lost in the 1 mOhm series resistance), and the axis field reaches 2020 T
at 3.44 us. Bank + grid energy closes to 1e-14; the hydrodynamic energy error is 1.2e-11 after 50 440 steps
(round-off accumulation, 2.4e-16 per step), and the inductance from the work done and from the liner's radius
stay within 2.4%. At turnaround the field radius is 0.50 mm, only 7 cells, against a 9.5-cell liner surface,
so that peak field is resolution-limited (`results/fcg_rlc`).

Limits: the coasting runs prescribe the liner velocity (the circuit option above replaces it with a bank, but
there is no explosive drive and no load circuit beyond the liner); the field is purely axial (no B_theta), so there is no current-driven instability; the 0.1 kg/m^3,
0.1 bar fill stands in for an evacuated cavity, whose Alfven speed would make the time step vanish. At
turnaround the cavity spans about 19 cells while the smeared liner surface is 16 to 17 cells wide, so the field
radius (where the cavity flux sits at the on-axis field) and the tracer radius differ by about 0.6 mm; both
are in the summary. Mixed cells in that layer use the tangent stiffened-gas closure. A thick liner must start
with v ~ 1/r (incompressible): a uniform v0 makes the shell ring radially at about rho c v0 dR/R, a few GPa,
which spalled it at 0.9 us before the field mattered. The parameters are illustrative, not a specific device.

## Benchmark: high-Mach MHD Richtmyer-Meshkov instability

A Mach-5 shock hits a single-mode interface between a light gas and one ten times denser. Baroclinic
vorticity deposited at the interface drives Richtmyer-Meshkov growth, spike roll-up and mixing. A uniform
field along the shock normal (B_x) suppresses it: the vorticity is carried away from the interface along
field lines by Alfven waves (Wheatley, Pullin & Samtaney 2005), so the interface amplitude stays bounded.

| | Hydro (B = 0) | MHD (B_x = 2.5) |
| :--- | :--- | :--- |
| Shock and interface | Mach 5, ρ₂/ρ₁ = 10, A = 0.82 (0.73 after compression) | same |
| Amplitude a(t) at t = 1.5 | 0.317, still growing | 0.064 (peak 0.092 at t ≈ 0.6), 80% below hydro |
| Interface | spike roll-up (mushroom) | bounded single mode |
| max \|ω\| at t = 1.5 (peak after impact) | 17.5 (57.8) | 11.6 (55.3) |
| Energy conservation error | 2.9 × 10⁻¹² | 1.6 × 10⁻¹² |
| h max\|∇·B\|/max\|B\| | n/a | 2.4% at shock passage, ~0.2% after |

2D, 1024 × 128 cells, single realization; no resolution study yet for this benchmark. Reproduce with
`python scripts/run_rmi_mhd.py && python scripts/plot_rmi_mhd.py`.

![RMI density and field lines](results/rmi_mhd/rmi_density_fieldlines.gif)
*Log density at matching times, hydrodynamic (left) and B_x = 2.5 (right); white lines are magnetic field
lines, the cyan line is the interface (heavy fraction 0.5).*

### Running it

```
python scripts/run_rmi_mhd.py            # configs/rmi_mhd.json: hydro and B_x = 2.5, 1024 x 128, ~20 min
python scripts/run_rmi_mhd.py --quick    # 256 x 32 check, ~1 min (results/rmi_mhd_quick)
python scripts/plot_rmi_mhd.py           # PNG/GIF into results/rmi_mhd
python main.py --test rmi_mhd            # smoke test
```

The set-up is in `aegis/problems/rmi_mhd.py`; the shock travels in +x and the interface is
x(y) = x0 + eta0 cos(k y). Jumps come from the Rankine-Hugoniot conditions, and the shock-interface
interaction is solved exactly in 1D for the Richtmyer and Meyer-Blewett reference growth rates. With B_x,
the incident, reflected and transmitted shocks must all be fast MHD shocks, so B0 above ~3.4 is rejected at
these parameters. Each case writes `diagnostics.csv`, `summary.json`, `frames.npz` and VTK frames
(`vtk/frame_NNNN.vti`, `rmi.pvd` for ParaView) to `results/rmi_mhd/<case>/`. The VTK frames and
`frames.npz` (about 140 MB per case) are not tracked in git.

## Model and numerical limitations

- Ideal gas (γ = 5/3) by default; liner strength is elastic-perfectly plastic (constant G and Y, no work or rate hardening, no thermal softening) with a threshold spall model, and there is no phase change. The stiffened-gas fit to a Hugoniot is exact at one particle velocity only. Mixed cells use the isobaric mixture rule, which is exact for stiffened gases and approximate for Mie-Gruneisen. Viscosity is the Braginskii parallel part plus the gyroviscous (FLR) stress, without the dissipative perpendicular terms (eta_1, eta_2), and gyroviscosity is 2D Cartesian only; thermal conduction is electron (Braginskii k_perp fit, no Righi-Leduc term) and, in two-temperature mode, ion (1/(1 + x_i^2) cross-field form), the liner is treated as an isothermal heat sink, and at grid-oblique field angles the limited scheme adds numerical cross-field diffusion that falls at second order with resolution. Radiation is optically thin bremsstrahlung or grey flux-limited diffusion (one group, no radiation pressure, power-law opacities unless a table is supplied, single gas temperature), coupled by first-order operator splitting; in mixed cells the temperature is taken from the mixture pressure and density.
- The Hall term and anisotropic resistivity are 2D Cartesian only (not r-z or 3D), without the electron pressure gradient term of Ohm's law; the cross-field resistivity uses the strongly magnetized ratio wherever B is nonzero. The `material` model takes eta from the passive tracer, so it is only as sharp as the numerically smeared liner/fuel interface; the `spitzer` model needs its temperature floor and cap, and in the cold, dense default of the examples it sits at the cap.
- Constrained transport keeps div B at round-off; 3D cylindrical (r, theta, z) covers ideal MHD only (resistive, transport, strength and radiation are Cartesian in 3D) and the near-axis time step needs the azimuthal filter (`cyl_filter`), which removes azimuthal structure finer than about dr on the inner rings, and the Cartesian 2D MTIF examples are still cross-sections with a four-fold grid imprint (use `examples/maglif_rz.py` for an axisymmetric implosion).
- Pressure balance of the striped field holds exactly in the initial data but only to second order in the discrete scheme.
- Shock-parallel stripes are restricted to the fast-shock regime, B0 < 6.55 sqrt(p_bar) at Mach 10; switch-on shocks are not modelled.
- Most sweeps use one resolution and one realisation. The lock boundary was rechecked at 2x and 4x resolution (`experiments/convergence_lock.py`): interface roughness converges, mixing-layer enstrophy of the hydrodynamic reference does not (no physical viscosity), so enstrophy ratios against hydro are not grid-converged.
- The legacy campaign in `main.py --campaign` uses locking criteria (window enstrophy, raw gap/bar momentum) that this study found unreliable for MHD. It is kept for reference and should not be run as is.

## Citation

If you use this code or its results, please cite it; GitHub shows a "Cite this repository" button generated
from `CITATION.cff`.

## License

No license has been chosen yet. Until one is added, the code is all rights reserved: you may read it, but
not reuse, modify or redistribute it without the author's permission.
