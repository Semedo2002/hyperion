"""Build paper/magnetic_sieve.docx with python-docx.

Two-column body; title, abstract, figures and wide tables span the full width.

    python paper/make_figures.py   # first
    python paper/make_paper.py
"""
import os
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "magnetic_sieve.docx")

TITLE = ("Striped magnetic fields and the Richtmyer-Meshkov instability: "
         "a trade-off between suppression and collimation")
AUTHOR = "Abdelrahman Shaltout"
AFFIL = "Independent researcher"

MARGIN, GAP = 2.2, 0.8                 # cm
TEXT_W = 21.0 - 2 * MARGIN             # 16.6 cm
COL_W = (TEXT_W - GAP) / 2             # 7.9 cm
FIG_WIDE, FIG_SINGLE = 15.8, 11.0      # cm
BODY_PT = 10.5

# ---------------------------------------------------------------- document and styles

doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.left_margin = sec.right_margin = Cm(MARGIN)
sec.top_margin = sec.bottom_margin = Cm(2.3)

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
normal.font.size = Pt(BODY_PT)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.25

for lvl, size, before in ((1, 12.5, 18), (2, 10.5, 11)):
    st = doc.styles[f"Heading {lvl}"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.italic = lvl == 2
    st.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(5)
    st.paragraph_format.keep_with_next = True
    rfonts = st.element.rPr.find(qn("w:rFonts"))
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), "Times New Roman")
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rfonts.get(qn(attr)) is not None:
            del rfonts.attrib[qn(attr)]


def set_columns(section, n):
    sectPr = section._sectPr
    for old in sectPr.findall(qn("w:cols")):
        sectPr.remove(old)
    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(int(GAP / 2.54 * 1440)))
    sectPr.append(cols)


def new_section(ncols):
    s = doc.add_section(WD_SECTION.CONTINUOUS)
    set_columns(s, ncols)
    return s


# ---------------------------------------------------------------- text helpers

SUB = re.compile(r"(?<=[A-Za-zΔεψΨ])_([A-Za-z0-9]+)")


def _spans_bi(text):
    out, buf, i, fmt = [], "", 0, None
    while i < len(text):
        if text.startswith("**", i):
            out.append((buf, fmt)); buf = ""; fmt = None if fmt == "b" else "b"; i += 2
        elif text[i] == "*":
            out.append((buf, fmt)); buf = ""; fmt = None if fmt == "i" else "i"; i += 1
        else:
            buf += text[i]; i += 1
    out.append((buf, fmt))
    return [(c, f) for c, f in out if c]


def _spans(text):
    """**bold**, *italic*, X_sub -> subscript"""
    out = []
    for chunk, fmt in _spans_bi(text):
        pos = 0
        for m in SUB.finditer(chunk):
            out.append((chunk[pos:m.start()], fmt))
            out.append((m.group(1), (fmt or "") + "s"))
            pos = m.end()
        out.append((chunk[pos:], fmt))
    return [(c, f) for c, f in out if c]


def _runs(p, text, italic=False, bold=False, size=None):
    for chunk, fmt in _spans(text):
        fmt = fmt or ""
        r = p.add_run(chunk)
        r.italic = italic or "i" in fmt
        r.bold = bold or "b" in fmt
        r.font.subscript = "s" in fmt
        if size:
            r.font.size = Pt(size)


def para(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, italic=False, size=None, after=None, bold=False):
    p = doc.add_paragraph()
    p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    _runs(p, text, italic, bold, size)
    return p


def heading(text, level=1):
    doc.add_heading(text, level=level)


def equation(text, number):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Cm(0.4)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.tab_stops.add_tab_stop(Cm(COL_W - 0.4), alignment=2)
    _runs(p, text, italic=True)
    p.add_run(f"\t({number})")


_wide = False


def wide_begin():
    global _wide
    new_section(1)
    _wide = True


def wide_end():
    global _wide
    new_section(2)
    _wide = False


def figure(name, caption, width_cm=FIG_WIDE):
    """Full-width figure on its own band, with room above and below."""
    standalone = not _wide
    if standalone:
        wide_begin()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.add_run().add_picture(os.path.join(FIG, name), width=Cm(width_cm))
    c = para(caption, size=9, after=16)
    c.paragraph_format.line_spacing = 1.15
    c.paragraph_format.left_indent = c.paragraph_format.right_indent = Cm(1.0)
    if standalone:
        wide_end()


def _cell_border(cell, **edges):
    """edges: name -> (val, size in eighths of a point)"""
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge, (val, sz) in edges.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tcPr.append(borders)


def _cell_shade(cell, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def _cell_margins(cell, top=60, bottom=60, left=90, right=90):
    tcPr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for side, v in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(v)); el.set(qn("w:type"), "dxa")
        mar.append(el)
    tcPr.append(mar)


def table(label, caption, header, rows, widths_cm, size=9.5):
    """Complete table on a full-width band: never split across columns (own one-column
    section) or pages (rows cannot split and every row is kept with the next)."""
    standalone = not _wide
    if standalone:
        wide_begin()
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.LEFT
    c.paragraph_format.space_before = Pt(12)
    c.paragraph_format.space_after = Pt(5)
    c.paragraph_format.keep_with_next = True
    c.paragraph_format.left_indent = Cm((TEXT_W - sum(widths_cm)) / 2)
    _runs(c, f"**{label}.** ", size=9.5)
    _runs(c, caption, size=9.5)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for row in t.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    last = len(rows)
    for r_i, values in enumerate([header] + rows):
        for c_i, v in enumerate(values):
            cell = t.cell(r_i, c_i)
            cell.width = Cm(widths_cm[c_i])
            cell.text = ""
            par = cell.paragraphs[0]
            _runs(par, str(v), bold=r_i == 0, size=size)
            par.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.space_before = Pt(0)
            par.paragraph_format.space_after = Pt(0)
            par.paragraph_format.line_spacing = 1.0
            par.paragraph_format.keep_with_next = r_i < last
            _cell_margins(cell)
            edges = {"left": ("nil", 0), "right": ("nil", 0), "top": ("nil", 0), "bottom": ("nil", 0)}
            if r_i == 0:
                edges.update(top=("single", 12), bottom=("single", 6))
                _cell_shade(cell, "EFEFEF")
            if r_i == last:
                edges["bottom"] = ("single", 12)
            _cell_border(cell, **edges)
    tail = doc.add_paragraph()
    tail.paragraph_format.space_after = Pt(10)
    if standalone:
        wide_end()


# ---------------------------------------------------------------- front matter

set_columns(sec, 1)
para(TITLE, align=WD_ALIGN_PARAGRAPH.CENTER, size=16, bold=True, after=12)
para(AUTHOR, align=WD_ALIGN_PARAGRAPH.CENTER, size=11, after=2)
para(AFFIL, align=WD_ALIGN_PARAGRAPH.CENTER, size=10, italic=True, after=18)

heading("Abstract")
p = para(
    "We examine whether a spatially striped magnetic field can suppress the Richtmyer-Meshkov "
    "instability (RMI) of a shocked density interface while channelling the post-shock flow through the "
    "weak-field gaps between the stripes, a configuration we refer to as a magnetic sieve. The study uses a "
    "GPU-based finite-volume MHD solver in two and three dimensions with an HLLD Riemann solver, MUSCL reconstruction, "
    "SSP-RK3 time integration and GLM divergence cleaning. An audit of the first implementation found a "
    "non-solenoidal initial field, an upstream state that was not in pressure balance, a resistive energy "
    "source that created energy, and an energy diagnostic that confused boundary inflow with conservation "
    "error. After correction, the discrete energy budget closes to round-off, and initial states are restricted "
    "to the fast-shock regime, which for a shock-parallel field at Mach 10 requires B₀ < 6.55√p̅. A "
    "uniform shock-parallel field suppresses the instability. Striped fields leave the interface rougher than a "
    "uniform field: a grid-convergence study at three resolutions shows that the stripe-scale deformation grows "
    "linearly with the bar-gap magnetic-pressure contrast, a₈ ≈ κΔP_B with κ converging to about 0.028. "
    "A lock threshold defined by mixing-layer enstrophy relative to the hydrodynamic run is not grid-converged, "
    "because the inviscid hydrodynamic enstrophy keeps growing with resolution. The deformation comes from a "
    "steady, field-aligned shear behind the shock: the shock compresses the gas but not a shock-parallel field, "
    "and magnetic tension does not oppose shear along the field. The contrast needed to steer flow into the gaps "
    "therefore roughens the interface in proportion. When the stripe wavenumber is half the interface wavenumber, the stripe-induced "
    "displacement interferes with the RMI mode with the phase dependence predicted from the jump conditions, "
    "producing a zero crossing of the mode followed by inverted growth rather than a sustained lock. This "
    "zero crossing is the magnetic counterpart of the phase-dependent interaction between non-uniform shocks "
    "and interface perturbations known from hydrodynamic studies; we report it as an observation. "
    "Three-dimensional resistive runs at Lundquist number S ≈ 75 show no tearing or breakdown of the "
    "shear layers.",
    after=6)
p.paragraph_format.left_indent = p.paragraph_format.right_indent = Cm(1.0)
para("*Keywords:* Richtmyer-Meshkov instability; magnetohydrodynamics; magnetic suppression; "
     "HLLD; divergence cleaning; shock admissibility", size=9.5, after=10).paragraph_format.left_indent = Cm(1.0)

new_section(2)

# ---------------------------------------------------------------- 1 introduction

heading("1. Introduction")
para(
    "When a shock crosses a perturbed interface between fluids of different density, baroclinic vorticity "
    "deposited on the interface drives the growth of the perturbation: the Richtmyer-Meshkov instability "
    "[1, 2]. The instability matters in inertial confinement fusion, where it seeds mixing between capsule "
    "layers, and in astrophysical shocks, where it can amplify magnetic fields. A magnetic field can suppress "
    "it (Section 2)."
)
para(
    "This work asks what happens when the field is structured on the scale of the perturbation on purpose. "
    "The hypothesis was that a field organised in stripes, with strong-field bars separated by weak-field "
    "gaps, would act as a sieve: the bars would hold the interface in place, while the gaps would admit the "
    "post-shock flow as collimated jets. The critical field would then follow a scaling "
    "B_crit = Ψ√ρ̅ V_hydro, with V_hydro the linear Richtmyer growth rate and Ψ a geometric factor."
)
para(
    "The first implementation of the solver reported a total energy change of 329% over a run and a maximum "
    "|∇·B| of 241, figures incompatible with a reliable calculation. Tracing them to their causes "
    "changed both the numerical method and the physical set-up (Sections 3 and 4.1). With a "
    "pressure-balanced, admissible configuration and interface-local diagnostics, the study then tests the "
    "hypothesis over the field strength, the bar-gap contrast, and the wavenumber and phase of the stripes "
    "(Section 4). Section 5 relates the results to previous work and states their limits."
)

# ---------------------------------------------------------------- 2 literature review

heading("2. Literature Review")
heading("2.1 Magnetic suppression of the RMI", 2)
para(
    "In ideal MHD a magnetic field can suppress the RMI. For a field aligned with the shock normal, Samtaney "
    "[3] showed numerically that the interface remains stable. Wheatley, Pullin and Samtaney [4] solved the "
    "linearised initial-value problem and showed that the interface amplitude stays bounded, because the "
    "vorticity deposited by the shock is carried away from the interface along field lines. Sano, Inoue and "
    "Nishihara [5] derived the field strength required for suppression in plasmas and found that it depends "
    "mainly on the shock Mach number."
)
para(
    "Fields with spatial structure have been studied mainly in converging geometry. Mostert et al. [6] "
    "applied seed fields of prescribed shape to spherical implosions; the suppression varied over the "
    "interface, and an octahedrally symmetric field gave a nearly uniform suppression pattern. In these "
    "studies the field structure is a consequence of the geometry or is chosen for symmetry; it is not used "
    "to steer the flow."
)
heading("2.2 Phase-dependent interaction and freeze-out", 2)
para(
    "In hydrodynamics, perturbation growth depends on the relative phase of a non-uniform drive and the "
    "interface. Ishizaki et al. [7] showed that a flat contact surface becomes unstable behind a rippled shock, "
    "with a growth rate that depends on the phase of the shock oscillation at impact. Zou et al. [8] studied "
    "flat interfaces impacted by rippled shocks in a shock tube. Freeze-out, the vanishing of the asymptotic "
    "normal velocity of the interface, results from the interaction of the interface with rippled wave fronts "
    "[9]. Recent work designs targets so that such interactions cancel growth, by optimising the target "
    "geometry [10] or by placing sub-surface voids that split the shock into a sequence of weaker waves [11]."
)
heading("2.3 Gap addressed", 2)
para(
    "We did not find studies of a magnetic field that is striped across a planar interface, nor of how such a "
    "field behaves when it must be in pressure balance with the gas ahead of the shock. The present work "
    "examines that configuration, and Section 5.1 compares its findings with the work above."
)

# ---------------------------------------------------------------- 3 methodology

heading("3. Methodology")
heading("3.1 Equations and discretisation", 2)
para(
    "We solve the two-dimensional compressible MHD equations for an ideal gas (γ = 5/3) with a passive "
    "scalar C marking the heavy fluid and an optional constant resistivity η. Variables are cell-centred. "
    "Interface states are reconstructed with MUSCL and the van Leer limiter [12], fluxes use the HLLD solver of "
    "Miyoshi and Kusano [13] in a CUDA kernel (one thread per face), and time integration uses three-stage "
    "SSP Runge-Kutta [14]. The divergence constraint is controlled by GLM cleaning [15] with Powell source "
    "terms [16]."
)
para(
    "The GLM speed c_h is set to the largest MHD signal speed; the earlier choice of 1.5 times the sum of fast "
    "speed and |v| reduced the time step by a factor of two to three without improving the cleaning. The "
    "damping parameter was reduced from α = 1.5 to 0.4: in a test with a striped field, "
    "h max|∇·B|/max|B| was 0.050, 0.042 and 0.038 for α = 1.5, 0.4 and 0, consistent with [17]."
)
heading("3.2 Striped field, pressure balance and admissibility", 2)
para("The field is aligned with the shock normal and modulated across the interface,")
equation("B_x = B₀√[ε² + (1 − ε²)cos²(k_m y + φ)],  B_y = B_z = 0", 1)
para(
    "which is solenoidal for any profile in y. Bars (B_x = B₀) alternate with gaps (B_x = εB₀); ε = 0 "
    "gives field-free gaps and ε → 1 a uniform field. The upstream gas is in pressure balance,"
)
equation("p(y) = P_tot − B_x²/2,   P_tot = p̅ + B₀²/2", 2)
para(
    "so the bar centres keep the reference state ρ = 1, p = p̅ = 1 (Fig. 1). With uniform upstream "
    "density, a planar shock of speed v_s = M c_s(bar) is imposed and each row receives the hydrodynamic jump at "
    "its local Mach number, since the normal field is continuous across the shock."
)
figure("fig01_setup.png",
       "Figure 1. Initial condition for B₀ = 6, ε = 0.3. (a) Density at t = 0: post-shock gas to the left "
       "of the shock at x = 1.2, the rippled interface at x = 1.5, and uniform heavy gas beyond it. (b) Upstream "
       "profiles across the stripes: gas and magnetic pressure vary, their sum does not.")
para(
    "The hydrodynamic jump is an admissible fast MHD shock only if the downstream flow stays super-Alfvénic "
    "in the shock frame, u₂ = v_s/r > v_A2. At the bar centres this gives"
)
equation("B₀ < v_s√(ρ/r) = 6.55√p̅   (M = 10, r = 3.88)", 3)
para(
    "Beyond this limit the shock would be of switch-on type. With uniform density the hotter gaps must also stay "
    "supersonic, B₀ < √[2p̅(M² − 1)] = 14.1√p̅ (Fig. 2b). The solver rejects inadmissible "
    "states; only B₀/√p̅ matters, so raising the reference pressure does not extend the range. The "
    "pressure balance is exact in the initial data but only second-order in the scheme, because p and B_x are "
    "reconstructed separately (Fig. 2a)."
)
figure("fig02_equilibrium_admissibility.png",
       "Figure 2. (a) Residual flow far upstream of the shock at t = 0.05 for B₀ = 5, against resolution; the "
       "square is the same case without pressure balance. (b) Admissible field strength against Mach number: the "
       "fast-shock limit of Eq. (3) and the gap Mach-number limit. The point marks the configuration of this study.")
heading("3.3 Energy accounting", 2)
para(
    "In a domain with driven inflow, the change of total energy is not a measure of conservation. We accumulate "
    "the energy flux through the domain faces from the same HLLD fluxes that update the energy, weighted by the "
    "SSP-RK3 stage coefficients (1/6, 1/6, 2/3), and the integral of the Powell energy source. The conservation "
    "error is"
)
equation("ε_E = [E(t) − E(0) − ∫F_E dA dt − ∫S_P dV dt]/E(0)", 4)
para(
    "It stays below 10⁻¹¹ in all runs and near 10⁻¹² in the sweeps of Section 4 (Fig. 3). The "
    "resistive terms are written in conservative form: −∇×(ηJ) in the induction equation with all "
    "three current components, and the divergence of the face-centred resistive Poynting flux ηJ×B in the "
    "energy equation, so Ohmic heating is not counted twice. A parabolic limit "
    "Δt ≤ [2η(Δx⁻² + Δy⁻²)]⁻¹ applies when η > 0."
)
figure("fig03_energy_budget.png",
       "Figure 3. Energy budget for B₀ = 3.5, ε = 0.9. (a) The total energy grows to several times its "
       "initial value, and the boundary inflow accounts for all of it. (b) The conservation error of Eq. (4).")
heading("3.4 Boundary conditions", 2)
para(
    "The post-shock state is held fixed in the ghost cells at x_min. For a super-fast inflow all seven "
    "characteristics enter the domain, so this is the characteristic boundary condition, and the solver checks "
    "it at start-up. The outflow is transmissive with ψ extrapolated, and y is periodic. Small fast and "
    "Alfvén pulses passed through the outflow leave residuals of 3.5×10⁻⁵ and 6.9×10⁻¹⁰ of the "
    "incident amplitude."
)
heading("3.5 Diagnostics and lock criterion", 2)
para(
    "Enstrophy over a fixed window around the interface, the metric used initially, is unsuitable for MHD: the "
    "Alfvénic waves that remove vorticity from the interface stay inside the window for some time. A "
    "uniform field gives a window enstrophy well above the hydrodynamic value while the mixing-layer enstrophy "
    "is far below it (Fig. 4). We therefore use the enstrophy in the mixing layer, 0.05 < C < 0.95, and the "
    "interface circulation per half wavelength. The interface position in each row follows from the gradient of "
    "C; its Fourier transform in y gives the seeded-mode amplitude a₄ and the rms roughness over all modes."
)
figure("fig04_enstrophy_location.png",
       "Figure 4. Enstrophy (a) in a fixed window around the interface and (b) in the mixing layer only, for "
       "the hydrodynamic run, a uniform field and deep stripes.")
para(
    "In sweeps A-F a case was called locked if, relative to the hydrodynamic run with identical settings, the "
    "late-time mixing-layer enstrophy was below 0.2 and the interface growth below 0.5 (growth measured as "
    "a₄(t_end)/a₄(0.05) in sweeps A-D and as the roughness over all modes from sweep E). Section 4.6 shows that "
    "the enstrophy part of this criterion is not grid-converged, so the converged comparison uses the interface "
    "roughness and the stripe-scale amplitude a₈ instead."
)
heading("3.6 Configuration and verification", 2)
para(
    "The domain is 12 × 2 with 400 × 100 cells. A Mach-10 shock starts at x = 1.2; the interface at "
    "x = 1.5 separates ρ = 1 from ρ = 3 with a perturbation a₀ sin(ky), k = 4π, a₀ = 0.02 "
    "(ka₀ = 0.25). Runs end at t = 0.8 with CFL number 0.3. Table 1 lists the verification suite, which is "
    "run before each study."
)
table("Table 1", "Verification tests, run before each study.",
      ["Test", "Quantity", "Result"],
      [["Brio-Wu shock tube [18]", "wave structure", "pass"],
       ["Alfv\u00e9n wave", "convergence order", "1.96"],
       ["Outflow reflection", "residual of fast / Alfv\u00e9n pulse", "3.5\u00d710\u207b\u2075 / 6.9\u00d710\u207b\u00b9\u2070"],
       ["Energy budget", "conservation error \u03b5_E", "\u2264 1.2\u00d710\u207b\u00b9\u2074"],
       ["Divergence", "h max|\u2207\u00b7B|/max|B|", "0 to 0.025"],
       ["Upstream equilibrium", "max|v|/c_s ahead of the shock", "3.9\u00d710\u207b\u00b3"]],
      [4.6, 6.2, 4.6])

# ---------------------------------------------------------------- 4 results

heading("4. Results")
heading("4.1 Consequences of the corrected set-up", 2)
para(
    "The original stripes B = B₀cos(k_m y)ŷ have ∇·B = −B₀k_m sin(k_m y), so the initial "
    "field violated the solenoidal constraint, by more behind the shock where B_y is compressed (Fig. 5). For "
    "B₀ = 85.4 the initial max|∇·B| was 3064, larger than the final 241: GLM was reducing an error "
    "imposed by the initial and inflow conditions. The Powell source, which acts only where ∇·B ≠ 0, "
    "removed 22% of the initial energy at B₀ = 20, so earlier estimates of the critical field were influenced "
    "by forces with no physical counterpart. The reported 329% energy change was boundary inflow: a run without "
    "field showed a 263% change and 0.00% after correcting for the boundary flux."
)
figure("fig05_legacy_divergence.png",
       "Figure 5. |∇·B| at t = 0 for B₀ = 6: (a) the original B_y stripes and (b) the shock-parallel "
       "stripes used in this study.")
para(
    "A first probe with the corrected set-up and the original perturbation (ka₀ ≈ 1.9, outside the "
    "linear regime assumed by V_hydro) met the original momentum criterion at B₀ = 6 (gap-to-bar momentum flux "
    "0.763 < 0.8). The imposed inflow alone gives 0.593, because the hotter gaps receive a weaker shock; relative "
    "to it, the flow moved momentum into the gaps. Momentum ratios below are normalised by their inflow value."
)
heading("4.2 Field-free gaps and choice of closure", 2)
para(
    "With field-free gaps (ε = 0, sweep A) the late-time growth rate of a₄ falls from 0.121 "
    "(hydrodynamic) to 0.018 at B₀ = 3 and 0.016 at B₀ = 6 (Fig. 6), but the mixing-layer enstrophy at "
    "t = 0.8 rises from 8.7 to 340 at B₀ = 6. A uniform-temperature closure keeps the Mach number equal "
    "in every row but leaves the dense gaps with about five times the post-shock pressure of the bars; the "
    "amplitude then grew 93-fold (sweep B). With uniform density the strong-shock pressure barely depends on the "
    "upstream pressure, and this closure is used throughout."
)
figure("fig06_growth.png",
       "Figure 6. Seeded-mode amplitude for the hydrodynamic run and for stripes with field-free gaps (sweep A).",
       width_cm=FIG_SINGLE)
heading("4.3 Suppression and collimation", 2)
para(
    "A uniform field suppresses the instability: at the base resolution, B = 2 and B = 3 reduce the "
    "mixing-layer enstrophy to 0.03 of the hydrodynamic value, while B = 1 does not halve the growth (0.73). "
    "Figure 7 shows sweeps C and D against the bar-gap magnetic-pressure contrast ΔP_B = B₀²(1 − ε²)/2. Every "
    "striped field reduces seeded-mode growth to 0.21-0.32 of the hydrodynamic value, but the mixing-layer "
    "enstrophy rises monotonically with ΔP_B. At this resolution the enstrophy criterion is met for "
    "ΔP_B ≲ 1.2; Section 4.6 shows that this crossing moves to higher contrast on refinement, so it is not a "
    "physical threshold. The monotonic increase with contrast is robust at every resolution."
)
figure("fig07_lock_map.png",
       "Figure 7. Late-time mixing-layer enstrophy relative to the hydrodynamic run, against the bar-gap "
       "magnetic-pressure contrast, at the base resolution (sweeps C and D). Green: meets the original lock "
       "criterion. The red point at the smallest contrast is a uniform field B = 1, which meets the enstrophy "
       "criterion but not the growth criterion. The position of the crossing depends on resolution (Section 4.6).",
       width_cm=FIG_SINGLE)
para(
    "The interface spectrum shows where the excess vorticity goes (Fig. 8). At t = 0.8 the hydrodynamic run "
    "has a₄ = 0.137 and roughness 0.098; a uniform field B = 3 gives 0.038 and 0.027, and stripes with 10% "
    "field contrast (B₀ = 3.5, ε = 0.9) 0.035 and 0.029. Deep stripes (B₀ = 6, ε = 0.3) also give a₄ = 0.035, but the stripe-scale mode "
    "a₈ grows to 0.255 and the roughness to 0.200, twice the hydrodynamic value."
)
figure("fig08_spectrum.png",
       "Figure 8. Interface spectrum: (a) seeded mode a₄ and (b) stripe-scale mode a₈.")

heading("4.4 Mechanism", 2)
para(
    "The shock raises the gas pressure about 125-fold at Mach 10, while a shock-parallel field passes through "
    "unchanged. A striped field balanced upstream therefore cannot stay balanced downstream. With uniform "
    "density the gaps hold more gas pressure, receive a weaker shock (Mach 2.4 against 10 in the bars at "
    "B₀ = 6) and acquire a lower post-shock velocity, so the bars advance faster than the gaps (Fig. 9). The "
    "shear is aligned with the field, and tension resists only displacements across field lines, so nothing "
    "opposes it. Its vorticity enters the mixing layer continuously and scales with ΔP_B."
)
figure("fig09_snapshots.png",
       "Figure 9. Density (left) and vorticity (right) at t = 0.8 for the hydrodynamic run (top), stripes "
       "with 10% field contrast, B₀ = 3.5, ε = 0.9 (middle) and deep stripes B₀ = 6, ε = 0.3 (bottom). The bands in the "
       "bottom-right panel are field-aligned shear layers between bars and gaps.", width_cm=TEXT_W)
para(
    "The same contrast produces collimation. The normalised gap-to-bar momentum flux increases with field "
    "strength (1.02, 1.08 and 1.29 for B₀ = 0, 3 and 6 in the first probe; 1.12 for B₀ = 6, ε = 0.3 "
    "in sweep C) and is indistinguishable from 1 in the locked cases. Suppression and collimation are therefore "
    "mutually exclusive in this regime."
)
heading("4.5 Phase-tuned interference and zero crossing", 2)
para(
    "With faster bar flow, the stripe-induced displacement follows B_x² ∝ cos(2k_m y + 2φ). It projects "
    "onto the seeded mode sin(ky) only if k_m/k = 0.5, and then opposes RMI growth for φ = π/4 and "
    "reinforces it for φ = 3π/4. Sweep E (B₀ = 6, ε = 0.3) confirms the phase dependence, with "
    "minimum roughness at φ = π/4 and maximum at 3π/4 (Fig. 10), but all phases are 2.5-3.0 times "
    "rougher than the hydrodynamic run. Writing the final seeded amplitude as |S ± R|, the extremes "
    "(0.347 and 0.407) give a stripe component S ≈ 0.38 and a residual RMI component R ≈ 0.03."
)
figure("fig10_phase.png",
       "Figure 10. Final interface roughness against stripe phase for k_m/k = 0.5, B₀ = 6, ε = 0.3 "
       "(sweep E). Dotted lines mark the predicted phases of cancellation and reinforcement.",
       width_cm=FIG_SINGLE)
para(
    "Sweep F reduced the contrast to bring S closer to R. At 20% contrast and φ = π/4 the final roughness "
    "is 0.14 of the hydrodynamic value, against 0.66 at φ = 3π/4 and 0.27 for a uniform field, but the "
    "time history shows no lock (Fig. 11): a₄ passes through zero at t ≈ 0.55 and grows again with "
    "inverted sign; at 30% the crossing occurs at t ≈ 0.31 and the subsequent growth exceeds the "
    "hydrodynamic rate. The mixing-layer enstrophy is 0.23 and 0.21 for the two phases at 20%, so the phase "
    "changes the displacement but not the injected vorticity. The RMI is an impulse that the field saturates, "
    "whereas the shear is a steady drive; the two cancel at one instant, not for all time."
)
figure("fig11_zero_crossing.png",
       "Figure 11. Seeded-mode amplitude for phase-aligned (φ = π/4) and anti-aligned (φ = 3π/4) "
       "stripes with k_m/k = 0.5 (sweep F). The aligned cases pass through zero and then grow with inverted sign.",
       width_cm=FIG_SINGLE)
heading("4.6 Resolution and grid convergence", 2)
para(
    "Doubling the cells in y changes the hydrodynamic a₄(0.8) only from 0.137 to 0.143. "
    "Deep-stripe cases are more sensitive (Fig. 12). For k_m/k = 2 the stripe pattern has about six cells per wavelength; "
    "at twice the resolution the roughness rose from 0.62 to 1.87 times the hydrodynamic value and the "
    "mixing-layer enstrophy from about 10 to 212 times. The apparent benefit of k_m/k = 2 in sweep E was "
    "numerical smoothing. Higher resolution sharpens the shear layers and strengthens the conclusion that deep "
    "stripes do not lock."
)
figure("fig12_resolution.png",
       "Figure 12. Resolution check for k_m/k = 2, B₀ = 6, ε = 0.3 at n_y = 100 and 200. (a) Hydrodynamic "
       "a₄ and stripe roughness relative to hydrodynamic. (b) Stripe mixing-layer enstrophy relative to "
       "hydrodynamic.")
para(
    "To test the base-resolution lock boundary, the cases nearest it were rerun at two and four times the "
    "resolution (800 × 200 and 1600 × 400 cells; Fig. 13, Table 2). Under the original criterion every case "
    "locks at the finer grids, including 20% field contrast, and the enstrophy ratios roughly halve with each "
    "refinement. The cause is the reference rather than the stripes: the mixing-layer enstrophy of the "
    "hydrodynamic run grows from 9.7 to 23.9 to 61.6, because without physical viscosity the inviscid mixing "
    "layer keeps forming finer vortices, while the striped runs stay between 1.1 and 4.5 at all resolutions. "
    "An enstrophy threshold measured against the hydrodynamic run therefore cannot converge."
)
para(
    "The interface roughness of the hydrodynamic and striped runs converges. The hydrodynamic value changes from "
    "0.132 to 0.140 between the two finest grids, and the striped values by at most 4%; relative to the "
    "hydrodynamic run they settle at 0.20, 0.25 and 0.32 for increasing contrast. The uniform field is the "
    "exception: its roughness keeps decreasing (0.0269, 0.0208, 0.0178, roughly first order), so the excess "
    "roughness of the stripes over a uniform field widens with resolution, from 1.1-1.4 times at the base grid "
    "to 1.6-2.5 times at the finest. The stripe-scale mode is proportional to the magnetic-pressure "
    "contrast at every resolution, a₈ = κΔP_B, with κ = 0.0176, 0.0250 and 0.0270 at the three grids. "
    "Richardson extrapolation of κ (observed order 1.9) gives κ ≈ 0.028, and even the small residual a₈ of the "
    "nearly uniform field (ε = 0.99, ΔP_B = 0.09) follows it, with a₈/ΔP_B = 0.029. This linear law, not a threshold, is "
    "the converged statement of the trade-off: the interface deformation produced by the stripes scales with the "
    "contrast that makes them collimate."
)
figure("fig13_convergence.png",
       "Figure 13. Grid convergence at 1×, 2× and 4× the base resolution. (a) Mixing-layer enstrophy: the "
       "hydrodynamic value keeps growing, the striped values do not. (b) Interface roughness, which converges. "
       "(c) Stripe-scale amplitude a₈ against the bar-gap magnetic-pressure contrast, with the least-squares "
       "line a₈ = κΔP_B at each resolution. B₀ = 3 was not run at 4×.")
table("Table 2", "Interface roughness relative to the hydrodynamic run and stripe-scale amplitude at t = 0.8, at "
      "three resolutions. Dashes: not run at that resolution.",
      ["Field (B₀, ε)", "ΔP_B", "rough./hydro 1×", "2×", "4×", "a₈ (4×)",
       "a₈/ΔP_B (4×)"],
      [["uniform, B = 3", "0.09", "0.274", "0.158", "0.127", "0.0026", "0.029"],
       ["3, 0.9", "0.86", "0.310", "0.199", "-", "-", "-"],
       ["3.5, 0.9", "1.16", "0.298", "0.216", "0.204", "0.0315", "0.027"],
       ["4, 0.9", "1.52", "0.309", "0.253", "0.245", "0.0424", "0.028"],
       ["3.5, 0.8", "2.21", "0.386", "0.332", "0.324", "0.0586", "0.027"]],
      [3.0, 1.6, 2.4, 1.8, 1.8, 2.2, 2.7])

# ---------------------------------------------------------------- 5 discussion

heading("5. Discussion")
heading("5.1 Relation to previous work and novelty", 2)
para(
    "The suppression by a uniform shock-parallel field agrees with [3-5] and serves as a check of the "
    "solver. The phase-dependent zero crossing of Section 4.5 is not a new phenomenon: it is the magnetic "
    "counterpart of the dependence of RMI growth on the phase between a non-uniform drive and the interface "
    "[7-9], which has also been used deliberately to reduce growth [10, 11]. What differs here is the origin "
    "of the non-uniform drive, which is the stripe-induced shear rather than a rippled shock or a shaped target. "
    "We therefore report the zero crossing as an observation."
)
para(
    "We found no previous report of the mechanism of Section 4.4, in which a pressure-balanced shock-parallel "
    "field structure is converted by the shock into a field-aligned shear that magnetic tension cannot oppose, "
    "nor of the resulting conflict between suppression and collimation. Studies of structured fields [6] "
    "concern implosion symmetry rather than flow steering. This statement is based on a search of the open "
    "literature and the indexes available to us; it should be read as “not found” rather than "
    "“shown to be absent”."
)
heading("5.2 Implications for the sieve concept", 2)
para(
    "The trade-off follows from two properties that do not depend on the numerical details: a shock-parallel "
    "field is not compressed by the shock, and magnetic tension does not resist shear along the field. Any "
    "stripe pattern strong enough to redirect the post-shock flow therefore also drives the shear that deforms "
    "the interface, and the converged deformation grows in proportion to the contrast (a₈ = κΔP_B). A field "
    "that leaves the interface as smooth as a uniform field must be nearly uniform, and a field that collimates "
    "the flow cannot do so. Phase alignment can choose the time at which the seeded mode is flat but cannot hold "
    "it there."
)
heading("5.3 Limitations", 2)
para(
    "The results apply to ideal MHD with shock-parallel stripes, uniform upstream density, Mach 10, density "
    "ratio 3 and the fast-shock regime of Eq. (3); stronger fields with switch-on shocks, other closures and "
    "oblique fields were not examined. The convergence study covered the cases nearest the former boundary at "
    "three resolutions, but the uniform-field roughness was still decreasing at the finest grid, the other "
    "sweeps used one resolution, and all used one realisation; the zero-crossing times in particular are base-resolution "
    "values. Thresholds based on enstrophy would need a viscous treatment in which enstrophy converges. "
    "Divergence errors of a few percent remain at shocks because the scheme does not use constrained transport."
)
para(
    "A three-dimensional extension of the solver (verified against the two-dimensional solver to 10⁻¹⁴ for "
    "z-invariant states) was used to ask whether the field-aligned shear layers break down through secondary "
    "Kelvin-Helmholtz or tearing instabilities. With explicit resistivity at the largest Lundquist number the grid "
    "resolves (S ≈ 75), field-reversing stripes and a frame moving with the interface to t = 5, the seeded "
    "three-dimensional runs were indistinguishable from their z-invariant controls: the z-velocity decayed, the "
    "current sheets stayed z-uniform and did not tear. Tearing at plasma-like S ≥ 10³ remains untested."
)

# ---------------------------------------------------------------- 6 conclusion

heading("6. Conclusion")
para(
    "Deep magnetic stripes cannot both hold a shocked interface and channel the flow through their gaps when the "
    "field is shock-parallel and the shock is in the fast regime. Reaching this result required correcting the "
    "initial field, the upstream equilibrium, the resistive energy term and the energy diagnostic, and "
    "restricting initial states to admissible shocks. A uniform field suppresses the instability; stripes "
    "create a field-aligned shear that tension cannot oppose, and the resulting stripe-scale deformation of the "
    "interface grows linearly with the bar-gap magnetic-pressure contrast, converged over three resolutions to "
    "a₈ ≈ 0.028 ΔP_B. Suppression and collimation therefore compete. Phase-aligned stripes produce a zero crossing of the seeded mode at a time set by "
    "the contrast, the magnetic analogue of known hydrodynamic phase interference, but no sustained lock. "
    "A lock threshold measured by enstrophy relative to the hydrodynamic case is not a converged quantity in "
    "inviscid simulations and should not be used to compare magnetized and unmagnetized runs."
)

heading("Code and data availability")
para("The solver, sweeps A-H, the convergence study, the three-dimensional sweeps, saved time series and the "
     "scripts that generate the figures and this document are available at https://github.com/Semedo2002/MagSieve-3D.",
     align=WD_ALIGN_PARAGRAPH.LEFT)
heading("Acknowledgements")
para("The author thanks Professor Emeritus Kenneth Morgan (Swansea University) for sharing his work, which has "
     "led the way towards a better future for computational fluid dynamics and computational engineering, and "
     "for a mentorship that continued long beyond the author's time at Swansea University and into his "
     "retirement. All simulations were run on the author's own workstation with a single NVIDIA GPU.")

heading("References")
refs = [
    "R. D. Richtmyer, Taylor instability in shock acceleration of compressible fluids, *Commun. Pure Appl. Math.* 13, 297-319 (1960).",
    "E. E. Meshkov, Instability of the interface of two gases accelerated by a shock wave, *Fluid Dyn.* 4, 101-104 (1969).",
    "R. Samtaney, Suppression of the Richtmyer-Meshkov instability in the presence of a magnetic field, *Phys. Fluids* 15, L53-L56 (2003).",
    "V. Wheatley, D. I. Pullin and R. Samtaney, Stability of an impulsively accelerated density interface in magnetohydrodynamics, *Phys. Rev. Lett.* 95, 125002 (2005).",
    "T. Sano, T. Inoue and K. Nishihara, Critical magnetic field strength for suppression of the Richtmyer-Meshkov instability in plasmas, *Phys. Rev. Lett.* 111, 205001 (2013).",
    "W. Mostert, D. I. Pullin, V. Wheatley and R. Samtaney, Magnetohydrodynamic implosion symmetry and suppression of Richtmyer-Meshkov instability in an octahedrally symmetric field, *Phys. Rev. Fluids* 2, 013701 (2017).",
    "R. Ishizaki, K. Nishihara, H. Sakagami and Y. Ueshima, Instability of a contact surface driven by a nonuniform shock wave, *Phys. Rev. E* 53, R5592 (1996).",
    "L. Zou, J. Liu, S. Liao, X. Zheng, Z. Zhai and X. Luo, Richtmyer-Meshkov instability of a flat interface subjected to a rippled shock wave, *Phys. Rev. E* 95, 013107 (2017).",
    "J. G. Wouchuk and T. Sano, Normal velocity freeze-out of the Richtmyer-Meshkov instability when a rarefaction is reflected, *Phys. Rev. E* 91, 023005 (2015).",
    "D. M. Sterbentz, C. F. Jekel, D. White, S. Aubry and J. L. Belof, Design optimization for Richtmyer-Meshkov instability suppression at shock-compressed material interfaces, *Phys. Fluids* 34, 082109 (2022).",
    "J. Strucka, D. M. Sterbentz, B. Lukic et al., Passive freeze-out of the Richtmyer-Meshkov instability, arXiv:2602.21375 (2026).",
    "B. van Leer, Towards the ultimate conservative difference scheme. II. Monotonicity and conservation combined in a second-order scheme, *J. Comput. Phys.* 14, 361-370 (1974).",
    "T. Miyoshi and K. Kusano, A multi-state HLL approximate Riemann solver for ideal magnetohydrodynamics, *J. Comput. Phys.* 208, 315-344 (2005).",
    "S. Gottlieb and C.-W. Shu, Total variation diminishing Runge-Kutta schemes, *Math. Comp.* 67, 73-85 (1998).",
    "A. Dedner, F. Kemm, D. Kröner, C.-D. Munz, T. Schnitzer and M. Wesenberg, Hyperbolic divergence cleaning for the MHD equations, *J. Comput. Phys.* 175, 645-673 (2002).",
    "K. G. Powell, P. L. Roe, T. J. Linde, T. I. Gombosi and D. L. De Zeeuw, A solution-adaptive upwind scheme for ideal magnetohydrodynamics, *J. Comput. Phys.* 154, 284-309 (1999).",
    "A. Mignone and P. Tzeferacos, A second-order unsplit Godunov scheme for cell-centered MHD: the CTU-GLM scheme, *J. Comput. Phys.* 229, 2117-2138 (2010).",
    "M. Brio and C. C. Wu, An upwind differencing scheme for the equations of ideal magnetohydrodynamics, *J. Comput. Phys.* 75, 400-422 (1988).",
]
for i, r in enumerate(refs, 1):
    p = para(f"[{i}] {r}", align=WD_ALIGN_PARAGRAPH.LEFT, size=8.5, after=3)
    p.paragraph_format.line_spacing = 1.1
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.first_line_indent = Cm(-0.6)

# page numbers
for s in doc.sections[1:]:
    s.footer.is_linked_to_previous = True
fp = doc.sections[0].footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = fp.add_run()
for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
    if tag:
        el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag)
    else:
        el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = text
    run._r.append(el)

doc.save(OUT)
print(f"saved {OUT}")
