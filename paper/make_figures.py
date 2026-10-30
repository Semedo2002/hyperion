"""Paper figures from experiments/data (no GPU needed).

    python paper/make_figures.py

Inputs: exp_A..H.npz (sweeps), initial_conditions.npz, equilibrium_convergence.npz,
showcase_*_t0.8.png (final GIF frames of experiments/showcase.py).
Figures are drawn for full page width (two panels) or ~11 cm (single panel).
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "experiments", "data")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 250,
                     "legend.fontsize": 8.5, "axes.labelsize": 10, "axes.titlesize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9})
WIDE, SINGLE = 6.3, 4.6     # inches: two-panel and single-panel figures
INK, MUTED = "#222222", "#8a8a8a"
C1, C2, C3, C4 = "#1f5f99", "#c0392b", "#2e8b57", "#8e6cbf"
GAMMA, MACH = 5 / 3, 10.0


def load(tag):
    d = np.load(os.path.join(DATA, f"exp_{tag}.npz"))
    cases = sorted({k.split("__")[0] for k in d.files})
    return {c: {k.split("__")[1]: d[k] for k in d.files if k.startswith(c + "__")} for c in cases}


def tail(x, t, frac=1/3, t0=0.05):
    m = t > t0
    x = x[m]
    return float(np.mean(x[-max(int(len(x) * frac), 1):]))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def title(ax, s):
    ax.set_title(s, loc="left", fontsize=10)


# ------------------------------------------------------------ set-up and numerics

def fig01_setup():
    d = np.load(os.path.join(DATA, "initial_conditions.npz"))
    x, y = d["new_x"], d["new_y"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.5), gridspec_kw={"width_ratios": [1.45, 1]})
    sl = x < 4
    im = a.pcolormesh(x[sl], y, d["new_rho"][sl].T, cmap="inferno", shading="auto")
    a.set_aspect("equal"); a.set_xlabel("$x$"); a.set_ylabel("$y$")
    a.axvline(1.2, color="white", lw=0.6, ls=":")
    a.text(1.12, 0.12, "shock", ha="right", fontsize=8.5, color="white")
    a.text(1.62, 0.12, "interface", ha="left", fontsize=8.5, color="white")
    fig.colorbar(im, ax=a, shrink=0.75, label=r"$\rho$", pad=0.02)
    title(a, "(a) density at $t=0$")
    i0 = np.argmin(np.abs(x - 3.0))
    Bx, pr = d["new_Bx"][i0], d["new_p"][i0]
    b.plot(y, Bx**2 / 2, color=C1, lw=1.4, label=r"$B_x^2/2$")
    b.plot(y, pr, color=C2, lw=1.4, label="$p$")
    b.plot(y, pr + Bx**2 / 2, color=INK, lw=1.4, ls="--", label=r"$p+B_x^2/2$")
    b.set_xlabel("$y$"); b.set_ylabel("pressure"); b.set_ylim(0, 27)
    b.legend(frameon=False, ncol=3, loc="upper center", fontsize=8, columnspacing=1.0)
    title(b, "(b) upstream profiles, $x=3$")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig01_setup.png")


def fig02_equilibrium_admissibility():
    e = np.load(os.path.join(DATA, "equilibrium_convergence.npz"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.6))
    a.loglog(e["nx"], e["vmax"], "o-", color=C1, ms=5, lw=1.3, label="pressure-balanced")
    ref = e["vmax"][0] * (e["nx"][0] / e["nx"])**2
    a.loglog(e["nx"], ref, color=MUTED, ls=":", lw=1.1, label="second order")
    a.loglog([200], [float(e["unbalanced_vmax"])], "s", color=C2, ms=6, label="no balance")
    a.set_xticks([100, 200, 400]); a.set_xticklabels(["100", "200", "400"]); a.minorticks_off()
    a.set_xlabel("$n_x$"); a.set_ylabel(r"max $|v|/c_s$ upstream"); a.set_ylim(5e-4, 20); a.legend(frameon=False, loc="upper right", fontsize=8)
    title(a, "(a) discrete equilibrium")
    M = np.linspace(1.5, 20, 200)
    r = (GAMMA + 1) * M**2 / ((GAMMA - 1) * M**2 + 2)
    b_fast = M * np.sqrt(GAMMA) / np.sqrt(r)
    b_gap = np.sqrt(2 * (M**2 - 1))
    b.plot(M, b_fast, color=C1, lw=1.4, label="fast-shock limit")
    b.plot(M, b_gap, color=C2, lw=1.4, ls="--", label="gap Mach number > 1")
    b.fill_between(M, 0, np.minimum(b_fast, b_gap), color=C3, alpha=0.12, lw=0)
    b.plot([MACH], [6.55], "o", color=INK, ms=5)
    b.annotate(r"$B_0<6.55\sqrt{\bar p}$ at $M=10$", (MACH, 6.55), xytext=(12.5, 2.6),
               fontsize=8.5, arrowprops=dict(arrowstyle="-", lw=0.7, color=INK))
    b.text(3.2, 0.7, "admissible", color=C3, fontsize=9)
    b.set_xlabel("Mach number $M$"); b.set_ylabel(r"$B_{0,\max}/\sqrt{\bar p}$"); b.set_ylim(0, 25)
    b.legend(frameon=False, loc="upper left")
    title(b, "(b) admissible field strength")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig02_equilibrium_admissibility.png")


def fig03_energy():
    d = load("H")["B0=3.5 eps=0.9"]
    t, E, inflow, pw = d["t"], d["E"], d["inflow"], d["powell"]
    E0 = E[0]
    res = np.abs(E - E0 - inflow - pw) / E0
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.5))
    a.plot(t, (E - E0) / E0, color=INK, lw=1.8, label="total energy change")
    a.plot(t, inflow / E0, color=C1, lw=1.2, ls="--", label="boundary inflow")
    a.set_xlabel("$t$"); a.set_ylabel("relative to $E(0)$"); a.legend(frameon=False)
    title(a, "(a) energy change and inflow")
    b.semilogy(t[1:], np.maximum(res[1:], 1e-17), color=C2, lw=1.3)
    b.set_xlabel("$t$"); b.set_ylabel(r"$|\varepsilon_E|$"); b.set_ylim(1e-16, 1e-11)
    title(b, "(b) conservation error, Eq. (4)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig03_energy_budget.png")


def fig04_enstrophy_location():
    H = load("H")
    cases = (("hydro", INK, "hydrodynamic"), ("uniform Bx=3", C3, "uniform $B=3$"),
             ("B0=6 eps=0.3", C2, r"stripes $B_0=6$, $\epsilon=0.3$"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.6), sharey=True)
    for name, col, lab in cases:
        d = H[name]
        a.semilogy(d["t"][1:], d["en"][1:], color=col, lw=1.4, label=lab)
        b.semilogy(d["t"][1:], np.maximum(d["emix"][1:], 1e-3), color=col, lw=1.4)
    title(a, r"(a) fixed window $|x-x_c|<0.5$"); title(b, "(b) mixing layer only")
    a.set_ylabel(r"enstrophy $\int\rho\omega^2\,dA$"); a.set_xlabel("$t$"); b.set_xlabel("$t$")
    a.legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig04_enstrophy_location.png")


# ------------------------------------------------------------ results

def fig05_legacy_div():
    d = np.load(os.path.join(DATA, "initial_conditions.npz"))
    x, y = d["old_x"], d["old_y"]
    sl = x < 4
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.2), sharey=True)
    norm = LogNorm(vmin=1.0, vmax=np.abs(d["old_divB"]).max())
    for ax, key, ttl in ((axes[0], "old", "(a) legacy $B_y=B_0\\cos k_m y$"),
                         (axes[1], "new", "(b) shock-parallel $B_x(y)$")):
        val = np.abs(d[f"{key}_divB"][sl]).T
        im = ax.pcolormesh(x[sl], y, np.where(val > 0, val, np.nan), norm=norm, cmap="magma", shading="auto")
        ax.set_aspect("equal"); ax.set_xlabel("$x$")
        title(ax, ttl + f"\nmax $|\\nabla\\cdot\\mathbf{{B}}|$ = {np.abs(d[f'{key}_divB']).max():.3g}")
    axes[1].text(2.0, 1.0, r"$\nabla\cdot\mathbf{B}=0$ everywhere", ha="center", va="center", fontsize=10)
    axes[0].set_ylabel("$y$")
    fig.colorbar(im, ax=axes, shrink=0.8, label=r"$|\nabla\cdot\mathbf{B}|$", pad=0.02)
    save(fig, "fig05_legacy_divergence.png")


def fig06_growth():
    A = load("A")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))
    for name, col, lab in (("hydro", INK, "hydrodynamic"), ("Bx cos B0=3", C1, r"$B_0=3$, field-free gaps"),
                           ("Bx cos B0=6", C2, r"$B_0=6$, field-free gaps")):
        ax.plot(A[name]["t"], A[name]["a"], color=col, lw=1.6, label=lab)
    ax.set_xlabel("$t$"); ax.set_ylabel("seeded-mode amplitude $a_4$"); ax.legend(frameon=False)
    save(fig, "fig06_growth.png")


def fig07_lock_map():
    """C and D predate the roughness diagnostic: lock = emix < 0.2 hydro and
    mode-4 growth a(t_end)/a(0.05) < 0.5 hydro, the criterion they were judged by."""
    pts = []
    for tag in ("C", "D"):
        S = load(tag)
        h = S["hydro"]; en_h = tail(h["emix"], h["t"])
        growth = lambda d: d["a"][-1] / d["a"][np.argmin(np.abs(d["t"] - 0.05))]
        g_h = growth(h)
        for name, d in S.items():
            m = re.match(r"(?:uniform Bx=(?P<u>[\d.]+))|(?:B0=(?P<b>[\d.]+) eps=(?P<e>[\d.]+))", name)
            if not m:
                continue
            b, e = (float(m["u"]), 0.99) if m["u"] else (float(m["b"]), float(m["e"]))
            pts.append((0.5 * b**2 * (1 - e**2), tail(d["emix"], d["t"]) / en_h, growth(d) / g_h, name))
    pts.sort()
    fig, ax = plt.subplots(figsize=(SINGLE, 3.1))
    for dp, er, gr, name in pts:
        ax.scatter(dp, er, s=36, color=C3 if (er < 0.2 and gr < 0.5) else C2, edgecolor="white", lw=0.6, zorder=3)
    ax.axhline(0.2, color=MUTED, ls="--", lw=1.0)
    ax.text(0.012, 0.24, "lock threshold", color=MUTED, fontsize=8.5)
    ax.set_xscale("symlog", linthresh=0.1); ax.set_yscale("log")
    ax.set_xlabel(r"bar$-$gap contrast $\Delta P_B = B_0^2(1-\epsilon^2)/2$")
    ax.set_ylabel("mixing-layer enstrophy / hydro")
    ax.scatter([], [], s=36, color=C3, label="locked"); ax.scatter([], [], s=36, color=C2, label="not locked")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig07_lock_map.png")
    return pts


def fig08_spectrum():
    H = load("H")
    cases = (("hydro", INK, "hydrodynamic"), ("uniform Bx=3", C3, "uniform $B=3$"),
             ("B0=3.5 eps=0.9", C1, r"$B_0=3.5$, $\epsilon=0.9$"), ("B0=6 eps=0.3", C2, r"$B_0=6$, $\epsilon=0.3$"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.6), sharey=True)
    for name, col, lab in cases:
        d = H[name]
        a.plot(d["t"], d["m4"], color=col, lw=1.4, label=lab)
        b.plot(d["t"], d["m8"], color=col, lw=1.4)
    title(a, "(a) seeded mode $a_4$"); title(b, "(b) stripe-scale mode $a_8$")
    a.set_xlabel("$t$"); b.set_xlabel("$t$"); a.set_ylabel("amplitude")
    a.legend(frameon=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig08_spectrum.png")


def fig09_snapshots():
    """Final density and vorticity (hydro / 10% stripes / deep stripes)."""
    ims = [Image.open(os.path.join(DATA, f"showcase_{f}_t0.8.png")) for f in ("density", "vorticity")]
    crops = [im.crop((0, int(0.09 * im.height), im.width, im.height)) for im in ims]  # drop suptitle band
    gap = 40
    w = sum(c.width for c in crops) + gap
    h = max(c.height for c in crops)
    out = Image.new("RGB", (w, h), "white")
    out.paste(crops[0], (0, 0)); out.paste(crops[1], (crops[0].width + gap, 0))
    out.save(os.path.join(FIG, "fig09_snapshots.png"))


def fig10_phase():
    E = load("E")
    rms_h = E["hydro"]["rms"][-1]
    ph = [j / 8 for j in range(8)]
    rr = [E[f"r=0.5 phi={j}pi/8"]["rms"][-1] / rms_h for j in range(8)]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    ax.plot(ph, rr, "o-", color=C1, lw=1.4, ms=5)
    ax.axvline(0.25, color=C3, ls=":", lw=1.1); ax.axvline(0.75, color=C2, ls=":", lw=1.1)
    ax.text(0.27, max(rr), "predicted\ncancellation", color=C3, fontsize=8.5, va="top")
    ax.text(0.77, min(rr), "predicted\nreinforcement", color=C2, fontsize=8.5, va="bottom")
    ax.set_xlabel(r"stripe phase $\varphi/\pi$"); ax.set_ylabel("interface roughness / hydro")
    save(fig, "fig10_phase.png")


def fig11_zero_crossing():
    F = load("F")
    fig, ax = plt.subplots(figsize=(SINGLE, 3.0))
    for name, col, ls, lab in (("hydro", INK, "-", "hydrodynamic"),
                               ("20% phi=pi/4", C3, "-", r"20% contrast, $\varphi=\pi/4$"),
                               ("20% phi=3pi/4", C2, "-", r"20% contrast, $\varphi=3\pi/4$"),
                               ("30% phi=pi/4", C4, "--", r"30% contrast, $\varphi=\pi/4$")):
        ax.plot(F[name]["t"], F[name]["m4"], color=col, ls=ls, lw=1.5, label=lab)
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$|a_4|$"); ax.legend(frameon=False)
    save(fig, "fig11_zero_crossing.png")


def fig12_resolution():
    E, G = load("E"), load("G")
    lo_h, hi_h = E["hydro"], G["hydro"]
    lo_s, hi_s = E["r=2 phi=0pi/8"], G["r=2 ny=200"]
    vals = {
        "hydro a4": (lo_h["m4"][-1], hi_h["m4"][-1]),
        "stripe roughness/hydro": (lo_s["rms"][-1] / lo_h["rms"][-1], hi_s["rms"][-1] / hi_h["rms"][-1]),
    }
    en = (tail(lo_s["emix"], lo_s["t"]) / tail(lo_h["emix"], lo_h["t"]),
          tail(hi_s["emix"], hi_s["t"]) / tail(hi_h["emix"], hi_h["t"]))
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.5))
    xs = np.arange(len(vals))
    a.bar(xs - 0.18, [v[0] for v in vals.values()], 0.36, color=C1, label="$n_y=100$")
    a.bar(xs + 0.18, [v[1] for v in vals.values()], 0.36, color=C2, label="$n_y=200$")
    a.set_xticks(xs); a.set_xticklabels(["hydrodynamic\n$a_4$", "stripe roughness\n/ hydro"])
    a.legend(frameon=False)
    title(a, "(a) amplitude and roughness")
    b.bar([0, 1], en, 0.5, color=[C1, C2])
    b.set_yscale("log"); b.set_xticks([0, 1]); b.set_xticklabels(["$n_y=100$", "$n_y=200$"])
    b.set_ylabel("mixing-layer enstrophy / hydro")
    title(b, "(b) stripe enstrophy")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig12_resolution.png")
    return vals, en


CONV_CASES = (("uniform Bx=3", 3.0, 0.99), ("B0=3 eps=0.9", 3.0, 0.9), ("B0=3.5 eps=0.9", 3.5, 0.9),
              ("B0=4 eps=0.9", 4.0, 0.9), ("B0=3.5 eps=0.8", 3.5, 0.8))


def convergence_data():
    """Interface metrics at 1x, 2x, 4x resolution (experiments/convergence_lock.py)."""
    out = {}
    for r in (1, 2, 4):
        S = load(f"conv{r}")
        h = S["hydro"]
        row = {"hydro": {"emix": tail(h["emix"], h["t"]), "rough": float(h["rms"][-1])}}
        for name, b, e in CONV_CASES:
            if name in S:
                d = S[name]
                row[name] = {"emix": tail(d["emix"], d["t"]), "rough": float(d["rms"][-1]),
                             "a8": float(d["m8"][-1]), "dP": 0.5 * b**2 * (1 - e**2)}
        out[r] = row
    kappa = {}
    for r, row in out.items():
        pts = [(v["dP"], v["a8"]) for k, v in row.items() if k != "hydro" and not k.startswith("uniform")]
        x, y = np.array(pts).T
        kappa[r] = float(np.sum(x * y) / np.sum(x * x))      # least squares through the origin
    k1, k2, k4 = kappa[1], kappa[2], kappa[4]
    ratio = (k2 - k1) / (k4 - k2)
    kappa_inf = k4 + (k4 - k2) / (ratio - 1)
    return out, kappa, kappa_inf, np.log2(ratio)


def fig13_convergence():
    out, kappa, kappa_inf, order = convergence_data()
    res = (1, 2, 4)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(WIDE, 2.4))
    cols = {"hydro": INK, "uniform Bx=3": C3, "B0=3 eps=0.9": "#f0a04b", "B0=3.5 eps=0.9": C1,
            "B0=4 eps=0.9": C4, "B0=3.5 eps=0.8": C2}
    labs = {"hydro": "hydrodynamic", "uniform Bx=3": "uniform $B=3$", "B0=3 eps=0.9": r"$3, 0.9$",
            "B0=3.5 eps=0.9": r"$3.5, 0.9$", "B0=4 eps=0.9": r"$4, 0.9$", "B0=3.5 eps=0.8": r"$3.5, 0.8$"}
    for name in ("hydro",) + tuple(n for n, _, _ in CONV_CASES):
        rs = [r for r in res if name in out[r]]
        a.semilogy(rs, [out[r][name]["emix"] for r in rs], "o-", color=cols[name], ms=3.5, lw=1.1, label=labs[name])
        b.plot(rs, [out[r][name]["rough"] for r in rs], "o-", color=cols[name], ms=3.5, lw=1.1)
    for ax in (a, b):
        ax.set_xscale("log", base=2); ax.set_xticks(res); ax.set_xticklabels(["1x", "2x", "4x"]); ax.minorticks_off()
        ax.set_xlabel("resolution")
    title(a, "(a) mixing-layer enstrophy"); title(b, "(b) interface roughness")
    b.legend(*a.get_legend_handles_labels(), frameon=False, fontsize=6.5, title=r"$B_0, \epsilon$",
             title_fontsize=6.5, loc="center", bbox_to_anchor=(0.62, 0.56))
    for r, mk, cc in ((1, "s", "#bbbbbb"), (2, "^", "#888888"), (4, "o", INK)):
        pts = [(v["dP"], v["a8"]) for k, v in out[r].items() if k != "hydro"]
        x, y = np.array(pts).T
        c.plot(x, y, mk, color=cc, ms=4.5, label=f"{r}x, $\\kappa$ = {kappa[r]:.4f}")
        xx = np.linspace(0, 2.4, 10)
        c.plot(xx, kappa[r] * xx, color=cc, lw=0.9)
    c.set_xlabel(r"$\Delta P_B$"); c.set_ylabel("$a_8$ at $t=0.8$")
    c.legend(frameon=False, fontsize=6.5, loc="upper left")
    title(c, r"(c) stripe-scale mode, $a_8 = \kappa\,\Delta P_B$")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig13_convergence.png")
    return out, kappa, kappa_inf, order


if __name__ == "__main__":
    fig01_setup()
    fig02_equilibrium_admissibility()
    fig03_energy()
    fig04_enstrophy_location()
    fig05_legacy_div()
    fig06_growth()
    fig07_lock_map()
    fig08_spectrum()
    fig09_snapshots()
    fig10_phase()
    fig11_zero_crossing()
    vals, en = fig12_resolution()
    out, kappa, kappa_inf, order = fig13_convergence()
    print("kappa per resolution:", {r: round(k, 5) for r, k in kappa.items()},
          f"Richardson kappa_inf = {kappa_inf:.4f} (observed order {order:.2f})")
    for r, row in out.items():
        h = row["hydro"]
        print(f"{r}x hydro emix {h['emix']:.2f} rough {h['rough']:.4f} | " + "; ".join(
            f"{k}: rough/h {v['rough'] / h['rough']:.3f} a8 {v['a8']:.4f}" for k, v in row.items() if k != "hydro"))
    print("resolution:", {k: tuple(round(float(x), 4) for x in v) for k, v in vals.items()},
          "emix ratio", tuple(round(float(x), 1) for x in en))
