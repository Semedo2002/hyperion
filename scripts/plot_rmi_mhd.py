"""Plots for the high-Mach RMI benchmark.

    python scripts/plot_rmi_mhd.py                       # results/rmi_mhd
    python scripts/plot_rmi_mhd.py --dir results/rmi_mhd_quick

Writes into the results directory:
  rmi_amplitude.png           a(t) for each case with the Richtmyer and Meyer-Blewett lines
  rmi_diagnostics.png         max vorticity, energy error, div B, bubble/spike positions
  rmi_density_fieldlines.png  side-by-side log density at the last frame, field lines for MHD
  rmi_density_fieldlines.gif  the same over all frames
Field lines are contours of the flux function A(x, y), with dA/dy = B_x and dA/dx = -B_y.
"""
import os
import sys
import csv
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"hydro": "#222222", "mhd_parallel": "#c0392b", "mhd_perpendicular": "#1f5f99"}
LABELS = {"hydro": "hydro ($B=0$)", "mhd_parallel": "MHD, $B_x$", "mhd_perpendicular": "MHD, $B_y$"}


def load_case(path):
    with open(os.path.join(path, "summary.json")) as fh:
        summary = json.load(fh)
    with open(os.path.join(path, "diagnostics.csv")) as fh:
        rows = list(csv.DictReader(fh))
    diag = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    frames = np.load(os.path.join(path, "frames.npz"))
    return summary, diag, frames


def label(name, summary):
    p = summary["params"]
    base = LABELS.get(name, name)
    return base if p["B0"] == 0 else base.replace("$B_x$", f"$B_x={p['B0']:g}$").replace("$B_y$", f"$B_y={p['B0']:g}$")


def plot_amplitude(cases, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ref = cases.get("hydro", next(iter(cases.values())))[0]
    lin, p = ref["linear"], ref["params"]
    t = np.linspace(0, p["t_end"], 300)
    for key, ls, name in (("v_impulsive", "--", "Richtmyer (impulsive)"), ("v_meyer_blewett", ":", "Meyer-Blewett")):
        a = np.where(t < lin["t_impact"], lin["a0"], lin["a0_post"] + lin[key] * (t - lin["t_impact"]))
        ax.plot(t, a, color="#8a8a8a", ls=ls, lw=1.2, label=f"{name}, hydro linear")
    for name, (summary, diag, _) in cases.items():
        ax.plot(diag["t"], diag["a"], color=COLORS.get(name, None), lw=2.0, label=label(name, summary))
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$a(t) = (x_{bubble} - x_{spike})/2$")
    ax.set_xlim(0, p["t_end"]); ax.set_ylim(bottom=0)
    ax.set_title(f"RMI amplitude, Mach {p['mach']:g}, $\\rho_2/\\rho_1={p['density_ratio']:g}$ "
                 f"($A_{{post}}={lin['A_post']:.2f}$)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rmi_amplitude.png"), dpi=220)
    plt.close(fig)


def plot_diagnostics(cases, out):
    fig, ax = plt.subplots(2, 2, figsize=(9.0, 6.0))
    for name, (summary, d, _) in cases.items():
        kw = dict(color=COLORS.get(name, None), lw=1.4, label=label(name, summary))
        ax[0, 0].plot(d["t"], d["omega_max"], **kw)
        ax[0, 1].semilogy(d["t"][1:], np.maximum(np.abs(d["energy_error"][1:]), 1e-17), **kw)
        if summary["params"]["B0"] > 0:
            ax[1, 0].plot(d["t"], d["divB_rel"], **kw)
        ax[1, 1].plot(d["t"], d["x_bubble"] - d["x_mean"], **kw)
        ax[1, 1].plot(d["t"], d["x_spike"] - d["x_mean"], color=kw["color"], lw=1.4, ls="--")
    titles = (r"(a) max $|\omega_z|$", "(b) energy conservation error", r"(c) $h\,\max|\nabla\cdot B|/\max|B|$",
              "(d) bubble (solid) and spike (dashed) relative to mean interface")
    for a, ttl in zip(ax.flat, titles):
        a.set_title(ttl, loc="left", fontsize=10); a.set_xlabel("$t$")
    ax[0, 0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rmi_diagnostics.png"), dpi=200)
    plt.close(fig)


def flux_function(Bx, By, dx, dy):
    """A(x, y) = int_0^y B_x dy' - int_0^x B_y(x', 0) dx' on cell centres."""
    A = np.cumsum(Bx, axis=1) * dy
    A -= (np.cumsum(By[:, 0]) * dx)[:, None]
    return A


def render_frame(cases, k, window, vlim, dpi):
    names = list(cases)
    fig, axes = plt.subplots(1, len(names), figsize=(6.2 * len(names), 2.9), squeeze=False)
    for ax, name in zip(axes[0], names):
        summary, diag, fr = cases[name]
        x, y = fr["x"], fr["y"]
        rho, C = fr[f"rho{k:03d}"], fr[f"C{k:03d}"]
        t = float(fr[f"t{k:03d}"])
        xm = float(np.interp(t, diag["t"], diag["x_mean"]))
        lo, hi = xm - window / 2, xm + window / 2
        m = (x >= lo) & (x <= hi)
        im = ax.pcolormesh(x[m], y, np.log10(rho[m]).T, cmap="inferno", vmin=vlim[0], vmax=vlim[1],
                           shading="auto", rasterized=True)
        ax.contour(x[m], y, C[m].T, levels=[0.5], colors="#5fd3f3", linewidths=0.8)
        if summary["params"]["B0"] > 0:
            dx, dy = x[1] - x[0], y[1] - y[0]
            A = flux_function(fr[f"Bx{k:03d}"], fr[f"By{k:03d}"], dx, dy)
            ax.contour(x[m], y, A[m].T, levels=14, colors="white", linewidths=0.6, alpha=0.85)
        ax.set_aspect("equal"); ax.set_xlim(lo, hi); ax.set_ylim(y[0] - 0.5 * (y[1] - y[0]), y[-1] + 0.5 * (y[1] - y[0]))
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
        ax.set_title(f"{label(name, summary)},  $t = {t:.3f}$", loc="left", fontsize=10)
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.9, pad=0.015)
    cb.set_label(r"$\log_{10}\rho$")
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
    return fig, img.convert("RGB")


def plot_frames(cases, out, window=2.5, fps=6):
    n = min(sum(1 for k in fr.files if k.startswith("t") and k[1:].isdigit()) for _, _, fr in cases.values())
    rmax = max(float(fr[f"rho{k:03d}"].max()) for _, _, fr in cases.values() for k in range(n))
    vlim = (np.log10(0.9), np.log10(rmax))
    images = []
    for k in range(n):
        fig, img = render_frame(cases, k, window, vlim, dpi=110)
        images.append(img)
        if k == n - 1:
            fig.savefig(os.path.join(out, "rmi_density_fieldlines.png"), dpi=250, bbox_inches="tight")
        plt.close(fig)
    images[0].save(os.path.join(out, "rmi_density_fieldlines.gif"), save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "rmi_mhd"))
    args = ap.parse_args(argv)
    names = sorted(d for d in os.listdir(args.dir) if os.path.isfile(os.path.join(args.dir, d, "summary.json")))
    names.sort(key=lambda n: (n != "hydro", n))
    if not names:
        sys.exit(f"no cases found in {args.dir}")
    cases = {n: load_case(os.path.join(args.dir, n)) for n in names}
    plot_amplitude(cases, args.dir)
    plot_diagnostics(cases, args.dir)
    n = plot_frames(cases, args.dir)
    print(f"wrote rmi_amplitude.png, rmi_diagnostics.png, rmi_density_fieldlines.png/.gif ({n} frames) in {args.dir}")


if __name__ == "__main__":
    main()
