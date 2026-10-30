"""3D rendering of AEGIS-MHD snapshots with PyVista (off screen): ray-cast volume rendering of density,
liner / density isosurfaces, temperature hot-spot contours and magnetic field lines, one PNG per frame and an
MP4 of the series.

    python scripts/render_3d.py results/run/series.pvd               # ParaView series written by aegis.io3d
    python scripts/render_3d.py results/run                          # a run directory (series.pvd, vtk/, h5/)
    python scripts/render_3d.py frame_0003.vti frame_0004.h5 --out renders --size 1920 1080 --fps 12

Needs pyvista (pip install pyvista imageio imageio-ffmpeg; h5py for .h5 input).
"""
import os
import sys
import glob
import argparse
import xml.etree.ElementTree as ET

import numpy as np


def frames_from(paths):
    """expand the inputs into an ordered list of (time, file)"""
    out = []
    for p in paths:
        if os.path.isdir(p):
            if os.path.exists(os.path.join(p, "series.pvd")):
                out += frames_from([os.path.join(p, "series.pvd")])
            else:
                out += [(None, f) for f in sorted(glob.glob(os.path.join(p, "vtk", "*.vti")) +
                                                  glob.glob(os.path.join(p, "h5", "*.h5")) +
                                                  glob.glob(os.path.join(p, "*.vti")) + glob.glob(os.path.join(p, "*.h5")))]
        elif p.endswith(".pvd"):
            root = os.path.dirname(p)
            for ds in ET.parse(p).getroot().iter("DataSet"):
                out.append((float(ds.get("timestep")), os.path.join(root, ds.get("file"))))
        else:
            out.append((None, p))
    return out


def load(path):
    """pyvista ImageData with cell arrays from a .vti or an .h5 written by aegis.io3d"""
    import pyvista as pv
    if path.endswith(".h5"):
        import h5py
        with h5py.File(path, "r") as f:
            o, h, t = f.attrs["origin"], f.attrs["spacing"], float(f.attrs.get("time", 0.0))
            arrays = {k: f[k][()] for k in f.keys()}
        n = next(iter(arrays.values())).shape[:3]
        g = pv.ImageData(dimensions=(n[0] + 1, n[1] + 1, n[2] + 1), spacing=tuple(h), origin=tuple(o))
        for k, a in arrays.items():
            g.cell_data[k] = a.reshape(-1, 3, order="F") if a.ndim == 4 else a.ravel(order="F")
        g.field_data["time"] = [t]
        return g
    return pv.read(path)


def render(frames, out, size=(1600, 1200), fps=10, volume=True, streamlines=True, iso=None, hot=(0.5, 0.8),
           field="rho", log=True):
    import pyvista as pv
    os.makedirs(out, exist_ok=True)
    pv.OFF_SCREEN = True
    grids = [load(f) for _, f in frames]
    # fixed colour ranges over the series, so the movie does not flicker
    def rng(name, fn=lambda a: a):
        vals = [fn(np.asarray(g.cell_data[name])) for g in grids if name in g.cell_data]
        return (float(min(v.min() for v in vals)), float(max(v.max() for v in vals))) if vals else None
    lg = (lambda a: np.log10(np.maximum(a, 1e-30))) if log else (lambda a: a)
    fr = rng(field, lg)
    Tr = rng("T")
    pngs = []
    for k, ((t, f), g) in enumerate(zip(frames, grids)):
        pd = g.cell_data_to_point_data()
        p = pv.Plotter(off_screen=True, window_size=list(size))
        p.set_background("#0b0d12")
        scal = lg(np.asarray(pd.point_data[field]))
        pd.point_data["_f"] = scal
        b = pd.bounds
        c0 = ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
        # cut away the quadrant facing the camera (x > centre, y < centre) to look inside the liner
        cut = pd.clip_box((c0[0], b[1], b[2], c0[1], b[4], b[5]), invert=True)
        if volume:
            p.add_volume(pd, scalars="_f", cmap="inferno", clim=fr, opacity=list(np.linspace(0.0, 0.12, 12)), shade=False,
                         show_scalar_bar=False)
        if "liner" in pd.point_data and iso is None:
            surf = cut.contour([0.5], scalars="liner")
            if surf.n_points:
                p.add_mesh(surf, color="#b8c4d6", opacity=0.55, smooth_shading=True, specular=0.4)
        elif iso is not None:
            surf = cut.contour(list(iso), scalars="_f")
            if surf.n_points:
                p.add_mesh(surf, cmap="bone", opacity=0.5, smooth_shading=True, show_scalar_bar=False)
        if "T" in pd.point_data and Tr is not None:
            Tm = float(np.max(pd.point_data["T"]))
            hs = pd.contour([h * Tm for h in hot], scalars="T")
            if hs.n_points:
                p.add_mesh(hs, cmap="autumn", clim=(hot[0] * Tm, Tm), opacity=0.45, smooth_shading=True,
                           scalar_bar_args={"title": "T (hot spot)", "color": "white"})
        if streamlines and "B" in pd.point_data and float(np.max(np.linalg.norm(pd.point_data["B"], axis=1))) > 0:
            c = c0
            r = 0.25 * min(b[1] - b[0], b[3] - b[2])
            lines = pd.streamlines("B", source_center=c, source_radius=r, n_points=120, max_length=4 * (b[5] - b[4]),
                                   integration_direction="both", progress_bar=False)
            if lines.n_points:
                p.add_mesh(lines.tube(radius=0.004 * (b[1] - b[0])), color="#5fd3ff", opacity=0.85)
        tt = t if t is not None else (float(g.field_data["time"][0]) if "time" in g.field_data else None)
        p.add_text(f"{os.path.basename(f)}" + (f"   t = {tt:.4g}" if tt is not None else ""), font_size=10,
                   color="white")
        p.camera_position = "iso"
        p.camera.zoom(1.2)
        png = os.path.join(out, f"render_{k:04d}.png")
        p.screenshot(png)
        p.close()
        pngs.append(png)
    mp4 = None
    if len(pngs) > 1:
        import imageio.v2 as imageio
        mp4 = os.path.join(out, "render.mp4")
        try:
            with imageio.get_writer(mp4, fps=fps, codec="libx264", quality=8, macro_block_size=8) as w:
                for png in pngs:
                    w.append_data(imageio.imread(png))
        except Exception as e:                       # no ffmpeg: fall back to a GIF
            print(f"  MP4 failed ({e}); writing a GIF instead")
            mp4 = os.path.join(out, "render.gif")
            imageio.mimsave(mp4, [imageio.imread(p_) for p_ in pngs], duration=1.0 / fps)
    return pngs, mp4


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default=None, help="output directory (default: <first input dir>/renders)")
    ap.add_argument("--size", type=int, nargs=2, default=[1600, 1200])
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--field", default="rho", help="volume-rendered cell array")
    ap.add_argument("--linear", action="store_true", help="linear instead of log10 scale for the volume field")
    ap.add_argument("--iso", type=float, nargs="*", default=None, help="isosurface levels of the field (log10 if log)")
    ap.add_argument("--no-volume", action="store_true")
    ap.add_argument("--no-streamlines", action="store_true")
    a = ap.parse_args()
    frames = frames_from(a.inputs)
    if not frames:
        sys.exit("no frames found")
    first = a.inputs[0] if os.path.isdir(a.inputs[0]) else os.path.dirname(os.path.abspath(a.inputs[0]))
    out = a.out or os.path.join(first, "renders")
    pngs, mp4 = render(frames, out, tuple(a.size), a.fps, not a.no_volume, not a.no_streamlines, a.iso,
                       field=a.field, log=not a.linear)
    print(f"{len(pngs)} frames -> {out}" + (f", {os.path.basename(mp4)}" if mp4 else ""))


if __name__ == "__main__":
    main()
