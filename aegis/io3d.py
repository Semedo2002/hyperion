# io3d.py - 3D snapshot output: VTK XML ImageData (.vti, no VTK dependency), ParaView time series (.pvd), and
# HDF5 (.h5, when h5py is installed). scripts/render_3d.py reads all three.
import os
import struct

import numpy as np

from .config import iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iCLR
from .physics import cons_to_prim


def write_vti(path, cell_arrays, origin, spacing):
    """cell_arrays: name -> (nx, ny, nz) scalar or (nx, ny, nz, 3) vector; raw appended float32, x fastest"""
    first = next(iter(cell_arrays.values()))
    nx, ny, nz = first.shape[:3]
    blobs, header, offset = [], [], 0
    for name, a in cell_arrays.items():
        a = np.asarray(a, dtype="<f4")
        ncomp = 3 if a.ndim == 4 else 1
        data = (a.transpose(2, 1, 0, 3) if ncomp == 3 else a.transpose(2, 1, 0)).tobytes(order="C")
        header.append(f'        <DataArray type="Float32" Name="{name}" NumberOfComponents="{ncomp}" '
                      f'format="appended" offset="{offset}"/>')
        blobs.append(struct.pack("<Q", len(data)) + data)
        offset += 8 + len(data)
    ext = f"0 {nx} 0 {ny} 0 {nz}"
    xml = (f'<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
           f'header_type="UInt64">\n  <ImageData WholeExtent="{ext}" Origin="{origin[0]} {origin[1]} {origin[2]}" '
           f'Spacing="{spacing[0]} {spacing[1]} {spacing[2]}">\n    <Piece Extent="{ext}">\n      <CellData>\n'
           + "\n".join(header) + '\n      </CellData>\n    </Piece>\n  </ImageData>\n  <AppendedData encoding="raw">\n   _')
    with open(path, "wb") as fh:
        fh.write(xml.encode())
        for b in blobs:
            fh.write(b)
        fh.write(b"\n  </AppendedData>\n</VTKFile>\n")


def write_pvd(path, entries):
    rows = "\n".join(f'    <DataSet timestep="{t:.6g}" file="{f}"/>' for t, f in entries)
    with open(path, "w") as fh:
        fh.write(f'<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n  <Collection>\n'
                 f'{rows}\n  </Collection>\n</VTKFile>\n')


def write_h5(path, cell_arrays, origin, spacing, t=0.0):
    import h5py
    with h5py.File(path, "w") as f:
        f.attrs["origin"], f.attrs["spacing"], f.attrs["time"] = origin, spacing, t
        for name, a in cell_arrays.items():
            f.create_dataset(name, data=np.asarray(a, dtype="f4"), compression="gzip", compression_opts=4)


def fields(s):
    """standard snapshot fields of a 3D solver (interior cells, host float32): density, pressure, temperature
    p/rho, liner fraction, velocity, magnetic field, |B|, and E_r with FLD"""
    from .solver import to_numpy
    ng = s.ng
    I = (slice(None),) + (slice(ng, -ng),) * 3
    W = cons_to_prim(s.U, s.eos)[I]
    h = lambda a: to_numpy(a).astype(np.float32)
    out = {"rho": h(W[iRHO]), "p": h(W[iPR]), "T": h(W[iPR] / W[iRHO]), "liner": h(W[iCLR]),
           "v": np.stack([h(W[iVX]), h(W[iVY]), h(W[iVZ])], axis=-1),
           "B": np.stack([h(W[iBX]), h(W[iBY]), h(W[iBZ])], axis=-1)}
    out["Bmag"] = np.linalg.norm(out["B"], axis=-1)
    if getattr(s, "Er", None) is not None:
        out["Er"] = h(s.Er)
    return out


class Recorder:
    """writes numbered 3D snapshots of a solver into out_dir (vtk/frame_NNNN.vti + series.pvd, and/or
    h5/frame_NNNN.h5)"""

    def __init__(self, out_dir, fmt=("vti",)):
        self.out, self.fmt, self.entries, self.k = out_dir, tuple(fmt), [], 0
        for f in self.fmt:
            os.makedirs(os.path.join(out_dir, "vtk" if f == "vti" else f), exist_ok=True)

    def __call__(self, s):
        cfg = s.cfg
        arr = fields(s)
        origin, spacing = (cfg.x_min, cfg.y_min, cfg.z_min), (cfg.dx, cfg.dy, cfg.dz)
        if "vti" in self.fmt:
            name = f"vtk/frame_{self.k:04d}.vti"
            write_vti(os.path.join(self.out, name), arr, origin, spacing)
            self.entries.append((s.t, name))
            write_pvd(os.path.join(self.out, "series.pvd"), self.entries)
        if "h5" in self.fmt:
            write_h5(os.path.join(self.out, f"h5/frame_{self.k:04d}.h5"), arr, origin, spacing, s.t)
        self.k += 1
