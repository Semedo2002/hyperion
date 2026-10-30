# parallel.py - slab domain decomposition along x for MHDSolver (2D Cartesian) and MHDSolver3D, over MPI.
#
# Each rank runs an ordinary solver on its own x-range (same dx to the last bit) and swaps ng ghost layers
# with its neighbours: cells (U), the constrained-transport face fields and the liner stress. HLLD fluxes on
# a shared face then come out identical on both ranks, so mass, momentum and energy stay conserved across
# the cut. The edge EMFs on a shared x-face are taken from the left rank on both sides (they depend on a
# local stagnation tolerance), so both ranks update the shared face field identically and the global face
# divergence stays at round-off. dt is the global minimum.
#
# Halo transport: NCCL device-to-device when CuPy has it (Linux) and backend="nccl"/"auto", otherwise MPI with
# the halos staged through host memory (or passed as device buffers if the MPI is CUDA-aware, cuda_aware=True).
#
#   from aegis.parallel import distributed
#   s = distributed(MHDSolver3D, cfg)            # under mpiexec -n P; cfg describes the whole grid
#   ... set s.U on the local grid (s.x holds this rank's cell centres) and march as usual ...
#   rho = s.gather_x(rho_interior)                # whole field on rank 0
#
# Not decomposed (raise): cylindrical geometry, the split-step stencils (conduction, viscosity, FLD), driven-inflow
# x boundaries, sponges.
import dataclasses

import numpy as np

from .config import Config
from .physics import xp


def _split(n, p):
    base, extra = divmod(n, p)
    sizes = [base + (1 if r < extra else 0) for r in range(p)]
    return sizes, [sum(sizes[:r]) for r in range(p)]


def local_config(cfg, rank, size):
    """this rank's Config: nx and the x-range of its slab, global dx kept exactly; returns (cfg, offset)"""
    sizes, offs = _split(cfg.nx, size)
    if min(sizes) < 4:
        raise ValueError(f"nx = {cfg.nx} over {size} ranks leaves slabs under 4 cells")
    dx = cfg.dx
    grid = dataclasses.replace(cfg.grid, nx=sizes[rank], x_min=cfg.x_min + offs[rank] * dx,
                               x_max=cfg.x_min + (offs[rank] + sizes[rank]) * dx, x_spacing=dx)
    phys = dataclasses.replace(cfg.physics, use_cuda_graphs=False)   # halos are swapped inside the RK stages
    out = Config.__new__(Config)
    out.grid, out.physics, out.io = grid, phys, cfg.io
    return out, offs[rank]


class _Halo:
    """swaps slabs with the left and right neighbours (None at a non-periodic domain end)"""

    def __init__(self, comm, left, right, backend, cuda_aware):
        from mpi4py import MPI
        self.comm, self.left, self.right, self.MPI = comm, left, right, MPI
        self.cuda_aware = cuda_aware
        self.nccl = None
        if backend in ("nccl", "auto"):
            try:
                from cupy.cuda import nccl
                if not nccl.available:
                    raise ImportError
                uid = nccl.get_unique_id() if comm.rank == 0 else None
                uid = comm.bcast(uid, root=0)
                self.nccl = nccl.NcclCommunicator(comm.size, uid, comm.rank)
                self._nc = nccl
            except Exception:
                if backend == "nccl":
                    raise RuntimeError("backend='nccl' needs CuPy built with NCCL (Linux)")
        self.backend = "nccl" if self.nccl is not None else "mpi"

    def swap(self, to_left, to_right):
        """send to_left / to_right (device arrays, same shape on every rank); returns (from_left, from_right),
        None where there is no neighbour"""
        to_left, to_right = xp.ascontiguousarray(to_left), xp.ascontiguousarray(to_right)
        if self.backend == "nccl":
            import cupy as cp
            nc = self._nc
            fl = cp.empty_like(to_right) if self.left is not None else None
            fr = cp.empty_like(to_left) if self.right is not None else None
            st = cp.cuda.get_current_stream().ptr
            nc.groupStart()
            if self.right is not None:
                self.nccl.send(to_right.data.ptr, to_right.size, nc.NCCL_FLOAT64, self.right, st)
                self.nccl.recv(fr.data.ptr, fr.size, nc.NCCL_FLOAT64, self.right, st)
            if self.left is not None:
                self.nccl.send(to_left.data.ptr, to_left.size, nc.NCCL_FLOAT64, self.left, st)
                self.nccl.recv(fl.data.ptr, fl.size, nc.NCCL_FLOAT64, self.left, st)
            nc.groupEnd()
            return fl, fr
        MPI = self.MPI
        host = (lambda a: a) if (self.cuda_aware or xp.__name__ != "cupy") else (lambda a: a.get())
        L = self.left if self.left is not None else MPI.PROC_NULL
        Rr = self.right if self.right is not None else MPI.PROC_NULL
        sr, sl = host(to_right), host(to_left)
        bl = np.empty(sr.shape) if not self.cuda_aware else xp.empty(sr.shape)
        br = np.empty(sl.shape) if not self.cuda_aware else xp.empty(sl.shape)
        if xp.__name__ == "cupy" and not self.cuda_aware:
            xp.cuda.Device().synchronize()
        self.comm.Sendrecv(sr, dest=Rr, sendtag=1, recvbuf=bl, source=L, recvtag=1)
        self.comm.Sendrecv(sl, dest=L, sendtag=2, recvbuf=br, source=Rr, recvtag=2)
        fl = xp.asarray(bl) if self.left is not None else None
        fr = xp.asarray(br) if self.right is not None else None
        return fl, fr


class _Distributed:
    """mixin over MHDSolver / MHDSolver3D (built by distributed())"""

    def _xsl(self, A, ax, s):
        idx = [slice(None)] * A.ndim
        idx[ax] = s
        return tuple(idx)

    def _swap_cells(self, A, ax):
        """ghost layers of a cell-centred array along its x axis ax"""
        ng, n = self.ng, self.cfg.nx
        fl, fr = self.halo.swap(A[self._xsl(A, ax, slice(ng, 2 * ng))], A[self._xsl(A, ax, slice(n, n + ng))])
        if fl is not None:
            A[self._xsl(A, ax, slice(0, ng))] = fl
        if fr is not None:
            A[self._xsl(A, ax, slice(ng + n, n + 2 * ng))] = fr

    def apply_bc(self, U):
        super().apply_bc(U)                          # y, z and the physical x ends
        self._swap_cells(U, 1)
        return U

    def _ghost_faces(self, Bf, U):
        super()._ghost_faces(Bf, U)
        ng, n = self.ng, self.cfg.nx
        # x-faces: index = cell on the high side; my ghost faces 0..ng-1 are the left rank's faces n..n+ng-1,
        # my faces ng+n+1.. are the right rank's ng+1..2ng
        f = Bf[0]
        fl, fr = self.halo.swap(f[ng + 1:2 * ng + 1], f[n:n + ng])
        if fl is not None:
            f[0:ng] = fl
        if fr is not None:
            f[ng + n + 1:n + 2 * ng + 1] = fr
        for f in Bf[1:]:
            self._swap_cells(f, 0)

    def _halo(self, A):
        self._swap_cells(A, 1)

    def _emf_hook(self, E):
        """EMFs on the edges of a shared x-face: the left rank's values on both sides"""
        n = self.cfg.nx
        Es = E if isinstance(E, list) else [E]
        for k, e in enumerate(Es):
            if isinstance(E, list) and k == 0:       # 3D: E_x edges run along x, none lie on an x-face
                continue
            fl, _ = self.halo.swap(e[0:1], e[n:n + 1])
            if fl is not None:
                e[0:1] = fl
        return E

    def _ct_sync(self):
        if self.ct and self._faces_owner is not self.U:
            self.apply_bc(self.U)                    # faces from cells need the neighbours' ghost cells
        super()._ct_sync()

    def compute_dt(self, W):
        dt = super().compute_dt(W)
        dt = self.comm.allreduce(dt, op=self.halo.MPI.MIN)
        if hasattr(self.cfg, "glm_ch"):
            try:
                self.cfg.glm_ch = self.comm.allreduce(self.cfg.glm_ch, op=self.halo.MPI.MAX)
            except AttributeError:
                pass
        if hasattr(self, "_ch"):
            self._ch = self.comm.allreduce(self._ch, op=self.halo.MPI.MAX)
        return dt

    def div_b_max(self):
        return self.comm.allreduce(super().div_b_max(), op=self.halo.MPI.MAX)

    def global_sum(self, v):
        return self.comm.allreduce(float(v), op=self.halo.MPI.SUM)

    def gather_x(self, a, ax=0):
        """interior array of this slab (x along axis ax) -> the whole field on rank 0 (None elsewhere)"""
        from .solver import to_numpy
        parts = self.comm.gather(to_numpy(a), root=0)
        return np.concatenate(parts, axis=ax) if self.comm.rank == 0 else None


def distributed(cls, cfg, comm=None, backend="auto", cuda_aware=False, device=None):
    """an instance of cls (MHDSolver or MHDSolver3D) running this rank's slab of cfg's grid"""
    from mpi4py import MPI
    comm = comm or MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    if cfg.geometry != "cartesian":
        raise NotImplementedError("domain decomposition: Cartesian grids only (r-z needs the axis on one rank)")
    if cfg.cond_kappa0 > 0 or cfg.visc_coeff > 0 or cfg.fld:
        raise NotImplementedError("domain decomposition: the conduction / viscosity / FLD split steps are not decomposed")
    bcx = cfg.get_bc_x()
    if bcx not in ("periodic", "extrapolation"):
        raise NotImplementedError(f"domain decomposition: x boundary {bcx!r} (periodic or extrapolation only)")
    if xp.__name__ == "cupy":
        import cupy as cp
        cp.cuda.Device(device if device is not None else rank % cp.cuda.runtime.getDeviceCount()).use()
    lcfg, off = local_config(cfg, rank, size)
    per = bcx == "periodic"
    left = (rank - 1) % size if (rank > 0 or per) else None
    right = (rank + 1) % size if (rank < size - 1 or per) else None
    if size == 1:
        left = right = None
    Dist = type("Distributed" + cls.__name__, (_Distributed, cls), {})
    s = Dist(lcfg)
    s.comm, s.offset, s.nx_global = comm, off, cfg.nx
    s.halo = _Halo(comm, left, right, backend, cuda_aware)
    if size == 1 and per:                            # one rank, periodic: the base boundary conditions do it
        s.halo.left = s.halo.right = None
    return s
