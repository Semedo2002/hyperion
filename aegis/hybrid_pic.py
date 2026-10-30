# hybrid_pic.py - adaptive kinetic-fluid hybrid: cells where the fluid description breaks down carry their ions
# as particles (particle-in-cell), the rest stays MHD.
#
# Classification (every step, on the device): the local Knudsen number Kn = lambda / L with the ion mean free
# path lambda = pic_mfp0 theta^2 / rho (theta = p / rho; the Spitzer scaling lambda ~ T^2 / n) and L the shortest
# of the density and pressure gradient lengths and the domain size; and the ion Larmor radius in cells,
# r_L / dx with r_L = pic_di sqrt(theta) / |B|. A cell turns kinetic when Kn > pic_kn_on or r_L > pic_rl_on dx
# (vacuum feeds, current sheets, sheaths) and returns to the fluid when both drop below pic_hysteresis times
# their thresholds.
#
# One step (a split step after the MHD stages, so the fluid step U* is conservative everywhere):
#   1. cells that turned fluid drop their particles (their state is already the particle moments); cells that
#      turned kinetic get pic_ppc particles sampled from their fluid state;
#   2. particles move for dt with the Boris pusher in the MHD fields, B and E = -u x B (ideal Ohm's law, u the
#      fluid velocity), bilinear in the cell-centred values; q/m = 1 / pic_di;
#   3. particles that left the kinetic region (or the domain) are dropped; kinetic cells left with fewer than
#      pic_ppc / 4 particles are resampled from U*;
#   4. conservation: kinetic cells on the edge of the kinetic region (the buffer, which exchanges fluxes with
#      the fluid through the HLLD faces) and resampled cells are matched cell by cell to U*; every connected
#      region of the remaining kinetic cells is matched as a whole: particle weights scaled to its U* mass, one
#      velocity shift to its U* momentum, and the peculiar velocities (relative to each cell's mean) scaled to
#      its U* energy. Mass, momentum and total energy of the domain therefore equal those of the conservative
#      fluid step to round-off, while inside a region the particles, not the fluid, decide where mass, momentum
#      and heat go (free streaming, gyration, non-Maxwellian distributions);
#   5. the particle moments (NGP deposit) replace rho, rho u and E - B^2/2 in the kinetic cells; the tracers keep
#      their mass fractions. B is never touched (CT keeps div B at round-off).
#
# Model: the particles carry the whole thermal energy (p = 2/3 of the thermal energy density, so gamma = 5/3);
# electrons are not separated (cold-electron hybrid) and E has no Hall or electron-pressure term. The fluid
# sees the particles' scalar pressure (the trace); anisotropy and heat flux live in the particles only.
import numpy as np

from .config import FLOOR_RHO, RHO, MX, MY, MZ, BX, BY, BZ, EN, RHOC, RHOSE
from .physics import xp

def _bincount(idx, w, n):
    return xp.bincount(idx, weights=w, minlength=n)[:n]


class HybridPIC:
    def __init__(self, solver):
        s, cfg = solver, solver.cfg
        self.s, self.cfg = s, cfg
        self.nx, self.ny, self.ng = cfg.nx, cfg.ny, s.ng
        self.dx, self.dy = cfg.dx, cfg.dy
        self.dV = cfg.dx * cfg.dy
        self.qm = 1.0 / cfg.pic_di if cfg.pic_di > 0 else 0.0
        self.rng = xp.random.default_rng(cfg.pic_seed)
        self.K = xp.zeros((self.nx, self.ny), dtype=bool)  # kinetic cells (interior)
        self.force = None                                   # optional boolean array: these cells always kinetic
        self.P = {k: xp.zeros(0) for k in ("x", "y", "vx", "vy", "vz", "w")}
        self.n_switch_on = self.n_switch_off = 0
        self.n_resampled = 0
        self.n_energy_clamped = 0
        self.Kn = self.rL = None

    # ---------------- classification ----------------
    def criteria(self, W):
        """(Kn, r_L / dx) on the interior cells from primitives with ghost cells"""
        ng, nx, ny, cfg = self.ng, self.nx, self.ny, self.cfg
        rho = xp.maximum(W[0], FLOOR_RHO)
        p = xp.maximum(W[7], 1e-300)
        theta = p / rho
        inner = (slice(ng, ng + nx), slice(ng, ng + ny))
        if cfg.pic_mfp0 > 0:
            lr, lp = xp.log(rho), xp.log(p)
            g = 0.0
            for f in (lr, lp):
                gx = (f[ng + 1:ng + nx + 1, ng:ng + ny] - f[ng - 1:ng + nx - 1, ng:ng + ny]) / (2 * self.dx)
                gy = (f[ng:ng + nx, ng + 1:ng + ny + 1] - f[ng:ng + nx, ng - 1:ng + ny - 1]) / (2 * self.dy)
                g = xp.maximum(g, xp.sqrt(gx**2 + gy**2))
            Lbox = max(cfg.x_max - cfg.x_min, cfg.y_max - cfg.y_min)
            Kn = cfg.pic_mfp0 * theta[inner]**2 / rho[inner] * (g + 1.0 / Lbox)
        else:
            Kn = xp.zeros((nx, ny))
        if cfg.pic_di > 0:
            B = xp.sqrt(W[4]**2 + W[5]**2 + W[6]**2)[inner]
            rL = cfg.pic_di * xp.sqrt(theta[inner]) / xp.maximum(B, 1e-300) / min(self.dx, self.dy)
        else:
            rL = xp.zeros((nx, ny))
        return Kn, rL

    def classify(self, W):
        cfg = self.cfg
        self.Kn, self.rL = Kn, rL = self.criteria(W)
        on = xp.zeros_like(self.K)
        off = xp.ones_like(self.K)
        if cfg.pic_mfp0 > 0:
            on |= Kn > cfg.pic_kn_on
            off &= Kn < cfg.pic_hysteresis * cfg.pic_kn_on
        if cfg.pic_di > 0:
            on |= rL > cfg.pic_rl_on
            off &= rL < cfg.pic_hysteresis * cfg.pic_rl_on
        K = on | (self.K & ~off)
        if self.force is not None:
            K |= self.force
        return K

    # ---------------- particles ----------------
    def _cell(self, x, y):
        cfg = self.cfg
        i = xp.floor((x - cfg.x_min) / self.dx).astype(xp.int64)
        j = xp.floor((y - cfg.y_min) / self.dy).astype(xp.int64)
        return i, j

    def _keep(self, m):
        for k in self.P:
            self.P[k] = self.P[k][m]

    def _append(self, d):
        for k in self.P:
            self.P[k] = xp.concatenate([self.P[k], d[k]])

    def sample(self, cells, Ui):
        """pic_ppc particles in each cell of the boolean map `cells`, with exactly the cell's mass, momentum and
        thermal energy (Ui: interior conserved state)"""
        n_cell = int(xp.count_nonzero(cells))
        if n_cell == 0:
            return
        cfg, ppc = self.cfg, self.cfg.pic_ppc
        ci = xp.nonzero(cells.ravel())[0]
        c = xp.repeat(ci, ppc)
        i, j = c // self.ny, c % self.ny
        n = c.size
        x = cfg.x_min + (i + self.rng.random(n)) * self.dx
        y = cfg.y_min + (j + self.rng.random(n)) * self.dy
        rho = xp.maximum(Ui[RHO].ravel()[ci], FLOOR_RHO)
        u = [Ui[m].ravel()[ci] / rho for m in (MX, MY, MZ)]
        B2 = Ui[BX].ravel()[ci]**2 + Ui[BY].ravel()[ci]**2 + Ui[BZ].ravel()[ci]**2
        eth = xp.maximum(Ui[EN].ravel()[ci] - 0.5 * rho * (u[0]**2 + u[1]**2 + u[2]**2) - 0.5 * B2, 0.0)
        G = self.rng.standard_normal((3, n))
        # exact moments: zero-mean, unit-energy noise per cell, then scaled
        loc = xp.repeat(xp.arange(n_cell), ppc)
        G -= xp.stack([_bincount(loc, g, n_cell)[loc] / ppc for g in G])
        e = _bincount(loc, 0.5 * (G**2).sum(axis=0), n_cell)
        scale = xp.sqrt(xp.where(e > 0, eth * self.dV / xp.maximum(e * rho * self.dV / ppc, 1e-300), 0.0))
        v = [u[k][loc] + G[k] * scale[loc] for k in range(3)]
        w = xp.repeat(rho * self.dV / ppc, ppc)
        self._append(dict(x=x, y=y, vx=v[0], vy=v[1], vz=v[2], w=w))

    def fields(self, U):
        """cell-centred B and E = -u x B on all cells (ghosts included)"""
        rho = xp.maximum(U[RHO], FLOOR_RHO)
        u = [U[m] / rho for m in (MX, MY, MZ)]
        B = [U[BX], U[BY], U[BZ]]
        E = [-(u[1] * B[2] - u[2] * B[1]), -(u[2] * B[0] - u[0] * B[2]), -(u[0] * B[1] - u[1] * B[0])]
        return E, B

    def _interp(self, F, x, y):
        cfg, ng = self.cfg, self.ng
        fx = (x - cfg.x_min) / self.dx - 0.5 + ng
        fy = (y - cfg.y_min) / self.dy - 0.5 + ng
        i0 = xp.clip(xp.floor(fx).astype(xp.int64), 0, F[0].shape[0] - 2)
        j0 = xp.clip(xp.floor(fy).astype(xp.int64), 0, F[0].shape[1] - 2)
        ax, ay = fx - i0, fy - j0
        out = []
        for f in F:
            out.append((1 - ax) * (1 - ay) * f[i0, j0] + ax * (1 - ay) * f[i0 + 1, j0]
                       + (1 - ax) * ay * f[i0, j0 + 1] + ax * ay * f[i0 + 1, j0 + 1])
        return out

    def push(self, dt, U):
        """Boris step in E, B (cell-centred, bilinear), then the position update and the domain boundaries"""
        P, qm = self.P, self.qm
        if P["x"].size == 0:
            return
        v = [P["vx"], P["vy"], P["vz"]]
        if qm != 0.0:
            E, B = self.fields(U)
            Ep = self._interp(E, P["x"], P["y"])
            Bp = self._interp(B, P["x"], P["y"])
            h = 0.5 * qm * dt
            vm = [v[k] + h * Ep[k] for k in range(3)]
            t = [h * b for b in Bp]
            t2 = t[0]**2 + t[1]**2 + t[2]**2
            sv = [2 * tk / (1 + t2) for tk in t]
            vp = [vm[0] + vm[1] * t[2] - vm[2] * t[1], vm[1] + vm[2] * t[0] - vm[0] * t[2],
                  vm[2] + vm[0] * t[1] - vm[1] * t[0]]
            vq = [vm[0] + vp[1] * sv[2] - vp[2] * sv[1], vm[1] + vp[2] * sv[0] - vp[0] * sv[2],
                  vm[2] + vp[0] * sv[1] - vp[1] * sv[0]]
            v = [vq[k] + h * Ep[k] for k in range(3)]
        P["vx"], P["vy"], P["vz"] = v
        cfg = self.cfg
        P["x"] = P["x"] + v[0] * dt
        P["y"] = P["y"] + v[1] * dt
        Lx, Ly = cfg.x_max - cfg.x_min, cfg.y_max - cfg.y_min
        keep = xp.ones(P["x"].shape, dtype=bool)
        if cfg.get_bc_x() == "periodic":
            P["x"] = cfg.x_min + xp.mod(P["x"] - cfg.x_min, Lx)
        else:
            keep &= (P["x"] >= cfg.x_min) & (P["x"] < cfg.x_max)
        if cfg.bc_y_type == "periodic":
            P["y"] = cfg.y_min + xp.mod(P["y"] - cfg.y_min, Ly)
        else:
            keep &= (P["y"] >= cfg.y_min) & (P["y"] < cfg.y_max)
        self._keep(keep)

    def kinetic_energy(self):
        P = self.P
        return float(xp.sum(0.5 * P["w"] * (P["vx"]**2 + P["vy"]**2 + P["vz"]**2)))

    # ---------------- the coupled step ----------------
    def _match(self, lab, nlab, Ui):
        """scale weights, shift and stretch velocities so that each labelled group of cells holds exactly the
        U* mass, momentum and energy (E - B^2/2) of those cells"""
        P, n_c = self.P, self.nx * self.ny
        i, j = self._cell(P["x"], P["y"])
        c = i * self.ny + j
        pl = lab.ravel()[c] - 1                              # particle group (-1: none)
        has = pl >= 0
        labc = lab.ravel() - 1
        inc = labc >= 0
        dV = self.dV
        B2 = (Ui[BX]**2 + Ui[BY]**2 + Ui[BZ]**2).ravel()
        tgt_M = _bincount(labc[inc], Ui[RHO].ravel()[inc] * dV, nlab)
        tgt_P = [_bincount(labc[inc], Ui[m].ravel()[inc] * dV, nlab) for m in (MX, MY, MZ)]
        tgt_E = _bincount(labc[inc], (Ui[EN].ravel() - 0.5 * B2)[inc] * dV, nlab)
        g, w = pl[has], P["w"][has]
        v = [P["vx"][has], P["vy"][has], P["vz"][has]]
        M = _bincount(g, w, nlab)
        f = tgt_M / xp.maximum(M, 1e-300)
        w = w * f[g]
        dv = [(tgt_P[k] - _bincount(g, w * v[k], nlab)) / xp.maximum(tgt_M, 1e-300) for k in range(3)]
        v = [v[k] + dv[k][g] for k in range(3)]
        cc = c[has]
        mc = _bincount(cc, w, n_c)
        uc = [_bincount(cc, w * v[k], n_c) / xp.maximum(mc, 1e-300) for k in range(3)]
        pec = [v[k] - uc[k][cc] for k in range(3)]
        e_bulk = _bincount(g, 0.5 * w * (uc[0][cc] * v[0] + uc[1][cc] * v[1] + uc[2][cc] * v[2]), nlab)
        e_th = _bincount(g, 0.5 * w * (pec[0]**2 + pec[1]**2 + pec[2]**2), nlab)
        a2 = (tgt_E - e_bulk) / xp.maximum(e_th, 1e-300)
        self.n_energy_clamped = int(xp.count_nonzero(a2 < 0))
        a = xp.sqrt(xp.maximum(a2, 0.0))
        v = [uc[k][cc] + a[g] * pec[k] for k in range(3)]
        P["w"][has] = w
        P["vx"][has], P["vy"][has], P["vz"][has] = v

    def step(self, dt):
        s, ng, nx, ny = self.s, self.ng, self.nx, self.ny
        from .physics import cons_to_prim
        from cupyx.scipy import ndimage
        U = s.apply_bc(s.U)
        W = cons_to_prim(U, s.eos)
        inner = (slice(None), slice(ng, ng + nx), slice(ng, ng + ny))
        Ui = U[inner]
        # 1. classification and switches
        K = self.classify(W)
        off, on = self.K & ~K, K & ~self.K
        self.n_switch_on += int(xp.count_nonzero(on))
        self.n_switch_off += int(xp.count_nonzero(off))
        # 2. push in the fields of the fluid step
        self.push(dt, U)
        # 3. drop particles outside the kinetic region; resample sparse cells
        i, j = self._cell(self.P["x"], self.P["y"])
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        i, j = xp.clip(i, 0, nx - 1), xp.clip(j, 0, ny - 1)
        self._keep(ok & K[i, j])
        i, j = self._cell(self.P["x"], self.P["y"])
        cnt = _bincount(i * ny + j, None, nx * ny).reshape(nx, ny) if self.P["x"].size else xp.zeros((nx, ny))
        sparse = K & (cnt < max(self.cfg.pic_ppc // 4, 2))
        if self.P["x"].size:
            self._keep(~sparse[i, j])
        self.sample(sparse, Ui)
        self.n_resampled += int(xp.count_nonzero(sparse))
        self.K = K
        if not bool(K.any()):
            return
        # 4. conservation: buffer and resampled cells one by one, the other kinetic cells by connected region
        nb = xp.zeros_like(K)
        per = (s.cfg.get_bc_x() == "periodic", s.cfg.bc_y_type == "periodic")
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            nbr = xp.roll(K, sh, axis=ax)
            if not per[ax]:                          # outside a non-periodic edge counts as fluid
                edge = [slice(None), slice(None)]
                edge[ax] = 0 if sh == 1 else -1
                nbr[tuple(edge)] = False
            nb |= ~nbr
        single = K & (nb | sparse)
        bulk = K & ~single
        lab, nreg = ndimage.label(bulk)
        nreg = int(nreg)
        lab = lab.astype(xp.int64)
        ns = int(xp.count_nonzero(single))
        lab[single] = nreg + 1 + xp.arange(ns)
        self._match(lab, nreg + ns, Ui)
        # 5. deposit: particle moments replace the ion state of the kinetic cells
        P = self.P
        i, j = self._cell(P["x"], P["y"])
        c = i * ny + j
        n_c = nx * ny
        M = _bincount(c, P["w"], n_c).reshape(nx, ny) / self.dV
        mom = [_bincount(c, P["w"] * P[k], n_c).reshape(nx, ny) / self.dV for k in ("vx", "vy", "vz")]
        Ek = _bincount(c, 0.5 * P["w"] * (P["vx"]**2 + P["vy"]**2 + P["vz"]**2), n_c).reshape(nx, ny) / self.dV
        rho_old = xp.maximum(Ui[RHO], FLOOR_RHO)
        B2 = Ui[BX]**2 + Ui[BY]**2 + Ui[BZ]**2
        nK = int(xp.count_nonzero(K))
        for var, new in ((RHOC, M * Ui[RHOC] / rho_old), (RHOSE, M * Ui[RHOSE] / rho_old), (RHO, M),
                         (MX, mom[0]), (MY, mom[1]), (MZ, mom[2]), (EN, Ek + 0.5 * B2)):
            if var not in (RHOC, RHOSE):
                # the matched particle sums equal the U* totals only up to the round-off of the (unordered,
                # atomic) per-cell sums over many particles; put that residual back, spread over the kinetic
                # cells, using ordered device sums, so the domain totals stay at round-off over long runs
                new = new + (xp.sum(xp.where(K, Ui[var], 0.0)) - xp.sum(xp.where(K, new, 0.0))) / nK
            Ui[var] = xp.where(K, new, Ui[var])
        U[inner] = Ui
        s.apply_bc(U)

    def dt_limit(self):
        """gyro-period limit Omega_i dt <= 0.5 in kinetic cells (inf without a Lorentz force)"""
        if self.qm == 0.0 or not bool(self.K.any()):
            return np.inf
        ng, nx, ny = self.ng, self.nx, self.ny
        U = self.s.U[:, ng:ng + nx, ng:ng + ny]
        B = xp.sqrt(U[BX]**2 + U[BY]**2 + U[BZ]**2)
        bmax = float(xp.max(xp.where(self.K, B, 0.0)))
        return 0.5 / (self.qm * bmax) if bmax > 0 else np.inf

    def report(self):
        return dict(particles=int(self.P["x"].size), kinetic_cells=int(xp.count_nonzero(self.K)),
                    switched_on=self.n_switch_on, switched_off=self.n_switch_off, resampled=self.n_resampled)
