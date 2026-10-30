# fno_subgrid.py - a small Fourier neural operator (FNO, Li et al. 2021) run inside the time step as a learned
# predictor for an expensive implicit solve, with an exactness guard.
#
# Model: lift (1x1 linear) -> L Fourier layers, each GELU(spectral_conv(x) + W x + b) -> projection (two 1x1
# linear layers). The spectral convolution keeps the lowest `modes` Fourier modes in each direction
# (kx = -(m-1)..m-1, ky = 0..m-1) and multiplies them by learned complex channel-mixing weights. The truncated
# transforms are written as real matrix products with cosine / sine bases, so the same code runs on CuPy or
# NumPy, is resolution independent (the bases are built per grid) and is differentiated by aegis/autodiff.py's
# tape (training uses the reverse mode of Feature 1, no external ML framework).
#
# Use in the solver: radiation_multigroup.step asks the model for the implicit solution E^{n+1} of every group
# before its Krylov solve (cfg.fld_fno_model). The guard is the residual of that linear system:
#   - relative residual <= cfg.fld_fno_tol: the prediction is accepted and BiCGStab is skipped (in the first
#     Picard pass; later passes, which update the flux limiter, solve as usual starting from it);
#   - otherwise BiCGStab runs, starting from the prediction if its residual is below half that of E^n, else from
#     E^n as without the predictor (a residual is only a rough guide to the iteration count).
# Either way the step finishes with the conservative face-flux + exchange update built from the solution, so gas +
# radiation energy is conserved to round-off whatever the network predicts; the tolerance bounds the error of the
# radiation field itself. Training data come from the exact solver (tools/train_fno_radiation.py).
import json

import numpy as np

from . import autodiff as ad


def _bases(n, m, xp, full):
    """cos / sin of 2 pi k i / n for the kept wavenumbers (full: -(m-1)..m-1, else 0..m-1), shape (K, n)"""
    k = np.arange(-(m - 1), m) if full else np.arange(m)
    ang = 2 * np.pi * np.outer(k, np.arange(n)) / n
    return xp.asarray(np.cos(ang)), xp.asarray(np.sin(ang)), k


class FNO:
    def __init__(self, cin, cout, width=16, modes=8, layers=3, hidden=32, seed=0, xp=np):
        self.cin, self.cout, self.width, self.modes, self.layers, self.hidden = cin, cout, width, modes, layers, hidden
        self.xp = xp
        rng = np.random.default_rng(seed)
        kx, ky = 2 * modes - 1, modes
        sc = 1.0 / (width * width)
        P = dict(lift_w=rng.normal(0, 1 / np.sqrt(cin), (cin, width)), lift_b=np.zeros(width),
                 proj1_w=rng.normal(0, 1 / np.sqrt(width), (width, hidden)), proj1_b=np.zeros(hidden),
                 proj2_w=rng.normal(0, 1 / np.sqrt(hidden), (hidden, cout)) * 0.1, proj2_b=np.zeros(cout))
        for l in range(layers):
            P[f"sr{l}"] = sc * rng.random((width, width, kx, ky))
            P[f"si{l}"] = sc * rng.random((width, width, kx, ky))
            P[f"pw{l}"] = rng.normal(0, 1 / np.sqrt(width), (width, width))
            P[f"pb{l}"] = np.zeros(width)
        self.params = {k: xp.asarray(v) for k, v in P.items()}
        self.norm = dict(in_mu=np.zeros(cin), in_sd=np.ones(cin), out_mu=np.zeros(cout), out_sd=np.ones(cout))
        self.meta = {}
        self._bcache = {}

    def _b(self, nx, ny):
        key = (nx, ny)
        if key not in self._bcache:
            Cx, Sx, _ = _bases(nx, self.modes, self.xp, True)
            Cy, Sy, ky = _bases(ny, self.modes, self.xp, False)
            c = self.xp.asarray(np.where(ky == 0, 1.0, 2.0) / (nx * ny))     # Hermitian completion in ky
            self._bcache[key] = (Cx, Sx, Cy, Sy, c)
        return self._bcache[key]

    def _spectral(self, x, wr, wi):
        """x: (B, C, nx, ny) -> (B, O, nx, ny), lowest modes mixed by the complex weights wr + i wi"""
        E = ad.einsum
        Cx, Sx, Cy, Sy, c = self._b(x.shape[2], x.shape[3])
        xc, xs = E("bcij,ki->bckj", x, Cx), E("bcij,ki->bckj", x, Sx)
        Xr = E("bckj,lj->bckl", xc, Cy) - E("bckj,lj->bckl", xs, Sy)
        Xi = -(E("bckj,lj->bckl", xs, Cy) + E("bckj,lj->bckl", xc, Sy))
        Yr = E("bckl,cokl->bokl", Xr, wr) - E("bckl,cokl->bokl", Xi, wi)
        Yi = E("bckl,cokl->bokl", Xi, wr) + E("bckl,cokl->bokl", Xr, wi)
        Yr, Yi = Yr * c, Yi * c
        # y = Re sum Y e^{+i(a + b)}: cos(a+b) = CxCy - SxSy, sin(a+b) = SxCy + CxSy
        ar, ai = E("bokl,ki->boil", Yr, Cx), E("bokl,ki->boil", Yi, Cx)
        br, bi = E("bokl,ki->boil", Yr, Sx), E("bokl,ki->boil", Yi, Sx)
        return (E("boil,lj->boij", ar, Cy) - E("boil,lj->boij", br, Sy)
                - E("boil,lj->boij", bi, Cy) - E("boil,lj->boij", ai, Sy))

    def forward(self, x, P=None):
        """x: (B, cin, nx, ny) normalised features (Tensor or array) -> normalised outputs (B, cout, nx, ny)"""
        P = P or {k: ad.const(v) for k, v in self.params.items()}
        E = ad.einsum
        h = E("bcij,cd->bdij", x, P["lift_w"]) + ad.reshape(P["lift_b"], (1, -1, 1, 1))
        for l in range(self.layers):
            h = ad.gelu(self._spectral(h, P[f"sr{l}"], P[f"si{l}"]) + E("bcij,cd->bdij", h, P[f"pw{l}"])
                        + ad.reshape(P[f"pb{l}"], (1, -1, 1, 1)))
        h = ad.gelu(E("bcij,cd->bdij", h, P["proj1_w"]) + ad.reshape(P["proj1_b"], (1, -1, 1, 1)))
        return E("bcij,cd->bdij", h, P["proj2_w"]) + ad.reshape(P["proj2_b"], (1, -1, 1, 1))

    def predict(self, feats):
        """raw features (B, cin, nx, ny) -> raw outputs (normalisation applied both ways)"""
        xp = self.xp
        mu, sd = (xp.asarray(self.norm[k]).reshape(1, -1, 1, 1) for k in ("in_mu", "in_sd"))
        y = self.forward(ad.const((feats - mu) / sd)).v
        mo, so = (xp.asarray(self.norm[k]).reshape(1, -1, 1, 1) for k in ("out_mu", "out_sd"))
        return y * so + mo

    def fit_normalisation(self, X, Y):
        to = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
        X, Y = to(X), to(Y)
        self.norm = dict(in_mu=X.mean(axis=(0, 2, 3)), in_sd=X.std(axis=(0, 2, 3)) + 1e-8,
                         out_mu=Y.mean(axis=(0, 2, 3)), out_sd=Y.std(axis=(0, 2, 3)) + 1e-8)

    def train(self, X, Y, epochs=200, batch=8, lr=3e-3, seed=0, log=None, Xv=None, Yv=None):
        """Adam on the mean squared error of the normalised outputs; returns the loss history"""
        xp = self.xp
        mu, sd = (xp.asarray(self.norm[k]).reshape(1, -1, 1, 1) for k in ("in_mu", "in_sd"))
        mo, so = (xp.asarray(self.norm[k]).reshape(1, -1, 1, 1) for k in ("out_mu", "out_sd"))
        Xn, Yn = (xp.asarray(X) - mu) / sd, (xp.asarray(Y) - mo) / so
        m = {k: xp.zeros_like(v) for k, v in self.params.items()}
        v2 = {k: xp.zeros_like(v) for k, v in self.params.items()}
        rng = np.random.default_rng(seed)
        n, it, hist = Xn.shape[0], 0, []
        for ep in range(epochs):
            perm = rng.permutation(n)
            tot = 0.0
            lr_ep = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))        # cosine schedule
            for s in range(0, n, batch):
                idx = xp.asarray(perm[s:s + batch])
                with ad.record():
                    P = {k: ad.leaf(v) for k, v in self.params.items()}
                    d = self.forward(ad.const(Xn[idx]), P) - ad.const(Yn[idx])
                    loss = ad.tsum(d * d) * (1.0 / d.v.size)
                    ad.backward(loss)
                it += 1
                for k, p in P.items():
                    g = p.g
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v2[k] = 0.999 * v2[k] + 0.001 * g * g
                    self.params[k] = self.params[k] - lr_ep * (m[k] / (1 - 0.9**it)) / (xp.sqrt(v2[k] / (1 - 0.999**it)) + 1e-8)
                tot += float(loss.v) * len(idx)
            rec = dict(epoch=ep, train=tot / n)
            if Xv is not None:
                rec["val"] = self.loss(Xv, Yv)
            hist.append(rec)
            if log and (ep % max(epochs // 10, 1) == 0 or ep == epochs - 1):
                log(rec)
        return hist

    def loss(self, X, Y):
        xp = self.xp
        mo, so = (xp.asarray(self.norm[k]).reshape(1, -1, 1, 1) for k in ("out_mu", "out_sd"))
        d = (self.predict(xp.asarray(X)) - xp.asarray(Y)) / so
        return float(xp.mean(d * d))

    # ---------------- persistence ----------------
    def save(self, path):
        to = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
        arch = dict(cin=self.cin, cout=self.cout, width=self.width, modes=self.modes, layers=self.layers,
                    hidden=self.hidden, meta=self.meta)
        np.savez(path, arch=json.dumps(arch), **{f"p_{k}": to(v) for k, v in self.params.items()},
                 **{f"n_{k}": np.asarray(v) for k, v in self.norm.items()})

    @classmethod
    def load(cls, path, xp=np):
        z = np.load(path, allow_pickle=False)
        arch = json.loads(str(z["arch"]))
        meta = arch.pop("meta", {})
        m = cls(xp=xp, **arch)
        m.params = {k[2:]: xp.asarray(z[k]) for k in z.files if k.startswith("p_")}
        m.norm = {k[2:]: z[k] for k in z.files if k.startswith("n_")}
        m.meta = meta
        return m


# ============================================================
# multigroup radiation: features, predictor, cache
# ============================================================
_TINY = 1e-30


def radiation_features(E0, theta, rho, B, dt, xp):
    """per-cell inputs of the radiation predictor: log E_g^n, log of the Planck group energies a T^4 b_g,
    log theta, log rho, log dt (G + G + 3 channels), shape (1, C, nx, ny)"""
    G = E0.shape[0]
    f = [xp.log(xp.maximum(E0, _TINY)), xp.log(xp.maximum(B, _TINY)), xp.log(xp.maximum(theta, _TINY))[None],
         xp.log(xp.maximum(rho, _TINY))[None], xp.full((1,) + theta.shape, np.log(max(dt, _TINY)))]
    return xp.concatenate(f, axis=0)[None]


def radiation_target(E0, E1, xp):
    """what the network learns: log(E^{n+1} / E^n) per group"""
    return (xp.log(xp.maximum(E1, _TINY)) - xp.log(xp.maximum(E0, _TINY)))[None]


_MODELS = {}


def radiation_guess(cfg, E0, theta, rho, B, dt):
    """the model's E^{n+1} (G, ...) for a 2D multigroup step (models are loaded once per path)"""
    from .physics import xp
    if cfg.fld_fno_model not in _MODELS:
        _MODELS[cfg.fld_fno_model] = FNO.load(cfg.fld_fno_model, xp=xp)
    m = _MODELS[cfg.fld_fno_model]
    if m.cout != E0.shape[0]:
        raise ValueError(f"{cfg.fld_fno_model} predicts {m.cout} groups, the run has {E0.shape[0]}")
    y = m.predict(radiation_features(E0, theta, rho, B, dt, xp))[0]
    return xp.maximum(E0, _TINY) * xp.exp(y)


class Stats:
    """counts of accepted predictions and Krylov iterations (reset per run)"""
    accepted = 0
    fallback = 0
    iterations = 0
    residuals = []
    default_residuals = []

    @classmethod
    def reset(cls):
        cls.accepted = cls.fallback = cls.iterations = 0
        cls.residuals = []
        cls.default_residuals = []
