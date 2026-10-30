"""Train the Fourier-neural-operator predictor for the multigroup radiation solve (aegis/fno_subgrid.py) and
benchmark it inside the solver step.

Training data come from the exact solver: random smooth temperature, density and radiation fields (log-normal
Fourier series) on a periodic grid, a few implicit multigroup steps each (Kramers group opacities with an edge),
recording E^n, theta, rho, the Planck group energies, dt and the converged E^{n+1}. The network learns
log(E^{n+1} / E^n) per group.

The benchmark runs held-out states through radiation_multigroup.step three ways: exact (BiCGStab from E^n),
predictor with fallback (fld_fno_tol), and predictor as warm start only (fld_fno_tol = 0+), and reports the
accepted fraction, Krylov iterations, wall time per step, the error against the exact solution and the energy
conservation of each.

    python tools/train_fno_radiation.py                    # writes results/fno/radiation_fno.npz + report
    python tools/train_fno_radiation.py --quick
"""
import os
import sys
import json
import time
import argparse
import contextlib
import io

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config, NVAR, EN, iRHO, iPR
    from aegis.physics import xp, prim_to_cons, cons_to_prim
from aegis import radiation_multigroup as mg, fno_subgrid as fs


def config(a, **kw):
    c = dict(nx=a.n, ny=a.n, x_max=1.0, y_max=1.0, gamma=5 / 3, bc_x_type="periodic", bc_y_type="periodic",
             diag_interval=10**9, snapshot_times=[], fld=True, fld_c=10.0, fld_a=1.0, fld_advect=False,
             fld_tol=1e-8, fld_groups=a.groups, fld_mg_opacity="kramers", fld_kp0=20.0, fld_kr0=2.0, fld_edge=3.0,
             fld_edge_jump=10.0, fld_group_emin=0.1, fld_group_emax=20.0)
    c.update(kw)
    return Config(**c)


def smooth_field(rng, n, kmax, amp):
    x = np.arange(n) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    f = np.zeros((n, n))
    for kx in range(-kmax, kmax + 1):
        for ky in range(0, kmax + 1):
            if kx == 0 and ky == 0:
                continue
            a = rng.normal() / (kx * kx + ky * ky)
            f += a * np.cos(2 * np.pi * (kx * X + ky * Y) + rng.uniform(0, 2 * np.pi))
    return amp * f / (np.abs(f).max() + 1e-12)


def random_state(rng, a, cfg):
    n = a.n
    theta = np.exp(np.log(rng.uniform(0.5, 3.0)) + smooth_field(rng, n, 4, rng.uniform(0.2, 1.0)))
    rho = np.exp(smooth_field(rng, n, 3, rng.uniform(0.0, 0.7)))
    W = xp.zeros((NVAR, n, n))
    W[iRHO] = xp.asarray(rho)
    W[iPR] = xp.asarray(rho * theta)
    U = prim_to_cons(W, 5 / 3)
    trad = np.exp(np.log(rng.uniform(0.3, 3.0)) + smooth_field(rng, n, 4, rng.uniform(0.2, 1.0)))
    B, _ = mg.planck_groups(mg.edges_of(cfg), xp.asarray(trad), 1.0)
    E = B * xp.asarray(np.exp(smooth_field(rng, n, 2, 0.3)))[None]
    dt = float(np.exp(rng.uniform(np.log(a.dt_min), np.log(a.dt_max))))
    return U, E.copy(), dt


def generate(a, cfg, n_states, seed):
    rng = np.random.default_rng(seed)
    mg.RECORD = []
    h, per = (1 / a.n, 1 / a.n), (True, True)
    for s in range(n_states):
        U, E, dt = random_state(rng, a, cfg)
        for _ in range(a.steps_per_state):
            mg.step(E, U, 5 / 3, cfg, h, per, dt, advect=False)
    rec, mg.RECORD = mg.RECORD, None
    X = xp.concatenate([fs.radiation_features(E0, th, rho, B, dt, xp) for E0, th, rho, B, dt, _ in rec])
    Y = xp.concatenate([fs.radiation_target(E0, E1, xp) for E0, th, rho, B, dt, E1 in rec])
    return X, Y


def benchmark(a, cfg, model_path, seed):
    rng = np.random.default_rng(seed)
    h, per = (1 / a.n, 1 / a.n), (True, True)
    modes = {"exact": config(a), "predictor (accept <= tol, else warm start)": config(a, fld_fno_model=model_path,
                                                                                       fld_fno_tol=a.tol),
             "warm start only": config(a, fld_fno_model=model_path, fld_fno_tol=1e-300)}
    out = {}
    states = [random_state(rng, a, cfg) for _ in range(a.bench_states)]
    import cupy as cp
    for name, c in modes.items():
        fs.Stats.reset()
        its, t_tot, err, cons = 0, 0.0, [], []
        for U0, E0, dt in states:
            U, E = U0.copy(), E0.copy()
            Ux, Ex = U0.copy(), E0.copy()
            tot0 = float(xp.sum(U[EN]) + xp.sum(E))
            for _ in range(a.steps_per_state):
                mg.step(Ex, Ux, 5 / 3, modes["exact"], h, per, dt, advect=False)   # reference
                cp.cuda.Device().synchronize()
                t0 = time.perf_counter()
                _, _, it = mg.step(E, U, 5 / 3, c, h, per, dt, advect=False)
                cp.cuda.Device().synchronize()
                t_tot += time.perf_counter() - t0
                its += it
                err.append(float(xp.max(xp.abs(E - Ex)) / xp.max(xp.abs(Ex))))
            cons.append(abs(float(xp.sum(U[EN]) + xp.sum(E)) - tot0) / tot0)
        nstep = a.bench_states * a.steps_per_state
        out[name] = dict(iterations_per_step=its / nstep, ms_per_step=1e3 * t_tot / nstep,
                         max_rel_error_vs_exact=max(err), energy_conservation=max(cons),
                         accepted=fs.Stats.accepted, fallback=fs.Stats.fallback,
                         median_predictor_residual=float(np.median(fs.Stats.residuals)) if fs.Stats.residuals else None,
                         median_start_residual=float(np.median(fs.Stats.default_residuals)) if fs.Stats.default_residuals else None)
    # the predictor alone vs one BiCGStab solve
    m = fs.FNO.load(model_path, xp=xp)
    U0, E0, dt = states[0]
    W = cons_to_prim(U0, 5 / 3)
    th = W[iPR] / W[iRHO]
    B, _ = mg.planck_groups(mg.edges_of(cfg), th, cfg.fld_a)
    f = fs.radiation_features(E0, th, W[iRHO], B, dt, xp)
    m.predict(f)
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        m.predict(f)
    cp.cuda.Device().synchronize()
    out["inference_ms"] = 1e3 * (time.perf_counter() - t0) / 20
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--groups", type=int, default=6)
    ap.add_argument("--states", type=int, default=60)
    ap.add_argument("--steps-per-state", type=int, default=3)
    ap.add_argument("--dt-min", type=float, default=2e-3)
    ap.add_argument("--dt-max", type=float, default=2e-2)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--modes", type=int, default=10)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--tol", type=float, default=1e-3, help="acceptance tolerance (relative residual)")
    ap.add_argument("--bench-states", type=int, default=4)
    ap.add_argument("--bench-only", action="store_true", help="benchmark the saved model, no training")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "fno"))
    a = ap.parse_args()
    if a.quick:
        a.states, a.epochs, a.bench_states, a.width, a.modes, a.layers = 16, 10, 2, 8, 6, 2
    os.makedirs(a.out, exist_ok=True)
    cfg = config(a)
    path = os.path.join(a.out, "radiation_fno.npz")
    if a.bench_only:
        bench = benchmark(a, cfg, path, seed=3)
        for k, v in bench.items():
            print(f"  {k}: {v}")
        rep_path = os.path.join(a.out, "radiation_fno_report.json")
        rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
        rep["benchmark"] = bench
        with open(rep_path, "w") as f:
            json.dump(rep, f, indent=1)
        return
    t0 = time.time()
    X, Y = generate(a, cfg, a.states, seed=1)
    Xv, Yv = generate(a, cfg, max(a.states // 6, 2), seed=2)
    print(f"data: {X.shape[0]} train / {Xv.shape[0]} validation samples from the exact solver ({time.time() - t0:.0f}s)")
    m = fs.FNO(X.shape[1], Y.shape[1], width=a.width, modes=a.modes, layers=a.layers, xp=xp)
    m.fit_normalisation(X, Y)
    t0 = time.time()
    hist = m.train(X, Y, epochs=a.epochs, Xv=Xv, Yv=Yv, log=lambda r: print(f"  epoch {r['epoch']:4d}: train "
                                                                            f"{r['train']:.3e}  val {r['val']:.3e}"))
    print(f"training: {time.time() - t0:.0f}s")
    m.meta = dict(groups=a.groups, n=a.n, opacity="kramers", kp0=20.0, kr0=2.0, edge=3.0, edge_jump=10.0,
                  dt_range=[a.dt_min, a.dt_max], note="trained for config() in tools/train_fno_radiation.py")
    m.save(path)
    bench = benchmark(a, cfg, path, seed=3)
    for k, v in bench.items():
        print(f"  {k}: {v}")
    with open(os.path.join(a.out, "radiation_fno_report.json"), "w") as f:
        json.dump(dict(args=vars(a), history=hist, benchmark=bench), f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
