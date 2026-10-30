"""Parameter sweeps A-H of the sieve study.

    python experiments/sweeps.py C        # one sweep
    python experiments/sweeps.py all      # everything (~2.5 h on one GPU)

Results go to experiments/data/exp_<tag>.npz. All runs use runner.BASE unless overridden.
"""
import sys
import math

from runner import run_sweep

HYDRO = {"hydro": {"B_transverse": 0.0, "B_field_type": "uniform"}}
UNIFORM_BX = 0.99  # gap fraction that makes the x field effectively uniform

SWEEPS = {
    # A: small ripple, long domain, field-free-gap stripes (Bx = B0 cos k y)
    "A": ({"B_field_type": "striped"}, {
        **HYDRO,
        "Bx cos B0=3": {"B_transverse": 3.0},
        "Bx cos B0=6": {"B_transverse": 6.0},
    }),
    # B: uniform-temperature closure (rejected: post-shock pressure imbalance) + uniform-field control
    "B": ({"B_field_type": "striped", "equilibrium_closure": "uniform_temperature"}, {
        **HYDRO,
        "UT cos B0=3": {"B_transverse": 3.0},
        "UT cos B0=6": {"B_transverse": 6.0},
        "uniform Bx=3": {"B_transverse": 3.0, "B_gap_fraction": UNIFORM_BX},
    }),
    # C: gap-fraction sweep
    "C": ({"B_field_type": "striped"}, {
        **HYDRO,
        **{f"B0={b:g} eps={e}": {"B_transverse": b, "B_gap_fraction": e}
           for b in (4.0, 6.0) for e in (0.3, 0.5, 0.7)},
        "uniform Bx=3": {"B_transverse": 3.0, "B_gap_fraction": UNIFORM_BX},
    }),
    # D: lock threshold (uniform) and weak modulation
    "D": ({"B_field_type": "striped"}, {
        **HYDRO,
        "uniform Bx=1": {"B_transverse": 1.0, "B_gap_fraction": UNIFORM_BX},
        "uniform Bx=2": {"B_transverse": 2.0, "B_gap_fraction": UNIFORM_BX},
        "B0=3 eps=0.9": {"B_transverse": 3.0, "B_gap_fraction": 0.9},
        "B0=3.5 eps=0.9": {"B_transverse": 3.5, "B_gap_fraction": 0.9},
        "B0=3.5 eps=0.8": {"B_transverse": 3.5, "B_gap_fraction": 0.8},
        "B0=4 eps=0.9": {"B_transverse": 4.0, "B_gap_fraction": 0.9},
    }),
    # E: k_mod/k_int and phase, deep stripes (B0=6, eps=0.3)
    "E": ({"B_field_type": "striped", "B_transverse": 6.0, "B_gap_fraction": 0.3}, {
        **HYDRO,
        **{f"r=0.5 phi={j}pi/8": {"B_modulation_mode": 2, "B_stripe_phase": j * math.pi / 8} for j in range(8)},
        **{f"r={r} phi={j}pi/8": {"B_modulation_mode": m, "B_stripe_phase": j * math.pi / 8}
           for m, r in ((4, "1"), (8, "2")) for j in (0, 2)},
    }),
    # F: does phase move the lock boundary? r = 0.5, 20% and 30% modulation
    "F": ({"B_field_type": "striped", "B_modulation_mode": 2}, {
        **HYDRO,
        "20% phi=pi/4": {"B_transverse": 3.5, "B_gap_fraction": 0.8, "B_stripe_phase": math.pi / 4},
        "20% phi=3pi/4": {"B_transverse": 3.5, "B_gap_fraction": 0.8, "B_stripe_phase": 3 * math.pi / 4},
        "30% phi=pi/4": {"B_transverse": 4.0, "B_gap_fraction": 0.7, "B_stripe_phase": math.pi / 4},
        "30% phi=3pi/4": {"B_transverse": 4.0, "B_gap_fraction": 0.7, "B_stripe_phase": 3 * math.pi / 4},
    }),
    # G: resolution check of the r = 2 case (ny 100 -> 200)
    "G": ({"ny": 200, "B_field_type": "striped", "B_transverse": 6.0, "B_gap_fraction": 0.3,
           "B_modulation_mode": 8}, {
        **HYDRO,
        "r=2 ny=200": {},
    }),
    # H: paper figure set (spectrum, enstrophy location, energy budget time series)
    "H": ({"B_field_type": "striped"}, {
        **HYDRO,
        "uniform Bx=3": {"B_transverse": 3.0, "B_gap_fraction": UNIFORM_BX},
        "B0=3.5 eps=0.9": {"B_transverse": 3.5, "B_gap_fraction": 0.9},
        "B0=6 eps=0.3": {"B_transverse": 6.0, "B_gap_fraction": 0.3},
        "cos B0=6": {"B_transverse": 6.0},
    }),
}

if __name__ == "__main__":
    tags = list(SWEEPS) if sys.argv[1:] == ["all"] else sys.argv[1:]
    if not tags or any(t not in SWEEPS for t in tags):
        sys.exit(f"usage: python experiments/sweeps.py <{'|'.join(SWEEPS)}|all>")
    for tag in tags:
        common, cases = SWEEPS[tag]
        run_sweep(tag, common, cases)
