"""Run the high-Mach RMI benchmark cases defined in a JSON config.

    python scripts/run_rmi_mhd.py                       # configs/rmi_mhd.json, full resolution
    python scripts/run_rmi_mhd.py --quick               # low-resolution check (config "quick" block)
    python scripts/run_rmi_mhd.py --cases mhd_parallel  # a subset
Outputs go to <out_dir>/<case>/ (results/rmi_mhd/ by default); see aegis/problems/rmi_mhd.py.
"""
import os
import sys
import json
import argparse
import contextlib
import io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.problems.rmi_mhd import RMIParams, run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "rmi_mhd.json"))
    ap.add_argument("--quick", action="store_true", help="apply the config's 'quick' overrides")
    ap.add_argument("--cases", nargs="*", help="case names to run (default: all)")
    ap.add_argument("--out", help="output directory (default: config out_dir)")
    args = ap.parse_args(argv)

    with open(args.config) as fh:
        cfg = json.load(fh)
    out_root = args.out or os.path.join(ROOT, cfg.get("out_dir", "results/rmi_mhd"))
    if args.quick and not args.out:
        out_root = out_root.rstrip("/\\") + "_quick"
    names = args.cases or list(cfg["cases"])
    for name in names:
        params = {**cfg["common"], **(cfg.get("quick", {}) if args.quick else {}), **cfg["cases"][name], "name": name}
        print(f"=== {name}: {params}", flush=True)
        run(RMIParams(**params), os.path.join(out_root, name))
    print(f"outputs in {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
