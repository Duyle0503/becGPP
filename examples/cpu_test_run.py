"""Reference test run for becGPP on a CPU (a few minutes, no GPU needed).

It exercises every component of the program on small grids -- the analytic
validation gate, the 2D and 3D spectral operators, rotation with a triangular
seed, the three free-space kernels (Newton 2D/3D, quasi-2D) and the local
contact/quintic terms -- and writes the usual outputs (CSV, per-run JSON,
figures) to an output directory, plus a compact ``results.json``.

    python examples/cpu_test_run.py                       # run, write ./becgpp_test_run
    python examples/cpu_test_run.py --check               # also compare with the shipped reference
    python examples/cpu_test_run.py --out DIR --check REF.json

The shipped reference ``data/sample_output/cpu_test_run/expected_results.json``
was produced with this script on a CPU by the version named inside it. The
check compares energies and chemical potentials to a relative tolerance of
1e-6 (1e-5 for the rotating case, whose vortex arrangement is sensitive to
floating-point details), radii to 1e-3 and the vortex number exactly. Results
on a GPU differ in the last digits and in the random seed noise, so the check
is meant for CPU runs.
"""
import argparse
import json
import os
import platform
import sys
import time

import torch

import becgpp
from becgpp import default_cfg, paths, run
from becgpp.constants import CODE_VERSION

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REF = os.path.join(HERE, "..", "data", "sample_output", "cpu_test_run",
                           "expected_results.json")

COMMON = dict(mode="single", show_inline=False, save_ckpt=False, save_figs=True,
              res_tol=1e-6, maxit=6000, nseeds=1, rng=0, tag="cpu_test_run")

CASES = [
    ("2D_harmonic", dict(dimension="2D", s=2, Omega=0.0, beta2=100.0, beta3=0.0,
                         G_C=0.0, kernel="none", L=8.0, Ngrid=64, seed="tf")),
    ("2D_rotating_lattice", dict(dimension="2D", s=2, Omega=0.8, beta2=100.0, beta3=0.0,
                                 G_C=0.0, kernel="none", L=9.0, Ngrid=96, seed="triangular")),
    ("2D_selfgrav_newton", dict(dimension="2D", s=2, Omega=0.0, beta2=50.0, beta3=0.0,
                                G_C=5.0, kernel="newton", L=8.0, Ngrid=64, seed="tf")),
    ("2D_free_cubic_quintic", dict(dimension="2D", s=0, Omega=0.0, beta2=-50.0, beta3=50.0,
                                   G_C=0.0, kernel="none", L=3.0, Ngrid=64, seed="tf")),
    ("quasi2D_q2d", dict(dimension="quasi2D", s=2, Omega=0.0, beta2=50.0, beta3=0.0,
                         G_C=5.0, kernel="q2d", l_z=0.5, L=8.0, Ngrid=64, seed="tf")),
    ("3D_harmonic", dict(dimension="3D", s=2, Omega=0.0, beta2=100.0, beta3=0.0,
                         G_C=0.0, kernel="none", L=6.0, Ngrid=32, seed="tf")),
    ("3D_schrodinger_newton", dict(dimension="3D", s=0, Omega=0.0, beta2=0.0, beta3=0.0,
                                   G_C=10.0, kernel="newton", L=3.0, Ngrid=32,
                                   seed="gaussian")),
]

KEYS = ("E", "mu", "Lz", "Nv", "R90", "virial_rel", "resid_rel", "converged",
        "stop_reason", "iters", "tf_kind", "tf_R90")


def _clean(v):
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def run_all(out):
    paths.configure(out)
    t0 = time.time()
    gate = run(default_cfg(mode="validate", validate_N=256, show_inline=False))
    results = dict(
        code_version=CODE_VERSION, package_version=becgpp.__version__
        if hasattr(becgpp, "__version__") else CODE_VERSION,
        torch=torch.__version__, python=platform.python_version(),
        platform=platform.platform(), device=str(becgpp.DEV),
        validation=dict(passed=bool(gate["passed"]),
                        **{k: _clean(v) for k, v in gate.items() if k != "passed"}),
        cases={})
    for name, over in CASES:
        t1 = time.time()
        d = run(default_cfg(**dict(COMMON, **over)))
        rec = {k: _clean(d.get(k)) for k in KEYS}
        rec["walltime_s"] = round(time.time() - t1, 2)
        results["cases"][name] = dict(config=over, result=rec)
    results["total_walltime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def _rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def check(results, ref):
    rows, ok = [], bool(results["validation"]["passed"])
    rows.append(("validation gate", "passed", results["validation"]["passed"], True, ok))
    for name, exp in ref["cases"].items():
        got = results["cases"].get(name)
        if got is None:
            rows.append((name, "present", False, True, False)); ok = False
            continue
        g, e = got["result"], exp["result"]
        tol = 1e-5 if "rotating" in name else 1e-6
        for key, good in (("E", _rel(g["E"], e["E"]) <= tol),
                          ("mu", _rel(g["mu"], e["mu"]) <= tol),
                          ("R90", abs(g["R90"] - e["R90"]) <= 1e-3),
                          ("Nv", int(g["Nv"]) == int(e["Nv"])),
                          ("converged", bool(g["converged"]) == bool(e["converged"]))):
            rows.append((name, key, g[key], e[key], good))
            ok = ok and good
    width = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"{r[0]:<{width}}  {r[1]:<9} got={r[2]!s:<22} expected={r[3]!s:<22} "
              f"{'ok' if r[4] else 'MISMATCH'}")
    print(f"\nCHECK {'PASSED' if ok else 'FAILED'} against reference "
          f"{ref.get('code_version')} ({ref.get('platform')})")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="./becgpp_test_run", help="output directory")
    p.add_argument("--check", nargs="?", const=DEFAULT_REF, default=None,
                   help="compare with a reference results.json (default: the shipped one)")
    a = p.parse_args(argv)
    results = run_all(a.out)
    print(f"\nWrote {os.path.join(a.out, 'results.json')} "
          f"({results['total_walltime_s']} s on {results['device']})")
    if a.check:
        with open(a.check) as f:
            ref = json.load(f)
        return 0 if check(results, ref) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
