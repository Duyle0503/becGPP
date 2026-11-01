# Sample output: CPU reference test run

Produced with becGPP 1.2.2 by `examples/cpu_test_run.py` on a CPU
(GitHub Actions `ubuntu-latest`, 3.12.15 / PyTorch 2.14.1+cpu; see `environment.txt`).
Total wall time 47.6 s, of which the validation gate takes most.

Reproduce and compare, from the repository root:

```bash
python -m pip install -e ".[yaml,test]"
python examples/cpu_test_run.py --check        # writes ./becgpp_test_run and compares
```

The check passes when the validation gate passes, energies and chemical potentials
agree to 1e-6 relative (1e-5 for the rotating case), R90 to 1e-3 and the vortex
number exactly. GPU runs differ in the last digits and in the random seed noise,
so the reference is meant for CPU runs.

## Contents

| file | content |
|---|---|
| `expected_results.json` | compact reference: environment, validation-gate errors, per-case configuration and results |
| `single_summary.csv` | one row per run with the full diagnostic set (energies, mu, Lz, Nv, w_LLL, radii, virial, TF comparison, stop reason) |
| `cpu_test_run_<rid>.json` | per-run record: complete configuration, every diagnostic, code version and device |
| `validation.csv` | analytic gate: relative errors of every kernel test, LLL identities, flat-top TF |
| `fig/` | density / phase / Thomas-Fermi overlays (PNG; the run also writes PDF) |
| `test_run.log` | console output of the run |

## Cases

Validation gate (`validate_N=256`): passed = True.

| case | configuration | E | mu | Nv | R90 | iterations | time (s) |
|---|---|---|---|---|---|---|---|
| `2D_harmonic` | dimension=2D, s=2, Omega=0.0, beta2=100.0, beta3=0.0, G_C=0.0, kernel=none, L=8.0, Ngrid=64 | 3.94594419 | 5.759754 | 0 | 2.8504 | 117 | 1.0 |
| `2D_rotating_lattice` | dimension=2D, s=2, Omega=0.8, beta2=100.0, beta3=0.0, G_C=0.0, kernel=none, L=9.0, Ngrid=96 | 3.19037110 | 4.349420 | 7 | 3.6213 | 2444 | 6.72 |
| `2D_selfgrav_newton` | dimension=2D, s=2, Omega=0.0, beta2=50.0, beta3=0.0, G_C=5.0, kernel=newton, L=8.0, Ngrid=64 | 0.95872008 | 0.633669 | 0 | 2.0616 | 97 | 1.04 |
| `2D_free_cubic_quintic` | dimension=2D, s=0, Omega=0.0, beta2=-50.0, beta3=50.0, G_C=0.0, kernel=none, L=3.0, Ngrid=64 | -5.66331850 | -7.648381 | 0 | 0.7656 | 134 | 0.93 |
| `quasi2D_q2d` | dimension=quasi2D, s=2, Omega=0.0, beta2=50.0, beta3=0.0, G_C=5.0, kernel=q2d, L=8.0, Ngrid=64, l_z=0.5 | 1.29993158 | 1.216793 | 0 | 2.2361 | 131 | 1.17 |
| `3D_harmonic` | dimension=3D, s=2, Omega=0.0, beta2=100.0, beta3=0.0, G_C=0.0, kernel=none, L=6.0, Ngrid=32 | 2.86792041 | 3.713216 | 0 | 2.5156 | 116 | 1.71 |
| `3D_schrodinger_newton` | dimension=3D, s=0, Omega=0.0, beta2=0.0, beta3=0.0, G_C=10.0, kernel=newton, L=3.0, Ngrid=32 | -5.31688552 | -15.761825 | 0 | 0.7012 | 166 | 3.12 |

All cases use `res_tol=1e-6`, one seed (`rng=0`) and the seed type listed in
`expected_results.json`. Together they exercise the 2D, 3D and quasi-2D grids,
the spectral kinetic and L_z operators (rotating lattice, triangular seed), the
Newton kernel in 2D and 3D, the finite-width quasi-2D kernel, the cubic and
quintic contact terms, the free (untrapped) and trapped Thomas-Fermi references
and the virial diagnostic.
