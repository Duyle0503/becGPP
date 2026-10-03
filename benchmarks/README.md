# Benchmarks

Scripts that regenerate every table and figure of the becGPP paper. Each script
is self-contained so that it can be pasted into a single Kaggle (or Colab) cell
with a GPU: it installs the tagged becGPP release it was run with, records the
hardware, and zips its outputs.

| script | release | paper | outputs (`results/<name>/`) |
|---|---|---|---|
| `verification.py` | v1.1.0 | Secs. 5, 7: Tables validation, pcg, bench, cpugpu; Figs. convergence, performance, sweeps | `validation.csv`, `kernels_log_q2d.csv`, `sweep_*.csv`, `cq_ladder.csv`, `pcg_variants.csv`, `perf_*.csv` |
| `selfgravity_limits.py` | v1.1.0 | Sec. 5: Table sn, Fig. mass–radius | `ref_schrodinger_newton.csv`, `ref_mass_radius.csv`, `fig_mass_radius.pdf` |
| `itp_agreement.py` | v1.1.0 | Sec. 5: column δE of Table itp | `itp_vs_pcg.csv` |
| `itp_timing.py` | v1.2.0 | Sec. 7: time columns of Table itp, Fig. itp | `timing.csv`, `traces.csv` |
| `make_paper_numbers.py` | — | prints the LaTeX rows and draws the figures from `results/` | — |

```bash
cd benchmarks
python make_paper_numbers.py --figdir figs
```

## Notes on the recorded results

* All runs: Kaggle notebook, NVIDIA Tesla T4 (16 GB), PyTorch 2.10, CUDA 12.8,
  double precision; the exact environment is in `results/*/hardware.json`.
  The T4 executes double precision at 1/32 of its single-precision rate, so the
  wall times are upper bounds for GPUs with full double-precision throughput.
* The v1.1 scripts pin `cg_beta="pr"` (the default in 1.0–1.1); from 1.2.0 the
  default is `"pr_precond"`. Ground states are the same for both.
* `results/selfgravity_limits/` is the second run of that script, after the box
  was sized from an interpolated R99 estimate instead of the Thomas–Fermi radius
  (in the first run the beta2 = 32, 128, 512 points were box-limited).
* `lll_param` in `results/verification/sweep_*.csv` was written by 1.1.0 with the
  gap `2(1-Omega)`; the paper quotes `beta2*peak/(1+Omega)` (definition since
  1.1.1), recomputed from the `peak` column.
* In `results/itp_timing/` the becGPP runs of the lattice and the self-gravitating
  cases end at `maxit` with `iters`=19999: with 1.2.0 the solver kept retrying
  failed line searches once the energy had settled (fixed in 1.2.1). The times
  to each accuracy are read from `traces.csv` and are unaffected.
