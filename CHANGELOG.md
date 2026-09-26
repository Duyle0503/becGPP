# Changelog

## 1.1.0 (2026-09-24)

### Fixed
- **Virial identity for the `-ln r` kernel.** The long-range term is `-G_C/2`
  (unit mass), not `p*E_g` with `p=0`; the old formula reported a virial residual
  of order 0.5 for correct `-ln r` ground states.
- **Unit conversion of the gravitational coupling.** `G_C = G m^2 N /(a_ho hbar w)`
  (one factor of N, per-particle energy, like `beta2 = g N`); 1.0.0 used `N^2`.
- **Sweeps at fast rotation** could stay on a vortex-free (metastable) branch,
  because each point was warm-started from the previous one only. A sweep point
  now tries continuation *and* `nseeds` fresh seeds and keeps the lowest energy
  (`branch`, `candidate_E` columns). Multi-seed selection is now variational
  (lowest energy wins); 1.0.0 preferred any converged candidate over a lower-energy
  one that had not yet reached `res_tol`.
- Exact self-cell average for the `-ln r` kernel (was a Gauss-Legendre estimate).
- Cache keys: `trap_coeff`, `l_z`, `nseeds`, `cg_beta` now enter the run id and the
  Thomas-Fermi cache (a changed `trap_coeff` could reuse a stale TF/checkpoint).
- Config keys `step`, `step_max`, `linesearch_max`, `precond_shift(_min)` are now
  actually used by the solver (defaults reproduce 1.0.0 exactly).

### Changed
- **Safe defaults**: `Omega=0`, `G_C=0`, `L=10`, `Ngrid=128` (1.0.0 defaulted to
  critical rotation with a long-range term on a 256-point grid).
- `w_LLL` is reported only for rotating 2D runs in the harmonic trap; new
  `lll_param = beta2*n_peak / 2(1-Omega)` (the LLL regime needs `lll_param << 1`).
- Validation gate: adds `-ln r` and quasi-2D Gaussian tests and the `-ln r` kernel
  order; 3D threshold tightened from 5e-2 to 5e-3.
- Triangular seeds are used only for rotating runs (`|Omega| >= 0.3`).
- Estimated peak memory is printed as a warning before oversized grids.
- Lists can be passed on the command line as JSON (`--sweep_values "[5,10]"`).
- `zip_output` defaults to False.

### Added
- **Quasi-2D kernel `q2d`**: the Newton kernel averaged over a frozen Gaussian
  axial profile of width `l_z`, `K_eff(u) = e^x K0(x)/(sqrt(2 pi) l_z)`,
  `x = u^2/4 l_z^2`; `dimension="quasi2D"` selects it when `l_z > 0` (thin-disc
  `1/r` when `l_z = 0`). Unit mapping `regime="selfgrav_pancake"`.
- `cg_beta` option: `pr` (default, as 1.0.0), `pr_precond`, `none`.
- Diagnostics: `stop_reason`, `seed_used`, `points_per_R90`, `virial_lr_term`.
- `benchmarks/`: Kaggle scripts that regenerate the paper's tables and figures, and
  `benchmarks/results_v1.1/`: the raw results (Tesla T4) quoted in the paper.
- Tests for kernel virials, kernel limits, safe defaults and unit scaling.
