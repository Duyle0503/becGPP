# becGPP

Version **1.2.2**, the version described in the accompanying *Computer Physics
Communications* article. Install it from this directory (see [Install](#install)).

**A GPU-accelerated spectral ground-state solver for trapped, rotating and self-gravitating Bose–Einstein condensates in 2D, 3D and quasi-2D.**

`becGPP` computes the ground state of the general Gross–Pitaevskii–Poisson (GPP)
energy functional. Dimension, trap power, contact and three-body couplings,
rotation, and the long-range kernel are all independent run-time parameters — the
same code path serves an ordinary trapped condensate, a rotating vortex lattice,
a self-bound cubic–quintic droplet, and a Newtonian boson star. Nothing about a
particular physical regime is hard-coded.

The energy functional (trap units ℏ = m = ω⊥ = 1, ∫|ψ|² = 1, d = 2 or 3):

```
E[ψ] = ∫ [ ½|∇ψ|² + c r^s |ψ|² − Ω ψ* L_z ψ ] dr      (c = trap_coeff, default ½)
     + (β₂/2) ∫|ψ|⁴ dr        (two-body contact)
     + (β₃/3) ∫|ψ|⁶ dr        (three-body / quintic)
     − (G_C/2) ∬ ρ(r) K(|r−r′|) ρ(r′) dr dr′   (long range)
```

with kernel `K = 1/r` (Newton, 2D/3D), `K = −ln r` (Poisson, 2D), the quasi-2D
Newton kernel of a pancake with axial width `l_z`
(`K_eff(u) = e^{u²/4l_z²} K₀(u²/4l_z²)/(√(2π) l_z)`, → `1/u` as `l_z → 0`), or none.

## Method (one paragraph)

Ground states are found by a **normalized preconditioned conjugate-gradient**
minimization on the unit-norm manifold: an **adaptive Sobolev preconditioner**
(shift = median bulk potential), **Polak–Ribière CG** with periodic restart, a
**Barzilai–Borwein** trial step, and a backtracking **Armijo** line search. The
kinetic and L_z operators are **spectral (FFT)**; the long-range potential is a
**zero-padded free-space convolution** with a Gauss–Legendre cell-averaged kernel
and an exact self-cell average. A **Thomas–Fermi** reference is extracted in every
mode, and an **analytic validation gate** (Coulomb integrals + Landau-level
identities, all kernels, second-order convergence) certifies the framework. See the accompanying paper for details.

## Install

From the program archive (the version described in the paper), inside this directory:

```bash
python -m pip install -e .              # runtime deps: torch>=2.0, numpy, matplotlib
python -m pip install -e ".[yaml,test]" # + PyYAML configs, pytest and SciPy
```

From GitHub, check out the same release tag rather than `main`:

```bash
git clone https://github.com/Duyle0503/becGPP.git
cd becGPP
git checkout v1.2.2                     # the release described in the paper
python -m pip install -e ".[yaml,test]"
```

Install a PyTorch build matching your CPU/CUDA first if `pip` would otherwise pick
an incompatible wheel.

A CUDA-capable GPU is used automatically when available; the solver falls back to
the CPU otherwise. Double precision is used throughout (required: the
gravitational coupling can be far weaker than the contact term).

**Regime of validity.** A ground state exists only if the energy is bounded below.
With rotation this needs confinement: `Omega = 0`, a trap with `s > 2`, or `s = 2`
with `Omega**2 <= 2*trap_coeff` (|Omega| <= 1 in the default harmonic trap).
`default_cfg` emits a `RuntimeWarning` outside this range; a finite box then hides,
but does not remove, the instability.

## Quick start

**Python API**

```python
from becgpp import default_cfg, run

# Rotating 2D condensate -> Abrikosov vortex lattice
cfg = default_cfg(mode="single", dimension="2D", s=2, Omega=0.9,
                  beta2=200, beta3=0, G_C=0, kernel="none",
                  seed="triangular", nseeds=3, L=12, Ngrid=384)
diag = run(cfg)
print(diag["E"], diag["Nv"], diag["w_LLL"])
```

**Command line**

```bash
becgpp --config examples/config_2D_rotating_lattice.yaml --outdir ./out
becgpp --mode single --dimension 3D --G_C 20 --kernel newton --outdir ./out
python -m becgpp --config examples/config_2D_harmonic.yaml
```

**Build a dataset**

Assemble a figure set by scripting `single` and `sweep` over your own list of
configurations — nothing is hard-coded. For example, to reproduce the showcase
states, loop over their configs and call `run(default_cfg(mode="single", ...))`
for each.

## Modes

| mode          | what it does |
|---------------|--------------|
| `validate`    | analytic gates: 1/r, −ln r and quasi-2D kernels vs Gaussians, kernel orders, LLL identities |
| `smoke`       | validate + a tiny run + TF extraction (fast end-to-end check) |
| `single`      | one ground state, with figures, TF overlay and a summary row |
| `tf_only`     | Thomas–Fermi reference alone (no solve) |
| `sweep`       | vary any numeric parameter over a list, rest fixed; each point tries continuation + `nseeds` fresh seeds and keeps the lowest energy |
| `convergence` | refine grid N / box L / padding at fixed physics |

Set the residual tolerance yourself with `res=` (alias of `res_tol`); other
aliases are `etol=`, `N=`, `dim=`.

The defaults describe a small, safe problem (2D, `Omega=0`, `G_C=0`, `Ngrid=128`).
Switch rotation, the long-range term and larger grids on explicitly. For rotating
runs and rotating sweeps use `seed="triangular"` and/or `nseeds>=2`.
Lists on the command line are JSON: `--sweep_values "[5, 10, 20]"`.

The CG coefficient is `cg_beta="pr_precond"` by default since 1.2.0
(`"pr"` reproduces 1.0–1.1 and the v1.1 benchmark scripts). `record_trace=True`
stores the per-iteration history `(time, iteration, E, residual)` in `obs["trace"]`.

A solve ends with one `stop_reason`: `converged` (relative residual below
`res_tol`), `energy_converged` (relative energy change over `conv_window`
iterations below `energy_tol`), `line_search_stalled` (`linesearch_stall`
consecutive failed safeguarded Armijo searches), or
`maxit`. In double precision the energy decrease of a step, which scales as the
squared residual, reaches the rounding level of `E` at a relative residual of
about 1e-9–1e-8, so very tight tolerances can encounter a rounding floor. A failed CG search
is restarted along the preconditioned gradient with a full backtrack.
`line_search_stalled` means both searches failed repeatedly; it does not
prove that rounding is the cause; the default `res_tol=1e-4` is far above it. `iters` is the number of iterations performed. `converged` is true only
for the first two, or when the final residual is below `res_tol`.
Energy convergence is a plateau criterion, not a certificate that the residual tolerance was met.

Configuration keys are validated by `default_cfg`, `run`, `ground_state` and
`make_grid`, and when reading config files or the CLI:
an unknown key, an unknown `dimension` or a non-numeric `sweep_param` is an error,
and the aliases (`res=`, `etol=`, `N=`, `dim=`) work in config files too.

## Output

Runs write to the output directory (`--outdir`, else `GPP_OUTDIR`, else
`./becgpp_out`):

- CSV rows with the full diagnostic set — `E, mu, Lz, Nv, w_LLL, lll_param, oblateness,
  R50/R90/R99, points_per_R90, virial_rel, resid_rel, stop_reason, tf_kind, tf_R90, tf_mu,
  tf_rho0, walltime, iters` (sweeps also record `branch` and `candidate_E`);
- a per-run JSON record (`<rid>.json`) with the full config + every diagnostic;
- publication-style figures (`fig/`, PDF + PNG, no titles, labelled colour bars);
- checkpoints (`ckpt/`) — a rerun with the same config resumes from them.

## Package layout

```
becgpp/
  constants.py     device, dtype, physical constants
  paths.py         output-directory management
  config.py        default configuration (CFG_DEFAULTS, default_cfg)
  units.py         physical <-> trap-unit conversion
  grid.py          grid, spectral operators, auto-box, resampling
  interactions.py  free-space convolution kernels
  operators.py     energy, Hamiltonian, residual, observables
  fields.py        radii, radial averaging, smoothing
  thomasfermi.py   generic TF dispatcher
  seeds.py         gaussian / TF / triangular seeds
  vortex.py        vortex counting and LLL weight
  solvers.py       preconditioned-CG ground-state solver
  diagnostics.py   the per-run diagnostic bundle
  figures.py       publication figures + inline display
  io.py            run ids and CSV output
  modes.py         validate/smoke/single/tf_only/sweep/convergence + dispatch
  cli.py           command-line entry point
examples/          ready-made YAML configs, run.py, Kaggle quick-start
tests/             pytest: validation gate, kernel virials/limits, smoke, audit
benchmarks/        scripts that regenerate the paper's tables and figures
  results/         their raw outputs, one folder per script
data/sample_output/  sample I/O (format reference; produced with version 1.0)
docs/              change notes and review records
tools/cpc/         one-cell GPU runners and packaging helpers for the CPC bundle
CHANGELOG.md       changes in each release
CITATION.cff       citation metadata
pyproject.toml     package metadata and dependencies
```

## Reproducibility / tests

```bash
pytest -q          # validation gate, kernel virials/limits, smoke and audit tests
```

The `validate` gate checks all three free-space kernels against analytic
Gaussians (a few parts in 10⁴–10³ at N=256–512, second order in Δx) and the
Landau-level identities (machine precision). The audit tests check the energy
gradient against the Hamiltonian by finite differences for every kernel and
dimension, kernel symmetry, `mu = <H>`, exact non-interacting energies, vortex
counting, and hold a regression test for every fixed defect. All tests run in CI
on every push to `main` (Python 3.9, 3.11, 3.12; CPU).

The paper's numbers are regenerated by the single-cell Kaggle scripts in
`benchmarks/` (each installs the exact historical commit it was run with); their raw
outputs are in `benchmarks/results/`, and `benchmarks/make_paper_numbers.py`
turns them into the tables and figures. See [`benchmarks/README.md`](benchmarks/README.md).

| script | paper | produces |
|---|---|---|
| `verification.py` | Secs. 5, 7 | validation gate, −ln r / quasi-2D virials, sweeps, cubic–quintic ladder, CG variants, performance |
| `selfgravity_limits.py` | Sec. 5 | Schrödinger–Newton and Thomas–Fermi limits of the untrapped self-gravitating condensate, mass–radius curve |
| `itp_agreement.py` | Sec. 5 | agreement with an independent imaginary-time split-step solver |
| `itp_timing.py` | Sec. 7 | time to a given energy accuracy, PCG versus imaginary time on the same GPU |

See [`CHANGELOG.md`](CHANGELOG.md) for the changes in each release.

## Citing

See [`CITATION.cff`](CITATION.cff). Please cite both the software release and the
accompanying *Computer Physics Communications* paper.

## License

MIT — see [`LICENSE`](LICENSE).


## Versions of the paper's data

The article states, table by table, which version of the program produced each
result; data from versions 1.0 and 1.1 keep their original labels and are not
timings of this version. See `benchmarks/README.md` and `CHANGELOG.md`. Test and
review records are in [`docs/`](docs/); the one-cell GPU runners and packaging
helpers are in [`tools/cpc/`](tools/cpc/README.md).

The trapped attractive cubic–repulsive quintic TF branch uses liquid/vacuum
coexistence. Its discontinuity may prevent exact unit mass on a finite grid;
`mass_error` and `converged` report this instead of renormalizing a physical
root into a state that violates the TF equation. Zero-pressure flat-top TF is
used only for a constant effective potential. All wavefunction inputs and grids
are evaluated in double precision. Multi-seed `walltime` sums all seed solves;
`selected_seed_walltime` records the winning solve separately.

Single and sweep modes write the per-run JSON and optional state outputs listed above;
validate and convergence modes write their diagnostic tables, and tf_only writes
its TF table and optional figures. Output switches control figure/checkpoint creation.
