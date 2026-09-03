# becGPP

**A GPU-accelerated spectral ground-state solver for trapped, rotating and self-gravitating Bose–Einstein condensates in 2D, 3D and quasi-2D.**

`becGPP` computes the ground state of the general Gross–Pitaevskii–Poisson (GPP)
energy functional. Dimension, trap power, contact and three-body couplings,
rotation, and the long-range kernel are all independent run-time parameters — the
same code path serves an ordinary trapped condensate, a rotating vortex lattice,
a self-bound cubic–quintic droplet, and a Newtonian boson star. Nothing about a
particular physical regime is hard-coded.

The energy functional (trap units ℏ = m = ω⊥ = 1, ∫|ψ|² = 1, d = 2 or 3):

```
E[ψ] = ∫ [ ½|∇ψ|² + ½ r^s |ψ|² − Ω ψ* L_z ψ ] dr
     + (β₂/2) ∫|ψ|⁴ dr        (two-body contact)
     + (β₃/3) ∫|ψ|⁶ dr        (three-body / quintic)
     − (G_C/2) ∬ ρ(r) K(|r−r′|) ρ(r′) dr dr′   (long range)
```

with kernel `K = 1/r` (Newton, 3D & quasi-2D), `K = −ln r` (Poisson, 2D), or none.

## Method (one paragraph)

Ground states are found by a **normalized preconditioned conjugate-gradient**
minimization on the unit-norm manifold: an **adaptive Sobolev preconditioner**
(shift = median bulk potential), **Polak–Ribière CG** with periodic restart, a
**Barzilai–Borwein** trial step, and a backtracking **Armijo** line search. The
kinetic and Lᵤ operators are **spectral (FFT)**; the long-range potential is a
**zero-padded free-space convolution** with a Gauss–Legendre cell-averaged kernel
and an analytic self-term. A **Thomas–Fermi** reference is extracted in every
mode, and an **analytic validation gate** (Coulomb integrals + Landau-level
identities) certifies the framework. See the accompanying paper for details.

## Install

```bash
git clone https://github.com/Duyle0503/becGPP.git
cd becGPP
pip install -e .            # runtime deps: torch, numpy, matplotlib
pip install -e ".[yaml,test]"   # + PyYAML configs and pytest
```

A CUDA-capable GPU is used automatically when available; the solver falls back to
the CPU otherwise. Double precision is used throughout (required: the
gravitational coupling can be far weaker than the contact term).

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

**One-button paper data**

```bash
becgpp --mode scan_all --scan_profile full --outdir ./out
```

runs validation, eight showcase singles, four parameter sweeps, the grid
convergence study and the GPU benchmark, writing CSVs and publication-style
figures. Use `--scan_profile quick` for a fast dry run first.

## Modes

| mode          | what it does |
|---------------|--------------|
| `validate`    | analytic gates: Coulomb Gaussian, LLL identities, flat-top TF |
| `smoke`       | validate + a tiny run + TF extraction (fast end-to-end check) |
| `single`      | one ground state, with figures, TF overlay and a summary row |
| `tf_only`     | Thomas–Fermi reference alone (no solve) |
| `sweep`       | vary any numeric parameter over a list |
| `convergence` | refine grid N / box L / padding at fixed physics |
| `scan_all`    | the entire paper dataset in one run |
| `refig`       | re-render all figures from saved checkpoints (no re-solve) |

## Output

Every run writes to the output directory (`GPP_OUTDIR`, else `--outdir`, else
`./becgpp_out`):

- CSV rows with the full diagnostic set — `E, mu, Lz, Nv, w_LLL, oblateness,
  R50/R90/R99, virial_rel, resid_rel, tf_kind, tf_R90, tf_mu, tf_rho0, walltime, iters`;
- publication-style figures (`fig/`, PDF + PNG, no titles, labelled colour bars);
- checkpoints (`ckpt/`) for resume and for `refig`;
- `scan/scan_manifest.json` recording code version, device and wall time.

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
  modes.py         validate/single/sweep/convergence/scan_all/refig + dispatch
  cli.py           command-line entry point
examples/          ready-made YAML configs + run.py
tests/             pytest (analytic validation gate)
data/sample_output/  sample I/O for one run
```

## Reproducibility / tests

```bash
pytest -q          # runs the analytic validation gate as a unit test
```

The `validate` gate checks the Coulomb solver and Landau-level algebra against
closed forms to ≤ 10⁻³ (LLL residuals to machine precision); the same gate runs
in CI on every push.

## Citing

See [`CITATION.cff`](CITATION.cff). Please cite both the software release and the
accompanying *Computer Physics Communications* paper.

## License

MIT — see [`LICENSE`](LICENSE).
