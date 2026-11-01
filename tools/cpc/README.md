# CPC verification and submission tools

These tools operate on the separate code/data/latex bundle. Updated 1 November
2026. Numerical modules and the 109-test suite still match tag v1.2.2 at
1c7eef613d147b6857375f9a61342b38cae1bc86; later commits add verification tools,
benchmark protocols and documentation only.

## Current evidence

- Release: 109 CPU tests and seven reference cases pass, including the fresh
  local audit on 29 October. A source-pinned 31 October run now passes all
  109 CUDA tests on one T4 (28.991 s JUnit, no skips/errors/failures).
- rc1: the historical T4 rerun passed all 97 tests in 29.52 seconds, with no
  skipped/excluded cases. This is not a release CUDA run.
- Original 71-job batch: 56 PASS, 14 REVIEW, 1 FAIL; labels are preserved.
- Controlled CPU study: 48 solves, 45 residual passes, three iteration caps.
  All returned states were recomputed; no universally best shift/BB option.

## Runners

`gpu/GPU_RELEASE_RETEST.ipynb` runs the 109 release tests in one CUDA cell.
`gpu/GPU_TRAPPED3D_REFINEMENT.ipynb` now supplies protocol v2: eleven physics
solves (grid/box/padding/seed/tighter residual) and the 109 CUDA tests by default.
It includes checkpoint/resume, per-job hashes and a standalone offline reader.
Read `gpu/TRAPPED3D_REFINEMENT_README.md`. The supplied GPU dataset passes all
eleven physics jobs and the CUDA suite. The updated reader passes 34 harness
checks; it tolerates last-bit summary reduction differences across NumPy
versions while recomputing unchanged scientific thresholds and exact verdicts.
Both runners pin source content and forbid substituting CPU evidence.

The four-grid trapped-3D energies have observed orders 1.979 and 1.987;
finest scaled virial residual is 1.93e-4. Padding, fixed-spacing box,
Gaussian/TF and tighter-residual controls pass. These finite-grid results do
not prove continuum accuracy at 1e-8 or global minimality. Independent
NumPy/SciPy evaluations of the N=96,128 states agree at roundoff. Portable
scripts are benchmarks/recheck_trapped3d_states.py and
benchmarks/plot_trapped3d_refinement.py. Raw provenance and detailed review:
docs/CPC_FINAL_REVIEW_2026_11_01.md.

`benchmarks/cpc_ablation.py` is CPU-only. Its script used for the recorded run
is preserved with the evidence as runner_executed.py; the shipped version
additionally permits provenance capture when a source ZIP has no Git history.
The controls do not reproduce another published solver's shift/step rules.

The older GPU_CHECK_ONE_CELL and GPU_RETEST_ONLY notebooks reproduce rc1.
Read gpu/GPU_ONE_CELL_README.md to distinguish their scopes.

## Audit and packaging

The manuscript source layout is now one `latex/main.tex`, a separate
`latex/references.bib`, and the image directory `latex/figs/`. Both verification
sections are included directly in main.tex. The LaTeX source ZIP includes no
subsidiary .tex or .bbl files; build with pdflatex, BibTeX, then pdflatex twice.

Use code/tools in the portable bundle, or these matching tools/cpc copies in
an attached checkout. Set BECGPP_CPC_ROOT if automatic location is ambiguous.

- verify_final_project.py checks raw hashes, release numerical/test hashes,
  historical snapshots, CPU/CUDA test identifiers, notebooks and LaTeX links.
  Integrity PASS is separate from release CUDA evidence and author approval.
- pinned_source.py verifies captured historical bytes against original run
  hashes when .git is absent. Snapshots are not signed Git attestations.
- review_cuda_retest.py audits the original 97-test rerun; review_release also
  checks full 109-test identifiers, hashes, worker and metadata if supplied.
- prepare_final_submission.py excludes caches, validates every ZIP entry and
  reuses an unchanged reproducibility ZIP. The original refinement ZIP is
  included separately once; RESTORE.py restores it to its canonical directory.
  --master-dir selects storage for the large master ZIP. SHA256.json includes
  relative paths. A complete source manifest identifies shipped files.
- finalize_release.ps1 checks the current bundle, runs optional CPU/TeX checks
  and rebuilds archives. It never changes refs, discards edits or publishes a
  tag, and stops on a failed reference check.

Extract the master archive into a new directory and run python RESTORE.py.
It verifies nested CRC/hashes and refuses path escapes or different existing
files; Verification.zip is restored last to overlay the historical data README.
Build latex/main.tex into latex/build, copy its PDF to latex/main.pdf, then run
python code/tools/verify_final_project.py. Git is not required for this audit.
No script submits the article or completes Program Library deposition.
