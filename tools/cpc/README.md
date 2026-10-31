# CPC verification and submission tools

These tools operate on the separate code/data/latex bundle. Updated 31 October
2026. Numerical modules and the 109-test suite still match tag v1.2.2 at
1c7eef613d147b6857375f9a61342b38cae1bc86; later commits add verification tools,
benchmark protocols and documentation only.

## Current evidence

- Release: 109 CPU tests and seven reference cases pass, including the fresh
  local audit on 29 October. CUDA coverage of these 109 tests is pending.
- rc1: the historical T4 rerun passed all 97 tests in 29.52 seconds, with no
  skipped/excluded cases. This is not a release CUDA run.
- Original 71-job batch: 56 PASS, 14 REVIEW, 1 FAIL; labels are preserved.
- Controlled CPU study: 48 solves, 45 residual passes, three iteration caps.
  All returned states were recomputed; no universally best shift/BB option.

## Runners

`gpu/GPU_RELEASE_RETEST.ipynb` runs the 109 release tests in one CUDA cell.
`gpu/GPU_TRAPPED3D_REFINEMENT.ipynb` supplies six grid/box/padding jobs for the
trapped 3D showcase. The latter protocol has not been executed; syntax review
is not GPU validation. Both pin source content and forbid CPU substitution.

`benchmarks/cpc_ablation.py` is CPU-only. Its script used for the recorded run
is preserved with the evidence as runner_executed.py; the shipped version
additionally permits provenance capture when a source ZIP has no Git history.
The controls do not reproduce another published solver's shift/step rules.

The older GPU_CHECK_ONE_CELL and GPU_RETEST_ONLY notebooks reproduce rc1.
Read gpu/GPU_ONE_CELL_README.md to distinguish their scopes.

## Audit and packaging

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
  reuses an unchanged reproducibility ZIP. New raw evidence requires
  --rebuild-reproducibility. A complete source manifest identifies shipped files.
- finalize_release.ps1 checks the current bundle, runs optional CPU/TeX checks
  and rebuilds archives. It never changes refs, discards edits or publishes a
  tag, and stops on a failed reference check.

Restore the source, LaTeX and raw-data ZIPs into one root, then Verification.zip
last (its current data README overlays the historical raw archive README).
Build latex/main.tex into latex/build, copy its PDF to latex/main.pdf, then run
python code/tools/verify_final_project.py. Git is not required for this audit.
No script submits the article or completes Program Library deposition.
