# CPC verification and submission tools

These scripts accompany the separate CPC bundle organized as `code/`, `data/`
and `latex/`. The repository contains the tools and a compact review record;
the manuscript, raw physics checkpoints and submission ZIPs are supplied
separately. Installing becGPP does not require this bundle.

## Release 1.2.2

`gpu/GPU_RELEASE_RETEST.ipynb` (or the identical `.py`) runs the complete test
suite of release `v1.2.2` (109 tests) on CUDA in one Kaggle/Colab cell. It
downloads the tagged source and refuses to run if its file hashes differ from
those pinned in the cell. Return the result ZIP; extracted, it belongs in
`data/raw/release_1_2_2_cuda/` of the bundle, where `verify_final_project.py`
picks it up (until then the release CUDA check is reported as pending).

`finalize_release.ps1` brings a Windows bundle to the release in one command:
checks out the tag, runs the CPU tests and the reference test run, compiles the
manuscript, verifies the bundle and rebuilds the submission archives.

## GPU notebooks (release candidate evidence)

`gpu/GPU_CHECK_ONE_CELL.ipynb` (or the identical `.py`) runs the full 71-job
verification plan in one Kaggle/Colab cell after enabling CUDA.
`gpu/GPU_RETEST_ONLY.ipynb` runs only the repaired 97-test CUDA regression job.
Read `gpu/GPU_ONE_CELL_README.md` for profiles, dependencies, thresholds and
the difference between PASS, REVIEW and FAIL.

The runners intentionally pin tested commit
`75f0a304acf2ff61dceb3c5ac4df10a9ef84b197`, rather than the moving branch head.
Later documentation/tooling commits do not change that tested numerical source.
The verified T4 rerun passed 97/97 tests in 29.52 s with no excluded cases or
skips. This does not remove the 14 physical-convergence/method REVIEW flags in
the original batch. See `../../docs/CUDA_REVIEW_2026_10_17.md` from this directory
or the repository's `docs/CUDA_REVIEW_2026_10_17.md`.

## Audit and packaging scripts

| Script | Purpose |
|---|---|
| `review_cuda_retest.py` | Check ZIP CRC, file/source hashes, worker, result consistency and all 97 JUnit test identifiers |
| `review_uploaded_gpu.py` | Audit the 71-job batch, recompute saved states on CPU and regenerate figures |
| `verify_final_project.py` | Check the organized bundle's code, data, CPU/CUDA evidence, notebooks and LaTeX references |
| `prepare_final_submission.py` | Package code, LaTeX, evidence and data; read back every ZIP and verify its contents |
| `workspace_paths.py` | Find the separate bundle or use the explicitly configured bundle root |
| `pinned_source.py` | Confirm hashes pinned by earlier evidence runs against the working tree or the pinned git commit |
| `finalize_release.ps1` | Windows: check out the release tag, test, compile, verify and package in one command |

When the checkout is at `<bundle>/code/becGPP`, the bundle is found automatically.
From another clone, set `BECGPP_CPC_ROOT` to the directory containing
`code/becGPP`, `data/raw`, and `latex/main.tex`. This prevents the tools from
mistaking an arbitrary checkout for the submission workspace.

Example from the repository root in PowerShell:

```powershell
$env:BECGPP_CPC_ROOT = 'D:\CPC'
python tools/cpc/review_cuda_retest.py
python tools/cpc/verify_final_project.py
python tools/cpc/prepare_final_submission.py
```

The audit/packaging scripts use the Python standard library. The saved-state
review additionally needs the candidate's PyTorch, NumPy, SciPy and Matplotlib
dependencies. Its explicit inputs can be shown with
`python tools/cpc/review_uploaded_gpu.py --help`. Use only trusted author-supplied
checkpoints. Compile the bundle's LaTeX before running the final project check.

Both raw runs are retained separately. The original full batch's failed fixture
job remains unchanged; the successful regression rerun supplies the repair
evidence. Same-grid energy agreement does not establish continuum accuracy or
global minimality. Final author declarations and the journal submission process
remain separate from code verification.
