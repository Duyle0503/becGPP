# CUDA review and portable regression fixtures

The supplied 71-job full run used numerical commit
`5f66bfc8635a1791078fdd503160e341fb12e82c` on one Tesla T4 with
PyTorch 2.11.0+cu128 and Python 3.13.15. Its original status is retained:
56 PASS, 14 REVIEW, 1 FAIL. All ten independent energy comparisons passed;
four CPU/CUDA parity studies and eight selected showcase configurations passed
their declared targets. None of the ten strict reference residual requests
of 1e-8 was attained. This is not a claim of an entirely passing CUDA suite.

The CUDA regression job passed 74 of 90 selected tests. Sixteen tests failed
before evaluating their numerical assertions because CPU random tensors were
combined with CUDA grids. Seven other CPU-only reference cases were excluded
by the original runner.

The repaired fixtures create generators and tensors on the grid's device.
Independent NumPy comparisons explicitly transfer computed fields to CPU.
All 97 tests can now be selected on either backend, without excluding those
seven references. These changes affect tests only; every numerical solver
module remains unchanged from the measured commit.

## Repaired CUDA rerun verified on 2026-10-18

A separate regression-only run at commit
`75f0a304acf2ff61dceb3c5ac4df10a9ef84b197` passed all 97 tests on a Tesla T4
in 29.52 seconds (31.96 seconds for the worker job). JUnit reports zero failures,
errors and skips, and the runner excludes zero cases. The 97 unique test
identifiers match the CPU suite: the former 16 failures and all seven previously
excluded reference cases now pass. Python was 3.13.15, PyTorch 2.11.0+cu128 and
CUDA 12.8.

Archive CRC and 13 file checksums were verified; 25 normalized numerical/test
source hashes and the embedded worker match the prepared runner. The review
record is in `docs/evidence/CUDA_RETEST_REVIEW.json`. The raw archive SHA256 is
`8d2ad7d9cdf152c769cb36b359da0cb043619855f42f2cc69a62cbd67d3dee6d`.
Raw data and the full manuscript remain in the separate CPC bundle.

The original 71-job result is retained with its FAIL and 14 REVIEW flags. This
rerun resolves the regression-fixture defect, without rerunning the physics jobs
or establishing that the strict residual targets were attained. CPU and GPU
suite times come from different environments and are not a speedup benchmark.
The newly published bundle tools and documentation do not change the numerical
solver or test assertions.
