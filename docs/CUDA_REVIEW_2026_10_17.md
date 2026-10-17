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
module remains unchanged from the measured commit. A CUDA rerun of the
repaired fixtures is pending; the original results must not be relabelled PASS.
