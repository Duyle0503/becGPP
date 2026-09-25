"""Analytic validation gate as a unit test.

Runs the same closed-form checks the solver ships with (free-space 1/r, -ln r and
quasi-2D kernels vs analytic Gaussians, single-particle Landau-level identities,
the flat-top Thomas-Fermi bookkeeping, the 3D Gaussian central potential, and the
kernel convergence orders) and asserts they pass. Requires torch; skipped
automatically if torch is not installed so the rest of the suite can still run.
"""
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

from becgpp import default_cfg, paths
from becgpp.modes import mode_validate

_RES = {}


@pytest.fixture(scope="module", autouse=True)
def _tmp_outdir():
    d = tempfile.mkdtemp(prefix="becgpp_test_")
    paths.configure(d)
    yield d


def _gate():
    # one gate run (N=256) shared by all tests of this module
    if "r" not in _RES:
        _RES["r"] = mode_validate(default_cfg(validate_N=256, show_inline=False))
    return _RES["r"]


def test_validation_gate_passes():
    res = _gate()
    assert res["passed"], res


def test_validation_error_magnitudes():
    res = _gate()
    assert res["mg"] < 5e-3       # 2D 1/r energy
    assert res["mp"] < 5e-3       # 2D 1/r central potential
    assert res["mlg"] < 5e-3      # 2D -ln r energy
    assert res["mlp"] < 5e-3      # 2D -ln r central potential
    assert res["mq"] < 5e-3       # quasi-2D kernel K_eff(l_z)
    assert res["ml"] < 1e-12      # LLL identities (machine precision)
    assert res["mtf"] < 1e-12     # flat-top TF bookkeeping
    assert res["m3"] < 5e-3       # 3D 1/r central potential
    for name in ("2D", "2D_log", "3D"):
        assert res["kernel_orders"][name] > 1.7, res["kernel_orders"]


def test_validation_csv_written():
    _gate()
    assert os.path.isfile(os.path.join(paths.BASE, "validation.csv"))
