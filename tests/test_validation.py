"""Analytic validation gate as a unit test.

Runs the same closed-form checks the solver ships with (free-space Coulomb kernel
vs an analytic Gaussian, single-particle Landau-level identities, the flat-top
Thomas-Fermi relation, and the 3D Gaussian central potential) and asserts they
pass the publication thresholds. Requires torch; skipped automatically if torch
is not installed so the rest of the suite can still run.
"""
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

from becgpp import default_cfg, paths
from becgpp.modes import mode_validate


@pytest.fixture(scope="module", autouse=True)
def _tmp_outdir():
    d = tempfile.mkdtemp(prefix="becgpp_test_")
    paths.configure(d)
    yield d


def test_validation_gate_passes():
    # A modest grid keeps the test fast while still clearing the thresholds.
    cfg = default_cfg(validate_N=256, show_inline=False, zip_output=False)
    res = mode_validate(cfg)
    assert res["passed"], res


def test_validation_error_magnitudes():
    cfg = default_cfg(validate_N=256, show_inline=False, zip_output=False)
    res = mode_validate(cfg)
    assert res["mg"] < 5e-3       # 2D Coulomb energy
    assert res["mp"] < 5e-3       # 2D Coulomb central potential
    assert res["ml"] < 1e-4       # LLL identities (near machine precision)
    assert res["mtf"] < 1e-2      # flat-top TF relation
    assert res["m3"] < 5e-2       # 3D Coulomb central potential


def test_validation_csv_written():
    cfg = default_cfg(validate_N=192, show_inline=False, zip_output=False)
    mode_validate(cfg)
    assert os.path.isfile(os.path.join(paths.BASE, "validation.csv"))
