"""End-to-end smoke tests: a tiny ground-state solve and the unit-conversion maps.

These exercise the full module wiring (grid -> operators -> solver -> diagnostics
-> Thomas-Fermi) on a small grid, so a broken import or a misplaced function
surfaces immediately. Requires torch; skipped if unavailable.
"""
import math
import tempfile

import pytest

torch = pytest.importorskip("torch")

from becgpp import (default_cfg, paths, make_grid, ground_state, diagnostics,
                    extract_tf, sim_params_from_physical)


@pytest.fixture(scope="module", autouse=True)
def _tmp_outdir():
    paths.configure(tempfile.mkdtemp(prefix="becgpp_smoke_"))
    yield


def test_single_ground_state_converges():
    cfg = default_cfg(mode="single", dimension="2D", s=2, Omega=0.0,
                      beta2=100, beta3=0, G_C=0, kernel="none",
                      L=10, Ngrid=96, seed="tf", maxit=4000, res_tol=1e-4,
                      show_inline=False, save_figs=False, save_ckpt=False, zip_output=False)
    G = make_grid(cfg)
    psi, G, obs = ground_state(cfg, G=G, verbose=False)
    assert obs["converged"]
    assert obs["resid_rel"] < 2e-4
    d = diagnostics(psi, G, cfg)
    # repulsive trapped gas -> inverted-parabola TF reference exists
    assert d["tf_kind"] == "repulsive_parabola"
    assert math.isfinite(d["E"]) and d["virial_rel"] < 1e-2


def test_flattop_tf_exact():
    # Self-bound cubic-quintic: flat-top density rho0 = -3 b2 / (4 b3).
    cfg = default_cfg(dimension="2D", s=2, Omega=1.0, beta2=-1.0, beta3=1.0,
                      G_C=0, kernel="none", L=20, Ngrid=192)
    G = make_grid(cfg)
    tf = extract_tf(cfg, G)
    assert tf is not None and tf["kind"] == "local_flattop"
    assert abs(tf["rho0"] - 0.75) < 1e-9        # -3*(-1)/(4*1) = 0.75


def test_unit_conversion_selfgrav():
    out = sim_params_from_physical("selfgrav3d", m=1e-26, omega_perp=100.0,
                                   N=1e5, a_s=0.0, verbose=False)
    assert out["dimension"] == "3D" and out["kernel"] == "newton"
    assert out["G_C"] > 0.0
