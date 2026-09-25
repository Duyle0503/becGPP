"""Long-range kernels beyond the validation gate: the virial identity of the
-ln r kernel, the thin-disc limit of the quasi-2D kernel, and safe defaults.
Requires torch; skipped if unavailable."""
import math
import tempfile

import pytest

torch = pytest.importorskip("torch")

from becgpp import default_cfg, paths, make_grid, ground_state, diagnostics, q2d_kernel
from becgpp.config import CFG_DEFAULTS
from becgpp.units import sim_params_from_physical


@pytest.fixture(scope="module", autouse=True)
def _tmp_outdir():
    paths.configure(tempfile.mkdtemp(prefix="becgpp_kern_"))
    yield


def _small(**kw):
    base = dict(mode="single", dimension="2D", s=2, Omega=0.0, beta2=50, beta3=0,
                L=8, Ngrid=64, seed="tf", maxit=6000, res_tol=1e-6,
                show_inline=False, save_figs=False, save_ckpt=False)
    base.update(kw)
    return default_cfg(**base)


def test_log_kernel_virial():
    # For K = -ln r the virial term is -G_C/2 (unit mass), not p*E_g with p=0.
    cfg = _small(G_C=5.0, kernel="log")
    psi, G, obs = ground_state(cfg, verbose=False)
    d = diagnostics(psi, G, cfg)
    assert abs(d["virial_lr_term"] + 2.5) < 1e-6
    assert d["virial_rel"] < 5e-3, d["virial_rel"]


def test_newton_2d_virial():
    cfg = _small(G_C=5.0, kernel="newton")
    psi, G, obs = ground_state(cfg, verbose=False)
    d = diagnostics(psi, G, cfg)
    assert d["virial_rel"] < 5e-3, d["virial_rel"]


def test_q2d_virial():
    cfg = _small(dimension="quasi2D", G_C=5.0, kernel="q2d", l_z=0.5)
    psi, G, obs = ground_state(cfg, verbose=False)
    d = diagnostics(psi, G, cfg)
    assert d["virial_rel"] < 5e-3, d["virial_rel"]


def test_q2d_kernel_limits():
    u = torch.tensor([5.0, 10.0], dtype=torch.float64)
    # far field: K_eff -> 1/u
    k = q2d_kernel(u, 0.05)
    assert torch.allclose(k, 1.0 / u, rtol=1e-4)
    # near field: log singularity, K_eff(u) ~ (2/(sqrt(2pi) l)) [-ln u + ln(2 sqrt2 l) - gamma/2]
    l = 1.0
    small = torch.tensor([1e-3], dtype=torch.float64)
    ref = (2.0 / (math.sqrt(2 * math.pi) * l)) * (-math.log(1e-3) + math.log(2 * math.sqrt(2) * l)
                                                  - 0.5 * 0.5772156649015329)
    assert abs(q2d_kernel(small, l).item() - ref) / ref < 1e-4


def test_quasi2d_auto_kernel():
    G = make_grid(default_cfg(dimension="quasi2D", G_C=1.0, l_z=0.3, Ngrid=32))
    assert G["kernel"] == "q2d"
    G = make_grid(default_cfg(dimension="quasi2D", G_C=1.0, l_z=0.0, Ngrid=32))
    assert G["kernel"] == "newton"
    G = make_grid(default_cfg(dimension="2D", G_C=1.0, Ngrid=32))
    assert G["kernel"] == "log"


def test_safe_defaults():
    assert CFG_DEFAULTS["Omega"] == 0.0
    assert CFG_DEFAULTS["G_C"] == 0.0
    assert CFG_DEFAULTS["Ngrid"] <= 128


def test_units_per_particle_gravity():
    # G_C carries one factor of N (per-particle energy), like beta2 = g N.
    a = sim_params_from_physical("selfgrav3d", m=1e-26, omega_perp=100.0, N=1e4, verbose=False)
    b = sim_params_from_physical("selfgrav3d", m=1e-26, omega_perp=100.0, N=2e4, verbose=False)
    assert abs(b["G_C"] / a["G_C"] - 2.0) < 1e-12
