"""Audit tests: consistency checks that must hold, plus regression tests for
defects found in a code review.

The first group checks internal consistency and exact limits (energy gradient
vs. Hamiltonian, kernel symmetry, non-interacting ground states, mu, vortex
counting). The second group documents known defects; each is marked
``xfail(strict=True)`` so the suite stays green while the defect exists and
turns red (XPASS) once it is fixed -- then remove the marker.
Requires torch; skipped if unavailable.
"""
import json
import math
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

import becgpp.cli as cli
import becgpp.modes as modes
from becgpp import (default_cfg, paths, make_grid, geometry, auto_grid, resample_state,
                    ground_state, diagnostics, extract_tf, kernel_convolution, lll_weight,
                    vortex_diagnostic)
from becgpp.io import run_id
from becgpp.operators import energy_components, apply_H, observables
from becgpp.seeds import triangular_seed


@pytest.fixture(scope="module", autouse=True)
def _tmp_outdir():
    paths.configure(tempfile.mkdtemp(prefix="becgpp_audit_"))
    yield


def _cfg(**kw):
    base = dict(mode="single", L=6, Ngrid=48, show_inline=False, save_figs=False,
                save_ckpt=False)
    base.update(kw)
    return default_cfg(**base)


def _dv(G):
    return G["dx"] ** G["ndim"]


def _random_state(G, seed=1):
    g = torch.Generator().manual_seed(seed)
    env = torch.exp(-0.5 * G["R2"])
    sh = G["R2"].shape
    re = env * (1.0 + 0.3 * torch.randn(sh, generator=g))
    im = 0.3 * env * torch.randn(sh, generator=g)
    return (re + 1j * im).to(torch.complex128), g


_PHYSICS = [
    dict(dimension="2D", beta2=30, beta3=5, Omega=0.4),
    dict(dimension="2D", beta2=30, G_C=3, kernel="newton", Omega=0.3),
    dict(dimension="2D", beta2=30, G_C=3, kernel="log"),
    dict(dimension="quasi2D", beta2=30, G_C=3, kernel="q2d", l_z=0.5),
    dict(dimension="3D", beta2=20, beta3=2, G_C=3, kernel="newton", Omega=0.5, Ngrid=24, L=5),
    dict(dimension="2D", s=4, trap_coeff=0.25, beta2=10),
]


# =============================================================================
#  Consistency checks (expected to pass)
# =============================================================================
@pytest.mark.parametrize("kw", _PHYSICS)
def test_energy_gradient_matches_hamiltonian(kw):
    # dE[psi + eps v]/d eps = 2 Re <H psi, v>: the solver's gradient is the
    # derivative of the energy it minimises, for every term and kernel.
    c = _cfg(**kw)
    G = make_grid(c)
    psi, g = _random_state(G)
    env = torch.exp(-0.5 * G["R2"])
    v = (env * torch.randn(G["R2"].shape, generator=g)
         + 1j * env * torch.randn(G["R2"].shape, generator=g)).to(torch.complex128)

    def E(p):
        return energy_components(p, G, c["beta2"], c["beta3"], c["Omega"], c["G_C"])["E"].item()

    eps = 1e-5
    fd = (E(psi + eps * v) - E(psi - eps * v)) / (2 * eps)
    an = 2.0 * ((apply_H(psi, G, c).conj() * v).real.sum() * _dv(G)).item()
    assert abs(fd - an) / abs(an) < 1e-7


@pytest.mark.parametrize("kw", _PHYSICS[1:5])
def test_kernel_convolution_is_symmetric(kw):
    G = make_grid(_cfg(**kw))
    g = torch.Generator().manual_seed(2)
    a = torch.rand(G["R2"].shape, generator=g, dtype=torch.float64)
    b = torch.rand(G["R2"].shape, generator=g, dtype=torch.float64)
    lhs = (a * kernel_convolution(b, G)).sum().item()
    rhs = (kernel_convolution(a, G) * b).sum().item()
    assert abs(lhs - rhs) <= 1e-12 * abs(lhs)


@pytest.mark.parametrize("kw", _PHYSICS)
def test_mu_equals_expectation_of_H(kw):
    c = _cfg(**kw)
    G = make_grid(c)
    psi, _ = _random_state(G)
    psi = psi / torch.sqrt((psi.abs() ** 2).sum() * _dv(G))
    mu_H = ((psi.conj() * apply_H(psi, G, c)).real.sum() * _dv(G)).item()
    assert abs(observables(psi, G, c)["mu"] - mu_H) < 1e-10 * max(1.0, abs(mu_H))


@pytest.mark.parametrize("kw,exact", [
    (dict(dimension="2D"), 1.0),
    (dict(dimension="3D", Ngrid=32), 1.5),
    (dict(dimension="2D", trap_coeff=0.125), 0.5),       # omega = sqrt(2*0.125) = 1/2
    (dict(dimension="2D", Omega=0.6), 1.0),              # rotation below the trap frequency
])
def test_noninteracting_ground_state_energy(kw, exact):
    L = 9.0 if kw.get("trap_coeff") else 6.0
    c = _cfg(beta2=0.0, res_tol=1e-7, maxit=600, L=L, **kw)
    _, _, obs = ground_state(c, verbose=False)
    assert abs(obs["E"] - exact) < 1e-7, obs["E"]


@pytest.mark.parametrize("m", [1, -1, 2])
def test_vortex_count_of_imprinted_vortex(m):
    c = _cfg(Ngrid=64, L=6)
    G = make_grid(c)
    psi = (torch.exp(-0.5 * G["R2"] / 4.0) * torch.tanh(torch.sqrt(G["R2"]) / 0.5)).to(torch.complex128)
    psi = psi * torch.exp(1j * m * torch.atan2(G["Y"] + 1e-9, G["X"] + 1e-9))
    d = vortex_diagnostic(psi, G, 0.5)
    assert d["Nnet"] == m


def test_lll_weight_of_lll_state():
    G = make_grid(_cfg(Ngrid=96, L=8))
    z = (G["X"] + 1j * G["Y"]).to(torch.complex128)
    psi = z ** 3 * torch.exp(-0.5 * G["R2"])
    assert abs(lll_weight(psi, G) - 1.0) < 1e-8


def test_resample_identity_even_grid():
    G = make_grid(_cfg(Ngrid=64))
    psi = torch.exp(-((G["X"] - 1.0) ** 2 + G["Y"] ** 2)).to(torch.complex128)
    psi = psi / torch.sqrt((psi.abs() ** 2).sum() * _dv(G))
    assert (resample_state(psi, G, G) - psi).abs().max().item() < 1e-12


def test_default_cfg_rejects_unknown_and_maps_aliases():
    assert default_cfg(res=1e-7)["res_tol"] == 1e-7
    with pytest.raises(ValueError):
        default_cfg(omega=0.9)
    with pytest.raises(ValueError):
        default_cfg(res=1e-7, res_tol=1e-6)


def test_cli_coerces_command_line_types(monkeypatch):
    monkeypatch.setattr(cli, "run", lambda cfg: cfg)
    cfg = cli.main(["--beta2", "1e3", "--Ngrid", "64", "--save_figs", "false",
                    "--sweep_values", "[1, 2]", "--s", "2.5"])
    assert cfg["beta2"] == 1000.0 and cfg["Ngrid"] == 64 and cfg["save_figs"] is False
    assert cfg["sweep_values"] == [1, 2] and cfg["s"] == 2.5


# =============================================================================
#  Known defects (xfail until fixed)
# =============================================================================
@pytest.mark.xfail(strict=True, reason="solver: after the energy has settled, every Armijo "
                   "search fails; the Barzilai-Borwein mix keeps alpha >= 2/3*bb_min so "
                   "'line_search_stalled' is unreachable, and the energy-window stop is only "
                   "evaluated on accepted steps -> runs to maxit (8 energy evals/iteration)")
def test_solver_stops_when_line_search_stalls():
    # Exact answer (E = 1) is reached in ~100 iterations; the residual floor set by
    # the box (e^{-L^2/2}) is above res_tol, so only the stall / energy stops can end it.
    c = _cfg(beta2=0.0, res_tol=1e-12, energy_tol=1e-9, conv_window=200, maxit=1000)
    _, _, obs = ground_state(c, verbose=False)
    assert abs(obs["E"] - 1.0) < 1e-10
    assert obs["stop_reason"] != "maxit", obs["stop_reason"]


@pytest.mark.xfail(strict=True, reason="io.run_id ignores seed_winding, seed_noise, maxit, "
                   "energy_tol, ...: a rerun with different settings silently resumes the "
                   "old checkpoint in mode 'single'")
def test_run_id_distinguishes_seed_winding():
    assert run_id(_cfg(seed_winding=0)) != run_id(_cfg(seed_winding=2))


@pytest.mark.xfail(strict=True, reason="cli: config-file keys are merged with dict.update, "
                   "bypassing default_cfg's alias mapping and unknown-key check")
def test_cli_config_file_aliases_and_unknown_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "run", lambda cfg: cfg)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(dict(mode="single", res=1e-2)))
    assert cli.main(["--config", str(p)])["res_tol"] == 1e-2
    p.write_text(json.dumps(dict(mode="single", omega=0.9)))   # typo of Omega
    with pytest.raises((ValueError, SystemExit)):
        cli.main(["--config", str(p)])


@pytest.mark.xfail(strict=True, reason="cli: PyYAML reads '1e3' as a string; config-file "
                   "values are not coerced, so beta2: 1e3 crashes later with a TypeError")
def test_cli_yaml_scientific_notation(monkeypatch, tmp_path):
    pytest.importorskip("yaml")
    monkeypatch.setattr(cli, "run", lambda cfg: cfg)
    p = tmp_path / "c.yaml"
    p.write_text("mode: single\nbeta2: 1e3\n")
    assert isinstance(cli.main(["--config", str(p)])["beta2"], float)


@pytest.mark.xfail(strict=True, reason="mode_sweep: sweep_param is not validated; a typo "
                   "('omega') runs every point with identical physics")
def test_sweep_rejects_unknown_parameter():
    c = _cfg(mode="sweep", sweep_param="omega", sweep_values=[0.0], Ngrid=32, maxit=20)
    with pytest.raises(ValueError):
        modes.mode_sweep(c)


@pytest.mark.xfail(strict=True, reason="mode_sweep: max() of an empty list when "
                   "sweep_values=[] (ValueError instead of an empty result)")
def test_sweep_with_no_values():
    assert modes.mode_sweep(_cfg(mode="sweep", sweep_values=[])) == []


@pytest.mark.xfail(strict=True, reason="mode_smoke: 'd.get(\"converged\") is not None' is "
                   "always True, so OVERALL ignores whether the run converged")
def test_smoke_ok_reflects_convergence(monkeypatch):
    monkeypatch.setattr(modes, "mode_validate", lambda cfg: dict(passed=True))
    monkeypatch.setattr(modes, "mode_single", lambda cfg: dict(converged=False, E=0.0))
    monkeypatch.setattr(modes, "mode_tf_only", lambda cfg: None)
    assert modes.mode_smoke(_cfg())["ok"] is False


@pytest.mark.xfail(strict=True, reason="seeds.triangular_seed uses abs(Omega) and always "
                   "imprints +1 vortices: for Omega<0 the seed rotates the wrong way")
def test_triangular_seed_follows_rotation_sign():
    c = _cfg(Omega=-0.9, beta2=200, L=10, Ngrid=96, seed="triangular")
    G = make_grid(c)
    Lz = energy_components(triangular_seed(G, c), G, 200.0, 0.0, -0.9, 0.0)["Lz"].item()
    assert Lz < 0.0, Lz


@pytest.mark.xfail(strict=True, reason="diagnostics: w_LLL projects on z^m exp(-r^2/2) for "
                   "any Omega; for Omega<0 the LLL is conj(z)^m, so the mirror state scores ~0")
def test_lll_weight_negative_rotation():
    c = _cfg(Omega=-0.9, beta2=0.0, Ngrid=96, L=8)
    G = make_grid(c)
    zb = (G["X"] - 1j * G["Y"]).to(torch.complex128)
    psi = zb ** 3 * torch.exp(-0.5 * G["R2"])
    psi = psi / torch.sqrt((psi.abs() ** 2).sum() * _dv(G))
    assert abs(diagnostics(psi, G, c)["w_LLL"] - 1.0) < 1e-6


@pytest.mark.xfail(strict=True, reason="grid.resample_state assumes the grid starts at -L; "
                   "for odd Ngrid it starts at -(N-1)/2*dx, so the identity map is shifted")
def test_resample_identity_odd_grid():
    G = make_grid(_cfg(Ngrid=65))
    psi = torch.exp(-((G["X"] - 1.0) ** 2 + G["Y"] ** 2)).to(torch.complex128)
    psi = psi / torch.sqrt((psi.abs() ** 2).sum() * _dv(G))
    assert (resample_state(psi, G, G) - psi).abs().max().item() < 1e-12


@pytest.mark.xfail(strict=True, reason="grid.geometry maps any unrecognised dimension "
                   "('1D', '3d ', typos) silently to 2D")
def test_geometry_rejects_unknown_dimension():
    with pytest.raises(ValueError):
        geometry(dict(dimension="1D"))


@pytest.mark.xfail(strict=True, reason="grid.auto_grid: R_est = 0.5*beta2/G_C is negative for "
                   "a repulsive long-range term (G_C<0)")
def test_auto_grid_repulsive_long_range():
    _, _, R_est = auto_grid(_cfg(G_C=-1.0, beta2=10.0))
    assert R_est > 0


@pytest.mark.xfail(strict=True, reason="thomasfermi: trapped repulsive pure quintic "
                   "(beta2=0, beta3>0) has a TF profile sqrt((mu-V)/beta3) but gets None")
def test_tf_trapped_pure_quintic():
    c = _cfg(beta2=0.0, beta3=50.0)
    assert extract_tf(c, make_grid(c)) is not None


@pytest.mark.xfail(strict=True, reason="solver: obs['iters'] is the last 0-based loop index, "
                   "one less than the number of iterations performed")
def test_iteration_count():
    c = _cfg(beta2=50.0, res_tol=1e-14, maxit=5)
    _, _, obs = ground_state(c, verbose=False)
    assert obs["iters"] == 5
