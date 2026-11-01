"""Free (translation-invariant) states are returned centred in the periodic box."""
import math

import torch

from becgpp import default_cfg, make_grid, energy_components, ground_state, mass_quantiles
from becgpp.solvers import recentre_periodic


def _free_cq(**over):
    return default_cfg(**dict(dict(dimension="2D", s=0, Omega=0.0, beta2=-50.0, beta3=50.0,
                                   G_C=0.0, kernel="none", L=3.0, Ngrid=48, seed="tf",
                                   res_tol=1e-6, maxit=3000, nseeds=1), **over))


def test_recentre_is_an_exact_symmetry():
    cfg = _free_cq()
    G = make_grid(cfg)
    psi = torch.exp(-((G["X"] - 0.4) ** 2 + (G["Y"] + 0.3) ** 2) / 0.2).to(torch.complex128)
    wrapped = torch.roll(psi, shifts=(G["N"] // 2, 5), dims=(0, 1))
    back, shifts = recentre_periodic(wrapped, G)
    e0 = energy_components(wrapped, G, cfg["beta2"], cfg["beta3"], 0.0, 0.0)["E"].item()
    e1 = energy_components(back, G, cfg["beta2"], cfg["beta3"], 0.0, 0.0)["E"].item()
    assert any(shifts)
    assert abs(e1 - e0) <= 1e-12 * max(1.0, abs(e0))
    i, j = divmod(int(torch.argmax(back.abs()).item()), G["N"])
    assert abs(i - G["N"] // 2) <= 3 and abs(j - G["N"] // 2) <= 3


def test_free_droplet_returned_centred():
    cfg = _free_cq()
    G = make_grid(cfg)
    psi_c, _, obs_c = ground_state(cfg, G=G, verbose=False)
    seed = torch.roll(psi_c, shifts=(G["N"] // 2, G["N"] // 2), dims=(0, 1))   # across the boundary
    psi_w, _, obs_w = ground_state(cfg, G=G, psi0=seed, verbose=False)
    R90_c = mass_quantiles(psi_c.abs() ** 2, G, (0.90,))[0]
    R90_w = mass_quantiles(psi_w.abs() ** 2, G, (0.90,))[0]
    R_tf = 1.0 / math.sqrt(math.pi * 0.75)               # flat-top radius, rho0 = 0.75
    assert R90_c < 1.5 * R_tf and R90_w < 1.5 * R_tf
    assert abs(obs_w["E"] - obs_c["E"]) <= 1e-7 * max(1.0, abs(obs_c["E"]))


def test_trapped_state_not_shifted():
    cfg = default_cfg(dimension="2D", s=2, beta2=50.0, L=6.0, Ngrid=32, maxit=500, res_tol=1e-5)
    _, _, obs = ground_state(cfg, verbose=False)
    assert obs["recentred_shift"] == [0, 0]
