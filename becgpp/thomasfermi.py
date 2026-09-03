"""Generic Thomas-Fermi reference, dispatched by the active parameters.

extract_tf() returns, for whatever parameters are set:
  * G_C != 0                  -> nonlocal compressed-TF (fixed point),
  * local, self-bound (b2<0<b3) -> zero-pressure flat-top TF,
  * local, repulsive + trap    -> inverted-profile TF (parabola if b3=0),
or None where no TF reference is defined.
"""
import math

import torch

from .grid import dV
from .interactions import long_range_phi
from .fields import mass_quantiles

_TF_CACHE = {}


def tf_effective_potential(G, cfg):
    """Co-rotating confining potential for the TF construction. The centrifugal
    term acts IN-PLANE only (perpendicular to the rotation axis z):
        V_eff = 1/2 r^s - 1/2 Omega^2 (x^2 + y^2).
    In 2D at s=2, Omega=1 this vanishes (self-bound, critical rotation).
    In 3D at s=2, Omega=1 it leaves 1/2 z^2: the in-plane trap is gone but the
    axial trap remains, i.e. a quasi-2D pancake -- NOT an isotropic free cloud."""
    O = float(cfg.get("Omega", 1.0))
    rperp2 = G["X"]**2 + G["Y"]**2
    return G["V"] - 0.5 * O * O * rperp2


def _invert_local(mu, Veff, b2, b3):
    """Upper physical root rho>=0 of  b3 rho^2 + b2 rho = mu - Veff  (b3>=0).
    Where no real root exists (discriminant<0, i.e. beyond the droplet edge) the
    density is 0 -- NOT the branch floor -b2/(2 b3), which for attractive cubic
    (b2<0) would leave a spurious nonzero density outside the support."""
    rhs = mu - Veff
    if b3 > 1e-30:
        disc = b2 * b2 + 4.0 * b3 * rhs
        root = (-b2 + torch.sqrt(torch.clamp(disc, min=0.0))) / (2.0 * b3)
        return torch.where(disc > 0.0, torch.clamp(root, min=0.0), torch.zeros_like(rhs))
    if abs(b2) > 1e-30:
        return torch.clamp(rhs / b2, min=0.0)
    return torch.zeros_like(rhs)


def _bisect_mu_for_norm(f_rho, G, lo, hi, steps=90):
    dv = dV(G)
    guard = 0
    while (f_rho(hi).sum() * dv).item() < 1.0:                # expand upper bracket
        hi = lo + 2.0 * (hi - lo) + 1e-9
        guard += 1
        if guard > 200:                                    # cannot reach norm 1 (degenerate) -> give up
            return float("nan")
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if (f_rho(mid).sum() * dv).item() < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def extract_tf(cfg, G, iters=400, relax=0.4):
    """Generic Thomas-Fermi reference for the CURRENT parameters. Returns a dict
    with the TF density n, radii, mu and a 'kind' label, or None if no TF
    reference is defined (e.g. purely repulsive gas with no trap)."""
    b2 = float(cfg["beta2"])
    b3 = float(cfg["beta3"])
    gc = float(cfg["G_C"])
    dv = dV(G)
    Veff = tf_effective_potential(G, cfg)
    nonlocal_on = (G["kernel"] != "none" and abs(gc) > 1e-15)
    key = ("tf", G["kernel"], round(b2, 10), round(b3, 10), round(gc, 10),
           round(float(cfg["Omega"]), 10), int(G["N"]), round(float(G["L"]), 10),
           int(G["ndim"]), int(G["s"]))
    if cfg.get("tf_cache", True) and key in _TF_CACHE:
        return _TF_CACHE[key]
    tf = None

    if nonlocal_on and b2 <= 0 and b3 <= 0:
        # No local pressure to balance the attraction: there is no compressed-TF
        # regime (the state is kinetic-pressure supported). No TF reference.
        tf = None
    elif nonlocal_on:
        # Compressed TF:  mu = Veff + b2 n + b3 n^2 + phi[n],  int n = 1.
        sigma = 0.15 * G["L"]
        n = torch.exp(-G["R2"] / (2 * sigma**2))
        n = n / (n.sum() * dv)
        conv = False
        dn = float("inf")
        mu = float("nan")
        for it in range(int(iters)):
            phi = long_range_phi(n, G, gc)
            base = Veff + phi
            f = lambda mu: _invert_local(mu, base, b2, b3)
            mu = _bisect_mu_for_norm(f, G, base.min().item(),
                                     base.max().item() + abs(b2) + abs(b3) + 1.0)
            if not math.isfinite(mu):
                break               # degenerate -> abort TF
            n_new = f(mu)
            ssum = (n_new.sum() * dv).item()
            if ssum <= 0:
                break
            n_new = n_new / ssum
            dn = (n_new - n).abs().sum().item() * dv
            n = (1 - relax) * n + relax * n_new
            n = n / (n.sum() * dv)
            if dn < 1e-9:
                conv = True
                break
        if math.isfinite(mu):
            R90, R99 = mass_quantiles(n, G, (0.90, 0.99))
            tf = dict(kind="nonlocal_compressed", n=n, R90=R90, R99=R99, mu=mu,
                      rho0=float("nan"), converged=conv, final_dn=dn)
        else:
            tf = None

    else:
        confined = bool(torch.max(Veff).item() > 1e-12)
        if b2 < 0 and b3 > 0:
            # Self-bound droplet: attractive cubic + repulsive quintic. The TF
            # reference is the zero-pressure flat top rho0 = -3 b2/(4 b3), radius
            # set by int n = 1. A weak residual trap (Omega<1) only compresses the
            # edge, so flat-top is the correct leading-order reference at any Omega
            # -- NOT the inverted profile (which cannot even be normalized here).
            rho0 = -3.0 * b2 / (4.0 * b3)
            mu = b2 * rho0 + b3 * rho0**2
            R = 1.0 / math.sqrt(math.pi * rho0) if G["ndim"] == 2 \
                else (3.0 / (4.0 * math.pi * rho0))**(1.0 / 3.0)
            rr = torch.sqrt(G["R2"])
            n = torch.where(rr <= R, torch.full_like(rr, rho0), torch.zeros_like(rr))
            n = n / (n.sum() * dv)
            R90, R99 = mass_quantiles(n, G, (0.90, 0.99))
            tf = dict(kind="local_flattop", n=n, R90=R90, R99=R99, mu=mu,
                      rho0=rho0, R=R, converged=True, final_dn=0.0)
        elif confined and b2 > 0:
            # Repulsive trapped gas: inverted-profile TF (parabola if b3=0).
            f = lambda mu: _invert_local(mu, Veff, b2, b3)
            mu = _bisect_mu_for_norm(f, G, Veff.min().item(),
                                     Veff.max().item() + abs(b2) + abs(b3) + 1.0)
            n = f(mu)
            n = n / (n.sum() * dv)
            R90, R99 = mass_quantiles(n, G, (0.90, 0.99))
            kind = "repulsive_parabola" if b3 <= 0 else "local_trapped_repulsive"
            tf = dict(kind=kind, n=n, R90=R90, R99=R99, mu=mu,
                      rho0=float("nan"), converged=True, final_dn=0.0)
        else:
            tf = None                                     # unbounded / no TF reference

    if tf is not None and cfg.get("tf_cache", True):
        _TF_CACHE[key] = tf
    return tf
