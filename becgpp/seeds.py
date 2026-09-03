"""Initial-state seeds: Gaussian, Thomas-Fermi envelope, triangular vortex lattice."""
import math

import numpy as np
import torch

from .grid import norm_of, dV
from .thomasfermi import extract_tf


def gaussian_seed(G, cfg):
    torch.manual_seed(cfg["rng"])
    env = torch.exp(-0.5 * G["R2"])
    noise = 1.0 + cfg.get("seed_noise", 1e-2) * torch.randn_like(env)
    psi = (env * noise).to(torch.complex128)
    m = int(cfg.get("seed_winding", 0))
    if m != 0:
        psi = psi * torch.exp(1j * m * torch.atan2(G["Y"], G["X"]))
    return psi / norm_of(psi, G)


def tf_seed(G, cfg):
    tf = extract_tf(cfg, G)
    if tf is None or tf.get("n") is None:
        return gaussian_seed(G, cfg)
    amp = torch.sqrt(torch.clamp(tf["n"], min=0.0))
    torch.manual_seed(cfg["rng"])
    noise = 1.0 + cfg.get("seed_noise", 1e-2) * torch.randn_like(amp)
    pn = cfg.get("seed_phase_noise", 5e-2) * torch.randn_like(amp)
    psi = (amp * noise).to(torch.complex128) * torch.exp(1j * pn)
    m = int(cfg.get("seed_winding", 0))
    if m != 0:
        psi = psi * torch.exp(1j * m * torch.atan2(G["Y"], G["X"]))
    return psi / norm_of(psi, G)


def _inplane_radius(n, G, frac=0.90):
    """In-plane (x-y) radius enclosing a mass fraction; used to fill the lattice."""
    w = (torch.clamp(n, min=0.0) * dV(G)).reshape(-1)
    rperp = torch.sqrt(G["X"]**2 + G["Y"]**2).reshape(-1)
    order = torch.argsort(rperp)
    cum = torch.cumsum(w[order], dim=0)
    jj = int(torch.searchsorted(cum, float(frac) * cum[-1]).clamp(max=cum.numel() - 1).item())
    return rperp[order[jj]].item()


def triangular_seed(G, cfg):
    """Abrikosov triangular vortex lattice on the TF envelope. In 3D the vortices
    are STRAIGHT LINES parallel to the rotation axis z (the phase and cores use
    the in-plane coordinates only), arranged triangularly in the x-y plane -- the
    standard rotating-condensate vortex lattice. The envelope is the (oblate) TF
    density, so the imprinted state already carries angular momentum."""
    tf = extract_tf(cfg, G)
    if tf is None or not tf.get("converged", False):
        return tf_seed(G, cfg)
    amp = torch.sqrt(torch.clamp(tf["n"], min=0.0))
    O = max(float(cfg.get("Omega", 1.0)), 1e-12)
    av = math.sqrt(2.0 * math.pi / (math.sqrt(3.0) * O))
    dy = math.sqrt(3.0) * av / 2.0
    Rperp = _inplane_radius(tf["n"], G, 0.90)
    Rv = max(av, 0.98 * Rperp)
    rng = np.random.default_rng(int(cfg.get("rng", 0)))
    torch.manual_seed(int(cfg.get("rng", 0)))
    angle = float(rng.uniform(0.0, math.pi / 3.0))
    shift = rng.uniform(-0.25, 0.25, size=2) * av
    ca, sa = math.cos(angle), math.sin(angle)
    pos = []
    jmax = int(math.ceil(Rv / dy)) + 2
    imax = int(math.ceil(Rv / av)) + 3
    for jj in range(-jmax, jmax + 1):
        for ii in range(-imax, imax + 1):
            x0 = (ii + 0.5 * (jj & 1)) * av + shift[0]
            y0 = jj * dy + shift[1]
            xx = ca * x0 - sa * y0
            yy = sa * x0 + ca * y0
            if xx * xx + yy * yy < Rv * Rv:
                pos.append((xx, yy))
    phase = torch.zeros_like(amp)
    core = torch.ones_like(amp)
    xi = 0.25 * av
    for xx, yy in pos:                                    # in-plane distances -> straight lines along z in 3D
        dxt = G["X"] - xx
        dyt = G["Y"] - yy
        dist = torch.sqrt(dxt * dxt + dyt * dyt)
        phase = phase + torch.atan2(dyt, dxt)
        core = core * torch.tanh(dist / max(xi, 1e-12))
    noise = cfg.get("seed_noise", 1e-2) * torch.randn_like(amp)
    psi = (amp * core * (1.0 + noise)).to(torch.complex128) * torch.exp(1j * phase)
    return psi / norm_of(psi, G)


def make_seed(G, cfg):
    kind = cfg.get("seed", "tf")
    if kind == "triangular":
        return triangular_seed(G, cfg)
    if kind == "tf":
        return tf_seed(G, cfg)
    return gaussian_seed(G, cfg)
