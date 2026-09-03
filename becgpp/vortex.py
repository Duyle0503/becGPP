"""Vortex counting and lowest-Landau-level weight (2D / quasi-2D; 3D via midplane)."""
import math

import numpy as np
import torch

from .grid import dV, norm_of


def _count_vortices_2d(psi2d, X2d, Y2d, dx, Omega=1.0, dens_thresh_frac=0.15):
    """Count +-1 phase windings inside the droplet on a single 2D field/slice."""
    psi2d = psi2d.contiguous()
    rho = psi2d.abs()**2
    dv2 = dx * dx
    mass = (rho.sum() * dv2).item()
    if mass <= 0:
        return dict(Nv=0, Nplus=0, Nminus=0, Nnet=0)
    xcm = (X2d * rho).sum().item() * dv2 / mass
    ycm = (Y2d * rho).sum().item() * dv2 / mass
    rr = torch.sqrt((X2d - xcm)**2 + (Y2d - ycm)**2)
    w = (rho * dv2).reshape(-1)
    order = torch.argsort(rr.reshape(-1))
    cum = torch.cumsum(w[order], 0)
    j = int(torch.searchsorted(cum, 0.99 * cum[-1]).clamp(max=cum.numel() - 1).item())
    R = rr.reshape(-1)[order[j]].item()
    if not np.isfinite(R) or R <= 0:
        return dict(Nv=0, Nplus=0, Nminus=0, Nnet=0)
    theta = torch.angle(psi2d)

    def wrap(dd):
        return (dd + math.pi) % (2 * math.pi) - math.pi
    d1 = wrap(theta[1:, :-1] - theta[:-1, :-1])
    d2 = wrap(theta[1:, 1:] - theta[1:, :-1])
    d3 = wrap(theta[:-1, 1:] - theta[1:, 1:])
    d4 = wrap(theta[:-1, :-1] - theta[:-1, 1:])
    wind = torch.round((d1 + d2 + d3 + d4) / (2 * math.pi))
    N = psi2d.shape[0]
    k = 2 * math.pi * torch.fft.fftfreq(N, d=dx).to(psi2d.device)
    KX, KY = torch.meshgrid(k, k, indexing="ij")
    K2 = KX**2 + KY**2
    ell = min(2.0 * math.sqrt(math.pi / max(Omega, 1e-9)), max(R / 3.0, 2.0 * dx))
    nbar = torch.clamp(torch.fft.ifftn(torch.fft.fftn(rho) * torch.exp(-0.5 * ell**2 * K2)).real, min=0.0)
    cell = 0.25 * (nbar[:-1, :-1] + nbar[1:, :-1] + nbar[1:, 1:] + nbar[:-1, 1:])
    rc = torch.sqrt((X2d[:-1, :-1] - xcm)**2 + (Y2d[:-1, :-1] - ycm)**2)
    mask = (cell > dens_thresh_frac * nbar.max()) & (rc < R)
    nplus = int(((wind >= 1) & mask).sum().item())
    nminus = int(((wind <= -1) & mask).sum().item())
    return dict(Nv=nplus + nminus, Nplus=nplus, Nminus=nminus, Nnet=nplus - nminus)


def vortex_diagnostic(psi, G, Omega=1.0, dens_thresh_frac=0.15):
    """Vortex count. In 3D the vortices are straight lines along z, counted where
    they pierce the z=0 midplane."""
    if G["ndim"] == 2:
        return _count_vortices_2d(psi, G["X"], G["Y"], G["dx"], Omega, dens_thresh_frac)
    mid = G["N"] // 2
    return _count_vortices_2d(psi[:, :, mid], G["X"][:, :, mid], G["Y"][:, :, mid],
                              G["dx"], Omega, dens_thresh_frac)


def lll_weight(psi, G, Mmax=60):
    if G["ndim"] != 2:
        return float("nan")
    dv = dV(G)
    psi_n = psi / norm_of(psi, G)
    r = torch.sqrt(G["R2"])
    theta = torch.atan2(G["Y"], G["X"])
    r2mean = (G["R2"] * psi_n.abs()**2).sum().item() * dv
    estimate = int(math.ceil(r2mean + 10.0 * math.sqrt(max(r2mean, 1.0)) + 30.0))
    Mcap = min(max(int(Mmax), estimate), max(int(Mmax), int((0.80 * G["L"])**2)))
    w = 0.0
    for m in range(Mcap + 1):
        if m == 0:
            mag = torch.exp(-0.5 * G["R2"]) / math.sqrt(math.pi)
        else:
            lm = m * torch.log(torch.clamp(r, min=1e-300)) - 0.5 * G["R2"]
            lm = lm - 0.5 * (math.log(math.pi) + math.lgamma(m + 1.0))
            mag = torch.exp(lm)
        q = mag.to(torch.complex128) * torch.exp(1j * m * theta)
        q = q / norm_of(q, G)
        w += ((q.conj() * psi_n).sum() * dv).abs().item()**2
    return min(max(w, 0.0), 1.0)
