"""Density-field utilities: centre of mass, mass radii, radial averaging, smoothing."""
import numpy as np
import torch

from .constants import DEV
from .grid import dV


def center_of_mass(rho, G):
    r = rho.detach().real if torch.is_tensor(rho) else torch.as_tensor(rho, device=DEV)
    mass = (r.sum() * dV(G)).item()
    if mass <= 0:
        return tuple(0.0 for _ in range(G["ndim"]))
    return tuple((c * r).sum().item() * dV(G) / mass for c in G["coords"])


def mass_quantiles(rho, G, fractions=(0.50, 0.90, 0.99), center=None):
    r = rho.detach().real if torch.is_tensor(rho) else torch.as_tensor(rho, device=DEV)
    w = (r.clamp(min=0.0) * dV(G)).reshape(-1)
    if center is None:
        center = center_of_mass(r, G)
    rad2 = sum((c - center[i])**2 for i, c in enumerate(G["coords"]))
    radii = torch.sqrt(rad2).reshape(-1)
    order = torch.argsort(radii)
    cum = torch.cumsum(w[order], dim=0)
    out = []
    for f in fractions:
        jj = int(torch.searchsorted(cum, float(f) * cum[-1]).clamp(max=cum.numel() - 1).item())
        out.append(radii[order[jj]].item())
    return out


def mass_radius(rho, G, fraction=0.99, center=None):
    return mass_quantiles(rho, G, (fraction,), center=center)[0]


def radial_average(field, G, bins=None, rmax=None, center=None):
    f = field.detach().cpu().numpy() if torch.is_tensor(field) else np.asarray(field)
    coords = [c.cpu().numpy() for c in G["coords"]]
    if center is None:
        center = tuple(0.0 for _ in coords)
    r = np.sqrt(sum((c - center[i])**2 for i, c in enumerate(coords)))
    if rmax is None:
        rmax = G["L"]
    if bins is None:
        bins = int(min(400, max(20, rmax / (2.0 * G["dx"]))))
    edges = np.linspace(0, rmax, bins + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    prof = np.full(bins, np.nan)
    rf = r.ravel()
    ff = f.ravel()
    idx = np.clip(np.digitize(rf, edges) - 1, 0, bins - 1)
    for i in range(bins):
        m = idx == i
        if np.any(m):
            prof[i] = ff[m].mean()
    return ctr, prof


def droplet_radius(rho, G, frac=1e-2):
    ctr, prof = radial_average(rho, G, center=center_of_mass(rho, G))
    pk = np.nanmax(prof)
    if not np.isfinite(pk) or pk <= 0:
        return float("nan")
    valid = prof > frac * pk
    return ctr[valid][-1] if np.any(valid) else float("nan")


def gaussian_smooth(field, G, sigma):
    filt = torch.exp(-0.5 * float(sigma)**2 * G["K2"])
    return torch.fft.ifftn(torch.fft.fftn(field) * filt).real
