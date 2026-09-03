"""Grid construction, spectral operators, and grid utilities."""
import math

import torch
import torch.nn.functional as F

from .constants import DEV


def geometry(cfg):
    """Return (ndim, quasi) for the configured geometry."""
    g = str(cfg.get("dimension", "2D")).lower()
    if g == "3d":
        return 3, False
    if g == "quasi2d":
        return 2, True
    return 2, False


def resolve_kernel(cfg):
    """Resolve kernel choice into ('newton'|'log'|'none', exponent p)."""
    ndim, quasi = geometry(cfg)
    k = str(cfg.get("kernel", "auto")).lower()
    if abs(float(cfg.get("G_C", 0.0))) <= 1e-15:
        return "none", 0
    if k == "none":
        return "none", 0
    if k == "auto":
        k = "log" if (ndim == 2 and not quasi) else "newton"
    if k == "newton":
        return "newton", 1          # 1/|r|,  scales as lambda^1
    if k == "log":
        return "log", 0             # -ln|r| (2D only); virial anomaly, see note
    raise ValueError(f"unknown kernel {k!r}")


def make_grid(cfg):
    ndim, quasi = geometry(cfg)
    L, N, s = float(cfg["L"]), int(cfg["Ngrid"]), float(cfg["s"])
    kkind, kexp = resolve_kernel(cfg)
    dx = 2.0 * L / N
    x = (torch.arange(N, device=DEV, dtype=torch.get_default_dtype()) - N // 2) * dx
    k1 = 2 * math.pi * torch.fft.fftfreq(N, d=dx).to(DEV)
    if ndim == 2:
        X, Y = torch.meshgrid(x, x, indexing="ij")
        coords = (X, Y)
        R2 = X**2 + Y**2
        KX, KY = torch.meshgrid(k1, k1, indexing="ij")
        kcoords = (KX, KY)
        K2 = KX**2 + KY**2
    else:
        X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
        coords = (X, Y, Z)
        R2 = X**2 + Y**2 + Z**2
        KX, KY, KZ = torch.meshgrid(k1, k1, k1, indexing="ij")
        kcoords = (KX, KY, KZ)
        K2 = KX**2 + KY**2 + KZ**2
    tc = float(cfg.get("trap_coeff", 0.5))               # trap strength: V = tc * r^s
    V = tc * R2 ** (s / 2.0) if s > 0 else torch.zeros_like(R2)
    return dict(ndim=ndim, quasi=quasi, coords=coords, kcoords=kcoords,
                X=coords[0], Y=coords[1], KX=kcoords[0], KY=kcoords[1],
                R2=R2, V=V, K2=K2, s=s, dx=dx, N=N, L=L,
                kernel=kkind, kexp=kexp, pad=max(2, int(cfg.get("pad", 2))))


def dV(G):
    return G["dx"] ** G["ndim"]


def norm_of(p, G):
    return torch.sqrt((p.abs()**2).sum() * dV(G))


def Lz_op(p, G):
    fp = torch.fft.fftn(p)
    dpx = torch.fft.ifftn(1j * G["KX"] * fp)
    dpy = torch.fft.ifftn(1j * G["KY"] * fp)
    return -1j * (G["X"] * dpy - G["Y"] * dpx)


def auto_grid(cfg):
    """Rough half-box L and grid N from a radius estimate (for autobox sweeps)."""
    b2 = float(cfg["beta2"])
    b3 = float(cfg["beta3"])
    gc = float(cfg["G_C"])
    ndim, _ = geometry(cfg)
    if abs(gc) > 1e-15 and b2 > 0:
        R_est = 0.5 * b2 / gc                              # gravitylike-ish 1/gamma
    elif b2 < 0 and b3 > 0:
        rho0 = max(-3 * b2 / (4 * b3), 1e-6)
        R_est = 1.0 / math.sqrt(math.pi * rho0) if ndim == 2 else (3 / (4 * math.pi * rho0))**(1 / 3)
    else:
        R_est = 4.0
    L = max(6.0, 2.5 * R_est)
    dxt = min(0.10, max(R_est / 20.0, 0.01))
    Nn = int(math.ceil(2.0 * L / dxt))
    cap = 512 if ndim == 2 else 224
    N = int(min(cap, max(128, 32 * math.ceil(Nn / 32))))
    return float(L), N, R_est


def resample_state(psi_old, G_old, G_new):
    """Interpolate a complex wavefunction between two physical grids (bi/tri-linear),
    for gamma-continuation across box sizes. Renormalized on the new grid; returns
    None if the interpolation is degenerate."""
    d = G_old["ndim"]
    x0 = -float(G_old["L"])
    x1 = float(G_old["L"]) - float(G_old["dx"])
    if not (x1 > x0):
        return None
    nrm = [2.0 * (c - x0) / (x1 - x0) - 1.0 for c in G_new["coords"]]   # normalized new coords
    src = torch.stack((psi_old.real, psi_old.imag), dim=0).unsqueeze(0).to(psi_old.real.dtype)
    if d == 2:
        grid = torch.stack((nrm[1], nrm[0]), dim=-1).unsqueeze(0)
    else:
        grid = torch.stack((nrm[2], nrm[1], nrm[0]), dim=-1).unsqueeze(0)
    dst = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    psi = dst[0].to(torch.complex128) + 1j * dst[1].to(torch.complex128)
    nn = norm_of(psi, G_new)
    if not torch.isfinite(nn) or nn.item() <= 1e-14:
        return None
    return psi / nn
