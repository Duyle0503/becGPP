"""Grid construction, spectral operators, and grid utilities."""
import math

import torch
import torch.nn.functional as F

from .constants import DEV


def geometry(cfg):
    """Return (ndim, quasi) for the configured geometry: "2D", "3D" or "quasi2D"
    (case-insensitive)."""
    g = str(cfg.get("dimension", "2D")).strip().lower()
    if g == "2d":
        return 2, False
    if g == "3d":
        return 3, False
    if g == "quasi2d":
        return 2, True
    raise ValueError(f"unknown dimension {cfg.get('dimension')!r}; use '2D', '3D' or 'quasi2D'")


def resolve_kernel(cfg):
    """Resolve the kernel choice into (kind, tag) with kind in
    'newton' | 'log' | 'q2d' | 'none'.

    ``kernel="auto"`` picks the natural kernel of the geometry: ``log`` (the
    planar Green function) in 2D, ``newton`` in 3D, and in ``quasi2D`` the
    reduced Newton kernel ``q2d`` of axial width ``l_z`` (or its thin-disc limit
    ``newton`` = 1/r when ``l_z`` = 0)."""
    ndim, quasi = geometry(cfg)
    k = str(cfg.get("kernel", "auto")).lower()
    if abs(float(cfg.get("G_C", 0.0))) <= 1e-15 or k == "none":
        return "none", 0
    lz = float(cfg.get("l_z", 0.0))
    if k == "auto":
        if ndim == 3:
            k = "newton"
        elif quasi:
            k = "q2d" if lz > 0 else "newton"
        else:
            k = "log"
    if k == "newton":
        return "newton", 1          # 1/|r|: homogeneous of degree -1
    if k == "log":
        if ndim != 2:
            raise ValueError("kernel='log' is two-dimensional only")
        return "log", 0             # -ln|r|: virial term -G_C/2 (see operators.observables)
    if k == "q2d":
        if ndim != 2:
            raise ValueError("kernel='q2d' needs dimension '2D' or 'quasi2D'")
        if lz <= 0:
            raise ValueError("kernel='q2d' needs an axial width l_z > 0")
        return "q2d", 0             # not homogeneous: virial uses the W = u K' kernel
    raise ValueError(f"unknown kernel {k!r}")


def _memory_estimate_gb(ndim, N, pad, kernel):
    """Rough peak device memory (GB) of one ground-state solve."""
    n_field = float(N) ** ndim
    base = 14 * 16 * n_field                               # ~14 complex128 fields
    if kernel != "none":
        m = float(pad * N) ** ndim
        base += 6 * 16 * m                                 # padded convolution buffers
        base += 4 * 8 * m                                  # kernel construction (meshgrids)
    return base / 1e9


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
    pad = max(2, int(cfg.get("pad", 2)))
    mem = _memory_estimate_gb(ndim, N, pad, kkind)
    if mem > float(cfg.get("mem_warn_gb", 12.0)):
        print(f"[warn] estimated peak memory ~{mem:.0f} GB for N={N} ({ndim}D, kernel={kkind}, "
              f"pad={pad}); reduce Ngrid if the device runs out of memory.")
    return dict(ndim=ndim, quasi=quasi, coords=coords, kcoords=kcoords,
                X=coords[0], Y=coords[1], KX=kcoords[0], KY=kcoords[1],
                R2=R2, V=V, K2=K2, s=s, trap_coeff=tc, dx=dx, N=N, L=L,
                kernel=kkind, kexp=kexp, l_z=float(cfg.get("l_z", 0.0)), pad=pad,
                mem_estimate_gb=mem)


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
    if gc > 1e-15 and b2 > 0:
        R_est = 0.5 * b2 / gc                              # attractive long range: ~ 1/gamma
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
    n_old, dx_old = int(G_old["N"]), float(G_old["dx"])
    x0 = -(n_old // 2) * dx_old                    # first/last node, as in make_grid
    x1 = x0 + (n_old - 1) * dx_old                 # (x0 = -L only for even N)
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
