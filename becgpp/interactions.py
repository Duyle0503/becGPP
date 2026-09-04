"""Long-range interaction: free-space convolution with a cell-averaged kernel.

The kernel is averaged over each grid cell by a 4x4 Gauss-Legendre rule (removing
the O(dx) error of point sampling near the origin), with an analytic self-cell
term. The convolution uses zero-padding (pad >= 2) so the periodic FFT returns
the free-space (open boundary) result.
"""
import math

import torch

from .constants import DEV
from .grid import dV

_KERNEL_CACHE = {}


def _build_kernel_fft(G):
    key = ("kfft", G["kernel"], G["ndim"], int(G["N"]), round(float(G["dx"]), 14),
           int(G["pad"]), str(DEV))
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    N, dx, ndim, pad = G["N"], G["dx"], G["ndim"], G["pad"]
    M = pad * N
    j = torch.arange(M, device=DEV, dtype=torch.get_default_dtype())
    dj = torch.where(j <= M // 2, j, j - M) * dx
    nodes = (-0.8611363115940526, -0.3399810435848563,
             0.3399810435848563, 0.8611363115940526)
    wts = (0.3478548451374538, 0.6521451548625461,
           0.6521451548625461, 0.3478548451374538)
    if ndim == 2:
        DX, DY = torch.meshgrid(dj, dj, indexing="ij")
        if G["kernel"] == "newton":                       # 1/r, GL cell-average
            ker = torch.zeros_like(DX)
            for u, wu in zip(nodes, wts):
                for v, wv in zip(nodes, wts):
                    ker += 0.25 * wu * wv / torch.sqrt((DX + 0.5 * dx * u)**2 + (DY + 0.5 * dx * v)**2)
            ker[0, 0] = 4.0 * math.asinh(1.0) / dx        # exact <1/r> over the cell
        else:                                             # -ln r, GL cell-average
            ker = torch.zeros_like(DX)
            for u, wu in zip(nodes, wts):
                for v, wv in zip(nodes, wts):
                    ker += 0.25 * wu * wv * (-0.5) * torch.log((DX + 0.5 * dx * u)**2 + (DY + 0.5 * dx * v)**2)
        Kf = torch.fft.fftn(ker.to(torch.complex128))
    else:                                                 # 3D Newton 1/r, GL cell-average
        DX, DY, DZ = torch.meshgrid(dj, dj, dj, indexing="ij")
        ker = torch.zeros_like(DX)
        for u, wu in zip(nodes, wts):
            for v, wv in zip(nodes, wts):
                for w, ww in zip(nodes, wts):
                    ker += (wu * wv * ww / 8.0) / torch.sqrt(
                        (DX + 0.5 * dx * u)**2 + (DY + 0.5 * dx * v)**2 + (DZ + 0.5 * dx * w)**2)
        ker[0, 0, 0] = 2.380077 / dx                      # analytic <1/r> over the self cell
        Kf = torch.fft.fftn(ker.to(torch.complex128))
    _KERNEL_CACHE[key] = (Kf, M)
    return Kf, M


def long_range_phi(rho, G, gc):
    """Attractive long-range potential phi = -gc (K * rho), free-space BC."""
    if G["kernel"] == "none" or abs(gc) <= 1e-15:
        return torch.zeros_like(rho)
    N, ndim = G["N"], G["ndim"]
    Kf, M = _build_kernel_fft(G)
    i0 = (M - N) // 2
    shape = (M,) * ndim
    rho_pad = torch.zeros(shape, dtype=torch.complex128, device=rho.device)
    sl = tuple(slice(i0, i0 + N) for _ in range(ndim))
    rho_pad[sl] = rho.to(torch.complex128)
    conv = torch.fft.ifftn(torch.fft.fftn(rho_pad) * Kf).real
    return -gc * dV(G) * conv[sl]
