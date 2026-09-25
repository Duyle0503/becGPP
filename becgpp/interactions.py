"""Long-range interaction: free-space convolution with a cell-averaged kernel.

Kernels (selected by ``G["kernel"]``):

* ``newton`` -- K(r) = 1/r, in 2D or 3D;
* ``log``    -- K(r) = -ln r, 2D (Green function of the planar Laplacian);
* ``q2d``    -- the quasi-two-dimensional reduction of the 3D Newton kernel for
  a frozen Gaussian axial profile of width ``l_z`` (2D only),

      K_eff(u) = exp(u^2/4l_z^2) K_0(u^2/4l_z^2) / (sqrt(2 pi) l_z),

  whose 2D Fourier transform is (2 pi/k) exp(k^2 l_z^2/2) erfc(k l_z/sqrt 2);
  it tends to 1/u for u >> l_z and to a regularized -ln u for u << l_z.

Every kernel is averaged over each grid cell by a 4x4 (4x4x4 in 3D)
Gauss-Legendre rule, which removes the O(dx) error of point sampling near the
origin. The singular self-cell is replaced by its exact cell average:
analytic for ``newton`` and ``log``, a high-order polar quadrature for ``q2d``.
The convolution uses zero padding (pad >= 2) so that the periodic FFT returns
the free-space (open-boundary) result.
"""
import math

import numpy as np
import torch

from .constants import DEV
from .grid import dV

_KERNEL_CACHE = {}

_GL_NODES = (-0.8611363115940526, -0.3399810435848563,
             0.3399810435848563, 0.8611363115940526)
_GL_WTS = (0.3478548451374538, 0.6521451548625461,
           0.6521451548625461, 0.3478548451374538)

# exact cell averages of the singular kernels over a cell of side dx centred on 0
#   <1/r>_2D  = 4 asinh(1) / dx
#   <1/r>_3D  = C3 / dx,   C3 = int_{[-1/2,1/2]^3} d^3u / |u|
#   <-ln r>_2D = -ln dx + 3/2 + (ln 2)/2 - pi/4
SELF_NEWTON_2D = 4.0 * math.asinh(1.0)
SELF_NEWTON_3D = 2.3800773640
SELF_LOG_2D_CONST = 1.5 + 0.5 * math.log(2.0) - 0.25 * math.pi


def _k0e(x):
    """exp(x) K_0(x) for a float64 tensor x >= 0."""
    try:
        return torch.special.scaled_modified_bessel_k0(x)
    except AttributeError:                                  # torch < 2.0
        from scipy.special import k0e
        return torch.as_tensor(k0e(x.cpu().numpy()), device=x.device, dtype=x.dtype)


def _k1e(x):
    """exp(x) K_1(x) for a float64 tensor x > 0."""
    try:
        return torch.special.scaled_modified_bessel_k1(x)
    except AttributeError:                                  # torch < 2.0
        from scipy.special import k1e
        return torch.as_tensor(k1e(x.cpu().numpy()), device=x.device, dtype=x.dtype)


def q2d_kernel(u, l_z):
    """Quasi-2D Newton kernel K_eff(u) for a Gaussian axial width l_z."""
    x = u * u / (4.0 * l_z * l_z)
    return _k0e(x) / (math.sqrt(2.0 * math.pi) * l_z)


def q2d_virial_kernel(u, l_z):
    """W(u) = u dK_eff/du, the kernel entering the virial identity for q2d."""
    x = u * u / (4.0 * l_z * l_z)
    xs = torch.clamp(x, min=1e-300)
    return 2.0 * xs * (_k0e(xs) - _k1e(xs)) / (math.sqrt(2.0 * math.pi) * l_z)


def _polar_self_cell_2d(f, dx, n=48):
    """Exact-to-quadrature average of a radial kernel f(r) over the square cell
    [-dx/2, dx/2]^2, integrating in polar coordinates over its 8 triangles.
    The r dr measure removes the (integrable) singularity at the origin; the
    substitution r = R t^2 smooths r ln r behaviour."""
    t, wt = np.polynomial.legendre.leggauss(n)
    th = 0.5 * (math.pi / 4.0) * (t + 1.0)                   # theta in [0, pi/4]
    wth = 0.5 * (math.pi / 4.0) * wt
    tt = 0.5 * (t + 1.0)                                    # t in [0, 1]
    wtt = 0.5 * wt
    total = 0.0
    for thj, wj in zip(th, wth):
        R = 0.5 * dx / math.cos(thj)
        r = torch.as_tensor(R * tt**2, device=DEV, dtype=torch.float64)
        jac = torch.as_tensor(2.0 * R * R * tt**3, device=DEV, dtype=torch.float64)  # r dr = 2 R^2 t^3 dt
        vals = f(r) * jac
        total += wj * float((vals * torch.as_tensor(wtt, device=DEV)).sum().item())
    return 8.0 * total / (dx * dx)


def _kernel_values(which, G, DXs):
    """Point value of the requested kernel at displacement components DXs."""
    kind = G["kernel"]
    r2 = sum(d * d for d in DXs)
    if which == "W":                                        # virial kernel (q2d only)
        return q2d_virial_kernel(torch.sqrt(r2), G["l_z"])
    if kind == "newton":
        return 1.0 / torch.sqrt(r2)
    if kind == "log":
        return -0.5 * torch.log(r2)
    if kind == "q2d":
        return q2d_kernel(torch.sqrt(r2), G["l_z"])
    raise ValueError(f"unknown kernel {kind!r}")


def _build_kernel_fft(G, which="K"):
    key = ("kfft", which, G["kernel"], G["ndim"], int(G["N"]), round(float(G["dx"]), 14),
           int(G["pad"]), round(float(G.get("l_z", 0.0)), 14), str(DEV))
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    N, dx, ndim, pad = G["N"], G["dx"], G["ndim"], G["pad"]
    M = pad * N
    j = torch.arange(M, device=DEV, dtype=torch.float64)
    dj = torch.where(j <= M // 2, j, j - M) * dx
    if ndim == 2:
        DX, DY = torch.meshgrid(dj, dj, indexing="ij")
        ker = torch.zeros_like(DX)
        for u, wu in zip(_GL_NODES, _GL_WTS):
            for v, wv in zip(_GL_NODES, _GL_WTS):
                ker += 0.25 * wu * wv * _kernel_values(
                    which, G, (DX + 0.5 * dx * u, DY + 0.5 * dx * v))
        if which == "W":
            ker[0, 0] = _polar_self_cell_2d(lambda r: q2d_virial_kernel(r, G["l_z"]), dx)
        elif G["kernel"] == "newton":
            ker[0, 0] = SELF_NEWTON_2D / dx                  # exact <1/r> over the cell
        elif G["kernel"] == "log":
            ker[0, 0] = -math.log(dx) + SELF_LOG_2D_CONST    # exact <-ln r> over the cell
        else:                                               # q2d: polar quadrature of the cell
            ker[0, 0] = _polar_self_cell_2d(lambda r: q2d_kernel(r, G["l_z"]), dx)
    else:
        if G["kernel"] != "newton":
            raise ValueError("only the Newton kernel is available in 3D")
        DX, DY, DZ = torch.meshgrid(dj, dj, dj, indexing="ij")
        ker = torch.zeros_like(DX)
        for u, wu in zip(_GL_NODES, _GL_WTS):
            for v, wv in zip(_GL_NODES, _GL_WTS):
                for w, ww in zip(_GL_NODES, _GL_WTS):
                    ker += (wu * wv * ww / 8.0) / torch.sqrt(
                        (DX + 0.5 * dx * u)**2 + (DY + 0.5 * dx * v)**2 + (DZ + 0.5 * dx * w)**2)
        ker[0, 0, 0] = SELF_NEWTON_3D / dx                   # exact <1/r> over the self cell
        del DX, DY, DZ
    Kf = torch.fft.fftn(ker.to(torch.complex128))
    del ker
    _KERNEL_CACHE[key] = (Kf, M)
    return Kf, M


def kernel_convolution(rho, G, which="K"):
    """Discrete free-space convolution (K * rho)(r_i) = sum_j K_ij rho_j dV."""
    N, ndim = G["N"], G["ndim"]
    Kf, M = _build_kernel_fft(G, which)
    i0 = (M - N) // 2
    shape = (M,) * ndim
    rho_pad = torch.zeros(shape, dtype=torch.complex128, device=rho.device)
    sl = tuple(slice(i0, i0 + N) for _ in range(ndim))
    rho_pad[sl] = rho.to(torch.complex128)
    conv = torch.fft.ifftn(torch.fft.fftn(rho_pad) * Kf).real
    return dV(G) * conv[sl]


def long_range_phi(rho, G, gc):
    """Attractive long-range potential phi = -gc (K * rho), free-space BC."""
    if G["kernel"] == "none" or abs(gc) <= 1e-15:
        return torch.zeros_like(rho)
    return -gc * kernel_convolution(rho, G, "K")


def clear_kernel_cache():
    """Free the cached kernel transforms (e.g. before a large 3D run)."""
    _KERNEL_CACHE.clear()
