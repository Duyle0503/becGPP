"""Publication-style figures and optional inline notebook display.

Saved figures have no titles, concise labels, and a labelled colour bar on every
map. Importing this module sets the shared Matplotlib style.
"""
import os
import io
import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import paths
from .fields import center_of_mass, mass_radius, radial_average, droplet_radius, gaussian_smooth
from .thomasfermi import extract_tf
from .vortex import vortex_diagnostic

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "font.family": "serif", "mathtext.fontset": "cm",
    "axes.linewidth": 0.8, "axes.labelsize": 10,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.4,
})


def _savefig(fig, base):
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png")
    plt.close(fig)


def _imshow(ax, field2d, L, cmap="inferno", label=r"$n$", xl=r"$x$", yl=r"$y$",
            vmin=None, vmax=None):
    im = ax.imshow(field2d.T, origin="lower", extent=[-L, L, -L, L], cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label)
    return im


def save_state_figs(psi, G, cfg, rid):
    # Publication style: no titles, concise labels, a labelled colorbar on every map.
    L = G["L"]
    rho = psi.abs()**2
    if G["ndim"] == 2:
        dens = rho.cpu().numpy()
        fig, ax = plt.subplots(figsize=(3.4, 2.9))
        _imshow(ax, dens, L, label=r"$n$")
        _savefig(fig, os.path.join(paths.FIG_DIR, f"{rid}_dens"))
        ph = torch.angle(psi).cpu().numpy()
        fig, ax = plt.subplots(figsize=(3.4, 2.9))
        im = ax.imshow(ph.T, origin="lower", extent=[-L, L, -L, L], cmap="twilight",
                       vmin=-math.pi, vmax=math.pi, aspect="equal")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"$\arg\psi$")
        cb.set_ticks([-math.pi, 0, math.pi])
        cb.set_ticklabels([r"$-\pi$", "$0$", r"$\pi$"])
        _savefig(fig, os.path.join(paths.FIG_DIR, f"{rid}_phase"))
    else:
        N = G["N"]
        mid = N // 2
        dens = rho.cpu().numpy()
        vmax = float(dens.max())
        slabs = [("xy", dens[:, :, mid], r"$x$", r"$y$"),
                 ("xz", dens[:, mid, :], r"$x$", r"$z$"),
                 ("yz", dens[mid, :, :], r"$y$", r"$z$")]
        fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.8), constrained_layout=True)
        im = None
        for ax, (nm, sl, xl, yl) in zip(axs, slabs):
            im = ax.imshow(sl.T, origin="lower", extent=[-L, L, -L, L], cmap="inferno",
                           vmin=0.0, vmax=vmax, aspect="equal")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
        cb = fig.colorbar(im, ax=axs, fraction=0.026, pad=0.02)
        cb.set_label(r"$n$")
        fig.savefig(os.path.join(paths.FIG_DIR, f"{rid}_dens_slices") + ".pdf")
        fig.savefig(os.path.join(paths.FIG_DIR, f"{rid}_dens_slices") + ".png")
        plt.close(fig)


def save_tf_comparison(psi, G, cfg, rid):
    tf = extract_tf(cfg, G)
    rho = psi.abs()**2
    center = center_of_mass(rho, G)
    R90 = mass_radius(rho, G, 0.90, center=center)
    R = droplet_radius(rho, G)
    rmax = 1.3 * R if (R and np.isfinite(R)) else (1.3 * R90 if np.isfinite(R90) and R90 > 0 else G["L"])
    # Coarse-grain over the vortex lattice ONLY when a lattice is present (Nv>=6),
    # and cap the smoothing scale well below the cloud (0.15 R90) -- otherwise a
    # compact/vortex-free droplet is washed into a flat line. Radial averaging
    # already does the azimuthal averaging, so the raw density is the envelope.
    Nv = vortex_diagnostic(psi, G, cfg.get("Omega", 1.0))["Nv"] if G["ndim"] == 2 else 0
    if G["ndim"] == 2 and Nv >= 6 and np.isfinite(R90) and R90 > 0:
        O = max(float(cfg.get("Omega", 1.0)), 1e-12)
        a_v = math.sqrt(2.0 * math.pi / (math.sqrt(3.0) * O))
        sig = min(0.8 * a_v, 0.15 * R90)
        rho_cg = torch.clamp(gaussian_smooth(rho, G, sig), min=0.0)
    else:
        rho_cg = rho
    r_p, rho_p = radial_average(rho_cg, G, rmax=rmax, center=center)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.plot(r_p, rho_p, "-", color="#1f4e79", label="GPE")
    if tf is not None and tf.get("n") is not None:
        rt, pt = radial_average(tf["n"], G, rmax=rmax)
        ax.plot(rt, pt, "--", color="#c1272d", label="TF")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$n(r)$")
    ax.set_xlim(0, rmax)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", handlelength=1.4)
    _savefig(fig, os.path.join(paths.FIG_DIR, f"{rid}_TF"))


def show_density(psi, G, cfg, title=None):
    """Display the density inline in a notebook (Kaggle/Jupyter). No-op elsewhere.
    2D: single density map. 3D: xy / xz / yz central slices."""
    if not cfg.get("show_inline", True):
        return
    try:
        from IPython.display import Image, display
    except Exception:
        return
    L = G["L"]
    rho = psi.abs()**2
    if G["ndim"] == 2:
        fig, ax = plt.subplots(figsize=(4.0, 3.4))
        _imshow(ax, rho.cpu().numpy(), L)
    else:
        N = G["N"]
        mid = N // 2
        dens = rho.cpu().numpy()
        vmax = float(dens.max())
        slabs = [(dens[:, :, mid], r"$x$", r"$y$"), (dens[:, mid, :], r"$x$", r"$z$"),
                 (dens[mid, :, :], r"$y$", r"$z$")]
        fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.0), constrained_layout=True)
        im = None
        for ax, (sl, xl, yl) in zip(axs, slabs):
            im = ax.imshow(sl.T, origin="lower", extent=[-L, L, -L, L], cmap="inferno",
                           vmin=0.0, vmax=vmax, aspect="equal")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
        cb = fig.colorbar(im, ax=axs, fraction=0.026, pad=0.02)
        cb.set_label(r"$n$")
    # 'title' is a monitoring label only (inline display, never a saved paper figure)
    if title:
        fig.suptitle(title, fontsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    display(Image(data=buf.getvalue()))


def show_tf_density(tf, G, cfg, title=None):
    """Display the TF reference density inline (2D map, or 3D xy slice)."""
    if not cfg.get("show_inline", True) or tf is None or tf.get("n") is None:
        return
    try:
        from IPython.display import Image, display
    except Exception:
        return
    L = G["L"]
    n = tf["n"]
    if G["ndim"] == 2:
        fig, ax = plt.subplots(figsize=(4.2, 3.5))
        _imshow(ax, n.cpu().numpy(), L, label=r"$n_{\rm TF}$")
    else:
        N = G["N"]
        mid = N // 2
        fig, ax = plt.subplots(figsize=(4.2, 3.5))
        im = ax.imshow(n.cpu().numpy()[:, :, mid].T, origin="lower", extent=[-L, L, -L, L], cmap="inferno")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if title:
        ax.set_title(title, fontsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    display(Image(data=buf.getvalue()))


_PARAM_SYM = {"G_C": r"$G_C$", "Omega": r"$\Omega$", "beta2": r"$\beta_2$",
              "beta3": r"$\beta_3$", "s": r"$s$", "L": r"$L$", "Ngrid": r"$N$"}


def sweep_figs(rows, param, label=None):
    if not rows:
        return
    label = label or param
    xlab = _PARAM_SYM.get(param, param)
    rows = sorted(rows, key=lambda r: r[param])          # plot in ascending param order
    x = [r[param] for r in rows]
    logx = all(isinstance(v, (int, float)) and v > 0 for v in x) and (max(x) / max(min(x), 1e-30) > 20)
    for key, lab, name in (("E", r"$E$", "E"), ("R90", r"$R_{90}$", "R90"),
                           ("Lz", r"$\langle L_z\rangle$", "Lz")):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        ax.plot(x, [r[key] for r in rows], "o-", ms=4, label="GPE")
        if key == "R90":
            ax.plot(x, [r["tf_R90"] for r in rows], "s--", ms=3, color="#c1272d", label="TF")
            ax.legend(loc="best", handlelength=1.4)
        if logx:
            ax.set_xscale("log")
        if key == "R90" and logx:
            ax.set_yscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel(lab)
        _savefig(fig, os.path.join(paths.FIG_DIR, f"sweep_{label}_{name}"))
