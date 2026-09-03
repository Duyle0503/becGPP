"""Optional unit conversion: physical (SI) <-> dimensionless "trap units".

The solver runs in trap units (hbar = m = omega_perp = 1), with all couplings
dimensionless and int|psi|^2 = 1. These helpers convert real physical inputs to
the dimensionless CFG couplings, and simulation outputs back to SI. They do NOT
touch the solver -- the runs stay dimensionless; use them only to prepare inputs
and interpret outputs. Energy convention: E_contact = (beta2/2) int rho^2.

Length scale a_ho = sqrt(hbar/(m*omega_perp)); energy scale hbar*omega_perp;
time scale 1/omega_perp.  Standard mean-field maps:
    3D contact      : beta2 = 4*pi*N*a_s/a_ho
    2D pancake      : beta2 = sqrt(8*pi)*N*a_s/l_z,  l_z = sqrt(hbar/(m*omega_z))
    3D self-gravity : G_C   = G*m^2*N^2/(a_ho*hbar*omega_perp)   (Newton 1/r)
    quintic/3-body  : beta3 supplied via G3 (regime specific; see the paper)
NOTE the cubic-quintic literature often writes G2=2*beta2, G3=2*beta3 (energy
(G2/4)rho^2 + (G3/6)rho^3); to reproduce it set beta2=G2/2, beta3=G3/2.
"""
import math

from .constants import HBAR, G_NEWTON


def physical_scales(m, omega_perp):
    """SI scales for trap units: length a_ho (m), energy (J and Hz), time (s)."""
    a_ho = math.sqrt(HBAR / (m * omega_perp))
    return dict(a_ho_m=a_ho, energy_J=HBAR * omega_perp, energy_Hz=omega_perp / (2 * math.pi),
                time_s=1.0 / omega_perp, m_kg=m, omega_perp=omega_perp)


def sim_params_from_physical(regime, m, omega_perp, N, a_s=0.0, omega_z=None,
                             G=G_NEWTON, beta3=0.0, verbose=True):
    """Return dimensionless CFG couplings for a chosen physical regime, plus the
    SI scales for converting outputs back.
       regime: 'bec3d' | 'bec2d_pancake' | 'selfgrav3d'
       m [kg], omega_perp [rad/s], N atoms, a_s [m], omega_z [rad/s] (pancake), G [SI]."""
    sc = physical_scales(m, omega_perp)
    a_ho = sc["a_ho_m"]
    out = dict(beta2=0.0, beta3=float(beta3), G_C=0.0, Omega=0.0, scales=sc, regime=regime)
    if regime == "bec3d":
        out["beta2"] = 4.0 * math.pi * N * a_s / a_ho
        out["dimension"] = "3D"
    elif regime == "bec2d_pancake":
        if omega_z is None:
            raise ValueError("bec2d_pancake needs omega_z")
        l_z = math.sqrt(HBAR / (m * omega_z))
        out["beta2"] = math.sqrt(8.0 * math.pi) * N * a_s / l_z
        out["dimension"] = "quasi2D"
        out["l_z_m"] = l_z
    elif regime == "selfgrav3d":
        out["G_C"] = G * m * m * N * N / (a_ho * HBAR * omega_perp)
        out["kernel"] = "newton"
        out["dimension"] = "3D"
        if a_s:
            out["beta2"] = 4.0 * math.pi * N * a_s / a_ho
    else:
        raise ValueError(f"unknown regime {regime!r}")
    if verbose:
        print(f"[units:{regime}] a_ho={a_ho:.3e} m  hbar*w={sc['energy_J']:.3e} J "
              f"({sc['energy_Hz']:.3g} Hz)  1/w={sc['time_s']:.3e} s")
        print(f"[units:{regime}] -> beta2={out['beta2']:.4g}  beta3={out['beta3']:.4g}  G_C={out['G_C']:.4g}")
    return out


def outputs_to_physical(scales, E=None, mu=None, R=None):
    """Convert dimensionless outputs to SI using scales from physical_scales().
    Energies are PER PARTICLE (times hbar*omega_perp); lengths times a_ho."""
    res = {}
    if E is not None:
        res["E_J"] = E * scales["energy_J"]
        res["E_Hz"] = E * scales["energy_Hz"]
    if mu is not None:
        res["mu_J"] = mu * scales["energy_J"]
        res["mu_Hz"] = mu * scales["energy_Hz"]
    if R is not None:
        res["R_m"] = R * scales["a_ho_m"]
    return res
