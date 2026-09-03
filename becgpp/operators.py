"""Energy functional, mean-field Hamiltonian, residual and observables."""
import math

import torch

from .grid import dV, Lz_op
from .interactions import long_range_phi


def energy_components(p, G, beta2, beta3, Omega, G_C):
    dv = dV(G)
    rho = p.abs()**2
    Ntot = float(p.numel())
    grad2 = (G["K2"] * torch.fft.fftn(p).abs()**2).sum() * dv / Ntot
    Ekin = 0.5 * grad2
    Etrap = (G["V"] * rho).sum() * dv
    Econtact = 0.5 * beta2 * (rho**2).sum() * dv
    Ethree = (beta3 / 3.0) * (rho**3).sum() * dv
    Lz = (p.conj() * Lz_op(p, G)).real.sum() * dv
    Erot = -Omega * Lz
    if G["kernel"] != "none" and abs(G_C) > 1e-15:
        phi = long_range_phi(rho, G, G_C)
        Egrav = 0.5 * (phi * rho).sum() * dv
    else:
        phi = torch.zeros_like(rho)
        Egrav = torch.tensor(0.0, device=p.device)
    E = Ekin + Etrap + Econtact + Ethree + Erot + Egrav
    return dict(E=E, Ekin=Ekin, Etrap=Etrap, Econtact=Econtact,
                Ethree=Ethree, Erot=Erot, Egrav=Egrav, Lz=Lz, phi=phi)


def energy_t(p, G, beta2, beta3, Omega, G_C):
    return energy_components(p, G, beta2, beta3, Omega, G_C)["E"]


def apply_H(psi, G, cfg):
    b2, b3, O, gc = cfg["beta2"], cfg["beta3"], cfg["Omega"], cfg["G_C"]
    kin = torch.fft.ifftn(0.5 * G["K2"] * torch.fft.fftn(psi))
    rho = psi.abs()**2
    phi = long_range_phi(rho, G, gc) if (G["kernel"] != "none" and abs(gc) > 1e-15) else 0.0
    return kin + (G["V"] + b2 * rho + b3 * rho**2 + phi) * psi - O * Lz_op(psi, G)


def compute_residual(psi, G, cfg):
    Hpsi = apply_H(psi, G, cfg)
    mu = (psi.conj() * Hpsi).real.sum() * dV(G)
    resid = torch.sqrt(((Hpsi - mu * psi).abs()**2).sum() * dV(G)).item()
    return resid, mu.item(), resid / max(1.0, abs(mu.item()))


def observables(p, G, cfg):
    b2, b3, O, gc = cfg["beta2"], cfg["beta3"], cfg["Omega"], cfg["G_C"]
    d = float(G["ndim"])
    s = float(G["s"])
    p_exp = float(G["kexp"])
    dv = dV(G)
    comp = energy_components(p, G, b2, b3, O, gc)
    E = comp["E"].item()
    rho = p.abs()**2
    # mu = <psi|H_GP|psi> = E + E_contact + 2 E_three + E_grav  (any dimension)
    mu = E + comp["Econtact"].item() + 2.0 * comp["Ethree"].item() + comp["Egrav"].item()
    Lz = comp["Lz"].item()
    r2mean = (G["R2"] * rho).sum().item() * dv
    # d-dimensional virial:  2Ekin - s Etrap + d Econtact + 2d Ethree + p Egrav = 0
    virial = (2.0 * comp["Ekin"] - s * comp["Etrap"] + d * comp["Econtact"]
              + 2.0 * d * comp["Ethree"] + p_exp * comp["Egrav"]).item()
    vscale = sum(abs(comp[k].item()) for k in ("Ekin", "Etrap", "Econtact", "Ethree", "Egrav"))
    binding_ref = 1.0 if (abs(O - 1.0) < 1e-12 and s == 2 and G["ndim"] == 2) else float("nan")
    out = dict(E=E, mu=mu, Lz=Lz, rrms=math.sqrt(max(r2mean, 0.0)), peak=rho.max().item(),
               E_bind=(E - binding_ref) if math.isfinite(binding_ref) else float("nan"),
               virial=virial, virial_rel=abs(virial) / max(1.0, vscale))
    for k in ("Ekin", "Etrap", "Econtact", "Ethree", "Erot", "Egrav"):
        out[k] = comp[k].item()
    # oblateness: R_perp / R_z (normalized so an isotropic sphere gives 1)
    if G["ndim"] == 3:
        X, Y, Z = G["coords"]
        rperp2 = (((X**2 + Y**2) * rho).sum().item()) * dv
        z2 = ((Z**2 * rho).sum().item()) * dv
        out["R_perp_rms"] = math.sqrt(max(rperp2, 0.0))
        out["R_z_rms"] = math.sqrt(max(z2, 0.0))
        out["oblateness"] = (out["R_perp_rms"] / math.sqrt(2.0)) / max(out["R_z_rms"], 1e-30)
    else:
        out["R_perp_rms"] = out["R_z_rms"] = float("nan")
        out["oblateness"] = float("nan")
    return out
