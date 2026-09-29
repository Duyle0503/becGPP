"""Diagnostic bundle: writes the TF reference into every record."""
import math

from .operators import observables
from .fields import mass_quantiles, droplet_radius
from .vortex import vortex_diagnostic, lll_weight
from .thomasfermi import extract_tf


def _lll_applicable(G, cfg):
    """The LLL basis z^m exp(-r^2/2) is that of the isotropic harmonic trap
    V = r^2/2 in 2D; the weight is only reported for rotating runs there."""
    return (G["ndim"] == 2 and abs(float(G["s"]) - 2.0) < 1e-12
            and abs(float(G.get("trap_coeff", 0.5)) - 0.5) < 1e-12
            and abs(float(cfg.get("Omega", 0.0))) > 0.0)


def diagnostics(psi, G, cfg):
    d = observables(psi, G, cfg)
    R50, R90, R99 = mass_quantiles(psi.abs()**2, G, (0.50, 0.90, 0.99))
    d.update(R50=R50, R90=R90, R99=R99, R_1pct=droplet_radius(psi.abs()**2, G))
    if cfg.get("want_vortices", True):
        d.update(vortex_diagnostic(psi, G, cfg.get("Omega", 0.0)))
    else:
        d.update(Nv=0, Nplus=0, Nminus=0, Nnet=0)
    lll_ok = cfg.get("want_lll", True) and _lll_applicable(G, cfg)
    d["w_LLL"] = lll_weight(psi, G, 60, float(cfg.get("Omega", 0.0))) if lll_ok else float("nan")
    # Standard mean-field LLL criterion (Ho 2001; Fetter 2009): the interaction
    # energy g*n_peak must be small compared with the gap to the next Landau level,
    # which in the rotating frame is (1 + Omega) hbar*omega_perp (-> 2 at Omega = 1);
    # 1 - Omega is the spacing *within* the LLL, not the gap. LLL regime: lll_param << 1.
    O = abs(float(cfg.get("Omega", 0.0)))
    d["lll_param"] = (abs(float(cfg["beta2"])) * d["peak"] / (1.0 + O)) if lll_ok else float("nan")
    tf = extract_tf(cfg, G)
    if tf is not None:
        d.update(tf_kind=tf.get("kind", ""), tf_R90=tf.get("R90", float("nan")),
                 tf_R99=tf.get("R99", float("nan")), tf_mu=tf.get("mu", float("nan")),
                 tf_rho0=tf.get("rho0", float("nan")), tf_converged=bool(tf.get("converged", False)))
    else:
        d.update(tf_kind="", tf_R90=float("nan"), tf_R99=float("nan"),
                 tf_mu=float("nan"), tf_rho0=float("nan"), tf_converged=False)
    d["mem_estimate_gb"] = G.get("mem_estimate_gb", float("nan"))
    if not math.isfinite(d["R90"]):
        d["points_per_R90"] = float("nan")
    else:
        d["points_per_R90"] = d["R90"] / G["dx"]
    return d
