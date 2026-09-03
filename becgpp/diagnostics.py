"""Diagnostic bundle: writes the TF reference into every record."""
from .operators import observables
from .fields import mass_quantiles, droplet_radius
from .vortex import vortex_diagnostic, lll_weight
from .thomasfermi import extract_tf


def diagnostics(psi, G, cfg):
    d = observables(psi, G, cfg)
    R50, R90, R99 = mass_quantiles(psi.abs()**2, G, (0.50, 0.90, 0.99))
    d.update(R50=R50, R90=R90, R99=R99, R_1pct=droplet_radius(psi.abs()**2, G))
    if cfg.get("want_vortices", True):
        d.update(vortex_diagnostic(psi, G, cfg.get("Omega", 1.0)))
    else:
        d.update(Nv=0, Nplus=0, Nminus=0, Nnet=0)
    d["w_LLL"] = lll_weight(psi, G, 60) if cfg.get("want_lll", False) else float("nan")
    tf = extract_tf(cfg, G)
    if tf is not None:
        d.update(tf_kind=tf.get("kind", ""), tf_R90=tf.get("R90", float("nan")),
                 tf_R99=tf.get("R99", float("nan")), tf_mu=tf.get("mu", float("nan")),
                 tf_rho0=tf.get("rho0", float("nan")), tf_converged=bool(tf.get("converged", False)))
    else:
        d.update(tf_kind="", tf_R90=float("nan"), tf_R99=float("nan"),
                 tf_mu=float("nan"), tf_rho0=float("nan"), tf_converged=False)
    return d
