"""becGPP -- a general Gross-Pitaevskii(-Poisson) ground-state solver for
Bose-Einstein condensates in 2D, 3D and quasi-2D geometries.

The same code covers ordinary trapped condensates, rotating vortex states,
self-gravitating (Newtonian) clouds, two- and three-body condensates and their
combinations. Nothing about a particular physical regime is hard-coded.

Quick start::

    from becgpp import default_cfg, run
    cfg = default_cfg(mode="single", dimension="2D", s=2, Omega=0.9,
                      beta2=200, G_C=0, kernel="none", seed="triangular", nseeds=3)
    diag = run(cfg)

See ``becgpp.config.CFG_DEFAULTS`` for every parameter, and ``examples/`` for
ready-made configuration files.
"""
from .constants import CODE_VERSION, DEV
from .config import CFG_DEFAULTS, default_cfg
from . import paths
from .grid import make_grid, geometry, resolve_kernel, auto_grid, resample_state
from .interactions import long_range_phi
from .operators import energy_components, apply_H, compute_residual, observables
from .fields import mass_quantiles, mass_radius, radial_average, droplet_radius
from .thomasfermi import extract_tf
from .seeds import make_seed
from .solvers import ground_state, ground_state_multiseed
from .vortex import vortex_diagnostic, lll_weight
from .diagnostics import diagnostics
from .units import physical_scales, sim_params_from_physical, outputs_to_physical
from .modes import (run, DISPATCH, zip_results,
                    mode_validate, mode_smoke, mode_single, mode_tf_only,
                    mode_sweep, mode_convergence, mode_scan_all, mode_refig)

__version__ = "1.0.0"

__all__ = [
    "CODE_VERSION", "DEV", "CFG_DEFAULTS", "default_cfg", "paths",
    "make_grid", "geometry", "resolve_kernel", "auto_grid", "resample_state",
    "long_range_phi", "energy_components", "apply_H", "compute_residual", "observables",
    "mass_quantiles", "mass_radius", "radial_average", "droplet_radius",
    "extract_tf", "make_seed", "ground_state", "ground_state_multiseed",
    "vortex_diagnostic", "lll_weight", "diagnostics",
    "physical_scales", "sim_params_from_physical", "outputs_to_physical",
    "run", "DISPATCH", "zip_results",
    "mode_validate", "mode_smoke", "mode_single", "mode_tf_only",
    "mode_sweep", "mode_convergence", "mode_scan_all", "mode_refig",
]
