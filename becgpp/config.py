"""Default run configuration.

``CFG_DEFAULTS`` holds every tunable parameter with its default. Build a run
configuration by copying it and overriding the keys you need, e.g.::

    from becgpp import default_cfg
    cfg = default_cfg(mode="single", dimension="3D", G_C=20, kernel="newton")
"""

CFG_DEFAULTS = dict(
    mode         = "smoke",        # validate | smoke | single | tf_only | sweep | convergence | scan_all | refig
    scan_profile = "full",         # scan_all: "full" (paper data) | "quick" (fast dry run)

    # ---- geometry ----
    dimension    = "2D",           # "2D" | "3D" | "quasi2D"
    s            = 2,              # trap power  V = trap_coeff * r^s   (0 = no trap)
    trap_coeff   = 0.5,            # trap prefactor (0.5 = standard 1/2 r^s; e.g. 0.25 = softer quartic)
    Omega        = 1.0,            # rotation about z

    # ---- interactions (free signs; set to 0 to switch a term off) ----
    beta2        = 100.0,          # 2-body contact
    beta3        = 0.0,            # 3-body / quintic
    G_C          = 5.0,            # long-range coupling (>0 attractive)
    kernel       = "auto",         # auto | newton | log | none

    # ---- grid ----
    L            = 16.0,           # half-box: domain [-L, L)^d
    Ngrid        = 256,
    pad          = 2,              # >=2 : free-space (linear) convolution

    # ---- solver ----
    maxit        = 60000,
    res_tol      = 1e-4,           # KKT residual ||(H-mu)psi|| / max(1,|mu|)
    energy_tol   = 1e-9,
    step         = 0.5,
    step_max     = 2.0,
    precond_shift = 1.0,
    linesearch_max = 12,
    check        = 200,
    conv_window  = 2000,

    # ---- seeding ----
    seed         = "tf",           # gaussian | tf | triangular  (triangular: 2D only)
    seed_winding = 0,
    seed_noise   = 1e-2,
    seed_phase_noise = 5e-2,
    rng          = 0,
    nseeds       = 1,              # >1 : multi-seed, keep lowest-energy converged branch

    # ---- diagnostics ----
    want_vortices = True,          # 2D / quasi2D only
    want_lll      = True,          # LLL weight; meaningful at Omega=1, 2D

    # ---- sweep mode: vary ANY numeric CFG key over a list ----
    sweep_param  = "G_C",
    sweep_values = [1.0, 2.0, 4.0, 8.0, 16.0],
    sweep_autobox = False,         # if True, pick L,N per point from a radius estimate

    # ---- convergence mode ----
    conv_Ngrids   = [192, 256, 384],
    conv_box_factors = [1.0, 1.2],
    conv_pads     = [2, 3],

    # ---- output ----
    tag          = "becgpp",
    show_inline  = True,           # display the density inline in a notebook after each run
    save_figs    = True,
    save_ckpt    = True,
    zip_output   = True,
)


def default_cfg(**overrides):
    """Return a fresh copy of the default configuration, with ``overrides`` applied."""
    cfg = dict(CFG_DEFAULTS)
    cfg.update(overrides)
    return cfg
