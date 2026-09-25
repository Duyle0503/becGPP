"""Default run configuration.

``CFG_DEFAULTS`` holds every tunable parameter with its default. Build a run
configuration by copying it and overriding the keys you need, e.g.::

    from becgpp import default_cfg
    cfg = default_cfg(mode="single", dimension="3D", G_C=20, kernel="newton")

The defaults describe a small, safe problem: a non-rotating repulsive 2D
condensate in a harmonic trap with no long-range term, on a 128-point grid.
Rotation (``Omega``), the long-range coupling (``G_C``) and larger grids are
switched on explicitly by the user.
"""

CFG_DEFAULTS = dict(
    mode         = "smoke",        # smoke | validate | single | tf_only | sweep | convergence

    # ---- geometry ----
    dimension    = "2D",           # "2D" | "3D" | "quasi2D"
    s            = 2,              # trap power  V = trap_coeff * r^s   (0 = no trap)
    trap_coeff   = 0.5,            # trap prefactor (0.5 = standard 1/2 r^s; e.g. 0.25 = softer quartic)
    Omega        = 0.0,            # rotation about z (in units of the trap frequency)

    # ---- interactions (free signs; set to 0 to switch a term off) ----
    beta2        = 100.0,          # 2-body contact
    beta3        = 0.0,            # 3-body / quintic
    G_C          = 0.0,            # long-range coupling (>0 attractive); 0 = off
    kernel       = "auto",         # auto | newton | log | q2d | none
    l_z          = 0.0,            # quasi2D: axial Gaussian width for kernel q2d (0 -> thin-disc 1/r)

    # ---- grid ----
    L            = 10.0,           # half-box: domain [-L, L)^d
    Ngrid        = 128,
    pad          = 2,              # >=2 : free-space (linear) convolution
    mem_warn_gb  = 12.0,           # warn when the estimated peak memory exceeds this

    # ---- solver ----
    maxit        = 60000,
    res_tol      = 1e-4,           # KKT residual ||(H-mu)psi|| / max(1,|mu|); alias: res=
    energy_tol   = 1e-9,           # windowed relative-energy stop; alias: etol=
    conv_window  = 2000,           # window (iterations) of the energy-plateau stop
    step         = 1.0,            # initial trial step of the line search
    step_max     = 3.0,            # upper clamp of the (Barzilai-Borwein) trial step
    linesearch_max = 8,            # Armijo halvings per iteration
    precond_shift_min = 0.5,       # lower bound on the adaptive Sobolev shift sigma
    cg_beta      = "pr",           # pr | pr_precond | none   (Polak-Ribiere variant)
    cg_restart   = 30,             # reset the CG direction every cg_restart steps
    check        = 200,

    # ---- seeding ----
    seed         = "tf",           # gaussian | tf | triangular  (triangular: 2D, quasi2D and 3D lines)
    seed_winding = 0,
    seed_noise   = 1e-2,
    seed_phase_noise = 5e-2,
    rng          = 0,
    nseeds       = 1,              # >1 : multi-seed, keep the lowest-energy result

    # ---- diagnostics ----
    want_vortices = True,
    want_lll      = True,          # LLL weight; computed only for 2D, s=2, trap_coeff=0.5, Omega>0
    tf_cache      = True,          # cache the Thomas-Fermi extraction per configuration

    # ---- validate mode ----
    validate_N    = 256,           # grid used by the analytic gate (use 512 for publication numbers)

    # ---- sweep mode: vary ANY numeric CFG key over a list ----
    sweep_param  = "G_C",
    sweep_values = [1.0, 2.0, 4.0, 8.0, 16.0],
    sweep_autobox = False,         # if True, pick L,N per point from a radius estimate
    sweep_continuation = True,     # also try the previous point's state as a seed
    # (a sweep uses nseeds exactly like single: nseeds>1 adds fresh seeds per point
    #  and keeps the lowest-energy candidate)

    # ---- convergence mode ----
    conv_Ngrids   = [96, 128, 192],
    conv_box_factors = [1.0, 1.2],
    conv_pads     = [2, 3],

    # ---- output ----
    tag          = "becgpp",
    show_inline  = True,           # display the density inline in a notebook after each run
    save_figs    = True,
    save_ckpt    = True,
    zip_output   = False,
)


# short, user-facing aliases -> canonical CFG keys
_ALIASES = dict(res="res_tol", etol="energy_tol", N="Ngrid", dim="dimension",
                precond_shift="precond_shift_min")


def default_cfg(**overrides):
    """Return a fresh copy of the default configuration, with ``overrides`` applied.

    A few convenience aliases are accepted and mapped to their canonical keys, so
    the residual tolerance can be set directly as ``res=`` (equivalently
    ``res_tol=``); also ``etol`` -> ``energy_tol``, ``N`` -> ``Ngrid``,
    ``dim`` -> ``dimension``. Passing both an alias and its canonical key is an error.
    """
    cfg = dict(CFG_DEFAULTS)
    for alias, canonical in _ALIASES.items():
        if alias in overrides:
            if canonical in overrides:
                raise ValueError(f"pass only one of {alias!r} or {canonical!r}")
            overrides[canonical] = overrides.pop(alias)
    unknown = set(overrides) - set(CFG_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config key(s): {sorted(unknown)}; "
                         f"see becgpp.config.CFG_DEFAULTS for valid keys")
    cfg.update(overrides)
    return cfg
