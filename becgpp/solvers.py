"""Ground-state solver: normalized preconditioned conjugate-gradient descent.

Adaptive Sobolev preconditioner, Polak-Ribiere CG with periodic restart, a
Barzilai-Borwein trial step, and a backtracking Armijo line search on the full
rotating GP energy (no LLL projection). Multi-seed selection keeps the lowest
converged branch.
"""
import math
import time

import torch

from .constants import DEV
from .config import default_cfg
from .grid import make_grid, dV, norm_of
from .interactions import long_range_phi
from .operators import apply_H, energy_components, compute_residual, observables
from .seeds import make_seed


def _inner_re(u, v, dv):
    return ((u.conj() * v).real.sum() * dv).item()


def _proj_tangent(v, psi, dv):
    return v - psi * ((psi.conj() * v).sum() * dv)


def _preconditioner(grad, psi, G, cfg, phi=None):
    """Adaptive Sobolev preconditioner: z = (1/2 k^2 + sigma)^-1 grad, with sigma
    the median of the local potential over the bulk. Projected onto the tangent
    space at psi. This shift tracks the actual stiffness (beta2 n dominates),
    which fixed shift=1 does not -- essential for fast 3D convergence."""
    rho = psi.abs()**2
    b2, b3, gc = cfg["beta2"], cfg["beta3"], cfg["G_C"]
    local = G["V"] + 2.0 * b2 * rho + 3.0 * b3 * rho**2
    if G["kernel"] != "none" and abs(gc) > 1e-15:
        local = local + (phi if phi is not None else long_range_phi(rho, G, gc))
    mask = rho > 0.05 * rho.max()
    vals = local[mask]
    floor = float(cfg.get("precond_shift_min", 0.5))
    sigma = float(torch.median(vals).item()) if vals.numel() else floor
    sigma = max(sigma, floor)                      # keep the metric positive definite
    denom = 0.5 * G["K2"] + sigma
    z = torch.fft.ifftn(torch.fft.fftn(grad) / denom)
    return _proj_tangent(z, psi, dV(G)), sigma


def _translation_invariant(G, cfg):
    """True when the discrete energy is invariant under periodic shifts of the
    grid: no trap, no rotation and no (zero-padded, hence non-periodic)
    long-range term."""
    no_trap = float(G["s"]) == 0.0 or float(G.get("trap_coeff", 0.5)) == 0.0
    no_lr = G["kernel"] == "none" or abs(float(cfg["G_C"])) <= 1e-15
    return no_trap and no_lr and float(cfg.get("Omega", 0.0)) == 0.0


def recentre_periodic(psi, G):
    """Shift psi by whole grid cells so that its periodic (circular) centre of
    mass sits at the box centre. For a translation-invariant problem this is an
    exact symmetry of the discrete energy; it only undoes drift along the
    translation mode, which otherwise can carry a free droplet across the
    periodic boundary and spoil radii and figures. Returns (psi, shifts)."""
    rho = psi.abs() ** 2
    shifts = []
    for ax in range(psi.dim()):
        other = tuple(d for d in range(psi.dim()) if d != ax)
        m = rho.sum(dim=other)
        n = m.numel()
        th = 2.0 * math.pi * torch.arange(n, device=m.device, dtype=m.dtype) / n
        c = (m * torch.cos(th)).sum().item()
        s = (m * torch.sin(th)).sum().item()
        if c * c + s * s <= 1e-24 * m.sum().item() ** 2:   # no preferred position
            shifts.append(0)
            continue
        j_com = (math.atan2(s, c) % (2.0 * math.pi)) * n / (2.0 * math.pi)
        shifts.append(int(round(n // 2 - j_com)) % n)
    if any(shifts):
        psi = torch.roll(psi, shifts=tuple(shifts), dims=tuple(range(psi.dim())))
    return psi, tuple(shifts)


def ground_state(cfg, G=None, psi0=None, verbose=True):
    """Riemannian preconditioned conjugate-gradient minimization of the rotating
    GP energy on the unit-norm sphere: adaptive Sobolev preconditioner,
    Polak-Ribiere CG with periodic restart, a Barzilai-Borwein trial step, and a
    backtracking Armijo line search. Converges in O(1e3) steps where plain
    preconditioned gradient descent stalls at res ~ 1e-2 (esp. in 3D)."""
    cfg = default_cfg(**cfg)
    if G is None:
        G = make_grid(cfg)
    psi = psi0.to(device=DEV,dtype=torch.complex128) if psi0 is not None else make_seed(G, cfg)
    if tuple(psi.shape) != tuple(G['R2'].shape) or not torch.isfinite(psi).all().item():
        raise ValueError('psi0 must be finite and have the same shape as the grid')
    dv = dV(G)
    initial_norm = norm_of(psi,G)
    if not torch.isfinite(initial_norm).item() or initial_norm.item() <= 0:
        raise ValueError('psi0 must have a finite, nonzero norm')
    psi = psi / initial_norm
    maxit = int(cfg["maxit"])
    res_tol = float(cfg["res_tol"])
    etol = float(cfg.get("energy_tol", 1e-9))
    check = int(cfg.get("check", 200))
    alpha = float(cfg.get("step", 1.0))                  # initial trial step
    alpha_min, alpha_max = min(1e-12,float(cfg['step_max'])), float(cfg.get("step_max", 3.0))
    ls_max = int(cfg.get("linesearch_max", 8))          # Armijo halvings per iteration
    # A failed CG search is retried with a fresh preconditioned gradient and a
    # full backtrack to alpha_min. Do not recycle stale CG/BB history on retries.
    stall_max = max(1, int(cfg.get("linesearch_stall", 10)))
    c1 = 1e-4
    shrink, growth = 0.5, 1.1
    bb_min, bb_max, bb_mix = 0.05, alpha_max, 0.5
    restart_period = int(cfg.get("cg_restart", 30))
    # Polak-Ribiere coefficient:
    # "pr_precond" (default since 1.2.0) uses the preconditioned (Sobolev) inner product,
    #   beta = <z, g - g_old> / <z_old, g_old>          (standard preconditioned PR)
    # "pr" (default in 1.0-1.1, used for the v1.1 paper runs) uses the plain gradient,
    #   beta = <g, g - g_old> / <g_old, g_old>
    cg_beta = str(cfg.get("cg_beta", "pr_precond")).lower()
    if cg_beta not in ("pr", "pr_precond", "none"):
        raise ValueError(f"cg_beta must be 'pr', 'pr_precond' or 'none', got {cg_beta!r}")
    # windowed energy convergence: stop when the NET relative energy drop over a
    # window falls below energy_tol. This is a fallback witness for under-resolved
    # grids, where the residual ||(H-mu)psi|| can stall at a resolution-set floor
    # above res_tol while the energy has already settled. On adequately resolved
    # grids the residual criterion fires first (stop_reason="converged").
    conv_window = int(cfg.get("conv_window", 2000))
    E_ref = None
    it_ref = 0
    b2, b3, O, gc = cfg["beta2"], cfg["beta3"], cfg["Omega"], cfg["G_C"]

    def energy_phi(p):
        # one zero-padded convolution per call; its potential is reused by the
        # residual and the preconditioner of the same state (1.2.0)
        comp = energy_components(p, G, b2, b3, O, gc)
        return comp["E"].item(), comp["phi"]

    def residual(p, phi=None):
        Hp = apply_H(p, G, cfg, phi=phi)
        mu = (p.conj() * Hp).sum().real * dv
        g = Hp - mu * p
        ra = torch.sqrt((g.abs()**2).sum() * dv).item()
        return g, mu, ra, ra / max(1.0, abs(mu.item()))

    E, phi = energy_phi(psi)
    t0 = time.time()
    # optional convergence history (wall time, iteration, energy, residual); the
    # .item() calls above/below synchronize the device, so the times are honest
    record = bool(cfg.get("record_trace", False))
    grad, mu, _, resid_rel = residual(psi, phi)
    trace = [(0.0,0,E,resid_rel)] if record else None
    z, sigma = _preconditioner(grad, psi, G, cfg, phi)
    direction = z.clone()
    z_prev = z
    converged = resid_rel < res_tol
    stop = 'converged' if converged else 'maxit'
    it = 0
    E_prev = E
    psi_prev = psi
    grad_prev = grad
    best_resid = resid_rel
    best_psi = psi.clone()
    E_ref = E
    n_fail = 0                                           # consecutive failed line searches
    n_iter = 0                                           # iterations performed
    force_restart = False
    ls_restarts = 0
    for it in range(0 if converged else maxit):
        n_iter = it + 1
        z, sigma = _preconditioner(grad, psi, G, cfg, phi)
        restarted = force_restart
        force_restart = False
        if it == 0 or restarted or cg_beta == "none" or (restart_period > 0 and it % restart_period == 0):
            beta = 0.0
        else:
            y = grad - grad_prev
            if cg_beta == "pr_precond":
                den = max(_inner_re(z_prev, grad_prev, dv), 1e-300)
                beta = max(0.0, _inner_re(z, y, dv) / den)
            else:
                den = max(_inner_re(grad_prev, grad_prev, dv), 1e-300)
                beta = max(0.0, _inner_re(grad, y, dv) / den)
            if not math.isfinite(beta):
                beta = 0.0
        dirc = _proj_tangent(z + beta * direction, psi, dv)
        gd = _inner_re(grad, dirc, dv)
        if gd <= 0:
            dirc = z
            gd = _inner_re(grad, dirc, dv)
        if gd <= 0:
            dirc = _proj_tangent(grad, psi, dv)
            gd = _inner_re(grad, dirc, dv)
        if it > 0 and not restarted:                         # Barzilai-Borwein trial step
            svec = psi - psi_prev
            yvec = grad - grad_prev
            sy = _inner_re(svec, yvec, dv)
            if sy > 1e-30 and math.isfinite(sy):
                bb = _inner_re(svec, svec, dv) / sy
                if math.isfinite(bb):
                    bb = min(max(bb, bb_min), bb_max)
                    alpha = (1.0 - bb_mix) * alpha + bb_mix * bb
        alpha = min(max(alpha, alpha_min), alpha_max)
        accepted = False
        trial = alpha
        psi_t = psi
        Et = E
        phi_t = phi
        # First try CG with its configured budget, then discard that direction
        # and search a fresh descent direction over the complete step range.
        fallback_steps = max(ls_max, int(math.ceil(math.log2(max(alpha_max/alpha_min,1.0))))+1)
        for attempt,budget in enumerate((ls_max,fallback_steps)):
            if attempt:
                ls_restarts += 1
                force_restart = True
                dirc = z
                gd = _inner_re(grad,dirc,dv)
                trial = min(alpha,1.0)
            if not math.isfinite(gd) or gd <= 0:
                continue
            for _ in range(budget):
                psi_t = psi - trial * dirc
                psi_t = psi_t / norm_of(psi_t, G)
                Et, phi_t = energy_phi(psi_t)
                if math.isfinite(Et) and Et < E and Et <= E - 2.0*c1 * trial * gd:
                    accepted = True
                    break
                if trial <= alpha_min:
                    break
                trial = max(trial*shrink,alpha_min)
            if accepted:
                break
        if not accepted:
            n_fail += 1
            alpha = max(trial,alpha_min)
            direction = z.clone()
            force_restart = True
            if n_fail >= stall_max:
                stop = "line_search_stalled"
                if verbose:
                    print(f"[gs] line search stalled it={it}: E={E:.10g} res={resid_rel:.3e} "
                          f"best_res={best_resid:.3e}")
                break
            continue
        n_fail = 0
        psi_prev = psi
        grad_prev = grad
        z_prev = z
        E_prev = E
        psi = psi_t
        E = Et
        phi = phi_t
        alpha = min(alpha_max, trial * growth)
        direction = dirc
        grad, mu, resid_abs, resid_rel = residual(psi, phi)
        if record:
            trace.append((time.time() - t0, it + 1, E, resid_rel))
        if resid_rel < best_resid:                       # keep the lowest-residual iterate
            best_resid = resid_rel
            best_psi = psi.clone()
        dE_rel = abs(E - E_prev) / max(1.0, abs(E))
        if verbose and it % max(1, check) == 0:
            print(f"  [gs] it={it} E={E:.8f} res={resid_rel:.2e} best={best_resid:.2e} step={trial:.2e}")
        if resid_rel < res_tol:                          # strict residual convergence (best in 2D)
            converged = True
            stop = "converged"
            if verbose:
                print(f"[gs] converged it={it}: E={E:.10g} res={resid_rel:.3e}")
            break
        if it - it_ref >= conv_window:                   # windowed energy convergence (3D witness)
            dE_win = abs(E - E_ref) / max(1.0, abs(E))
            if dE_win < etol:
                converged = True
                stop = "energy_converged"
                if verbose:
                    print(f"[gs] energy converged it={it}: dE_window={dE_win:.2e} "
                          f"res={resid_rel:.3e} best_res={best_resid:.3e}")
                break
            E_ref = E
            it_ref = it
    # Return the lowest-residual iterate seen. Accepted Armijo steps never raise
    # the energy, so this iterate differs from the last one only when the residual
    # oscillated; its energy is reported as computed, not assumed minimal.
    if best_resid < resid_rel:
        psi = best_psi
    recentred = (0,) * psi.dim()
    if _translation_invariant(G, cfg):              # undo drift along the zero mode
        psi, recentred = recentre_periodic(psi, G)
    resid_abs, mu_f, resid_rel = compute_residual(psi, G, cfg)
    converged = bool(converged or (resid_rel < res_tol))
    obs = observables(psi, G, cfg)
    obs.update(iters=n_iter, walltime=time.time() - t0, resid_abs=resid_abs,
               resid_rel=resid_rel, min_resid=best_resid, converged=converged,
               stop_reason=stop, preconditioner_shift=sigma,line_search_restarts=ls_restarts,
               recentred_shift=list(recentred))
    if record:
        obs["trace"] = trace
    return psi.detach(), G, obs


def ground_state_multiseed(cfg, G=None, verbose=False):
    cfg = default_cfg(**cfg)
    n = int(cfg.get("nseeds", 1))
    if G is None:
        G = make_grid(cfg)
    rotating = abs(float(cfg.get("Omega", 0.0))) >= 0.3
    # triangular first: needed for rotating branches; vortex-free seeds otherwise
    seed_cycle = ("triangular", "tf", "gaussian") if rotating else ("tf", "gaussian")
    best = None
    Es = []
    seed_times = []
    for j in range(max(1, n)):
        c = dict(cfg)
        c["rng"] = int(cfg.get("rng", 0)) + j
        c["seed"] = seed_cycle[j % len(seed_cycle)]
        psi, G, obs = ground_state(c, G=G, verbose=verbose)
        Es.append(obs["E"])
        seed_times.append(obs['walltime'])
        # Variational selection: the energy of any normalized state bounds the ground
        # state from above, so the lowest energy wins even if that candidate stopped
        # before res_tol (its 'converged' flag is reported with it).
        score = (obs["E"],)
        if best is None or score < best[0]:
            best = (score, psi, obs, c)
    _, psi, obs, cbest = best
    obs['selected_seed_walltime'] = obs['walltime']
    obs['walltime'] = sum(seed_times)
    obs['seed_walltimes'] = seed_times
    obs["seed_energy_spread"] = (max(Es) - min(Es)) if len(Es) > 1 else 0.0
    obs["seed_used"] = cbest["seed"]
    obs["seed_energies"] = ";".join(f"{e:.10g}" for e in Es)
    return psi, G, obs
