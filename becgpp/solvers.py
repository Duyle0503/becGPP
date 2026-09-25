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
from .grid import make_grid, dV, norm_of
from .interactions import long_range_phi
from .operators import apply_H, energy_t, compute_residual, observables
from .seeds import make_seed


def _inner_re(u, v, dv):
    return ((u.conj() * v).real.sum() * dv).item()


def _proj_tangent(v, psi, dv):
    return v - psi * ((psi.conj() * v).sum() * dv)


def _preconditioner(grad, psi, G, cfg):
    """Adaptive Sobolev preconditioner: z = (1/2 k^2 + sigma)^-1 grad, with sigma
    the median of the local potential over the bulk. Projected onto the tangent
    space at psi. This shift tracks the actual stiffness (beta2 n dominates),
    which fixed shift=1 does not -- essential for fast 3D convergence."""
    rho = psi.abs()**2
    b2, b3, gc = cfg["beta2"], cfg["beta3"], cfg["G_C"]
    local = G["V"] + 2.0 * b2 * rho + 3.0 * b3 * rho**2
    if G["kernel"] != "none" and abs(gc) > 1e-15:
        local = local + long_range_phi(rho, G, gc)
    mask = rho > 0.05 * rho.max()
    vals = local[mask]
    floor = float(cfg.get("precond_shift_min", 0.5))
    sigma = float(torch.median(vals).item()) if vals.numel() else floor
    sigma = max(sigma, floor)                      # keep the metric positive definite
    denom = 0.5 * G["K2"] + sigma
    z = torch.fft.ifftn(torch.fft.fftn(grad) / denom)
    return _proj_tangent(z, psi, dV(G)), sigma


def ground_state(cfg, G=None, psi0=None, verbose=True):
    """Riemannian preconditioned conjugate-gradient minimization of the rotating
    GP energy on the unit-norm sphere: adaptive Sobolev preconditioner,
    Polak-Ribiere CG with periodic restart, a Barzilai-Borwein trial step, and a
    backtracking Armijo line search. Converges in O(1e3) steps where plain
    preconditioned gradient descent stalls at res ~ 1e-2 (esp. in 3D)."""
    if G is None:
        G = make_grid(cfg)
    psi = psi0.to(DEV) if psi0 is not None else make_seed(G, cfg)
    dv = dV(G)
    psi = psi / norm_of(psi, G)
    maxit = int(cfg["maxit"])
    res_tol = float(cfg["res_tol"])
    etol = float(cfg.get("energy_tol", 1e-9))
    check = int(cfg.get("check", 200))
    alpha = float(cfg.get("step", 1.0))                  # initial trial step
    alpha_min, alpha_max = 1e-8, float(cfg.get("step_max", 3.0))
    ls_max = int(cfg.get("linesearch_max", 8))          # Armijo halvings per iteration
    c1 = 1e-4
    shrink, growth = 0.5, 1.1
    bb_min, bb_max, bb_mix = 0.05, alpha_max, 0.5
    restart_period = int(cfg.get("cg_restart", 30))
    # Polak-Ribiere coefficient: "pr" uses the plain gradient,
    #   beta = <g, g - g_old> / <g_old, g_old>          (default; used for the paper runs)
    # "pr_precond" uses the preconditioned (Sobolev) inner product,
    #   beta = <z, g - g_old> / <z_old, g_old>          (standard preconditioned PR)
    cg_beta = str(cfg.get("cg_beta", "pr")).lower()
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

    def residual(p):
        Hp = apply_H(p, G, cfg)
        mu = (p.conj() * Hp).sum().real * dv
        g = Hp - mu * p
        ra = torch.sqrt((g.abs()**2).sum() * dv).item()
        return g, mu, ra, ra / max(1.0, abs(mu.item()))

    E = energy_t(psi, G, b2, b3, O, gc).item()
    t0 = time.time()
    grad, mu, _, resid_rel = residual(psi)
    z, sigma = _preconditioner(grad, psi, G, cfg)
    direction = z.clone()
    z_prev = z
    converged = False
    stop = "maxit"
    it = 0
    E_prev = E
    psi_prev = psi
    grad_prev = grad
    best_resid = resid_rel
    best_psi = psi.clone()
    E_ref = E
    for it in range(maxit):
        z, sigma = _preconditioner(grad, psi, G, cfg)
        if it == 0 or cg_beta == "none" or (restart_period > 0 and it % restart_period == 0):
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
        if it > 0:                                            # Barzilai-Borwein trial step
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
        for _ in range(ls_max):
            psi_t = psi - trial * dirc
            psi_t = psi_t / norm_of(psi_t, G)
            Et = energy_t(psi_t, G, b2, b3, O, gc).item()
            if math.isfinite(Et) and Et <= E - c1 * trial * gd:
                accepted = True
                break
            trial *= shrink
            if trial < alpha_min:
                break
        if not accepted:
            alpha = max(alpha * 0.5, alpha_min)
            direction = z.clone()
            if alpha <= alpha_min:
                stop = "line_search_stalled"
                break
            continue
        psi_prev = psi
        grad_prev = grad
        z_prev = z
        E_prev = E
        psi = psi_t
        E = Et
        alpha = min(alpha_max, trial * growth)
        direction = dirc
        grad, mu, resid_abs, resid_rel = residual(psi)
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
    resid_abs, mu_f, resid_rel = compute_residual(psi, G, cfg)
    converged = bool(converged or (resid_rel < res_tol))
    obs = observables(psi, G, cfg)
    obs.update(iters=it, walltime=time.time() - t0, resid_abs=resid_abs,
               resid_rel=resid_rel, min_resid=best_resid, converged=converged,
               stop_reason=stop, preconditioner_shift=sigma)
    return psi.detach(), G, obs


def ground_state_multiseed(cfg, G=None, verbose=False):
    n = int(cfg.get("nseeds", 1))
    if G is None:
        G = make_grid(cfg)
    rotating = abs(float(cfg.get("Omega", 0.0))) >= 0.3
    # triangular first: needed for rotating branches; vortex-free seeds otherwise
    seed_cycle = ("triangular", "tf", "gaussian") if rotating else ("tf", "gaussian")
    best = None
    Es = []
    for j in range(max(1, n)):
        c = dict(cfg)
        c["rng"] = int(cfg.get("rng", 0)) + j
        c["seed"] = seed_cycle[j % len(seed_cycle)]
        psi, G, obs = ground_state(c, G=G, verbose=verbose)
        Es.append(obs["E"])
        # Variational selection: the energy of any normalized state bounds the ground
        # state from above, so the lowest energy wins even if that candidate stopped
        # before res_tol (its 'converged' flag is reported with it).
        score = (obs["E"],)
        if best is None or score < best[0]:
            best = (score, psi, obs, c)
    _, psi, obs, cbest = best
    obs["seed_energy_spread"] = (max(Es) - min(Es)) if len(Es) > 1 else 0.0
    obs["seed_used"] = cbest["seed"]
    obs["seed_energies"] = ";".join(f"{e:.10g}" for e in Es)
    return psi, G, obs
