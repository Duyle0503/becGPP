"""Run modes: validate, smoke, single, tf_only, sweep, convergence."""
import os
import json
import math
import time
import shutil

import torch
import numpy as np
import matplotlib.pyplot as plt

from . import paths
from .constants import DEV, CODE_VERSION
from .grid import make_grid, geometry, dV, norm_of, resolve_kernel, auto_grid
from .interactions import long_range_phi
from .operators import energy_components
from .fields import radial_average
from .thomasfermi import extract_tf
from .diagnostics import diagnostics
from .solvers import ground_state, ground_state_multiseed
from .figures import (_savefig, _imshow, save_state_figs, save_tf_comparison,
                      show_density, show_tf_density, sweep_figs)
from .io import run_id, write_csv


# =============================================================================
#  MODE: validate  --  analytic gates certifying the numerical framework
# =============================================================================
def mode_validate(cfg):
    print("=" * 70)
    print("VALIDATE: analytic gates (framework certification)")
    rows = []
    Nv = int(cfg.get("validate_N", 256))

    # ---- 2D battery ----
    G = make_grid(dict(cfg, dimension="2D", s=2, kernel="newton", G_C=1.0, L=12.0, Ngrid=Nv))
    dv = dV(G)
    print("-" * 70)
    print("[2D-1] free-space 1/|r| kernel vs analytic Gaussian")
    mg = mp = 0.0
    for a in (0.5, 1.0, 2.0):
        psi = torch.exp(-G["R2"] / (2 * a * a)).to(torch.complex128) / (math.sqrt(math.pi) * a)
        psi = psi / norm_of(psi, G)
        rho = psi.abs()**2
        phi = long_range_phi(rho, G, 1.0)
        Eg = 0.5 * (phi * rho).sum().item() * dv
        phi0 = phi[Nv // 2, Nv // 2].item()
        eg = abs(Eg + math.sqrt(math.pi / 8.0) / a) / (math.sqrt(math.pi / 8.0) / a)
        ep = abs(phi0 + math.sqrt(math.pi) / a) / (math.sqrt(math.pi) / a)
        mg = max(mg, eg)
        mp = max(mp, ep)
        print(f"    a={a}: Egrav rel={eg:.2e}  phi0 rel={ep:.2e}")
        rows += [dict(test="2D_gauss", case=f"a={a}", quantity="Egrav", rel_error=eg),
                 dict(test="2D_gauss", case=f"a={a}", quantity="phi0", rel_error=ep)]
    print("[2D-2] LLL single-particle identities (E_sp=1, Lz=m, <r2>=m+1)")
    ml = 0.0
    for m in (0, 1, 10, 30):
        z = (G["X"] + 1j * G["Y"]).to(torch.complex128)
        psi = (z**m) * torch.exp(-0.5 * G["R2"])
        psi = psi / norm_of(psi, G)
        comp = energy_components(psi, G, 0.0, 0.0, 1.0, 0.0)
        Esp = (comp["Ekin"] + comp["Etrap"] + comp["Erot"]).item()
        r2 = (G["R2"] * psi.abs()**2).sum().item() * dv
        Lz = comp["Lz"].item()
        e = max(abs(Esp - 1.0), abs(Lz - m), abs(r2 - (m + 1)))
        ml = max(ml, e)
        print(f"    m={m:2d}: E_sp={Esp:.6f} Lz={Lz:.5f} <r2>={r2:.5f} maxerr={e:.2e}")
        rows.append(dict(test="2D_LLL", case=f"m={m}", quantity="ids", rel_error=e))

    # ---- local flat-top TF closed form (2D & 3D) ----
    print("[TF] local flat-top: rho0=-3b2/(4b3); mu=b2 rho0+b3 rho0^2; R from int=1")
    mtf = 0.0
    for ndim in (2, 3):
        Gf = make_grid(dict(cfg, dimension=("2D" if ndim == 2 else "3D"), s=2, Omega=1.0,
                            kernel="none", beta2=-1.0, beta3=1.0, G_C=0.0,
                            L=(20.0 if ndim == 2 else 8.0), Ngrid=(192 if ndim == 2 else 96)))
        for (b2, b3) in ((-0.5, 1.0), (-1.5, 1.0)):
            tf = extract_tf(dict(cfg, dimension=("2D" if ndim == 2 else "3D"), s=2, Omega=1.0,
                                 kernel="none", beta2=b2, beta3=b3, G_C=0.0), Gf)
            rho0_th = -3 * b2 / (4 * b3)
            mu_th = b2 * rho0_th + b3 * rho0_th**2
            e = max(abs(tf["rho0"] - rho0_th) / rho0_th, abs(tf["mu"] - mu_th) / max(abs(mu_th), 1e-12))
            mtf = max(mtf, e)
            print(f"    {ndim}D b2={b2} b3={b3}: rho0 rel={abs(tf['rho0'] - rho0_th) / rho0_th:.2e} "
                  f"mu rel={abs(tf['mu'] - mu_th) / max(abs(mu_th), 1e-12):.2e}")
            rows.append(dict(test=f"{ndim}D_flat_tf", case=f"b2={b2}", quantity="rho0_mu", rel_error=e))

    # ---- 3D free-space kernel: for psi ~ exp(-r^2/2 sig^2), phi(0) = -2/(sqrt(pi) sig) ----
    print("[3D] free-space 1/|r| kernel: phi(0) of a Gaussian")
    G3 = make_grid(dict(cfg, dimension="3D", s=2, kernel="newton", G_C=1.0, L=8.0, Ngrid=96))
    m3 = 0.0
    for sig in (1.0, 1.5):
        psi = torch.exp(-G3["R2"] / (2 * sig * sig)).to(torch.complex128)
        psi = psi / norm_of(psi, G3)
        rho = psi.abs()**2
        phi = long_range_phi(rho, G3, 1.0)
        N3 = G3["N"]
        phi0 = phi[N3 // 2, N3 // 2, N3 // 2].item()
        phi0_ex = -2.0 / (math.sqrt(math.pi) * sig)
        e = abs(phi0 - phi0_ex) / abs(phi0_ex)
        m3 = max(m3, e)
        print(f"    sigma={sig}: phi0={phi0:.5f} exact={phi0_ex:.5f} rel={e:.2e}")
        rows.append(dict(test="3D_gauss", case=f"sig={sig}", quantity="phi0", rel_error=e))

    # ---- free-space kernel CONVERGENCE ORDER (2D & 3D) against the analytic phi(0) ----
    # Independent of the solver: fit p in |phi0(dx)-phi0_exact| ~ dx^p on a fixed Gaussian.
    # The Gauss-Legendre cell-averaged kernels are second order in both dimensions.
    print("[order] free-space kernel convergence order (phi(0) of a Gaussian)")
    kernel_orders = {}
    for ndim, Ns, sig, L in ((2, (192, 256, 384, 512), 1.0, 10.0),
                             (3, (64, 96, 128, 160), 1.0, 8.0)):
        dxs, errs = [], []
        ex = (-math.sqrt(math.pi) / sig) if ndim == 2 else (-2.0 / (math.sqrt(math.pi) * sig))
        for Ni in Ns:
            Gk = make_grid(dict(cfg, dimension=("2D" if ndim == 2 else "3D"), s=2,
                                kernel="newton", G_C=1.0, L=L, Ngrid=int(Ni)))
            psi = torch.exp(-Gk["R2"] / (2 * sig * sig)).to(torch.complex128)
            psi = psi / norm_of(psi, Gk)
            phi = long_range_phi(psi.abs()**2, Gk, 1.0)
            c = Gk["N"] // 2
            p0 = (phi[c, c].item() if ndim == 2 else phi[c, c, c].item())
            dxs.append(Gk["dx"])
            errs.append(abs(p0 - ex) / abs(ex))
        order = float(np.polyfit(np.log(dxs), np.log(errs), 1)[0])
        kernel_orders[f"{ndim}D"] = order
        print(f"    {ndim}D: fitted order = {order:.2f}   errs={['%.2e' % e for e in errs]}")
        for dxi, ei in zip(dxs, errs):
            rows.append(dict(test=f"{ndim}D_kernel_order", case=f"dx={dxi:.4f}",
                             quantity="phi0_relerr", rel_error=ei, order=order))

    write_csv(os.path.join(paths.BASE, "validation.csv"), rows)
    passed = (mg < 5e-3 and mp < 5e-3 and ml < 1e-4 and mtf < 1e-2 and m3 < 5e-2)
    order_ok = all(o > 1.7 for o in kernel_orders.values())
    print("-" * 70)
    print(f"[validate] 2D-Coulomb Eg={mg:.2e} phi0={mp:.2e} | LLL={ml:.2e} | flatTF={mtf:.2e} | "
          f"3D-phi0={m3:.2e} | kernel_order={kernel_orders} | PASS={passed and order_ok}")
    print("[validate] thresholds relax on coarse N; use validate_N=512 for publication numbers")
    return dict(passed=passed and order_ok, mg=mg, mp=mp, ml=ml, mtf=mtf, m3=m3,
                kernel_orders=kernel_orders)


# =============================================================================
#  MODE: single
# =============================================================================
def mode_single(cfg):
    if abs(float(cfg.get("Omega", 0.0))) >= 0.5 and cfg.get("seed") == "tf" and int(cfg.get("nseeds", 1)) == 1:
        print("[hint] rotating run (Omega>=0.5) with a vortex-free 'tf' seed: the true ground")
        print("       state usually carries vortices, and the solver would nucleate them one")
        print("       by one from noise (slow; seen as residual spikes + slowly falling E).")
        print("       Use seed='triangular' or nseeds>=3 to start on the vortex-lattice branch.")
    rid = run_id(cfg)
    G = make_grid(cfg)
    psi0 = None
    ck = os.path.join(paths.CKPT_DIR, f"{rid}.pt")
    if os.path.isfile(ck):
        try:
            psi0 = torch.load(ck, map_location=DEV, weights_only=False)["psi"].to(DEV)
            print(f"[resume] {ck}")
        except Exception as exc:
            print(f"[resume-warn] {exc}")
    if int(cfg.get("nseeds", 1)) > 1 and psi0 is None:
        psi, G, obs = ground_state_multiseed(cfg, G=G, verbose=True)
    else:
        psi, G, obs = ground_state(cfg, G=G, psi0=psi0, verbose=True)
    diag = diagnostics(psi, G, cfg)
    diag.update(rid=rid, iters=obs["iters"], walltime=obs["walltime"],
                resid_rel=obs["resid_rel"], converged=obs["converged"],
                dimension=cfg["dimension"], kernel=G["kernel"])
    if cfg.get("save_ckpt", True):
        torch.save(dict(psi=psi.cpu(), cfg=cfg, obs=diag), ck)
    if cfg.get("save_figs", True):
        save_state_figs(psi, G, cfg, rid)
        save_tf_comparison(psi, G, cfg, rid)
    show_density(psi, G, cfg, title=f"{cfg['dimension']} density  "
                 f"(beta2={cfg['beta2']}, G_C={cfg['G_C']}, Omega={cfg['Omega']})")
    write_csv(os.path.join(paths.BASE, "single_summary.csv"), [diag])
    # reusable machine-readable record: full config + every diagnostic, one file per run.
    with open(os.path.join(paths.BASE, f"{rid}.json"), "w") as f:
        json.dump(dict(code_version=CODE_VERSION, device=DEV, cfg=cfg, diagnostics=diag),
                  f, indent=2, default=str)
    print("-" * 70)
    print(f"[single] {cfg['dimension']} s={cfg['s']} Omega={cfg['Omega']} "
          f"beta2={cfg['beta2']} beta3={cfg['beta3']} G_C={cfg['G_C']} kernel={G['kernel']}")
    print(f"         E={diag['E']:.8f} mu={diag['mu']:.6f} Lz={diag['Lz']:.5f} Nv={diag['Nv']}")
    print(f"         R90={diag['R90']:.4f} TF({diag['tf_kind']}) R90={diag['tf_R90']:.4f} "
          f"virial_rel={diag['virial_rel']:.2e} w_LLL={diag['w_LLL']}")
    if diag.get("oblateness") == diag.get("oblateness"):  # not NaN -> 3D
        print(f"         oblateness R_perp/R_z(norm)={diag['oblateness']:.3f} "
              f"(1=sphere, >1 oblate)  R_perp={diag['R_perp_rms']:.3f} R_z={diag['R_z_rms']:.3f}")
    print(f"         resid={diag['resid_rel']:.2e} converged={diag['converged']} it={diag['iters']}")
    return diag


# =============================================================================
#  MODE: tf_only
# =============================================================================
def mode_tf_only(cfg):
    rid = run_id(cfg) + "_tf"
    G = make_grid(cfg)
    tf = extract_tf(cfg, G)
    if tf is None:
        print("[tf_only] no TF reference for these parameters")
        return None
    if cfg.get("save_figs", True):
        if G["ndim"] == 2:
            fig, ax = plt.subplots(figsize=(3.6, 3.0))
            _imshow(ax, tf["n"].cpu().numpy(), G["L"], label=r"$n_{\rm TF}$")
            _savefig(fig, os.path.join(paths.FIG_DIR, f"{rid}_density"))
        rt, pt = radial_average(tf["n"], G)
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        ax.plot(rt, pt, "-", color="#c1272d")
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$n_{\rm TF}(r)$")
        ax.set_ylim(bottom=0)
        _savefig(fig, os.path.join(paths.FIG_DIR, f"{rid}_radial"))
    show_tf_density(tf, G, cfg, title=f"TF density ({tf.get('kind', '')})")
    print(f"[tf_only] kind={tf.get('kind')} R90={tf.get('R90'):.4f} mu={tf.get('mu'):.5f} "
          f"rho0={tf.get('rho0', float('nan'))} converged={tf.get('converged')}")
    write_csv(os.path.join(paths.BASE, "tf_only.csv"),
              [dict(rid=rid, kind=tf.get("kind"), R90=tf.get("R90"), R99=tf.get("R99"),
                    mu=tf.get("mu"), rho0=tf.get("rho0", float("nan")))])
    return tf


# =============================================================================
#  MODE: sweep  --  vary ANY numeric CFG key over a list; TF + all diagnostics
# =============================================================================
def mode_sweep(cfg):
    param = cfg.get("sweep_param", "G_C")
    values = list(cfg.get("sweep_values", []))
    print("=" * 70)
    print(f"SWEEP over {param!r} = {values}")
    rows = []
    prev = None
    for val in values:
        c = dict(cfg)
        c[param] = val
        if cfg.get("sweep_autobox", False):
            L, N, _ = auto_grid(c)
            c["L"], c["Ngrid"] = L, N
        c["tag"] = f"{cfg.get('tag', 'becgpp')}_{param}{val:g}"
        G = make_grid(c)
        psi0 = prev if (prev is not None and tuple(prev.shape) == tuple(G["R2"].shape)) else None
        print(f"\n--- {param}={val}  L={c['L']:.3g} N={c['Ngrid']} {c['dimension']} ---")
        psi, G, obs = ground_state(c, G=G, psi0=psi0, verbose=False)
        diag = diagnostics(psi, G, c)
        diag.update({param: val})
        diag.update(rid=run_id(c), resid_rel=obs["resid_rel"],
                    iters=obs["iters"], converged=obs["converged"], walltime=obs["walltime"])
        rows.append(diag)
        prev = psi.detach()
        if cfg.get("save_figs", True):
            save_state_figs(psi, G, c, diag["rid"])
            save_tf_comparison(psi, G, c, diag["rid"])
        show_density(psi, G, c, title=f"{param}={val}")
        write_csv(os.path.join(paths.BASE, f"sweep_{param}.csv"), rows)
        print(f"    E={diag['E']:.6f} mu={diag['mu']:.5f} Lz={diag['Lz']:.4f} Nv={diag['Nv']} "
              f"R90={diag['R90']:.4f} TF_R90={diag['tf_R90']:.4f} vir={diag['virial_rel']:.2e} "
              f"res={diag['resid_rel']:.2e}")
    sweep_figs(rows, param)
    return rows


# =============================================================================
#  MODE: convergence  --  refine N, box, padding at fixed physics
# =============================================================================
def mode_convergence(cfg):
    rows = []
    L0 = float(cfg["L"])
    dx0 = 2.0 * L0 / int(cfg["Ngrid"])
    variants = [("grid", L0, int(N), int(cfg.get("pad", 2))) for N in cfg.get("conv_Ngrids", [192, 256, 384])]
    for f in cfg.get("conv_box_factors", [1.0, 1.2]):
        L = L0 * f
        N = max(96, 32 * math.ceil((2 * L / dx0) / 32))
        variants.append(("box", L, N, int(cfg.get("pad", 2))))
    for pp in cfg.get("conv_pads", [2, 3]):
        variants.append(("pad", L0, int(cfg["Ngrid"]), int(pp)))
    print("=" * 70)
    print(f"CONVERGENCE  {cfg['dimension']} s={cfg['s']} Omega={cfg['Omega']} "
          f"beta2={cfg['beta2']} beta3={cfg['beta3']} G_C={cfg['G_C']}")
    seen = set()
    prev = None
    for axis, L, N, pad in variants:
        key = (round(L, 10), N, pad)
        if key in seen:
            continue
        seen.add(key)
        c = dict(cfg, L=float(L), Ngrid=int(N), pad=int(pad), tag=f"{cfg.get('tag', 'becgpp')}_{axis}_N{N}_p{pad}")
        G = make_grid(c)
        psi0 = prev if (prev is not None and tuple(prev.shape) == tuple(G["R2"].shape)) else None
        psi, G, obs = ground_state(c, G=G, psi0=psi0, verbose=False)
        d = diagnostics(psi, G, c)
        d.update(axis=axis, L=L, N=N, pad=pad, dx=G["dx"], resid_rel=obs["resid_rel"],
                 iters=obs["iters"], converged=obs["converged"], points_per_R90=d["R90"] / G["dx"])
        rows.append(d)
        if axis == "grid" and N == max(cfg.get("conv_Ngrids", [256])):
            prev = psi.detach()
        write_csv(os.path.join(paths.BASE, "convergence.csv"), rows)
        print(f"    {axis} L={L:.4g} N={N} p={pad}: E={d['E']:.9g} R90={d['R90']:.6g} "
              f"vir={d['virial_rel']:.2e} ppR90={d['points_per_R90']:.1f} res={d['resid_rel']:.2e}")
    grid = [r for r in rows if r["axis"] == "grid"]
    if len(grid) >= 2:
        base = grid[-1]
        print("-" * 70)
        for r in grid:
            print(f"    N={r['N']}: dE_vs_finest={r['E'] - base['E']:+.3e} "
                  f"dR90_rel={(r['R90'] - base['R90']) / max(abs(base['R90']), 1e-30):+.3e}")
    return rows


# =============================================================================
#  MODE: smoke  --  fast end-to-end check in the configured geometry
# =============================================================================
def mode_smoke(cfg):
    print("=" * 70)
    print("SMOKE: validate gate + a tiny run + TF extraction")
    gate = mode_validate(dict(cfg, validate_N=192))
    small = dict(cfg, L=10.0, Ngrid=(128 if geometry(cfg)[0] == 2 else 72),
                 maxit=1500, res_tol=3e-3, nseeds=1, save_figs=True, want_lll=True, tag="smoke")
    print("-" * 70)
    print(f"[smoke] single run ({cfg['dimension']})")
    d = mode_single(small)
    print("-" * 70)
    print("[smoke] tf_only")
    mode_tf_only(small)
    ok = bool(gate["passed"] and d and d.get("converged", False) is not None)
    print("-" * 70)
    print(f"[smoke] validation_passed={gate['passed']} run_done={bool(d)} OVERALL={ok}")
    return dict(gate=gate, run=d, ok=ok)


# =============================================================================
#  DISPATCH + ARCHIVE
# =============================================================================
def zip_results(name="becgpp_results"):
    parent = os.path.dirname(os.path.abspath(paths.BASE))
    arch = shutil.make_archive(os.path.join(parent, name), "zip", root_dir=paths.BASE)
    print(f"[zip] {arch} ({os.path.getsize(arch) / 1e6:.1f} MB)")
    return arch


DISPATCH = dict(validate=mode_validate, smoke=mode_smoke, single=mode_single,
                tf_only=mode_tf_only, sweep=mode_sweep, convergence=mode_convergence)


def run(cfg):
    """Dispatch a configuration to its mode and return the result."""
    mode = cfg.get("mode", "smoke")
    fn = DISPATCH.get(mode)
    if fn is None:
        raise ValueError(f"unknown mode {mode!r}; choose {sorted(DISPATCH)}")
    kkind = resolve_kernel(cfg)[0]
    requested = str(cfg.get("kernel", "auto")).lower()
    if requested in ("newton", "log") and kkind == "none" and abs(float(cfg.get("G_C", 0.0))) <= 1e-15:
        print(f"[warn] kernel={requested!r} is set but G_C=0, so the long-range term is OFF "
              f"(reported kernel='none'). Set G_C>0 to activate it, e.g. G_C=20.")
    print(f"becGPP {CODE_VERSION} | device={DEV} | mode={mode} | "
          f"dim={cfg.get('dimension')} kernel={kkind}")
    result = fn(cfg)
    if cfg.get("zip_output", False):
        try:
            zip_results()
        except Exception as exc:
            print(f"[zip-warn] {exc}")
    print(f"\nDone. Output in: {paths.BASE}")
    return result
