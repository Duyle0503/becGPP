# =============================================================================
#  becGPP v1.1 -- STEP 3: comparison with independent (literature/analytic) references
#  (Kaggle, ONE cell, GPU accelerator ON)
# -----------------------------------------------------------------------------
#  All runs are the UNTRAPPED self-gravitating condensate (s = 0), i.e. the
#  genuine boson-star / dark-matter-core problem, in the code's per-particle
#  units (hbar = m = 1, int |psi|^2 = 1):
#
#     E[psi] = int 1/2 |grad psi|^2 + beta2/2 int rho^2 - G_C/2 int int rho rho'/|r-r'|
#
#  R1  Non-interacting limit (beta2 = 0): the Schrodinger-Newton ground state.
#      Literature (Membrado et al. 1989, as quoted by Chavanis 2011 Eqs. 29-33 and
#      Chavanis & Delfini 2011):
#         E_tot = -0.05426 G^2 M^3 m^2 / hbar^2   ->  E / G_C^2  = -0.05426
#         eigenvalue -0.16278 G^2 M^2 m^3/hbar^2  ->  mu / G_C^2 = -0.16278
#         R99 = 9.946 hbar^2/(G M m^2)             ->  R99 * G_C  =  9.946
#      Grid sequence + Richardson extrapolation in dx^2.
#  R2  Thomas-Fermi limit (beta2 large): the n=1 polytrope (Chavanis 2011 Eq. 39),
#         rho = rho_c sin(kr)/(kr),  k^2 = 4 pi G_C / beta2,  R = pi/k,
#         R99 = 0.954242 R,  mu_TF = -G_C k/pi,  E_TF = -beta2 k^3/(8 pi^2).
#  R3  Mass-radius curve R99*G_C versus chi = G_C beta2 / pi (the variable
#      chi = 4 G M^2 m a / hbar^2 of Chavanis & Delfini), bridging R1 and R2.
#
#  Outputs (in /kaggle/working/v11_step3, zipped): ref_schrodinger_newton.csv,
#  ref_mass_radius.csv, fig_mass_radius.(pdf|png), SUMMARY.txt, hardware.json
# =============================================================================
import os, sys, json, time, math, shutil, subprocess, platform, csv

REPO, TAG = "https://github.com/Duyle0503/becGPP.git", "v1.1.0"


def _install():
    try:
        import becgpp
        if becgpp.__version__.startswith("1.1"):
            return
    except Exception:
        pass
    for ref in (f"@{TAG}", ""):
        if subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                           "--force-reinstall", f"git+{REPO}{ref}"]).returncode == 0:
            return
    raise RuntimeError("could not install becGPP")


_install()
import numpy as np
import torch
import becgpp
from becgpp import default_cfg, paths, make_grid, ground_state, diagnostics
from becgpp.interactions import clear_kernel_cache

ROOT = "/kaggle/working/v11_step3" if os.path.isdir("/kaggle/working") else os.path.abspath("v11_step3")
os.makedirs(ROOT, exist_ok=True)
paths.configure(ROOT)
SUMMARY = []

# literature values (see header)
SN_E, SN_MU, SN_R99 = -0.05426, -0.16278, 9.946
TF_R99_OVER_R = 0.954242


def note(s):
    print(s, flush=True)
    SUMMARY.append(s)


def write_rows(path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {path}")


def hardware():
    cpu = platform.processor() or ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info = dict(torch=torch.__version__, becgpp=becgpp.__version__, device=becgpp.DEV,
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                cpu_model=cpu, torch_cpu_threads=torch.get_num_threads())
    json.dump(info, open(os.path.join(ROOT, "hardware.json"), "w"), indent=2)
    print(info)


def solve(cfg):
    G = make_grid(cfg)
    t = time.time()
    psi, G, obs = ground_state(cfg, G=G, verbose=False)
    d = diagnostics(psi, G, cfg)
    d.update(iters=obs["iters"], resid=obs["resid_rel"], stop=obs["stop_reason"],
             converged=obs["converged"], wall_s=time.time() - t, dx=G["dx"], N=G["N"], L=G["L"])
    del psi
    clear_kernel_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return d


def base_cfg(**kw):
    c = dict(mode="single", dimension="3D", s=0, Omega=0.0, beta3=0.0, kernel="newton",
             seed="gaussian", nseeds=1, res_tol=1e-7, maxit=8000, want_lll=False,
             want_vortices=False, show_inline=False, save_figs=False, save_ckpt=False)
    c.update(kw)
    return default_cfg(**c)


hardware()

# ----------------------------------------------------------------------------- R1
GC = 10.0
rows = []
for N in (96, 128, 160, 192):
    d = solve(base_cfg(beta2=0.0, G_C=GC, L=3.0, Ngrid=N))
    r = dict(N=N, dx=d["dx"], E_over_GC2=d["E"] / GC**2, mu_over_GC2=d["mu"] / GC**2,
             R99_times_GC=d["R99"] * GC, virial_rel=d["virial_rel"], resid=d["resid"],
             iters=d["iters"], wall_s=d["wall_s"], stop=d["stop"])
    rows.append(r)
    note(f"R1 N={N} dx={d['dx']:.4f}: E/G^2={r['E_over_GC2']:.6f} ({(r['E_over_GC2'] / SN_E - 1):+.2e}) "
         f"mu/G^2={r['mu_over_GC2']:.6f} ({(r['mu_over_GC2'] / SN_MU - 1):+.2e}) "
         f"R99*G={r['R99_times_GC']:.4f} ({(r['R99_times_GC'] / SN_R99 - 1):+.2e}) "
         f"vir={r['virial_rel']:.1e} it={r['iters']} {r['wall_s']:.0f}s")


def richardson(xs, ys):
    """Least-squares fit y = y0 + c dx^2 on the finest three grids."""
    x = np.asarray(xs[-3:]) ** 2
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, np.asarray(ys[-3:]), rcond=None)[0][0])


dxs = [r["dx"] for r in rows]
E0 = richardson(dxs, [r["E_over_GC2"] for r in rows])
M0 = richardson(dxs, [r["mu_over_GC2"] for r in rows])
R0 = richardson(dxs, [r["R99_times_GC"] for r in rows])
rows.append(dict(N="extrap", dx=0.0, E_over_GC2=E0, mu_over_GC2=M0, R99_times_GC=R0))
note(f"R1 extrapolated dx->0: E/G^2={E0:.6f} (lit {SN_E}, dev {E0 / SN_E - 1:+.2e})  "
     f"mu/G^2={M0:.6f} (lit {SN_MU}, dev {M0 / SN_MU - 1:+.2e})  "
     f"R99*G={R0:.4f} (lit {SN_R99}, dev {R0 / SN_R99 - 1:+.2e})")
write_rows(os.path.join(ROOT, "ref_schrodinger_newton.csv"), rows)

# ------------------------------------------------------------------------ R2 + R3
mr = []
for b2 in (0.0, 0.5, 2.0, 8.0, 32.0, 128.0, 512.0, 2048.0):
    k = math.sqrt(4 * math.pi * GC / b2) if b2 > 0 else float("nan")
    R_tf = math.pi / k if b2 > 0 else float("nan")
    # box from an interpolated R99 estimate (the true radius exceeds the TF radius
    # at moderate chi; sizing from R_TF alone left the chi~1e2-1e3 runs box-limited)
    chi_est = GC * b2 / math.pi
    R99_est = 1.1 * (SN_R99 + 1.4989 * math.sqrt(chi_est)) / GC
    L = max(3.0, 1.4 * R99_est)
    N = 160 if L <= 8 else 192
    d = solve(base_cfg(beta2=b2, G_C=GC, L=L, Ngrid=N, seed=("tf" if b2 > 0 else "gaussian")))
    chi = GC * b2 / math.pi
    row = dict(beta2=b2, chi=chi, L=L, N=N, dx=d["dx"], R99_times_GC=d["R99"] * GC, E=d["E"],
               mu=d["mu"], virial_rel=d["virial_rel"], resid=d["resid"], converged=d["converged"],
               points_per_R99=d["R99"] / d["dx"])
    if b2 > 0:
        row.update(R99_TF_times_GC=TF_R99_OVER_R * R_tf * GC,
                   mu_TF=-GC * k / math.pi, E_TF=-b2 * k**3 / (8 * math.pi**2))
    mr.append(row)
    msg = (f"R3 beta2={b2:7.1f} chi={chi:8.2f}: R99*G={row['R99_times_GC']:.4f} "
           f"E={d['E']:.6f} mu={d['mu']:.6f} vir={d['virial_rel']:.1e} pts/R99={row['points_per_R99']:.0f}")
    if b2 > 0:
        msg += (f" | TF: R99*G={row['R99_TF_times_GC']:.4f} ({row['R99_times_GC'] / row['R99_TF_times_GC'] - 1:+.2e})"
                f" mu={row['mu_TF']:.5f} ({d['mu'] / row['mu_TF'] - 1:+.2e})"
                f" E={row['E_TF']:.5f} ({d['E'] / row['E_TF'] - 1:+.2e})")
    note(msg)
write_rows(os.path.join(ROOT, "ref_mass_radius.csv"), mr)

# figure: R99*G_C versus chi with both exact limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "font.family": "serif", "mathtext.fontset": "cm",
                     "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
fig, ax = plt.subplots(figsize=(4.4, 3.4))
pts = [r for r in mr if r["chi"] > 0]
chis = np.array([r["chi"] for r in pts])
ax.plot(chis, [r["R99_times_GC"] for r in pts], "o", color="#1f5c8b", ms=5, label=r"becGPP")
cc = np.logspace(math.log10(chis.min() / 2), math.log10(chis.max() * 2), 200)
ax.plot(cc, 1.4989 * np.sqrt(cc), "--", color="#b3282d", label=r"TF: $1.499\,\chi^{1/2}$")
ax.axhline(SN_R99, color="0.4", ls=":", label=r"non-interacting: $9.946$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\chi = G_{\rm C}\beta_2/\pi$")
ax.set_ylabel(r"$R_{99}\,G_{\rm C}$")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(ROOT, f"fig_mass_radius.{ext}"), dpi=160)

open(os.path.join(ROOT, "SUMMARY.txt"), "w").write("\n".join(SUMMARY) + "\n")
arch = shutil.make_archive("/kaggle/working/becgpp_v11_step3" if os.path.isdir("/kaggle/working")
                           else os.path.abspath("becgpp_v11_step3"), "zip", root_dir=ROOT)
print("\n" + "=" * 78 + "\n" + "\n".join(SUMMARY) + f"\n\nArchive: {arch}")
