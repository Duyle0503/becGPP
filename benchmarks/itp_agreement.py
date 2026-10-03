# =============================================================================
#  becGPP benchmark: agreement of becGPP (preconditioned CG) with the standard
#  imaginary-time split-step Fourier method (ITP-TSSP) on the same GPU
#  (Kaggle, ONE cell, GPU accelerator ON)
# -----------------------------------------------------------------------------
#  The reference solver below is written from scratch in plain PyTorch and does
#  NOT call the becGPP solver: it is the normalized gradient flow discretized by
#  Strang time splitting (Bao & Du 2004; Bao, Wang & Markowich 2005 for rotation,
#  whose Lz term is split in alternating directions x / y and applied exactly by
#  1D FFTs). It is the explicit imaginary-time method of split-step codes such as
#  GPUE (GPELab uses an implicit backward-Euler gradient flow, BEC2HPC the
#  preconditioned CG), and serves as an independent accuracy check: same
#  discretization, same energy, different minimizer.
#
#  Only the long-range potential of case D reuses becGPP's free-space
#  convolution (the ITP method has no kernel of its own); cases A-C are fully
#  independent. The energy functional is evaluated by becGPP's energy routine
#  for both methods so that the numbers are directly comparable.
#
#  Cases (trap units):
#   A  2D harmonic,           beta2=200, Omega=0,   L=10, N=256
#   B  2D rotating lattice,   beta2=200, Omega=0.9, L=12, N=256 (same seed)
#   C  3D harmonic,           beta2=200, Omega=0,   L=8,  N=96
#   D  3D self-gravitating,   beta2=100, G_C=20,    L=8,  N=96
#
#  For ITP, dt is decreased (0.02 -> 0.005 -> 0.00125 -> 0.0003125) and at each dt the flow
#  is run to stationarity; the Strang splitting leaves an O(dt^2) bias in the
#  stationary energy, removed by Richardson extrapolation of the last two dt.
#
#  Outputs (/kaggle/working/itp_agreement): itp_vs_pcg.csv, SUMMARY.txt, hardware.json
# =============================================================================
import os, sys, json, time, math, shutil, subprocess, platform, csv

REPO, TAG = "https://github.com/Duyle0503/becGPP.git", "v1.1.0"


def _install():
    try:
        import becgpp
        if becgpp.__version__.startswith(("1.1", "1.2")):
            return
    except Exception:
        pass
    for ref in (f"@{TAG}", ""):
        if subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                           "--force-reinstall", f"git+{REPO}{ref}"]).returncode == 0:
            return
    raise RuntimeError("could not install becGPP")


_install()
import torch
import becgpp

# The paper runs (v1.1) used cg_beta="pr" (the default in 1.0-1.1); pin it so that this
# script reproduces them exactly under becGPP >= 1.2 (whose default is "pr_precond").
import becgpp.config as _bc
_bc.CFG_DEFAULTS["cg_beta"] = "pr"
from becgpp import default_cfg, paths, make_grid, ground_state, diagnostics, make_seed
from becgpp.operators import energy_components
from becgpp.interactions import long_range_phi, clear_kernel_cache

DEV = becgpp.DEV
ROOT = "/kaggle/working/itp_agreement" if os.path.isdir("/kaggle/working") else os.path.abspath("itp_agreement")
os.makedirs(ROOT, exist_ok=True)
paths.configure(ROOT)
SUMMARY = []


def note(s):
    print(s, flush=True)
    SUMMARY.append(s)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def hardware():
    cpu = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info = dict(torch=torch.__version__, becgpp=becgpp.__version__, device=DEV,
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                cpu_model=cpu)
    json.dump(info, open(os.path.join(ROOT, "hardware.json"), "w"), indent=2)
    print(info)
    return info


# ------------------------------------------------------------------ ITP-TSSP solver
class ITP:
    """Normalized gradient flow, Strang-split; rotation by x/y alternating directions."""

    def __init__(self, cfg, G):
        self.cfg, self.G = cfg, G
        self.b2, self.b3 = float(cfg["beta2"]), float(cfg["beta3"])
        self.O, self.gc = float(cfg["Omega"]), float(cfg["G_C"])
        self.ndim = G["ndim"]
        N, dx = G["N"], G["dx"]
        k = 2 * math.pi * torch.fft.fftfreq(N, d=dx).to(DEV)
        self.k = k
        self.X, self.Y = G["coords"][0], G["coords"][1]
        self.dv = dx ** self.ndim

    def _nonlin(self, psi):
        rho = psi.abs() ** 2
        V = self.G["V"] + self.b2 * rho + self.b3 * rho ** 2
        if self.gc != 0.0 and self.G["kernel"] != "none":
            V = V + long_range_phi(rho, self.G, self.gc)
        return V

    def _kin(self, psi, dt):
        k = self.k
        if self.O == 0.0:
            return torch.fft.ifftn(torch.exp(-0.5 * dt * self.G["K2"]) * torch.fft.fftn(psi))
        # H_x = -1/2 d_xx - i Omega y d_x  -> symbol 1/2 kx^2 + Omega y kx   (FFT along x)
        # H_y = -1/2 d_yy + i Omega x d_y  -> symbol 1/2 ky^2 - Omega x ky   (FFT along y)
        sh = [1] * self.ndim
        shx = list(sh); shx[0] = -1
        shy = list(sh); shy[1] = -1
        kx, ky = k.reshape(shx), k.reshape(shy)
        ex = torch.exp(-0.5 * dt * (0.5 * kx ** 2 + self.O * self.Y * kx))   # half step of H_x
        ey = torch.exp(-dt * (0.5 * ky ** 2 - self.O * self.X * ky))        # full step of H_y
        psi = torch.fft.ifft(ex * torch.fft.fft(psi, dim=0), dim=0)
        psi = torch.fft.ifft(ey * torch.fft.fft(psi, dim=1), dim=1)
        psi = torch.fft.ifft(ex * torch.fft.fft(psi, dim=0), dim=0)
        if self.ndim == 3:
            shz = list(sh); shz[2] = -1
            kz = k.reshape(shz)
            psi = torch.fft.ifft(torch.exp(-0.5 * dt * kz ** 2) * torch.fft.fft(psi, dim=2), dim=2)
        return psi

    def _normalize(self, psi):
        return psi / torch.sqrt((psi.abs() ** 2).sum() * self.dv)

    def energy(self, psi):
        return energy_components(psi, self.G, self.b2, self.b3, self.O, self.gc)["E"].item()

    def run_dt(self, psi, dt, etol=1e-11, check=25, maxsteps=40000, E_ref=None, tol=None):
        """Run the flow at fixed dt to stationarity. If E_ref is given, also return
        the wall time (since the start of this call) at which |E-E_ref| <= tol|E_ref|
        was first observed (checked every `check` steps), else nan."""
        sync()
        t0 = time.time()
        t_hit = float("nan")
        E_old = self.energy(psi)
        steps = 0
        while steps < maxsteps:
            for _ in range(check):
                psi = psi * torch.exp(-0.5 * dt * self._nonlin(psi))
                psi = self._kin(psi, dt)
                psi = self._normalize(psi * torch.exp(-0.5 * dt * self._nonlin(psi)))
            steps += check
            E = self.energy(psi)
            if E_ref is not None and t_hit != t_hit and abs(E - E_ref) <= tol * abs(E_ref):
                sync()
                t_hit = time.time() - t0
            if abs(E - E_old) <= etol * max(1.0, abs(E)):
                break
            E_old = E
        return psi, E, steps, t_hit


CASES = [
    ("A_2D_harmonic", dict(dimension="2D", s=2, Omega=0.0, beta2=200, G_C=0, kernel="none",
                           L=10, Ngrid=256, seed="tf")),
    ("B_2D_lattice_0.9", dict(dimension="2D", s=2, Omega=0.9, beta2=200, G_C=0, kernel="none",
                              L=12, Ngrid=256, seed="triangular")),
    ("C_3D_harmonic", dict(dimension="3D", s=2, Omega=0.0, beta2=200, G_C=0, kernel="none",
                           L=8, Ngrid=96, seed="tf")),
    ("D_3D_selfgrav", dict(dimension="3D", s=2, Omega=0.0, beta2=100, G_C=20, kernel="newton",
                           L=8, Ngrid=96, seed="tf")),
]
DTS = (0.02, 0.005, 0.00125, 0.0003125)
TARGET = 1e-5           # relative energy error at which both methods are timed

hardware()
rows = []
for name, kw in CASES:
    cfg = default_cfg(mode="single", beta3=0.0, rng=0, nseeds=1, res_tol=1e-8, maxit=15000,
                      show_inline=False, save_figs=False, save_ckpt=False, **kw)
    G = make_grid(cfg)
    psi0 = make_seed(G, cfg)                      # identical starting state for both methods
    # --- becGPP PCG: reference at residual 1e-8, and timed at residual 1e-5
    ground_state(dict(cfg, maxit=3), G=G, psi0=psi0.clone(), verbose=False)   # warm caches
    sync(); t = time.time()
    psi_pt, _, obs_t = ground_state(dict(cfg, res_tol=1e-5), G=G, psi0=psi0.clone(), verbose=False)
    sync(); t_pcg_t = time.time() - t
    sync(); t = time.time()
    psi_p, _, obs = ground_state(cfg, G=G, psi0=psi0.clone(), verbose=False)
    sync(); t_pcg9 = time.time() - t
    E_pcg = obs["E"]
    # --- ITP-TSSP from the same seed
    itp = ITP(cfg, G)
    psi = psi0.clone()
    E_dt, t_itp, steps_tot = [], 0.0, 0
    t_to_1e6 = float("nan")
    for dt in DTS:
        sync(); t = time.time()
        psi, E, steps, t_hit = itp.run_dt(psi, dt, E_ref=E_pcg, tol=TARGET)
        sync()
        if math.isnan(t_to_1e6) and t_hit == t_hit:
            t_to_1e6 = t_itp + t_hit                   # time since the ITP start
        t_itp += time.time() - t
        steps_tot += steps
        E_dt.append(E)
    # Richardson in dt^2 from the last two dt
    d1, d2 = DTS[-2], DTS[-1]
    E_itp_ex = (E_dt[-1] * d1 ** 2 - E_dt[-2] * d2 ** 2) / (d1 ** 2 - d2 ** 2)
    dg = diagnostics(psi_p, G, cfg)
    row = dict(case=name, N=G["N"], E_pcg=E_pcg, resid_pcg=obs["resid_rel"], iters_pcg=obs["iters"],
               t_pcg_target_s=t_pcg_t, iters_pcg_target=obs_t["iters"], E_pcg_target=obs_t["E"],
               t_pcg_1e9_s=t_pcg9, Nv_pcg=dg["Nv"], Lz_pcg=dg["Lz"],
               E_itp_dt=";".join(f"{e:.12f}" for e in E_dt), E_itp_extrap=E_itp_ex,
               rel_diff_extrap=(E_itp_ex - E_pcg) / abs(E_pcg), steps_itp=steps_tot,
               t_itp_total_s=t_itp, t_itp_to_target_s=t_to_1e6,
               speedup_to_target=(t_to_1e6 / t_pcg_t) if t_to_1e6 == t_to_1e6 else float("nan"))
    rows.append(row)
    note(f"{name}: E_PCG={E_pcg:.12f} (res {obs['resid_rel']:.1e}, {obs['iters']} it) | "
         f"E_ITP(dt)={[round(e, 10) for e in E_dt]} extrap={E_itp_ex:.12f} "
         f"rel diff={row['rel_diff_extrap']:+.2e} | time to 1e-5: PCG {t_pcg_t:.2f}s "
         f"vs ITP {t_to_1e6:.2f}s (x{row['speedup_to_target']:.1f}); ITP total {t_itp:.1f}s, {steps_tot} steps")
    del psi, psi_p, psi_pt, psi0
    clear_kernel_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

with open(os.path.join(ROOT, "itp_vs_pcg.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
open(os.path.join(ROOT, "SUMMARY.txt"), "w").write("\n".join(SUMMARY) + "\n")
arch = shutil.make_archive("/kaggle/working/becgpp_itp_agreement" if os.path.isdir("/kaggle/working")
                           else os.path.abspath("becgpp_itp_agreement"), "zip", root_dir=ROOT)
print("\n" + "=" * 78 + "\n" + "\n".join(SUMMARY) + f"\n\nArchive: {arch}")
