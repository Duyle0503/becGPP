# =============================================================================
#  becGPP v1.2 -- MA-4: like-for-like time to solution, becGPP (PCG) vs an
#  independent imaginary-time split-step solver (ITP), on the same GPU
#  (Kaggle, ONE cell, GPU accelerator ON; ~40-70 min on a T4)
# -----------------------------------------------------------------------------
#  Answers referee point MA-4: in the v1.1 comparison becGPP was timed to a
#  residual of 1e-5 and ITP to an energy error of 1e-5. Here BOTH methods are
#  timed by the SAME criterion: the first wall time at which the relative energy
#  error |E - E_ref|/|E_ref| drops below a target, read from a per-iteration
#  convergence history (becGPP: cfg record_trace=True; ITP: every 25 steps).
#
#  E_ref = lowest energy found by any method at the tightest settings (the
#  becGPP runs to residual 1e-9). Methods:
#     PCG-pp : becGPP, cg_beta="pr_precond"  (default since 1.2.0)
#     PCG-pr : becGPP, cg_beta="pr"          (default in 1.0-1.1, used for the v1.1 paper runs)
#     ITP    : Strang split-step normalized gradient flow, dt = 0.02 -> 0.005 ->
#              0.00125 -> 0.0003125 (rotation by x/y alternating directions)
#  All start from the SAME seed; kernel caches are warmed before timing.
#
#  Cases (trap units), identical to the v1.1 step-4 run:
#   A  2D harmonic,          beta2=200, Omega=0,   L=10, N=256
#   B  2D rotating lattice,  beta2=200, Omega=0.9, L=12, N=256
#   C  3D harmonic,          beta2=200, Omega=0,   L=8,  N=96
#   D  3D self-gravitating,  beta2=100, G_C=20,    L=8,  N=96
#
#  Outputs (/kaggle/working/v12_ma4, zipped): ma4_timing.csv (times to 1e-4,
#  1e-5, 1e-6 for each method), ma4_traces.csv (full histories),
#  fig_ma4_convergence.(pdf|png) (energy error vs wall time, 4 panels),
#  SUMMARY.txt, hardware.json
# =============================================================================
import os, sys, json, time, math, shutil, subprocess, platform, csv

REPO, TAG = "https://github.com/Duyle0503/becGPP.git", "v1.2.0"


def _install():
    try:
        import becgpp
        if becgpp.__version__.startswith("1.2"):
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
from becgpp import default_cfg, paths, make_grid, ground_state, make_seed
from becgpp.operators import energy_components
from becgpp.interactions import long_range_phi, clear_kernel_cache

assert becgpp.__version__.startswith("1.2"), f"need becGPP 1.2.x (record_trace), got {becgpp.__version__}"
DEV = becgpp.DEV
ROOT = "/kaggle/working/v12_ma4" if os.path.isdir("/kaggle/working") else os.path.abspath("v12_ma4")
os.makedirs(ROOT, exist_ok=True)
paths.configure(ROOT)
TARGETS = (1e-4, 1e-5, 1e-6)
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
                cpu_model=cpu, python=platform.python_version())
    json.dump(info, open(os.path.join(ROOT, "hardware.json"), "w"), indent=2)
    print(info)


# ------------------------------------------------------------------ ITP-TSSP solver
class ITP:
    """Normalized gradient flow, Strang-split; rotation by x/y alternating directions.
    Written independently of the becGPP solver (only the long-range potential of
    case D reuses becGPP's free-space convolution)."""

    def __init__(self, cfg, G):
        self.cfg, self.G = cfg, G
        self.b2, self.b3 = float(cfg["beta2"]), float(cfg["beta3"])
        self.O, self.gc = float(cfg["Omega"]), float(cfg["G_C"])
        self.ndim = G["ndim"]
        N, dx = G["N"], G["dx"]
        self.k = 2 * math.pi * torch.fft.fftfreq(N, d=dx).to(DEV)
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

    def run(self, psi, dts, etol=1e-11, check=25, maxsteps=40000):
        """Returns final psi, list of stage-final energies, and the history
        [(wall time, step, E)] sampled every `check` steps (timer excludes nothing:
        the energy evaluations are part of the method's cost, as for becGPP)."""
        hist = [(0.0, 0, self.energy(psi))]
        sync()
        t0 = time.time()
        steps = 0
        E_stage = []
        for dt in dts:
            E_old = hist[-1][2]
            n = 0
            while n < maxsteps:
                for _ in range(check):
                    psi = psi * torch.exp(-0.5 * dt * self._nonlin(psi))
                    psi = self._kin(psi, dt)
                    psi = self._normalize(psi * torch.exp(-0.5 * dt * self._nonlin(psi)))
                n += check
                steps += check
                E = self.energy(psi)                    # .item() synchronizes
                hist.append((time.time() - t0, steps, E))
                if abs(E - E_old) <= etol * max(1.0, abs(E)):
                    break
                E_old = E
            E_stage.append(hist[-1][2])
        return psi, E_stage, hist


def first_time(hist, E_ref, tol):
    for t, _, E in hist:
        if abs(E - E_ref) <= tol * abs(E_ref):
            return t
    return float("nan")


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

hardware()
rows, traces = [], []
for name, kw in CASES:
    base = default_cfg(mode="single", beta3=0.0, rng=0, nseeds=1, res_tol=1e-9, maxit=20000,
                       record_trace=True, show_inline=False, save_figs=False, save_ckpt=False, **kw)
    G = make_grid(base)
    psi0 = make_seed(G, base)                      # identical starting state for every method
    ground_state(dict(base, maxit=3), G=G, psi0=psi0.clone(), verbose=False)   # warm caches
    runs = {}
    for label, beta in (("PCG-pp", "pr_precond"), ("PCG-pr", "pr")):
        sync()
        psi, _, obs = ground_state(dict(base, cg_beta=beta), G=G, psi0=psi0.clone(), verbose=False)
        runs[label] = dict(E=obs["E"], hist=[(t, it, E) for t, it, E, _ in obs["trace"]],
                           iters=obs["iters"], resid=obs["resid_rel"], stop=obs["stop_reason"])
        del psi
    itp = ITP(base, G)
    psi, E_stage, hist = itp.run(psi0.clone(), DTS)
    d1, d2 = DTS[-2], DTS[-1]
    E_itp_ex = (E_stage[-1] * d1 ** 2 - E_stage[-2] * d2 ** 2) / (d1 ** 2 - d2 ** 2)
    runs["ITP"] = dict(E=E_stage[-1], hist=hist, iters=hist[-1][1], resid=float("nan"),
                       stop="stages_done", E_extrap=E_itp_ex)
    E_ref = min(runs["PCG-pp"]["E"], runs["PCG-pr"]["E"])
    for label, r in runs.items():
        row = dict(case=name, method=label, N=G["N"], E_final=r["E"], E_ref=E_ref,
                   rel_err_final=(r["E"] - E_ref) / abs(E_ref), iters_or_steps=r["iters"],
                   resid=r["resid"], stop=r["stop"], wall_total_s=r["hist"][-1][0],
                   E_itp_extrap=r.get("E_extrap", float("nan")))
        for tol in TARGETS:
            row[f"t_to_{tol:.0e}"] = first_time(r["hist"], E_ref, tol)
        rows.append(row)
        for t, it, E in r["hist"]:
            traces.append(dict(case=name, method=label, t=t, it=it, E=E,
                               rel_err=abs(E - E_ref) / abs(E_ref)))
    msg = [f"{name} (E_ref={E_ref:.12f}):"]
    for r in [x for x in rows if x["case"] == name]:
        msg.append(f"   {r['method']:7s} t(1e-4)={r['t_to_1e-04']:.2f}s t(1e-5)={r['t_to_1e-05']:.2f}s "
                   f"t(1e-6)={r['t_to_1e-06']:.2f}s  final rel err={r['rel_err_final']:+.1e} "
                   f"[{r['iters_or_steps']} it/steps, {r['stop']}]")
    note("\n".join(msg))
    note(f"   ITP dt-extrapolated energy: rel diff {(E_itp_ex - E_ref) / abs(E_ref):+.2e}")
    del psi, psi0
    clear_kernel_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write(path, rr):
    keys = list(dict.fromkeys(k for r in rr for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rr)


write(os.path.join(ROOT, "ma4_timing.csv"), rows)
write(os.path.join(ROOT, "ma4_traces.csv"), traces)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm",
                     "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
style = {"PCG-pp": ("-", "#1f5c8b"), "PCG-pr": ("--", "#6a9fcb"), "ITP": ("-", "#b3282d")}
for ax, (name, _) in zip(axs.ravel(), CASES):
    for m, (ls, col) in style.items():
        tr = [x for x in traces if x["case"] == name and x["method"] == m and x["t"] > 0]
        if tr:
            ax.loglog([x["t"] for x in tr], [max(x["rel_err"], 1e-16) for x in tr], ls, color=col,
                      lw=1.2, label={"PCG-pp": "becGPP (pr_precond)", "PCG-pr": "becGPP (pr)",
                                     "ITP": "imaginary time"}[m])
    ax.axhline(1e-5, color="0.6", lw=0.6, ls=":")
    ax.set_title(name.split("_", 1)[1].replace("_", " "), fontsize=9)
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel(r"$|E-E_{\rm ref}|/|E_{\rm ref}|$")
axs[0, 0].legend(fontsize=7.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(ROOT, f"fig_ma4_convergence.{ext}"), dpi=150)

open(os.path.join(ROOT, "SUMMARY.txt"), "w").write("\n".join(SUMMARY) + "\n")
arch = shutil.make_archive("/kaggle/working/becgpp_v12_ma4" if os.path.isdir("/kaggle/working")
                           else os.path.abspath("becgpp_v12_ma4"), "zip", root_dir=ROOT)
print("\n" + "=" * 78 + "\n" + "\n".join(SUMMARY) + f"\n\nArchive: {arch}")
