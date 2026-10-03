# =============================================================================
#  becGPP benchmark: verification, sweeps, CG variants and performance
#  (Kaggle, ONE cell, GPU accelerator ON: Settings -> Accelerator -> GPU)
# -----------------------------------------------------------------------------
#  Installs becGPP (tag v1.1.0, the version used for the paper) from GitHub and
#  produces the numbers of the verification and performance sections:
#
#   S0 hardware.json              GPU/CPU model, driver, torch threads (Sec. perf)
#   S1 validation.csv             analytic gate at N=512: 1/r, -ln r, q2D, LLL, 3D,
#                                 kernel orders (Table validation, Fig. convergence)
#   S2 kernels_log_q2d.csv        -ln r virial (fixed identity) + quasi-2D l_z crossover
#   S3 sweep_gc2d_G_C.csv         2D Omega=1 G_C sweep, continuation + 3 seeds
#   S4 sweep_omega_Omega.csv      rotation sweep with nseeds=3 and the LLL parameter
#   S5 cq_ladder.csv              cubic-quintic droplets approaching the flat top
#   S6 pcg_variants.csv           Polak-Ribiere variants (pr / pr_precond / none)
#   S7 perf_gpu.csv, perf_cpu_gpu.csv   performance with hardware recorded
#   -> everything zipped to /kaggle/working/becgpp_verification.zip
#
#  Turn sections on/off in SECTIONS below. Rough GPU time (P100/T4): S1 ~5 min,
#  S2 ~5 min, S3 ~40-70 min, S4 ~15 min, S5 ~15 min, S6 ~10 min, S7 ~10 min.
# =============================================================================
import os, sys, json, time, math, glob, shutil, subprocess, platform, textwrap, csv

SECTIONS = dict(S1=True, S2=True, S3=True, S4=True, S5=True, S6=True, S7=True)

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
from becgpp import default_cfg, run, paths, make_grid, ground_state, diagnostics
assert becgpp.__version__.startswith(("1.1", "1.2")), f"need becGPP >= 1.1, got {becgpp.__version__}"

ROOT = "/kaggle/working/verification" if os.path.isdir("/kaggle/working") else os.path.abspath("verification")
os.makedirs(ROOT, exist_ok=True)
QUIET = dict(show_inline=False, zip_output=False)


def section_dir(name):
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    paths.configure(d)
    return d


def write_rows(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {path}")


def hardware(path):
    cpu = platform.processor() or ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info = dict(python=platform.python_version(), torch=torch.__version__,
                becgpp=becgpp.__version__, cuda=getattr(torch.version, "cuda", None),
                device=becgpp.DEV,
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                gpu_mem_GB=(round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                            if torch.cuda.is_available() else None),
                cpu_model=cpu, cpu_logical_cores=os.cpu_count(),
                torch_cpu_threads=torch.get_num_threads())
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm",
             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        info["nvidia_smi"] = None
    json.dump(info, open(path, "w"), indent=2)
    print(json.dumps(info, indent=2))
    return info


SUMMARY = []


def note(line):
    print(line)
    SUMMARY.append(line)


def guard(name, fn):
    t = time.time()
    try:
        fn()
        note(f"[{name}] done in {(time.time() - t) / 60:.1f} min")
    except Exception as exc:                      # keep going; report at the end
        import traceback
        traceback.print_exc()
        note(f"[{name}] FAILED: {exc!r}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------- S0
HW = hardware(os.path.join(ROOT, "hardware.json"))
if not torch.cuda.is_available():
    print("\n[WARN] no GPU detected -- enable the GPU accelerator; timings will be CPU.\n")


# ---------------------------------------------------------------------------- S1
def s1_validate():
    d = section_dir("S1_validate")
    r = run(default_cfg(mode="validate", validate_N=512, **QUIET))
    note(f"S1 gate PASS={r['passed']}  2D-1/r Eg={r['mg']:.2e} phi0={r['mp']:.2e}  "
         f"2D-log Eg={r['mlg']:.2e} phi0={r['mlp']:.2e}  q2D={r['mq']:.2e}  "
         f"LLL={r['ml']:.1e}  3D phi0={r['m3']:.2e}  orders={r['kernel_orders']}")
    shutil.copy(os.path.join(d, "validation.csv"), os.path.join(ROOT, "validation.csv"))


# ---------------------------------------------------------------------------- S2
def s2_kernels():
    section_dir("S2_kernels")
    rows = []
    # (a) -ln r kernel in a harmonic trap: virial with the corrected -G_C/2 term
    for gc in (2.0, 5.0, 10.0):
        cfg = default_cfg(mode="single", dimension="2D", s=2, Omega=0.0, beta2=100, G_C=gc,
                          kernel="log", L=10, Ngrid=256, seed="tf", res_tol=1e-7, maxit=20000,
                          save_figs=False, save_ckpt=False, **QUIET)
        psi, G, obs = ground_state(cfg, verbose=False)
        dg = diagnostics(psi, G, cfg)
        rows.append(dict(case="log", G_C=gc, l_z=float("nan"), E=dg["E"], mu=dg["mu"],
                         R90=dg["R90"], Egrav=dg["Egrav"], virial_lr_term=dg["virial_lr_term"],
                         virial_rel=dg["virial_rel"],
                         virial_rel_old_p0=abs(dg["virial"] - dg["virial_lr_term"]) /
                         max(1.0, sum(abs(dg[k]) for k in ("Ekin", "Etrap", "Econtact", "Egrav"))),
                         resid=obs["resid_rel"], iters=obs["iters"]))
        note(f"S2 log G_C={gc}: E={dg['E']:.10f} virial_rel={dg['virial_rel']:.2e} "
             f"(old p=0 formula would give {rows[-1]['virial_rel_old_p0']:.2e})")
    # (b) quasi-2D kernel: l_z -> 0 approaches the thin-disc 1/r kernel
    for lz in (2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.0):
        kern = "q2d" if lz > 0 else "newton"
        cfg = default_cfg(mode="single", dimension="quasi2D", s=2, Omega=0.0, beta2=100,
                          G_C=10.0, kernel=kern, l_z=lz, L=10, Ngrid=384, seed="tf",
                          res_tol=1e-7, maxit=20000, save_figs=False, save_ckpt=False, **QUIET)
        psi, G, obs = ground_state(cfg, verbose=False)
        dg = diagnostics(psi, G, cfg)
        rows.append(dict(case=kern, G_C=10.0, l_z=lz, E=dg["E"], mu=dg["mu"], R90=dg["R90"],
                         Egrav=dg["Egrav"], virial_lr_term=dg["virial_lr_term"],
                         virial_rel=dg["virial_rel"], resid=obs["resid_rel"], iters=obs["iters"],
                         dx=G["dx"]))
        note(f"S2 quasi2D l_z={lz}: E={dg['E']:.10f} Egrav={dg['Egrav']:.6f} "
             f"R90={dg['R90']:.4f} virial_rel={dg['virial_rel']:.2e}")
    write_rows(os.path.join(ROOT, "kernels_log_q2d.csv"), rows)


# ---------------------------------------------------------------------------- S3
def s3_gc_sweep():
    section_dir("S3_gc_sweep")
    rows = run(default_cfg(mode="sweep", dimension="2D", s=2, Omega=1.0, beta2=100, beta3=0,
                           kernel="newton", sweep_param="G_C", sweep_values=[32, 16, 10, 8, 5],
                           sweep_autobox=True, nseeds=3, sweep_continuation=True,
                           maxit=80000, tag="gc2d", **QUIET))
    for r in rows:
        note(f"S3 G_C={r['G_C']}: [{r['branch']}] E={r['E']:.8f} Nv={r['Nv']} Lz={r['Lz']:.3f} "
             f"R90={r['R90']:.3f} TF={r['tf_R90']:.3f} w_LLL={r['w_LLL']:.3f} "
             f"conv={r['converged']} res={r['resid_rel']:.1e} L={r['L']} N={r['N']}")
    write_rows(os.path.join(ROOT, "sweep_gc2d_G_C.csv"), rows)


# ---------------------------------------------------------------------------- S4
def s4_omega_sweep():
    section_dir("S4_omega_sweep")
    rows = run(default_cfg(mode="sweep", dimension="2D", s=2, beta2=200, G_C=0, kernel="none",
                           sweep_param="Omega", sweep_values=[0.0, 0.5, 0.7, 0.85, 0.9, 0.95],
                           L=12, Ngrid=384, nseeds=3, sweep_continuation=True, tag="omega",
                           **QUIET))
    for r in rows:
        note(f"S4 Omega={r['Omega']}: [{r['branch']}] E={r['E']:.8f} Lz={r['Lz']:.3f} "
             f"Nv={r['Nv']} w_LLL={r['w_LLL']:.3f} lll_param={r['lll_param']:.2f} "
             f"R90={r['R90']:.3f} TF={r['tf_R90']:.3f} conv={r['converged']}")
    write_rows(os.path.join(ROOT, "sweep_omega_Omega.csv"), rows)


# ---------------------------------------------------------------------------- S5
def s5_cq_ladder():
    """Self-bound cubic-quintic droplets in free space (s=0, Omega=0) at fixed
    flat-top density rho0 = -3 b2/(4 b3) = 0.75 and fixed TF radius, with the
    healing length shrinking as |b2| grows: the numerical droplet approaches
    the zero-pressure flat top as R/xi increases."""
    section_dir("S5_cq_ladder")
    rows = []
    rho0 = 0.75
    for b2 in (-25.0, -100.0, -250.0, -1000.0, -4000.0):
        b3 = -b2                                           # keeps rho0 = 0.75
        mu_tf = b2 * rho0 + b3 * rho0**2
        eps_bulk = 0.5 * b2 * rho0 + b3 * rho0**2 / 3.0     # energy per particle of the bulk
        R_tf = 1.0 / math.sqrt(math.pi * rho0)
        xi = 1.0 / math.sqrt(abs(mu_tf))                     # healing length at the flat top
        cfg = default_cfg(mode="single", dimension="2D", s=0, Omega=0.0, beta2=b2, beta3=b3,
                          G_C=0, kernel="none", L=2.5, Ngrid=1024, seed="tf", res_tol=1e-6,
                          save_figs=True, save_ckpt=False, tag=f"cq{int(-b2)}", **QUIET)
        dg = run(cfg)
        rows.append(dict(beta2=b2, beta3=b3, rho0=rho0, R_over_xi=R_tf / xi, peak=dg["peak"],
                         peak_over_rho0=dg["peak"] / rho0, mu=dg["mu"], mu_TF=mu_tf,
                         mu_rel_dev=(dg["mu"] - mu_tf) / abs(mu_tf), E=dg["E"], eps_bulk=eps_bulk,
                         E_rel_dev=(dg["E"] - eps_bulk) / abs(eps_bulk), R90=dg["R90"],
                         R90_TF=dg["tf_R90"], virial_rel=dg["virial_rel"],
                         resid=dg["resid_rel"], converged=dg["converged"]))
        note(f"S5 b2={b2}: R/xi={R_tf / xi:.1f} peak/rho0={dg['peak'] / rho0:.4f} "
             f"mu/mu_TF-1={rows[-1]['mu_rel_dev']:+.3e} E/eps-1={rows[-1]['E_rel_dev']:+.3e} "
             f"R90={dg['R90']:.4f} (TF {dg['tf_R90']:.4f})")
    write_rows(os.path.join(ROOT, "cq_ladder.csv"), rows)


# ---------------------------------------------------------------------------- S6
def s6_pcg_variants():
    section_dir("S6_pcg")
    cases = [
        ("2D_harmonic", dict(dimension="2D", s=2, Omega=0.0, beta2=200, G_C=0, kernel="none",
                             L=10, Ngrid=256, seed="tf")),
        ("2D_lattice_0.9", dict(dimension="2D", s=2, Omega=0.9, beta2=200, G_C=0, kernel="none",
                                L=12, Ngrid=384, seed="triangular")),
        ("3D_selfgrav", dict(dimension="3D", s=2, Omega=0.0, beta2=100, G_C=20, kernel="newton",
                             L=8, Ngrid=128, seed="tf")),
    ]
    rows = []
    for name, kw in cases:
        for beta in ("pr", "pr_precond", "none"):
            cfg = default_cfg(mode="single", res_tol=1e-6, maxit=60000, cg_beta=beta, rng=0,
                              save_figs=False, save_ckpt=False, **kw, **QUIET)
            G = make_grid(cfg)
            ground_state(dict(cfg, maxit=3), G=G, verbose=False)        # warm caches
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.time()
            psi, G, obs = ground_state(cfg, G=G, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rows.append(dict(case=name, cg_beta=beta, iters=obs["iters"], wall_s=time.time() - t,
                             E=obs["E"], resid=obs["resid_rel"], stop=obs["stop_reason"],
                             converged=obs["converged"]))
            note(f"S6 {name} {beta:>10}: it={obs['iters']:6d} t={rows[-1]['wall_s']:.1f}s "
                 f"E={obs['E']:.10f} res={obs['resid_rel']:.1e} {obs['stop_reason']}")
    write_rows(os.path.join(ROOT, "pcg_variants.csv"), rows)


# ---------------------------------------------------------------------------- S7
WORKER = textwrap.dedent('''
    import sys, json, time, torch, becgpp
    import becgpp.config as _bc; _bc.CFG_DEFAULTS["cg_beta"] = "pr"   # as in the v1.1 runs
    from becgpp import default_cfg, make_grid
    from becgpp.solvers import ground_state
    from becgpp.interactions import clear_kernel_cache
    out = []
    for ndim, Ns, iters in json.loads(sys.argv[1]):
        for N in Ns:
            cfg = default_cfg(mode="single", dimension=("3D" if ndim == 3 else "2D"), s=2,
                              Omega=0.0, beta2=100, G_C=5, kernel="newton",
                              L=(8.0 if ndim == 3 else 12.0), Ngrid=int(N), maxit=int(iters),
                              res_tol=0.0, energy_tol=0.0, seed="gaussian",
                              want_vortices=False, want_lll=False, show_inline=False)
            G = make_grid(cfg)
            ground_state(dict(cfg, maxit=3), G=G, verbose=False)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.time()
            psi, G, obs = ground_state(cfg, G=G, verbose=False)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            wall = time.time() - t0; it = max(int(obs["iters"]), 1); pts = int(N) ** ndim
            out.append(dict(device=becgpp.DEV, ndim=ndim, N=int(N), points=pts, iters=it,
                            ms_per_iter=1e3 * wall / it, ns_per_point_iter=1e9 * wall / (it * pts),
                            torch_threads=torch.get_num_threads()))
            del psi, G; clear_kernel_cache()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("BENCH_JSON:" + json.dumps(out))
''')


def _bench(conf, env_extra):
    wf = os.path.join(ROOT, "_bench_worker.py")
    open(wf, "w").write(WORKER)
    env = dict(os.environ)
    env.update(env_extra)
    p = subprocess.run([sys.executable, wf, json.dumps(conf)], capture_output=True, text=True, env=env)
    tag = [l for l in p.stdout.splitlines() if l.startswith("BENCH_JSON:")]
    if not tag:
        print(p.stdout[-2000:], p.stderr[-2000:])
        raise RuntimeError("benchmark worker failed")
    return json.loads(tag[0].split(":", 1)[1])


def s7_perf():
    gpu = _bench([[2, [256, 384, 512, 768, 1024], 200], [3, [64, 96, 128, 160], 200]], {})
    for r in gpu:
        r.update(gpu=HW["gpu"])
    write_rows(os.path.join(ROOT, "perf_gpu.csv"), gpu)
    ns = [r["ns_per_point_iter"] for r in gpu]
    note(f"S7 GPU ({HW['gpu']}): ns/pt/iter min={min(ns):.1f} max={max(ns):.1f} "
         f"mean={sum(ns) / len(ns):.1f} over {len(ns)} grids")
    conf = [[2, [128, 256, 384], 60], [3, [48, 64], 40]]
    g = {(r["ndim"], r["N"]): r for r in _bench(conf, {})}
    c = _bench(conf, {"CUDA_VISIBLE_DEVICES": ""})
    rows = []
    for r in c:
        gg = g[(r["ndim"], r["N"])]
        rows.append(dict(ndim=r["ndim"], N=r["N"], points=r["points"], gpu_ms=gg["ms_per_iter"],
                         cpu_ms=r["ms_per_iter"], speedup=r["ms_per_iter"] / gg["ms_per_iter"],
                         gpu=HW["gpu"], cpu=HW["cpu_model"], cpu_threads=r["torch_threads"]))
        note(f"S7 {r['ndim']}D N={r['N']}: GPU {gg['ms_per_iter']:.2f} ms  CPU {r['ms_per_iter']:.2f} ms "
             f"({r['torch_threads']} threads)  speedup {rows[-1]['speedup']:.1f}x")
    write_rows(os.path.join(ROOT, "perf_cpu_gpu.csv"), rows)


for key, fn in (("S1", s1_validate), ("S2", s2_kernels), ("S3", s3_gc_sweep),
                ("S4", s4_omega_sweep), ("S5", s5_cq_ladder), ("S6", s6_pcg_variants),
                ("S7", s7_perf)):
    if SECTIONS.get(key, False):
        guard(key, fn)

open(os.path.join(ROOT, "SUMMARY.txt"), "w").write("\n".join(SUMMARY) + "\n")
arch = shutil.make_archive("/kaggle/working/becgpp_verification" if os.path.isdir("/kaggle/working")
                           else os.path.abspath("becgpp_verification"), "zip", root_dir=ROOT)
print("\n" + "=" * 78 + "\n" + "\n".join(SUMMARY))
print(f"\nArchive: {arch}  -- download it and send it back for the manuscript update.")
