# =============================================================================
#  becGPP on Kaggle -- install from GitHub and run the paper's showcase cases.
#  Turn on the GPU accelerator (Settings -> Accelerator -> GPU) before running.
#  Paste each numbered block into its own Kaggle cell.
# =============================================================================


# ---- Cell 1 -- install the package straight from GitHub -----------------
# (double precision + GPU auto-select; the only run-time deps are torch/numpy/matplotlib,
#  which Kaggle already ships, so the install is fast.)
!pip install -q git+https://github.com/Duyle0503/becGPP.git

import becgpp
from becgpp import default_cfg, run, paths
print("becGPP", becgpp.__version__, "| device:", becgpp.DEV)
paths.configure("/kaggle/working/out")   # CSVs, figures and checkpoints land here


# ---- Cell 2 -- validation gate (run this first) --------------------------
# Checks the Coulomb solver, the Landau-level identities and the flat-top relation
# against closed forms, and reports the kernel convergence order in 2D and 3D.
run(default_cfg(mode="validate", validate_N=512))


# ---- Cell 3 -- CASE A: harmonic repulsive BEC (2D) -----------------------
# Plain contact gas, no rotation, no long-range term. TF = inverted parabola.
# res= sets the convergence tolerance yourself (KKT residual ||(H-mu)psi||/max(1,|mu|)).
# Default res=1e-4; use a smaller res for tighter convergence (slower), larger for a quick look.
# (res is an alias for res_tol; etol= aliases energy_tol, N= aliases Ngrid, dim= aliases dimension.)
diag = run(default_cfg(
    mode="single", dimension="2D", s=2, Omega=0.0,
    beta2=200, beta3=0, G_C=0, kernel="none",
    seed="tf", L=12, N=384,
    res=1e-5))                    # <- user-set residual tolerance
print("E =", diag["E"], " mu =", diag["mu"], " R90 =", diag["R90"],
      " resid =", diag["resid_rel"], " converged =", diag["converged"])


# ---- Cell 4 -- CASE B: rotating vortex lattice (2D, Omega = 0.9) ----------
# Triangular seed + multi-seed selection so the solver lands on the Abrikosov branch.
diag = run(default_cfg(
    mode="single", dimension="2D", s=2, Omega=0.9,
    beta2=200, beta3=0, G_C=0, kernel="none",
    seed="triangular", nseeds=3, L=12, Ngrid=384))
print("E =", diag["E"], " Lz =", diag["Lz"], " Nv =", diag["Nv"], " w_LLL =", diag["w_LLL"])


# ---- Cell 5 -- CASE C: cubic-quintic flat-top droplet (2D) ---------------
# Attractive 2-body + repulsive 3-body -> self-bound; density saturates at rho0 = -3 beta2 / 4 beta3.
diag = run(default_cfg(
    mode="single", dimension="2D", s=2, Omega=1.0,
    beta2=-250, beta3=250, G_C=0, kernel="none",
    seed="tf", L=8, Ngrid=256))
print("E =", diag["E"], " mu =", diag["mu"], " peak rho =", diag["peak"], " R90 =", diag["R90"])


# ---- Cell 6 -- CASE D: boson star (3D Newton self-gravity) ---------------
# Long-range 1/r attraction balanced by contact repulsion; spherical ground state.
# IMPORTANT: the kernel is only active when G_C > 0. If you set kernel="newton"
# but leave G_C=0, there is no long-range term and the log reports kernel="none"
# (the code now prints a [warn] in that case). So set BOTH kernel AND G_C.
diag = run(default_cfg(
    mode="single", dimension="3D", s=2, Omega=0.0,
    beta2=100, beta3=0, G_C=20, kernel="newton",   # <- G_C>0 activates the kernel
    seed="tf", L=8, Ngrid=128))
print("E =", diag["E"], " mu =", diag["mu"], " R90 =", diag["R90"], " oblateness =", diag.get("oblateness"))


# ---- Cell 7 -- CASE E: rotating oblate self-gravitating cloud (3D) --------
# Same star at Omega = 0.9: it flattens along z and threads a few vortex lines.
diag = run(default_cfg(
    mode="single", dimension="3D", s=2, Omega=0.9,
    beta2=100, beta3=0, G_C=20, kernel="newton",
    seed="tf", L=8, Ngrid=128))
print("E =", diag["E"], " Lz =", diag["Lz"], " oblateness =", diag.get("oblateness"))


# ---- Cell 8 -- CASE F: quasi-2D Newton (pancake 1/r) ---------------------
# In-plane reduction of the 3D Newtonian interaction; kernel = "newton" in a 2D geometry.
diag = run(default_cfg(
    mode="single", dimension="quasi2D", s=2, Omega=0.0,
    beta2=100, beta3=0, G_C=10, kernel="newton",
    seed="tf", L=10, Ngrid=256))
print("E =", diag["E"], " R90 =", diag["R90"], " w_LLL =", diag["w_LLL"])


# ---- Cell 9 -- a parameter sweep (mass-radius trend of the 3D star) -------
# sweep varies exactly ONE key (sweep_param) over sweep_values; EVERY OTHER key
# stays FIXED at whatever you pass in the cfg. Here dimension, s, Omega, beta2,
# beta3, kernel, L, Ngrid are all held fixed and only G_C moves. sweep_param can
# be ANY numeric key -- "Omega", "beta2", "beta3", "L", "Ngrid", ... -- so e.g.
# sweep_param="Omega", sweep_values=[0,0.5,0.9] fixes the interactions and scans rotation.
# (When sweeping a kernel case, keep kernel set AND G_C in the swept list > 0.)
run(default_cfg(
    mode="sweep", dimension="3D", s=2, Omega=0.0,
    beta2=100, beta3=0, kernel="newton", seed="tf", L=8, Ngrid=96,
    sweep_param="G_C", sweep_values=[5, 10, 20, 40, 80]))


# ---- Cell 10 -- Thomas-Fermi extractor alone (no solve) ------------------
# Just the TF envelope for a configuration -- instant, useful as a seed/sanity check.
run(default_cfg(
    mode="tf_only", dimension="2D", s=2, Omega=0.0,
    beta2=200, beta3=0, G_C=0, kernel="none", L=12, Ngrid=384))


# ---- Cell 11 -- show a saved figure inline -------------------------------
from IPython.display import Image, display
import glob, os
figs = sorted(glob.glob("/kaggle/working/out/fig/*.png"))
print("\n".join(os.path.basename(f) for f in figs))
if figs:
    display(Image(figs[-1]))          # newest figure


# ---- Cell 12 -- reuse the outputs (no re-solving) ------------------------
# Every `single` run writes THREE reusable artifacts under out/:
#   * out/<rid>.json      -- full cfg + all diagnostics, machine-readable
#   * out/ckpt/<rid>.pt   -- the wavefunction (torch), for resume / re-plot
#   * out/single_summary.csv -- one row per run appended across the session
# Load a result back without recomputing:
import json, glob
jf = sorted(glob.glob("/kaggle/working/out/*.json"))[-1]
rec = json.load(open(jf))
print("kernel:", rec["cfg"]["kernel"], "G_C:", rec["cfg"]["G_C"])
print("E:", rec["diagnostics"]["E"], " R90:", rec["diagnostics"]["R90"],
      " peak(max density):", rec["diagnostics"]["peak"])
# Re-running the SAME cfg auto-resumes from out/ckpt/<rid>.pt instead of starting cold.


# ---- Cell 13 -- smoke: fast end-to-end sanity check ----------------------
# setup = just pick a geometry. It runs the validation gate (N=192) + one tiny
# solve (L=10, N=128/72, maxit=1500) + a TF extraction, all in the geometry you
# name. Use it to confirm a fresh install works before a long run. ~1 minute.
run(default_cfg(mode="smoke", dimension="3D"))


# ---- Cell 14 -- scan_all: reproduce the entire paper batch ---------------
# setup = mode + scan_profile only; it fixes all its own physics internally
# (validation + 8 showcase singles + 4 sweeps + convergence + GPU benchmark).
#   scan_profile="quick" -> small grids / fewer points, a few minutes (dry run)
#   scan_profile="full"  -> paper-grade data, ~2 hours on one GPU
# Writes CSVs, all figures, and out/scan/scan_manifest.json (code version,
# device, timings). Then zip out/ and download it.
run(default_cfg(mode="scan_all", scan_profile="quick"))
# from becgpp import zip_results; zip_results()   # -> becgpp_results.zip
