"""Turn the benchmark outputs (benchmarks/results/) into manuscript numbers.

Usage :

    python make_paper_numbers.py --figdir figs        # from benchmarks/

Prints the LaTeX rows / numbers quoted in the paper and
writes fig_convergence.pdf (kernel orders incl. -ln r) and fig_mass_radius.pdf
into --figdir. Missing folders are skipped.
"""
import argparse
import csv
import json
import math
import os
import shutil
from collections import defaultdict


def rows(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def sci(x, d=1):
    x = f(x)
    if not math.isfinite(x):
        return "--"
    if x == 0:
        return "$0$"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10**e
    return f"${m:.{d}f}\\times10^{{{e}}}$"


def header(t):
    print("\n" + "=" * 78 + f"\n% {t}\n" + "=" * 78)


def verification(d, figdir):
    hw = os.path.join(d, "hardware.json")
    if os.path.isfile(hw):
        header("Hardware (Sec. perf, first sentence)")
        print(json.dumps(json.load(open(hw)), indent=2))
    val = rows(os.path.join(d, "validation.csv"))
    if val:
        header("Table validation")
        by = defaultdict(dict)
        for r in val:
            by[(r["test"], r["case"])][r["quantity"]] = f(r["rel_error"])
        for (t, c), q in by.items():
            if "kernel_order" in t:
                continue
            print(f"{t:14s} {c:14s} " + "  ".join(f"{k}={sci(v)}" for k, v in q.items()))
        header("Kernel orders + fig_convergence.pdf")
        order = defaultdict(list)
        for r in val:
            if r["test"].endswith("_kernel_order"):
                order[r["test"].replace("_kernel_order", "")].append(
                    (f(r["case"].split("=")[1]), f(r["rel_error"]), f(r["order"])))
        for k, v in order.items():
            print(f"{k}: fitted order = {v[0][2]:.2f}")
        if order and figdir:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams.update({"font.size": 11, "font.family": "serif", "mathtext.fontset": "cm",
                                 "xtick.direction": "in", "ytick.direction": "in",
                                 "legend.frameon": False})
            fig, ax = plt.subplots(figsize=(4.2, 3.3))
            style = {"2D": ("o-", "#1f5c8b", r"$1/r$, 2D"), "2D_log": ("^-", "#2e7d32", r"$-\ln r$, 2D"),
                     "3D": ("s-", "#b3282d", r"$1/r$, 3D")}
            for k, v in order.items():
                mk, col, lab = style.get(k, ("x-", "k", k))
                dx = [a for a, _, _ in v]
                er = [b for _, b, _ in v]
                ax.loglog(dx, er, mk, color=col, ms=5, label=f"{lab} ($p={v[0][2]:.2f}$)")
            xs = [min(a for v in order.values() for a, _, _ in v), max(a for v in order.values() for a, _, _ in v)]
            y0 = max(b for v in order.values() for _, b, _ in v)
            ax.loglog(xs, [y0 * (x / xs[1])**2 for x in xs], "k--", lw=0.8, label=r"$\propto\Delta x^2$")
            ax.set_xlabel(r"$\Delta x$")
            ax.set_ylabel(r"relative error of $\Phi(0)$")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(figdir, "fig_convergence.pdf"))
            print(f"[fig] {os.path.join(figdir, 'fig_convergence.pdf')}")
    kr = rows(os.path.join(d, "kernels_log_q2d.csv"))
    if kr:
        header("-ln r virial and quasi-2D crossover (Sec. verify / model)")
        for r in kr:
            print(f"{r['case']:7s} G_C={r['G_C']:>5} l_z={r['l_z']:>6}  E={f(r['E']):.8f}  "
                  f"virial_rel={sci(r['virial_rel'])}  old(p=0)={sci(r.get('virial_rel_old_p0'))}")
    for name, key in (("sweep_gc2d_G_C.csv", "G_C"), ("sweep_omega_Omega.csv", "Omega")):
        sw = rows(os.path.join(d, name))
        if sw:
            header(f"{name}")
            for r in sw:
                print(f"{key}={r[key]:>5}  [{r.get('branch')}] conv={r['converged']} E={f(r['E']):.6f} "
                      f"Nv={r['Nv']} Lz={f(r['Lz']):.2f} R90={f(r['R90']):.3f} TF={f(r['tf_R90']):.3f} "
                      f"w_LLL={f(r['w_LLL']):.3f} lll={f(r.get('lll_param')):.2f} res={sci(r['resid_rel'])}")
    cq = rows(os.path.join(d, "cq_ladder.csv"))
    if cq:
        header("Cubic-quintic ladder (Sec. cq)")
        for r in cq:
            print(f"beta2={r['beta2']:>7}  R/xi={f(r['R_over_xi']):6.1f}  peak/rho0-1={f(r['peak_over_rho0']) - 1:+.3e}  "
                  f"mu/muTF-1={f(r['mu_rel_dev']):+.3e}  E/eps-1={f(r['E_rel_dev']):+.3e}")
    pc = rows(os.path.join(d, "pcg_variants.csv"))
    if pc:
        header("Table pcg (iter & s per variant)")
        by = defaultdict(dict)
        for r in pc:
            by[r["case"]][r["cg_beta"]] = r
        for case, v in by.items():
            cells = " & ".join(f"{v[b]['iters']} & {f(v[b]['wall_s']):.1f}" if b in v else "-- & --"
                               for b in ("pr", "pr_precond", "none"))
            print(f"{case} & {cells} \\\\")
    pg = rows(os.path.join(d, "perf_gpu.csv"))
    if pg:
        header("Table bench")
        for r in pg:
            print(f"{r['ndim']}D & {r['N']} & {int(r['points']):,} & {f(r['ms_per_iter']):.1f} & "
                  f"{f(r['ns_per_point_iter']):.1f} \\\\".replace(",", "\\,"))
        ns = [f(r["ns_per_point_iter"]) for r in pg]
        print(f"% ns/pt/iter range {min(ns):.0f}-{max(ns):.0f}, mean {sum(ns) / len(ns):.0f}")
    pc2 = rows(os.path.join(d, "perf_cpu_gpu.csv"))
    if pc2:
        header("Table cpugpu")
        for r in pc2:
            print(f"{r['ndim']}D & {r['N']} & {int(r['points']):,} & {f(r['gpu_ms']):.1f} & "
                  f"{f(r['cpu_ms']):.1f} & {f(r['speedup']):.1f} \\\\".replace(",", "\\,"))


def selfgravity_limits(d, figdir):
    sn = rows(os.path.join(d, "ref_schrodinger_newton.csv"))
    if sn:
        header("Table sn (Schrodinger-Newton)")
        for r in sn:
            print(f"{r['N']} & {f(r['dx']):.4f} & {f(r['E_over_GC2']):.6f} & {f(r['mu_over_GC2']):.6f} & "
                  f"{f(r['R99_times_GC']):.4f} \\\\")
        ex = sn[-1]
        print(f"% deviations of extrapolation: E {f(ex['E_over_GC2']) / -0.05426 - 1:+.1e}, "
              f"mu {f(ex['mu_over_GC2']) / -0.16278 - 1:+.1e}, R99 {f(ex['R99_times_GC']) / 9.946 - 1:+.1e}")
    mr = rows(os.path.join(d, "ref_mass_radius.csv"))
    if mr:
        header("Mass-radius / TF limit")
        for r in mr:
            line = f"beta2={r['beta2']:>7} chi={f(r['chi']):9.2f} R99*G={f(r['R99_times_GC']):.4f}"
            if r.get("R99_TF_times_GC"):
                line += (f"  R99/TF-1={f(r['R99_times_GC']) / f(r['R99_TF_times_GC']) - 1:+.2e}"
                         f"  mu/TF-1={f(r['mu']) / f(r['mu_TF']) - 1:+.2e}  E/TF-1={f(r['E']) / f(r['E_TF']) - 1:+.2e}")
            print(line)
    src = os.path.join(d, "fig_mass_radius.pdf")
    if os.path.isfile(src) and figdir:
        shutil.copy(src, os.path.join(figdir, "fig_mass_radius.pdf"))
        print(f"[fig] copied fig_mass_radius.pdf -> {figdir}")


def itp_agreement(d):
    it = rows(os.path.join(d, "itp_vs_pcg.csv"))
    if it:
        header("Table itp: energies, Delta-t-extrapolated agreement (v1.1 timings unused)")
        for r in it:
            print(f"{r['case']} & {f(r['E_pcg']):.8f} & {sci(r['rel_diff_extrap'])} & "
                  f"{f(r['t_pcg_target_s']):.2f} & {f(r['t_itp_to_target_s']):.2f} \\\\"
                  f"   % speedup x{f(r['speedup_to_target']):.1f}")


def itp_timing(d, figdir):
    """Times to 1e-4 / 1e-6 (Table itp) and the convergence figure (Fig. itp)."""
    tm = rows(os.path.join(d, "timing.csv"))
    if tm:
        header("Table itp: time to 1e-4 / 1e-6 (s)")
        for case in dict.fromkeys(r["case"] for r in tm):
            cells = []
            for m in ("PCG-pp", "PCG-pr", "ITP"):
                r = next((x for x in tm if x["case"] == case and x["method"] == m), None)
                a, b = (f(r["t_to_1e-04"]), f(r["t_to_1e-06"])) if r else (float("nan"),) * 2
                cells.append(f"{a:.2f}\\,/\\,{b:.2f}" if math.isfinite(b) else "--")
            print(f"{case} & " + " & ".join(cells) + " \\\\")
    tr = rows(os.path.join(d, "traces.csv"))
    if not (tr and figdir):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm",
                         "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
    cases = list(dict.fromkeys(r["case"] for r in tr))
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
    style = {"PCG-pp": ("-", "#1f5c8b", "becGPP (pr_precond)"),
             "PCG-pr": ("--", "#6a9fcb", "becGPP (pr)"),
             "ITP": ("-", "#b3282d", "imaginary time")}
    for k, (ax, c) in enumerate(zip(axs.ravel(), cases)):
        for m, (ls, col, lab) in style.items():
            pts = [r for r in tr if r["case"] == c and r["method"] == m and f(r["t"]) > 0]
            ax.loglog([f(r["t"]) for r in pts], [max(f(r["rel_err"]), 1e-16) for r in pts],
                      ls, color=col, lw=1.2, label=lab)
        ax.axhline(1e-5, color="0.6", lw=0.6, ls=":")
        ax.text(0.04, 0.06, f"({'abcd'[k]})", transform=ax.transAxes)
        ax.set_xlabel("wall time (s)")
        ax.set_ylabel(r"$|E-E_{\rm ref}|/|E_{\rm ref}|$")
    axs[0, 0].legend(fontsize=7.5, loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_itp_timing.pdf"))
    print(f"[saved] {os.path.join(figdir, 'fig_itp_timing.pdf')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("verification", nargs="?", default="results/verification")
    ap.add_argument("selfgravity_limits", nargs="?", default="results/selfgravity_limits")
    ap.add_argument("itp_agreement", nargs="?", default="results/itp_agreement")
    ap.add_argument("itp_timing", nargs="?", default="results/itp_timing")
    ap.add_argument("--figdir", default=None)
    a = ap.parse_args()
    if a.figdir:
        os.makedirs(a.figdir, exist_ok=True)
    verification(a.verification, a.figdir)
    selfgravity_limits(a.selfgravity_limits, a.figdir)
    itp_agreement(a.itp_agreement)
    itp_timing(a.itp_timing, a.figdir)
