"""Controlled CPU sensitivity study of becGPP's shift and BB choices.

This is NOT a recreation of the energy-based shift or second-variation step
of another solver. It compares this implementation with two deliberately
simple controls: fixed positive shift sigma=1, and removal of the BB trial
blend while keeping its accepted-step growth/Armijo safeguards. No production
module is changed. See each run's provenance and source hashes.

python benchmarks/cpc_ablation.py --out PATH --repeats 3
"""
import argparse
import ast
import csv
import hashlib
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Use this source checkout, even when another becGPP is installed in the venv.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import torch
torch.set_num_threads(2)
from becgpp import default_cfg, make_grid, make_seed
from becgpp import solvers
from becgpp.constants import DEV
from becgpp.grid import dV
from becgpp.interactions import clear_kernel_cache

CASES = [
    ("harmonic2d", dict(dimension="2D", Ngrid=64, L=8., beta2=200., seed="tf")),
    ("rotating2d", dict(dimension="2D", Ngrid=64, L=9., beta2=100., Omega=.8,
                        seed="triangular", maxit=6000)),
    ("harmonic3d", dict(dimension="3D", Ngrid=32, L=6., beta2=100., seed="tf")),
    ("selfgrav3d", dict(dimension="3D", Ngrid=32, L=6., beta2=100., G_C=20.,
                        kernel="newton", seed="tf")),
]
VARIANTS = [("median_bb", True, True), ("median_no_bb", True, False),
            ("fixed1_bb", False, True), ("fixed1_no_bb", False, False)]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def normalized_source(path):
    return path.read_bytes().replace(b"\r\n", b"\n")

def variant_function(adaptive, bb):
    source = inspect.getsource(solvers.ground_state)
    marker = "if it > 0 and not restarted:"
    assert source.count(marker) == 1
    derived = source if bb else source.replace(marker, "if False:")
    state = dict(energy_evaluations=0, hamiltonian_evaluations=0, shifts=[])
    namespace = dict(vars(solvers))
    original_energy = solvers.energy_components
    original_hamiltonian = solvers.apply_H

    def energy(*args, **kwargs):
        state["energy_evaluations"] += 1
        return original_energy(*args, **kwargs)

    def hamiltonian(*args, **kwargs):
        state["hamiltonian_evaluations"] += 1
        return original_hamiltonian(*args, **kwargs)

    def preconditioner(grad, psi, grid, cfg, phi=None):
        if adaptive:
            z, sigma = solvers._preconditioner(grad, psi, grid, cfg, phi)
        else:
            sigma = 1.0
            z = torch.fft.ifftn(torch.fft.fftn(grad) / (.5 * grid["K2"] + sigma))
            z = solvers._proj_tangent(z, psi, dV(grid))
        state["shifts"].append(sigma)
        return z, sigma

    namespace.update(energy_components=energy, apply_H=hamiltonian,
                     _preconditioner=preconditioner)
    exec(compile(derived, "<audited-ablation-ground-state>", "exec"), namespace)
    return namespace["ground_state"], state, derived

def summarize(out):
    records = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json"))]
    table = []
    for case, _ in CASES:
        for label, _, _ in VARIANTS:
            rows = [r for r in records if r["case"] == case and r["variant"] == label]
            if not rows:
                continue
            table.append(dict(case=case, variant=label, repeats=len(rows),
                              residual_passes=sum(r["strict_residual_pass"] for r in rows),
                              median_seconds=statistics.median(r["seconds"] for r in rows),
                              iterations=[r["iters"] for r in rows],
                              trial_energy_evaluations=[r["trial_energy_evaluations"] for r in rows],
                              energies=[r["energy"] for r in rows],
                              residuals=[r["residual"] for r in rows],
                              stops=[r["stop_reason"] for r in rows],
                              final_shift=rows[-1]["final_shift"]))
    save(out / "summary.json", dict(rows=table, scope="CPU same-grid sensitivity; not GPU speed or continuum accuracy"))
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "variant", "repeats", "residual_passes", "median_seconds", "iterations", "trial_energy_evaluations", "energies", "residuals", "stops", "final_shift"])
        writer.writeheader()
        writer.writerows(table)
    return table

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--case", choices=[c[0] for c in CASES])
    args = p.parse_args()
    assert DEV == "cpu", "This protocol is explicitly a CPU experiment"
    assert args.repeats >= 1
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    hashes = {p.relative_to(REPO).as_posix(): digest(normalized_source(p))
              for p in sorted((REPO / "becgpp").glob("*.py"))}
    try:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        source_commit = None  # Source ZIPs have no .git; hashes below remain available.
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                      git_commit=source_commit,
                      source_hashes=hashes, benchmark_hash=digest(Path(__file__).read_bytes()),
                      python=sys.version, torch=torch.__version__, numpy=np.__version__,
                      platform=platform.platform(), device=DEV, torch_threads=torch.get_num_threads(),
                      precision="float64/complex128", residual_target=1e-6,
                      energy_stop_disabled=True, fixed_shift=1., variants=VARIANTS,
                      timing_scope="Ground-state call excluding grid/seed construction; includes final residual/observables. Three-iteration variant-specific warmup precedes each call.",
                      count_scope="Trial energy calls include the initial energy and line-search candidates, not final diagnostic-only energy; Hamiltonian counts refer to in-loop residuals.",
                      limitations=["Sensitivity controls are not another published solver.",
                                   "No assertion, tolerance, Armijo condition or CG safeguard is changed.",
                                   "Same-grid agreement does not prove continuum convergence or global minimality.",
                                   "CPU medians are not GPU speedups."])
    save(out / "provenance.json", provenance)
    (out / "runs").mkdir(exist_ok=True)
    (out / "states").mkdir(exist_ok=True)
    for case_index, (case, over) in enumerate(CASES):
        if args.case and case != args.case:
            continue
        c = dict(dimension="2D", s=2, trap_coeff=.5, Omega=0., beta2=100., beta3=0.,
                 G_C=0., kernel="none", L=8., Ngrid=64, pad=2, maxit=3000,
                 res_tol=1e-6, energy_tol=0., conv_window=2000, step=1., step_max=3.,
                 linesearch_max=8, precond_shift_min=.5, cg_beta="pr_precond", cg_restart=30,
                 seed="tf", seed_noise=.01, seed_phase_noise=.05, rng=0, nseeds=1,
                 save_figs=False, save_ckpt=False, show_inline=False, want_vortices=False,
                 want_lll=False, record_trace=True)
        c.update(over); c = default_cfg(**c)
        clear_kernel_cache()
        grid = make_grid(c)
        initial = make_seed(grid, c)
        initial /= torch.sqrt((initial.abs() ** 2).sum() * dV(grid))
        initial_bytes = initial.cpu().numpy().tobytes()
        np.savez_compressed(out / "states" / (case + "_initial.npz"), psi=initial.cpu().numpy())
        save(out / (case + "_config.json"), dict(cfg=c, initial_array_sha256=digest(initial_bytes)))
        for repeat in range(args.repeats):
            # Rotate the order across repeats; this prevents one label always going first.
            order = VARIANTS[(repeat + case_index) % 4:] + VARIANTS[:(repeat + case_index) % 4]
            for label, adaptive, bb in order:
                target = out / "runs" / f"{case}_{label}_{repeat}.json"
                if target.exists():
                    continue
                fn, state, derived = variant_function(adaptive, bb)
                (out / (label + "_derived_ground_state.py")).write_text(derived, encoding="utf-8")
                fn(dict(c, maxit=3, res_tol=0., energy_tol=0.), G=grid, psi0=initial.clone(), verbose=False)
                state.update(energy_evaluations=0, hamiltonian_evaluations=0, shifts=[])
                start = time.perf_counter()
                psi, _, obs = fn(c, G=grid, psi0=initial.clone(), verbose=False)
                elapsed = time.perf_counter() - start
                trace = obs.pop("trace")
                mass_error = abs((psi.abs() ** 2).sum().item() * dV(grid) - 1.)
                assert torch.isfinite(psi).all().item() and mass_error < 1e-12
                assert psi.dtype == torch.complex128 and psi.device.type == "cpu"
                assert all(trace[i][2] <= trace[i-1][2] + 1e-13 * max(1., abs(trace[i-1][2])) for i in range(1, len(trace)))
                with target.with_suffix(".csv").open("w", newline="") as f:
                    writer = csv.writer(f); writer.writerow(["seconds", "iteration", "energy", "relative_residual"]); writer.writerows(trace)
                np.savez_compressed(out / "states" / f"{case}_{label}_{repeat}.npz", psi=psi.cpu().numpy())
                record = dict(case=case, variant=label, repeat=repeat, seconds=elapsed,
                              energy=obs["E"], residual=obs["resid_rel"], strict_residual_pass=obs["resid_rel"] < c["res_tol"],
                              iters=obs["iters"], stop_reason=obs["stop_reason"],
                              trial_energy_evaluations=state["energy_evaluations"],
                              hamiltonian_evaluations=state["hamiltonian_evaluations"],
                              line_search_restarts=obs["line_search_restarts"],
                              final_shift=state["shifts"][-1], shift_min=min(state["shifts"]), shift_max=max(state["shifts"]),
                              mass_error=mass_error, virial_relative=obs["virial_rel"],
                              derived_function_sha256=digest(derived.encode()), initial_array_sha256=digest(initial_bytes))
                save(target, record)
                print(json.dumps(record), flush=True)
                summarize(out)
                del psi, trace
        del grid, initial
        clear_kernel_cache()
    table = summarize(out)
    print("COMPLETE", len(table), "case/variant groups", flush=True)

if __name__ == "__main__":
    main()
