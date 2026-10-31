"""Orchestration and offline assessment for the self-contained Kaggle cell.

No solver module is modified. This module has no GPU/import side effects.
"""
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
import urllib.request
import uuid
import zipfile
import xml.etree.ElementTree as ET

PROTOCOL_VERSION = "becgpp-trapped3d-confirm-v2"
CRITERIA = dict(mass_error=1e-10, edge_mass_two_cells=1e-6, outer_shell_mass=1e-8,
                finest_virial_relative=5e-4, points_per_R90=12.,
                last_grid_energy_change=1e-3, last_grid_mu_change=1e-3,
                last_grid_rrms_change=5e-3, last_grid_peak_change=2e-2,
                h2_extrapolation_energy_correction=1e-3, observed_order_low=1.5, observed_order_high=2.5,
                padding_energy_change=2e-11, padding_density_l2=1e-5,
                box_energy_change=1e-5, box_density_l2=1e-4,
                seed_energy_change=1e-8, seed_density_l2=5e-5,
                tighter_solve_energy_change=1e-9, grid_increment_over_optimization_change=100.)
DEFAULT_SETTINGS = dict(run_root=None, source_dir=None, resume_zip=None, gpu_index=0,
                        run_release_tests=True, session_budget_hours=8., job_timeout_seconds=7200,
                        heartbeat_seconds=30, memory_fraction=.82, retry_failed_jobs=True)
TERMINAL = {"PASS", "REVIEW", "FAIL"}


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def safe_path(root, name):
    assert isinstance(name, str) and "\\" not in name and ":" not in name
    assert not Path(name).is_absolute() and ".." not in Path(name).parts
    p = (Path(root) / name).resolve()
    assert p.is_relative_to(Path(root).resolve())
    return p


def plan_jobs(include_tests=True):
    jobs = []
    if include_tests:
        jobs.append(dict(name="release_cuda_109", kind="regression"))
    for n in (96, 128, 160, 192):
        jobs.append(dict(name="grid_N" + str(n), kind="solve", N=n, L=8., pad=2,
                         seed="gaussian", rng=0, residual_target=1e-6, max_iterations=4096))
    for L, n in ((6., 96), (10., 160), (12., 192)):
        jobs.append(dict(name="box_L" + str(int(L)), kind="solve", N=n, L=L, pad=2,
                         seed="gaussian", rng=0, residual_target=1e-6, max_iterations=4096))
    for n in (96, 128):
        jobs.append(dict(name="padding_N" + str(n) + "_pad3", kind="solve", N=n, L=8., pad=3,
                         seed="gaussian", rng=0, residual_target=1e-6, max_iterations=4096))
    jobs.append(dict(name="seed_TF_N128", kind="solve", N=128, L=8., pad=2,
                     seed="tf", rng=0, residual_target=1e-6, max_iterations=4096))
    jobs.append(dict(name="tight_N192", kind="solve", N=192, L=8., pad=2,
                     seed="gaussian", rng=0, residual_target=1e-7, max_iterations=2048, initial_from="grid_N192"))
    return jobs


def make_cfg(job):
    return dict(dimension="3D", s=2, trap_coeff=.5, Omega=0., beta2=100., beta3=0., G_C=20., kernel="newton",
                Ngrid=job["N"], L=job["L"], pad=job["pad"], seed=job["seed"], rng=job["rng"], nseeds=1,
                seed_noise=.01 if job["seed"] == "tf" else 0., seed_phase_noise=.05 if job["seed"] == "tf" else 0.,
                seed_winding=0, res_tol=job["residual_target"], energy_tol=0., maxit=job["max_iterations"],
                step=1., step_max=3., linesearch_max=8, precond_shift_min=.5, cg_beta="pr_precond", cg_restart=30,
                save_figs=False, save_ckpt=False, show_inline=False, want_vortices=False, want_lll=False, record_trace=True)


def definition(spec, worker_source, core_source, include_tests=True):
    return dict(protocol=PROTOCOL_VERSION, source_commit=spec["source_commit"], source_hashes=spec["source_hashes"],
                expected_tests=spec["expected_tests"], expected_test_ids_sha256=spec["expected_test_ids_sha256"],
                worker_sha256=digest(worker_source.encode()), core_sha256=digest(core_source.encode()),
                jobs=plan_jobs(include_tests), criteria=CRITERIA, chunk_iterations=256,
                limitations=["Same-discretization checks and finite-resolution extrapolations do not certify a global minimum.",
                    "R90 is a trap-centered histogram estimate with bin width dx/8.",
                    "Box comparisons use dx=0.125, not the finest-grid spacing; boundary mass is also tested on every grid.",
                    "Chunked continuation restarts CG/BB history. Timings are diagnostics, not speedup benchmarks.",
                    "Completed cached jobs retain their original GPU/software environment; no cross-session timing comparison.",
                    "PASS is for this nonrotating beta2=100, G_C=20 trapped Newton case and its declared tolerances only."])


def job_signature(defn, job, input_sha=None):
    return digest(canonical(dict(protocol_fingerprint=digest(canonical(defn)), job=job, input_state_sha256=input_sha)))


def validate_result(result, defn, job, signature):
    assert result["job_signature"] == signature
    assert result["protocol_fingerprint"] == digest(canonical(defn))
    assert result["source_commit"] == defn["source_commit"]
    assert result["job"] == job and result["package_version"] == "1.2.2"
    assert result["scope"] == "CUDA_EVIDENCE" and result["backend"] == "cuda"
    assert result.get("worker_exit_code", 0) == 0 or result["status"] != "PASS"
    if result["status"] not in ("PASS", "REVIEW"):
        return
    assert result["gpu_executed"] is True
    if job["kind"] == "regression":
        assert result["status"] == "PASS" and result["exit_code"] == 0
        evidence = result["junit"]
        assert evidence["passed"] and evidence["test_count"] == defn["expected_tests"]
        assert evidence["test_ids_sha256"] == defn["expected_test_ids_sha256"]
        assert all(evidence[k] == 0 for k in ("failures", "errors", "skipped"))
        return
    for key, value in make_cfg(job).items():
        assert result["cfg"][key] == value, key
    assert result["actual_device"].startswith("cuda") and result["dtype"] == "torch.complex128"
    assert all(isinstance(result[k], (int, float)) and math.isfinite(result[k]) for k in
               ("energy", "mu", "residual", "mass", "virial_relative", "rrms", "R90", "peak_density", "dx"))
    if result["status"] == "PASS":
        assert result["checks"] and all(result["checks"].values())
        assert result["residual"] < job["residual_target"]
        assert abs(result["mass"] - 1.) < CRITERIA["mass_error"]
        assert result["edge_mass_two_cells"] < CRITERIA["edge_mass_two_cells"]
        assert result["outer_one_unit_mass"] < CRITERIA["outer_shell_mass"]


def inspect_receipt(root, defn, job, signature, complete_only=True):
    root = Path(root)
    folder = root / "jobs" / job["name"]
    receipt_path = folder / "receipt.json"
    if not receipt_path.exists():
        return None
    receipt = read(receipt_path)
    if receipt.get("job_signature") != signature:
        return None
    for name, value in receipt["artifacts"].items():
        assert sha(safe_path(root, name)) == value, "Cached artifact changed: " + name
    result = read(folder / "result.json")
    validate_result(result, defn, job, signature)
    if complete_only and result["status"] not in TERMINAL:
        return None
    if result.get("state_path"):
        assert sha(safe_path(root, result["state_path"])) == result["state_sha256"]
    if job["kind"] == "regression" and result["status"] == "PASS":
        xml = ET.parse(safe_path(root, result["junit_path"])).getroot()
        cases = xml.findall(".//testcase")
        ids = sorted([[c.get("classname", ""), c.get("name", "")] for c in cases])
        suites = [xml] if xml.tag == "testsuite" else xml.findall(".//testsuite")
        assert len(ids) == len({tuple(x) for x in ids}) == defn["expected_tests"]
        assert digest(canonical(ids)) == defn["expected_test_ids_sha256"]
        assert sum(int(s.get("tests", 0)) for s in suites) == defn["expected_tests"]
        assert all(int(s.get(k, 0)) == 0 for s in suites for k in ("failures", "errors", "skipped"))
        assert not any(xml.findall(".//" + k) for k in ("failure", "error", "skipped"))
        assert result["peak_allocated_bytes"] > 0
    return result


def find_checkpoint(root, defn, job, signature):
    folder = Path(root) / "jobs" / job["name"]
    for p in sorted(folder.glob("attempt_*/progress.json"), reverse=True):
        progress = read(p)
        if (progress.get("job_signature") != signature or progress.get("protocol_fingerprint") != digest(canonical(defn))
                or progress.get("scope") != "CUDA_EVIDENCE" or progress.get("backend") != "cuda"
                or progress.get("source_commit") != defn["source_commit"]):
            continue
        state = safe_path(p.parent, progress["state_file"])
        if state.is_file() and sha(state) == progress["state_sha256"]:
            return dict(path=str(state), sha256=progress["state_sha256"], total_iterations=progress["total_iterations"],
                        chunk=progress["chunk"], consecutive_stalls=progress.get("consecutive_stalls", 0),
                        original_initial_array_sha256=progress["original_initial_array_sha256"],
                        original_initial_energy=progress["original_initial_energy"])
    return None


def observed_order(energies, widths):
    a, b, c = energies
    x, y, z = widths
    if not all(math.isfinite(t) for t in energies + widths) or not (x > y > z > 0):
        return None
    if (a - b) * (b - c) <= 0:
        return None
    ratio = (a - b) / (b - c)
    def f(p):
        return ((x / z) ** p - (y / z) ** p) / ((y / z) ** p - 1.) - ratio
    lo, hi = .25, 6.
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(70):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def scaled_delta(a, b):
    return abs(a - b) / max(1., abs(b))


def positive_relative_delta(a, b):
    return abs(a - b) / max(1e-300, abs(b))


def h2_fit(energies, widths):
    xs = [h * h for h in widths]
    xm, ym = sum(xs) / len(xs), sum(energies) / len(energies)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, energies)) / sum((x - xm) ** 2 for x in xs)
    return dict(intercept=ym - slope * xm, slope=slope,
                max_fit_residual=max(abs(y - (ym + slope * (x - xm))) for x, y in zip(xs, energies)))


def density_delta(root, a, b):
    import numpy as np
    # Exact coordinate-aligned crop for the fixed-dx box study; no interpolant.
    if abs(a["dx"] - b["dx"]) > 1e-14:
        raise ValueError("Density comparison requires equal grid spacing")
    arrays = []
    n = min(a["job"]["N"], b["job"]["N"])
    for result in (a, b):
        size = result["job"]["N"]
        assert (size - n) % 2 == 0
        p = np.load(safe_path(root, result["state_path"]), mmap_mode="r", allow_pickle=False)
        assert p.shape == (size,) * 3 and p.dtype == np.complex128
        off = (size - n) // 2
        section = p[off:off + n, off:off + n, off:off + n]
        arrays.append(section.real ** 2 + section.imag ** 2)
    return float(np.sqrt(np.sum((arrays[0] - arrays[1]) ** 2) / max(1e-300, float(np.sum(arrays[1] ** 2)))))


def assess(root, defn, outcomes, density_fn=density_delta):
    root = Path(root)
    jobs = {j["name"]: j for j in defn["jobs"]}
    assert len(jobs) == len(defn["jobs"])
    assert len({r["job"]["name"] for r in outcomes}) == len(outcomes)
    by = {r["job"]["name"]: r for r in outcomes}
    assert set(by) <= set(jobs)
    metrics, criteria = {}, {}
    solve_names = [j["name"] for j in defn["jobs"] if j["kind"] == "solve"]
    available = all(name in by and by[name]["status"] in ("PASS", "REVIEW") for name in solve_names)
    if available:
        g = [by["grid_N" + str(n)] for n in (96, 128, 160, 192)]
        e, h = [r["energy"] for r in g], [r["dx"] for r in g]
        orders = [observed_order(e[:3], h[:3]), observed_order(e[1:], h[1:])]
        fit = h2_fit(e[1:], h[1:])
        finest = g[-1]
        metrics["grid"] = dict(energies=e, widths=h, observed_orders=orders,
            last_energy_change=scaled_delta(e[-2], e[-1]), last_mu_change=scaled_delta(g[-2]["mu"], finest["mu"]),
            last_rrms_change=positive_relative_delta(g[-2]["rrms"], finest["rrms"]),
            last_peak_change=positive_relative_delta(g[-2]["peak_density"], finest["peak_density"]),
            h2_fit=fit, h2_energy_correction=scaled_delta(e[-1], fit["intercept"]))
        grid = metrics["grid"]
        criteria.update(grid_energy=grid["last_energy_change"] < CRITERIA["last_grid_energy_change"],
            grid_mu=grid["last_mu_change"] < CRITERIA["last_grid_mu_change"],
            grid_rrms=grid["last_rrms_change"] < CRITERIA["last_grid_rrms_change"],
            grid_peak=grid["last_peak_change"] < CRITERIA["last_grid_peak_change"],
            h2_extrapolation=grid["h2_energy_correction"] < CRITERIA["h2_extrapolation_energy_correction"],
            observed_order=all(p is not None and CRITERIA["observed_order_low"] < p < CRITERIA["observed_order_high"] for p in orders),
            finest_virial=finest["virial_relative"] < CRITERIA["finest_virial_relative"],
            finest_resolution=finest["points_per_R90"] >= CRITERIA["points_per_R90"])
        padding = []
        for n in (96, 128):
            ref, padded = by["grid_N" + str(n)], by["padding_N" + str(n) + "_pad3"]
            pair = dict(N=n, energy_change=scaled_delta(padded["energy"], ref["energy"]),
                        density_l2=density_fn(root, padded, ref), identical_initial_array=padded["initial_array_sha256"] == ref["initial_array_sha256"])
            padding.append(pair)
        metrics["padding"] = padding
        criteria["padding"] = all(p["identical_initial_array"] and p["energy_change"] < CRITERIA["padding_energy_change"] and p["density_l2"] < CRITERIA["padding_density_l2"] for p in padding)
        boxes = [by["box_L6"], by["grid_N128"], by["box_L10"], by["box_L12"]]
        box_pairs = [dict(L_from=a["job"]["L"], L_to=b["job"]["L"],
                          energy_change=scaled_delta(a["energy"], b["energy"]), density_l2=density_fn(root, a, b)) for a, b in zip(boxes[:-1], boxes[1:])]
        metrics["box"] = dict(dx=.125, comparisons=box_pairs,
                              note="Density differences use the exact common centered grid region, without renormalization or interpolation.")
        criteria["box"] = all(p["energy_change"] < CRITERIA["box_energy_change"] and p["density_l2"] < CRITERIA["box_density_l2"] for p in box_pairs)
        a, b = by["seed_TF_N128"], by["grid_N128"]
        metrics["seed"] = dict(energy_change=scaled_delta(a["energy"], b["energy"]), density_l2=density_fn(root, a, b),
                               note="Two finite starts are a sensitivity check, not a global-minimum certificate.")
        criteria["seed"] = metrics["seed"]["energy_change"] < CRITERIA["seed_energy_change"] and metrics["seed"]["density_l2"] < CRITERIA["seed_density_l2"]
        tight = by["tight_N192"]
        tight_delta = scaled_delta(tight["energy"], finest["energy"])
        numerical_floor = max(tight_delta, 64 * sys.float_info.epsilon)
        increments = [scaled_delta(e[i], e[i + 1]) for i in range(3)]
        metrics["optimizer"] = dict(tighter_residual=tight["residual"], tighter_energy_change=tight_delta,
                                     smallest_grid_increment=min(increments), numerical_comparison_floor=numerical_floor)
        criteria["tighter_solve"] = tight["residual"] < 1e-7 and tight_delta < CRITERIA["tighter_solve_energy_change"]
        criteria["grid_above_optimizer_change"] = min(increments) > CRITERIA["grid_increment_over_optimization_change"] * numerical_floor
    all_solve_pass = available and all(by[n]["status"] == "PASS" for n in solve_names)
    refinement = "PASS" if all_solve_pass and criteria and all(criteria.values()) else ("REVIEW" if available else "INCOMPLETE")
    release = by.get("release_cuda_109", {}).get("status", "PENDING" if "release_cuda_109" in jobs else "NOT_REQUESTED")
    overall = "PASS" if refinement == "PASS" and release == "PASS" else "REVIEW_OR_INCOMPLETE"
    if any(r["status"] == "FAIL" for r in outcomes):
        overall = "FAIL"
    return dict(protocol=PROTOCOL_VERSION, overall=overall, refinement_status=refinement, release_cuda_status=release,
                planned_jobs=len(jobs), completed_jobs=sum(r["status"] in TERMINAL for r in outcomes),
                recorded_jobs=len(outcomes), pending_jobs=[n for n in jobs if n not in by or by[n]["status"] not in TERMINAL],
                gpu_executed=any(r.get("gpu_executed") is True for r in outcomes),
                status_counts={s: sum(r["status"] == s for r in outcomes) for s in sorted({r["status"] for r in outcomes})},
                failed_criteria=[k for k, v in criteria.items() if not v], criteria_pass=criteria,
                metrics=metrics, limitations=defn["limitations"])


def verify_source(source, spec):
    source = Path(source).resolve()
    expected = spec["source_hashes"]
    actual_names = {p.relative_to(source).as_posix() for folder in ("becgpp", "tests") for p in (source / folder).glob("*.py")}
    assert actual_names == set(expected), "Numerical/test file list differs from the pinned release"
    for name, value in expected.items():
        assert digest((source / name).read_bytes().replace(b"\r\n", b"\n")) == value, name
    return source


def restore_archive(archive, root, fingerprint=None):
    root = Path(root)
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == len(set(z.namelist())), "Duplicate archive paths"
        assert z.testzip() is None
        manifest = json.loads(z.read("SHA256.json"))
        assert {n for n in z.namelist() if not n.endswith("/")} == set(manifest) | {"SHA256.json"}
        if fingerprint is not None:
            assert json.loads(z.read("protocol.json"))["fingerprint"] == fingerprint, "Resume protocol differs"
        expanded = sum(i.file_size for i in z.infolist())
        assert expanded + 256 * 1024**2 < shutil.disk_usage(root.parent).free, "Not enough disk space to restore"
        for info in z.infolist():
            assert not stat.S_ISLNK(info.external_attr >> 16), "Symlinks are not accepted"
            target = safe_path(root, info.filename.rstrip("/"))
            if not info.is_dir():
                assert digest(z.read(info.filename)) == manifest[info.filename] if info.filename != "SHA256.json" else True
            if target.exists() and not info.is_dir():
                assert sha(target) == digest(z.read(info.filename)), "Resume would overwrite different local content: " + info.filename
            else:
                z.extract(info, root)


def prepare_source(root, spec, settings):
    if settings["source_dir"]:
        return verify_source(settings["source_dir"], spec)
    base = root / "source"
    source = base / ("becGPP-" + spec["source_commit"])
    if source.exists():
        return verify_source(source, spec)
    url = "https://codeload.github.com/Duyle0503/becGPP/zip/" + spec["source_commit"]
    archive_bytes = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                archive_bytes = response.read()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    base.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
        assert z.testzip() is None
        for info in z.infolist():
            safe_path(base, info.filename.rstrip("/"))
            assert not stat.S_ISLNK(info.external_attr >> 16)
            z.extract(info, base)
    atomic_json(root / "download.json", dict(url=url, source_zip_sha256=digest(archive_bytes)))
    return verify_source(source, spec)


def ensure_dependencies():
    # Never install or replace PyTorch/CUDA. Only absent verification packages.
    missing = [m for m in ("numpy", "matplotlib", "pytest", "scipy") if importlib.util.find_spec(m) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", *missing], check=True)


def stop_process(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        proc.terminate()
    else:
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def run_process(command, log, env, cwd, timeout, heartbeat, progress_path=None):
    proc = None
    reason = None
    started, last_heartbeat = time.monotonic(), 0.
    with Path(log).open("w", encoding="utf-8") as stream:
        try:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=os.name != "nt")
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > timeout:
                    reason = "TIMEOUT"
                    stop_process(proc)
                    break
                if elapsed - last_heartbeat >= heartbeat:
                    progress = read(progress_path) if progress_path and Path(progress_path).is_file() else {}
                    print("  heartbeat", int(elapsed), "s", "iters", progress.get("total_iterations", "starting"),
                          "res", progress.get("residual", "--"), flush=True)
                    last_heartbeat = elapsed
                time.sleep(.5)
        except BaseException:
            if proc is not None:
                stop_process(proc)
            raise
    return proc.returncode, reason


def write_receipt(root, job, signature, record):
    folder = Path(root) / "jobs" / job["name"]
    atomic_json(folder / "result.json", record)
    artifacts = {p.relative_to(root).as_posix(): sha(p) for p in sorted(folder.rglob("*"))
                 if p.is_file() and p.name != "receipt.json" and not p.name.endswith(".tmp")}
    atomic_json(folder / "receipt.json", dict(job_signature=signature, artifacts=artifacts))


def run_job(root, source, defn, job, settings, session, env, input_result=None, remaining=None):
    input_sha = input_result.get("state_sha256") if input_result else None
    signature = job_signature(defn, job, input_sha)
    cached = inspect_receipt(root, defn, job, signature)
    if cached and not (cached["status"] == "FAIL" and settings["retry_failed_jobs"]):
        print("REUSE", job["name"], cached["status"], flush=True)
        return cached
    folder = root / "jobs" / job["name"]
    folder.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name.split("_")[-1]) for p in folder.glob("attempt_*") if p.is_dir()]
    attempt = folder / ("attempt_" + str(max(numbers, default=0) + 1).zfill(3))
    attempt.mkdir()
    request = dict(job=job, source=str(source), source_commit=defn["source_commit"], expected_source_hashes=defn["source_hashes"],
                   job_signature=signature, protocol_fingerprint=digest(canonical(defn)), backend="cuda", scope="CUDA_EVIDENCE",
                   attempt_dir=str(attempt), session=session, expected_tests=defn["expected_tests"],
                   expected_test_ids_sha256=defn["expected_test_ids_sha256"], memory_fraction=settings["memory_fraction"],
                   criteria=CRITERIA, chunk_iterations=defn["chunk_iterations"])
    if job["kind"] == "solve":
        request["cfg"] = make_cfg(job)
        request["resume_checkpoint"] = find_checkpoint(root, defn, job, signature)
        if input_result:
            request.update(input_state_path=str(safe_path(root, input_result["state_path"])), input_state_sha256=input_sha)
    atomic_json(attempt / "request.json", request)
    print("RUN", job["name"], "resume iterations", (request.get("resume_checkpoint") or {}).get("total_iterations", 0), flush=True)
    record = dict(job=job, job_signature=signature, protocol_fingerprint=digest(canonical(defn)), scope="CUDA_EVIDENCE",
                  backend="cuda", gpu_executed=False, source_commit=defn["source_commit"], package_version="1.2.2", session=session)
    try:
        code, reason = run_process([sys.executable, str(root / "worker.py"), str(attempt / "request.json")],
                                  attempt / "stdout.log", env, source,
                                  min(settings["job_timeout_seconds"], max(1., remaining - 30.)) if remaining else settings["job_timeout_seconds"],
                                  settings["heartbeat_seconds"], attempt / "progress.json")
        result_file = attempt / "result.json"
        if result_file.exists():
            record = read(result_file)
        else:
            record.update(status="FAIL", error="Worker did not produce a result; inspect stdout.log")
        record["worker_exit_code"] = code
        if reason:
            record.update(status=reason, error="Parent timeout; committed checkpoints are retained")
        elif code != 0:
            record["status"] = "FAIL"
        if record.get("state_file"):
            state = safe_path(attempt, record["state_file"])
            assert sha(state) == record["state_sha256"]
            record["state_path"] = state.relative_to(root).as_posix()
        if job["kind"] == "regression" and (attempt / "pytest.xml").exists():
            record["junit_path"] = (attempt / "pytest.xml").relative_to(root).as_posix()
        if record["status"] in ("PASS", "REVIEW"):
            validate_result(record, defn, job, signature)
    except KeyboardInterrupt:
        record.update(status="INTERRUPTED", error="User interrupted; committed checkpoints are retained")
        atomic_json(attempt / "parent_result.json", record)
        write_receipt(root, job, signature, record)
        raise
    except Exception as error:
        traceback.print_exc()
        record.update(status="FAIL", error=repr(error))
    atomic_json(attempt / "parent_result.json", record)
    write_receipt(root, job, signature, record)
    return record


def collect_results(root, defn):
    results, by = [], {}
    for job in defn["jobs"]:
        dependency = job.get("initial_from")
        input_sha = by.get(dependency, {}).get("state_sha256") if dependency else None
        signature = job_signature(defn, job, input_sha)
        result = inspect_receipt(root, defn, job, signature, complete_only=False)
        if result:
            results.append(result)
            by[job["name"]] = result
    return results


def report_outputs(root, defn, fatal=None):
    outcomes = collect_results(root, defn)
    summary = assess(root, defn, outcomes)
    if fatal:
        summary.update(overall="REVIEW_OR_INCOMPLETE", orchestration_error=fatal)
    atomic_json(root / "outcomes.json", outcomes)
    atomic_json(root / "SUMMARY.json", summary)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        columns = ["name", "status", "N", "L", "pad", "seed", "energy", "mu", "residual", "virial_relative", "rrms", "R90", "dx", "points_per_R90", "total_iterations", "attempt_seconds"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for result in outcomes:
            row = {key: result.get(key) for key in columns}
            row.update({key: result["job"].get(key) for key in ("N", "L", "pad", "seed")})
            row.update(name=result["job"]["name"], status=result["status"])
            writer.writerow(row)
    with (root / "refinement_table.tex").open("w", encoding="utf-8") as f:
        f.write("% Generated finite-grid results. Preserve REVIEW/FAIL/INCOMPLETE statuses.\n")
        f.write("\\begin{tabular}{lrrrrrrl}\nCase & N & L & E & mu & residual & virial & status \\\\\n")
        for result in outcomes:
            if result["job"]["kind"] != "solve":
                continue
            values = [result["job"]["name"].replace("_", r"\_"), str(result["job"]["N"]), str(result["job"]["L"])]
            values += [format(result[key], ".9g") if isinstance(result.get(key), (int, float)) else "--" for key in ("energy", "mu", "residual", "virial_relative")]
            values.append(result["status"].replace("_", r"\_"))
            f.write(" & ".join(values) + " \\\\\n")
        f.write("\\end{tabular}\n")
    if summary["metrics"].get("grid"):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            grid = summary["metrics"]["grid"]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            x = [h * h for h in grid["widths"]]
            axes[0].plot(x, grid["energies"], "o-", label="computed")
            fit = grid["h2_fit"]
            xx = [0., max(x)]
            axes[0].plot(xx, [fit["intercept"] + fit["slope"] * t for t in xx], "--", label="last-three h^2 fit")
            axes[0].set(xlabel="dx squared", ylabel="Energy (code units)")
            axes[0].legend()
            by = {r["job"]["name"]: r for r in outcomes}
            axes[1].loglog(grid["widths"], [by["grid_N" + str(n)]["virial_relative"] for n in (96, 128, 160, 192)], "o-")
            axes[1].axhline(CRITERIA["finest_virial_relative"], color="gray", linestyle="--")
            axes[1].set(xlabel="dx", ylabel="Scaled virial residual")
            fig.savefig(root / "convergence.png", dpi=180)
            plt.close(fig)
        except Exception as error:
            atomic_json(root / "plot_warning.json", dict(error=repr(error)))
    return summary


def archive_outputs(root):
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and "source" not in p.relative_to(root).parts
             and p.name not in ("SHA256.json", "run.lock") and not p.name.endswith(".tmp")]
    manifest = {p.relative_to(root).as_posix(): sha(p) for p in files}
    atomic_json(root / "SHA256.json", manifest)
    target = root.with_name(root.name + ".zip")
    temp = target.with_name(target.name + ".partial")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for p in files + [root / "SHA256.json"]:
            z.write(p, p.relative_to(root).as_posix())
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
    temp.replace(target)
    checksum = sha(target)
    target.with_name(target.name + ".sha256").write_text(checksum + "  " + target.name + "\n", encoding="utf-8")
    print("DOWNLOAD:", target, "SHA256:", checksum, flush=True)
    return target


def process_identity(pid):
    try:
        # A new Kaggle VM/kernel can reuse the previous kernel's PID.
        start = Path("/proc/" + str(pid) + "/stat").read_text().rsplit(")", 1)[1].split()[19]
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + start
    except OSError:
        return None


def acquire_lock(root):
    lock = root / "run.lock"
    if lock.exists():
        old = read(lock)
        try:
            os.kill(int(old["pid"]), 0)
        except OSError:
            lock.unlink()
        else:
            identity = process_identity(int(old["pid"]))
            if old.get("process_identity") != identity and identity is not None:
                lock.unlink()
            else:
                raise RuntimeError("Another runner is active (PID " + str(old["pid"]) + "). Stop it before resuming.")
    with lock.open("x", encoding="utf-8") as f:
        json.dump(dict(pid=os.getpid(), process_identity=process_identity(os.getpid()), created_utc=datetime.now(timezone.utc).isoformat()), f)
    return lock


def main(settings, spec, worker_source, core_source):
    unknown = set(settings) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError("Unknown setting: " + str(sorted(unknown)))
    settings = dict(DEFAULT_SETTINGS, **settings)
    assert isinstance(settings["run_release_tests"], bool)
    assert 0 < settings["session_budget_hours"] <= 48 and settings["job_timeout_seconds"] > 0
    assert 0 < settings["memory_fraction"] < 1 and settings["heartbeat_seconds"] > 0
    defn = definition(spec, worker_source, core_source, settings["run_release_tests"])
    fingerprint = digest(canonical(defn))
    base = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
    root = Path(settings["run_root"]).expanduser().resolve() if settings["run_root"] else base / ("becgpp_trapped3d_v2_" + fingerprint[:12])
    root.parent.mkdir(parents=True, exist_ok=True)
    if settings["resume_zip"]:
        restore_archive(settings["resume_zip"], root, fingerprint)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        assert read(root / "protocol.json")["fingerprint"] == fingerprint, "Different protocol: choose a new run_root"
    lock = acquire_lock(root)
    started = time.monotonic()
    fatal = None
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    print("RESULT DIRECTORY:", root, "\nPROTOCOL:", fingerprint, "\nPLANNED:", len(defn["jobs"]), "jobs", flush=True)
    try:
        atomic_json(root / "protocol.json", dict(fingerprint=fingerprint, definition=defn))
        (root / "worker.py").write_bytes(worker_source.encode("utf-8"))
        (root / "core.py").write_bytes(core_source.encode("utf-8"))
        atomic_json(root / "source_manifest.json", dict(commit=spec["source_commit"], normalized_module_sha256=spec["source_hashes"]))
        atomic_json(root / "settings.json", settings)
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Enable a Kaggle GPU; CPU substitution is forbidden.")
        index = int(settings["gpu_index"])
        assert 0 <= index < torch.cuda.device_count()
        ensure_dependencies()
        torch.cuda.set_device(index)
        probe = torch.ones((8, 8, 8), dtype=torch.complex128, device="cuda:" + str(index))
        assert (torch.fft.ifftn(torch.fft.fftn(probe)) - probe).abs().max().item() < 1e-13
        del probe
        torch.cuda.empty_cache()
        environment = dict(session=session, created_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                           torch=torch.__version__, cuda_runtime=torch.version.cuda, selected_gpu_index=index,
                           gpu=torch.cuda.get_device_name(index), gpu_count=torch.cuda.device_count(), gpu_executed=True)
        atomic_json(root / "sessions" / session / "environment.json", environment)
        try:
            info = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
            path = root / "sessions" / session / "nvidia_smi.txt"
            path.write_text(info.stdout + "\n" + info.stderr, encoding="utf-8")
        except (OSError, subprocess.TimeoutExpired):
            pass
        source = prepare_source(root, spec, settings)
        # CUDA indices are logical in the parent. Preserve an existing visibility
        # mapping when selecting the one physical device visible to each worker.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        selection = visible.split(",")[index].strip() if visible and visible.strip() not in ("-1", "") else str(index)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=selection, MPLBACKEND="Agg", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
        by = {}
        for job in defn["jobs"]:
            dep = job.get("initial_from")
            dependency = by.get(dep) if dep else None
            if dep and (not dependency or not dependency.get("state_path")):
                print("PENDING", job["name"], "requires a saved state from", dep, flush=True)
                continue
            remaining = settings["session_budget_hours"] * 3600 - (time.monotonic() - started)
            signature = job_signature(defn, job, dependency.get("state_sha256") if dependency else None)
            cached = inspect_receipt(root, defn, job, signature)
            if cached and not (cached["status"] == "FAIL" and settings["retry_failed_jobs"]):
                print("REUSE", job["name"], cached["status"], flush=True)
                by[job["name"]] = cached
                continue
            if remaining < 60:
                print("SESSION BUDGET reached. Completed jobs/checkpoints are retained.", flush=True)
                break
            result = run_job(root, source, defn, job, settings, session, env, dependency, remaining)
            by[job["name"]] = result
            summary = report_outputs(root, defn)
            print("DONE", job["name"], result["status"], "|", summary["completed_jobs"], "/", len(defn["jobs"]), flush=True)
            archive_outputs(root)
    except KeyboardInterrupt:
        fatal = "Interrupted by user. Resume this same cell to continue from committed checkpoints."
        print(fatal, flush=True)
    except Exception as error:
        fatal = repr(error)
        atomic_json(root / "sessions" / session / "orchestration_error.json", dict(error=fatal, traceback=traceback.format_exc()))
        traceback.print_exc()
    finally:
        try:
            try:
                summary = report_outputs(root, defn, fatal)
            except Exception as error:
                summary = dict(protocol=PROTOCOL_VERSION, overall="FAIL_INTEGRITY", refinement_status="UNVERIFIED",
                    release_cuda_status="UNVERIFIED", failed_criteria=["integrity_or_assessment_error"],
                    pending_jobs=[j["name"] for j in defn["jobs"]], error=repr(error), traceback=traceback.format_exc())
                atomic_json(root / "SUMMARY.json", summary)
            archive = archive_outputs(root)
            print("FINAL:", summary["overall"], "| refinement:", summary["refinement_status"], "| release CUDA:", summary["release_cuda_status"], flush=True)
            print("Failed criteria:", summary["failed_criteria"], "\nPending:", summary["pending_jobs"], flush=True)
            print("Save the Kaggle output and download the ZIP. A hard VM shutdown cannot be recovered without retained output files.", flush=True)
            try:
                from IPython.display import FileLink, display
                display(FileLink(str(archive)))
                display(FileLink(str(root / "summary.csv")))
            except ImportError:
                pass
        finally:
            if lock.exists() and read(lock).get("pid") == os.getpid():
                lock.unlink()
    return summary
