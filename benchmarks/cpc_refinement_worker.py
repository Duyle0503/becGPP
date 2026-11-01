"""Child process for the one-cell refinement runner; generated into the notebook.

Production requests always use CUDA. The explicit HARNESS_SELFTEST scope exists
only for local protocol checks and can never produce a GPU PASS receipt.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import xml.etree.ElementTree as ET


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(clean(value), f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def junit_evidence(path, count, expected_ids_sha):
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    ids = sorted([[c.get("classname", ""), c.get("name", "")] for c in cases])
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    good = (len(ids) == count and len({tuple(x) for x in ids}) == count
            and hashlib.sha256(canonical(ids)).hexdigest() == expected_ids_sha
            and sum(int(s.get("tests", 0)) for s in suites) == count
            and all(int(s.get(k, 0)) == 0 for s in suites for k in ("failures", "errors", "skipped"))
            and not any(root.findall(".//" + k) for k in ("failure", "error", "skipped")))
    return dict(passed=bool(good), test_count=len(ids), test_ids_sha256=hashlib.sha256(canonical(ids)).hexdigest(),
                failures=len(root.findall(".//failure")), errors=len(root.findall(".//error")),
                skipped=len(root.findall(".//skipped")))


def main(request_path):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    out = Path(request["attempt_dir"])
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    scope = request.get("scope", "CUDA_EVIDENCE")
    backend = request.get("backend", "cuda")
    assert (scope, backend) in (("CUDA_EVIDENCE", "cuda"), ("HARNESS_SELFTEST", "cpu"))
    import numpy as np
    import torch
    torch.set_num_threads(2)
    if backend == "cuda":
        assert torch.cuda.is_available(), "CUDA required; CPU fallback is forbidden"
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    source = Path(request["source"]).resolve()
    for name, value in request["expected_source_hashes"].items():
        assert hashlib.sha256((source / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == value, name
    sys.path.insert(0, str(source))
    import becgpp
    assert Path(becgpp.__file__).resolve().is_relative_to(source)
    assert becgpp.__version__ == "1.2.2" and becgpp.DEV == backend
    header = dict(job=request["job"], job_signature=request["job_signature"], protocol_fingerprint=request["protocol_fingerprint"],
                  scope=scope, backend=backend, gpu_executed=(backend == "cuda"), package_version=becgpp.__version__,
                  source_commit=request["source_commit"], session=request["session"],
                  input_state_sha256=request.get("input_state_sha256"),
                  environment=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__, platform=platform.platform(),
                                   cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if backend == "cuda" else None,
                                   torch_threads=torch.get_num_threads(), deterministic_algorithms_warn_only=True))
    job = request["job"]

    def finish(record):
        record.update(header)
        record["attempt_seconds"] = time.monotonic() - start
        if backend == "cuda":
            torch.cuda.synchronize()
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        atomic_json(out / "result.json", record)
        print("RESULT", job["name"], record["status"], record.get("energy"), record.get("residual"), flush=True)

    try:
        if job["kind"] == "regression":
            import pytest
            xml = out / "pytest.xml"
            code = int(pytest.main([str(source / "tests"), "-q", "--junitxml=" + str(xml)]))
            checked = junit_evidence(xml, request["expected_tests"], request["expected_test_ids_sha256"])
            status = "PASS" if code == 0 and checked["passed"] else "FAIL"
            if scope != "CUDA_EVIDENCE":
                status = "SELFTEST_COMPLETE"
            finish(dict(status=status, exit_code=code, junit=checked))
            return

        from becgpp import default_cfg, make_grid, make_seed, ground_state
        from becgpp.grid import dV, _memory_estimate_gb
        from becgpp.operators import energy_components, apply_H
        cfg = default_cfg(**request["cfg"])
        atomic_json(out / "config.json", cfg)
        estimate = _memory_estimate_gb(3, cfg["Ngrid"], cfg["pad"], "newton")
        if backend == "cuda":
            free, total = torch.cuda.mem_get_info()
            if estimate * 1e9 > request["memory_fraction"] * free:
                finish(dict(status="BLOCKED_MEMORY", estimated_peak_gb=estimate, free_bytes=free, total_bytes=total))
                return
            torch.cuda.reset_peak_memory_stats()
        G = make_grid(cfg)
        dv = dV(G)
        resume = request.get("resume_checkpoint")
        if resume:
            state_path = Path(resume["path"])
            assert sha(state_path) == resume["sha256"]
            p = torch.as_tensor(np.load(state_path, allow_pickle=False), device=backend, dtype=torch.complex128)
            total_iters = int(resume["total_iterations"])
            chunk_id = int(resume["chunk"])
            stalls = int(resume.get("consecutive_stalls", 0))
        elif request.get("input_state_path"):
            state_path = Path(request["input_state_path"])
            assert sha(state_path) == request["input_state_sha256"]
            p = torch.as_tensor(np.load(state_path, allow_pickle=False), device=backend, dtype=torch.complex128)
            total_iters = chunk_id = stalls = 0
        else:
            p = make_seed(G, cfg)
            total_iters = chunk_id = stalls = 0
        assert tuple(p.shape) == (cfg["Ngrid"],) * 3 and torch.isfinite(p).all().item()
        p /= torch.sqrt((p.abs() ** 2).sum() * dv)
        attempt_initial_hash = hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
        initial_hash = resume["original_initial_array_sha256"] if resume else attempt_initial_hash
        trace_path = out / "trace.csv"
        with trace_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["chunk", "attempt_seconds", "total_attempted_iteration", "local_iteration", "energy", "relative_residual"])
        last = None
        initial_energy = float(energy_components(p, G, cfg["beta2"], cfg["beta3"], cfg["Omega"], cfg["G_C"])["E"].item())
        original_initial_energy = resume["original_initial_energy"] if resume else initial_energy
        selected_previous = initial_energy
        while total_iters < job["max_iterations"]:
            chunk_id += 1
            count = min(request["chunk_iterations"], job["max_iterations"] - total_iters)
            c = dict(cfg, maxit=count, record_trace=True, energy_tol=0.)
            p, _, obs = ground_state(c, G=G, psi0=p, verbose=False)
            trace = obs.pop("trace")
            assert all(math.isfinite(row[2]) and math.isfinite(row[3]) for row in trace)
            assert all(trace[i][2] <= trace[i - 1][2] + 2e-13 * max(1., abs(trace[i - 1][2])) for i in range(1, len(trace)))
            assert obs["E"] <= selected_previous + 2e-13 * max(1., abs(selected_previous))
            selected_previous = obs["E"]
            with trace_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for seconds, it, energy, resid in trace:
                    w.writerow([chunk_id, time.monotonic() - start - obs["walltime"] + seconds, total_iters + it, it, energy, resid])
                f.flush()
            total_iters += int(obs["iters"])
            stalls = stalls + 1 if obs["stop_reason"] == "line_search_stalled" else 0
            # Two alternating slots preserve the previously committed checkpoint
            # if a session dies between the state write and progress.json update.
            slot = out / ("state_" + str(chunk_id % 2) + ".npy")
            temp = slot.with_name(slot.name + ".tmp")
            with temp.open("wb") as f:
                np.save(f, p.detach().cpu().numpy(), allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
            temp.replace(slot)
            state_hash = sha(slot)
            progress = dict(job_signature=request["job_signature"], protocol_fingerprint=request["protocol_fingerprint"],
                            scope=scope, backend=backend, source_commit=request["source_commit"],
                            state_file=slot.name, state_sha256=state_hash, total_iterations=total_iters, chunk=chunk_id,
                            consecutive_stalls=stalls, energy=obs["E"], residual=obs["resid_rel"], stop_reason=obs["stop_reason"],
                            original_initial_array_sha256=initial_hash, original_initial_energy=original_initial_energy)
            atomic_json(out / "progress.json", progress)
            last = obs
            print("CHUNK", chunk_id, "iters", total_iters, "E", obs["E"], "res", obs["resid_rel"], "stop", obs["stop_reason"], flush=True)
            if obs["resid_rel"] < cfg["res_tol"] or stalls >= 2 or obs["iters"] == 0:
                break
        # A prior attempt may already have consumed the complete budget.
        comp = energy_components(p, G, cfg["beta2"], cfg["beta3"], cfg["Omega"], cfg["G_C"])
        Hp = apply_H(p, G, cfg, phi=comp["phi"])
        rho = p.abs() ** 2
        mu_h = float(((p.conj() * Hp).sum().real * dv).item())
        energy = float(comp["E"].item())
        mu_e = energy + float(comp["Econtact"].item()) + 2 * float(comp["Ethree"].item()) + float(comp["Egrav"].item())
        resid = float(torch.sqrt(((Hp - mu_h * p).abs() ** 2).sum() * dv).item()) / max(1., abs(mu_h))
        mass = float((rho.sum() * dv).item())
        coords = G["coords"]
        boundary = torch.zeros_like(rho, dtype=torch.bool)
        shell = torch.zeros_like(rho, dtype=torch.bool)
        for axis in range(3):
            ids = [slice(None)] * 3
            ids[axis] = slice(0, 2)
            boundary[tuple(ids)] = True
            ids[axis] = slice(-2, None)
            boundary[tuple(ids)] = True
            shell |= coords[axis].abs() >= cfg["L"] - 1.
        edge_mass = float(rho[boundary].sum().item() * dv)
        shell_mass = float(rho[shell].sum().item() * dv)
        peak = float(rho.max().item())
        face_relative_peak = float(rho[boundary].max().item()) / peak
        moments = [float((coord.square() * rho).sum().item() * dv) for coord in coords]
        center = [float((coord * rho).sum().item() * dv) / mass for coord in coords]
        rrms = math.sqrt(sum(moments) / mass)
        # R90 uses a shell histogram whose bin width is dx/8. Its quantization
        # is reported, rather than presenting it as a smooth fitted radius.
        dr = G["dx"] / 8
        radii = torch.sqrt(G["R2"])
        bins = torch.floor(radii / dr).to(torch.int64).reshape(-1)
        hist = torch.bincount(bins, weights=rho.reshape(-1) * dv)
        cumulative = hist.cumsum(0)
        qindex = int(torch.searchsorted(cumulative, .9 * cumulative[-1]).item())
        r90 = (qindex + .5) * dr
        ec = {key: float(comp[key].item()) for key in ("Ekin", "Etrap", "Econtact", "Ethree", "Erot", "Egrav")}
        virial = 2 * ec["Ekin"] - 2 * ec["Etrap"] + 3 * ec["Econtact"] + 6 * ec["Ethree"] + ec["Egrav"]
        virial_relative = abs(virial) / max(1., sum(abs(ec[k]) for k in ("Ekin", "Etrap", "Econtact", "Ethree", "Egrav")) + abs(ec["Egrav"]))
        checks = dict(actual_device=p.device.type == backend, precision=p.dtype == torch.complex128,
                      finite=bool(torch.isfinite(p).all().item()) and all(math.isfinite(x) for x in [energy, mu_h, resid, rrms, peak]),
                      residual=resid < cfg["res_tol"], mass=abs(mass - 1.) < request["criteria"]["mass_error"],
                      boundary=edge_mass < request["criteria"]["edge_mass_two_cells"],
                      outer_shell=shell_mass < request["criteria"]["outer_shell_mass"],
                      mu_identity=abs(mu_h - mu_e) / max(1., abs(mu_h)) < 2e-12,
                      energy_recomputed=last is None or abs(energy - last["E"]) / max(1., abs(energy)) < 2e-12)
        mid = cfg["Ngrid"] // 2
        with (out / "center_lines.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["x", "rho_x", "rho_y", "rho_z", "phi_x"])
            arrays = [coords[0][:, mid, mid], rho[:, mid, mid], rho[mid, :, mid], rho[mid, mid, :], comp["phi"][:, mid, mid]]
            for row in zip(*(a.detach().cpu().tolist() for a in arrays)):
                w.writerow(row)
        progress = json.loads((out / "progress.json").read_text(encoding="utf-8")) if (out / "progress.json").exists() else resume
        if not (out / "progress.json").exists() and resume:
            # Preserve a fully exhausted resume state inside the new attempt.
            slot = out / "state_0.npy"
            import shutil
            shutil.copyfile(resume["path"], slot)
            progress = dict(state_file=slot.name, state_sha256=sha(slot), total_iterations=total_iters, chunk=chunk_id,
                            original_initial_array_sha256=initial_hash, original_initial_energy=original_initial_energy)
            atomic_json(out / "progress.json", dict(progress, job_signature=request["job_signature"],
                protocol_fingerprint=request["protocol_fingerprint"], scope=scope, backend=backend, source_commit=request["source_commit"]))
        status = "PASS" if all(checks.values()) else "REVIEW"
        if scope != "CUDA_EVIDENCE":
            status = "SELFTEST_COMPLETE"
        finish(dict(status=status, checks=checks, cfg=cfg, energy=energy, mu=mu_h, residual=resid,
                    actual_device=str(p.device), dtype=str(p.dtype),
                    energy_components=ec, virial=virial, virial_relative=virial_relative,
                    mass=mass, edge_mass_two_cells=edge_mass, outer_one_unit_mass=shell_mass,
                    boundary_relative_peak=face_relative_peak, rrms=rrms, R90=r90, R90_bin_width=dr,
                    points_per_R90=r90 / G["dx"], peak_density=peak, center_of_mass=center,
                    moment_anisotropy=(max(moments) - min(moments)) / max(1e-300, sum(moments) / 3),
                    dx=G["dx"], total_iterations=total_iters, last_stop_reason=last["stop_reason"] if last else "budget_exhausted_on_resume",
                    initial_array_sha256=initial_hash, initial_energy=original_initial_energy,
                    attempt_initial_array_sha256=attempt_initial_hash, attempt_initial_energy=initial_energy,
                    state_file=progress["state_file"], state_sha256=progress["state_sha256"],
                    checkpoint_restart_note="Each chunk calls the unchanged ground_state; CG/BB history restarts between chunks. These are not timing comparisons with uninterrupted solves."))
    except torch.cuda.OutOfMemoryError as error:
        finish(dict(status="BLOCKED_MEMORY", error=str(error)))
    except Exception as error:
        traceback.print_exc()
        finish(dict(status="FAIL", error=repr(error)))
        raise


if __name__ == "__main__":
    main(sys.argv[1])
