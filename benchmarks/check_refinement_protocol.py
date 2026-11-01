"""Run the protocol's structural checks and an explicit small CPU worker check.

These are harness tests, never CUDA/refinement evidence for the paper.
python benchmarks/check_refinement_protocol.py --output REPORT.json
"""
import argparse
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

import cpc_refinement_core as core
from cpc_refinement_worker import junit_evidence

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
SPEC = core.read(BASE / "cpc_refinement_spec.json")
WORKER = (BASE / "cpc_refinement_worker.py").read_text(encoding="utf-8")
CORE = (BASE / "cpc_refinement_core.py").read_text(encoding="utf-8")
DEFINITION = core.definition(SPEC, WORKER, CORE)
WORKSPACE = REPO.parents[1]
REVIEW = WORKSPACE / "data/review/refinement_runner_v2_2026_10_31"


class ProtocolChecks(unittest.TestCase):
    def test_summary_comparison_accepts_last_bit_reduction_difference(self):
        from review_trapped3d_results import metrics_compatible
        self.assertTrue(metrics_compatible({"density_l2": 2.625097576646383e-7}, {"density_l2": 2.625097576646382e-7}))

    def test_summary_comparison_rejects_meaningful_change_or_verdict_change(self):
        from review_trapped3d_results import metrics_compatible
        self.assertFalse(metrics_compatible({"energy": -3.44}, {"energy": -3.45}))
        self.assertFalse(metrics_compatible({"pass": True}, {"pass": 1}))
        self.assertFalse(metrics_compatible({"pass": True}, {"pass": False}))

    def setUp(self):
        REVIEW.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="protocol_test_", dir=REVIEW)
        self.root = Path(self.tmp.name).resolve()
        assert self.root.is_relative_to(REVIEW.resolve())

    def tearDown(self):
        for name in (self.root.name + ".zip", self.root.name + ".zip.sha256", self.root.name + "_OFFLINE_AUDIT.json"):
            p = self.root.parent / name
            assert p.resolve().parent == REVIEW.resolve() and name.startswith(self.root.name)
            if p.exists():p.unlink()
        self.tmp.cleanup()

    def header(self, job, status="PASS", input_sha=None):
        return dict(job=copy.deepcopy(job), status=status, job_signature=core.job_signature(DEFINITION, job, input_sha),
                    protocol_fingerprint=core.digest(core.canonical(DEFINITION)), source_commit=SPEC["source_commit"],
                    scope="CUDA_EVIDENCE", backend="cuda", gpu_executed=True, package_version="1.2.2", worker_exit_code=0)

    def examples(self):
        rows = []
        for job in DEFINITION["jobs"]:
            r = self.header(job)
            if job["kind"] == "regression":
                r.update(exit_code=0, junit=dict(passed=True, test_count=109,
                    test_ids_sha256=SPEC["expected_test_ids_sha256"], failures=0, errors=0, skipped=0))
            else:
                dx = 2 * job["L"] / job["N"]
                r.update(cfg=core.make_cfg(job), energy=-3.45 + .10 * dx ** 2,
                         mu=-7. + .1 * dx ** 2, residual=1e-8, mass=1., checks=dict(finite=True, residual=True),
                         actual_device="cuda:0", dtype="torch.complex128", virial_relative=1e-4,
                         rrms=1.25 + .001 * dx ** 2, R90=1.8, points_per_R90=1.8 / dx,
                         peak_density=.15 + .001 * dx ** 2, dx=dx, edge_mass_two_cells=0., outer_one_unit_mass=0.,
                         initial_array_sha256="same_analytic_array_at_given_N", state_path="unused_in_synthetic_comparison.npy")
            rows.append(r)
        return rows

    def assess_examples(self, rows):
        return core.assess(self.root, DEFINITION, rows, density_fn=lambda *_: 0.)

    def test_plan_is_complete_and_fixed_physics(self):
        self.assertEqual(len(DEFINITION["jobs"]), 12)
        self.assertEqual(len(core.plan_jobs(False)), 11)
        boxes = [j for j in DEFINITION["jobs"] if j["name"].startswith("box_") or j["name"] == "grid_N128"]
        self.assertEqual(len(boxes), 4)
        self.assertTrue(all(2 * j["L"] / j["N"] == .125 for j in boxes))
        for j in DEFINITION["jobs"]:
            if j["kind"] == "solve":
                c = core.make_cfg(j)
                self.assertEqual((c["dimension"], c["s"], c["trap_coeff"], c["beta2"], c["G_C"], c["Omega"]), ("3D", 2, .5, 100., 20., 0.))
                self.assertEqual(c["energy_tol"], 0.)

    def test_nonuniform_grid_order_and_extrapolation(self):
        h = [16 / n for n in (96, 128, 160, 192)]
        e = [-3.45 + .2 * x ** 2 for x in h]
        self.assertAlmostEqual(core.observed_order(e[:3], h[:3]), 2., places=8)
        self.assertAlmostEqual(core.observed_order(e[1:], h[1:]), 2., places=8)
        self.assertAlmostEqual(core.h2_fit(e[1:], h[1:])["intercept"], -3.45, places=12)

    def test_nonmonotonic_energy_does_not_invent_an_order(self):
        self.assertIsNone(core.observed_order([1., 2., 1.5], [.3, .2, .1]))

    def test_synthetic_converged_case_passes(self):
        self.assertEqual(self.assess_examples(self.examples())["overall"], "PASS")

    def test_report_exports_csv_latex_and_plot(self):
        rows = self.examples()
        original = core.assess
        with patch.object(core, "collect_results", return_value=rows), patch.object(core, "assess", side_effect=lambda root, defn, outcomes: original(root, defn, outcomes, density_fn=lambda *_: 0.)):
            result = core.report_outputs(self.root, DEFINITION)
        self.assertEqual(result["overall"], "PASS")
        self.assertTrue((self.root / "summary.csv").is_file())
        self.assertTrue((self.root / "convergence.png").is_file(), (self.root / "plot_warning.json").read_text() if (self.root / "plot_warning.json").exists() else "Plot missing")
        tex = (self.root / "refinement_table.tex").read_text(encoding="utf-8")
        self.assertIn("grid\\_N192", tex)
        self.assertNotIn("None", tex)

    def test_empty_run_is_incomplete(self):
        result = self.assess_examples([])
        self.assertEqual(result["refinement_status"], "INCOMPLETE")
        self.assertEqual(result["completed_jobs"], 0)
        self.assertNotEqual(result["overall"], "PASS")

    def test_missing_tight_solve_never_passes(self):
        rows = [r for r in self.examples() if r["job"]["name"] != "tight_N192"]
        self.assertNotEqual(self.assess_examples(rows)["overall"], "PASS")

    def test_failed_regression_blocks_overall_pass(self):
        rows = self.examples();rows[0]["status"] = "FAIL"
        result = self.assess_examples(rows)
        self.assertEqual(result["overall"], "FAIL")
        self.assertEqual(result["refinement_status"], "PASS")

    def test_residual_failure_is_not_hidden_by_energy_agreement(self):
        rows = self.examples();rows[-1].update(status="REVIEW", residual=2e-7)
        result = self.assess_examples(rows)
        self.assertIn("tighter_solve", result["failed_criteria"])
        self.assertNotEqual(result["overall"], "PASS")

    def test_virial_failure_is_not_hidden_by_small_energy_changes(self):
        rows = self.examples()
        next(r for r in rows if r["job"]["name"] == "grid_N192")["virial_relative"] = .01
        self.assertIn("finest_virial", self.assess_examples(rows)["failed_criteria"])

    def test_padding_requires_identical_initial_state(self):
        rows = self.examples()
        next(r for r in rows if r["job"]["name"] == "padding_N128_pad3")["initial_array_sha256"] = "different"
        self.assertIn("padding", self.assess_examples(rows)["failed_criteria"])

    def test_duplicate_case_is_rejected(self):
        rows = self.examples();rows.append(rows[-1])
        with self.assertRaises(AssertionError):
            self.assess_examples(rows)

    def test_cpu_result_cannot_be_reused_as_gpu_pass(self):
        job = DEFINITION["jobs"][1];r = self.examples()[1]
        r.update(scope="HARNESS_SELFTEST", backend="cpu", gpu_executed=False)
        with self.assertRaises(AssertionError):
            core.validate_result(r, DEFINITION, job, core.job_signature(DEFINITION, job))

    def test_wrong_precision_is_rejected(self):
        r = self.examples()[1];r["dtype"] = "torch.complex64"
        with self.assertRaises(AssertionError):
            core.validate_result(r, DEFINITION, r["job"], r["job_signature"])

    def test_job_fingerprint_includes_tolerance_and_initial_state(self):
        job = DEFINITION["jobs"][-1]
        a = core.job_signature(DEFINITION, job, "a")
        self.assertNotEqual(a, core.job_signature(DEFINITION, job, "b"))
        self.assertNotEqual(a, core.job_signature(DEFINITION, dict(job, residual_target=1e-4), "a"))

    def test_source_bytes_match_immutable_release(self):
        self.assertEqual(core.verify_source(REPO, SPEC), REPO.resolve())
        changed = copy.deepcopy(SPEC)
        changed["source_hashes"]["becgpp/solvers.py"] = "0" * 64
        with self.assertRaises(AssertionError):
            core.verify_source(REPO, changed)

    def test_path_traversal_rejected(self):
        for name in ("../escape", "a/../../escape", "/absolute", "C:/absolute", "a\\..\\bad"):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                core.safe_path(self.root, name)

    def test_fixed_dx_density_crop_has_no_interpolation(self):
        import numpy as np
        rows = []
        for n in (4, 8):
            x = np.arange(n) - n // 2
            p = np.exp(-.1 * (x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)).astype(np.complex128)
            name = "state" + str(n) + ".npy";np.save(self.root / name, p)
            rows.append(dict(dx=1., job=dict(N=n), state_path=name))
        self.assertEqual(core.density_delta(self.root, *rows), 0.)

    def test_receipt_detects_artifact_change(self):
        job = DEFINITION["jobs"][0];r = self.header(job, "FAIL");r["gpu_executed"] = False
        folder = self.root / "jobs" / job["name"]
        core.atomic_json(folder / "attempt_001/parent_result.json", r)
        core.write_receipt(self.root, job, r["job_signature"], r)
        self.assertEqual(core.inspect_receipt(self.root, DEFINITION, job, r["job_signature"])["status"], "FAIL")
        (folder / "attempt_001/parent_result.json").write_text("changed", encoding="utf-8")
        with self.assertRaises(AssertionError):
            core.inspect_receipt(self.root, DEFINITION, job, r["job_signature"])

    def test_failed_job_without_gpu_allocation_can_be_reported(self):
        job = DEFINITION["jobs"][0];r = self.header(job, "FAIL");r["gpu_executed"] = False
        core.write_receipt(self.root, job, r["job_signature"], r)
        result = core.report_outputs(self.root, DEFINITION)
        self.assertEqual(result["overall"], "FAIL")
        self.assertFalse(result["gpu_executed"])

    def test_corrupt_last_checkpoint_falls_back_to_previous_attempt(self):
        job = DEFINITION["jobs"][1];sig = core.job_signature(DEFINITION, job)
        for i in (1, 2):
            folder = self.root / "jobs" / job["name"] / ("attempt_00" + str(i));folder.mkdir(parents=True)
            (folder / "state_1.npy").write_bytes(b"recorded state")
            core.atomic_json(folder / "progress.json", dict(job_signature=sig,
                protocol_fingerprint=core.digest(core.canonical(DEFINITION)), source_commit=SPEC["source_commit"],
                scope="CUDA_EVIDENCE", backend="cuda", state_file="state_1.npy", state_sha256=core.sha(folder / "state_1.npy"),
                original_initial_array_sha256="original-seed", original_initial_energy=-3., total_iterations=i * 256, chunk=i))
        (folder / "state_1.npy").write_bytes(b"incomplete write")
        checkpoint = core.find_checkpoint(self.root, DEFINITION, job, sig)
        self.assertEqual(checkpoint["total_iterations"], 256)

    def test_timeout_stops_child_and_returns_incomplete(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code, reason = core.run_process([sys.executable, "-c", "import time;time.sleep(20)"],
                self.root / "stdout.log", os.environ.copy(), self.root, .2, .05)
        self.assertEqual(reason, "TIMEOUT")
        self.assertNotEqual(code, 0)

    def test_interrupt_preserves_attempt_and_can_be_archived(self):
        job = DEFINITION["jobs"][1]
        with patch.object(core, "run_process", side_effect=KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                core.run_job(self.root, REPO, DEFINITION, job, core.DEFAULT_SETTINGS, "test", os.environ.copy())
            summary = core.report_outputs(self.root, DEFINITION)
            archive = core.archive_outputs(self.root)
        self.assertNotEqual(summary["overall"], "PASS")
        self.assertTrue((self.root / "jobs/grid_N96/attempt_001/parent_result.json").exists())
        # The generated ZIP is a sibling; remove only this test-owned file.
        self.assertTrue(archive.parent == self.root.parent and archive.name.startswith(self.root.name))
        archive.unlink();archive.with_name(archive.name + ".sha256").unlink()

    def test_no_cuda_main_archives_without_cpu_substitution(self):
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        out = self.root / "run"
        with patch.dict(sys.modules, {"torch": fake}), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = core.main(dict(run_root=str(out)), SPEC, WORKER, CORE)
        self.assertNotEqual(result["overall"], "PASS")
        self.assertFalse(result["gpu_executed"])
        self.assertFalse((out / "run.lock").exists())
        self.assertTrue((self.root / "run.zip").exists())

    def test_restore_hash_and_path_validation(self):
        out = self.root / "run";out.mkdir();(out / "example.txt").write_text("preserved", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):archive = core.archive_outputs(out)
        dest = self.root / "restored"
        disk = types.SimpleNamespace(free=10**12)
        with patch.object(core.shutil, "disk_usage", return_value=disk):
            core.restore_archive(archive, dest)
        self.assertEqual((dest / "example.txt").read_text(), "preserved")
        (dest / "example.txt").write_text("different")
        with patch.object(core.shutil, "disk_usage", return_value=disk), self.assertRaises(AssertionError):core.restore_archive(archive, dest)
        bad = self.root / "traversal.zip"
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("../escape", b"bad")
            z.writestr("SHA256.json", json.dumps({"../escape": core.digest(b"bad")}))
        with patch.object(core.shutil, "disk_usage", return_value=disk), self.assertRaises(AssertionError):core.restore_archive(bad, self.root / "safe")

    def test_restore_refuses_insufficient_disk(self):
        out = self.root / "run";out.mkdir();(out / "example.txt").write_text("small", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):archive = core.archive_outputs(out)
        with patch.object(core.shutil, "disk_usage", return_value=types.SimpleNamespace(free=0)), self.assertRaises(AssertionError):
            core.restore_archive(archive, self.root / "restored")

    def test_corrupt_archive_payload_is_rejected(self):
        path = self.root / "corrupt.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("artifact.txt", b"altered")
            z.writestr("SHA256.json", json.dumps({"artifact.txt": core.digest(b"original")}))
        with patch.object(core.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10**12)), self.assertRaises(AssertionError):
            core.restore_archive(path, self.root / "restored")

    def test_pass_metadata_cannot_hide_failed_junit(self):
        job = DEFINITION["jobs"][0];r = self.header(job)
        folder = self.root / "jobs" / job["name"] / "attempt_001";folder.mkdir(parents=True)
        xml = folder / "pytest.xml"
        shutil.copyfile(WORKSPACE / "data/raw/release_1_2_2_ci/pytest_cpu.xml", xml)
        r.update(exit_code=0, peak_allocated_bytes=1024, junit=junit_evidence(xml, 109, SPEC["expected_test_ids_sha256"]),
                 junit_path=xml.relative_to(self.root).as_posix())
        core.write_receipt(self.root, job, r["job_signature"], r)
        self.assertEqual(core.inspect_receipt(self.root, DEFINITION, job, r["job_signature"])["status"], "PASS")
        tree = ET.parse(xml);ET.SubElement(tree.getroot().find(".//testcase"), "failure");tree.write(xml)
        # Even recomputing the outer receipt cannot make inconsistent test data pass.
        core.write_receipt(self.root, job, r["job_signature"], r)
        with self.assertRaises(AssertionError):core.inspect_receipt(self.root, DEFINITION, job, r["job_signature"])

    def test_offline_reader_retains_scientific_failure(self):
        from review_trapped3d_results import review
        job = DEFINITION["jobs"][0];r = self.header(job, "FAIL");r["gpu_executed"] = False
        core.atomic_json(self.root / "protocol.json", dict(fingerprint=core.digest(core.canonical(DEFINITION)), definition=DEFINITION))
        core.atomic_json(self.root / "source_manifest.json", dict(commit=SPEC["source_commit"], normalized_module_sha256=SPEC["source_hashes"]))
        (self.root / "worker.py").write_bytes(WORKER.encode("utf-8"))
        (self.root / "core.py").write_bytes(CORE.encode("utf-8"))
        core.write_receipt(self.root, job, r["job_signature"], r)
        with contextlib.redirect_stdout(io.StringIO()):
            core.report_outputs(self.root, DEFINITION)
            archive = core.archive_outputs(self.root)
            result = review(self.root)
        self.assertEqual(result["artifact_integrity"], "PASS")
        self.assertEqual(result["overall"], "FAIL")
        for file in (archive, archive.with_name(archive.name + ".sha256"), self.root.parent / (self.root.name + "_OFFLINE_AUDIT.json")):
            self.assertTrue(file.parent == self.root.parent and file.name.startswith(self.root.name));file.unlink()

    def test_actual_junit_ids_and_corruptions(self):
        xml = WORKSPACE / "data/raw/release_1_2_2_ci/pytest_cpu.xml"
        self.assertTrue(junit_evidence(xml, 109, SPEC["expected_test_ids_sha256"])["passed"])
        for mutation in ("duplicate", "failure", "skip_count", "missing"):
            tree = ET.parse(xml);root = tree.getroot();cases = root.findall(".//testcase")
            if mutation == "duplicate":cases[1].attrib.update(cases[0].attrib)
            elif mutation == "failure":ET.SubElement(cases[0], "failure")
            elif mutation == "skip_count":root.find(".//testsuite").set("skipped", "1")
            else:root.find(".//testsuite").remove(cases[-1])
            bad = self.root / (mutation + ".xml");tree.write(bad)
            self.assertFalse(junit_evidence(bad, 109, SPEC["expected_test_ids_sha256"])["passed"])

    def test_notebook_has_exactly_one_matching_code_cell(self):
        script = (BASE / "cpc_refinement_gpu.py").read_text(encoding="utf-8")
        nb = core.read(REPO / "tools/cpc/gpu/GPU_TRAPPED3D_REFINEMENT.ipynb")
        self.assertEqual(len(nb["cells"]), 1)
        self.assertEqual(nb["cells"][0]["cell_type"], "code")
        self.assertEqual("".join(nb["cells"][0]["source"]), script)
        tree = ast.parse(script)
        strings = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   for t in n.targets if isinstance(t, ast.Name) and t.id in ("CORE_SOURCE", "WORKER_SOURCE")}
        self.assertEqual(strings["CORE_SOURCE"], CORE)
        self.assertEqual(strings["WORKER_SOURCE"], WORKER)

    def test_cpu_worker_checkpoint_roundtrip_is_never_gpu_evidence(self):
        job = dict(name="cpu_harness_only", kind="solve", N=12, L=4., pad=2, seed="gaussian", rng=0, residual_target=1e-6, max_iterations=8)
        first = self.root / "first";first.mkdir()
        request = dict(job=job, source=str(REPO), source_commit=SPEC["source_commit"], expected_source_hashes=SPEC["source_hashes"],
            job_signature="explicit-cpu-selftest", protocol_fingerprint="explicit-cpu-selftest", backend="cpu", scope="HARNESS_SELFTEST",
            attempt_dir=str(first), session="LOCAL_CPU_HARNESS_CHECK", expected_tests=109,
            expected_test_ids_sha256=SPEC["expected_test_ids_sha256"], memory_fraction=.82, criteria=core.CRITERIA,
            chunk_iterations=4, cfg=core.make_cfg(job))
        core.atomic_json(self.root / "request1.json", request)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
        with (self.root / "worker1.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, str(BASE / "cpc_refinement_worker.py"), str(self.root / "request1.json")], env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
        self.assertEqual(result.returncode, 0, (self.root / "worker1.log").read_text(errors="replace"))
        data = core.read(first / "result.json");progress = core.read(first / "progress.json")
        self.assertEqual(data["status"], "SELFTEST_COMPLETE");self.assertFalse(data["gpu_executed"])
        self.assertEqual(data["actual_device"], "cpu");self.assertEqual(data["dtype"], "torch.complex128")
        self.assertLess(abs(data["mass"] - 1.), 1e-12)
        self.assertEqual(progress["total_iterations"], 8)
        self.assertTrue((first / "state_0.npy").exists() and (first / "state_1.npy").exists())
        second = self.root / "second";second.mkdir()
        request.update(attempt_dir=str(second), resume_checkpoint=dict(path=str(first / progress["state_file"]),
            sha256=progress["state_sha256"], total_iterations=progress["total_iterations"], chunk=progress["chunk"],
            original_initial_array_sha256=progress["original_initial_array_sha256"], original_initial_energy=progress["original_initial_energy"]))
        core.atomic_json(self.root / "request2.json", request)
        with (self.root / "worker2.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, str(BASE / "cpc_refinement_worker.py"), str(self.root / "request2.json")], env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
        self.assertEqual(result.returncode, 0, (self.root / "worker2.log").read_text(errors="replace"))
        resumed = core.read(second / "result.json")
        self.assertEqual(resumed["status"], "SELFTEST_COMPLETE")
        self.assertFalse(resumed["gpu_executed"])
        self.assertEqual(resumed["state_sha256"], progress["state_sha256"])
        self.assertEqual(resumed["initial_array_sha256"], data["initial_array_sha256"])
        self.assertEqual(resumed["initial_energy"], data["initial_energy"])
        self.assertAlmostEqual(resumed["energy"], data["energy"], places=12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REVIEW / "PROTOCOL_CHECKS.json"))
    args = parser.parse_args()
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(scope="Local harness checks and N=12 CPU worker checkpoint roundtrip; NOT CUDA/refinement paper evidence",
                  tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  seconds=time.monotonic() - started, gpu_executed=False, numerical_release_unchanged=True,
                  protocol_fingerprint=core.digest(core.canonical(DEFINITION)))
    core.atomic_json(args.output, report)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":main()
