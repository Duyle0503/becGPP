"""Audit the author-provided repaired CUDA regression evidence (stdlib only)."""
from pathlib import Path
import argparse
import ast
import collections
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from workspace_paths import find_workspace

ROOT = find_workspace()
COMMIT = "75f0a304acf2ff61dceb3c5ac4df10a9ef84b197"
RAW_NAME = "becgpp_gpu_20261017T141120Z_cwk2teki"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def test_ids(path):
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    assert len(cases) == 97
    assert not root.findall(".//failure") and not root.findall(".//error") and not root.findall(".//skipped")
    ids = [(t.get("classname"), t.get("name")) for t in cases]
    assert len(set(ids)) == len(ids), "Duplicate test identifiers"
    return ids

def review(raw, archive, output):
    manifest = read(raw / "SHA256.json")
    assert len(manifest) == 13
    for name, value in manifest.items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert sha(raw / name) == value, name
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        prefix = next(n[:-len("SHA256.json")] for n in z.namelist() if n.endswith("SHA256.json"))
        assert json.loads(z.read(prefix + "SHA256.json")) == manifest
        for name, value in manifest.items():
            assert hashlib.sha256(z.read(prefix + name)).hexdigest() == value, name
        assert {n[len(prefix):] for n in z.namelist() if not n.endswith("/")} == set(manifest) | {"SHA256.json"}
    source = read(raw / "source_manifest.json")
    assert source["commit"] == COMMIT
    expected = source["normalized_module_sha256"]
    assert len(expected) == 25
    for name, value in expected.items():
        content = (ROOT / "code/becGPP" / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(content).hexdigest() == value, name
    script = ast.parse((ROOT / "code/gpu/GPU_RETEST_ONLY.py").read_text(encoding="utf-8"))
    worker_node = next(n for n in script.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "WORKER" for t in n.targets))
    worker = ast.literal_eval(worker_node.value)
    assert worker.replace("\r\n", "\n") == (raw / "worker.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    result = read(raw / "jobs/02_cuda_regressions/result.json")
    summary = read(raw / "SUMMARY.json")
    environment = read(raw / "environment.json")
    outcomes = read(raw / "outcomes.json")
    plan = read(raw / "plan.json")
    assert len(outcomes) == 1 and outcomes[0] == result
    assert len(plan["jobs"]) == 1 and plan["jobs"][0] == result["job"]
    assert summary["environment"] == environment
    assert summary["overall"] == "PASS" and summary["status_counts"] == {"PASS": 1}
    assert summary["planned_jobs"] == summary["completed_jobs"] == 1
    assert summary["gpu_executed"] is True and environment["gpu_evidence"] is True
    assert environment["solver_commit"] == COMMIT and environment["regressions_only"] is True
    assert result["status"] == "PASS" and result["backend"] == result["job"]["backend"] == "cuda"
    assert result["expected_tests"] == 97 and result["excluded_gpu_reference_cases"] == 0
    stats = result["stats"]
    assert int(stats["tests"]) == 97 and all(int(stats[k]) == 0 for k in ("failures", "errors", "skipped"))
    ids = test_ids(raw / "jobs/02_cuda_regressions/pytest.xml")
    cpu_ids = test_ids(ROOT / "data/review/portable_suite.xml")
    assert collections.Counter(ids) == collections.Counter(cpu_ids), "CUDA and CPU test coverage differs"
    old_xml = ET.parse(ROOT / "data/raw/gpu_2026_10_17/jobs/02_cuda_regressions/pytest.xml").getroot()
    old_ids = {(t.get("classname"), t.get("name")) for t in old_xml.findall(".//testcase")}
    old_failed = {(t.get("classname"), t.get("name")) for t in old_xml.findall(".//testcase") if t.find("failure") is not None or t.find("error") is not None}
    assert len(old_ids) == 90 and len(old_failed) == 16 and old_ids <= set(ids)
    assert len(set(ids) - old_ids) == 7
    assert result["cuda"]["name"] == environment["gpu_name"] == "Tesla T4"
    assert result["cuda"]["runtime"] == environment["cuda_runtime"] == "12.8"
    assert result["cuda"]["peak_allocated_bytes"] > 0
    assert "97 passed in 29.52s" in (raw / "jobs/02_cuda_regressions/stdout.log").read_text()
    report = dict(review_date="2026-10-18", run_created_utc=environment["created_utc"],
                  commit=COMMIT, archive_sha256=sha(archive), archive_crc_valid=True,
                  raw_files_verified=len(manifest), source_and_test_hashes_verified=len(expected),
                  embedded_worker_matches=True, cuda_tests_passed=97, failures=0, errors=0, skipped=0,
                  excluded_cases=0, same_97_test_ids_as_cpu=True, former_16_failed_cases_now_pass=True,
                  former_7_excluded_cases_now_pass=True, pytest_seconds=float(stats["time"]),
                  job_seconds=result["seconds"], gpu=environment["gpu_name"], python=environment["python"],
                  torch=environment["torch"], cuda_runtime=environment["cuda_runtime"],
                  peak_allocated_bytes=result["cuda"]["peak_allocated_bytes"],
                  peak_reserved_bytes=result["cuda"]["peak_reserved_bytes"],
                  result="PASS", scope="Repaired CUDA regressions only; original 71-job batch and its 14 review flags are unchanged.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=ROOT / "data/raw/gpu_retest_2026_10_17")
    p.add_argument("--archive", type=Path, default=ROOT / "data/raw/archives" / (RAW_NAME + ".zip"))
    p.add_argument("--out", type=Path, default=ROOT / "data/review/CUDA_RETEST_REVIEW.json")
    args = p.parse_args()
    print(json.dumps(review(args.raw, args.archive, args.out), indent=2))
