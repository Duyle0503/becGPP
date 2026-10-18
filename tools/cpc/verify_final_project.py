"""Check the organized project and preserve the distinction between evidence and pass status.

Release 1.2.2: the historical GPU evidence pins commits 5f66bfc (71-job batch)
and 75f0a30 (CUDA regression rerun); their hashes are confirmed against git
objects where the release changed a file. The release itself is checked
against its tag, its CPU CI record and, once supplied, its CUDA test run.

Run from any directory: python code/tools/verify_final_project.py
This is an artifact-integrity check, not a replacement for pytest or a CUDA run.
"""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from review_cuda_retest import review as audit_cuda_retest
from workspace_paths import find_workspace
from pinned_source import verify_pinned

ROOT = find_workspace()
HEAD = "75f0a304acf2ff61dceb3c5ac4df10a9ef84b197"   # verified CUDA rerun (rc1)
BATCH = "5f66bfc8635a1791078fdd503160e341fb12e82c"  # 71-job GPU batch
RELEASE = "1.2.2"
RELEASE_TAG = "v" + RELEASE
RELEASE_TESTS = 109
# Files changed between the GPU batch and the release (see CHANGELOG 1.2.2).
CHANGED_SINCE_BATCH = {
    "becgpp/__init__.py": "version string",
    "becgpp/constants.py": "version string",
    "becgpp/config.py": "warning for rotation without confinement",
    "becgpp/solvers.py": "recentring of translation-invariant states",
}
RAW_ARCHIVE_SHA = "e82d1ce28157f893455892162f4702b17c9f87f6f6f5563901e1b5fa1b05b9ad"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def tex_sources():
    latex = ROOT / "latex"
    pending = [latex / "main.tex"]
    visited = set()
    texts = []
    while pending:
        p = pending.pop()
        if p in visited:
            continue
        visited.add(p)
        text = p.read_text(encoding="utf-8")
        texts.append(text)
        for name in re.findall(r"\\input\{([^}]+)\}", text):
            pending.append(latex / (name if name.endswith(".tex") else name + ".tex"))
    return "\n".join(texts)

def main():
    raw = ROOT / "data/raw/gpu_2026_10_17"
    review = ROOT / "data/review"
    repo = ROOT / "code/becGPP"
    checks = {}
    manifest = read(raw / "SHA256.json")
    for name, expected in manifest.items():
        assert sha(raw / name) == expected, name
    checks["raw_files_verified"] = len(manifest)
    archive = ROOT / "data/raw/archives/becgpp_gpu_20261017T071146Z_umse17xn.zip"
    if archive.exists():
        assert sha(archive) == RAW_ARCHIVE_SHA
        checks["original_zip_sha256"] = "verified"
    else:
        checks["original_zip_sha256"] = "ZIP omitted from portable package; extracted file hashes verified"
    summary = read(raw / "SUMMARY.json")
    assert summary["overall"] == "FAIL"
    assert summary["status_counts"] == {"PASS": 56, "REVIEW": 14, "FAIL": 1}
    checks["original_gpu_status_preserved"] = summary["overall"]
    source = {k: v for k, v in read(raw / "source_manifest.json")["normalized_module_sha256"].items() if k.startswith("becgpp/")}
    unchanged = {}
    for name, expected in source.items():
        content = (repo / name).read_bytes().replace(b"\r\n", b"\n")
        same = hashlib.sha256(content).hexdigest() == expected
        assert same or name in CHANGED_SINCE_BATCH, f"{name} changed since the GPU batch without a recorded reason"
        unchanged[name] = same
    checks["modules_unchanged_since_gpu_batch"] = sum(unchanged.values())
    checks["modules_changed_since_gpu_batch"] = {k: CHANGED_SINCE_BATCH[k] for k, v in unchanged.items() if not v}
    checks["gpu_batch_sources_confirmed"] = verify_pinned(repo, BATCH, source)
    version = re.search(r'^version\s*=\s*"([^"]+)"', (repo / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
    assert version == RELEASE, version
    becver = re.search(r"\\newcommand\{\\becver\}\{([^}]+)\}", (ROOT / "latex/main.tex").read_text(encoding="utf-8")).group(1)
    assert becver == RELEASE, f"manuscript \\becver={becver}, package {version}"
    checks["release_version"] = RELEASE
    if (repo / ".git").exists():
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        tagged = subprocess.check_output(["git", "rev-parse", RELEASE_TAG + "^{commit}"], cwd=repo, text=True).strip()
        assert head == tagged, f"checkout {head[:12]} is not the release tag {RELEASE_TAG} ({tagged[:12]})"
        for ancestor in (BATCH, HEAD):
            subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, "HEAD"], cwd=repo, check=True)
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
        checks["git_head"] = head
        checks["release_tag"] = RELEASE_TAG
        checks["verified_solver_test_commit"] = HEAD
    else:
        checks["git_head"] = "source distribution; numerical hashes checked"
    tests = ET.parse(review / "portable_suite.xml").getroot().findall(".//testcase")
    assert len(tests) == 97
    assert all(not list(t) for t in tests)
    checks["cpu_regressions_passed"] = 97
    # Release CPU evidence: GitHub CI (3 Python versions) and the full suite with
    # JUnit XML from the test-run workflow, both on the tagged commit.
    release_dir = ROOT / "data/raw/release_1_2_2_ci"
    ci = read(release_dir / "github_checks.json")
    release_sha = ci["headRevision"]
    if checks["git_head"] != "source distribution; numerical hashes checked":
        assert release_sha == checks["git_head"]
    ci_jobs = [j for j in ci["jobs"] if j["workflow"] == "CI"]
    assert len(ci_jobs) == 3 and all(j["conclusion"] == "success" and j["head_sha"] == release_sha for j in ci_jobs)
    checks["github_ci_success_jobs"] = len(ci_jobs)
    suite = ET.parse(release_dir / "pytest_cpu.xml").getroot()
    cases = suite.findall(".//testcase")
    assert len(cases) == RELEASE_TESTS and all(not list(t) for t in cases)
    checks["release_cpu_tests_passed"] = len(cases)
    reference = read(repo / "data/sample_output/cpu_test_run/expected_results.json")
    assert reference["code_version"] == "becGPP_" + RELEASE and reference["validation"]["passed"]
    checks["cpu_reference_run_cases"] = len(reference["cases"])
    for stem, pin in (("GPU_CHECK_ONE_CELL", HEAD), ("GPU_RETEST_ONLY", HEAD), ("GPU_RELEASE_RETEST", RELEASE_TAG)):
        script = (ROOT / "code/gpu" / (stem + ".py")).read_text(encoding="utf-8")
        syntax = ast.parse(script)
        hashes = next(n for n in syntax.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and "HASH" in t.id for t in n.targets))
        expected = ast.literal_eval(hashes.value)
        if pin == RELEASE_TAG:
            for name, value in expected.items():
                content = (repo / name).read_bytes().replace(b"\r\n", b"\n")
                assert hashlib.sha256(content).hexdigest() == value, name
        else:
            verify_pinned(repo, pin, expected)
        assert pin in script
        nb = read(ROOT / "code/gpu" / (stem + ".ipynb"))
        cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(cells) == 1
        assert "".join(cells[0]["source"]) == script
    checks["one_cell_gpu_notebooks"] = 3
    text = tex_sources()
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    assert len(labels) == len(set(labels)), "Duplicate LaTeX labels"
    refs = re.findall(r"\\(?:eqref|ref|autoref)\{([^}]+)\}", text)
    assert set(refs) <= set(labels), sorted(set(refs) - set(labels))
    figures = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)))
    for name in figures:
        assert (ROOT / "latex/figs" / name).is_file(), name
    cites = set()
    for group in re.findall(r"\\cite[a-zA-Z]*(?:\[[^\]]*\])*\{([^}]+)\}", text):
        cites.update(k.strip() for k in group.split(","))
    bibkeys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", (ROOT / "latex/references.bib").read_text(encoding="utf-8")))
    assert cites <= bibkeys, sorted(cites - bibkeys)
    log = (ROOT / "latex/build/main.log").read_text(encoding="utf-8", errors="replace")
    assert not re.search(r"Overfull \\hbox|undefined references|Citation .* undefined|Reference .* undefined|^!", log, re.M)
    assert "Output written on" in log
    assert (ROOT / "latex/main.pdf").read_bytes().startswith(b"%PDF-")
    checks.update(figures=len(figures), labels=len(labels), cited_bibkeys=len(cites))
    cuda = audit_cuda_retest(ROOT / "data/raw/gpu_retest_2026_10_17",
                            ROOT / "data/raw/archives/becgpp_gpu_20261017T141120Z_cwk2teki.zip",
                            review / "CUDA_RETEST_REVIEW.json")
    checks["cuda_after_fixture_fix"] = "PASS"
    checks["cuda_regressions_passed"] = cuda["cuda_tests_passed"]
    checks["cuda_regression_seconds"] = cuda["pytest_seconds"]
    checks["cuda_test_ids_match_cpu"] = cuda["same_97_test_ids_as_cpu"]
    cuda_release = ROOT / "data/raw/release_1_2_2_cuda"
    if (cuda_release / "SUMMARY.json").is_file():
        summary = read(cuda_release / "SUMMARY.json")
        result = read(cuda_release / "jobs/02_cuda_regressions/result.json")
        assert summary["overall"] == "PASS" and summary["gpu_executed"] is True
        assert result["status"] == "PASS" and int(result["stats"]["tests"]) == RELEASE_TESTS
        assert read(cuda_release / "source_manifest.json")["commit"] == RELEASE_TAG
        checks["release_cuda_tests_passed"] = RELEASE_TESTS
    else:
        checks["release_cuda_tests_passed"] = "pending: run code/gpu/GPU_RELEASE_RETEST.ipynb on a CUDA GPU"
    checks["integrity_status"] = "PASS"
    out = review / "FINAL_PROJECT_CHECK.json"
    out.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))

if __name__ == "__main__":
    main()
