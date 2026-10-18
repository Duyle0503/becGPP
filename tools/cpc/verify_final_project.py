"""Check the organized project and preserve the distinction between evidence and pass status.

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

ROOT = find_workspace()
HEAD = "75f0a304acf2ff61dceb3c5ac4df10a9ef84b197"
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
    for name, expected in source.items():
        content = (repo / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(content).hexdigest() == expected, name
    checks["unchanged_numerical_modules"] = len(source)
    if (repo / ".git").exists():
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        subprocess.run(["git", "merge-base", "--is-ancestor", HEAD, "HEAD"], cwd=repo, check=True)
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
        checks["git_head"] = head
        checks["verified_solver_test_commit"] = HEAD
    else:
        checks["git_head"] = "source distribution; numerical hashes checked"
    tests = ET.parse(review / "portable_suite.xml").getroot().findall(".//testcase")
    assert len(tests) == 97
    assert all(not list(t) for t in tests)
    checks["cpu_regressions_passed"] = 97
    ci = read(review / "LATEST_GITHUB_CHECKS.json")
    assert ci["headRevision"] == HEAD
    assert len(ci["jobs"]) == 3
    assert all(j["conclusion"] == "success" and j["head_sha"] == HEAD for j in ci["jobs"])
    checks["github_ci_success_jobs"] = 3
    for stem in ("GPU_CHECK_ONE_CELL", "GPU_RETEST_ONLY"):
        script = (ROOT / "code/gpu" / (stem + ".py")).read_text(encoding="utf-8")
        syntax = ast.parse(script)
        hashes = next(n for n in syntax.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and "HASH" in t.id for t in n.targets))
        for name, expected in ast.literal_eval(hashes.value).items():
            content = (repo / name).read_bytes().replace(b"\r\n", b"\n")
            assert hashlib.sha256(content).hexdigest() == expected, name
        assert HEAD in script
        nb = read(ROOT / "code/gpu" / (stem + ".ipynb"))
        cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(cells) == 1
        assert "".join(cells[0]["source"]) == script
    checks["one_cell_gpu_notebooks"] = 2
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
    checks["integrity_status"] = "PASS"
    out = review / "FINAL_PROJECT_CHECK.json"
    out.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))

if __name__ == "__main__":
    main()
