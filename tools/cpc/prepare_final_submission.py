"""Build portable submission archives from code/, data/, latex/; use only stdlib.

The package is a submission candidate. Corrected CPU and CUDA regressions have
passed; author confirmations remain pending. See README.md and GPU_ASSESSMENT.md.
"""
from pathlib import Path
import hashlib
import json
import re
import zipfile
from workspace_paths import find_workspace

ROOT = find_workspace()
OUT = ROOT / "data/submission"
EXCLUDED = {".git", "__pycache__", ".pytest_cache", "build", "dist", "becgpp.egg-info", "becgpp_out"}

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def tree(folder, prefix, exclusions=()):
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder)
        if p.is_file() and not set(rel.parts) & set(exclusions):
            yield p, (Path(prefix) / rel).as_posix()

def make_zip(name, members):
    target = OUT / name
    members = list(members)
    assert len({arc for _, arc in members}) == len(members)
    records = {}
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for p, arc in members:
            records[arc] = sha(p)
            compression = zipfile.ZIP_STORED if p.suffix.lower() in {".zip", ".whl", ".gz", ".png", ".jpg"} else zipfile.ZIP_DEFLATED
            z.write(p, arc, compress_type=compression)
        z.writestr("CHECKSUMS/" + name.removesuffix(".zip") + ".json", json.dumps(records, indent=2) + "\n")
    # Decompress and hash every entry, not just archive metadata.
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for arc, expected in records.items():
            h = hashlib.sha256()
            with z.open(arc) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            assert h.hexdigest() == expected, arc
    print(f"{name}: {len(members)} files, {target.stat().st_size:,} bytes; CRC and contents verified", flush=True)
    return target

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check = json.loads((ROOT / "data/review/FINAL_PROJECT_CHECK.json").read_text(encoding="utf-8"))
    assert check["integrity_status"] == "PASS"
    source = make_zip("Software_source.zip", tree(ROOT / "code/becGPP", "code/becGPP", EXCLUDED))
    latex = ROOT / "latex"
    tex = (latex / "main.tex").read_text(encoding="utf-8") + (latex / "candidate_verification.tex").read_text(encoding="utf-8")
    figures = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))
    latex_members = [(latex / name, "latex/" + name) for name in ("main.tex", "references.bib", "candidate_verification.tex")]
    latex_members += [(latex / "build/main.bbl", "latex/main.bbl")]
    latex_members += [(latex / "figs" / name, "latex/figs/" + name) for name in figures]
    latex_source = make_zip("LaTeX_source.zip", latex_members)
    evidence = make_zip("Verification.zip", list(tree(ROOT / "data/review", "data/review")) + list(tree(ROOT / "data/verification", "data/verification")))
    raw = make_zip("Reproducibility_data.zip", list(tree(ROOT / "data/raw/gpu_2026_10_17", "data/raw/gpu_2026_10_17")) + list(tree(ROOT / "data/raw/gpu_retest_2026_10_17", "data/raw/gpu_retest_2026_10_17")) + list(tree(ROOT / "data/history/legacy_v1_0", "data/history/legacy_v1_0", {"__pycache__"})) + [(ROOT / "data/README.md", "data/README.md"), (ROOT / "data/raw/archives/becgpp_gpu_20261017T141120Z_cwk2teki.zip", "data/raw/archives/becgpp_gpu_20261017T141120Z_cwk2teki.zip")])
    payload = [(ROOT / "README.md", "README.md"), (source, "code/Software_source.zip"), (latex_source, "latex/LaTeX_source.zip"),
               (latex / "main.pdf", "latex/main.pdf"), (evidence, "data/Verification.zip"), (raw, "data/Reproducibility_data.zip"),
               (ROOT / "data/review/GPU_ASSESSMENT.md", "data/GPU_ASSESSMENT.md")]
    payload += list(tree(ROOT / "code/gpu", "code/gpu"))
    payload += list(tree(ROOT / "code/releases", "code/releases"))
    payload += list(tree(latex / "submission", "latex/submission"))
    payload += list(tree(ROOT / "code/tools", "code/tools", {"__pycache__"}))
    final = make_zip("CPC_submission_package.zip", payload)
    products = [source, latex_source, evidence, raw, final]
    checksums = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in products}
    (OUT / "SHA256.json").write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
