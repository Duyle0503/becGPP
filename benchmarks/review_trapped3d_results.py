"""Verify/reassess the one-cell v2 ZIP without importing PyTorch or trusting PASS labels.

python benchmarks/review_trapped3d_results.py RESULTS.zip --out extracted_results
"""
import argparse
import json
import math
from pathlib import Path
from cpc_refinement_core import (assess, canonical, collect_results, definition, digest,
                                 read, restore_archive, atomic_json, sha)


def metrics_compatible(recorded, recomputed):
    """Allow last-bit reduction differences across NumPy/CPU implementations.

    This tolerance applies only to reproduction of the stored summary. The
    scientific acceptance thresholds are recomputed unchanged, and verdicts
    and criterion booleans must still agree exactly.
    """
    if isinstance(recorded, dict) and isinstance(recomputed, dict):
        return set(recorded) == set(recomputed) and all(metrics_compatible(recorded[k], recomputed[k]) for k in recorded)
    if isinstance(recorded, list) and isinstance(recomputed, list):
        return len(recorded) == len(recomputed) and all(metrics_compatible(a, b) for a, b in zip(recorded, recomputed))
    if isinstance(recorded, bool) or isinstance(recomputed, bool):
        return type(recorded) is type(recomputed) and recorded == recomputed
    if isinstance(recorded, (int, float)) and isinstance(recomputed, (int, float)):
        return math.isclose(recorded, recomputed, rel_tol=1e-12, abs_tol=1e-24)
    return recorded == recomputed


def review(path, output=None):
    path = Path(path).resolve()
    root = path if path.is_dir() else Path(output or path.with_suffix("")).resolve()
    if not path.is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        restore_archive(path, root)
    base = Path(__file__).resolve().parent
    spec = read(base / "cpc_refinement_spec.json")
    worker = (base / "cpc_refinement_worker.py").read_text(encoding="utf-8")
    core = (base / "cpc_refinement_core.py").read_text(encoding="utf-8")
    stored = read(root / "protocol.json")
    include_tests = any(j["kind"] == "regression" for j in stored["definition"]["jobs"])
    expected = definition(spec, worker, core, include_tests)
    assert stored["definition"] == expected and stored["fingerprint"] == digest(canonical(expected)), "Protocol does not match this reviewed runner"
    assert sha(root / "worker.py") == expected["worker_sha256"]
    assert sha(root / "core.py") == expected["core_sha256"]
    source = read(root / "source_manifest.json")
    assert source["commit"] == spec["source_commit"] and source["normalized_module_sha256"] == spec["source_hashes"]
    manifest = read(root / "SHA256.json")
    from cpc_refinement_core import safe_path
    for name, value in manifest.items():
        assert sha(safe_path(root, name)) == value, name
    outcomes = collect_results(root, expected)
    result = assess(root, expected, outcomes)
    recorded = read(root / "SUMMARY.json")
    assert metrics_compatible(recorded["metrics"], result["metrics"]), "Stored assessment differs: metrics"
    for key in ("criteria_pass", "refinement_status", "release_cuda_status"):
        assert recorded[key] == result[key], "Stored assessment differs: " + key
    assert recorded["overall"] != "PASS" or result["overall"] == "PASS", "Stored PASS is unsupported"
    result.update(artifact_integrity="PASS", archive_or_directory=str(path),
                  metric_reproduction_tolerance=dict(relative=1e-12, absolute=1e-24, scientific_thresholds_changed=False),
                  scope="Content/hash/metadata and numerical-summary audit; this reader does not itself execute CUDA")
    # Write beside the dataset so its original manifest remains immutable.
    destination = root.parent / (root.name + "_OFFLINE_AUDIT.json")
    atomic_json(destination, result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    parser.add_argument("--out")
    args = parser.parse_args()
    review(args.results, args.out)
