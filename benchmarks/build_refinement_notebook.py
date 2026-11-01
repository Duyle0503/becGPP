"""Regenerate the self-contained one-cell script/notebook from reviewed sources."""
import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def build():
    spec = json.loads((BASE / "cpc_refinement_spec.json").read_text(encoding="utf-8"))
    worker = (BASE / "cpc_refinement_worker.py").read_text(encoding="utf-8")
    core = (BASE / "cpc_refinement_core.py").read_text(encoding="utf-8")
    assert "'''" not in worker and "'''" not in core
    preamble = '''# ONE Kaggle Python cell: exact release CUDA tests + trapped-3D refinement.
# Enable Internet and a GPU. Do not pip-install another becGPP or PyTorch.
# Optional settings: leave defaults for the complete 12-job protocol.
# To resume on a NEW session, attach the previous result ZIP as Kaggle input
# and set resume_zip to its path. Within the same session just rerun this cell.
BECGPP_REFINE_SETTINGS = globals().get("BECGPP_REFINE_SETTINGS", {})
'''
    text = preamble + "SOLVER_COMMIT = " + repr(spec["source_commit"]) + "\n"
    text += "EXPECTED_MODULE_HASHES = " + repr(spec["source_hashes"]) + "\n"
    text += "EXPECTED_TEST_COUNT = " + str(spec["expected_tests"]) + "\n"
    text += "EXPECTED_TEST_IDS_SHA256 = " + repr(spec["expected_test_ids_sha256"]) + "\n"
    text += "WORKER_SOURCE = r'''" + worker + "'''\n\n"
    text += "CORE_SOURCE = r'''" + core + "'''\n\n"
    text += '''if __name__ == "__main__":
    namespace = {"__name__": "_becgpp_refinement_core"}
    exec(compile(CORE_SOURCE, "<refinement-core>", "exec"), namespace)
    namespace["main"](BECGPP_REFINE_SETTINGS, dict(source_commit=SOLVER_COMMIT,
        source_hashes=EXPECTED_MODULE_HASHES, expected_tests=EXPECTED_TEST_COUNT,
        expected_test_ids_sha256=EXPECTED_TEST_IDS_SHA256), WORKER_SOURCE, CORE_SOURCE)
'''
    ast.parse(text)
    compile(worker, "<worker>", "exec")
    compile(core, "<core>", "exec")
    script = BASE / "cpc_refinement_gpu.py"
    script.write_text(text, encoding="utf-8", newline="\n")
    notebook = dict(cells=[dict(cell_type="code", id="trapped3d-confirm-v2", metadata={}, execution_count=None,
                               outputs=[], source=text.splitlines(keepends=True))],
                    metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                  language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
    nb = json.dumps(notebook, indent=1) + "\n"
    destinations = [BASE.parent / "tools/cpc/gpu"]
    workspace = BASE.parents[2]
    if (workspace / "code/gpu").is_dir() and (workspace / "latex/main.tex").is_file():
        destinations.append(workspace / "code/gpu")
    for dest in destinations:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "GPU_TRAPPED3D_REFINEMENT.py").write_text(text, encoding="utf-8", newline="\n")
        (dest / "GPU_TRAPPED3D_REFINEMENT.ipynb").write_text(nb, encoding="utf-8", newline="\n")
    print("Built one code cell; SHA256", hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    build()
