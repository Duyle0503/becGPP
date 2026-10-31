# Paste this ENTIRE short loader into ONE Kaggle Python cell.
# GPU + Internet must be enabled. It downloads the frozen, verified full cell.
import hashlib
import urllib.request

BECGPP_REFINE_SETTINGS = {
    "resume_zip": None,          # New session: path to the previously saved ZIP.
    "gpu_index": 0,
    "session_budget_hours": 8.0,
}

url = "https://raw.githubusercontent.com/Duyle0503/becGPP/9d40eb0e3723a05670552388cea6fffeafe1bff7/benchmarks/cpc_refinement_gpu.py"
expected = "1be2a7aa0d7b424381db283e416e3be62ea511e23e6bb8336dba4b559be0ee8a"
with urllib.request.urlopen(url, timeout=120) as response:
    code = response.read().replace(b"\r\n", b"\n")
if hashlib.sha256(code).hexdigest() != expected:
    raise RuntimeError("Runner hash mismatch; refusing to execute altered code")
namespace = {"__name__": "__main__", "BECGPP_REFINE_SETTINGS": BECGPP_REFINE_SETTINGS}
exec(compile(code, url, "exec"), namespace)
