"""Shared helpers for the Kaggle benchmark scripts (install, hardware record, IO).

Each benchmark script inlines an identical copy of the preamble it needs, so a
script can be pasted into a single Kaggle cell; this module is the reference
version kept in the repository.
"""
import json
import os
import platform
import subprocess
import sys

REPO = "https://github.com/Duyle0503/becGPP.git"
TAG = "v1.1.0"


def install(min_version="1.1"):
    try:
        import becgpp
        if becgpp.__version__.startswith(min_version):
            return
    except Exception:
        pass
    for ref in (f"@{TAG}", ""):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                            "--force-reinstall", f"git+{REPO}{ref}"])
        if r.returncode == 0:
            return
    raise RuntimeError("could not install becGPP from GitHub")


def hardware_record(path):
    import torch
    import becgpp
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    info = dict(python=platform.python_version(), torch=torch.__version__,
                becgpp=becgpp.__version__, code_version=becgpp.CODE_VERSION,
                cuda=getattr(torch.version, "cuda", None), device=becgpp.DEV,
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                gpu_mem_GB=(round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                            if torch.cuda.is_available() else None),
                cpu_model=cpu, cpu_logical_cores=os.cpu_count(),
                torch_cpu_threads=torch.get_num_threads(), platform=platform.platform())
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm",
             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        info["nvidia_smi"] = None
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    return info
