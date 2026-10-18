"""Locate the separate CPC bundle; never assume a clone contains the data/paper."""
import os
from pathlib import Path

def find_workspace():
    explicit = os.environ.get("BECGPP_CPC_ROOT")
    candidates = [Path(explicit).expanduser()] if explicit else [
        *Path(__file__).resolve().parents, Path.cwd(), *Path.cwd().parents
    ]
    for candidate in candidates:
        root = candidate.resolve()
        if ((root / "code/becGPP/becgpp").is_dir()
                and (root / "data/raw").is_dir()
                and (root / "latex/main.tex").is_file()):
            return root
    raise RuntimeError(
        "CPC bundle not found. Set BECGPP_CPC_ROOT to the directory containing "
        "code/becGPP, data/raw and latex/main.tex (see tools/cpc/README.md)."
    )
