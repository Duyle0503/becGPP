"""Run identifiers and incremental CSV output."""
import csv
import hashlib
import json

from .constants import CODE_VERSION


def run_id(cfg):
    keys = ["dimension", "s", "Omega", "beta2", "beta3", "G_C", "kernel",
            "L", "Ngrid", "pad", "seed", "res_tol", "rng"]
    raw = json.dumps(dict(code=CODE_VERSION, **{k: cfg.get(k) for k in keys}), sort_keys=True)
    return f"{cfg.get('tag', 'becgpp')}_{hashlib.md5(raw.encode()).hexdigest()[:10]}"


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
