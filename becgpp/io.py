"""Run identifiers and incremental CSV output."""
import csv
import hashlib
import json

from .config import CFG_DEFAULTS
from .constants import CODE_VERSION

# keys that cannot change the computed state: output, display, diagnostics, and the
# settings of the other modes. Everything else (physics, grid, seed, solver) enters
# the run id, so a checkpoint is only resumed by a run that would reproduce it.
# maxit is excluded on purpose: rerunning with a larger maxit continues the run.
_RID_EXCLUDE = frozenset({
    "mode", "tag", "maxit", "check", "record_trace", "mem_warn_gb",
    "want_vortices", "want_lll", "tf_cache", "validate_N",
    "sweep_param", "sweep_values", "sweep_autobox", "sweep_continuation",
    "conv_Ngrids", "conv_box_factors", "conv_pads",
    "show_inline", "save_figs", "save_ckpt", "zip_output",
})


def run_id(cfg):
    keys = sorted(k for k in CFG_DEFAULTS if k not in _RID_EXCLUDE)
    raw = json.dumps(dict(code=CODE_VERSION, **{k: cfg.get(k, CFG_DEFAULTS[k]) for k in keys}),
                     sort_keys=True, default=str)
    return f"{cfg.get('tag', 'becgpp')}_{hashlib.md5(raw.encode()).hexdigest()[:10]}"


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
