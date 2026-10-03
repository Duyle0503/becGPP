"""Command-line entry point.

Usage::

    becgpp --config examples/config_2D_rotating_lattice.yaml
    becgpp --mode single --dimension 3D --G_C 20 --kernel newton --outdir ./out
    python -m becgpp --config examples/config_2D_harmonic.yaml

A config file (YAML if PyYAML is installed, otherwise JSON) supplies overrides on
top of ``CFG_DEFAULTS``; any ``--key value`` pair on the command line overrides
the file in turn. ``--outdir`` sets the output directory.
"""
import argparse
import json
import os

from .config import default_cfg, canonical_key, CFG_DEFAULTS
from . import paths
from .modes import run


def _load_config_file(path):
    with open(path, "r") as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("PyYAML is required to read .yaml configs; "
                             "install it or use a .json config") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _coerce(value, template):
    """Coerce a CLI string to the type of the matching default value."""
    if isinstance(template, bool):
        return str(value).lower() in ("1", "true", "yes", "on")
    if isinstance(template, int) and not isinstance(template, bool):
        try:
            return int(value)
        except ValueError:
            return float(value)
    if isinstance(template, float):
        return float(value)
    if isinstance(template, (list, tuple)):
        # lists on the command line are JSON, e.g. --sweep_values "[5, 10, 20]"
        out = json.loads(value)
        if not isinstance(out, list):
            raise SystemExit(f"expected a JSON list, got {value!r}")
        return out
    return value


def build_parser():
    p = argparse.ArgumentParser(prog="becgpp", description="General GP(-Poisson) ground-state solver.")
    p.add_argument("--config", help="Path to a YAML or JSON config file with CFG overrides.")
    p.add_argument("--outdir", help="Output directory (overrides GPP_OUTDIR).")
    # allow --<key> for every known CFG key
    for key, val in CFG_DEFAULTS.items():
        p.add_argument(f"--{key}", default=None,
                       help=f"override CFG['{key}'] (default {val!r})")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    file_cfg = {}
    if args.config:
        loaded = _load_config_file(args.config)
        if not isinstance(loaded, dict):
            parser.error(f"{args.config}: expected a mapping of CFG keys")
        # numbers such as 1e3 are strings for PyYAML (YAML 1.1): type them like the
        # CLI flags, from the default of the matching key
        for key, v in loaded.items():
            tmpl = CFG_DEFAULTS.get(canonical_key(key))
            file_cfg[key] = _coerce(v, tmpl) if (isinstance(v, str) and tmpl is not None) else v
    try:
        cfg = default_cfg(**file_cfg)       # validates keys and maps aliases (res=, N=, ...)
    except ValueError as exc:
        parser.error(f"{args.config}: {exc}")
    # apply any explicit CLI overrides (typed by the default's type)
    for key in CFG_DEFAULTS:
        v = getattr(args, key, None)
        if v is not None:
            cfg[key] = _coerce(v, CFG_DEFAULTS[key])

    outdir = args.outdir or os.environ.get("GPP_OUTDIR")
    if outdir:
        paths.configure(outdir)

    return run(cfg)


def console(argv=None):
    """Console-script entry point: run, then exit with status 0. ``main`` returns
    the run's result, which ``sys.exit`` would print and turn into status 1."""
    main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(console())
