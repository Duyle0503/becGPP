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

from .config import default_cfg, CFG_DEFAULTS
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

    cfg = default_cfg()
    if args.config:
        cfg.update(_load_config_file(args.config))
    # apply any explicit CLI overrides (typed by the default's type)
    for key in CFG_DEFAULTS:
        v = getattr(args, key, None)
        if v is not None:
            cfg[key] = _coerce(v, CFG_DEFAULTS[key])

    outdir = args.outdir or os.environ.get("GPP_OUTDIR")
    if outdir:
        paths.configure(outdir)

    return run(cfg)


if __name__ == "__main__":
    main()
