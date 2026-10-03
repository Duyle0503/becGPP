#!/usr/bin/env python3
"""Thin example entry point equivalent to the ``becgpp`` console script.

    python examples/run.py --config examples/config_2D_rotating_lattice.yaml --outdir ./out

It simply forwards to becgpp.cli.main; installing the package also gives you the
``becgpp`` command directly.
"""
from becgpp.cli import console

if __name__ == "__main__":
    raise SystemExit(console())
