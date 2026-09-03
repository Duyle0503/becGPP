"""Output directory management.

The base output directory is chosen at import time from, in order:
  1. the ``GPP_OUTDIR`` environment variable,
  2. ``/kaggle/working/becgpp`` when running on Kaggle,
  3. ``./becgpp_out`` otherwise.

Call :func:`configure` to point the package at a different directory at run time
(e.g. from a config file). Other modules reference the attributes through the
module object (``from . import paths; paths.FIG_DIR``) so that reconfiguration
propagates.
"""
import os


def _base_dir():
    env = os.environ.get("GPP_OUTDIR")
    if env:
        return env
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/becgpp"
    return "./becgpp_out"


def _make_dirs(base):
    ckpt = os.path.join(base, "ckpt")
    fig = os.path.join(base, "fig")
    for d in (base, ckpt, fig):
        os.makedirs(d, exist_ok=True)
    return base, ckpt, fig


BASE, CKPT_DIR, FIG_DIR = _make_dirs(_base_dir())


def configure(base):
    """Switch the output directory to ``base`` (created if missing)."""
    global BASE, CKPT_DIR, FIG_DIR
    BASE, CKPT_DIR, FIG_DIR = _make_dirs(base)
    return BASE
