"""Global backend configuration and physical constants for becGPP.

Importing this module sets the default torch dtype to float64 (double precision
is required: the long-range coupling can be many orders of magnitude weaker than
the contact term and single precision loses it) and selects the compute device.
"""
import torch

torch.set_default_dtype(torch.float64)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CODE_VERSION = "becGPP_1.0"

# --- SI physical constants (used only by the optional unit-conversion helpers) ---
HBAR = 1.054571817e-34       # J s
G_NEWTON = 6.67430e-11       # m^3 kg^-1 s^-2
AMU = 1.66053906660e-27      # kg
A0 = 5.29177210903e-11       # Bohr radius, m
