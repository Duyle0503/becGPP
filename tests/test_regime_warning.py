"""Rotation without sufficient confinement has no ground state: default_cfg warns."""
import warnings

import pytest

from becgpp import default_cfg


@pytest.mark.parametrize("over, should_warn", [
    (dict(), False),
    (dict(Omega=0.9), False),
    (dict(Omega=1.0), False),                       # critical rotation, harmonic trap
    (dict(s=4, Omega=3.0), False),                  # quartic trap confines any Omega
    (dict(s=0, Omega=0.0), False),                  # untrapped, non-rotating
    (dict(Omega=1.01), True),                       # over-critical harmonic trap
    (dict(Omega=-1.2), True),
    (dict(s=0, Omega=0.5), True),                   # untrapped and rotating
    (dict(s=2, trap_coeff=0.25, Omega=0.8), True),  # Omega^2 > 2 c
])
def test_unbounded_rotation_warning(over, should_warn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        default_cfg(**over)
    hits = [w for w in rec if issubclass(w.category, RuntimeWarning) and "unbounded below" in str(w.message)]
    assert bool(hits) == should_warn
