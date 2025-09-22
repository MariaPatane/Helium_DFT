import numpy as np
import pytest


# test norm(u)

def test_norm_integral_matches_Z(dft_solver):
    """
        The test verify that the normalization procedure conserves the electron count.
        After calling `norm(u)`, the integrated electron density should equal Z,
        consistent with the convention ρ(r) = Z * |u_norm(r)|^2. This test constructs
        a non-trivial trial wavefunction u, normalizes it, and checks that the density
        integrates to the correct number of electrons.
    """
    x = dft_solver.x
    # non trivial u and different from zero
    u = np.sin(np.pi * x / x[-1])
    den = dft_solver.norm(u)
    total = dft_solver.h * np.sum(den)
    assert np.isfinite(total)
    assert np.isclose(total, dft_solver.Z, rtol=1e-6, atol=1e-9)


def test_norm_zero_input(dft_solver):
    """
       The test verify that if the function u=0, 
       then the normalized function should be equal
       to zero. 
       If u=0 -> norm=0
    """
    u = np.zeros(dft_solver.N)
    den = dft_solver.norm(u)
    assert np.all(den == 0.0)
    assert np.isfinite(den).all()
