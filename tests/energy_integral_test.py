import pytest
import numpy as np 


# test energy_integral(v)


def test_energy_integral_constant_potential(dft_solver):
    """
        Check that for a constant potential v=1, the energy integral 
        matches 0.5 * h * sum(den), consistent with the implementation.
    """
    u = np.exp(-dft_solver.x)
    den = dft_solver.norm(u)
    v = np.ones_like(den)
    E = dft_solver.energy_integral(v)
    expected = 0.5 * dft_solver.h * np.sum(den)
    assert np.isfinite(E)
    assert np.isclose(E, expected, rtol=1e-12, atol=1e-12)


def test_energy_integral_zero_cases(dft_solver):
    """
        If potential=0 or den=0, energy must be 0.
    """
    # caso v=0
    _ = dft_solver.norm(np.exp(-dft_solver.x))
    E_zero_v = dft_solver.energy_integral(np.zeros(dft_solver.N))
    assert E_zero_v == 0.0

    # caso den=0
    dft_solver.den[:] = 0.0
    E_any_v = dft_solver.energy_integral(np.ones(dft_solver.N))
    assert E_any_v == 0.0