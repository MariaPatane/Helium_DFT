import numpy as np
import pytest
from DFT.DFT_solver import DFTHelium

def test_scf_end_to_end_converges():
    """
        this test of the SCF loop for helium (Z=2) aims to verify that:
        - The final energy is finite and negative (bound state).
        - Convergence is reached in a reasonable number of iterations.
        - The total electron density integrates to ~Z (normalization check).
    """
    m = DFTHelium(Z=2, r_max=5.0, h=0.01)
    E,iters = m.run(max_iteration=50, tol=1e-6)
    assert np.isfinite(E)
    assert E < 0.0
    assert 1 <= iters <= 50
    total_density = m.h * np.sum(m.den)
    assert np.isclose(total_density, m.Z, rtol=1e-5, atol=1e-7)

def test_scf_tighter_tol_requires_more_iters():
    """
        This test check that stricter convergence tolerance requires more iterations.
    """
    m1 = DFTHelium(Z=2, r_max=5.0, h=0.01)
    _, it_loose = m1.run(max_iteration=100, tol=1e-3)

    m2 = DFTHelium(Z=2, r_max=5.0, h=0.01)
    _, it_tight = m2.run(max_iteration=100, tol=1e-7)

    assert it_loose <= it_tight

@pytest.mark.parametrize(("rmax","h"), [(3.0, 0.02), (8.0, 0.02), (8.0, 0.005)])
def test_scf_robust_grid_settings(rmax, h):
    """
        Verify SCF robustness across different radial grids.
    """
    m = DFTHelium(Z=2, r_max=rmax, h=h)
    E, iters= m.run(max_iteration=80, tol=5e-6)
    assert np.isfinite(E)
    assert iters <= 80
