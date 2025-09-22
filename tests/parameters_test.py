import pytest
from DFT.DFT_solver import DFTHelium

def test_Z_zero_raises_error(dft_solver):
    with pytest.raises(ValueError, match="Z"):
        DFTHelium(Z = 0)

def test_r_max_zero_raises_error(dft_solver):
    with pytest.raises(ValueError, match="r_max"):
        DFTHelium(r_max=0)

def test_h_negative_raises_error(dft_solver):
    with pytest.raises(ValueError, match="h"):
        DFTHelium(h = -0.5)

def test_h_positive_ok(dft_solver):
    m = DFTHelium(r_max=10.0, h=0.1)
    assert m.h == 0.1
    assert len(m.x) == int(10.0/0.1) 

def test_lda_params_defined(dft_solver):
    """
        Check that all required LDA parameters are defined.
    """
    solver = dft_solver
    required_params = ["A", "B", "C", "D", "gamma", "beta1", "beta2"]

    missing = [p for p in required_params if p not in solver.lda_params]
    assert not missing, f"Missing LDA parameters: {missing}"
