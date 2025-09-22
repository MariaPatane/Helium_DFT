import numpy as np
#test correlation potential

def test_correlation_no_nan(dft_solver):
    """
       to test the proper functioning of correlation potential for different value of density
    """
    # 0,low, medium, high value of density to test the correct functioning of correlation potential
    dft_solver.den = np.array([0.0, 1e-12, 0.01, 0.5, 10.0])
    Vc = dft_solver.correlation(None, None)
    assert np.all(np.isfinite(Vc))


def test_correlation_continuity_near_rs1(dft_solver):
    """
    test the continuity of Vc around rs=1
    """
    # density able to produce rs around 1 
    # (rs = (3 / (4π den))^(1/3)  ->  den = 3/(4π rs^3))
    den_rs_below_1 = 3 / (4 * np.pi * (0.99**3))  # rs ≈ 0.99
    den_rs_above_1 = 3 / (4 * np.pi * (1.01**3))  # rs ≈ 1.01

    dft_solver.den = np.array([den_rs_below_1])
    Vc_low = dft_solver.correlation(None, None)[0]

    dft_solver.den = np.array([den_rs_above_1])
    Vc_high = dft_solver.correlation(None, None)[0]

    diff = abs(Vc_high - Vc_low)

    assert diff < 1.0, f"too large discontinuity around rs=1: diff={diff}"
   

def test_correlation_zero_density(dft_solver):
    """
        test aims to prove that if density = 0 -> Vc=0
    """
    dft_solver.den = np.zeros(5)
    Vc = dft_solver.correlation(None, None)
    # With zero density, correlation potential must be zero
    assert np.all(Vc == 0.0)

def test_correlation_high_density(dft_solver):
    """
        test aims to prove that correlation potential at high density is finite and negative
    """   
    dft_solver.den = np.array([10.0, 20.0, 50.0])
    Vc = dft_solver.correlation(None, None)
    # Values must be finite and typically negative
    assert np.all(np.isfinite(Vc))
    assert np.all(Vc <= 0)

