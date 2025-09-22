
import pytest
import numpy as np
from DFT.DFT_solver import DFTHelium

@pytest.fixture
def dft_solver():
    """
       Generic DFTHelium solver for all tests
    """
    solver = DFTHelium(Z=2, r_max=5.0, h=0.1)
    return solver

@pytest.fixture
def hydrogen_solver():
    """
       Generic DFT solver for hydrogen 
    """
    solver = DFTHelium(Z=1, r_max=5.0, h=0.01)  
    # density 1s for H
    solver.den = 4 *(solver.x)**2* np.exp(-2 * solver.x)  # P(r) = 4 r^2 e^{-2r}
    return solver    

@pytest.fixture
def hydrogen_solver_physical_test():
    """ 
        DFT solver for hydrogen atom with a denser 
        grid (4000 points)to obtain accurate physical results
    """
    solver = DFTHelium(Z=1, r_max=20.0, h=0.005)  
    solver.den = 4 *(solver.x)**2* np.exp(-2 * solver.x)  # P(r) = 4 r^2 e^{-2r}
    return solver  