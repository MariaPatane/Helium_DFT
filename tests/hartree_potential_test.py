import numpy as np

def test_shape_hartree_potential(dft_solver):
    """ 
        Hartree potential array must have the same size of the redial grid
    """
    Vh = dft_solver.hartree(nn = int, l = int)
    assert Vh.shape == dft_solver.x.shape

def test_NaN_or_inf_hartree_potential(dft_solver):
    """ 
        Hartree potential array contains no NaN values
    """
    Vh = dft_solver.hartree(nn = int, l = int)
    assert np.all(np.isfinite(Vh))

def test_hydrogen_poisson_solution(hydrogen_solver_physical_test):
    """
    Check that the numerical solution U(r) of hartree energy, is 
    compatible with the solution of Poisson equation for H  
    U(r) = 1 - (r+1) * exp(-2r)
    """
    solver = hydrogen_solver_physical_test
    #hartree potential for nn=1 l=0 (1s)
    solver.hartree(1, 0)

    # Analitical solution
    U_exact = 1 - (solver.x + 1) * np.exp(-2 * solver.x)

    #confront of numerical and analitical solution
    assert np.allclose(solver.U, U_exact, atol=1e-2), \
        f"Max error = {np.max(np.abs(solver.U - U_exact))}"
