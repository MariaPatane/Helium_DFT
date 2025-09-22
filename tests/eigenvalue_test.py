import numpy as np
#tests eingenvalues solver 

def test_output_shape(dft_solver):
    """
        Check that u_eff has the same length as N
    """
    u_eff, nodes = dft_solver.eigenvalue_Eff_iterative(l=0, E_eff=-1.0)
    assert u_eff.shape[0] == dft_solver.N

def test_nodes_nonnegative(dft_solver):
    """
        Number of nodes should not be negative
    """
    _, nodes = dft_solver.eigenvalue_Eff_iterative(l=0, E_eff=-1.0)
    assert nodes >= 0

def test_values_finite(dft_solver):
    """
       All values of the radial wavefunction should be finite
    """
    u_eff, _ = dft_solver.eigenvalue_Eff_iterative(l=0, E_eff=-1.0)
    assert np.all(np.isfinite(u_eff))

def test_nodes_for_known_pattern(dft_solver):
    """
        Check the node count in a simple case, for zero potential and 
        l=0, small negative energy u_eff should have 0 nodes
    """
    _, nodes = dft_solver.eigenvalue_Eff_iterative(l=0, E_eff=-0.1)
    assert nodes == 0
